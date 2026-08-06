// Feuille de marque — le déroulé d'une partie, donne par donne.
//
// Une *donne* était lisible partout (Rejouer, « Mes donnes ») ; une *partie* ne
// l'était nulle part. Même l'écran de fin ne montre que les deux totaux : pour
// savoir sur quelle donne une partie s'est jouée, il fallait rouvrir ses donnes
// une par une, sans jamais voir le cumul. C'est cette lecture-là que la page
// rend, et elle n'était **pas calculable** avant la migration v16 — le score
// marqué d'une donne n'était enregistré nulle part.
//
// Deux échelles de points cohabitent dans une donne et ne s'additionnent
// jamais : les points **marqués** (contrat compris, `games.score_ns/ew`) et les
// points **cartes** (0-252). Les colonnes de la feuille sont les premiers ; les
// seconds ne sont qu'en infobulle de ligne, précisément pour qu'on ne puisse
// pas lire une colonne pour l'autre.
//
// Le repère est **N-S / E-O**, pas « nous / eux ». La page est partageable par
// lien, donc son lecteur n'est pas forcément un joueur de la table : la colonne
// du camp du lecteur est simplement marquée quand on sait qui il est, ce qui
// n'oblige à retourner ni les chiffres ni le vocabulaire.

import { contractChipHtml } from '../shared/cards.js';
// Noms *neutres* des camps : la page n'est pas une table où le lecteur est
// assis (c'est là que `TEAM_NAMES_REL` — « Nous / Eux » — a cours), c'est un
// document qui se partage. Même famille que Regarder et Rejouer.
import { SEAT_NAMES_FR, TEAM_NAMES_FR, TEAM_INITIALS_FR } from '../shared/seats.js';
import { botLabel } from '../shared/agents.js';
import { navigateTo } from '../router.js';

const base = () => document.querySelector('base')?.getAttribute('href') || '/';

// Le rythme est un réglage de *partie* (`matches.pacing`, migration v8), et il
// nomme à la fois le tempo et le bot des trois autres sièges — c'est le couple
// que `pacing.resolve` impose côté serveur. Une partie d'avant la colonne n'en
// a pas : on ne dit alors rien plutôt que d'inventer.
const PACING_LABELS = {
    standard: 'tempo standard, derrière Dédé',
    rapide: 'tempo rapide, derrière DouDou50',
};

const TEMPLATE = `
<div class="compte-page pt-page">
    <p class="pt-back"><a href="/compte">← Mes parties</a></p>
    <div id="pt-body"><div class="an-loading">Chargement…</div></div>
</div>`;

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

function fmtDate(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    return d.toLocaleDateString('fr-FR', { day: '2-digit', month: 'long', year: 'numeric' })
        + ` à ${d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}`;
}

/** « +7,4 » — un delta d'Elo se lit avec son signe, toujours. */
function fmtDelta(x) {
    const v = Math.round(x * 10) / 10;
    return (v > 0 ? '+' : '') + v.toString().replace('.', ',');
}

/** Le nom d'un siège : le pseudo pour un humain, la position pour un bot.
 *
 *  Même règle que `GameTable.playerName()` : les trois sièges IA d'une table
 *  sont tenus par le *même* bot, donc « Dédé » ne désigne personne. Le nom du
 *  bot ne survit qu'en qualificatif sur l'étiquette du siège. */
function seatLabel(seats, seat) {
    if (seat === null || seat === undefined) return '';
    const s = seats && seats[seat];
    if (!s) return SEAT_NAMES_FR[seat];
    return s.bot ? SEAT_NAMES_FR[seat] : s.name;
}

/** Le siège du lecteur, ou null.
 *
 *  Il se déduit des sièges nommés, pas de `matches.human_seat` : celui-ci
 *  désigne le joueur solo *propriétaire* de la partie, or n'importe qui peut
 *  ouvrir ce lien. Un anonyme, un joueur qui regarde la partie d'un autre et un
 *  invité de salon retombent donc tous les trois sur le repère neutre. */
function viewerSeat(seats, username) {
    if (!seats || !username) return null;
    const i = seats.findIndex(s => s && !s.bot && s.name === username);
    return i < 0 ? null : i;
}

function headHtml(m, mySeat) {
    const myTeam = mySeat === null ? null : mySeat % 2;
    let verdict;
    if (m.abandoned) {
        verdict = '<span class="pt-verdict pt-abandoned">Partie abandonnée</span>';
    } else if (m.winner === null || m.winner === undefined) {
        verdict = '';
    } else if (myTeam === null) {
        verdict = `<span class="pt-verdict">${TEAM_NAMES_FR[m.winner]} l'emporte</span>`;
    } else {
        const won = m.winner === myTeam;
        verdict = `<span class="pt-verdict ${won ? 'victory' : 'defeat'}">`
            + `${won ? 'Partie gagnée' : 'Partie perdue'}</span>`;
    }

    const scores = `<div class="pt-totals">
        <span class="team-ns${myTeam === 0 ? ' pt-mine' : ''}">${TEAM_NAMES_FR[0]}
            <b class="num">${m.points_ns}</b></span>
        <span class="pt-sep">—</span>
        <span class="team-ew${myTeam === 1 ? ' pt-mine' : ''}">${TEAM_NAMES_FR[1]}
            <b class="num">${m.points_ew}</b></span>
    </div>`;

    const deals = `${m.deals} donne${m.deals > 1 ? 's' : ''}`;
    const where = m.mode === 'multi' ? 'en salon' : 'en solo';
    const meta = [fmtDate(m.created_at), deals, where, PACING_LABELS[m.pacing]]
        .filter(Boolean).join(' · ');

    // L'Elo se note à la **partie** depuis la migration v14, et seulement en
    // 2000 points : c'est donc ici, et nulle part sur une donne, qu'une
    // variation a un sens.
    let elo = '';
    if (m.elo && m.elo.length) {
        elo = '<p class="pt-elo">' + m.elo.map(e =>
            `${esc(e.name || 'Joueur')} <b class="${e.delta >= 0 ? 'pt-up' : 'pt-down'}">`
            + `${fmtDelta(e.delta)}</b> Elo <span class="pt-elo-after">`
            + `(${Math.round(e.elo_after)})</span>`).join(' · ') + '</p>';
    } else if (m.target === 2000) {
        elo = '<p class="pt-elo pt-unrated">Partie non classée.</p>';
    } else {
        elo = `<p class="pt-elo pt-unrated">Les parties en ${m.target} ne sont pas
            classées — seul le 2000 l'est.</p>`;
    }

    const seats = (m.seats || []).map((s, i) =>
        `<span class="pt-seat${i === mySeat ? ' pt-mine' : ''}">${SEAT_NAMES_FR[i]}
            <b>${esc(s.bot ? botLabel(s.name) : s.name)}</b></span>`).join('');

    return `<section class="compte-card pt-head">
        <h2 class="compte-title">Partie en ${m.target} points</h2>
        ${verdict}
        ${scores}
        <p class="pt-meta">${esc(meta)}</p>
        ${seats ? `<div class="pt-seats">${seats}</div>` : ''}
        ${elo}
    </section>`;
}

/** Une cellule de score : ce que la donne a marqué, et le cumul en dessous. */
function scoreCell(pts, cum, cls) {
    const val = pts === null || pts === undefined ? '—' : pts;
    return `<td class="pt-score ${cls}">
        <span class="pt-pts num">${val}</span>
        <span class="pt-cum num">${cum}</span></td>`;
}

function rowHtml(g, mySeat, seats) {
    const contract = g.contract && g.contract.value;
    let label, outcome = '';
    if (!contract) {
        // Quatre passes : personne ne prend, personne ne marque. La ligne existe
        // quand même — elle a coûté une donne et fait tourner le donneur.
        label = '<span class="pt-passed">Donne passée</span>';
    } else {
        label = contractChipHtml(g.contract);
        const takerTeam = g.contract.team;
        const takerScore = takerTeam === 0 ? g.score_ns : g.score_ew;
        // « Contrat tenu ⟺ le camp preneur marque » : sous ce barème une chute
        // lui donne exactement 0, et un contrat réussi au moins 3V − 162. Même
        // équivalence que `stats.my_stats`, et c'est une propriété du barème,
        // pas une approximation.
        if (takerScore !== null && takerScore !== undefined) {
            outcome = takerScore > 0
                ? '<span class="pt-outcome pt-made">tenu</span>'
                : '<span class="pt-outcome pt-down-c">chuté</span>';
        }
    }

    // Le preneur est un **siège**, pas un camp : en solo trois sièges sur quatre
    // sont des bots, donc « Nord-Sud a pris » ne dit pas qui. Le camp reste le
    // repli quand le journal ne porte pas la phase de ses actions.
    const taker = contract
        ? (g.taker !== null && g.taker !== undefined
            ? `${esc(seatLabel(seats, g.taker))}${g.taker === mySeat ? ' <i>(vous)</i>' : ''}`
            : TEAM_NAMES_FR[g.contract.team])
        : '';
    const mineNs = mySeat !== null && mySeat % 2 === 0;
    const mineEw = mySeat !== null && mySeat % 2 === 1;

    return `<tr class="pt-row" data-game="${esc(g.id)}"
        title="Points cartes : ${g.points_ns} – ${g.points_ew} · donneur ${SEAT_NAMES_FR[g.dealer]}">
        <td class="pt-no"><a class="pt-open" href="/analyse/rejouer?game=${esc(g.id)}"
            >${g.deal_no || ''}</a></td>
        <td class="pt-contract">${label} ${outcome}
            <span class="pt-taker-inline">${taker}</span></td>
        <td class="pt-taker">${taker}</td>
        ${scoreCell(g.score_ns, g.total_ns, 'team-ns' + (mineNs ? ' pt-mine' : ''))}
        ${scoreCell(g.score_ew, g.total_ew, 'team-ew' + (mineEw ? ' pt-mine' : ''))}
    </tr>`;
}

/** Le cumul recalculé ne retombe pas toujours sur le score de la partie.
 *
 *  Trois causes réelles, et il faut dire laquelle : une donne pas encore
 *  rattrapée par `integrity.backfill_scores`, une donne mise en quarantaine
 *  après coup, et le **barème** — la ligne `matches` a été écrite au fil de la
 *  partie sous le barème du jour, alors que les scores par donne d'une vieille
 *  partie ont été rejoués sous le barème courant, qui a changé deux fois
 *  (2026-04-16, 2026-07-31). Le total de la partie fait foi : c'est lui qui a
 *  désigné le vainqueur et nourri l'Elo. */
function gapHtml(m) {
    if (m.sheet_ns === m.points_ns && m.sheet_ew === m.points_ew) return '';
    const causes = [];
    if (m.unscored_deals) {
        causes.push(`${m.unscored_deals} donne${m.unscored_deals > 1 ? 's' : ''} `
            + `sans score enregistré`);
    }
    if (m.invalid_deals) {
        causes.push(`${m.invalid_deals} donne${m.invalid_deals > 1 ? 's' : ''} `
            + `écartée${m.invalid_deals > 1 ? 's' : ''} pour incohérence`);
    }
    causes.push('un barème qui a changé depuis');
    return `<p class="pt-gap">Le cumul de la feuille
        (<b class="num">${m.sheet_ns} – ${m.sheet_ew}</b>) ne retombe pas sur le score
        de la partie (<b class="num">${m.points_ns} – ${m.points_ew}</b>) :
        ${causes.join(', ')}. C'est le score de la partie qui fait foi — c'est lui
        qui a désigné le vainqueur.</p>`;
}

/** L'en-tête d'une colonne de camp, en deux longueurs.
 *
 *  « Nord-Sud » écrit en toutes lettres fixe à lui seul la largeur de la
 *  colonne, et à 390px les deux colonnes de score sortaient de l'écran — or ce
 *  sont elles qu'on vient lire. « N-S » est de toute façon ce qu'on écrit sur
 *  une feuille de marque en papier. */
function teamTh(team) {
    return `<span class="pt-long">${TEAM_NAMES_FR[team]}</span>`
        + `<span class="pt-short">${TEAM_INITIALS_FR[team]}</span>`;
}

function sheetHtml(m, mySeat) {
    if (!m.games.length) {
        return `<section class="compte-card"><div class="history-empty">
            Aucune donne terminée dans cette partie.</div></section>`;
    }
    const mine = i => (mySeat !== null && mySeat % 2 === i ? ' pt-mine' : '');
    return `<section class="compte-card pt-sheet-card">
        <h3 class="compte-subtitle">Feuille de marque</h3>
        <div class="pt-scroll"><table class="pt-sheet">
            <thead><tr>
                <th class="pt-no">#</th>
                <th>Contrat</th>
                <th class="pt-taker">Preneur</th>
                <th class="pt-score team-ns${mine(0)}">${teamTh(0)}</th>
                <th class="pt-score team-ew${mine(1)}">${teamTh(1)}</th>
            </tr></thead>
            <tbody>${m.games.map(g => rowHtml(g, mySeat, m.seats)).join('')}</tbody>
        </table></div>
        <p class="pt-legend">Points <b>marqués</b> — le grand chiffre est ce que la
        donne a rapporté, le petit le cumul après elle. Cliquez une donne pour la
        rejouer carte par carte.</p>
        ${gapHtml(m)}
    </section>`;
}

export async function mount(container) {
    container.innerHTML = TEMPLATE;
    const body = document.getElementById('pt-body');
    const id = (new URLSearchParams(location.search).get('id') || '').trim();
    if (!id) {
        body.innerHTML = `<section class="compte-card"><div class="history-empty">
            Aucune partie demandée — vos parties sont sur
            <a href="/compte">votre compte</a>.</div></section>`;
        return;
    }

    let m;
    try {
        const resp = await fetch(`${base()}api/matches/${encodeURIComponent(id)}`);
        if (!resp.ok) throw new Error();
        m = await resp.json();
    } catch {
        body.innerHTML = `<section class="compte-card"><div class="history-empty">
            Partie introuvable. Seules les parties terminées ont une feuille de
            marque ; une partie en cours se reprend depuis
            <a href="/jouer">Jouer</a>.</div></section>`;
        return;
    }

    // Qui regarde : sans compte, ou en regardant la partie d'un autre, on lit la
    // feuille au repère neutre. Une erreur ici ne doit rien coûter de plus.
    let username = null;
    try {
        const meResp = await fetch(`${base()}api/me`);
        if (meResp.ok) username = (await meResp.json()).user?.username || null;
    } catch { /* anonyme */ }
    const mySeat = viewerSeat(m.seats, username);

    body.innerHTML = headHtml(m, mySeat) + sheetHtml(m, mySeat);

    for (const row of body.querySelectorAll('.pt-row')) {
        row.addEventListener('click', (e) => {
            // Une vraie ancre vit dans la première cellule : c'est elle qui
            // porte Ctrl+clic et le clic-milieu, et le routeur la gère déjà.
            // Ce gestionnaire n'est que le confort « toute la ligne clique ».
            if (e.target.closest('a')) return;
            if (e.button !== 0 || e.metaKey || e.ctrlKey || e.shiftKey || e.altKey) return;
            navigateTo(`/analyse/rejouer?game=${encodeURIComponent(row.dataset.game)}`);
        });
    }
}

export function unmount() {}
