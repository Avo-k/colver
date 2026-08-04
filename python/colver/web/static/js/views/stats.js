// Mes stats — le portrait chiffré d'un joueur.
//
// Deux moitiés, et la séparation est le sujet de la page.
//
// Le haut est **gratuit** : du SQL sur des colonnes déjà écrites. Combien de
// donnes, quelle série, à quelle fréquence vous prenez, si c'est plutôt vous ou
// votre partenaire, quelle couleur vous préférez. Ça s'affiche toujours et ça ne
// coûte rien.
//
// Le bas est **payé** : une recherche double-mort par décision. Il ne se calcule
// que si le joueur appuie sur le bouton, comme la demande d'analyse de lichess.
// Rien ne se déclenche au chargement — un GET qui lancerait des solves partirait
// au premier préchargeur de navigateur.
//
// Règle d'affichage qui traverse tout le fichier : un taux ne se montre jamais
// sans son n et son intervalle, et pas du tout sous cinq observations. Sur
// quelques dizaines de donnes, « 62 % » sans rien d'autre est une affirmation
// que les données ne soutiennent pas.

const base = () => document.querySelector('base')?.getAttribute('href') || '/';

// Sous ce seuil, l'intervalle de Wilson couvre à peu près tout [0, 100] : le
// chiffre n'informe sur rien, et sa seule fonction serait de se faire lire
// comme une mesure.
const MIN_OBS = 5;

const CAT_LABELS = {
    parfait: 'Sans perte',
    bon: 'Bon',
    imprecision: 'Imprécision',
    erreur: 'Erreur',
    faute: 'Faute',
};
const CAT_ORDER = ['parfait', 'bon', 'imprecision', 'erreur', 'faute'];

const TEMPLATE = `
<div class="compte-page stats-page">
    <div class="compte-card">
        <h2 class="compte-title">Mes statistiques</h2>
        <div id="stats-body"><div class="an-loading">Chargement…</div></div>
    </div>
    <div class="compte-card" id="stats-oracle-card">
        <h3 class="compte-subtitle">Face à l'oracle</h3>
        <div id="stats-oracle"><div class="an-loading">Chargement…</div></div>
    </div>
</div>`;

let poller = null;

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

// ---- Briques d'affichage ---------------------------------------------------

function row(label, value, hint) {
    if (value === null || value === undefined || value === '') return '';
    return '<div class="cs-row">' +
        `<span class="cs-label">${label}</span>` +
        `<span class="cs-val">${value}</span>` +
        `<span class="cs-hint">${hint || ''}</span></div>`;
}

/** Un taux avec son intervalle, ou rien du tout s'il repose sur trop peu. */
function pct(label, blob, unit = '') {
    if (!blob || !blob.n) return '';
    if (blob.n < MIN_OBS) return row(label, '—', `seulement ${blob.n} obs.`);
    return row(label, `${blob.pct} %`,
        `n = ${blob.n}${unit} · IC95 ${blob.lo}–${blob.hi} %`);
}

/** Une barre proportionnelle : plus lisible qu'une colonne de pourcentages. */
function bars(segments) {
    const total = segments.reduce((s, x) => s + x.n, 0);
    if (!total) return '';
    const bar = segments.filter(s => s.n).map(s =>
        `<span class="cs-bar-seg cs-bar-${s.key}" style="flex:${s.n}" ` +
        `title="${esc(s.label)} : ${s.n}"></span>`).join('');
    const legend = segments.filter(s => s.n).map(s =>
        `<span class="cs-legend"><i class="cs-bar-${s.key}"></i>${esc(s.label)} ` +
        `<b>${Math.round(100 * s.n / total)} %</b></span>`).join('');
    return `<div class="cs-bar">${bar}</div><div class="cs-legends">${legend}</div>`;
}

function section(title, inner) {
    const body = inner.filter(Boolean).join('');
    return body ? `<h4 class="compte-section-title">${title}</h4>${body}` : '';
}

// ---- La moitié gratuite ----------------------------------------------------

function renderFree(st) {
    if (!st || !st.deals) {
        return '<div class="history-empty">Aucune donne terminée pour l\'instant — ' +
            '<a href="/jouer/humain">jouez-en une !</a></div>';
    }
    const out = [];

    out.push(section('Au compteur', [
        row('Donnes jouées', st.deals,
            st.modes && st.modes.multi ? `dont ${st.modes.multi} en salon` : 'en solo'),
        row('Jours de jeu', st.days,
            st.density ? `${st.density} donnes par jour joué` : ''),
        row('Série en cours', st.streak ? `${st.streak} jour${st.streak > 1 ? 's' : ''}` : '—',
            st.streak ? 'jours consécutifs' : 'reprenez aujourd\'hui pour la relancer'),
    ]));

    out.push(section('Résultats', [
        pct('Donnes gagnées', st.won),
        st.margin && st.margin.n >= MIN_OBS
            ? row('Écart moyen par donne',
                `${st.margin.mean > 0 ? '+' : ''}${st.margin.mean}`,
                `n = ${st.margin.n} · ±${st.margin.ci} · médiane ${st.margin.median}`)
            : '',
        st.deals !== st.scored
            ? `<p class="compte-hint">${st.deals - st.scored} donne(s) en attente de ` +
              'calcul, exclues des taux ci-dessus.</p>'
            : '',
    ]));

    // Qui prend : la dynamique la plus parlante de la page. Un joueur qui ne
    // prend jamais et dont le partenaire prend tout ne joue pas le même jeu que
    // celui qui prend une fois sur trois.
    const w = st.who_takes || {};
    out.push(section('Qui prend', [
        w.n ? bars([
            { key: 'me', label: 'Vous', n: w.me },
            { key: 'partner', label: 'Votre partenaire', n: w.partner },
            { key: 'opp', label: 'Les adversaires', n: w.opponents },
        ]) : '',
        w.n ? row('Sur les donnes contractées', w.n,
            w.passed ? `${w.passed} donne(s) passée(s) en plus` : '') : '',
    ]));

    const t = st.takes || {};
    const trumps = (t.trumps || []).filter(x => x.n);
    out.push(section('Quand vous prenez', [
        t.n ? row('Contrats pris', t.n,
            `${t.per_100} pour 100 donnes · hauteur moyenne ${t.avg_value ?? '—'}`) : '',
        pct('Contrats tenus', t.held, ' prises'),
        trumps.length ? row('Atout préféré',
            trumps.slice().sort((a, b) => b.n - a.n)[0].suit,
            trumps.map(x => `${x.suit} ${x.n}`).join(' · ')) : '',
        st.bidding && st.bidding.capots
            ? row('Capots annoncés', st.bidding.capots, '') : '',
    ]));

    const b = st.bidding || {};
    out.push(section('À l\'enchère', [
        pct('Vous passez', b.pass, ' décisions'),
        b.height_n >= MIN_OBS
            ? row('Hauteur annoncée', b.avg_height,
                `moyenne sur ${b.height_n} annonces`) : '',
        // Le trio prises / hauteur / tenus se lit ensemble : un taux de
        // contrats tenus se maximise en n'annonçant plus que 80.
        (t.n >= MIN_OBS && b.height_n >= MIN_OBS)
            ? '<p class="compte-hint">Un taux de contrats tenus très haut avec une ' +
              'hauteur basse décrit surtout de la prudence — les deux se lisent ensemble.</p>'
            : '',
    ]));

    out.push(section('Défense et faits d\'armes', [
        pct('Chutes infligées', st.defense, ' défenses'),
        st.capots_for ? row('Capots réalisés', st.capots_for, '') : '',
        st.capots_against ? row('Capots subis', st.capots_against, '') : '',
        st.belotes ? row('Belotes en main', st.belotes,
            `sur ${st.deals} donnes`) : '',
        st.coinches ? row('Coinches annoncées', st.coinches, '') : '',
        st.surcoinches ? row('Surcoinches', st.surcoinches, '') : '',
        st.contres_played ? row('Donnes contrées', st.contres_played,
            'tous camps confondus') : '',
    ]));

    const partners = st.partners || [];
    out.push(section('Vos partenaires', [
        partners.length
            ? partners.map(p => row(esc(p.name), p.deals, 'donnes ensemble')).join('')
            : '<p class="compte-hint">Vous n\'avez encore joué qu\'avec des bots. ' +
              'En <a href="/jouer/salon">salon</a>, votre partenaire apparaîtra ici.</p>',
    ]));

    const tempo = st.tempo || {};
    if (tempo.n >= MIN_OBS) {
        const dropped = tempo.dropped
            ? ` · ${tempo.dropped} interruption${tempo.dropped > 1 ? 's' : ''} exclue${tempo.dropped > 1 ? 's' : ''}`
            : '';
        out.push(section('Tempo', [
            row('Temps par donne', `${tempo.median} s`,
                `médiane sur ${tempo.n} donnes en partie · ${tempo.p25}–${tempo.p75} s${dropped}`),
        ]));
    }

    return `<div class="compte-stats-list">${out.filter(Boolean).join('')}</div>`;
}

// ---- La moitié payée -------------------------------------------------------

function renderOracle(o) {
    if (!o || !o.total) {
        return '<p class="compte-hint">Rien à analyser pour l\'instant : ' +
            'jouez quelques donnes d\'abord.</p>';
    }
    const out = [];
    const job = o.job;

    // La couverture d'abord, toujours. Sans elle, un joueur qui n'a analysé
    // qu'une poignée de donnes lirait une moyenne sans savoir sur quoi elle
    // porte — et le choix des donnes analysées lui appartient.
    const pctDone = Math.round(100 * o.analysed / o.total);
    out.push(`<div class="cs-cover"><div class="cs-cover-bar">` +
        `<span style="width:${pctDone}%"></span></div>` +
        `<span class="cs-cover-txt">${o.analysed} / ${o.total} donnes analysées ` +
        `(${pctDone} %)</span></div>`);

    if (job && job.running) {
        const p = job.total ? Math.round(100 * job.done / job.total) : 0;
        out.push(`<p class="compte-hint">Analyse en cours — ${job.done} / ${job.total} ` +
            `(${p} %). Vous pouvez quitter la page, elle continue.</p>`);
    } else if (o.pending > 0) {
        out.push(`<button id="stats-analyse" class="compte-submit">` +
            `Analyser mes ${o.pending} donne${o.pending > 1 ? 's' : ''} restante${o.pending > 1 ? 's' : ''}</button>`);
        out.push('<p class="compte-hint">Chaque donne passe au solveur double-mort, ' +
            'qui calcule le meilleur coup possible à chaque décision. Comptez ' +
            'environ un quart de seconde par donne. Le résultat sert aussi à ' +
            '<a href="/analyse/rejouer">Rejouer</a>, qui devient instantané.</p>');
    }
    if (job && !job.running && job.errors) {
        out.push(`<p class="compte-hint compte-hint-warn">${job.errors} donne(s) ` +
            'n\'ont pas pu être analysées.</p>');
    }

    if (!o.decisions) {
        return out.join('');
    }

    const counts = o.counts || {};
    const segments = CAT_ORDER
        .map(k => ({ key: k, label: CAT_LABELS[k], n: counts[k] || 0 }))
        .filter(s => s.n);

    out.push('<div class="compte-stats-list">');
    out.push(row('Points perdus par décision', o.avg_cost,
        `n = ${o.decisions} décisions${o.cost_ci ? ` · ±${o.cost_ci}` : ''}`));
    out.push(pct('Décisions sans perte', o.clean, ' décisions'));
    out.push(row('Coups sans choix', o.forced,
        'une seule carte jouable — hors du calcul'));
    out.push('</div>');
    out.push(bars(segments));
    out.push('<p class="compte-hint">« Sans perte » veut dire que le solveur ne ' +
        'valorise aucune autre carte plus haut — pas que vous avez joué sa carte ' +
        'préférée : plus d\'une position sur deux a plusieurs cartes également ' +
        'bonnes. Les coups forcés sont exclus, sinon le taux gonflerait d\'un ' +
        'tiers sans rien dire de vous.</p>');
    return out.join('');
}

// ---- Chargement ------------------------------------------------------------

async function loadOracle() {
    const box = document.getElementById('stats-oracle');
    if (!box) return null;
    let o = null;
    try {
        const resp = await fetch(`${base()}api/me/oracle`);
        o = resp.ok ? await resp.json() : null;
        box.innerHTML = renderOracle(o);
    } catch {
        box.innerHTML = '<div class="an-loading">Indisponible</div>';
        return null;
    }
    const btn = document.getElementById('stats-analyse');
    if (btn) {
        btn.addEventListener('click', async () => {
            btn.disabled = true;
            btn.textContent = 'Lancement…';
            try {
                await fetch(`${base()}api/me/oracle`, { method: 'POST' });
            } catch { /* le sondage rattrapera */ }
            startPolling();
        });
    }
    return o;
}

/** Sonder tant qu'un balayage tourne. Une seconde : le travail se compte en
 *  centaines de millisecondes par donne, et la page doit donner signe de vie. */
function startPolling() {
    stopPolling();
    poller = setInterval(async () => {
        const o = await loadOracle();
        if (!o || !o.job || !o.job.running) stopPolling();
    }, 1000);
    loadOracle();
}

function stopPolling() {
    if (poller) { clearInterval(poller); poller = null; }
}

export async function mount(container) {
    container.innerHTML = TEMPLATE;
    const body = document.getElementById('stats-body');

    let resp;
    try {
        resp = await fetch(`${base()}api/me/stats`);
    } catch {
        body.innerHTML = '<div class="an-loading">Statistiques indisponibles</div>';
        return;
    }
    if (resp.status === 401) {
        // Pas de compte, pas de stats : elles n'existent que rattachées à
        // quelqu'un. On le dit et on propose la porte d'entrée.
        body.innerHTML = '<div class="history-empty">Ces statistiques sont celles ' +
            'de votre compte — <a href="/compte">connectez-vous</a> pour les voir.</div>';
        document.getElementById('stats-oracle-card').classList.add('hidden');
        return;
    }
    body.innerHTML = renderFree(await resp.json());

    const o = await loadOracle();
    if (o && o.job && o.job.running) startPolling();
}

export function unmount() {
    stopPolling();
}
