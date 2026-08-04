// Mes stats — le portrait chiffré d'un joueur, en profil deux colonnes.
//
// Le rail de gauche est **qui vous êtes** : pseudo, classement, compteurs,
// activité, partenaires, faits d'armes. Il reste à l'écran (sticky) pendant
// qu'on parcourt l'analyse à droite, qui est **comment vous jouez**. Sous
// 900 px il passe simplement au-dessus.
//
// La page se divise aussi en deux moitiés d'un autre genre, et c'est celle-là
// qui compte vraiment :
//
// - tout ce qui vient de `/api/me/stats` est **gratuit** — du SQL sur des
//   colonnes déjà écrites. Ça s'affiche toujours, sans attente.
// - le bloc « Face à l'oracle » est **payé** : une recherche double-mort par
//   décision. Il ne se calcule que si le joueur appuie sur le bouton, comme la
//   demande d'analyse de lichess. Rien ne se déclenche au chargement — un GET
//   qui lancerait des solves partirait au premier préchargeur de navigateur.
//
// Règle d'affichage qui traverse tout le fichier : un taux ne se montre jamais
// sans son n et son intervalle, et pas du tout sous cinq observations. Sur
// quelques dizaines de donnes, « 62 % » tout seul est une affirmation que les
// données ne soutiennent pas — d'où les intervalles *dessinés* (`ciRow`) : la
// barre est la précision, et on voit d'un coup d'œil qu'un chiffre ne vaut rien.

const base = () => document.querySelector('base')?.getAttribute('href') || '/';

// Sous ce seuil, l'intervalle de Wilson couvre à peu près tout [0, 100].
const MIN_OBS = 5;

const CAT_LABELS = {
    parfait: 'Sans perte', bon: 'Bon', imprecision: 'Imprécision',
    erreur: 'Erreur', faute: 'Faute',
};
const CAT_ORDER = ['parfait', 'bon', 'imprecision', 'erreur', 'faute'];
const WEEKDAYS = ['lundi', 'mardi', 'mercredi', 'jeudi', 'vendredi', 'samedi', 'dimanche'];

// Le filtre de format. « Toutes » est le défaut et le reste : c'est le seul
// périmètre qui ne demande rien à comprendre. Les trois autres répondent à des
// questions différentes — on n'annonce pas pareil en tournoi qu'en donne seule,
// et le score de partie entre dans l'observation du bidder.
const SCOPES = [
    { id: 'all', label: 'Toutes' },
    { id: '2000', label: 'Parties 2000' },
    { id: '1000', label: 'Parties 1000' },
    { id: 'deal', label: 'Donnes seules' },
];

const TEMPLATE = `
<div class="st-page">
    <div class="st-filter" id="st-filter" role="radiogroup" aria-label="Format de jeu">
        ${SCOPES.map(s => `<button class="pc-seg-btn" role="radio" data-scope="${s.id}"
            aria-checked="${s.id === 'all'}">${s.label}</button>`).join('')}
    </div>
    <aside class="st-rail" id="st-rail"></aside>
    <main class="st-main" id="st-main">
        <div class="an-loading">Chargement…</div>
    </main>
</div>`;

let poller = null;
let scope = 'all';

/** Le périmètre vit dans la query string : un rechargement, un signet ou un
 *  lien partagé retombent sur la même vue. Jamais dans un `#fragment` — le
 *  routeur le traite comme une URL héritée et le redirigerait. */
function syncUrl() {
    const url = new URL(location.href);
    if (scope === 'all') url.searchParams.delete('f');
    else url.searchParams.set('f', scope);
    history.replaceState(history.state, '', url);
}

function qs() { return scope === 'all' ? '' : `?scope=${scope}`; }

function esc(s) {
    return String(s).replace(/[&<>"']/g, c => ({
        '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;',
    }[c]));
}

// ---- Briques ---------------------------------------------------------------

function card(title, inner, cls = '') {
    const body = inner.filter(Boolean).join('');
    if (!body) return '';
    return `<section class="st-card ${cls}">` +
        (title ? `<h3 class="st-h">${title}</h3>` : '') + body + '</section>';
}

/** Un chiffre et son libellé. `tone` : 'gold' pour l'accent, 'neg' pour un écart négatif. */
function stat(v, k, n, tone = '', size = '') {
    if (v === null || v === undefined || v === '') return '';
    return `<div class="st-stat">
        <span class="st-v ${tone} ${size} num">${v}</span>
        <span class="st-k">${k}</span>
        ${n ? `<span class="st-n num">${n}</span>` : ''}</div>`;
}

/** Un taux, avec son intervalle **dessiné**. La barre est la précision. */
function ciRow(label, b, unit = '') {
    if (!b || !b.n) return '';
    if (b.n < MIN_OBS) {
        return `<div class="st-stat"><span class="st-v sm">—</span>
            <span class="st-k">${label}</span>
            <span class="st-n">seulement ${b.n} observation${b.n > 1 ? 's' : ''}</span></div>`;
    }
    return `<div class="st-stat" title="Intervalle de confiance à 95 % : ${b.lo}–${b.hi} %">
        <span class="st-v gold num">${b.pct} %</span>
        <span class="st-k">${label}</span>
        <div class="st-ci"><span class="st-ci-span" style="left:${b.lo}%;width:${b.hi - b.lo}%"></span>
            <span class="st-ci-dot" style="left:${b.pct}%"></span></div>
        <span class="st-n num">n = ${b.n}${unit} · IC95 ${b.lo}–${b.hi} %</span></div>`;
}

/** Barre segmentée + légende directe : l'identité n'est jamais portée par la
 *  couleur seule. */
function segbar(segs) {
    const tot = segs.reduce((s, x) => s + x.n, 0);
    if (!tot) return '';
    const pct = n => Math.round(100 * n / tot);
    return `<div class="st-bar">${segs.map(s =>
        `<span style="flex:${s.n};background:var(${s.c})" title="${esc(s.k)} : ${s.n}"></span>`).join('')}</div>
    <div class="st-legend">${segs.map(s =>
        `<span><i style="background:var(${s.c})"></i>${esc(s.k)} <b class="num">${pct(s.n)} %</b></span>`).join('')}</div>`;
}

/** Barres par catégorie. */
function bars(items, { wide = false, fill = '--c-series-you' } = {}) {
    if (!items.length) return '';
    const max = Math.max(...items.map(i => i.n), 1);
    return `<div class="st-rows">${items.map(i =>
        `<div class="st-row ${wide ? 'wide' : ''}" title="${esc(i.tip || i.lab)}">
            <span class="st-row-lab ${i.cls || ''}">${i.lab}</span>
            <span class="st-track"><span class="st-fill" style="width:${Math.max(3, 100 * i.n / max)}%;background:var(${i.fill || fill})"></span></span>
            <span class="st-row-val num">${i.n}</span></div>`).join('')}</div>`;
}

/** Calendrier d'activité. Le seul objet de la page qui montre le *temps*. */
function heatmap(a) {
    if (!a || !a.days) return '';
    const start = new Date(a.start + 'T00:00:00Z');
    const cells = [];
    for (let w = 0; w < a.weeks; w++) {
        let col = '';
        for (let d = 0; d < 7; d++) {
            const i = w * 7 + d, v = a.days[i];
            if (v === null || v === undefined) { col += '<i class="st-cell void"></i>'; continue; }
            // Quatre paliers relatifs au meilleur jour : une échelle absolue
            // laisserait la grille éteinte pour qui joue trois donnes par jour.
            const lvl = v === 0 ? 0 : Math.min(4, Math.ceil(4 * v / Math.max(a.max, 1)));
            const day = new Date(start.getTime() + i * 86400000);
            const txt = day.toLocaleDateString('fr-FR', { day: 'numeric', month: 'long' });
            col += `<i class="st-cell" data-l="${lvl}" title="${WEEKDAYS[d]} ${txt} — ${
                v ? `${v} donne${v > 1 ? 's' : ''}` : 'aucune donne'}"></i>`;
        }
        cells.push(`<div class="st-heat-col">${col}</div>`);
    }
    return `<div class="st-heat-wrap"><div class="st-heat">${cells.join('')}</div>
        <div class="st-heat-foot"><span>${a.weeks} dernières semaines</span>
        <span class="st-heat-key">moins ${[0, 1, 2, 3, 4]
            .map(l => `<i class="st-cell" data-l="${l}"></i>`).join('')} plus</span></div></div>`;
}

/** Anneau de progression : une proportion unique, sans axe. */
function ring(pct, val, key, note) {
    const r = 30, c = 2 * Math.PI * r, on = c * Math.max(0, Math.min(100, pct)) / 100;
    return `<div class="st-ring">
        <svg width="76" height="76" viewBox="0 0 76 76" aria-hidden="true">
            <circle cx="38" cy="38" r="${r}" fill="none" stroke="var(--c-surface-2)" stroke-width="8"/>
            <circle cx="38" cy="38" r="${r}" fill="none" stroke="var(--c-q1)" stroke-width="8"
                stroke-linecap="round" stroke-dasharray="${on} ${c - on}" transform="rotate(-90 38 38)"/>
        </svg>
        <div><div class="st-ring-v num">${val}</div>
        <div class="st-ring-k">${key}</div>
        <div class="st-ring-n num">${note}</div></div></div>`;
}

// ---- Le rail : qui vous êtes -----------------------------------------------

function renderRail(me, st) {
    const elo = me && me.stats && me.stats.elo;
    const since = me && me.user ? new Date(me.user.created_at)
        .toLocaleDateString('fr-FR', { month: 'long', year: 'numeric' }) : null;

    const identity = `<div class="st-id">
        <span class="st-name">${esc(me.user.username)}</span>
        <span class="st-since">${elo && elo.games
            ? `Classement <b class="num">${Math.round(elo.elo)}</b>${elo.ranked ? '' : ' (provisoire)'} · `
            : ''}membre depuis ${since}</span>
        <div class="st-mini">
            ${stat(st.deals, 'donnes', '', 'gold', 'sm')}
            ${stat(st.days, 'jours', '', '', 'sm')}
            ${stat(st.streak || '—', 'série', '', '', 'sm')}
        </div></div>`;

    const arms = [
        ['Capots réussis', st.capots_for], ['Capots subis', st.capots_against],
        ['Belotes', st.belotes], ['Coinches', st.coinches],
        ['Surcoinches', st.surcoinches], ['Donnes contrées', st.contres_played],
    ].filter(([, v]) => v);

    return card('', [identity], 'st-card-id')
        + card('Activité', [heatmap(st.activity)])
        + card('Partenaires', [
            st.partners.length
                ? bars(st.partners.map(p => ({
                    lab: esc(p.name), n: p.deals,
                    tip: `${p.deals} donnes avec ${p.name}`,
                })), { wide: true, fill: '--c-series-mate' })
                : '',
            st.partners.length
                ? '<p class="st-hint">En salon. En solo, votre partenaire est un bot.</p>'
                : '<p class="st-hint">Vous n\'avez encore joué qu\'avec des bots. '
                  + 'En <a href="/jouer/salon">salon</a>, votre partenaire apparaîtra ici.</p>',
        ])
        + card("Faits d'armes", [
            arms.length
                ? `<div class="st-chips">${arms.map(([k, v]) =>
                    `<span class="st-chip">${k} <b class="num">${v}</b></span>`).join('')}</div>`
                : '<p class="st-hint">Ni capot, ni belote, ni coinche pour l\'instant.</p>',
        ]);
}

// ---- Le corps : comment vous jouez -----------------------------------------

function renderMain(st) {
    const t = st.takes || {};
    const b = st.bidding || {};
    const w = st.who_takes || {};
    const tempo = st.tempo || {};
    const trumps = (t.trumps || []).filter(x => x.n);

    const results = card('Résultats', [
        ciRow('donnes gagnées', st.won),
        st.margin && st.margin.n >= MIN_OBS
            ? stat(`${st.margin.mean > 0 ? '+' : ''}${st.margin.mean}`, 'écart moyen par donne',
                `n = ${st.margin.n} · ±${st.margin.ci} · médiane ${st.margin.median}`,
                st.margin.mean < 0 ? 'neg' : '', 'sm')
            : '',
        st.deals !== st.scored
            ? `<p class="st-hint">${st.deals - st.scored} donne(s) en attente de calcul, `
              + 'exclues des taux ci-dessus.</p>'
            : '',
    ]);

    const who = card('Qui prend', [
        w.n ? segbar([
            { k: 'Vous', n: w.me, c: '--c-series-you' },
            { k: 'Votre partenaire', n: w.partner, c: '--c-series-mate' },
            { k: 'Les adversaires', n: w.opponents, c: '--c-series-rest' },
        ]) : '<p class="st-hint">Aucune donne contractée pour l\'instant.</p>',
        w.n ? `<p class="st-hint">Sur ${w.n} donnes contractées`
            + (w.passed ? `, ${w.passed} passée${w.passed > 1 ? 's' : ''} en plus` : '')
            + '. En solo, trois sièges sur quatre sont des bots : « mon camp a pris » '
            + 'ne dit pas « j\'ai pris ».</p>' : '',
    ]);

    const taking = card('Quand vous prenez', [
        t.n ? stat(t.n, 'contrats pris',
            `${t.per_100} pour 100 donnes · hauteur moyenne ${t.avg_value ?? '—'}`, '', 'sm') : '',
        ciRow('contrats tenus', t.held, ' prises'),
        trumps.length ? '<h4 class="st-h4">Atout</h4>' + bars(trumps.map(x => ({
            lab: x.suit, n: x.n,
            cls: (x.suit === '♥' || x.suit === '♦') ? 'st-suit-r' : 'st-suit-b',
            tip: `${x.n} contrat${x.n > 1 ? 's' : ''} pris à ${x.suit}`,
        }))) : '',
        b.capots ? stat(b.capots, 'capots annoncés', '', '', 'sm') : '',
    ]);

    const bidding = card('À l\'enchère & en défense', [
        ciRow('vous passez', b.pass, ' décisions'),
        b.height_n >= MIN_OBS
            ? stat(b.avg_height, 'hauteur annoncée', `moyenne sur ${b.height_n} annonces`, '', 'sm')
            : '',
        ciRow('chutes infligées', st.defense, ' défenses'),
        (t.n >= MIN_OBS && b.height_n >= MIN_OBS)
            ? '<p class="st-hint">Un taux de contrats tenus très haut avec une hauteur '
              + 'basse décrit surtout de la prudence — les deux se lisent ensemble.</p>'
            : '',
    ]);

    const pace = tempo.n >= MIN_OBS ? card('Tempo', [
        stat(`${tempo.median} s`, 'temps par donne',
            `médiane sur ${tempo.n} donnes en partie · ${tempo.p25}–${tempo.p75} s`
            + (tempo.dropped ? ` · ${tempo.dropped} interruption${tempo.dropped > 1 ? 's' : ''} exclue${tempo.dropped > 1 ? 's' : ''}` : ''),
            '', 'sm'),
    ]) : '';

    return `<div class="st-duo">${results}${who}</div>`
        + `<div class="st-duo">${taking}${bidding}</div>`
        + '<section class="st-card" id="st-oracle-card">'
        + '<h3 class="st-h">Face à l\'oracle</h3>'
        + '<div id="st-oracle"><div class="an-loading">Chargement…</div></div></section>'
        + pace;
}

// ---- Le bloc payé ----------------------------------------------------------

function renderOracle(o) {
    if (!o || !o.total) {
        return '<p class="st-hint">Rien à analyser pour l\'instant : jouez quelques donnes d\'abord.</p>';
    }
    const out = [];
    const job = o.job;
    const cov = Math.round(100 * o.analysed / o.total);

    // La couverture d'abord, toujours : c'est le joueur qui choisit quand
    // analyser, donc une moyenne calculée sur un dixième de ses donnes doit se
    // lire comme telle.
    out.push(`<div class="st-cover"><div class="st-cover-bar"><span style="width:${cov}%"></span></div>
        <span class="st-n num">${o.analysed} / ${o.total} donnes analysées (${cov} %)</span></div>`);

    if (job && job.running) {
        const p = job.total ? Math.round(100 * job.done / job.total) : 0;
        out.push(`<p class="st-hint">Analyse en cours — ${job.done} / ${job.total} (${p} %). `
            + 'Vous pouvez quitter la page, elle continue.</p>');
    } else if (o.pending > 0) {
        out.push(`<button id="st-analyse" class="compte-submit">Analyser mes ${o.pending} `
            + `donne${o.pending > 1 ? 's' : ''} restante${o.pending > 1 ? 's' : ''}</button>`);
        out.push('<p class="st-hint">Chaque donne passe au solveur double-mort, qui rejoue '
            + 'chaque décision en voyant les quatre mains. Comptez environ un quart de '
            + 'seconde par donne. Le résultat sert aussi à '
            + '<a href="/analyse/rejouer">Rejouer</a>, qui devient instantané.</p>');
    }
    if (job && !job.running && job.errors) {
        out.push(`<p class="st-hint compte-hint-warn">${job.errors} donne(s) n'ont pas pu être analysées.</p>`);
    }
    if (!o.decisions) return out.join('');

    const counts = o.counts || {};
    const segs = CAT_ORDER.map((k, i) => ({ k: CAT_LABELS[k], n: counts[k] || 0, c: `--c-q${i + 1}` }))
        .filter(s => s.n);

    // Pas de barre pour « points perdus » : une barre a besoin d'une échelle que
    // le lecteur comprenne, et il n'y en a pas ici (perdre 2 points par décision,
    // c'est beaucoup ou peu ?). Une barre à un seul item est en plus toujours
    // pleine — elle aurait l'air d'un maximum atteint.
    out.push('<div class="st-oracle-split">');
    out.push('<div>' + ring(o.clean.pct, `${o.clean.pct} %`, 'décisions sans perte',
        `n = ${o.clean.n} · IC95 ${o.clean.lo}–${o.clean.hi} %`) + '</div>');
    out.push('<div>' + segbar(segs)
        + stat(o.avg_cost, 'points perdus par décision',
            `sur ${o.decisions} décisions${o.cost_ci ? ` · ±${o.cost_ci}` : ''}`, '', 'sm')
        + `<p class="st-hint">${o.forced} coups n'offraient qu'une carte : ils sont `
        + 'hors du calcul, sinon le taux gonflerait sans rien dire de vous.</p></div>');
    out.push('</div>');
    out.push('<p class="st-hint">« Sans perte » veut dire que le solveur ne valorise aucune '
        + 'autre carte plus haut — pas que vous avez joué sa carte préférée : plus d\'une '
        + 'position sur deux a plusieurs cartes également bonnes.</p>');
    return out.join('');
}

// ---- Chargement ------------------------------------------------------------

async function loadOracle() {
    const box = document.getElementById('st-oracle');
    if (!box) return null;
    let o = null;
    try {
        const resp = await fetch(`${base()}api/me/oracle${qs()}`);
        o = resp.ok ? await resp.json() : null;
        box.innerHTML = renderOracle(o);
    } catch {
        box.innerHTML = '<p class="st-hint">Indisponible.</p>';
        return null;
    }
    const btn = document.getElementById('st-analyse');
    if (btn) {
        btn.addEventListener('click', async () => {
            btn.disabled = true;
            btn.textContent = 'Lancement…';
            try { await fetch(`${base()}api/me/oracle${qs()}`, { method: 'POST' }); }
            catch { /* le sondage rattrapera */ }
            startPolling();
        });
    }
    return o;
}

/** Sonder tant qu'un balayage tourne : le travail se compte en centaines de
 *  millisecondes par donne, et la page doit donner signe de vie. */
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

let cachedMe = null;

/** Charger et rendre pour le périmètre courant. Le filtre, lui, ne bouge pas :
 *  il doit rester atteignable même quand un format ne contient aucune donne,
 *  sinon on s'y enferme. */
async function load() {
    const rail = document.getElementById('st-rail');
    const main = document.getElementById('st-main');
    if (!main) return;
    stopPolling();
    main.innerHTML = '<div class="an-loading">Chargement…</div>';
    rail.innerHTML = '';

    let st;
    try {
        const [rs, rm] = await Promise.all([
            fetch(`${base()}api/me/stats${qs()}`),
            cachedMe ? Promise.resolve(null) : fetch(`${base()}api/me`),
        ]);
        if (rs.status === 401) {
            // Ces chiffres n'existent que rattachés à quelqu'un.
            document.getElementById('st-filter').classList.add('hidden');
            main.innerHTML = '<div class="history-empty">Ces statistiques sont celles de '
                + 'votre compte — <a href="/compte">connectez-vous</a> pour les voir.</div>';
            return;
        }
        st = await rs.json();
        if (rm && rm.ok) cachedMe = await rm.json();
    } catch {
        main.innerHTML = '<div class="an-loading">Statistiques indisponibles</div>';
        return;
    }

    if (!st || !st.deals) {
        main.innerHTML = '<div class="history-empty">'
            + (scope === 'all'
                ? 'Aucune donne terminée pour l\'instant — <a href="/jouer/humain">jouez-en une !</a>'
                : 'Aucune donne dans ce format. Choisissez « Toutes » pour tout revoir.')
            + '</div>';
        return;
    }

    if (cachedMe && cachedMe.user) rail.innerHTML = renderRail(cachedMe, st);
    main.innerHTML = renderMain(st);

    const o = await loadOracle();
    if (o && o.job && o.job.running) startPolling();
}

export async function mount(container) {
    container.innerHTML = TEMPLATE;
    cachedMe = null;

    const wanted = new URL(location.href).searchParams.get('f');
    scope = SCOPES.some(x => x.id === wanted) ? wanted : 'all';

    const bar = document.getElementById('st-filter');
    bar.querySelectorAll('[data-scope]').forEach(b => {
        b.setAttribute('aria-checked', String(b.dataset.scope === scope));
        b.addEventListener('click', () => {
            if (b.dataset.scope === scope) return;
            scope = b.dataset.scope;
            bar.querySelectorAll('[data-scope]').forEach(x =>
                x.setAttribute('aria-checked', String(x.dataset.scope === scope)));
            syncUrl();
            load();
        });
    });
    syncUrl();
    await load();
}

export function unmount() {
    stopPolling();
}
