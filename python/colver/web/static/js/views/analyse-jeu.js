// Analyse du jeu de la carte — une décision, pesée contre les mondes que le
// joueur ne voyait pas.
//
// La page annonces demande « que vaut cette main ? » ; celle-ci descend d'un
// cran : « à cette position, que valait chaque carte jouable ? ». Deux
// questions distinctes cohabitent et ne doivent jamais être fusionnées —
// l'Oracle sur la vraie donne (exact, information parfaite, déjà dans Rejouer)
// et l'Oracle sur les mondes de l'information set (ce que ce siège pouvait
// savoir). Une carte deuxième dans la vraie donne mais meilleure dans 70 % des
// mondes était un bon choix contre de la malchance ; confondre les deux
// colonnes cacherait exactement ça.
//
// L'état de la page est le CFN complet 4 sections + un index d'action, donc
// l'URL est partageable et le lien depuis Rejouer n'a rien à recalculer.

import { send, onMessage, offMessage, onOpen, offOpen } from '../ws.js';
import {
    SEAT_NAMES_FR, cardChipHtml, renderHand, renderHandMini, renderTrick,
    contractChipHtml, bidChipHtml,
} from '../shared/cards.js';
import { SEAT_COLOR_VARS } from '../shared/seats.js';
import { renderBackLink } from '../shared/analyse-back.js';

const docLink = (section) =>
    `<a class="doc-link" href="/about?s=${section}" title="Comment ça marche ?">?</a>`;

// Les trois avis, dans l'ordre où la table les montre.
const AVIS = [
    ['doudou', 'DouDou50', 'Le réseau Q, une passe avant, sans recherche'],
    ['isdd', 'Dédé', "L'agent de production : mondes playgen + solve DD"],
    ['oracle', 'Oracle', 'Double-dummy sur la vraie donne — un plafond, pas un joueur'],
];

const TEMPLATE = `
<div id="aj-wrap">
    <a id="aj-back" class="analyse-back hidden" href="#"></a>
    <div id="aj-import" class="prob-box">
        <div class="annonces-title">Analyse du jeu de la carte</div>
        <div class="aj-import-row">
            <input id="aj-cfn" type="text" placeholder="Coller le CFN d'une partie…" spellcheck="false">
            <input id="aj-idx" type="number" min="0" placeholder="coup">
            <button id="aj-load-btn" class="primary-btn">Analyser</button>
        </div>
        <p class="aj-hint">Depuis <a href="/analyse/rejouer">Rejouer une partie</a>, chaque carte
        porte un lien « Analyser cette carte » qui remplit cette page.</p>
    </div>

    <div id="aj-empty" class="aj-placeholder">
        Colle le CFN d'une partie, ou arrive ici depuis Rejouer.
    </div>

    <div id="aj-error" class="annonces-error hidden"></div>

    <div id="aj-main" class="hidden">
        <div id="aj-position" class="prob-box">
            <div id="aj-info-bar">
                <span id="aj-contract"></span>
                <span id="aj-trick-no"></span>
                <span id="aj-scores"></span>
                <span id="aj-turn"></span>
            </div>
            <div id="aj-position-grid">
                <div>
                    <div class="section-title">Pli en cours</div>
                    <div id="aj-trick-area">
                        <div class="trick-card" id="aj-trick-n"></div>
                        <div class="trick-card" id="aj-trick-w"></div>
                        <div class="trick-card" id="aj-trick-e"></div>
                        <div class="trick-card" id="aj-trick-s"></div>
                    </div>
                </div>
                <div>
                    <div class="section-title" id="aj-hand-title">Main du joueur</div>
                    <div class="hand" id="aj-hand"></div>
                    <div id="aj-bid-history"></div>
                </div>
            </div>
        </div>

        <div id="aj-forced" class="aj-placeholder hidden">
            Une seule carte jouable ici : aucune décision à analyser.
        </div>

        <div id="aj-table-panel" class="annonces-result-panel">
            <div class="section-title" id="aj-table-header">
                <span>Cartes jouables</span>
                ${docLink('jeu-carte')}
                <div id="aj-progress" class="sim-progress hidden">
                    <div class="sim-progress-bar"><div class="sim-progress-fill"></div></div>
                    <span class="sim-progress-text"></span>
                </div>
                <span id="aj-worlds-badge"></span>
            </div>
            <div id="aj-table-body"></div>
        </div>

        <div class="annonces-result-panel hidden" id="aj-samples-wrap">
            <details id="aj-samples">
                <summary>Voir 10 mondes échantillonnés</summary>
                <div id="aj-samples-content"></div>
            </details>
        </div>
    </div>
</div>
`;

// ── état ──

let reqId = 0;
let position = null;   // payload card_analysis_position.position
let truth = null;      // {best_card, ns, cost}
let opinions = null;   // {doudou, isdd, oracle}
let rows = null;       // {card: {...}} — agrégats par carte
let progress = null;   // {completed, total, real_total, elapsed_ms}
let worldsSource = null;
// Requête en attente : `send()` jette silencieusement quand le socket n'est pas
// encore OPEN. Sur une arrivée directe par URL (le cas normal ici, puisqu'on
// vient d'un lien de Rejouer ou d'un signet) la vue se monte avant l'ouverture,
// et sans ce relais la page restait vide sans rien dire.
let pending = null;
// Partie d'où l'on vient, quand on arrive depuis Rejouer. Conservée à travers
// les réécritures d'URL — cf. syncUrl.
let backGame = null;

function currentReq() { return `aj-${reqId}`; }

function syncUrl(cfn, idx) {
    const q = new URLSearchParams({ cfn, i: String(idx) });
    // Partie d'origine : ne décrit pas la position, mais l'effacer priverait
    // l'URL de son chemin de retour au premier changement de coup.
    if (backGame) q.set('from', backGame);
    history.replaceState(null, '', `${window.location.pathname}?${q}`);
}

// ── chargement ──

function load(cfn, idx) {
    cfn = (cfn || '').trim();
    if (!cfn) return;
    reqId += 1;
    position = null; truth = null; opinions = null; rows = null;
    progress = null; worldsSource = null;

    document.getElementById('aj-empty').classList.add('hidden');
    document.getElementById('aj-error').classList.add('hidden');
    document.getElementById('aj-main').classList.add('hidden');
    document.getElementById('aj-samples-wrap').classList.add('hidden');
    document.getElementById('aj-table-body').innerHTML =
        '<div class="dd-loader"><div class="dd-loader-text">Analyse…</div></div>';

    syncUrl(cfn, idx);
    pending = { cfn, idx, req: currentReq() };
    send({ type: 'card_analysis', cfn, idx, req_id: currentReq() });
}

// Rejoué à l'ouverture du socket. Si la requête est déjà passée, la position est
// arrivée et `pending` a été vidé, donc il n'y a rien à refaire.
function flushPending() {
    if (!pending) return;
    send({ type: 'card_analysis', cfn: pending.cfn, idx: pending.idx,
           req_id: pending.req });
}

// ── rendu de la position ──

function renderPosition() {
    const p = position;
    document.getElementById('aj-main').classList.remove('hidden');

    document.getElementById('aj-contract').innerHTML = p.contract
        ? contractChipHtml(p.contract) : '';
    document.getElementById('aj-trick-no').textContent = `Pli ${p.trick_no}/8`;
    document.getElementById('aj-scores').textContent =
        `Nord-Sud ${p.points[0]} · Est-Ouest ${p.points[1]}`;

    const turn = document.getElementById('aj-turn');
    turn.innerHTML = `À <strong style="color:${SEAT_COLOR_VARS[p.seat]}">${SEAT_NAMES_FR[p.seat]}</strong> de jouer`;

    renderTrick('aj-trick', p.current_trick, p.trick_lead);

    document.getElementById('aj-hand-title').textContent =
        `Main de ${SEAT_NAMES_FR[p.seat]} — ${p.hands[p.seat].length} cartes`;
    const trump = p.contract ? p.contract.trump : -1;
    renderHand(document.getElementById('aj-hand'), p.hands[p.seat], false, null,
               new Set(p.legal), trump);

    const bh = document.getElementById('aj-bid-history');
    bh.innerHTML = p.bid_history.map(([seat, action]) =>
        `<span class="watch-bid-entry ${seat % 2 === 0 ? 'team-ns' : 'team-ew'}">` +
        `${SEAT_NAMES_FR[seat]} ${bidChipHtml(action)}</span>`).join('');

    document.getElementById('aj-forced').classList.toggle('hidden', !p.forced);
    document.getElementById('aj-table-panel').classList.toggle('hidden', !!p.forced);
}

// ── la table ──

// Fiabilité d'une cellule par l'opacité, comme le Jeu réel des annonces : un
// tableau où chaque nombre a un corps différent est illisible.
function confidenceClass(n) {
    if (!n) return 'conf-lo';
    if (n < 20) return 'conf-lo';
    if (n < 80) return 'conf-mid';
    return 'conf-hi';
}

function num(v, digits = 0) {
    return v === null || v === undefined ? '·' : v.toFixed(digits);
}

function pct(v) {
    return v === null || v === undefined ? '·' : `${v}<span class="aj-unit">%</span>`;
}

// Le regret note la ligne : 0 = aussi bon que la meilleure carte du tableau.
function regretClass(regret) {
    if (regret === null || regret === undefined) return '';
    if (regret <= 1) return 'aj-best';
    if (regret <= 8) return 'aj-ok';
    if (regret <= 20) return 'aj-meh';
    return 'aj-bad';
}

function riskClass(p) {
    if (p === null || p === undefined) return '';
    if (p >= 50) return 'aj-bad';
    if (p >= 20) return 'aj-meh';
    return 'aj-ok';
}

function avisBadges(card) {
    if (!opinions) return '';
    return AVIS.map(([key, label, help]) =>
        opinions[key] === card
            ? `<span class="aj-avis aj-avis-${key}" title="${help}">${label}</span>`
            : '').filter(Boolean).join('');
}

function renderTable() {
    const p = position;
    if (!p || p.forced) return;

    const takerTeam = p.contract ? p.contract.team : 0;
    const defending = (p.seat % 2) !== takerTeam;
    // « Contrat réussi » est l'issue du preneur : sur un siège en défense, un
    // taux élevé désigne la pire carte. La colonne est donc lue du côté du
    // siège qui joue, et son titre le dit.
    const winLabel = defending ? 'Contrat chuté' : 'Contrat réussi';

    let html = '<div class="aj-table-scroll"><table id="aj-table">';
    html += `<thead>
        <tr class="aj-group-row">
            <th></th>
            <th colspan="4" class="aj-grp aj-grp-oracle">Mondes de l'information set
                <span class="aj-grp-sub">${SEAT_NAMES_FR[p.seat]} ne voit que sa main</span></th>
            <th colspan="2" class="aj-grp aj-grp-real">Jeu réel <span class="aj-grp-sub">DouDou50 finit la donne</span></th>
            <th colspan="1" class="aj-grp aj-grp-truth">Vrai monde <span class="aj-grp-sub">information parfaite</span></th>
            <th class="aj-grp">Avis</th>
        </tr>
        <tr class="aj-head-row">
            <th>Carte</th>
            <th>Points N-S <span class="aj-grp-sub">moy.</span></th>
            <th>Méd.</th>
            <th title="Part des mondes où cette carte est la meilleure pour son équipe">Meilleure</th>
            <th title="Part des mondes où cette carte perd au moins 10 points face à la meilleure">Risque</th>
            <th title="Écart de points de donne marqués, Nord-Sud moins Est-Ouest">Écart N-S</th>
            <th>${winLabel}</th>
            <th title="Points perdus face au meilleur coup sur la donne réelle">Coût</th>
            <th></th>
        </tr></thead><tbody>`;

    // Tri : meilleure carte de l'information set en tête. À défaut d'agrégats
    // (début de streaming), on garde l'ordre des candidates.
    const cards = p.candidates.slice().sort((a, b) => {
        const ra = rows?.[a]?.regret, rb = rows?.[b]?.regret;
        if (ra === undefined || ra === null) return 1;
        if (rb === undefined || rb === null) return -1;
        return ra - rb;
    });

    for (const card of cards) {
        const r = rows?.[String(card)] || {};
        const cost = truth?.cost?.[String(card)];
        const played = card === p.played_action;
        const conf = confidenceClass(r.n);
        const realConf = confidenceClass(r.real_n);
        const winPct = r.real_n ? Math.round((defending ? r.real_n - r.real_ok : r.real_ok) / r.real_n * 100) : null;

        // Une carte absente de l'ensemble réduit est équivalente à une autre de
        // la même couleur : les deux se valent, et leurs lignes porteront des
        // chiffres quasi identiques. On le signale plutôt que de laisser croire
        // à deux options distinctes. Laquelle exactement, la réduction ne le dit
        // pas, donc on ne le prétend pas.
        const equiv = p.reduced && !p.reduced.includes(card);

        html += `<tr class="${played ? 'aj-played' : ''} ${regretClass(r.regret)}">`;
        html += `<td class="aj-card-cell">${cardChipHtml(card)}` +
                (played ? '<span class="aj-played-tag">jouée</span>' : '') +
                (equiv ? '<span class="aj-equiv" title="Équivalente à une autre carte de la même couleur : jouer l\'une ou l\'autre ne change rien pour la suite de la donne">≡</span>' : '') +
                '</td>';
        html += `<td class="${conf}">${num(r.mean_ns, 1)}` +
                (r.regret ? `<span class="aj-regret">−${num(r.regret, 1)}</span>` : '') + '</td>';
        html += `<td class="${conf}">${num(r.median_ns, 0)}</td>`;
        html += `<td class="${conf}">${pct(r.best_pct)}</td>`;
        html += `<td class="${conf} ${riskClass(r.risk_pct)}">${pct(r.risk_pct)}</td>`;
        const diff = r.real_diff;
        html += `<td class="${realConf}">${diff === null || diff === undefined ? '·'
            : (diff >= 0 ? '+' : '−') + Math.abs(diff).toFixed(0)}</td>`;
        html += `<td class="${realConf}" title="${r.real_n || 0} donnes jouées">${pct(winPct)}</td>`;
        html += `<td class="aj-truth-cell">${cost === null || cost === undefined ? '·'
            : (cost === 0 ? '0' : `−${cost}`)}</td>`;
        html += `<td class="aj-avis-cell">${avisBadges(card)}</td>`;
        html += '</tr>';
    }
    html += '</tbody></table></div>';

    html += renderLegend(defending);
    document.getElementById('aj-table-body').innerHTML = html;
}

function renderLegend(defending) {
    const p = position;
    const n = progress ? progress.total : null;
    const realN = progress ? progress.real_total : null;
    const parts = [];
    if (n) {
        parts.push(`<strong>${n}</strong> mondes résolus en double-dummy`);
        parts.push(`<strong>${realN}</strong> d'entre eux joués jusqu'au bout par DouDou50`);
    }
    parts.push(`« Coût » et « Vrai monde » portent sur la donne telle qu'elle était : ` +
        `c'est ce que montre déjà Rejouer, et ça ne dit pas si le choix était raisonnable.`);
    parts.push(`Les colonnes de gauche ne voient que ce que ${SEAT_NAMES_FR[p.seat]} pouvait savoir — ` +
        `c'est là qu'on juge la décision.`);
    if (defending) {
        parts.push(`${SEAT_NAMES_FR[p.seat]} défend : faire chuter le contrat est l'issue favorable.`);
    }
    return `<div class="aj-legend">${parts.map(s => `<p>${s}</p>`).join('')}</div>`;
}

function renderProgress() {
    if (!progress) return;
    const wrap = document.getElementById('aj-progress');
    if (!wrap) return;
    wrap.classList.remove('hidden');
    const { completed, total, elapsed_ms } = progress;
    const p = total > 0 ? Math.round(completed / total * 100) : 0;
    wrap.querySelector('.sim-progress-fill').style.width = `${p}%`;
    wrap.querySelector('.sim-progress-text').textContent =
        `${completed}/${total} — ${(elapsed_ms / 1000).toFixed(1)}s`;
    wrap.classList.toggle('done', completed >= total);

    const badge = document.getElementById('aj-worlds-badge');
    if (badge && worldsSource) {
        badge.className = `aj-worlds-badge ${worldsSource === 'uniform' ? 'aj-worlds-warn' : ''}`;
        badge.textContent = worldsSource === 'uniform'
            ? 'mondes uniformes — les coupes révélées par le jeu sont ignorées'
            : 'mondes playgen v2';
    }
}

function renderSamples(sampleHands) {
    if (!sampleHands || !sampleHands.length) return;
    const wrap = document.getElementById('aj-samples-wrap');
    wrap.classList.remove('hidden');
    const p = position;
    const trump = p.contract ? p.contract.trump : -1;

    let html = '';
    sampleHands.forEach((hands, i) => {
        html += `<details class="sim-deal-details"><summary>Monde ${i + 1}</summary>
            <div class="sim-deal-hands">`;
        for (let seat = 0; seat < 4; seat++) {
            if (seat === p.seat) continue;
            html += `<div class="sim-hand-section">
                <span class="sim-hand-label">${SEAT_NAMES_FR[seat]}</span>
                <div class="sim-hand" id="aj-sample-${i}-${seat}"></div>
            </div>`;
        }
        html += '</div></details>';
    });
    document.getElementById('aj-samples-content').innerHTML = html;

    // Les mains du monde sont les mains *initiales* reconstruites ; on n'affiche
    // que ce qu'il reste, sinon on montrerait des cartes déjà sur la table.
    const stillHeld = new Set();
    for (let s = 0; s < 4; s++) for (const c of p.hands[s]) stillHeld.add(c);
    sampleHands.forEach((hands, i) => {
        for (let seat = 0; seat < 4; seat++) {
            if (seat === p.seat) continue;
            const el = document.getElementById(`aj-sample-${i}-${seat}`);
            if (!el) continue;
            const remaining = hands[seat].filter(c => !p.played[seat].includes(c));
            renderHandMini(el, remaining, 34, trump);
        }
    });
}

// ── messages ──

function stale(data) { return data.req_id && data.req_id !== currentReq(); }

function onPosition(data) {
    if (stale(data)) return;
    pending = null;
    position = data.position;
    progress = { completed: 0, total: data.plan.oracle_worlds,
                 real_total: data.plan.real_worlds, elapsed_ms: 0 };
    renderPosition();
    if (position.forced) {
        document.getElementById('aj-table-body').innerHTML = '';
        return;
    }
    renderTable();
    renderProgress();
    const idxEl = document.getElementById('aj-idx');
    if (idxEl) idxEl.value = String(data.idx);
    const cfnEl = document.getElementById('aj-cfn');
    if (cfnEl) cfnEl.value = data.cfn;
}

function onTruth(data) {
    if (stale(data)) return;
    truth = data.truth;
    renderTable();
}

function onOpinions(data) {
    if (stale(data)) return;
    opinions = data.opinions;
    renderTable();
}

function onUpdate(data) {
    if (stale(data)) return;
    rows = data.rows;
    worldsSource = data.worlds_source;
    progress = { completed: data.completed, total: data.total,
                 real_total: data.real_total, elapsed_ms: data.elapsed_ms };
    renderTable();
    renderProgress();
}

function onDone(data) {
    if (stale(data)) return;
    onUpdate(data);
    renderSamples(data.sample_hands);
}

function onError(data) {
    if (stale(data)) return;
    pending = null;
    const el = document.getElementById('aj-error');
    el.classList.remove('hidden');
    el.innerHTML = data.phase === 0
        ? `${data.error} — <a href="/analyse/annonces">analyser une annonce</a>`
        : data.error;
    document.getElementById('aj-main').classList.add('hidden');
    document.getElementById('aj-empty').classList.add('hidden');
}

// ── cycle de vie ──

export function mount(container) {
    container.innerHTML = TEMPLATE;
    reqId = 0;
    position = null; truth = null; opinions = null; rows = null;
    progress = null; worldsSource = null; pending = null; backGame = null;

    onOpen(flushPending);
    onMessage('card_analysis_position', onPosition);
    onMessage('card_analysis_truth', onTruth);
    onMessage('card_analysis_opinions', onOpinions);
    onMessage('card_analysis_update', onUpdate);
    onMessage('card_analysis_done', onDone);
    onMessage('card_analysis_error', onError);

    document.getElementById('aj-load-btn').addEventListener('click', () => {
        const cfn = document.getElementById('aj-cfn').value;
        const idx = parseInt(document.getElementById('aj-idx').value, 10);
        load(cfn, Number.isFinite(idx) ? idx : 0);
    });
    document.getElementById('aj-cfn').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') document.getElementById('aj-load-btn').click();
    });

    const params = new URLSearchParams(window.location.search);
    const cfn = params.get('cfn');
    const idx = parseInt(params.get('i'), 10);

    // Retour explicite vers la partie. Ne dépend pas de l'historique du
    // navigateur, donc marche aussi sur un lien partagé ou un signet — là où le
    // bouton Retour ramènerait ailleurs, voire hors du site.
    backGame = params.get('from');
    renderBackLink('aj-back', backGame, params.get('i'));

    if (cfn) {
        document.getElementById('aj-cfn').value = cfn;
        document.getElementById('aj-idx').value = Number.isFinite(idx) ? String(idx) : '0';
        load(cfn, Number.isFinite(idx) ? idx : 0);
    }
}

export function unmount() {
    offOpen(flushPending);
    offMessage('card_analysis_position', onPosition);
    offMessage('card_analysis_truth', onTruth);
    offMessage('card_analysis_opinions', onOpinions);
    offMessage('card_analysis_update', onUpdate);
    offMessage('card_analysis_done', onDone);
    offMessage('card_analysis_error', onError);
    reqId += 1;  // les messages en vol deviennent périmés
    position = null; truth = null; opinions = null; rows = null; pending = null;
    backGame = null;
}
