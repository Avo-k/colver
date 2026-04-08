// Problèmes d'annonce view — single-bid practice problems

import { send, onMessage, offMessage } from '../ws.js';
import { SUITS, SEAT_NAMES_FR, renderHand, renderFaceDownHand, actionName, encodeBidAction, bidActionHtml } from '../shared/cards.js';
import * as xgbExplain from '../xgb-explain.js';

const TEMPLATE = `
<div id="pa-config">
    <span class="annonces-title">Probl\u00e8mes d'annonce</span>
    <button id="pa-generate-btn">Nouveau probl\u00e8me</button>
    <span id="pa-loading" class="hidden" style="color:#888;font-size:.85rem">G\u00e9n\u00e9ration\u2026</span>
</div>
<div id="pa-table" class="hidden">
    <div class="seats">
        <div class="seat north">
            <div class="seat-label">Nord (partenaire)</div>
            <div class="hand" id="pa-hand-north"></div>
        </div>
        <div class="seat west">
            <div class="seat-label">Ouest</div>
            <div class="hand" id="pa-hand-west"></div>
        </div>
        <div id="pa-center" class="pa-center-area">
            <div id="pa-bidding-panel">
                <div id="pa-bid-history-entries"></div>
                <div id="pa-bid-panel">
                    <div id="pa-bid-selectors">
                        <select id="pa-bid-value">
                            <option value="">Valeur...</option>
                            <option value="80">80</option><option value="90">90</option>
                            <option value="100">100</option><option value="110">110</option>
                            <option value="120">120</option><option value="130">130</option>
                            <option value="140">140</option><option value="150">150</option>
                            <option value="160">160</option><option value="250">Capot</option>
                        </select>
                        <select id="pa-bid-suit">
                            <option value="0">\u2660</option><option value="1">\u2665</option>
                            <option value="2">\u2666</option><option value="3">\u2663</option>
                        </select>
                        <button id="pa-bid-submit" class="bid-btn bid-action">Annoncer</button>
                    </div>
                    <div id="pa-bid-special">
                        <button id="pa-pass-btn" class="bid-btn pass hidden">Passe</button>
                        <button id="pa-coinche-btn" class="bid-btn coinche hidden">Coinche</button>
                        <button id="pa-surcoinche-btn" class="bid-btn coinche hidden">Surcoinche</button>
                    </div>
                    <button id="pa-hint-btn" class="pa-hint-btn hidden">Indice</button>
                    <div id="pa-hint-content" class="pa-hint-content hidden"></div>
                </div>
            </div>
        </div>
        <div class="seat east">
            <div class="seat-label">Est</div>
            <div class="hand" id="pa-hand-east"></div>
        </div>
        <div class="seat south">
            <div class="seat-label">Sud (vous)</div>
            <div class="hand" id="pa-hand-south"></div>
        </div>
    </div>
    <div id="pa-correction" class="prob-correction hidden">
        <div class="section-title">Correction</div>
        <div id="pa-player-badge" class="prob-badge"></div>
        <div class="prob-section">
            <div class="prob-label">Bid \u00e0 D\u00e9d\u00e9 (NN)</div>
            <div id="pa-nn-best" class="prob-best"></div>
            <div id="pa-nn-bars"></div>
        </div>
        <div class="prob-section">
            <div class="prob-label">Heuristique (am\u00e9lior\u00e9)</div>
            <div id="pa-heuristic-best" class="prob-best"></div>
        </div>
        <div class="prob-section hidden" id="pa-xgb-section">
            <div class="prob-label">Facteurs cl\u00e9s <span style="font-size:0.7rem;color:#777">(XGBoost, pas le NN)</span></div>
            <select id="pa-xgb-suit-select" style="margin-bottom:4px"></select>
            <div id="pa-xgb-waterfall"></div>
            <div id="pa-xgb-probability"></div>
        </div>
        <div class="prob-section">
            <div class="prob-label">Analyse Double-Dummy</div>
            <table class="dd-table">
                <tr><th></th><th>NS</th><th>EO</th></tr>
                <tr><td>\u2660</td><td id="pa-dd-s-ns">-</td><td id="pa-dd-s-ew">-</td></tr>
                <tr><td class="red">\u2665</td><td id="pa-dd-h-ns">-</td><td id="pa-dd-h-ew">-</td></tr>
                <tr><td>\u2663</td><td id="pa-dd-c-ns">-</td><td id="pa-dd-c-ew">-</td></tr>
                <tr><td class="red">\u2666</td><td id="pa-dd-d-ns">-</td><td id="pa-dd-d-ew">-</td></tr>
            </table>
            <div id="pa-dd-elapsed" class="visit-total"></div>
        </div>
        <button id="pa-next-btn" class="prob-next-btn">Probl\u00e8me suivant</button>
    </div>
</div>
`;

const SUIT_SYMBOLS = ['\u2660', '\u2665', '\u2666', '\u2663'];

let paLegalActions = [];
let paLocked = false;
let paHand = [];
let paBidHistory = [];
let paXgbResults = null;
let paHintData = null;

function bidActionName(action) {
    if (action === 0) return 'Passe';
    if (action >= 37 && action <= 40) return `Capot ${SUITS[action - 37]}`;
    if (action === 41) return 'Coinche';
    if (action === 42) return 'Surcoinche';
    if (action >= 1 && action <= 36) {
        const bidIdx = action - 1;
        const valueIdx = Math.floor(bidIdx / 4);
        const suitIdx = bidIdx % 4;
        const value = 80 + valueIdx * 10;
        return `${value} ${SUITS[suitIdx]}`;
    }
    return `Action ${action}`;
}

async function prepareHint(hand, bidHistory, qValues) {
    paHintData = null;
    const hintBtn = document.getElementById('pa-hint-btn');
    const hintContent = document.getElementById('pa-hint-content');
    hintBtn.classList.add('hidden');
    hintContent.classList.add('hidden');
    hintContent.innerHTML = '';

    if (!qValues || !qValues.length || hand.length !== 8) return;

    try {
        const results = await xgbExplain.analyzeAllSuits(hand, bidHistory, qValues);
        if (!results || results.length === 0) return;

        // Pick the suit with highest probability of bidding
        const best = results[0]; // already sorted by probability
        const entries = Object.entries(best.contributions)
            .filter(([, v]) => Math.abs(v) > 0.001)
            .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]))
            .slice(0, 3);

        if (entries.length === 0) return;

        const suitCls = (best.suit === 1 || best.suit === 2) ? 'suit-red' : 'suit-black';
        const suitSym = SUIT_SYMBOLS[best.suit];

        paHintData = { entries, suit: best.suit, suitSym, suitCls, features: best.features };
        hintBtn.classList.remove('hidden');
    } catch (err) {
        console.warn('[pa-hint] XGB analysis failed:', err);
    }
}

function revealHint() {
    if (!paHintData) return;
    const { entries, suitSym, suitCls, features } = paHintData;
    const hintContent = document.getElementById('pa-hint-content');
    const hintBtn = document.getElementById('pa-hint-btn');

    let html = `<div class="pa-hint-title">Facteurs cl\u00e9s pour <span class="${suitCls}">${suitSym}</span></div>`;
    for (const [feat, val] of entries) {
        const label = xgbExplain.featureLabel(feat);
        const featVal = features[feat];
        const arrow = val > 0 ? '\u2191' : '\u2193';
        const cls = val > 0 ? 'pa-hint-pos' : 'pa-hint-neg';
        const abs = Math.abs(val);
        const strength = abs >= 1.0 ? '\u25cf\u25cf\u25cf' : abs >= 0.3 ? '\u25cf\u25cf' : '\u25cf';
        const valDisplay = featVal !== undefined ? ` = ${featVal}` : '';
        html += `<div class="pa-hint-row ${cls}"><span class="pa-hint-arrow">${arrow}</span><span class="pa-hint-label">${label}<span class="pa-hint-val">${valDisplay}</span></span><span class="pa-hint-strength">${strength}</span></div>`;
    }
    hintContent.innerHTML = html;
    hintContent.classList.remove('hidden');
    hintBtn.classList.add('hidden');
}

function handleProblemReady(data) {
    document.getElementById('pa-loading').classList.add('hidden');
    document.getElementById('pa-table').classList.remove('hidden');
    document.getElementById('pa-correction').classList.add('hidden');
    document.getElementById('pa-bid-panel').style.display = '';
    paLegalActions = data.legal_actions;
    paLocked = false;
    paHand = data.south_hand || [];
    paXgbResults = null;
    paHintData = null;
    // Extract bid actions from history for XGB scenario detection
    paBidHistory = (data.bid_history || []).map(e => e.action);

    // Render face-down hands for N/E/W
    renderFaceDownHand(document.getElementById('pa-hand-north'), 8);
    renderFaceDownHand(document.getElementById('pa-hand-east'), 8);
    renderFaceDownHand(document.getElementById('pa-hand-west'), 8);

    // Render bid history in center panel
    const entries = document.getElementById('pa-bid-history-entries');
    entries.innerHTML = '';
    for (const e of data.bid_history) {
        const sp = document.createElement('span');
        sp.className = 'bid-entry ' + (e.player % 2 === 0 ? 'team-partner' : 'team-opponent');
        sp.innerHTML = `<span class="pa-bid-seat">${SEAT_NAMES_FR[e.player]}</span> ${bidActionHtml(e.action)}`;
        entries.appendChild(sp);
    }
    const you = document.createElement('span');
    you.className = 'bid-entry pa-your-turn';
    you.textContent = 'Sud ?';
    entries.appendChild(you);

    // Render South's hand
    renderHand(document.getElementById('pa-hand-south'), data.south_hand);

    // Configure bid controls
    const legalSet = new Set(data.legal_actions);
    const valSel = document.getElementById('pa-bid-value');
    valSel.value = '';
    for (const opt of valSel.options) {
        if (!opt.value) continue;
        const v = parseInt(opt.value);
        let ok = false;
        for (let s = 0; s < 4; s++) {
            if (legalSet.has(encodeBidAction(v, s))) { ok = true; break; }
        }
        opt.disabled = !ok;
        if (ok && !valSel.value) valSel.value = String(opt.value);
    }

    const showBid = [...legalSet].some(a => a >= 1 && a <= 40);
    document.getElementById('pa-bid-selectors').style.display = showBid ? 'flex' : 'none';
    document.getElementById('pa-pass-btn').classList.toggle('hidden', !legalSet.has(0));
    document.getElementById('pa-coinche-btn').classList.toggle('hidden', !legalSet.has(41));
    document.getElementById('pa-surcoinche-btn').classList.toggle('hidden', !legalSet.has(42));

    // Prepare hint from NN Q-values (async, non-blocking)
    if (data.nn_q_values) {
        prepareHint(paHand, paBidHistory, data.nn_q_values);
    }
}

function renderPaXgbWaterfall(result, containerId, probId) {
    const container = document.getElementById(containerId);
    const probEl = document.getElementById(probId);
    if (!container || !result) return;
    const entries = Object.entries(result.contributions)
        .filter(([, v]) => Math.abs(v) > 0.001)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));
    if (entries.length === 0) {
        container.innerHTML = '';
        probEl.innerHTML = '';
        return;
    }
    const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)));
    let html = '<div class="xgb-waterfall-chart">';
    for (const [feat, val] of entries) {
        const label = xgbExplain.featureLabel(feat);
        const featVal = result.features[feat];
        const pct = Math.abs(val) / maxAbs * 100;
        const cls = val > 0 ? 'xgb-bar-pos' : 'xgb-bar-neg';
        const sign = val > 0 ? '+' : '';
        const valDisplay = featVal !== undefined ? ` = ${featVal}` : '';
        html += `<div class="xgb-row">
            <span class="xgb-feat-name" title="${feat}${valDisplay}">${label}<span class="xgb-feat-val">${valDisplay}</span></span>
            <div class="xgb-bar-wrap"><div class="xgb-bar ${cls}" style="width:${pct.toFixed(0)}%"></div></div>
            <span class="xgb-contrib">${sign}${val.toFixed(3)}</span>
        </div>`;
    }
    html += '</div>';
    container.innerHTML = html;
    const pctVal = (result.probability * 100).toFixed(0);
    const pcls = result.probability >= 0.5 ? 'xgb-prob-high' : 'xgb-prob-low';
    probEl.innerHTML = `<span class="${pcls}">P(enchérir) : ${pctVal}%</span>`;
}

async function runPaXgbAnalysis(qValues) {
    try {
        const results = await xgbExplain.analyzeAllSuits(paHand, paBidHistory, qValues);
        if (!results) return;
        paXgbResults = results;
        const section = document.getElementById('pa-xgb-section');
        if (section) section.classList.remove('hidden');
        const select = document.getElementById('pa-xgb-suit-select');
        select.innerHTML = '';
        for (let i = 0; i < results.length; i++) {
            const r = results[i];
            const opt = document.createElement('option');
            opt.value = i;
            opt.innerHTML = `${SUIT_SYMBOLS[r.suit]} (${(r.probability * 100).toFixed(0)}%)`;
            select.appendChild(opt);
        }
        select.value = '0';
        renderPaXgbWaterfall(results[0], 'pa-xgb-waterfall', 'pa-xgb-probability');
    } catch (err) {
        console.warn('[pa-xgb] Analysis failed:', err);
    }
}

function handleCorrection(data) {
    document.getElementById('pa-bid-panel').style.display = 'none';
    document.getElementById('pa-correction').classList.remove('hidden');

    const correct = data.nn_action !== null && data.player_action === data.nn_action;
    const badge = document.getElementById('pa-player-badge');
    badge.className = 'prob-badge ' + (correct ? 'prob-badge-correct' : 'prob-badge-wrong');
    badge.innerHTML = 'Votre annonce : ' + bidActionHtml(data.player_action);

    const nnBestEl = document.getElementById('pa-nn-best');
    const nnBarsEl = document.getElementById('pa-nn-bars');
    nnBarsEl.innerHTML = '';
    if (data.nn_q_values && data.nn_q_values.length) {
        nnBestEl.innerHTML = 'NN : ' + (data.nn_action != null ? bidActionHtml(data.nn_action) : '\u2014');
        const sorted = [...data.nn_q_values].sort((a, b) => b[1] - a[1]).slice(0, 8);
        const mn = Math.min(...sorted.map(x => x[1]));
        const mx = Math.max(...sorted.map(x => x[1]));
        const rng = mx - mn || 1;
        const div = document.createElement('div');
        div.className = 'visit-bars';
        for (const [a, q] of sorted) {
            const isBest = a === data.nn_action;
            const isPlayer = a === data.player_action;
            const row = document.createElement('div');
            row.className = 'visit-row' + (isBest ? ' best' : '') + (isPlayer ? ' prob-player-pick' : '');
            const pct = ((q - mn) / rng * 100).toFixed(0);
            row.innerHTML = `<span class="visit-name">${bidActionHtml(a)}</span>` +
                `<div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>` +
                `<span class="visit-count">${q.toFixed(3)}</span>`;
            div.appendChild(row);
        }
        nnBarsEl.appendChild(div);
    } else {
        nnBestEl.textContent = 'NN : non disponible';
    }

    document.getElementById('pa-heuristic-best').innerHTML = 'Am\u00e9lior\u00e9 : ' + bidActionHtml(data.heuristic_action);

    const keys = ['s', 'h', 'd', 'c'];
    if (data.dd_suits) {
        for (let i = 0; i < 4; i++) {
            document.getElementById(`pa-dd-${keys[i]}-ns`).textContent = data.dd_suits[i][0];
            document.getElementById(`pa-dd-${keys[i]}-ew`).textContent = data.dd_suits[i][1];
        }
    }
    document.getElementById('pa-dd-elapsed').textContent = `DD : ${data.dd_elapsed_ms}ms`;

    // XGB interpretability
    if (data.nn_q_values && data.nn_q_values.length && paHand.length === 8) {
        runPaXgbAnalysis(data.nn_q_values);
    }
}

function paBidSubmit(action) {
    if (paLocked) return;
    if (!new Set(paLegalActions).has(action)) return;
    paLocked = true;
    send({ type: 'bid_problem_submit', action });
}

export function mount(container) {
    container.innerHTML = TEMPLATE;
    paLegalActions = [];
    paLocked = false;

    document.getElementById('pa-hint-btn').addEventListener('click', revealHint);

    document.getElementById('pa-generate-btn').addEventListener('click', () => {
        paLocked = false;
        paHintData = null;
        document.getElementById('pa-loading').classList.remove('hidden');
        document.getElementById('pa-table').classList.add('hidden');
        send({ type: 'bid_problem_generate' });
    });

    document.getElementById('pa-bid-submit').onclick = () => {
        const v = parseInt(document.getElementById('pa-bid-value').value);
        const s = parseInt(document.getElementById('pa-bid-suit').value);
        if (isNaN(v)) return;
        const a = encodeBidAction(v, s);
        paBidSubmit(a);
    };
    document.getElementById('pa-pass-btn').onclick = () => paBidSubmit(0);
    document.getElementById('pa-coinche-btn').onclick = () => paBidSubmit(41);
    document.getElementById('pa-surcoinche-btn').onclick = () => paBidSubmit(42);
    document.getElementById('pa-next-btn').onclick = () => document.getElementById('pa-generate-btn').click();
    document.getElementById('pa-xgb-suit-select').addEventListener('change', (e) => {
        if (paXgbResults) {
            renderPaXgbWaterfall(paXgbResults[parseInt(e.target.value)], 'pa-xgb-waterfall', 'pa-xgb-probability');
        }
    });

    onMessage('bid_problem_ready', handleProblemReady);
    onMessage('bid_problem_correction', handleCorrection);

    // Auto-load first problem
    document.getElementById('pa-generate-btn').click();
}

export function unmount() {
    offMessage('bid_problem_ready', handleProblemReady);
    offMessage('bid_problem_correction', handleCorrection);
    paLegalActions = [];
    paLocked = false;
}
