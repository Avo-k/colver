// Problèmes d'annonce view — single-bid practice problems

import { send, onMessage, offMessage } from '../ws.js';
import { SUITS, renderHand, actionName, encodeBidAction } from '../shared/cards.js';

const TEMPLATE = `
<div id="pa-config">
    <span class="annonces-title">Probl\u00e8mes d'annonce</span>
    <button id="pa-generate-btn">Nouveau probl\u00e8me</button>
    <span id="pa-loading" class="hidden" style="color:#888;font-size:.85rem">G\u00e9n\u00e9ration\u2026</span>
</div>
<div id="pa-main" class="hidden">
    <div id="pa-left">
        <div class="prob-box">
            <div class="section-title">Ench\u00e8res pr\u00e9c\u00e9dentes</div>
            <div id="pa-bid-history-entries" style="display:flex;gap:4px;flex-wrap:wrap;min-height:20px"></div>
        </div>
        <div class="prob-box">
            <div class="section-title">Votre main (Sud)</div>
            <div class="hand" id="pa-hand-display"></div>
        </div>
        <div class="prob-box" id="pa-bid-panel">
            <div id="pa-bid-selectors" style="display:flex;gap:6px;flex-wrap:wrap;align-items:center">
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
            <div style="display:flex;gap:6px;margin-top:6px">
                <button id="pa-pass-btn" class="bid-btn pass hidden">Passe</button>
                <button id="pa-coinche-btn" class="bid-btn coinche hidden">Coinche</button>
                <button id="pa-surcoinche-btn" class="bid-btn coinche hidden">Surcoinche</button>
            </div>
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

let paLegalActions = [];
let paLocked = false;

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

function handleProblemReady(data) {
    document.getElementById('pa-loading').classList.add('hidden');
    document.getElementById('pa-main').classList.remove('hidden');
    document.getElementById('pa-correction').classList.add('hidden');
    document.getElementById('pa-bid-panel').style.display = '';
    paLegalActions = data.legal_actions;
    paLocked = false;

    // Render bid history
    const entries = document.getElementById('pa-bid-history-entries');
    entries.innerHTML = '';
    for (const e of data.bid_history) {
        const sp = document.createElement('span');
        sp.className = 'watch-bid-entry ' + (e.player % 2 === 0 ? 'team-ns' : 'team-ew');
        sp.textContent = ['N', 'E', 'S', 'O'][e.player] + ':' + e.name;
        entries.appendChild(sp);
    }
    const you = document.createElement('span');
    you.className = 'watch-bid-entry';
    you.style.cssText = 'color:#d4af37;font-style:italic';
    you.textContent = 'S:?';
    entries.appendChild(you);

    // Render South's hand
    renderHand(document.getElementById('pa-hand-display'), data.south_hand);

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
}

function handleCorrection(data) {
    document.getElementById('pa-bid-panel').style.display = 'none';
    document.getElementById('pa-correction').classList.remove('hidden');

    const correct = data.nn_action !== null && data.player_action === data.nn_action;
    const badge = document.getElementById('pa-player-badge');
    badge.className = 'prob-badge ' + (correct ? 'prob-badge-correct' : 'prob-badge-wrong');
    badge.textContent = 'Votre annonce : ' + data.player_action_name;

    const nnBestEl = document.getElementById('pa-nn-best');
    const nnBarsEl = document.getElementById('pa-nn-bars');
    nnBarsEl.innerHTML = '';
    if (data.nn_q_values && data.nn_q_values.length) {
        nnBestEl.textContent = 'NN : ' + (data.nn_action_name || '\u2014');
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
            row.innerHTML = `<span class="visit-name">${bidActionName(a)}</span>` +
                `<div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>` +
                `<span class="visit-count">${q.toFixed(3)}</span>`;
            div.appendChild(row);
        }
        nnBarsEl.appendChild(div);
    } else {
        nnBestEl.textContent = 'NN : non disponible';
    }

    document.getElementById('pa-heuristic-best').textContent = 'Am\u00e9lior\u00e9 : ' + data.heuristic_action_name;

    const keys = ['s', 'h', 'd', 'c'];
    if (data.dd_suits) {
        for (let i = 0; i < 4; i++) {
            document.getElementById(`pa-dd-${keys[i]}-ns`).textContent = data.dd_suits[i][0];
            document.getElementById(`pa-dd-${keys[i]}-ew`).textContent = data.dd_suits[i][1];
        }
    }
    document.getElementById('pa-dd-elapsed').textContent = `DD : ${data.dd_elapsed_ms}ms`;
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

    document.getElementById('pa-generate-btn').addEventListener('click', () => {
        paLocked = false;
        document.getElementById('pa-loading').classList.remove('hidden');
        document.getElementById('pa-main').classList.add('hidden');
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
