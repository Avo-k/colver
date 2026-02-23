// Annonces view — hand builder + bidding NN evaluation

import { send, onMessage, offMessage } from '../ws.js';
import { RANKS, SUITS, cardSvgPath, cardRank, cardSuit, renderHand, actionName } from '../shared/cards.js';

const SUIT_SYMBOLS = ['\u2660', '\u2665', '\u2666', '\u2663'];
const SEAT_NAMES = ['N', 'E', 'S', 'O'];
const SEAT_COLORS = ['#82cfff', '#82e0aa', '#d4af37', '#f0b429'];

function suitHtml(suitIdx) {
    const cls = (suitIdx === 1 || suitIdx === 2) ? 'suit-red' : 'suit-black';
    return `<span class="${cls}">${SUIT_SYMBOLS[suitIdx]}</span>`;
}

function bidActionHtml(action) {
    if (action === 0) return 'Passe';
    if (action >= 37 && action <= 40) return `Capot ${suitHtml(action - 37)}`;
    if (action === 41) return 'Coinche';
    if (action === 42) return 'Surcoinche';
    if (action >= 1 && action <= 36) {
        const bidIdx = action - 1;
        const valueIdx = Math.floor(bidIdx / 4);
        const suitIdx = bidIdx % 4;
        const value = 80 + valueIdx * 10;
        return `${value} ${suitHtml(suitIdx)}`;
    }
    return `Action ${action}`;
}

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

const TEMPLATE = `
<div id="annonces-config">
    <div id="annonces-header">
        <span class="annonces-title">\u00c9valuation des annonces</span>
        <span id="annonces-count">0/8 cartes</span>
        <button id="annonces-random-btn" class="secondary-btn">Main al\u00e9atoire</button>
        <button id="annonces-clear-btn" class="secondary-btn">Vider la main</button>
    </div>
    <div id="annonces-palette"></div>
    <div id="annonces-history-section">
        <div id="annonces-history-header">
            <span class="annonces-subtitle">Ench\u00e8res pr\u00e9c\u00e9dentes</span>
            <button id="annonces-history-clear-btn" class="secondary-btn">Vider</button>
        </div>
        <div id="annonces-history-list"></div>
        <div id="annonces-history-add">
            <select id="annonces-action-select"></select>
            <button id="annonces-history-add-btn">+ Ajouter</button>
        </div>
    </div>
    <div id="annonces-eval-row">
        <button id="annonces-eval-btn" disabled>\u00c9valuer</button>
        <label class="annonces-sim-label">Simulations DD :
            <input type="number" id="annonces-sim-count" value="10" min="1" max="200" style="width:55px">
        </label>
    </div>
    <div id="annonces-loading" class="hidden">Calcul en cours\u2026</div>
    <div id="annonces-results-row" class="hidden">
        <div id="annonces-results" class="annonces-result-col">
            <div id="annonces-results-header" class="section-title"></div>
            <div id="annonces-results-body"></div>
        </div>
        <div id="annonces-dd-results" class="annonces-result-col">
            <div id="annonces-dd-header" class="section-title"></div>
            <div id="annonces-dd-body"></div>
        </div>
    </div>
</div>
<div id="annonces-hand-preview">
    <div class="section-title">Votre main</div>
    <div class="hand" id="annonces-hand-display"></div>
</div>
`;

let annoncesHand = new Set();
let annoncesHistory = [];
let ddTimerId = null;
let ddStartTime = 0;
let ddEstimatedMs = 0;

function annoncesPlayerSeat(turnIdx, historyLen) {
    return (2 - historyLen + turnIdx + 32) % 4;
}

function initAnnoncesGrid() {
    const palette = document.getElementById('annonces-palette');
    palette.innerHTML = '';
    for (let suit = 0; suit < 4; suit++) {
        const label = document.createElement('div');
        label.className = 'palette-suit-label';
        label.innerHTML = suitHtml(suit);
        palette.appendChild(label);

        for (let rank = 0; rank < 8; rank++) {
            const idx = suit * 8 + rank;
            const el = document.createElement('div');
            el.className = 'card annonces-card';
            el.id = `annonces-card-${idx}`;

            const img = document.createElement('img');
            img.src = cardSvgPath(idx);
            img.alt = `${RANKS[rank]}${SUITS[suit]}`;
            img.draggable = false;
            el.appendChild(img);

            el.addEventListener('click', () => toggleAnnoncesCard(idx));
            palette.appendChild(el);
        }
    }
}

function initActionSelect() {
    const select = document.getElementById('annonces-action-select');
    select.innerHTML = '';
    const addOpt = (value, text) => {
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = text;
        select.appendChild(opt);
    };
    addOpt(0, 'Passe');
    for (let valIdx = 0; valIdx < 9; valIdx++) {
        const value = 80 + valIdx * 10;
        for (let suitIdx = 0; suitIdx < 4; suitIdx++) {
            addOpt(valIdx * 4 + suitIdx + 1, `${value} ${SUIT_SYMBOLS[suitIdx]}`);
        }
    }
    for (let suitIdx = 0; suitIdx < 4; suitIdx++) {
        addOpt(37 + suitIdx, `Capot ${SUIT_SYMBOLS[suitIdx]}`);
    }
    addOpt(41, 'Coinche');
    addOpt(42, 'Surcoinche');
}

function toggleAnnoncesCard(idx) {
    if (annoncesHand.has(idx)) {
        annoncesHand.delete(idx);
    } else {
        if (annoncesHand.size >= 8) return;
        annoncesHand.add(idx);
    }
    updateAnnoncesDisplay();
}

function updateAnnoncesDisplay() {
    const count = annoncesHand.size;
    const full = count === 8;
    for (let i = 0; i < 32; i++) {
        const el = document.getElementById(`annonces-card-${i}`);
        if (!el) continue;
        const selected = annoncesHand.has(i);
        el.classList.toggle('ann-selected', selected);
        el.classList.toggle('ann-faded', full && !selected);
    }
    document.getElementById('annonces-count').textContent = `${count}/8 cartes`;
    document.getElementById('annonces-eval-btn').disabled = count !== 8;

    const handEl = document.getElementById('annonces-hand-display');
    renderHand(handEl, Array.from(annoncesHand));
}

function renderAnnoncesHistory() {
    const list = document.getElementById('annonces-history-list');
    list.innerHTML = '';
    const n = annoncesHistory.length;

    annoncesHistory.forEach((action, i) => {
        const seat = annoncesPlayerSeat(i, n);
        const row = document.createElement('div');
        row.className = 'ann-history-row';

        const badge = document.createElement('span');
        badge.className = 'ann-seat-badge';
        badge.textContent = SEAT_NAMES[seat];
        badge.style.color = SEAT_COLORS[seat];

        const actionSpan = document.createElement('span');
        actionSpan.className = 'ann-action-name';
        actionSpan.innerHTML = bidActionHtml(action);

        const delBtn = document.createElement('button');
        delBtn.className = 'ann-del-btn';
        delBtn.textContent = '\u00d7';
        delBtn.title = 'Supprimer';
        delBtn.addEventListener('click', () => {
            annoncesHistory.splice(i, 1);
            renderAnnoncesHistory();
        });

        row.appendChild(badge);
        row.appendChild(actionSpan);
        row.appendChild(delBtn);
        list.appendChild(row);
    });

    const yourRow = document.createElement('div');
    yourRow.className = 'ann-history-row ann-your-turn';
    const yourBadge = document.createElement('span');
    yourBadge.className = 'ann-seat-badge';
    yourBadge.textContent = 'S';
    yourBadge.style.color = SEAT_COLORS[2];
    const yourLabel = document.createElement('span');
    yourLabel.className = 'ann-action-name';
    yourLabel.textContent = 'Votre tour';
    yourRow.appendChild(yourBadge);
    yourRow.appendChild(yourLabel);
    list.appendChild(yourRow);
}

function startDdTimer(numSims) {
    ddEstimatedMs = numSims * 150;
    ddStartTime = Date.now();
    const estSec = (ddEstimatedMs / 1000).toFixed(1);
    document.getElementById('annonces-dd-header').textContent = `Oracle DD`;
    document.getElementById('annonces-dd-body').innerHTML =
        `<div class="dd-loader">
            <div class="dd-loader-text">R\u00e9solution de ${numSims} donnes (~${estSec}s)\u2026</div>
            <div class="dd-progress-bar"><div class="dd-progress-fill" id="dd-progress-fill"></div></div>
            <div class="dd-loader-pct" id="dd-loader-pct">0%</div>
        </div>`;
    ddTimerId = setInterval(updateDdProgress, 100);
}

function updateDdProgress() {
    const elapsed = Date.now() - ddStartTime;
    const pct = Math.min(99, Math.round((elapsed / ddEstimatedMs) * 100));
    const fill = document.getElementById('dd-progress-fill');
    const label = document.getElementById('dd-loader-pct');
    if (fill) fill.style.width = pct + '%';
    if (label) label.textContent = pct + '%';
}

function stopDdTimer() {
    if (ddTimerId) {
        clearInterval(ddTimerId);
        ddTimerId = null;
    }
}

function ddSuggestedBid(avgNs) {
    const thresholds = [160, 150, 140, 130, 120, 110, 100, 90, 80];
    for (const t of thresholds) {
        if (avgNs >= t) return t;
    }
    return null;
}

function handleBidEvalResult(data) {
    if (data.error) {
        document.getElementById('annonces-results-body').innerHTML =
            `<div class="annonces-error">${data.error}</div>`;
        document.getElementById('annonces-results-header').textContent = 'Erreur';
        return;
    }

    const qValues = data.q_values.slice().sort((a, b) => b[1] - a[1]);
    const bestAction = data.best_action;
    const minQ = Math.min(...qValues.map(([, q]) => q));
    const maxQ = Math.max(...qValues.map(([, q]) => q));
    const range = maxQ - minQ || 1;

    document.getElementById('annonces-results-header').innerHTML =
        `Le Bide \u00e0 D\u00e9d\u00e9 : ${bidActionHtml(bestAction)}`;

    let html = '<div class="visit-bars ann-qvalues-scroll">';
    for (const [action, q] of qValues) {
        const pct = ((q - minQ) / range * 100).toFixed(0);
        const isBest = action === bestAction;
        const name = bidActionHtml(action);
        html += `<div class="visit-row${isBest ? ' best' : ''}">
            <span class="visit-name">${name}</span>
            <div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>
            <span class="visit-count">${q.toFixed(3)}</span>
        </div>`;
    }
    html += '</div>';
    document.getElementById('annonces-results-body').innerHTML = html;
}

function handleDdSimResult(data) {
    stopDdTimer();

    if (data.error) {
        document.getElementById('annonces-dd-body').innerHTML =
            `<div class="annonces-error">${data.error}</div>`;
        document.getElementById('annonces-dd-header').textContent = 'Erreur';
        return;
    }

    const suits = data.suits;
    const elapsed = data.elapsed_ms;
    const numSims = data.num_sims;

    document.getElementById('annonces-dd-header').textContent =
        `Oracle DD (${numSims} donnes, ${(elapsed / 1000).toFixed(1)}s)`;

    let html = '<table class="dd-sim-table"><tr><th>Atout</th><th>Moy. NS</th><th>Moy. EO</th><th>Annonce</th></tr>';
    for (const s of suits) {
        const symbol = suitHtml(s.suit);
        const bid = ddSuggestedBid(s.avg_ns);
        const bidText = bid ? `${bid} ${suitHtml(s.suit)}` : '\u2014';
        const bidClass = bid ? (bid >= 100 ? 'dd-bid-high' : 'dd-bid-ok') : 'dd-bid-none';
        html += `<tr>
            <td>${symbol}</td>
            <td>${s.avg_ns.toFixed(1)}</td>
            <td>${s.avg_ew.toFixed(1)}</td>
            <td class="${bidClass}">${bidText}</td>
        </tr>`;
    }
    html += '</table>';
    document.getElementById('annonces-dd-body').innerHTML = html;
}

export function mount(container) {
    container.innerHTML = TEMPLATE;

    annoncesHand = new Set();
    annoncesHistory = [];

    initAnnoncesGrid();
    initActionSelect();
    renderAnnoncesHistory();
    updateAnnoncesDisplay();

    // Event handlers
    document.getElementById('annonces-history-add-btn').addEventListener('click', () => {
        const select = document.getElementById('annonces-action-select');
        const action = parseInt(select.value);
        annoncesHistory.push(action);
        renderAnnoncesHistory();
    });

    document.getElementById('annonces-history-clear-btn').addEventListener('click', () => {
        annoncesHistory = [];
        renderAnnoncesHistory();
    });

    document.getElementById('annonces-random-btn').addEventListener('click', () => {
        const indices = Array.from({ length: 32 }, (_, i) => i);
        for (let i = 31; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        annoncesHand = new Set(indices.slice(0, 8));
        updateAnnoncesDisplay();
        document.getElementById('annonces-results-row').classList.add('hidden');
        stopDdTimer();
    });

    document.getElementById('annonces-clear-btn').addEventListener('click', () => {
        annoncesHand.clear();
        updateAnnoncesDisplay();
        document.getElementById('annonces-results-row').classList.add('hidden');
        stopDdTimer();
    });

    document.getElementById('annonces-eval-btn').addEventListener('click', () => {
        const hand = Array.from(annoncesHand);
        const numSims = Math.max(1, Math.min(200, parseInt(document.getElementById('annonces-sim-count').value) || 10));

        document.getElementById('annonces-results-row').classList.remove('hidden');
        document.getElementById('annonces-loading').classList.add('hidden');

        document.getElementById('annonces-results-header').textContent = 'Le Bide \u00e0 D\u00e9d\u00e9';
        document.getElementById('annonces-results-body').innerHTML =
            '<div class="dd-loader"><div class="dd-loader-text">Calcul\u2026</div></div>';

        startDdTimer(numSims);

        send({ type: 'bid_eval', hand, prior_actions: annoncesHistory });
        send({ type: 'dd_sim', hand, prior_actions: annoncesHistory, num_sims: numSims });
    });

    onMessage('bid_eval_result', handleBidEvalResult);
    onMessage('dd_sim_result', handleDdSimResult);
}

export function unmount() {
    offMessage('bid_eval_result', handleBidEvalResult);
    offMessage('dd_sim_result', handleDdSimResult);
    stopDdTimer();
    annoncesHand = new Set();
    annoncesHistory = [];
}
