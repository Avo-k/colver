// Annonces view — hand builder + bidding NN evaluation

import { send, onMessage, offMessage } from '../ws.js';
import { RANKS, SUITS, cardSvgPath, cardRank, cardSuit, renderHand, actionName } from '../shared/cards.js';

const SUIT_SYMBOLS = ['\u2660', '\u2665', '\u2666', '\u2663'];
const SEAT_NAMES = ['N', 'E', 'S', 'O'];
const SEAT_COLORS = ['#82cfff', '#82e0aa', '#d4af37', '#f0b429'];
const THRESHOLDS = [80, 90, 100, 110, 120, 130, 140, 150, 160, 162];
const THRESHOLD_LABELS = ['80', '90', '100', '110', '120', '130', '140', '150', '160', 'Cap'];

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
<div id="annonces-top-row">
    <div id="annonces-left">
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
        </div>
    </div>
    <div id="annonces-right">
        <div id="annonces-hand-preview">
            <div class="section-title">Votre main</div>
            <div class="hand" id="annonces-hand-display"></div>
            <div id="annonces-eval-row">
                <button id="annonces-eval-btn" disabled>\u00c9valuer</button>
                <label class="annonces-sim-label">Simulations :
                    <input type="number" id="annonces-sim-count" value="200" min="1" max="1000" style="width:55px">
                </label>
            </div>
        </div>
        <div id="annonces-results-area" class="hidden">
            <div class="annonces-result-panel" id="annonces-nn-panel">
                <div id="annonces-results-header" class="section-title"></div>
                <p class="nn-explainer">R\u00e9seau de neurones entra\u00een\u00e9 par renforcement sur des millions de parties en jeu parfait (double-dummy).</p>
                <div id="annonces-results-body"></div>
            </div>
            <div class="annonces-result-panel" id="annonces-sim-panel">
                <div id="annonces-sim-header" class="section-title">Oracle</div>
                <p class="oracle-explainer">Des mains adverses al\u00e9atoires sont g\u00e9n\u00e9r\u00e9es et r\u00e9solues en jeu parfait (double-dummy). Chaque cellule indique le % de mondes o\u00f9 le contrat est r\u00e9alisable. C\u2019est un plafond th\u00e9orique\u00a0: en partie r\u00e9elle, le taux de r\u00e9ussite sera plus bas, mais cela permet de jauger le potentiel de la main.</p>
                <div id="annonces-sim-body"></div>
                <div class="hidden" id="annonces-sim-viewer-wrap">
                    <details id="annonces-sim-viewer">
                        <summary>Voir 10 exemples de distribution</summary>
                        <div id="annonces-sim-viewer-content"></div>
                    </details>
                </div>
            </div>
        </div>
    </div>
</div>
`;

let annoncesHand = new Set();
let annoncesHistory = [];

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

function renderOracleTable(successCounts, completed, total, elapsedMs) {
    const header = document.getElementById('annonces-sim-header');
    const elapsed = elapsedMs != null ? `, ${(elapsedMs / 1000).toFixed(1)}s` : '';
    header.textContent = `Oracle (${completed}/${total} donnes${elapsed})`;

    const body = document.getElementById('annonces-sim-body');

    // Build header row
    let html = '<table id="oracle-table"><thead><tr><th></th>';
    for (const label of THRESHOLD_LABELS) {
        html += `<th>${label}</th>`;
    }
    html += '</tr></thead><tbody>';

    // 4 suit rows
    for (let suit = 0; suit < 4; suit++) {
        html += `<tr><td>${suitHtml(suit)}</td>`;
        for (let t = 0; t < THRESHOLDS.length; t++) {
            const count = successCounts[suit][t];
            const pct = completed > 0 ? Math.round(count / completed * 100) : 0;
            let cls;
            if (pct === 0) cls = 'oracle-zero';
            else if (pct >= 60) cls = 'oracle-high';
            else if (pct >= 30) cls = 'oracle-mid';
            else cls = 'oracle-low';
            html += `<td class="${cls}">${pct}</td>`;
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    body.innerHTML = html;
}

function renderSimViewer(deals, numSims) {
    const wrap = document.getElementById('annonces-sim-viewer-wrap');
    wrap.classList.remove('hidden');
    const content = document.getElementById('annonces-sim-viewer-content');
    const viewer = document.getElementById('annonces-sim-viewer');

    // Show at most 10 randomly sampled deals
    let sampled = deals;
    if (deals.length > 10) {
        const indices = Array.from({ length: deals.length }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        sampled = indices.slice(0, 10).sort((a, b) => a - b).map(i => deals[i]);
    }

    const shown = sampled.length;
    viewer.querySelector('summary').textContent = `Voir 10 exemples de distribution`;

    let html = '';
    for (let d = 0; d < sampled.length; d++) {
        const deal = sampled[d];
        html += `<details class="sim-deal-details">
            <summary>Donne ${d + 1}</summary>
            <div class="sim-deal-hands">`;
        for (const seat of [0, 1, 3]) {
            const cards = deal[String(seat)];
            if (!cards) continue;
            html += `<div class="sim-hand-section">
                <span class="sim-hand-label">${SEAT_NAMES[seat]}</span>
                <div class="hand sim-hand" id="sim-hand-${d}-${seat}"></div>
            </div>`;
        }
        html += '</div></details>';
    }
    content.innerHTML = html;

    // Render card images into each sim hand container
    for (let d = 0; d < sampled.length; d++) {
        const deal = sampled[d];
        for (const seat of [0, 1, 3]) {
            const cards = deal[String(seat)];
            if (!cards) continue;
            const el = document.getElementById(`sim-hand-${d}-${seat}`);
            if (el) renderHand(el, cards);
        }
    }
}

function handleSimUpdate(data) {
    if (data.error) {
        document.getElementById('annonces-sim-body').innerHTML =
            `<div class="annonces-error">${data.error}</div>`;
        document.getElementById('annonces-sim-header').textContent = 'Erreur';
        return;
    }
    renderOracleTable(data.success_counts, data.completed, data.total, data.elapsed_ms);
}

function handleSimDone(data) {
    renderOracleTable(data.success_counts, data.completed, data.total, data.elapsed_ms);
    if (data.sampled_deals && data.sampled_deals.length > 0) {
        renderSimViewer(data.sampled_deals, data.completed);
    }
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
        document.getElementById('annonces-results-area').classList.add('hidden');
    });

    document.getElementById('annonces-clear-btn').addEventListener('click', () => {
        annoncesHand.clear();
        updateAnnoncesDisplay();
        document.getElementById('annonces-results-area').classList.add('hidden');
    });

    document.getElementById('annonces-eval-btn').addEventListener('click', () => {
        const hand = Array.from(annoncesHand);
        const numSims = Math.max(1, Math.min(1000, parseInt(document.getElementById('annonces-sim-count').value) || 200));

        document.getElementById('annonces-results-area').classList.remove('hidden');

        // Reset NN panel
        document.getElementById('annonces-results-header').textContent = 'Le Bide \u00e0 D\u00e9d\u00e9';
        document.getElementById('annonces-results-body').innerHTML =
            '<div class="dd-loader"><div class="dd-loader-text">Calcul\u2026</div></div>';

        // Reset oracle panel with empty table
        document.getElementById('annonces-sim-viewer-wrap').classList.add('hidden');
        const emptyCountsSeed = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                                 [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
        renderOracleTable(emptyCountsSeed, 0, numSims, null);

        send({ type: 'bid_eval', hand, prior_actions: annoncesHistory });
        send({ type: 'annonces_sim', hand, prior_actions: annoncesHistory, num_sims: numSims });
    });

    onMessage('bid_eval_result', handleBidEvalResult);
    onMessage('annonces_sim_update', handleSimUpdate);
    onMessage('annonces_sim_done', handleSimDone);
}

export function unmount() {
    offMessage('bid_eval_result', handleBidEvalResult);
    offMessage('annonces_sim_update', handleSimUpdate);
    offMessage('annonces_sim_done', handleSimDone);
    annoncesHand = new Set();
    annoncesHistory = [];
}
