// Annonces view — hand builder + bidding NN evaluation
// Supports local WASM computation (BidNet + Oracle) and server fallback.

import { send, onMessage, offMessage } from '../ws.js';
import { RANKS, SUITS, cardSvgPath, cardRank, cardSuit, renderHand, actionName, bidActionHtml, SUIT_DISPLAY_ORDER } from '../shared/cards.js';
import * as wasmBridge from '../wasm-bridge.js';
import * as xgbExplain from '../xgb-explain.js';

const SUIT_SYMBOLS = ['\u2660', '\u2665', '\u2666', '\u2663'];
const SEAT_NAMES = ['Nord', 'Est', 'Sud', 'Ouest'];
const SEAT_COLORS = ['#82cfff', '#82e0aa', '#d4af37', '#f0b429'];
const THRESHOLDS = [80, 90, 100, 110, 120, 130, 140, 150, 160, 162];
const THRESHOLD_LABELS = ['80', '90', '100', '110', '120', '130', '140', '150', '160', 'Cap'];

const SUIT_EMOJI = ['♠️', '♥️', '♦️', '♣️'];
function suitHtml(suitIdx) {
    return SUIT_EMOJI[suitIdx];
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
                <div class="annonces-toggle-wrap">
                    <span class="annonces-toggle-label" id="annonces-label-server">Serveur</span>
                    <label class="annonces-toggle">
                        <input type="checkbox" id="annonces-local-toggle">
                        <span class="annonces-toggle-track"></span>
                    </label>
                    <span class="annonces-toggle-label" id="annonces-label-local">Calcul local</span>
                </div>
            </div>
        </div>
        <div id="annonces-results-area" class="hidden">
            <div class="annonces-result-panel" id="annonces-nn-panel">
                <div id="annonces-results-header" class="section-title"></div>
                <p class="nn-explainer">R\u00e9seau de neurones entra\u00een\u00e9 par renforcement sur des millions de parties en jeu parfait (double-dummy).</p>
                <div id="annonces-results-body"></div>
                <div class="hidden" id="annonces-sim-viewer-wrap">
                    <details id="annonces-sim-viewer">
                        <summary>Voir 10 exemples de distribution</summary>
                        <div id="annonces-sim-viewer-content"></div>
                    </details>
                </div>
                <div class="hidden" id="annonces-xgb-panel">
                    <div id="annonces-xgb-header" class="section-title">
                        <span>Facteurs cl\u00e9s</span>
                        <select id="xgb-suit-select"></select>
                    </div>
                    <p class="xgb-explainer">Mod\u00e8le XGBoost distill\u00e9 du NN \u2014 il <em>approxime</em> les d\u00e9cisions du r\u00e9seau \u00e0 l\u2019aide de caract\u00e9ristiques interpr\u00e9tables. Ces contributions ne proviennent <strong>pas</strong> du r\u00e9seau de neurones.</p>
                    <div id="xgb-waterfall"></div>
                    <div id="xgb-probability"></div>
                </div>
            </div>
            <div class="annonces-result-panel" id="annonces-sim-panel">
                <div id="annonces-sim-header" class="section-title">
                    <span>Oracle</span>
                    <div id="oracle-progress" class="sim-progress hidden">
                        <div class="sim-progress-bar"><div class="sim-progress-fill"></div></div>
                        <span class="sim-progress-text"></span>
                    </div>
                </div>
                <p class="oracle-explainer">Des mains adverses al\u00e9atoires sont g\u00e9n\u00e9r\u00e9es et r\u00e9solues en jeu parfait (double-dummy). Chaque cellule indique le % de mondes o\u00f9 le contrat est r\u00e9alisable. C\u2019est un plafond th\u00e9orique\u00a0: en partie r\u00e9elle, le taux de r\u00e9ussite sera plus bas, mais cela permet de jauger le potentiel de la main.</p>
                <div id="annonces-sim-body"></div>
                <div class="hidden" id="annonces-doudou-panel">
                    <div id="annonces-doudou-header" class="section-title">
                        <span>DouDou50</span>
                        <div id="doudou-progress" class="sim-progress hidden">
                            <div class="sim-progress-bar"><div class="sim-progress-fill"></div></div>
                            <span class="sim-progress-text"></span>
                        </div>
                        <span id="doudou-stats-text"></span>
                    </div>
                    <p class="doudou-explainer">Distributions al\u00e9atoires jou\u00e9es en partie compl\u00e8te par le r\u00e9seau de neurones (ench\u00e8res NN + jeu DMC). Chaque cellule montre combien de fois ce contrat est ench\u00e9ri et le taux de r\u00e9ussite. La taille du chiffre refl\u00e8te la fiabilit\u00e9\u00a0: plus il y a d\u2019observations, plus le score est lisible. La couleur est d\u00e9termin\u00e9e par un intervalle de confiance (Wilson).</p>
                    <div id="annonces-doudou-body"></div>
                </div>
            </div>
        </div>
    </div>
</div>
`;

let annoncesHand = new Set();
let annoncesHistory = [];
let xgbResults = null; // cached XGB analysis results

function isLocalMode() {
    const toggle = document.getElementById('annonces-local-toggle');
    return toggle && toggle.checked;
}

function setLocalMode(on) {
    const toggle = document.getElementById('annonces-local-toggle');
    if (toggle) toggle.checked = on;
    updateToggleLabels();
    try { localStorage.setItem('annonces-local', on ? '1' : '0'); } catch(e) {}
}

function updateToggleLabels() {
    const local = isLocalMode();
    const labelLocal = document.getElementById('annonces-label-local');
    const labelServer = document.getElementById('annonces-label-server');
    if (labelLocal) labelLocal.classList.toggle('active', local);
    if (labelServer) labelServer.classList.toggle('active', !local);
}

function annoncesPlayerSeat(turnIdx, historyLen) {
    return (2 - historyLen + turnIdx + 32) % 4;
}

function initAnnoncesGrid() {
    const palette = document.getElementById('annonces-palette');
    palette.innerHTML = '';
    for (const suit of SUIT_DISPLAY_ORDER) {
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
    yourBadge.textContent = 'Sud';
    yourBadge.style.color = SEAT_COLORS[2];
    const yourLabel = document.createElement('span');
    yourLabel.className = 'ann-action-name';
    yourLabel.textContent = 'Votre tour';
    yourRow.appendChild(yourBadge);
    yourRow.appendChild(yourLabel);
    list.appendChild(yourRow);
}

// ── XGBoost interpretability ──

function renderXgbWaterfall(result) {
    const container = document.getElementById('xgb-waterfall');
    const probEl = document.getElementById('xgb-probability');
    if (!container || !result) return;

    // Sort contributions by absolute value, descending
    const entries = Object.entries(result.contributions)
        .filter(([, v]) => Math.abs(v) > 0.001)
        .sort((a, b) => Math.abs(b[1]) - Math.abs(a[1]));

    if (entries.length === 0) {
        container.innerHTML = '<div class="xgb-empty">Pas assez de donn\u00e9es</div>';
        probEl.innerHTML = '';
        return;
    }

    const maxAbs = Math.max(...entries.map(([, v]) => Math.abs(v)));

    let html = '<div class="xgb-waterfall-chart">';
    for (const [feat, val] of entries) {
        const label = xgbExplain.featureLabel(feat);
        const featVal = result.features[feat];
        const pct = Math.abs(val) / maxAbs * 100;
        const isPos = val > 0;
        const cls = isPos ? 'xgb-bar-pos' : 'xgb-bar-neg';
        const sign = isPos ? '+' : '';
        const valDisplay = featVal !== undefined ? ` = ${featVal}` : '';

        html += `<div class="xgb-row">
            <span class="xgb-feat-name" title="${feat}${valDisplay}">${label}<span class="xgb-feat-val">${valDisplay}</span></span>
            <div class="xgb-bar-wrap">
                <div class="xgb-bar ${cls}" style="width:${pct.toFixed(0)}%"></div>
            </div>
            <span class="xgb-contrib">${sign}${val.toFixed(3)}</span>
        </div>`;
    }
    html += '</div>';
    container.innerHTML = html;

    // Show probability
    const pct = (result.probability * 100).toFixed(0);
    const cls = result.probability >= 0.5 ? 'xgb-prob-high' : 'xgb-prob-low';
    probEl.innerHTML = `<span class="${cls}">Probabilit\u00e9 d\u2019ench\u00e9rir : ${pct}%</span>`;
}

function populateXgbSuitSelect(results) {
    const select = document.getElementById('xgb-suit-select');
    if (!select || !results) return;
    select.innerHTML = '';
    for (let i = 0; i < results.length; i++) {
        const r = results[i];
        const opt = document.createElement('option');
        opt.value = i;
        opt.innerHTML = `${SUIT_SYMBOLS[r.suit]} (${(r.probability * 100).toFixed(0)}%)`;
        select.appendChild(opt);
    }
    select.value = '0';
}

async function runXgbAnalysis(hand, qValues) {
    try {
        const results = await xgbExplain.analyzeAllSuits(hand, annoncesHistory, qValues);
        if (!results) return;
        xgbResults = results;

        const panel = document.getElementById('annonces-xgb-panel');
        panel.classList.remove('hidden');

        populateXgbSuitSelect(results);
        renderXgbWaterfall(results[0]);
    } catch (err) {
        console.warn('[xgb] Analysis failed:', err);
    }
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
        `Bid \u00e0 D\u00e9d\u00e9 : ${bidActionHtml(bestAction)}`;

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

    // Trigger XGB interpretability analysis
    const hand = Array.from(annoncesHand);
    if (hand.length === 8) {
        runXgbAnalysis(hand, data.q_values);
    }
}

function updateProgressBar(id, completed, total, elapsedMs) {
    const wrap = document.getElementById(id);
    wrap.classList.remove('hidden');
    const fill = wrap.querySelector('.sim-progress-fill');
    const text = wrap.querySelector('.sim-progress-text');
    const pct = total > 0 ? Math.round(completed / total * 100) : 0;
    fill.style.width = `${pct}%`;
    const elapsed = elapsedMs != null ? ` \u2014 ${(elapsedMs / 1000).toFixed(1)}s` : '';
    text.textContent = `${completed}/${total}${elapsed}`;
    if (completed >= total) {
        wrap.classList.add('done');
    } else {
        wrap.classList.remove('done');
    }
}

function renderOracleTable(successCounts, completed, total, elapsedMs) {
    updateProgressBar('oracle-progress', completed, total, elapsedMs);

    const body = document.getElementById('annonces-sim-body');

    // Build header row
    let html = '<table id="oracle-table"><thead><tr><th></th>';
    for (const label of THRESHOLD_LABELS) {
        html += `<th>${label}</th>`;
    }
    html += '</tr></thead><tbody>';

    // 4 suit rows
    for (const suit of SUIT_DISPLAY_ORDER) {
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

const DOUDOU_COLS = ['80', '90', '100', '110', '120', '130', '140', '150', '160', 'Cap'];

// Wilson score lower bound (z=1.645 for 90% confidence).
// Returns a value in [0, 1] — usable as a "meaningful winrate" that
// penalises small sample sizes.
function wilsonLower(successes, n) {
    if (n === 0) return 0;
    const z = 1.645;
    const p = successes / n;
    const denom = 1 + z * z / n;
    const centre = p + z * z / (2 * n);
    const spread = z * Math.sqrt((p * (1 - p) + z * z / (4 * n)) / n);
    return Math.max(0, (centre - spread) / denom);
}

// Font-size scale: ranges from 0.65rem (1 obs) to 0.85rem (≥20 obs).
function pctFontSize(count) {
    const t = Math.min(count, 20) / 20;          // 0 → 1
    return (0.65 + t * 0.20).toFixed(2);          // rem
}

function renderDoudouTable(doudouCells, doudouStats, completed, total, elapsedMs) {
    const panel = document.getElementById('annonces-doudou-panel');
    if (!doudouCells) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');

    updateProgressBar('doudou-progress', completed, total, elapsedMs);

    const v = doudouStats.voids;
    const nsC = doudouStats.ns_contracts;
    const nsA = doudouStats.ns_achieved;
    const ewC = doudouStats.ew_contracts;
    const ewA = doudouStats.ew_achieved;
    const nsPct = nsC > 0 ? Math.round(nsA / nsC * 100) : 0;
    const ewPct = ewC > 0 ? Math.round(ewA / ewC * 100) : 0;
    document.getElementById('doudou-stats-text').textContent =
        `${v} passe, NS ${nsPct}% (${nsA}/${nsC}), EW ${ewPct}% (${ewA}/${ewC})`;

    const body = document.getElementById('annonces-doudou-body');
    let html = '<table id="doudou-table"><thead><tr><th></th>';
    for (const label of DOUDOU_COLS) {
        html += `<th>${label}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (const suit of SUIT_DISPLAY_ORDER) {
        html += `<tr><td>${suitHtml(suit)}</td>`;
        for (let col = 0; col < 10; col++) {
            const [count, achieved] = doudouCells[suit][col];
            if (count === 0) {
                html += '<td class="doudou-empty">\u00b7</td>';
            } else {
                const pct = Math.round(achieved / count * 100);
                const wlb = Math.round(wilsonLower(achieved, count) * 100);
                let cls;
                if (wlb >= 60) cls = 'doudou-high';
                else if (wlb >= 30) cls = 'doudou-mid';
                else cls = 'doudou-low';
                const fs = pctFontSize(count);
                html += `<td class="${cls}"><span class="doudou-count">${count}</span><span class="doudou-pct" style="font-size:${fs}rem">${pct}</span></td>`;
            }
        }
        html += '</tr>';
    }
    html += '</tbody></table>';
    body.innerHTML = html;
}

function handleSimUpdate(data) {
    if (data.error) {
        document.getElementById('annonces-sim-body').innerHTML =
            `<div class="annonces-error">${data.error}</div>`;
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

// --- DouDou-only server handlers (used in local mode for DouDou part) ---

function handleDoudouUpdate(data) {
    if (data.error) return; // Silently ignore — DouDou is optional in local mode
    renderDoudouTable(data.doudou_cells, data.doudou_stats, data.completed, data.total, data.elapsed_ms);
}

function handleDoudouDone(data) {
    renderDoudouTable(data.doudou_cells, data.doudou_stats, data.completed, data.total, data.elapsed_ms);
}

// --- Eval paths ---

function resetPanels(numSims) {
    document.getElementById('annonces-results-area').classList.remove('hidden');
    document.getElementById('annonces-results-header').textContent = 'Bid \u00e0 D\u00e9d\u00e9';
    document.getElementById('annonces-results-body').innerHTML =
        '<div class="dd-loader"><div class="dd-loader-text">Calcul\u2026</div></div>';
    // Reset XGB panel
    document.getElementById('annonces-xgb-panel').classList.add('hidden');
    xgbResults = null;
    document.getElementById('annonces-sim-viewer-wrap').classList.add('hidden');
    const emptyCountsSeed = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
    renderOracleTable(emptyCountsSeed, 0, numSims, null);
    // Reset DouDou panel — show it but with empty state
    const doudouPanel = document.getElementById('annonces-doudou-panel');
    doudouPanel.classList.remove('hidden');
    document.getElementById('annonces-doudou-body').innerHTML = '';
    document.getElementById('doudou-stats-text').textContent = '';
    const dp = document.getElementById('doudou-progress');
    dp.classList.add('hidden');
    dp.classList.remove('done');
    dp.querySelector('.sim-progress-fill').style.width = '0%';
    dp.querySelector('.sim-progress-text').textContent = '';
}

async function evalLocal(hand, numSims) {
    try {
        await wasmBridge.ensureReady();
    } catch (err) {
        console.warn('[annonces] WASM init failed, falling back to server:', err);
        setLocalMode(false);
        evalServer(hand, numSims);
        return;
    }

    // 1. BidNet eval (main thread, sub-ms)
    try {
        const result = wasmBridge.evaluateBid(hand, annoncesHistory);
        handleBidEvalResult(result);
    } catch (err) {
        handleBidEvalResult({ error: `WASM BidNet: ${err.message || err}` });
    }

    // 2. Oracle via Worker (streaming)
    wasmBridge.runOracleSim(hand, numSims,
        (data) => {
            renderOracleTable(data.success_counts, data.completed, data.total, data.elapsed_ms);
        },
        (data) => {
            if (data.error) {
                document.getElementById('annonces-sim-body').innerHTML =
                    `<div class="annonces-error">${data.error}</div>`;
                return;
            }
            renderOracleTable(data.success_counts, data.completed, data.total, data.elapsed_ms);
            if (data.sampled_deals && data.sampled_deals.length > 0) {
                renderSimViewer(data.sampled_deals, data.completed);
            }
        }
    );

    // 3. DouDou via WebSocket (server-side, needs DMC model)
    send({ type: 'annonces_doudou', hand, prior_actions: annoncesHistory, num_sims: numSims });
}

function evalServer(hand, numSims) {
    send({ type: 'bid_eval', hand, prior_actions: annoncesHistory });
    send({ type: 'annonces_sim', hand, prior_actions: annoncesHistory, num_sims: numSims });
}

export function mount(container) {
    container.innerHTML = TEMPLATE;

    annoncesHand = new Set();
    annoncesHistory = [];

    initAnnoncesGrid();
    initActionSelect();
    renderAnnoncesHistory();
    updateAnnoncesDisplay();

    // Restore toggle state from localStorage
    const toggle = document.getElementById('annonces-local-toggle');
    const stored = localStorage.getItem('annonces-local');
    toggle.checked = stored !== '0'; // default on
    updateToggleLabels();
    toggle.addEventListener('change', updateToggleLabels);

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

        // Cancel any previous simulation before starting a new one
        wasmBridge.cancelOracle();

        resetPanels(numSims);

        if (isLocalMode()) {
            evalLocal(hand, numSims);
        } else {
            evalServer(hand, numSims);
        }
    });

    // XGB suit dropdown
    document.getElementById('xgb-suit-select').addEventListener('change', (e) => {
        if (xgbResults) {
            renderXgbWaterfall(xgbResults[parseInt(e.target.value)]);
        }
    });

    onMessage('bid_eval_result', handleBidEvalResult);
    onMessage('annonces_sim_update', handleSimUpdate);
    onMessage('annonces_sim_done', handleSimDone);
    onMessage('annonces_doudou_update', handleDoudouUpdate);
    onMessage('annonces_doudou_done', handleDoudouDone);
}

export function unmount() {
    offMessage('bid_eval_result', handleBidEvalResult);
    offMessage('annonces_sim_update', handleSimUpdate);
    offMessage('annonces_sim_done', handleSimDone);
    offMessage('annonces_doudou_update', handleDoudouUpdate);
    offMessage('annonces_doudou_done', handleDoudouDone);
    wasmBridge.cancelOracle();
    annoncesHand = new Set();
    annoncesHistory = [];
    xgbResults = null;
}
