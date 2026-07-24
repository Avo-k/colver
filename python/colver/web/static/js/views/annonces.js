// Annonces view — hand builder + bidding NN evaluation
// Supports local WASM computation (BidNet + Oracle) and server fallback.

import { send, onMessage, offMessage } from '../ws.js';
import { RANKS, SUITS, cardSvgPath, renderHand, renderHandMini, actionName, bidActionHtml, SUIT_DISPLAY_ORDER, cardCode, parseCardToken } from '../shared/cards.js';
import { suitHtml, createSuitPicker } from '../shared/suits.js';
import { SEAT_COLOR_VARS } from '../shared/seats.js';
import * as wasmBridge from '../wasm-bridge.js';
import * as xgbExplain from '../xgb-explain.js';

const SEAT_NAMES = ['Nord', 'Est', 'Sud', 'Ouest'];
const THRESHOLDS = [80, 90, 100, 110, 120, 130, 140, 150, 160, 162];
const THRESHOLD_LABELS = ['80', '90', '100', '110', '120', '130', '140', '150', '160', 'Cap'];


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
                    <span id="annonces-action-select"></span>
                    <button id="annonces-history-add-btn">+ Ajouter</button>
                </div>
            </div>
        </div>
        <div class="annonces-result-panel hidden" id="annonces-nn-panel">
            <div id="annonces-results-header" class="section-title"></div>
            <p class="nn-explainer">R\u00e9seau de neurones entra\u00een\u00e9 par renforcement sur des millions de parties en jeu parfait (double-dummy).</p>
            <div id="annonces-results-body"></div>
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
    </div>
    <div id="annonces-right">
        <div id="annonces-hand-preview">
            <div class="section-title">Votre main</div>
            <div class="hand" id="annonces-hand-display"></div>
            <div id="annonces-eval-row">
                <button id="annonces-eval-btn" disabled>\u00c9valuer</button>
            </div>
        </div>
        <div id="annonces-verdict" class="hidden">
            <span class="verdict-label">Dans cette situation, Bid V6 joue</span>
            <span id="annonces-verdict-action"></span>
            <span class="verdict-alt">
                <label for="annonces-alt-select" class="verdict-alt-label">Analyser une autre annonce :</label>
                <span id="annonces-alt-select"></span>
                <button id="annonces-alt-btn" class="secondary-btn">Analyser</button>
                <label class="annonces-sim-label">Simulations :
                    <input type="number" id="annonces-sim-count" value="200" min="1" max="1000" style="width:55px">
                </label>
            </span>
            <span id="annonces-alt-status" class="hidden"></span>
        </div>
        <div id="annonces-results-area" class="hidden">
            <div class="annonces-result-panel hidden" id="annonces-doudou-panel">
                <div id="annonces-doudou-header" class="section-title">
                    <span>DouDou50<span id="doudou-forced-label"></span></span>
                    <div id="doudou-progress" class="sim-progress hidden">
                        <div class="sim-progress-bar"><div class="sim-progress-fill"></div></div>
                        <span class="sim-progress-text"></span>
                    </div>
                    <span id="doudou-stats-text"></span>
                </div>
                <p class="doudou-explainer">Distributions al\u00e9atoires jou\u00e9es en partie compl\u00e8te par le r\u00e9seau de neurones (ench\u00e8res NN + jeu DMC). Chaque cellule montre combien de fois ce contrat est ench\u00e9ri et le taux de r\u00e9ussite. La taille du chiffre refl\u00e8te la fiabilit\u00e9\u00a0: plus il y a d\u2019observations, plus le score est lisible. La couleur est d\u00e9termin\u00e9e par un intervalle de confiance (Wilson).</p>
                <div id="annonces-doudou-body"></div>
            </div>
            <div class="annonces-result-panel" id="annonces-oracle-panel">
                <div id="annonces-sim-header" class="section-title">
                    <span>Oracle</span>
                    <div id="oracle-progress" class="sim-progress hidden">
                        <div class="sim-progress-bar"><div class="sim-progress-fill"></div></div>
                        <span class="sim-progress-text"></span>
                    </div>
                </div>
                <p class="oracle-explainer">Des mains adverses al\u00e9atoires sont g\u00e9n\u00e9r\u00e9es et r\u00e9solues en jeu parfait (double-dummy). Chaque cellule indique le % de mondes o\u00f9 le contrat est r\u00e9alisable. C\u2019est un plafond th\u00e9orique\u00a0: en partie r\u00e9elle, le taux de r\u00e9ussite sera plus bas, mais cela permet de jauger le potentiel de la main.</p>
                <div id="annonces-sim-body"></div>
            </div>
        </div>
        <div class="annonces-result-panel hidden" id="annonces-sim-viewer-wrap">
            <details id="annonces-sim-viewer">
                <summary>Voir 10 exemples de distribution</summary>
                <div id="annonces-sim-viewer-content"></div>
            </details>
        </div>
    </div>
</div>
`;

let annoncesHand = new Set();
let annoncesHistory = [];
let xgbResults = null; // cached XGB analysis results
let forcedAction = null; // alternative bid being analysed (null = Bid V6's own choice)
let actionSelector = null; // paired bid selector for the history-add row
let altSelector = null;    // paired bid selector for "analyser une autre annonce"

// Keep the URL in sync with the current hand/history, hand as two-char card
// codes ("7S,KH,...") rather than raw indices.
function syncUrl() {
    const parts = [];
    if (annoncesHand.size > 0) {
        parts.push('hand=' + Array.from(annoncesHand).sort((a, b) => a - b).map(cardCode).join(','));
    }
    if (annoncesHistory.length > 0) {
        parts.push('history=' + annoncesHistory.join(','));
    }
    history.replaceState(null, '', window.location.pathname + (parts.length ? '?' + parts.join('&') : ''));
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

// Build a paired bid selector (niveau + couleur) inside `container`.
// Levels: Passe · 80…160 · Capot · Coinche · Surcoinche. The suit dropdown is
// only relevant for a numeric value or Capot, and is disabled for the others.
// Returns { read(): actionCode, set(actionCode): void }.
function buildBidSelector(container) {
    container.innerHTML = '';
    container.classList.add('bid-selector');

    const levelSel = document.createElement('select');
    levelSel.className = 'bid-level-select';
    // Segmented control : les <option> ne portent pas de couleur de façon
    // portable, d'où les emoji qu'on utilisait ici. Plus besoin.
    const suitSel = createSuitPicker({ value: 0, name: 'atout' });
    suitSel.classList.add('bid-suit-select');

    const addOpt = (sel, value, text, color) => {
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = text;
        if (color) opt.style.color = color;
        sel.appendChild(opt);
    };

    addOpt(levelSel, 'pass', 'Passe');
    for (let valIdx = 0; valIdx < 9; valIdx++) {
        addOpt(levelSel, String(valIdx), String(80 + valIdx * 10));
    }
    addOpt(levelSel, 'capot', 'Capot');
    addOpt(levelSel, 'coinche', 'Coinche');
    addOpt(levelSel, 'surcoinche', 'Surcoinche');

    const isSpecial = (v) => v === 'pass' || v === 'coinche' || v === 'surcoinche';
    const sync = () => { suitSel.disabled = isSpecial(levelSel.value); };
    levelSel.addEventListener('change', sync);
    sync();

    container.appendChild(levelSel);
    container.appendChild(suitSel);

    return {
        read() {
            const lvl = levelSel.value;
            if (lvl === 'pass') return 0;
            if (lvl === 'coinche') return 41;
            if (lvl === 'surcoinche') return 42;
            const suit = parseInt(suitSel.value);
            if (lvl === 'capot') return 37 + suit;
            return parseInt(lvl) * 4 + suit + 1;
        },
        set(action) {
            if (action === 0) { levelSel.value = 'pass'; }
            else if (action === 41) { levelSel.value = 'coinche'; }
            else if (action === 42) { levelSel.value = 'surcoinche'; }
            else if (action >= 37 && action <= 40) {
                levelSel.value = 'capot';
                suitSel.value = String(action - 37);
            } else if (action >= 1 && action <= 36) {
                const a = action - 1;
                levelSel.value = String(Math.floor(a / 4));
                suitSel.value = String(a % 4);
            }
            sync();
        },
    };
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
    syncUrl();
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
        badge.style.color = SEAT_COLOR_VARS[seat];

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
    yourBadge.style.color = SEAT_COLOR_VARS[2];
    const yourLabel = document.createElement('span');
    yourLabel.className = 'ann-action-name';
    yourLabel.textContent = 'Votre tour';
    yourRow.appendChild(yourBadge);
    yourRow.appendChild(yourLabel);
    list.appendChild(yourRow);
    syncUrl();
}

// ── XGBoost interpretability ──

let xgbExpanded = false;

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

    let html = `<div class="xgb-waterfall-chart${xgbExpanded ? '' : ' ann-collapsed'}" id="xgb-chart">`;
    entries.forEach(([feat, val], i) => {
        const label = xgbExplain.featureLabel(feat);
        const featVal = result.features[feat];
        const pct = Math.abs(val) / maxAbs * 100;
        const isPos = val > 0;
        const cls = isPos ? 'xgb-bar-pos' : 'xgb-bar-neg';
        const sign = isPos ? '+' : '';
        const valDisplay = featVal !== undefined ? ` = ${featVal}` : '';

        html += `<div class="xgb-row${i >= 5 ? ' ann-extra' : ''}">
            <span class="xgb-feat-name" title="${feat}${valDisplay}">${label}<span class="xgb-feat-val">${valDisplay}</span></span>
            <div class="xgb-bar-wrap">
                <div class="xgb-bar ${cls}" style="width:${pct.toFixed(0)}%"></div>
            </div>
            <span class="xgb-contrib">${sign}${val.toFixed(3)}</span>
        </div>`;
    });
    html += '</div>';
    if (entries.length > 5) {
        html += `<button class="ann-see-more" id="xgb-more">${xgbExpanded ? 'Voir moins' : `Voir plus (${entries.length - 5})`}</button>`;
    }
    container.innerHTML = html;
    const xgbMoreBtn = document.getElementById('xgb-more');
    if (xgbMoreBtn) {
        xgbMoreBtn.addEventListener('click', () => {
            xgbExpanded = !xgbExpanded;
            document.getElementById('xgb-chart').classList.toggle('ann-collapsed', !xgbExpanded);
            xgbMoreBtn.textContent = xgbExpanded ? 'Voir moins' : `Voir plus (${entries.length - 5})`;
        });
    }

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
        opt.innerHTML = `${SUITS[r.suit]} (${(r.probability * 100).toFixed(0)}%)`;
        select.appendChild(opt);
    }
    select.value = '0';
}

async function runXgbAnalysis(hand, qValues) {
    try {
        const results = await xgbExplain.analyzeAllSuits(hand, annoncesHistory, qValues);
        if (!results) return;
        xgbResults = results;
        xgbExpanded = false;

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
        document.getElementById('annonces-verdict').classList.add('hidden');
        return;
    }

    const qValues = data.q_values.slice().sort((a, b) => b[1] - a[1]);
    const bestAction = data.best_action;
    const minQ = Math.min(...qValues.map(([, q]) => q));
    const maxQ = Math.max(...qValues.map(([, q]) => q));
    const range = maxQ - minQ || 1;

    document.getElementById('annonces-verdict').classList.remove('hidden');
    document.getElementById('annonces-verdict-action').innerHTML = bidActionHtml(bestAction);
    if (altSelector) altSelector.set(bestAction);

    document.getElementById('annonces-results-header').innerHTML =
        `Bid V6 : ${bidActionHtml(bestAction)}`;

    let html = '<div class="visit-bars ann-qvalues-scroll ann-collapsed" id="ann-qvalues">';
    qValues.forEach(([action, q], i) => {
        const pct = ((q - minQ) / range * 100).toFixed(0);
        const isBest = action === bestAction;
        const name = bidActionHtml(action);
        html += `<div class="visit-row${isBest ? ' best' : ''}${i >= 5 ? ' ann-extra' : ''}">
            <span class="visit-name">${name}</span>
            <div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>
            <span class="visit-count">${q.toFixed(3)}</span>
        </div>`;
    });
    html += '</div>';
    if (qValues.length > 5) {
        html += `<button class="ann-see-more" id="ann-qvalues-more">Voir plus (${qValues.length - 5})</button>`;
    }
    if (data.playgen_policy && data.playgen_policy.length) {
        const pol = data.playgen_policy.slice().sort((a, b) => b[1] - a[1]).slice(0, 5);
        const polBest = pol[0][0];
        html += '<div class="oracle-variant-label">Playgen v2 <span class="oracle-quant-sub">p(annonce)</span></div>';
        html += '<div class="visit-bars">';
        pol.forEach(([action, p]) => {
            const pct = (p * 100).toFixed(1);
            html += `<div class="visit-row${action === polBest ? ' best' : ''}">
                <span class="visit-name">${bidActionHtml(action)}</span>
                <div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${Math.max(2, p * 100)}%"></div></div>
                <span class="visit-count">${pct}%</span>
            </div>`;
        });
        html += '</div>';
    }
    document.getElementById('annonces-results-body').innerHTML = html;
    const moreBtn = document.getElementById('ann-qvalues-more');
    if (moreBtn) {
        moreBtn.addEventListener('click', () => {
            const list = document.getElementById('ann-qvalues');
            const collapsed = list.classList.toggle('ann-collapsed');
            moreBtn.textContent = collapsed ? `Voir plus (${qValues.length - 5})` : 'Voir moins';
        });
    }

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

// Success % → strip cell color. Lightness rises monotonically with pct so the
// strip stays readable for colorblind users; hue sweeps red → green.
// ≤5%: near-black — this contract essentially never makes it.
function oraclePctColor(pct) {
    if (pct <= 5) return 'hsl(0, 0%, 10%)';
    return `hsl(${8 + 1.32 * pct}, 58%, ${24 + 0.32 * pct}%)`;
}

// Highest threshold index with pct >= level, or -1.
function oracleCrossing(pcts, level) {
    let idx = -1;
    for (let t = 0; t < pcts.length; t++) {
        if (pcts[t] >= level) idx = t;
    }
    return idx;
}

const ORACLE_MARKER_LEVELS = [80, 50, 20];

function oraclePcts(successCounts, suit, completed) {
    return THRESHOLDS.map((_, t) =>
        completed > 0 ? Math.round(successCounts[suit][t] / completed * 100) : 0);
}

function renderOracleStrips(successCounts, completed) {
    let html = '<div id="oracle-strips"><div class="oracle-strip-header"><span></span>';
    for (const label of THRESHOLD_LABELS) {
        html += `<span>${label}</span>`;
    }
    html += '</div>';
    for (const suit of SUIT_DISPLAY_ORDER) {
        const pcts = oraclePcts(successCounts, suit, completed);
        html += `<div class="oracle-strip-row"><span class="oracle-strip-suit">${suitHtml(suit)}</span>`;
        for (let t = 0; t < pcts.length; t++) {
            html += `<span class="oracle-strip-cell" style="background:${oraclePctColor(pcts[t])}"` +
                    ` title="${THRESHOLD_LABELS[t]}${SUITS[suit]} : ${pcts[t]} %"></span>`;
        }
        html += '</div><div class="oracle-strip-markers"><span></span>';
        const markers = THRESHOLDS.map(() => []);
        for (const level of ORACLE_MARKER_LEVELS) {
            const idx = oracleCrossing(pcts, level);
            if (idx >= 0) markers[idx].push(level);
        }
        for (const m of markers) {
            html += `<span>${m.length ? '▴' + m.join('·') : ''}</span>`;
        }
        html += '</div>';
    }
    html += '</div>';
    return html;
}

// Per-suit synthesis: average/median NS double-dummy points, % of worlds where
// this suit is NS's best trump, plus compact Sûr/Tendu thresholds (ex-Paliers).
function renderOracleSynth(synth, successCounts, completed) {
    let html = '<table class="oracle-quant-table"><thead><tr><th></th>' +
        '<th>Points NS <span class="oracle-quant-sub">moy. DD</span></th>' +
        '<th>Méd.</th>' +
        '<th>Meilleure couleur <span class="oracle-quant-sub">% mondes</span></th>' +
        '<th class="oracle-mini-col">Sûr <span class="oracle-quant-sub">≥80%</span></th>' +
        '<th class="oracle-mini-col">Tendu <span class="oracle-quant-sub">≥20%</span></th>' +
        '</tr></thead><tbody>';
    for (const suit of SUIT_DISPLAY_ORDER) {
        const avg = Math.round(synth.ns_sums[suit] / completed);
        const med = synth.ns_medians ? Math.round(synth.ns_medians[suit]) : null;
        const bestPct = Math.round(synth.best_counts[suit] / completed * 100);
        const pcts = oraclePcts(successCounts, suit, completed);
        const sur = oracleCrossing(pcts, 80);
        const tendu = oracleCrossing(pcts, 20);
        html += `<tr><td>${suitHtml(suit)}</td><td>${avg}</td><td>${med !== null ? med : '—'}</td>` +
            `<td>${bestPct} %</td>` +
            `<td class="oracle-mini-col">${sur >= 0 ? THRESHOLD_LABELS[sur] : '—'}</td>` +
            `<td class="oracle-mini-col">${tendu >= 0 ? THRESHOLD_LABELS[tendu] : '—'}</td></tr>`;
    }
    html += '</tbody></table>';
    return html;
}

let worldsSource = 'uniform';
let worldsCounts = null;

const WORLDS_SOURCE_LABELS = {
    playgen: 'mondes playgen v2 — conditionn\u00e9s \u00e0 l\u2019ench\u00e8re',
    mixte: 'mondes playgen v2 + compl\u00e9ment uniforme',
    uniform: 'mondes uniformes',
};

function renderOracleTable(successCounts, completed, total, elapsedMs, oracleSynth) {
    updateProgressBar('oracle-progress', completed, total, elapsedMs);

    const body = document.getElementById('annonces-sim-body');

    let html = '';
    if (worldsSource !== 'uniform') {
        let label = WORLDS_SOURCE_LABELS[worldsSource] || worldsSource;
        if (worldsCounts) {
            const pg = worldsCounts.playgen || 0;
            const un = worldsCounts.uniform || 0;
            label += un > 0 ? ` \u2014 ${pg} playgen + ${un} uniformes` : ` \u2014 ${pg}/${pg + un}`;
        }
        html += `<div class="oracle-worlds-badge">${label}</div>`;
    }
    if (oracleSynth && completed > 0) {
        html += '<div class="oracle-variant-label">Moyennes</div>';
        html += renderOracleSynth(oracleSynth, successCounts, completed);
    }
    html += '<div class="oracle-variant-label">Réussite par contrat</div>';
    html += renderOracleStrips(successCounts, completed);
    body.innerHTML = html;
    highlightOracleCell(forcedAction);
}

function renderSimViewer(deals, numSims, sources) {
    const wrap = document.getElementById('annonces-sim-viewer-wrap');
    wrap.classList.remove('hidden');
    const content = document.getElementById('annonces-sim-viewer-content');
    const viewer = document.getElementById('annonces-sim-viewer');

    // Show at most 10 randomly sampled deals
    let sampled = deals;
    let sampledSources = sources || null;
    if (deals.length > 10) {
        const indices = Array.from({ length: deals.length }, (_, i) => i);
        for (let i = indices.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [indices[i], indices[j]] = [indices[j], indices[i]];
        }
        const keep = indices.slice(0, 10).sort((a, b) => a - b);
        sampled = keep.map(i => deals[i]);
        sampledSources = sources ? keep.map(i => sources[i]) : null;
    }

    const shown = sampled.length;
    viewer.querySelector('summary').textContent = `Voir 10 exemples de distribution`;

    let html = '';
    for (let d = 0; d < sampled.length; d++) {
        const deal = sampled[d];
        const srcChip = sampledSources && sampledSources[d]
            ? `<span class="sim-deal-src${sampledSources[d] === 'playgen' ? ' pg' : ''}">${sampledSources[d] === 'playgen' ? 'playgen' : 'uniforme'}</span>`
            : '';
        html += `<details class="sim-deal-details">
            <summary>Donne ${d + 1} ${srcChip}</summary>
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
            if (el) renderHandMini(el, cards, 34);
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

const DOUDOU_TEAM_LABELS = { all: 'Tous', ns: 'NS', ew: 'EW' };
let doudouTeamFilter = 'all';
let lastDoudouData = null; // cached last render args, for filter switching

// Cell = [ns_count, ns_achieved, ew_count, ew_achieved] (legacy 2-tuple tolerated).
function doudouCellCounts(cell, filter) {
    if (cell.length === 2) return [cell[0], cell[1]];
    if (filter === 'ns') return [cell[0], cell[1]];
    if (filter === 'ew') return [cell[2], cell[3]];
    return [cell[0] + cell[2], cell[1] + cell[3]];
}

// Consolidated auction/outcome synthesis over all sims.
function renderDoudouSynth(stats, completed) {
    const contracts = stats.ns_contracts + stats.ew_contracts;
    if (!contracts || stats.taker_seats === undefined) return '';
    const pct = (n, d) => d > 0 ? Math.round(n / d * 100) : 0;
    const rows = [];

    let trumpHtml = SUIT_DISPLAY_ORDER.map(s =>
        `${suitHtml(s)} ${pct(stats.trump_counts[s], contracts)}%`).join(' \u00b7 ');
    if (stats.voids > 0) {
        trumpHtml += ` \u00b7 <span class="synth-dim">pass\u00e9e ${pct(stats.voids, completed)}%</span>`;
    }
    rows.push(['Couleur jou\u00e9e', trumpHtml]);

    rows.push(['Qui prend le contrat', [['Sud', 2], ['Nord', 0], ['Est', 1], ['Ouest', 3]]
        .map(([name, s]) => `${name} ${pct(stats.taker_seats[s], contracts)}%`).join(' \u00b7 ')]);

    rows.push(['Contrats r\u00e9ussis',
        `NS ${pct(stats.ns_achieved, stats.ns_contracts)}% (${stats.ns_achieved}/${stats.ns_contracts})` +
        ` \u00b7 EW ${pct(stats.ew_achieved, stats.ew_contracts)}% (${stats.ew_achieved}/${stats.ew_contracts})`]);

    if (stats.south_bids > 0) {
        const sb = stats.south_bids;
        rows.push(['Nord apr\u00e8s votre annonce',
            `soutient ${pct(stats.partner_support, sb)}% \u00b7 autre couleur ${pct(stats.partner_other, sb)}%` +
            ` \u00b7 passe ${pct(stats.partner_pass, sb)}% <span class="synth-dim">(${sb} donnes)</span>`]);
        rows.push(['Surench\u00e8re adverse', `${pct(stats.opp_overbid, sb)}%`]);
    }

    if (stats.coinche > 0) {
        let c = `${pct(stats.coinche, contracts)}% des contrats (r\u00e9ussis ${stats.coinche_achieved}/${stats.coinche})`;
        if (stats.surcoinche > 0) c += ` \u00b7 surcoinch\u00e9 ${stats.surcoinche}\u00d7`;
        rows.push(['Coinch\u00e9', c]);
    } else {
        rows.push(['Coinch\u00e9', '<span class="synth-dim">jamais</span>']);
    }

    const avgParts = [];
    if (stats.ns_contracts > 0) {
        avgParts.push(`contrat NS moyen ${Math.round(stats.ns_value_sum / stats.ns_contracts)}`);
    }
    if (stats.pts_n > 0) {
        avgParts.push(`points NS ${Math.round(stats.pts_ns_sum / stats.pts_n)} / EW ${Math.round(stats.pts_ew_sum / stats.pts_n)}`);
    }
    if (avgParts.length) rows.push(['Moyennes', avgParts.join(' \u00b7 ')]);

    return '<div class="doudou-synth">' + rows.map(([label, value]) =>
        `<div class="synth-row"><span class="synth-label">${label}</span><span class="synth-value">${value}</span></div>`
    ).join('') + '</div>';
}

function renderDoudouTable(doudouCells, doudouStats, completed, total, elapsedMs) {
    const panel = document.getElementById('annonces-doudou-panel');
    if (!doudouCells) {
        panel.classList.add('hidden');
        return;
    }
    panel.classList.remove('hidden');
    lastDoudouData = { doudouCells, doudouStats, completed, total, elapsedMs };

    updateProgressBar('doudou-progress', completed, total, elapsedMs);
    document.getElementById('doudou-stats-text').textContent = '';

    const body = document.getElementById('annonces-doudou-body');

    let html = renderDoudouSynth(doudouStats, completed);

    // Team filter (column pruning uses total counts so columns stay stable)
    html += '<div id="doudou-team-filter"><span class="synth-label">Contrats pris par</span>' +
        ['all', 'ns', 'ew'].map(f =>
            `<button class="doudou-filter-btn${f === doudouTeamFilter ? ' active' : ''}" data-filter="${f}">${DOUDOU_TEAM_LABELS[f]}</button>`
        ).join('') + '</div>';

    // Prune leading/trailing columns with no observation in any suit
    let firstCol = 0, lastCol = DOUDOU_COLS.length - 1;
    const colUsed = DOUDOU_COLS.map((_, col) =>
        SUIT_DISPLAY_ORDER.some(suit => doudouCellCounts(doudouCells[suit][col], 'all')[0] > 0));
    if (colUsed.some(Boolean)) {
        firstCol = colUsed.indexOf(true);
        lastCol = colUsed.lastIndexOf(true);
    }

    html += '<table id="doudou-table"><thead><tr><th></th>';
    for (let col = firstCol; col <= lastCol; col++) {
        html += `<th>${DOUDOU_COLS[col]}</th>`;
    }
    html += '</tr></thead><tbody>';

    for (const suit of SUIT_DISPLAY_ORDER) {
        html += `<tr><td>${suitHtml(suit)}</td>`;
        for (let col = firstCol; col <= lastCol; col++) {
            const [count, achieved] = doudouCellCounts(doudouCells[suit][col], doudouTeamFilter);
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

    body.querySelectorAll('.doudou-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            doudouTeamFilter = btn.dataset.filter;
            const d = lastDoudouData;
            if (d) renderDoudouTable(d.doudouCells, d.doudouStats, d.completed, d.total, d.elapsedMs);
        });
    });
}

// Highlight the bandeau cell corresponding to the forced annonce (bid actions 1-40).
function highlightOracleCell(action) {
    const strips = document.getElementById('oracle-strips');
    if (!strips) return;
    strips.querySelectorAll('.oracle-forced').forEach(el => el.classList.remove('oracle-forced'));
    if (action === null || action < 1 || action > 40) return;
    let suit, col;
    if (action <= 36) {
        suit = (action - 1) % 4;
        col = Math.floor((action - 1) / 4);
    } else {
        suit = action - 37;
        col = 9; // Capot
    }
    const row = strips.querySelectorAll('.oracle-strip-row')[SUIT_DISPLAY_ORDER.indexOf(suit)];
    if (!row) return;
    const cell = row.children[col + 1];
    if (cell) cell.classList.add('oracle-forced');
}

function resetDoudouPanel() {
    lastDoudouData = null;
    const panel = document.getElementById('annonces-doudou-panel');
    panel.classList.remove('hidden');
    document.getElementById('annonces-doudou-body').innerHTML = '';
    document.getElementById('doudou-stats-text').textContent = '';
    const dp = document.getElementById('doudou-progress');
    dp.classList.add('hidden');
    dp.classList.remove('done');
    dp.querySelector('.sim-progress-fill').style.width = '0%';
    dp.querySelector('.sim-progress-text').textContent = '';
    const forcedLabel = document.getElementById('doudou-forced-label');
    if (forcedLabel) {
        forcedLabel.innerHTML = forcedAction !== null
            ? ` — annonce forcée : ${bidActionHtml(forcedAction)}` : '';
    }
}

// Rerun the DouDou50 simulation with South's bid forced to `action`
// (subsequent bids and play stay NN-driven). Also highlights the matching
// oracle cell — the oracle table itself is bid-independent, so no rerun needed.
function runAltAnalysis(action) {
    if (annoncesHand.size !== 8) return;
    forcedAction = action;

    const statusEl = document.getElementById('annonces-alt-status');
    statusEl.classList.remove('hidden');
    statusEl.innerHTML = `Analyse de ${bidActionHtml(action)} en cours…`;

    highlightOracleCell(action);
    resetDoudouPanel();

    const numSims = Math.max(1, Math.min(1000, parseInt(document.getElementById('annonces-sim-count').value) || 200));
    send({
        type: 'annonces_doudou',
        hand: Array.from(annoncesHand),
        prior_actions: annoncesHistory,
        num_sims: numSims,
        forced_action: action,
    });
}

function handleSimUpdate(data) {
    if (data.error) {
        document.getElementById('annonces-sim-body').innerHTML =
            `<div class="annonces-error">${data.error}</div>`;
        return;
    }
    if (data.worlds_source) worldsSource = data.worlds_source;
    if (data.worlds_counts) worldsCounts = data.worlds_counts;
    renderOracleTable(data.success_counts, data.completed, data.total, data.elapsed_ms, data.oracle_synth);
}

function handleSimDone(data) {
    if (data.worlds_source) worldsSource = data.worlds_source;
    if (data.worlds_counts) worldsCounts = data.worlds_counts;
    renderOracleTable(data.success_counts, data.completed, data.total, data.elapsed_ms, data.oracle_synth);
    if (data.sampled_deals && data.sampled_deals.length > 0) {
        renderSimViewer(data.sampled_deals, data.completed, data.sampled_sources);
    }
}

// --- DouDou-only server handlers (used in local mode for DouDou part) ---

function handleDoudouUpdate(data) {
    if (data.error) {
        if (forcedAction !== null) {
            const statusEl = document.getElementById('annonces-alt-status');
            statusEl.classList.remove('hidden');
            statusEl.innerHTML = `<span class="annonces-error">${data.error}</span>`;
        }
        return; // Otherwise silently ignore — DouDou is optional in local mode
    }
    renderDoudouTable(data.doudou_cells, data.doudou_stats, data.completed, data.total, data.elapsed_ms);
}

function handleDoudouDone(data) {
    if (data.error) return;
    renderDoudouTable(data.doudou_cells, data.doudou_stats, data.completed, data.total, data.elapsed_ms);
    if (forcedAction !== null) {
        const statusEl = document.getElementById('annonces-alt-status');
        statusEl.classList.remove('hidden');
        statusEl.innerHTML = `DouDou50 simulé avec annonce forcée : ${bidActionHtml(forcedAction)}`;
    }
}

// --- Eval paths ---

// Hide all result panels (hand cleared / redrawn).
function hideResults() {
    document.getElementById('annonces-results-area').classList.add('hidden');
    document.getElementById('annonces-nn-panel').classList.add('hidden');
    document.getElementById('annonces-verdict').classList.add('hidden');
    document.getElementById('annonces-sim-viewer-wrap').classList.add('hidden');
}

function resetPanels(numSims) {
    document.getElementById('annonces-results-area').classList.remove('hidden');
    document.getElementById('annonces-nn-panel').classList.remove('hidden');
    document.getElementById('annonces-verdict').classList.add('hidden');
    document.getElementById('annonces-results-header').textContent = 'Bid V6';
    document.getElementById('annonces-results-body').innerHTML =
        '<div class="dd-loader"><div class="dd-loader-text">Calcul\u2026</div></div>';
    // Reset XGB panel
    document.getElementById('annonces-xgb-panel').classList.add('hidden');
    xgbResults = null;
    document.getElementById('annonces-sim-viewer-wrap').classList.add('hidden');
    const emptyCountsSeed = [[0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                             [0, 0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]];
    renderOracleTable(emptyCountsSeed, 0, numSims, null);
    // Reset alternative-annonce state and DouDou panel (shown, empty state)
    forcedAction = null;
    document.getElementById('annonces-alt-status').classList.add('hidden');
    resetDoudouPanel();
}

async function evalLocal(hand, numSims) {
    try {
        await wasmBridge.ensureReady();
    } catch (err) {
        console.warn('[annonces] WASM init failed, falling back to server:', err);
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
            renderOracleTable(data.success_counts, data.completed, data.total, data.elapsed_ms, data.oracle_synth);
        },
        (data) => {
            if (data.error) {
                document.getElementById('annonces-sim-body').innerHTML =
                    `<div class="annonces-error">${data.error}</div>`;
                return;
            }
            renderOracleTable(data.success_counts, data.completed, data.total, data.elapsed_ms, data.oracle_synth);
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
    actionSelector = buildBidSelector(document.getElementById('annonces-action-select'));
    altSelector = buildBidSelector(document.getElementById('annonces-alt-select'));

    document.getElementById('annonces-alt-btn').addEventListener('click', () => {
        runAltAnalysis(altSelector.read());
    });

    // Pre-fill from URL params (e.g. ?hand=7S,KH,... or legacy ?hand=0,1,2,...; &history=5,0,17)
    const params = new URLSearchParams(window.location.search);
    const handParam = params.get('hand');
    const histParam = params.get('history');
    if (handParam) {
        annoncesHand = new Set(handParam.split(',').map(parseCardToken).filter(n => n >= 0 && n < 32));
    }
    if (histParam) {
        annoncesHistory = histParam.split(',').map(Number).filter(n => n >= 0 && n <= 42);
    }

    renderAnnoncesHistory();
    updateAnnoncesDisplay();

    // Auto-evaluate if pre-filled with 8 cards
    if (annoncesHand.size === 8) {
        setTimeout(() => document.getElementById('annonces-eval-btn').click(), 100);
    }

    // Event handlers
    document.getElementById('annonces-history-add-btn').addEventListener('click', () => {
        annoncesHistory.push(actionSelector.read());
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
        hideResults();
    });

    document.getElementById('annonces-clear-btn').addEventListener('click', () => {
        annoncesHand.clear();
        updateAnnoncesDisplay();
        hideResults();
    });

    document.getElementById('annonces-eval-btn').addEventListener('click', () => {
        const hand = Array.from(annoncesHand);
        const numSims = Math.max(1, Math.min(1000, parseInt(document.getElementById('annonces-sim-count').value) || 200));

        // Cancel any previous simulation before starting a new one
        wasmBridge.cancelOracle();

        resetPanels(numSims);

        // Local WASM by default (BidNet + Oracle); falls back to the server
        // if WASM init fails. DouDou always runs server-side (10MB DMC model).
        evalLocal(hand, numSims);
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
