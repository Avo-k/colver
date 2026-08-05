// Annonces view — hand builder + bidding NN evaluation
// Supports local WASM computation (BidNet + Oracle) and server fallback.

import { send, onMessage, offMessage, onOpen, offOpen } from '../ws.js';
import { RANKS, SUITS, cardSvgPath, renderHand, renderHandMini, actionName, bidChipHtml, SUIT_DISPLAY_ORDER, cardCode, parseCardToken } from '../shared/cards.js';
import { suitHtml, createSuitPicker } from '../shared/suits.js';
import { SEAT_COLOR_VARS } from '../shared/seats.js';
import { renderBackLink } from '../shared/analyse-back.js';
import * as wasmBridge from '../wasm-bridge.js';
import * as xgbExplain from '../xgb-explain.js';

const SEAT_NAMES = ['Nord', 'Est', 'Sud', 'Ouest'];

// Lien vers la doc. Le routeur redirige tout hash connu (#annonces →
// /analyse/annonces), donc l'ancre passe par ?s= et non par un fragment.
const docLink = (section) =>
    `<a class="doc-link" href="/about?s=${section}" title="Comment ça marche ?">?</a>`;
// Nombre de donnes tir\u00e9es par tableau. Le Jeu r\u00e9el en demande beaucoup plus :
// une donne jou\u00e9e est bien plus rapide qu'un solve double-dummy, et son
// r\u00e9sultat est binaire (r\u00e9ussi/chut\u00e9) donc plus bruit\u00e9 \u00e0 \u00e9chantillon \u00e9gal.
const ORACLE_SIMS = 200;
const REAL_SIMS = 1000;

const THRESHOLDS = [80, 90, 100, 110, 120, 130, 140, 150, 160, 162];
const THRESHOLD_LABELS = ['80', '90', '100', '110', '120', '130', '140', '150', '160', 'Cap'];


const TEMPLATE = `
<a id="annonces-back" class="analyse-back hidden" href="#"></a>
<div id="annonces-match" class="match-bar hidden"></div>
<div id="annonces-top-row">
    <aside id="annonces-saved">
        <div id="annonces-saved-head">
            <button id="annonces-saved-toggle" class="ann-saved-toggle" aria-expanded="true"
                    title="Replier / déplier"><span class="ann-saved-chevron">‹</span></button>
            <span class="ann-saved-title">Mains analysées</span>
            <button id="annonces-saved-add" class="secondary-btn ann-saved-add"
                    title="Sauvegarder la main courante">+</button>
        </div>
        <div id="annonces-saved-list"></div>
        <div id="annonces-saved-foot">
            <button id="annonces-saved-clear" class="secondary-btn">Tout effacer</button>
        </div>
        <button id="annonces-saved-rail" class="ann-saved-rail" title="Mains analysées">Mains analysées</button>
    </aside>
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
            <div id="annonces-results-body"></div>
            <div class="hidden" id="annonces-xgb-panel">
                <div id="annonces-xgb-header" class="section-title">
                    <span>Facteurs cl\u00e9s</span>
                    <select id="xgb-suit-select"></select>
                    ${docLink('annonces')}
                </div>
                <p class="xgb-explainer">Approximation des d\u00e9cisions du r\u00e9seau, pas le r\u00e9seau lui-m\u00eame.</p>
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
            </span>
            <span id="annonces-alt-status" class="hidden"></span>
        </div>
        <div id="annonces-tabs" class="hidden" role="tablist"></div>
        <div id="annonces-results-area" class="hidden">
            <div class="annonces-result-panel hidden" id="annonces-doudou-panel">
                <div id="annonces-doudou-header" class="section-title">
                    <span>Jeu réel<span id="doudou-forced-label"></span></span>
                    ${docLink('jeu-reel')}
                    <div id="doudou-progress" class="sim-progress hidden">
                        <div class="sim-progress-bar"><div class="sim-progress-fill"></div></div>
                        <span class="sim-progress-text"></span>
                    </div>
                    <span id="doudou-stats-text"></span>
                </div>
                <div id="doudou-headline"></div>
                <div id="annonces-doudou-body"></div>
            </div>
            <div class="annonces-result-panel" id="annonces-oracle-panel">
                <div id="annonces-sim-header" class="section-title">
                    <span>Jeu parfait</span>
                    ${docLink('jeu-parfait')}
                    <div id="oracle-progress" class="sim-progress hidden">
                        <div class="sim-progress-bar"><div class="sim-progress-fill"></div></div>
                        <span class="sim-progress-text"></span>
                    </div>
                </div>
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
let actionSelector = null; // paired bid selector for the history-add row
let altSelector = null;    // paired bid selector for "analyser une autre annonce"

// ── Onglets d'analyse ──
// Un onglet = une annonce analysée sur la main courante. Le Jeu parfait ne
// dépend pas de l'annonce (l'Oracle résout les quatre couleurs quoi qu'il
// arrive) : il est partagé par tous les onglets, seul le Jeu réel est simulé
// par onglet. Une seule simulation tourne à la fois côté serveur — ouvrir un
// onglet interrompt celle en cours, qui garde son résultat partiel et peut
// être relancée.
let tabs = [];
let activeTabId = null;
let tabSeq = 0;
let v6BestAction = null;   // l'annonce choisie par Bid V6 sur la main courante
let oracleState = null;    // Jeu parfait, partagé : {counts, completed, total, elapsedMs, synth}

function activeTab() {
    return tabs.find(t => t.id === activeTabId) || null;
}

function tabById(id) {
    return id === undefined || id === null ? null : tabs.find(t => t.id === id) || null;
}

// L'annonce de l'onglet courant, ou null quand c'est celle de Bid V6.
function currentForced() {
    const t = activeTab();
    return t ? t.forced : null;
}

// Partie d'origine, quand on arrive depuis Rejouer. Conservée telle quelle à
// travers les réécritures d'URL — cf. syncUrl.
let backParams = {};

// ── Le score de partie ──
// Bid V6 lit une observation *score-aware* : la même main s'annonce autrement à
// 900-200 qu'à 0-0. Analyser une annonce hors de son score, c'est donc poser
// une autre question que celle que le joueur s'est posée à la table.
//
// `?s=<ns>-<ew>` est **dans le repère de cette page**, qui assied toujours le
// siège analysé en Sud : « nous » y est Nord-Sud quel que soit le siège
// d'origine. La rotation est faite une fois pour toutes par Rejouer au moment
// de fabriquer le lien, comme `rooms.rotate_state` le fait à la diffusion.
//
// 0-0 hors de ce chemin : une main tapée à la main n'a pas de partie derrière
// elle, et c'est le cas nominal de la page.
let matchScores = [0, 0];

function parseScoreParam(raw) {
    const m = /^(\d{1,4})-(\d{1,4})$/.exec((raw || '').trim());
    if (!m) return [0, 0];
    return [Number(m[1]), Number(m[2])];
}

function hasMatchScore() {
    return matchScores[0] > 0 || matchScores[1] > 0;
}

function scoreSig() {
    return hasMatchScore() ? `${matchScores[0]}-${matchScores[1]}` : '';
}

// Un bandeau, pas un réglage : le score vient de la partie d'origine et ne se
// modifie pas ici. Il doit être lisible, parce qu'il explique pourquoi la même
// main peut recevoir deux réponses différentes.
function renderMatchScore() {
    const el = document.getElementById('annonces-match');
    if (!el) return;
    if (!hasMatchScore()) {
        el.classList.add('hidden');
        el.innerHTML = '';
        return;
    }
    el.innerHTML =
        `<span class="match-bar-label">Score de partie</span>` +
        `<span class="match-bar-score"><span class="team-ns">Nous ${matchScores[0]}</span>` +
        ` – <span class="team-ew">Eux ${matchScores[1]}</span></span>` +
        `<span class="match-bar-when">Bid V6 est évalué à ce score</span>`;
    el.classList.remove('hidden');
}

// ── La vraie donne ──
// Quand on arrive depuis Rejouer, on connaît les quatre mains : le solveur dit
// exactement ce que cette distribution-là permettait. C'est une ligne de plus
// dans le Jeu parfait, jamais un ingrédient des mondes échantillonnés — les
// conditionner sur la donne réelle changerait la question de « cette annonce
// était-elle bonne ? » en « a-t-elle marché ? », la seule des deux qui
// n'apprend rien. Absente hors de ce chemin : une main tapée à la main n'a pas
// de donne derrière elle, c'est le cas nominal de la page.
let trueWorld = null;        // {pts: [4], best, seat, hand}
let trueWorldPending = null; // requête à rejouer si le socket n'était pas ouvert

function requestTrueWorld() {
    if (!backParams.from || backParams.i === null || backParams.i === undefined) return;
    trueWorldPending = { game_id: backParams.from, action_idx: Number(backParams.i) };
    send({ type: 'annonces_true_world', ...trueWorldPending });
}

// `send()` jette en silence tant que le socket n'est pas ouvert : à froid (lien
// partagé, signet, F5) la requête partirait dans le vide.
function flushTrueWorld() {
    if (trueWorldPending) send({ type: 'annonces_true_world', ...trueWorldPending });
}

function handleTrueWorld(data) {
    trueWorldPending = null;
    // Pas de message d'erreur : la ligne est un bonus du chemin « depuis
    // Rejouer », son absence ne doit rien coûter au reste de la page.
    trueWorld = data.error ? null : data;
    renderOracle();
}

// La ligne ne vaut que pour la donne d'où elle vient : dès que la main à
// l'écran n'est plus celle du siège analysé, elle décrit une autre donne.
function trueWorldShown() {
    if (!trueWorld || annoncesHand.size !== 8) return null;
    const cur = Array.from(annoncesHand).sort((a, b) => a - b).join(',');
    return cur === trueWorld.hand.join(',') ? trueWorld : null;
}

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
    // `from`/`i` désignent la partie d'où l'on vient : ils ne décrivent pas la
    // main, mais les effacer à la première carte cliquée rendrait l'URL non
    // rechargeable et non partageable avec son chemin de retour.
    for (const key of ['from', 'i']) {
        const v = backParams[key];
        if (v) parts.push(`${key}=${encodeURIComponent(v)}`);
    }
    // Le score fait partie de la question posée, donc de l'URL partageable :
    // sans lui, rouvrir le lien répondrait à 0-0 sans le dire.
    if (hasMatchScore()) parts.push(`s=${scoreSig()}`);
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
    document.getElementById('annonces-saved-add').disabled = count !== 8;
    markCurrentSaved();
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
        actionSpan.innerHTML = bidChipHtml(action);

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
    markCurrentSaved();
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
    document.getElementById('annonces-verdict-action').innerHTML = bidChipHtml(bestAction);
    if (altSelector) altSelector.set(bestAction);

    // L'onglet de base porte « Bid V6 » tant que le réseau n'a pas répondu :
    // son étiquette devient l'annonce dès qu'on la connaît.
    v6BestAction = bestAction;
    renderTabs();
    const base = tabs.find(t => t.forced === null);
    if (base && base.id === activeTabId) renderForcedLabel(base);

    document.getElementById('annonces-results-header').innerHTML =
        `Bid V6 : ${bidChipHtml(bestAction)} ${docLink('annonces')}`;

    let html = '<div class="visit-bars ann-qvalues-scroll ann-collapsed" id="ann-qvalues">';
    qValues.forEach(([action, q], i) => {
        const pct = ((q - minQ) / range * 100).toFixed(0);
        const isBest = action === bestAction;
        const name = bidChipHtml(action);
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
                <span class="visit-name">${bidChipHtml(action)}</span>
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

// Plus haut palier tenu par un total de points cartes, ou -1 sous 80.
function trueWorldLevel(pts) {
    let idx = -1;
    for (let t = 0; t < THRESHOLDS.length; t++) {
        if (pts >= THRESHOLDS[t]) idx = t;
    }
    return idx;
}

// Cellule « vraie donne » : une valeur exacte, donc aucun des codes visuels du
// tableau échantillonné (Wilson, opacité de confiance, taille variable) — sans
// quoi n = 1 se lirait comme la mesure la plus sûre de la page.
function trueWorldCell(tw, suit) {
    const pts = tw.pts[suit];
    const lvl = trueWorldLevel(pts);
    const isBest = pts === Math.max(...tw.pts);
    return `<td class="ow-cell${isBest ? ' ow-best' : ''}" ` +
        `title="Sur cette donne, ${pts} points cartes en double-dummy">` +
        `<span class="ow-pts">${pts}</span>` +
        `<span class="ow-level">${lvl >= 0 ? THRESHOLD_LABELS[lvl] : '—'}</span></td>`;
}

// Per-suit synthesis: average/median NS double-dummy points, % of worlds where
// this suit is NS's best trump, plus compact Sûr/Tendu thresholds (ex-Paliers).
function renderOracleSynth(synth, successCounts, completed) {
    const tw = trueWorldShown();
    let html = '<table class="oracle-quant-table"><thead><tr><th></th>' +
        '<th>Points Nord-Sud <span class="oracle-quant-sub">moy. DD</span></th>' +
        '<th>Méd.</th>' +
        (tw ? '<th class="ow-col">Vraie donne <span class="oracle-quant-sub">pts · contrat</span></th>' : '') +
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
            (tw ? trueWorldCell(tw, suit) : '') +
            `<td>${bestPct} %</td>` +
            `<td class="oracle-mini-col">${sur >= 0 ? THRESHOLD_LABELS[sur] : '—'}</td>` +
            `<td class="oracle-mini-col">${tendu >= 0 ? THRESHOLD_LABELS[tendu] : '—'}</td></tr>`;
    }
    html += '</tbody></table>';
    if (tw) {
        html += '<div class="ow-note">Vraie donne : la distribution telle qu’elle ' +
            'était, résolue en double-dummy — une valeur exacte, sur une seule donne. ' +
            'Elle dit ce que cette donne-là permettait, pas si l’annonce était bonne.</div>';
    }
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
    highlightOracleCell(currentForced());
}

// Le Jeu parfait est commun à tous les onglets : on le mémorise ici et on le
// repeint tel quel à chaque changement d'onglet, seule la case surlignée change.
function applyOracle(data) {
    oracleState = {
        counts: data.success_counts,
        completed: data.completed,
        total: data.total,
        elapsedMs: data.elapsed_ms,
        synth: data.oracle_synth,
    };
    renderOracle();
}

function renderOracle() {
    if (!oracleState) return;
    renderOracleTable(oracleState.counts, oracleState.completed, oracleState.total,
                      oracleState.elapsedMs, oracleState.synth);
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
// Fiabilité d'une cellule, portée par la TEINTE et non par la taille.
// Auparavant chaque cellule recevait sa propre font-size (0.65 → 0.85rem) :
// un tableau où chaque chiffre a un corps différent est illisible, et les
// cellules peu observées finissaient sous le seuil de lecture. L'opacité dit
// la même chose sans toucher au rythme typographique.
// Rendu d'une cellule « taux de réussite ». La couleur vient de la borne
// inférieure de Wilson, l'opacité du nombre d'observations.
function doudouCellHtml(count, achieved, extraCls = '') {
    if (count === 0) return `<td class="doudou-empty ${extraCls}">\u00b7</td>`;
    const pct = Math.round(achieved / count * 100);
    const wlb = Math.round(wilsonLower(achieved, count) * 100);
    const cls = wlb >= 60 ? 'doudou-high' : wlb >= 30 ? 'doudou-mid' : 'doudou-low';
    return `<td class="${cls} ${confidenceClass(count)} ${extraCls}" ` +
        `title="${count} donnes observées, ${achieved} réussies">` +
        `<span class="doudou-pct">${pct}<span class="doudou-unit">%</span></span>` +
        `<span class="doudou-count">${count}</span></td>`;
}

function confidenceClass(count) {
    if (count < 5) return 'conf-lo';
    if (count < 20) return 'conf-mid';
    return 'conf-hi';
}

const DOUDOU_TEAM_LABELS = { all: 'Tous', ns: 'Nord-Sud', ew: 'Est-Ouest' };
// Préférence d'affichage, volontairement globale : deux onglets comparés
// doivent montrer la même tranche de contrats.
let doudouTeamFilter = 'all';

// Cell = [ns_count, ns_achieved, ew_count, ew_achieved] (legacy 2-tuple tolerated).
function doudouCellCounts(cell, filter) {
    if (cell.length === 2) return [cell[0], cell[1]];
    if (filter === 'ns') return [cell[0], cell[1]];
    if (filter === 'ew') return [cell[2], cell[3]];
    return [cell[0] + cell[2], cell[1] + cell[3]];
}

// Chiffre-phare : part des donnes où Nord-Sud marque plus que l'adversaire.
// Le dénominateur inclut les donnes passées (comptées nulles), sinon le taux
// serait calculé sur un sous-ensemble différent de celui du reste du panneau.
function renderDoudouHeadline(stats) {
    if (!stats || stats.deal_wins_ns === undefined) return '';
    const decided = stats.deal_wins_ns + stats.deal_wins_ew + stats.deal_draws;
    if (!decided) return '';

    const pct = Math.round(stats.deal_wins_ns / decided * 100);
    const cls = pct >= 55 ? 'dh-good' : pct <= 45 ? 'dh-bad' : 'dh-even';

    // Les donnes passées ne rapportent rien : `decided` est le bon diviseur
    // pour une moyenne par donne, pas `pts_n` qui les exclut.
    const diff = Math.round((stats.pts_ns_sum - stats.pts_ew_sum) / decided);

    // Une seule ligne, rien de plus : le nombre de donnes est déjà sur la barre
    // de progression juste au-dessus, et le taux de donnes passées dans la
    // synthèse juste en dessous.
    const sub = `Espérance ${diff >= 0 ? '+' : '−'}${Math.abs(diff)} pts`;

    const forced = currentForced();
    const after = forced !== null
        ? `après ${bidChipHtml(forced)}` : 'sur cette main';

    return `<div class="doudou-headline ${cls}">` +
        `<span class="dh-pct">${pct}<span class="dh-unit">%</span></span>` +
        '<span class="dh-text">' +
        `<span class="dh-main">Nord-Sud gagne la donne ${after}</span>` +
        `<span class="dh-sub">${sub}</span>` +
        '</span></div>';
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
        `Nord-Sud ${pct(stats.ns_achieved, stats.ns_contracts)}% (${stats.ns_achieved}/${stats.ns_contracts})` +
        ` \u00b7 Est-Ouest ${pct(stats.ew_achieved, stats.ew_contracts)}% (${stats.ew_achieved}/${stats.ew_contracts})`]);

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
        avgParts.push(`contrat Nord-Sud moyen ${Math.round(stats.ns_value_sum / stats.ns_contracts)}`);
    }
    if (stats.pts_n > 0) {
        avgParts.push(`points Nord-Sud ${Math.round(stats.pts_ns_sum / stats.pts_n)} / Est-Ouest ${Math.round(stats.pts_ew_sum / stats.pts_n)}`);
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

    updateProgressBar('doudou-progress', completed, total, elapsedMs);
    document.getElementById('doudou-stats-text').textContent = '';

    const body = document.getElementById('annonces-doudou-body');

    document.getElementById('doudou-headline').innerHTML = renderDoudouHeadline(doudouStats);

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
    html += '<th class="doudou-total-col">Total</th>';
    html += '</tr></thead><tbody>';

    for (const suit of SUIT_DISPLAY_ORDER) {
        html += `<tr><td>${suitHtml(suit)}</td>`;
        // Le total agrège TOUTES les colonnes, pas seulement celles affichées :
        // l'élagage ne retire aujourd'hui que des colonnes vides, mais le total
        // reste juste si cette règle change.
        let totalCount = 0, totalAchieved = 0;
        for (let col = 0; col < DOUDOU_COLS.length; col++) {
            const [c, a] = doudouCellCounts(doudouCells[suit][col], doudouTeamFilter);
            totalCount += c;
            totalAchieved += a;
        }
        for (let col = firstCol; col <= lastCol; col++) {
            const [count, achieved] = doudouCellCounts(doudouCells[suit][col], doudouTeamFilter);
            html += doudouCellHtml(count, achieved);
        }
        html += doudouCellHtml(totalCount, totalAchieved, 'doudou-total-col');
        html += '</tr>';
    }
    html += '</tbody></table>';
    body.innerHTML = html;

    body.querySelectorAll('.doudou-filter-btn').forEach(btn => {
        btn.addEventListener('click', () => {
            doudouTeamFilter = btn.dataset.filter;
            renderDoudou(activeTab());
        });
    });
}

// Jeu réel de l'onglet `t` : tableau si la simulation a produit quelque chose,
// message d'attente/erreur sinon.
function renderDoudou(t) {
    const panel = document.getElementById('annonces-doudou-panel');
    panel.classList.remove('hidden');
    renderForcedLabel(t);

    const body = document.getElementById('annonces-doudou-body');

    if (!t || (!t.doudou && !t.error)) {
        clearDoudouBody();
        if (t && t.status === 'running') {
            body.innerHTML = '<div class="dd-loader"><div class="dd-loader-text">Simulation…</div></div>';
        } else if (t && t.status === 'partial') {
            body.appendChild(partialNote(t));
        }
        return;
    }
    if (t.error) {
        clearDoudouBody();
        body.innerHTML = `<div class="annonces-error">${t.error}</div>`;
        return;
    }
    const d = t.doudou;
    renderDoudouTable(d.cells, d.stats, d.completed, d.total, d.elapsedMs);
    if (t.status === 'partial') body.appendChild(partialNote(t));
}

// Une seule simulation tourne à la fois : un onglet laissé en plan garde ce
// qu'il avait et propose de reprendre.
function partialNote(t) {
    const note = document.createElement('div');
    note.className = 'ann-tab-note';
    note.innerHTML = '<span>Analyse interrompue par une autre annonce.</span>';
    const btn = document.createElement('button');
    btn.className = 'secondary-btn';
    btn.textContent = 'Relancer';
    btn.addEventListener('click', () => runDoudouFor(t));
    note.appendChild(btn);
    return note;
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

function clearDoudouBody() {
    document.getElementById('annonces-doudou-body').innerHTML = '';
    document.getElementById('doudou-headline').innerHTML = '';
    document.getElementById('doudou-stats-text').textContent = '';
    const dp = document.getElementById('doudou-progress');
    dp.classList.add('hidden');
    dp.classList.remove('done');
    dp.querySelector('.sim-progress-fill').style.width = '0%';
    dp.querySelector('.sim-progress-text').textContent = '';
}

function renderForcedLabel(t) {
    // Rien à dire pour l'onglet de Bid V6 : l'onglet porte déjà son annonce, et
    // un titre plus long chasse la barre de progression hors de l'en-tête.
    const el = document.getElementById('doudou-forced-label');
    if (!el) return;
    const forced = t ? t.forced : null;
    el.innerHTML = forced !== null ? ` — annonce forcée : ${bidChipHtml(forced)}` : '';
}

// ── Onglets : cycle de vie ──

// Crée un onglet pour `forced` (null = l'annonce que Bid V6 choisira lui-même)
// et l'active, sans lancer la simulation.
function createTab(forced) {
    const t = {
        id: ++tabSeq,
        forced,
        hand: Array.from(annoncesHand),
        history: annoncesHistory.slice(),
        doudou: null,
        error: null,
        status: 'idle',
    };
    tabs.push(t);
    activeTabId = t.id;
    return t;
}

function activateTab(id) {
    const t = tabById(id);
    if (!t) return;
    activeTabId = id;
    renderTabs();
    renderActiveTab();
}

function closeTab(id) {
    const i = tabs.findIndex(t => t.id === id);
    if (i < 0 || tabs.length <= 1) return;
    const wasActive = tabs[i].id === activeTabId;
    tabs.splice(i, 1);
    if (wasActive) activeTabId = tabs[Math.min(i, tabs.length - 1)].id;
    renderTabs();
    renderActiveTab();
}

function tabLabelHtml(t) {
    if (t.forced === null) {
        const chip = v6BestAction !== null ? bidChipHtml(v6BestAction) : '';
        return `<span class="ann-tab-tag">V6</span>${chip || '<span>Bid V6</span>'}`;
    }
    return bidChipHtml(t.forced);
}

function renderTabs() {
    const bar = document.getElementById('annonces-tabs');
    if (!bar) return;
    bar.classList.toggle('hidden', tabs.length === 0);
    bar.innerHTML = '';
    for (const t of tabs) {
        const el = document.createElement('div');
        el.className = 'ann-tab' + (t.id === activeTabId ? ' active' : '') +
                       (t.status === 'running' ? ' running' : '') +
                       (t.status === 'partial' ? ' partial' : '') +
                       (t.status === 'error' ? ' error' : '');
        el.setAttribute('role', 'tab');
        el.setAttribute('tabindex', '0');
        el.setAttribute('aria-selected', t.id === activeTabId ? 'true' : 'false');
        if (t.status === 'partial') el.title = 'Analyse interrompue';
        if (t.status === 'error') el.title = t.error || 'Analyse impossible';

        const label = document.createElement('span');
        label.className = 'ann-tab-label';
        label.innerHTML = tabLabelHtml(t);
        el.appendChild(label);

        if (t.doudou && t.status !== 'running') {
            const done = document.createElement('span');
            done.className = 'ann-tab-count';
            done.textContent = `${t.doudou.completed}`;
            el.appendChild(done);
        }
        if (t.status === 'running') {
            const spin = document.createElement('span');
            spin.className = 'ann-tab-spin';
            el.appendChild(spin);
        }
        if (t.status === 'error') {
            const warn = document.createElement('span');
            warn.className = 'ann-tab-warn';
            warn.textContent = '!';
            el.appendChild(warn);
        }

        el.addEventListener('click', () => activateTab(t.id));
        el.addEventListener('keydown', (e) => {
            if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); activateTab(t.id); }
        });

        if (tabs.length > 1) {
            const close = document.createElement('button');
            close.className = 'ann-tab-close';
            close.innerHTML = '×';
            close.title = 'Fermer cet onglet';
            close.addEventListener('click', (e) => { e.stopPropagation(); closeTab(t.id); });
            el.appendChild(close);
        }
        bar.appendChild(el);
    }
}

function renderActiveTab() {
    const t = activeTab();
    renderDoudou(t);
    renderOracle();
    highlightOracleCell(t ? t.forced : null);
}

// Lance (ou relance) la simulation Jeu réel de l'onglet `t`. Une seule tourne
// à la fois côté serveur : les autres passent en « partial ».
function runDoudouFor(t) {
    for (const o of tabs) {
        if (o !== t && o.status === 'running') o.status = 'partial';
    }
    t.status = 'running';
    t.error = null;
    t.doudou = null;

    const msg = {
        type: 'annonces_doudou',
        req_id: t.id,
        hand: t.hand,
        prior_actions: t.history,
        num_sims: REAL_SIMS,
    };
    if (t.forced !== null) msg.forced_action = t.forced;
    send(msg);

    renderTabs();
    if (t.id === activeTabId) renderActiveTab();
}

// « Analyser une autre annonce » : ouvre un onglet pour `action` — ou revient
// à celui qui l'a déjà analysée — et simule le Jeu réel avec l'annonce de Sud
// forcée (la suite des enchères et le jeu restent pilotés par les réseaux).
// Le Jeu parfait ne bouge pas : il ne dépend pas de l'annonce.
function runAltAnalysis(action) {
    if (annoncesHand.size !== 8) return;
    const statusEl = document.getElementById('annonces-alt-status');
    statusEl.classList.add('hidden');
    statusEl.innerHTML = '';

    const existing = tabs.find(t =>
        t.forced === action || (t.forced === null && v6BestAction === action));
    if (existing) {
        activateTab(existing.id);
        return;
    }

    const t = createTab(action);
    runDoudouFor(t);
}

// Un message du serveur porte l'onglet qui l'a demandé. Sans `req_id` (mode
// local, où l'Oracle tourne dans le Worker), on retombe sur l'onglet en cours
// de simulation — jamais sur l'onglet actif, qui peut avoir changé.
function targetTab(data) {
    if (data.req_id !== undefined && data.req_id !== null) return tabById(data.req_id);
    return tabs.find(t => t.status === 'running') || null;
}

function handleSimUpdate(data) {
    if (data.error) {
        document.getElementById('annonces-sim-body').innerHTML =
            `<div class="annonces-error">${data.error}</div>`;
        return;
    }
    if (data.req_id !== undefined && data.req_id !== null && !tabById(data.req_id)) return;
    if (data.worlds_source) worldsSource = data.worlds_source;
    if (data.worlds_counts) worldsCounts = data.worlds_counts;
    applyOracle(data);
}

function handleSimDone(data) {
    if (data.req_id !== undefined && data.req_id !== null && !tabById(data.req_id)) return;
    if (data.worlds_source) worldsSource = data.worlds_source;
    if (data.worlds_counts) worldsCounts = data.worlds_counts;
    applyOracle(data);
    if (data.sampled_deals && data.sampled_deals.length > 0) {
        renderSimViewer(data.sampled_deals, data.completed, data.sampled_sources);
    }
}

// --- DouDou-only server handlers (used in local mode for DouDou part) ---

function handleDoudouUpdate(data) {
    const t = targetTab(data);
    if (!t) return;
    if (data.error) {
        t.error = data.error;
        t.status = 'error';
        renderTabs();
        if (t.id === activeTabId) renderDoudou(t);
        return;
    }
    t.doudou = {
        cells: data.doudou_cells, stats: data.doudou_stats,
        completed: data.completed, total: data.total, elapsedMs: data.elapsed_ms,
    };
    if (t.id === activeTabId) renderDoudou(t);
}

function handleDoudouDone(data) {
    const t = targetTab(data);
    if (!t || data.error) return;
    t.doudou = {
        cells: data.doudou_cells, stats: data.doudou_stats,
        completed: data.completed, total: data.total, elapsedMs: data.elapsed_ms,
    };
    t.status = 'done';
    renderTabs();
    if (t.id === activeTabId) renderDoudou(t);
}

// --- Eval paths ---

// Hide all result panels (hand cleared / redrawn).
function hideResults() {
    tabs = [];
    activeTabId = null;
    v6BestAction = null;
    oracleState = null;
    document.getElementById('annonces-tabs').classList.add('hidden');
    document.getElementById('annonces-results-area').classList.add('hidden');
    document.getElementById('annonces-nn-panel').classList.add('hidden');
    document.getElementById('annonces-verdict').classList.add('hidden');
    document.getElementById('annonces-sim-viewer-wrap').classList.add('hidden');
}

function resetPanels() {
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
    oracleState = { counts: emptyCountsSeed, completed: 0, total: ORACLE_SIMS,
                    elapsedMs: null, synth: null };
    document.getElementById('annonces-alt-status').classList.add('hidden');
    clearDoudouBody();
}

// Nouvelle \u00e9valuation : les onglets d\u00e9crivent la main courante, on repart d'un
// seul onglet \u2014 celui de l'annonce que Bid V6 va choisir.
function runEvaluation() {
    const hand = Array.from(annoncesHand);
    if (hand.length !== 8) return;

    wasmBridge.cancelOracle();
    tabs = [];
    tabSeq = 0;
    activeTabId = null;
    v6BestAction = null;

    resetPanels();
    const base = createTab(null);
    base.status = 'running';
    renderTabs();
    renderActiveTab();

    recordSavedHand(hand, annoncesHistory);

    // Local WASM by default (BidNet + Oracle); falls back to the server
    // if WASM init fails. DouDou always runs server-side (10MB DMC model).
    evalLocal(hand, base);
}

async function evalLocal(hand, tab) {
    try {
        await wasmBridge.ensureReady();
    } catch (err) {
        console.warn('[annonces] WASM init failed, falling back to server:', err);
        evalServer(hand, tab);
        return;
    }

    // 1. BidNet eval (main thread, sub-ms), au score de la partie d'origine.
    //    Un bundle WASM antérieur au score n'a pas de quoi répondre : plutôt
    //    que des Q silencieusement calculés à 0-0, on repasse par le serveur,
    //    qui sait le faire. L'Oracle et le Jeu réel, eux, ne lisent pas le
    //    score — c'est le bidder qui est score-aware, pas le solveur.
    try {
        handleBidEvalResult(wasmBridge.evaluateBid(hand, annoncesHistory, matchScores));
    } catch (err) {
        if (hasMatchScore()) {
            console.warn('[annonces] BidNet WASM sans score, repli serveur:', err);
            send({ type: 'bid_eval', hand, prior_actions: annoncesHistory, scores: matchScores });
        } else {
            handleBidEvalResult({ error: `WASM BidNet: ${err.message || err}` });
        }
    }

    // 2. Oracle via Worker (streaming). Partagé par tous les onglets : ouvrir
    //    une autre annonce ne l'interrompt pas — il ne dépend pas de l'annonce.
    wasmBridge.runOracleSim(hand, ORACLE_SIMS,
        (data) => {
            applyOracle(data);
        },
        (data) => {
            if (data.error) {
                document.getElementById('annonces-sim-body').innerHTML =
                    `<div class="annonces-error">${data.error}</div>`;
                return;
            }
            applyOracle(data);
            if (data.sampled_deals && data.sampled_deals.length > 0) {
                renderSimViewer(data.sampled_deals, data.completed);
            }
        }
    );

    // 3. DouDou via WebSocket (server-side, needs DMC model)
    runDoudouFor(tab);
}

function evalServer(hand, tab) {
    send({ type: 'bid_eval', hand, prior_actions: annoncesHistory, scores: matchScores });
    // Le serveur enchaîne Oracle puis Dédé sur un même pool de mondes : le
    // Jeu réel de ce flux appartient à l'onglet de base.
    tab.status = 'running';
    send({ type: 'annonces_sim', req_id: tab.id, hand, prior_actions: annoncesHistory,
           oracle_sims: ORACLE_SIMS, doudou_sims: REAL_SIMS });
    renderTabs();
}

// ── Mains sauvegardées ──
// Une main analysée, c'est la main, les enchères qui la précèdent *et* le score
// de partie : les trois forment la situation. La même main après « 100♥ » de
// l'adversaire n'est pas la même question, et la même main à 1900-200 non plus
// — Bid V6 lit le score. La clé de déduplication porte donc sur les trois.

const SAVED_KEY = 'colver:annonces:saved';
const SIDEBAR_KEY = 'colver:annonces:sidebar';
const SAVED_MAX = 40;

// `scores` absent d'une entrée écrite avant que le score existe = 0-0, ce qui
// est exactement ce sous quoi elle a été analysée.
function handSig(hand, history, scores) {
    const s = (scores && (scores[0] || scores[1])) ? `${scores[0]}-${scores[1]}` : '';
    return Array.from(hand).slice().sort((a, b) => a - b).join(',') + '|' + history.join(',') + '|' + s;
}

function loadSaved() {
    try {
        const raw = localStorage.getItem(SAVED_KEY);
        const list = raw ? JSON.parse(raw) : [];
        return Array.isArray(list) ? list.filter(e => Array.isArray(e.hand) && e.hand.length === 8) : [];
    } catch { return []; }
}

function storeSaved(list) {
    try { localStorage.setItem(SAVED_KEY, JSON.stringify(list.slice(0, SAVED_MAX))); } catch { /* quota */ }
}

// Enregistre (ou remonte en tête) la situation courante.
function recordSavedHand(hand, history) {
    if (hand.length !== 8) return;
    const sig = handSig(hand, history, matchScores);
    const list = loadSaved().filter(e => handSig(e.hand, e.history || [], e.scores) !== sig);
    list.unshift({
        hand: Array.from(hand).sort((a, b) => a - b),
        history: history.slice(),
        scores: hasMatchScore() ? matchScores.slice() : null,
        ts: Date.now(),
    });
    storeSaved(list);
    renderSavedList();
}

function renderSavedList() {
    const list = document.getElementById('annonces-saved-list');
    if (!list) return;
    const entries = loadSaved();
    list.innerHTML = '';

    if (entries.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'ann-saved-empty';
        empty.textContent = 'Les mains évaluées s’enregistrent ici.';
        list.appendChild(empty);
        document.getElementById('annonces-saved-clear').classList.add('hidden');
        return;
    }
    document.getElementById('annonces-saved-clear').classList.remove('hidden');

    const current = handSig(Array.from(annoncesHand), annoncesHistory, matchScores);
    entries.forEach((entry, i) => {
        const row = document.createElement('div');
        row.className = 'ann-saved-row';
        row.dataset.sig = handSig(entry.hand, entry.history || [], entry.scores);
        if (row.dataset.sig === current) row.classList.add('current');

        const body = document.createElement('button');
        body.className = 'ann-saved-body';
        body.title = 'Charger et évaluer cette main';

        // Pas de classe `hand` ici : elle impose la hauteur d'une carte pleine
        // (cf. cards.css), qui laisserait un grand vide sous ces miniatures.
        const cards = document.createElement('div');
        cards.className = 'ann-saved-cards';
        renderHandMini(cards, entry.hand, 18);
        body.appendChild(cards);

        const hist = document.createElement('div');
        hist.className = 'ann-saved-hist';
        hist.innerHTML = (entry.history && entry.history.length)
            ? entry.history.map(bidChipHtml).join('')
            : '<span class="ann-saved-first">Premier à parler</span>';
        // Sans ce repère, deux entrées qui ne diffèrent que par le score sont
        // indiscernables — or elles ne portent pas la même réponse.
        if (entry.scores && (entry.scores[0] || entry.scores[1])) {
            hist.innerHTML += `<span class="ann-saved-score" title="Score de partie">` +
                `${entry.scores[0]}–${entry.scores[1]}</span>`;
        }
        body.appendChild(hist);

        body.addEventListener('click', () => loadSavedEntry(entry));
        row.appendChild(body);

        const del = document.createElement('button');
        del.className = 'ann-saved-del';
        del.innerHTML = '×';
        del.title = 'Supprimer';
        del.addEventListener('click', (e) => {
            e.stopPropagation();
            const kept = loadSaved();
            kept.splice(i, 1);
            storeSaved(kept);
            renderSavedList();
        });
        row.appendChild(del);
        list.appendChild(row);
    });
}

// Repeindre la liste à chaque carte cliquée reconstruirait des centaines
// d'images : seul le liseré « situation courante » change vraiment.
function markCurrentSaved() {
    const cur = handSig(Array.from(annoncesHand), annoncesHistory, matchScores);
    document.querySelectorAll('#annonces-saved-list .ann-saved-row').forEach(row => {
        row.classList.toggle('current', row.dataset.sig === cur);
    });
}

function loadSavedEntry(entry) {
    // Une main enregistrée ne porte pas la donne dont elle vient : sa clé est
    // (main, enchères précédentes, score). On perd donc la vraie donne au
    // rechargement, plutôt que de risquer de la rattacher à une autre. Le
    // score, lui, est dans l'entrée : sans lui la même main reviendrait
    // analysée à 0-0 sous le même libellé.
    trueWorld = null;
    annoncesHand = new Set(entry.hand);
    annoncesHistory = (entry.history || []).slice();
    matchScores = (entry.scores || [0, 0]).slice();
    renderMatchScore();
    syncUrl();
    renderAnnoncesHistory();
    updateAnnoncesDisplay();
    renderSavedList();
    runEvaluation();
}

function setSidebar(open) {
    const el = document.getElementById('annonces-saved');
    if (!el) return;
    el.classList.toggle('collapsed', !open);
    document.getElementById('annonces-saved-toggle')
        .setAttribute('aria-expanded', open ? 'true' : 'false');
    try { localStorage.setItem(SIDEBAR_KEY, open ? 'open' : 'closed'); } catch { /* private mode */ }
}

function sidebarInitiallyOpen() {
    let stored = null;
    try { stored = localStorage.getItem(SIDEBAR_KEY); } catch { /* private mode */ }
    if (stored === 'open') return true;
    if (stored === 'closed') return false;
    // Par défaut : dépliée seulement là où elle ne mange rien à l'analyse.
    return window.innerWidth >= 1400;
}

export function mount(container) {
    container.innerHTML = TEMPLATE;

    annoncesHand = new Set();
    annoncesHistory = [];
    tabs = [];
    tabSeq = 0;
    activeTabId = null;
    v6BestAction = null;
    oracleState = null;
    trueWorld = null;
    trueWorldPending = null;

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
    // Retour vers la partie d'où l'on vient, quand on arrive depuis Rejouer.
    backParams = { from: params.get('from'), i: params.get('i') };
    matchScores = parseScoreParam(params.get('s'));
    renderBackLink('annonces-back', backParams.from, backParams.i);
    renderMatchScore();
    requestTrueWorld();
    if (handParam) {
        annoncesHand = new Set(handParam.split(',').map(parseCardToken).filter(n => n >= 0 && n < 32));
    }
    if (histParam) {
        annoncesHistory = histParam.split(',').map(Number).filter(n => n >= 0 && n <= 42);
    }

    renderAnnoncesHistory();
    updateAnnoncesDisplay();

    // Barre latérale des mains sauvegardées
    setSidebar(sidebarInitiallyOpen());
    renderSavedList();
    document.getElementById('annonces-saved-toggle').addEventListener('click', () => {
        const el = document.getElementById('annonces-saved');
        setSidebar(el.classList.contains('collapsed'));
    });
    document.getElementById('annonces-saved-rail').addEventListener('click', () => setSidebar(true));
    document.getElementById('annonces-saved-add').addEventListener('click', () => {
        if (annoncesHand.size !== 8) return;
        recordSavedHand(Array.from(annoncesHand), annoncesHistory);
    });
    document.getElementById('annonces-saved-clear').addEventListener('click', () => {
        storeSaved([]);
        renderSavedList();
    });

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

    document.getElementById('annonces-eval-btn').addEventListener('click', runEvaluation);

    // XGB suit dropdown
    document.getElementById('xgb-suit-select').addEventListener('change', (e) => {
        if (xgbResults) {
            renderXgbWaterfall(xgbResults[parseInt(e.target.value)]);
        }
    });

    onOpen(flushTrueWorld);
    onMessage('annonces_true_world', handleTrueWorld);
    onMessage('bid_eval_result', handleBidEvalResult);
    onMessage('annonces_sim_update', handleSimUpdate);
    onMessage('annonces_sim_done', handleSimDone);
    onMessage('annonces_doudou_update', handleDoudouUpdate);
    onMessage('annonces_doudou_done', handleDoudouDone);
}

export function unmount() {
    offOpen(flushTrueWorld);
    offMessage('annonces_true_world', handleTrueWorld);
    offMessage('bid_eval_result', handleBidEvalResult);
    offMessage('annonces_sim_update', handleSimUpdate);
    offMessage('annonces_sim_done', handleSimDone);
    offMessage('annonces_doudou_update', handleDoudouUpdate);
    offMessage('annonces_doudou_done', handleDoudouDone);
    wasmBridge.cancelOracle();
    annoncesHand = new Set();
    annoncesHistory = [];
    xgbResults = null;
    tabs = [];
    activeTabId = null;
    v6BestAction = null;
    backParams = {};
    oracleState = null;
    trueWorld = null;
    trueWorldPending = null;
}
