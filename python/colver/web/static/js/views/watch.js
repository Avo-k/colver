// Watch view: live AI vs AI spectating with thinking stats + integrated deal builder
// ES module with mount(container) / unmount() exports

import { send, onMessage, offMessage } from '../ws.js';
import * as SFX from '../sounds.js';
import {
    RANKS, SUITS, SEAT_NAMES_FR, cardSuit, cardRank, cardSvgPath, cardToHtml,
    renderHand, renderTrick, renderLastTrick, contractStr, actionName,
    _prevTrick, _animatingTrick, detectTrickCompletion, animateTrickFlush
} from '../shared/cards.js';
import { BoardRenderer } from '../shared/board.js';
import { initCfnBox } from '../shared/cfn-box.js';
import { setGameId, openBugReport } from '../shared/bug-report.js';

// ── Module state ──────────────────────────────────────────────────────────────

let watchBoard = null;
let watchAutoStarted = false;

// Deal builder state
let dealHands = [[], [], [], []];
let assignedCards = new Set();
let dragCardIdx = null;
let dragSource = null; // 'palette' or player index string

const PLAYER_NAMES_FR = ['Nord', 'Est', 'Sud', 'Ouest'];

// WS handler references (for offMessage in unmount)
let _onWatchStarted = null;
let _onWatchMove = null;
let _onDealSaved = null;

// ── HTML Template ─────────────────────────────────────────────────────────────

function template() {
    return `
<div id="watch-config">
    <div class="watch-agents">
        <label>N\u00a0: <select class="agent-select" data-seat="0">
            <option value="dede">D\u00e9d\u00e9 (IS-DD)</option>
            <option value="doudou">DouDou27</option>
            <option value="oracle_dd">Oracle (DD)</option>
        </select></label>
        <label>E\u00a0: <select class="agent-select" data-seat="1">
            <option value="dede">D\u00e9d\u00e9 (IS-DD)</option>
            <option value="doudou">DouDou27</option>
            <option value="oracle_dd">Oracle (DD)</option>
        </select></label>
        <label>S\u00a0: <select class="agent-select" data-seat="2">
            <option value="dede">D\u00e9d\u00e9 (IS-DD)</option>
            <option value="doudou">DouDou27</option>
            <option value="oracle_dd">Oracle (DD)</option>
        </select></label>
        <label>O\u00a0: <select class="agent-select" data-seat="3">
            <option value="dede">D\u00e9d\u00e9 (IS-DD)</option>
            <option value="doudou">DouDou27</option>
            <option value="oracle_dd">Oracle (DD)</option>
        </select></label>
    </div>
    <label id="watch-difficulty-label">Niveau\u00a0:
        <select id="watch-difficulty">
            <option value="facile">Facile</option>
            <option value="normal">Normal</option>
            <option value="difficile" selected>Difficile</option>
            <option value="expert">Expert</option>
        </select>
    </label>
    <button id="watch-start">Lancer</button>
    <span class="config-separator">|</span>
    <input type="text" id="cfn-input" placeholder="Coller un CFN..." spellcheck="false">
    <button id="cfn-load">Charger</button>

    <details id="deal-builder-accordion" class="deal-accordion">
        <summary>Donne personnalis\u00e9e</summary>
        <div id="deal-builder-content">
            <div id="card-palette"></div>
            <div id="deal-table">
                <div class="deal-row">
                    <div class="drop-zone ns" data-player="0">
                        <div class="drop-zone-label">Nord <span class="dz-count">(0/8)</span></div>
                        <div class="drop-zone-cards"></div>
                    </div>
                </div>
                <div class="deal-row deal-row-middle">
                    <div class="drop-zone ew" data-player="3">
                        <div class="drop-zone-label">Ouest <span class="dz-count">(0/8)</span></div>
                        <div class="drop-zone-cards"></div>
                    </div>
                    <div id="deal-center-area"></div>
                    <div class="drop-zone ew" data-player="1">
                        <div class="drop-zone-label">Est <span class="dz-count">(0/8)</span></div>
                        <div class="drop-zone-cards"></div>
                    </div>
                </div>
                <div class="deal-row">
                    <div class="drop-zone ns" data-player="2">
                        <div class="drop-zone-label">Sud <span class="dz-count">(0/8)</span></div>
                        <div class="drop-zone-cards"></div>
                    </div>
                </div>
            </div>
            <div id="deal-bottom-bar">
                <label>Donneur\u00a0:
                    <select id="deal-dealer">
                        <option value="0">Nord</option>
                        <option value="1">Est</option>
                        <option value="2">Sud</option>
                        <option value="3">Ouest</option>
                    </select>
                </label>
                <button id="random-deal">Donne al\u00e9atoire</button>
                <button id="clear-deal" class="secondary-btn">Vider</button>
                <button id="save-deal">Enregistrer et regarder</button>
            </div>
            <div id="deal-feedback" class="hidden"></div>
        </div>
    </details>
</div>

<div id="watch-main" class="hidden">
    <div id="watch-left">
        <div id="watch-score-bar">
            <span id="watch-score-ns">NS\u00a0: 0</span>
            <span id="watch-game-id" class="game-id-tag hidden"></span>
            <span id="watch-contract-display"></span>
            <span id="watch-score-ew">EO\u00a0: 0</span>
            <button id="watch-report-btn" class="report-btn hidden" title="Signaler un bug">Bug</button>
        </div>
        <div id="watch-cfn" class="cfn-box hidden" title="Cliquer pour copier"></div>

        <div class="seats">
            <div class="seat north">
                <div class="seat-label" id="watch-label-n">Nord</div>
                <div class="hand" id="watch-hand-north"></div>
            </div>
            <div class="seat west">
                <div class="seat-label" id="watch-label-w">Ouest</div>
                <div class="hand" id="watch-hand-west"></div>
            </div>
            <div id="watch-trick-area">
                <div class="trick-card" id="watch-trick-n"></div>
                <div class="trick-card" id="watch-trick-w"></div>
                <div class="trick-card" id="watch-trick-e"></div>
                <div class="trick-card" id="watch-trick-s"></div>
            </div>
            <div id="watch-bid-overlay" class="hidden">
                <div class="bid-overlay-title">Ench\u00e8res</div>
                <div id="watch-bid-overlay-entries"></div>
            </div>
            <div class="seat east">
                <div class="seat-label" id="watch-label-e">Est</div>
                <div class="hand" id="watch-hand-east"></div>
            </div>
            <div class="seat south">
                <div class="seat-label" id="watch-label-s">Sud</div>
                <div class="hand" id="watch-hand-south"></div>
            </div>
        </div>

    </div>

    <div id="watch-right">
        <div id="watch-dd-box" class="dd-oracle-box hidden">
            <div class="dd-title">DD Oracle</div>
            <table class="dd-table">
                <tr><th></th><th>NS</th><th>EO</th></tr>
                <tr><td>\u2660</td><td id="dd-s-ns">-</td><td id="dd-s-ew">-</td></tr>
                <tr><td class="red">\u2665</td><td id="dd-h-ns">-</td><td id="dd-h-ew">-</td></tr>
                <tr><td class="red">\u2666</td><td id="dd-d-ns">-</td><td id="dd-d-ew">-</td></tr>
                <tr><td>\u2663</td><td id="dd-c-ns">-</td><td id="dd-c-ew">-</td></tr>
            </table>
        </div>
        <div id="watch-transport">
            <div class="transport-row">
                <button id="watch-prev-btn" title="Coup pr\u00e9c\u00e9dent">|\u25C0</button>
                <button id="watch-step-btn" title="Prochain coup">\u25B6|</button>
            </div>
            <div class="transport-row">
                <button id="watch-start-btn" title="Retour au d\u00e9but">|\u25C0\u25C0</button>
                <button id="watch-auto-btn" title="Auto-play">\u25B6</button>
                <button id="watch-end-btn" title="Fin de partie">\u25B6\u25B6|</button>
            </div>
        </div>

        <div id="watch-stats-panel">
            <div id="watch-stats-header"></div>
            <div id="watch-stats-body"></div>
        </div>

        <div id="watch-bid-history">
            <div class="section-title">Ench\u00e8res</div>
            <div id="watch-bid-entries"></div>
        </div>

        <div id="watch-tricks-history">
            <div class="section-title">Plis</div>
            <div id="watch-tricks-list"></div>
        </div>
    </div>
</div>
`;
}

// ── Stats rendering (watch-specific) ─────────────────────────────────────────

function watchRenderMoveStats(move, state) {
    const header = watchBoard.el('stats-header');
    const body = watchBoard.el('stats-body');

    if (!move) {
        header.innerHTML = '';
        body.innerHTML = '';
        return;
    }

    const stats = move.stats;
    const seatName = SEAT_NAMES_FR[move.player];
    const teamClass = move.player % 2 === 0 ? 'team-ns' : 'team-ew';

    header.innerHTML = `<span class="stats-player ${teamClass}">${seatName}</span>` +
        `<span class="stats-agent">${stats.agent_label}</span>` +
        `<span class="stats-action">${move.name}</span>` +
        (stats.elapsed_ms !== undefined ? `<span class="stats-time">${stats.elapsed_ms}ms</span>` : '');

    body.innerHTML = '';

    // Bidding: show NN Q-values
    if (move.phase === 0 && stats.bid_nn) {
        const nn = stats.bid_nn;
        const sorted = [...nn.q_values].sort((a, b) => b[1] - a[1]);
        const top = sorted.slice(0, 8);
        const vals = top.map(x => x[1]);
        const maxQ = Math.max(...vals);
        const minQ = Math.min(...vals);
        const range = maxQ - minQ || 1;

        const div = document.createElement('div');
        div.className = 'visit-bars';

        for (const [action, q] of top) {
            const row = document.createElement('div');
            const isBest = action === nn.best_action;
            row.className = 'visit-row' + (isBest ? ' best' : '');
            const pct = ((q - minQ) / range * 100).toFixed(0);
            const name = actionName(action, 0);
            row.innerHTML = `<span class="visit-name">${name}</span>` +
                `<div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>` +
                `<span class="visit-count">${q.toFixed(3)}</span>`;
            div.appendChild(row);
        }
        body.appendChild(div);
        return;
    }

    // IS-DD (Dede): card score bars
    if (stats.card_scores && stats.card_scores.length > 0) {
        renderDdScoreBars(body, stats.card_scores, move.action, stats.determinizations);
        return;
    }

    // IS-MCTS: visit count bars
    if (stats.visit_counts && stats.visit_counts.length > 0) {
        renderVisitBars(body, stats.visit_counts, move.action, stats.root_visits);
        return;
    }

    // DouDou: Q-value bars
    if (stats.q_values && stats.q_values.length > 0) {
        renderQValueBars(body, stats.q_values, move.action);
        return;
    }
}

// ── Card annotations (watch-only) ────────────────────────────────────────────

function watchRenderCardAnnotations(data, state) {
    if (!data || !data.move || !data.move.stats) return;

    const move = data.move;
    const stats = move.stats;

    if (state.phase === 0 || move.phase === 0) return;

    // IS-DD card scores
    if (stats.card_scores && stats.card_scores.length > 0) {
        const scoreMap = buildAnnotationMap(stats.card_scores);
        annotatePlayerHand(move.player, state, scoreMap);
        annotateTrickCard(move.player, move.action, scoreMap);
        return;
    }

    // DouDou Q-values
    if (stats.q_values && stats.q_values.length > 0) {
        const qMap = buildAnnotationMap(stats.q_values);
        annotatePlayerHand(move.player, state, qMap);
        annotateTrickCard(move.player, move.action, qMap);
    }
}

function buildAnnotationMap(entries) {
    const map = new Map();
    const sorted = [...entries].sort((a, b) => b[1] - a[1]);
    const bestAction = sorted[0] ? sorted[0][0] : -1;
    const vals = sorted.map(x => x[1]);
    const maxV = Math.max(...vals);
    const minV = Math.min(...vals);
    const range = maxV - minV || 1;

    for (const [action, val] of entries) {
        const norm = (val - minV) / range;
        const isBest = action === bestAction;
        map.set(action, {
            text: Number.isInteger(val) || Math.abs(val) >= 10 ? val.toFixed(0) : val.toFixed(2),
            cls: 'card-qval' + (isBest ? ' best' : ''),
            style: { opacity: String(0.5 + norm * 0.5) }
        });
    }
    return map;
}

function annotatePlayerHand(player, state, annotationMap) {
    const handEls = {
        0: document.getElementById('watch-hand-north'),
        1: document.getElementById('watch-hand-east'),
        2: document.getElementById('watch-hand-south'),
        3: document.getElementById('watch-hand-west'),
    };
    const handEl = handEls[player];
    if (handEl) {
        const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;
        renderHand(handEl, state.hands[player], false, null, null, trumpSuit, annotationMap);
    }
}

function annotateTrickCard(player, playedCard, annotationMap) {
    const seatMap = { 0: 'n', 1: 'e', 2: 's', 3: 'w' };
    const trickEl = document.getElementById(`watch-trick-${seatMap[player]}`);
    if (trickEl && annotationMap.has(playedCard)) {
        const cardDiv = trickEl.querySelector('.card');
        if (cardDiv) {
            const ann = annotationMap.get(playedCard);
            const badge = document.createElement('span');
            badge.className = `card-annotation ${ann.cls}`;
            badge.textContent = ann.text;
            if (ann.style) Object.assign(badge.style, ann.style);
            cardDiv.appendChild(badge);
        }
    }
}

// ── Stats bar helpers ─────────────────────────────────────────────────────────

function renderVisitBars(container, visitCounts, bestAction, rootVisits) {
    const sorted = [...visitCounts].sort((a, b) => b[1] - a[1]);
    const maxVisits = sorted[0] ? sorted[0][1] : 1;
    const top = sorted.slice(0, 10);

    const div = document.createElement('div');
    div.className = 'visit-bars';

    for (const [action, visits] of top) {
        const row = document.createElement('div');
        const isBest = action === bestAction;
        row.className = 'visit-row' + (isBest ? ' best' : '');
        const pct = (visits / maxVisits * 100).toFixed(0);
        const name = actionName(action, 1);
        row.innerHTML = `<span class="visit-name">${name}</span>` +
            `<div class="visit-bar-bg"><div class="visit-bar-fill" style="width:${pct}%"></div></div>` +
            `<span class="visit-count">${visits}</span>`;
        div.appendChild(row);
    }

    if (rootVisits) {
        const total = document.createElement('div');
        total.className = 'visit-total';
        total.textContent = `Total\u00a0: ${rootVisits} visites`;
        div.appendChild(total);
    }

    container.appendChild(div);
}

function renderQValueBars(container, qValues, bestAction) {
    const sorted = [...qValues].sort((a, b) => b[1] - a[1]);
    const vals = sorted.map(x => x[1]);
    const maxQ = Math.max(...vals);
    const minQ = Math.min(...vals);
    const range = maxQ - minQ || 1;

    const div = document.createElement('div');
    div.className = 'visit-bars';

    for (const [action, q] of sorted) {
        const row = document.createElement('div');
        const isBest = action === bestAction;
        row.className = 'visit-row' + (isBest ? ' best' : '');
        const pct = ((q - minQ) / range * 100).toFixed(0);
        const name = actionName(action, 1);
        row.innerHTML = `<span class="visit-name">${name}</span>` +
            `<div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>` +
            `<span class="visit-count">${q.toFixed(3)}</span>`;
        div.appendChild(row);
    }

    container.appendChild(div);
}

function renderDdScoreBars(container, cardScores, bestAction, determinizations) {
    const sorted = [...cardScores].sort((a, b) => b[1] - a[1]);
    const vals = sorted.map(x => x[1]);
    const maxS = Math.max(...vals);
    const minS = Math.min(...vals);
    const range = maxS - minS || 1;

    const div = document.createElement('div');
    div.className = 'visit-bars';

    for (const [action, score] of sorted) {
        const row = document.createElement('div');
        const isBest = action === bestAction;
        row.className = 'visit-row' + (isBest ? ' best' : '');
        const pct = ((score - minS) / range * 100).toFixed(0);
        const name = actionName(action, 1);
        row.innerHTML = `<span class="visit-name">${name}</span>` +
            `<div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>` +
            `<span class="visit-count">${score.toFixed(1)}</span>`;
        div.appendChild(row);
    }

    if (determinizations) {
        const total = document.createElement('div');
        total.className = 'visit-total';
        total.textContent = `${determinizations} d\u00e9terminisations`;
        div.appendChild(total);
    }

    container.appendChild(div);
}

// ── DD Oracle box ─────────────────────────────────────────────────────────────

function renderDdOracleBox(ddScores) {
    const box = document.getElementById('watch-dd-box');
    if (!ddScores || ddScores.length !== 4) {
        box.classList.add('hidden');
        return;
    }
    box.classList.remove('hidden');
    const suitKeys = ['s', 'h', 'd', 'c'];
    for (let i = 0; i < 4; i++) {
        document.getElementById(`dd-${suitKeys[i]}-ns`).textContent = ddScores[i][0];
        document.getElementById(`dd-${suitKeys[i]}-ew`).textContent = ddScores[i][1];
    }
}

// ── Game ID / bug report helpers ──────────────────────────────────────────────

function setWatchGameId(id) {
    setGameId(id);
    const el = document.getElementById('watch-game-id');
    if (id) {
        el.textContent = id;
        el.classList.remove('hidden');
        document.getElementById('watch-report-btn').classList.remove('hidden');
    } else {
        el.classList.add('hidden');
        document.getElementById('watch-report-btn').classList.add('hidden');
    }
}

// ── Deal builder logic ────────────────────────────────────────────────────────

function createDraggableCard(cardIdx, source) {
    const el = document.createElement('div');
    el.className = 'card';
    el.draggable = true;

    const img = document.createElement('img');
    img.src = cardSvgPath(cardIdx);
    img.alt = `${RANKS[cardRank(cardIdx)]}${SUITS[cardSuit(cardIdx)]}`;
    img.draggable = false;
    el.appendChild(img);

    el.dataset.card = cardIdx;
    el.dataset.source = source;

    el.addEventListener('dragstart', (e) => {
        dragCardIdx = cardIdx;
        dragSource = source;
        el.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', String(cardIdx));
    });

    el.addEventListener('dragend', () => {
        el.classList.remove('dragging');
        dragCardIdx = null;
        dragSource = null;
        document.querySelectorAll('.drag-over').forEach(z => z.classList.remove('drag-over'));
    });

    // Cards in drop zones: click to remove
    if (source !== 'palette') {
        el.addEventListener('click', () => {
            removeCardFromPlayer(cardIdx);
            updateCardDisplay();
        });
    }

    return el;
}

function initCardPalette() {
    const palette = document.getElementById('card-palette');
    if (!palette) return;
    palette.innerHTML = '';
    for (let suit = 0; suit < 4; suit++) {
        const label = document.createElement('div');
        label.className = 'palette-suit-label';
        label.textContent = SUITS[suit];
        label.style.color = (suit === 1 || suit === 2) ? '#ef9a9a' : '#ddd';
        palette.appendChild(label);

        for (let rank = 0; rank < 8; rank++) {
            const idx = suit * 8 + rank;
            const card = createDraggableCard(idx, 'palette');
            card.id = `palette-card-${idx}`;
            if (assignedCards.has(idx)) card.classList.add('assigned');
            palette.appendChild(card);
        }
    }
}

function initDropZones() {
    document.querySelectorAll('#deal-builder-content .drop-zone').forEach(zone => {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            zone.classList.add('drag-over');
        });

        zone.addEventListener('dragleave', (e) => {
            if (!zone.contains(e.relatedTarget)) {
                zone.classList.remove('drag-over');
            }
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            const cardIdx = parseInt(e.dataTransfer.getData('text/plain'));
            if (isNaN(cardIdx)) return;
            const playerIdx = parseInt(zone.dataset.player);

            if (dragSource === 'palette') {
                assignCardToPlayer(cardIdx, playerIdx);
            } else {
                const srcPlayer = parseInt(dragSource);
                if (srcPlayer === playerIdx) return;
                removeCardFromPlayer(cardIdx);
                assignCardToPlayer(cardIdx, playerIdx);
            }
            updateCardDisplay();
        });
    });

    // Palette: drag from drop zone back = remove
    const palette = document.getElementById('card-palette');
    if (!palette) return;

    palette.addEventListener('dragover', (e) => {
        if (dragSource !== 'palette') {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            palette.classList.add('drag-over');
        }
    });

    palette.addEventListener('dragleave', (e) => {
        if (!palette.contains(e.relatedTarget)) {
            palette.classList.remove('drag-over');
        }
    });

    palette.addEventListener('drop', (e) => {
        e.preventDefault();
        palette.classList.remove('drag-over');
        if (dragSource === 'palette') return;
        const cardIdx = parseInt(e.dataTransfer.getData('text/plain'));
        if (isNaN(cardIdx)) return;
        removeCardFromPlayer(cardIdx);
        updateCardDisplay();
    });
}

function assignCardToPlayer(cardIdx, playerIdx) {
    if (dealHands[playerIdx].length >= 8) return false;
    if (assignedCards.has(cardIdx)) return false;
    dealHands[playerIdx].push(cardIdx);
    assignedCards.add(cardIdx);
    return true;
}

function removeCardFromPlayer(cardIdx) {
    for (let p = 0; p < 4; p++) {
        const i = dealHands[p].indexOf(cardIdx);
        if (i >= 0) {
            dealHands[p].splice(i, 1);
            break;
        }
    }
    assignedCards.delete(cardIdx);
}

function updateCardDisplay() {
    // Palette: fade assigned cards
    for (let i = 0; i < 32; i++) {
        const el = document.getElementById(`palette-card-${i}`);
        if (el) {
            el.classList.toggle('assigned', assignedCards.has(i));
            el.draggable = !assignedCards.has(i);
        }
    }

    // Drop zones
    document.querySelectorAll('#deal-builder-content .drop-zone').forEach(zone => {
        const playerIdx = parseInt(zone.dataset.player);
        const cards = dealHands[playerIdx];
        const countEl = zone.querySelector('.dz-count');
        if (countEl) countEl.textContent = `(${cards.length}/8)`;

        zone.classList.toggle('full', cards.length === 8);

        const container = zone.querySelector('.drop-zone-cards');
        if (!container) return;
        container.innerHTML = '';
        const sorted = [...cards].sort((a, b) => a - b);
        for (const c of sorted) {
            container.appendChild(createDraggableCard(c, String(playerIdx)));
        }
    });
}

function resetDealBuilder() {
    dealHands = [[], [], [], []];
    assignedCards = new Set();
    dragCardIdx = null;
    dragSource = null;
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function getSelectedAgents() {
    const agents = {};
    document.querySelectorAll('#watch-config .agent-select').forEach(sel => {
        agents[sel.dataset.seat] = sel.value;
    });
    return agents;
}

function updateWatchDifficultyVisibility() {
    const hasDede = Array.from(document.querySelectorAll('#watch-config .agent-select')).some(sel => sel.value === 'dede');
    const label = document.getElementById('watch-difficulty-label');
    const sel = document.getElementById('watch-difficulty');
    if (label) label.classList.toggle('hidden', !hasDede);
    if (!hasDede && sel) sel.value = 'difficile';
}

function loadFromCfn() {
    const input = document.getElementById('cfn-input');
    const cfn = input.value.trim();
    if (!cfn) return;
    watchAutoStarted = true;
    const agents = getSelectedAgents();
    const difficulty = document.getElementById('watch-difficulty').value;
    send({ type: 'watch_cfn', cfn, agents, difficulty });
}

// ── Mount / Unmount ───────────────────────────────────────────────────────────

export function mount(container) {
    // Render template
    container.innerHTML = template();

    // Create board renderer
    watchBoard = new BoardRenderer({
        prefix: 'watch',
        isReplay: false,
        renderMoveStats: watchRenderMoveStats,
        renderCardAnnotations: watchRenderCardAnnotations,
        onRequestStep: () => {
            send({ type: 'watch_step' });
        },
    });

    // Bind transport buttons and keyboard
    watchBoard.bindTransport();
    watchBoard.bindKeyboard();

    // CFN load
    document.getElementById('cfn-load').addEventListener('click', loadFromCfn);
    document.getElementById('cfn-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') loadFromCfn();
    });

    // Difficulty visibility
    document.querySelectorAll('#watch-config .agent-select').forEach(sel => {
        sel.addEventListener('change', updateWatchDifficultyVisibility);
    });
    updateWatchDifficultyVisibility();

    // Start game button
    document.getElementById('watch-start').addEventListener('click', () => {
        const agents = getSelectedAgents();
        const difficulty = document.getElementById('watch-difficulty').value;
        send({ type: 'watch_start', agents, difficulty });
        document.getElementById('watch-start').disabled = true;
        document.getElementById('watch-start').textContent = 'D\u00e9marrage...';
    });

    // Bug report button
    document.getElementById('watch-report-btn').addEventListener('click', openBugReport);

    // Init CFN box
    initCfnBox('watch-cfn');

    // ── Deal builder bindings ─────────────────────────────────────────────

    resetDealBuilder();
    initCardPalette();
    initDropZones();
    updateCardDisplay();

    // Random deal
    document.getElementById('random-deal').addEventListener('click', () => {
        const undealt = [];
        for (let i = 0; i < 32; i++) {
            if (!assignedCards.has(i)) undealt.push(i);
        }
        for (let i = undealt.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [undealt[i], undealt[j]] = [undealt[j], undealt[i]];
        }
        let idx = 0;
        for (let p = 0; p < 4; p++) {
            const needed = 8 - dealHands[p].length;
            for (let k = 0; k < needed && idx < undealt.length; k++) {
                dealHands[p].push(undealt[idx]);
                assignedCards.add(undealt[idx]);
                idx++;
            }
        }
        initCardPalette();
        updateCardDisplay();
    });

    // Clear deal
    document.getElementById('clear-deal').addEventListener('click', () => {
        resetDealBuilder();
        initCardPalette();
        updateCardDisplay();
        const feedback = document.getElementById('deal-feedback');
        if (feedback) feedback.classList.add('hidden');
    });

    // Save deal -> auto-start watching
    document.getElementById('save-deal').addEventListener('click', () => {
        for (let p = 0; p < 4; p++) {
            if (dealHands[p].length !== 8) {
                alert(`${PLAYER_NAMES_FR[p]} doit avoir exactement 8 cartes (en a ${dealHands[p].length})`);
                return;
            }
        }

        const dealer = parseInt(document.getElementById('deal-dealer').value);
        const agents = getSelectedAgents();

        send({
            type: 'save_custom_deal',
            dealer,
            hands: dealHands,
            agents,
        });
    });

    // ── WS message handlers ───────────────────────────────────────────────

    _onWatchStarted = (data) => {
        watchBoard.reset(data.state);

        document.getElementById('watch-main').classList.remove('hidden');
        document.getElementById('watch-start').disabled = false;
        document.getElementById('watch-start').textContent = 'Relancer';
        if (data.game_id) setWatchGameId(data.game_id);

        renderDdOracleBox(data.dd_scores);

        // Disable DouDou if not available
        if (!data.doudou_available) {
            document.querySelectorAll('#watch-config .agent-select option[value="doudou"]').forEach(opt => {
                opt.disabled = true;
                opt.textContent = 'DouDou27 (non dispo)';
            });
        }

        watchBoard.renderHistoryEntry(-1);
    };

    _onWatchMove = (data) => {
        watchBoard.waitingForStep = false;

        if (data.finished && !data.move) {
            watchBoard.finished = true;
            watchBoard.stopAutoPlay();
            watchBoard.updateTransportButtons();
            return;
        }

        watchBoard.pushMove(data);
        watchBoard.renderHistoryEntry(watchBoard.historyIndex);
        watchBoard.handleBeloteEvent(data);

        if (data.finished) {
            watchBoard.finished = true;
            watchBoard.stopAutoPlay();
            watchBoard.updateTransportButtons();
            return;
        }

        if (watchBoard.autoPlayMode) {
            watchBoard.continueAutoPlay(data);
        }
    };

    _onDealSaved = (data) => {
        const feedback = document.getElementById('deal-feedback');
        feedback.classList.remove('hidden');
        feedback.innerHTML = '';

        const msg = document.createElement('span');
        msg.textContent = 'Donne enregistr\u00e9e\u00a0: ';
        feedback.appendChild(msg);

        const idTag = document.createElement('span');
        idTag.className = 'game-id-tag';
        idTag.textContent = data.game_id;
        feedback.appendChild(idTag);

        // Auto-start watching the custom deal (no tab switch needed)
        const agents = getSelectedAgents();
        send({ type: 'watch_custom', game_id: data.game_id, agents });

        // Collapse the deal builder accordion
        const accordion = document.getElementById('deal-builder-accordion');
        if (accordion) accordion.removeAttribute('open');
    };

    onMessage('watch_started', _onWatchStarted);
    onMessage('watch_move', _onWatchMove);
    onMessage('deal_saved', _onDealSaved);

    // ── Auto-start first game on mount ────────────────────────────────────

    if (!watchAutoStarted) {
        watchAutoStarted = true;
        setTimeout(() => {
            const btn = document.getElementById('watch-start');
            if (btn && !btn.disabled) btn.click();
        }, 100);
    }
}

export function unmount() {
    // Unregister WS handlers
    if (_onWatchStarted) { offMessage('watch_started', _onWatchStarted); _onWatchStarted = null; }
    if (_onWatchMove) { offMessage('watch_move', _onWatchMove); _onWatchMove = null; }
    if (_onDealSaved) { offMessage('deal_saved', _onDealSaved); _onDealSaved = null; }

    // Stop auto-play and keyboard
    if (watchBoard) {
        watchBoard.stopAutoPlay();
        watchBoard.unbindKeyboard();
        watchBoard.active = false;
    }

    // Clear drag state
    dragCardIdx = null;
    dragSource = null;
}
