// Watch mode: live AI vs AI spectating with thinking stats

// Watch-specific stats rendering (full Q-values, visit bars, DD scores)
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

// Q-value annotations on cards (watch-only)
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

// Create the watch board renderer
const watchBoard = new BoardRenderer({
    prefix: 'watch',
    isReplay: false,
    renderMoveStats: watchRenderMoveStats,
    renderCardAnnotations: watchRenderCardAnnotations,
    onRequestStep: () => {
        send({ type: 'watch_step' });
    },
});

// Load from CFN
document.getElementById('cfn-load').addEventListener('click', loadFromCfn);
document.getElementById('cfn-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') loadFromCfn();
});

function loadFromCfn() {
    const input = document.getElementById('cfn-input');
    const cfn = input.value.trim();
    if (!cfn) return;
    watchAutoStarted = true;
    const agents = {};
    document.querySelectorAll('.agent-select').forEach(sel => {
        agents[sel.dataset.seat] = sel.value;
    });
    const difficulty = document.getElementById('watch-difficulty').value;
    send({ type: 'watch_cfn', cfn, agents, difficulty });
}

// Show difficulty selector only when at least one agent is IS-DD
function updateWatchDifficultyVisibility() {
    const hasDede = Array.from(document.querySelectorAll('.agent-select')).some(sel => sel.value === 'dede');
    const label = document.getElementById('watch-difficulty-label');
    const sel = document.getElementById('watch-difficulty');
    label.classList.toggle('hidden', !hasDede);
    if (!hasDede) sel.value = 'difficile';
}
document.querySelectorAll('.agent-select').forEach(sel => {
    sel.addEventListener('change', updateWatchDifficultyVisibility);
});
updateWatchDifficultyVisibility();

// Start game
document.getElementById('watch-start').addEventListener('click', () => {
    const agents = {};
    document.querySelectorAll('.agent-select').forEach(sel => {
        agents[sel.dataset.seat] = sel.value;
    });
    const difficulty = document.getElementById('watch-difficulty').value;
    send({ type: 'watch_start', agents, difficulty });
    document.getElementById('watch-start').disabled = true;
    document.getElementById('watch-start').textContent = 'Demarrage...';
});

// Stats bar rendering functions

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
        total.textContent = `Total : ${rootVisits} visites`;
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
        total.textContent = `${determinizations} determinisations`;
        div.appendChild(total);
    }

    container.appendChild(div);
}

// DD Oracle box rendering
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

// Auto-start a game when the Regarder tab is first activated
let watchAutoStarted = false;
document.querySelector('[data-tab="watch"]').addEventListener('click', () => {
    if (!watchAutoStarted) {
        watchAutoStarted = true;
        setTimeout(() => {
            const btn = document.getElementById('watch-start');
            if (btn && !btn.disabled) btn.click();
        }, 100);
    }
});

// Message handlers
onMessage('watch_started', (data) => {
    watchBoard.reset(data.state);

    document.getElementById('watch-main').classList.remove('hidden');
    document.getElementById('watch-start').disabled = false;
    document.getElementById('watch-start').textContent = 'Relancer';
    if (data.game_id) setWatchGameId(data.game_id);

    renderDdOracleBox(data.dd_scores);

    // Disable DouDou if not available
    if (!data.doudou_available) {
        document.querySelectorAll('.agent-select option[value="doudou"]').forEach(opt => {
            opt.disabled = true;
            opt.textContent = 'DouDou35 (non dispo)';
        });
    }

    watchBoard.renderHistoryEntry(-1);
});

onMessage('watch_move', (data) => {
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
});
