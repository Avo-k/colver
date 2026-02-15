// Watch mode: AI vs AI spectating with thinking stats

let watchActive = false;
let watchFinished = false;
let autoPlayMode = null; // null, 'trick', 'game'
let autoPlayTimer = null;
let waitingForStep = false;
let trickCountBefore = 0;

const SUIT_LABELS = ['\u2660', '\u2665', '\u2666', '\u2663'];

// Start game
document.getElementById('watch-start').addEventListener('click', () => {
    const agents = {};
    document.querySelectorAll('.agent-select').forEach(sel => {
        agents[sel.dataset.seat] = sel.value;
    });
    const time_ms = parseInt(document.getElementById('watch-time').value);
    send({ type: 'watch_start', agents, time_ms });
    document.getElementById('watch-start').disabled = true;
    document.getElementById('watch-start').textContent = 'Demarrage...';
});

// Transport controls
document.getElementById('watch-step-btn').addEventListener('click', () => {
    if (!watchActive || watchFinished || waitingForStep) return;
    requestStep();
});

document.getElementById('watch-trick-btn').addEventListener('click', () => {
    if (!watchActive || watchFinished || waitingForStep) return;
    startAutoPlay('trick');
});

document.getElementById('watch-auto-btn').addEventListener('click', () => {
    if (autoPlayMode) {
        stopAutoPlay();
    } else if (watchActive && !watchFinished) {
        startAutoPlay('game');
    }
});

document.getElementById('watch-end-btn').addEventListener('click', () => {
    if (!watchActive || watchFinished || waitingForStep) return;
    startAutoPlay('end');
});

function requestStep() {
    waitingForStep = true;
    send({ type: 'watch_step' });
}

function startAutoPlay(mode) {
    autoPlayMode = mode;
    if (mode === 'trick') {
        trickCountBefore = getCurrentTrickCount();
    }
    document.getElementById('watch-auto-btn').textContent = '\u23F8';
    requestStep();
}

function stopAutoPlay() {
    autoPlayMode = null;
    if (autoPlayTimer) {
        clearTimeout(autoPlayTimer);
        autoPlayTimer = null;
    }
    document.getElementById('watch-auto-btn').textContent = '\u25B6';
}

function getCurrentTrickCount() {
    const list = document.getElementById('watch-tricks-list');
    return list ? list.children.length : 0;
}

function continueAutoPlay(data) {
    if (!autoPlayMode || data.finished) {
        stopAutoPlay();
        return;
    }

    if (autoPlayMode === 'trick') {
        const currentTricks = (data.completed_tricks || []).length;
        if (currentTricks > trickCountBefore) {
            stopAutoPlay();
            return;
        }
    }

    const delay = autoPlayMode === 'end' ? 0 : parseInt(document.getElementById('watch-speed').value);
    autoPlayTimer = setTimeout(() => {
        if (autoPlayMode) requestStep();
    }, delay);
}

// Render state
function renderWatchState(state) {
    // Score bar
    document.getElementById('watch-score-ns').textContent = `NS : ${state.points[0]} (${state.tricks_won[0]}P)`;
    document.getElementById('watch-score-ew').textContent = `EO : ${state.points[1]} (${state.tricks_won[1]}P)`;
    document.getElementById('watch-contract-display').textContent = contractStr(state.contract);

    // All 4 hands visible
    const handEls = {
        0: document.getElementById('watch-hand-north'),
        1: document.getElementById('watch-hand-east'),
        2: document.getElementById('watch-hand-south'),
        3: document.getElementById('watch-hand-west'),
    };
    for (let seat = 0; seat < 4; seat++) {
        renderHand(handEls[seat], state.hands[seat]);
    }

    // Trick
    renderTrick('watch-trick', state.current_trick);

    // Last trick
    renderLastTrick(
        document.getElementById('watch-last-trick'),
        state.last_trick,
        state.last_trick_winner,
        state.last_trick_points,
        0  // NS perspective for team coloring
    );

    // Highlight current player
    const labelMap = { 0: 'watch-label-n', 1: 'watch-label-e', 2: 'watch-label-s', 3: 'watch-label-w' };
    for (let s = 0; s < 4; s++) {
        const el = document.getElementById(labelMap[s]);
        if (el) el.classList.toggle('active-player', s === state.current_player && !state.is_terminal);
    }
}

// Render stats panel
function renderStats(move) {
    const header = document.getElementById('watch-stats-header');
    const body = document.getElementById('watch-stats-body');

    if (!move) {
        header.innerHTML = '';
        body.innerHTML = '';
        return;
    }

    const stats = move.stats;
    const seatName = SEAT_NAMES_FR[move.player];
    const team = move.player % 2 === 0 ? 'NS' : 'EO';
    const teamClass = move.player % 2 === 0 ? 'team-ns' : 'team-ew';

    header.innerHTML = `<span class="stats-player ${teamClass}">${seatName}</span>` +
        `<span class="stats-agent">${stats.agent_label}</span>` +
        `<span class="stats-action">${move.name}</span>` +
        (stats.elapsed_ms !== undefined ? `<span class="stats-time">${stats.elapsed_ms}ms</span>` : '');

    body.innerHTML = '';

    // Bidding: show hand evaluation
    if (move.phase === 0 && stats.bid_eval) {
        const eval_ = stats.bid_eval;
        const evalDiv = document.createElement('div');
        evalDiv.className = 'bid-eval-stats';
        for (let s = 0; s < 4; s++) {
            const row = document.createElement('div');
            row.className = 'eval-row' + (s === eval_.best_suit ? ' best' : '');
            const barWidth = Math.min(100, eval_.scores[s] * 3.5);
            row.innerHTML = `<span class="eval-suit">${SUIT_LABELS[s]}</span>` +
                `<div class="eval-bar-bg"><div class="eval-bar-fill" style="width:${barWidth}%"></div></div>` +
                `<span class="eval-score">${eval_.scores[s]}</span>`;
            evalDiv.appendChild(row);
        }
        body.appendChild(evalDiv);
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

function renderVisitBars(container, visitCounts, bestAction, rootVisits) {
    const sorted = [...visitCounts].sort((a, b) => b[1] - a[1]);
    const maxVisits = sorted[0] ? sorted[0][1] : 1;
    const top = sorted.slice(0, 10); // Show top 10

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

// Render bid history (watch mode)
function renderWatchBidHistory(bidHistory) {
    const container = document.getElementById('watch-bid-entries');
    container.innerHTML = '';
    if (!bidHistory || bidHistory.length === 0) return;

    for (const bid of bidHistory) {
        const el = document.createElement('span');
        const team = bid.player % 2 === 0 ? 'team-ns' : 'team-ew';
        el.className = `watch-bid-entry ${team}`;
        const seatLetter = ['N', 'E', 'S', 'O'][bid.player];
        el.textContent = `${seatLetter}:${bid.name}`;
        container.appendChild(el);
    }
}

// Render tricks history
function renderTricksHistory(tricks) {
    const container = document.getElementById('watch-tricks-list');
    container.innerHTML = '';
    if (!tricks || tricks.length === 0) return;

    for (let i = 0; i < tricks.length; i++) {
        const t = tricks[i];
        const row = document.createElement('div');
        const winnerTeam = t.winner % 2 === 0 ? 'team-ns' : 'team-ew';
        row.className = `trick-history-row ${winnerTeam}`;

        const cards = t.cards.map(c => {
            if (c >= 0 && c < 32) return `${RANKS[cardRank(c)]}${SUITS[cardSuit(c)]}`;
            return '?';
        }).join(' ');

        const winnerName = SEAT_NAMES_FR[t.winner];
        row.innerHTML = `<span class="trick-num">#${i + 1}</span>` +
            `<span class="trick-cards">${cards}</span>` +
            `<span class="trick-winner">${winnerName} +${t.points}</span>`;
        container.appendChild(row);
    }

    // Auto-scroll to bottom
    container.scrollTop = container.scrollHeight;
}

// Message handlers
onMessage('watch_started', (data) => {
    watchActive = true;
    watchFinished = false;
    waitingForStep = false;
    autoPlayMode = null;
    document.getElementById('watch-main').classList.remove('hidden');
    document.getElementById('watch-start').disabled = false;
    document.getElementById('watch-start').textContent = 'Relancer';

    // Disable DouDou if not available
    if (!data.doudou_available) {
        document.querySelectorAll('.agent-select option[value="doudou"]').forEach(opt => {
            opt.disabled = true;
            opt.textContent = 'DouDou NN (non dispo)';
        });
    }

    renderWatchState(data.state);
    renderWatchBidHistory([]);
    renderTricksHistory([]);
    document.getElementById('watch-stats-header').innerHTML = '';
    document.getElementById('watch-stats-body').innerHTML = '<div class="stats-placeholder">Cliquez sur un bouton pour avancer</div>';
});

onMessage('watch_move', (data) => {
    waitingForStep = false;

    if (data.finished && !data.move) {
        watchFinished = true;
        stopAutoPlay();
        return;
    }

    renderWatchState(data.state);
    if (data.move) renderStats(data.move);
    renderWatchBidHistory(data.bid_history);
    renderTricksHistory(data.completed_tricks);

    if (data.finished) {
        watchFinished = true;
        stopAutoPlay();
        // Show result
        const state = data.state;
        const header = document.getElementById('watch-stats-header');
        const body = document.getElementById('watch-stats-body');
        const nsWon = state.points[0] > state.points[1];
        header.innerHTML = `<span class="stats-result">${nsWon ? 'NS gagne' : 'EO gagne'} ${state.points[0]}-${state.points[1]}</span>`;
        body.innerHTML = `<div class="stats-final">NS: ${state.points[0]}pts (${state.tricks_won[0]}P) / EO: ${state.points[1]}pts (${state.tricks_won[1]}P)</div>`;
        return;
    }

    // Continue auto-play if active
    if (autoPlayMode) {
        continueAutoPlay(data);
    }
});
