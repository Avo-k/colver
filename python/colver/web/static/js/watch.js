// Watch mode: AI vs AI spectating with thinking stats

let watchActive = false;
let watchFinished = false;
let autoPlayMode = null; // null, 'game', 'end'
let autoPlayTimer = null;
let waitingForStep = false;

// History buffer for bidirectional navigation
let moveHistory = [];    // Array of all received watch_move/replay_move data
let historyIndex = -1;   // Cursor (-1 = initial state before any move)
let initialState = null; // State from watch_started/replay_loaded

const SUIT_LABELS = ['\u2660', '\u2665', '\u2666', '\u2663'];

// Start game
document.getElementById('watch-start').addEventListener('click', () => {
    const agents = {};
    document.querySelectorAll('.agent-select').forEach(sel => {
        agents[sel.dataset.seat] = sel.value;
    });
    send({ type: 'watch_start', agents });
    document.getElementById('watch-start').disabled = true;
    document.getElementById('watch-start').textContent = 'Demarrage...';
});

// Transport controls
document.getElementById('watch-prev-btn').addEventListener('click', () => {
    if (!watchActive) return;
    stopAutoPlay();
    goToPreviousMove();
});

document.getElementById('watch-step-btn').addEventListener('click', () => {
    if (!watchActive || waitingForStep) return;
    stopAutoPlay();
    goToNextMove();
});

document.getElementById('watch-start-btn').addEventListener('click', () => {
    if (!watchActive) return;
    stopAutoPlay();
    goToStart();
});

document.getElementById('watch-auto-btn').addEventListener('click', () => {
    if (autoPlayMode) {
        stopAutoPlay();
    } else if (watchActive && !isAtEnd()) {
        startAutoPlay('game');
    }
});

document.getElementById('watch-end-btn').addEventListener('click', () => {
    if (!watchActive || waitingForStep) return;
    stopAutoPlay();
    if (isAtEnd()) return;
    startAutoPlay('end');
});

// Keyboard navigation
document.addEventListener('keydown', (e) => {
    // Only when watch panel is active and not focused on input elements
    if (!document.getElementById('watch-panel').classList.contains('active')) return;
    const tag = document.activeElement?.tagName;
    if (tag === 'INPUT' || tag === 'SELECT' || tag === 'TEXTAREA') return;
    if (!watchActive) return;

    if (e.key === 'ArrowLeft') {
        e.preventDefault();
        stopAutoPlay();
        goToPreviousMove();
    } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        stopAutoPlay();
        goToNextMove();
    }
});

function isAtEnd() {
    // At end if we're at the last buffered entry and that entry is finished, or game is finished
    if (historyIndex >= 0 && historyIndex === moveHistory.length - 1) {
        const last = moveHistory[historyIndex];
        if (last && last.finished) return true;
    }
    return watchFinished && historyIndex === moveHistory.length - 1;
}

function requestStep() {
    waitingForStep = true;
    send({ type: replaySession ? 'replay_step' : 'watch_step' });
}

function startAutoPlay(mode) {
    autoPlayMode = mode;
    document.getElementById('watch-auto-btn').textContent = '\u23F8';
    goToNextMove();
}

function stopAutoPlay() {
    autoPlayMode = null;
    if (autoPlayTimer) {
        clearTimeout(autoPlayTimer);
        autoPlayTimer = null;
    }
    document.getElementById('watch-auto-btn').textContent = '\u25B6';
}

// Navigation functions
function goToPreviousMove() {
    if (historyIndex <= -1) return;
    historyIndex--;
    renderHistoryEntry(historyIndex);
}

function goToNextMove() {
    if (historyIndex < moveHistory.length - 1) {
        // We have buffered data ahead
        historyIndex++;
        renderHistoryEntry(historyIndex);
        // Continue auto-play from buffer
        if (autoPlayMode) {
            continueAutoPlayFromBuffer();
        }
    } else if (!watchFinished && !isAtEnd()) {
        // Need to request from server
        requestStep();
    } else if (autoPlayMode) {
        stopAutoPlay();
    }
}

function goToStart() {
    historyIndex = -1;
    renderHistoryEntry(-1);
}

function renderHistoryEntry(index) {
    if (index < 0) {
        // Render initial state
        if (initialState) {
            renderWatchState(initialState);
            renderWatchBidHistory([]);
            renderTricksHistory([]);
            document.getElementById('watch-stats-header').innerHTML = '';
            document.getElementById('watch-stats-body').innerHTML = '<div class="stats-placeholder">Cliquez sur un bouton pour avancer</div>';
            renderCardAnnotations(null, initialState);
        }
    } else if (index < moveHistory.length) {
        const data = moveHistory[index];
        renderWatchState(data.state);
        if (data.move) renderStats(data.move);
        else {
            document.getElementById('watch-stats-header').innerHTML = '';
            document.getElementById('watch-stats-body').innerHTML = '';
        }
        renderWatchBidHistory(data.bid_history);
        renderTricksHistory(data.completed_tricks);

        if (data.finished) {
            const state = data.state;
            const header = document.getElementById('watch-stats-header');
            const body = document.getElementById('watch-stats-body');
            const nsWon = state.points[0] > state.points[1];
            if (replaySession) {
                header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> <span class="stats-result">${nsWon ? 'NS gagne' : 'EO gagne'} ${state.points[0]}-${state.points[1]}</span>`;
            } else {
                header.innerHTML = `<span class="stats-result">${nsWon ? 'NS gagne' : 'EO gagne'} ${state.points[0]}-${state.points[1]}</span>`;
            }
            body.innerHTML = `<div class="stats-final">NS: ${state.points[0]}pts (${state.tricks_won[0]}P) / EO: ${state.points[1]}pts (${state.tricks_won[1]}P)</div>`;
        }

        renderCardAnnotations(data, data.state);
    }
    updateTransportButtons();
}

function updateTransportButtons() {
    const prevBtn = document.getElementById('watch-prev-btn');
    const startBtn = document.getElementById('watch-start-btn');
    const stepBtn = document.getElementById('watch-step-btn');
    const endBtn = document.getElementById('watch-end-btn');

    const canGoBack = historyIndex >= 0;
    const canGoForward = !isAtEnd() || historyIndex < moveHistory.length - 1;

    prevBtn.disabled = !canGoBack;
    startBtn.disabled = !canGoBack;
    stepBtn.disabled = !canGoForward || waitingForStep;
    endBtn.disabled = !canGoForward || waitingForStep;
}

function continueAutoPlayFromBuffer() {
    if (!autoPlayMode) return;
    const data = moveHistory[historyIndex];
    if (!data) { stopAutoPlay(); return; }

    if (data.finished) {
        stopAutoPlay();
        return;
    }

    const delay = autoPlayMode === 'end' ? 0 : 1000;
    autoPlayTimer = setTimeout(() => {
        if (autoPlayMode) goToNextMove();
    }, delay);
}

function continueAutoPlay(data) {
    if (!autoPlayMode || data.finished) {
        stopAutoPlay();
        return;
    }

    const delay = autoPlayMode === 'end' ? 0 : 1000;
    autoPlayTimer = setTimeout(() => {
        if (autoPlayMode) goToNextMove();
    }, delay);
}

// Q-value annotations on cards
function renderCardAnnotations(data, state) {
    if (!data || !data.move || !data.move.stats) return;

    const move = data.move;
    const stats = move.stats;

    // No annotations during bidding phase
    if (state.phase === 0 || move.phase === 0) return;

    // During play phase with DouDou Q-values
    if (stats.q_values && stats.q_values.length > 0) {
        const qMap = new Map();
        const sorted = [...stats.q_values].sort((a, b) => b[1] - a[1]);
        const bestAction = sorted[0] ? sorted[0][0] : -1;
        const vals = sorted.map(x => x[1]);
        const maxQ = Math.max(...vals);
        const minQ = Math.min(...vals);
        const range = maxQ - minQ || 1;

        for (const [action, q] of stats.q_values) {
            const norm = (q - minQ) / range;
            const isBest = action === bestAction;
            qMap.set(action, {
                text: q.toFixed(2),
                cls: 'card-qval' + (isBest ? ' best' : ''),
                style: { opacity: String(0.5 + norm * 0.5) }
            });
        }

        // Annotate the player's hand and the played card in trick area
        const player = move.player;
        const handEls = {
            0: document.getElementById('watch-hand-north'),
            1: document.getElementById('watch-hand-east'),
            2: document.getElementById('watch-hand-south'),
            3: document.getElementById('watch-hand-west'),
        };
        const handEl = handEls[player];
        if (handEl) {
            const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;
            renderHand(handEl, state.hands[player], false, null, null, trumpSuit, qMap);
        }

        // Also annotate the just-played card in the trick area
        const playedCard = move.action;
        const seatMap = { 0: 'n', 1: 'e', 2: 's', 3: 'w' };
        const trickEl = document.getElementById(`watch-trick-${seatMap[player]}`);
        if (trickEl && qMap.has(playedCard)) {
            const cardDiv = trickEl.querySelector('.card');
            if (cardDiv) {
                const ann = qMap.get(playedCard);
                const badge = document.createElement('span');
                badge.className = `card-annotation ${ann.cls}`;
                badge.textContent = ann.text;
                if (ann.style) Object.assign(badge.style, ann.style);
                cardDiv.appendChild(badge);
            }
        }
    }
}

function reRenderHandsWithAnnotations(state, annotations) {
    const handEls = {
        0: document.getElementById('watch-hand-north'),
        1: document.getElementById('watch-hand-east'),
        2: document.getElementById('watch-hand-south'),
        3: document.getElementById('watch-hand-west'),
    };
    const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;
    for (let seat = 0; seat < 4; seat++) {
        renderHand(handEls[seat], state.hands[seat], false, null, null, trumpSuit, annotations);
    }
}

// Render state
function renderWatchState(state) {
    // Score bar
    document.getElementById('watch-score-ns').textContent = `NS : ${state.points[0]} (${state.tricks_won[0]}P)`;
    document.getElementById('watch-score-ew').textContent = `EO : ${state.points[1]} (${state.tricks_won[1]}P)`;
    document.getElementById('watch-contract-display').textContent = contractStr(state.contract);

    // Belote badges
    if (state.belote) {
        renderBeloteBadge('watch-score-ns', state.belote[0]);
        renderBeloteBadge('watch-score-ew', state.belote[1]);
    }

    // All 4 hands visible
    const handEls = {
        0: document.getElementById('watch-hand-north'),
        1: document.getElementById('watch-hand-east'),
        2: document.getElementById('watch-hand-south'),
        3: document.getElementById('watch-hand-west'),
    };
    const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;
    for (let seat = 0; seat < 4; seat++) {
        renderHand(handEls[seat], state.hands[seat], false, null, null, trumpSuit);
    }

    // Trick
    renderTrick('watch-trick', state.current_trick);

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
        const name = actionName(bid.action, 0);
        el.textContent = `${seatLetter}:${name}`;
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

        const SEAT_L = ['N', 'E', 'S', 'O'];
        const leadSeat = t.lead !== undefined ? t.lead : -1;

        // Show cards in play order starting from lead
        let orderedCards = '';
        if (leadSeat >= 0) {
            for (let j = 0; j < 4; j++) {
                const seat = (leadSeat + j) % 4;
                const c = t.cards[seat];
                if (c >= 0 && c < 32) {
                    orderedCards += `${RANKS[cardRank(c)]}${SUITS[cardSuit(c)]} `;
                }
            }
            orderedCards = orderedCards.trim();
        } else {
            orderedCards = t.cards.map(c => {
                if (c >= 0 && c < 32) return `${RANKS[cardRank(c)]}${SUITS[cardSuit(c)]}`;
                return '?';
            }).join(' ');
        }

        const leadLabel = leadSeat >= 0 ? SEAT_L[leadSeat] : '?';
        const winnerName = SEAT_NAMES_FR[t.winner];
        row.innerHTML = `<span class="trick-num">#${i + 1}</span>` +
            `<span class="trick-lead-label">${leadLabel}</span>` +
            `<span class="trick-cards">${orderedCards}</span>` +
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
    replaySession = false;
    moveHistory = [];
    historyIndex = -1;
    initialState = data.state;

    document.getElementById('watch-main').classList.remove('hidden');
    document.getElementById('watch-start').disabled = false;
    document.getElementById('watch-start').textContent = 'Relancer';
    if (data.game_id) setWatchGameId(data.game_id);

    // Disable DouDou if not available
    if (!data.doudou_available) {
        document.querySelectorAll('.agent-select option[value="doudou"]').forEach(opt => {
            opt.disabled = true;
            opt.textContent = 'DouDou (non dispo)';
        });
    }

    renderHistoryEntry(-1);
});

onMessage('watch_move', (data) => {
    waitingForStep = false;

    if (data.finished && !data.move) {
        watchFinished = true;
        stopAutoPlay();
        updateTransportButtons();
        return;
    }

    // Push to history buffer
    moveHistory.push(data);
    historyIndex = moveHistory.length - 1;

    renderHistoryEntry(historyIndex);

    if (data.belote_event) {
        const text = data.belote_event === 'belote' ? 'Belote !' : 'Rebelote !';
        showBeloteAnnouncement('watch-trick-area', text);
    }

    if (data.finished) {
        watchFinished = true;
        stopAutoPlay();
        updateTransportButtons();
        return;
    }

    // Continue auto-play if active
    if (autoPlayMode) {
        continueAutoPlay(data);
    }
});
