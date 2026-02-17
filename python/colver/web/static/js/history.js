// Game history, replay, and bug report logic

let currentGameId = null;       // active game ID (play or watch)
let currentActionIdx = 0;       // for bug reports
let replaySession = false;      // true when replaying from DB
let replayTotalActions = 0;

// ===== Game history sidebar =====

async function loadGameHistory() {
    try {
        const base = document.querySelector('base')?.getAttribute('href') || '/';
        const resp = await fetch(`${base}api/games?limit=50`);
        if (!resp.ok) return;
        const games = await resp.json();
        renderGameHistory(games);
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function renderGameHistory(games) {
    const list = document.getElementById('history-list');
    list.innerHTML = '';
    if (games.length === 0) {
        list.innerHTML = '<div class="history-empty">Aucune partie</div>';
        return;
    }
    for (const g of games) {
        const row = document.createElement('div');
        row.className = 'history-row';
        row.addEventListener('click', () => loadReplay(g.id));

        const id = document.createElement('span');
        id.className = 'history-id';
        id.textContent = g.id;

        const mode = document.createElement('span');
        mode.className = 'history-mode';
        mode.textContent = g.mode === 'play' ? 'J' : 'R';
        mode.title = g.mode === 'play' ? 'Jouer' : 'Regarder';

        const info = document.createElement('span');
        info.className = 'history-info';
        if (g.is_complete && g.points_ns !== null) {
            const nsWon = g.points_ns > g.points_ew;
            info.textContent = `${g.points_ns}-${g.points_ew}`;
            info.classList.add(nsWon ? 'ns-won' : 'ew-won');
        } else {
            info.textContent = '...';
        }

        const date = document.createElement('span');
        date.className = 'history-date';
        const d = new Date(g.created_at);
        date.textContent = `${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
        date.title = d.toLocaleString();

        row.appendChild(id);
        row.appendChild(mode);
        row.appendChild(info);
        row.appendChild(date);
        list.appendChild(row);
    }
}

function loadReplay(gameId) {
    send({ type: 'replay_load', game_id: gameId });
}

// Search by ID
document.getElementById('history-search-btn').addEventListener('click', () => {
    const input = document.getElementById('history-search-input');
    const id = input.value.trim().toLowerCase();
    if (id.length >= 1) loadReplay(id);
});

document.getElementById('history-search-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const id = e.target.value.trim().toLowerCase();
        if (id.length >= 1) loadReplay(id);
    }
});

// Load history when Regarder tab is activated
const watchTab = document.querySelector('.tab[data-tab="watch"]');
if (watchTab) {
    watchTab.addEventListener('click', () => {
        loadGameHistory();
    });
}

// ===== Game ID display =====

function setPlayGameId(id) {
    currentGameId = id;
    const el = document.getElementById('play-game-id');
    if (id) {
        el.textContent = id;
        el.classList.remove('hidden');
        document.getElementById('play-report-btn').classList.remove('hidden');
    } else {
        el.classList.add('hidden');
        document.getElementById('play-report-btn').classList.add('hidden');
    }
}

function setWatchGameId(id) {
    currentGameId = id;
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

// ===== Replay messages =====

onMessage('replay_loaded', (data) => {
    replaySession = true;
    replayTotalActions = data.total_actions || 0;
    currentActionIdx = 0;
    setWatchGameId(data.game_id);

    // Reset watch state for replay
    watchActive = true;
    watchFinished = false;
    waitingForStep = false;
    autoPlayMode = null;

    document.getElementById('watch-main').classList.remove('hidden');
    document.getElementById('watch-start').disabled = false;
    document.getElementById('watch-start').textContent = 'Relancer';

    renderWatchState(data.state);
    renderWatchBidHistory([]);
    renderTricksHistory([]);

    const header = document.getElementById('watch-stats-header');
    const body = document.getElementById('watch-stats-body');
    header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> <span class="stats-agent">${data.game_id}</span>`;
    body.innerHTML = '<div class="stats-placeholder">Cliquez sur un bouton pour avancer</div>';
});

onMessage('replay_move', (data) => {
    waitingForStep = false;
    currentActionIdx = data.action_idx || 0;

    if (data.finished && !data.move) {
        watchFinished = true;
        stopAutoPlay();
        return;
    }

    renderWatchState(data.state);
    if (data.move) renderStats(data.move);
    renderWatchBidHistory(data.bid_history);
    renderTricksHistory(data.completed_tricks);

    if (data.belote_event) {
        const text = data.belote_event === 'belote' ? 'Belote !' : 'Rebelote !';
        showBeloteAnnouncement('watch-trick-area', text);
    }

    if (data.finished) {
        watchFinished = true;
        stopAutoPlay();
        const state = data.state;
        const header = document.getElementById('watch-stats-header');
        const body = document.getElementById('watch-stats-body');
        const nsWon = state.points[0] > state.points[1];
        header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> <span class="stats-result">${nsWon ? 'NS gagne' : 'EO gagne'} ${state.points[0]}-${state.points[1]}</span>`;
        body.innerHTML = `<div class="stats-final">NS: ${state.points[0]}pts (${state.tricks_won[0]}P) / EO: ${state.points[1]}pts (${state.tricks_won[1]}P)</div>`;
        return;
    }

    if (autoPlayMode) continueAutoPlay(data);
});

// ===== Bug report =====

function openBugReport() {
    if (!currentGameId) return;
    document.getElementById('report-game-label').textContent = `Partie : ${currentGameId}`;
    document.getElementById('report-message').value = '';
    document.getElementById('report-status').textContent = '';
    document.getElementById('report-modal').classList.remove('hidden');
    document.getElementById('report-message').focus();
}

document.getElementById('play-report-btn').addEventListener('click', openBugReport);
document.getElementById('watch-report-btn').addEventListener('click', openBugReport);

document.getElementById('report-cancel').addEventListener('click', () => {
    document.getElementById('report-modal').classList.add('hidden');
});

document.getElementById('report-submit').addEventListener('click', async () => {
    const message = document.getElementById('report-message').value.trim();
    if (!message) return;
    const statusEl = document.getElementById('report-status');
    statusEl.textContent = 'Envoi...';
    try {
        const base = document.querySelector('base')?.getAttribute('href') || '/';
        const resp = await fetch(`${base}api/games/${currentGameId}/report`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ message, action_idx: currentActionIdx }),
        });
        if (resp.ok) {
            statusEl.textContent = 'Envoye !';
            setTimeout(() => {
                document.getElementById('report-modal').classList.add('hidden');
            }, 1000);
        } else {
            statusEl.textContent = 'Erreur';
        }
    } catch (e) {
        statusEl.textContent = 'Erreur reseau';
    }
});

// Close modal on backdrop click
document.getElementById('report-modal').addEventListener('click', (e) => {
    if (e.target === e.currentTarget) {
        document.getElementById('report-modal').classList.add('hidden');
    }
});
