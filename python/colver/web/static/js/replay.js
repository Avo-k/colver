// Replay tab: browse and replay saved games

let replayTotalActions = 0;

// Simplified stats: just player + action name (no Q-values/DD scores)
function replayRenderMoveStats(move, state) {
    const header = replayBoard.el('stats-header');
    const body = replayBoard.el('stats-body');

    if (!move) {
        header.innerHTML = '';
        body.innerHTML = '';
        return;
    }

    const seatName = SEAT_NAMES_FR[move.player];
    const team = move.player % 2 === 0 ? 'NS' : 'EO';
    const teamClass = move.player % 2 === 0 ? 'team-ns' : 'team-ew';

    header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> ` +
        `<span class="stats-player ${teamClass}">${seatName}</span>` +
        `<span class="stats-action">${move.name}</span>`;
    body.innerHTML = '';
}

const replayBoard = new BoardRenderer({
    prefix: 'replay',
    isReplay: true,
    renderMoveStats: replayRenderMoveStats,
    renderCardAnnotations: () => {},
    onRequestStep: () => {
        send({ type: 'replay_step' });
    },
});

// ===== Game history sidebar =====

let replayTabVisited = false;

async function loadGameHistory(autoLoadFirst = false) {
    try {
        const base = document.querySelector('base')?.getAttribute('href') || '/';
        const resp = await fetch(`${base}api/games?limit=50`);
        if (!resp.ok) return;
        const games = await resp.json();
        renderGameHistory(games);
        if (autoLoadFirst && games.length > 0) {
            loadReplay(games[0].id);
        }
    } catch (e) {
        console.error('Failed to load history:', e);
    }
}

function renderGameHistory(games) {
    const list = document.getElementById('replay-list');
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

        const info = document.createElement('span');
        info.className = 'history-info';
        const nsWon = g.points_ns > g.points_ew;
        info.textContent = `${g.points_ns}-${g.points_ew}`;
        info.classList.add(nsWon ? 'ns-won' : 'ew-won');

        const date = document.createElement('span');
        date.className = 'history-date';
        const d = new Date(g.created_at);
        date.textContent = `${d.getHours().toString().padStart(2,'0')}:${d.getMinutes().toString().padStart(2,'0')}`;
        date.title = d.toLocaleString();

        row.appendChild(id);
        row.appendChild(info);
        row.appendChild(date);
        list.appendChild(row);
    }
}

function loadReplay(gameId) {
    send({ type: 'replay_load', game_id: gameId });
}

// Search by ID
document.getElementById('replay-search-btn').addEventListener('click', () => {
    const input = document.getElementById('replay-search-input');
    const id = input.value.trim().toLowerCase();
    if (id.length >= 1) loadReplay(id);
});

document.getElementById('replay-search-input').addEventListener('keydown', (e) => {
    if (e.key === 'Enter') {
        const id = e.target.value.trim().toLowerCase();
        if (id.length >= 1) loadReplay(id);
    }
});

// Load history when Replay tab is activated (auto-load first game on initial visit)
const replayTab = document.querySelector('.tab[data-tab="replay"]');
if (replayTab) {
    replayTab.addEventListener('click', () => {
        const firstVisit = !replayTabVisited;
        replayTabVisited = true;
        loadGameHistory(firstVisit);
    });
}

// ===== Game ID display =====

function setReplayGameId(id) {
    currentGameId = id;
    const el = document.getElementById('replay-game-id');
    if (id) {
        el.textContent = id;
        el.classList.remove('hidden');
        document.getElementById('replay-report-btn').classList.remove('hidden');
    } else {
        el.classList.add('hidden');
        document.getElementById('replay-report-btn').classList.add('hidden');
    }
}

// ===== Replay message handlers =====

onMessage('replay_loaded', (data) => {
    replayTotalActions = data.total_actions || 0;
    currentActionIdx = 0;
    setReplayGameId(data.game_id);

    replayBoard.reset(data.state);

    document.getElementById('replay-main').classList.remove('hidden');

    replayBoard.renderHistoryEntry(-1);
    const header = replayBoard.el('stats-header');
    header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> <span class="stats-agent">${data.game_id}</span>`;
});

onMessage('replay_move', (data) => {
    replayBoard.waitingForStep = false;
    currentActionIdx = data.action_idx || 0;

    if (data.finished && !data.move) {
        replayBoard.finished = true;
        replayBoard.stopAutoPlay();
        replayBoard.updateTransportButtons();
        return;
    }

    replayBoard.pushMove(data);
    replayBoard.renderHistoryEntry(replayBoard.historyIndex);
    replayBoard.handleBeloteEvent(data);

    if (data.finished) {
        replayBoard.finished = true;
        replayBoard.stopAutoPlay();
        replayBoard.updateTransportButtons();
        return;
    }

    if (replayBoard.autoPlayMode) replayBoard.continueAutoPlay(data);
});

// Bug report button for replay tab
document.getElementById('replay-report-btn').addEventListener('click', openBugReport);
