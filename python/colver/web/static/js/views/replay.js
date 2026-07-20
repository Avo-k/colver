// Replay view — browse and replay saved games

import { send, onMessage, offMessage } from '../ws.js';
import { SEAT_NAMES_FR, RANKS, SUITS, cardRank, cardSuit, actionName, bidActionHtml } from '../shared/cards.js';
import { BoardRenderer } from '../shared/board.js';
import { initCfnBox } from '../shared/cfn-box.js';
import { setGameId, setActionIdx, openBugReport } from '../shared/bug-report.js';

const TEMPLATE = `
<div id="replay-main">
    <div id="replay-history">
        <div class="section-title">Historique</div>
        <div id="replay-search">
            <input type="text" id="replay-search-input" placeholder="ID..." maxlength="4">
            <button id="replay-search-btn">Charger</button>
        </div>
        <div id="replay-list"></div>
    </div>

    <div id="replay-left">
        <div id="replay-score-bar">
            <span id="replay-score-ns">NS : 0</span>
            <span id="replay-game-id" class="game-id-tag hidden"></span>
            <span id="replay-contract-display"></span>
            <span id="replay-score-ew">EO : 0</span>
            <button id="replay-report-btn" class="report-btn hidden" title="Signaler un bug">Bug</button>
        </div>
        <div id="replay-cfn" class="cfn-box hidden" title="Cliquer pour copier"></div>

        <div class="seats">
            <div class="seat north">
                <div class="seat-label" id="replay-label-n">Nord</div>
                <div class="hand" id="replay-hand-north"></div>
            </div>
            <div class="seat west">
                <div class="seat-label" id="replay-label-w">Ouest</div>
                <div class="hand" id="replay-hand-west"></div>
            </div>
            <div id="replay-trick-area">
                <div class="trick-card" id="replay-trick-n"></div>
                <div class="trick-card" id="replay-trick-w"></div>
                <div class="trick-card" id="replay-trick-e"></div>
                <div class="trick-card" id="replay-trick-s"></div>
            </div>
            <div class="seat east">
                <div class="seat-label" id="replay-label-e">Est</div>
                <div class="hand" id="replay-hand-east"></div>
            </div>
            <div class="seat south">
                <div class="seat-label" id="replay-label-s">Sud</div>
                <div class="hand" id="replay-hand-south"></div>
            </div>
        </div>

        <div id="replay-last-trick" class="hidden"></div>
    </div>

    <div id="replay-right">
        <div id="replay-transport">
            <div class="transport-row">
                <button id="replay-prev-btn" title="Coup precedent">|\u25C0</button>
                <button id="replay-step-btn" title="Prochain coup">\u25B6|</button>
            </div>
            <div class="transport-row">
                <button id="replay-start-btn" title="Retour au debut">|\u25C0\u25C0</button>
                <button id="replay-auto-btn" title="Auto-play">\u25B6</button>
                <button id="replay-end-btn" title="Fin de partie">\u25B6\u25B6|</button>
            </div>
        </div>

        <div id="replay-stats-panel">
            <div id="replay-stats-header"></div>
            <div id="replay-stats-body"></div>
        </div>

        <div id="replay-analysis">
            <div class="section-title">Analyse Oracle</div>
            <div id="replay-analysis-body" class="analysis-body"></div>
        </div>

        <div id="replay-bid-history">
            <div class="section-title">Encheres</div>
            <div id="replay-bid-entries"></div>
        </div>

        <div id="replay-tricks-history">
            <div class="section-title">Plis</div>
            <div id="replay-tricks-list"></div>
        </div>
    </div>
</div>
`;

let replayBoard = null;
let replayTotalActions = 0;
let _pendingLoadId = null;
let _analysisByIdx = null;   // action_idx -> move analysis
let _analysisSummary = null;

const CATEGORY_UI = {
    parfait:     { tag: '✓',  cls: 'an-best',   label: 'Meilleur coup' },
    bon:         { tag: '✓',  cls: 'an-good',   label: 'Bon coup' },
    imprecision: { tag: '?!', cls: 'an-inacc',  label: 'Imprécision' },
    erreur:      { tag: '?',  cls: 'an-error',  label: 'Erreur' },
    faute:       { tag: '??', cls: 'an-blund',  label: 'Faute' },
};

function cardLabel(c) {
    const suit = cardSuit(c);
    const red = suit === 1 || suit === 2;
    return `<span class="${red ? 'an-red' : ''}">${RANKS[cardRank(c)]}${SUITS[suit]}</span>`;
}

function replayRenderMoveStats(move, state) {
    const header = replayBoard.el('stats-header');
    const body = replayBoard.el('stats-body');

    if (!move) {
        header.innerHTML = '';
        body.innerHTML = '';
        return;
    }

    const seatName = SEAT_NAMES_FR[move.player];
    const teamClass = move.player % 2 === 0 ? 'team-ns' : 'team-ew';

    header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> ` +
        `<span class="stats-player ${teamClass}">${seatName}</span>` +
        `<span class="stats-action">${move.phase === 0 ? bidActionHtml(move.action) : move.name}</span>`;
    body.innerHTML = '';

    // Oracle annotation for this move (history entry i == action index i)
    const an = _analysisByIdx && _analysisByIdx[replayBoard.historyIndex];
    if (an && move.phase === 1) {
        if (an.forced) {
            body.innerHTML = `<div class="an-move an-forced">Carte forcée</div>`;
        } else {
            const ui = CATEGORY_UI[an.category] || CATEGORY_UI.bon;
            let html = `<div class="an-move ${ui.cls}">` +
                `<span class="an-tag">${ui.tag}</span> ${ui.label}`;
            if (an.cost > 0) {
                html += ` <span class="an-cost">−${an.cost} pts</span>` +
                    `<span class="an-alt">Oracle : ${cardLabel(an.best)}</span>`;
            }
            html += '</div>';
            body.innerHTML = html;
        }
    }
}

function renderAnalysisSummary() {
    const el = document.getElementById('replay-analysis-body');
    if (!el) return;
    if (!_analysisSummary) {
        el.innerHTML = '<div class="an-loading">Analyse en cours…</div>';
        return;
    }
    let html = '<table class="an-table"><tr><th></th><th title="Coût total en points">Coût</th>' +
        '<th title="Coût moyen par décision">Moy.</th><th>?!</th><th>?</th><th>??</th></tr>';
    for (const p of _analysisSummary.players) {
        if (p.moves === 0) continue;
        const team = p.player % 2 === 0 ? 'team-ns' : 'team-ew';
        const c = p.counts;
        html += `<tr><td class="${team}">${SEAT_NAMES_FR[p.player]}</td>` +
            `<td>${p.total_cost}</td><td>${p.avg_cost}</td>` +
            `<td class="an-inacc">${c.imprecision || 0}</td>` +
            `<td class="an-error">${c.erreur || 0}</td>` +
            `<td class="an-blund">${c.faute || 0}</td></tr>`;
    }
    html += '</table>';
    el.innerHTML = html;
}

async function loadAnalysis(gameId) {
    _analysisByIdx = null;
    _analysisSummary = null;
    renderAnalysisSummary();
    try {
        const base = document.querySelector('base')?.getAttribute('href') || '/';
        const resp = await fetch(`${base}api/games/${gameId}/analysis`);
        if (!resp.ok) {
            const el = document.getElementById('replay-analysis-body');
            if (el) el.innerHTML = '<div class="an-loading">Analyse indisponible</div>';
            return;
        }
        const data = await resp.json();
        _analysisByIdx = {};
        for (const m of data.moves) _analysisByIdx[m.idx] = m;
        _analysisSummary = data.summary;
        renderAnalysisSummary();
        // Refresh the current move annotation now that data is here
        if (replayBoard) replayBoard.renderHistoryEntry(replayBoard.historyIndex);
    } catch {
        const el = document.getElementById('replay-analysis-body');
        if (el) el.innerHTML = '<div class="an-loading">Analyse indisponible</div>';
    }
}

function handleReplayLoaded(data) {
    replayTotalActions = data.total_actions || 0;
    setActionIdx(0);
    setReplayGameId(data.game_id);

    replayBoard.reset(data.state);

    document.getElementById('replay-main').classList.remove('hidden');

    replayBoard.renderHistoryEntry(-1);
    const header = replayBoard.el('stats-header');
    header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> <span class="stats-agent">${data.game_id}</span>`;

    loadAnalysis(data.game_id);
}

function handleReplayMove(data) {
    replayBoard.waitingForStep = false;
    setActionIdx(data.action_idx || 0);

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
}

function setReplayGameId(id) {
    setGameId(id);
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

function loadReplay(gameId) {
    send({ type: 'replay_load', game_id: gameId });
}

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

// Public API for cross-view navigation (play -> replay)
export function loadReplayById(gameId) {
    _pendingLoadId = gameId;
    // If already mounted, load immediately
    if (replayBoard) {
        loadGameHistory(false);
        loadReplay(gameId);
        _pendingLoadId = null;
    }
}

export function mount(container) {
    container.innerHTML = TEMPLATE;

    replayBoard = new BoardRenderer({
        prefix: 'replay',
        isReplay: true,
        renderMoveStats: replayRenderMoveStats,
        renderCardAnnotations: () => {},
        onRequestStep: () => {
            send({ type: 'replay_step' });
        },
    });

    replayBoard.bindTransport();
    replayBoard.bindKeyboard();

    initCfnBox('replay-cfn');

    // Search
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

    // Bug report
    document.getElementById('replay-report-btn').addEventListener('click', openBugReport);

    // Register WS handlers
    onMessage('replay_loaded', handleReplayLoaded);
    onMessage('replay_move', handleReplayMove);

    // Load history; if pending load from another view, use that
    if (_pendingLoadId) {
        loadGameHistory(false);
        loadReplay(_pendingLoadId);
        _pendingLoadId = null;
    } else {
        loadGameHistory(true);
    }
}

export function unmount() {
    offMessage('replay_loaded', handleReplayLoaded);
    offMessage('replay_move', handleReplayMove);

    if (replayBoard) {
        replayBoard.stopAutoPlay();
        replayBoard.unbindKeyboard();
        replayBoard.active = false;
        replayBoard = null;
    }
}
