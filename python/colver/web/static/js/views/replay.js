// Replay view — browse and replay saved games with oracle analysis.
// The whole game is preloaded at replay_load, so navigation (buttons,
// arrows, clicking any move in the "Coups" list) is instant and local.

import { send, onMessage, offMessage } from '../ws.js';
import { navigateTo } from '../router.js';
import {
    SEAT_NAMES_FR, SUITS, cardCode, cardChipHtml,
    bidChipHtml, actionName, SUIT_DISPLAY_ORDER, _animatingTrick
} from '../shared/cards.js';
import { BoardRenderer } from '../shared/board.js';
import { initCfnBox, updateCfnBox } from '../shared/cfn-box.js';
import { setGameId, setActionIdx, openBugReport } from '../shared/bug-report.js';

const SEAT_INITIALS = ['N', 'E', 'S', 'O'];

const TEMPLATE = `
<div id="replay-main">
    <div id="replay-history">
        <div class="section-title" id="replay-history-title">Historique</div>
        <div id="replay-search">
            <input type="text" id="replay-search-input" placeholder="ID..." maxlength="4">
            <button id="replay-search-btn">Charger</button>
        </div>
        <div id="replay-list"></div>
    </div>

    <div id="replay-left">
        <div id="replay-score-bar">
            <span id="replay-score-ns">Nord-Sud : 0</span>
            <span id="replay-game-id" class="game-id-tag hidden"></span>
            <span id="replay-contract-display"></span>
            <span id="replay-score-ew">Est-Ouest : 0</span>
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
                <button id="replay-prev-btn" title="Coup precedent">|◀</button>
                <button id="replay-step-btn" title="Prochain coup">▶|</button>
            </div>
            <div class="transport-row">
                <button id="replay-start-btn" title="Retour au debut">|◀◀</button>
                <button id="replay-auto-btn" title="Auto-play">▶</button>
                <button id="replay-end-btn" title="Fin de partie">▶▶|</button>
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

        <div id="replay-moves">
            <div class="section-title">Coups</div>
            <div id="replay-moves-list"></div>
        </div>
    </div>
</div>
`;

let replayBoard = null;
let replayTotalActions = 0;
let _pendingLoadId = null;
let _analysisByIdx = null;   // action_idx -> move analysis
let _analysisSummary = null;
let _bidsByIdx = null;       // action_idx -> bid analysis (model annonce)
let _oracleBids = null;      // deal-level DD contracts {suits, best}
let _initialHands = null;    // all 4 hands at deal start (replay = full info)
let _agentsByIdx = null;     // action_idx -> {doudou, oracle, isdd} card choices
let _agentsPending = false;  // the review is still being computed server-side
let _agentsDone = 0;         // cards reviewed so far (streamed, in play order)
let _agentsTotal = 0;        // cards the review will cover
let _currentGameId = null;   // guards late answers from a previously-open game

// The three reference bots, in the order they are shown under a played card.
const REVIEW_BOTS = [
    { key: 'doudou', label: 'DouDou50', title: 'Réseau Q direct, sans recherche' },
    { key: 'oracle', label: 'Oracle',   title: 'Solveur double-dummy — voit les 4 mains' },
    { key: 'isdd',   label: 'Dédé',     title: 'IS-DD : mondes playgen résolus en double-dummy' },
];

const CATEGORY_UI = {
    parfait:     { tag: '✓',  cls: 'an-best',   label: 'Meilleur coup' },
    bon:         { tag: '✓',  cls: 'an-good',   label: 'Bon coup' },
    imprecision: { tag: '?!', cls: 'an-inacc',  label: 'Imprécision' },
    erreur:      { tag: '?',  cls: 'an-error',  label: 'Erreur' },
    faute:       { tag: '??', cls: 'an-blund',  label: 'Faute' },
};

// Navigate to the annonces analysis page pre-filled with the acting player's
// hand and the auction history up to (not including) this bid.
function openBidAnalysis(idx) {
    if (!replayBoard || !_initialHands) return;
    const data = replayBoard.moveHistory[idx];
    if (!data || !data.move || data.move.phase !== 0) return;
    const hand = _initialHands[data.move.player];
    if (!hand || hand.length !== 8) return;
    const history = [];
    for (let i = 0; i < idx; i++) {
        const m = replayBoard.moveHistory[i].move;
        if (m && m.phase === 0) history.push(m.action);
    }
    let url = `/analyse/annonces?hand=${hand.map(cardCode).join(',')}`;
    if (history.length) url += `&history=${history.join(',')}`;
    navigateTo(url);
}

// ===== "Qui aurait joué quoi" (the three reference bots) =====

// One row of chips: what each bot would have played at this exact position,
// whoever actually played it. A chip is green when the bot agrees with the
// card that was played, grey otherwise.
function botsHtml(idx, played) {
    const entry = _agentsByIdx && _agentsByIdx[idx];
    if (!entry) {
        // Cards are reviewed in play order, so an absent entry on a pending
        // review just means the search has not reached this one yet.
        if (!_agentsPending) return '';
        const progress = _agentsTotal ? ` ${_agentsDone}/${_agentsTotal}` : '';
        return `<div class="an-bots an-bots-loading">Analyse des bots…${progress}</div>`;
    }
    if (entry.forced) return '';

    const chips = REVIEW_BOTS.map(bot => {
        const card = entry[bot.key];
        if (card === null || card === undefined) return '';
        const same = card === played;
        return `<span class="an-bot ${same ? 'an-bot-same' : 'an-bot-diff'}" ` +
            `title="${bot.title}">` +
            `<span class="an-bot-name">${bot.label}</span>` +
            `<span class="an-bot-card">${cardChipHtml(card)}</span></span>`;
    }).filter(Boolean).join('');

    return chips ? `<div class="an-bots">${chips}</div>` : '';
}

// ===== Move stats (current move annotation) =====

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
        `<span class="stats-action">${move.phase === 0 ? bidChipHtml(move.action) : move.name}</span>`;
    body.innerHTML = '';

    // Bid move: model annonce + oracle annonce + link to the annonces page
    if (move.phase === 0) {
        const idx = replayBoard.historyIndex;
        let html = '';
        const bid = _bidsByIdx && _bidsByIdx[idx];
        if (bid) {
            const agree = bid.model_best === move.action;
            html += `<div class="an-move ${agree ? 'an-best' : 'an-inacc'}">` +
                `<span class="an-tag">${agree ? '✓' : '≠'}</span>` +
                `Bid V6 : ${bidChipHtml(bid.model_best)}` +
                `<span class="an-bid-q">Q ${bid.q_best.toFixed(2)}` +
                (!agree && bid.q_played !== null && bid.q_played !== undefined
                    ? ` · joué ${bid.q_played.toFixed(2)}` : '') +
                `</span></div>`;
        }
        html += `<button class="an-bid-analyse-btn" id="replay-bid-analyse-btn">Analyser cette annonce →</button>`;
        body.innerHTML = html;
        const btn = document.getElementById('replay-bid-analyse-btn');
        if (btn) btn.addEventListener('click', () => openBidAnalysis(idx));
        return;
    }

    if (move.phase !== 1) return;

    const idx = replayBoard.historyIndex;
    // Oracle annotation for this move (history entry i == action index i)
    const an = _analysisByIdx && _analysisByIdx[idx];
    let html = '';
    if (an) {
        if (an.forced) {
            html = `<div class="an-move an-forced">Carte forcée</div>`;
        } else {
            const ui = CATEGORY_UI[an.category] || CATEGORY_UI.bon;
            html = `<div class="an-move ${ui.cls}">` +
                `<span class="an-tag">${ui.tag}</span> ${ui.label}`;
            if (an.cost > 0) {
                html += ` <span class="an-cost">−${an.cost} pts</span>` +
                    `<span class="an-alt">Oracle : ${cardChipHtml(an.best)}</span>`;
            }
            html += '</div>';
        }
    }
    body.innerHTML = html + botsHtml(idx, move.action);
}

// ===== Navigable moves list =====

function jumpTo(i) {
    if (!replayBoard || !replayBoard.active) return;
    if (_animatingTrick === 'replay-trick') return;
    replayBoard.stopAutoPlay();
    replayBoard._pendingFlushData = null;
    replayBoard.historyIndex = i;
    replayBoard.renderHistoryEntry(i);
}

// Garder le coup courant en vue en ne bougeant QUE l'ascenseur de la liste.
// scrollIntoView() remonte tous les conteneurs défilants : sur mobile, où la
// liste s'affiche en entier dans la page, il faisait sauter la page sur la
// liste à chaque coup — le tapis disparaissait de l'écran.
function scrollIntoList(list, el) {
    const lr = list.getBoundingClientRect();
    const er = el.getBoundingClientRect();
    if (er.top < lr.top) list.scrollTop -= lr.top - er.top;
    else if (er.bottom > lr.bottom) list.scrollTop += er.bottom - lr.bottom;
}

function updateMovesHighlight() {
    const list = document.getElementById('replay-moves-list');
    if (!list || !replayBoard) return;
    const idx = replayBoard.historyIndex;
    setActionIdx(idx + 1);
    let current = null;
    list.querySelectorAll('[data-idx]').forEach(el => {
        const isCurrent = parseInt(el.dataset.idx) === idx;
        el.classList.toggle('mv-current', isCurrent);
        if (isCurrent) current = el;
    });
    if (current) scrollIntoList(list, current);
}

function buildMovesList() {
    const list = document.getElementById('replay-moves-list');
    if (!list || !replayBoard) return;
    list.innerHTML = '';

    const bids = document.createElement('div');
    bids.className = 'mv-bids';
    let trickRow = null;
    let cardsInRow = 0;
    let trickNum = 0;

    replayBoard.moveHistory.forEach((data, i) => {
        const m = data.move;
        if (!m) return;
        if (m.phase === 0) {
            const chip = document.createElement('span');
            chip.className = 'mv-bid ' + (m.player % 2 === 0 ? 'team-ns' : 'team-ew');
            chip.dataset.idx = i;
            chip.innerHTML = `${SEAT_INITIALS[m.player]}&nbsp;${bidChipHtml(m.action)}`;
            chip.title = SEAT_NAMES_FR[m.player];
            const bid = _bidsByIdx && _bidsByIdx[i];
            if (bid && bid.model_best !== m.action) {
                chip.classList.add('mv-bid-diff');
                chip.title += ` — Bid V6 : ${actionName(bid.model_best, 0)}`;
            }
            chip.addEventListener('click', () => jumpTo(i));
            bids.appendChild(chip);
        } else {
            if (cardsInRow === 0) {
                trickNum++;
                trickRow = document.createElement('div');
                trickRow.className = 'mv-trick';
                const num = document.createElement('span');
                num.className = 'mv-num';
                num.textContent = trickNum;
                trickRow.appendChild(num);
                list.appendChild(trickRow);
            }
            const an = _analysisByIdx && _analysisByIdx[i];
            const cardEl = document.createElement('span');
            let cls = 'mv-card';
            let tip = SEAT_NAMES_FR[m.player];
            if (an) {
                if (an.forced) {
                    cls += ' mv-forced';
                    tip += ' — carte forcée';
                } else {
                    cls += ' mv-' + an.category;
                    const ui = CATEGORY_UI[an.category];
                    tip += ` — ${ui ? ui.label : an.category}`;
                    if (an.cost > 0) tip += ` (−${an.cost} pts)`;
                }
            }
            cardEl.className = cls;
            cardEl.dataset.idx = i;
            cardEl.innerHTML = cardChipHtml(m.action);
            cardEl.title = tip;
            cardEl.addEventListener('click', () => jumpTo(i));
            trickRow.appendChild(cardEl);
            cardsInRow++;

            if (cardsInRow === 4) {
                cardsInRow = 0;
                const w = data.state.last_trick_winner;
                if (w !== null && w !== undefined) {
                    const win = document.createElement('span');
                    win.className = 'mv-winner ' + (w % 2 === 0 ? 'team-ns' : 'team-ew');
                    win.textContent = `${SEAT_INITIALS[w]} +${data.state.last_trick_points}`;
                    trickRow.appendChild(win);
                }
            }
        }
    });

    if (bids.children.length) list.prepend(bids);
    updateMovesHighlight();
}

// ===== Oracle analysis =====

function renderAnalysisSummary() {
    const el = document.getElementById('replay-analysis-body');
    if (!el) return;
    if (!_analysisSummary) {
        el.innerHTML = '<div class="an-loading">Analyse en cours…</div>';
        return;
    }
    let html = '';
    if (_oracleBids && _oracleBids.suits) {
        // Deal-level DD scores per trump suit — static info, no bid "decision"
        html += '<table class="an-dd-table" title="Points réalisables en jeu parfait (double-dummy) pour chaque atout">' +
            '<tr><th></th><th class="team-ns">Nord-Sud</th><th class="team-ew">Est-Ouest</th></tr>';
        for (const suit of SUIT_DISPLAY_ORDER) {
            const [ns, ew] = _oracleBids.suits[suit];
            const red = suit === 1 || suit === 2;
            html += `<tr><td class="${red ? 'an-red' : ''}">${SUITS[suit]}</td>` +
                `<td>${ns}</td><td>${ew}</td></tr>`;
        }
        html += '</table>';
    }
    html += '<table class="an-table"><tr><th></th><th title="Coût total en points">Coût</th>' +
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
    _bidsByIdx = null;
    _oracleBids = null;
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
        _bidsByIdx = {};
        for (const b of data.bids || []) _bidsByIdx[b.idx] = b;
        _oracleBids = data.oracle_bids || null;
        _analysisSummary = data.summary;
        renderAnalysisSummary();
        // Recolor the moves list and refresh the current annotation
        if (replayBoard) {
            buildMovesList();
            replayBoard.renderHistoryEntry(replayBoard.historyIndex);
        }
    } catch {
        const el = document.getElementById('replay-analysis-body');
        if (el) el.innerHTML = '<div class="an-loading">Analyse indisponible</div>';
    }
}

// ===== Bot review (streamed over the WebSocket) =====

// Re-render only the annotation panel. Going through renderHistoryEntry would
// redraw the board, and it carries navigation state (flush pending, forward vs
// backward) that a mid-playback refresh would trample.
function refreshMoveStats() {
    if (!replayBoard || !replayBoard.active) return;
    const data = replayBoard.moveHistory[replayBoard.historyIndex];
    if (!data || !data.move || data.finished) return;
    replayRenderMoveStats(data.move, data.state);
}

// The review is ~9s of IS-DD search on first load and cached server-side
// afterwards. It streams card by card in play order, so the opening lead is
// annotated while the endgame is still being searched.
function requestAgentReview(gameId) {
    _agentsByIdx = null;
    _agentsPending = true;
    _agentsDone = 0;
    _agentsTotal = 0;
    send({ type: 'replay_agents', game_id: gameId });
}

function handleAgentReviewStart(data) {
    if (data.game_id !== _currentGameId) return;
    _agentsByIdx = {};
    _agentsPending = true;
    _agentsDone = 0;
    _agentsTotal = data.total || 0;
    refreshMoveStats();
}

function handleAgentReviewMove(data) {
    if (data.game_id !== _currentGameId || !data.move) return;
    if (_agentsByIdx === null) _agentsByIdx = {};
    _agentsByIdx[data.move.idx] = data.move;
    _agentsDone++;
    // Repaint when this is the card on screen, or while the panel is still
    // showing the progress counter for a card that has not landed yet.
    const shown = _agentsByIdx[replayBoard ? replayBoard.historyIndex : -1];
    if (!shown || shown === data.move) refreshMoveStats();
}

function handleAgentReviewDone(data) {
    if (data.game_id !== _currentGameId) return;
    const byIdx = {};
    for (const m of data.moves || []) byIdx[m.idx] = m;
    _agentsByIdx = byIdx;
    _agentsPending = false;
    refreshMoveStats();
}

function handleAgentReviewError(data) {
    if (data.game_id !== _currentGameId) return;
    _agentsPending = false;
    refreshMoveStats();
}

// ===== Load / history =====

function handleReplayLoaded(data) {
    replayTotalActions = data.total_actions || 0;
    _currentGameId = data.game_id;
    setActionIdx(0);
    setReplayGameId(data.game_id);
    // Replay states expose all 4 hands; deal-start hands feed the annonces link
    _initialHands = (data.state && data.state.hands) || null;

    replayBoard.reset(data.state);
    if (data.moves) {
        for (const m of data.moves) replayBoard.moveHistory.push(m);
        replayBoard.finished = true;
    }
    replayBoard.historyIndex = -1;
    replayBoard._prevHistoryIndex = -1;

    document.getElementById('replay-main').classList.remove('hidden');
    replayBoard.renderHistoryEntry(-1);

    // Full-game CFN (auction + play): click-to-copy, paste into the belief page.
    updateCfnBox('replay-cfn', data.game_cfn);

    const header = replayBoard.el('stats-header');
    header.innerHTML = `<span class="stats-replay-tag">REPLAY</span> <span class="stats-agent">${data.game_id}</span>`;

    buildMovesList();
    loadAnalysis(data.game_id);
    requestAgentReview(data.game_id);
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
        let mine = false;
        try {
            const meResp = await fetch(`${base}api/me`);
            mine = meResp.ok && !!(await meResp.json()).user;
        } catch { /* anonymous */ }
        const title = document.getElementById('replay-history-title');
        if (title) title.textContent = mine ? 'Mes parties' : 'Historique';
        const url = mine ? `${base}api/me/games?limit=50` : `${base}api/games?limit=50`;
        const resp = await fetch(url);
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

function contractLabel(g) {
    const c = g.contract;
    if (!c || !c.value) return '<span class="history-nocontract">passée</span>';
    const suit = c.trump;
    const red = suit === 1 || suit === 2;
    const mult = c.coinche === 2 ? ' ×3' : c.coinche === 1 ? ' ×2' : '';
    const val = c.value === 250 ? 'Capot' : c.value;
    return `${val}<span class="${red ? 'an-red' : ''}">${SUITS[suit]}</span>${mult}`;
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

        const contract = document.createElement('span');
        contract.className = 'history-contract';
        contract.innerHTML = contractLabel(g);

        // Score from the viewer's team perspective when known
        const seat = g.user_seat ?? g.human_seat;
        const mine = seat !== null && seat !== undefined;
        const isNS = mine ? seat % 2 === 0 : true;
        const a = isNS ? g.points_ns : g.points_ew;
        const b = isNS ? g.points_ew : g.points_ns;
        const info = document.createElement('span');
        info.className = 'history-info';
        info.textContent = `${a}-${b}`;
        info.classList.add(a > b ? 'ns-won' : 'ew-won');

        const date = document.createElement('span');
        date.className = 'history-date';
        const d = new Date(g.created_at);
        date.textContent = d.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' });

        row.title = `${g.id} — ${d.toLocaleString()}`;
        row.appendChild(contract);
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
        renderCardAnnotations: () => updateMovesHighlight(),
        onRequestStep: () => {},
        // Rejoué : ramassage de pli 3× plus rapide, et pause écourtée sur la
        // frame « 4 cartes sur la table ».
        flushDuration: 533,
        flushHoldDelay: 333,
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
    onMessage('agent_review_start', handleAgentReviewStart);
    onMessage('agent_review_move', handleAgentReviewMove);
    onMessage('agent_review_done', handleAgentReviewDone);
    onMessage('agent_review_error', handleAgentReviewError);

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
    offMessage('agent_review_start', handleAgentReviewStart);
    offMessage('agent_review_move', handleAgentReviewMove);
    offMessage('agent_review_done', handleAgentReviewDone);
    offMessage('agent_review_error', handleAgentReviewError);

    if (replayBoard) {
        replayBoard.stopAutoPlay();
        replayBoard.unbindKeyboard();
        replayBoard.active = false;
        replayBoard = null;
    }
    _analysisByIdx = null;
    _analysisSummary = null;
    _bidsByIdx = null;
    _oracleBids = null;
    _initialHands = null;
    _agentsByIdx = null;
    _agentsPending = false;
    _agentsDone = 0;
    _agentsTotal = 0;
    _currentGameId = null;
}
