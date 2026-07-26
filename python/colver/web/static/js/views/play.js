// Play view (solo humain vs IA) — thin wrapper around the shared GameTable

import { send, onMessage, offMessage } from '../ws.js';
import { GameTable, TABLE_TEMPLATE, MY_SEAT } from '../shared/table.js';
import { navigateTo } from '../router.js';

// ===== Template =====

const MODE_KEY = 'colver_play_mode';
const MODES = {
    standard: { label: 'Standard', bot: 'Dédé', hint: '≈ 40 s la donne' },
    rapide: { label: 'Rapide', bot: 'DouDou50', hint: '≈ 15 s la donne' },
};

const TEMPLATE = `
<div id="play-config">
    <p id="play-intro">Jouez à la Belote Contrée contre l'IA.</p>
    <div id="mode-choice" class="mode-choice">
        ${Object.entries(MODES).map(([key, m]) => `
        <button type="button" class="mode-btn" data-mode="${key}">
            <span class="mode-btn-label">${m.label}</span>
            <span class="mode-btn-sub">${m.bot} · ${m.hint}</span>
        </button>`).join('')}
    </div>
    <p id="mode-note" class="mode-note hidden"></p>
    <button id="start-game">Nouvelle Partie</button>
</div>
` + TABLE_TEMPLATE;

let table = null;
// The mode picks the tempo *and* the bot: a fast tempo only makes sense behind
// a bot that answers instantly, so the two are one choice. Server-side truth
// lives in python/colver/web/pacing.py.
let currentMode = 'standard';

function setMode(mode) {
    currentMode = Object.hasOwn(MODES, mode) ? mode : 'standard';
    localStorage.setItem(MODE_KEY, currentMode);
    for (const btn of document.querySelectorAll('#mode-choice .mode-btn')) {
        btn.classList.toggle('mode-btn-active', btn.dataset.mode === currentMode);
    }
}

// ===== WS message handlers (stored for offMessage) =====

function handleGameState(data) {
    table.handleGameState(data);
    // The server resolves the mode: say so when it could not seat the bot the
    // mode advertises, instead of silently playing a different opponent.
    const note = document.getElementById('mode-note');
    if (note && data.mode) {
        note.classList.toggle('hidden', !data.mode_degraded);
        if (data.mode_degraded) {
            note.textContent = 'DouDou50 est indisponible sur le serveur : '
                + 'Dédé prend sa place, avec un budget de réflexion réduit.';
        }
    }
}

function handleAiMove(data) {
    table.handleMove(data);
}

function handleError(data) {
    console.error('Erreur serveur:', data.msg);
    const statusEl = document.getElementById('play-status');
    if (statusEl) statusEl.textContent = `Erreur : ${data.msg}`;
}

// ===== Lifecycle =====

export function mount(container) {
    container.innerHTML = TEMPLATE;

    table = new GameTable({
        sendMove: (action) => send({
            type: 'play', action, human_seat: MY_SEAT,
        }),
        localEchoBids: true,
        resultButtons: [
            {
                label: 'Nouvelle partie', className: 'result-restart',
                onClick: () => document.getElementById('start-game').click(),
            },
            {
                label: 'Analyser', className: 'result-analyse',
                onClick: (gameId) => {
                    if (!gameId) return;
                    navigateTo('/analyse/rejouer');
                    import('../views/replay.js').then(m => m.loadReplayById(gameId));
                },
            },
        ],
    });
    table.bind();

    for (const btn of document.querySelectorAll('#mode-choice .mode-btn')) {
        btn.addEventListener('click', () => setMode(btn.dataset.mode));
    }
    setMode(localStorage.getItem(MODE_KEY));

    document.getElementById('play-config-toggle').addEventListener('click', () => {
        document.getElementById('play-config').classList.toggle('config-shown');
    });

    document.getElementById('start-game').addEventListener('click', () => {
        table.reset();
        send({
            type: 'start_game', mode: currentMode, human_seat: MY_SEAT,
        });
        table.show();
        document.getElementById('play-status').textContent = 'Lancement de la partie...';
    });

    onMessage('game_state', handleGameState);
    onMessage('ai_move', handleAiMove);
    onMessage('error', handleError);
}

export function unmount() {
    offMessage('game_state', handleGameState);
    offMessage('ai_move', handleAiMove);
    offMessage('error', handleError);
    if (table) {
        table.unbind();
        table = null;
    }
}
