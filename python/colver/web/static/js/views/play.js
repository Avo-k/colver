// Play view (solo humain vs IA) — thin wrapper around the shared GameTable

import { send, onMessage, offMessage } from '../ws.js';
import { GameTable, TABLE_TEMPLATE, MY_SEAT } from '../shared/table.js';
import { navigateTo } from '../router.js';

// ===== Template =====

const MODE_KEY = 'colver_play_mode';
const TARGET_KEY = 'colver_play_target';
const MODES = {
    standard: { label: 'Standard', bot: 'Dédé', hint: '≈ 40 s la donne' },
    rapide: { label: 'Rapide', bot: 'DouDou50', hint: '≈ 15 s la donne' },
};
// Format : une donne isolée, ou une partie jusqu'à un score. Les cibles doivent
// rester alignées sur match_state.TARGETS côté serveur, qui refuse le reste.
const TARGETS = [
    { value: 0, label: 'Une donne', hint: 'une main, un résultat' },
    { value: 1000, label: '1000 points', hint: '≈ 7 donnes' },
    { value: 2000, label: '2000 points', hint: '≈ 14 donnes' },
];

const TEMPLATE = `
<div id="play-config">
    <p id="play-intro">Jouez à la Belote Contrée contre l'IA.</p>
    <div class="config-group">
        <span class="config-group-label">Rythme</span>
        <div id="mode-choice" class="mode-choice">
            ${Object.entries(MODES).map(([key, m]) => `
            <button type="button" class="mode-btn" data-mode="${key}">
                <span class="mode-btn-label">${m.label}</span>
                <span class="mode-btn-sub">${m.bot} · ${m.hint}</span>
            </button>`).join('')}
        </div>
    </div>
    <div class="config-group">
        <span class="config-group-label">Format</span>
        <div id="target-choice" class="mode-choice">
            ${TARGETS.map(t => `
            <button type="button" class="mode-btn" data-target="${t.value}">
                <span class="mode-btn-label">${t.label}</span>
                <span class="mode-btn-sub">${t.hint}</span>
            </button>`).join('')}
        </div>
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
let currentTarget = 0;
// Une nouvelle donne se reconnaît à son identifiant : c'est lui qui déclenche
// la remise à zéro de la table, que la donne vienne d'un lancement ou de
// l'enchaînement d'une partie.
let lastGameId = null;

function setMode(mode) {
    currentMode = Object.hasOwn(MODES, mode) ? mode : 'standard';
    localStorage.setItem(MODE_KEY, currentMode);
    for (const btn of document.querySelectorAll('#mode-choice .mode-btn')) {
        btn.classList.toggle('mode-btn-active', btn.dataset.mode === currentMode);
    }
}

function setTarget(target) {
    const value = Number(target);
    currentTarget = TARGETS.some(t => t.value === value) ? value : 0;
    localStorage.setItem(TARGET_KEY, String(currentTarget));
    for (const btn of document.querySelectorAll('#target-choice .mode-btn')) {
        btn.classList.toggle('mode-btn-active',
            Number(btn.dataset.target) === currentTarget);
    }
}

// ===== WS message handlers (stored for offMessage) =====

function handleGameState(data) {
    if (data.game_id && data.game_id !== lastGameId) {
        // Donne suivante d'une partie : même table, tout est à refaire.
        lastGameId = data.game_id;
        table.reset();
        table.show();
    }
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

    const analyseButton = {
        label: 'Analyser', className: 'result-analyse',
        onClick: (gameId) => {
            if (!gameId) return;
            navigateTo('/analyse/rejouer');
            import('../views/replay.js').then(m => m.loadReplayById(gameId));
        },
    };

    table = new GameTable({
        sendMove: (action) => send({
            type: 'play', action, human_seat: MY_SEAT,
        }),
        localEchoBids: true,
        // Tant que la partie n'est pas jouée, le bouton principal enchaîne la
        // donne suivante ; sinon il en relance une (avec le format choisi).
        resultButtons: ({ match }) => [
            match && match.target > 0 && !match.finished
                ? {
                    label: 'Donne suivante', className: 'result-restart',
                    onClick: () => send({ type: 'next_deal' }),
                }
                : {
                    label: 'Nouvelle partie', className: 'result-restart',
                    onClick: () => document.getElementById('start-game').click(),
                },
            analyseButton,
        ],
    });
    table.bind();

    for (const btn of document.querySelectorAll('#mode-choice .mode-btn')) {
        btn.addEventListener('click', () => setMode(btn.dataset.mode));
    }
    setMode(localStorage.getItem(MODE_KEY));
    for (const btn of document.querySelectorAll('#target-choice .mode-btn')) {
        btn.addEventListener('click', () => setTarget(btn.dataset.target));
    }
    setTarget(localStorage.getItem(TARGET_KEY));

    document.getElementById('play-config-toggle').addEventListener('click', () => {
        document.getElementById('play-config').classList.toggle('config-shown');
    });

    document.getElementById('start-game').addEventListener('click', () => {
        table.reset();
        lastGameId = null;
        send({
            type: 'start_game', mode: currentMode, target: currentTarget,
            human_seat: MY_SEAT,
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
    lastGameId = null;
    if (table) {
        table.unbind();
        table = null;
    }
}
