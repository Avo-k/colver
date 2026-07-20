// Play view (solo humain vs IA) — thin wrapper around the shared GameTable

import { send, onMessage, offMessage } from '../ws.js';
import { GameTable, TABLE_TEMPLATE, MY_SEAT } from '../shared/table.js';
import { navigateTo } from '../router.js';

// ===== Template =====

const TEMPLATE = `
<div id="play-config">
    <p id="play-intro">Jouez à la Belote Contrée contre l'IA.</p>
    <label>Adversaires :
        <select id="opponent-ai">
            <option value="dede">Dédé (IS-DD)</option>
            <option value="doudou">DouDou50</option>
            <option value="oracle_dd">Oracle (DD)</option>
        </select>
    </label>
    <label>Partenaire :
        <select id="partner-ai">
            <option value="dede">Dédé (IS-DD)</option>
            <option value="doudou">DouDou50</option>
            <option value="oracle_dd">Oracle (DD)</option>
        </select>
    </label>
    <label>Pause :
        <input type="range" id="move-delay" min="1" max="8" value="2" step="1" style="width:80px">
        <span id="move-delay-val">2s</span>
    </label>
    <button id="start-game">Nouvelle Partie</button>
</div>
` + TABLE_TEMPLATE;

let table = null;

function getMoveDelay() {
    return parseInt(document.getElementById('move-delay').value);
}

// ===== WS message handlers (stored for offMessage) =====

function handleGameState(data) {
    table.handleGameState(data);
    // Disable DouDou options if not available on server
    if (data.doudou_available === false) {
        for (const selId of ['opponent-ai', 'partner-ai']) {
            const opt = document.querySelector(`#${selId} option[value="doudou"]`);
            if (opt) {
                opt.disabled = true;
                opt.textContent = 'DouDou50 (non dispo)';
            }
            const sel = document.getElementById(selId);
            if (sel.value === 'doudou') sel.value = 'smart';
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
            type: 'play', action, human_seat: MY_SEAT, move_delay: getMoveDelay(),
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

    document.getElementById('move-delay').addEventListener('input', (e) => {
        document.getElementById('move-delay-val').textContent = `${e.target.value}s`;
    });

    document.getElementById('play-config-toggle').addEventListener('click', () => {
        document.getElementById('play-config').classList.toggle('config-shown');
    });

    document.getElementById('start-game').addEventListener('click', () => {
        const opponentAi = document.getElementById('opponent-ai').value;
        const partnerAi = document.getElementById('partner-ai').value;
        table.reset();
        send({
            type: 'start_game', opponent_ai: opponentAi, partner_ai: partnerAi,
            human_seat: MY_SEAT, move_delay: getMoveDelay(),
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
