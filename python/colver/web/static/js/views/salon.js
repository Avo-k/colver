// Salon view — multiplayer rooms: create/join, lobby, then shared GameTable

import { send, onMessage, offMessage, onOpen, offOpen } from '../ws.js';
import { GameTable, TABLE_TEMPLATE } from '../shared/table.js';
import { navigateTo } from '../router.js';

const base = () => document.querySelector('base')?.getAttribute('href') || '/';

const BOT_LABELS = { dede: 'Dédé', doudou: 'DouDou', oracle_dd: 'Oracle' };
const SEAT_TITLES = ['Nord', 'Est', 'Sud', 'Ouest'];

const TEMPLATE = `
<div id="salon-root">
    <div id="salon-entry" class="salon-panel hidden">
        <h2 class="compte-title">Salon multijoueur</h2>
        <p class="salon-desc">Jouez à la Belote Contrée entre humains — les sièges vides sont tenus par l'IA.</p>
        <button id="salon-create" class="compte-submit">Créer un salon</button>
        <div class="salon-join-row">
            <input id="salon-code-input" class="compte-input" maxlength="4"
                   placeholder="code" autocomplete="off" spellcheck="false">
            <button id="salon-join" class="compte-submit">Rejoindre</button>
        </div>
        <div id="salon-entry-error" class="compte-error hidden"></div>
    </div>

    <div id="salon-login" class="salon-panel hidden">
        <h2 class="compte-title">Salon multijoueur</h2>
        <p class="salon-desc">Il faut un compte pour jouer en salon.</p>
        <button id="salon-goto-login" class="compte-submit">Se connecter / créer un compte</button>
    </div>

    <div id="salon-lobby" class="salon-panel hidden">
        <h2 class="compte-title">Salon <span id="salon-code" class="salon-code"></span></h2>
        <p class="salon-desc">Partagez ce code pour inviter d'autres joueurs.</p>
        <div class="salon-seats" id="salon-seats"></div>
        <div id="salon-host-controls" class="salon-host-controls hidden">
            <label>Bots :
                <select id="salon-bot-type">
                    <option value="dede">Dédé (IS-DD)</option>
                    <option value="doudou">DouDou50</option>
                    <option value="oracle_dd">Oracle (DD)</option>
                </select>
            </label>
            <label>Pause :
                <input type="range" id="salon-move-delay" min="1" max="8" value="2" step="1" style="width:80px">
                <span id="salon-move-delay-val">2s</span>
            </label>
            <button id="salon-start" class="compte-submit">Lancer la partie</button>
        </div>
        <div id="salon-lobby-status" class="salon-lobby-status"></div>
        <button id="salon-leave" class="compte-logout">Quitter le salon</button>
    </div>

    <div id="salon-game" class="hidden">${TABLE_TEMPLATE}</div>
    <div id="salon-toast" class="salon-toast hidden"></div>
</div>`;

let table = null;
let roomState = null;
let lastGameId = null;
let mounted = false;

function show(panelId) {
    for (const id of ['salon-entry', 'salon-login', 'salon-lobby', 'salon-game']) {
        document.getElementById(id).classList.toggle('hidden', id !== panelId);
    }
    if (panelId === 'salon-game') {
        document.getElementById('play-table').classList.remove('hidden');
    }
}

function toast(msg) {
    const el = document.getElementById('salon-toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.remove('hidden');
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.add('hidden'), 4000);
}

// ===== Lobby rendering =====

function renderLobby(data) {
    document.getElementById('salon-code').textContent = data.code;

    const seatsEl = document.getElementById('salon-seats');
    seatsEl.innerHTML = '';
    for (let i = 0; i < 4; i++) {
        const seat = data.seats[i];
        const div = document.createElement('div');
        div.className = 'salon-seat' + (i % 2 === 0 ? ' team-a' : ' team-b');
        const title = `<div class="salon-seat-title">${SEAT_TITLES[i]}</div>`;
        if (seat) {
            const you = data.you_seat === i;
            div.classList.add('taken');
            if (you) div.classList.add('you');
            div.innerHTML = title +
                `<div class="salon-seat-name">${seat.username}${you ? ' (vous)' : ''}</div>` +
                `<div class="salon-seat-sub">${seat.is_host ? 'hôte' : ''}${seat.connected ? '' : ' · déconnecté'}</div>`;
            if (you && data.status !== 'playing') {
                div.title = 'Cliquez pour libérer le siège';
                div.addEventListener('click', () => send({ type: 'room_stand' }));
            }
        } else {
            div.innerHTML = title +
                `<div class="salon-seat-name salon-seat-free">${BOT_LABELS[data.bot_type] || 'Bot'} 🤖</div>` +
                `<div class="salon-seat-sub">cliquer pour s'asseoir</div>`;
            div.addEventListener('click', () => send({ type: 'room_sit', seat: i }));
        }
        seatsEl.appendChild(div);
    }

    const hostControls = document.getElementById('salon-host-controls');
    hostControls.classList.toggle('hidden', !data.is_host);
    if (data.is_host) {
        document.getElementById('salon-bot-type').value = data.bot_type;
        document.getElementById('salon-move-delay').value = data.move_delay;
        document.getElementById('salon-move-delay-val').textContent = `${data.move_delay}s`;
    }

    const statusEl = document.getElementById('salon-lobby-status');
    if (data.status === 'playing') {
        statusEl.textContent = 'Partie en cours…';
    } else if (data.status === 'finished') {
        statusEl.textContent = 'Partie terminée — l\'hôte peut relancer.';
    } else {
        statusEl.textContent = data.is_host
            ? 'Lancez quand tout le monde est assis — les sièges vides seront des bots.'
            : `En attente du lancement par l'hôte…`;
    }
}

// ===== WS handlers =====

function handleRoomState(data) {
    roomState = data;
    renderLobby(data);
    // Only steer navigation between lobby and game panels; an in-progress
    // game keeps its table visible (room_game_state manages that panel).
    const gameVisible = !document.getElementById('salon-game').classList.contains('hidden');
    if (data.status === 'playing') {
        if (data.you_seat === null) show('salon-lobby'); // spectator member
        // seated: wait for room_game_state
    } else if (!gameVisible) {
        show('salon-lobby');
    }
}

function handleRoomNone() {
    if (!mounted) return;
    show('salon-entry');
}

function handleGameState(data) {
    if (data.game_id && data.game_id !== lastGameId) {
        lastGameId = data.game_id;
        table.reset();
        show('salon-game');
        table.show();
    } else if (document.getElementById('salon-game').classList.contains('hidden')) {
        show('salon-game');
        table.show();
    }
    if (data.seat_names) {
        table.setSeatLabels(data.seat_names.map(n => BOT_LABELS[n] || n));
    }
    table.handleGameState(data);
}

function handleMove(data) {
    table.handleMove(data);
}

function handleRoomError(data) {
    toast(data.msg);
}

function handleRoomLeft() {
    roomState = null;
    lastGameId = null;
    show('salon-entry');
}

function handleWsOpen() {
    // After a reconnect, ask the server where we stand (rebinds our socket)
    send({ type: 'room_status' });
}

// ===== Lifecycle =====

export async function mount(container) {
    container.innerHTML = TEMPLATE;
    mounted = true;

    table = new GameTable({
        sendMove: (action) => send({ type: 'room_play', action }),
        localEchoBids: false,
        resultButtons: [
            {
                label: 'Revanche', className: 'result-restart',
                onClick: () => send({ type: 'room_start' }),
            },
            {
                label: 'Salon', className: 'result-analyse',
                onClick: () => show('salon-lobby'),
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
    // No config panel in salon mode — hide the solo-only gear button
    document.getElementById('play-config-toggle').style.display = 'none';

    document.getElementById('salon-create').addEventListener('click', () => {
        send({ type: 'room_create' });
    });
    document.getElementById('salon-join').addEventListener('click', () => {
        const code = document.getElementById('salon-code-input').value.trim().toLowerCase();
        if (code) send({ type: 'room_join', code });
    });
    document.getElementById('salon-code-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') document.getElementById('salon-join').click();
    });
    document.getElementById('salon-goto-login').addEventListener('click', () => {
        navigateTo('/compte');
    });
    document.getElementById('salon-leave').addEventListener('click', () => {
        send({ type: 'room_leave' });
    });
    document.getElementById('salon-start').addEventListener('click', () => {
        send({ type: 'room_start' });
    });
    document.getElementById('salon-bot-type').addEventListener('change', (e) => {
        send({ type: 'room_config', bot_type: e.target.value });
    });
    document.getElementById('salon-move-delay').addEventListener('input', (e) => {
        document.getElementById('salon-move-delay-val').textContent = `${e.target.value}s`;
        send({ type: 'room_config', move_delay: parseFloat(e.target.value) });
    });

    onMessage('room_state', handleRoomState);
    onMessage('room_none', handleRoomNone);
    onMessage('room_game_state', handleGameState);
    onMessage('room_move', handleMove);
    onMessage('room_error', handleRoomError);
    onMessage('room_left', handleRoomLeft);
    onOpen(handleWsOpen);

    // Logged in? Then probe for an existing room membership.
    try {
        const resp = await fetch(`${base()}api/me`);
        const me = resp.ok ? await resp.json() : { user: null };
        if (!mounted) return;
        if (!me.user) {
            show('salon-login');
            return;
        }
    } catch { /* fall through: the probe below still works via ws */ }
    send({ type: 'room_status' });
}

export function unmount() {
    mounted = false;
    offMessage('room_state', handleRoomState);
    offMessage('room_none', handleRoomNone);
    offMessage('room_game_state', handleGameState);
    offMessage('room_move', handleMove);
    offMessage('room_error', handleRoomError);
    offMessage('room_left', handleRoomLeft);
    offOpen(handleWsOpen);
    if (table) {
        table.unbind();
        table = null;
    }
    roomState = null;
    lastGameId = null;
}
