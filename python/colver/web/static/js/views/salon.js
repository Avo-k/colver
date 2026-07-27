// Salon view — multiplayer rooms: create/join, lobby, then shared GameTable

import { send, onMessage, offMessage, onOpen, offOpen } from '../ws.js';
import { GameTable, TABLE_TEMPLATE } from '../shared/table.js';
import { navigateTo } from '../router.js';

const base = () => document.querySelector('base')?.getAttribute('href') || '/';

const BOT_LABELS = { dede: 'Dédé', doudou: 'DouDou', oracle_dd: 'Oracle' };
// Same bundles as the solo view and pacing.py: one host choice sets both the
// tempo and which bot fills the empty seats.
const MODES = {
    standard: { label: 'Standard', bot: 'Dédé', hint: '≈ 40 s la donne' },
    rapide: { label: 'Rapide', bot: 'DouDou50', hint: '≈ 15 s la donne' },
};
// Mêmes formats qu'en solo (cf. match_state.TARGETS côté serveur).
const TARGETS = [
    { value: 0, label: 'Une donne', hint: 'une main, un résultat' },
    { value: 1000, label: '1000 points', hint: '≈ 7 donnes' },
    { value: 2000, label: '2000 points', hint: '≈ 14 donnes' },
];
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
            <div class="config-group">
                <span class="config-group-label">Rythme</span>
                <div id="salon-mode-choice" class="mode-choice">
                    ${Object.entries(MODES).map(([key, m]) => `
                    <button type="button" class="mode-btn" data-mode="${key}">
                        <span class="mode-btn-label">${m.label}</span>
                        <span class="mode-btn-sub">${m.bot} · ${m.hint}</span>
                    </button>`).join('')}
                </div>
            </div>
            <div class="config-group">
                <span class="config-group-label">Format</span>
                <div id="salon-target-choice" class="mode-choice">
                    ${TARGETS.map(t => `
                    <button type="button" class="mode-btn" data-target="${t.value}">
                        <span class="mode-btn-label">${t.label}</span>
                        <span class="mode-btn-sub">${t.hint}</span>
                    </button>`).join('')}
                </div>
            </div>
            <p id="salon-mode-note" class="mode-note hidden"></p>
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
let isHostSeat = false;        // ce client tient-il le siège de l'hôte
let awaitingNextDeal = false;  // donne finie, partie en cours

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
        for (const btn of document.querySelectorAll('#salon-mode-choice .mode-btn')) {
            btn.classList.toggle('mode-btn-active', btn.dataset.mode === data.mode);
        }
        for (const btn of document.querySelectorAll('#salon-target-choice .mode-btn')) {
            btn.classList.toggle('mode-btn-active',
                Number(btn.dataset.target) === (data.target || 0));
        }
        const note = document.getElementById('salon-mode-note');
        note.classList.toggle('hidden', !data.mode_degraded);
        if (data.mode_degraded) {
            note.textContent = 'DouDou50 est indisponible sur le serveur : '
                + 'Dédé prend sa place, avec un budget de réflexion réduit.';
        }
    }

    const statusEl = document.getElementById('salon-lobby-status');
    if (data.status === 'playing' && data.awaiting_next_deal) {
        statusEl.textContent = 'Donne terminée — l\'hôte lance la suivante.';
    } else if (data.status === 'playing') {
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
    // Qui peut enchaîner la donne suivante : lu ici plutôt que dans le lobby,
    // parce que c'est ce message qui accompagne le panneau de fin de donne.
    isHostSeat = !!data.is_host;
    awaitingNextDeal = !!data.awaiting_next_deal;
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
    awaitingNextDeal = false;
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

    const salonButton = {
        label: 'Salon', className: 'result-analyse',
        onClick: () => show('salon-lobby'),
    };
    const analyseButton = {
        label: 'Analyser', className: 'result-analyse',
        onClick: (gameId) => {
            if (!gameId) return;
            navigateTo('/analyse/rejouer');
            import('../views/replay.js').then(m => m.loadReplayById(gameId));
        },
    };

    table = new GameTable({
        sendMove: (action) => send({ type: 'room_play', action }),
        localEchoBids: false,
        // Entre deux donnes d'une partie, seul l'hôte enchaîne — les autres
        // voient pourquoi rien ne bouge plutôt qu'un bouton qui échouerait.
        resultButtons: () => {
            if (awaitingNextDeal) {
                return [
                    isHostSeat
                        ? {
                            label: 'Donne suivante', className: 'result-restart',
                            onClick: () => send({ type: 'room_next_deal' }),
                        }
                        : {
                            label: 'En attente de l\'hôte…',
                            className: 'result-restart', disabled: true,
                        },
                    analyseButton,
                ];
            }
            return [
                {
                    label: 'Revanche', className: 'result-restart',
                    onClick: () => send({ type: 'room_start' }),
                },
                salonButton,
                analyseButton,
            ];
        },
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
    for (const btn of document.querySelectorAll('#salon-mode-choice .mode-btn')) {
        btn.addEventListener('click', () => {
            send({ type: 'room_config', mode: btn.dataset.mode });
        });
    }
    for (const btn of document.querySelectorAll('#salon-target-choice .mode-btn')) {
        btn.addEventListener('click', () => {
            send({ type: 'room_config', target: Number(btn.dataset.target) });
        });
    }

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
    isHostSeat = false;
    awaitingNextDeal = false;
}
