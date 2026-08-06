// Page Jouer — un seul écran pour toutes les façons de lancer une partie.
//
// Elle remplace `play.js` (solo) et `salon.js` (multijoueur) : c'étaient deux
// pages pour un même geste, avec les mêmes réglages écrits deux fois et deux
// vocabulaires pour la même chose.
//
// **Le nombre de bots n'est pas un choix, c'est une conséquence** — les sièges
// vides sont toujours tenus par l'IA, donc « contre trois bots », « à deux
// contre deux » et « quatre humains » ne sont pas trois modes à construire mais
// trois résultats du même. Le seul vrai réglage est *qui a le droit de
// s'asseoir* : personne (Seul), ceux à qui on envoie le lien (Entre amis).
//
// Deux pilotes vivent toujours derrière, et c'est délibéré : le solo est adossé
// à la base et se reprend au coup près, un salon vit en mémoire et meurt avec
// le processus. Les fusionner ferait perdre la reprise ou obligerait à
// persister les salons. Ce module est donc un aiguillage — un seul `GameTable`,
// deux protocoles (`play` / `room_play`).

import { send, onMessage, offMessage, onOpen, offOpen } from '../ws.js';
import { GameTable, TABLE_TEMPLATE, MY_SEAT } from '../shared/table.js';
import { navigateTo } from '../router.js';
import { botLabel } from '../shared/agents.js';

const base = () => document.querySelector('base')?.getAttribute('href') || '/';

// ===== Réglages =====

const WHO_KEY = 'colver_play_who';
const MODE_KEY = 'colver_play_mode';
const TARGET_KEY = 'colver_play_target';

// Qui peut s'asseoir. `shared` marque les tables partagées — celles qui ont un
// code et un lien. Elles portent la seule règle d'accès du site : **sans compte
// on y joue une donne, pas une partie**. Une partie dure une demi-heure, trois
// autres personnes en dépendent, et c'est elle qu'on classe. Le solo, lui,
// reste ouvert à tous quel que soit le format.
const WHO = {
    solo: {
        label: 'Seul', hint: 'contre l\'IA, départ immédiat',
        cta: 'Jouer', shared: false,
    },
    amis: {
        label: 'Entre amis', hint: 'un lien à envoyer',
        cta: 'Créer la table', shared: true,
    },
};

// Le mode choisit le tempo **et** le bot : un tempo rapide n'est honnête que
// derrière un bot qui répond tout de suite. La vérité est dans `pacing.py`.
const MODES = {
    standard: { label: 'Standard', bot: 'Dédé', hint: '≈ 40 s la donne' },
    rapide: { label: 'Rapide', bot: 'DouDou50', hint: '≈ 15 s la donne' },
};
// Doit rester aligné sur `match_state.TARGETS`, qui refuse le reste.
const TARGETS = [
    { value: 0, label: 'Une donne', hint: 'une main, un résultat' },
    { value: 1000, label: '1000 points', hint: '≈ 7 donnes' },
    { value: 2000, label: '2000 points', hint: '≈ 14 donnes' },
];
const SEAT_TITLES = ['Nord', 'Est', 'Sud', 'Ouest'];

/** Les deux réglages communs. `p` préfixe les ids : l'écran de configuration et
 *  le salon les montrent tous les deux, et deux ids identiques dans le document
 *  feraient répondre le mauvais bloc à `getElementById`. */
function configBlocks(p) {
    return `
    <div class="config-group">
        <span class="config-group-label">Rythme</span>
        <div id="${p}-mode-choice" class="mode-choice">
            ${Object.entries(MODES).map(([key, m]) => `
            <button type="button" class="mode-btn" data-mode="${key}">
                <span class="mode-btn-label">${m.label}</span>
                <span class="mode-btn-sub">${m.bot} · ${m.hint}</span>
            </button>`).join('')}
        </div>
    </div>
    <div class="config-group">
        <span class="config-group-label">Format</span>
        <div id="${p}-target-choice" class="mode-choice">
            ${TARGETS.map(t => `
            <button type="button" class="mode-btn" data-target="${t.value}">
                <span class="mode-btn-label">${t.label}</span>
                <span class="mode-btn-sub">${t.hint}</span>
            </button>`).join('')}
        </div>
    </div>`;
}

const TEMPLATE = `
<div id="play-config">
    <div id="play-resume" class="play-resume hidden"></div>
    <div class="config-group">
        <span class="config-group-label">Qui joue</span>
        <div id="who-choice" class="mode-choice">
            ${Object.entries(WHO).map(([key, w]) => `
            <button type="button" class="mode-btn" data-who="${key}">
                <span class="mode-btn-label">${w.label}</span>
                <span class="mode-btn-sub">${w.hint}</span>
            </button>`).join('')}
        </div>
    </div>
    ${configBlocks('cfg')}
    <p id="mode-note" class="mode-note hidden"></p>
    <div class="config-start">
        <button id="start-game">Jouer</button>
        <p id="who-note" class="mode-note hidden"></p>
    </div>
    <div id="join-code" class="join-code hidden">
        <span class="config-group-label">Un ami vous a envoyé un code&nbsp;?</span>
        <div class="salon-join-row">
            <input id="salon-code-input" class="compte-input" maxlength="4"
                   placeholder="code" autocomplete="off" spellcheck="false">
            <button id="salon-join" class="compte-submit">Rejoindre</button>
        </div>
    </div>
</div>

<div id="salon-login" class="salon-panel hidden">
    <h2 class="compte-title">Jouer avec d'autres joueurs</h2>
    <p class="salon-desc">Une table partagée réserve un siège à chacun&nbsp;: il
        faut un compte pour qu'on puisse vous y reconnaître — et vous y
        reconnecter si vous coupez.</p>
    <button id="salon-goto-login" class="compte-submit">Se connecter / créer un compte</button>
    <button id="salon-back" class="compte-logout">Retour</button>
</div>

<div id="salon-lobby" class="salon-panel hidden">
    <h2 class="compte-title">Table <span id="salon-code" class="salon-code"></span></h2>
    <div class="salon-invite">
        <button id="salon-copy" class="compte-submit">Copier le lien d'invitation</button>
        <p class="salon-desc">Envoyez-le à qui vous voulez&nbsp;: le lien assied
            directement à cette table.</p>
    </div>
    <div class="salon-seats" id="salon-seats"></div>
    <div id="salon-host-controls" class="salon-host-controls hidden">
        ${configBlocks('salon')}
        <p id="salon-mode-note" class="mode-note hidden"></p>
        <button id="salon-start" class="compte-submit">Lancer la partie</button>
    </div>
    <div id="salon-lobby-status" class="salon-lobby-status"></div>
    <button id="salon-leave" class="compte-logout">Quitter la table</button>
</div>
` + TABLE_TEMPLATE + `
<div id="salon-toast" class="salon-toast hidden"></div>`;

// ===== État du module =====

let table = null;
let currentWho = 'solo';
let currentMode = 'standard';
let currentTarget = 0;
let loggedIn = false;

// Quel protocole porte la partie à l'écran. C'est lui qui aiguille `sendMove`,
// l'écho local des annonces et les boutons de fin de donne — pas le réglage
// `currentWho`, qui peut avoir changé pendant qu'une partie tourne.
let inRoom = false;

// Solo
let lastGameId = null;
let resumeWanted = null;
let gameOnScreen = false;
// Salon
let roomWanted = null;      // code demandé par `?table=`
let lastRoomGameId = null;
let isHostSeat = false;
let awaitingNextDeal = false;
let mounted = false;

// ===== Panneaux =====

const PANELS = ['play-config', 'salon-login', 'salon-lobby'];

function showPanel(id) {
    for (const p of PANELS) {
        document.getElementById(p)?.classList.toggle('hidden', p !== id);
    }
    // La table est masquée par les panneaux de configuration, jamais l'inverse :
    // une partie en cours reste à l'écran tant qu'on ne la quitte pas.
    if (id !== null) document.getElementById('play-table').classList.add('hidden');
}

function showTable() {
    for (const p of PANELS) {
        // ⚠️ `#play-config` est masqué par le CSS, pas par nous : `board.css`
        // le retire dès qu'une table est visible et le rend au `.config-shown`
        // du bouton ⚙. Lui poser `.hidden` — qui est `!important` — rendrait ce
        // bouton inerte, sans rien afficher ni signaler.
        if (p !== 'play-config') document.getElementById(p)?.classList.add('hidden');
    }
    document.getElementById('play-config').classList.remove('hidden');
    document.getElementById('play-table').classList.remove('hidden');
    table.show();
}

function toast(msg) {
    const el = document.getElementById('salon-toast');
    if (!el) return;
    el.textContent = msg;
    el.classList.remove('hidden');
    clearTimeout(el._timer);
    el._timer = setTimeout(() => el.classList.add('hidden'), 4000);
}

// ===== Sélecteurs =====

function paint(containerId, attr, value) {
    for (const btn of document.querySelectorAll(`#${containerId} .mode-btn`)) {
        btn.classList.toggle('mode-btn-active', btn.dataset[attr] === String(value));
    }
}

/** Une partie longue à une table partagée demande un compte — pas une donne, et
 *  pas le solo. La règle vit ici, et le serveur la refait de son côté. */
function needsAccount() {
    return WHO[currentWho].shared && currentTarget !== 0 && !loggedIn;
}

function refreshWhoNote() {
    const note = document.getElementById('who-note');
    note.classList.toggle('hidden', !needsAccount());
    if (needsAccount()) {
        note.textContent = 'Sans compte, on peut jouer une donne à plusieurs — '
            + 'pas une partie en 1000 ou 2000 points.';
    }
}

function setWho(who) {
    currentWho = Object.hasOwn(WHO, who) ? who : 'solo';
    localStorage.setItem(WHO_KEY, currentWho);
    paint('who-choice', 'who', currentWho);
    const spec = WHO[currentWho];
    document.getElementById('start-game').textContent = spec.cta;
    // Le code d'invitation ne concerne que les tables partagées ; l'offrir en
    // solo poserait une question à laquelle la colonne choisie répond déjà.
    document.getElementById('join-code').classList.toggle('hidden', !spec.shared);
    refreshWhoNote();
}

function setMode(mode) {
    currentMode = Object.hasOwn(MODES, mode) ? mode : 'standard';
    localStorage.setItem(MODE_KEY, currentMode);
    paint('cfg-mode-choice', 'mode', currentMode);
}

function setTarget(target) {
    const value = Number(target);
    currentTarget = TARGETS.some(t => t.value === value) ? value : 0;
    localStorage.setItem(TARGET_KEY, String(currentTarget));
    paint('cfg-target-choice', 'target', currentTarget);
    refreshWhoNote();
}

// ===== Reprendre (solo) =====

function matchDate(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    return `${d.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' })}`
        + ` à ${d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}`;
}

// Une donne laissée en plan, hors partie. Elle se reprend au coup près, donc
// elle se présente comme les parties : le même bandeau, la même paire de
// boutons — sauf qu'ici « Abandonner » n'efface qu'une donne.
function lonePendingRow(deal) {
    const moves = deal.moves > 0
        ? `${deal.moves} coup${deal.moves > 1 ? 's' : ''} joué${deal.moves > 1 ? 's' : ''}`
        : 'à peine commencée';
    return `
    <div class="resume-row resume-deal" data-deal="${deal.game_id}">
        <span class="resume-score"><b>Donne</b></span>
        <span class="resume-meta">${moves} · ${matchDate(deal.created_at)}</span>
        <button type="button" class="resume-go">Reprendre</button>
        <button type="button" class="resume-drop">Abandonner</button>
    </div>`;
}

function renderOpenMatches(matches, deal) {
    const el = document.getElementById('play-resume');
    if (!el) return;
    const nothing = (!matches || matches.length === 0) && !deal;
    if (gameOnScreen || nothing) {
        el.classList.add('hidden');
        el.innerHTML = '';
        return;
    }
    el.classList.remove('hidden');
    el.innerHTML = '<span class="config-group-label">Reprendre</span>'
        + (deal ? lonePendingRow(deal) : '')
        + (matches || []).map(m => {
            // Le score est lu du côté du joueur, comme le bandeau de la table.
            const us = (m.human_seat ?? 2) % 2 === 0 ? m.points_ns : m.points_ew;
            const them = (m.human_seat ?? 2) % 2 === 0 ? m.points_ew : m.points_ns;
            const deals = `${m.deals} donne${m.deals > 1 ? 's' : ''}`;
            return `
    <div class="resume-row" data-id="${m.id}">
        <span class="resume-score">
            <b class="${us === them ? '' : (us > them ? 'resume-ahead' : 'resume-behind')}">${us}</b>
            <span class="resume-sep">–</span><b>${them}</b>
        </span>
        <span class="resume-meta">objectif ${m.target} · ${deals} · ${matchDate(m.created_at)}${
            m.pending ? '<br><span class="resume-warn">une donne était en cours :'
                + ' elle reprend où elle s\'est arrêtée</span>' : ''}</span>
        <button type="button" class="resume-go">Reprendre</button>
        <button type="button" class="resume-drop">Abandonner</button>
    </div>`;
        }).join('');

    for (const row of el.querySelectorAll('.resume-row')) {
        // Une ligne porte soit une partie (`data-id`), soit la donne isolée
        // (`data-deal`) : deux messages, un seul câblage.
        const id = row.dataset.id;
        row.querySelector('.resume-go').addEventListener('click', () => {
            send(id ? { type: 'resume_match', match_id: id }
                    : { type: 'resume_deal' });
        });
        // Abandonner est irréversible : le bouton demande confirmation sur place
        // plutôt que d'ouvrir une boîte de dialogue du navigateur.
        const drop = row.querySelector('.resume-drop');
        drop.addEventListener('click', () => {
            if (drop.dataset.armed) {
                send(id ? { type: 'abandon_match', match_id: id }
                        : { type: 'drop_deal' });
                return;
            }
            drop.dataset.armed = '1';
            drop.textContent = 'Confirmer ?';
            drop.classList.add('resume-drop-armed');
        });
    }
}

// ===== Salon =====

function inviteUrl(code) {
    return `${location.origin}${base()}jouer?table=${code}`;
}

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
                `<div class="salon-seat-name salon-seat-free">${botLabel(data.bot_type)} 🤖</div>` +
                `<div class="salon-seat-sub">cliquer pour s'asseoir</div>`;
            div.addEventListener('click', () => send({ type: 'room_sit', seat: i }));
        }
        seatsEl.appendChild(div);
    }

    const hostControls = document.getElementById('salon-host-controls');
    hostControls.classList.toggle('hidden', !data.is_host);
    if (data.is_host) {
        paint('salon-mode-choice', 'mode', data.mode);
        paint('salon-target-choice', 'target', data.target || 0);
        // Deux choses peuvent être à dire ici, et l'invité passe devant : il
        // explique un choix *refusé*, alors que le mode dégradé n'annonce
        // qu'une substitution.
        const note = document.getElementById('salon-mode-note');
        const msg = data.has_guest
            ? 'Un invité est à table : cette table ne peut jouer qu\'une donne. '
              + 'Une partie en 1000 ou 2000 points demande un compte à chacun.'
            : (data.mode_degraded
                ? 'DouDou50 est indisponible sur le serveur : Dédé prend sa '
                  + 'place, avec un budget de réflexion réduit.'
                : '');
        note.classList.toggle('hidden', !msg);
        note.textContent = msg;
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
            ? 'Lancez quand vous voulez — les sièges vides seront tenus par l\'IA.'
            : 'En attente du lancement par l\'hôte…';
    }
}

// ===== Messages WS — solo =====

function handleGameState(data) {
    inRoom = false;
    table.localEchoBids = true;
    document.getElementById('play-config-toggle').style.display = '';
    if (!gameOnScreen) {
        gameOnScreen = true;
        renderOpenMatches(null);
    }
    if (data.game_id && data.game_id !== lastGameId) {
        // Donne suivante d'une partie : même table, tout est à refaire.
        lastGameId = data.game_id;
        table.reset();
        showTable();
    } else {
        showTable();
    }
    table.handleGameState(data);
    // Le serveur tranche le mode : le dire quand il n'a pas pu asseoir le bot
    // annoncé, plutôt que de faire jouer un autre adversaire en silence.
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
    if (!inRoom) table.handleMove(data);
}

function handleOpenMatches(data) {
    // La réponse prouve que la demande est passée : `play_status` répond
    // toujours par la liste, même quand il reprend une partie au passage.
    resumeWanted = null;
    renderOpenMatches(data.matches, data.deal);
}

function handleError(data) {
    console.error('Erreur serveur:', data.msg);
    const statusEl = document.getElementById('play-status');
    if (statusEl) statusEl.textContent = `Erreur : ${data.msg}`;
}

// ===== Messages WS — salon =====

function handleRoomState(data) {
    roomWanted = null;
    renderLobby(data);
    // On ne pilote la navigation qu'entre configuration et salon : une partie
    // en cours garde sa table (c'est `room_game_state` qui tient ce panneau).
    const gameVisible = !document.getElementById('play-table').classList.contains('hidden');
    if (data.status === 'playing') {
        if (data.you_seat === null) showPanel('salon-lobby');  // membre non assis
    } else if (!gameVisible || !inRoom) {
        showPanel('salon-lobby');
    }
}

function handleRoomNone() {
    if (!mounted) return;
    // Aucun salon : on reste sur la configuration. Ce message arrive à chaque
    // arrivée sur la page, il ne doit donc chasser aucune partie solo.
    if (!inRoom && !gameOnScreen) showPanel('play-config');
}

function handleRoomGameState(data) {
    inRoom = true;
    table.localEchoBids = false;
    // Pas de panneau de réglages pendant une partie de salon : ils appartiennent
    // à l'hôte et vivent dans le salon.
    document.getElementById('play-config-toggle').style.display = 'none';
    isHostSeat = !!data.is_host;
    awaitingNextDeal = !!data.awaiting_next_deal;
    if (data.game_id && data.game_id !== lastRoomGameId) {
        lastRoomGameId = data.game_id;
        table.reset();
    }
    showTable();
    // Entrées `{name, bot}` : c'est la table qui décide comment rendre un bot
    // (par sa position) et un humain (par son pseudo).
    if (data.seat_names) table.setSeatLabels(data.seat_names);
    table.handleGameState(data);
}

function handleRoomMove(data) {
    if (inRoom) table.handleMove(data);
}

function handleRoomError(data) {
    toast(data.msg);
}

function handleRoomLeft() {
    inRoom = false;
    lastRoomGameId = null;
    awaitingNextDeal = false;
    document.getElementById('play-config-toggle').style.display = '';
    showPanel('play-config');
}

// ===== Sondes à l'ouverture =====

function probe() {
    send({ type: 'play_status', resume: resumeWanted });
    // Tout le monde a une identité de salon depuis les invités — compte ou
    // jeton posé à la poignée de main — donc `room_status` aboutit toujours et
    // rend `room_none` quand il n'y a rien à retrouver.
    if (roomWanted) send({ type: 'room_join', code: roomWanted });
    else send({ type: 'room_status' });
}

// ===== Cycle de vie =====

export async function mount(container) {
    container.innerHTML = TEMPLATE;
    mounted = true;

    const analyseButton = {
        label: 'Analyser', className: 'result-analyse',
        onClick: (gameId) => {
            if (!gameId) return;
            navigateTo('/analyse/rejouer');
            import('../views/replay.js').then(m => m.loadReplayById(gameId));
        },
    };
    const lobbyButton = {
        label: 'Table', className: 'result-analyse',
        onClick: () => showPanel('salon-lobby'),
    };

    table = new GameTable({
        // Un seul plateau, deux protocoles : c'est `inRoom` qui tranche, et il
        // suit la partie à l'écran — pas le réglage, qui peut avoir été changé
        // pendant qu'une partie tourne.
        sendMove: (action) => send(inRoom
            ? { type: 'room_play', action }
            : { type: 'play', action, human_seat: MY_SEAT }),
        localEchoBids: true,
        resultButtons: ({ match }) => {
            if (inRoom) {
                // Entre deux donnes d'une partie, seul l'hôte enchaîne — les
                // autres voient pourquoi rien ne bouge plutôt qu'un bouton qui
                // échouerait.
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
                    lobbyButton,
                    analyseButton,
                ];
            }
            // Solo : tant que la partie n'est pas jouée, le bouton principal
            // enchaîne la donne suivante ; sinon il en relance une.
            return [
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
            ];
        },
    });
    table.bind();

    // ---- réglages ----
    for (const btn of document.querySelectorAll('#who-choice .mode-btn')) {
        btn.addEventListener('click', () => setWho(btn.dataset.who));
    }
    for (const btn of document.querySelectorAll('#cfg-mode-choice .mode-btn')) {
        btn.addEventListener('click', () => setMode(btn.dataset.mode));
    }
    for (const btn of document.querySelectorAll('#cfg-target-choice .mode-btn')) {
        btn.addEventListener('click', () => setTarget(btn.dataset.target));
    }
    setWho(localStorage.getItem(WHO_KEY));
    setMode(localStorage.getItem(MODE_KEY));
    setTarget(localStorage.getItem(TARGET_KEY));

    // Dans le salon, les mêmes réglages appartiennent à l'hôte et passent par
    // le serveur : c'est lui qui les rediffuse à toute la table.
    for (const btn of document.querySelectorAll('#salon-mode-choice .mode-btn')) {
        btn.addEventListener('click', () =>
            send({ type: 'room_config', mode: btn.dataset.mode }));
    }
    for (const btn of document.querySelectorAll('#salon-target-choice .mode-btn')) {
        btn.addEventListener('click', () =>
            send({ type: 'room_config', target: Number(btn.dataset.target) }));
    }

    document.getElementById('play-config-toggle').addEventListener('click', () => {
        document.getElementById('play-config').classList.toggle('config-shown');
    });

    // ---- lancer ----
    document.getElementById('start-game').addEventListener('click', () => {
        if (WHO[currentWho].shared) {
            if (needsAccount()) { showPanel('salon-login'); return; }
            // Le salon naît avec les réglages de l'écran : ils sont posés juste
            // après la création, le serveur les rediffusant à la table.
            send({ type: 'room_create' });
            send({ type: 'room_config', mode: currentMode, target: currentTarget });
            return;
        }
        table.reset();
        inRoom = false;
        gameOnScreen = true;
        renderOpenMatches(null);
        lastGameId = null;
        send({
            type: 'start_game', mode: currentMode, target: currentTarget,
            human_seat: MY_SEAT,
        });
        showTable();
        document.getElementById('play-status').textContent = 'Lancement de la partie...';
    });

    // ---- salon ----
    document.getElementById('salon-join').addEventListener('click', () => {
        // Pas de garde de compte ici : une table peut très bien jouer une seule
        // donne, auquel cas un invité y a sa place. C'est le serveur qui sait
        // ce que joue *cette* table, et qui le dit s'il refuse.
        const code = document.getElementById('salon-code-input').value.trim().toLowerCase();
        if (code) send({ type: 'room_join', code });
    });
    document.getElementById('salon-code-input').addEventListener('keydown', (e) => {
        if (e.key === 'Enter') document.getElementById('salon-join').click();
    });
    document.getElementById('salon-goto-login').addEventListener('click', () => {
        navigateTo('/compte');
    });
    document.getElementById('salon-back').addEventListener('click', () => {
        showPanel('play-config');
    });
    document.getElementById('salon-leave').addEventListener('click', () => {
        send({ type: 'room_leave' });
    });
    document.getElementById('salon-start').addEventListener('click', () => {
        send({ type: 'room_start' });
    });
    document.getElementById('salon-copy').addEventListener('click', async () => {
        const url = inviteUrl(document.getElementById('salon-code').textContent);
        try {
            // `navigator.clipboard` n'existe qu'en contexte sécurisé et peut
            // être refusé : on montre le lien plutôt que d'échouer en silence.
            await navigator.clipboard.writeText(url);
            toast('Lien copié');
        } catch {
            toast(url);
        }
    });

    onMessage('game_state', handleGameState);
    onMessage('ai_move', handleAiMove);
    onMessage('play_open', handleOpenMatches);
    onMessage('error', handleError);
    onMessage('room_state', handleRoomState);
    onMessage('room_none', handleRoomNone);
    onMessage('room_game_state', handleRoomGameState);
    onMessage('room_move', handleRoomMove);
    onMessage('room_error', handleRoomError);
    onMessage('room_left', handleRoomLeft);

    // Les deux paramètres d'URL sont retirés dès qu'ils sont lus : reprendre ou
    // rejoindre une table est une action, pas un état de page, et un F5 ne doit
    // pas la rejouer.
    const params = new URLSearchParams(location.search);
    resumeWanted = params.get('resume');
    roomWanted = (params.get('table') || '').trim().toLowerCase() || null;
    if (resumeWanted || roomWanted) {
        history.replaceState(null, '', location.pathname);
    }

    // Connecté ? Le savoir avant de sonder : un anonyme n'a pas de salon, et le
    // lui demander lui renvoie une erreur qui ne le concerne pas.
    try {
        const resp = await fetch(`${base()}api/me`);
        const me = resp.ok ? await resp.json() : { user: null };
        if (!mounted) return;
        loggedIn = !!me.user;
    } catch { /* on continue : la sonde solo marche sans compte */ }
    setWho(currentWho);   // l'avertissement « compte nécessaire » en dépend
    onOpen(probe);
    probe();
}

export function unmount() {
    mounted = false;
    offMessage('game_state', handleGameState);
    offMessage('ai_move', handleAiMove);
    offMessage('play_open', handleOpenMatches);
    offMessage('error', handleError);
    offMessage('room_state', handleRoomState);
    offMessage('room_none', handleRoomNone);
    offMessage('room_game_state', handleRoomGameState);
    offMessage('room_move', handleRoomMove);
    offMessage('room_error', handleRoomError);
    offMessage('room_left', handleRoomLeft);
    offOpen(probe);
    resumeWanted = null;
    roomWanted = null;
    gameOnScreen = false;
    inRoom = false;
    lastGameId = null;
    lastRoomGameId = null;
    isHostSeat = false;
    awaitingNextDeal = false;
    if (table) {
        table.unbind();
        table = null;
    }
}
