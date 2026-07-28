// Play view (solo humain vs IA) — thin wrapper around the shared GameTable

import { send, onMessage, offMessage, onOpen, offOpen } from '../ws.js';
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
    <div id="play-resume" class="play-resume hidden"></div>
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

// Partie à reprendre demandée par l'URL (`?resume=<id>`), envoyée dès que la
// socket est ouverte. Le paramètre est retiré de l'URL aussitôt lu : reprendre
// abandonne la donne en cours, ce n'est pas une action qu'un F5 doit rejouer.
let resumeWanted = null;
// La liste des parties à reprendre appartient à l'écran de configuration, pas
// à la table : dès qu'une donne est affichée elle disparaît, y compris derrière
// le bouton ⚙ qui rouvre les réglages en cours de partie.
let gameOnScreen = false;

// ===== Parties en cours =====

function matchDate(iso) {
    const d = new Date(iso);
    if (Number.isNaN(d.getTime())) return '';
    return `${d.toLocaleDateString('fr-FR', { day: '2-digit', month: '2-digit' })}`
        + ` à ${d.toLocaleTimeString('fr-FR', { hour: '2-digit', minute: '2-digit' })}`;
}

function renderOpenMatches(matches) {
    const el = document.getElementById('play-resume');
    if (!el) return;
    if (gameOnScreen || !matches || matches.length === 0) {
        el.classList.add('hidden');
        el.innerHTML = '';
        return;
    }
    el.classList.remove('hidden');
    el.innerHTML = '<span class="config-group-label">Reprendre</span>'
        + matches.map(m => {
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
                + ' elle sera abandonnée</span>' : ''}</span>
        <button type="button" class="resume-go">Reprendre</button>
        <button type="button" class="resume-drop">Abandonner</button>
    </div>`;
        }).join('');

    for (const row of el.querySelectorAll('.resume-row')) {
        const id = row.dataset.id;
        row.querySelector('.resume-go').addEventListener('click', () => {
            send({ type: 'resume_match', match_id: id });
        });
        // Concéder est irréversible : le bouton demande confirmation sur place
        // plutôt que d'ouvrir une boîte de dialogue du navigateur.
        const drop = row.querySelector('.resume-drop');
        drop.addEventListener('click', () => {
            if (drop.dataset.armed) {
                send({ type: 'abandon_match', match_id: id });
                return;
            }
            drop.dataset.armed = '1';
            drop.textContent = 'Confirmer ?';
            drop.classList.add('resume-drop-armed');
        });
    }
}

function handleOpenMatches(data) {
    // La réponse prouve que la demande est passée : `play_status` répond
    // toujours par la liste, même quand il reprend une partie au passage.
    resumeWanted = null;
    renderOpenMatches(data.matches);
}

function probeStatus() {
    send({ type: 'play_status', resume: resumeWanted });
}

// ===== WS message handlers (stored for offMessage) =====

function handleGameState(data) {
    if (!gameOnScreen) {
        gameOnScreen = true;
        renderOpenMatches(null);
    }
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
        gameOnScreen = true;
        renderOpenMatches(null);
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
    onMessage('play_open', handleOpenMatches);
    onMessage('error', handleError);

    // Demander au serveur où on en est : une partie encore vivante sur cette
    // socket revient telle quelle à l'écran (aller-retour vers l'analyse), et
    // sinon on récupère la liste des parties à reprendre. `send()` est muet
    // quand la socket n'est pas encore ouverte (chargement à froid), d'où le
    // rappel sur `onOpen`.
    const params = new URLSearchParams(location.search);
    resumeWanted = params.get('resume');
    if (resumeWanted) {
        history.replaceState(null, '', location.pathname);
    }
    onOpen(probeStatus);
    probeStatus();
}

export function unmount() {
    offMessage('game_state', handleGameState);
    offMessage('ai_move', handleAiMove);
    offMessage('play_open', handleOpenMatches);
    offMessage('error', handleError);
    offOpen(probeStatus);
    resumeWanted = null;
    gameOnScreen = false;
    lastGameId = null;
    if (table) {
        table.unbind();
        table = null;
    }
}
