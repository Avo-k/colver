// Croyances view — belief network visualization
// Step through a game and see how NN/heuristic beliefs evolve

import { send, onMessage, offMessage, onOpen, offOpen } from '../ws.js';
import { RANKS, SUITS, cardSvgPath, cardRank, cardSuit, SEAT_NAMES_FR, SUIT_DISPLAY_ORDER } from '../shared/cards.js';

const SUIT_SYMBOLS = ['\u2660', '\u2665', '\u2666', '\u2663'];
const SEAT_NAMES = ['N', 'E', 'S', 'O'];
const SEAT_COLORS = ['#5b9ff5', '#e06080', '#d4af37', '#8c6fe6'];

// Display order within a suit: A 10 R D V 9 8 7
const DISPLAY_ORDER = [7, 6, 5, 4, 3, 2, 1, 0];

const SUIT_EMOJI = ['♠️', '♥️', '♦️', '♣️'];
function suitHtml(suitIdx) {
    return SUIT_EMOJI[suitIdx];
}

function actionHtml(action, phase) {
    if (phase === 0) {
        if (action === 0) return 'Passe';
        if (action === 41) return 'Coinche';
        if (action === 42) return 'Surcoinche';
        if (action >= 37 && action <= 40) return `Capot ${suitHtml(action - 37)}`;
        if (action >= 1 && action <= 36) {
            const bidIdx = action - 1;
            const valueIdx = Math.floor(bidIdx / 4);
            const suitIdx = bidIdx % 4;
            const value = 80 + valueIdx * 10;
            return `${value}${suitHtml(suitIdx)}`;
        }
        return `?${action}`;
    }
    const rank = cardRank(action);
    const suit = cardSuit(action);
    return `${RANKS[rank]}${suitHtml(suit)}`;
}

// Module state
let totalActions = 0;
let numBidActions = 0;
let allActions = null; // [{player, action, phase}] — full game history
let gameCfn = null; // full-game CFN (auction + play), for copy/share
let precompute = null; // {observer, done, total} — playgen cache warming progress
let currentIdx = 0;
let currentState = null;
let currentLastAction = null;
let initialHands = null;
let dealer = 0;
let observer = 0;
let viewMode = 'nn'; // 'nn', 'playgen', 'heuristic', 'compare'
let nnWeights = null;
let heuristicWeights = null;
let playgenWeights = null; // null pendant les enchères (contrat inconnu)
let groundTruth = null;
let loading = false;

const TEMPLATE = `
<div id="belief-config">
    <button id="belief-generate-btn" class="primary-btn">Nouvelle partie aléatoire</button>
    <button id="belief-import-btn" class="belief-nav-btn" title="Coller le CFN d'une partie">Importer une partie custom</button>
    <button id="belief-copy-btn" class="belief-nav-btn" title="Copier le CFN de cette partie" style="display:none">Copier le CFN</button>
    <span id="belief-loading"></span>
</div>
<div id="belief-main" class="hidden">
    <div id="belief-topbar">
        <div id="belief-nav">
            <button id="belief-start-btn" class="belief-nav-btn" title="Debut">|&#9665;</button>
            <button id="belief-prev-btn" class="belief-nav-btn" title="Precedent">&#9665;</button>
            <input id="belief-slider" type="range" min="0" max="0" value="0">
            <button id="belief-next-btn" class="belief-nav-btn" title="Suivant">&#9655;</button>
            <button id="belief-end-btn" class="belief-nav-btn" title="Fin">&#9655;|</button>
        </div>
        <div id="belief-info">
            <span id="belief-action-info">Action 0/0</span>
            <span id="belief-contract-info"></span>
            <span id="belief-last-action"></span>
        </div>
        <div id="belief-controls">
            <div id="belief-observer-btns">
                <span class="belief-label">Obs:</span>
                <button class="belief-obs-btn active" data-obs="0">N</button>
                <button class="belief-obs-btn" data-obs="1">E</button>
                <button class="belief-obs-btn" data-obs="2">S</button>
                <button class="belief-obs-btn" data-obs="3">O</button>
            </div>
            <div id="belief-mode-btns">
                <button class="belief-mode-btn active" data-mode="nn">NN</button>
                <button class="belief-mode-btn" data-mode="playgen">Playgen</button>
                <button class="belief-mode-btn" data-mode="heuristic">Heuristique</button>
                <button class="belief-mode-btn" data-mode="compare">Comparer</button>
            </div>
            <span id="belief-precompute"></span>
        </div>
    </div>
    <div id="belief-history"></div>
    <div id="belief-grid"></div>
    <div id="belief-stats"></div>
</div>
`;

function setLoading(msg) {
    const el = document.getElementById('belief-loading');
    if (el) el.textContent = msg || '';
    loading = !!msg;
    const btn = document.getElementById('belief-generate-btn');
    if (btn) btn.disabled = loading;
}

function requestWeights() {
    // Le calcul playgen (~2s de MC) n'est demandé que si le mode l'affiche
    const withPlaygen = viewMode === 'playgen' || viewMode === 'compare';
    send({ type: 'belief_get_weights', observer, playgen: withPlaygen });
}

function onGenerated(data) {
    setLoading('');
    initialHands = data.initial_hands;
    dealer = data.dealer;
    totalActions = data.total_actions;
    numBidActions = data.num_bid_actions;
    allActions = data.actions || null;
    gameCfn = data.game_cfn || null;
    const copyBtn = document.getElementById('belief-copy-btn');
    if (copyBtn) copyBtn.style.display = gameCfn ? '' : 'none';
    precompute = null;
    renderHistory();
    renderPrecompute();
    currentIdx = 0;
    currentState = null;
    currentLastAction = null;
    nnWeights = null;
    heuristicWeights = null;
    playgenWeights = null;
    groundTruth = null;

    const slider = document.getElementById('belief-slider');
    if (slider) {
        slider.max = totalActions;
        slider.value = 0;
    }

    document.getElementById('belief-main').classList.remove('hidden');
    updateInfo();
    // Request initial state (which triggers weight request via onState)
    stepTo(0);
}

function onState(data) {
    currentIdx = data.action_idx;
    currentState = data.state;
    currentLastAction = data.last_action;

    const slider = document.getElementById('belief-slider');
    if (slider) slider.value = currentIdx;

    updateInfo();
    updateHistoryPosition();
    requestWeights();
}

function onPrecompute(data) {
    precompute = data;
    renderPrecompute();
}

function renderPrecompute() {
    const el = document.getElementById('belief-precompute');
    if (!el) return;
    if (!precompute || precompute.observer !== observer || precompute.total === 0) {
        el.textContent = '';
        return;
    }
    if (precompute.done >= precompute.total) {
        el.innerHTML = `Playgen <span class="belief-precompute-done">✓</span>`;
    } else {
        el.textContent = `Playgen ⏳ ${precompute.done}/${precompute.total}`;
    }
}

function onWeights(data) {
    if (data.observer !== observer) return;
    nnWeights = data.nn;
    heuristicWeights = data.heuristic;
    playgenWeights = data.playgen || null;
    groundTruth = data.ground_truth;
    renderGrid();
    renderStats();
}

function updateInfo() {
    const actionInfo = document.getElementById('belief-action-info');
    if (actionInfo) {
        let label = `Action ${currentIdx}/${totalActions}`;
        if (currentState) {
            if (currentState.phase === 0) {
                label += ', Encheres';
            } else {
                const tricksPlayed = currentState.tricks_won[0] + currentState.tricks_won[1];
                label += `, Pli ${tricksPlayed + 1}`;
            }
        }
        actionInfo.textContent = label;
    }

    const contractInfo = document.getElementById('belief-contract-info');
    if (contractInfo && currentState && currentState.contract && currentState.contract.value) {
        const c = currentState.contract;
        const team = c.team === 0 ? 'NS' : 'EO';
        const coinche = c.coinche === 1 ? ' x' : c.coinche === 2 ? ' xx' : '';
        contractInfo.innerHTML = `Contrat: ${c.value}${suitHtml(c.trump)} ${team}${coinche}`;
    } else if (contractInfo) {
        contractInfo.textContent = '';
    }

    const lastAction = document.getElementById('belief-last-action');
    if (lastAction && currentLastAction) {
        lastAction.innerHTML = `${SEAT_NAMES[currentLastAction.player]} joue ${actionHtml(currentLastAction.action, currentLastAction.phase)}`;
    } else if (lastAction) {
        lastAction.textContent = '';
    }
}

function renderHistory() {
    const el = document.getElementById('belief-history');
    if (!el) return;
    el.innerHTML = '';
    if (!allActions) return;

    // Position 0 (before any action)
    const start = document.createElement('button');
    start.className = 'belief-hist-item belief-hist-start';
    start.textContent = 'Début';
    start.dataset.idx = '0';
    start.title = 'Avant la première annonce';
    el.appendChild(start);

    let trickDiv = null;
    for (let i = 0; i < allActions.length; i++) {
        const { player, action, phase } = allActions[i];
        const item = document.createElement('button');
        item.className = 'belief-hist-item';
        item.dataset.idx = String(i + 1); // position AFTER this action
        item.style.setProperty('--seat-color', SEAT_COLORS[player]);
        item.title = `${SEAT_NAMES[player]} : ${actionHtml(action, phase).replace(/<[^>]*>/g, '')}`;

        if (phase === 0) {
            item.classList.add('belief-hist-bid');
            item.innerHTML = `<span class="belief-hist-seat">${SEAT_NAMES[player]}</span>${actionHtml(action, phase)}`;
            el.appendChild(item);
        } else {
            // Group play actions 4 by 4 (tricks)
            const playIdx = i - numBidActions;
            if (playIdx % 4 === 0) {
                trickDiv = document.createElement('span');
                trickDiv.className = 'belief-hist-trick';
                el.appendChild(trickDiv);
            }
            item.classList.add('belief-hist-card');
            item.innerHTML = `<span class="belief-hist-seat">${SEAT_NAMES[player]}</span>${actionHtml(action, phase)}`;
            trickDiv.appendChild(item);
        }
    }
    updateHistoryPosition();
}

function updateHistoryPosition() {
    const el = document.getElementById('belief-history');
    if (!el) return;
    el.querySelectorAll('.belief-hist-item').forEach(item => {
        const idx = parseInt(item.dataset.idx);
        item.classList.toggle('belief-hist-done', idx <= currentIdx);
        item.classList.toggle('belief-hist-current', idx === currentIdx);
    });
    // Keep the current item in view
    const cur = el.querySelector('.belief-hist-current');
    if (cur) cur.scrollIntoView({ block: 'nearest', inline: 'nearest', behavior: 'smooth' });
}

function onHistoryClick(e) {
    const item = e.target.closest('.belief-hist-item');
    if (!item) return;
    stepTo(parseInt(item.dataset.idx));
}

function renderGrid() {
    const grid = document.getElementById('belief-grid');
    if (!grid) return;
    grid.innerHTML = '';

    if (!groundTruth) return;

    // Sources to display: single bar in nn/playgen/heuristic modes, stacked in compare
    const sources = [];
    if (viewMode === 'nn') sources.push({ w: nnWeights, label: null });
    else if (viewMode === 'playgen') sources.push({ w: playgenWeights, label: null });
    else if (viewMode === 'heuristic') sources.push({ w: heuristicWeights, label: null });
    else {
        if (nnWeights) sources.push({ w: nnWeights, label: 'NN' });
        if (playgenWeights) sources.push({ w: playgenWeights, label: 'PG' });
        if (heuristicWeights) sources.push({ w: heuristicWeights, label: 'Heur.' });
    }
    const activeSources = sources.filter(s => s.w);

    // Determine known/played cards for observer
    const observerHand = new Set(currentState ? currentState.hands[observer] : (initialHands ? initialHands[observer] : []));
    const playedCards = new Set();
    if (currentState && currentState.current_trick) {
        for (const c of currentState.current_trick) {
            if (c >= 0 && c < 32) playedCards.add(c);
        }
    }
    // Cards in hands that are smaller than initial = played
    if (initialHands && currentState && currentState.hands) {
        const allCurrent = new Set();
        for (const h of currentState.hands) {
            for (const c of h) allCurrent.add(c);
        }
        for (const h of initialHands) {
            for (const c of h) {
                if (!allCurrent.has(c)) playedCards.add(c);
            }
        }
    }

    // Relative player names from observer's perspective
    const relNames = ['Moi', 'Gauche', 'Partenaire', 'Droite'];
    const absFromRel = (rel) => (observer + rel) % 4;

    for (const suit of SUIT_DISPLAY_ORDER) {
        const suitDiv = document.createElement('div');
        suitDiv.className = 'belief-suit';

        const suitLabel = document.createElement('div');
        suitLabel.className = 'belief-suit-label';
        const cls = (suit === 1 || suit === 2) ? 'suit-red' : 'suit-black';
        suitLabel.innerHTML = `<span class="${cls}">${SUIT_SYMBOLS[suit]}</span>`;
        suitDiv.appendChild(suitLabel);

        const cardsRow = document.createElement('div');
        cardsRow.className = 'belief-cards-row';

        for (const rankIdx of DISPLAY_ORDER) {
            const cardIdx = suit * 8 + rankIdx;
            const cardDiv = document.createElement('div');
            cardDiv.className = 'belief-card-col';

            // Card image
            const img = document.createElement('img');
            img.src = cardSvgPath(cardIdx);
            img.alt = `${RANKS[rankIdx]}${SUITS[suit]}`;
            img.className = 'belief-card-img';
            img.draggable = false;

            const isInHand = observerHand.has(cardIdx);
            const isPlayed = playedCards.has(cardIdx);

            if (isInHand || isPlayed) {
                cardDiv.classList.add(isInHand ? 'belief-known' : 'belief-played');
                cardDiv.appendChild(img);
                const tag = document.createElement('div');
                tag.className = 'belief-tag';
                tag.textContent = isInHand ? 'Moi' : 'Joue';
                cardDiv.appendChild(tag);
            } else {
                // Unknown card — show belief bars
                cardDiv.appendChild(img);

                for (const src of activeSources) {
                    cardDiv.appendChild(createBeliefBar(src.w, cardIdx, observer, groundTruth, src.label));
                }
                if (activeSources.length === 0) {
                    const noData = document.createElement('div');
                    noData.className = 'belief-no-data';
                    noData.textContent = '-';
                    cardDiv.appendChild(noData);
                }
            }

            cardsRow.appendChild(cardDiv);
        }

        suitDiv.appendChild(cardsRow);
        grid.appendChild(suitDiv);
    }
}

function createBeliefBar(weights, cardIdx, observer, groundTruth, label) {
    const bar = document.createElement('div');
    bar.className = 'belief-bar-container';

    if (label) {
        const labelEl = document.createElement('div');
        labelEl.className = 'belief-bar-label';
        labelEl.textContent = label;
        bar.appendChild(labelEl);
    }

    const barInner = document.createElement('div');
    barInner.className = 'belief-bar';

    // Get probabilities for the 3 other players (relative order)
    const segments = [];
    for (let rel = 1; rel <= 3; rel++) {
        const abs = (observer + rel) % 4;
        const prob = weights[abs][cardIdx];
        segments.push({ rel, abs, prob });
    }

    // Find ground truth holder
    let truthHolder = -1;
    if (groundTruth) {
        for (let p = 0; p < 4; p++) {
            if (p === observer) continue;
            if (groundTruth[p].includes(cardIdx)) {
                truthHolder = p;
                break;
            }
        }
    }

    for (const seg of segments) {
        if (seg.prob <= 0.001) continue;
        const pct = Math.max(seg.prob * 100, 1);
        const segEl = document.createElement('div');
        segEl.className = 'belief-bar-seg';
        segEl.style.width = pct + '%';
        segEl.style.backgroundColor = SEAT_COLORS[seg.abs];
        segEl.title = `${SEAT_NAMES[seg.abs]}: ${(seg.prob * 100).toFixed(1)}%`;
        barInner.appendChild(segEl);
    }

    bar.appendChild(barInner);

    // Ground truth marker
    if (truthHolder >= 0) {
        const marker = document.createElement('div');
        marker.className = 'belief-truth-marker';
        marker.style.color = SEAT_COLORS[truthHolder];
        marker.title = `Vraie position: ${SEAT_NAMES[truthHolder]}`;
        marker.textContent = SEAT_NAMES[truthHolder];
        bar.appendChild(marker);
    }

    return bar;
}

function renderStats() {
    const statsEl = document.getElementById('belief-stats');
    if (!statsEl) return;
    statsEl.innerHTML = '';

    const modes = [];
    if (viewMode === 'nn' || viewMode === 'compare') modes.push({ w: nnWeights, label: 'NN' });
    if (viewMode === 'playgen' || viewMode === 'compare') modes.push({ w: playgenWeights, label: 'Playgen' });
    if (viewMode === 'heuristic' || viewMode === 'compare') modes.push({ w: heuristicWeights, label: 'Heuristique' });

    // Le playgen v2 échantillonne aussi pendant l'enchère ; s'il manque, c'est
    // qu'il n'est pas encore calculé (ou indisponible) pour cette position.
    if ((viewMode === 'playgen' || viewMode === 'compare') && !playgenWeights
        && currentState && !currentState.is_terminal) {
        const note = document.createElement('div');
        note.className = 'belief-stat-group';
        note.innerHTML = `<span class="belief-stat-label">Playgen</span><span class="belief-stat-item">en cours de calcul…</span>`;
        statsEl.appendChild(note);
    }

    // Determine unknown cards
    const handsSource = currentState ? currentState.hands : initialHands;
    const observerHand = new Set(handsSource ? handsSource[observer] : []);
    const allCurrentCards = new Set();
    if (handsSource) {
        for (const h of handsSource) for (const c of h) allCurrentCards.add(c);
    }
    const playedCards = new Set();
    if (initialHands) {
        for (const h of initialHands) {
            for (const c of h) {
                if (!allCurrentCards.has(c)) playedCards.add(c);
            }
        }
    }
    if (currentState && currentState.current_trick) {
        for (const c of currentState.current_trick) {
            if (c >= 0 && c < 32) playedCards.add(c);
        }
    }

    const unknowns = [];
    for (let c = 0; c < 32; c++) {
        if (!observerHand.has(c) && !playedCards.has(c)) unknowns.push(c);
    }

    if (unknowns.length === 0) {
        statsEl.textContent = 'Aucune carte inconnue';
        return;
    }

    for (const { w, label } of modes) {
        if (!w || !groundTruth) continue;

        let correct = 0;
        let totalEntropy = 0;

        for (const c of unknowns) {
            // Find ground truth holder
            let truthHolder = -1;
            for (let p = 0; p < 4; p++) {
                if (p === observer) continue;
                if (groundTruth[p].includes(c)) { truthHolder = p; break; }
            }
            if (truthHolder < 0) continue;

            // Argmax (excluding observer)
            let maxProb = -1, maxPlayer = -1;
            for (let p = 0; p < 4; p++) {
                if (p === observer) continue;
                if (w[p][c] > maxProb) { maxProb = w[p][c]; maxPlayer = p; }
            }
            if (maxPlayer === truthHolder) correct++;

            // Entropy
            let entropy = 0;
            for (let p = 0; p < 4; p++) {
                if (p === observer) continue;
                const prob = w[p][c];
                if (prob > 0.001) entropy -= prob * Math.log2(prob);
            }
            totalEntropy += entropy;
        }

        const accuracy = unknowns.length > 0 ? (correct / unknowns.length * 100).toFixed(0) : '0';
        const avgEntropy = unknowns.length > 0 ? (totalEntropy / unknowns.length).toFixed(2) : '0';

        const statDiv = document.createElement('div');
        statDiv.className = 'belief-stat-group';
        statDiv.innerHTML = `
            <span class="belief-stat-label">${label}</span>
            <span class="belief-stat-item">Precision: <b>${accuracy}%</b></span>
            <span class="belief-stat-item">Top-1: <b>${correct}/${unknowns.length}</b></span>
            <span class="belief-stat-item">Entropie moy.: <b>${avgEntropy} bits</b></span>
        `;
        statsEl.appendChild(statDiv);
    }
}

function stepTo(target) {
    send({ type: 'belief_step_to', target });
}

// Event handlers
function onGenerateClick() {
    setLoading('Generation en cours...');
    send({ type: 'belief_generate' });
}

function onImportClick() {
    const cfn = prompt('Coller le CFN de la partie à analyser :');
    if (!cfn || !cfn.trim()) return;
    setLoading('Import en cours...');
    send({ type: 'belief_import', cfn: cfn.trim() });
}

async function onCopyClick() {
    if (!gameCfn) return;
    const btn = document.getElementById('belief-copy-btn');
    try {
        await navigator.clipboard.writeText(gameCfn);
        if (btn) { const t = btn.textContent; btn.textContent = 'CFN copié ✓'; setTimeout(() => { btn.textContent = t; }, 1500); }
    } catch (e) {
        // Clipboard blocked (e.g. non-HTTPS) — fall back to a prompt for manual copy.
        prompt('Copier le CFN :', gameCfn);
    }
}

function onPrevClick() { if (currentIdx > 0) stepTo(currentIdx - 1); }
function onNextClick() { if (currentIdx < totalActions) stepTo(currentIdx + 1); }
function onStartClick() { stepTo(0); }
function onEndClick() { stepTo(totalActions); }
function onSliderInput(e) { stepTo(parseInt(e.target.value)); }

function onObserverClick(e) {
    const btn = e.target.closest('.belief-obs-btn');
    if (!btn) return;
    observer = parseInt(btn.dataset.obs);
    document.querySelectorAll('.belief-obs-btn').forEach(b => b.classList.toggle('active', parseInt(b.dataset.obs) === observer));
    // Re-warm the playgen cache for the new observer
    precompute = null;
    renderPrecompute();
    send({ type: 'belief_precompute', observer });
    requestWeights();
}

function onModeClick(e) {
    const btn = e.target.closest('.belief-mode-btn');
    if (!btn) return;
    viewMode = btn.dataset.mode;
    document.querySelectorAll('.belief-mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === viewMode));
    // Playgen pas encore calculé pour cette position → re-demander
    if ((viewMode === 'playgen' || viewMode === 'compare') && !playgenWeights) {
        requestWeights();
    }
    renderGrid();
    renderStats();
}

// The server keeps the belief session per WebSocket connection. On reconnect
// that session is gone, but we still hold the full deal client-side — rebuild
// it at the current position so navigation keeps working seamlessly.
function restoreSession() {
    if (!allActions || totalActions === 0) return false;
    send({
        type: 'belief_restore',
        dealer,
        initial_hands: initialHands,
        actions: allActions,
        target: currentIdx,
        observer,
    });
    return true;
}

function onWsOpen() {
    // Only fires on a *re*connect while this view is mounted (the initial
    // connect predates mount). Silently restore the lost server session.
    restoreSession();
}

function onError(data) {
    setLoading('');
    console.error('Belief error:', data.msg);
    // Self-heal if the server lost our session (e.g. a click raced a reconnect
    // before onWsOpen restored it).
    if (data.msg && data.msg.includes('session croyances') && restoreSession()) {
        return;
    }
}

// Keyboard navigation
function onKeyDown(e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'TEXTAREA') return;
    if (e.key === 'ArrowLeft') { e.preventDefault(); onPrevClick(); }
    else if (e.key === 'ArrowRight') { e.preventDefault(); onNextClick(); }
    else if (e.key === 'Home') { e.preventDefault(); onStartClick(); }
    else if (e.key === 'End') { e.preventDefault(); onEndClick(); }
}

export function mount(container) {
    container.innerHTML = TEMPLATE;

    // Reset state
    totalActions = 0; currentIdx = 0; currentState = null;
    currentLastAction = null; initialHands = null;
    allActions = null; gameCfn = null; precompute = null;
    nnWeights = null; heuristicWeights = null; playgenWeights = null; groundTruth = null;
    observer = 0; viewMode = 'nn'; loading = false;

    // Wire events
    document.getElementById('belief-generate-btn').addEventListener('click', onGenerateClick);
    document.getElementById('belief-import-btn').addEventListener('click', onImportClick);
    document.getElementById('belief-copy-btn').addEventListener('click', onCopyClick);
    document.getElementById('belief-prev-btn').addEventListener('click', onPrevClick);
    document.getElementById('belief-next-btn').addEventListener('click', onNextClick);
    document.getElementById('belief-start-btn').addEventListener('click', onStartClick);
    document.getElementById('belief-end-btn').addEventListener('click', onEndClick);
    document.getElementById('belief-slider').addEventListener('input', onSliderInput);
    document.getElementById('belief-observer-btns').addEventListener('click', onObserverClick);
    document.getElementById('belief-mode-btns').addEventListener('click', onModeClick);
    document.getElementById('belief-history').addEventListener('click', onHistoryClick);
    document.addEventListener('keydown', onKeyDown);

    // WebSocket handlers
    onMessage('belief_generated', onGenerated);
    onMessage('belief_state', onState);
    onMessage('belief_weights', onWeights);
    onMessage('belief_precompute', onPrecompute);
    onMessage('error', onError);
    onOpen(onWsOpen);

    // Auto-load a game on tab entry
    onGenerateClick();
}

export function unmount() {
    offMessage('belief_generated', onGenerated);
    offMessage('belief_state', onState);
    offMessage('belief_weights', onWeights);
    offMessage('belief_precompute', onPrecompute);
    offMessage('error', onError);
    offOpen(onWsOpen);
    document.removeEventListener('keydown', onKeyDown);
}
