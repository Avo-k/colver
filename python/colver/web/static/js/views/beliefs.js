// Croyances view — belief network visualization
// Step through a game and see how NN/heuristic beliefs evolve

import { send, onMessage, offMessage } from '../ws.js';
import { RANKS, SUITS, cardSvgPath, cardRank, cardSuit, SEAT_NAMES_FR, SUIT_DISPLAY_ORDER } from '../shared/cards.js';

const SUIT_SYMBOLS = ['\u2660', '\u2665', '\u2666', '\u2663'];
const SEAT_NAMES = ['N', 'E', 'S', 'O'];
const SEAT_COLORS = ['#5b9ff5', '#e06080', '#d4af37', '#8c6fe6'];

// Display order within a suit: A 10 R D V 9 8 7
const DISPLAY_ORDER = [7, 6, 5, 4, 3, 2, 1, 0];

function suitHtml(suitIdx) {
    const cls = (suitIdx === 1 || suitIdx === 2) ? 'suit-red' : 'suit-black';
    return `<span class="${cls}">${SUIT_SYMBOLS[suitIdx]}</span>`;
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
let currentIdx = 0;
let currentState = null;
let currentLastAction = null;
let initialHands = null;
let dealer = 0;
let observer = 0;
let viewMode = 'nn'; // 'nn', 'heuristic', 'compare'
let nnWeights = null;
let heuristicWeights = null;
let groundTruth = null;
let loading = false;

const TEMPLATE = `
<div id="belief-config">
    <button id="belief-generate-btn" class="primary-btn">Nouvelle partie</button>
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
                <button class="belief-mode-btn" data-mode="heuristic">Heuristique</button>
                <button class="belief-mode-btn" data-mode="compare">Comparer</button>
            </div>
        </div>
    </div>
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
    send({ type: 'belief_get_weights', observer });
}

function onGenerated(data) {
    setLoading('');
    initialHands = data.initial_hands;
    dealer = data.dealer;
    totalActions = data.total_actions;
    numBidActions = data.num_bid_actions;
    currentIdx = 0;
    currentState = null;
    currentLastAction = null;
    nnWeights = null;
    heuristicWeights = null;
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
    requestWeights();
}

function onWeights(data) {
    if (data.observer !== observer) return;
    nnWeights = data.nn;
    heuristicWeights = data.heuristic;
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

function renderGrid() {
    const grid = document.getElementById('belief-grid');
    if (!grid) return;
    grid.innerHTML = '';

    if (!groundTruth) return;

    const weights = viewMode === 'heuristic' ? heuristicWeights : nnWeights;
    const weights2 = viewMode === 'compare' ? heuristicWeights : null;

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

                if (weights) {
                    const bar = createBeliefBar(weights, cardIdx, observer, groundTruth, viewMode === 'compare' ? 'NN' : null);
                    cardDiv.appendChild(bar);
                }
                if (weights2) {
                    const bar2 = createBeliefBar(weights2, cardIdx, observer, groundTruth, 'Heur.');
                    cardDiv.appendChild(bar2);
                }
                if (!weights && !weights2) {
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
    if (viewMode === 'heuristic' || viewMode === 'compare') modes.push({ w: heuristicWeights, label: 'Heuristique' });

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
    requestWeights();
}

function onModeClick(e) {
    const btn = e.target.closest('.belief-mode-btn');
    if (!btn) return;
    viewMode = btn.dataset.mode;
    document.querySelectorAll('.belief-mode-btn').forEach(b => b.classList.toggle('active', b.dataset.mode === viewMode));
    renderGrid();
    renderStats();
}

function onError(data) {
    setLoading('');
    console.error('Belief error:', data.msg);
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
    nnWeights = null; heuristicWeights = null; groundTruth = null;
    observer = 0; viewMode = 'nn'; loading = false;

    // Wire events
    document.getElementById('belief-generate-btn').addEventListener('click', onGenerateClick);
    document.getElementById('belief-prev-btn').addEventListener('click', onPrevClick);
    document.getElementById('belief-next-btn').addEventListener('click', onNextClick);
    document.getElementById('belief-start-btn').addEventListener('click', onStartClick);
    document.getElementById('belief-end-btn').addEventListener('click', onEndClick);
    document.getElementById('belief-slider').addEventListener('input', onSliderInput);
    document.getElementById('belief-observer-btns').addEventListener('click', onObserverClick);
    document.getElementById('belief-mode-btns').addEventListener('click', onModeClick);
    document.addEventListener('keydown', onKeyDown);

    // WebSocket handlers
    onMessage('belief_generated', onGenerated);
    onMessage('belief_state', onState);
    onMessage('belief_weights', onWeights);
    onMessage('error', onError);

    // Auto-load a game on tab entry
    onGenerateClick();
}

export function unmount() {
    offMessage('belief_generated', onGenerated);
    offMessage('belief_state', onState);
    offMessage('belief_weights', onWeights);
    offMessage('error', onError);
    document.removeEventListener('keydown', onKeyDown);
}
