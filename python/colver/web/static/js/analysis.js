// Analysis mode logic — drag-and-drop card assignment

let analysisHands = [[], [], [], []];
let assignedCards = new Set();

const PLAYER_NAMES_FR = ['Nord', 'Est', 'Sud', 'Ouest'];

// Currently dragged card info
let dragCardIdx = null;
let dragSource = null; // 'palette' or player index (0-3)

function createDraggableCard(cardIdx, source) {
    const el = document.createElement('div');
    el.className = 'card';
    el.draggable = true;

    const img = document.createElement('img');
    img.src = cardSvgPath(cardIdx);
    img.alt = `${RANKS[cardRank(cardIdx)]}${SUITS[cardSuit(cardIdx)]}`;
    img.draggable = false;
    el.appendChild(img);

    el.dataset.card = cardIdx;
    el.dataset.source = source;

    el.addEventListener('dragstart', (e) => {
        dragCardIdx = cardIdx;
        dragSource = source;
        el.classList.add('dragging');
        e.dataTransfer.effectAllowed = 'move';
        e.dataTransfer.setData('text/plain', String(cardIdx));
    });

    el.addEventListener('dragend', () => {
        el.classList.remove('dragging');
        dragCardIdx = null;
        dragSource = null;
        // Clean up all drag-over highlights
        document.querySelectorAll('.drag-over').forEach(z => z.classList.remove('drag-over'));
    });

    // Cards in drop zones: click to remove
    if (source !== 'palette') {
        el.addEventListener('click', () => {
            removeCardFromPlayer(cardIdx);
            updateCardDisplay();
        });
    }

    return el;
}

function initCardPalette() {
    const palette = document.getElementById('card-palette');
    palette.innerHTML = '';
    for (let suit = 0; suit < 4; suit++) {
        for (let rank = 0; rank < 8; rank++) {
            const idx = suit * 8 + rank;
            const card = createDraggableCard(idx, 'palette');
            card.id = `palette-card-${idx}`;
            if (assignedCards.has(idx)) card.classList.add('assigned');
            palette.appendChild(card);
        }
    }
}

function initDropZones() {
    // Drop zones accept cards
    document.querySelectorAll('.drop-zone').forEach(zone => {
        zone.addEventListener('dragover', (e) => {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            zone.classList.add('drag-over');
        });

        zone.addEventListener('dragleave', (e) => {
            // Only remove if leaving the zone itself (not entering a child)
            if (!zone.contains(e.relatedTarget)) {
                zone.classList.remove('drag-over');
            }
        });

        zone.addEventListener('drop', (e) => {
            e.preventDefault();
            zone.classList.remove('drag-over');
            const cardIdx = parseInt(e.dataTransfer.getData('text/plain'));
            if (isNaN(cardIdx)) return;
            const playerIdx = parseInt(zone.dataset.player);

            if (dragSource === 'palette') {
                // From palette to drop zone
                assignCardToPlayer(cardIdx, playerIdx);
            } else {
                // From one drop zone to another (or same)
                const srcPlayer = parseInt(dragSource);
                if (srcPlayer === playerIdx) return;
                // Remove from source player
                removeCardFromPlayer(cardIdx);
                // Add to target player
                assignCardToPlayer(cardIdx, playerIdx);
            }
            updateCardDisplay();
        });
    });

    // Palette accepts cards back (drag from drop zone to palette = remove)
    const palette = document.getElementById('card-palette');
    palette.addEventListener('dragover', (e) => {
        if (dragSource !== 'palette') {
            e.preventDefault();
            e.dataTransfer.dropEffect = 'move';
            palette.classList.add('drag-over');
        }
    });

    palette.addEventListener('dragleave', (e) => {
        if (!palette.contains(e.relatedTarget)) {
            palette.classList.remove('drag-over');
        }
    });

    palette.addEventListener('drop', (e) => {
        e.preventDefault();
        palette.classList.remove('drag-over');
        if (dragSource === 'palette') return;
        const cardIdx = parseInt(e.dataTransfer.getData('text/plain'));
        if (isNaN(cardIdx)) return;
        removeCardFromPlayer(cardIdx);
        updateCardDisplay();
    });
}

function assignCardToPlayer(cardIdx, playerIdx) {
    if (analysisHands[playerIdx].length >= 8) return false;
    if (assignedCards.has(cardIdx)) return false;
    analysisHands[playerIdx].push(cardIdx);
    assignedCards.add(cardIdx);
    return true;
}

function removeCardFromPlayer(cardIdx) {
    for (let p = 0; p < 4; p++) {
        const i = analysisHands[p].indexOf(cardIdx);
        if (i >= 0) {
            analysisHands[p].splice(i, 1);
            break;
        }
    }
    assignedCards.delete(cardIdx);
}

function updateCardDisplay() {
    // Update palette: fade assigned cards
    for (let i = 0; i < 32; i++) {
        const el = document.getElementById(`palette-card-${i}`);
        if (el) {
            el.classList.toggle('assigned', assignedCards.has(i));
            el.draggable = !assignedCards.has(i);
        }
    }

    // Update each drop zone
    document.querySelectorAll('.drop-zone').forEach(zone => {
        const playerIdx = parseInt(zone.dataset.player);
        const cards = analysisHands[playerIdx];
        const countEl = zone.querySelector('.dz-count');
        countEl.textContent = `(${cards.length}/8)`;

        zone.classList.toggle('full', cards.length === 8);

        const container = zone.querySelector('.drop-zone-cards');
        container.innerHTML = '';
        const sorted = [...cards].sort((a, b) => a - b);
        for (const c of sorted) {
            container.appendChild(createDraggableCard(c, String(playerIdx)));
        }
    });
}

document.getElementById('random-deal').addEventListener('click', () => {
    const cards = Array.from({ length: 32 }, (_, i) => i);
    for (let i = cards.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [cards[i], cards[j]] = [cards[j], cards[i]];
    }
    analysisHands = [
        cards.slice(0, 8),
        cards.slice(8, 16),
        cards.slice(16, 24),
        cards.slice(24, 32),
    ];
    assignedCards = new Set(cards);
    initCardPalette();
    updateCardDisplay();
});

document.getElementById('setup-analysis').addEventListener('click', () => {
    for (let p = 0; p < 4; p++) {
        if (analysisHands[p].length !== 8) {
            alert(`${PLAYER_NAMES_FR[p]} doit avoir exactement 8 cartes (en a ${analysisHands[p].length})`);
            return;
        }
    }

    const trump = parseInt(document.getElementById('analysis-trump').value);
    const value = parseInt(document.getElementById('analysis-value').value);
    const team = parseInt(document.getElementById('analysis-team').value);

    send({
        type: 'setup_analysis',
        dealer: 0,
        hands: analysisHands,
        contract: { trump, value, team, coinche: 0 },
    });
});

document.getElementById('run-analysis').addEventListener('click', () => {
    const agent = document.getElementById('analysis-agent').value;
    const timeMs = parseInt(document.getElementById('analysis-time').value);
    document.getElementById('run-analysis').disabled = true;
    document.getElementById('run-analysis').textContent = 'Analyse en cours...';
    send({ type: 'analyze', agent, time_ms: timeMs });
});

function renderAnalysisState(state) {
    document.getElementById('analysis-table').classList.remove('hidden');

    const handEls = {
        0: document.getElementById('analysis-hand-north'),
        1: document.getElementById('analysis-hand-east'),
        2: document.getElementById('analysis-hand-south'),
        3: document.getElementById('analysis-hand-west'),
    };
    const trumpSuit = (state.contract && state.contract.trump !== undefined) ? state.contract.trump : -1;
    for (let seat = 0; seat < 4; seat++) {
        renderHand(handEls[seat], state.hands[seat], false, null, null, trumpSuit);
    }
    renderTrick('analysis-trick', state.current_trick);
}

// Message handlers
onMessage('analysis_ready', (data) => {
    renderAnalysisState(data.state);
});

onMessage('analysis_result', (data) => {
    document.getElementById('run-analysis').disabled = false;
    document.getElementById('run-analysis').textContent = 'Analyser';

    const el = document.getElementById('analysis-result');
    el.innerHTML = '';

    const best = document.createElement('div');
    best.className = 'best-action';
    best.textContent = `Meilleur coup : ${data.name}`;
    el.appendChild(best);

    if (data.legal_actions) {
        const list = document.createElement('div');
        list.className = 'action-list';
        for (const a of data.legal_actions) {
            const item = document.createElement('div');
            item.className = 'action-item' + (a.action === data.best ? ' best' : '');
            item.textContent = a.name;
            list.appendChild(item);
        }
        el.appendChild(list);
    }
});

// Init
initCardPalette();
initDropZones();
updateCardDisplay();
