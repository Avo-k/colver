// Deal mode logic — drag-and-drop card assignment + save to DB

let dealHands = [[], [], [], []];
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
    if (dealHands[playerIdx].length >= 8) return false;
    if (assignedCards.has(cardIdx)) return false;
    dealHands[playerIdx].push(cardIdx);
    assignedCards.add(cardIdx);
    return true;
}

function removeCardFromPlayer(cardIdx) {
    for (let p = 0; p < 4; p++) {
        const i = dealHands[p].indexOf(cardIdx);
        if (i >= 0) {
            dealHands[p].splice(i, 1);
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
        const cards = dealHands[playerIdx];
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
    dealHands = [
        cards.slice(0, 8),
        cards.slice(8, 16),
        cards.slice(16, 24),
        cards.slice(24, 32),
    ];
    assignedCards = new Set(cards);
    initCardPalette();
    updateCardDisplay();
});

document.getElementById('save-deal').addEventListener('click', () => {
    for (let p = 0; p < 4; p++) {
        if (dealHands[p].length !== 8) {
            alert(`${PLAYER_NAMES_FR[p]} doit avoir exactement 8 cartes (en a ${dealHands[p].length})`);
            return;
        }
    }

    const dealer = parseInt(document.getElementById('deal-dealer').value);
    const agents = {};
    document.querySelectorAll('.deal-agent-select').forEach(sel => {
        agents[sel.dataset.seat] = sel.value;
    });

    send({
        type: 'save_custom_deal',
        dealer,
        hands: dealHands,
        agents,
    });
});

// Message handlers
onMessage('deal_saved', (data) => {
    const feedback = document.getElementById('deal-feedback');
    feedback.classList.remove('hidden');
    feedback.innerHTML = '';

    const msg = document.createElement('span');
    msg.textContent = `Donne enregistree : `;
    feedback.appendChild(msg);

    const idTag = document.createElement('span');
    idTag.className = 'game-id-tag';
    idTag.textContent = data.game_id;
    feedback.appendChild(idTag);

    const watchBtn = document.createElement('button');
    watchBtn.textContent = 'Regarder';
    watchBtn.className = 'deal-watch-btn';
    watchBtn.addEventListener('click', () => {
        // Switch to watch tab
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        document.querySelector('[data-tab="watch"]').classList.add('active');
        document.getElementById('watch-panel').classList.add('active');

        // Send watch_custom to start a live session with this deal
        send({ type: 'watch_custom', game_id: data.game_id });
    });
    feedback.appendChild(watchBtn);
});

// Init
initCardPalette();
initDropZones();
updateCardDisplay();
