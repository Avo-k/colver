// Analysis mode logic

let analysisHands = [[], [], [], []];
let assignedCards = new Set();

const PLAYER_NAMES_FR = ['Nord', 'Est', 'Sud', 'Ouest'];

function initCardPalette() {
    const palette = document.getElementById('card-palette');
    palette.innerHTML = '';
    for (let suit = 0; suit < 4; suit++) {
        for (let rank = 0; rank < 8; rank++) {
            const idx = suit * 8 + rank;
            const card = cardToHtml(idx, true, () => toggleCard(idx));
            card.id = `palette-card-${idx}`;
            if (assignedCards.has(idx)) card.classList.add('assigned');
            palette.appendChild(card);
        }
    }
}

function toggleCard(cardIdx) {
    const player = parseInt(document.getElementById('edit-player').value);
    if (assignedCards.has(cardIdx)) {
        for (let p = 0; p < 4; p++) {
            const i = analysisHands[p].indexOf(cardIdx);
            if (i >= 0) {
                analysisHands[p].splice(i, 1);
                break;
            }
        }
        assignedCards.delete(cardIdx);
    } else {
        if (analysisHands[player].length >= 8) return;
        analysisHands[player].push(cardIdx);
        assignedCards.add(cardIdx);
    }
    updateCardDisplay();
}

function updateCardDisplay() {
    for (let i = 0; i < 32; i++) {
        const el = document.getElementById(`palette-card-${i}`);
        if (el) {
            if (assignedCards.has(i)) el.classList.add('assigned');
            else el.classList.remove('assigned');
        }
    }
    const container = document.getElementById('assigned-cards');
    container.innerHTML = '';
    for (let p = 0; p < 4; p++) {
        const div = document.createElement('div');
        div.className = 'player-cards';
        div.innerHTML = `<div class="label">${PLAYER_NAMES_FR[p]} (${analysisHands[p].length})</div>`;
        const sorted = [...analysisHands[p]].sort((a, b) => a - b);
        for (const c of sorted) {
            div.appendChild(cardToHtml(c));
        }
        container.appendChild(div);
    }
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
    for (let seat = 0; seat < 4; seat++) {
        renderHand(handEls[seat], state.hands[seat]);
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
updateCardDisplay();
