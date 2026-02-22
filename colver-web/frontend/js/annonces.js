// Annonces tab — hand builder + bidding NN evaluation

let annoncesHand = new Set();

function initAnnoncesGrid() {
    const palette = document.getElementById('annonces-palette');
    palette.innerHTML = '';
    for (let suit = 0; suit < 4; suit++) {
        // Suit label
        const label = document.createElement('div');
        label.className = 'palette-suit-label';
        label.textContent = SUITS[suit];
        label.style.color = (suit === 1 || suit === 2) ? '#ef9a9a' : '#ddd';
        palette.appendChild(label);

        for (let rank = 0; rank < 8; rank++) {
            const idx = suit * 8 + rank;
            const el = document.createElement('div');
            el.className = 'card annonces-card';
            el.id = `annonces-card-${idx}`;

            const img = document.createElement('img');
            img.src = cardSvgPath(idx);
            img.alt = `${RANKS[rank]}${SUITS[suit]}`;
            img.draggable = false;
            el.appendChild(img);

            el.addEventListener('click', () => toggleAnnoncesCard(idx));
            palette.appendChild(el);
        }
    }
}

function toggleAnnoncesCard(idx) {
    if (annoncesHand.has(idx)) {
        annoncesHand.delete(idx);
    } else {
        if (annoncesHand.size >= 8) return;
        annoncesHand.add(idx);
    }
    updateAnnoncesDisplay();
}

function updateAnnoncesDisplay() {
    const count = annoncesHand.size;
    const full = count === 8;
    for (let i = 0; i < 32; i++) {
        const el = document.getElementById(`annonces-card-${i}`);
        if (!el) continue;
        const selected = annoncesHand.has(i);
        el.classList.toggle('ann-selected', selected);
        el.classList.toggle('ann-faded', full && !selected);
    }
    document.getElementById('annonces-count').textContent = `${count}/8 cartes`;
    document.getElementById('annonces-eval-btn').disabled = count !== 8;

    // Update hand preview
    const handEl = document.getElementById('annonces-hand-display');
    renderHand(handEl, Array.from(annoncesHand));
}

function bidActionName(action) {
    if (action === 0) return 'Passe';
    if (action >= 37 && action <= 40) return `Capot ${SUITS[action - 37]}`;
    if (action === 41) return 'Coinche';
    if (action === 42) return 'Surcoinche';
    if (action >= 1 && action <= 36) {
        const bidIdx = action - 1;
        const valueIdx = Math.floor(bidIdx / 4);
        const suitIdx = bidIdx % 4;
        const value = 80 + valueIdx * 10;
        return `${value} ${SUITS[suitIdx]}`;
    }
    return `Action ${action}`;
}

document.getElementById('annonces-clear-btn').addEventListener('click', () => {
    annoncesHand.clear();
    updateAnnoncesDisplay();
    document.getElementById('annonces-results').classList.add('hidden');
});

document.getElementById('annonces-eval-btn').addEventListener('click', () => {
    const priorPasses = parseInt(document.getElementById('annonces-passes').value);
    const hand = Array.from(annoncesHand);
    document.getElementById('annonces-results').classList.add('hidden');
    document.getElementById('annonces-loading').classList.remove('hidden');
    send({ type: 'bid_eval', hand, prior_passes: priorPasses });
});

onMessage('bid_eval_result', (data) => {
    document.getElementById('annonces-loading').classList.add('hidden');
    const resultsEl = document.getElementById('annonces-results');
    resultsEl.classList.remove('hidden');

    if (data.error) {
        document.getElementById('annonces-results-body').innerHTML =
            `<div class="annonces-error">${data.error}</div>`;
        document.getElementById('annonces-results-header').textContent = 'Erreur';
        return;
    }

    const qValues = data.q_values;
    const bestAction = data.best_action;
    const top = qValues.slice(0, 10);
    const minQ = Math.min(...top.map(([, q]) => q));
    const maxQ = Math.max(...top.map(([, q]) => q));
    const range = maxQ - minQ || 1;

    document.getElementById('annonces-results-header').textContent =
        `Meilleure annonce : ${bidActionName(bestAction)}`;

    let html = '<div class="visit-bars">';
    for (const [action, q] of top) {
        const pct = ((q - minQ) / range * 100).toFixed(0);
        const isBest = action === bestAction;
        const name = bidActionName(action);
        html += `<div class="visit-row${isBest ? ' best' : ''}">
            <span class="visit-name">${name}</span>
            <div class="visit-bar-bg"><div class="visit-bar-fill q-fill" style="width:${pct}%"></div></div>
            <span class="visit-count">${q.toFixed(3)}</span>
        </div>`;
    }
    html += '</div>';
    document.getElementById('annonces-results-body').innerHTML = html;
});

// Init
initAnnoncesGrid();
updateAnnoncesDisplay();
