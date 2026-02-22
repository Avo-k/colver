// Annonces tab — hand builder + bidding NN evaluation

let annoncesHand = new Set();
let annoncesHistory = []; // array of action indices (prior bids before our turn)
let ddTimerId = null;
let ddStartTime = 0;
let ddEstimatedMs = 0;

const SEAT_NAMES = ['N', 'E', 'S', 'O'];
// N=light blue, E=light green, S=gold, O=orange
const SEAT_COLORS = ['#82cfff', '#82e0aa', '#d4af37', '#f0b429'];

function annoncesPlayerSeat(turnIdx, historyLen) {
    return (2 - historyLen + turnIdx + 32) % 4;
}

function initAnnoncesGrid() {
    const palette = document.getElementById('annonces-palette');
    palette.innerHTML = '';
    for (let suit = 0; suit < 4; suit++) {
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

function initActionSelect() {
    const select = document.getElementById('annonces-action-select');
    select.innerHTML = '';

    const addOpt = (value, text) => {
        const opt = document.createElement('option');
        opt.value = value;
        opt.textContent = text;
        select.appendChild(opt);
    };

    addOpt(0, 'Passe');
    for (let valIdx = 0; valIdx < 9; valIdx++) {
        const value = 80 + valIdx * 10;
        for (let suitIdx = 0; suitIdx < 4; suitIdx++) {
            addOpt(valIdx * 4 + suitIdx + 1, `${value} ${SUITS[suitIdx]}`);
        }
    }
    for (let suitIdx = 0; suitIdx < 4; suitIdx++) {
        addOpt(37 + suitIdx, `Capot ${SUITS[suitIdx]}`);
    }
    addOpt(41, 'Coinche');
    addOpt(42, 'Surcoinche');
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

function renderAnnoncesHistory() {
    const list = document.getElementById('annonces-history-list');
    list.innerHTML = '';
    const n = annoncesHistory.length;

    annoncesHistory.forEach((action, i) => {
        const seat = annoncesPlayerSeat(i, n);
        const row = document.createElement('div');
        row.className = 'ann-history-row';

        const badge = document.createElement('span');
        badge.className = 'ann-seat-badge';
        badge.textContent = SEAT_NAMES[seat];
        badge.style.color = SEAT_COLORS[seat];

        const actionSpan = document.createElement('span');
        actionSpan.className = 'ann-action-name';
        actionSpan.textContent = bidActionName(action);

        const delBtn = document.createElement('button');
        delBtn.className = 'ann-del-btn';
        delBtn.textContent = '×';
        delBtn.title = 'Supprimer';
        delBtn.addEventListener('click', () => {
            annoncesHistory.splice(i, 1);
            renderAnnoncesHistory();
        });

        row.appendChild(badge);
        row.appendChild(actionSpan);
        row.appendChild(delBtn);
        list.appendChild(row);
    });

    // "Your turn" row always at end
    const yourRow = document.createElement('div');
    yourRow.className = 'ann-history-row ann-your-turn';
    const yourBadge = document.createElement('span');
    yourBadge.className = 'ann-seat-badge';
    yourBadge.textContent = 'S';
    yourBadge.style.color = SEAT_COLORS[2];
    const yourLabel = document.createElement('span');
    yourLabel.className = 'ann-action-name';
    yourLabel.textContent = 'Votre tour';
    yourRow.appendChild(yourBadge);
    yourRow.appendChild(yourLabel);
    list.appendChild(yourRow);
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

document.getElementById('annonces-history-add-btn').addEventListener('click', () => {
    const select = document.getElementById('annonces-action-select');
    const action = parseInt(select.value);
    annoncesHistory.push(action);
    renderAnnoncesHistory();
});

document.getElementById('annonces-history-clear-btn').addEventListener('click', () => {
    annoncesHistory = [];
    renderAnnoncesHistory();
});

document.getElementById('annonces-clear-btn').addEventListener('click', () => {
    annoncesHand.clear();
    updateAnnoncesDisplay();
    document.getElementById('annonces-results-row').classList.add('hidden');
    stopDdTimer();
});

// DD loading progress
function startDdTimer(numSims) {
    ddEstimatedMs = numSims * 150; // ~150ms per sim estimate
    ddStartTime = Date.now();
    const estSec = (ddEstimatedMs / 1000).toFixed(1);
    document.getElementById('annonces-dd-header').textContent = `Oracle DD`;
    document.getElementById('annonces-dd-body').innerHTML =
        `<div class="dd-loader">
            <div class="dd-loader-text">Résolution de ${numSims} donnes (~${estSec}s)…</div>
            <div class="dd-progress-bar"><div class="dd-progress-fill" id="dd-progress-fill"></div></div>
            <div class="dd-loader-pct" id="dd-loader-pct">0%</div>
        </div>`;
    ddTimerId = setInterval(updateDdProgress, 100);
}

function updateDdProgress() {
    const elapsed = Date.now() - ddStartTime;
    // Allow progress to go slightly past 100% (cap display at 99% until done)
    const pct = Math.min(99, Math.round((elapsed / ddEstimatedMs) * 100));
    const fill = document.getElementById('dd-progress-fill');
    const label = document.getElementById('dd-loader-pct');
    if (fill) fill.style.width = pct + '%';
    if (label) label.textContent = pct + '%';
}

function stopDdTimer() {
    if (ddTimerId) {
        clearInterval(ddTimerId);
        ddTimerId = null;
    }
}

// Eval button fires both NN eval and DD simulation
document.getElementById('annonces-eval-btn').addEventListener('click', () => {
    const hand = Array.from(annoncesHand);
    const numSims = Math.max(1, Math.min(200, parseInt(document.getElementById('annonces-sim-count').value) || 10));

    // Show results row immediately with loading state in DD column
    document.getElementById('annonces-results-row').classList.remove('hidden');
    document.getElementById('annonces-loading').classList.add('hidden');

    // Clear Q-values column — instant result incoming
    document.getElementById('annonces-results-header').textContent = 'Le Bide à Dédé';
    document.getElementById('annonces-results-body').innerHTML =
        '<div class="dd-loader"><div class="dd-loader-text">Calcul…</div></div>';

    // Start DD loader with progress bar
    startDdTimer(numSims);

    send({ type: 'bid_eval', hand, prior_actions: annoncesHistory });
    send({ type: 'dd_sim', hand, prior_actions: annoncesHistory, num_sims: numSims });
});

onMessage('bid_eval_result', (data) => {
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
        `Le Bide à Dédé : ${bidActionName(bestAction)}`;

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

// DD simulation result
const DD_SUIT_SYMBOLS = ['♠', '♥', '♦', '♣'];
const DD_SUIT_CLASSES = ['', 'red', 'red', ''];

function ddSuggestedBid(avgNs) {
    const thresholds = [160, 150, 140, 130, 120, 110, 100, 90, 80];
    for (const t of thresholds) {
        if (avgNs >= t) return t;
    }
    return null;
}

onMessage('dd_sim_result', (data) => {
    stopDdTimer();

    if (data.error) {
        document.getElementById('annonces-dd-body').innerHTML =
            `<div class="annonces-error">${data.error}</div>`;
        document.getElementById('annonces-dd-header').textContent = 'Erreur';
        return;
    }

    const suits = data.suits;
    const elapsed = data.elapsed_ms;
    const numSims = data.num_sims;

    document.getElementById('annonces-dd-header').textContent =
        `Oracle DD (${numSims} donnes, ${(elapsed / 1000).toFixed(1)}s)`;

    let html = '<table class="dd-sim-table"><tr><th>Atout</th><th>Moy. NS</th><th>Moy. EO</th><th>Annonce</th></tr>';
    for (const s of suits) {
        const suitClass = DD_SUIT_CLASSES[s.suit];
        const symbol = DD_SUIT_SYMBOLS[s.suit];
        const bid = ddSuggestedBid(s.avg_ns);
        const bidText = bid ? `${bid} ${symbol}` : '—';
        const bidClass = bid ? (bid >= 100 ? 'dd-bid-high' : 'dd-bid-ok') : 'dd-bid-none';
        html += `<tr>
            <td class="${suitClass}">${symbol}</td>
            <td>${s.avg_ns.toFixed(1)}</td>
            <td>${s.avg_ew.toFixed(1)}</td>
            <td class="${bidClass}">${bidText}</td>
        </tr>`;
    }
    html += '</table>';
    document.getElementById('annonces-dd-body').innerHTML = html;
});

// Init
initAnnoncesGrid();
initActionSelect();
renderAnnoncesHistory();
updateAnnoncesDisplay();
