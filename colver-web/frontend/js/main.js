// WebSocket connection and shared utilities

let ws = null;
let messageHandlers = {};

function connect() {
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const base = document.querySelector('base')?.getAttribute('href') || '/';
    ws = new WebSocket(`${proto}://${location.host}${base}ws`);

    ws.onopen = () => console.log('Connecte');
    ws.onclose = () => {
        console.log('Deconnecte, reconnexion...');
        setTimeout(connect, 1000);
    };
    ws.onmessage = (evt) => {
        const data = JSON.parse(evt.data);
        const handler = messageHandlers[data.type];
        if (handler) handler(data);
        else console.log('Non gere:', data);
    };
}

function send(msg) {
    if (ws && ws.readyState === WebSocket.OPEN) {
        ws.send(JSON.stringify(msg));
    }
}

function onMessage(type, handler) {
    messageHandlers[type] = handler;
}

// Card rendering
const RANKS = ['7', '8', '9', 'V', 'D', 'R', '10', 'A'];
const SUITS = ['\u2660', '\u2665', '\u2666', '\u2663']; // spade heart diamond club
const SUIT_NAMES_EN = ['spades', 'hearts', 'diamonds', 'clubs'];
const RANK_NAMES_EN = ['7', '8', '9', 'jack', 'queen', 'king', '10', 'ace'];

function cardSuit(idx) { return idx >> 3; }
function cardRank(idx) { return idx & 7; }

// Map card index to SVG filename
function cardSvgPath(cardIdx) {
    const suit = cardSuit(cardIdx);
    const rank = cardRank(cardIdx);
    // Use version 2 art for face cards (jack=3, queen=4, king=5)
    const suffix = (rank >= 3 && rank <= 5) ? '2' : '';
    return `cards/${RANK_NAMES_EN[rank]}_of_${SUIT_NAMES_EN[suit]}${suffix}.svg`;
}

function cardToHtml(cardIdx, clickable = false, onClick = null, illegal = false) {
    const el = document.createElement('div');
    let cls = 'card';
    if (clickable) cls += ' clickable raised';
    if (illegal) cls += ' illegal';
    el.className = cls;

    const img = document.createElement('img');
    img.src = cardSvgPath(cardIdx);
    img.alt = `${RANKS[cardRank(cardIdx)]}${SUITS[cardSuit(cardIdx)]}`;
    img.draggable = false;
    el.appendChild(img);

    el.dataset.card = cardIdx;
    if (clickable && onClick) {
        el.addEventListener('click', () => onClick(cardIdx));
    }
    return el;
}

function faceDownCard() {
    const el = document.createElement('div');
    el.className = 'card face-down';
    return el;
}

// Sort keys for display order within a suit (lower = shown first / strongest)
// Plain: A > 10 > K > Q > J > 9 > 8 > 7
const PLAIN_ORDER = [7, 6, 5, 4, 3, 2, 1, 0]; // rank -> sort key
// Trump: J > 9 > A > 10 > K > Q > 8 > 7
const TRUMP_ORDER = [7, 6, 1, 0, 5, 4, 3, 2]; // rank -> sort key

function renderHand(container, cards, clickable = false, onClick = null, legalSet = null, trumpSuit = -1) {
    container.innerHTML = '';
    const sorted = [...cards].sort((a, b) => {
        const suitA = cardSuit(a), suitB = cardSuit(b);
        if (suitA !== suitB) return suitA - suitB;
        const orderA = suitA === trumpSuit ? TRUMP_ORDER : PLAIN_ORDER;
        const orderB = suitB === trumpSuit ? TRUMP_ORDER : PLAIN_ORDER;
        return orderA[cardRank(a)] - orderB[cardRank(b)];
    });
    for (const c of sorted) {
        const isLegal = !legalSet || legalSet.has(c);
        const cardClickable = clickable && isLegal;
        const illegal = clickable && !isLegal;
        container.appendChild(cardToHtml(c, cardClickable, onClick, illegal));
    }
}

function renderFaceDownHand(container, count) {
    container.innerHTML = '';
    for (let i = 0; i < count; i++) {
        container.appendChild(faceDownCard());
    }
}

function renderTrick(prefix, trick) {
    const seatMap = { 0: 'n', 1: 'e', 2: 's', 3: 'w' };
    for (let seat = 0; seat < 4; seat++) {
        const el = document.getElementById(`${prefix}-${seatMap[seat]}`);
        el.innerHTML = '';
        const c = trick[seat];
        if (c >= 0 && c < 32) {
            el.appendChild(cardToHtml(c));
        }
    }
}

function renderLastTrick(container, trick, trickWinner, trickPoints, humanSeat) {
    container.innerHTML = '';
    if (!trick || trick.every(c => c < 0 || c >= 32)) {
        container.classList.add('hidden');
        return;
    }
    container.classList.remove('hidden');
    // Team color: green if partner team won, red if opponent
    const isPartnerWin = trickWinner !== null && (trickWinner % 2) === (humanSeat % 2);
    const teamClass = trickWinner !== null ? (isPartnerWin ? 'team-partner' : 'team-opponent') : '';
    const label = document.createElement('div');
    label.className = 'last-trick-label ' + teamClass;
    const pts = trickPoints || 0;
    label.textContent = trickWinner !== null
        ? `${SEAT_NAMES_FR[trickWinner]} +${pts}`
        : `Pli +${pts}`;
    container.appendChild(label);
    // Compass grid: N top, W left, E right, S bottom
    const grid = document.createElement('div');
    grid.className = 'last-trick-grid';
    const positions = { 0: 'lt-n', 1: 'lt-e', 2: 'lt-s', 3: 'lt-w' };
    for (let seat = 0; seat < 4; seat++) {
        const c = trick[seat];
        const cell = document.createElement('div');
        cell.className = positions[seat];
        if (c >= 0 && c < 32) {
            cell.appendChild(cardToHtml(c));
        }
        grid.appendChild(cell);
    }
    container.appendChild(grid);
}

const SEAT_NAMES_FR = ['Nord', 'Est', 'Sud', 'Ouest'];

function showBeloteAnnouncement(trickAreaId, text) {
    const area = document.getElementById(trickAreaId);
    if (!area) return;
    const el = document.createElement('div');
    el.className = 'belote-announcement';
    el.textContent = text;
    area.style.position = 'relative';
    area.appendChild(el);
    setTimeout(() => el.remove(), 1800);
}

function renderBeloteBadge(scoreElId, beloteVal) {
    const scoreEl = document.getElementById(scoreElId);
    if (!scoreEl) return;
    const existing = scoreEl.querySelector('.belote-badge');
    if (existing) existing.remove();
    if (beloteVal === 2) {
        const badge = document.createElement('span');
        badge.className = 'belote-badge';
        badge.textContent = '+20 belote';
        scoreEl.appendChild(badge);
    }
}

function contractStr(contract) {
    if (!contract || Object.keys(contract).length === 0) return '';
    const val = contract.value;
    const suit = SUITS[contract.trump];
    const team = contract.team === 0 ? 'NS' : 'EO';
    const coinche = contract.coinche === 1 ? ' x' : contract.coinche === 2 ? ' xx' : '';
    return `${val}${suit} par ${team}${coinche}`;
}

function actionName(action, phase) {
    if (phase === 0) {
        if (action === 0) return 'Passe';
        if (action === 41) return 'Coinche';
        if (action === 42) return 'Surcoinche';
        if (action >= 1 && action <= 40) {
            let valIdx, suitIdx;
            if (action <= 36) {
                const idx = action - 1;
                valIdx = Math.floor(idx / 4);
                suitIdx = idx % 4;
                const values = [80,90,100,110,120,130,140,150,160];
                return `${values[valIdx]}${SUITS[suitIdx]}`;
            } else {
                suitIdx = action - 37;
                return `Capot${SUITS[suitIdx]}`;
            }
        }
        return `?${action}`;
    }
    return `${RANKS[cardRank(action)]}${SUITS[cardSuit(action)]}`;
}

// Tab switching
document.querySelectorAll('.tab').forEach(btn => {
    btn.addEventListener('click', () => {
        document.querySelectorAll('.tab').forEach(b => b.classList.remove('active'));
        document.querySelectorAll('.panel').forEach(p => p.classList.remove('active'));
        btn.classList.add('active');
        document.getElementById(`${btn.dataset.tab}-panel`).classList.add('active');
    });
});

// Range value displays
document.getElementById('ai-time').addEventListener('input', e => {
    document.getElementById('ai-time-val').textContent = e.target.value;
});
document.getElementById('analysis-time').addEventListener('input', e => {
    document.getElementById('analysis-time-val').textContent = e.target.value;
});
document.getElementById('watch-time').addEventListener('input', e => {
    document.getElementById('watch-time-val').textContent = e.target.value;
});
document.getElementById('watch-speed').addEventListener('input', e => {
    document.getElementById('watch-speed-val').textContent = e.target.value;
});

// Connect on load
connect();
