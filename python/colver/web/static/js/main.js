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

// Trick flush animation state
let _prevTrick = {};       // prefix -> array of 4 card indices from previous render
let _animatingTrick = null; // prefix currently animating, or null

function detectTrickCompletion(prefix, newTrick) {
    const prev = _prevTrick[prefix];
    _prevTrick[prefix] = newTrick ? [...newTrick] : [];
    if (!prev) return null;
    // Was previous trick full (4 valid cards)?
    const prevFull = prev.filter(c => c >= 0 && c < 32).length === 4;
    const newFull = newTrick ? newTrick.filter(c => c >= 0 && c < 32).length : 0;
    if (prevFull && newFull < 4) {
        return prev; // Return the 4 cards that were in the completed trick
    }
    return null;
}

function animateTrickFlush(prefix, onComplete, winner) {
    const trickAreaId = prefix === 'trick' ? 'trick-area' : 'watch-trick-area';
    const lastTrickId = prefix === 'trick' ? 'last-trick' : 'watch-last-trick';
    const handPrefix = prefix === 'trick' ? 'hand' : 'watch-hand';
    const trickArea = document.getElementById(trickAreaId);
    if (!trickArea) { if (onComplete) onComplete(); return; }

    _animatingTrick = prefix;

    const seatMap = { 0: 'n', 1: 'e', 2: 's', 3: 'w' };
    const seatDirMap = { 0: 'north', 1: 'east', 2: 'south', 3: 'west' };
    const faceClones = [];
    const backClones = [];
    const rects = [];

    for (let seat = 0; seat < 4; seat++) {
        const slotEl = document.getElementById(`${prefix}-${seatMap[seat]}`);
        const cardEl = slotEl ? slotEl.querySelector('.card') : null;
        if (cardEl) {
            const rect = cardEl.getBoundingClientRect();
            rects.push(rect);

            // Face clone (card image)
            const face = cardEl.cloneNode(true);
            face.style.position = 'absolute';
            face.style.left = rect.left + 'px';
            face.style.top = rect.top + 'px';
            face.style.width = rect.width + 'px';
            face.style.height = rect.height + 'px';
            face.style.margin = '0';
            faceClones.push(face);

            // Back clone (face-down card)
            const back = document.createElement('div');
            back.className = 'card face-down';
            back.style.position = 'absolute';
            back.style.left = rect.left + 'px';
            back.style.top = rect.top + 'px';
            back.style.width = rect.width + 'px';
            back.style.height = rect.height + 'px';
            back.style.margin = '0';
            backClones.push(back);
        }
    }

    if (faceClones.length === 0) {
        _animatingTrick = null;
        if (onComplete) onComplete();
        return;
    }

    // Create fixed overlay: backs behind, faces on top
    const overlay = document.createElement('div');
    overlay.className = 'trick-flush-overlay';
    for (const c of backClones) overlay.appendChild(c);
    for (const c of faceClones) overlay.appendChild(c);
    document.body.appendChild(overlay);

    // Clear original trick slots
    for (let seat = 0; seat < 4; seat++) {
        const slotEl = document.getElementById(`${prefix}-${seatMap[seat]}`);
        if (slotEl) slotEl.innerHTML = '';
    }

    // Calculate center of trick area
    const areaRect = trickArea.getBoundingClientRect();
    const centerX = areaRect.left + areaRect.width / 2;
    const centerY = areaRect.top + areaRect.height / 2;

    // Fly toward the winner's seat direction
    let targetX, targetY;
    if (winner !== undefined && winner >= 0 && winner < 4) {
        const winnerEl = document.getElementById(`${handPrefix}-${seatDirMap[winner]}`);
        if (winnerEl) {
            const wRect = winnerEl.getBoundingClientRect();
            targetX = wRect.left + wRect.width / 2;
            targetY = wRect.top + wRect.height / 2;
        }
    }
    if (targetX === undefined) {
        // Fallback: top-right of trick area
        targetX = areaRect.right - 40;
        targetY = areaRect.top + 20;
    }

    // Hide last-trick box during animation — revealed in onComplete callback
    const lastTrickEl = document.getElementById(lastTrickId);
    if (lastTrickEl) {
        lastTrickEl.classList.add('hidden');
        lastTrickEl.innerHTML = '';
    }

    const duration = 1600;
    let finished = 0;
    const totalAnims = faceClones.length * 2;
    const rotations = ['-3deg', '4deg', '-2deg', '5deg'];

    function onAnimFinish() {
        finished++;
        if (finished === totalAnims) {
            overlay.remove();
            _animatingTrick = null;
            if (onComplete) onComplete();
        }
    }

    for (let i = 0; i < faceClones.length; i++) {
        const rect = rects[i];
        const origX = rect.left;
        const origY = rect.top;
        const pileX = centerX - rect.width / 2;
        const pileY = centerY - rect.height / 2;
        const flyX = targetX - rect.width / 4;
        const flyY = targetY - rect.height / 4;
        const rot = rotations[i];

        // Face: slides to pile, fades out at flip point
        faceClones[i].animate([
            { left: origX+'px', top: origY+'px', transform: 'scale(1) rotate(0deg)', opacity: 1 },
            { left: pileX+'px', top: pileY+'px', transform: `scale(0.95) rotate(${rot})`, opacity: 1, offset: 0.30 },
            { left: pileX+'px', top: pileY+'px', transform: `scale(0.95) rotate(${rot})`, opacity: 0, offset: 0.35 },
            { left: flyX+'px', top: flyY+'px', transform: 'scale(0.5) rotate(0deg)', opacity: 0 },
        ], { duration, easing: 'ease-in-out', fill: 'forwards' }).onfinish = onAnimFinish;

        // Back: follows same path, fades in at flip point, flies to target
        backClones[i].animate([
            { left: origX+'px', top: origY+'px', transform: 'scale(1) rotate(0deg)', opacity: 0 },
            { left: pileX+'px', top: pileY+'px', transform: `scale(0.95) rotate(${rot})`, opacity: 0, offset: 0.30 },
            { left: pileX+'px', top: pileY+'px', transform: `scale(0.95) rotate(${rot})`, opacity: 1, offset: 0.35 },
            { left: flyX+'px', top: flyY+'px', transform: 'scale(0.5) rotate(0deg)', opacity: 0 },
        ], { duration, easing: 'ease-in-out', fill: 'forwards' }).onfinish = onAnimFinish;
    }
}

// Card rendering
const RANKS = ['7', '8', '9', 'V', 'D', 'R', '10', 'A'];
const SUITS = ['\u2660', '\u2665', '\u2666', '\u2663']; // spade heart diamond club

// Card point values per rank index: [7, 8, 9, J, Q, K, 10, A]
const PLAIN_POINTS = [0, 0, 0, 2, 3, 4, 10, 11];
const TRUMP_POINTS = [0, 0, 14, 20, 3, 4, 10, 11];
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

function cardToHtml(cardIdx, clickable = false, onClick = null, illegal = false, annotation = null) {
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

    if (annotation) {
        const badge = document.createElement('span');
        badge.className = `card-annotation ${annotation.cls || ''}`;
        badge.textContent = annotation.text;
        if (annotation.style) Object.assign(badge.style, annotation.style);
        el.appendChild(badge);
    }

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

function renderHand(container, cards, clickable = false, onClick = null, legalSet = null, trumpSuit = -1, annotations = null) {
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
        const ann = annotations ? annotations.get(c) : null;
        container.appendChild(cardToHtml(c, cardClickable, onClick, illegal, ann));
    }
}

function renderFaceDownHand(container, count) {
    // Skip re-render if the count of face-down cards hasn't changed
    const current = container.children.length;
    if (current === count && count > 0 && container.firstChild && container.firstChild.classList.contains('face-down')) {
        return;
    }
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

// CFN box: click to copy
function updateCfnBox(elementId, cfn) {
    const el = document.getElementById(elementId);
    if (!el) return;
    if (cfn) {
        el.textContent = cfn;
        el.classList.remove('hidden');
    } else {
        el.classList.add('hidden');
    }
}

function initCfnBox(elementId) {
    const el = document.getElementById(elementId);
    if (!el) return;
    el.addEventListener('click', () => {
        const text = el.textContent;
        if (!text) return;
        navigator.clipboard.writeText(text).then(() => {
            el.classList.add('copied');
            const prev = el.textContent;
            el.textContent = 'Copie !';
            setTimeout(() => {
                el.textContent = prev;
                el.classList.remove('copied');
            }, 1000);
        });
    });
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

// Connect on load
connect();
initCfnBox('play-cfn');
initCfnBox('watch-cfn');
