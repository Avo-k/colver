// Card rendering utilities — shared across all views

import * as SFX from '../sounds.js';

export const RANKS = ['7', '8', '9', 'V', 'D', 'R', '10', 'A'];
export const SUITS = ['\u2660', '\u2665', '\u2666', '\u2663'];
export const PLAIN_POINTS = [0, 0, 0, 2, 3, 4, 10, 11];
export const TRUMP_POINTS = [0, 0, 14, 20, 3, 4, 10, 11];
export const SUIT_NAMES_EN = ['spades', 'hearts', 'diamonds', 'clubs'];
export const RANK_NAMES_EN = ['7', '8', '9', 'jack', 'queen', 'king', '10', 'ace'];
export const SEAT_NAMES_FR = ['Nord', 'Est', 'Sud', 'Ouest'];

// Display order: ♠ ♥ ♣ ♦ (alternating black-red)
export const SUIT_DISPLAY_ORDER = [0, 1, 3, 2];

// Build per-hand suit sort keys that maximize color alternation.
// With all 4 suits: ♠ ♥ ♣ ♦. With fewer, reorders to avoid adjacent same-color.
export function suitSortKeys(cards) {
    const present = new Set();
    for (const c of cards) present.add(c >> 3);
    const blacks = [0, 3].filter(s => present.has(s)); // ♠ ♣
    const reds = [1, 2].filter(s => present.has(s));   // ♥ ♦
    const order = [];
    let bi = 0, ri = 0, pickBlack = blacks.length >= reds.length;
    while (bi < blacks.length || ri < reds.length) {
        if (pickBlack && bi < blacks.length) order.push(blacks[bi++]);
        else if (!pickBlack && ri < reds.length) order.push(reds[ri++]);
        else if (bi < blacks.length) order.push(blacks[bi++]);
        else order.push(reds[ri++]);
        pickBlack = !pickBlack;
    }
    const keys = [4, 4, 4, 4];
    for (let i = 0; i < order.length; i++) keys[order[i]] = i;
    return keys;
}

export function cardSuit(idx) { return idx >> 3; }
export function cardRank(idx) { return idx & 7; }

export function cardSvgPath(cardIdx) {
    const suit = cardSuit(cardIdx);
    const rank = cardRank(cardIdx);
    const suffix = (rank >= 3 && rank <= 5) ? '2' : '';
    return `cards/${RANK_NAMES_EN[rank]}_of_${SUIT_NAMES_EN[suit]}${suffix}.svg`;
}

export function cardToHtml(cardIdx, clickable = false, onClick = null, illegal = false, annotation = null) {
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

export function faceDownCard() {
    const el = document.createElement('div');
    el.className = 'card face-down';
    return el;
}

// Sort keys for display order within a suit
const PLAIN_ORDER = [7, 6, 5, 4, 3, 2, 1, 0];
const TRUMP_ORDER = [7, 6, 1, 0, 5, 4, 3, 2];

export function renderHand(container, cards, clickable = false, onClick = null, legalSet = null, trumpSuit = -1, annotations = null) {
    container.innerHTML = '';
    const sortKeys = suitSortKeys(cards);
    const sorted = [...cards].sort((a, b) => {
        const suitA = cardSuit(a), suitB = cardSuit(b);
        if (suitA !== suitB) return sortKeys[suitA] - sortKeys[suitB];
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

export function renderFaceDownHand(container, count) {
    const current = container.children.length;
    if (current === count && count > 0 && container.firstChild && container.firstChild.classList.contains('face-down')) {
        return;
    }
    container.innerHTML = '';
    for (let i = 0; i < count; i++) {
        container.appendChild(faceDownCard());
    }
}

export function renderTrick(prefix, trick) {
    const seatMap = { 0: 'n', 1: 'e', 2: 's', 3: 'w' };
    for (let seat = 0; seat < 4; seat++) {
        const el = document.getElementById(`${prefix}-${seatMap[seat]}`);
        if (!el) continue;
        el.innerHTML = '';
        const c = trick[seat];
        if (c >= 0 && c < 32) {
            el.appendChild(cardToHtml(c));
        }
    }
}

export function renderLastTrick(container, trick, trickWinner, trickPoints, humanSeat) {
    container.innerHTML = '';
    if (!trick || trick.every(c => c < 0 || c >= 32)) {
        container.classList.add('hidden');
        return;
    }
    container.classList.remove('hidden');
    const isPartnerWin = trickWinner !== null && (trickWinner % 2) === (humanSeat % 2);
    const teamClass = trickWinner !== null ? (isPartnerWin ? 'team-partner' : 'team-opponent') : '';
    const label = document.createElement('div');
    label.className = 'last-trick-label ' + teamClass;
    const pts = trickPoints || 0;
    label.textContent = trickWinner !== null
        ? `${SEAT_NAMES_FR[trickWinner]} +${pts}`
        : `Pli +${pts}`;
    container.appendChild(label);
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

export function contractStr(contract) {
    if (!contract || Object.keys(contract).length === 0) return '';
    const val = contract.value;
    const suit = SUITS[contract.trump];
    const team = contract.team === 0 ? 'NS' : 'EO';
    const coinche = contract.coinche === 1 ? ' x' : contract.coinche === 2 ? ' xx' : '';
    return `${val}${suit} par ${team}${coinche}`;
}

export function actionName(action, phase) {
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

export function showBeloteAnnouncement(trickAreaId, text) {
    SFX.belote();
    const area = document.getElementById(trickAreaId);
    if (!area) return;
    const el = document.createElement('div');
    el.className = 'belote-announcement';
    el.textContent = text;
    area.style.position = 'relative';
    area.appendChild(el);
    setTimeout(() => el.remove(), 1800);
}

export function renderBeloteBadge(scoreElId, beloteVal) {
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

// Trick flush animation state (module-level, shared)
export let _prevTrick = {};
export let _animatingTrick = null;

export function setAnimatingTrick(val) { _animatingTrick = val; }

export function detectTrickCompletion(prefix, newTrick) {
    const prev = _prevTrick[prefix];
    _prevTrick[prefix] = newTrick ? [...newTrick] : [];
    if (!prev) return null;
    const prevFull = prev.filter(c => c >= 0 && c < 32).length === 4;
    const newFull = newTrick ? newTrick.filter(c => c >= 0 && c < 32).length : 0;
    if (prevFull && newFull < 4) {
        return prev;
    }
    return null;
}

export function animateTrickFlush(prefix, onComplete, winner) {
    let trickAreaId, lastTrickId, handPrefix;
    if (prefix === 'trick') {
        trickAreaId = 'trick-area';
        lastTrickId = 'last-trick';
        handPrefix = 'hand';
    } else {
        trickAreaId = prefix + '-area';
        const tabPrefix = prefix.replace('-trick', '');
        lastTrickId = tabPrefix + '-last-trick';
        handPrefix = tabPrefix + '-hand';
    }
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

            const face = cardEl.cloneNode(true);
            face.style.position = 'absolute';
            face.style.left = rect.left + 'px';
            face.style.top = rect.top + 'px';
            face.style.width = rect.width + 'px';
            face.style.height = rect.height + 'px';
            face.style.margin = '0';
            faceClones.push(face);

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

    SFX.trickWon();

    const overlay = document.createElement('div');
    overlay.className = 'trick-flush-overlay';
    for (const c of backClones) overlay.appendChild(c);
    for (const c of faceClones) overlay.appendChild(c);
    document.body.appendChild(overlay);

    for (let seat = 0; seat < 4; seat++) {
        const slotEl = document.getElementById(`${prefix}-${seatMap[seat]}`);
        if (slotEl) slotEl.innerHTML = '';
    }

    const areaRect = trickArea.getBoundingClientRect();
    const centerX = areaRect.left + areaRect.width / 2;
    const centerY = areaRect.top + areaRect.height / 2;

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
        targetX = areaRect.right - 40;
        targetY = areaRect.top + 20;
    }

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

        faceClones[i].animate([
            { left: origX+'px', top: origY+'px', transform: 'scale(1) rotate(0deg)', opacity: 1 },
            { left: pileX+'px', top: pileY+'px', transform: `scale(0.95) rotate(${rot})`, opacity: 1, offset: 0.30 },
            { left: pileX+'px', top: pileY+'px', transform: `scale(0.95) rotate(${rot})`, opacity: 0, offset: 0.35 },
            { left: flyX+'px', top: flyY+'px', transform: 'scale(0.5) rotate(0deg)', opacity: 0 },
        ], { duration, easing: 'ease-in-out', fill: 'forwards' }).onfinish = onAnimFinish;

        backClones[i].animate([
            { left: origX+'px', top: origY+'px', transform: 'scale(1) rotate(0deg)', opacity: 0 },
            { left: pileX+'px', top: pileY+'px', transform: `scale(0.95) rotate(${rot})`, opacity: 0, offset: 0.30 },
            { left: pileX+'px', top: pileY+'px', transform: `scale(0.95) rotate(${rot})`, opacity: 1, offset: 0.35 },
            { left: flyX+'px', top: flyY+'px', transform: 'scale(0.5) rotate(0deg)', opacity: 0 },
        ], { duration, easing: 'ease-in-out', fill: 'forwards' }).onfinish = onAnimFinish;
    }
}

export function encodeBidAction(value, suitIdx) {
    const values = [80,90,100,110,120,130,140,150,160];
    if (value === 250) return 37 + suitIdx;
    const valIdx = values.indexOf(value);
    if (valIdx < 0) return -1;
    return valIdx * 4 + suitIdx + 1;
}
