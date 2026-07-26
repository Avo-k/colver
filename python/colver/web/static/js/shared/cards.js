// Card rendering utilities — shared across all views

import * as SFX from '../sounds.js';

import {
    SUIT_GLYPHS, SUIT_NAMES_EN, SUIT_NAMES_FR, SUIT_DISPLAY_ORDER, SUIT_IS_RED,
    suitHtml, suitClass,
} from './suits.js';
import { TEAM_NAMES_FR, TEAM_NAMES_REL, teamName, teamNameMid } from './seats.js';

export const RANKS = ['7', '8', '9', 'V', 'D', 'R', '10', 'A'];
export const PLAIN_POINTS = [0, 0, 0, 2, 3, 4, 10, 11];
export const TRUMP_POINTS = [0, 0, 14, 20, 3, 4, 10, 11];
export const RANK_NAMES_EN = ['7', '8', '9', 'jack', 'queen', 'king', '10', 'ace'];
export const SEAT_NAMES_FR = ['Nord', 'Est', 'Sud', 'Ouest'];

// Ré-exports : les vues importent historiquement ces symboles depuis cards.js.
// `SUITS` est le glyphe NU — réservé au texte pur (alt, title, textContent).
// Pour tout rendu HTML, passer par suitHtml() ou cardChipHtml().
export const SUITS = SUIT_GLYPHS;
export { SUIT_NAMES_EN, SUIT_DISPLAY_ORDER, SUIT_IS_RED, suitHtml, suitClass };
export { TEAM_NAMES_FR, TEAM_NAMES_REL, teamName, teamNameMid };

/**
 * « A♥ » sur une pastille claire — rang en noir, glyphe rouge ou noir, comme
 * une vraie carte. C'est le rendu par défaut partout où on DÉSIGNE une carte
 * (le coup qu'un bot aurait joué, une barre de score par carte, un pli rejoué)
 * : sur le feutre sombre, le texte nu se lit comme une étiquette, la pastille
 * se lit comme une carte. `on-light` bascule les glyphes sur les tokens de
 * surface claire — aucune couleur n'est écrite en dur ici (cf. tokens.css).
 */
export function cardChipHtml(cardIdx) {
    return `<span class="card-chip on-light">` +
        `<span class="card-chip__rank">${RANKS[cardRank(cardIdx)]}</span>` +
        `${suitHtml(cardSuit(cardIdx))}</span>`;
}

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

// Two-char card codes for URLs (CFN-style): rank char + suit letter, e.g. "7S", "KH", "TS".
const CODE_RANKS = ['7', '8', '9', 'J', 'Q', 'K', 'T', 'A'];
const CODE_SUITS = ['S', 'H', 'D', 'C'];

export function cardCode(idx) {
    return CODE_RANKS[cardRank(idx)] + CODE_SUITS[cardSuit(idx)];
}

// Parse a hand URL token: either a two-char code ("KH") or a legacy integer index.
// Returns the card index 0-31, or -1 if invalid.
export function parseCardToken(tok) {
    tok = tok.trim().toUpperCase();
    if (/^\d+$/.test(tok)) {
        const n = parseInt(tok, 10);
        return n >= 0 && n < 32 ? n : -1;
    }
    if (tok.length !== 2) return -1;
    const rank = CODE_RANKS.indexOf(tok[0]);
    const suit = CODE_SUITS.indexOf(tok[1]);
    if (rank < 0 || suit < 0) return -1;
    return suit * 8 + rank;
}

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
        // Une carte jouable est un contrôle : atteignable au Tab, actionnable
        // à Entrée ou Espace, et annoncée par un lecteur d'écran. Elle n'était
        // qu'un <div> cliquable — le jeu au clavier était impossible.
        el.tabIndex = 0;
        el.setAttribute('role', 'button');
        el.setAttribute('aria-label', `Jouer ${RANKS[cardRank(cardIdx)]} de ${SUIT_NAMES_FR[cardSuit(cardIdx)]}`);
        el.addEventListener('click', () => onClick(cardIdx));
        el.addEventListener('keydown', (e) => {
            if (e.key !== 'Enter' && e.key !== ' ') return;
            e.preventDefault();
            onClick(cardIdx);
        });
    } else if (illegal) {
        el.setAttribute('aria-disabled', 'true');
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

// La main se recompose à chaque rendu : les cartes restantes se regroupent et
// se recentrent. La stabilité du plateau vient du conteneur, dimensionné en
// CSS pour 8 cartes quoi qu'il arrive (cf. .hand dans cards.css) — rien ici
// n'a besoin de connaître ni de mémoriser des positions.
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

// Shared hand ordering (suit alternation + rank order, trump-aware).
export function sortHand(cards, trumpSuit = -1) {
    const sortKeys = suitSortKeys(cards);
    return [...cards].sort((a, b) => {
        const suitA = cardSuit(a), suitB = cardSuit(b);
        if (suitA !== suitB) return sortKeys[suitA] - sortKeys[suitB];
        const orderA = suitA === trumpSuit ? TRUMP_ORDER : PLAIN_ORDER;
        const orderB = suitB === trumpSuit ? TRUMP_ORDER : PLAIN_ORDER;
        return orderA[cardRank(a)] - orderB[cardRank(b)];
    });
}

// Compact "corner index" mini card — pure CSS text (no image), legible at small sizes.
// Used where cards are shown small and spread (e.g. annonces sim hands).
export function miniCardIndex(cardIdx, w = 34) {
    const suit = cardSuit(cardIdx), rank = cardRank(cardIdx);
    const el = document.createElement('div');
    // Face claire : `on-light` bascule le glyphe sur les variantes sombres des
    // tokens — c'est le seul endroit de l'app où ♠/♣ se rendent en vrai noir.
    let cls = 'mini-card-index on-light';
    if (suit === 1 || suit === 2) cls += ' red';
    if (RANKS[rank].length > 1) cls += ' ten';
    el.className = cls;
    el.style.width = w + 'px';
    el.style.height = Math.round(w * 1.4) + 'px';
    el.style.fontSize = w + 'px';
    el.dataset.card = cardIdx;
    el.innerHTML = `<span class="mc-corner"><span class="mc-rank">${RANKS[rank]}</span>${suitHtml(suit, 'mc-suit')}</span>`;
    return el;
}

// Render a spread hand of corner-index mini cards (no overlap).
export function renderHandMini(container, cards, w = 34, trumpSuit = -1) {
    container.innerHTML = '';
    for (const c of sortHand(cards, trumpSuit)) {
        container.appendChild(miniCardIndex(c, w));
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

/**
 * Contrat en HTML, sur une ligne : la pastille du contrat suivie du camp
 * preneur. `relative` : nommer les camps depuis la chaise du joueur
 * (« par nous ») plutôt que de l'extérieur (« par Nord-Sud »).
 *
 * Le contrat lui-même passe par contractChipHtml() — c'est la même pastille
 * claire que dans le bandeau de la table de jeu et que dans l'historique des
 * annonces. Seul le « par X » reste du texte, dans la couleur ambiante.
 */
export function contractStr(contract, relative = false) {
    if (!contract || !contract.value) return '';
    const team = teamNameMid(contract.team, relative);
    return `${contractChipHtml(contract)} par ${team}`;
}

/** Contrat en texte pur (title, alt, textContent). */
export function contractText(contract, relative = false) {
    if (!contract || Object.keys(contract).length === 0) return '';
    const team = teamNameMid(contract.team, relative);
    const coinche = contract.coinche === 1 ? ' x' : contract.coinche === 2 ? ' xx' : '';
    return `${contract.value}${SUITS[contract.trump]} par ${team}${coinche}`;
}

/**
 * L'annonce sur une pastille claire — « 100 ♣ », « Capot ♠ », « Passe » :
 * valeur et mot en sombre, glyphe rouge ou noir, comme sur une carte.
 *
 * ORIGINE UNIQUE du rendu d'une annonce : tout ce qui affiche une annonce dans
 * l'app passe par ici, et l'apparence tient dans `.bid-chip` (cards.css) — pour
 * changer la tête des annonces partout, il n'y a que ces deux endroits à
 * toucher. `on-light` bascule les glyphes sur les tokens de surface claire, donc
 * aucune couleur n'est écrite en dur (cf. tokens.css). Le pendant côté cartes
 * est cardChipHtml(). Pour du texte nu (title, alt), voir actionName(a, 0).
 */
export function bidChipHtml(action) {
    return `<span class="bid-chip on-light">${bidChipBody(action)}</span>`;
}

/**
 * Le contrat sur la même pastille que les annonces — « 140 ♠ », « Capot ♥ »,
 * suivi du multiplicateur quand le contrat est contré (×2) ou surcontré (×3).
 *
 * Le contrat *est* une annonce, celle qui a tenu : il se lit donc comme les
 * autres, sur fond clair, et pas en accent sur le fond sombre du bandeau — où
 * la valeur prenait la couleur d'accent et le glyphe celle des surfaces
 * sombres, deux teintes sans rapport collées l'une à l'autre.
 * Le camp preneur n'entre pas dans la pastille : il est écrit à côté.
 */
export function contractChipHtml(contract, extraClass = '') {
    if (!contract || !contract.value) return '';
    const val = contract.value === 250 ? 'Capot' : contract.value;
    const mult = contract.coinche === 1 ? '<span class="bid-chip__mult">&times;2</span>'
               : contract.coinche === 2 ? '<span class="bid-chip__mult">&times;3</span>'
               : '';
    const cls = `bid-chip on-light${extraClass ? ' ' + extraClass : ''}`;
    return `<span class="${cls}"><span class="bid-chip__val">${val}</span>${suitHtml(contract.trump)}${mult}</span>`;
}

function bidChipBody(action) {
    if (action === 0) return 'Passe';
    if (action === 41) return 'Coinche';
    if (action === 42) return 'Surcoinche';
    if (action >= 1 && action <= 36) {
        const idx = action - 1;
        const values = [80, 90, 100, 110, 120, 130, 140, 150, 160];
        return `<span class="bid-chip__val">${values[Math.floor(idx / 4)]}</span>${suitHtml(idx % 4)}`;
    }
    if (action >= 37 && action <= 40) {
        return `<span class="bid-chip__val">Capot</span>${suitHtml(action - 37)}`;
    }
    return `?${action}`;
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

export const TRICK_FLUSH_DURATION = 1600;

export function animateTrickFlush(prefix, onComplete, winner, durationMs) {
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

    const duration = durationMs || TRICK_FLUSH_DURATION;
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
