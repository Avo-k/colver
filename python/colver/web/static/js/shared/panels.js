// Panneaux latéraux partagés — historique d'enchères et historique de plis.
//
// Ces deux rendus existaient en trois exemplaires quasi identiques : deux dans
// board.js (panneau + surcouche d'enchères) et un dans table.js. Les copies
// avaient déjà divergé — table.js testait `lead !== null` et board.js non, si
// bien qu'un `lead` nul y produisait `SEAT_L[null]` → « undefined » affiché.
// La version stricte est celle retenue ici.
//
// Le nom de siège passe par un callback : le plateau solo/salon affiche les
// pseudos des joueurs, Watch et Replay les noms de sièges.

import { cardTextHtml, SEAT_NAMES_FR, bidActionHtml } from './cards.js';
import { teamClass } from './seats.js';

const SEAT_INITIALS = ['N', 'E', 'S', 'O'];
const defaultPlayerName = (seat) => SEAT_NAMES_FR[seat];

/** Une suite de puces « Nord 100♣ », colorées par équipe. */
export function renderBidEntries(container, bidHistory, playerName = defaultPlayerName) {
    if (!container) return;
    container.innerHTML = '';
    if (!bidHistory) return;
    for (const bid of bidHistory) {
        const el = document.createElement('span');
        el.className = `watch-bid-entry ${teamClass(bid.player)}`;
        el.innerHTML = `${playerName(bid.player)} ${bidActionHtml(bid.action)}`;
        container.appendChild(el);
    }
}

/** Les plis terminés, cartes dans l'ordre de jeu à partir de l'entameur. */
export function renderTrickHistory(container, tricks, playerName = defaultPlayerName) {
    if (!container) return;
    container.innerHTML = '';
    if (!tricks || tricks.length === 0) return;

    for (let i = 0; i < tricks.length; i++) {
        const t = tricks[i];
        const row = document.createElement('div');
        row.className = `trick-history-row ${teamClass(t.winner)}`;

        const leadSeat = (t.lead !== undefined && t.lead !== null) ? t.lead : -1;

        let orderedCards;
        if (leadSeat >= 0) {
            const parts = [];
            for (let j = 0; j < 4; j++) {
                const c = t.cards[(leadSeat + j) % 4];
                if (c >= 0 && c < 32) parts.push(cardTextHtml(c));
            }
            orderedCards = parts.join(' ');
        } else {
            orderedCards = t.cards
                .map(c => (c >= 0 && c < 32) ? cardTextHtml(c) : '?')
                .join(' ');
        }

        const leadLabel = leadSeat >= 0 ? SEAT_INITIALS[leadSeat] : '?';
        row.innerHTML = `<span class="trick-num">#${i + 1}</span>` +
            `<span class="trick-lead-label">${leadLabel}</span>` +
            `<span class="trick-cards">${orderedCards}</span>` +
            `<span class="trick-winner">${playerName(t.winner)} +${t.points}</span>`;
        container.appendChild(row);
    }
}
