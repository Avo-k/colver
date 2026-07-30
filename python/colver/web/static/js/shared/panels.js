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

import { cardChipHtml, SEAT_NAMES_FR, bidChipHtml } from './cards.js';
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
        el.innerHTML = `${playerName(bid.player)} ${bidChipHtml(bid.action)}`;
        container.appendChild(el);
    }
}

/**
 * L'enchère lue comme à la table : une colonne par joueur dans l'ordre de
 * parole, une ligne par tour.
 *
 * La suite de puces (`renderBidEntries`) dit ce qui a été annoncé, mais pas
 * qui a parlé au-dessus de qui — c'est pourtant toute la question quand on
 * relit une enchère au milieu du jeu (« mon partenaire a-t-il soutenu, ou
 * juste passé après leur 110 ? »). Les colonnes partent du premier parleur,
 * donc la lecture verticale suit les tours de parole.
 */
export function renderAuctionTable(container, bidHistory, playerName = defaultPlayerName, mySeat = null) {
    if (!container) return;
    container.innerHTML = '';
    if (!bidHistory || bidHistory.length === 0) {
        const empty = document.createElement('div');
        empty.className = 'auction-empty';
        empty.textContent = 'Aucune annonce';
        container.appendChild(empty);
        return;
    }

    const first = bidHistory[0].player;
    const order = [0, 1, 2, 3].map(i => (first + i) % 4);

    for (const seat of order) {
        const head = document.createElement('div');
        head.className = `auction-col-head ${teamClass(seat)}`;
        if (seat === mySeat) head.classList.add('is-me');
        head.textContent = playerName(seat);
        container.appendChild(head);
    }

    // Une annonce dont le siège ne suit pas le tour attendu ne peut pas
    // arriver (le serveur les émet dans l'ordre de jeu), mais si ça arrivait
    // on remplirait des cases vides plutôt que de décaler toute la grille.
    let slot = 0;
    for (const bid of bidHistory) {
        const target = order.indexOf(bid.player);
        while (slot % 4 !== target) {
            container.appendChild(emptyAuctionCell());
            slot++;
        }
        const cell = document.createElement('div');
        cell.className = `auction-cell ${teamClass(bid.player)}`;
        cell.innerHTML = auctionCellHtml(bid.action);
        container.appendChild(cell);
        slot++;
    }
}

function emptyAuctionCell() {
    const el = document.createElement('div');
    el.className = 'auction-cell';
    return el;
}

/** Passe et coinche en texte, les vraies annonces en pastille : une colonne de
 *  pastilles claires pour trois passes se lit comme trois annonces. */
function auctionCellHtml(action) {
    if (action === 0) return '<span class="auction-pass">Passe</span>';
    if (action === 41) return '<span class="auction-coinche">Coinche</span>';
    if (action === 42) return '<span class="auction-coinche">Surcoinche</span>';
    return bidChipHtml(action);
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
                if (c >= 0 && c < 32) parts.push(cardChipHtml(c));
            }
            orderedCards = parts.join(' ');
        } else {
            orderedCards = t.cards
                .map(c => (c >= 0 && c < 32) ? cardChipHtml(c) : '?')
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
