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

import { cardChipHtml, bidChipHtml } from './cards.js';
import { teamClass, SEAT_NAMES_FR, SEAT_INITIALS_FR } from './seats.js';

const defaultPlayerName = (seat) => SEAT_NAMES_FR[seat];

/**
 * L'enchère lue comme à la table : une colonne par joueur dans l'ordre de
 * parole, une ligne par tour. **Le seul rendu d'une enchère complète du site**
 * — partout où les annonces se donnaient à la suite (panneau d'enchère, fin de
 * donne, Regarder, Rejouer, `/analyse/jeu`, les deux pages de problèmes), elles
 * défilaient en puces qui passaient à la ligne au petit bonheur de la largeur.
 * Une puce dit ce qui a été annoncé, mais pas qui a parlé au-dessus de qui —
 * c'est pourtant toute la question d'une enchère (« mon partenaire a-t-il
 * soutenu, ou juste passé après leur 110 ? »). Les colonnes partent du premier
 * parleur, donc la lecture verticale suit les tours de parole.
 *
 * opts :
 *   playerName(seat) — libellé de colonne (pseudo à une table, nom de siège ailleurs)
 *   mySeat           — souligne la colonne du lecteur
 *   heads            — 'names' | 'initials' (colonne étroite) | 'none'
 *   compact          — variante resserrée, pour une colonne latérale
 *   pending          — siège dont on attend l'annonce : une case « ? » de plus
 *   emptyText        — quand il n'y a ni annonce ni `pending`
 *   decorate(cell, bid, i) — Rejouer y accroche ses clics et ses liserés
 */
export function renderAuctionTable(container, bidHistory, opts = {}) {
    if (!container) return;
    const {
        playerName = defaultPlayerName,
        mySeat = null,
        heads = 'names',
        compact = false,
        pending = null,
        emptyText = 'Aucune annonce',
        decorate = null,
    } = opts;

    container.classList.add('auction-grid');
    container.classList.toggle('auction-grid--compact', !!compact);
    container.innerHTML = '';

    const bids = bidHistory || [];
    // Sans annonce, c'est le siège attendu qui ancre la première colonne : au
    // premier tour de parole il *est* le premier parleur.
    const first = bids.length ? bids[0].player : pending;
    if (first === null || first === undefined) {
        const empty = document.createElement('div');
        empty.className = 'auction-empty';
        empty.textContent = emptyText;
        container.appendChild(empty);
        return;
    }

    const order = [0, 1, 2, 3].map(i => (first + i) % 4);

    if (heads !== 'none') {
        for (const seat of order) {
            const head = document.createElement('div');
            head.className = `auction-col-head ${teamClass(seat)}`;
            if (seat === mySeat) head.classList.add('is-me');
            head.textContent = heads === 'initials' ? SEAT_INITIALS_FR[seat] : playerName(seat);
            head.title = playerName(seat);
            container.appendChild(head);
        }
    }

    // Une annonce dont le siège ne suit pas le tour attendu ne peut pas
    // arriver (le serveur les émet dans l'ordre de jeu), mais si ça arrivait
    // on remplirait des cases vides plutôt que de décaler toute la grille.
    let slot = 0;
    const align = (seat) => {
        const target = order.indexOf(seat);
        while (slot % 4 !== target) {
            container.appendChild(emptyAuctionCell());
            slot++;
        }
    };

    for (let i = 0; i < bids.length; i++) {
        const bid = bids[i];
        align(bid.player);
        const cell = document.createElement('div');
        cell.className = `auction-cell ${teamClass(bid.player)}`;
        cell.innerHTML = auctionCellHtml(bid.action);
        cell.title = playerName(bid.player);
        if (decorate) decorate(cell, bid, i);
        container.appendChild(cell);
        slot++;
    }

    if (pending !== null && pending !== undefined) {
        align(pending);
        const cell = document.createElement('div');
        cell.className = `auction-cell auction-pending ${teamClass(pending)}`;
        cell.textContent = '?';
        cell.title = `${playerName(pending)} doit annoncer`;
        container.appendChild(cell);
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

        const leadLabel = leadSeat >= 0 ? SEAT_INITIALS_FR[leadSeat] : '?';
        row.innerHTML = `<span class="trick-num">#${i + 1}</span>` +
            `<span class="trick-lead-label">${leadLabel}</span>` +
            `<span class="trick-cards">${orderedCards}</span>` +
            `<span class="trick-winner">${playerName(t.winner)} +${t.points}</span>`;
        container.appendChild(row);
    }
}
