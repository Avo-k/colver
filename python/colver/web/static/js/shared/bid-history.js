// Shared bid history chip rendering — deduplicated from 4 locations

import { actionName } from './cards.js';

export function renderBidHistoryChips(container, bidHistory) {
    container.innerHTML = '';
    if (!bidHistory || bidHistory.length === 0) return;

    for (const bid of bidHistory) {
        const el = document.createElement('span');
        const team = bid.player % 2 === 0 ? 'team-ns' : 'team-ew';
        el.className = `watch-bid-entry ${team}`;
        const seatLetter = ['N', 'E', 'S', 'O'][bid.player];
        const name = bid.name || actionName(bid.action, 0);
        el.textContent = `${seatLetter}:${name}`;
        container.appendChild(el);
    }
}
