// Shared bid history chip rendering — deduplicated from 4 locations

import { SEAT_NAMES_FR, bidActionHtml } from './cards.js';

export function renderBidHistoryChips(container, bidHistory) {
    container.innerHTML = '';
    if (!bidHistory || bidHistory.length === 0) return;

    for (const bid of bidHistory) {
        const el = document.createElement('span');
        const team = bid.player % 2 === 0 ? 'team-ns' : 'team-ew';
        el.className = `watch-bid-entry ${team}`;
        el.innerHTML = `${SEAT_NAMES_FR[bid.player]} : ${bidActionHtml(bid.action)}`;
        container.appendChild(el);
    }
}
