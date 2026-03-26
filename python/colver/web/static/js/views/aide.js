// Aide-mémoire — visual reference sheet for Belote Contrée

import { cardSvgPath } from '../shared/cards.js';

const HEART = 1;  // Red suit for trump examples
const SPADE = 0;  // Black suit for plain examples

const TRUMP_CARDS = [
    { rank: 3, pts: 20, promoted: true  },  // Valet
    { rank: 2, pts: 14, promoted: true  },  // 9
    { rank: 7, pts: 11, promoted: false },  // As
    { rank: 6, pts: 10, promoted: false },  // 10
    { rank: 5, pts: 4,  promoted: false },  // Roi
    { rank: 4, pts: 3,  promoted: false },  // Dame
    { rank: 1, pts: 0,  promoted: false },  // 8
    { rank: 0, pts: 0,  promoted: false },  // 7
];

const PLAIN_CARDS = [
    { rank: 7, pts: 11 },  // As
    { rank: 6, pts: 10 },  // 10
    { rank: 5, pts: 4  },  // Roi
    { rank: 4, pts: 3  },  // Dame
    { rank: 3, pts: 2  },  // Valet
    { rank: 2, pts: 0  },  // 9
    { rank: 1, pts: 0  },  // 8
    { rank: 0, pts: 0  },  // 7
];

function ptsClass(pts) {
    if (pts >= 14) return 'aide-pts-gold';
    if (pts >= 10) return 'aide-pts-bright';
    if (pts > 0) return 'aide-pts-dim';
    return 'aide-pts-zero';
}

function buildCardRow(container, cards, suit) {
    for (const card of cards) {
        const cardIdx = suit * 8 + card.rank;
        const item = document.createElement('div');
        item.className = 'aide-card-item' + (card.promoted ? ' aide-promoted' : '');

        const img = document.createElement('img');
        img.src = cardSvgPath(cardIdx);
        img.className = 'aide-card-img';
        img.draggable = false;
        item.appendChild(img);

        const badge = document.createElement('div');
        badge.className = `aide-pts ${ptsClass(card.pts)}`;
        badge.textContent = card.pts;
        item.appendChild(badge);

        container.appendChild(item);
    }
}

const TEMPLATE = `
<div id="aide-content">
    <div class="aide-header">
        <h2>Aide-M\u00e9moire</h2>
        <p class="aide-subtitle">Ordre de force et valeur des cartes en Belote Contr\u00e9e</p>
    </div>

    <div class="aide-sections">
        <div class="aide-section">
            <div class="aide-section-head">
                <h3><span class="suit-red">\u2665</span> Atout</h3>
                <span class="aide-section-total">62 pts / couleur</span>
            </div>
            <p class="aide-hint">Du plus fort au plus faible \u2014 le Valet et le 9 deviennent les plus forts</p>
            <div class="aide-card-row" id="aide-trump-row"></div>
        </div>

        <div class="aide-section">
            <div class="aide-section-head">
                <h3><span class="suit-black">\u2660</span> Non-Atout</h3>
                <span class="aide-section-total">30 pts / couleur</span>
            </div>
            <p class="aide-hint">Du plus fort au plus faible</p>
            <div class="aide-card-row" id="aide-plain-row"></div>
        </div>

        <div class="aide-section aide-totals-section">
            <h3>Points de la donne</h3>
            <div class="aide-totals-grid">
                <div class="aide-stat">
                    <div class="aide-stat-val">152</div>
                    <div class="aide-stat-label">Points carte</div>
                    <div class="aide-stat-detail">62 + 3\u00d730</div>
                </div>
                <div class="aide-stat">
                    <div class="aide-stat-val aide-stat-plus">+10</div>
                    <div class="aide-stat-label">Dix de der</div>
                    <div class="aide-stat-detail">Dernier pli</div>
                </div>
                <div class="aide-stat aide-stat-highlight">
                    <div class="aide-stat-val">162</div>
                    <div class="aide-stat-label">Total normal</div>
                </div>
                <div class="aide-stat">
                    <div class="aide-stat-val aide-stat-plus">+100</div>
                    <div class="aide-stat-label">Capot</div>
                    <div class="aide-stat-detail">8 plis gagn\u00e9s</div>
                </div>
                <div class="aide-stat aide-stat-highlight">
                    <div class="aide-stat-val">252</div>
                    <div class="aide-stat-label">Total capot</div>
                </div>
                <div class="aide-stat">
                    <div class="aide-stat-val aide-stat-plus">+20</div>
                    <div class="aide-stat-label">Belote</div>
                    <div class="aide-stat-detail">Roi + Dame d'atout</div>
                </div>
            </div>
        </div>

        <div class="aide-section aide-bids-section">
            <h3>Ench\u00e8res</h3>
            <div class="aide-bids-row">
                <span class="aide-bid">80</span>
                <span class="aide-bid">90</span>
                <span class="aide-bid">100</span>
                <span class="aide-bid">110</span>
                <span class="aide-bid">120</span>
                <span class="aide-bid">130</span>
                <span class="aide-bid">140</span>
                <span class="aide-bid">150</span>
                <span class="aide-bid">160</span>
                <span class="aide-bid aide-bid-capot">Capot</span>
            </div>
            <div class="aide-bids-info">
                <span class="aide-bid-tag">Coinche \u00d72</span>
                <span class="aide-bid-tag">Surcoinche \u00d74</span>
                <span class="aide-bid-sep">\u2014</span>
                <span class="aide-bid-match">Partie en 2000 pts</span>
            </div>
        </div>
    </div>
</div>
`;

export function mount(container) {
    container.innerHTML = TEMPLATE;
    buildCardRow(document.getElementById('aide-trump-row'), TRUMP_CARDS, HEART);
    buildCardRow(document.getElementById('aide-plain-row'), PLAIN_CARDS, SPADE);
}

export function unmount() {}
