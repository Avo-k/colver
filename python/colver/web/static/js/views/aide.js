// Aide-mémoire — visual reference sheet for Belote Contrée

import { cardSvgPath } from '../shared/cards.js';

const HEART = 1;  // Red suit for trump examples
const SPADE = 0;  // Black suit for plain examples

// 10 rank-aligned slots. Promoted trump cards (Valet, 9) occupy slots 0-1;
// non-atout Valet/9 stay in their native rank slots (6-7). Placeholders fill
// the empty cells so rows align across both columns.
const ALIGNED_SLOTS = [
    { trump: { rank: 3, pts: 20, promoted: true }, plain: null },                 // Valet (atout promu)
    { trump: { rank: 2, pts: 14, promoted: true }, plain: null },                 // 9 (atout promu)
    { trump: { rank: 7, pts: 11 },                 plain: { rank: 7, pts: 11 } }, // As
    { trump: { rank: 6, pts: 10 },                 plain: { rank: 6, pts: 10 } }, // 10
    { trump: { rank: 5, pts: 4  },                 plain: { rank: 5, pts: 4  } }, // Roi
    { trump: { rank: 4, pts: 3  },                 plain: { rank: 4, pts: 3  } }, // Dame
    { trump: null,                                 plain: { rank: 3, pts: 2  } }, // Valet (non-atout)
    { trump: null,                                 plain: { rank: 2, pts: 0  } }, // 9 (non-atout)
    { trump: { rank: 1, pts: 0  },                 plain: { rank: 1, pts: 0  } }, // 8
    { trump: { rank: 0, pts: 0  },                 plain: { rank: 0, pts: 0  } }, // 7
];

function ptsClass(pts) {
    if (pts >= 14) return 'aide-pts-gold';
    if (pts >= 10) return 'aide-pts-bright';
    if (pts > 0) return 'aide-pts-dim';
    return 'aide-pts-zero';
}

function buildColumn(container, slots, side, suit) {
    for (const slot of slots) {
        const card = slot[side];
        const row = document.createElement('div');
        row.className = 'aide-col-row';

        if (!card) {
            row.classList.add('aide-col-empty');
            const ph = document.createElement('div');
            ph.className = 'aide-col-placeholder';
            row.appendChild(ph);

            const dash = document.createElement('div');
            dash.className = 'aide-col-pts aide-pts-zero';
            dash.textContent = '\u2013';
            row.appendChild(dash);
        } else {
            if (card.promoted) row.classList.add('aide-promoted');
            const cardIdx = suit * 8 + card.rank;
            const img = document.createElement('img');
            img.src = cardSvgPath(cardIdx);
            img.className = 'aide-col-card-img';
            img.draggable = false;
            row.appendChild(img);

            const badge = document.createElement('div');
            badge.className = `aide-col-pts ${ptsClass(card.pts)}`;
            badge.textContent = card.pts;
            row.appendChild(badge);
        }
        container.appendChild(row);
    }
}

const TEMPLATE = `
<div id="aide-content">
    <div class="aide-header">
        <h2>Aide-M\u00e9moire</h2>
        <p class="aide-subtitle">Ordre de force et valeur des cartes en Belote Contr\u00e9e</p>
    </div>

    <div class="aide-sections">
        <div class="aide-columns">
            <div class="aide-section aide-col">
                <div class="aide-col-head">
                    <h3><span class="suit-red">\u2665</span> Atout</h3>
                    <span class="aide-col-total">62 pts</span>
                </div>
                <div class="aide-col-cards" id="aide-trump-col"></div>
            </div>

            <div class="aide-section aide-col">
                <div class="aide-col-head">
                    <h3><span class="suit-black">\u2660</span> Non-Atout</h3>
                    <span class="aide-col-total">30 pts</span>
                </div>
                <div class="aide-col-cards" id="aide-plain-col"></div>
            </div>
        </div>

        <div class="aide-section aide-totals-section">
            <h3>Points de la donne</h3>
            <div class="aide-formulas">
                <div class="aide-formula">
                    <span class="aide-formula-label">Normal</span>
                    <span class="aide-formula-eq">
                        <span class="aide-term"><span class="aide-num">152</span><span class="aide-sub">cartes</span></span>
                        <span class="aide-op">+</span>
                        <span class="aide-term"><span class="aide-num">10</span><span class="aide-sub">dix de der</span></span>
                        <span class="aide-op">+</span>
                        <span class="aide-term aide-term-opt"><span class="aide-num">20</span><span class="aide-sub">belote</span></span>
                        <span class="aide-op">=</span>
                        <span class="aide-total">162<span class="aide-total-alt">ou 182</span></span>
                    </span>
                </div>
                <div class="aide-formula aide-formula-capot">
                    <span class="aide-formula-label">Capot</span>
                    <span class="aide-formula-eq">
                        <span class="aide-term"><span class="aide-num">152</span><span class="aide-sub">cartes</span></span>
                        <span class="aide-op">+</span>
                        <span class="aide-term"><span class="aide-num">100</span><span class="aide-sub">dix de der</span></span>
                        <span class="aide-op">+</span>
                        <span class="aide-term aide-term-opt"><span class="aide-num">20</span><span class="aide-sub">belote</span></span>
                        <span class="aide-op">=</span>
                        <span class="aide-total">252<span class="aide-total-alt">ou 272</span></span>
                    </span>
                </div>
            </div>
            <p class="aide-hint aide-formula-hint">Belote = Roi + Dame d'atout annonc\u00e9s par la m\u00eame \u00e9quipe.</p>
        </div>

        <div class="aide-section aide-scoring-section">
            <h3>Calcul du score</h3>

            <div class="aide-score-grid">
                <div class="aide-score-card aide-score-win">
                    <div class="aide-score-head">Contrat r\u00e9ussi</div>
                    <div class="aide-score-row">
                        <span class="aide-score-team aide-team-pren">Preneurs</span>
                        <span class="aide-score-val">points cartes + contrat + belote</span>
                    </div>
                    <div class="aide-score-row">
                        <span class="aide-score-team aide-team-def">D\u00e9fense</span>
                        <span class="aide-score-val">leurs points cartes + belote</span>
                    </div>
                </div>
                <div class="aide-score-card aide-score-lose">
                    <div class="aide-score-head">Chute</div>
                    <div class="aide-score-row">
                        <span class="aide-score-team aide-team-pren">Preneurs</span>
                        <span class="aide-score-val">0</span>
                    </div>
                    <div class="aide-score-row">
                        <span class="aide-score-team aide-team-def">D\u00e9fense</span>
                        <span class="aide-score-val">162 + contrat + belote</span>
                    </div>
                </div>
            </div>

            <div class="aide-score-sublabel">Coinch\u00e9 (\u00d72) / Surcoinch\u00e9 (\u00d73)</div>
            <div class="aide-score-grid">
                <div class="aide-score-card aide-score-win">
                    <div class="aide-score-head">Contrat r\u00e9ussi</div>
                    <div class="aide-score-row">
                        <span class="aide-score-team aide-team-pren">Preneurs</span>
                        <span class="aide-score-val">162<sup>\u2217</sup> + <span class="aide-mult">(contrat \u00d7 mult)</span> + belote</span>
                    </div>
                    <div class="aide-score-row">
                        <span class="aide-score-team aide-team-def">D\u00e9fense</span>
                        <span class="aide-score-val">0</span>
                    </div>
                </div>
                <div class="aide-score-card aide-score-lose">
                    <div class="aide-score-head">Chute</div>
                    <div class="aide-score-row">
                        <span class="aide-score-team aide-team-pren">Preneurs</span>
                        <span class="aide-score-val">0</span>
                    </div>
                    <div class="aide-score-row">
                        <span class="aide-score-team aide-team-def">D\u00e9fense</span>
                        <span class="aide-score-val">162 + <span class="aide-mult">(contrat \u00d7 mult)</span> + belote</span>
                    </div>
                </div>
            </div>
            <p class="aide-hint aide-formula-hint"><sup>\u2217</sup> 252 si capot r\u00e9alis\u00e9. <strong>mult</strong> = \u00d72 si coinch\u00e9, \u00d73 si surcoinch\u00e9 \u2014 s'applique uniquement \u00e0 la valeur du contrat. En chute, la d\u00e9fense prend le contrat et <em>tous</em> les points cartes de la donne (162), quel que soit le partage r\u00e9el des plis. Les scores marqu\u00e9s sont arrondis \u00e0 la dizaine.</p>
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
                <span class="aide-bid-tag">Surcoinche \u00d73</span>
                <span class="aide-bid-sep">\u2014</span>
                <span class="aide-bid-match">Partie en 2000 pts</span>
            </div>
        </div>
    </div>
</div>
`;

export function mount(container) {
    container.innerHTML = TEMPLATE;
    buildColumn(document.getElementById('aide-trump-col'), ALIGNED_SLOTS, 'trump', HEART);
    buildColumn(document.getElementById('aide-plain-col'), ALIGNED_SLOTS, 'plain', SPADE);
}

export function unmount() {}
