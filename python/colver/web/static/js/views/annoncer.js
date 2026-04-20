// Stratégie d'annonce — guide visuel dérivé du bot v5 (probe ML)

import { cardSvgPath } from '../shared/cards.js';

const HEART = 1;
const SPADE = 0;

// Rank indexing: 7=0, 8=1, 9=2, J=3, Q=4, K=5, 10=6, A=7

// Points par carte quand la couleur est ATOUT (formule NN-native)
// Valeurs "effectives" (rank + length bonus intégré) — directement additionnables
const TRUMP_CARDS_POINTS = [
    { rank: 3, pts: 11, label: 'V' },   // Valet
    { rank: 2, pts:  4, label: '9' },   // 9
    { rank: 6, pts:  3, label: '10' },  // 10
    { rank: 5, pts:  2, label: 'R' },   // Roi
    { rank: 4, pts:  2, label: 'D' },   // Dame
    { rank: 1, pts:  2, label: '8' },   // 8
    { rank: 0, pts:  2, label: '7' },   // 7
    { rank: 7, pts:  1, label: 'A' },   // As (faible !)
];

// Points par carte en LATÉRAL (couleur non-atout)
const SIDE_CARDS_POINTS = [
    { rank: 7, pts:  0, label: 'A' },
    { rank: 3, pts: -2, label: 'V' },
    { rank: 2, pts: -2, label: '9' },
    { rank: 5, pts: -1, label: 'R' },
    { rank: 4, pts: -1, label: 'D' },
    { rank: 6, pts: -1, label: '10' },
    { rank: 1, pts: -1, label: '8' },
    { rank: 0, pts: -1, label: '7' },
];

function ptsClass(pts) {
    if (pts >= 8) return 'annonce-pts-gold';
    if (pts >= 3) return 'annonce-pts-bright';
    if (pts >= 1) return 'annonce-pts-dim';
    if (pts === 0) return 'annonce-pts-zero';
    return 'annonce-pts-neg';
}

function ptsText(pts) {
    if (pts > 0) return `+${pts}`;
    if (pts === 0) return '0';
    return `${pts}`;
}

function buildCardRow(container, cards, suit, highlight) {
    container.innerHTML = '';
    for (const card of cards) {
        const cardIdx = suit * 8 + card.rank;
        const item = document.createElement('div');
        item.className = 'annonce-card-item' + (highlight && highlight(card) ? ' annonce-card-highlight' : '');

        const img = document.createElement('img');
        img.src = cardSvgPath(cardIdx);
        img.className = 'annonce-card-img';
        img.draggable = false;
        item.appendChild(img);

        const badge = document.createElement('div');
        badge.className = `annonce-pts ${ptsClass(card.pts)}`;
        badge.textContent = ptsText(card.pts);
        item.appendChild(badge);

        container.appendChild(item);
    }
}

const TEMPLATE = `
<div id="annonce-content">
    <div class="annonce-header">
        <h2>Stratégie d'annonce</h2>
        <p class="annonce-subtitle">Évalue une main et décide si tu annonces — poids appris par le bot v5</p>
        <p class="annonce-badge">≈ 90% d'accord avec le NN champion</p>
    </div>

    <div class="annonce-sections">

        <!-- ÉTAPE 1 : Score d'une couleur comme atout -->
        <div class="annonce-section">
            <div class="annonce-step">Étape 1</div>
            <h3>Calcule le score de <u>chaque</u> couleur comme atout potentiel</h3>
            <p class="annonce-hint">Pour chaque couleur, additionne ces points. Retiens le meilleur des 4.</p>

            <div class="annonce-subhead">
                <span class="suit-red">♥</span> Cartes dans la couleur <strong>atout</strong>
                <span class="annonce-subhead-hint">(le Valet domine, l'As presque rien)</span>
            </div>
            <div class="annonce-card-row annonce-card-row-8" id="annonce-trump-row"></div>

            <div class="annonce-subhead annonce-subhead-spaced">
                <span class="suit-black">♠</span> Cartes des 3 autres couleurs (<strong>latéral</strong>)
                <span class="annonce-subhead-hint">(l'As latéral = 0, le Valet/9 latéral pénalise)</span>
            </div>
            <div class="annonce-card-row annonce-card-row-8" id="annonce-side-row"></div>

            <div class="annonce-bonus-grid">
                <div class="annonce-bonus">
                    <div class="annonce-bonus-label">Coupe (couleur latérale vide)</div>
                    <div class="annonce-bonus-val annonce-pts-bright">+2 chacune</div>
                </div>
                <div class="annonce-bonus">
                    <div class="annonce-bonus-label">Singleton (1 carte latérale)</div>
                    <div class="annonce-bonus-val annonce-pts-bright">+1 chacun</div>
                </div>
            </div>

            <div class="annonce-corrections">
                <div class="annonce-correction">
                    <span class="annonce-correction-label">J + 9 du même atout</span>
                    <span class="annonce-correction-val annonce-pts-neg">−2</span>
                    <span class="annonce-correction-hint">anti-synergie</span>
                </div>
                <div class="annonce-correction">
                    <span class="annonce-correction-label">J + A du même atout</span>
                    <span class="annonce-correction-val annonce-pts-bright">+1</span>
                    <span class="annonce-correction-hint">synergie mineure</span>
                </div>
            </div>
        </div>

        <!-- EXEMPLE -->
        <div class="annonce-section annonce-example-section">
            <div class="annonce-step">Exemple</div>
            <h3>Main réelle, calcul pas-à-pas</h3>
            <p class="annonce-hint">Main : <strong class="annonce-hand">♠97 &nbsp;&nbsp; ♥V987 &nbsp;&nbsp; ♦R &nbsp;&nbsp; ♣7</strong> — 8 cartes</p>

            <div class="annonce-example-steps">
                <div class="annonce-example-step">
                    <span class="annonce-example-expr">V<sub>♥</sub>+9<sub>♥</sub>+8<sub>♥</sub>+7<sub>♥</sub></span>
                    <span class="annonce-example-calc">= 11+4+2+2</span>
                    <span class="annonce-example-val annonce-pts-bright">+19</span>
                </div>
                <div class="annonce-example-step">
                    <span class="annonce-example-expr">J×9 correction</span>
                    <span class="annonce-example-calc">−2</span>
                    <span class="annonce-example-val annonce-pts-neg">+17</span>
                </div>
                <div class="annonce-example-step">
                    <span class="annonce-example-expr">Latéral ♠ (9 et 7)</span>
                    <span class="annonce-example-calc">−2 − 1</span>
                    <span class="annonce-example-val annonce-pts-neg">+14</span>
                </div>
                <div class="annonce-example-step">
                    <span class="annonce-example-expr">Latéral ♦ (R)</span>
                    <span class="annonce-example-calc">−1</span>
                    <span class="annonce-example-val annonce-pts-neg">+13</span>
                </div>
                <div class="annonce-example-step">
                    <span class="annonce-example-expr">Latéral ♣ (7)</span>
                    <span class="annonce-example-calc">−1</span>
                    <span class="annonce-example-val annonce-pts-neg">+12</span>
                </div>
                <div class="annonce-example-step">
                    <span class="annonce-example-expr">2 singletons (♦, ♣)</span>
                    <span class="annonce-example-calc">+2</span>
                    <span class="annonce-example-val annonce-pts-bright">+14</span>
                </div>
                <div class="annonce-example-step annonce-example-total">
                    <span class="annonce-example-expr">Score ♥ final</span>
                    <span class="annonce-example-calc">—</span>
                    <span class="annonce-example-val annonce-pts-gold">14</span>
                </div>
            </div>

            <div class="annonce-example-concl">
                Autres couleurs : ♠ = 0, ♦ = −7, ♣ = −7. <strong>Meilleur = ♥ à 14.</strong>
                <br>
                <span class="annonce-nn-choice">⟶ Le NN v5 annonce <strong>110 ♥</strong> sur cette main.</span>
            </div>
        </div>

        <!-- ÉTAPE 2 : Décision selon position -->
        <div class="annonce-section">
            <div class="annonce-step">Étape 2</div>
            <h3>Décision selon ta position</h3>
            <p class="annonce-hint">Ton meilleur score (calculé en étape 1) dit s'il faut annoncer. Seuils selon où tu parles.</p>

            <div class="annonce-pos-grid">
                <div class="annonce-pos-card">
                    <div class="annonce-pos-num">1</div>
                    <div class="annonce-pos-label">Ouverture</div>
                    <div class="annonce-pos-rule">
                        <div class="annonce-pos-threshold">Score ≥ <span class="annonce-big">7</span></div>
                        <div class="annonce-pos-or">ou</div>
                        <div>Score ≥ 5 <span class="annonce-and">+ 3 atouts</span></div>
                        <div class="annonce-pos-or">ou</div>
                        <div>J + coupe + 2 atouts</div>
                        <div class="annonce-pos-or">ou</div>
                        <div>5+ atouts</div>
                    </div>
                </div>
                <div class="annonce-pos-card">
                    <div class="annonce-pos-num">2</div>
                    <div class="annonce-pos-label">1 passe</div>
                    <div class="annonce-pos-rule">
                        <div class="annonce-pos-threshold">Score ≥ <span class="annonce-big">8</span></div>
                        <div class="annonce-pos-or">ou</div>
                        <div>Score ≥ 5 <span class="annonce-and">+ 3 atouts</span></div>
                        <div class="annonce-pos-or">ou</div>
                        <div>J + coupe + 2 atouts</div>
                        <div class="annonce-pos-or">ou</div>
                        <div>4+ atouts</div>
                    </div>
                </div>
                <div class="annonce-pos-card annonce-pos-highlight">
                    <div class="annonce-pos-num">3</div>
                    <div class="annonce-pos-label">2 passes — protection</div>
                    <div class="annonce-pos-rule">
                        <div class="annonce-pos-threshold">Score ≥ <span class="annonce-big">6</span></div>
                        <div class="annonce-pos-or">ou</div>
                        <div>Score ≥ 4 <span class="annonce-and">+ 3 atouts</span></div>
                        <div class="annonce-pos-or">ou</div>
                        <div>J + 2 atouts</div>
                        <div class="annonce-pos-or">ou</div>
                        <div>3+ atouts</div>
                    </div>
                </div>
                <div class="annonce-pos-card">
                    <div class="annonce-pos-num">4</div>
                    <div class="annonce-pos-label">3 passes — dernier</div>
                    <div class="annonce-pos-rule">
                        <div class="annonce-pos-threshold">Score ≥ <span class="annonce-big">7</span></div>
                        <div class="annonce-pos-or">ou</div>
                        <div>Score ≥ 5 <span class="annonce-and">+ 3 atouts</span></div>
                        <div class="annonce-pos-or">ou</div>
                        <div>J + coupe + 2 atouts</div>
                        <div class="annonce-pos-or">ou</div>
                        <div>4+ atouts</div>
                    </div>
                </div>
            </div>

            <div class="annonce-mnemonic">
                <strong>Mnémonique :</strong> <span class="annonce-mono">7/5 · 8/5 · 6/4 · 7/5</span>
                <span class="annonce-mnemonic-hint">— seuils principaux (pos1 · pos2 · pos3 · pos4)</span>
            </div>
        </div>

        <!-- Réponses -->
        <div class="annonce-section">
            <div class="annonce-step">Étape 3</div>
            <h3>Après une annonce adverse</h3>

            <div class="annonce-response-grid">
                <div class="annonce-response annonce-response-partner">
                    <div class="annonce-response-head">
                        <span class="annonce-response-icon">🤝</span>
                        <span>Partenaire a dit <strong>80</strong></span>
                    </div>
                    <div class="annonce-response-body">
                        <div class="annonce-response-default annonce-pts-bright">Annonce presque toujours</div>
                        <div class="annonce-response-exception">
                            <span class="annonce-tag">Sauf</span>
                            0-1 carte dans sa couleur<br>
                            <strong>ET</strong> score < 5<br>
                            <strong>ET</strong> < 3 atouts
                        </div>
                        <div class="annonce-response-tip">
                            Si 3+ cartes dans sa couleur → soutiens dans sa couleur.
                            Sinon annonce dans ta meilleure.
                        </div>
                    </div>
                </div>

                <div class="annonce-response annonce-response-opp">
                    <div class="annonce-response-head">
                        <span class="annonce-response-icon">⚔️</span>
                        <span>Adversaire a dit <strong>80</strong> — règle du miroir</span>
                    </div>
                    <div class="annonce-response-body">
                        <div class="annonce-mirror-rule">
                            <strong>Ignore sa couleur.</strong><br>
                            Recalcule le score sur <u>tes 3 autres couleurs</u> → <span class="annonce-mono">score_alt</span>.
                        </div>
                        <div class="annonce-mirror-table">
                            <div class="annonce-mirror-row">
                                <span class="annonce-mirror-cond">4+ cartes dans SA couleur</span>
                                <span class="annonce-mirror-act annonce-act-coinche">COINCHE</span>
                            </div>
                            <div class="annonce-mirror-row">
                                <span class="annonce-mirror-cond">score_alt ≥ 6</span>
                                <span class="annonce-mirror-act annonce-act-bid">Contre-annonce</span>
                            </div>
                            <div class="annonce-mirror-row">
                                <span class="annonce-mirror-cond">score_alt 4-5 + 3 atouts</span>
                                <span class="annonce-mirror-act annonce-act-bid">Contre-annonce</span>
                            </div>
                            <div class="annonce-mirror-row">
                                <span class="annonce-mirror-cond">3+ cartes sa couleur + score_alt &lt; 5</span>
                                <span class="annonce-mirror-act annonce-act-coinche">COINCHE</span>
                            </div>
                            <div class="annonce-mirror-row annonce-mirror-default">
                                <span class="annonce-mirror-cond">Sinon</span>
                                <span class="annonce-mirror-act annonce-act-pass">passe</span>
                            </div>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- Niveaux -->
        <div class="annonce-section">
            <div class="annonce-step">Étape 4</div>
            <h3>À quel niveau annoncer ?</h3>
            <p class="annonce-hint">Distribution réelle du NN sur 80 000 mains par scénario.</p>

            <div class="annonce-level-table">
                <div class="annonce-level-head">
                    <span>Score</span>
                    <span>pos 1-2</span>
                    <span>pos 3</span>
                    <span>pos 4</span>
                </div>
                <div class="annonce-level-row"><span>&lt; 7</span><span>80</span><span>80</span><span>80</span></div>
                <div class="annonce-level-row"><span>7-9</span><span>80</span><span>80</span><span>80</span></div>
                <div class="annonce-level-row"><span>10-13</span><span>90</span><span>100</span><span>100</span></div>
                <div class="annonce-level-row"><span>14-17</span><span>100-110</span><span>110</span><span>100-110</span></div>
                <div class="annonce-level-row"><span>18-21</span><span>110</span><span>110</span><span>110</span></div>
                <div class="annonce-level-row"><span>22+</span><span>110-120</span><span>110-120</span><span>110</span></div>
            </div>

            <div class="annonce-level-note">
                <strong>⚠️ Nuance :</strong> si ta 2e couleur a aussi un score décent (≥10), annonce <em>plus bas</em> pour laisser parler le partenaire. Règle pratique : <strong>−10 au niveau</strong> quand tu as 2 couleurs compétitives.
            </div>
        </div>

        <!-- Pièges -->
        <div class="annonce-section annonce-traps-section">
            <h3>⚠️ Pièges à connaître</h3>
            <ul class="annonce-traps">
                <li><strong>L'As d'atout vaut presque rien</strong> (+1 comme un 7). Les aces latéraux non plus (A = 0 net). Le J et le 9 latéraux sont carrément <em>négatifs</em>.</li>
                <li><strong>Le 10 d'atout est un poids mort</strong> (+3, battable par J/9/A).</li>
                <li><strong>J+9 ensemble :</strong> −2 d'anti-synergie. Reste très fort mais pas "J seul" + "9 seul" additionnés.</li>
                <li><strong>Belote (R+D atout) n'aide pas à annoncer</strong> mais permet de monter d'un palier (+20 pts garantis à l'annonce).</li>
                <li><strong>"Annoncer aux As"</strong> (3 As latéraux sans J ni 9) : passe. Les As latéraux ne compensent pas le manque d'atout.</li>
            </ul>
        </div>

        <!-- Footer -->
        <div class="annonce-footer">
            Règles dérivées par régression ML directe sur les poids du bot v5 champion.
            <a href="https://github.com/avo-k/colver" target="_blank" rel="noopener">Code source</a>.
        </div>
    </div>
</div>
`;

export function mount(container) {
    container.innerHTML = TEMPLATE;
    // Highlight J (rank=3) and 9 (rank=2) as the "promoted" / important trump cards
    buildCardRow(
        document.getElementById('annonce-trump-row'),
        TRUMP_CARDS_POINTS,
        HEART,
        (c) => c.rank === 3 || c.rank === 2,
    );
    buildCardRow(
        document.getElementById('annonce-side-row'),
        SIDE_CARDS_POINTS,
        SPADE,
        null,
    );
}

export function unmount() {}
