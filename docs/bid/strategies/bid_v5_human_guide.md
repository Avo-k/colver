# Bid v5 — guide humain

Évaluation pratique d'une main et décision d'annoncer. Règles tirées du bot champion v5 par analyse ML directe des poids du réseau — elles reproduisent sa décision à **88-94 %** selon le scénario.

## 1. Calculer le `score` d'une couleur comme atout potentiel

Pour chaque couleur, additionne :

**Cartes dans la couleur (atout)**
| J | 9 | 10 | K, Q, 8, 7 | A |
|--:|--:|--:|--:|--:|
| +11 | +4 | +3 | +2 chacune | +1 |

**Cartes des 3 autres couleurs (latéral)** — valeur nette (longueur incluse)
| A | J ou 9 | K, Q, 10, 8, 7 |
|--:|--:|--:|
| 0 | −2 chacune | −1 chacune |

**Distribution**
- +2 par coupe (couleur latérale vide)
- +1 par singleton (couleur latérale à 1 carte)

**Corrections**
- J **et** 9 du même atout ensemble : **−2**
- J **et** A du même atout ensemble : +1

Retiens ton meilleur `score` sur les 4 couleurs.

### Exemple concret (main réelle, décision réelle du NN)

Main : **♠97 &nbsp; ♥J987 &nbsp; ♦K &nbsp; ♣7** (8 cartes)

Évaluation comme atout ♥ :
- Atout : J(+11) + 9(+4) + 8(+2) + 7(+2) = **+19**
- Correction J×9 : **−2** → **+17**
- Latéral ♠ : 9(−2) + 7(−1) = **−3**
- Latéral ♦ : K(−1) = **−1**
- Latéral ♣ : 7(−1) = **−1**
- Distribution : 2 singletons (♦, ♣) = **+2**
- **Score ♥ = 17 − 3 − 1 − 1 + 2 = 14**

Autres couleurs : ♠ = 0, ♦ = −7, ♣ = −7. Meilleur = ♥ à 14.

**Décision du NN v5** sur cette main en ouverture : **annonce 110 ♥**. Notre règle (§2) : score 14 ≥ 7 → annonce. Niveau (§5) : 14 → 110. ✓

## 2. Décision selon la position

| Position | Annonce si |
|----------|-----------|
| **1 (ouverture)**      | score ≥ 7 &nbsp; OU &nbsp; (score ≥ 5 + 3 atouts) &nbsp; OU &nbsp; (J + coupe + 2 atouts) &nbsp; OU &nbsp; 5+ atouts |
| **2 (1 passe)**        | score ≥ 8 &nbsp; OU &nbsp; (score ≥ 5 + 3 atouts) &nbsp; OU &nbsp; (J + coupe + 2 atouts) &nbsp; OU &nbsp; 4+ atouts |
| **3 (2 passes)**       | score ≥ 6 &nbsp; OU &nbsp; (score ≥ 4 + 3 atouts) &nbsp; OU &nbsp; (J + 2 atouts) &nbsp; OU &nbsp; 3+ atouts |
| **4 (3 passes)**       | score ≥ 7 &nbsp; OU &nbsp; (score ≥ 5 + 3 atouts) &nbsp; OU &nbsp; (J + coupe + 2 atouts) &nbsp; OU &nbsp; 4+ atouts |

Mnémonique : `7/5 · 8/5 · 6/4 · 7/5` (seuil principal / avec 3+ atouts).

## 3. Réponse au partenaire (il a dit 80)

**Annonce toujours**, sauf : `0-1 carte dans sa couleur ET score < 5 ET < 3 atouts` → passe.

Si 3+ cartes dans sa couleur, soutiens dans sa couleur. Sinon annonce dans ta meilleure couleur.

## 4. Défense (adversaire a dit 80) — règle du miroir

**Ignore sa couleur** avant d'évaluer. Calcule le score sur tes 3 autres couleurs → `score_alt`.

| Condition | Action |
|-----------|--------|
| 4+ cartes dans SA couleur | **Coinche** (tu tiens ses atouts) |
| score_alt ≥ 6 | Contre-annonce dans cette couleur |
| score_alt 4-5 **et** 3+ atouts dedans | Contre-annonce |
| 3+ cartes dans sa couleur **et** score_alt < 5 | Coinche |
| Sinon | Passe |

## 5. Niveau à annoncer

Distribution réelle du niveau choisi par le NN selon `nn_score` (couleur annoncée), mesurée sur 80k mains :

| score | pos1-2 | pos3 | pos4 |
|------:|:------:|:----:|:----:|
| < 7   | 80 | 80 | 80 |
| 7-9   | 80 | 80 | 80 |
| 10-13 | 90 | 100 | 100 |
| 14-17 | 100-110 | 110 | 100-110 |
| 18-21 | 110 | 110 | 110 |
| 22+   | 110-120 | 110-120 | 110 |

**⚠️ Nuance** : si ta **2e** couleur a aussi un score décent (≥10), le NN préfère souvent annoncer **plus bas** pour laisser parler le partenaire. Règle pratique : retire 10 au niveau quand tu as 2 couleurs compétitives.

## 6. Pièges à connaître

- **L'As d'atout vaut presque rien** (+1, comme un 7). Les honneurs latéraux non plus (A = 0 net, J et 9 latéraux sont carrément *négatifs*).
- **Le 10 d'atout est un poids mort** (+3, battable par J/9/A).
- **J+9 ensemble** : −2 d'anti-synergie. Reste très fort mais pas "J seul" + "9 seul" additionnés.
- **Belote (K+Q atout)** n'aide **pas** à annoncer mais permet de monter d'un palier (+20 pts garantis).
- **"Annoncer aux As"** (3 As latéraux sans J ni 9) : passe. Les As latéraux ne compensent pas le manque d'atout.

## 7. Précision de ces règles

Mesurée sur 80 000 mains par scénario contre la décision réelle du NN v5 :

| Scénario | Accord NN | XGBoost plafond |
|----------|----------:|----------------:|
| pos1 (ouverture) | **89.9 %** | 95 % |
| pos2 (après passe) | **91.3 %** | 95 % |
| pos3 (protection) | **94.1 %** | 96 % |
| pos4 (dernière chance) | 90.7 % | 97 % |
| Partenaire 80 | 85-91 % | 99 % |
| Défense opp 80 | 78-88 % | 96-97 % |
| Coinche | 89 % | 80 % |

Les ~10pp d'écart au plafond XGBoost tiennent à des interactions à 5 variables et moyennes de 300 arbres qu'un humain ne peut pas faire mentalement.

---

Dérivation détaillée : [bid_v5_simplified_rules.md](bid_v5_simplified_rules.md), [probe_morning_report.md](../interpretability/probe_morning_report.md).
