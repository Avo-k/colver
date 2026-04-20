# Bid Rules — Distilled from NN V2

Rules extracted by training interpretable models (Decision Trees, XGBoost) on 200,000 random deals per scenario, evaluated by the bid NN V2 (512-hidden, 3-layer dueling DQN trained on DD oracle rewards).

**Accuracy of the distilled rules vs the NN:** ~93% for opening, ~85-90% for responses.

## Quick Reference: trump_score

`trump_score = evaluate_for_trump(hand, suit)` is the single most predictive feature. It combines:

| Card (as trump) | Points | Card (side suit) | Points |
|-----------------|--------|------------------|--------|
| Valet           | 8      | As               | +3     |
| 9               | 6      | Coupe (0 cartes) | +3     |
| As              | 4      | Singleton        | +1     |
| 10              | 3      |                  |        |
| Roi             | 1      |                  |        |
| Dame            | 1      |                  |        |

**+ bonus longueur atout:** max(0, nb_atouts - 2) × 2

Exemples: J9 + 2 petits + A latéral = 8+6+2+3 = **19**. J seul + 3 As latéraux = 8+9 = **17**.

---

## 1. Ouverture (Position 1)

**Taux global:** 80% des mains sont annoncées, 20% passent.

### Règle simple (92.8% de concordance avec le NN)

```
1. As-tu le VALET d'atout ?
   OUI → aller en 2
   NON → aller en 3

2. Avec Valet:
   - J + ≥ 2 atouts (sans As d'atout)  → ANNONCE
   - J + As atout + ≥ 2 atouts + score > 15  → ANNONCE
   - J seul (1 atout) + score > 13.5 + 2e couleur forte  → ANNONCE (rare)
   - J seul (1 atout) sinon  → PASSE

3. Sans Valet:
   - 9 + ≥ 3 atouts (sans As d'atout)  → ANNONCE
   - 9 + As atout + ≥ 3 atouts + coupe  → ANNONCE
   - Ni J ni 9 + coupe + peu d'as totaux (≤1)  → ANNONCE (distribution)
   - Sinon  → PASSE
```

### Table de référence par composition d'atout

| Atout      | 1 carte | 2 cartes | 3 cartes | 4 cartes | 5+ cartes |
|------------|---------|----------|----------|----------|-----------|
| **J + 9**  | —       | 99%      | 100%     | 100%     | 100%      |
| **J seul** | 28%     | **91%**  | **99%**  | 100%     | 100%      |
| **9 seul** | 1%      | 10%      | **73%**  | **93%**  | 100%      |
| **Ni J/9** | —       | 1%       | 9%       | 47%      | 94%       |

### Importance des features (XGBoost)

1. **has_jack** — 47.6% (domine de loin)
2. **has_nine** — 13.3%
3. **trump_count** — 12.1%
4. **has_ace** — 7.8% (attention: négatif avec J !)
5. **side_voids** — 7.6%

### Le piège de l'As d'atout

Surprise du NN: **l'As d'atout combiné au Valet est souvent un signal de passe**, pas d'annonce. Avec J+A d'atout et 2 cartes, le NN annonce moins (91%) que J sans A (91%). Avec 4+ atouts et J+A, c'est encore pire.

Explication probable: l'As prend de la place sans apporter de contrôle (contrairement au 9 qui vaut 14 points en atout). Le NN préfère une main concentrée (J+9+petits) à une main étalée (J+A+petits).

---

## 2. Après passes (Positions 2-4)

### Position 2 (après 1 passe)

**Taux:** 66% annonce, 34% passe — plus sélectif qu'en ouverture.

Le NN est **plus strict** en position 2 car l'adversaire suivant peut encore enchérir. Les seuils sont plus hauts qu'en ouverture (score ≥ 14 au lieu de ≥ 10).

### Position 3 (après 2 passes — partenaire a passé)

**Taux:** 95% annonce — le NN ouvre quasi-systématiquement.

C'est la position "protective": si tu passes, la donne est void (4 passes = redistribution). Le NN annonce même avec des mains très faibles (score ≥ 5).

Les niveaux montent: 90 est le plus fréquent (71k), devant 100 (43k) et 80 (40k).

### Position 4 (après 3 passes — dernier à parler)

**Taux:** 75.5% annonce — position de "sauvetage".

Seuils intermédiaires. Les niveaux sont conservateurs: 80 dominant (65k).

### Table combinée (positions 2-4 après passes)

| Atout      | 2 cartes | 3 cartes | 4 cartes | 5+ cartes |
|------------|----------|----------|----------|-----------|
| **J + 9**  | 95%      | 100%     | 100%     | 100%      |
| **J seul** | 75%      | **96%**  | 100%     | 100%      |
| **9 seul** | 40%      | **58%**  | **86%**  | 100%      |
| **Ni J/9** | 12%      | 31%      | 53%      | 91%       |

---

## 3. Réponse au partenaire (Partner bid 80)

**Taux:** 82% annonce (pos3: 92%, pos4: 72%).

### Importance des features

1. **is_partner_suit** — 41.8% (le NN surenchérit quasi-systématiquement dans la couleur du partenaire)
2. **trump_score** — 19.8%
3. **has_jack** — 12.5%
4. **trump_count** — 6.7%
5. **partner_support** — 3.7% (combien de cartes dans la couleur du partenaire)

### Règles de réponse

```
Partenaire annonce 80♠:

1. Dans SA couleur (♠):
   - J'ai le J d'atout  → 90 (surenchère obligatoire si ≥ 2 atouts)
   - J'ai le 9 + ≥ 2 cartes  → 90
   - ≥ 3 cartes dans sa couleur + As latéral  → 90
   - 0-1 cartes dans sa couleur + score < 7  → PASSE

2. Dans UNE AUTRE couleur:
   - J + ≥ 2 atouts + score > 12  → ANNONCE dans ma couleur
   - Score > 16 dans ma couleur  → ANNONCE
   - Sinon  → PASSE ou soutien dans sa couleur
```

### Support par nombre de cartes dans la couleur du partenaire

| Cartes dans sa couleur | Taux d'annonce |
|------------------------|---------------|
| 0                      | 87%           |
| 1                      | 71%           |
| 2                      | 79%           |
| 3                      | 91%           |
| 4                      | 97%           |
| 5                      | 99%           |

Observation intéressante: 0 cartes → 87% (le NN annonce dans une autre couleur !). Le creux à 1 carte est surprenant.

---

## 4. Défense (Adversaire bid 80)

**Taux:** 55% contre-annonce, 17% coinche, 28% passe.

### Importance des features

1. **opp_suit_cards** — 24.7% (combien de cartes J'AI dans la couleur de l'adversaire)
2. **side_voids** — 16.0%
3. **best_side_length** — 10.9%
4. **is_opp_suit** — 8.8%
5. **has_jack** — 7.3%
6. **trump_score** — 5.1%

### La clé: combien de cartes dans la couleur adverse

```
Adversaire annonce 80♠, combien ai-je de ♠ ?

- 0-1 cartes ♠  → annonce dans une autre couleur (souvent 90)
- 2 cartes ♠    → annonce si score > 14 dans ma couleur OU coupe
- 3 cartes ♠    → difficile, passe sauf score > 14 + coupe
- 4+ cartes ♠   → PASSE (ses cartes sont des atouts contre moi)
```

### Le coinche

Le NN coinche dans 17% des cas quand l'adversaire annonce 80. Profil du coinche:
- **Score atout moyen: 11.8** (pas besoin d'un gros jeu !)
- **Nb atouts moyen: 2.0** — souvent avec peu d'atout
- **29% ont le J, 25% ont le 9** dans la couleur adverse
- **3.3 cartes en moyenne dans la couleur adverse** — le coinche se fait avec BEAUCOUP de cartes dans la couleur de l'adversaire (c'est logique: ça veut dire qu'il a peu d'atouts)

**Règle de coinche simplifiée:**
```
Coinche si:
  - ≥ 3 cartes dans la couleur adverse (tu tiens ses atouts)
  - Ou J/9 de sa couleur + ≥ 2 cartes
  - Position 3 coinche plus facilement (28%) que position 2 (11%) ou 4 (11%)
```

---

## 5. Niveaux d'annonce

Quand le NN annonce, voici la distribution des niveaux en ouverture:

| Niveau | Fréquence | Conditions typiques |
|--------|-----------|---------------------|
| 80     | 48%       | J + 2 atouts, ou 9 + 3-4 atouts |
| 90     | 22%       | J + 2-3 atouts + 1 As latéral |
| 100    | 15%       | J + 3+ atouts + As/coupe, ou J9 + 3 atouts |
| 110    | 11%       | J9 + 4 atouts, ou J + 4+ avec distribution |
| 120    | 3.2%      | J9 + 4-5 atouts + coupes, score > 25 |
| 130+   | 0.3%      | J9 + 5+ atouts + 2 coupes, score > 28 |

**Seuils approximatifs par trump_score:**

| trump_score | Niveau typique |
|-------------|---------------|
| 10-14       | 80            |
| 14-17       | 80 (voire 90 avec As latéral) |
| 17-20       | 80-100        |
| 20-25       | 100-110       |
| 25-30       | 110-120       |
| 30+         | 120-130+      |

---

## 6. SHAP: Quelles cartes comptent vraiment ?

Analyse SHAP directe sur le réseau de neurones (contribution marginale Monte Carlo sur 20k mains).

### Contribution marginale de chaque rang en tant qu'atout

| Rang | Contribution | Points atout | Verdict |
|------|-------------|-------------|---------|
| **Valet** | **+0.28** | 20 | Roi absolu, loin devant tout |
| **9** | **+0.15** | 14 | Essentiel, mais 2× moins que J |0?
| 7 | +0.07 | 0 | Masse d'atout (positif !) |
| 8 | +0.06 | 0 | Idem |
| Dame | +0.07 | 3 | Comme un petit |
| Roi | +0.06 | 4 | Comme un petit |
| **10** | **+0.05** | 10 | **Pire qu'un 7 !** Vulnérable |
| **As** | **-0.02** | 11 | **NEGATIF.** L'As d'atout nuit. |

### L'As d'atout: le piège confirmé par 3 méthodes

L'As d'atout a une contribution **négative** à l'annonce. Confirmé par :
- XGBoost SHAP: `has_ace` high → -1.95 (fortement anti-bid)
- Monte Carlo marginal: A♠ = -0.04, A♥ = -0.03, A♦ = -0.03, A♣ = -0.03
- Dependence plot: has_ace=1 crée des SHAP entre -1 et -4

**Pourquoi l'As d'atout est toxique:**
1. L'As est **battu** par le J et le 9 (qui valent J=20, 9=14 pts)
2. Il **occupe une place** sans apporter de contrôle au jeu
3. Il **remplace** une carte latérale qui pourrait être une coupe ou un As latéral
4. 11 points dans les mains de l'adversaire s'il a J ou 9

**Règle pour le joueur humain:** Ne pas compter l'As d'atout comme un atout. Le traiter comme une carte neutre voire légèrement négative pour la décision d'annoncer. Un 7 d'atout de plus est plus utile qu'un As d'atout.

### Contribution "overall" (bid advantage, toutes couleurs)

```
J:   +0.043  ███████████████████  (le seul vrai moteur)
9:   +0.010  ████                 (4× moins que J)
7,8: ~0.000                       (neutre)
Q,K: ~0.000                       (neutre)
10:  -0.014  -------              (poids mort)
A:   -0.032  ---------------      (toxique)
```

### Leçon contre-intuitive: le 10 est pire qu'un 7

Un 7 d'atout (+0.07) contribue **plus** à l'annonce qu'un 10 d'atout (+0.05). Le 10 vaut 10 points mais est vulnérable — capturable par J, 9, et As adverses. Le 7 ne vaut rien en points mais c'est un atout de plus qui fait la longueur.

### Plots SHAP

Tous dans `data/shap/`: `shap_card_heatmap.png`, `shap_card_contributions.png`, `shap_xgb_summary.png`, `shap_xgb_dep_has_ace.png`.

---

## 7. Analyse par combinaisons de cartes

Expériences contrôlées : on fixe les atouts et on moyenne sur des milliers de mains aléatoires.

### Valet + quel 2e atout ?

Avec exactement 2 atouts (J + X), le reste aléatoire :

| Combo | Advantage | Verdict |
|-------|-----------|---------|
| **J+9** | **+0.085** | De loin le meilleur duo |
| J+7 | +0.054 | Un petit > un gros ! |
| J+Q | +0.052 | |
| J+8 | +0.051 | |
| J+K | +0.047 | |
| J+10 | +0.024 | Moitié d'un petit |
| **J+A** | **-0.030** | **Pire que J seul !** |

Le J seul (+0.043) est meilleur que J+A (-0.030). C'est-à-dire que l'As d'atout fait activement du mal, même avec le Valet.

### 3 atouts : classement des trios

| Trio | Advantage |
|------|-----------|
| **J+9+7** | **+0.144** |
| J+9+K | +0.142 |
| J+9+8 | +0.141 |
| J+7+8 | +0.135 |
| J+K+Q | +0.133 |
| J+9+10 | +0.125 |
| J+10+K | +0.116 |
| 9+7+8 | +0.096 |
| 9+K+Q | +0.092 |
| **J+9+A** | **+0.077** |
| J+A+7 | +0.070 |
| J+A+K | +0.069 |

**J+9+7 > J+9+A** : le 7 vaut mieux que l'As comme 3e atout ! (delta = +0.07 en faveur du 7)

### Ajout d'un 4e atout à J+9+7

| Ajout | Total | Delta vs base |
|-------|-------|---------------|
| **+K** | +0.172 | **+0.039** |
| +8 | +0.171 | +0.038 |
| +Q | +0.170 | +0.037 |
| +10 | +0.156 | +0.023 |
| **+A** | **+0.104** | **-0.028** |

Ajouter l'As fait **baisser** la valeur de la main de -0.028 ! Tout petit (K, 8, Q) apporte +0.04 de bonus de longueur, mais l'As détruit cet avantage.

### Longueur d'atout

Avec J+9 + N petits :

| Total trump | Advantage | Delta par carte |
|-------------|-----------|-----------------|
| 2 (J+9) | +0.123 | — |
| 3 | +0.154 | +0.031 |
| 4 | +0.176 | +0.022 |
| 5 | +0.199 | +0.022 |
| 6 | +0.218 | +0.019 |

Rendement décroissant : la 3e carte vaut +0.031, la 6e vaut +0.019. Mais chaque carte compte.

Avec J seul + petits :

| Total trump | Advantage |
|-------------|-----------|
| 1 (J seul) | +0.043 |
| 2 | +0.096 |
| 3 | +0.133 |
| 4 | +0.172 |
| 5 | +0.218 |

Note : **J + 4 petits (5 atouts) = J9 + 4 petits (6 atouts)**. La longueur compense le manque du 9.

### Coupes vs As latéraux

Avec J+9+7 fixés, comparaison directe :

| Configuration | Advantage |
|---------------|-----------|
| **0 as, 2 coupes** | **+0.222** |
| **0 as, 1 coupe** | **+0.202** |
| 1 as, 1 coupe | +0.122 |
| 1 as, 0 coupe | +0.114 |
| 2 as, 0 coupe | +0.075 |
| 3 as, 0 coupe | +0.044 |

**Une coupe vaut ~3 As latéraux.** Et les As latéraux sont à rendement décroissant (le 2e As vaut moins que le 1er, le 3e encore moins). Les coupes sont à rendement croissant !

Les As latéraux **nuisent** au-delà d'un certain point : ils prennent des places dans les couleurs latérales, empêchant les coupes et la courte distribution.

### 10 latéraux

Les 10 sont légèrement négatifs aussi : 0 → +0.151, 1 → +0.139, 2 → +0.121, 3 → +0.094.

### Archétypes de mains

| Archétype | Advantage | Description |
|-----------|-----------|-------------|
| 6 trump J9 | +0.212 | La machine |
| 5 trump J9 | +0.190 | Très fort |
| J9 + 2 coupes | +0.176 | Distribution > longueur |
| J9 + 2 petits | +0.171 | J9 garbage |
| J9 + belote (KQ) | +0.167 | Belote bonus marginal |
| J9 + 1 coupe | +0.163 | Standard |
| 5 trump sans J/9 | +0.130 | La masse compense |
| **Monster J9A10** | **+0.088** | **Pire que J9+2 petits !** |
| 9 + 2 petits | +0.086 | Marginal |
| J9 + 0 coupe + 2 as | +0.076 | Les as ne compensent pas |
| J seul | +0.045 | Fragile |
| 3 as latéraux, no J | -0.017 | **"Aux as" = passe** |

**Le "monster" J9A10 (+0.088) est nettement pire que J9+2 petits (+0.171).** L'As et le 10 d'atout prennent des places de longueur sans apporter assez de contrôle.

### Belote (K+Q d'atout)

| Contexte | K+Q ensemble | K + petit | Q + petit | 2 petits | Synergie K×Q |
|----------|-------------|-----------|-----------|----------|-------------|
| Avec J+9 | +0.170 | +0.172 | +0.170 | +0.166 | **-0.006** |
| Avec J seul | +0.127 | +0.116 | +0.119 | +0.109 | **+0.001** |
| Avec 9 seul | +0.070 | +0.061 | +0.063 | — | — |

**La belote n'aide pas pour la décision bid/pass** — le K et la Q valent chacun ~+0.007 (comme des petits). Mais la belote a un **effet majeur sur le niveau** :

Avec J+9 et 4 atouts, la belote shift massivement vers les niveaux hauts :

| Config | Avg level | % à 120 |
|--------|-----------|---------|
| J9+KQ | 114 | **37%** |
| J9+87 | 112 | 22% |

Le delta Q entre belote et non-belote **augmente avec le niveau** : +0.03 à 110, +0.04 à 120, +0.05 à 130, +0.08 à 140. Le NN a appris que la belote garantit 20 points bonus, ce qui sécurise un contrat plus haut.

**Règle pour le joueur : la belote ne change pas la décision d'annoncer, mais elle autorise +10 à +20 points de plus sur le niveau.**

### Combos latéraux

| Combo latéral | Delta vs sans |
|---------------|---------------|
| **K+Q même couleur** | **+0.025** (positif !) |
| As latéral | -0.054 (négatif) |
| A+10 même couleur | -0.055 (pire) |

**K+Q latéral est le seul combo de cartes hautes qui aide.** C'est une source de plis défensive. L'As seul et surtout A+10 ensemble sont des poids morts — trop de cartes dans une couleur qu'on voudrait courte pour couper.

### Résumé pour le joueur humain

1. **Longueur > Honneurs** (sauf J et 9 qui sont spéciaux)
2. **Coupes > As latéraux** — une coupe vaut 3 As
3. **L'As d'atout est toxique** — un 7 d'atout est meilleur
4. **Le 10 d'atout est un poids mort** — mieux vaut un 8
5. **J+9 est le duo magique** — le gap avec J+A ou J+10 est énorme
6. **"Annoncer aux As" ne fonctionne pas** — 3 As latéraux sans J ni 9 = passe
7. **La belote d'atout n'aide pas à annoncer** — K+Q = deux petits pour la décision
8. **K+Q latéral aide** (+0.025) — seul combo de cartes hautes qui apporte quelque chose
9. **A+10 latéral nuit** — trop de cartes dans une couleur = pas de coupe

---

## 8. Le 9 sans Valet — deep dive par position

### Taux d'annonce par position et longueur

| Trump | Pos1 | Pos2 | Pos3 | Pos4 |
|-------|------|------|------|------|
| 1 (9 seul) | 70% | 75% | **65%** | 96% |
| 2 | 71% | 82% | **64%** | 97% |
| 3 | 79% | 95% | **75%** | 99% |
| 4 | 98% | 100% | **97%** | 100% |
| 5 | 100% | 100% | 100% | 100% |

**Position 3 est la plus prudente pour le 9.** Contrairement au J qui annonce à 95% en pos3 ("protection"), le NN sait que le 9 seul n'est pas assez fort pour protéger. Il passe dans 25-36% des cas.

**Position 4 annonce quasi-tout** — dernier à parler, mieux vaut tenter que laisser mourir la donne.

**Position 2 est agressive** — plus que Pos1 ! Le NN profite de l'information que l'adversaire a passé.

### Le gap J vs 9 varie énormément selon la position

| Trump | Pos1 | Pos2 | Pos3 | Pos4 |
|-------|------|------|------|------|
| 2 | J+23pp | J+17pp | **J+29pp** | J+3pp |
| 3 | J+20pp | J+5pp | **J+25pp** | J+1pp |
| 4 | J+3pp | J+0pp | J+4pp | J+0pp |

**Pos4 annule le gap** — avec 3+ atouts, le 9 annonce autant que le J.
**Pos3 creuse le gap** — le J protège, le 9 non.
**À 4+ atouts**, la longueur domine et le gap disparaît partout.

Mais le **niveau** reste très différent : même quand le 9 annonce, il annonce plus bas.

| Trump | J avg level (Pos3) | 9 avg level (Pos3) | Gap |
|-------|---------------------|---------------------|-----|
| 3 | 87 | 82 | -5 |
| 4 | 94 | 82 | **-12** |
| 5 | 107 | 83 | **-24** |

Le 9 annonce toujours timidement (80 dominant). Le J ose monter.

### 9 + quel compagnon ? (2 atouts)

| Combo | Pos1 bid | Pos1 avg | Verdict |
|-------|----------|----------|---------|
| **9+A** | **85%** | **87** | Le meilleur ! (inverse du J) |
| 9+10 | 75% | 85 | |
| 9+K | 72% | 84 | |
| 9+7 | 70% | 84 | |

**L'As aide le 9** (contrairement au J). Le 9 a besoin d'un preneur de plis côté atout, l'As remplit ce rôle. Pour le J, l'As est redondant (le J est déjà le plus fort).

### Les coupes transforment le 9

Avec 9+7+8 (3 atouts), les coupes latérales changent tout :

| Configuration | Pos1 bid | Pos3 bid |
|---------------|----------|----------|
| 0 ace, 0 void | 49% | 48% |
| 1 ace, 0 void | 83% | 60% |
| **0 ace, 1 void** | **93%** | **98%** |
| 1 ace, 1 void | 100% | 100% |
| 2 aces, 0 voids | 100% | 100% |
| **0 ace, 2 voids** | **100%** (avg 93) | **100%** (avg 96) |

**Une coupe fait passer de 49% à 93%.** C'est encore plus spectaculaire que pour le J.

**Règle pour le joueur humain avec un 9 sans Valet :**
1. **Pos1 (ouverture)** : 3+ atouts + coupe → annonce. 4+ atouts → annonce toujours. 2 atouts → passe sauf 9+A ou distribution.
2. **Pos2** : plus agressif, 3 atouts suffisent.
3. **Pos3** : être **prudent** — le 9 ne protège pas bien. 4+ atouts ou coupe nécessaire.
4. **Pos4** : annoncer quasi-tout (97%+ avec 2+ atouts).

---

## Méthode

- **Données:** 200k mains aléatoires × 9 scénarios = 1.8M décisions du NN V2
- **SHAP:** Monte Carlo marginal contributions (20k deals, perturbation-based)
- **Modèles proxy:** Decision Tree (depth 5-6, ~93%) + XGBoost (~94%) + TreeExplainer SHAP
- **Fichiers source:**
  - `colver-core/src/bin/distill_bid.rs` — génération CSV (Rust)
  - `scripts/analysis/distill_bid.py` — entraînement des modèles proxy (Python)
  - `scripts/analysis/shap_bid.py` — analyse SHAP (XGBoost + NN direct)
  - `data/distill/bid_distill.csv` — données brutes (7.2M lignes)
  - `data/distill/bid_distill_analysis.log` — log complet avec tous les arbres de décision
