# Bid Rules — Distilled from NN V5 (champion, 2026-04-19)

Rules extracted by training interpretable models (Decision Trees depth 5-6, XGBoost depth 5 / 300 trees) on 200,000 random deals per scenario, evaluated by the **bid_v5_isdd** NN (512-hidden, 3-layer dueling DQN, 113-dim score-aware v2 obs, 25M training steps on IS-DD-only reward pool). Distillation performed at **neutral match score** (my_score = opp_score = 0).

**Accuracy of the distilled rules vs the NN:** XGBoost per-deal: 92% opening, 94-97% passes, 80% partner/opp responses.

For the historical v2 analysis (Bid a Dede, DD oracle reward, pre-rule-change), see [bid_rules_xgb_v2.md](bid_rules_xgb_v2.md).

## TL;DR — What changed between v2 and v5

Three concurrent shifts happened between v2 (end of March 2026) and v5 (April 2026):

1. **Reward signal:** DD oracle Q-values → real DMC/IS-DD points (v5_max uses `max(DMC, ISDD)`, v5_isdd — production — uses IS-DD only).
2. **Score context:** 5 precomputed features `(s_me/2000, s_opp/2000, win_prob, leader_dist, diff)` appended to the 108-dim base obs.
3. **Scoring rules** (FFB update 2026-04-16): surcoinche ×3 (was ×4), contré/surcontré base 160 + contrat×mult (was 320/640 + contrat×mult), capot is a regular 250-contract (was flat 500/1000/2000).

The distilled rules show **three qualitative shifts**:

- **Hand strength dominates single features.** In v2, `has_jack` alone drove 47.6% of the opening decision. In v5 it's `trump_score` (37.2%) with `has_jack` at 14.2% — the network uses the holistic evaluator as its primary signal instead of gating on one card.
- **Much more selective opening on marginal hands.** J alone + 2 atouts went from 91% bid → **57% bid**. 9 alone + 2 atouts went from 10% → **1%**. v5 pass-through on weak trump is almost total.
- **Stronger partnership in response.** Holding 3+ cards in partner's suit: v2 91%, **v5 99.3%**. Partner support is now almost automatic when long.

Defense is largely unchanged: coinche rate 17% → 15%, same trump profile (~2 atouts, 3.5 cartes dans la couleur adverse). The main shift is within defense: pos3 coinche rate dropped (28% → 18%), pos4 rose (11% → 16%).

---

## 1. Ouverture (Position 1)

**v5: 75.8% annonce, 24.2% passe** (v2: 80% / 20%).

### Importance des features (XGBoost per-deal)

| # | Feature (v5) | v5 imp | v2 imp | Δ |
|---|--------------|-------:|-------:|---|
| 1 | **trump_score** | **37.2%** | — | — (pas top-5 en v2) |
| 2 | has_jack | 14.2% | **47.6%** | −33 pp |
| 3 | trump_count | 13.3% | 12.1% | +1 |
| 4 | trump_points | 11.3% | — | — |
| 5 | has_nine | 7.1% | 13.3% | −6 pp |
| 6 | has_ace | 6.4% | 7.8% | −1 |

**Lecture :** le v5 ne gate plus sur "ai-je le Valet ?" en premier. Il lit une évaluation continue (`trump_score` = combine J/9/A, bonus longueur, coupes/as latéraux) puis affine avec `has_jack`. Cohérent avec son entraînement sur points réels : moins d'asymétrie 0/1 à apprendre.

### Table de référence par composition d'atout (pos1)

| Atout | 1 carte | 2 cartes | 3 cartes | 4 cartes | 5+ cartes |
|-------|--------:|---------:|---------:|---------:|----------:|
| **J + 9** | — | 96% (v2 99%) | 100% | 100% | 100% |
| **J seul** | 2% (v2 **28%**) | **57%** (v2 **91%**) | 98% (v2 99%) | 100% | 100% |
| **9 seul** | — | **1%** (v2 10%) | 65% (v2 73%) | 98% (v2 93%) | 100% |
| **Ni J/9** | — | 0% (v2 1%) | 5% (v2 9%) | **64%** (v2 47%) | 100% (v2 94%) |

**Deux shifts opposés :**
- **Plus prudent sur les mains courtes J/9-only** : le v5 sait qu'une main "J seul + 2 atouts" (57%) doit être filtrée par `trump_score` — sans as latéraux ni coupes c'est un piège.
- **Plus agressif sur la longueur sans honneur** : ni J ni 9 + 4 atouts passe de 47% à 64%. La longueur pure suffit désormais.

### Seuils trump_score (pos1)

| trump_score | v5 bid rate |
|-------------|-------------|
| [0, 5)  | 0% |
| [5, 10) | 15% |
| [10, 14) | 49% |
| [14, 17) | 73% |
| [17, 20) | 92% |
| [20, 25) | 99.4% |
| [25+)    | 100% |

Le "coude" est autour de `trump_score = 12` (50% bid). En v2 le coude dépendait fortement de `has_jack` plus que du score composite.

---

## 2. Après passes (Positions 2-4)

**v5: 80.6% annonce** (v2 combined: ~78%). Globalement similaire, mais redistribution interne.

### Évolution par position

| Position | v2 bid | v5 bid | Δ |
|----------|-------:|-------:|---|
| pos2 (1 passe) | 66% | **74.4%** | +8 pp (plus agressif) |
| pos3 (2 passes) | **95%** | 86.7% | −8 pp (moins agressif) |
| pos4 (3 passes) | 75% | 80.9% | +6 pp (plus agressif) |

**Lecture :** le v5 a rééquilibré. v2 était quasi-obligatoire en pos3 ("protect"), le v5 sait qu'annoncer aux cartes faibles en pos3 coûte plus que ça ne rapporte — même position 3 mérite un minimum de jeu.

### Tables de référence par position (v5)

| Atout | pos1 | pos2 | pos3 | pos4 |
|-------|-----:|-----:|-----:|-----:|
| **J + 9** (2 atouts) | 96% | 97% | 100% | 99% |
| **J seul** (1 atout) | 2% | 2% | **20%** | 10% |
| **J seul** (2) | 57% | 52% | **89%** | 70% |
| **J seul** (3) | 98% | 99% | 100% | 100% |
| **9 seul** (2) | 1% | 2% | **22%** | 15% |
| **9 seul** (3) | 65% | 57% | 85% | 69% |
| **9 seul** (4) | 98% | 99% | 100% | 100% |
| **Ni J/9** (3) | 5% | 5% | **24%** | 20% |
| **Ni J/9** (4) | 64% | 55% | 92% | 76% |

Position 3 reste la plus offensive sur les mains très faibles (le "sauvetage" de la donne), mais bien moins systématique qu'en v2.

Position 2 reste plus stricte que pos1 (seuils similaires en v5 vs pos1, cohérent avec v2 : "l'adversaire suivant peut encore enchérir").

---

## 3. Réponse au partenaire (Partner bid 80)

**v5: 87.3% annonce** (v2: 82%). Globalement plus actif, mais surtout beaucoup plus **coopératif**.

### Importance des features

| # | Feature | v5 imp | v2 imp |
|---|---------|-------:|-------:|
| 1 | **is_partner_suit** | 36.5% | 41.8% |
| 2 | trump_score | 18.8% | 19.8% |
| 3 | trump_count | 12.3% | 6.7% |
| 4 | has_jack | 7.6% | 12.5% |
| 5 | partner_support | 3.9% | 3.7% |
| — | total_aces | 5.4% | — |

Redistribution : `trump_count` double d'importance (6.7% → 12.3%) parce que le v5 reconnaît mieux quand supporter avec de la longueur, et `has_jack` perd du poids (12.5% → 7.6%) — la possession du Valet en réponse n'est plus la seule raison de monter.

### Support par nombre de cartes dans la couleur du partenaire

| Cartes dans sa couleur | v2 bid | v5 bid | Δ |
|------------------------|-------:|-------:|---|
| 0 | 87% | 92% | +5 |
| 1 | **71%** | **74%** | +3 |
| 2 | 79% | 86% | +7 |
| 3 | 91% | **99.3%** | **+8** |
| 4 | 97% | 99.9% | +3 |
| 5 | 99% | 99.2% | — |

**Shift majeur :** avec 3 cartes du partenaire, le v5 soutient quasi-systématiquement (99%). En v2 il y avait 9% de passes / contre-annonces dans cette config. Le v5 lit ça comme "on va gagner ce contrat ensemble" et monte.

Le creux à 1 carte subsiste (moins que 0 ou 2) — c'est une main où ni le support ni le contre-annonce ne sont clairement bons.

---

## 4. Défense (Adversaire bid 80)

**v5: 58.5% contre-annonce, 14.9% coinche, 26.6% passe** (v2: 55% / 17% / 28%).

Le volume d'action reste quasi-identique (73% actif vs 72% en v2). Mais la **distribution par position change** :

| Position | v5 coinche | v2 coinche | Δ |
|----------|----------:|----------:|---|
| pos2_opp80 | ~11% | 11% | 0 |
| pos3_opp80 | **~18%** | **28%** | **−10 pp** |
| pos4_opp80 | ~16% | 11% | +5 pp |

**pos3 défense était sur-coinche en v2.** Avec des points réels (IS-DD au lieu de DD oracle), le v5 a réalisé que le coinche en pos3 sur adversaire qui ouvre à 80 est souvent perdant — le partenaire n'a pas signalé et on joue avec peu d'information. Il préfère maintenant contre-annoncer dans sa couleur (niveaux 90-100 fréquents).

### Importance des features

| # | Feature | v5 imp | v2 imp |
|---|---------|-------:|-------:|
| 1 | **is_opp_suit** | **32.7%** | 8.8% |
| 2 | trump_score | 31.6% | 5.1% |
| 3 | has_jack | 11.2% | 7.3% |
| 4 | trump_count | 7.2% | — |
| 5 | has_ace | 4.4% | — |
| — | opp_suit_cards | (top-10) | 24.7% |
| — | side_voids | (top-10) | 16.0% |
| — | best_side_length | 4.3% | 10.9% |

**Shift majeur :** le v5 regarde d'abord **la couleur de l'adversaire** (est-ce la mienne ?) puis son propre `trump_score`. `opp_suit_cards` (combien de cartes j'ai dans sa couleur) perd du poids relatif : en v2 c'était LE signal du coinche, en v5 c'est intégré dans le mix.

### Profil du coinche (v5)

| Mesure | v5 | v2 |
|--------|---:|---:|
| Taux global (opp80) | 14.9% | 17% |
| Avg trump_score | 11.5 | 11.8 |
| Avg trump_count | 2.0 | 2.0 |
| % has J | 29.1% | 29% |
| % has 9 | 26.3% | 25% |
| Avg cards in opp suit | 3.5 | 3.3 |

**Quasi-identique.** Le v5 coinche dans les mêmes situations que v2, juste légèrement moins souvent et à des positions redistribuées.

---

## 5. Niveaux d'annonce

Distribution sur scénario `opp80` (défense) avec contre-annonce (v5) :

| Position | Niveau dominant | 90% | 100 | 110 | 120+ |
|----------|----------------:|----:|----:|----:|-----:|
| pos2_opp80 | 90 | 70% | 16% | 13% | <1% |
| pos3_opp80 | 90 | 50% | 26% | 22% | 2% |
| pos4_opp80 | 90 | 65% | 28% | 10% | 1% |

Le 90 reste le niveau-par-défaut en défense. Les contres à 110 sont plus fréquents en pos3 (22%) : quand le v5 n'a pas coinché mais a une main qui surpasse clairement, il monte plus haut.

---

## 6. Limites de cette analyse

- **Score neutre.** Toutes les évaluations sont faites avec `my_score = opp_score = 0`. Le v5 possède 5 features score qui sont neutralisées ici. En fin de match (par ex. 1800-1400), le comportement peut diverger sensiblement — ce cas n'est pas couvert. Pour explorer : `./target/release/distill_bid models/bid_v5_isdd/bid_nn_final.bin 50000 out.csv 1800 1400`.
- **Pas de SHAP.** Le doc v2 ([bid_rules_xgb_v2.md](bid_rules_xgb_v2.md) §6-7) contient une analyse SHAP / Monte Carlo détaillée (As toxique, 10 poids mort, belote, combos) non re-faite ici. La plupart des conclusions qualitatives restent valables (l'As d'atout n'est toujours pas un gros facteur positif dans les importances v5), mais les magnitudes numériques datent de v2.
- **Scénarios fixés.** Seuls 9 scénarios sont générés (pos1_open, pos2-4 after passes, partner80, opp80 × 4 suits). Pas de situations post-coinche, post-surcoinche, ou après deux bids adverses.

---

## 7. Méthode

- **Données:** 200k mains aléatoires × 9 scénarios = 7.2M lignes (4 suits × 1.8M décisions du v5)
- **Modèle cible:** `models/bid_v5_isdd/bid_nn_final.bin` (113-dim score-aware v2 obs)
- **Modèles proxy:** Decision Tree (depth 5-6, min_samples_leaf 200-300, class_weight balanced), XGBoost (n_estimators 200-300, max_depth 4-5, learning_rate 0.1-0.15)
- **Fichiers source:**
  - `colver-core/src/bin/distill_bid.rs` — génération CSV (adapté v5: dispatch sur obs_dim 108/110/113)
  - `scripts/analysis/distill_bid.py` — arbres et tables (log auto-nommé depuis le stem du CSV)
  - `scripts/export/export_xgb_models.py` — JSON d'interprétabilité pour le web frontend
  - `data/distill/bid_v5_distill.csv` — données brutes (1 GB)
  - `data/distill/bid_v5_distill_analysis.log` — log complet avec arbres de décision et importances
  - `python/colver/web/static/data/xgb_models.json` — modèles exportés pour SHAP path-based

### Commandes de reproduction

```bash
# 1. Générer le CSV de distillation (~45 min, mono-thread)
cargo build --release --bin distill_bid
./target/release/distill_bid models/bid_v5_isdd/bid_nn_final.bin 200000 data/distill/bid_v5_distill.csv

# 2. Analyser et extraire les tables / arbres
uv run python scripts/analysis/distill_bid.py data/distill/bid_v5_distill.csv

# 3. Exporter les modèles pour le frontend web
uv run python scripts/export/export_xgb_models.py data/distill/bid_v5_distill.csv
```

Pour explorer un autre état de score, passer `my_score` et `opp_score` en 4e/5e args du binaire Rust (par ex. `... out.csv 1800 1400`).
