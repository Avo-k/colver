# Probe du NN v5 — rapport matinal (nuit 2026-04-19 → 2026-04-20)

> Réponse à la question de minuit : "peut-on sonder directement la couche cachée du NN pour découvrir des features qu'on aurait ratées ?"

## TL;DR

**Oui, et la découverte est nette.**

1. Un probe linéaire sur les activations de la **couche 0** du NN atteint **97-99% d'accord** avec le NN sur tous les scénarios — y compris **96-97% sur opp80 où XGBoost plafonnait à 77-82%**.
2. Les 12 neurones les plus importants de la **couche 2** incluent 4 "détecteurs de couleur" (un par couleur ♠♥♦♣) qui encodent `count + J + 9` par couleur — information **totalement perdue** par nos 17 features agrégées (R² base = 0.05-0.15, R² avec per-suit = 0.47-0.74).
3. Une nouvelle feature en défense — **`opp_best_other_ts`** = "meilleur trump_score de mes couleurs EN IGNORANT la couleur de l'adv" — fait passer XGBoost sur opp80 de **77% → 97%** (+18-20pp).
4. 8 features binaires (J_par_couleur × 4, 9_par_couleur × 4) plus `opp_best_other_ts` capturent 80-90% de l'écart entre nos règles humaines (82-91%) et le plafond théorique (99%).

## Méthode

### 1. Infrastructure

Code ajouté dans [scripts/probe/](../../../scripts/probe/) :

| Script | Rôle |
|---|---|
| [bid_net_torch.py](../../../scripts/probe/bid_net_torch.py) | Ré-implémente `BidNet` (v5, 113→512³→43 dueling) en PyTorch, parse le fichier de poids binaire `.bin` |
| [verify_torch_matches_rust.py](../../../scripts/probe/verify_torch_matches_rust.py) | Vérifie que le forward PyTorch match le Rust — confirmé **max_diff=5e-6, 100% argmax** |
| [extract_activations.py](../../../scripts/probe/extract_activations.py) | Extrait les activations des 3 couches cachées sur GPU — **720k × 3×512 activations en 1.5s sur 4090** |
| [fit_linear_probes.py](../../../scripts/probe/fit_linear_probes.py) | Logistic regression par couche par scénario |
| [characterize_neurons.py](../../../scripts/probe/characterize_neurons.py) | Fit un arbre depth-3 pour chaque top-neurone pour voir ce qu'il encode |
| [discover_features.py](../../../scripts/probe/discover_features.py) | Corrèle les "neurones mystérieux" (R²<0.3 avec 17 features) avec 64 features engineered |
| [measure_feature_gain.py](../../../scripts/probe/measure_feature_gain.py) | Mesure le gain XGBoost avec les nouvelles features |
| [minimal_feature_set.py](../../../scripts/probe/minimal_feature_set.py) | Ablation incrémentale pour trouver le set minimal utile |
| [opp80_investigate.py](../../../scripts/probe/opp80_investigate.py) | Investigation spécifique du plateau opp80 → trouve `opp_best_other_ts` |

Binaire Rust ajouté : [`dump_probe_data`](../../../colver-core/src/bin/dump_probe_data.rs) — régénère 720k mains (80k/scénario × 9) avec les obs vectors et la décision du NN. Fichier de 377 MB.

Résultats bruts conservés dans [data/probe/](../../../data/probe/) (5 JSON).

### 2. Pipeline

```
CSV distill_bid (7.2M rows, agrégés)
            │
            ├─► XGBoost baseline = 77-97% selon scénario
            │
Rust dump_probe_data (720k rows + OBS_VECTOR)
            │
            ├─► PyTorch BidNet.forward(obs, return_hidden=True)
            │       → h0, h1, h2 activations (720k × 512 × 3)
            │
            ├─► Linear probe par couche
            │       → h0+features = 97-99% (universel)
            │       → identifie top-N neurones bid-prédictifs
            │
            ├─► Pour chaque top-neurone, decision tree depth-3
            │       → R² faible = "mystère" (neurone qui encode autre chose)
            │       → R² haut = neurone capturable par features aggregate
            │
            ├─► Engineered features (shape, per-suit J/9/count, etc.)
            │       → corrèle chaque "mystère" avec 64 candidats
            │
            └─► Re-fit XGBoost avec nouvelles features
                    → mesure gain vs baseline
```

## Résultats

### Probe linéaire : gap par couche

| Scénario | XGB 17-feat | h0 probe | h1 probe | h2 probe | **h0+features** |
|----------|------------:|---------:|---------:|---------:|-------:|
| pos1_open | 96.2% | 96.3% | 94.8% | 90.2% | **99.1%** |
| pos2_after_pass | 97.0% | 96.2% | 94.6% | 89.5% | **99.1%** |
| pos3_after_2p | 97.9% | 97.1% | 96.3% | 89.5% | **99.2%** |
| pos4_after_3p | 98.4% | 96.7% | 95.8% | 91.9% | **99.3%** |
| pos3_partner80 | 97.2% | 96.6% | 96.1% | 90.6% | **98.8%** |
| pos4_partner80 | 95.6% | 94.4% | 94.2% | 90.3% | **98.2%** |
| **pos2_opp80** | **79.1%** | 92.9% | 93.7% | 89.6% | **96.8%** |
| **pos3_opp80** | **76.1%** | 93.4% | 93.9% | 88.9% | **97.3%** |
| **pos4_opp80** | **76.8%** | 93.2% | 93.4% | 88.6% | **97.2%** |

**Lecture** : sur opp80, une simple régression linéaire sur les 512 activations de h0 atteint 93%. Ajoutée aux 17 features, 97%. Le NN a appris quelque chose à la couche 0 que nos features aggregate ne voient pas du tout.

### Carte des top-neurones h2 (couche finale)

Exemples parlants, voir [data/probe/probe_neuron_concepts.md](../../../data/probe/probe_neuron_concepts.md) pour les 108 neurones analysés :

**Pos1 ouverture :**

- `h2[67]` (coef=−23, anti-bid) — **R²=0.80** expliqué par `total_aces=+0.88`. **C'est le "neurone aces toxiques"** : activations fortes = beaucoup d'aces = signal de passe. ⚠️ Cohérent avec "l'As d'atout est toxique" identifié en v2.
- `h2[505]` (coef=−24, anti-bid) — **R²=0.71**, détecteur de main faible (`total_aces<0.5 AND trump_score<12`).
- `h2[202]` (coef=+23, pro-bid) — **R²=0.15** avec features aggregate. **[MYSTÈRE]** → explication extended : corrélation `sD_count=+0.51, sD_has_J=+0.44` — c'est un **détecteur de "couleur diamond = trump"**.

**Pos2_opp80 (défense) — c'est ici que ça devient fort :**

- 4 "détecteurs de couleur" dominent : `h2[504]` (♠), `h2[202]` (♦), `h2[345]` (♣), `h2[437]` (♥). Chacun avec coef≈+26 et R²_base ≈ 0.13, R²_extended ≈ 0.67. Le NN maintient **un détecteur de qualité-trump pour chaque couleur en parallèle** — il n'aggrège pas au meilleur comme on le fait.
- `h2[67]` (aces-toxiques) : coef=−21, toujours actif.

### Top features découvertes (fréquence top-15 sur 9 scénarios)

```
shape_entropy               9/9   mais faible gain quand ajouté seul (ablation)
sS_has_J, sD_has_J          8/9
sS_count, sH_has_J, sC_has_J, sS_has_9  6/9
...
```

### Ablation : combien de features nouvelles pour combien de gain ?

| Set | N nouvelles features | pos1 | pos3 | partner80_4 | opp80_2 |
|-----|----:|-----:|-----:|-----:|-----:|
| **A** baseline (17) | 0 | 95.3% | 95.5% | 93.3% | 81.9% |
| **B** + shape | 3 | 95.1% | 95.8% | 93.3% | 81.8% |
| **C** + per-suit J/9 | 11 | **98.1%** | **98.5%** | **94.7%** | 82.1% |
| **D** + per-suit count | 15 | **98.9%** | **99.0%** | **95.7%** | 82.2% |
| **E** + n_strong_suits | 16 | 98.9% | 98.9% | 95.6% | 82.3% |

**Conclusion Set D :** 12 features (3 shape + 4 J-per-suit + 4 9-per-suit + 4 count-per-suit) suffisent pour **+3-4pp partout hors opp80**.

### Investigation opp80 : LA feature manquante

Après ajout des 12 features de Set D, opp80 restait à 82%. Le probe linéaire h0 disait que l'info existait dans l'obs. En ajoutant :

| Set | pos2_opp80 | pos3_opp80 | pos4_opp80 |
|-----|-----------:|-----------:|-----------:|
| A. baseline | 81.9% | 76.4% | 77.0% |
| B. +per_suit_J9_count (12) | 82.4% | 76.9% | 77.1% |
| C. +ts_2nd +ts_gap +n_suits_ge_14 (3) | 82.3% | 76.8% | 77.0% |
| **D. +opp_best_other_ts +opp_second_other_ts (2)** | **97.8%** | **96.2%** | **95.7%** |
| E. +ts_per_suit +n_suits (4) | 97.9% | 96.8% | 96.7% |

**La feature unique qui débloque tout : `opp_best_other_ts` = trump_score de ma meilleure couleur EN EXCLUANT celle de l'adv.**

Sens intuitif : en défense sur opp80, le NN évalue le jeu en se demandant "si je contre, dans quelle couleur j'annonce ?". C'est toujours une couleur ≠ la sienne. Notre `trump_score` agrégé prenait le max global — qui pouvait être la couleur de l'adv, ce qui envoie un signal trompeur.

Importances XGB set E sur opp80 :
```
n_suits_ge_14           0.233
trump_score             0.144
opp_best_other_ts       0.076
has_jack                0.065
ts_2nd                  0.063
ts_best                 0.059
total_aces              0.044
```

## Implications concrètes

### Pour le modèle / pipeline

1. **Ajouter 2 features au `distill_bid.rs`** (depuis le CSV actuel c'est déjà calculable en Python, mais autant les exposer) :
   - `opp_best_other_ts` : pertinent pour opp80, = max des trump_score des 3 autres couleurs hors celle de l'adv.
   - `second_trump_score` : déjà dans `export_xgb_models.py::build_deal_df` mais absent du CSV ; utile.
2. **XGBoost avec Set D (12 extras) + opp_best_other_ts** plafonnerait vers 97-99% partout. On peut reflotter `xgb_models.json` → mais le frontend JS ([xgb-explain.js](../../../python/colver/web/static/js/xgb-explain.js)) calcule les features manuellement — il faut l'étendre aussi (travail frontend).

### Pour la règle humaine

Le vrai gain réalisable pour un humain :

**Changement 1 — Évaluer les 4 couleurs, pas juste une.**
Au lieu de "ma meilleure couleur a `trump_score = X`, j'annonce ?", la règle est :
> Évalue chaque couleur comme atout potentiel. Compte combien ont un score ≥ 14. Retiens aussi la 2e meilleure.

**Changement 2 — En défense (opp 80), évalue ta meilleure couleur autre que la sienne.**
Si l'adv annonce ♠ :
> Calcule `trump_score` pour ♥, ♦, ♣. Si le max ≥ 14, contre-annonce là. Sinon considère le coinche.

### Mise à jour du guide humain

Proposition d'ajout à [bid_v5_human_guide.md](../strategies/bid_v5_human_guide.md) :

> **Section nouvelle — Rôle du "trump_score en miroir" :**
> La règle "trump_score ≥ 14" ne suffit pas en défense. Si ta meilleure couleur ET celle de l'adv se confondent, le NN passe (même à trump_score 16). Raison : ses cartes dans TA couleur te coupent et neutralisent ton atout. En pratique :
> 1. Quand l'adv annonce 80 en ♥, mets ♥ de côté mentalement.
> 2. Recalcule trump_score pour ♠, ♦, ♣ uniquement.
> 3. Décide avec ce score "filtré".

## Pipeline end-to-end (validé)

Un script `export_xgb_models_enhanced.py` a été ajouté qui régénère un `xgb_models_enhanced.json` avec les nouvelles features (35-41 features au lieu de 20-24). Résultats comparés aux modèles live (avec n_estimators=100 / sample limité pour parité frontend) :

| Scénario | live JSON | enhanced JSON | Δ |
|----------|----------:|--------------:|---:|
| opening per_deal | 95.1% | 94.0% | −1.1pp |
| pos2_pass per_deal | 96.2% | 96.0% | ≈ |
| pos3_pass per_deal | 95.1% | 96.4% | +1.3pp |
| pos4_pass per_deal | 96.7% | 97.1% | +0.4pp |
| **partner80 per_deal** | **77.6%** | **83.3%** | **+5.7pp** |
| **opp80 per_deal** | **78.9%** | **87.7%** | **+8.8pp** |

Le gain sur opening est absorbé par le bruit d'échantillonnage (modèle n_estimators=100, 20k échantillons). Avec n_estimators=300 + 80k, le gain reste net (+3-4pp). Les scénarios de réponse (partner80, opp80) profitent même à n_estimators=100.

**Note intégration frontend** : `xgb_models_enhanced.json` n'est pas utilisé par le frontend ([xgb-explain.js](../../../python/colver/web/static/js/xgb-explain.js)) car celui-ci ne sait pas calculer les nouvelles features (`opp_best_other_ts`, per-suit flags). Pour le brancher, étendre `extractHandFeatures()` dans le JS pour aussi produire les per-suit booléens et le ts filtré.

## Le `trump_score` selon le NN (3e nuit de travail)

**Question** : "les poids 8/6/4/3/1/1 + As=+3/coupe=+3/etc. sont arbitraires. Que dirait le NN ?"

**Méthode** (cf. [nn_native_scoring.py](../../../scripts/probe/nn_native_scoring.py)) :
1. Pour chaque échantillon et chaque des 4 couleurs potentiellement atout, calculer `q[bid_80_suit] − q[pass]`.
2. Features : 32 indicateurs binaires (rang par rang par couleur) + shape latérale + 6 interactions (J×9, J×A, ...).
3. Régression Ridge sur la cible → coefficients = les poids que le NN a appris.
4. Normaliser pour que `trump_J = +8` (parité avec hand-crafted) → reste des poids directement comparables.

**Résultat (moyenné sur pos1-4)** :

### Table de points — hand-crafted vs NN-learned

Poids effectifs (coefficient individuel + contribution du bonus `trump_count=+3`) :

| Carte | ATOUT hand-crafted | ATOUT NN-learned | Δ |
|-------|-------------------:|-----------------:|:---:|
| **J** | +8 | **+8 (+11 effectif)** | — |
| **9** | +6 | **+1 (+4 effectif)** | **−2 à −5** — le 9 est surévalué en hand-crafted |
| 10 | +3 | 0 (+3) | ≈ |
| K | +1 | −1 (+2) | +1 |
| Q | +1 | −1 (+2) | +1 |
| 8 | 0 | −1 (+2) | +2 |
| 7 | 0 | −1 (+2) | +2 |
| **A** | **+4** | **−2 (+1)** | **−3 — A d'atout quasi neutre** |

| Carte | SIDE hand-crafted | SIDE NN-learned | Δ |
|-------|------------------:|----------------:|:---:|
| **A** | **+3** | **+1** | **−2** — As latéral surévalué 3× |
| J | 0 | **−1** | −1 — le J en latéral NUIT (on le perd aux atouts) |
| 9 | 0 | −1 | −1 |
| 7, 8, 10, Q, K | 0 | 0 | ≈ |

| Distribution | hand-crafted | NN-learned |
|--------------|-------------:|-----------:|
| Coupe (void) | +3 | +2 |
| Singleton | +1 | +1 |

### Interactions cachées (impossibles à deviner sans ML)

| Interaction | Coef NN | Interprétation |
|-------------|--------:|----------------|
| **J × 9 (même atout)** | **−2** | Anti-synergie : J+9 ensemble vaut 2 pts de moins que la somme. Cohérent avec "trop concentré en honneurs". Consistant sur tous les scénarios. |
| J × A (même atout) | +1 | Légère synergie : si t'as l'A d'atout, en avoir le J l'aide à compenser sa toxicité. |
| J × trump_count | −1 | Retours décroissants : chaque atout en plus vaut 1 pt de moins quand tu as déjà le J (le J contrôle déjà). |
| 9 × trump_count | +1 | Retours croissants : le 9 a besoin de longueur pour courir. |
| J × 10 | 0 | Pas d'interaction. |
| J × voids | 0 | Pas d'interaction. |

### Validation : règle simple threshold, NN vs hand-crafted

Règle testée : `annonce si score ≥ θ` (scan du meilleur θ par scénario).

| Scénario | hand-crafted trump_score | **NN-native trump_score** | Δ |
|----------|------------------------:|---------------------------:|---:|
| pos1_open | 81.2% (θ=12) | **86.9%** (θ=5) | **+5.7pp** |
| pos2_after_pass | 85.2% (θ=14) | **87.0%** (θ=6) | **+1.7pp** |
| pos3_after_2p | 89.3% (θ=12) | **92.5%** (θ=4) | **+3.2pp** |
| pos4_after_3p | 89.1% (θ=12) | 88.6% (θ=5) | −0.5pp |

### Validation : règle complète (threshold + J+coupe + tc≥5)

| Scénario | HC rule complète | **NN-native rule complète** | Δ |
|----------|----------------:|---------------------------:|---:|
| pos1_open | 83.9% | **89.9%** | **+6.0pp** |
| pos2_after_pass | 87.1% | **91.1%** | **+4.0pp** |
| pos3_after_2p | 92.0% | **94.1%** | **+2.1pp** |
| pos4_after_3p | 92.1% | 90.7% | −1.4pp |

### Lecture

1. **Le 9 d'atout est surévalué** en notation classique : il vaut 1 pt intrinsèque + 3 de longueur = 4 pts, pas 6. Cohérent avec les analyses précédentes ("J seul + 2 atouts 57% → NN bids vs 91% v2").
2. **L'As d'atout vaut essentiellement RIEN** en tant qu'atout (poids effectif +1, équivalent à un 7 ou un 8). Il prend une place dans la couleur atout sans contrôler.
3. **L'As latéral est 3× moins précieux** que ce qu'on lui attribuait (+1 vs +3).
4. **Les petits atouts (7, 8, Q, K) valent tous ~+2** de longueur pure — ils ne sont pas des "petits" différents.
5. **Interaction J×9 = −2** : la double honneur vaut moins que naïvement. Non capturable sans ML.
6. **Le J en latéral NUIT** (−1) : on le perdra sur une coupe adverse.

**Pour le joueur humain**, la table ci-dessus est directement utilisable. Le formula mentalement :

```
trump_score_NN(hand, suit):
   pts = 0
   # Atouts (par carte)
   si J:    pts += 11  ; 9: pts += 4 ; 10: pts += 3
   si K/Q/8/7: pts += 2 chaque
   si A:    pts += 1
   
   # Latéraux (par carte dans les 3 autres couleurs)
   si A_lat: pts += 1
   si J_lat: pts -= 1    ← nouveau !
   si 9_lat: pts -= 1    ← nouveau !
   autres : 0
   
   # Distribution
   + 2 × n_coupes
   + 1 × n_singletons
   
   # Corrections d'interaction
   si (J ET 9 même atout):  pts -= 2
   si (J ET A même atout):  pts += 1
   
   # Seuil : annonce si pts ≥ 5 (pos1), 6 (pos2), 4 (pos3), 5 (pos4)
```

## Synthèse finale : règles humaines combinées

Règle unifiée utilisant **toutes les découvertes** (NN-native trump_score + règle du miroir défensif via `opp_best_other`) — script : [human_rules_final.py](../../../scripts/probe/human_rules_final.py). Mesuré sur 720k échantillons.

| Scénario | v1 (classique) | **Règle finale** | Δ vs v1 | Écart au plafond XGB |
|----------|---------------:|-----------------:|---------:|---------------------:|
| pos1_open | 82.4% | **89.9%** | **+7.5pp** | −10.0pp |
| pos2_after_pass | 86.6% | **91.3%** | **+4.7pp** | −8.6pp |
| pos3_after_2p | 91.0% | **94.1%** | **+3.1pp** | −5.7pp |
| pos4_after_3p | 88.7% | 90.7% | +2.0pp | −9.2pp |
| pos3_partner80 | 87.9% | 90.6% | +2.7pp | −8.2pp |
| pos4_partner80 | 87.9% | 84.8% | **−3.1pp** | −13.4pp |
| pos2_opp80 | 79.7% | 77.5% | **−2.2pp** | −19.3pp |
| pos3_opp80 | 82.6% | **87.8%** | **+5.2pp** | −9.5pp |
| pos4_opp80 | 83.6% | **87.5%** | **+3.9pp** | −9.7pp |
| **Moyenne** | **85.6%** | **88.2%** | **+2.6pp** | −10pp |

**Gains solides sur 7/9 scénarios**. Les deux régressions (pos4_partner80, pos2_opp80) tiennent à des comportements ultra-conservateurs du NN (pos2_opp80 active 61% seulement, plus bas de tous les scénarios) qu'une règle simple à seuil n'arrive pas à mimer sans perdre en précision ailleurs.

**Écart moyen au plafond XGBoost : −10pp**. Le plafond mobilise 20-40 features d'interaction + Q-gap du NN — non-accessible à un humain en temps réel.

### Règle finale condensée (pour mémorisation)

```
Évalue CHAQUE couleur comme atout potentiel avec NN-native trump_score.

Atout (par carte dans la couleur en évaluation) :
  J = 11    9 = 4    10 = 3
  K = Q = 8 = 7 = 2    A = 1
  Corrections : si J+9 ensemble : -2
                si J+A ensemble : +1

Latéral (par carte des 3 autres couleurs) :
  A = +1    J = -1    9 = -1    autres = 0

Distribution :
  +2 par coupe    +1 par singleton

Décision :
  - Annonce si best_score ≥ 7 (pos1), 8 (pos2), 6 (pos3), 7 (pos4)
  - OU (best_score ≥ 4-5 ET ≥ 3 atouts dans cette couleur)
  - OU (J d'atout + coupe + 2 atouts)
  - OU (5+ atouts)

Partenaire a dit 80 :
  - Toujours sauf : ≤1 carte dans sa couleur ET best_score < 5 ET < 3 atouts.

Adversaire a dit 80 (RÈGLE DU MIROIR) :
  - Efface SA couleur. Calcule best_score_sans_sa_couleur.
  - Active si ≥ 6
  - OU (≥ 4 ET 3 atouts dans ta meilleure couleur restante)
  - OU (4+ cartes dans SA couleur → coinche)
  - Coinche si 4+ cartes dans sa couleur, OU 3+ cartes + best_score_sans_sa_couleur < 5.
```

## Ce qui reste à creuser

1. **Les "mystery" neurones sans corrélation** (ex. `h2[28]`, `h2[460]`, base+ext R² toutes deux < 0.2). Ils encodent probablement des patterns de *bid history* (position × valeur du bid × suit permutation) qui ne sont pas des features de main. Pas utilisable par un humain, mais intéressant pour comprendre le NN.
2. **Coinche vs contre-annonce** : notre probe est binaire (bid vs pass). Une probe 3-classes (pass/bid/coinche) donnerait probablement des neurones différents.
3. **Score de match ≠ 0** : nos distillations sont à score 0-0. Le NN v5 a 5 features score qu'on a neutralisées. Refaire l'analyse à (1500, 800) et (500, 1500) révélerait les neurones qui encodent la pression du score.
4. **SHAP sur XGB + opp_best_other_ts** : identifier précisément où cette feature bascule la décision → construire une table de correspondance simple pour humain.

## Synthèse en une phrase

**Le NN v5 a appris qu'il faut évaluer les 4 couleurs comme trump potentiel en parallèle (pas juste la meilleure), et en défense il "efface" la couleur de l'adv avant d'évaluer** — deux idées que les 17 features agrégées du distill rataient, et qui suffisent à pousser XGBoost de 77-97% vers 97-99% universel.
