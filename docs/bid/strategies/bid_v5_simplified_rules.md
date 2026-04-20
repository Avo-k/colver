# Bid v5 — règles ultra-simplifiées (compression de XGBoost)

> Comment loin peut-on aller avec une règle minimaliste tout en restant proche
> du NN ? Ce doc répond en mesurant précisément le trade-off simplicité ↔ accuracy.

## TL;DR

Un **seul arbre de décision de profondeur 3** utilisant **5 features** atteint **88-91% d'accord avec le NN v5**, contre 95-97% pour XGBoost complet (300 arbres, 17 features). L'écart de 5-7pp est littéralement "ce qu'un humain ne peut pas faire à la main" : moyennage de 300 arbres, interactions à 5 variables, seuils à 0.5 près.

Les 5 features qui comptent :
1. **`nn_best`** — trump_score NN-native de la meilleure couleur
2. **`hc_best`** — trump_score classique de la meilleure couleur (non-redondant avec 1)
3. **`tc_best`** — nombre d'atouts dans cette couleur
4. **`has_jack`** — J d'atout dans cette couleur
5. **`n_voids`** — nombre de coupes (couleurs latérales vides)

Les features suivantes ont été **éliminées sans perte** de la baseline XGBoost (toutes < 0.01 importance) :

```
n_suits_ge_5, n_suits_ge_8, shape_l1, shape_l4, nn_2nd, nn_3rd
```

## Trade-off simplicité ↔ accuracy (pos1-pos4)

| Configuration | pos1 | pos2 | pos3 | pos4 | # features | complexité humaine |
|---------------|-----:|-----:|-----:|-----:|:---:|:---:|
| XGBoost full (17 feats, depth=5, 300 arbres) | 95.2% | 94.9% | 95.9% | 96.6% | 17 | irréalisable |
| XGBoost minimal (~9 feats) | 95.1% | 94.5% | 95.5% | 96.2% | ~9 | irréalisable |
| XGBoost (9 feats, depth=4, 100 arbres) | 94.9% | 94.9% | 94.9% | 95.5% | 9 | irréalisable |
| **XGBoost (9 feats, depth=2, 50 arbres)** | 90.5% | 92.1% | 92.9% | 92.5% | 9 | en théorie simulable (avg de 50 petits arbres) |
| **1 arbre depth-3 (5 feats)** | **87.6%** | **91.3%** | **89.9%** | **91.2%** | 5 | ~7 branches if-else → **mémorisable** |
| 1 arbre depth-2 (5 feats) | 83-90% | 89-92% | 89-91% | 88-90% | 5 | 3 branches → ultra-simple |

**Sweet spot pour un humain : 1 arbre depth-3** (91% sur pos2 et pos4, 88-90% sur pos1/3). Les règles ci-dessous.

## Les règles if-else directement utilisables

### Position 1 (ouverture) — 87.6% d'accord

```
Calcule nn_best.
  si nn_best ≤ 4      → PASSE
  si nn_best entre 5-8:
      si pas de J d'atout dans la meilleure couleur → ANNONCE (63% du temps)
      si J présent → PASSE (25% — le J seul ne suffit pas)
  si nn_best ≥ 9      → ANNONCE (>89% du temps)
```

### Position 2 (après 1 passe) — 91.3% d'accord

```
Calcule nn_best et hc_best.
  si nn_best ≤ 8:
      si hc_best > 14 ET nn_best > 4  → ANNONCE
      sinon                             → PASSE
  si nn_best ≥ 9                        → ANNONCE
```

### Position 3 (après 2 passes) — 89.9% d'accord

```
Calcule nn_best et hc_best.
  si nn_best ≤ 6:
      si hc_best > 14 ET nn_best > 4  → ANNONCE
      sinon                             → PASSE
  si nn_best ≥ 7                        → ANNONCE
```

Pos3 a le seuil le plus bas car c'est la position "protect" (sinon la donne meurt).

### Position 4 (après 3 passes) — 91.2% d'accord

```
Calcule nn_best et hc_best.
  si hc_best ≤ 14:
      si hc_best ≤ 11                  → PASSE
      si hc_best 12-14 ET nn_best ≥ 8  → ANNONCE
      sinon                             → PASSE
  si hc_best > 14:
      si nn_best ≤ 4 ET hc_best ≤ 16   → PASSE (main "flashy" fausse)
      sinon                             → ANNONCE
```

## Ce que XGBoost fait et que l'humain ne peut pas (les 5-7pp manquants)

Les 5-7pp entre arbre unique depth-3 (91%) et XGBoost complet (95-97%) viennent de trois sources génuinement inhumaines :

### 1. Moyennage de 300 arbres

Chaque arbre de XGBoost attrape un pattern local différent (surtout via boosting : chaque arbre corrige les erreurs des précédents). Le vote pondéré final = 300 × rules, impossible à tenir mentalement. Passer de 1 → 50 arbres = +2-3pp, de 50 → 300 = +1-2pp.

### 2. Interactions à ≥ 4 variables simultanées

Un arbre de profondeur 5 peut dire : "si (nn_best ∈ [6,9]) ET (hc_best ≥ 15) ET (tc_best = 3) ET (has_J = 1) ET (n_voids = 0) → bid avec proba 62%". Un humain peut garder 2-3 conditions en tête, pas 5. Les arbres depth-3 (notre limite lisible) ratent ces raffinements.

### 3. Seuils fins (12.5, 13.5, 14.5)

XGBoost split à `nn_best ≤ 8.5` puis `≤ 9.5` dans différents arbres. Un humain arrondit à 10 ou 5. Chaque demi-point de précision vaut ~0.2-0.5pp sur l'accuracy finale.

**Ce que l'humain ne perd PAS en simplifiant** :
- Les règles monotoniques (score élevé → annonce)
- Les seuils ronds (5, 10, 15)
- Les conditions binaires (a le J, a une coupe)

## Feature importance résumée (toutes scénarios)

Classement moyen après 4 scénarios pos1-pos4 :

```
1. nn_best             40-50%   (dominant)
2. hc_best             10-15%   (non-redondant avec 1 : les deux scores capturent
                                  des aspects différents — erreurs décorrélées)
3. tc_best             5-10%
4. has_jack / has_nine 3-5%
5. has_ace             1-3%
6. n_J_in_hand         2-4%   (nombre de suits où on a le J — inutile en v1)
7. n_voids             1-3%
8. n_singletons        1-2%
9. hc_2nd              1-3%   (2e meilleur hand-crafted — utile quelquefois)
10. nn_2nd             1-2%
11+ (shape_*, n_suits_ge_*)  ≈ 0
```

### Pourquoi garder `hc_best` alors qu'on a `nn_best` ?

C'est l'observation la plus contre-intuitive. On pourrait penser que `nn_best` (appris par le NN) remplace complètement `hc_best`. En fait non — les deux scores pondèrent différemment :
- `hc_best` donne beaucoup de poids au 9 (+6), à l'As atout (+4), à l'As latéral (+3).
- `nn_best` donne peu au 9 (effectif +4), presque rien à l'As atout (+1), peu à l'As latéral (+1).

Les **mains de type "flash"** (J + plusieurs As + main plate) ont `hc_best` élevé mais `nn_best` modeste. Les **mains "distributives"** (J + longueur + coupes) ont les deux scores élevés. Combiner les deux = savoir si la main est *vraiment* forte ou juste flashy.

**C'est pour ça que les arbres utilisent les deux** : un joueur humain pourrait faire la même chose en calculant les deux scores et en ne bidant que si les DEUX sont décents.

## Règle ULTRA-simple (si tu veux encore plus compact, ~85% d'accord)

Si tu ne veux qu'une seule feature et un seul seuil :

```
Évalue chaque couleur avec nn_trump_score (formule dans §0 de bid_v5_human_guide.md).
Annonce si max(4 couleurs) ≥ 7 (pos1/4) ou ≥ 8 (pos2) ou ≥ 6 (pos3).
Sinon passe.
```

Accord avec NN : ~85% pos1, 87% pos2, 92% pos3, 88% pos4. -2 à -4pp vs arbre depth-3, mais tient sur une ligne.

## Infrastructure

- [scripts/probe/simplify_xgb.py](../../../scripts/probe/simplify_xgb.py) : backward elimination + depth/n_est sweep
- [scripts/probe/final_tree_rules.py](../../../scripts/probe/final_tree_rules.py) : extraction des arbres depth-3
- [/tmp/xgb_simplify.json](file:///tmp/xgb_simplify.json) : résultats quantitatifs

## Comment re-générer sur un autre modèle

```bash
# Régénérer le probe dataset si nouveau modèle
./target/release/dump_probe_data models/<new_bid>.bin 80000 /tmp/probe_data.bin
PYTHONPATH=scripts/probe uv run python scripts/probe/extract_activations.py

# Relancer la simplification
PYTHONPATH=scripts/probe uv run python scripts/probe/simplify_xgb.py
PYTHONPATH=scripts/probe uv run python scripts/probe/final_tree_rules.py
```
