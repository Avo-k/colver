# Bidding Documentation

**En cours :** [bid_v7_plan.md](bid_v7_plan.md) — plan d'entraînement v7 (acquis
mesurés, questions ouvertes, pistes). Trois défauts de v6 y sont chiffrés : non-équivariance
aux couleurs (24,6 % des annonces basculent sous renommage), Q plate au sommet, capot
jamais annoncé (0 sur 3000 enchères).

## Model zoo (`bid/bid_net.rs`)

Dueling DQN, hidden size auto-detected from the weight file (tries 256, 512,
1024). Strategies live in [bid_eval/](../../colver-core/src/bid/bid_eval/):
`BidADd` (NN, default), `Improved`, `Heuristic`, `Smart`, `Roro`, `Maxi`,
`BidParams` (parametric) — one file each.

| Model | Weights | Obs | Notes |
|---|---|---|---|
| Bid a Doudou (v1) | `bid_nn_final.bin` | 114→256²→43 | DouZero self-play |
| Bid a Dede (v2) | `bid_v2/bid_nn_final.bin` | 108→512³→43 | DD solver + 24× suit augmentation |
| Bumblebid | — | transformer d=64 L=2 H=4 | experimental, abandoned for MLP |
| Bid v3 Max | `bid_v3_max_20M/bid_nn_final.bin` | 20M steps | `max(DMC, ISDD)` real points |
| Bid v5 ISDD | `bid_v5_isdd/bid_nn_final.bin` | 113-dim score-aware v2 | Δ-winprob reward, 1M ISDD pool; first to dominate v2 in both play modes |
| **Bid v6 ISDD** (default) | `bid_v6_isdd_resume/bid_nn_final.bin` | 117-dim score-aware v3 | 75M steps (45M + 30M resume); +4 belote bits, belote-aware reward (Q+K trump = +20), match simulation (cumulative scores, dealer rotation, reset @ 2000), 5M ISDD pool. Arena vs v5: 55.8% (DMC play) / 57.3% +181 (IS-DD play) |

## Strategies

Deterministic and learned bidders. Implementation: [colver-core/src/bid/bid_eval/](../../colver-core/src/bid/bid_eval/).

**`playgen`** — playgen v2's own 43-way auction head used directly as a bidder
(`strategy = "playgen"`, `model = models/playgen/playgen_v2_final.bin`). Implemented
as `PlaygenBidPolicy` in [agent/bid.rs](../../colver-core/src/agent/bid.rs), *not* in
`bid_eval/` — it needs the whole visible prefix, so like a world source it tracks the
deal through `init_deal` / `observe`. A behaviour clone of v6 (the corpus it learned
from) and not score-aware, yet within ~1-3 pp of it: 48.2% h2h over 3000 matches,
61.6% vs `nn_v2_dmc50` where v6 scores 65.0%. Bot: `arena/bots/playgen_bid.toml`.
See [experiments/auction_conditioned_labels.md](experiments/auction_conditioned_labels.md).

- [strategies/guide_encheres.md](strategies/guide_encheres.md) — high-level guide to bidding strategies
- [strategies/bid_v2.md](strategies/bid_v2.md) — Bid a Dede (current production NN bidder, 20M steps DD-only)
- [strategies/bid_v3_max.md](strategies/bid_v3_max.md) — bid_v3_max_20M (max(DMC,ISDD) reward signal)
- [strategies/bid_v4_score_aware.md](strategies/bid_v4_score_aware.md) — bid_v4 score-aware (Δ win probability reward, match-context)
- [strategies/bid_v5.md](strategies/bid_v5.md) — `v5_isdd_25M`: score features v2 + reward clip + EMA + cosine LR, trained on IS-DD-pure pool (25M steps). Superseded as champion by v6 (see model zoo above).

## Architectures

NN architectures for bidding. See [colver-core/src/bid/bid_net.rs](../../colver-core/src/bid/bid_net.rs) (MLP) and [bid_candle.rs](../../colver-core/src/bid/bid_candle.rs) (training).

- [architectures/bumblebid.md](architectures/bumblebid.md) — transformer encoder (experimental, abandoned for MLP)

## Reward Studies

How the choice of reward signal affects bid model performance.

- [reward_studies/v3_reward_study.md](reward_studies/v3_reward_study.md) — full study (DD, real, blend, curriculum, max, ensembles)
- [reward_studies/v3_max_signal_results.md](reward_studies/v3_max_signal_results.md) — bid_v3_max_20M champion results

## Interpretability

Distilled rules + heuristic strategy documentation.

- [strategies/bid_v5_human_guide.md](strategies/bid_v5_human_guide.md) — **Guide humain** : règles simples v5 pour jouer soi-même (82-91% d'accord avec le NN)
- [strategies/bid_v5_simplified_rules.md](strategies/bid_v5_simplified_rules.md) — **Arbre depth-3 minimaliste** : 5 features, 88-91% d'accord, mémorisable
- [interpretability/bid_rules_xgb.md](interpretability/bid_rules_xgb.md) — **XGBoost-distilled rules from NN v5** (champion, score-aware, 2026-04-19) + diff vs v2
- [interpretability/probe_morning_report.md](interpretability/probe_morning_report.md) — **Hidden-layer probe** : deux features manquantes découvertes (per-suit J/9, `opp_best_other_ts`) qui ferment l'écart 77%→97%
- [interpretability/hand_classification.md](interpretability/hand_classification.md) — **Classer les mains** : index canonique exact (472 579 classes) + code lisible (`T5.J9AT.A1/A1/x1`, 80 codes utiles), et la mesure DD appariée qui dit ce que vaut chaque carte (J d'atout +49,2 ; Dame de côté −0,1)
- [interpretability/bid_rules_v6.md](interpretability/bid_rules_v6.md) — **Comment annonce v6, en familles de mains** : la politique d'ouverture, de défense et de soutien écrite en `HandCode`, chaque ligne avec son accord *et son plafond*. 73 familles retrouvent 87 % des décisions annoncer/passer. En défense, ce qui décide n'est pas la main dans la couleur adverse mais **la meilleure autre couleur**
- [interpretability/rule_ceiling.md](interpretability/rule_ceiling.md) — **Le plafond d'une règle humaine (2026-08-03)** : v6 n'étant pas équivariant, aucune règle insensible aux couleurs ne peut dépasser **97,4 % sur annoncer/passer ni 83,5 % sur l'action exacte** à l'ouverture. Contient un négatif (entraîner sur le réseau symétrisé ne vaut rien, +0,5 pt) et le seul ajout de features qui paie (le `trump_score` de la 2ᵉ couleur)
- [interpretability/bid_rules_xgb_v2.md](interpretability/bid_rules_xgb_v2.md) — historical v2 distillation (DD oracle, pre-rule-change)
- [interpretability/strategies_encheres.md](interpretability/strategies_encheres.md) — French doc of all heuristic bidders (heuristic, smart, roro, improved, ...)

## Experiments (archived)

Old experiments not currently used in production but kept for reference.

- [experiments/auction_conditioned_labels.md](experiments/auction_conditioned_labels.md) — **négatif, à lire avant de recommencer (2026-07-25)** : conditionner le label d'annonce sur le préfixe d'enchère resserre bien le posterior de 21,5 % autour du bon endroit, mais les deux bidders distillés qui en sortent perdent l'arène (47,3 % puis 43-44 % contre v6). Le pipeline est réutilisable, la définition de la cible est ce qui cloche. Contient aussi deux fausses pistes qui semblaient conclusives (« les mondes de playgen sont plus riches », « c'est la distribution des scores de match »)
- [experiments/enchere_cannes_roro.md](experiments/enchere_cannes_roro.md)
- [experiments/maxi_bid_roro.md](experiments/maxi_bid_roro.md)
