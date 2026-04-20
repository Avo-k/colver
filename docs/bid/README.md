# Bidding Documentation

## Strategies

Deterministic and learned bidders. Implementation: [colver-core/src/bid/bid_eval/](../../colver-core/src/bid/bid_eval/).

- [strategies/guide_encheres.md](strategies/guide_encheres.md) — high-level guide to bidding strategies
- [strategies/bid_v2.md](strategies/bid_v2.md) — Bid a Dede (current production NN bidder, 20M steps DD-only)
- [strategies/bid_v3_max.md](strategies/bid_v3_max.md) — bid_v3_max_20M (max(DMC,ISDD) reward signal)
- [strategies/bid_v4_score_aware.md](strategies/bid_v4_score_aware.md) — bid_v4 score-aware (Δ win probability reward, match-context)
- [strategies/bid_v5.md](strategies/bid_v5.md) — **current champion** `v5_isdd_25M`: score features v2 + reward clip + EMA + cosine LR, trained on IS-DD-pure pool (25M steps)

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
- [interpretability/bid_rules_xgb_v2.md](interpretability/bid_rules_xgb_v2.md) — historical v2 distillation (DD oracle, pre-rule-change)
- [interpretability/strategies_encheres.md](interpretability/strategies_encheres.md) — French doc of all heuristic bidders (heuristic, smart, roro, improved, ...)

## Experiments (archived)

Old experiments not currently used in production but kept for reference.

- [experiments/enchere_cannes_roro.md](experiments/enchere_cannes_roro.md)
- [experiments/maxi_bid_roro.md](experiments/maxi_bid_roro.md)
