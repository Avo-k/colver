# Bidding Documentation

## Strategies

Deterministic and learned bidders. Implementation: [colver-core/src/bid/bid_eval/](../../colver-core/src/bid/bid_eval/).

- [strategies/guide_encheres.md](strategies/guide_encheres.md) — high-level guide to bidding strategies
- [strategies/bid_v2.md](strategies/bid_v2.md) — Bid a Dede (current production NN bidder, 20M steps DD-only)
- [strategies/bid_v3_max.md](strategies/bid_v3_max.md) — bid_v3_max_20M (max(DMC,ISDD) reward signal)

## Architectures

NN architectures for bidding. See [colver-core/src/bid/bid_net.rs](../../colver-core/src/bid/bid_net.rs) (MLP) and [bid_candle.rs](../../colver-core/src/bid/bid_candle.rs) (training).

- [architectures/bumblebid.md](architectures/bumblebid.md) — transformer encoder (experimental, abandoned for MLP)

## Reward Studies

How the choice of reward signal affects bid model performance.

- [reward_studies/v3_reward_study.md](reward_studies/v3_reward_study.md) — full study (DD, real, blend, curriculum, max, ensembles)
- [reward_studies/v3_max_signal_results.md](reward_studies/v3_max_signal_results.md) — bid_v3_max_20M champion results

## Interpretability

Distilled rules + heuristic strategy documentation.

- [interpretability/bid_rules_xgb.md](interpretability/bid_rules_xgb.md) — XGBoost-distilled rules from NN v2 (~93% accuracy)
- [interpretability/strategies_encheres.md](interpretability/strategies_encheres.md) — French doc of all heuristic bidders (heuristic, smart, roro, improved, ...)

## Experiments (archived)

Old experiments not currently used in production but kept for reference.

- [experiments/enchere_cannes_roro.md](experiments/enchere_cannes_roro.md)
- [experiments/maxi_bid_roro.md](experiments/maxi_bid_roro.md)
