# Colver Documentation

Belote Contrée engine + RL training stack.

## Top-level

- [ARCHITECTURE.md](ARCHITECTURE.md) — workspace layout, key subsystems, observation formats
- [RULES.md](RULES.md) — game rules summary
- [BENCH.md](BENCH.md) — performance benchmarks
- [arena_results.md](arena_results.md) — **global arena leaderboard** (the main eval metric)
- [deal_bias.md](deal_bias.md) — traditional gather-cut dealing vs competition shuffling (bias study)
- `règles officielles belote contrée.pdf` — official FFB rules

## Subsystems

- [bid/](bid/) — bidding strategies, bid NN training, reward studies, interpretability
- [play/](play/) — play methods (DD solver, IS-DD, DMC, IS-MCTS), play NN training
- [belief/](belief/) — belief models for hidden card inference
- [data_gen/](data_gen/) — pool generation, enrichment methods, replay formats
- [training/](training/) — training pipelines and how to invoke them
- [superpowers/](superpowers/) — agent-driven plans and design specs

## Quick links by topic

| Topic | Main doc |
|-------|----------|
| Latest bid champion | [bid/reward_studies/v3_max_signal_results.md](bid/reward_studies/v3_max_signal_results.md) |
| Reward signal study | [bid/reward_studies/v3_reward_study.md](bid/reward_studies/v3_reward_study.md) |
| Distilled bid rules (XGBoost) | [bid/interpretability/bid_rules_xgb.md](bid/interpretability/bid_rules_xgb.md) |
| Triforge joint training | [play/experiments/triforge.md](play/experiments/triforge.md) |
| Belief net + IS-DD | [belief/bis_dd.md](belief/bis_dd.md) |
| Training commands | [training/overview.md](training/overview.md) |
