# Play Documentation

How a player plays cards once bidding ends. Implementations live in [colver-core/src/search/](../../colver-core/src/search/) and [colver-core/src/dmc/](../../colver-core/src/dmc/).

## Methods

- [dd_solver.md](dd_solver.md) — Double-dummy alpha-beta solver (sees all hands, oracle)
- [is_dd.md](is_dd.md) — **Information-Set DD, the production card player** (samples determinized worlds + one exact DD solve each)
- [../agents.md](../agents.md) — how a card player is built, configured and driven
- [smart_ismcts.md](smart_ismcts.md) — Belief-weighted IS-MCTS
- [dmc.md](dmc.md) — DouZero-style Q-network (DouDou35, DouDou50)

## World sampling

IS-DD and IS-MCTS both need determinized worlds. The samplers that produce them
(belief nets, playgen transformer, constraint-uniform) are documented under
[../belief/](../belief/) — see [../belief/playgen.md](../belief/playgen.md).

## Experiments

- [experiments/triforge.md](experiments/triforge.md) — joint bid+play training, alternating best response

## Performance

See [../BENCH.md](../BENCH.md) for raw rollout numbers.
