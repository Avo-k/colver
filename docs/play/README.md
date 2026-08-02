# Play Documentation

How a player plays cards once bidding ends. Implementations live in [colver-core/src/search/](../../colver-core/src/search/) and [colver-core/src/dmc/](../../colver-core/src/dmc/).

## Methods

- [dd_solver.md](dd_solver.md) — Double-dummy alpha-beta solver (sees all hands, oracle)
- [dd_solver_optimization.md](dd_solver_optimization.md) — **what was tried on the solver and what
  came back negative**, with the measurement and the command to replay each one. Read before
  attempting any solver optimisation: larger TT, MTD(f), window seeding and a slimmed
  `apply_play` are all measured dead ends.
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

See [../BENCH.md](../BENCH.md) for raw rollout numbers (agents, NN kernels, rollouts).

DD-solver numbers are **not** there — they live in [dd_solver.md](dd_solver.md), measured with
`bench_dd` on a fixed corpus and journalised in
[../measurements/index.jsonl](../measurements/index.jsonl). Anything quoting a solver time
without naming a corpus and a position shape predates 2026-08-02 and is unreliable: four
documents used to give four different figures for "a solve".
