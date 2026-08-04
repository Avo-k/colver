# Play Documentation

How a player plays cards once bidding ends. Implementations live in [colver-core/src/search/](../../colver-core/src/search/) and [colver-core/src/dmc/](../../colver-core/src/dmc/).

## Methods

- [dd_solver.md](dd_solver.md) — Double-dummy alpha-beta solver (sees all hands, oracle)
- [dd_solver_optimization.md](dd_solver_optimization.md) — **what was tried on the solver and what
  came back negative**, with the measurement and the command to replay each one. Read before
  attempting any solver optimisation: larger TT, MTD(f), window seeding and a slimmed
  `apply_play` are all measured dead ends.
- [is_dd.md](is_dd.md) — **Information-Set DD, the production card player** (samples determinized
  worlds + one exact DD solve each). Its cost table **multiplies** the solver's, it does not
  restate it — the per-shape costs live in [dd_solver.md](dd_solver.md#performance).
- [../agents.md](../agents.md) — how a card player is built, configured and driven
- [smart_ismcts.md](smart_ismcts.md) — Belief-weighted IS-MCTS
- [dmc.md](dmc.md) — DouZero-style Q-network (DouDou35, DouDou50)
- [doudou_next.md](doudou_next.md) — **ce qui doit monter à bord de la prochaine
  itération de DouDou**, à fermer avant de lancer : une modification d'obs
  invalide les poids. Première entrée, la **belote annoncée** — publique, lue par
  IS-DD et par l'obs d'enchère v6, absente de l'obs de jeu.

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
