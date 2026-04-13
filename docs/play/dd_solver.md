# Double-Dummy Solver

**Code:** [colver-core/src/search/solver.rs](../../colver-core/src/search/solver.rs)

Exact alpha-beta search assuming all 4 hands are visible. Used as the **oracle training signal** for bid models and as a benchmark for evaluating play strategies. Not a player — sees too much.

## Algorithm

- Alpha-beta with transposition table (~2MB per buffer, reusable)
- PVS (Principal Variation Search) at non-PV nodes
- Killer move heuristic (2 killers per ply)
- History heuristic for move ordering
- Move ordering: TT move → killers → captures → quiets

## Performance

| Position | Time |
|----------|------|
| Full deal (1 trump) | ~77 ms |
| Full deal (4 trumps) | ~310 ms |
| Mid-game (4-5 tricks left) | ~13.5 ms |

Pool generation throughput: ~244 deals/s on 32 cores with `RUSTFLAGS="-C target-cpu=native"` + workspace LTO. See [gen_pool.rs](../../colver-core/src/bin/gen_pool.rs).

## API

```rust
use colver_core::solver::{solve, new_tt_buffer, solve_all_suits};

let mut tt = new_tt_buffer();
let ns_pts = solve(&state, &mut tt);  // returns NS points with optimal play

let dd_pts: [u8; 4] = solve_all_suits(&state, &mut tt);  // all 4 trump suits
```

## Why not use it as a player

DD assumes the opponents play perfectly with full information. Real opponents don't, so its decisions are systematically over-aggressive in scenarios where information asymmetry matters. Use [is_dd.md](is_dd.md) for actual play with sampled determinizations.

## Key role: training signal

The bid model is trained on DD points as the reward target — see [bid/reward_studies/v3_reward_study.md](../bid/reward_studies/v3_reward_study.md). The 5M dd pool ([data_gen/pools.md](../data_gen/pools.md)) was solved offline.
