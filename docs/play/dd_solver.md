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

### Windowed solve (2026-07-26)

`solve_windowed_reuse_tt(&state, &mut tt, alpha, beta)` and
`solve_for_trump_windowed(hands, dealer, trump, &mut tt, alpha, beta)` expose the
alpha-beta window that the other entry points hardcode to `[0, 252]`.

Intended for a batch of near-identical positions — the sampled worlds of one hand,
whose DD values cluster — where the running mean of the worlds already solved is a
good seed. A full window rediscovers the value from scratch every time.

**The search is fail-soft, so the result is exact only when `alpha < v < beta`.**
Outside that range it is a bound (`v <= alpha` ⇒ upper bound, `v >= beta` ⇒ lower
bound) and the caller *must* re-search wider to get the exact value. Getting this
wrong is the same class of defect as `quick_tricks` below: a bound silently used as
a value. [bench_solve_window.rs](../../colver-core/src/bin/bench_solve_window.rs)
asserts every windowed result against the full-window value rather than assuming it.

[bench_tt_size.rs](../../colver-core/src/bin/bench_tt_size.rs) sweeps the TT size on
the same path: `new_tt_buffer()` is 2 MB, so 32 threads each carrying one is a 64 MB
working set, well past L3. The solver masks with `tt.len() - 1`, so any power-of-two
size is legal.

## BREAKING (2026-07-23): `quick_tricks` removed — it returned wrong DD values

`quick_tricks` credited a whole run of plain-suit master cards as guaranteed
points after checking only *once* that opponents could not ruff, ignoring that
they become void on the next round. That bogus lower bound raised `alpha`
above the true value and cut valid lines.

Measured on real deals: **25% of `solve_for_trump` values were wrong** (66% of
deals had ≥1 wrong suit; median error 6 pts, max 52), and 23–29% of play
positions had ≥1 wrong card score. Removing it also made the solver ~10%
*faster*, so there was no tradeoff.

**Any pre-2026-07-23 DD data is affected** — notably `data/deals/base_5M.bin`
and every score layer derived from it, plus arena and training results that
depend on them.

Cost in *playing strength* was small: the same bot with and without the buggy
bound scored 50.9% over 1600 matches, indistinguishable from zero. The defect
mattered for **data** (DD training targets, oracle trust) far more than for
direct play.

**Invariant:** a sound alpha-beta returns the same value whatever the move
order. `test_root_scores_match_independent_solve` enforces it by cross-checking
each root score against an independent solve — that is what caught the bug.
Keep it green when adding pruning.

## Why not use it as a player

DD assumes the opponents play perfectly with full information. Real opponents don't, so its decisions are systematically over-aggressive in scenarios where information asymmetry matters. Use [is_dd.md](is_dd.md) for actual play with sampled determinizations.

## Key role: training signal

The bid model is trained on DD points as the reward target — see [bid/reward_studies/v3_reward_study.md](../bid/reward_studies/v3_reward_study.md). The 5M dd pool ([data_gen/pools.md](../data_gen/README.md)) was solved offline.
