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

**This table is the single source for DD solver timings.** Every other document quotes at most
one rounded figure from it and links back here — if you find a second table of solver times
anywhere in `docs/`, one of them is stale. Consumers that reason from these numbers:
[is_dd.md](is_dd.md#performance) (world budgets), [../ARCHITECTURE.md](../ARCHITECTURE.md),
[../web_analyse_jeu.md](../web_analyse_jeu.md) (per-position sim budgets).

Measured 2026-08-02 with `bench_dd` on a fixed 2 120-position corpus, one thread, i9-13900K,
stock `x86-64` target (the flag buys nothing — see the negative-results table).
Journalised in [docs/measurements/index.jsonl](../measurements/index.jsonl). Every earlier
figure in the docs is superseded: four documents quoted 13.5 / 14.9 / 28 / 77 ms for things all
called "a solve", none with a stated corpus or shape.

| Shape (all via `solve_with_scores`) | n | nodes/pos | time/pos |
|---|---|---|---|
| Full deal, 4 suits | 800 | 1 448 045 | 32.3 ms |
| Mid-game, real games, 13-24 cards left | 360 | 9 061 | 169 µs |
| Endgame, real games, 2-12 cards left | 240 | 89 | 1.4 µs |
| Determinized worlds (the IS-DD unit) | 720 | 55 862 | 1.13 ms |

**Node counts are exact; the times are a floor, not a value.** They are the minimum of 5
alternating rounds, and the per-round spread on this box is **~9 %** even with the machine
otherwise idle. Quote them to two significant figures, and never compare a time here against a
time measured on another day — only within one alternating run. (An earlier revision of this
table said 34.6 ms for the full deal, from fewer rounds on a busier machine. That is the same
measurement, not a regression.)

Throughput is ~45 M nodes/s, i.e. **~22 ns per node**. That is already tight, and it is the
reason most per-node micro-optimisation attempts below came back negative: the search is
dominated by the transposition-table probe, which is a random access into 2 MB.

The distribution is heavily skewed — on full deals the **worst 10 % of solves hold 40 % of all
nodes** (p50 317 k nodes, p99 3.8 M, max 6.0 M). Anything that helps only the median deal is
worth little; the tail is where the batch time is.

**Value-only `solve_for_trump`, for comparison**: ~18 ms mean, 10.4 ms median, 69 ms P95 on a
full deal (via the older `dd_bench`). Do not line these up against the table above — one trump
instead of four, and a mean/median/P95 over deals instead of a min over repeats. The gap between
the mean and the P95 is the same tail as above, seen from the other harness.

## Benchmarking and the exactness gate

`bench_dd` (feature `solver_stats` for node counts) is both the benchmark and the guard:

```bash
cargo build --release --features "parallel solver_stats" --bin bench_dd
./target/release/bench_dd build --out data/analysis/dd_corpus_v1.bin \
    --pool data/deals/base_5M.bin --games data/training/heldout_20k_s90210.bin
./target/release/bench_dd run --corpus data/analysis/dd_corpus_v1.bin \
    --values cand.vals --repeats 5
./target/release/bench_dd diff --a baseline.vals --b cand.vals   # must say EXACT MATCH
scripts/analysis/dd_ab_revs.sh <baseline-rev> 3                  # alternating A/B of two git revs
scripts/analysis/dd_ab_flags.sh 5                                # alternating A/B of RUSTFLAGS targets
```

Three rules learned the hard way here:

- **Node counts first, wall-clock second.** On a hybrid P/E-core CPU under WSL2, wall time
  cannot separate better pruning from landing on a P-core.
- **Never compare two sequential runs.** A single binary measured twice on this machine
  differed by 20 % because another job started in between — larger than most of the wins.
  `--ab` interleaves within one process; `dd_ab_revs.sh` alternates two binaries built from
  two git revisions. Both keep the **minimum**, since competing load only ever adds time.
- **The corpus is a file, written once and kept.** No seeded generator here is reproducible:
  `gen_pool` hands slot indices out of an `AtomicUsize` to N workers, so the RNG stream that
  lands at a given index depends on thread scheduling.

The corpus draws its mid-game and endgame positions from **real played games** (COLVGM01), not
random legal play, and includes the determinized-world batches that are IS-DD's actual unit —
the shape nothing in the repo measured before. `base_5M.bin`'s `dd_pts` are stale, but its
`hands` are just a deal distribution and remain usable.

## Measured negative results — do not re-derive

| Idea | Verdict | Evidence |
|---|---|---|
| Larger transposition table | **No.** 2 MB → 134 MB buys 3 % fewer nodes and costs 2.4× the time (cache misses). `1<<18` is at the optimum; `1<<16` is marginally better. | Single-thread sweep over 6 sizes, 2026-08-02 |
| MTD(f) / binary search on the point value | **No.** A null-window probe costs 0.42× a full search — only 2.4× cheaper — and a cold binary search needs 7.3 probes, so **1.94× the full-window cost**. Bridge DD wins here because its value is a trick count (0-13); a point total (0-252) is too wide. | 320 solves, 2026-08-02 |
| Narrow window seeded from the running world mean | **No, 1.04× at best.** The premise is false: worlds of one hand are not clustered (36 % land >40 pts from the mean). | [bid_v7_plan.md](../bid/bid_v7_plan.md) §1.5 |
| Solver-only `apply_play` skipping voids, belote and `trick_history` | **No, 0.977× — it made things *slower*.** Node counts identical, values exact; removing real instructions still lost 2.3 %. This also undercuts the larger "compact 32-byte solver state" idea, whose easier half this was. | Alternating A/B vs HEAD, min-of-3, 2026-08-02 |
| `-C target-cpu=native` or `x86-64-v3` | **No, 0 %** — v3 lands within 0.03 % of baseline, `native` is 1.25 % *slower*. The reason is worth knowing: `tzcnt` is already emitted **32 times in the baseline binary** (LLVM uses the `F3`-prefixed encoding, which pre-BMI1 CPUs decode as plain `bsf`), and `popcnt` appears zero times even with `native`. The dominant bit primitive never needed the flag. | 3 binaries, same source, 5 alternating rounds, 2026-08-02 |
| Deduplicating DD-equivalent cards at the root of `solve_with_scores` | Not worth it: only **5.2 %** of root moves are redundant. | 5 991 decision points |

## Accepted changes

- **The TT stamps an epoch instead of being cleared** (2026-08-02). The table is valid for one
  (deal, trump) pair only, so it was `memset` on every entry point — a flat 28.8 ms per 1000
  solves, negligible against a 35 ms full deal and *dominant* below 12 cards left, where a
  solve searches 89 nodes. A 15-bit epoch in the spare entry bits invalidates in O(1); the
  clear happens once per 32 767 solves. **18.9× on endgames, 1.14× mid-game, node-for-node
  identical.** Keep `&mut [u64]` in the recursion — holding `&mut TtBuf` there re-loads the
  `Vec` pointer at every probe and costs more than the memset saved.
- **Card equivalence is a table, not two sorts** (2026-08-02). Derived from the point tables and
  proved by exhaustion against the original loops. No measurable speed change; kept for
  simplicity and because the derivation's assumptions are now asserted.

Pool generation throughput: ~244 deals/s on 32 cores with `RUSTFLAGS="-C target-cpu=native"` + workspace LTO — a compound claim on a *different* binary; `target-cpu=native` on the solver itself measures 0%, see the negative-results table. See [gen_pool.rs](../../colver-core/src/bin/gen_pool.rs).

## API

Every entry point returns `[NS_points, EW_points]` unless stated otherwise. The previous
version of this section was wrong on both counts it made: `solve` takes no table, and
`solver::solve_all_suits` does not exist — it is a private helper in
[bid_train_env.rs](../../colver-core/src/bid/bid_train_env.rs) looping the four trumps over
one table.

```rust
use colver_core::solver::{self, new_tt_buffer, TtBuf};

// One-shot. Allocates its own 2 MB table, so not for a loop.
let pts: [u8; 2] = solver::solve(&state);
let pts: [u8; 2] = solver::solve_for_trump(hands, dealer, trump);

// Reusing a table across many solves — always do this in a loop.
let mut tt: TtBuf = new_tt_buffer();
let pts = solver::solve_reuse_tt(&state, &mut tt);
let pts = solver::solve_for_trump_reuse_tt(hands, dealer, trump, &mut tt);

// Per-card values for every legal root move. This is IS-DD's unit and the dominant
// consumer of DD CPU in the project. `None` allocates a fresh table per call.
let sc = solver::solve_with_scores(&state, Some(&mut tt));
// sc.scores[..sc.count] is (card, ns_points), and sc.best_card is already the best move —
// do not follow this with solve_best_card, which searches the same tree a second time.
```

**`TtBuf`, not `Vec<u64>`** (changed 2026-08-02). The table carries a solve epoch so it can
invalidate itself in O(1) instead of being memset; see
[dd_solver_optimization.md](dd_solver_optimization.md) §1.1. `TtBuf::with_log2_size(n)` builds
any power-of-two size.

**Node counting** lives behind the `solver_stats` feature: `solver::take_nodes()` returns and
resets this thread's count, `solver::stats_enabled()` says whether it is compiled in. Check the
latter before reporting a count — a silent 0 reads like a perfect search.

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

**Single-threaded, the sweep is settled and the answer is "leave it alone"**: 2 MB → 134 MB
buys 3 % fewer nodes and costs 2.4× the time. `1<<18` is at the optimum. The 32-thread
question — 64 MB of working set against 36 MB of L3 — is the part that has **never been
run**, and it is the cheapest open lever in the repo. Numbers and method:
[dd_solver_optimization.md](dd_solver_optimization.md) §2.1.

**The premise behind the windowed solve is false**, incidentally: the sampled worlds of one
hand do *not* cluster (36 % land more than 40 points from the running mean), so seeding a
narrow window from that mean is worth 1.04× at best. Both windowed entry points therefore
have no production caller. §2.3 of the same document, and
[bid_v7_plan.md](../bid/bid_v7_plan.md) §1.5.

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
