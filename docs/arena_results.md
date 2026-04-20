# Arena Results — Global Leaderboard

The arena is the **king metric** for evaluating any bidding/play change. Each h2h plays both directions for variance reduction. Runs ≥1000 matches per direction for tournament-grade conclusions; 200 matches for quick iteration.

**Scripts:** [colver-core/src/bin/arena.rs](../colver-core/src/bin/arena.rs)
**Bot configs:** [arena/bots/](../arena/bots/) (TOML files)
**Raw results:** [arena/results/matches.csv](../arena/results/matches.csv)

## How to run

```bash
# Quick h2h
cargo run --bin arena --release -- h2h bid_v3_max_20M nn_v2_dmc50 --matches 1000

# Round-robin subset
cargo run --bin arena --release -- round-robin --matches 200 \
    --bots bid_v3_max_20M,nn_v2_dmc50,nn_v2_isdd

# View saved results
cargo run --bin arena --release -- results --bot bid_v3_max_20M
```

## Reference matchups

These are the canonical evals to run on any new bid or play model.

### Eval set 1 — DMC play (fast, cheap)

H2H against `nn_v2_dmc50` (Bid a Dede + DouDou50). Both bots use DMC for play, so this isolates the bid model.

| Bot | Win% | Margin | Matches |
|-----|------|--------|---------|
| `nn_v2_dmc50` (reference) | 50.0% | 0 | (self) |
| **`v5_isdd_25M`** 🏆 | **61.8%** | **+116** | 1000 |
| `v5_isdd_22p5M` | 56.3% | +45 | 1000 |
| `v5_max_16M` | 56.3% | +18 | 1000 |
| `v5_max_20M` | 54.9% | +19 | 1000 |
| `bid_v3_max_20M` | 49.9% | +11 | 2000 |
| `exp_e_curriculum` (5M) | 54.5% | +97 | 200 |
| `exp_d_blend25` (5M) | 53.0% | +41 | 200 |
| `exp_c_blend50` (5M) | 49.2% | -26 | 200 |
| `exp_b_blend75` (5M) | 48.5% | -50 | 200 |
| `exp_a_dd` (5M) | 46.5% | -81 | 200 |

*v5 rows: 2026-04-20 round-robin (5 bots, 500 matches/direction, 1000/H2H); win% and margin shown are the direct H2H vs `nn_v2_dmc50`. See [bid/strategies/bid_v5.md](bid/strategies/bid_v5.md).*

**Round-robin standing** (same run, total win% across all opponents):

| Rank | Bot | Win% | Margin |
|------|-----|------|--------|
| 1 | `v5_isdd_25M` 🏆 | 54.0% | +53 |
| 2 | `v5_isdd_22p5M` | 52.4% | +36 |
| 3 | `v5_max_16M` | 51.1% | −5 |
| 4 | `v5_max_20M` | 49.9% | −35 |
| 5 | `nn_v2_dmc50` | 42.7% | −50 |

### Eval set 2 — IS-DD play (slower, more realistic)

H2H against `nn_v2_isdd_no_belief` (Bid a Dede + IS-DD, no belief net). Both bots use IS-DD for play.

| Bot | Win% | Margin | Matches |
|-----|------|--------|---------|
| `nn_v2_isdd_no_belief` (reference) | 50.0% | 0 | (self) |
| **`v5_isdd_25M_isdd`** 🏆 | **57.3%** (vs `v3_max_isdd`) | **+44** | 1000 |
| `bid_v3_max_20M_isdd` | 50.8% | +37 | 2000 |
| `exp_a_dd_isdd` (5M) | 50.5% | +1 | 200 |
| `exp_b_blend75_isdd` | 48.8% | -66 | 200 |
| `exp_e_curriculum_isdd` | 48.8% | -70 | 200 |
| `exp_d_blend25_isdd` | 48.5% | -69 | 200 |
| `exp_c_blend50_isdd` | 47.5% | -76 | 200 |

### Cross-eval

| Matchup | Result | Margin |
|---------|--------|--------|
| `bid_v3_max_20M` (DMC) vs `nn_v2_isdd_no_belief` | — | — |
| `exp_e_curriculum` (DMC) vs `exp_a_dd_isdd` (IS-DD) | 49.5% / 50.5% | -36 |

## Champions by category

| Category | Bot | TOML |
|----------|-----|------|
| **Best overall (any play)** | `nn_v2_isdd` (belief net #1) | [arena/bots/nn_v2_isdd.toml](../arena/bots/nn_v2_isdd.toml) |
| **Best bid model (DMC play)** | `v5_isdd_25M` | [arena/bots/v5_isdd_25M.toml](../arena/bots/v5_isdd_25M.toml) |
| **Best bid model (IS-DD play)** | `v5_isdd_25M_isdd` | [arena/bots/v5_isdd_25M_isdd.toml](../arena/bots/v5_isdd_25M_isdd.toml) |
| Best fast bot | `v5_isdd_25M` | (DMC-based) |
| Previous champion bidder | `bid_v3_max_20M` | [arena/bots/bid_v3_max_20M.toml](../arena/bots/bid_v3_max_20M.toml) |

## Reading the results

- **Win%** is direct match win rate (first to 2000 points).
- **Margin** is mean point difference at end of match, from the row bot's perspective.
- A model with 50.5% win and +50 margin is winning more matches AND winning by larger margins.
- Std at 2000 matches ≈ ±1.1pp; at 200 matches ≈ ±2.5pp; at 50 matches ≈ ±5pp.
- **Always run ≥1000 matches before claiming a new champion.**

## Notes

- Results are reproducible with `--seed 42` (default).
- The CSV log keeps full provenance (bid_a/play_a/bid_b/play_b labels) so you can re-query historical matchups.
- Old bots may be stale — re-run if play model files change.
