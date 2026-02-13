# Colver Benchmark Report

**Date:** 2026-02-13
**Platform:** WSL2 (Linux 6.6.87), Intel i9-13900K, single-threaded
**Toolchain:** rustc 1.91.0, `--release` profile

---

## Raw Rollout Throughput

| Workload | Throughput | Latency |
|---|---|---|
| Play-phase rollout | 1.39M rollouts/sec | 720 ns |
| Full-deal rollout (bid + play) | 895K rollouts/sec | 1118 ns |

---

## Agent Architectures

### Random

Uniformly samples from legal actions. No search cost.

| Metric | Value |
|---|---|
| Time per move | ~0 (negligible) |
| Time per game | < 1 ms |

### Perfect-Info MCTS (1000 iterations)

Standard UCT with full hand visibility. Arena-based tree, rollout-to-terminal simulation.

| Metric | Value |
|---|---|
| Iterations | 1000 |
| Time per game (vs Random) | 8 ms |
| Win rate vs Random | 97% |
| Avg score vs Random | 1114 - 24 |

### Perfect-Info MCTS (4000 iterations)

Same architecture, higher budget.

| Metric | Value |
|---|---|
| Iterations | 4000 |
| Time per game (vs IS-MCTS 4000) | 67 ms |

### Naive IS-MCTS (Ensemble Determinization)

Samples D determinized worlds (respecting void constraints), runs standard MCTS on each, aggregates root visit counts. Only sees own hand + played cards.

| Config (D x I = total) | Time/game (vs Random) | Win% vs Random | Avg score (NS - EW) |
|---|---|---|---|
| 20 x 50 = 1000 | 8 ms | 92% | 1137 - 81 |
| 40 x 100 = 4000 | 32 ms | 90% | 1105 - 103 |
| 50 x 200 = 10000 | 78 ms | 82% | 1075 - 213 |

### Smart IS-MCTS (Belief-Weighted Determinization)

Same ensemble architecture as Naive IS-MCTS, but uses a `CardBeliefs` model to bias determinization. Maintains per-card per-player probability weights updated via hard constraints (voids, trump ceiling) and soft inference (bidding signals, play patterns). See [SMART_ISMCTS.md](SMART_ISMCTS.md) for detailed design.

| Config (D x I = total) | Opponent | Time/game | Win% (NS) | Avg score (NS - EW) |
|---|---|---|---|---|
| 20 x 50 = 1000 | Random | 9 ms | 88% | 1067 - 130 |
| 20 x 50 = 1000 | Naive IS-MCTS (same budget) | 17 ms | 46% | 536 - 647 |

---

## Head-to-Head: IS-MCTS vs Perfect-Info MCTS

Equal total budget (4000 iterations). IS-MCTS uses 40 determinizations x 100 iterations; MCTS uses 4000 iterations with full hand visibility.

| Metric | IS-MCTS (NS) | MCTS (EW) |
|---|---|---|
| Win rate | 7% | 93% |
| Avg score | 82 | 1138 |
| Time per game | 67 ms (combined) | |

The gap is expected: perfect-info MCTS "cheats" by seeing all four hands, giving it a large advantage in card play decisions.

---

## Notes

- All results over 100 games, `--release` build, single thread
- IS-MCTS win rate vs Random appears slightly lower at higher budgets due to variance in random opponent bidding (random sometimes bids contracts it fulfills by luck)
- GameState is 56 bytes (`Copy`), enabling cheap cloning for both determinization and MCTS tree rollouts
- Determinization uses greedy constraint-aware redistribution (respects void suits, card counts)
- Smart IS-MCTS adds belief overhead of <1ms per game; the cost is in the per-action weight updates (128 floats), not the search itself
- Smart vs Naive IS-MCTS at equal budget is roughly even (~46-54%), suggesting the soft inference weights need further tuning or that the hard constraints alone capture most of the useful information at this search budget
