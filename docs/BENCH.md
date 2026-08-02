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

Same ensemble architecture as Naive IS-MCTS, but uses a `CardBeliefs` model to bias determinization. Maintains per-card per-player probability weights updated via hard constraints (voids, trump ceiling) and soft inference (bidding signals, play patterns). See [play/smart_ismcts.md](play/smart_ismcts.md) for detailed design.

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

---

## NN inference kernels (`nn_kernels.rs`, 2026-07-23)

Shared `dot` / `linear` / `layer_norm` for the pure-Rust nets (`dmc_net`,
`bid_net`, `belief_net`). Sums are split across 8 independent accumulator
lanes, plus an AVX2-dispatched build.

Why it matters: a single-accumulator dot product serialises one ~4-cycle FP add
per element and cannot be auto-vectorised — Rust never grants float
reassociation, so the compiler is not allowed to reorder the sum.

| Net | Before | After | Speedup |
|---|---|---|---|
| DmcNet | 926 µs | 178 µs | 5.2× |
| BidNet | 219 µs | 33 µs | 6.6× |
| BeliefNet | 158 µs | 23 µs | 6.3× |

End to end: DouDou50 self-play went from 35 to **176 deals/s**.

Numerics move only in the last ulp (max abs deviation 6e-4, no argmax change
outside an exact tie). **Any new inference net should use these rather than an
inline loop.** `playgen/infer.rs` has its own equivalent, `dot8`.

## Playgen GPU inference (2026-07-23)

`playgen/gpu.rs` (candle CUDA) batches world generation across lanes. On a
4090 at B=12 the `bench_world_cred` phases went 52.8 s → 9.1 s (auctions,
5.8×) and 29.7 s → 9.2 s (play, 3.2×), bit-identical to CPU. Decode is ~6 ms
per step almost independently of batch size, so the GPU is worthless for a
single world and decisive for pools: 78 worlds/s at B=32, 328 at B=128, 951 at
B=512. See [belief/playgen.md](belief/playgen.md).

## Playgen decode + prefill, v2 (2026-08-02)

`bench_playgen_gpu` sur `models/playgen/playgen_v2_final.bin` (d=384 L=6 H=8,
10,6M params), préfixe 58 tokens, **1 monde = 64 pas de décodage** (2 tokens ×
32 cartes). 4090 + 32 cœurs.

```bash
CUDARC_CUDA_VERSION=13010 cargo build --release --bin bench_playgen_gpu --features dmc_train
./target/release/bench_playgen_gpu --playgen models/playgen/playgen_v2_final.bin \
    --batches 1,8,32,128,512 --steps 64 --prefix 58
```

| B | CPU 1 fil (ms/pas) | CPU mondes/s | GPU (ms/pas) | GPU mondes/s | GPU prefill 58 tok |
|---|---|---|---|---|---|
| 1 | 2,32 | 6,7 | 2,27 | 6,9 | 106,8 ms |
| 8 | 10,21 | 12,2 | 2,83 | 44,1 | 129,6 ms |
| 32 | 38,02 | 13,1 | 3,06 | 163,2 | 135,7 ms |
| 128 | 149,41 | 13,4 | 3,52 | 567,5 | 127,9 ms |
| 512 | *(trop lent)* | — | 7,96 | **1005,2** | 221,7 ms |

Sorties identiques CPU/GPU (colonne `sink`). Chiffres plus hauts que l'entrée
2026-07-23 ci-dessus, qui mesurait un autre chemin (`playgen/gpu.rs` via
`bench_world_cred`) — les deux ne se contredisent pas, ils ne mesurent pas la
même chose.

**Le fait structurant : le prefill est par *batch*, pas par lane, et il est
quasi plat en taille de batch** — 512× plus de lanes pour 2,1× le temps
(106,8 → 221,7 ms). Le décodage, lui, est bien amorti (3,06 ms/pas couvre 32
lanes).

Conséquence, et elle décide de la faisabilité de toute feature dérivée de
playgen : **pour un déroulement court, le prefill domine**. Un déroulement
d'enchère seule (jusqu'à la triple passe) fait ~16 pas contre 64 pour un monde
complet, donc à B=32 c'est 135,7 ms de prefill contre 49 ms de décodage. Le
seul levier est alors de **mettre plusieurs positions différentes dans un même
batch**, pas plus de lanes de la même position — et `KvCacheBatch::from_prefix`
prend un seul préfixe par batch, donc c'est un changement de code.

Ordre de grandeur pour précalculer une feature d'enchère sur un pool
(5M donnes × ~6 décisions d'enchère = 30M positions, 32 déroulements chacune) :

| stratégie | par position | pool 5M | pool 1M |
|---|---|---|---|
| API actuelle (1 préfixe/batch, B=32) | 185 ms | **1540 GPU-h (64 j)** | 308 GPU-h |
| batch inter-positions (B=512 = 16 pos × 32) | 22 ms | 182 GPU-h (7,6 j) | **36 GPU-h** |

Lecture : sans batch inter-positions, c'est hors de portée ; avec, c'est lourd
à 5M et confortable à 1M. Le nombre de déroulements par position ne change
presque rien tant que le prefill domine — c'est le nombre de *positions* qui
coûte. Contexte et usage : [bid/bid_v7_plan.md](bid/bid_v7_plan.md) §3.8.
