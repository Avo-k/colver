# Bid v3 Max Signal — Champion Results

> **Headline:** Training a bid model on `max(DMC, ISDD)` real points instead of pure DD oracle produces the only model in the v3 study that **doesn't lose to `nn_v2` in either DMC or IS-DD evaluation**.

For the broader study (DD, real, blend, curriculum, ensembles), see [v3_reward_study.md](v3_reward_study.md). For the model spec, see [../strategies/bid_v3_max.md](../strategies/bid_v3_max.md).

## Pool: dd_1.2M_max_dmc_isdd.bin

Built from 1.2M deals, each enriched with both DMC and IS-DD play points, then merged per suit:

```python
real_pts[s] = max(dmc_pts[s], isdd_pts[s])
```

| Source | File | Deals |
|--------|------|-------|
| DMC enrichment (sequential, offset 0) | `dd_5M_seq_enriched_dmc.bin` | 5M |
| IS-DD enrichment (12 batches, hard constraints) | `data/deals/tmp_isdd_batchN.bin` | 1.2M |
| Merged (intersection) | `dd_1.2M_max_dmc_isdd.bin` | 1.2M |

The match rate is 100% because both pools sample sequentially from `dd_5M.bin` starting at offset 0.

## Why max?

On 4.8M matched samples (1.2M deals × 4 suits):

| Signal | MAE vs DD | Mean pts | Total pts | % of DD |
|--------|-----------|----------|-----------|---------|
| DMC | 19.04 | 82.66 | 396.8 M | 96.5% |
| ISDD | 18.82 | 82.44 | 395.7 M | 96.2% |
| avg(DMC, ISDD) | 16.90 | 82.55 | 396.2 M | 96.3% |
| **max(DMC, ISDD)** | 18.41 | **91.56** | **439.5 M** | **106.8%** |
| min(DMC, ISDD) | 19.45 | 73.54 | 353.0 M | 85.8% |
| oracle best (closest to DD) | 11.96 | 83.02 | 398.5 M | 96.9% |

`max` is the **only signal that exceeds DD on average** (+5.86 pts/game, +6.84% total). This isn't a bug — DD assumes opponents play perfectly with full information; against realistic defenders, NS regularly scores more than DD theoretical max.

### Contract success rates (P(signal ≥ thr | DD ≥ thr))

| Threshold | DMC | ISDD | avg | **max** |
|-----------|-----|------|-----|---------|
| ≥ 80 | 87.0% | 85.7% | 88.6% | **94.4%** |
| ≥ 110 | 71.4% | 71.4% | 72.8% | **85.1%** |
| ≥ 120 | 63.2% | 64.5% | 64.0% | **79.5%** |
| ≥ 140 | 42.8% | 46.6% | 42.9% | **61.9%** |
| ≥ 160 | 23.8% | 32.0% | 42.3% | **42.7%** |

A bid of 120 passes 79.5% of the time with `max` as training signal vs 64% with single-method signals. The model learns to bid more aggressively because the data supports it.

## Training

| Hyperparam | Value |
|------------|-------|
| Architecture | Dueling MLP 108→512³→43 |
| Steps | 20M (matched to nn_v2 budget) |
| Envs | 64 |
| Batch | 512 |
| `--reward` | `real` (uses pool's `real_pts` directly) |
| Pool | `dd_1.2M_max_dmc_isdd.bin` |
| Wall time | ~10h |

```bash
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- \
    --hidden 512 --layers 3 --num-envs 64 --batch-size 512 \
    --steps 20000000 \
    --pool-file data/pools/dd_1.2M_max_dmc_isdd.bin \
    --reward real \
    --save-dir models/bid_v3_max_20M \
    --eval-freq 1000000 --save-freq 2000000
```

## Arena results

### Final h2h (2000 matches, ±1.1pp std)

| Eval | bid_v3_max_20M | nn_v2 | Margin/game |
|------|----------------|-------|-------------|
| vs `nn_v2_dmc50` (DMC play) | **49.9%** | 50.1% | +11 |
| vs `nn_v2_isdd_no_belief` (IS-DD play) | **50.8%** | 49.2% | +37 |

### Comparison with v3 study signals (5M steps each, 200 matches)

| Bot | vs nn_v2 DMC | vs nn_v2 IS-DD |
|-----|--------------|----------------|
| `exp_a_dd` (pure DD, 5M) | 46.5% (-81) | 50.5% (+1) |
| `exp_b_blend75` | 48.5% (-50) | 48.8% (-66) |
| `exp_c_blend50` | 49.2% (-26) | 47.5% (-76) |
| `exp_d_blend25` | 53.0% (+41) | 48.5% (-69) |
| `exp_e_curriculum` (95→30% DD) | 54.5% (+97) | 48.8% (-70) |
| **bid_v3_max_20M (20M)** | **49.9% (+11)** | **50.8% (+37)** |

### Per-checkpoint vs nn_v2_dmc50 (200 matches each, noisy)

| Step | Win% | Margin |
|------|------|--------|
| 2M | 42.8% | -163 |
| 4M | 51.5% | +21 |
| 6M | 50.2% | +66 |
| 8M | 49.5% | +5 |
| 10M | 49.0% | +23 |
| 12M | 49.5% | +16 |
| 14M | 51.8% | +10 |
| **16M** | **54.2%** | **+114** |
| 18M | 51.5% | +41 |
| 20M | 52.5% | +96 |

The model takes off after ~14M steps. The 200-match noise is large (±2.5pp) — see the 2000-match results above for reliable numbers.

## Key observations

1. **No single-method overfitting.** All other v3 signals (curriculum, blends, dd-pure) overfit to either DMC or IS-DD play and lose against the other. `max` is the only signal that holds in both.

2. **`max` is not an unbiased estimator of DD.** It systematically overestimates by ~6 pts. But that's the point — it represents what a strong realistic player can achieve against realistic defenders, not the theoretical perfect-play bound.

3. **Training budget matters.** At 5M steps, `max` looked unconvincing (47.5% / 49.0% vs 200-match reference). At 20M steps it matches nn_v2.

4. **Signal vs noise: 200 matches isn't enough.** Initial 400-match h2h reported 51.9% / 53.6%, but 2000-match runs settled at 49.9% / 50.8%. Always re-run with ≥1000 matches before claiming victory.

## Open questions

- Does the same approach work for `train_joint` (Triforge)? The Triforge bid phase uses live DD, not enriched real_pts — would need a different reward injection.
- Can `max` be combined with `oracle best` (lowest |error|) in a curriculum? `max` for the first half, `oracle best` for the second?
- Is there a non-linear combination that beats max? `0.7*max + 0.3*avg`?
