# Bid v3 Reward Study: DD vs Real vs Blend vs Curriculum

## Motivation

Bid v2 (Bid a Dede) trains on 100% DD solver rewards. The DD solver assumes perfect play, but real opponents don't play perfectly. This study investigates whether blending DD rewards with "real" play rewards (from DouDou50 or IS-DD) improves bidding.

## Datasets

### Current layout (`data/deals/`)

Deals and scores are now stored separately. One base file with deals + DD, plus separate score files per play method.

| File | Content | Format | Notes |
|------|---------|--------|-------|
| `base_5M.bin` | 5M deals (dealer + hands + dd_pts) | COLVDD01 | Master deal pool, source for all scores |
| `scores_dmc_5M.sc` | 5M DMC (DouDou50) scores | COLVSC01 | GPU-batched, covers all 5M deals |
| `scores_isdd_500k.sc` | 500K IS-DD scores | COLVSC01 | Offset 0, covers deals [0, 500K). Merged from 5 sequential batches: [0,200K) at 50ms, [200K,500K) at 20ms |
| `archive/` | Old pool files (COLVDD01/COLVDR01) | mixed | Kept for reference, not used in training |

**IS-DD time budget note:** 50ms and 20ms produce statistically identical quality (tested on 10K matched deals: MAE vs DD = 18.99 vs 18.87, correlation 50ms↔20ms = 0.83, same as DD↔real). The variance between runs is dominated by determinization sampling noise, not thinking time. 20ms is 2× faster with no quality loss.

### File formats

**COLVDD01** (deals + DD): `magic[8] + count[u64] + N × (dealer[1] + hands[16] + dd_pts[4])` = 21B/deal

**COLVDR01** (legacy enriched): `magic[8] + count[u64] + N × (dealer[1] + hands[16] + dd_pts[4] + real_pts[4])` = 25B/deal. Still loadable via `DealPool::load_enriched()`.

**COLVSC01** (score layer): `magic[8] + name_len[u16] + name[utf8] + count[u32] + offset[u32] + N × pts[4]`. Lightweight, can have partial coverage (offset + count ≤ base pool size). Multiple score files can be loaded onto the same base pool.

### Enrichment methods

- **DouDou50 (DMC)**: GPU-batched inference (`enrich_pool` binary). ~25K deals/s on RTX 4090.
- **IS-DD**: CPU parallel with rayon (`enrich_pool_isdd` binary). Smart IS-DD, 20 determinizations, soft inference, no belief net. ~16 deals/s at 20ms on 32 cores, ~7 deals/s at 50ms. Sequential `--offset` flag for deterministic ordering. Also outputs `.sc` score file alongside legacy `.bin`.

### Usage in training

```bash
# DD-only (no score file needed)
--pool data/deals/base_5M.bin --reward dd

# With DMC scores
--pool data/deals/base_5M.bin --scores data/deals/scores_dmc_5M.sc --reward blend:0.75

# With IS-DD scores
--pool data/deals/base_5M.bin --scores data/deals/scores_isdd_500k.sc --reward blend:0.75

# Multiple score layers (last one activated)
--pool data/deals/base_5M.bin --scores data/deals/scores_dmc_5M.sc --scores data/deals/scores_isdd_500k.sc --reward blend:0.5
```

## Reward Modes

The training binary (`train_bid_nn`) supports:

| Mode | Flag | Description |
|------|------|-------------|
| DD only | `--reward dd` | `reward = dd_pts` (bid v2 default) |
| Real only | `--reward real` | `reward = real_pts` |
| Blend | `--reward blend:0.75` | `reward = alpha * dd + (1-alpha) * real` |
| Curriculum | `--reward curriculum:0.95:0.3` | Blend alpha anneals linearly from 0.95 to 0.3 over training |

## Experiment Setup

- **Architecture**: Dueling MLP 108->512x3->43 (same as bid v2, ~607K params)
- **Steps**: 5M (vs 20M for bid v2 production)
- **Envs**: 64, batch 512
- **Pool**: dd_5M_enriched.bin (DouDou50 real points)
- **Opponent diversity**: 40%->15% anneal (improved_v2 + aggressive + conservative + random)
- **Epsilon**: 0.3->0.02 over 3M steps
- **24x suit augmentation**
- **PER**: alpha=0.6, beta 0.4->1.0

## Results: H2H vs nn_v2 (Bid a Dede, 20M steps)

### With DMC (DouDou50) play

| Exp | Reward | Win% | Margin |
|-----|--------|------|--------|
| E | **Curriculum 95->30% DD** | **54.5%** | **+97** |
| D | 25% DD / 75% real | 53.0% | +41 |
| C | 50/50 | 49.2% | -26 |
| B | 75/25 | 48.5% | -50 |
| A | 100% DD | 46.5% | -81 |

Trend: more real = better. Curriculum is best (+97 margin vs nn_v2 in only 5M steps).

### With IS-DD play (no belief)

| Exp | Reward | Win% | Margin |
|-----|--------|------|--------|
| **A** | **100% DD** | **50.5%** | **+1** |
| B | 75/25 | 48.8% | -66 |
| E | Curriculum 95->30 | 48.8% | -70 |
| D | 25/75 | 48.5% | -69 |
| C | 50/50 | 47.5% | -76 |

**The ranking reverses completely.** DD pure is best with IS-DD play. All blends degrade.

### Round-Robin (DMC play, all models + reference)

| Rank | Bot | Win% | Margin |
|------|-----|------|--------|
| 1 | exp_e_curriculum | 52.0% | +70 |
| 2 | exp_b_blend75 | 51.8% | +29 |
| 3 | exp_c_blend50 | 50.6% | +39 |
| 4 | exp_d_blend25 | 49.8% | +12 |
| 5 | nn_v2_dmc50 (ref) | 48.4% | -42 |
| 6 | exp_a_dd | 47.4% | -108 |

## Key Finding: Play Model Bias

The "real" rewards in the enriched pool come from DouDou50 (DMC). When the bid model is evaluated with the same play model (DMC), it exploits DouDou50-specific error patterns — overfitting to the play model rather than learning better bidding.

With IS-DD (a different, stronger play method), this advantage disappears entirely. The DD-pure model, which learned no play-model-specific biases, is the most robust.

**Direct test**: curriculum+DMC (54.5% vs nn_v2 w/ DMC) loses to DD-pure+IS-DD when pitted against each other: 49.5% vs 50.5%.

## DD vs DMC vs IS-DD: Score Analysis (62K matched deals)

### Overview

| Metric | DD | DMC (DouDou50) | IS-DD |
|--------|------|------|-------|
| Mean pts | 85.8 | 82.7 | 82.5 |
| Std | 53.0 | 39.9 | 43.3 |
| Correlation w/ DD | 1.000 | 0.828 | 0.832 |

DMC and IS-DD have nearly identical mean scores (~82.5 vs 85.8 DD), but DMC↔IS-DD correlation (0.807) is lower than either's correlation with DD (~0.83). **They make different errors.**

### Complementarity

For each game (deal x suit), classified by closeness to DD (±10 pts):

| Category | Count | % |
|----------|-------|---|
| Both close | 53K | 21.7% |
| DMC close only | 51K | 20.7% |
| ISDD close only | 56K | 22.6% |
| Both far | 86K | 35.1% |

43% of games, exactly one model is close to DD while the other is not. They are strongly complementary.

### Ensemble

| Method | MAE vs DD |
|--------|-----------|
| DMC alone | 19.12 |
| IS-DD alone | 18.53 |
| **avg(DMC, IS-DD)** | **16.85** |

The simple average reduces MAE by 12% over the best individual method.

### By DD point range

| DD range | n | MAE DMC | MAE ISDD | MAE ensemble |
|----------|---|---------|----------|--------------|
| [0, 40) | 46K | 18.3 | 15.3 | 15.5 |
| [40, 80) | 74K | 14.6 | 14.8 | 12.3 |
| [80, 100) | 39K | 13.9 | 14.8 | 11.7 |
| [100, 120) | 35K | 14.3 | 15.1 | 12.4 |
| [120, 160) | 41K | 17.1 | 18.4 | 16.3 |

Biggest ensemble gain is in [40,120) — the critical range for bid/pass decisions.

### Contract success rates

| Threshold | DD count | DMC ok% | ISDD ok% |
|-----------|----------|---------|----------|
| >= 80 | 126K | 87.1% | 86.0% |
| >= 100 | 88K | 78.1% | 77.3% |
| >= 120 | 52K | 63.4% | 64.8% |
| >= 140 | 24K | 42.6% | 47.2% |
| >= 160 | 11K | 23.3% | 34.1% |

IS-DD is better at realizing high contracts (>= 120). DMC is slightly better at mid-range contracts. This aligns with IS-DD being a near-exact solver vs DMC's NN approximation.

### Extreme disagreements

Largest gaps are all **capots** (DD=252). DMC sometimes fails capots that IS-DD nails, and vice versa, on different deals. Neither is strictly dominant.

## Mixed-Team Enrichment (in progress)

To isolate how each team's play quality affects NS points, we enrich the same 100K deals with asymmetric play methods:

| Pool | NS method | EW method | Expected NS pts | Status |
|------|-----------|-----------|------------------|--------|
| `dd_100k_enriched_isdd.bin` | IS-DD | IS-DD | ~82.5 | Done |
| `dd_5M_enriched.bin` (matched) | DMC | DMC | ~82.7 | Done (62K matched) |
| `dd_100k_ns_dmc_ew_isdd.bin` | DMC | IS-DD | ~73 (NS weaker vs strong EW) | Running |
| `dd_100k_ns_isdd_ew_dmc.bin` | IS-DD | DMC | ~90+ (NS stronger vs weak EW) | Running |

**Hypothesis**: If NS=IS-DD/EW=DMC gives higher NS points than NS=DMC/EW=IS-DD, it confirms IS-DD plays stronger defense (lower NS pts when it's EW). The cross-comparison will let us decompose the DD→real gap into "NS play quality" vs "EW play quality" components.

**Analysis plan**: With all 5 views on the same deals (DD, DMC×DMC, ISDD×ISDD, DMC×ISDD, ISDD×DMC), we can build a 5-way comparison table and assess whether an ensemble of mixed perspectives gives a better training signal than any single one.

## Enrichment Speed

| Method | Speed | 100K deals | 1M deals |
|--------|-------|------------|----------|
| DouDou50 (GPU) | ~25K deals/s | ~4s | ~40s |
| IS-DD (CPU, 50ms) | ~7 deals/s | ~4h | ~38h |
| Mixed DMC/IS-DD (CPU) | ~10 deals/s | ~2.7h | ~28h |

## Conclusions

1. **DD pure is the most robust training signal** for bidding. It transfers across play methods without bias.
2. **Blending with play-model rewards overfits to that specific play model.** The gain with DMC eval is an artifact, not a real improvement.
3. **DMC and IS-DD are complementary** — they make different errors, and their ensemble (simple average) is 12% better than either alone as an approximation of DD.
4. **Curriculum training helps with DMC eval** but the benefit doesn't transfer to IS-DD. The curriculum essentially learns to exploit DMC-specific patterns.
5. **For future work**: an ensemble reward signal `avg(DMC, ISDD)` might give the benefits of real play data without the play-model bias, since the ensemble is closer to DD truth. This requires enriching pools with both DMC and IS-DD scores (same deals).

## Files

- **Training binary**: `colver-core/src/bin/train_bid_nn.rs` (supports `--reward dd|real|blend:X|curriculum:S:E`, `--scores path.sc`)
- **Enrichment (DMC)**: `colver-core/src/bin/enrich_pool.rs` (GPU, DouDou50)
- **Enrichment (IS-DD)**: `colver-core/src/bin/enrich_pool_isdd.rs` (`--offset` for sequential, outputs both `.bin` and `.sc`)
- **Enrichment (mixed)**: `colver-core/src/bin/enrich_pool_mixed.rs` (`--ns-method dmc|isdd`, CPU parallel)
- **Migration**: `colver-core/src/bin/migrate_pools.rs` (converts old pools to new `data/deals/` layout)
- **Sweep script**: `scripts/training/bid_v3_sweep.sh` (`--steps N`, `--matches N`, resume-safe)
- **Arena bots**: `arena/bots/bid_v3_exp_*.toml` (DMC), `arena/bots/bid_v3_exp_*_isdd.toml` (IS-DD)
- **Sweep results**: `models/bid_v3_exp/sweep_results.txt`, per-exp logs in `models/bid_v3_exp/exp_*/training.log`
