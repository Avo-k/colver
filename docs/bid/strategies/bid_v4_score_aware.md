# Bid v4 Score-Aware — match-context reward shaping

**Status:** Design complete, training not yet run.

## Motivation

All prior bid models (v1, v2, v3_max) optimize for **per-deal** reward — they treat every deal identically regardless of the match score. In Contrée, the first team to reach 2000 cumulative points wins the match. This creates score-dependent risk dynamics:

- **Early game (200/200):** a chute at 100 costs ~260 pts to the defense — painful but survivable. Many deals remain to recover.
- **Endgame (1750/1750):** the same chute at 100 gifts 260 pts → opponent reaches 2010 → **match lost instantly**.
- **Leading (1800/1200):** conservative play protects a large lead. Risky bids that chute can hand momentum back.
- **Trailing (1200/1800):** aggressive bids are rational — the expected value of conservative play trends toward losing the match anyway.

A score-aware model should learn these dynamics from the reward signal alone, without hardcoded rules.

## Approach

### Observation: +2 features for match scores

Append `my_score / 2000.0` and `opp_score / 2000.0` to the bid observation vector.

- New obs dim: **110** (was 108)
- Existing blocks unchanged: hand (32) + bid history (72) + position (4) + match scores (2)

During training, match scores are sampled uniformly from `[0, 2000) × [0, 2000)` per episode. Uniform sampling ensures coverage of all score states, including rare but critical endgame situations.

### Reward: Δ win probability

Instead of raw deal score, the reward is the **change in match win probability** caused by the deal:

```
reward = (P(win | after) - P(win | before)) × scale
```

Where `P(win | s_me, s_opp)` is a calibrated sigmoid:

```
P(win) = σ(1.7 × (s_me - s_opp) / ((R_sum)^0.8 + 320))
```

with `R_sum = (2000 - s_me) + (2000 - s_opp)`.

Scoring rules: surcontré ×3 (not ×4), contré base 160 (not 320), capot = contract at 250.

If a deal ends the match (someone ≥ 2000): `P(win) = 1.0` if I won, `0.0` if I lost (with higher-total tiebreaker when both cross 2000).

### Architecture

Same Dueling MLP as v2/v3: input → 512³ → 43 actions. Only the input dim changes (110 vs 108).

## Win Probability Calibration

The `P(win)` function was calibrated empirically by simulating 10,000 full matches (0/0 → 2000) using bid_v3_max_20M + DouDou50 in self-play. This produced 88,272 intermediate score states, each labeled with the eventual match winner.

### Calibration process

1. **Data generation:** `calibrate_winprob --matches 10000` plays full matches, recording the cumulative score before each deal and who ultimately won the match. Each of the ~8.8 deals per match produces one (ns_cum, ew_cum, winner) data point.

2. **Model fitting:** Five sigmoid variants were fitted against the binned data (weighted MSE, 100pt bins, min 20 samples per bin):

| Model | Formula | wMSE |
|-------|---------|------|
| 1. min-based | σ(k×Δ / max(2000-min, δ)) | 0.000532 |
| 2. sum-based | σ(k×Δ / max(4000-sum, δ)) | 0.000421 |
| 3. product-based | σ(k×Δ×2000 / max(R_me×R_opp, δ)) | 0.001077 |
| **4. power-sum** | **σ(k×Δ / (R_sum^α + δ))** | **0.000172** |
| 5. max-based | σ(k×Δ / max(2000-max, δ)) | 0.000770 |

Model 4 wins by **3.1×** over the baseline (Model 1). Best-fit parameters: **k=1.1, α=0.8, δ=200**.

3. **Validation:** The key insight from calibration is that Contrée matches have enormous per-deal variance (σ=386 pts/deal) due to coinche/surcoinche swings. This means even a 200-point lead in endgame only gives ~65% win probability, not the 95% a naive model would predict.

### Match statistics (from 10k calibration matches)

- **Deals per match:** mean 8.8, median 9, range 1-16
- **Taker deal score:** mean 342, median 260
- **Defense deal score:** mean 21, median 20
- **Net score std per deal:** 386 (huge variance from coinché/surco)
- **Void deal rate:** 0.04% (negligible)

### Example reward values

At **1750/1750** (p₀ = 50%):

| Contract | Made | Chute |
|----------|------|-------|
| 80 | +0.14 | −0.43 |
| 100 | +0.20 | **−1.50** (match lost) |
| 130 | **+1.50** (match won) | **−1.50** (match lost) |
| 80 coinché | **+1.50** | **−1.50** |

At **200/200** (p₀ = 50%):

| Contract | Made | Chute |
|----------|------|-------|
| 80 | +0.08 | −0.23 |
| 100 | +0.13 | −0.25 |
| 130 | +0.22 | −0.27 |

The contrast is exactly what we want: early-game rewards are mild and symmetric, endgame rewards have cliffs.

## Data

- **Calibration CSV:** `data/winprob_points.csv` — 88,272 rows of (ns_cum, ew_cum, winner) from 10k full matches.
- **Match replays:** `data/winprob_replays.bin` — COLVMR01 format, 10k matches with full deal-level detail (hands, bids, cards played, scores). 5.2MB.
- **Binary:** `colver-core/src/bin/calibrate_winprob.rs`

## Training (planned)

```bash
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- \
    --hidden 512 --layers 3 \
    --num-envs 64 --batch-size 512 \
    --steps 20000000 \
    --pool-file data/pools/dd_1.2M_max_dmc_isdd.bin \
    --reward real \
    --score-aware \
    --save-dir models/bid_v4_score_aware \
    --eval-freq 1000000 --save-freq 2000000
```
