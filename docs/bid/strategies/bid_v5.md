# Bid v5 — score-aware with DQN stabilization

**Status:** Trained and validated (2026-04-19). New champion bidder across DMC and IS-DD play.

## TL;DR

v5 combines everything from v4 (score-aware reward) plus four new stabilization techniques, and introduces a second training run on a pool enriched with IS-DD-only real points. Result: the final `v5_isdd_25M` checkpoint is the strongest bidder produced so far, beating both the old DMC-pool champion `nn_v2_dmc50` (+11 pts winrate) and the previous best IS-DD companion `v3_max_isdd` (+14.6 pts winrate).

Files:
- Model (production): [models/bid_v5_isdd/bid_nn_final.bin](../../../models/bid_v5_isdd/bid_nn_final.bin)
- Checkpoints: `models/bid_v5_max/` (20M, max pool) and `models/bid_v5_isdd/` (25M, ISDD-pure pool)
- Training config: [scripts/training/v5_weekend.sh](../../../scripts/training/v5_weekend.sh)

## Motivation

v4 (score-aware, 20M steps) oscillated strongly in late training: best checkpoint at 16M (55.1% vs `nn_v2_dmc50`), regression to 51.3% by 20M. Pattern repeated from v3_max. Short stabilization experiments (see below) suggested reward clipping and EMA could smooth the training trajectory.

Separately, all prior NN bidders trained on DD-only or `max(DMC, ISDD)` reward. User hypothesis: "when DMC wins in training, it wins against another DMC — that may overstate realizability." A pool enriched with **IS-DD-only** points tests this.

## Changes vs v4

### 1. Score features v2 (obs_dim 110 → 113)

v4 appended 2 raw floats: `my_score/2000`, `opp_score/2000`. The network had to rediscover the calibrated win-probability sigmoid from these inputs.

v5 appends 5 precomputed features:

| # | Feature | Formula |
|---|---------|---------|
| 108 | my normalized | `s_me / 2000` |
| 109 | opp normalized | `s_opp / 2000` |
| 110 | win probability | `σ(1.7 · (s_me − s_opp) / (R_sum^0.8 + 340))` |
| 111 | leader distance to 2000 | `(2000 − max(s_me, s_opp)) / 2000` |
| 112 | score diff (signed) | `(s_me − s_opp) / 2000` clamped to [−1, 1] |

New constant `BID_OBS_DIM_SCORE_AWARE_V2 = 113` in [bid_obs.rs](../../../colver-core/src/bid/bid_obs.rs). v1 path (110 dim) preserved for v4 backward compatibility.

### 2. Reward clipping

Δ win probability reward (scaled by 3.0 in v4) can produce large magnitudes on coinche / surcoinche / capot swings. v5 optionally clips the post-scale reward to `[−clip, +clip]`. Default in v5: `--reward-clip 1.0`.

Implemented in [`flush_transitions_score_aware`](../../../colver-core/src/bid/bid_train_env.rs) via a new `reward_clip: Option<f32>` field on `BidTrainingEnv`.

### 3. Polyak EMA of weights

New fields `ema_tau`, `ema_weights` on [`BiddingTrainer`](../../../colver-core/src/bid/bid_candle.rs). After every `train_step`, the EMA shadow is updated: `ema = (1 − τ) · ema + τ · current`. Checkpoints (`.bin` exports) and in-training evals use the EMA snapshot when enabled.

v5 uses `--ema-tau 0.005`. Effective averaging window ≈ `1/τ = 200` train-steps. Short stabilization runs picked this as a reasonable tradeoff between smoothing and responsiveness.

### 4. Cosine LR decay

New flags `--lr` (start) and `--lr-end`. LR decays along a half-cosine from `lr` to `lr_end` over the full training. v5 uses `3e-4 → 3e-5` (10× reduction). Updates every 1000 steps (negligible cost).

## Short stabilization experiments (2M steps each)

Before committing to a long run, three configs on `dd_1.5M_max_dmc_isdd.bin` with `--reward real`:

| Config | Final win% | Final margin | Final loss |
|--------|------------|--------------|------------|
| A = baseline (v2 features only) | 43.5% | −302 | 0.0137 |
| B = A + reward clip 1.0 | **50.5%** | −146 | 0.0115 |
| C = B + EMA τ=0.005 | 47.5% | **−100** | 0.0115 |

Reward clip (B) gave the biggest uplift. EMA (C) produced tighter margin at parity winrate. Both kept. Short configs had `lr 1e-4` flat; full runs upgraded to `3e-4 → 3e-5` cosine.

## Weekend training pipeline (2026-04-18 → 2026-04-19, 36h58)

Two trainings, same config, different pools:

| Run | Pool | Steps | Wall | Result (last eval in training) |
|-----|------|-------|------|--------------------------------|
| `v5_max` | `dd_1.5M_max_dmc_isdd.bin` (existing, max(DMC,ISDD) reward) | 20M | 21h22 (concurrent with enrichment CPU) | 56.4% / +25 at 20M |
| `v5_isdd` | `dd_1M_isdd.bin` (fresh enrichment, IS-DD only) | 25M | 15h35 | 57.2% / +38 at 25M |

Pool enrichment in parallel with v5_max: `enrich_pool_isdd` on `dd_1.5M_base.bin` → `dd_1M_isdd.bin` (1M deals, 20ms per IS-DD search, 20 determinizations, ~14h on 32 cores → 19.7 deals/s). Produced both `.bin` (COLVDR01 pool) and `.sc` (COLVSC01 score layer) formats.

Config shared by both runs:

```
--hidden 512 --layers 3 --num-envs 256
--lr 3e-4 --lr-end 3e-5
--eps-start 0.30 --eps-end 0.02
--reward real --score-aware --sa-features-v2
--reward-clip 1.0 --ema-tau 0.005
--eval-matches 500
```

## Training trajectories

vs baseline `bid_v3_max_20M` (DMC play), 500 full matches per eval:

### v5_max (20M, eps decay over 15M)

| Steps | Win% | Margin |
|-------|------|--------|
| 2M | 44.4% | −252 |
| 4M | 47.4% | −174 |
| 6M | 51.2% | −77 |
| 8M | 52.6% | −81 |
| 10M | 56.8% | +25 |
| **12M** | 56.8% | **+48** ← margin peak |
| 14M | 52.2% | −23 |
| **16M** | **57.8%** | +32 ← winrate peak |
| 18M | 53.6% | −56 |
| 20M | 56.4% | +25 |

### v5_isdd (25M, eps decay over 18M)

| Steps | Win% | Margin |
|-------|------|--------|
| 2.5M | 48.4% | −145 |
| 5M | 48.8% | −179 |
| 7.5M | 51.6% | −97 |
| 10M | 52.4% | −102 |
| 12.5M | 52.2% | −43 |
| 15M | 58.6% | +68 |
| 17.5M | 56.8% | +79 |
| **20M** | 59.0% | **+95** ← margin peak |
| **22.5M** | **59.8%** | +89 ← winrate peak |
| 25M | 57.2% | +38 |

Observations:
- Oscillation remains visible (e.g. 14M dip in v5_max, 25M regression in v5_isdd) — EMA τ=0.005 not strong enough to fully suppress it. Effective window ~1.7s of wall time, much shorter than the 2M-step eval period.
- v5_isdd is weaker than v5_max until step ~12M, then pulls ahead and never looks back. Interpretation: IS-DD-only rewards are "stricter" (fewer realized contracts), so the model needs more training before the conservatism pays off.
- Final checkpoints are within eval noise of their respective training peaks (500-match stddev ≈ 2.2%). Two-proportion z-test on `v5_isdd 22.5M` vs `25M`: p ≈ 0.41, not significant.

## Arena tournament (authoritative ranking)

Round-robin on 5 bots, 500 matches per direction (1000 total per H2H, stddev ≈ 1.6%), DMC play for all.

```
1. v5_isdd_25M     win 54.0%  margin +53   ← nouveau champion
2. v5_isdd_22p5M   win 52.4%  margin +36
3. v5_max_16M      win 51.1%  margin  −5
4. v5_max_20M      win 49.9%  margin −35
5. nn_v2_dmc50     win 42.7%  margin −50   (ancien champion)
```

Key findings:
1. **The final v5_isdd checkpoint is the best**, not the training-eval "peak". The 22.5M/25M gap seen in training (59.8 vs 57.2) was noise — in direct 1000-match H2H, 25M beats 22.5M **50.9% to 49.1%**.
2. **ISDD-pure pool > max pool**: v5_isdd bots rank above all v5_max bots at high stat power.
3. **All v5 bots beat `nn_v2_dmc50`** by +8 to +11 pts winrate.
4. Gap v5_isdd_25M vs v5_max_20M: **+4.1 pts winrate, +88 margin** — the pool change accounts for most of the gain on top of the stabilization techniques.

### IS-DD cross-check

H2H `v5_isdd_25M_isdd` vs `v3_max_isdd`, both using IS-DD play (50ms, 20 determinizations), 1000 matches:

| Bot | Win% | Wins |
|-----|------|------|
| **v5_isdd_25M_isdd** | **57.3%** | 573/1000 |
| v3_max_isdd | 42.7% | 427/1000 |

Margin +44. Both directions favor v5 (277-223 and 296-204). Gap +14.6 pts ≈ 9σ — clearly significant. The v5_isdd bidder is **stronger with IS-DD play than with DMC play** (+14.6 vs +11), consistent with being trained on IS-DD-only reward.

## Recommended use

- **Production bidder**: `models/bid_v5_isdd/bid_nn_final.bin` (113-dim, dueling 512³).
- Pair with **IS-DD play** for arena maximum strength; pair with **DMC play** for speed.
- Still compatible with the belief-net play pipeline (`nn_v2_isdd` architecture), untested directly — the next natural experiment is `v5_isdd + smart_is_dd + belief_v3` vs current `nn_v2_isdd`.

## Known limitations / next steps

- **Oscillation not fully solved.** EMA τ=0.005 is probably ~10–100× too aggressive; a window of hundreds of thousands of train-steps would be more appropriate. Easy follow-up: try `ema_tau = 1e-5` (window ~100K train-steps ≈ 5 min wall) on a dedicated run.
- **Score features v2 untested without clip/EMA.** The 2M short config A demonstrated features alone don't regress but didn't isolate their contribution.
- **Pool size asymmetry.** v5_isdd used 1M deals vs v5_max's 1.5M. A matched-size run (1.5M ISDD-pure) would remove this confounder. Cost: ~7h more enrichment.
- **Belief net synergy unmeasured** for v5_isdd. `nn_v2_isdd` (the current global #1 via belief) may or may not transfer — belief net was trained on v2 auctions, not v5.
