# Bumblebid: Transformer Bidding Network

## Overview

Bumblebid is a transformer-based bidding model for Belote Contrée, designed to replace the MLP-based Bid a Dede (bid_v2). It uses an encoder-only architecture inspired by ModernBERT, with token-level suit embeddings that naturally represent the variable-length auction sequence.

## Architecture

**Token sequence:** `[CLS] [POS_x] [card_1..card_8] [bid_val bid_suit] [bid_val bid_suit] ...`

Each token embedding = `primary_emb[id] + suit_emb[suit] + pos_emb[position]`.

- **Pre-norm RMSNorm** (no bias), **GeGLU FFN** (2/3 × 4 × d_model intermediate)
- **Multi-head attention** with learned positional embeddings
- **Dueling heads**: CLS → RMSNorm → V(s) + A(s,a) - mean(A) → 43 Q-values
- Max sequence: 34 tokens (2 header + 8 cards + 12 bid rounds × 2 tokens)

### Token encoding

Primary IDs (27): CLS=1, POS0-3=2-5, RANK0-7=6-13, VAL0-8=14-22, CAPOT=23, PASS=24, COINCHE=25, SURCOINCHE=26.
Suit IDs (5): S=0, H=1, D=2, C=3, NULL=4.
Cards sorted by suit×8+rank. Position token encodes dealer-relative seat.

### Input pipeline

Takes the same 108-dim bid observation as the MLP (BidNet). Internally converts to token sequence via `obs_batch_to_tokens()` in `bumblebid_candle.rs`. Tokenisation verified lossless (exhaustive roundtrip test on all bid types, hand encodings, positions).

## Training Infrastructure

Uses the exact same Rust pipeline as nn_v2 (Bid a Dede) via `--transformer` flag on `train_bid_nn`:
- **PER** (500K buffer, alpha=0.6, beta 0.4→1.0)
- **Opponent diversity** (improved_v2 + aggressive + conservative + random, 40%→15% anneal, rest self-play)
- **ε-greedy** (0.3→0.02 over 3M steps)
- **24× suit augmentation** per sample
- **DQN loss** = MSE(Q(s, a_taken), return) with PER importance weights
- **DD oracle** for reward (or blend with DouDou50 real play via `--reward blend:0.75`)

```bash
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- \
    --transformer --d-model 256 --layers 2 --n-heads 8 \
    --num-envs 64 --steps 20000000 --batch-size 256 \
    --pool-file data/pools/dd_5M_enriched.bin \
    --reward blend:0.75 \
    --save-dir models/bumblebid/rust_d256_20M
```

## Key Learnings

### DD Oracle is a Training Signal, Not a Player
See CLAUDE.md. The model must play its own auctions — oracle targets supervise the loss but the model's own actions drive trajectories.

### Tokenisation is Lossless
Exhaustively verified: 108-dim obs → tokens → reconstruct = exact roundtrip for hand, position, and all bid types. No float precision issues.

### Loss Descent ≠ Arena Performance
Training loss descends continuously but arena performance oscillates. The model cycles between "conservative" phases (low coinches, better arena) and "aggressive" phases (high coinches, worse arena).

### Coinche Dominance is the Key Problem
nn_v2 (reference) coinches 0.11 times/game. Bumblebid coinches 0.3-0.8/game depending on checkpoint. Coinches multiply scores ×2/×4, creating outsized rewards in the PER buffer that bias learning. The model oscillates rather than converging monotonically.

### Python Training Failed, Rust Succeeded
Multiple Python approaches (supervised CE, DQN self-play, DQN with nn_v2 opponents) all failed or were too slow. The existing Rust infra (PER, opponent diversity, suit augmentation, speed) is what made training work. Lesson: reuse proven infrastructure.

## Experiment Log

### Exp 1: Python supervised (CE + advantage-MSE on oracle targets)
**Setup:** CE on best oracle action + advantage-MSE. ε-greedy, replay buffer, model plays own auctions.
**Result:** Loss plateaus at ~0.144 (= variance of DD targets seen from single hand). No virtuous cycle because oracle targets are fixed. **Abandoned.**

### Exp 2: Python DQN with episode returns (self-play)
**Setup:** 4× BB self-play, DD return from final contract.
**Result:** Loss descends (1.97→1.0) but arena terrible (~20-43/200 vs nn_v2). Degenerate auctions from bad model. **Abandoned.**

### Exp 3: Python DQN with nn_v2 opponents
**Setup:** BB vs nn_v2, DD episode returns.
**Result:** Loss ~0.95 but arena still bad. Too slow (~100 steps/s), no PER. **Abandoned → Rust.**

### Exp 4: Rust d=128 L=2 H=4 (dd-only, 5M steps)
**Setup:** 410K params. First Rust integration.
**Result:** Coinches too high (0.82/game at 2M). ~30% win vs nn_v2.

### Exp 5: Rust d=256 L=2 H=8 (dd-only, 2M steps)
**Setup:** 1.6M params.
**Result:** 39% win vs nn_v2 (diff -227), ~50% vs heuristic. Significant improvement.

### Exp 6: Rust d=256 L=2 H=8 (blend 75/25, 20M steps) — BEST SO FAR
**Setup:** 1.6M params, reward = 75% DD + 25% DouDou50, 5M enriched pool.
**Results (1000 games per checkpoint):**

| Step | vs nn_v2 win% | diff | vs heur win% | diff | Co/game |
|---|---|---|---|---|---|
| 1M | 34.3% | -258 | 43.6% | -95 | 0.61 |
| 2M | 31.4% | -325 | 39.5% | -115 | 0.70 |
| **3M** | **36.6%** | **-174** | **46.8%** | **-53** | **0.34** |
| **4M** | 35.9% | -191 | **45.6%** | **-40** | 0.42 |
| 5M | 34.4% | -279 | 44.8% | -69 | 0.63 |
| 6M | 38.5% | -209 | 45.2% | -51 | 0.49 |
| 8M | 30.1% | -351 | 40.8% | -106 | 0.78 |
| 10M | 37.9% | -225 | 45.1% | -61 | 0.43 |

**Best checkpoint: 3M-4M.** Model oscillates between conservative/aggressive phases. No monotonic improvement beyond 3M despite loss continuing to decrease.

### Reference: nn_v2 (Bid a Dede)
Dueling MLP 108→512³→43 (607K params). 20M steps, dd-only. Coinche rate: 0.11/game.

## Batch Size / Num Envs Sweep (d=256 L=3 H=8, candle, 50K steps)

All configs run **sequentially on the same GPU** (not in parallel), so wall-clock times are comparable.
Steps/s varied within each run (slower during buffer warmup, faster after). eps annealed 0.3→0.02 over 40K steps.

| Config | Steps/s (avg) | Loss @50K | Episodes | Voids | ~Wall-clock |
|---|---|---|---|---|---|
| e64 b256 | ~50 | 0.177 | 505K | 1.2K | ~17 min |
| **e64 b512** | ~30 | **0.131** | 486K | 1.5K | ~28 min |
| e128 b256 | ~44 | 0.188 | 992K | 1.5K | ~19 min |
| **e128 b512** | ~27 | **0.123** | 966K | 1.3K | ~30 min |
| e256 b256 | ~37 | 0.204 | 2.0M | 2.5K | ~22 min |

**Takeaways:**
- Bigger batch (512 vs 256) helps loss significantly (~30% lower) at ~2x speed cost
- More envs beyond 128 hurts: e256 is slower AND worse loss than e128
- e128 b512 is best for loss (0.123) with 2x more episodes than e64 (fresher buffer)
- e64 b256 is best for speed but worse loss
- **Chosen config for long runs: e128 b512**

## Speed Profiling: Transformer vs MLP (e128 b512, candle, RTX 4090)

Isolated each component to find the bottleneck. All runs: 5000 steps, sequential.

| Config | Steps/s | What it measures |
|---|---|---|
| Transformer, no training (inference only) | 65 | Env stepping + transformer forward on 128 envs |
| **MLP** 512³ L=3, full training | **109** | MLP forward + backward + envs (reference) |
| **Transformer** d=256 L=3, b512 | **13** | Transformer forward + backward + envs |
| Transformer d=256 L=3, b256 | 22 | Smaller batch helps ~2x |
| Transformer d=256 L=3, e64 b512 | 14 | Fewer envs doesn't help much |

**The transformer is 8.4x slower than the MLP** with the same training infra. This is not a candle inefficiency — multi-head attention on 34 tokens × 3 layers × 8 heads is fundamentally more compute than a 512³ MLP. The tokenisation (108-dim obs → token sequence, done on CPU) adds overhead too.

We investigated tch-rs (PyTorch C++ backend) as an alternative to candle but CUDA detection fails on WSL2 due to driver version mismatches (known issue: [tch-rs #988](https://github.com/LaurentMazare/tch-rs/issues/988)). Burn was considered but benchmarks show similar GPU performance to candle.

**Conclusion:** The speed limitation is architectural (attention vs MLP), not framework-related. Options:
1. Accept ~50 steps/s (e64 b256) and run longer (10M in ~55h)
2. Use smaller transformer (d=128 L=2: ~133 steps/s but weaker)
3. Return to MLP and focus on training improvements (reward clipping, target network)

## Best Results So Far

**Best checkpoint: d=256 L=2 H=8, blend 75/25, @ 3M steps** (from Exp 6)
- vs nn_v2: 36.6% win, diff -174 (1000 games)
- vs heuristic: 46.8% win, diff -53
- Coinches: 0.34/game (best balance)

For reference, nn_v2 (MLP 512³, 20M steps): coinches 0.11/game, the target to beat.

## Next Experiments

1. **Reward clipping** — clip returns to [-1, +1] to reduce coinche reward dominance in PER
2. **Target network** — frozen target Q-network updated every N steps (standard DQN stabilization against oscillation)
3. **MLP with deeper exploration** — use nn_v2's exact architecture but with longer training / different hyperparams
4. **Hybrid approach** — MLP trunk + attention on bid history tokens only (cheaper than full transformer)

## Files

- **Model (candle):** `colver-core/src/bid/bumblebid_candle.rs`
- **Model (PyTorch):** `scripts/bumblebid/model.py`
- **Training binary:** `colver-core/src/bin/train_bid_nn.rs` (`--transformer` flag)
- **Data/tokenizer:** `scripts/bumblebid/data.py`
- **Pool tokenizer:** `colver-core/src/bin/gen_bumblebid_pool.rs`
- **Checkpoints:** `models/bumblebid/rust_d256_20M/bid_nn_*.safetensors`
