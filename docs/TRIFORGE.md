# Triforge: Iterative Bid+Play Training

## The Problem

Training a bid NN and a play NN simultaneously fails because of coupled non-stationarity: each network learns against a moving target.

- The bid NN learns to bid for a play NN that changes every batch
- The play NN learns to play contracts that the bid NN wouldn't have bid yesterday
- Replay buffer churns completely every ~15K steps — any distribution shift causes loss cliffs
- Opponent pools made it worse, not better (destabilize the replay buffer faster)

We observed this empirically across 6+ joint training runs: loss plateaus and cliffs every time the NN bid fraction ramps up, even though eval win rates sometimes improved.

The previous pipeline also had a fundamental problem: the bid NN was trained on Double-Dummy (DD) oracle results. DD assumes perfect play by both sides, which produces unrealistic success rates — the bid NN learned that 84% of contracts succeed, but under realistic play only ~63% do (see [BID_RULES_STUDY.md](BID_RULES_STUDY.md)).

## The Solution: Iterative Best Response

Inspired by Fictitious Self-Play (Heinrich & Silver, 2015) and the Skyrim triforge analogy: **train one network at a time, freeze the other**.

```
Phase 0:  Play v0 = dmc_35.bin (DouDou35)
          Bid v0  = bid_nn_final.bin (Bid a Doudou)

Cycle 1:  bid-only  → train Bid v1 against frozen Play v0
          play-only → train Play v1 against frozen Bid v1

Cycle 2:  bid-only  → train Bid v2 (Bid a Dede) against frozen Play v1
          play-only → train Play v2 (DouDou50) against frozen Bid v2

          ...until eval stops improving
```

**Current defaults:** Bid a Dede (bid v2, 108→512³→43) + DouDou50 (play v2, 411→1024³→32 ResNet).

At each phase, one network is frozen (loaded from checkpoint, no gradient updates). This gives the training network a **stationary environment** — exactly the condition under which Q-learning converges reliably.

### Why this fixes the DD calibration problem

When we train the bid NN in bid-only mode against a frozen play NN:
- The play NN plays realistically (not perfectly like DD)
- The bid NN receives actual game outcomes as reward signal
- A bid that "should succeed" according to DD but fails under real play gets negative reward
- The bid NN learns calibrated contract success rates for the actual play level

And when we train the play NN against a frozen bid NN:
- The play NN sees the specific distribution of contracts that the bid NN actually bids
- No wasted capacity learning to play contracts that would never be bid
- The reward signal is stable (same bid distribution throughout training)

## Architecture Changes

### Play NN: ResNet + Canonical Encoding

**ResNet skip connections** on the Dueling Q-Network trunk. Same weights, different forward pass:

```
obs(411) → FC(1024) → LN → ReLU                    # layer 0: input projection
         → FC(1024) → LN + skip → ReLU              # layer 1: residual block
         → FC(1024) → LN + skip → ReLU              # layer 2: residual block
         → Value head (1) + Advantage head (32)      # dueling output
```

Skip connections improve gradient flow through the trunk without adding parameters. Proven effective in DouRN (2024) for the same game type.

**Canonical suit encoding (411 floats)**: trump always in slot 0, non-trump suits sorted by `(card_count, rank_pattern)` descending based on the player's initial hand. Two hands that differ only in which non-trump suit has which cards produce identical observations. No suit augmentation needed for the play NN.

### Bid NN: Scaled Up

From 2x256 (Bid a Doudou, v1) to **3x512** (Bid a Dede, v2). The Bridge 2024 baseline used 4x1024 for bidding — the old 2x256 was undersized. `BidNet::load` auto-detects hidden size (tries 256, 512, 1024), so old models still load.

Both BiddingTrainer and BidNet now accept a `layers` parameter. The `--bid-layers` and `--bid-hidden` args control this.

## Implementation

### Training modes

`train_joint` supports three `--mode` values:

| Mode | Play NN | Bid NN | Play buffer | Bid buffer | Play opponents |
|------|---------|--------|-------------|------------|----------------|
| `joint` | trains | trains | active | active | self-play + random |
| `play-only` | **trains** | frozen | active | **skipped** | self-play + random |
| `bid-only` | frozen | **trains** | **skipped** | active | **all frozen play** |

In `play-only` mode:
- `--resume-bid` required (the frozen bid model)
- `nn_bid_fraction = 1.0` — always use the frozen bid NN for bidding
- Bid transitions are not recorded, bid training is skipped
- Play opponents include 10% random for diversity

In `bid-only` mode:
- `--resume-play` required (the frozen play model)
- All play actions come from the frozen play NN (no random, no pool)
- Play transitions are not recorded, play training is skipped
- Bid diversity from heuristic opponents (phased warm-up still applies)

### Orchestration

```bash
# Single phase: train play against frozen bid
cargo run -p colver-core --bin train_joint --features dmc_train --release -- \
    --mode play-only \
    --resume-bid models/bid_nn_final.bin \
    --resume-play models/dmc_35.bin \
    --steps 10000000 \
    --save-dir models/triforge/cycle1_play

# Full triforge: 3 cycles of bid→play alternation
./scripts/triforge.sh --cycles 3 --play-steps 10000000 --bid-steps 5000000
```

The script chains phases, passing each phase's output as the next phase's frozen model. Results saved to `models/triforge/cycleN_{bid,play}/`.

### Evaluation

Eval always compares against the fixed reference:
- **vs random**: (our_play + frozen_bid) vs random play
- **vs DouDou35**: (our_play + our_bid) vs (DouDou35 + Bid a Doudou)

This measures absolute progress against the previous-generation system. 500 duplicate matches per eval (±2.2% precision).

## What Was Considered

### Joint training (what we tried first)

Full simultaneous training of both networks. Worked for ~2M steps then hit loss cliffs whenever the NN bid fraction ramped up. The replay buffer (2M entries) churns completely every ~15K steps, so any bid distribution shift causes all stored transitions to become stale within seconds.

We tried:
- Phased warm-up (heuristic → NN bidding): delayed the cliff but didn't prevent it
- Opponent pools (historical model checkpoints): made it worse
- Larger replay buffers (200K → 2M): slightly helped but didn't solve
- Disabling pools entirely: best result so far (92% vs random, 62% vs DouDou35 at 5.5M steps)

Joint training can work but requires careful tuning and the results are noisy. The triforge approach is simpler and more reliable.

### DD oracle for bid training (original pipeline)

The original approach: pre-solve deals with the DD solver, train the bid NN on DD success rates. Problems:
- DD assumes perfect play → unrealistic contract success rates
- 84% of contracts "succeed" under DD but only 63% under realistic play
- The bid NN learns to overbid because DD never punishes overcommitment
- DD solve time (~52ms/deal) limits training throughput

### Architecture alternatives considered

| Architecture | Pros | Cons | Verdict |
|---|---|---|---|
| **Transformer (play)** | SoTA on Hearts/Skat (GO-MCTS 2024) | Major rewrite, unclear gain for 32-card game | Future research |
| **LSTM history** | DouZero original used it | We already have positional encoding; adds sequential dependency | Not needed |
| **CNN on card grid** | Works for Mahjong (Suphx) | Belote card layout doesn't map to 2D grid naturally | Poor fit |
| **PPO / Actor-Critic** | More stable gradients, clipped policy updates | On-policy = can't reuse replay buffer, lower sample efficiency | Worth trying later |
| **Deeper MLP (5 layers)** | More capacity | Degrades without residual connections; diminishing returns | ResNet on 3 layers is the sweet spot |
| **ResNet MLP (chosen)** | Better gradient flow, proven in DouRN | Zero new parameters, trivial to implement | Implemented |
| **Larger bid NN (chosen)** | Bridge baseline uses 4×1024; our 2×256 was small | More GPU memory | 3×512 is the compromise |

### Canonical suit ordering vs augmentation

We went through three iterations:
1. **24× suit augmentation** (original): permute all 4 suits, applied at sample time
2. **6× non-trump augmentation**: trump in slot 0, permute 3 non-trump suits
3. **Canonical ordering** (final): sort non-trump suits by `(count, rank_pattern)` descending → zero augmentation needed

The canonical approach is both simpler (no augmentation code in the training loop) and more principled (the network literally cannot learn spurious suit-based patterns).

## Expected Training Budget

| Phase | Steps | Time (est. RTX 4090) | Purpose |
|---|---|---|---|
| Bid training | 5M | ~30 min | Learn calibrated bidding |
| Play training | 10M | ~1.5 hours | Learn to play bid contracts |
| Full cycle | 15M | ~2 hours | One bid→play iteration |
| 3 cycles | 45M | ~6 hours | Full triforge convergence |

Compare: joint training ran for 35M steps (~6 hours) with worse results.

## File Reference

- [train_joint.rs](../colver-core/src/tests/train_joint.rs) — training binary (`--mode` flag)
- [dmc_candle.rs](../colver-core/src/dmc_candle.rs) — play NN (ResNet DuelingQNet)
- [bid_candle.rs](../colver-core/src/bid_candle.rs) — bid NN (variable layers)
- [dmc_obs.rs](../colver-core/src/dmc_obs.rs) — canonical play encoding (411-dim)
- [joint_env.rs](../colver-core/src/joint_env.rs) — vectorized environment
- [scripts/triforge.sh](../scripts/triforge.sh) — orchestration script
- [scripts/monitor_joint.py](../scripts/monitor_joint.py) — training dashboard
