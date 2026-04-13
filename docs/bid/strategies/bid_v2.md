# Bid v2 (Bid a Dede) — production reference

**Code:** [colver-core/src/bid/bid_net.rs](../../../colver-core/src/bid/bid_net.rs) (inference), [bid_candle.rs](../../../colver-core/src/bid/bid_candle.rs) (training)

**Weights:** `models/bid_v2/bid_nn_final.bin`

The current production bidding NN. Used by `nn_v2_*` arena bots.

## Architecture

| Field | Value |
|-------|-------|
| Type | Dueling DQN, MLP |
| Input | 108-dim ([bid obs](../../../colver-core/src/bid/bid_obs.rs)) |
| Hidden | 512 × 3 layers |
| Output | 43 actions (PASS, 36 bids, 4 capot, coinche, surcoinche) |
| Params | ~607K |

## Training

| Hyperparam | Value |
|------------|-------|
| Steps | 20M |
| Reward | DD oracle (`--reward dd`) |
| Pool | `dd_2.5M.bin` |
| Augmentation | 24× suit permutation |
| Replay | PER (alpha=0.6, beta 0.4→1.0) |
| Opponents | improved_v2 + aggressive + conservative + random (40%→15% anneal) |
| Epsilon | 0.3 → 0.02 over 3M steps |

Train command:
```bash
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- \
    --hidden 512 --layers 3 --steps 20000000 --pool-file data/pools/dd_2.5M.bin
```

## Behavior characteristics

- Coinche rate: **0.11/game** (compared to 0.3-0.8 for less-trained variants)
- Calibrated to DD optimal — does not overbid in expectation
- Conservative compared to [bid_v3_max](bid_v3_max.md), which trains on `max(DMC, ISDD)` real points

## Successors

- [bid_v3_max](bid_v3_max.md) — same architecture, trained on `max(DMC, ISDD)` reward signal. Equal or better in arena across both DMC and IS-DD play.

See [../reward_studies/v3_reward_study.md](../reward_studies/v3_reward_study.md) for the full reward signal comparison.
