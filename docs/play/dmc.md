# DMC (Deep Monte Carlo) Play Network

**Code:** [colver-core/src/dmc/dmc_net.rs](../../colver-core/src/dmc/dmc_net.rs) (CPU inference), [dmc_candle.rs](../../colver-core/src/dmc/dmc_candle.rs) (GPU training)

DouZero-style Q-network for play. Trained via self-play with episodic returns (final NS points). Pure feed-forward MLP, no search at inference time.

## Models

| Name | Arch | Obs dim | Notes |
|------|------|---------|-------|
| **DouDou35** | 1024³ Dueling | 415 (legacy) | Original, ~1ms/move |
| **DouDou50** (default) | 1024³ ResNet Dueling | 411 (canonical) | Skip connections layers 1-2, trained 50M steps against bid_v2 auctions |
| **play_v3_max** | 1024³ ResNet Dueling | 411 (canonical) | Same arch as DouDou50, 50M steps against bid_v3_max_20M auctions. Reaches 51% in-training eval vs DouDou50+bid_v2 (near-parity). The bid_v3_max advantage doesn't transfer to DMC play — see [bid/strategies/bid_v3_max.md](../bid/strategies/bid_v3_max.md) "Synergy" section. Not superior to DouDou50 in arena. Weights: `models/play_v3_max/play_final.bin` |

Canonical obs (411-dim) puts trump in slot 0 and sorts non-trump suits by a stable ordering — eliminates need for suit augmentation. See [colver-core/src/dmc/dmc_obs.rs](../../colver-core/src/dmc/dmc_obs.rs).

## Inference

```rust
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs;

let mut net = DmcNet::load("models/play_v2/play_final.bin")?;
net.set_residual(true);  // required for DouDou50

dmc_obs::write_observation_tr(&mut obs, 0, &state, &tracking);
let order = dmc_obs::current_player_order(&state, &tracking);
let canonical_mask = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
let (canonical_action, _) = net.best_action(&obs, canonical_mask);
let physical = dmc_obs::card_to_physical(canonical_action, &order);
```

**Critical:** without `set_residual(true)`, DouDou50 weights are silently misinterpreted and play is much weaker (MAE ~25 vs DD instead of ~19).

## Training

See [TRIFORGE](experiments/triforge.md) for the joint pipeline. Solo DMC training:

```bash
cargo run -p colver-core --bin train_dmc --features dmc_train --release -- \
    --num-envs 256 --steps 50000000 \
    --bid-model models/bid_v2/bid_nn_final.bin
```

## Prochaine itération

Une modification de l'observation invalide les poids (la première couche change de forme : pas de reprise depuis DouDou50). Ce qui doit monter à bord se note **avant** de lancer, dans [doudou_next.md](doudou_next.md) — première entrée : **la belote annoncée**, absente des deux layouts alors qu'elle est publique.

## Comparison with IS-DD

DMC and IS-DD have very similar mean MAE vs DD (~19) but make **different errors** — see [bid/reward_studies/v3_reward_study.md](../bid/reward_studies/v3_reward_study.md). Hamming distance between their plays on the same deal is **29/32 cards**: they are stylistically opposed (DMC plays Aces immediately, IS-DD pulls trumps).
