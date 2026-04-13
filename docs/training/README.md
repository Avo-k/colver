# Training Documentation

How to train each model in the stack.

- [overview.md](overview.md) — full training/eval/experiment command reference

## Training binaries

| Binary | Trains | Feature flag |
|--------|--------|--------------|
| [train_bid_nn.rs](../../colver-core/src/bin/train_bid_nn.rs) | Bid NN (Dueling MLP, supports `--reward dd|real|blend|curriculum`) | `dmc_train` |
| [train_dmc.rs](../../colver-core/src/bin/train_dmc.rs) | Play NN (DMC Q-network) | `dmc_train` |
| [train_joint.rs](../../colver-core/src/bin/train_joint.rs) | Triforge: alternating bid/play best response | `dmc_train` |
| [train_belief_net.rs](../../colver-core/src/bin/train_belief_net.rs) | Belief net | `dmc_train` |

## Pipelines documented elsewhere

- [Triforge](../play/experiments/triforge.md) — joint bid+play training
- [Bumblebid](../bid/architectures/bumblebid.md) — transformer experiments
- [v3 reward study](../bid/reward_studies/v3_reward_study.md) — bid reward signal study
