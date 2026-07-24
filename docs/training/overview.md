# Training, Evaluation & Experiment Commands

## Common Commands

```bash
# MCTS demos
cargo run -p colver-core --bin mcts_demo --release -- 100
cargo run -p colver-core --bin smart_ismcts_demo --release -- 100

# DD solver benchmark
cargo run -p colver-core --bin dd_bench --release -- 1000

# Python bindings
uv sync
uv run python3 -c "import colver; env = colver.Env(); env.reset()"

# Web frontend
uv run python -m colver.web

# Docker
docker build -t colver . && docker run -p 8000:8000 colver
```

## Belief Network Training

```bash
# Generate game replay data (COLVGM01 format, preferred)
cargo run -p colver-core --bin generate_game_data --release --features parallel -- \
  --dmc-model models/dmc_final.bin --games 500000 --output data/games.bin

# V2 training (304-dim, standard architecture)
cargo run -p colver-core --bin train_belief_net --features dmc_train --release -- \
  --replays data/training/games_500k.bin --epochs 200 --batch-size 512 --lr 3e-4 \
  --v2 --augment --cosine-lr --warmup-epochs 10 --val-split 0.05 \
  --output models/belief_net_v2.bin

# V3 temporal features (380-dim)
cargo run -p colver-core --bin train_belief_net --features dmc_train --release -- \
  --replays data/training/games_500k.bin --epochs 15 --batch-size 512 --lr 3e-4 \
  --v3 --augment --cosine-lr --warmup-epochs 3 --output models/race_v3.bin

# V3 3-class output (observer excluded, 3 classes: left/partner/right)
# count-reg default is now 0.1; 300 epochs for overnight training
cargo run -p colver-core --bin train_belief_net --features dmc_train --release -- \
  --replays data/training/games_500k.bin --epochs 300 --batch-size 512 --lr 3e-4 \
  --cosine-lr --warmup-epochs 10 --seed 42 --v2 --augment \
  --count-reg 0.1 --output models/belief_v3.bin

# Architecture variants: --variant cross_attn | suit_shared | var_mlp | aux_loss
# Variable MLP: --variant var_mlp --num-layers 3 --hidden 256
# Card count regularization: --count-reg 0.1
# suit_shared doesn't need --augment (equivariant by construction)
```

### Belief Evaluation (7 modes)

```bash
cargo run --bin belief_eval --release -- --model models/belief_net_v2.bin \
  --replays data/training/games_500k.bin --mode MODE --games 5000
```

Modes: `offline` (accuracy/CE/calibration), `match` (IS-DD NN vs heuristic), `diagnose` (per-card predictions), `scenario` (hand-crafted tests), `per_trick` (per-trick accuracy), `ablation` (input block importance), `ensemble` (multi-model averaging via `--model m1.bin,m2.bin`).

## DMC Card Play Training

```bash
# Rust training (candle, ~474 steps/s on 4090)
cargo run -p colver-core --bin train_dmc --features dmc_train --release -- \
  --num-envs 256 --steps 35000000 \
  --bid-model models/bid_nn_final.bin \
  --nn-bid-start 0.75 --nn-bid-end 0.95 --nn-bid-anneal-steps 20000000 \
  --eval-freq 1000000 \
  --eval-random-matches 100 \
  --eval-isdd-matches 10 --eval-isdd-time-ms 20 \
  --eval-checkpoint models/dmc_35.bin --eval-checkpoint-matches 50

# Python training (slower, ~140 steps/s)
PYTHONPATH=scripts/training uv run python scripts/training/train_dmc.py --num-envs 256 --steps 20000000

# Evaluation
uv run python scripts/analysis/eval_dmc.py models/dmc_final.pt \
  --games 200 --baseline smart --time-ms 20 --both-sides

# Export PyTorch weights to Rust binary format
python scripts/export/export_dmc_weights.py models/dmc_final.pt models/dmc_final.bin
```

## NN Bidding Training

```bash
# Bid a Dede (v2, default): 3×512, DD solver + 24× suit augmentation
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- \
  --hidden 512 --layers 3 --steps 20000000 --pool-file data/pools/dd_2.5M.bin

# Bid a Doudou (v1, legacy): 2×256, DouZero self-play
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- \
  --num-envs 64 --steps 5000000 --pool-size 1000000
```

Phase 1: pre-solves deal pool (1M deals x 4 suits). Phase 2: trains Dueling DQN with PER + opponent diversity. `BidNet::load` auto-detects hidden size (tries 256, 512, 1024).

## Removed: NN value-function training

MCTS with a learned value function was the wrong architecture for a card game —
every node costs a network eval, so the tree stays tiny — and a DouZero-style
direct Q-network (DMC) does the same job orders of magnitude faster. The `nn`
cargo feature, `features.rs`, `value_net.rs`, `generate_value_data` and
`nn_experiment` were removed on 2026-07-24; recover them from git history if the
idea is ever worth revisiting.

## PyPI Publishing

Published as [`colver`](https://pypi.org/project/colver/) via GitHub Actions with trusted publishing.

```bash
git tag v0.2.1 && git push origin v0.2.1
# Builds wheels for: x86_64-linux, aarch64-linux, x86_64-macos, aarch64-macos, x86_64-windows
```

## Experiment Binaries

All binaries: `cargo run -p colver-core --bin NAME --release -- ARGS`

| Binary | Feature | Description |
|--------|---------|-------------|
| **Evaluation** | | |
| `arena` | rand / parallel | **Bot comparison framework** — `list`, `h2h`, `round-robin`, `results`, `trace` |
| `isdd_sweep` | rand | IS-DD parameter sweep (count / time) |
| `bench` | — | Performance benchmark (~1.3M rollouts/sec) |
| `dd_bench` | — | DD solver benchmark |
| `belief_eval` | rand | Belief network evaluation (7 modes) |
| `eval_beliefs` | rand | Belief model comparison |
| `bench_world_cred` | rand / dmc_train | **World-sampler credibility benchmark** (playgen vs belief vs uniform) |
| `bench_world_compress` | rand | World-sampler compression study |
| `bench_logp_cred` | rand | Log-prob vs credibility correlation |
| `bench_playgen_gpu` | dmc_train | Playgen GPU vs CPU forward parity + throughput |
| `calibrate_winprob` | rand | Contract win-probability calibration |
| `deal_bias` | rand | Gather-cut dealing vs competition shuffling |
| `retro_eval` | rand | Retrospective bid evaluation |
| `replay_dmc_vs_isdd` | parallel | DMC vs IS-DD on a replay corpus |
| **Serving** | | |
| `playgen_gpu_server` | gpu_server | **Playgen GPU sidecar** — IS-DD's world source in production |
| **Training** | | |
| `train_dmc` | dmc_train | DMC card-play training |
| `train_bid_nn` | dmc_train | Bid NN training |
| `train_joint` | dmc_train | Triforge joint bid+play training |
| `train_belief_net` | dmc_train | Belief network training |
| `train_playgen` | dmc_train | Playgen world-model training |
| `export_playgen` | dmc_train | Playgen safetensors → `.bin` inference weights |
| **Data generation** | | |
| `gen_pool` | rand | DD deal pool (COLVDD01) |
| `enrich_pool` / `enrich_pool_isdd` / `enrich_pool_mixed` | dmc_train / parallel | Score layers (COLVSC01) |
| `generate_belief_data` | rand | Belief training data (COLVBL01) |
| `gen_bid_belief_data` | parallel | Bid-belief training data (COLVBB01) |
| `generate_game_data` | rand | Game replays (COLVGM01) |
| `gen_bumblebid_pool` | rand | Bumblebid pool |
| `migrate_pools` | rand | Legacy pool → base + score layers |
| **Analysis / diagnostics** | | |
| `distill_bid` / `distill_play` | rand / parallel | Q-value dumps for interpretability |
| `dump_obs_q` / `dump_probe_data` | rand | Observation + Q-value probes |
| `audit_bid_v5` | rand | Bid v5 audit |
| `maxi_diagnose` | rand | Maxi vs DMC play-by-play |
| `bid_debug` / `bid_compare` / `bid_nn_eval` | rand | Bidding printouts and comparisons |
| **Legacy tournaments** (superseded by `arena`) | | |
| `bid_tournament` / `bid_nn_tournament` / `v2_tournament` | rand | Round-robin bidding |
| `match_experiment` / `bidding_experiment` / `oracle_experiment` / `strength_experiment` | rand | Older head-to-heads |
| `dd_calibrate` | rand | DD bidding calibration |
| `mcts_demo` / `smart_ismcts_demo` | rand | Demos |

## IS-DD Sweep Results

Recommended web configs based on sweep (200 deals, vs DouDou35 DMC play model):
- **20ms time-limited + soft inference**: ~48% vs DouDou35, ~230ms/deal
- **50ms time-limited**: ~57%, 515ms/deal (higher quality)
- Gains plateau sharply after D=8 determinizations
- Soft inference worth it at D≥16 (+3.5% for 7% more compute)

Note: The default play model is now **DouDou50** (411-dim canonical ResNet, 50M steps). DouDou35 (415-dim legacy, 35M steps) remains available for backward compatibility.
