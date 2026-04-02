# CLAUDE.md

## Build & Test Commands

```bash
cargo check                                    # Check compilation (both crates)
cargo test -p colver-core                      # Run all core tests
cargo test -p colver-core -- test_name         # Run a single test
cargo test -p colver-core --release            # Tests in release mode
cargo run -p colver-core --bin bench --release # Performance benchmark (~1.3M rollouts/sec)
cargo run -p colver-core --bin train_joint --features dmc_train --release -- --num-envs 256 --steps 35000000  # Joint bid+play training
cargo run -p colver-core --bin train_joint --features dmc_train --release -- --mode play-only --resume-bid models/bid_v2/bid_nn_final.safetensors --bid-hidden 512 --bid-layers 3 --num-envs 256 --steps 50000000 --eval-freq 1000000 --save-freq 2000000  # Triforge: play-only phase with bid_v2
./scripts/triforge.sh --cycles 3  # Full triforge: alternating bid/play training
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- --hidden 512 --layers 3 --steps 20000000 --pool-file data/dd_pool_1M.bin  # Standalone bid NN training
uv sync                                        # Build and install Python bindings
uv run python -m colver.web                    # Run web frontend → http://localhost:8000
```

**Cargo features:** `rand` (default), `parallel` (rayon), `nn` (NN value function), `dmc_train` (candle GPU training for DMC + bid NN + belief net)

See [docs/TRAINING.md](docs/TRAINING.md) for all training, evaluation, and experiment commands.

## Architecture

Belote Contrée game engine optimized for millions of RL rollouts/sec. Rust core with PyO3 Python bindings.

**Workspace:** `colver-core` (pure Rust, zero deps by default) + `colver-py` (PyO3/numpy FFI) + `python/colver/web/` (FastAPI/WebSocket frontend)

### Card Representation (`card.rs`)

`Card = u8` (0-31), `CardSet = u32` (bitmask). Bit layout: Spades[0-7], Hearts[8-15], Diamonds[16-23], Clubs[24-31]. Rank bits: 7=0, 8=1, 9=2, J=3, Q=4, K=5, 10=6, A=7 (plain strength order). Trump strength: J(7) > 9(6) > A(5) > 10(4) > K(3) > Q(2) > 8(1) > 7(0).

### GameState (`state.rs`)

`GameState` is `Copy` and ≤64 bytes (compile-time enforced). Players: 0=N, 1=E, 2=S, 3=W. Teams: 0=NS (players 0,2), 1=EW (players 1,3). Partner = `player ^ 2`.

### Action Encoding

**Bidding (43 actions, u64 mask):** 0=PASS, 1-36=bids (value_idx×4 + suit_idx + 1, values 80-160, suits 0-3 = S/H/D/C), 37-40=capot×4 suits, 41=COINCHE, 42=SURCOINCHE.

**Playing (32 actions, u32→u64 mask):** Action = card index 0-31 directly.

`GameState::legal_actions() -> u64` returns mask. `GameState::step(action: u8)` dispatches to bidding or play.

### Game Flow

Bidding → Playing → Done. Bidding ends on 3 passes after a bid, surcoinche, or 4 passes (void deal). Playing: 8 tricks of 4 cards. Dix de der: +10 (normal) or +100 (capot). Total card points = 152; with dix de der = 162 (normal) or 252 (capot).

### Key Rules (FFB official — see `REGLES-DE-LA-BELOTE-CONTREE.pdf`)

- Coinche **freezes** the contract (no more overbids, only surcoinche or pass)
- "Ne pisse pas": if can't overtrump opponent's cut, may discard instead of undertrumping
- Only 4 color suits (no Sans Atout / Tout Atout)
- Scoring: "points faits + demandés" (section 10.2 of PDF)

### Performance-Critical Path

`play.rs::legal_plays()` is the hottest function — all bitwise, no allocations. Target: >1M rollouts/sec single-threaded.

## Key Subsystems (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for details)

- **MCTS** (`mcts.rs`): Arena-based UCT, 1000 iters default, C=sqrt(2)
- **Smart IS-MCTS** (`smart_ismcts.rs` + `card_beliefs.rs`): Belief-weighted IS-MCTS, ~+7.5% vs naive
- **DD Solver** (`solver.rs`): Alpha-beta with TT, PVS, killer/history heuristics. ~13.5ms/solve avg
- **DMC Agent "DouDou35"** (`dmc_net.rs`): DouZero-style Q-network, 415→1024³→32 (legacy obs), pure Rust inference ~1ms. Supports `residual: bool` for skip connections (same weights, different forward). Superseded by **DouDou50** (411→1024³→32, canonical ResNet, trained 50M steps) as the default play model.
- **NN Bidder** (`bid_net.rs`): Dueling DQN, auto-detects hidden size (tries 256, 512, 1024). **Bid a Doudou** (v1): 114→256²→43, trained with DouZero self-play (`bid_nn_final.bin`). **Bid a Dede** (v2, default): 108→512³→43, trained with DD solver + 24x suit augmentation (`bid_v2/bid_nn_final.bin`).
- **Belief Network** (`belief_net.rs`): Card location prediction, V1/V2/V3 obs, multiple architecture variants
- **Bidding strategies** (`bid_eval.rs`): `BidADd` (NN, default), `Improved`, `Heuristic`, `Smart`, `Roro`, `Maxi`, `BidParams` (parametric)
- **Triforge Training** (`joint_env.rs` + `train_joint.rs`): Iterative best-response training — alternates bid-only and play-only phases with frozen partner. `--mode play-only|bid-only|joint`. Play NN: ResNet Dueling DQN (411→1024³→32, skip connections on layers 1-2). Bid NN: Dueling DQN (114→512³→43, configurable layers). Canonical play encoding (no suit augmentation), bid uses 24× augmentation. See [docs/TRIFORGE.md](docs/TRIFORGE.md).
  - **Weight formats:** Training checkpoints (candle) use `.safetensors` — required for `--resume-bid`/`--resume-play`. Inference weights use `.bin` (raw f32) — used by `BidNet::load`/`DmcNet::load` and arena TOML `model` paths. Triforge saves both formats at each checkpoint.
  - **Triforge play NN (DouDou50) in arena:** Use `residual = true` in TOML. Canonical obs (411-dim) auto-detected from weight file. Models saved to `models/play_v2/play_*.bin`.
- **Suit Augmentation** (`suit_perm.rs`): 24 suit permutations for data augmentation. Functions for belief obs (V1/V2/V3), DMC obs (415-dim), bid obs (108-dim), actions, and masks. TR variants (`permute_dmc_obs_tr` / `augment_play_batch_tr`) exist but unused since canonical ordering eliminates the need.

### Observation Layouts (for suit permutation / NN inputs)

**DMC play obs — legacy (415):** [0:32] hand, [32:160] trick 4×32, [160:256] played 3×32, [256:260] trump suit, [260:263] value/team/coinche, [263:275] voids 3×4, [275:279] scores, [279:351] bid history 12×6, [351:383] card trick idx, [383:415] card seq idx. Used by DouDou35 (legacy play model) and web inference.

**DMC play obs — canonical (411):** Fully canonical suit encoding: trump in slot 0, non-trump sorted by (card_count, rank_pattern) descending — `canonical_play_order(trump, initial_hand)`. No suit augmentation needed. [0:32] hand, [32:160] trick 4×32, [160:256] played 3×32, [256:259] value/team/coinche (no trump one-hot), [259:271] voids 3×4, [271:275] scores, [275:347] bid history 12×6, [347:379] card trick idx, [379:411] card seq idx. Used by DouDou50 (default play model) and joint training. `card_to_canonical`/`card_to_physical` convert between spaces; `current_player_order` computes the ordering from state+tracking.

**Bid obs (108):** [0:32] hand, [32:104] bid history 12×6, [104:108] position. Auction state (bid value, suit, coinche) removed — redundant with bid history.

**Replay buffers** (`dmc_replay.rs`): `PrioritizedReplayBuffer` is hardcoded to OBS_DIM=415/MASK=32. Use `FlexReplayBuffer` for other dims (e.g. joint training play: 411/32, bid: 114/43).

## Python Layer (`colver-py/` → `python/colver/`)

`Env` wraps GameState with IS-MCTS/DMC support. Built as `colver._colver`, re-exported from `colver.__init__`. See `python/colver/_colver.pyi` for type stubs.

## Web Frontend (`python/colver/web/`)

FastAPI + WebSocket + vanilla JS. Three modes: Play, Watch, Analysis. Models auto-downloaded at startup (DMC 10MB, bid NN 421KB, belief net 2MB).

**Annonces page** (`views/annonces.js`): BidNet Q-values + Oracle DD table + DouDou simulation table. Oracle shows raw success % per suit×threshold. DouDou table uses Wilson score lower bound (z=1.645) for color thresholds (green/gold/red) and scales font size by observation count (0.65rem at 1 obs → 0.85rem at 20+) so small-sample cells appear visually less prominent than well-sampled ones.

**Mobile (≤600px):** Play view hides N/E/W seats, shows only trick area + South hand. South hand spans full viewport width with dynamically computed card overlap (JS in `play.js` sets `--card-overlap` based on card count and available width). Card sizes use CSS custom properties (`--card-w`) — note `#play-table` overrides `:root` values, so mobile overrides must target `#play-table` specifically. Header is 61px on mobile (not 46px as on desktop).

## Arena: Bot Comparison Framework

Systematic head-to-head and round-robin evaluation of bot architectures on 2000-point matches. Bots are TOML configs — no recompilation needed to test new combinations.

**Directory structure:** `arena/bots/*.toml` (bot definitions), `arena/results/matches.csv` (persistent results). Binary: `colver-core/src/tests/arena.rs`.

```bash
cargo run --bin arena --release -- list                                          # List all bots
cargo run --bin arena --release -- h2h bot_a bot_b --matches 200                 # Head-to-head (200×2 with duplicate matching)
cargo run --bin arena --release -- round-robin --matches 100                     # Full round-robin
cargo run --bin arena --release -- round-robin --matches 50 --bots a,b,c         # Subset round-robin
cargo run --bin arena --release -- results                                       # Leaderboard from CSV
cargo run --bin arena --release -- results --bot nn_dmc35                        # Filter by bot
```

**Bot TOML format** (`arena/bots/<name>.toml`):
```toml
[bid]
strategy = "nn"                    # heuristic | improved | improved_v2 | smart | roro | maxi | petit_bide | moelleux | nn
model = "models/bid_nn_final.bin"  # required if strategy = "nn"
hidden = 256                       # hidden size for bid NN (default 256, bid_v2 uses 512)

[play]
method = "dmc"                     # naive_ismcts | smart_ismcts | is_dd | smart_is_dd | dmc | dmc_then_dd | oracle | heuristic
model = "models/dmc_35.bin"        # required if method = "dmc"
residual = false                   # skip connections for triforge models
time_ms = 20                       # time budget (ismcts/is_dd)
determinizations = 20              # for is_dd
switch_at = 5                      # for dmc_then_dd: switch to DD after N tricks (default 5)

[belief]                           # optional, for smart_ismcts / smart_is_dd
model = "models/belief_v3.bin"
use_hard_constraints = true
```

**Options:** `--matches N` (per direction, default 100), `--threads N` (default auto), `--seed N` (default 42). Each H2H runs both directions (duplicate matching) for variance reduction.

**Reference bots:** `nn_v2_isdd_no_belief` (Bid a Dede+SmartIsDd, **champion**), `nn_v2_isdd` (Bid a Dede+SmartIsDd+Belief), `nn_v2_dmc35` (Bid a Dede+DouDou35), `nn_dmc35` (Bid a Doudou+DouDou35), `nn_isdd` (Bid a Doudou+SmartIsDd+Belief).

**CSV format:** Columns include `bid_a,play_a,bid_b,play_b` labels. Parser auto-detects old (11-col) and new (15-col) formats.

**Iteration workflow:** Create a new `.toml` in `arena/bots/`, run `h2h` against champion, check `results`, iterate. No recompilation between experiments.

## Publishing & Deployment

**PyPI:** push `v*` tag → CI builds manylinux/macOS/Windows wheels via maturin → publishes automatically (trusted publishing).

**Docker:** `docker build -t colver . && docker run -p 8000:8000 colver`. Cross-builds for ARM64.
