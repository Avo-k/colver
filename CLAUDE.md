# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Check compilation (both crates)
cargo check

# Run all core tests (281 tests, plus more with --features nn or dmc_train)
cargo test -p colver-core

# Run a single test
cargo test -p colver-core -- test_name

# Run tests in release mode
cargo test -p colver-core --release

# Run the performance benchmark (~1.3M rollouts/sec target)
cargo run -p colver-core --bin bench --release

# Run MCTS vs random demo (default 100 games, ~8ms/game)
cargo run -p colver-core --bin mcts_demo --release
cargo run -p colver-core --bin mcts_demo --release -- 20  # custom game count

# Run Smart IS-MCTS demo (vs random + vs naive IS-MCTS)
cargo run -p colver-core --bin smart_ismcts_demo --release -- 100

# Run oracle experiment (bid achievability testing)
cargo run -p colver-core --bin oracle_experiment --release -- 200 2000

# Run bidding experiment (smart_bid vs heuristic)
cargo run -p colver-core --bin bidding_experiment --release -- 200 50

# Generate NN training data (fast mode with heuristic play)
cargo run -p colver-core --bin generate_value_data --release --features nn -- 10000 data/value_train.bin --fast

# Generate NN training data (slow mode with IS-MCTS play)
cargo run -p colver-core --bin generate_value_data --release --features nn -- 1000 data/value_train.bin

# Train value network (requires PyTorch)
python scripts/train_value_net.py --data data/value_train.bin --output models/value_net.bin

# Run NN evaluation experiment
cargo run -p colver-core --bin nn_experiment --release --features nn -- models/value_net.bin 50 --data data/value_train.bin

# Run double-dummy solver benchmark
cargo run -p colver-core --bin dd_bench --release -- 1000  # num_deals, default 1000

# Build and install Python bindings (via uv, preferred)
uv sync

# Test Python bindings
uv run python3 -c "import colver; env = colver.Env(); env.reset()"

# Run web frontend (FastAPI + WebSocket)
uv run python -m colver.web
# Or: uv run colver-web
# Then open http://localhost:8000

# Docker build and run
docker build -t colver .
docker run -p 8000:8000 colver

# Docker Compose
docker compose up -d

# Cross-build for Raspberry Pi (ARM64)
docker buildx build --platform linux/arm64 -t colver .

# Publish to PyPI (automated via GitHub Actions)
# Tag a release to trigger the publish workflow:
git tag v0.2.1
git push origin v0.2.1
# Builds wheels for Linux (x86_64/aarch64), macOS (Intel/Apple Silicon), Windows
# Publishes via trusted publishing (no API tokens needed)
# Manual trigger also available from GitHub Actions tab
```

## PyPI Publishing

Published as [`colver`](https://pypi.org/project/colver/) on PyPI. Uses GitHub Actions (`.github/workflows/publish.yml`) with [Trusted Publishing](https://docs.pypi.org/trusted-publishers/) — no API tokens needed.

**Release flow:** push a `v*` tag → CI builds manylinux/macOS/Windows wheels via `maturin` → publishes automatically.

**Targets:** `x86_64-unknown-linux-gnu`, `aarch64-unknown-linux-gnu`, `x86_64-apple-darwin`, `aarch64-apple-darwin`, `x86_64-pc-windows-msvc`.

## Architecture

Colver is a Belote Contrée game engine optimized for millions of RL rollouts/sec. Rust core with PyO3 Python bindings.

**Workspace:** `colver-core` (pure Rust, zero deps by default) + `colver-py` (PyO3/numpy FFI) + `colver-web` (FastAPI/WebSocket frontend)

**Features:** `rand` (default), `parallel` (rayon parallel determinization), `nn` (neural network value function — features, value_net, NN-guided MCTS), `dmc_train` (candle GPU training for DMC + bid NN)

### Card Representation (`card.rs`)

Cards use a bitmask system: `Card = u8` (index 0-31), `CardSet = u32` (bitmask). Bit layout: Spades[0-7], Hearts[8-15], Diamonds[16-23], Clubs[24-31]. Within each suit, rank bits go 7=0, 8=1, 9=2, J=3, Q=4, K=5, 10=6, A=7 (plain strength order). Trump strength differs: J(7) > 9(6) > A(5) > 10(4) > K(3) > Q(2) > 8(1) > 7(0).

### GameState (`state.rs`)

`GameState` is `Copy` and ≤64 bytes (compile-time enforced) for fast MCTS cloning. Contains hands[4], current_trick[4], contract, points/tricks per team, bidding state, played_cards bitmask, void tracking, and belote tracking. Players: 0=N, 1=E, 2=S, 3=W. Teams: 0=NS (players 0,2), 1=EW (players 1,3). Partner = `player ^ 2`.

### Action Encoding

**Bidding (43 actions, u64 mask):** 0=PASS, 1-36=bids (value_idx×4 + suit_idx + 1, values 80-160, suits 0-3 = S/H/D/C), 37-40=capot×4 suits, 41=COINCHE, 42=SURCOINCHE.

**Playing (32 actions, u32→u64 mask):** Action = card index 0-31 directly.

`GameState::legal_actions() -> u64` returns the appropriate mask for the current phase. `GameState::step(action: u8)` dispatches to bidding or play.

### Game Flow

Bidding → Playing → Done. Bidding ends on 3 passes after a bid, surcoinche, or 4 passes (void deal). Playing runs 8 tricks of 4 cards. Dix de der: +10 (normal) or +100 (capot). Total card points always = 152; with dix de der = 162 (normal) or 252 (capot).

### Key Rules (from FFB official rules PDF)

- Coinche **freezes** the contract (no more overbids, only surcoinche or pass)
- "Ne pisse pas": if can't overtrump opponent's cut, may discard instead of undertrumping
- Only 4 color suits (no Sans Atout / Tout Atout)
- Scoring mode: "points faits + demandés" (section 10.2 of PDF)

### MCTS Agent (`mcts.rs`, feature `rand`)

Perfect-information MCTS using UCT (UCB1 for trees). Arena-based tree with `Node`s and `Edge`s in flat `Vec`s for cache-friendliness. `MctsSearch` is reusable across searches (arenas are cleared between calls). Default: 1000 iterations, `C = sqrt(2)`.

**API:** `MctsSearch::search(&mut self, state, config, rng) -> u8` returns best action. `search_with_stats(...)` returns `SearchResult` with visit counts. `search_with_nn(state, config, value_net, rng)` uses NN leaf evaluation instead of rollouts (feature `nn`). `mcts_search(state, config, rng)` is a convenience one-shot wrapper.

**Algorithm:** Selection (UCB1 descent) → Expansion (enumerate legal actions as edges, create child node) → Simulation (`rollout_random` or NN eval) → Backpropagation. Rewards scaled by `1/2000` for rollouts (NN outputs already in [0,1], no scaling). Best action = most-visited root child.

### Smart IS-MCTS Agent (`smart_ismcts.rs` + `card_beliefs.rs`, feature `rand`)

Belief-weighted Information Set MCTS. Maintains a `CardBeliefs` model (`[[f32; 32]; 4]` weight matrix) that is updated after every action using hard constraints (voids, trump ceiling, played cards) and soft inference (bidding signals, play patterns). At search time, `determinize_weighted()` samples opponent hands biased by these beliefs, then standard MCTS runs on each determinized world. See [SMART_ISMCTS.md](SMART_ISMCTS.md) for detailed design.

**API:** `SmartIsMctsSearch::new()`, `init_deal(state, observer, use_soft)`, `record_action(state_before, player, action)`, `search(state, config, rng) -> u8`. Each player needs its own instance; both must observe all actions.

### Naive IS-MCTS Agent (`naive_ismcts.rs`, feature `rand`)

Ensemble determinization without beliefs. Samples D determinized worlds (uniform, respecting void constraints only), runs standard MCTS on each, aggregates root visit counts. Simpler but less informed than Smart IS-MCTS.

### Belief Network (`belief_net.rs`, `belief_obs.rs`, `belief_candle.rs`)

NN-based card location prediction: given a player's observable game state, predicts which player holds each unknown card. Replaces or augments heuristic `CardBeliefs` in IS-DD search.

**Architecture:** 330→512 (LN+ReLU) → 512 (LN+ReLU) → 128 linear output. Output is 32 cards × 4 player-relative slots (raw logits). `belief_to_weights()` applies per-card softmax, zeros observer slot, renormalizes, and remaps to absolute player indices. ~2MB weights, ~0.1ms/eval (CPU).

**Rust inference** (`belief_net.rs`) — Pure Rust forward pass, zero dependencies. `BeliefNet::load(path)` reads raw f32 binary (auto-infers obs_dim from file size). `BeliefNet::evaluate(&mut self, obs) -> [f32; 128]` returns logits. Uses scratch buffers (not thread-safe — one per thread).

**Weight file format** — Contiguous little-endian f32: for each of 2 hidden layers: W (in×H), b (H), gamma (H), beta (H); then output W (H×128), b (128). Row-major. Compatible between `BeliefTrainer::export_binary()` and `BeliefNet::load()`.

**Observation** (`belief_obs.rs`, 330 floats): own hand (32) + per-player played cards (128, player-relative [me,left,partner,right], includes current trick) + card trick index (32, trick_number/8.0) + card position-in-trick (32, position/4.0) + bid history (72, 12 slots × 6 floats, player-relative) + contract (8, trump one-hot + bid_value/250 + taker team one-hot + coinche/2) + known voids (12, 3 hidden players × 4 suits) + scoring context (4) + dealer-relative position (4, one-hot) + current trick lead suit (4, one-hot) + trick progress (2, trick_number/8 + cards_in_trick/4). `write_belief_observation()` is zero-allocation (writes into buffer at offset). Reuses `EnvTracking` from `dmc_obs.rs`.

**Training data — two paths:**
1. **Pre-extracted binary** (`COLVBL01` format): `generate_belief_data` binary plays games with DMC + NN bid, records (obs, target, mask) per play step. Header: magic (8B) + obs_dim (4B) + num_samples (8B). Per sample: obs (330×f32) + target (32×u8) + mask (u32). ~20GB for 500K games. Supports `--features parallel` for multi-threaded generation.
2. **Game replays** (`COLVGM01` format, preferred): `generate_game_data` binary stores compact replays (~62 bytes/game: dealer + hands + actions). `GameReplay::extract_belief_samples()` re-extracts belief training data on demand. ~28MB for 500K games vs ~20GB extracted. `extract_belief_samples_parallel()` behind `parallel` feature.

**Training** (`train_belief_net` binary, feature `dmc_train`): Candle-based supervised learning with masked cross-entropy loss (only unknown cards contribute). AdamW optimizer. Supports cosine LR schedule with linear warmup (`--cosine-lr --warmup-epochs 5`). Saves safetensors checkpoints + best binary export. Accepts either `--data` (COLVBL01) or `--replays` (COLVGM01, extracts samples on startup).

```bash
cargo run -p colver-core --bin train_belief_net --features dmc_train --release -- \
  --replays data/games_500k.bin --epochs 100 --batch-size 512 --lr 3e-4 \
  --cosine-lr --warmup-epochs 5 --val-split 0.05 --output models/belief_net.bin
```

**Evaluation** (`belief_eval` binary, 4 modes):
- `--mode offline`: accuracy/CE/calibration on held-out COLVBL01 data, per-trick accuracy breakdown, 10-bin calibration table.
- `--mode match`: IS-DD match play — NN beliefs vs heuristic CardBeliefs (duplicate pairs to 2000 pts, configurable `--time-ms`).
- `--mode diagnose`: per-card predictions on sample game positions from replay files, grouped by suit, showing predicted probabilities vs ground truth.
- `--mode scenario`: hand-crafted scenario tests (trump ceiling, void detection, bidding signals).

**Integration with IS-DD** (`is_dd.rs`): `IsDdConfig::use_nn_beliefs` flag. `IsDdSearch::load_belief_net(path)` loads the model. `compute_weights()` implements hybrid mode: NN soft predictions are filtered by heuristic `CardBeliefs` hard constraints (where heuristic weight = 0.0, NN weight is zeroed and renormalized). `use_hard_constraints` flag (default true) controls whether hard filtering is applied. Belief tracking via `EnvTracking` is auto-initialized when a `BeliefNet` is loaded.

**Key scenario test findings:** Model learned void detection (0% probability for void suits), trump ceiling (0% for impossible trumps above played rank), and bidding signals (J of trump ~100% for bidder, 9 of trump ~56% for bidder vs ~33% baseline).

**Python API:**
- `Env.load_belief_net(path)` / `Env.has_belief_net()` — load/check NN belief weights. When loaded, `action_dede()` and `action_dede_with_stats()` automatically use NN beliefs (sets `use_nn_beliefs: true`).
- `colver.belief_model_path()` / `colver.download_belief_model()` — model discovery/download (same pattern as DMC/bid models).
- Web frontend auto-downloads and loads belief net at startup; all IS-DD sessions (Play + Watch) use NN beliefs when available.

**Model distribution:** `models/belief_net.bin` (~2MB), auto-downloaded from GitHub releases (v0.3.1) at web server startup.

### Bidding Strategies (`bid_eval.rs`)

Six fixed bidding functions (`BidFunction` enum: `Heuristic`, `Smart`, `Improved`, `Roro`, `Maxi`, `BidADd`) plus a configurable `parametric_bid(state, &BidParams)`. All except `BidADd` are deterministic, ~200 ops, suitable for millions of rollouts/sec. `BidADd` uses DD determinization (~300ms/opening).

**Hand evaluation:** `evaluate_for_trump(hand, suit) -> u16` scores a hand assuming `suit` is trump. Trump honors (J=8, 9=6, A=4, 10=3, K=1, Q=1), trump length bonus ((count−2)×2 if count>2), side aces (+3 each), voids (+3), singletons (+1). Typical range 0–35.

**`improved_bid`** (default) — Tournament-winning balanced strategy. Quality gate (J/9/A/10 or 3+ cards in suit), then score→value mapping: 10→80, 13→90, 17→100, 20→110, 25→120. Opening cap 120, overcall cap 120, response cap 130. Coinches on J+9 in opponent's suit or 4+ trumps + side ace.
- Opening: best suit must pass quality gate + score threshold
- Partner response: raise in partner's suit based on own score, or bid alternative suit if score≥16
- Overcall: score≥13, quality gate, cap 120, won't compete above opponent's 120

**`heuristic_bid`** — Aggressive score-based. Maps score→value (10→80, 14→90, 17→100, 20→110, 23→120, 26→130). No quality gate, no cap. Boosts partner's suit +3. Never coinches. Takes ~50% of contracts with ~70% success rate.

**`smart_bid`** — Conservative convention-based. Requires J/9 for opening, J+9 signaling between partners. Very conservative (~10-13% contract take rate, ~78% success). Mostly historical.

**`roro_bid`** — Expert convention-based strategy. Position-aware openings, highest-level-first scan (130→80), structured partner responses, intervention (+10 light / +20 "la barre"), Théorème 3 coinche.

**`maxi_bid`** (`maxi.rs`) — Expert convention-linked bidding + structured card play (the "Maxi" agent). Hand classification into Cases A(80), B(90), C(100), D(110–130) based on J/9 honors, side strength, and loser count. 4 bidding phases: opening classification, partner response, suit change, competitive. Théorème 3 coinche (0 trumps in opponent's suit + 3 aces). Card play uses convention-aware leads, trump management, finesse patterns.

**`parametric_bid` + `BidParams`** — Configurable bidder for strategy sweeps. `BidParams` has: score thresholds[6] (for 80–130), opening/overcall/response caps, overcall_min_score, quality_gate flag. Presets: `ultra_conservative`, `conservative`, `moderate`, `balanced`, `aggressive`, `very_aggressive`. Used by `bid_tournament` binary.

### DD-Based Bidding (`dd_bid.rs`, feature `rand`)

Uses the double-dummy solver with determinization to estimate expected team points for each candidate trump suit. Replaces heuristic score→value mapping with principled point estimation.

**`DdBidder`** — Holds a pre-allocated TT buffer (2MB). `DdBidConfig` controls: determinizations per position (opening=8, response=4, overcall=4), confidence margin (15 pts), prefilter threshold (heuristic score≥6), quality gate, caps (120/120/130). ~300ms/opening on x86.

**Algorithm:** Pre-filter candidate suits (heuristic score + quality gate) → determinize opponent hands → DD-solve each suit per determinization → average NS points → map expected points to bid value (bid X if expected ≥ X + margin). Coinche uses heuristic logic (fast, well-validated).

### NN Bidding Agent — "Le Bide à Dédé" (`bid_net.rs`, `bid_obs.rs`, `bid_candle.rs`, `bid_train_env.rs`)

DD-oracle-trained Dueling DQN for bidding. Trained on 1M pre-solved deals with DD rewards. **Default bidder for all web bots** (replaces `improved_v2`). Beats `improved_v2` 70–76% in match play across all play engines (Smart IS-DD, DD oracle, DMC).

**Tournament results** (50 matches/pair, 20ms/move, 6 agents round-robin):
- NN vs V2 head-to-head: 70% (Smart IS-DD), 76% (DD oracle), 71% (DMC) — consistent +460-590 margin
- Rankings: NN+DD 89.8% > V2+DD 70.8% > NN+SDD 51.0% > NN+DMC 39.6% > V2+SDD 27.6% > V2+DMC 21.2%
- Play method hierarchy: DD >> Smart IS-DD >> DMC. NN bidding helps at every play level
- NN bidding partially compensates for weaker play engines (NN+SDD competitive with V2+DD)

**Architecture:** 114→256→256→43 Dueling DQN with LayerNorm. Q(s,a) = V(s) + A(s,a) - mean(A). ~421KB weights, ~0.1ms/eval (CPU).

**Observation** (`bid_obs.rs`, 114 floats): hand (32) + bid history (72, 12 slots × 6 floats, player-relative) + dealer-relative position (4) + auction state (6: bid_value/160, suit one-hot, coinche/2). `write_bid_observation()` is zero-allocation (writes into buffer at offset).

**Rust inference** (`bid_net.rs`) — Pure Rust forward pass, zero dependencies. Auto-detects standard vs dueling architecture from weight file size. `BidNet::load(path)` / `BidNet::evaluate(&mut self, obs) -> [f32; 43]` / `BidNet::best_action(&mut self, obs, legal_mask) -> (u8, Vec<(u8, f32)>)`.

**Training infrastructure:**
- `DealPool` (`bid_train_env.rs`): Pre-solves N deals × 4 suits using all CPU cores in parallel (~5 min for 1M deals on 16 cores). Binary format (`COLVDD01`), ~21B/deal. `load_or_generate()` caches to disk.
- `BidTrainingEnv`: Per-deal env that runs bidding in microseconds using pre-solved DD points. Buffers transitions, flushes with reward = (my_team_score - opp_score) / 500.0 on episode end.
- `VecBidEnv`: Vectorized multi-env with flat obs/mask buffers for GPU batching.
- `BidReplayBuffer`: SumTree-based PER buffer (same design as DMC replay).
- `BiddingQNet` / `BiddingTrainer` (`bid_candle.rs`): Candle-based Dueling DQN. `export_binary()` writes raw f32 weights compatible with `BidNet` CPU inference.

**Opponent diversity:** Configurable mix annealing from 40% → 15% non-self-play. Within non-self-play: improved_v2 (20%), aggressive (20%), conservative (20%), random (40%). Ensures the NN learns to exploit diverse bidding styles.

**Training command:**
```bash
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- \
  --num-envs 64 --steps 5000000 --pool-size 1000000
```

**Inline evaluation:** vs improved_v2 bidding with DD oracle scoring (both sides, alternating). Reports win rate + average margin.

**Python API:**
- `Env.bid_a_dd() -> int` — NN bid if model loaded, else `improved_v2` fallback. **Default for all web bots.**
- `Env.load_bid_model(path)` / `Env.has_bid_model()` — load/check NN weights.
- `Env.action_bid_nn() -> dict` — NN bid with Q-value statistics.
- `colver.bid_model_path()` / `colver.download_bid_model()` — model discovery/download (same pattern as DMC).

**Model distribution:** `models/bid_nn_final.bin` (421KB), auto-downloaded from GitHub releases at web server startup.

### Card Play Strategies

**Random play** (`rollout.rs: rollout_random`) — Uniform random legal moves. ~1.3M rollouts/sec.

**Heuristic play** (`rollout.rs: heuristic_play_action`) — Deterministic card play for rollouts (sees all hands). Decision tree: safe leads, partner feeding, minimum-winning-card, cheapest trump cut. ~769K full-deal rollouts/sec with heuristic bid.

**Naive IS-MCTS** (`naive_ismcts.rs`) — Ensemble determinization without beliefs. Samples D determinized worlds (uniform, void-aware), runs MCTS on each, aggregates root visit counts. Default: 20 determinizations × 50 iters, `HeuristicPlay` rollouts.

**Smart IS-MCTS** (`smart_ismcts.rs` + `card_beliefs.rs`) — Belief-weighted IS-MCTS. `CardBeliefs` weight matrix updated via hard constraints (voids, trump ceiling) and soft inference (bidding signals, play patterns). `determinize_weighted()` samples opponent hands biased by beliefs. ~+7.5% win rate vs Naive IS-MCTS in match play. Has `search_parallel()` behind `parallel` feature and `search_with_nn()` behind `nn` feature.

### Double-Dummy Solver (`solver.rs`)

Alpha-beta solver for perfect-information Belote: given 4 known hands and a trump suit, computes the exact maximum trick points each team can score with optimal play. No feature gate — zero external dependencies, always compiled.

**API:**
- `solver::solve(state) -> [u8; 2]` — solve a playing-phase state, returns `[ns_points, ew_points]`
- `solver::solve_for_trump(hands, dealer, trump) -> [u8; 2]` — convenience: creates DD state and solves
- `solver::solve_best_card(state) -> u8` — returns the optimal card for the current player
- `GameState::setup_dd(dealer, hands, trump)` — creates a playing-phase state for DD solving (bypasses bidding)

**Performance:** ~13.5ms/solve average, median ~7ms, P90 ~31ms (500 random deals × 4 suits). All results satisfy `ns + ew == 162` (or 252 for capot).

**Techniques (inspired by bridge DD solvers DDS/GIB):**
- Alpha-beta with fail-soft
- Transposition table: 256K entries (2MB, L2-cache friendly), packed u64 entries with relative future scores and hash move, always-replace
- Principal Variation Search (null-window for non-PV moves)
- Killer move heuristic (2 per ply, 32 plies)
- History heuristic (depth² bonus on cutoff, indexed by `[team][card]`)
- Card equivalence pruning — adjacent same-point cards with no outstanding card between them
- Quick tricks bounds — guaranteed future points from consecutive top trumps + unruffable plain masters
- Forced-move optimization for single legal cards
- Move ordering: hash move → killers → history + static score

**Benchmark:** `cargo run -p colver-core --bin dd_bench --release -- [num_deals]`

### DMC Q-Network Agent (`scripts/dmc_model.py`, `scripts/train_dmc.py`, `scripts/eval_dmc.py`)

DouZero-style Deep Monte-Carlo agent. A Q-network picks card plays with a single forward pass — no search tree. Bidding uses `improved_bid` (Python) or configurable strategies including NN bid (Rust). Trained with binary deal outcomes (win=1.0, loss=0.0, void=0.5), ε-greedy exploration.

**v3 architecture:** 444→1024→1024→1024→32 MLP with LayerNorm (~2.6M params). v3 observation extends v2 with 72-float bid history encoding (12 chronological slots in player-relative order, 6 floats each: action_type, bid_value, suit one-hot). Player-relative with per-player card tracking, trump ceiling inference, tactical features (master cards, partner winning, void info), and richer scoring context.

**Rust inference** (`dmc_net.rs`) — Pure Rust forward pass, zero dependencies. `DmcNet::load(path)` reads raw f32 binary weights (auto-detects obs_dim from file size). `DmcNet::evaluate(&mut self, obs) -> [f32; 32]` returns Q-values. `DmcNet::best_action(&mut self, obs, legal_mask) -> (u8, Vec<(u8, f32)>)` picks best legal action. `DmcNet::obs_dim()` returns expected input size. Uses scratch buffers (~1ms/eval on x86, no torch needed). Backward compatible: old 372-dim and new 444-dim models auto-detected.

**Weight export:** `python scripts/export_dmc_weights.py models/dmc_final.pt models/dmc_final.bin` converts PyTorch checkpoint to raw f32 binary. Weight layout: for each of 3 hidden layers: W (in×H), b (H), gamma (H), beta (H); then output W (H×32), b (32). ~10MB for H=1024.

**Training features:** Prioritized Experience Replay (SumTree-based PER), opponent pool (70% self-play, 20% past checkpoints, 10% random), 2M replay buffer, 20M steps default.

**Inline evaluation** (every `--eval-freq` steps, default 1M):
- Python trainer: deal win rate vs random (200 deals), match play vs random (100), vs naive IS-MCTS (10, 20ms), vs smart IS-MCTS (10, 20ms). Output: `[EVAL] deals 67% | rand 72% | naive 40% | smart 30% (45s)`.
- Rust trainer: match play vs random (100), vs frozen checkpoint (50), vs IS-DD (10, 20ms/move). Output: `[EVAL] rand 85% | ckpt 55% | isdd 35% | nn_bid 80% (210s)`.

**Python training:** `PYTHONPATH=scripts uv run python scripts/train_dmc.py --num-envs 256 --steps 20000000`
**Eval:** `PYTHONPATH=scripts uv run python scripts/eval_dmc.py models/dmc_final.pt --games 200 --baseline smart --time-ms 20 --both-sides`

**Rust training** (candle, feature `dmc_train`):
```bash
cargo run -p colver-core --bin train_dmc --features dmc_train --release -- \
  --num-envs 256 --steps 35000000 \
  --bid-model models/bid_nn_final.bin \
  --nn-bid-start 0.75 --nn-bid-end 0.95 --nn-bid-anneal-steps 20000000 \
  --eval-freq 1000000 \
  --eval-random-matches 100 \
  --eval-isdd-matches 10 --eval-isdd-time-ms 20 \
  --eval-checkpoint models/dmc_35.bin --eval-checkpoint-matches 50
```
- ~474 steps/s with 64 envs on 4090 (3x+ Python speedup)
- Dueling DQN: Q(s,a) = V(s) + A(s,a) - mean(A)
- **NN bid support**: `--bid-model` loads BidNet for training bidding (strategy 8). Bid fraction anneals linearly from `--nn-bid-start` (75%) to `--nn-bid-end` (95%) over `--nn-bid-anneal-steps`. Non-NN fraction split: 40% improved_v2 / 30% heuristic / 30% BidParams presets. Per-team assignment: each team independently draws a strategy.
- **IS-DD eval**: `--eval-isdd-matches` matches with `IsDdSearch` (belief tracking, `--eval-isdd-time-ms` per move). Replaces naive/smart IS-MCTS evals from earlier versions.
- **Checkpoint eval**: `--eval-checkpoint` loads a frozen DmcNet baseline for regression tracking.
- `dmc_candle.rs`: ManualLayerNorm, DuelingQNet, DuelingTrainer
- `dmc_obs.rs`: zero-alloc obs builder (writes into buffer at offset), EnvTracking struct
- `dmc_replay.rs`: SumTree + PER buffer (pure Rust)
- `dmc_env.rs`: VecTrainingEnv with pre-allocated obs_buf/mask_buf, per-team bid strategies `Vec<(u8, u8)>`, optional `BidNet` for NN bidding (strategy 8, falls back to improved_v2 if no model loaded)

**Python API (Env):** `action_naive_ismcts(time_ms)`, `action_smart_ismcts(time_ms)`, `smart_ismcts_init()`, `smart_ismcts_step(action)`, `bid_improved()`, `bid_roro()`, `deal_outcome()`, `rewards()`, `load_dmc_model(path)`, `action_dmc_with_stats()`, `get_bid_history()`, `get_observation()`.

### Neural Network Value Function (feature `nn`)

A learned MLP replaces rollouts for MCTS leaf evaluation. Train in Python (PyTorch), inference in pure Rust (hand-rolled matmul, zero deps).

**Feature extraction** (`features.rs`) — 278 floats from perfect-info GameState: 4 hands (128), current trick (128), trump suit (4), bid value (1), coinche (3), taker team (2), points (2), tricks (2), current player (4), trick lead (4). `extract_features(state, &mut buf)` — no allocations.

**Value network** (`value_net.rs`) — MLP: 278→256→256→1 (ReLU+Sigmoid). ~137K params. `ValueNet::load(path)` reads raw f32 binary. `ValueNet::evaluate(&mut self, features) -> f32` returns P(team 0 wins). Uses scratch buffers (not thread-safe — one instance per thread).

**Weight file format** — Contiguous little-endian f32: W1 (278×H), b1 (H), W2 (H×H), b2 (H), W3 (H×1), b3 (1). Row-major (matches PyTorch Linear weight layout).

**Training pipeline:**
1. `generate_value_data` — self-play data generation (IS-MCTS or `--fast` heuristic)
2. `scripts/train_value_net.py` — PyTorch training with BCELoss, exports raw f32 binary
3. `nn_experiment` — accuracy, speed, and strength evaluation

### Experiment Binaries

- **`oracle_experiment`**: Tests bid achievability — pairs each bidding strategy with perfect-info MCTS play.
- **`bidding_experiment`**: Head-to-head comparison of bidding strategies using IS-MCTS for play.
- **`match_experiment`**: Full match play (first to 2000 points), 5 experiments comparing IS-MCTS variants and bidding strategies. Reports contracts taken/made, deal score distributions, coinches.
- **`bid_tournament`**: Round-robin tournament of parameterized bidding strategies. Each pair plays both directions with Naive IS-MCTS for card play. Reports win matrix, margin matrix, rankings.
- **`bid_debug`**: Prints detailed bidding rounds showing each player's hand, suit evaluations, and decisions for both heuristic and improved bidders side-by-side.
- **`strength_experiment`**: Rollout policy comparison, D×I sweep, RAVE on/off.
- **`maxi_diagnose`**: Diagnostic tool — plays individual deals showing Maxi (NS) vs DMC (EW) with full play-by-play: hand evals, bidding reasoning (Cases A/B/C/D), DMC Q-value rankings, trick-by-trick results. Usage: `cargo run --bin maxi_diagnose --release -- [num_deals] [seed]`.
- **`v2_tournament`**: V2 bidding fine-tune tournament — compares `improved_bid` baseline vs `V2Config` variants using DMC + Oracle MCTS play in parallel. Round-robin match play with win/margin matrices.
- **`dd_bench`**: Double-dummy solver benchmark. Solves N random deals × 4 trump suits, reports timing distribution (percentiles, top-10% share), point totals, capot rates. Usage: `cargo run --bin dd_bench --release -- [num_deals]`.
- **`dd_calibrate`**: DD bidding calibration — generates pre-solved deal pools and analyzes DD-based bid value thresholds.
- **`train_bid_nn`** (feature `dmc_train`): NN bidding training with DD oracle rewards + Dueling DQN. Phase 1: pre-solve deal pool (1M deals). Phase 2: train with PER + opponent diversity. Usage: `cargo run --bin train_bid_nn --features dmc_train --release -- --num-envs 64 --steps 5000000`.
- **`train_dmc`** (feature `dmc_train`): DMC card play training with Dueling DQN. Supports NN bidding (`--bid-model`) with annealing, per-team bid strategies, IS-DD eval, checkpoint regression tracking. Usage: `cargo run --bin train_dmc --features dmc_train --release -- --num-envs 256 --steps 35000000 --bid-model models/bid_nn_final.bin`.
- **`bid_compare`**: Head-to-head comparison of bidding strategies using DD oracle for evaluation.
- **`bid_nn_eval`**: Evaluate a trained bid NN model against heuristic bidders with DD scoring.
- **`bid_nn_tournament`**: Round-robin tournament — NN bid vs improved_v2 across play methods (Smart IS-DD, DD oracle, DMC). Multi-threaded match play. Usage: `cargo run --bin bid_nn_tournament --release -- models/bid_nn_final.bin --dmc-model models/dmc_35.bin --matches 100`.
- **`generate_belief_data`** (feature `rand`): Plays games with DMC + NN bid, records belief training samples (obs + target + mask) in COLVBL01 format. Supports `--features parallel` for multi-threaded generation. Usage: `cargo run --bin generate_belief_data --release --features parallel -- --dmc-model models/dmc_final.bin --bid-model models/bid_nn_final.bin --games 500000 --output data/belief_train.bin`.
- **`generate_game_data`** (feature `rand`): Plays games with DMC + NN bid, stores compact replays (COLVGM01 format, ~62 bytes/game). Supports `--features parallel`. Usage: `cargo run --bin generate_game_data --release --features parallel -- --dmc-model models/dmc_final.bin --games 500000 --output data/games.bin`.
- **`train_belief_net`** (feature `dmc_train`): Supervised belief network training. Candle-based masked cross-entropy with cosine LR + warmup. Accepts `--data` (COLVBL01) or `--replays` (COLVGM01). Usage: `cargo run --bin train_belief_net --features dmc_train --release -- --replays data/games_500k.bin --epochs 100 --batch-size 512 --lr 3e-4 --cosine-lr --warmup-epochs 5`.
- **`belief_eval`** (feature `rand`): Belief network evaluation — 4 modes: `offline` (accuracy/CE/calibration), `match` (IS-DD NN vs heuristic), `diagnose` (per-card predictions), `scenario` (hand-crafted tests). Usage: `cargo run --bin belief_eval --release -- --model models/belief_net.bin --mode scenario`.
- **`generate_value_data`** (feature `nn`): Self-play data generation for NN training. Binary output format.
- **`nn_experiment`** (feature `nn`): NN value function evaluation — accuracy, speed, and strength tests.
- **`isdd_sweep`** (feature `rand`): IS-DD parameter sweep experiment. Three sections: count-based (D=1–64), time-based (5–50ms), soft inference comparison (D=8/16 hard vs soft). Each config plays N deals twice (IS-DD as NS vs random, and vs DouDou35 DMC), reporting win%, avg NS/EW points, ms/deal, and avg dets. Usage: `cargo run --bin isdd_sweep --release -- --dmc-model models/dmc_35.bin --deals 200 --threads 8`.

**IS-DD sweep results** (200 deals, 8 threads, vs DouDou35):
- Count sweep: D=1→44.5%, D=8→49%, D=16→52%, D=32→51.5%, D=64→50.5%. Gains plateau sharply after D=8 (~339ms/deal).
- Time sweep: 5ms→50.5% (24 dets), 10ms→46%, 20ms→48%, 50ms→**57%** (264 dets). 50ms is the clear sweet spot.
- Soft inference: at D=8, hard vs soft is a wash (46% vs 49%). At D=16, soft adds +3.5% (50.5%→54%) for only 7% more compute (704→752ms). Soft inference is worth it at D≥16.
- **Recommended web config**: 20ms time-limited with soft inference (~48% vs DouDou35, ~230ms/deal). For higher quality: 50ms (~57%, 515ms/deal).

### Performance-Critical Path

`play.rs::legal_plays()` is the hottest function — all bitwise, no allocations. `rollout.rs` runs games to completion with random moves; state is copied (~56 bytes memcpy) per rollout. Target: >1M rollouts/sec single-threaded.

### Python Layer (`colver-py/` → `python/colver/`)

`Env` wraps a single GameState with IS-MCTS search support. Uses `StdRng` (not `ThreadRng`) for PyO3 `Send` requirement. The native extension is built as `colver._colver` (private module convention) and re-exported from `colver.__init__`.

**Observation v4** (415 floats, player-relative): hand (32) + current trick per-player (128) + past tricks per-player (96) + contract (7) + void tracking (12) + scoring context (4) + bid history (72) + card trick index (32) + card sequence index (32). `get_observation()` returns the full vector. Legal action mask is 43 floats.

**Public API**: `Env`, `__version__`, `download_model()`, `model_path()`, `bid_model_path()`, `download_bid_model()`, `belief_model_path()`, `download_belief_model()`. See `python/colver/_colver.pyi` for full type stubs.

**Web frontend API** (on `Env`): `get_hands()`, `get_current_trick()`, `get_contract()`, `get_points()`, `get_tricks_won()`, `get_dealer()`, `get_trick_lead()`, `get_played_cards()`, `phase()`, `current_player()`, `is_terminal()`, `legal_actions()`. Static methods: `Env.card_name(idx)`, `Env.action_name(action, phase)`, `Env.deal_with_hands(dealer, hands)`. Setup: `set_contract(trump, value, team, coinche)`, `set_phase_playing()`.

### Web Frontend (`python/colver/web/` + `colver-web/`)

FastAPI + WebSocket backend with vanilla JS frontend. Bundled in the wheel under `colver[web]` optional dependency. Three modes: Play (human vs AI), Watch (spectate AI vs AI with thinking stats), Analysis (custom position setup + MCTS analysis).

**Play tab UX:** Instant card play (optimistic update), configurable pause slider (1–8s), trick flush animation (cards pile → flip face-down → fly toward winner's seat, 1.6s), end-of-game overlay (centered glassmorphism box with victory/defeat/draw theming, contract info, scores with belote annotation, confetti on victory, restart button). CFN box for copyable game state. Bug report button.

**Package layout** (`python/colver/web/`):
- `server.py` — FastAPI app, WebSocket handler, auto-downloads DMC/bid/belief models at startup.
- `game_manager.py` — `PlaySession` (human vs AI), `WatchSession` (spectate), `ReplaySession` (replay), `AnalysisSession` (custom position + MCTS analysis).
- `database.py` — SQLite game history, defaults to `~/.local/share/colver/colver.db`.
- `static/` — Frontend files (HTML, JS, CSS), copied from `colver-web/frontend/`.
- `cards/` — 67 SVG playing cards, copied from `images/cards/`.

**Development source** (`colver-web/`):
- `colver-web/frontend/` — Original frontend source (copied into `python/colver/web/static/` for builds).
- `colver-web/backend/` — Original backend source (adapted into `python/colver/web/`).

**Running:** `uv run python -m colver.web` or `uv run colver-web` → http://localhost:8000

### Docker Deployment

Multi-stage Dockerfile: `uv:python3.12-bookworm` builder (compiles PyO3 wheel with maturin) + `python:3.12-slim-bookworm` runtime. Web assets bundled in wheel (no separate COPY needed). No torch dependency — all inference is pure Rust (IS-MCTS + DMC Q-network + belief net). Three models auto-downloaded at startup: DMC (`dmc_27.bin`, 10MB), bid NN (`bid_nn_final.bin`, 421KB), belief net (`belief_net.bin`, 2MB). `docker-compose.yml` for single-service deployment. Cross-builds for ARM64 (Raspberry Pi) via `docker buildx`. CMD: `python -m colver.web`.

## Rules Reference

The official FFB rules are in `REGLES-DE-LA-BELOTE-CONTREE.pdf` at the repo root. Consult it for any rule ambiguities.
