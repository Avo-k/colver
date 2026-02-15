# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Check compilation (both crates)
cargo check

# Run all core tests (131 tests, 139 with --features nn)
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

# Build and install Python bindings
cd colver-py && maturin develop --release

# Or via uv (preferred)
uv sync

# Test Python bindings
uv run python3 -c "import colver; env = colver.Env(); env.reset()"

# Run web frontend (FastAPI + WebSocket)
uv run python colver-web/backend/server.py
# Then open http://localhost:8000
```

## Architecture

Colver is a Belote Contrée game engine optimized for millions of RL rollouts/sec. Rust core with PyO3 Python bindings.

**Workspace:** `colver-core` (pure Rust, zero deps by default) + `colver-py` (PyO3/numpy FFI) + `colver-web` (FastAPI/WebSocket frontend)

**Features:** `rand` (default), `parallel` (rayon parallel determinization), `nn` (neural network value function — features, value_net, NN-guided MCTS)

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

### Bidding Strategies (`bid_eval.rs`)

Three fixed bidding functions (`BidFunction` enum: `Heuristic`, `Smart`, `Improved`) plus a configurable `parametric_bid(state, &BidParams)`. All are deterministic, ~200 ops, suitable for millions of rollouts/sec.

**Hand evaluation:** `evaluate_for_trump(hand, suit) -> u16` scores a hand assuming `suit` is trump. Trump honors (J=8, 9=6, A=4, 10=3, K=1, Q=1), trump length bonus ((count−2)×2 if count>2), side aces (+3 each), voids (+3), singletons (+1). Typical range 0–35.

**`improved_bid`** (default) — Tournament-winning balanced strategy. Quality gate (J/9/A/10 or 3+ cards in suit), then score→value mapping: 10→80, 13→90, 17→100, 20→110, 25→120. Opening cap 120, overcall cap 120, response cap 130. Coinches on J+9 in opponent's suit or 4+ trumps + side ace.
- Opening: best suit must pass quality gate + score threshold
- Partner response: raise in partner's suit based on own score, or bid alternative suit if score≥16
- Overcall: score≥13, quality gate, cap 120, won't compete above opponent's 120

**`heuristic_bid`** — Aggressive score-based. Maps score→value (10→80, 14→90, 17→100, 20→110, 23→120, 26→130). No quality gate, no cap. Boosts partner's suit +3. Never coinches. Takes ~50% of contracts with ~70% success rate.

**`smart_bid`** — Conservative convention-based. Requires J/9 for opening, J+9 signaling between partners. Very conservative (~10-13% contract take rate, ~78% success). Mostly historical.

**`parametric_bid` + `BidParams`** — Configurable bidder for strategy sweeps. `BidParams` has: score thresholds[6] (for 80–130), opening/overcall/response caps, overcall_min_score, quality_gate flag. Presets: `ultra_conservative`, `conservative`, `moderate`, `balanced`, `aggressive`, `very_aggressive`. Used by `bid_tournament` binary.

### Card Play Strategies

**Random play** (`rollout.rs: rollout_random`) — Uniform random legal moves. ~1.3M rollouts/sec.

**Heuristic play** (`rollout.rs: heuristic_play_action`) — Deterministic card play for rollouts (sees all hands). Decision tree: safe leads, partner feeding, minimum-winning-card, cheapest trump cut. ~769K full-deal rollouts/sec with heuristic bid.

**Naive IS-MCTS** (`naive_ismcts.rs`) — Ensemble determinization without beliefs. Samples D determinized worlds (uniform, void-aware), runs MCTS on each, aggregates root visit counts. Default: 20 determinizations × 50 iters, `HeuristicPlay` rollouts.

**Smart IS-MCTS** (`smart_ismcts.rs` + `card_beliefs.rs`) — Belief-weighted IS-MCTS. `CardBeliefs` weight matrix updated via hard constraints (voids, trump ceiling) and soft inference (bidding signals, play patterns). `determinize_weighted()` samples opponent hands biased by beliefs. ~+7.5% win rate vs Naive IS-MCTS in match play. Has `search_parallel()` behind `parallel` feature and `search_with_nn()` behind `nn` feature.

### DMC Q-Network Agent (`scripts/dmc_model.py`, `scripts/train_dmc.py`, `scripts/eval_dmc.py`)

DouZero-style Deep Monte-Carlo agent. A Q-network picks card plays with a single forward pass — no search tree. Bidding uses `improved_bid`. Trained with binary deal outcomes (win=1.0, loss=0.0, void=0.5), ε-greedy exploration.

**v2 architecture:** 372→1024→1024→1024→32 MLP with LayerNorm (~3.2M params). v2 observation is player-relative with per-player card tracking, trump ceiling inference, tactical features (master cards, partner winning, void info), and richer scoring context.

**Training features:** Prioritized Experience Replay (SumTree-based PER), opponent pool (70% self-play, 20% past checkpoints, 10% random), 2M replay buffer, 20M steps default.

**Training:** `PYTHONPATH=scripts uv run python scripts/train_dmc.py --num-envs 256 --steps 20000000`
**Eval:** `PYTHONPATH=scripts uv run python scripts/eval_dmc.py models/dmc_final.pt --games 200 --baseline smart --time-ms 20 --both-sides`

**Python API (Env):** `action_naive_ismcts(time_ms)`, `action_smart_ismcts(time_ms)`, `smart_ismcts_init()`, `smart_ismcts_step(action)`, `bid_improved()`, `deal_outcome()`.
**Python API (VecEnv):** `current_players()`, `phases()`, `bid_improved()`. `step()` returns 5-tuple with `deal_outcomes (n,2)`.

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
- **`generate_value_data`** (feature `nn`): Self-play data generation for NN training. Binary output format.
- **`nn_experiment`** (feature `nn`): NN value function evaluation — accuracy, speed, and strength tests.

### Performance-Critical Path

`play.rs::legal_plays()` is the hottest function — all bitwise, no allocations. `rollout.rs` runs games to completion with random moves; state is copied (~56 bytes memcpy) per rollout. Target: >1M rollouts/sec single-threaded.

### Python Layer (`colver-py/`)

`Env` wraps a single GameState with IS-MCTS search support. `VecEnv(n)` wraps n parallel environments with NumPy array I/O. Uses `StdRng` (not `ThreadRng`) for PyO3 `Send` requirement.

**Observation v2** (372 floats, player-relative): hand (32) + trick (128) + per-player played cards (96) + all played (32) + card point values (32) + contract (7) + void tracking (12) + scoring context (12) + tactical features (21). Legal action mask is 43 floats. `VecEnv.step()` returns 5-tuple including `deal_outcomes (n,2)`.

**Web frontend API** (on `Env`): `get_hands()`, `get_current_trick()`, `get_contract()`, `get_points()`, `get_tricks_won()`, `get_dealer()`, `get_trick_lead()`, `get_played_cards()`, `phase()`, `current_player()`, `is_terminal()`, `legal_actions()`. Static methods: `Env.card_name(idx)`, `Env.action_name(action, phase)`, `Env.deal_with_hands(dealer, hands)`. Setup: `set_contract(trump, value, team, coinche)`, `set_phase_playing()`.

### Web Frontend (`colver-web/`)

FastAPI + WebSocket backend with vanilla JS frontend. Three modes: Play (human vs AI), Replay (step through recorded games), Analysis (custom position setup + MCTS analysis).

**Backend** (`colver-web/backend/`):
- `server.py` — FastAPI app, WebSocket handler, serves static files + SVG cards from `images/cards/`. Loads NN model once at startup for DouDou agent.
- `game_manager.py` — `PlaySession` (human vs AI with trick tracking and point calculation), `ReplaySession` (precomputed state replay), `AnalysisSession` (custom position + MCTS analysis).

**Frontend** (`colver-web/frontend/`):
- `index.html` — Three-tab layout (Jouer, Rejouer, Analyse) with compass-style table layout.
- `js/main.js` — WebSocket connection, card rendering (SVG images), shared utilities (`cardToHtml`, `renderHand`, `renderTrick`, `renderLastTrick`, `actionName`).
- `js/play.js` — Play mode: bidding panel with dropdown selectors, legal/illegal card states (raised/greyed), last trick display with compass grid and team-colored point labels.
- `js/replay.js` — Replay mode: step navigation, auto-play, log display.
- `js/analysis.js` — Analysis mode: card palette for hand editing, MCTS analysis results.
- `styles.css` — Dark green table theme, responsive card sizing via CSS variables, compass grid layouts.

**Card assets** (`images/cards/`): 67 SVG playing cards from [SVG-cards](https://svg-cards.sourceforge.net/). Face cards use version 2 art. Served at `/cards/<rank>_of_<suit>.svg`.

**Running:** `uv run python colver-web/backend/server.py` → http://localhost:8000

## Rules Reference

The official FFB rules are in `REGLES-DE-LA-BELOTE-CONTREE.pdf` at the repo root. Consult it for any rule ambiguities.
