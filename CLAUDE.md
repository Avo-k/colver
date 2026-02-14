# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Check compilation (both crates)
cargo check

# Run all core tests (104 tests)
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

# Build and install Python bindings
cd colver-py && maturin develop --release

# Or via uv (preferred)
uv sync

# Test Python bindings
uv run python3 -c "import colver; env = colver.Env(); env.reset()"
```

## Architecture

Colver is a Belote Contrée game engine optimized for millions of RL rollouts/sec. Rust core with PyO3 Python bindings.

**Workspace:** `colver-core` (pure Rust, zero deps by default) + `colver-py` (PyO3/numpy FFI)

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

**API:** `MctsSearch::search(&mut self, state, config, rng) -> u8` returns best action. `search_with_stats(...)` returns `SearchResult` with visit counts. `mcts_search(state, config, rng)` is a convenience one-shot wrapper.

**Algorithm:** Selection (UCB1 descent) → Expansion (enumerate legal actions as edges, create child node) → Simulation (`rollout_random`) → Backpropagation. Rewards scaled by `1/2000` to keep exploitation term in [0,1]. Best action = most-visited root child.

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

**Smart IS-MCTS** (`smart_ismcts.rs` + `card_beliefs.rs`) — Belief-weighted IS-MCTS. `CardBeliefs` weight matrix updated via hard constraints (voids, trump ceiling) and soft inference (bidding signals, play patterns). `determinize_weighted()` samples opponent hands biased by beliefs. ~+7.5% win rate vs Naive IS-MCTS in match play. Has `search_parallel()` behind `parallel` feature.

### Experiment Binaries

- **`oracle_experiment`**: Tests bid achievability — pairs each bidding strategy with perfect-info MCTS play.
- **`bidding_experiment`**: Head-to-head comparison of bidding strategies using IS-MCTS for play.
- **`match_experiment`**: Full match play (first to 2000 points), 5 experiments comparing IS-MCTS variants and bidding strategies. Reports contracts taken/made, deal score distributions, coinches.
- **`bid_tournament`**: Round-robin tournament of parameterized bidding strategies. Each pair plays both directions with Naive IS-MCTS for card play. Reports win matrix, margin matrix, rankings.
- **`bid_debug`**: Prints detailed bidding rounds showing each player's hand, suit evaluations, and decisions for both heuristic and improved bidders side-by-side.
- **`strength_experiment`**: Rollout policy comparison, D×I sweep, RAVE on/off.

### Performance-Critical Path

`play.rs::legal_plays()` is the hottest function — all bitwise, no allocations. `rollout.rs` runs games to completion with random moves; state is copied (~56 bytes memcpy) per rollout. Target: >1M rollouts/sec single-threaded.

### Python Layer (`colver-py/`)

`Env` wraps a single GameState. `VecEnv(n)` wraps n parallel environments with NumPy array I/O. Observation is a 222-float vector (hand + trick cards + played cards + contract info + scores + phase + position). Legal action mask is 43 floats. Uses `StdRng` (not `ThreadRng`) for PyO3 `Send` requirement.

## Rules Reference

The official FFB rules are in `REGLES-DE-LA-BELOTE-CONTREE.pdf` at the repo root. Consult it for any rule ambiguities.
