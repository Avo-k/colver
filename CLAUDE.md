# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test Commands

```bash
# Check compilation (both crates)
cargo check

# Run all core tests (93 tests)
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

### Performance-Critical Path

`play.rs::legal_plays()` is the hottest function — all bitwise, no allocations. `rollout.rs` runs games to completion with random moves; state is copied (~56 bytes memcpy) per rollout. Target: >1M rollouts/sec single-threaded.

### Python Layer (`colver-py/`)

`Env` wraps a single GameState. `VecEnv(n)` wraps n parallel environments with NumPy array I/O. Observation is a 222-float vector (hand + trick cards + played cards + contract info + scores + phase + position). Legal action mask is 43 floats. Uses `StdRng` (not `ThreadRng`) for PyO3 `Send` requirement.

## Rules Reference

The official FFB rules are in `REGLES-DE-LA-BELOTE-CONTREE.pdf` at the repo root. Consult it for any rule ambiguities.
