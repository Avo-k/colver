# CLAUDE.md

## Build & Test Commands

```bash
cargo check                                    # Check compilation (both crates)
cargo test -p colver-core                      # Run all core tests
cargo test -p colver-core -- test_name         # Run a single test
cargo test -p colver-core --release            # Tests in release mode
cargo run -p colver-core --bin bench --release # Performance benchmark (~1.3M rollouts/sec)
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
- **DMC Agent** (`dmc_net.rs`): DouZero-style Q-network, 415→1024³→32, pure Rust inference ~1ms
- **NN Bidder "Le Bide à Dédé"** (`bid_net.rs`): Dueling DQN 114→256²→43, default for all web bots
- **Belief Network** (`belief_net.rs`): Card location prediction, V1/V2/V3 obs, multiple architecture variants
- **Bidding strategies** (`bid_eval.rs`): `BidADd` (NN, default), `Improved`, `Heuristic`, `Smart`, `Roro`, `Maxi`, `BidParams` (parametric)

## Python Layer (`colver-py/` → `python/colver/`)

`Env` wraps GameState with IS-MCTS/DMC support. Built as `colver._colver`, re-exported from `colver.__init__`. See `python/colver/_colver.pyi` for type stubs.

## Web Frontend (`python/colver/web/`)

FastAPI + WebSocket + vanilla JS. Three modes: Play, Watch, Analysis. Models auto-downloaded at startup (DMC 10MB, bid NN 421KB, belief net 2MB).

**Annonces page** (`views/annonces.js`): BidNet Q-values + Oracle DD table + DouDou simulation table. Oracle shows raw success % per suit×threshold. DouDou table uses Wilson score lower bound (z=1.645) for color thresholds (green/gold/red) and scales font size by observation count (0.65rem at 1 obs → 0.85rem at 20+) so small-sample cells appear visually less prominent than well-sampled ones.

## Publishing & Deployment

**PyPI:** push `v*` tag → CI builds manylinux/macOS/Windows wheels via maturin → publishes automatically (trusted publishing).

**Docker:** `docker build -t colver . && docker run -p 8000:8000 colver`. Cross-builds for ARM64.
