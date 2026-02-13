<p align="center">
  <img src="images/colver.png" alt="Colver logo" width="200">
</p>

# Colver

Fast Belote Contree game environment for reinforcement learning. Rust core with Python bindings.

## Features

- **~1.3M rollouts/sec** single-threaded (play phase), ~856K full-deal rollouts/sec
- **56-byte `Copy` game state** for cheap MCTS cloning
- **MCTS agents** — perfect-info UCT and naive IS-MCTS (ensemble determinization)
- **Python bindings** via PyO3 — `Env` (single game) and `VecEnv` (batched) with NumPy I/O
- Zero dependencies in core (only `rand` behind a feature flag)

## Build & Run

Requires Rust 1.70+ and Python 3.8+.

```bash
# Run tests (75 tests)
cargo test -p colver-core

# Performance benchmark
cargo run -p colver-core --bin bench --release

# MCTS vs random demo
cargo run -p colver-core --bin mcts_demo --release -- 100

# IS-MCTS vs random demo
cargo run -p colver-core --bin ismcts_demo --release -- 100

# Python bindings (via uv)
uv sync
uv run python3 -c "import colver; env = colver.Env(); print(env.reset())"

# Or via maturin directly
cd colver-py && maturin develop --release
```

## Architecture

**Workspace:** `colver-core` (pure Rust) + `colver-py` (PyO3/NumPy FFI)

### Card Representation

Cards use a bitmask system: `Card = u8` (0-31), `CardSet = u32` (bitmask). Bit layout: Spades\[0-7\], Hearts\[8-15\], Diamonds\[16-23\], Clubs\[24-31\]. Within each suit, ranks go 7, 8, 9, J, Q, K, 10, A (plain strength order). Trump strength differs: J > 9 > A > 10 > K > Q > 8 > 7.

### Game State

`GameState` is `Copy` and 56 bytes (compile-time enforced <= 64) for fast MCTS cloning. Contains hands, current trick, contract, points/tricks per team, bidding state, played cards bitmask, void tracking, and belote tracking.

### Action Encoding

| Phase | Actions | Encoding |
|---|---|---|
| Bidding | 43 total | 0=PASS, 1-36=bids (9 values x 4 suits), 37-40=capot x 4 suits, 41=COINCHE, 42=SURCOINCHE |
| Playing | 32 total | Card index 0-31 directly |

### Game Flow

Bidding → Playing → Done. Bidding ends on 3 passes after a bid, surcoinche, or 4 passes (void deal). Playing runs 8 tricks of 4 cards. Total card points = 152; with dix de der = 162 (normal) or 252 (capot).

## Python API

```python
import colver

# Single environment
env = colver.Env()
obs, legal_actions = env.reset()
obs, reward, done, legal_actions = env.step(action)

env.current_player()    # 0-3
env.phase()             # 0=Bidding, 1=Playing, 2=Done
env.legal_action_mask() # numpy array (43,)
env.rewards()           # [NS_score, EW_score]
env.rollout(1000)       # average rewards from 1000 random rollouts

# Vectorized environment for batch RL
venv = colver.VecEnv(256)
obs, masks = venv.reset()                     # (256, 222), (256, 43)
obs, rewards, dones, masks = venv.step(actions)  # actions: list of 256 ints
```

**Observation** (222 floats): hand (32) + current trick (4x32) + played cards (32) + contract info (trump 4 + value 1 + coinche 3 + team 2) + points (2) + tricks (2) + phase (3) + relative position (4).

## Performance

| Workload | Throughput | Latency |
|---|---|---|
| Play-phase rollout | 1.3M/sec | ~770 ns |
| Full-deal rollout | 856K/sec | ~1170 ns |
| MCTS game (1000 iter) vs random | — | 8 ms |

See [BENCH.md](BENCH.md) for detailed benchmarks including IS-MCTS.

## Rules

Implements Belote Contree with 4 color suits (Spades, Hearts, Diamonds, Clubs). Scoring uses "points faits + points demandes" mode. See `REGLES-DE-LA-BELOTE-CONTREE.pdf` for the full FFB rulebook.
