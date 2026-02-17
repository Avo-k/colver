<p align="center">
  <img src="https://raw.githubusercontent.com/Avo-k/colver/master/images/colver.png" alt="Colver Logo" width="200">
</p>

<p align="center">
  <a href="https://pypi.org/project/colver/"><img src="https://img.shields.io/pypi/v/colver?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/colver/"><img src="https://img.shields.io/pypi/pyversions/colver" alt="Python"></a>
  <a href="https://avok.me/colver/"><img src="https://img.shields.io/badge/demo-avok.me%2Fcolver-green" alt="Live Demo"></a>
  <a href="https://github.com/Avo-k/colver/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/colver" alt="License"></a>
</p>

# Colver

**[Lire en francais](README.fr.md)**

Fast Belote Contree game environment for reinforcement learning. Rust core with Python bindings.

**Live demo: [avok.me/colver/](https://avok.me/colver/)** — running on a Raspberry Pi.

## Features

- **~1.4M rollouts/sec** single-threaded (play phase), ~895K rollouts/sec on a full deal
- **56-byte `Copy` game state** for fast MCTS cloning
- **Four AI agents** — perfect-info MCTS, Naive IS-MCTS, belief-weighted Smart IS-MCTS, and a Q-network (Deep Monte-Carlo)
- **Web interface** — play against AI in the browser (FastAPI + WebSocket)
- **Python bindings** via PyO3 — `Env` class with full type stubs, installable from PyPI
- Zero dependencies in the core (only `rand` behind a feature flag)

## Web Interface

Play against AI agents directly in your browser at **[avok.me/colver/](https://avok.me/colver/)**, or run it locally:

```bash
uv run python -m colver.web
# Or: uv run colver-web
# Open http://localhost:8000
```

The interface has four tabs:

### Play

Play as South against AI opponents. Choose the agent for your opponents (East/West) and your partner (North) independently — DouDou, Smart IS-MCTS, Naive IS-MCTS, or Oracle. The game follows official FFB Belote Contree rules: bidding with coinche/surcoinche, then 8 tricks. Playable cards are raised, illegal cards are greyed out. The last trick is shown with points and winner.

![Play tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-play.png)

### Watch

Spectate AI vs AI matches with all hands visible. Assign a different agent to each of the 4 seats (including Heuristic and Random). Step through actions one by one, play full tricks, or use auto-play with adjustable speed. The stats panel shows MCTS visit counts and Q-values for each decision, with full bidding and trick history.

![Watch tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-watch.png)

### Analysis

Set up a custom position by dragging cards into 4 player drop zones, or generate a random deal. Configure the contract (trump suit, value, declaring team), then run IS-MCTS analysis to find the best move and action rankings for the current player.

![Analysis tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-analysis.png)

### Docs

In-app documentation describing the game modes and each AI agent in detail.

![Docs tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-docs.png)

## Build & Run

Requires Rust 1.70+ and Python 3.10+.

```bash
# Tests (168 tests)
cargo test -p colver-core

# Performance benchmark
cargo run -p colver-core --bin bench --release

# MCTS vs random demo
cargo run -p colver-core --bin mcts_demo --release -- 100

# Smart IS-MCTS vs random + vs naive demo
cargo run -p colver-core --bin smart_ismcts_demo --release -- 100

# Python bindings (via uv)
uv sync
uv run python3 -c "import colver; env = colver.Env(); print(env.reset())"

# Web interface (play against AI)
uv run python -m colver.web

# DMC training (Q-network)
PYTHONPATH=scripts uv run python scripts/train_dmc.py --num-envs 256 --steps 20000000

# DMC evaluation vs IS-MCTS
PYTHONPATH=scripts uv run python scripts/eval_dmc.py models/dmc_final.pt --baseline smart --time-ms 20 --both-sides
```

## AI Agents

Colver includes four agents of increasing sophistication.

### 1. Perfect-info MCTS (`mcts.rs`)

Standard [UCT](https://link.springer.com/chapter/10.1007/11871842_29) (Upper Confidence bounds applied to Trees) with full hand visibility. This agent "cheats" by seeing all 4 hands — useful as an upper bound but unrealistic for real play.

- [UCB1](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf) tree policy: balances exploitation vs exploration
- Arena-based tree with `Node` and `Edge` in contiguous `Vec`s for cache locality
- Rollout simulation to completion with random legal moves
- Best action: most-visited root child

| Metric | 1000 iter | 4000 iter |
|---|---|---|
| Win rate vs Random | 97% | — |
| Time per game | 8 ms | 67 ms |

### 2. Naive IS-MCTS (`naive_ismcts.rs`)

Handles imperfect information via [ensemble determinization](https://doi.org/10.1109/CIG.2012.6374152). The key idea from [Cowling, Powley & Whitehouse (2012)](https://doi.org/10.1109/TCIAIG.2012.2200894):

> Instead of searching a single tree, sample multiple "determinized" worlds (each being a possible distribution of hidden cards), run standard MCTS on each, and aggregate the results.

The agent only sees its 8 cards and previously played cards. For each search:
1. Sample D determinized worlds — redistribute the 24 unknown cards among the 3 opponents, respecting known void constraints
2. Run standard MCTS (I iterations) on each world
3. Aggregate root visit counts across all D worlds
4. Choose the most-visited action

| Config (DxI) | Win rate vs Random | Avg score | Time/game |
|---|---|---|---|
| 20x50 = 1000 | 92% | 1137 - 81 | 8 ms |
| 40x100 = 4000 | 90% | 1105 - 103 | 32 ms |

### 3. Smart IS-MCTS (`smart_ismcts.rs` + `card_beliefs.rs`)

Extends Naive IS-MCTS with a **belief model** that biases determinization based on information revealed during bidding and play. Instead of sampling worlds uniformly, it samples worlds *consistent with what opponents have signaled*.

Based on [opponent modeling in imperfect information games](https://doi.org/10.1016/j.artint.2005.10.005). Each action reveals something about a player's hand:

- **Hard constraints** (weight = 0): known voids, trump ceiling, played/known cards
- **Soft constraints** (multiplicative weights): bidding signals (bidding Hearts makes the Jack of Hearts ~5x more likely), play patterns (leading an Ace suggests also holding 10 and King)

The belief model is a `[[f32; 32]; 4]` matrix — 128 floats — where `weights[player][card]` is the relative probability that `player` holds `card`.

See [SMART_ISMCTS.md](SMART_ISMCTS.md) for the full design document.

| Opponent | Win rate | Avg score | Time/game |
|---|---|---|---|
| Random | 88% | 1067 - 130 | 9 ms |
| Naive IS-MCTS (equal budget) | 46% | 536 - 647 | 17 ms |

### 4. DMC Agent (Deep Monte-Carlo) (`scripts/dmc_model.py`)

[DouZero](https://arxiv.org/abs/2106.06135)-style reinforcement learning agent. A Q-network picks card plays with a single forward pass — **no search tree**. Bidding uses `improved_bid` (not learned).

**Architecture v4**: MLP 415→1024→1024→1024→32 with LayerNorm (~2.6M parameters). Player-relative observation (415 floats): hand, trick, per-player played cards, contract, void tracking, scoring context, bid history.

**Training**: Deep Monte-Carlo (DMC) with Prioritized Experience Replay (PER), opponent pool (70% self-play, 20% past checkpoints, 10% random), 20M steps, 2M replay buffer.

**Inference**: pure Rust forward pass (~1ms/decision, no PyTorch needed).

### Bidding Strategies (`bid_eval.rs`)

Deterministic bidding strategies, fast enough (~200 ops) for use in MCTS rollouts.

**`improved_v2_bid`** (default) — Tournament-winning balanced strategy. Quality gate (J/9/A/10 or 3+ cards in suit), score→value mapping: 10→80, 13→90, 17→100, 20→110, 25→120. Opening cap 120, overcall cap 120, response cap 130.

**`heuristic_bid`** — Aggressive. No quality gate, no cap. Takes ~50% of contracts with ~70% success rate.

**`smart_bid`** — Conservative convention-based. J/9 signaling between partners. Very conservative (~10-13% take rate, ~78% success).

## Agent Comparison

| Agent | Type | Win rate vs Random | Speed/move | Bidding |
|---|---|---|---|---|
| Perfect MCTS | Search (cheats) | 97% | ~8ms | improved_bid |
| Naive IS-MCTS | Search | 92% | ~8ms | improved_bid |
| Smart IS-MCTS | Search + beliefs | 88% | ~9ms | improved_bid |
| **DMC Q-Network** | **Neural network** | **66%** | **<1ms** | improved_bid |
| Random | Baseline | 50% | ~0ms | — |

**Note**: Search-based agents (IS-MCTS) get stronger with more time budget. Numbers above use the default budget (~8-9ms/move). The DMC agent uses no search — one forward pass per decision.

## Architecture

**Workspace:** `colver-core` (pure Rust) + `colver-py` (PyO3/NumPy FFI) + `colver-web` (FastAPI/WebSocket)

### Card Representation

Bitmask system: `Card = u8` (0-31), `CardSet = u32` (bitmask). Layout: Spades\[0-7\], Hearts\[8-15\], Diamonds\[16-23\], Clubs\[24-31\]. Within each suit: 7, 8, 9, J, Q, K, 10, A (plain strength order). Trump strength: J > 9 > A > 10 > K > Q > 8 > 7.

### Game State

`GameState` is `Copy` and 56 bytes (compile-time enforced ≤64) for fast MCTS cloning. Contains hands, current trick, contract, points/tricks per team, bidding state, played cards bitmask, void tracking, and belote tracking.

### Action Encoding

| Phase | Actions | Encoding |
|---|---|---|
| Bidding | 43 total | 0=PASS, 1-36=bids (9 values x 4 suits), 37-40=capot x 4, 41=COINCHE, 42=SURCOINCHE |
| Playing | 32 total | Card index 0-31 directly |

### Game Flow

Bidding → Playing → Done. Bidding ends after 3 consecutive passes, a surcoinche, or 4 passes (void deal). Playing runs 8 tricks of 4 cards. Card point total = 152; with dix de der = 162 (normal) or 252 (capot).

## Python API

```python
import colver

print(colver.__version__)  # "0.2.0"

# Single environment
env = colver.Env()
obs, legal_actions = env.reset()
obs, reward, done, legal_actions = env.step(action)

env.current_player()       # 0-3
env.phase()                # 0=Bidding, 1=Playing, 2=Done
env.legal_action_mask()    # numpy array (43,)
env.rewards()              # [NS_score, EW_score]
env.bid_improved()         # improved_bid action
env.deal_outcome()         # [NS_outcome, EW_outcome] binary
env.get_observation()      # 415-float observation vector
env.action_naive_ismcts(20)  # naive IS-MCTS action (20ms)
env.action_smart_ismcts(20)  # smart IS-MCTS action (20ms)

# DMC Q-network (if model weights downloaded)
model = colver.model_path()  # ~/.cache/colver/models/dmc_final.bin
if model:
    env.load_dmc_model(str(model))
    result = env.action_dmc_with_stats()  # {"best_action": 5, "q_values": [...]}
```

## Performance

| Workload | Throughput | Latency |
|---|---|---|
| Play-phase rollout | 1.4M/sec | ~720 ns |
| Full-deal rollout | 895K/sec | ~1118 ns |
| MCTS game (1000 iter) vs random | — | 8 ms |
| Smart IS-MCTS game (20x50) vs random | — | 9 ms |
| DMC Q-Network inference | — | <1 ms |

## Docker

The Docker image lets you deploy the web interface on any machine, including a Raspberry Pi (ARM64).

```bash
# Build and run
docker build -t colver .
docker run -p 8000:8000 colver

# Or with Docker Compose
docker compose up -d

# Cross-build for Raspberry Pi (ARM64)
docker buildx build --platform linux/arm64 -t colver .
```

The image is ~257 MB (no PyTorch dependency). All agents run in pure Rust and work on all architectures.

## Rules

Implements Belote Contree with 4 suits (Spades, Hearts, Diamonds, Clubs). Scoring mode: "points faits + points demandes". See `REGLES-DE-LA-BELOTE-CONTREE.pdf` for the full FFB rulebook.

## References

- Kocsis, L. & Szepesvari, C. (2006). [Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29). *ECML*.
- Cowling, P.I., Powley, E.J. & Whitehouse, D. (2012). [Information Set Monte Carlo Tree Search](https://doi.org/10.1109/TCIAIG.2012.2200894). *IEEE Transactions on Computational Intelligence and AI in Games*.
- Zha, D. et al. (2021). [DouZero: Mastering DouDiZhu with Self-Play Deep Reinforcement Learning](https://arxiv.org/abs/2106.06135). *ICML*.
- Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). [Finite-time Analysis of the Multiarmed Bandit Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf). *Machine Learning*.
