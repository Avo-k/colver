<p align="center">
  <img src="images/colver.png" alt="Colver logo" width="200">
</p>

# Colver

Fast Belote Contree game environment for reinforcement learning. Rust core with Python bindings.

## Features

- **~1.4M rollouts/sec** single-threaded (play phase), ~895K full-deal rollouts/sec
- **56-byte `Copy` game state** for cheap MCTS cloning
- **Three AI agents** — perfect-info MCTS, naive IS-MCTS, and smart belief-weighted IS-MCTS
- **Python bindings** via PyO3 — `Env` (single game) and `VecEnv` (batched) with NumPy I/O
- Zero dependencies in core (only `rand` behind a feature flag)

## Build & Run

Requires Rust 1.70+ and Python 3.8+.

```bash
# Run tests (104 tests)
cargo test -p colver-core

# Performance benchmark
cargo run -p colver-core --bin bench --release

# MCTS vs random demo
cargo run -p colver-core --bin mcts_demo --release -- 100

# IS-MCTS vs random demo
cargo run -p colver-core --bin ismcts_demo --release -- 100

# Smart IS-MCTS vs random + vs naive demo
cargo run -p colver-core --bin smart_ismcts_demo --release -- 100

# Oracle experiment (bid achievability)
cargo run -p colver-core --bin oracle_experiment --release -- 200 2000

# Bidding experiment (smart_bid vs heuristic)
cargo run -p colver-core --bin bidding_experiment --release -- 200 50

# Python bindings (via uv)
uv sync
uv run python3 -c "import colver; env = colver.Env(); print(env.reset())"

# Or via maturin directly
cd colver-py && maturin develop --release
```

## AI Agents

Colver includes three search-based agents with increasing sophistication. All use [Monte Carlo Tree Search](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) (MCTS) at their core — a technique that builds a search tree by repeatedly simulating random games and using the results to guide future decisions.

### Perfect-Info MCTS (`mcts.rs`)

Standard [UCT](https://link.springer.com/chapter/10.1007/11871842_29) (Upper Confidence bounds applied to Trees) with full hand visibility. This agent "cheats" by seeing all four hands, making it useful as an upper-bound baseline but unrealistic for actual play.

- Uses [UCB1](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf) for tree policy: balances exploitation (play moves that scored well) vs exploration (try moves that haven't been tested enough)
- Arena-based tree with `Node`s and `Edge`s in flat `Vec`s for cache-friendliness
- Rollout-to-terminal simulation with random legal moves
- Rewards scaled by 1/2000 to keep the exploitation term in \[0, 1\]
- Best action chosen by **robust child selection** (most-visited root child)

| Metric | 1000 iter | 4000 iter |
|---|---|---|
| Win% vs Random | 97% | — |
| Time per game | 8 ms | 67 ms |

### Naive IS-MCTS (`naive_ismcts.rs`)

Handles imperfect information via [ensemble determinization](https://doi.org/10.1109/CIG.2012.6374152) (also called "root parallelization over information sets"). The key idea from [Cowling, Powley, and Whitehouse (2012)](https://doi.org/10.1109/TCIAIG.2012.2200894):

> Instead of searching one tree, sample many "determinized" worlds (each a possible assignment of the hidden cards), run standard MCTS on each, and aggregate the results.

The agent only sees its own 8 cards and cards already played. For each search:
1. Sample D determinized worlds — randomly redistribute the 24 unknown cards among the 3 opponents, respecting known void constraints
2. Run standard MCTS (I iterations) on each determinized world
3. Aggregate root visit counts across all D worlds
4. Pick the most-visited action

This is essentially "Multiple Observer Information Set MCTS" from the literature. The determinization only respects void constraints (which suits a player is known to lack).

| Config (DxI) | Win% vs Random | Avg score | Time/game |
|---|---|---|---|
| 20x50 = 1000 | 92% | 1137 - 81 | 8 ms |
| 40x100 = 4000 | 90% | 1105 - 103 | 32 ms |

### Smart IS-MCTS (`smart_ismcts.rs` + `card_beliefs.rs`)

Extends naive IS-MCTS with a **belief model** that biases determinization based on information revealed during bidding and play. Instead of sampling worlds uniformly, it samples worlds that are *consistent with what opponents have signaled*.

The core idea builds on the concept of [opponent modeling in imperfect-information games](https://doi.org/10.1016/j.artint.2005.10.005). Every action reveals something about a player's hand:

- **Hard constraints** (definitive, weight = 0): suit voids, trump voids from discard rules, trump ceiling from overtrump rules, played/known cards
- **Soft constraints** (probabilistic, multiplicative weights): bidding signals (e.g., bidding 80 Hearts makes the Jack of Hearts ~5x more likely), play patterns (leading an Ace suggests holding 10 and King of that suit)

The belief model is a `[[f32; 32]; 4]` weight matrix — 128 floats (512 bytes) — where `weights[player][card]` represents the relative likelihood that `player` holds `card`. After normalization, these weights drive a **weighted determinization** that assigns cards to players proportionally, handling tightly-constrained cards first to avoid dead ends.

See [SMART_ISMCTS.md](SMART_ISMCTS.md) for the full design document.

| Opponent | Win% (NS) | Avg score | Time/game |
|---|---|---|---|
| Random | 88% | 1067 - 130 | 9 ms |
| Naive IS-MCTS (equal budget) | 46% | 536 - 647 | 17 ms |

The Smart vs Naive matchup is roughly even at the default budget, suggesting the soft inference weights need further tuning or that hard constraints alone capture most useful information at low search budgets.

### Bidding Strategies (`bid_eval.rs`)

Colver includes two deterministic bidding strategies, both fast enough (~200 ops) for use inside MCTS rollouts.

**Heuristic bid** — Score-based. Evaluates trump strength (J=8, 9=6, A=4, 10=3, K/Q=1) plus length bonus and side suit features (aces, voids). Maps total score to bid value (80-130). Simple and effective. Avg bid ~117, ~72% achievable with perfect play on both sides.

**Smart bid** — Convention-based, mimicking human Belote Contrée signaling:

| Situation | Logic | Bid range |
|---|---|---|
| **Opening with J+9** | Scale by side aces / trump count | 80-100 |
| **Opening with J XOR 9** | Signal missing honor (need 3+ trumps) | 80 |
| **Opening "aux as"** | 2+ aces, no J/9 | 80 |
| **Partner response** | On partner's 80: respond 90 if holding missing J/9. On 90+: PASS | 90 max |
| **Overcall** | J+9 in different suit, score ≥ 14, cap at 100 | 80-100 |
| **Coinche** | J+9 in opponent's suit, 4+ trumps, or 3+ trumps + ace on bids ≥ 120 | — |

The key design principle is **one-shot communication**: partner responses are limited to a single raise (80→90), and overcalls cap at 100. This eliminates the escalation spirals common in naive convention implementations. Avg bid ~88, ~78% achievable with perfect play.

#### Oracle Experiment Results

The oracle experiment pairs each bidding strategy with perfect-info MCTS (sees all cards) to measure what % of contracts are inherently achievable:

| Bidding + Play | Avg Bid | Contract Success |
|---|---|---|
| smart_bid + Oracle vs Oracle | 88 | **78%** |
| heuristic + Oracle vs Oracle | 117 | 72% |
| smart_bid + Oracle vs Random | 88 | 98% |

Smart bid's lower, more conservative contracts are more frequently achievable even against perfect defense, while still being almost always achievable against imperfect defense.

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

Bidding -> Playing -> Done. Bidding ends on 3 passes after a bid, surcoinche, or 4 passes (void deal). Playing runs 8 tricks of 4 cards. Total card points = 152; with dix de der = 162 (normal) or 252 (capot).

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
| Play-phase rollout | 1.4M/sec | ~720 ns |
| Full-deal rollout | 895K/sec | ~1118 ns |
| MCTS game (1000 iter) vs random | — | 8 ms |
| Smart IS-MCTS game (20x50) vs random | — | 9 ms |

See [BENCH.md](BENCH.md) for detailed benchmarks.

## References

- Kocsis, L. and Szepesvari, C. (2006). [Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29). *ECML*.
- Cowling, P.I., Powley, E.J. and Whitehouse, D. (2012). [Information Set Monte Carlo Tree Search](https://doi.org/10.1109/TCIAIG.2012.2200894). *IEEE Transactions on Computational Intelligence and AI in Games*.
- Auer, P., Cesa-Bianchi, N. and Fischer, P. (2002). [Finite-time Analysis of the Multiarmed Bandit Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf). *Machine Learning*.
- Billings, D. et al. (2006). [Algorithms and Assessment in Computer Poker](https://doi.org/10.1016/j.artint.2005.10.005). *Artificial Intelligence*.

## Rules

Implements Belote Contree with 4 color suits (Spades, Hearts, Diamonds, Clubs). Scoring uses "points faits + points demandes" mode. See `REGLES-DE-LA-BELOTE-CONTREE.pdf` for the full FFB rulebook.
