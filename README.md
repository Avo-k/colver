<p align="center">
  <img src="https://raw.githubusercontent.com/Avo-k/colver/master/images/colver.png" alt="Colver Logo" width="200">
</p>

<p align="center">
  <a href="https://pypi.org/project/colver/"><img src="https://img.shields.io/pypi/v/colver?color=blue" alt="PyPI"></a>
  <a href="https://pypi.org/project/colver/"><img src="https://img.shields.io/pypi/pyversions/colver" alt="Python"></a>
  <a href="https://colver.net/"><img src="https://img.shields.io/badge/play-colver.net-green" alt="colver.net"></a>
  <a href="https://github.com/Avo-k/colver/blob/master/LICENSE"><img src="https://img.shields.io/pypi/l/colver" alt="License"></a>
</p>

# Colver

**[Lire en francais](README.fr.md)**

Fast Belote Contree game environment for reinforcement learning. Rust core with Python bindings.

**Play online: [colver.net](https://colver.net)** — a self-hosted public site: solo or multiplayer games, accounts, ratings and analysis.

## Quickstart

```bash
pip install colver
```

```python
import colver

env = colver.Env()
env.reset()
env.load_bid_model(colver.download_bid_model())   # fetched from the Hub on first use
print(colver.Env.action_name(env.action_bid_nn()["best_action"], 0))
```

Model weights live on the [Hugging Face Hub](https://huggingface.co/collections/Avo-k/colver-belote-contree-6a71df4a723e6734fe623a65)
— one repo per model, each with a card explaining what it does, what it is worth and what
it cannot do. `download_*()` caches them under `~/.cache/colver/models/` (GitHub Releases
is kept as a fallback); point `COLVER_MODEL_PATH` & co. at your own files to override.

## Features

- **~1.4M rollouts/sec** single-threaded (play phase), ~895K rollouts/sec on a full deal
- **≤96-byte `Copy` game state** for fast MCTS cloning
- **Six AI agents** — DMC Q-network, IS-DD, DD oracle, Smart/Naive IS-MCTS, and heuristic
- **NN bidding** — "Bid V6 IS-DD", a score- and belote-aware Dueling DQN (117→512³→43) trained 75M steps on real IS-DD points with full match simulation, used by all agents
- **ML interpretability** — XGBoost distillation + hidden-layer probe reveal the NN's implicit scoring system, translated into human-usable rules (88-94% agreement)
- **Playgen world model** — a causal transformer that continues a deal from what one seat can see; rolling it out reveals the hidden hands, which is where IS-DD's sampled worlds now come from
- **One agent layer** — a bot is a TOML spec (`[bid]` / `[play]` / `[worlds]`), built by the Rust side and driven identically by the arena, the web and `colver.Agent`
- **Web interface** — play solo or in multiplayer rooms, spectate, replay with per-card analysis, and train on problems (FastAPI + WebSocket)
- **Python bindings** via PyO3 — `Env`, `Agent`, `Analyst` and `Beliefs` with full type stubs, installable from PyPI
- Zero dependencies in the core (only `rand` behind a feature flag)

## Web Interface

Play against AI agents directly in your browser at **[colver.net](https://colver.net)**, or run it locally:

```bash
uv run python -m colver.web
# Or: uv run colver-web
# Open http://localhost:8000
```

Everything is in French, behind four destinations: **Jouer**, **Analyser**, **Apprendre**, **Classement**. An account is optional — it is what links games to you, and what makes matches resumable and rated.

### Jouer

**Humain vs IA** — Play as South against three bots. There are exactly two settings, both chosen before the deal:

- **Tempo** — `Standard` (Dede, ~40 s a deal) or `Rapide` (DouDou50, ~15 s). The bundle is deliberate: an IS-DD search costs real wall-clock per move, so a fast tempo is only honest behind a bot that answers instantly. All four AI seats run the same bot — a table where your partner is weaker than your opponents tells you nothing about how you played.
- **Match length** — a single deal (default), or a match to 1000 / 2000 points. The running score is passed to the bots, and Bid V6 reads it: it bids differently at 900-200 than at 0-0.

Cards are played instantly on click; the pause belongs to the position *before* a move and the bot thinks inside it, not on top of it. Forced passes are played for you, the last trick (where nobody has a choice left) runs itself, and the final trick is held 2 s before the end-of-deal panel. Signed-in players can leave a match and resume it later.

![Play tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-play.png)

**Salon multijoueur** — Rooms with a 4-character join code; bots fill whatever seats humans do not. The host picks the tempo and the match length. Every broadcast state is filtered to the viewer's own hand and rotated so the viewer always sits South. Seats are account-bound, so a disconnected player can come back to their seat.

![Salon](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-salon.png)

**IA vs IA** — Spectate AI vs AI matches with all hands visible. Assign a different agent to each of the 4 seats. Step through actions, play full tricks, or use auto-play. The stats panel shows Q-values, DD scores, or hand evaluations for each decision. Paste a CFN string to load a specific position.

![Watch tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-watch.png)

### Analyser

**Rejouer** — Browse and replay past deals (played, spectated or shared) and step through them card by card. Two independent analysis passes run on top, each cached and fetched separately so the slow one never blocks the fast one:

- the **DD pass** — the exact cost of every card, plus a bid review carrying two opinions per auction action: Bid V6's Q-values and playgen's 43-way auction head;
- the **agent review** — what DouDou50, the Oracle and Dede would have played at every non-forced card, streamed in as it computes.

Every card and every bid links out to its own analysis page, and back.

![Rejouer](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-replay.png)

**Annonces** — Compose an 8-card hand, set the bids that preceded your turn, and see what *Bid V6 IS-DD* would call — Q-values for every legal action, plus an XGBoost-distilled "key factors" panel (which features drove the decision). Two tables then play the hand out over hundreds of random deals of the other 24 cards: **Jeu parfait** (each deal solved double-dummy by the Oracle — a theoretical ceiling) and **Jeu reel** (the full auction bid by the NN at all four seats, then 8 tricks played by DouDou50 — what actually happens). One tab per bid analysed, so two bids on the same hand can be compared side by side, and analysed hands are kept in a sidebar. Runs server-side or fully in the browser via WASM ("Calcul local").

![Annonces tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-annonces.png)

**Jeu de la carte** — One row per playable card at a given position, answering two questions that must not be confused: *the worlds of the information set* (deals compatible with what the seat could know, each solved double-dummy — this is where the decision is judged) and *the real world* (a single exact solve on the deal as it actually was). A third block forces the card and lets DouDou50 finish the deal. A card that was second-best in the real deal but best in 70 % of the worlds was a good play against bad luck.

![Jeu de la carte](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-analyse-jeu.png)

**Croyances** — Watch *playgen* locate the hidden cards as a deal unfolds, during the auction as well as during play. Step through a generated deal and read the per-card probability bars against ground truth, from any of the four seats.

![Croyances tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-croyances.png)

**Problemes d'annonce** — Bidding practice problems. See a hand and bidding history, then find the right bid. The AI evaluates your answer against the NN bidder's recommendation.

**Problemes de jeu** — Card play practice problems. See a mid-game position and find the best card. Compare your choice to the DD solver's optimal play.

### Apprendre

**Aide-memoire** — Visual cheat sheet: card strength order and values (trump / non-trump), deal points, bidding rules.

**Guide des annonces** — Visual strategy guide derived from the bot via ML (per-card point weights, decision rules per position, mirror rule for defense). 88-94% agreement with the NN, memorizable in minutes.

![Annoncer tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-annoncer.png)

**Marquer les points** — Score keeper for real-life games. Pick a target score (1000-3000), add rounds with a form that computes exact FFB scores automatically (contract, coinche/surcoinche multiplier, points made, belote), and watch the live win-probability estimate update after every round. Everything lives in the browser — no server, no account.

![Marquer tab](https://raw.githubusercontent.com/Avo-k/colver/master/images/screenshots/tab-score.png)

### Classement

One Elo per rated entity — a human account or a bot type — updated deal by deal whenever all four seats are identifiable. Bots use a lower K: they are reference points and occupy up to three seats per game.

## Build & Run

Requires Rust 1.70+ and Python 3.10+.

```bash
# Tests (418 tests)
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

# Bot vs bot, 200 matches each way (bots are TOML files in arena/bots/)
cargo run --bin arena --release -- h2h v6_isdd_75M_belief v6_isdd_75M --matches 200

# Joint bid+play training (GPU, candle)
cargo run -p colver-core --bin train_joint --features dmc_train --release -- --num-envs 256 --steps 35000000

# Playgen GPU sidecar — where IS-DD gets its worlds from
cargo run -p colver-core --bin playgen_gpu_server --features gpu_server --release -- --playgen models/playgen/playgen_v2_final.bin --port 8003
export COLVER_PLAYGEN_GPU_URL=http://localhost:8003
```

Without `$COLVER_PLAYGEN_GPU_URL`, IS-DD bots fall back to constraint-uniform worlds (the web says so in the decision's stats; the arena refuses to build the bot unless the spec opts in).

## AI Agents

### Oracle — DD Solver (`solver.rs`)

Perfect-information double-dummy solver that sees all 4 hands — it *cheats*. Alpha-beta with transposition tables, PVS, killer moves, and card equivalence pruning. Computes the exact optimal card in ~35 ms from a full deal, ~190 us mid-game and ~1.5 us in the endgame (measured 2026-08-02, 1 thread). Useful as an upper bound.

### Dede — IS-DD (`is_dd.rs`)

Information Set Double-Dummy search: sample plausible deals consistent with what this seat can know, solve each one exactly with the alpha-beta DD solver, aggregate. Hard constraints (voids revealed by the play, trump ceiling, cards already gone) are facts and always apply. The sampled worlds come from a `WorldSource` the agent owns — **playgen over the GPU sidecar by default**, with a constraint-uniform sampler as the fallback. A **belief network** can additionally weight the sampling. IS-DD sounds like "is Dede" — hence the name.

### DouDou50 — DMC Q-Network (`dmc_net.rs`)

[DouZero](https://arxiv.org/abs/2106.06135)-style reinforcement learning agent. A Q-network picks card plays with a single forward pass — **no search tree**. Default play model, trained 50M steps with the NN bidder frozen (triforge play-only phase).

**Architecture**: ResNet Dueling DQN 411→1024→1024→1024→32 with LayerNorm and skip connections (~2.6M parameters). Uses canonical suit encoding (no augmentation needed). Inference in pure Rust (~1ms/decision, no PyTorch needed). Strongest overall agent.

The previous model **DouDou35** (415→1024³→32, legacy obs, 35M steps) is still supported for backward compatibility. *DouDou* = a reference to DouZero.

### Playgen — world model (`playgen/`)

A causal transformer (10.7M params) trained to continue a deal autoregressively from the prefix one observer can see. Rolling it out reveals the hidden hands, so a rollout *is* a determinized world drawn from a learned posterior rather than from a uniform shuffle — that is what IS-DD samples, and what the Croyances page displays. v2 also carries a 43-way auction head, which makes it usable for mid-auction sampling (and, as a curiosity, as a bidder in its own right). It runs on CPU or CUDA; in production a sidecar (`playgen_gpu_server`) serves it over HTTP, ~50× faster than sampling on CPU.

### Older search agents

**Smart IS-MCTS** (`smart_ismcts.rs`) — Belief-weighted [Information Set MCTS](https://doi.org/10.1109/TCIAIG.2012.2200894) with heuristic card beliefs. **Naive IS-MCTS** (`naive_ismcts.rs`) — Ensemble determinization without beliefs. Both are configurable and documented in [docs/play/smart_ismcts.md](docs/play/smart_ismcts.md).

### Bid V6 IS-DD — NN Bidder (`bid_net.rs`)

Dueling DQN **117→512→512→512→43** with score-aware v3 observation (match-score features + 4 belote bits). Trained **75M steps on real IS-DD points** (not DD oracle) with belote-aware reward and full **match simulation** (cumulative scores, dealer rotation, reset at 2000). Beats the previous champion Bid V5 in both eval sets (55.8% with DMC play, 57.3% / +181 pts with IS-DD play). `BidNet::load` auto-detects hidden size and obs_dim (108 / 110 / 113 / 117).

Past versions still supported via auto-detect:
- **Bid V5 IS-DD** — 113-dim score-aware obs, 25M steps on real IS-DD points
- **Bid V3 Max** — 108-dim, trained on `max(DMC, IS-DD)` real points (20M steps)
- **Bid à Dédé** (v2) — 108-dim, DD oracle reward
- **Bid à Doudou** (v1) — 114→256² dueling, DouZero self-play

**Interpretability**: XGBoost distillation and a hidden-layer linear probe revealed that the NN's implicit scoring differs sharply from the classical hand evaluation (e.g., J atout = +11 effective, 9 = +4, A atout = +1, side A = 0 net; plus an anti-synergy J×9 = −2). Translated into a mnemonic 5-feature decision tree reaching 88-94% NN agreement — see [docs/bid/strategies/bid_v5_human_guide.md](docs/bid/strategies/bid_v5_human_guide.md) and [docs/bid/interpretability/probe_morning_report.md](docs/bid/interpretability/probe_morning_report.md).

## Agent Comparison

| Agent | Type | Speed/move | Notes |
|---|---|---|---|
| Oracle (DD) | DD solver (cheats) | 35 ms -> 1.5 us (trick 1 -> trick 8) | Perfect info upper bound |
| Dede (IS-DD) | DD solver over sampled worlds | budget (1 s in the web) | Strongest search-based |
| **DouDou50** | **Q-network (ResNet)** | **<1ms** | Strongest overall, no search |
| Smart IS-MCTS | Search + beliefs | ~9ms | Configurable budget |
| Naive IS-MCTS | Search | ~8ms | Configurable budget |

**Note**: Search-based agents get stronger with more time budget. The DMC agent uses no search — one forward pass per decision.

## Bots are specs, not code paths

A bot is a TOML file describing a bidder, a card player and (for IS-DD) a world source. The arena, the web frontend and `colver.Agent` all read the same spec and get the same agent — there is no second implementation anywhere.

```toml
[bid]
strategy = "nn"                    # heuristic|improved|smart|roro|maxi|nn|playgen|…
model = "models/bid_v6_isdd_resume/bid_nn_final.bin"

[play]
method = "isdd"                    # isdd|dmc|dmc_then_isdd|ismcts|smart_ismcts|oracle_dd|heuristic
time_ms = 1000

[worlds]                           # IS-DD only; defaults to the sidecar
source = "sidecar"
url = "http://localhost:8003"
```

`arena/bots/*.toml` holds the reference bots; head-to-head and round-robin results accumulate in `arena/results/matches.csv`. Testing a new combination means writing a file, not recompiling.

## Architecture

**Workspace:** `colver-core` (pure Rust) + `colver-py` (PyO3/NumPy FFI) + `colver-wasm` (in-browser solver) + `python/colver/web` (FastAPI/WebSocket)

### Card Representation

Bitmask system: `Card = u8` (0-31), `CardSet = u32` (bitmask). Layout: Spades\[0-7\], Hearts\[8-15\], Diamonds\[16-23\], Clubs\[24-31\]. Within each suit: 7, 8, 9, J, Q, K, 10, A (plain strength order). Trump strength: J > 9 > A > 10 > K > Q > 8 > 7.

### Game State

`GameState` is `Copy` and ≤96 bytes (compile-time enforced) for fast MCTS cloning. Contains hands, current trick, contract, points/tricks per team, bidding state, played cards bitmask, void tracking, and belote tracking.

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

print(colver.__version__)  # "0.9.1"

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

# A whole bot from a spec — same TOML the arena reads
agent = colver.Agent(spec_toml, seat=2)
agent.init_deal(env)
agent.observe(env, action)       # every action, including your own
decision = agent.decide(env)     # {"action": …, plus per-method stats}

# Playgen as an analyst: hidden-card marginals, sampled worlds, auction policy
analyst = colver.Analyst("models/playgen/playgen_v2_final.bin")
analyst.init_deal(env, observer=2)
probs = analyst.marginals(env, n_worlds=50)
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

The Docker image lets you deploy the web interface on any machine, x86-64 or ARM64.

```bash
# Build and run
docker build -t colver .
docker run -p 8000:8000 colver

# Or with Docker Compose
docker compose up -d

# Cross-build for ARM64
docker buildx build --platform linux/arm64 -t colver .
```

The image is ~257 MB (no PyTorch dependency). All agents run in pure Rust and work on all architectures.

## Rules

Implements Belote Contree with 4 suits (Spades, Hearts, Diamonds, Clubs). Scoring mode: "points faits + points demandes", with the coinche (x2) and surcoinche (x3) multipliers applying to the contract value only. On a chute the defense takes the contract *and* every card point, whatever the real split of tricks.

Scores are **exact — nothing is rounded**. The FFB rounds the marque to the nearest 10; here a chute marks 162 + the contract, not 160, so the engine and the web score sheet always show the same number. See the [FFB rulebook](https://www.ffbelote.org/wp-content/uploads/2015/11/REGLES-DE-LA-BELOTE-CONTREE.pdf) for the full official text, and [docs/rules-survey/](docs/rules-survey/) for how ~594 published rulebooks actually differ from it.

## References

- Kocsis, L. & Szepesvari, C. (2006). [Bandit Based Monte-Carlo Planning](https://link.springer.com/chapter/10.1007/11871842_29). *ECML*.
- Cowling, P.I., Powley, E.J. & Whitehouse, D. (2012). [Information Set Monte Carlo Tree Search](https://doi.org/10.1109/TCIAIG.2012.2200894). *IEEE Transactions on Computational Intelligence and AI in Games*.
- Zha, D. et al. (2021). [DouZero: Mastering DouDiZhu with Self-Play Deep Reinforcement Learning](https://arxiv.org/abs/2106.06135). *ICML*.
- Auer, P., Cesa-Bianchi, N. & Fischer, P. (2002). [Finite-time Analysis of the Multiarmed Bandit Problem](https://homes.di.unimi.it/~cesabian/Pubblicazioni/ml-02.pdf). *Machine Learning*.

## Acknowledgments

Thanks to **Ronan Guillou**, seasoned coinche player, for his advice on the game and for being the first tester — his good sense guided many UI decisions.
