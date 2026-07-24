# Architecture Details

Detailed documentation for each subsystem. See [CLAUDE.md](../CLAUDE.md) for overview.

## Agents (`agent/`, `worlds.rs`, `game_loop.rs`) — start here

Every caller that plays a game — the arena, the web, any script — builds
`Box<dyn Player>` from an `AgentSpec` and drives it with `game_loop`. The
strategy modules below are the *implementations* behind that trait; you rarely
touch them directly. See **[agents.md](agents.md)**.

## MCTS Agent (`mcts.rs`, feature `rand`)

Perfect-information MCTS using UCT (UCB1 for trees). Arena-based tree with `Node`s and `Edge`s in flat `Vec`s for cache-friendliness. `MctsSearch` is reusable across searches (arenas are cleared between calls). Default: 1000 iterations, `C = sqrt(2)`.

**API:** `MctsSearch::search(&mut self, state, config, rng) -> u8` returns best action. `search_with_stats(...)` returns `SearchResult` with visit counts. `search_with_nn(state, config, value_net, rng)` uses NN leaf evaluation instead of rollouts (feature `nn`). `mcts_search(state, config, rng)` is a convenience one-shot wrapper.

**Algorithm:** Selection (UCB1 descent) → Expansion (enumerate legal actions as edges, create child node) → Simulation (`rollout_random` or NN eval) → Backpropagation. Rewards scaled by `1/2000` for rollouts (NN outputs already in [0,1], no scaling). Best action = most-visited root child.

## Smart IS-MCTS Agent (`smart_ismcts.rs` + `card_beliefs.rs`, feature `rand`)

Belief-weighted Information Set MCTS. Maintains a `CardBeliefs` model (`[[f32; 32]; 4]` weight matrix) updated after every action using hard constraints (voids, trump ceiling, played cards) and soft inference (bidding signals, play patterns). `determinize_weighted()` samples opponent hands biased by beliefs. See [play/smart_ismcts.md](play/smart_ismcts.md) for design.

**API:** `SmartIsMctsSearch::new()`, `init_deal(state, observer, use_soft)`, `record_action(state_before, player, action)`, `search(state, config, rng) -> u8`. Each player needs its own instance; both must observe all actions.

## Naive IS-MCTS Agent (`naive_ismcts.rs`, feature `rand`)

Ensemble determinization without beliefs. Samples D determinized worlds (uniform, void-aware), runs MCTS on each, aggregates root visit counts. Default: 20 determinizations × 50 iters.

## Card Play Strategies

- **Random** (`rollout.rs: rollout_random`) — Uniform random legal moves. ~1.3M rollouts/sec.
- **Heuristic** (`rollout.rs: heuristic_play_action`) — Deterministic, sees all hands. Safe leads, partner feeding, min-winning-card, cheapest trump cut. ~769K full-deal rollouts/sec.
- **Smart IS-MCTS** — Belief-weighted. ~+7.5% win rate vs Naive IS-MCTS. `search_parallel()` behind `parallel` feature, `search_with_nn()` behind `nn` feature.

## Double-Dummy Solver (`solver.rs`)

Alpha-beta solver for perfect-information Belote. No feature gate — zero external dependencies.

**API:**
- `solver::solve(state) -> [u8; 2]` — returns `[ns_points, ew_points]`
- `solver::solve_for_trump(hands, dealer, trump) -> [u8; 2]`
- `solver::solve_best_card(state) -> u8` — optimal card for current player
- `GameState::setup_dd(dealer, hands, trump)` — creates play-phase state for DD

**Performance:** ~13.5ms/solve avg, median ~7ms, P90 ~31ms. Invariant: `ns + ew == 162` (or 252 for capot).

**Techniques:** Alpha-beta fail-soft, TT (256K entries, 2MB, L2-friendly), PVS, killer moves (2/ply), history heuristic (depth²), card equivalence pruning, quick tricks bounds, forced-move optimization. Move ordering: hash move → killers → history + static score.

## Bidding Strategies (`bid_eval.rs`)

Six fixed bidding functions (`BidFunction` enum) plus configurable `parametric_bid(state, &BidParams)`.

**Hand evaluation:** `evaluate_for_trump(hand, suit) -> u16` — Trump honors (J=8, 9=6, A=4, 10=3, K=1, Q=1), trump length bonus, side aces (+3), voids (+3), singletons (+1). Range 0–35.

| Strategy | Description | Key traits |
|----------|-------------|------------|
| `BidADd` | NN Dueling DQN, DD-trained | **Default for web.** 70–76% vs improved_v2 |
| `Improved` | Tournament-winning balanced | Quality gate, caps 120/120/130, coinches |
| `Heuristic` | Aggressive score-based | No cap, no quality gate, ~50% take rate |
| `Smart` | Conservative J/9 conventions | ~10-13% take rate, mostly historical |
| `Roro` | Expert convention-based | Position-aware, Théorème 3 coinche |
| `Maxi` | Expert + structured card play | Cases A/B/C/D classification |
| `BidParams` | Configurable for sweeps | 6 presets: ultra_conservative → very_aggressive |

### DD-Based Bidding (`dd_bid.rs`, feature `rand`)

`DdBidder` uses DD solver + determinization to estimate expected points per trump suit. ~300ms/opening. Pre-filters candidates, DD-solves each determinization, maps expected points to bid value.

## NN Bidding (`bid_net.rs`, `bid_obs.rs`, `bid_candle.rs`)

Two generations of bid NN:

- **Bid a Doudou** (v1): 114→256→256→43, trained with DouZero self-play. ~421KB weights. Observation: hand (32) + bid history (72) + position (4) + auction state (6) = 114 floats.
- **Bid a Dede** (v2, default): 108→512→512→512→43, trained with DD solver + 24x suit augmentation. Observation: hand (32) + bid history (72) + position (4) = 108 floats (auction state removed as redundant).

Both use Dueling DQN with LayerNorm. ~0.1ms/eval.

**Rust inference:** `BidNet::load(path)` auto-detects hidden size (tries 256, 512, 1024). `BidNet::evaluate(&mut self, obs) -> [f32; 43]` / `BidNet::best_action(&mut self, obs, legal_mask) -> (u8, Vec<(u8, f32)>)`.

**Training infra:** `DealPool` (pre-solved deals, `COLVDD01` format), `BidTrainingEnv`, `VecBidEnv`, `BidReplayBuffer` (SumTree PER), `BiddingTrainer` (candle Dueling DQN). Opponent diversity annealing 40%→15% non-self-play.

## DMC Q-Network Agent (`dmc_net.rs`, `dmc_obs.rs`)

DouZero-style Deep Monte-Carlo. Q-network picks card plays with single forward pass — no search tree.

Two generations:

- **DouDou35** (legacy): 415→1024→1024→1024→32 MLP with LayerNorm, Dueling DQN (~2.6M params). Trained 35M steps with DouZero self-play. Uses legacy 415-dim observation.
- **DouDou50** (default): 411→1024→1024→1024→32 ResNet Dueling DQN with skip connections (~2.6M params). Trained 50M steps with Bid a Dede frozen (triforge play-only). Uses canonical 411-dim observation.

**Observation — legacy (415 floats, player-relative):** hand (32) + current trick per-player (128) + past tricks per-player (96) + contract (7) + void tracking (12) + scoring context (4) + bid history (72) + card trick index (32) + card sequence index (32).

**Observation — canonical (411 floats):** Same layout but trump always in suit slot 0, non-trump suits canonically sorted. No trump one-hot (3 fewer dims). No suit augmentation needed.

**Rust inference:** `DmcNet::load(path)` auto-detects obs_dim from file size. `DmcNet::evaluate(&mut self, obs) -> [f32; 32]`. ~1ms/eval, zero deps.

**Rust training** (`dmc_candle.rs`, `dmc_env.rs`, `dmc_replay.rs`, feature `dmc_train`): ~474 steps/s on 4090. NN bid support via `--bid-model`. PER buffer, opponent pool, IS-DD eval.

## Belief Network (`belief_net.rs`, `belief_obs.rs`, `belief_candle.rs`)

Predicts which player holds each unknown card from observable game state. Replaces/augments heuristic `CardBeliefs` in IS-DD search.

**Standard architecture (BeliefQNet):** obs_dim→512→512→128 (LN+ReLU). Output: 32 cards × 4 player-relative slots (logits). `belief_to_weights()` applies per-card softmax, zeros observer slot, renormalizes. ~1.9MB, ~0.1ms/eval.

**Architecture variants** (selected via `--variant` in training):
- **`standard`**: 2-layer MLP, ~480K params
- **`var_mlp`**: Variable-depth MLP, `--num-layers` and `--hidden`
- **`suit_shared`**: Per-suit weight-shared, suit-equivariant (no augmentation needed), ~60K params
- **`cross_attn`**: Global encoder + card embeddings + 4-head self-attention, ~120K params
- **`aux_loss`**: Standard trunk + trick_winner_head + void_head, auxiliary losses decay

### Observation Versions

**V1** (330 floats): own hand + per-player played cards (128) + trick/position indices + bid history + contract + voids + scoring + position + lead suit + trick progress.

**V2** (304 floats): Leaner with hard constraints. own hand (32) + played-by (32) + trick/position indices (64) + bid history (72) + contract (8) + hard constraints (96: 3 players × 32 cards, impossible=1.0).

**V3** (380 floats): V2 + per-card lead suit (32) + per-trick winner (32) + suit failure counts (12).

**Suit augmentation** (`suit_perm.rs`): 24 permutations for training data diversity. `--augment` flag.

**Training data formats:**
- `COLVBL01`: Pre-extracted (obs, target, mask). V1 only. ~20GB for 500K games.
- `COLVGM01` (preferred): Compact game replays (~62 bytes/game). Re-extract on demand for V1/V2/V3. ~28MB for 500K games.

**Integration with IS-DD** (`is_dd.rs`): `IsDdConfig::use_nn_beliefs` flag. Hybrid mode: NN predictions filtered by heuristic hard constraints.

**Rust inference:** `BeliefNet::load(path)` auto-detects V1/V2/V3 and 3-class/4-class from file size. `BeliefNet::evaluate(&mut self, obs) -> [f32; 128]`.

### Per-trick accuracy (100k games, 38.3M card predictions)

Évaluation sur 100k parties rejouées (data/games\_500k.bin). À chaque position de jeu, le modèle prédit pour chaque carte inconnue quel joueur la détient (3 candidats). V2 = 4-class (304-dim), V3 = 3-class + count regularization (304-dim).

| Trick | V2 Acc | V3 Acc | Δ Acc | V2 CE | V3 CE | Δ CE |
|-------|--------|--------|-------|-------|-------|------|
| 0 | 47.1% | 47.2% | +0.1 | 0.9848 | 0.9840 | -0.0008 |
| 1 | 50.0% | 50.2% | +0.2 | 0.9450 | 0.9432 | -0.0018 |
| 2 | 53.2% | 53.5% | +0.3 | 0.8989 | 0.8959 | -0.0030 |
| 3 | 56.8% | 57.1% | +0.3 | 0.8447 | 0.8402 | -0.0045 |
| 4 | 60.7% | 61.2% | +0.5 | 0.7808 | 0.7744 | -0.0064 |
| 5 | 65.3% | 65.9% | +0.6 | 0.7016 | 0.6925 | -0.0091 |
| 6 | 71.3% | 72.2% | +0.9 | 0.5873 | 0.5727 | -0.0146 |
| 7 | 81.1% | 82.2% | +1.1 | 0.3726 | 0.3563 | -0.0163 |
| **Total** | **54.68%** | **55.00%** | **+0.32** | | | |

V3 est systématiquement meilleur à chaque pli, en accuracy comme en cross-entropy. L'écart se creuse au fil du jeu : +0.1% au pli 0 → +1.1% au pli 7, et -0.0008 CE → -0.016 CE. La régularisation de comptage aide surtout en fin de partie quand le modèle doit respecter les contraintes combinatoires (chaque joueur a exactement N cartes restantes). Baseline random = 33.3%.

```bash
cargo run -p colver-core --bin belief_eval --release -- \
  --model models/belief_v3.bin --replays data/training/games_500k.bin \
  --mode per_trick --games 100000
```

## Removed subsystems

- **NN value function** (`features.rs`, `value_net.rs`, feature `nn`) — MCTS with
  a learned value function is the wrong architecture for a card game: each node
  costs a network eval, so the tree stays tiny. A DouZero-style direct
  Q-network (DMC) does the same job orders of magnitude faster. Removed
  2026-07-24.
- **BisDd** (`bis_dd.rs`) — a unified DD-based bid+play agent. Its bid-inference
  heuristic rejected reality 72% of the time against NN bidders
  ([belief/bis_dd.md](belief/bis_dd.md)); the bid belief net replaced it.
- **Elephant memory** (`elephant.rs`) — particle filter over past
  determinizations. Off by default everywhere, never won in the arena.
- **Single-tree IS-MCTS** (`single_tree_ismcts.rs`) — superseded by IS-DD.

All recoverable from git history.

## Python Layer (`colver-py/` → `python/colver/`)

`Env` wraps GameState with IS-MCTS/DMC support. `StdRng` (not `ThreadRng`) for PyO3 `Send`.

**Observation v4** (415 floats): same as DMC obs. `get_observation()` returns full vector.

**Public API**: `Env`, `__version__`, `download_model()`, `model_path()`, `bid_model_path()`, `download_bid_model()`, `belief_model_path()`, `download_belief_model()`. See `python/colver/_colver.pyi` for type stubs.

**Web frontend API** (on `Env`): `get_hands()`, `get_current_trick()`, `get_contract()`, `get_points()`, `get_tricks_won()`, `get_dealer()`, `get_trick_lead()`, `get_played_cards()`, `phase()`, `current_player()`, `is_terminal()`, `legal_actions()`. Static: `Env.card_name(idx)`, `Env.action_name(action, phase)`, `Env.deal_with_hands(dealer, hands)`.

## Web Frontend (`python/colver/web/`)

FastAPI + WebSocket + vanilla JS. Three modes: Play (human vs AI), Watch (spectate), Analysis (custom position + MCTS).

**Package layout:**
- `server.py` — FastAPI app, WebSocket handler, auto-downloads models at startup
- `game_manager.py` — `PlaySession`, `WatchSession`, `ReplaySession`, `AnalysisSession`
- `database.py` — SQLite game history (`~/.local/share/colver/colver.db`)
- `static/` — Frontend files; `cards/` — 67 SVG playing cards

## Docker Deployment

Multi-stage Dockerfile: uv builder + slim runtime. No torch — all inference is pure Rust. Three models auto-downloaded at startup. Cross-builds for ARM64 via `docker buildx`.
