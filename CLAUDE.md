# CLAUDE.md

## Build & Test Commands

```bash
cargo check                                    # Check compilation (both crates)
cargo test -p colver-core                      # Run all core tests
cargo test -p colver-core -- test_name         # Run a single test
cargo test -p colver-core --release            # Tests in release mode
cargo run -p colver-core --bin bench --release # Performance benchmark (~1.3M rollouts/sec)
cargo run -p colver-core --bin train_joint --features dmc_train --release -- --num-envs 256 --steps 35000000  # Joint bid+play training
cargo run -p colver-core --bin train_joint --features dmc_train --release -- --mode play-only --resume-bid models/bid_v2/bid_nn_final.safetensors --bid-hidden 512 --bid-layers 3 --num-envs 256 --steps 50000000 --eval-freq 1000000 --save-freq 2000000  # Triforge: play-only phase with bid_v2
./scripts/training/triforge.sh --cycles 3  # Full triforge: alternating bid/play training
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- --hidden 512 --layers 3 --steps 20000000 --pool-file data/deals/base_5M.bin --score-file data/deals/scores_isdd_5M.sc  # Standalone bid NN training (base pool + optional score layers)
RUSTFLAGS="-C target-cpu=native" cargo run -p colver-core --bin gen_pool --release -- -o data/deals/dd_pool.bin -n 1000000  # DD pool generation (no CUDA dep, ~244 deals/s)
cargo run -p colver-core --bin gen_bid_belief_data --release --features parallel -- --bid-model models/bid_v2/bid_nn_final.bin --bid-hidden 512 --deals 500000 --output data/belief/bid_belief_500k.bin  # Bid belief training data (COLVBB01, ~14M samples, ~65s)
CUDARC_CUDA_VERSION=13010 cargo build --release --bin playgen_gpu_server --features gpu_server && ./target/release/playgen_gpu_server --playgen models/playgen/playgen_v2_final.bin --port 8003  # Playgen GPU sidecar — IS-DD's world source
export COLVER_PLAYGEN_GPU_URL=http://localhost:8003          # required by any IS-DD agent (arena, web, scripts)
uv sync                                        # Build and install Python bindings
uv run python -m colver.web                    # Run web frontend → http://localhost:8000
```

**Cargo features:** `rand` (default), `parallel` (rayon), `dmc_train` (candle GPU training for DMC + bid NN + belief net), `gpu_server` (the playgen sidecar binary)

See [docs/](docs/) for all documentation. Key entry points:
- [docs/README.md](docs/README.md) — full doc index
- [docs/agents.md](docs/agents.md) — **the `Player` / `WorldSource` layer**: how bots are built and driven, and the bot-spec format
- [docs/training/overview.md](docs/training/overview.md) — training/eval commands
- [docs/arena_results.md](docs/arena_results.md) — global arena leaderboard (king metric)
- [docs/bid/](docs/bid/) — bidding strategies, NN bidders, reward studies, interpretability
- [docs/play/](docs/play/) — DD, IS-DD, DMC, IS-MCTS
- [docs/belief/](docs/belief/), [docs/data_gen/](docs/data_gen/)

## Architecture

Belote Contrée game engine optimized for millions of RL rollouts/sec. Rust core with PyO3 Python bindings.

**Workspace:** `colver-core` (pure Rust, zero deps by default) + `colver-py` (PyO3/numpy FFI) + `python/colver/web/` (FastAPI/WebSocket frontend)

**`colver-core/src/` module layout:**
- `agent/` — **`Player` trait + `AgentSpec` → `build(seat)`; the only place that knows how a seat plays.** mod (traits), spec (TOML), models (weight cache), isdd, dmc, bid, ismcts
- `worlds.rs` — `WorldSource` trait: sidecar (playgen on GPU, **default**), local playgen (CPU), constraint-uniform
- `game_loop.rs` — `play_deal` / `play_match` over `[Box<dyn Player>; 4]`
- `engine/` — card, state, bidding, trick, play, scoring, game, cfn (foundation, no external deps)
- `search/` — mcts, ismcts variants, is_dd, solver, determinize, rollout
- `bid/` — bid_eval (split into strategy files: heuristic, smart, roro, improved, parametric, petit_bide, moelleux), bid_obs, bid_net, bid_candle, dd_bid, maxi
- `dmc/` — dmc_net, dmc_obs, dmc_replay, dmc_env, dmc_candle, dmc_eval
- `belief/` — belief_net, belief_obs, belief_candle, card_beliefs (**load-bearing**: supplies IS-DD's hard constraints, despite the "deprecated" label it carried)
- `playgen/` — tokens (tokenizer v1/v2), model (candle transformer, dmc_train), infer (pure-Rust KV-cache inference, rand), analysis (read-only introspection)
- root — suit_perm, game_replay, joint_env, rule_player

All modules re-exported at crate root (`use colver_core::card` still works). Binaries in `src/bin/` (auto-discovered by Cargo). Scripts in `scripts/{training,analysis,export}/`.

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
- Scoring (FFB section 9.1): "points faits + demandés". Multiplier applies to **contract value only**, not base.
  - Normal réussi: card_pts + contrat + belote. Defense: their card_pts + belote.
  - Contré réussi: 160 (or 250 if capot réalisé) + contrat×2 + belote. Defense: 0.
  - Surcontré réussi: 160 (or 250 if capot réalisé) + contrat×3 + belote. Defense: 0.
  - Chute: defense gets 160 + contrat×mult + all belote. Preneurs: 0.
  - Capot = contrat à 250. Dix de der = 100 → 252 pts cartes.
- **BREAKING (2026-04-16):** two scoring rule changes. Any arena/training result from before this date must be re-run.
  1. **Surcoinche multiplier:** ×3 (was ×4). Affects surcontré réussi and chute.
  2. **Contré/surcontré scoring formula:** base is now 160 + contrat×mult (was 320/640 + contrat×mult). Capot is a regular contract at 250 (was flat 500/1000/2000).

### Terminologie (FR)

Vocabulaire à utiliser avec l'utilisateur (ne pas dire « chicane ») :
- **avoir une coupe** / **couper** / **couper à [couleur]** — ne pas/plus avoir de carte dans une couleur
- **coupe franche** — coupe présente dès la donne, avant même de commencer à jouer
- **grosses cartes** / **cartes à points** — les cartes qui valent des points (As, Dix, Roi, Dame, Valet) ; ne pas dire « honneurs »
- **avoir une longue** — avoir beaucoup de cartes dans une même couleur (ex. « une longue à cœur »)

### Performance-Critical Path

`play.rs::legal_plays()` is the hottest function — all bitwise, no allocations. Target: >1M rollouts/sec single-threaded.

## Key Subsystems (see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) for full details, [docs/play/](docs/play/) and [docs/bid/](docs/bid/) for per-component docs)

- **MCTS** (`search/mcts.rs`): Arena-based UCT, 1000 iters default, C=sqrt(2)
- **Smart IS-MCTS** (`search/smart_ismcts.rs` + `belief/card_beliefs.rs`): Belief-weighted IS-MCTS, ~+7.5% vs naive
- **DD Solver** (`search/solver.rs`): Alpha-beta with TT, PVS, killer/history heuristics. ~77ms/solve from full deal (4 suits ≈ 310ms), ~13.5ms mid-game. See [docs/play/dd_solver.md](docs/play/dd_solver.md).
  - **BREAKING (2026-07-23): `quick_tricks` removed — it returned wrong DD values (25% of `solve_for_trump` calls). All pre-2026-07-23 DD data is stale**, notably `data/deals/base_5M.bin` and every score layer derived from it. Details + the invariant that caught it: [docs/play/dd_solver.md](docs/play/dd_solver.md).
- **Pool generator** (`gen_pool` binary): Standalone DD pool generation, no CUDA dep. Uses `RUSTFLAGS="-C target-cpu=native"` + workspace `[profile.release] lto="fat", codegen-units=1` for 2.4× speedup. Checkpoints every 100k deals (resumable).
- **IS-DD** (`search/is_dd.rs` + `agent/isdd.rs`): Information Set DD — samples determinized worlds, solves each with DD, aggregates. **Hard constraints** (voids, trump ceiling, played cards) are facts and are always applied, with no flag. **Soft beliefs** (`use_soft_inference`, `use_nn_beliefs`) are **off by default**. `early_termination` is on by default. Worlds come from a `WorldSource` owned by `IsDdPlayer` — **playgen over the GPU sidecar by default**. `enrich_pool_isdd` generates play scores with IS-DD for training data. See [docs/play/is_dd.md](docs/play/is_dd.md).
  - **BREAKING (2026-07-24): the agent refactor.** World generation moved *inside* the agent (`worlds.rs`), so the arena and the web now run the same IS-DD. Consequences: (1) every IS-DD bot without an explicit `[worlds]` section now defaults to **sidecar playgen** where it previously sampled uniform, so **pre-2026-07-24 `matches.csv` rows for IS-DD bots are not comparable**; (2) an IS-DD bot needs `$COLVER_PLAYGEN_GPU_URL` or `worlds.url`, or it fails at construction — set `source = "uniform"` to opt out deliberately; (3) `IsDdSearch::set_injected_worlds` / `playgen_frac` / elephant memory are gone. See [docs/agents.md](docs/agents.md).
- **DMC Agent "DouDou35"** (`dmc/dmc_net.rs`): DouZero-style Q-network, 415→1024³→32 (legacy obs), pure Rust inference ~1ms. Supports `residual: bool` for skip connections (same weights, different forward). Superseded by **DouDou50** (411→1024³→32, canonical ResNet, trained 50M steps) as the default play model.
- **NN Bidder** (`bid/bid_net.rs`) + **bidding strategies** (`bid/bid_eval/`): Dueling DQN, hidden size auto-detected. Default is **Bid v6 ISDD** (`models/bid_v6_isdd_resume/bid_nn_final.bin`, 117-dim score-aware v3 obs). Full model zoo (v1→v6) and strategy list: [docs/bid/README.md](docs/bid/README.md).
- **Belief models** (`belief/`): `CardBeliefs` (heuristic, deprecated), `BeliefState` (soft weights, used by IS-DD), belief NN (`belief_v4_fix_v2.bin`, play) and bid belief NN (`bid_belief_v4.bin`, auctions). `belief_v3.bin` is **not usable** with NN bots. Eval binary: `eval_beliefs`. See [docs/belief/README.md](docs/belief/README.md).
- **Playgen GPU sidecar** (`playgen_gpu_server`, feature `gpu_server`): serves the playgen model over HTTP so agents get worlds ~50× faster than on CPU. **`worlds::SidecarWorldSource` is IS-DD's default world source**, so an IS-DD agent needs `$COLVER_PLAYGEN_GPU_URL` (or `[worlds] url`) or it refuses to build. Prod: systemd `playgen-gpu.service` on the moxxi host, `http://192.168.1.23:8003` — **keep its `--playgen` model aligned with the released one** (currently `playgen_v2_final.bin`; it silently ran an intermediate checkpoint for a day). See [docs/belief/playgen.md](docs/belief/playgen.md).
- **Playgen world sampler** (`playgen/`): causal transformer that continues a game autoregressively from the observer-visible prefix; rolling out reveals hidden hands = a determinized world from the learned posterior. Consumed by IS-DD through `worlds::SidecarWorldSource` (GPU, default) or `LocalPlaygenSource` (CPU), and by the web's analysis pages through `playgen::analysis::PlaygenAnalyst` / `colver.Analyst`. **v2** (COLVPG02, 10.7M params) also predicts auction actions through a 43-way bid head, enabling mid-auction deal sampling. `train_playgen [--v2]` / `export_playgen [--v2]`; inference auto-detects the format and runs on CPU or CUDA. See [docs/belief/playgen.md](docs/belief/playgen.md).
- **World-credibility benchmark** (`bench_world_cred`): compares world samplers (playgen / belief NN / uniform) by asking whether the reference policy would replay the observed hidden actions. `--bid-positions 100 --play-positions 100 --worlds 32 --seed 42`, ~1min30 on a 4090. See [docs/belief/playgen.md](docs/belief/playgen.md).
  - **BREAKING (2026-07-23): benchmark RNG fixed — all pre-fix cross-checkpoint numbers are void** (positions were drawn from the stream the samplers consumed). Within-run comparisons were never affected.
  - **Rule for any benchmark here**: never draw the questions from a stream the thing under test also consumes — generate all positions first, then answer them. Keep an untouched baseline as a control; it must stay bit-identical across runs. `bench_logp_cred.rs` and `bench_world_compress.rs` still have the old pattern.
- **Triforge Training** (`joint_env.rs` + `train_joint` binary): Iterative best-response training — alternates bid-only and play-only phases with frozen partner. `--mode play-only|bid-only|joint`. Play NN: ResNet Dueling DQN (411→1024³→32, skip connections on layers 1-2). Bid NN: Dueling DQN (114→512³→43, configurable layers). See [docs/play/experiments/triforge.md](docs/play/experiments/triforge.md).
  - **Weight formats:** training checkpoints (candle) use `.safetensors` — required for `--resume-bid`/`--resume-play`. Inference weights use `.bin` (raw f32) — used by `BidNet::load`/`DmcNet::load` and arena TOML `model` paths. Triforge saves both at each checkpoint.
  - **Resume gotcha:** `--resume-play`/`--resume-bid` reload weights only — NOT the step counter, replay buffer, or epsilon schedule. A naive resume injects ~25% random moves for millions of steps and degrades the policy. Override values in [docs/play/experiments/triforge.md](docs/play/experiments/triforge.md).
  - **Arena/eval:** obs_dim is auto-detected from weight-file size (411 canonical DouDou50 with residual vs 415 legacy DouDou35); use `residual = true` in TOML for triforge play models.
- **NN inference kernels** (`nn_kernels.rs`): shared `dot` / `linear` / `layer_norm` for the pure-Rust nets, 8 accumulator lanes + AVX2 dispatch, 5-6× on all three nets. **Any new inference net should use these rather than an inline loop.** `playgen/infer.rs` has its own equivalent `dot8`. Numbers: [docs/BENCH.md](docs/BENCH.md).
- **Suit Augmentation** (`suit_perm.rs`): 24 suit permutations for data augmentation. Functions for belief obs (V1/V2/V3), DMC obs (415-dim), bid obs (108-dim), actions, and masks. TR variants (`permute_dmc_obs_tr` / `augment_play_batch_tr`) exist but unused since canonical ordering eliminates the need.

### DD Oracle: Training Signal, Not a Player

DD solver values are a **training signal** (direction to optimize toward), never a substitute for the model's own policy during data collection. In Contrée, bidding is a communication game — players probe, signal holdings, and iteratively discover the best contract through dialogue. The DD oracle sees all 4 hands and knows the answer instantly, so it has no reason to communicate. Using oracle actions for data collection produces degenerate auctions (optimal bid → 3 passes → done) that teach the model nothing about the signaling dynamics it must learn.

**Rules for bid model training on DD pools:**
- The **model plays its own auctions** (ε-greedy on the model's policy). Oracle targets supervise the loss, but the model's own actions drive the auction trajectory.
- DD Q-values are an **approximation**: the solver assumes perfect play, but real opponents don't play perfectly. Treat DD targets as a useful direction, not ground truth.
- A single hand predicts only ~17% of DD outcome variance (R²). Most of the signal comes from bid history (partner/opponent communication) — which only exists if the model plays realistic auctions.

### Observation Layouts (for suit permutation / NN inputs)

**DMC play obs — legacy (415):** [0:32] hand, [32:160] trick 4×32, [160:256] played 3×32, [256:260] trump suit, [260:263] value/team/coinche, [263:275] voids 3×4, [275:279] scores, [279:351] bid history 12×6, [351:383] card trick idx, [383:415] card seq idx. Used by DouDou35 (legacy play model).

**Canonical obs critical:** When using 411-dim models for inference, you MUST convert legal masks via `cardset_to_canonical(mask, order)` and actions back via `card_to_physical(action, order)`. Without this, the model plays random legal moves. The PyO3 bridge (`action_dmc_with_stats`) and arena auto-detect obs_dim and branch accordingly.

**DMC play obs — canonical (411):** Fully canonical suit encoding: trump in slot 0, non-trump sorted by (card_count, rank_pattern) descending — `canonical_play_order(trump, initial_hand)`. No suit augmentation needed. [0:32] hand, [32:160] trick 4×32, [160:256] played 3×32, [256:259] value/team/coinche (no trump one-hot), [259:271] voids 3×4, [271:275] scores, [275:347] bid history 12×6, [347:379] card trick idx, [379:411] card seq idx. Used by DouDou50 (default play model) and joint training. `card_to_canonical`/`card_to_physical` convert between spaces; `current_player_order` computes the ordering from state+tracking.

**Bid obs (108):** [0:32] hand, [32:104] bid history 12×6, [104:108] position. Auction state (bid value, suit, coinche) removed — redundant with bid history.

**Replay buffers** (`dmc/dmc_replay.rs`): `PrioritizedReplayBuffer` is hardcoded to OBS_DIM=415/MASK=32. Use `FlexReplayBuffer` for other dims (e.g. joint training play: 411/32, bid: 114/43).

## Python Layer (`colver-py/` → `python/colver/`)

`Env` wraps GameState with IS-MCTS/DMC support. Built as `colver._colver`, re-exported from `colver.__init__`. See `python/colver/_colver.pyi` for type stubs.

**PyO3 rebuild:** `uv sync` may not recompile Rust changes. Use `touch colver-py/src/lib.rs && maturin develop --release` to force rebuild. The `.so` at `python/colver/_colver*.so` may be stale — check file timestamp if behavior doesn't match code.

## Web Frontend (`python/colver/web/`)

FastAPI + WebSocket + vanilla JS. Modes: Play (solo), Salon (multiplayer rooms), Watch, Analysis. Models auto-downloaded at startup (DMC 10MB, bid NN 421KB, belief net 2MB).

**Accounts** (`auth.py`): bcrypt passwords, DB-backed session cookies (`colver_session`, sha256-hashed, 30d). Games are linked to accounts (solo: `games.user_id`; multi: `game_players` per seat). SQLite schema migrations via `PRAGMA user_version` in `database.py` — append to `MIGRATIONS`, never edit past entries.

**Salon multiplayer** (`rooms.py`): in-memory rooms with 4-char join codes, bots fill empty seats, one driver task per room (humans awaited on a queue, bot moves in executor). All broadcast state is filtered to the viewer's hand and **rotated so the viewer is always display-seat 2 (South)** — the shared frontend table (`static/js/shared/table.js`, `GameTable` class, used by both solo play.js and salon.js) never handles physical seats. DB stores physical seats. Mid-game states must never include `cfn` or other seats' `legal_actions` (information leaks). Reconnection: seats are account-bound; any `room_*` message rebinds the member's socket.

**Annonces page** (`views/annonces.js`): BidNet Q-values + Oracle DD table + DouDou simulation table. Oracle shows raw success % per suit×threshold. DouDou table uses Wilson score lower bound (z=1.645) for color thresholds (green/gold/red) and scales font size by observation count (0.65rem at 1 obs → 0.85rem at 20+) so small-sample cells appear visually less prominent than well-sampled ones.

**Mobile (≤600px):** Play view hides N/E/W seats, shows only trick area + South hand. South hand spans full viewport width with dynamically computed card overlap (JS in `play.js` sets `--card-overlap` based on card count and available width). Card sizes use CSS custom properties (`--card-w`) — note `#play-table` overrides `:root` values, so mobile overrides must target `#play-table` specifically. Header is 61px on mobile (not 46px as on desktop).

## Arena: Bot Comparison Framework

Systematic head-to-head and round-robin evaluation of bot architectures on 2000-point matches. Bots are TOML configs — no recompilation needed to test new combinations.

**Directory structure:** `arena/bots/*.toml` (bot definitions), `arena/results/matches.csv` (persistent results). Binary: `colver-core/src/bin/arena.rs`.

```bash
cargo run --bin arena --release -- list                                          # List all bots
cargo run --bin arena --release -- h2h bot_a bot_b --matches 200                 # Head-to-head (200×2 with duplicate matching)
cargo run --bin arena --release -- round-robin --matches 100                     # Full round-robin
cargo run --bin arena --release -- round-robin --matches 50 --bots a,b,c         # Subset round-robin
cargo run --bin arena --release -- results                                       # Leaderboard from CSV
cargo run --bin arena --release -- results --bot nn_dmc35                        # Filter by bot
```

**Bot TOML format** (`arena/bots/<name>.toml`) — parsed by `AgentSpec`, used identically by the arena, the web and `colver.Agent`. Full reference: [docs/agents.md](docs/agents.md).
```toml
[bid]
strategy = "nn"                    # heuristic|improved|improved_v2|improved_v3|smart|roro|maxi|petit_bide|moelleux|nn
model = "models/bid_nn_final.bin"  # required if strategy = "nn"
hidden = 512                       # hidden-size hint (auto-detected from the file when possible)
score_aware = true                 # endgame adjustments for nets that can't see the match score

[play]
method = "isdd"                    # isdd|dmc|dmc_then_isdd|ismcts|smart_ismcts|oracle|oracle_dd|heuristic|rule
model = "models/doudou50.bin"      # required for dmc / dmc_then_isdd
residual = true                    # skip connections for DouDou50 / triforge models
time_ms = 1000                     # per-move budget; 0 = count mode
determinizations = 240             # used when time_ms = 0
switch_at = 5                      # dmc_then_isdd: trick at which IS-DD takes over

[worlds]                           # IS-DD only; defaults to the sidecar
source = "sidecar"                 # sidecar | playgen (CPU) | uniform
url = "http://192.168.1.23:8003"   # or $COLVER_PLAYGEN_GPU_URL
fallback = "strict"                # strict = error out; uniform = degrade and say so

[belief]                           # optional
model = "models/belief_v4_fix_v2.bin"
```
Legacy keys (`is_dd`/`smart_is_dd`/`dmc_then_dd` method names, `playgen_model`, `use_hard_constraints`) still parse.

**Options:** `--matches N` (per direction, default 100), `--threads N` (default auto), `--seed N` (default 42). Each H2H runs both directions (duplicate matching) for variance reduction.

**Reference bots:** `v6_isdd_75M_belief` (Bid v6+SmartIsDd+Belief, **#1**), `v6_isdd_75M_isdd` (Bid v6+SmartIsDd, #2), `v6_isdd_75M` (Bid v6+DouDou50, fast champion), `v5_isdd_25M` (previous bid champion), `nn_v2_isdd` (Bid v2+SmartIsDd+Belief), `nn_v2_dmc50` (Bid v2+DouDou50, fast baseline), `nn_v2_dmc35` (Bid v2+DouDou35).

**Post-solver-fix round-robin (2026-07-23, 100 matches/direction, `--no-save` so NOT in `matches.csv`):** `v6_isdd_75M_belief` 60.9% / +209 · `v6_isdd_75M_isdd` 55.1% / +69 · `v6_isdd_75M` 50.2% / +1 · `nn_v2_isdd` 47.7% · `v5_isdd_25M_isdd` 46.6% · `nn_v2_dmc50` 39.5%. Dedicated h2h: `v6_isdd_75M_belief` beats `v6_isdd_75M_isdd` **54.8% / +111** over 1000 matches — belief_v3 clearly contributes on v6. Do **not** read this against the 49.6% from 2026-04-26: that run predates the `use_nn_beliefs` fix (2026-07-21), so the belief net was loaded but never consulted. The +5pp is consistent with the 2026-07-22 finding that a properly-consulted belief net is worth +3–6pp; the solver fix's own share is unmeasured. Pre-2026-07-23 rows in `matches.csv` are stale for any bot using `is_dd`/`smart_is_dd`/`oracle`/`dmc_then_dd` — 52% of the file.

**Cost of the solver bug in playing strength is small:** same bot with vs without the buggy bound, 1600 matches, **50.9%** — not distinguishable from zero. The defect matters for *data* (DD training targets, oracle trust) far more than for direct play.

**Apples-to-apples comparisons:** v3 IS-DD bots (`bid_v3_*_isdd`) have no belief net — compare against `nn_v2_isdd_no_belief`, not `nn_v2_isdd`, to isolate the bidder effect from the belief-net effect.

**CSV format:** Columns include `bid_a,play_a,bid_b,play_b` labels. Parser auto-detects old (11-col) and new (15-col) formats.

**Iteration workflow:** Create a new `.toml` in `arena/bots/`, run `h2h` against champion, check `results`, iterate. No recompilation between experiments.

## Publishing & Deployment

**PyPI:** **bump `version` in `pyproject.toml` and commit it first**, then push the matching `v*` tag → CI builds manylinux/macOS/Windows wheels via maturin → publishes automatically (trusted publishing). Wheels are `abi3` (one per platform, Python 3.10+).

Tagging without the bump is what broke v0.3.0, v0.3.1, v0.4.0, v0.5.0 and v0.8.0 — the build reused the previous version's filenames and PyPI rejected them with `400 File already exists`. The `check-version` job now blocks that in ~10s. `colver.__version__` derives from package metadata, so it never needs a manual bump.

**Docker:** `docker build -t colver . && docker run -p 8000:8000 colver`. Cross-builds for ARM64.

## Data Directory Layout

Canonical layout: a single base pool (COLVDD01) plus per-method score layers (COLVSC01) loaded independently by `train_bid_nn` via `--score-file` (repeatable). Score files carry `offset` + `count`, so partial-coverage layers compose: load a 1M layer starting at offset 0, then a 1M layer at offset 1M, and the trainer sees 2M.

```
data/
  deals/              Base pools + score layers (modern layout)
    base_5M.bin           5M pre-solved DD deals (COLVDD01, 105MB)
    scores_dmc_5M.sc      DMC real-pts scores on all 5M (COLVSC01, 20MB)
    scores_isdd_5M.sc     IS-DD scores on all 5M, 20ms × 20 dets (COLVSC01, 20MB)
    archive/              Legacy COLVDR01 enriched pools + old experiments + historical logs
  belief/             Belief net training data
    belief_train_500k.bin  (COLVBL01, 20GB, play-phase samples)
    bid_belief_500k.bin    (COLVBB01, 6.3GB, 14.2M bid-phase samples from bid_v2)
  training/           Game replay / value data
    games_500k.bin       500K full game replays (28MB)
    value_train.bin      Value net training data (171MB)
  distill/            Bid distillation analysis
    bid_distill.csv      7.2M rows of bid NN Q-values + features (1GB)
    bid_distill_analysis.log
    bid_distill_console.log
  shap/               SHAP analysis plots
  colver.db           SQLite (web frontend)
```
