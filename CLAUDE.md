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
  - **Windowed solve** (2026-07-26): `solve_windowed_reuse_tt` / `solve_for_trump_windowed` take an explicit `[alpha, beta]` instead of the hardcoded `[0, 252]`, for batches of near-identical positions (the sampled worlds of one hand) where the running mean seeds a narrow window. **Fail-soft: the result is exact only when `alpha < v < beta`** — outside that range it is a bound and the caller must re-search wider. Treating a bound as a value is exactly the `quick_tricks` defect. Benches: `bench_solve_window` (asserts every windowed result against the full window), `bench_tt_size`.
  - **BREAKING (2026-07-23): `quick_tricks` removed — it returned wrong DD values (25% of `solve_for_trump` calls). All pre-2026-07-23 DD data is stale**, notably `data/deals/base_5M.bin` and every score layer derived from it. Details + the invariant that caught it: [docs/play/dd_solver.md](docs/play/dd_solver.md).
- **Pool generator** (`gen_pool` binary): Standalone DD pool generation, no CUDA dep. Uses `RUSTFLAGS="-C target-cpu=native"` + workspace `[profile.release] lto="fat", codegen-units=1` for 2.4× speedup. Checkpoints every 100k deals (resumable).
- **IS-DD** (`search/is_dd.rs` + `agent/isdd.rs`): Information Set DD — samples determinized worlds, solves each with DD, aggregates. **Hard constraints** (voids, trump ceiling, played cards) are facts and are always applied, with no flag. **Soft beliefs** (`use_soft_inference`, `use_nn_beliefs`) are **off by default**. `early_termination` is on by default. Worlds come from a `WorldSource` owned by `IsDdPlayer` — **playgen over the GPU sidecar by default**. `enrich_pool_isdd` generates play scores with IS-DD for training data. See [docs/play/is_dd.md](docs/play/is_dd.md).
  - **BREAKING (2026-07-24): the agent refactor.** World generation moved *inside* the agent (`worlds.rs`), so the arena and the web now run the same IS-DD. Consequences: (1) every IS-DD bot without an explicit `[worlds]` section now defaults to **sidecar playgen** where it previously sampled uniform, so **pre-2026-07-24 `matches.csv` rows for IS-DD bots are not comparable**; (2) an IS-DD bot needs `$COLVER_PLAYGEN_GPU_URL` or `worlds.url`, or it fails at construction — set `source = "uniform"` to opt out deliberately; (3) `IsDdSearch::set_injected_worlds` / `playgen_frac` / elephant memory are gone. See [docs/agents.md](docs/agents.md).
- **DMC Agent "DouDou35"** (`dmc/dmc_net.rs`): DouZero-style Q-network, 415→1024³→32 (legacy obs), pure Rust inference ~1ms. Supports `residual: bool` for skip connections (same weights, different forward). Superseded by **DouDou50** (411→1024³→32, canonical ResNet, trained 50M steps) as the default play model.
- **NN Bidder** (`bid/bid_net.rs`) + **bidding strategies** (`bid/bid_eval/`): Dueling DQN, hidden size auto-detected. Default is **Bid v6 ISDD** (`models/bid_v6_isdd_resume/bid_nn_final.bin`, 117-dim score-aware v3 obs). Full model zoo (v1→v6) and strategy list: [docs/bid/README.md](docs/bid/README.md).
  - **`strategy = "playgen"`** (`PlaygenBidPolicy` in `agent/bid.rs`, *not* in `bid_eval/`): playgen v2's own 43-way auction head as a bidder, masked by legal actions, argmax or softmax on `temperature`. It needs the whole visible prefix, so it tracks the deal via `init_deal` / `observe` like a world source — hence `build_bid` takes the seat. Falls back to `ImprovedV2` when the sampler can't answer. A behaviour clone of v6 and not score-aware (its corpus is standalone deals at 0-0), yet 48.2% h2h vs v6 over 3000 matches. Bot: `arena/bots/playgen_bid.toml`.
  - **Negative result — read before retrying auction-conditioned labels:** conditioning bid labels on the auction prefix narrows the posterior 21.5% *around the right place*, but both distilled bidders lose the arena (47.3%, then 43-44% vs v6). The pipeline is reusable; the target definition is what's wrong. Binaries (`gen_bid_labels`, `train_bid_distill`, `train_bid_cont`, `gen_dd_calibration`, `bench_label_variance`, `bench_bid_label_cond`, `bench_suit_prefilter`, `bench_world_richness`) and two false alarms that each looked conclusive: [docs/bid/experiments/auction_conditioned_labels.md](docs/bid/experiments/auction_conditioned_labels.md).
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

**Pacing modes** (`pacing.py`): a single host/player choice, `standard` or `rapide`, bundling the display tempo *and* the bot on all four AI seats — there is no per-seat opponent/partner choice and no pause slider. `standard` = Dédé (IS-DD, 1200 ms), `rapide` = DouDou50. The coupling is load-bearing, not cosmetic: an IS-DD search costs real wall-clock per move, so a fast tempo is only honest behind a bot that answers instantly. Card and trick pauses taper linearly over the 8 tricks (`standard` floors at 0.9 s / 1.2 s so a human can still read trick 8); bids get a flat, shorter pause. Measured on a full deal: **~41 s standard, ~14 s rapide**, same in solo and salon.
- **The pause belongs to the position *preceding* a move, and the bot thinks inside it, not on top of it.** Both drivers compute the move while the previous position is still on screen, then sleep only the remainder (`pacing.hold(target, elapsed)`). Get this wrong and Dédé's 1.2 s stacks onto every pause, running standard at nearly double its advertised tempo. Consequence: `_run_ai_turns` has **no trailing sleep** and its caller must not pause before handing over.
- Solo `_run_ai_turns` runs `play_ai_turn` in `asyncio.to_thread` (the Rust search releases the GIL); the salon already used an executor.
- `pacing.resolve(mode, doudou_available)` returns `(bot, think_ms, degraded)`. When DouDou's weights are missing, `rapide` seats Dédé at 400 ms and **says so** (`mode_degraded` → a note in the UI). `Room.bot_type` is derived from the mode, so seat labels and `games.agents` stay correct. Elo needs nothing: bots are rated as separate entities (`("bot", name)`), so mixing modes never pollutes a human's rating.
- The Oracle is deliberately **not** offered in solo play — it sees all 4 hands. It lives in Regarder and Rejouer only.

**Passe forcé:** when passing is the only legal bid (our bid coinched and the partner declined the surcoinche — you cannot surcoincher your own team; or over a partner's capot), there is no decision, so both drivers play it: `_run_ai_turns` (solo, flags the echo `auto: true` so the client shows it) and `Room._drive` (salon). Predicate: `game_manager.only_pass_is_legal()`. The frontend hides the bid panel in that state instead of offering a lone "Passer".

**Accounts** (`auth.py`): bcrypt passwords, DB-backed session cookies (`colver_session`, sha256-hashed, 30d). Games are linked to accounts (solo: `games.user_id`; multi: `game_players` per seat). SQLite schema migrations via `PRAGMA user_version` in `database.py` — append to `MIGRATIONS`, never edit past entries.

**Salon multiplayer** (`rooms.py`): in-memory rooms with 4-char join codes, bots fill empty seats, one driver task per room (humans awaited on a queue, bot moves in executor). All broadcast state is filtered to the viewer's hand and **rotated so the viewer is always display-seat 2 (South)** — the shared frontend table (`static/js/shared/table.js`, `GameTable` class, used by both solo play.js and salon.js) never handles physical seats. DB stores physical seats. Mid-game states must never include `cfn` or other seats' `legal_actions` (information leaks). Reconnection: seats are account-bound; any `room_*` message rebinds the member's socket.

**Annonces page** (`views/annonces.js`): BidNet Q-values + **Jeu parfait** (Oracle DD) + **Jeu réel** (DouDou50). Les deux boxes portent un lien `?` vers `/about?s=<section>` — **pas une ancre `#`**, que le routeur avalerait comme URL héritée. Les explications vivent dans `views/about.js`, plus sur la page. Comptes de simulations figés côté client (`ORACLE_SIMS = 200`, `REAL_SIMS = 1000`, plus de champ de saisie) : un solve DD coûte ~50× une donne jouée. Côté serveur, `annonces_sim` prend `oracle_sims` / `doudou_sims` et **partage un seul pool de mondes** (`world_total = max`) — l'Oracle résout les premiers, Dédé les joue tous, donc les deux tableaux décrivent le même échantillon. La génération de mondes survit à la boucle Oracle (drainée avant la phase 2) au lieu d'être annulée avec elle. La box Jeu réel s'ouvre sur un **chiffre-phare** (`#doudou-headline`) : % de donnes où Nord-Sud marque plus qu'Est-Ouest, + espérance de points. Dénominateur = toutes les sims terminées, **donnes passées comprises** (comptées nulles, `deal_draws`) — sinon le taux porterait sur un sous-ensemble différent du reste du panneau. Oracle shows raw success % per suit×threshold. DouDou table uses Wilson score lower bound (z=1.645) for color thresholds (green/gold/red) and scales font size by observation count (0.65rem at 1 obs → 0.85rem at 20+) so small-sample cells appear visually less prominent than well-sampled ones.
- **Onglets d'analyse** (`#annonces-tabs`) : un onglet = une annonce analysée sur la main courante ; « Analyser une autre annonce » en ouvre un nouveau au lieu d'écraser le précédent, et redemander une annonce déjà ouverte réactive son onglet. Le **Jeu parfait est partagé** par tous les onglets (l'Oracle résout les quatre couleurs, il ne dépend pas de l'annonce) — seul le Jeu réel est stocké par onglet ; changer d'onglet ne déplace que la case surlignée du bandeau. Une seule simulation tourne à la fois côté serveur (`_cancel_sim_task`) : ouvrir un onglet interrompt celle en cours, qui passe en `partial`, garde son résultat et propose « Relancer ». `annonces_sim` / `annonces_doudou` **renvoient le `req_id`** envoyé par le client (= l'id d'onglet) : sans lui, les derniers messages d'une simulation annulée atterriraient dans l'onglet suivant. Évaluer une nouvelle main remet la pile d'onglets à un seul.
- **Mains analysées** (`#annonces-saved`, localStorage `colver:annonces:saved`) : barre latérale gauche, dépliée par défaut ≥1400px, réduite à un rail ailleurs (état retenu dans `colver:annonces:sidebar`), bandeau horizontal scrollable sous 1024px. Une entrée = main **et** enchères précédentes — la même main après « 100♥ » n'est pas la même question, la clé de déduplication porte donc sur les deux. Alimentée à chaque évaluation ; cliquer une entrée recharge la situation et relance l'analyse. Les miniatures n'utilisent pas la classe `hand` (elle impose la hauteur d'une carte pleine), et tous ces boutons transparents doivent redéclarer leur fond au survol — `base.css` repeint tout `<button>:hover` en accent.

**Analyse du jeu de la carte** (`/analyse/jeu`, `views/analyse-jeu.js` + `card_analysis.py`, WS `card_analysis`): une ligne par carte jouable à une position. Design complet et pièges : [docs/web_analyse_jeu.md](docs/web_analyse_jeu.md).
- **État = CFN complet 4 sections + index d'action** (`?cfn=…&i=…`). Le CFN **cœur 3 sections ne suffit pas** : en phase de jeu `cfn.rs::format_contract` n'émet que le contrat résolu (`160hNS`) et l'enchère disparaît, alors que l'obs de jeu porte un historique d'enchères 12×6 et que playgen tokenise l'auction. Toujours passer par `game_notation.parse_full_cfn`. On arrive ici depuis Rejouer (`_gameCfn` + l'index du coup), qui n'a donc rien à recalculer.
- **Deux Oracles, jamais fusionnés.** (1) le *vrai monde* : un solve sur la donne réelle — exact, et déjà ce que Rejouer montre ; (2) les *mondes de l'information set* : mondes échantillonnés depuis ce que le siège pouvait savoir, chacun résolu. Seul (2) répond à « était-ce un bon choix ». Une carte deuxième dans la vraie donne mais meilleure dans 70 % des mondes était un bon coup contre de la malchance.
- **Lignes = `legal_actions()`, pas `legal_actions_reduced()`.** La réduction rend un représentant par classe **sans dire à quelle classe une carte appartient**, et les bots choisissent dans l'ensemble complet : un bot répondant 8♠ quand 9♠ a été gardé n'a pas de ligne, et son badge disparaît en silence. `reduced` ne sert plus qu'au test « y a-t-il une décision ? » (`len(reduced) < 2` = forcée) et au marqueur `≡`. Un solve couvre toutes les lignes, donc la complétude est gratuite côté Oracle ; le budget de déroulements divise par le nombre de lignes, donc auto-limitante côté Jeu réel. Contrôle : deux cartes équivalentes doivent afficher des chiffres Oracle identiques (leurs colonnes Jeu réel peuvent différer — DD-équivalent ≠ équivalent pour DouDou50).
- **Deux échelles à ne pas soustraire** : Oracle en points *cartes* DD (0-252), Jeu réel en points *de donne marqués* (contrat compris, >320 possible) — d'où l'écart signé N-S − E-O. Et « Contrat réussi » est l'issue du **preneur** : en défense un taux élevé désigne la pire carte, donc `real_win` relit l'événement du côté du siège qui joue et l'en-tête bascule en « Contrat chuté ».
- **Mondes** : `SidecarWorldSource` n'est pas utilisé ici ; la page passe par `playgen_gpu.play_worlds()` (route `/play_worlds`) puis `Analyst.play_worlds` (CPU, binding ajouté 2026-07-26), puis un mélange uniforme **qui ignore les coupes révélées par le jeu** — dégradé annoncé dans le badge. `play_worlds` rend les cartes *restantes* par siège, pas des donnes complètes : il faut y réinjecter les cartes déjà jouées pour obtenir une position résoluble.
- Une seule sim à la fois (`_cancel_sim_task`), `req_id` renvoyé comme pour les annonces. Le budget de mondes varie avec le nombre de cartes restantes (`WORLDS_BY_CARDS_LEFT`) : un solve coûte de moins en moins cher à mesure que la donne se vide.

**Rejouer page** (`views/replay.js`): two independent analysis passes, each cached in its own table and fetched separately so the slow one never blocks the fast one.
- `analysis.py` → `analysis` table, `/api/games/{id}/analysis`: DD cost of every card + bid review. A few seconds per game. The bid review carries **two** opinions per auction action: Bid V6's (`model_best`, a Q) and playgen v2's 43-way auction head (`playgen_best` + `playgen_p`, a probability — it is a world model, not a trained bidder). Like the IS-DD review, **one analyst instance per seat**: the head is read from the speaking seat's view, so a single instance would be conditioned on a hand that seat never saw. ~30 ms per bid on CPU; the sampler is kept only for the auction. `ANALYSIS_VERSION` 4. A row cached without a model is recomputed once it becomes available, or a load failure would leave the game permanently Playgen-less. Frontend: only V6 disagreement draws the `mv-bid-diff` outline (it is the reference bidder); playgen speaks in the tooltip. Both blocks test `!== undefined`, not just the row's presence.
- `agent_review.py` → `agent_review` table: what **DouDou50 / Oracle / Dédé (IS-DD)** would have played at every non-forced card, whoever actually played it. ~7-10s per deal at the `COLVER_REVIEW_ISDD_MS` (default 500) per-card IS-DD budget; one game at a time (module semaphore) so replay loads don't pile searches onto the playgen sidecar. **IS-DD is seat-bound** — four instances are built, one per seat, all shown every action, and the one whose seat is to play is the one asked. Asking a single instance would hand it information that seat never had.
  - `agent_review.stream()` is an async generator yielding `("start", total)`, `("move", entry)` per card **in play order**, `("done", blob)`. Each step runs in `asyncio.to_thread` (the Rust search releases the GIL), so the event loop stays free — measured WS round-trip during a review: <45ms. Abandoning the generator unwinds the lock and semaphore and caches nothing partial.
  - WS: client sends `replay_agents`, server streams `agent_review_{start,move,done,error}` from a cancellable task (`_agent_review_loop`), cancelled by the next `replay_load`/`replay_agents` or on disconnect. Sends in the replay branch go through `wsend` (the send lock) since the review task sends concurrently.
  - `/api/games/{id}/agents` is the same computation drained to completion, for non-WS callers.
  - Frontend refreshes via `refreshMoveStats()` (stats panel only) — **not** `renderHistoryEntry`, which carries navigation state (pending trick flush, forward/backward) that a mid-playback repaint would trample.

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
strategy = "nn"                    # heuristic|improved|improved_v2|improved_v3|smart|roro|maxi|petit_bide|moelleux|nn|playgen
model = "models/bid_nn_final.bin"  # required if strategy = "nn" or "playgen" (a playgen v2 .bin)
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
