# IS-DD (Information-Set Double-Dummy)

**Code:** [colver-core/src/search/is_dd.rs](../../colver-core/src/search/is_dd.rs)

Realistic player based on the [DD solver](dd_solver.md). Samples N "determinized worlds" consistent with current beliefs about hidden cards, solves each with DD, and aggregates per-card scores.

> **Naming:** `IsDdSearch` (the struct) is the unified search; "Smart IS-DD" in the arena/code refers to the same struct with a `BeliefNet` loaded.

## Algorithm

There is **one** search entry point — `search_with_stats` (`search` is a thin
wrapper that drops the stats). It runs a single pipeline, in chunks, until the
determinization count or time budget is hit:

```
1. GENERATE a chunk of determinized worlds        (sequential, stateful)
     ├─ injected worlds first (e.g. GPU sidecar), then
     ├─ playgen world  (prob. playgen_frac, from a lockstep batch), then
     ├─ belief-weighted world (prob. belief_frac when a belief source is on), else
     └─ constraint-uniform world  (determinize_greedy)
2. WEIGHT each world by credibility               (sequential, cred_alpha)
3. SOLVE each world with DD                        (parallel or sequential)
4. AGGREGATE Σ score·weight / Σ weight per card    (fixed order)
→ pick best card (max for NS, min for EW)
```

DD returns **exact** NS points per legal card per world, so far fewer samples are
needed than IS-MCTS (which uses noisy MCTS rollouts). 20 determinizations is
usually enough.

**Why generation and solving are split.** World *generation* is inherently
sequential — the playgen sampler carries a KV-cache, the injected-world queue and
the RNG are all stateful. DD *solving* is embarrassingly parallel: worlds are
independent and the transposition table is cleared at the start of every solve
(`solver.rs::solve_reuse_tt`), so nothing is shared between worlds anyway. The
pipeline therefore generates a chunk sequentially, then hands the whole chunk to
the solver — sequentially or across the rayon pool (see [Parallelism](#parallelism)).

## Hard constraints vs soft beliefs

These are **two completely different things** in IS-DD:

### Hard constraints (facts — always on)

Things we **know** to be true from the public game state. They are not configurable and are applied unconditionally in every code path:

- **Voids**: a player who couldn't follow suit no longer has any card of that suit
- **Trump ceiling**: a player who undertrumped (or discarded under "ne pisse pas") cannot have a higher trump
- **Played cards**: any card already played is no longer in any hand
- **Observer's hand**: the cards in our own hand are known

These constraints zero out probabilities for impossible (player, card) combinations. There is no `use_hard_constraints` flag — it would be like a flag for "use facts".

Hard constraints are computed inside `CardBeliefs::raw_weights()` and applied automatically:
- Heuristic path (`use_nn_beliefs=false`): `CardBeliefs::normalized_weights()` already excludes impossible cards
- NN path (`use_nn_beliefs=true`): NN soft predictions are masked by the same hard constraints (any card with `raw_weight==0.0` in CardBeliefs gets `nn_weight=0.0`)

### Soft beliefs (probabilistic guesses — all OFF by default)

Adjustments to probabilities based on **inferences** (which may be wrong). All disabled by default:

| Source | Flag | What it does |
|--------|------|--------------|
| **Heuristic soft inference** | `use_soft_inference` | Applies dominance reasoning ("player X followed without playing the highest → downweight their higher unknowns") and optional bid signal interpretation in `CardBeliefs` |
| **NN soft beliefs** | `use_nn_beliefs` | Loads a trained `BeliefNet` and uses its soft predictions for card locations. Hard constraints are still applied on top. |
| **Elephant memory** | `use_elephant_memory` | Particle filter that accumulates determinizations as particles, filters them by observed plays, and blends with base beliefs. See [section below](#elephant-memory). |

Multiple soft sources can be enabled simultaneously — e.g. `use_nn_beliefs=true` + `use_elephant_memory=true` gives NN soft + hard constraints + particle blend.

If all soft beliefs are off and no hard constraints zero out any unknown (early game), the determinizer falls back to uniform `determinize_greedy` over the remaining cards.

## Configuration (`IsDdConfig`)

| Field | Default | Effect |
|-------|---------|--------|
| `determinizations` | 20 | Number of worlds sampled (overridden by `time_limit_ms` if set) |
| `time_limit_ms` | None | Time budget per move; **scaled by cards remaining**: `effective_ms = ms × cards_left / 8`. Lets early tricks have more time and endgame finish quickly. |
| `use_soft_inference` | **false** | Soft heuristic from play (dominance, "ne pisse pas" weight adjustments). |
| `use_nn_beliefs` | **false** | Use a loaded `BeliefNet` for soft predictions. |
| `use_elephant_memory` | **false** | Particle filter from past determinizations. |
| `early_termination` | true | Skip search when forced (1 legal move) or when beliefs uniquely determine all hidden cards (single DD solve = exact answer). Always on by default. |
| `dominance_factor` | 1.0 | Used by `use_soft_inference`. When a player follows suit without playing the highest, downweight their higher unknown cards by this factor. 0.3 = aggressive, 1.0 = off (only relevant if soft inference is enabled) |
| `bid_function` | `ImprovedV2` | Used during bidding phase (IS-DD only acts during play) |
| `playgen_frac` | 0.0 | Fraction of worlds drawn from the playgen transformer (requires a loaded playgen model). 0 = none, 1 = all. |
| `playgen_temp` | 1.0 | Softmax temperature for playgen sampling (1.0 = posterior). |
| `belief_frac` | 1.0 | Among non-playgen worlds, fraction drawn belief-weighted (rest constraint-uniform for coverage). |
| `cred_alpha` | 0.0 | Credibility world-weighting exponent. See [Credibility weighting](#credibility-weighting). |
| `parallel` | **false** | Solve worlds across the rayon global pool. See [Parallelism](#parallelism). |

> Hard constraints (voids, trump ceiling, played cards) and `early_termination` are **always on** — they're correct by construction, no flag needed.

### Note on bid-derived beliefs

A previous experiment exposed soft bid inference (`partner bid 100 → likely strong trump`) in `BeliefState` for `BisDd`. It was **rejected**: against NN bidders, the heuristic interpretation rejected reality 72% of the time. See [BIS_DD.md](../belief/bis_dd.md). The bid belief NN v4 (`bid_belief_v4.bin`) replaced it. The dominance-based play heuristic in `CardBeliefs::use_soft_inference` is independent of bid interpretation.

## Early termination

Two cases skip the determinization loop entirely:

1. **Forced move** — only 1 legal action. Return immediately with score=81 (neutral midpoint).
2. **Resolved position** — `try_resolve_position()` checks if beliefs uniquely determine every hidden card's owner (via `raw_weights() > 0` test). If so, build the fully-known state and call `solve_with_scores` once. This is exact, no determinization needed.

These trigger more often than expected: late in a deal, voids accumulate and 4-5 cards become uniquely owned, so endgame becomes a single DD solve.

## Elephant memory

**Particle filter** for online belief refinement. Stores up to N "particles" (each = a `[u32; 4]` hand assignment) accumulated from past determinizations. After observing each card play, particles inconsistent with the play are filtered out.

| Field | Default | Effect |
|-------|---------|--------|
| `use_elephant_memory` | false | Enable the particle filter |
| `elephant_smoothing` | 0.05 | Blending factor when combining base beliefs with particle evidence. Lower = stronger particle influence |
| `elephant_dominance_penalty` | 0.5 | Soft penalty per dominant card not played (a player who didn't play their best is unlikely to have it) |
| `elephant_use_dominance` | true | Apply the dominance penalty to particle scoring |
| `elephant_decay` | 0.8 | Decay factor for old particles (0.8 = particles lose 20% influence per turn). 1.0 = no decay |

### Lifecycle

1. **`init_deal_with_config()`** — reset particle pool for new deal
2. **Each `search_with_stats()` call** — generated determinizations are added as new particles via `elephant.add_particles()`
3. **`record_action()` after each play** — `elephant.observe_play()` filters surviving particles consistent with the observed card. If dominance penalty is on, particles where the player held a dominant card they didn't play get downweighted
4. **Next search** — `compute_evidence()` aggregates surviving particle distributions, then `blend_with_evidence()` combines with base beliefs (NN or heuristic) using `elephant_smoothing`
5. **Resolved position case** — when a single DD solve is enough, the resolved hands are fed back as a particle (re-seeds the pool with ground truth)

Stats: `elephant_stats() → (surviving_particles, total_particles)` and `elephant_evidence(state) → Option<weights>`.

### When to use

Elephant memory is most useful **mid-to-late game** when belief uncertainty has narrowed and particles can converge. In early tricks the particle pool is sparse and noisy. It's **off by default** in production bots — primarily an experimental feature for belief studies.

## Performance

| Config | Time per move | Notes |
|--------|---------------|-------|
| 20 dets, no belief, full hand | ~50 ms | Default setup |
| 20 dets, with NN belief | ~70 ms | +20ms for NN forward + hybrid blend |
| 20 dets, mid-game (4 tricks left) | ~20 ms | Smaller search trees |
| With `time_limit_ms=50`, full hand | up to 50ms × 8/8 = 50ms | Auto-scaled |
| With `time_limit_ms=50`, last trick | up to 50ms × 1/8 = 6ms | Endgame finishes fast |
| Resolved position (early term) | ~5-10 ms | Single DD solve |
| Forced move (early term) | <1 µs | Constant return |

## Variants in the arena

Set in [arena/bots/](../../arena/bots/) TOML files:

```toml
[play]
method = "smart_is_dd"     # alternatives: "is_dd", "smart_ismcts"
time_ms = 50               # → time_limit_ms
determinizations = 20

[belief]
model = "models/belief_v3.bin"     # optional, loads BeliefNet (soft predictions)
```

Hard constraints are applied automatically — there is no flag for them.

Reference bots:
- `nn_v2_isdd_no_belief` — heuristic CardBeliefs only, no NN
- `nn_v2_isdd` — NN belief net + hard constraints (current production strongest)

## API

```rust
use colver_core::is_dd::{IsDdSearch, IsDdConfig};

let mut search = IsDdSearch::new();
// optional: search.load_belief_net("models/belief_v3.bin")?;

let config = IsDdConfig {
    determinizations: 20,
    time_limit_ms: Some(50),
    use_nn_beliefs: true,    // optional soft beliefs; hard constraints always on
    ..Default::default()
};

// Initialize beliefs at start of deal
search.init_deal_with_config(&state, observer, &config);

// Each turn (any player):
search.record_action(&state_before, player, action);  // update beliefs

// When it's our turn:
let action = search.search(&state, &config, &mut rng);
// Or with stats:
let result = search.search_with_stats(&state, &config, &mut rng);
// result.best_action, result.card_scores, result.determinizations
```

## Parallelism

Set `config.parallel = true` to solve the determinized worlds across the **rayon
global pool** instead of one at a time. There is no separate `search_parallel`
method any more — parallelism is a config flag on the single search path.

- **Bounded and shared.** The rayon global pool has `num_cpus` workers shared by
  *all* concurrent searches, so several web rooms solving at once cannot
  oversubscribe the machine — they share one pool via work-stealing.
- **Deterministic.** Results are bit-identical to sequential: world generation is
  always sequential (same RNG order regardless of the flag), DD is exact, and the
  aggregation reduces in a fixed input order. The `test_parallel_matches_sequential`
  test asserts `best_action` and `card_scores` match exactly.
- **Per-worker TT.** Each rayon worker keeps its own reusable transposition table
  (`map_init`); sequential mode reuses the search's own `tt_buf`. Since the TT is
  cleared per solve, no cross-world information is lost either way.
- **Requires the `parallel` cargo feature.** Without it the flag is ignored and
  the search falls back to sequential. `colver-py` enables the feature, so the
  web/PyO3 path (`action_dede*`) always runs parallel.

Chunk size is one world in sequential mode (tightest deadline adherence) and one
worker-pool round (`rayon::current_num_threads()`) in parallel mode.

## Credibility weighting

`cred_alpha > 0` weights each world in the DD aggregation by how *plausible* it
is, judged **self-supervisedly** against the observed history: replay the deal
holding the world's reconstructed hands and ask a reference policy, at each hidden
player's turn, "would you have played what actually happened?". This is the same
signal measured in aggregate by [`bench_world_cred`](../belief/playgen.md).

- **Auction** — judged by the bid net (`load_cred_bid_net` / `dede_load_cred_bid_net`).
- **Play** — judged by the canonical DMC net (`load_cred_play_net` / `dede_load_cred_play_net`).

Each judged action contributes a rank factor (argmax → 1.0, top-3 → 0.7, else →
0.35); the product across judged actions is flattened by `w.powf(alpha)`. Weight
1.0 when disabled, no judge loaded, or the world can't be reconstructed.

**Statistical caveat.** Playgen already samples ≈ the posterior, so its credibility
weight is the likelihood term of that same posterior — re-weighting playgen worlds
by credibility risks *double-counting*. Prefer credibility either (a) to correct a
mis-calibrated / over-dispersed sampler, or (b) with playgen treated as a proposal
only. Measure with `bench_world_cred` before enabling; kept **off by default**
pending that check. Cost scales with tricks played (one DMC eval per hidden play
per world), so keep world counts modest when the play judge is on.

## Comparison with DMC

DMC and IS-DD have very similar mean MAE vs DD (~19) but make **different errors** — see [bid/reward_studies/v3_reward_study.md](../bid/reward_studies/v3_reward_study.md). Hamming distance on the same deal is **29/32 cards**: stylistically opposed (DMC plays Aces immediately, IS-DD pulls trumps systematically with the J).

Both are realistic players. IS-DD is slightly stronger on extreme hands (capots, very weak); DMC is slightly stronger on standard mid-range hands. Their `max(DMC, ISDD)` is the basis for the [bid_v3_max_20M champion](../bid/strategies/bid_v3_max.md).
