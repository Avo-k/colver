# IS-DD (Information-Set Double-Dummy)

**Code:** [colver-core/src/search/is_dd.rs](../../colver-core/src/search/is_dd.rs)

Realistic player based on the [DD solver](dd_solver.md). Samples N "determinized worlds" consistent with current beliefs about hidden cards, solves each with DD, and aggregates per-card scores.

> **Naming:** `IsDdSearch` is the search. `IsDdPlayer` ([agents.md](../agents.md))
> is the agent that wraps it and owns its world source — that is what the arena
> and the web actually build. "Smart IS-DD" in old bot files and results means
> the same search with a `BeliefNet` loaded.

## Algorithm

Two entry points onto **one** pipeline:

- `search_with_source(state, config, rng, &mut dyn WorldSource)` — the real one.
  Worlds come from a [`WorldSource`](../agents.md), pulled in batches and
  refilled on demand. Returns `Result`: if the source fails, the error
  propagates rather than the search quietly continuing on weaker worlds.
- `search_with_stats(state, config, rng)` — no source, so infallible: worlds are
  sampled from beliefs / constraint-uniform only.

Both run in chunks until the determinization count or the time budget is hit:

```
0. REFILL from the WorldSource when the queue can't cover the round
     (count mode: ask for the whole remaining budget — one GPU round trip)
1. TAKE a chunk of worlds                          (sequential, stateful)
     ├─ from the source queue, else
     ├─ belief-weighted (prob. belief_frac when a belief source is on), else
     └─ constraint-uniform  (determinize_greedy)
2. WEIGHT each world by credibility                (sequential, cred_alpha)
3. SOLVE each world with DD                        (parallel or sequential)
4. AGGREGATE Σ score·weight / Σ weight per card    (fixed order)
→ pick best card (max for NS, min for EW)
```

`IsDdResult::worlds` reports how many solved worlds came from each branch, so a
run that should have been 100% playgen and wasn't says so instead of just
playing a few points per deal worse.

DD returns **exact** NS points per legal card per world, so far fewer samples are
needed than IS-MCTS (which uses noisy MCTS rollouts). 20 determinizations is
usually enough.

**Why generation and solving are split.** World *generation* is inherently
sequential — the world queue and the RNG are stateful, and the sidecar client
speaks one request at a time. DD *solving* is embarrassingly parallel: worlds are
independent, and every solve stamps a fresh **epoch** into the transposition table — an O(1)
invalidation rather than a memset since 2026-08-02 ([dd_solver.md](dd_solver.md)) — so no entry
from another world can ever be probed
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

Soft sources compose: `use_nn_beliefs = true` gives NN predictions *plus* the hard constraints, which are never optional.

If all soft beliefs are off and no hard constraints zero out any unknown (early game), the determinizer falls back to uniform `determinize_greedy` over the remaining cards.

## Configuration (`IsDdConfig`)

| Field | Default | Effect |
|-------|---------|--------|
| `determinizations` | 20 | Number of worlds sampled (overridden by `time_limit_ms` if set) |
| `time_limit_ms` | None | Time budget per move; **scaled by cards remaining**: `effective_ms = ms × cards_left / 8`. Lets early tricks have more time and endgame finish quickly. |
| `use_soft_inference` | **false** | Soft heuristic from play (dominance, "ne pisse pas" weight adjustments). |
| `use_nn_beliefs` | **false** | Use a loaded `BeliefNet` for soft predictions. |
| `early_termination` | true | Skip search when forced (1 legal move) or when beliefs uniquely determine all hidden cards (single DD solve = exact answer). Always on by default. |
| `dominance_factor` | 1.0 | Used by `use_soft_inference`. When a player follows suit without playing the highest, downweight their higher unknown cards by this factor. 0.3 = aggressive, 1.0 = off (only relevant if soft inference is enabled) |
| `bid_function` | `ImprovedV2` | Used during bidding phase (IS-DD only acts during play) |
| `world_batch` | 128 | Worlds requested per `WorldSource` refill under a time budget. In count mode the whole remaining budget is asked for at once. |
| `belief_frac` | 1.0 | **Fallback only** — when no source is attached or it runs dry, the fraction of worlds drawn belief-weighted (rest constraint-uniform for coverage). |
| `cred_alpha` | 0.0 | Credibility world-weighting exponent. See [Credibility weighting](#credibility-weighting). |
| `parallel` | **false** on `IsDdConfig`, **true** from an `AgentSpec` | Solve worlds across the rayon global pool. See [Parallelism](#parallelism). |

> Hard constraints (voids, trump ceiling, played cards) and `early_termination` are **always on** — they're correct by construction, no flag needed.

### Note on bid-derived beliefs

A previous experiment exposed soft bid inference (`partner bid 100 → likely strong trump`) in `BeliefState` for `BisDd` (both since removed). It was **rejected**: against NN bidders, the heuristic interpretation rejected reality 72% of the time. See [BIS_DD.md](../belief/bis_dd.md). The bid belief NN v4 (`bid_belief_v4.bin`) replaced it. The dominance-based play heuristic in `CardBeliefs::use_soft_inference` is independent of bid interpretation.

## Early termination

Two cases skip the determinization loop entirely:

1. **Forced move** — only 1 legal action. Return immediately with score=81 (neutral midpoint).
2. **Resolved position** — `try_resolve_position()` checks if beliefs uniquely determine every hidden card's owner (via `raw_weights() > 0` test). If so, build the fully-known state and call `solve_with_scores` once. This is exact, no determinization needed.

These trigger more often than expected: late in a deal, voids accumulate and 4-5 cards become uniquely owned, so endgame becomes a single DD solve.

## Removed: elephant memory

A particle filter that reused past determinizations as evidence. It was off by
default in every production bot, never beat plain IS-DD in the arena, and kept
five knobs alive in `IsDdConfig`. Removed 2026-07-24 — recover it from git
history (`colver-core/src/search/elephant.rs`) if the idea is worth revisiting.

## Performance

**This table multiplies; it does not measure.** Every entry is `n_dets ×` a per-shape solve cost
that is measured and maintained in **[dd_solver.md § Performance](dd_solver.md#performance)** —
the one place solver timings live. What IS-DD contributes is the multiplier and the shape
mapping, so that is all this table states. If a product below disagrees with the source table,
the source wins.

The previous version of this table was wrong by up to **four orders of magnitude**, always
optimistically — it implied ~2.5 ms for a full-deal solve where the measurement says ~32 ms, and
"~5-10 ms" for a resolved endgame that costs microseconds. That is what a table restating
numbers it does not own looks like once the numbers move.

`parallel` defaults to **true** from an `AgentSpec`, so the arena and the web divide the solve
column by the rayon pool — world *generation* stays sequential and does not.

| Config | Solve time per move (1 thread) | Notes |
|--------|---------------|-------|
| 20 dets, opening lead | 20 × *full deal* ≈ 650 ms | A trick-1 world **is** a full deal |
| 20 dets, mid-game (13-24 cards left) | 20 × *mid-game* ≈ 3.4 ms | Smaller search trees |
| 20 dets, with NN belief | + ~20 ms | NN forward + hybrid blend, independent of shape |
| With `time_limit_ms=50`, full hand | 50 ms × 8/8 = 50 ms | A deadline, not an amount of work |
| With `time_limit_ms=50`, last trick | 50 ms × 1/8 = 6 ms | idem |
| Resolved position (early term) | 1 × *endgame* ≈ 1.4 µs (≤ 12 cards left) | Single DD solve; 18.9× cheaper since the TT epoch change |
| Forced move (early term) | <1 µs | Constant return |

**Under a time budget the meaningful unit is worlds-per-budget, not milliseconds.** Per-world
solve cost falls **~23 000× across a deal** while `effective_ms` only falls 8×, so from mid-deal
onwards IS-DD is **world-generation-bound, not solver-bound**. Any future effort aimed at making
IS-DD faster in the endgame belongs in the sampler, not here — see
[../belief/playgen.md](../belief/playgen.md) for the sampler and
[dd_solver_optimization.md § 4](dd_solver_optimization.md#4-où-il-reste-du-temps-à-prendre) for
what is left on the solver side.

## Variants in the arena

Set in [arena/bots/](../../arena/bots/) TOML files:

```toml
[play]
method = "isdd"            # "is_dd" / "smart_is_dd" still parse (legacy names)
time_ms = 1000             # → time_limit_ms; 0 = count mode
determinizations = 240     # used when time_ms = 0

[worlds]                   # where the determinized worlds come from
source = "sidecar"         # sidecar (playgen on GPU, default) | playgen (CPU) | uniform
url = "http://gpu-host:8003"   # or $COLVER_PLAYGEN_GPU_URL

[belief]
model = "models/belief_v4_fix_v2.bin"   # optional soft predictions
```

Hard constraints are applied automatically — there is no flag for them.

**`[worlds]` is not optional in practice.** Omit it and the bot defaults to the
sidecar; with no URL anywhere it refuses to build rather than quietly sampling
uniform worlds. Write `source = "uniform"` when that is what you actually want.
Full reference: [agents.md](../agents.md).

Reference bots:
- `v6_isdd_75M_belief` — bid v6 + IS-DD + belief net (current champion)
- `v6_isdd_75M_isdd` — same without the belief net

## API

```rust
use colver_core::is_dd::{IsDdSearch, IsDdConfig};

let mut search = IsDdSearch::new();
// optional: search.load_belief_net("models/belief_v4_fix_v2.bin")?;

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

// When it's our turn — with a world source (the real configuration):
let result = search.search_with_source(&state, &config, &mut rng, &mut *source)?;
// result.best_action, result.card_scores, result.determinizations, result.worlds

// …or without one (beliefs / constraint-uniform only, infallible):
let action = search.search(&state, &config, &mut rng);
```

Most callers should not do any of this by hand — build an
[`IsDdPlayer`](../agents.md) from an `AgentSpec` and let it wire the world
source, the belief net and the credibility judges.

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
  agent path (`AgentSpec`, hence the arena and the web) turns it on by default.

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
