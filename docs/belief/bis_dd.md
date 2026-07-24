# Bis-DD: DD-Based Unified Bidding & Playing Agent

> **Removed (2026-07-24).** `bis_dd.rs` and the `bis_dd` bid/play methods are
> gone: the heuristic bid inference at the heart of the design rejected reality
> 72% of the time against NN bidders, and the bid belief net replaced it. This
> document is kept as the record of the experiment and its negative result —
> the code is recoverable from git history. Current agent architecture:
> [agents.md](../agents.md).

A pure-heuristic agent that uses the DD (double-dummy) solver for **both** bidding and playing decisions, with no trained neural network. Beliefs about other players' hands are built from auction and play observations, then used to filter/weight determinizations before solving.

## Concept

Instead of training a bid NN or play NN, sample possible card distributions consistent with what we've observed, solve them with the DD solver, and pick the action that maximizes expected value (EV).

```
Observe bid/play actions
        │
  BeliefState accumulates:
    - Soft weights (J/9 boosts for bidders, reductions for passers, play inference)
    - Hard constraints from play only (voids, trump ceiling)
        │
  determinize_weighted() → candidate world biased by soft weights
        │
  DD solver (solve_for_trump / solve_with_scores)
        │
  Aggregate EV across N worlds → pick best action
```

## Files

- `belief/belief_state.rs` — `BeliefState`: constraint accumulation, soft weights, determinization
- `bid/bis_dd.rs` — `BisDdAgent`: decision logic (bid EV + play IS-DD), config
- `bin/bis_dd_diag.rs` — diagnostic binary for belief quality and timing
- `bin/arena.rs` — arena integration (stateful agent pattern)
- `arena/bots/bis_dd.toml` — bot config for arena

## Current Config (defaults)

| Param | Value | Notes |
|-------|-------|-------|
| `min_dets` | 10 | Minimum determinizations before deciding |
| `bid_time_ms` | 500 | Time budget per bid decision |
| `play_time_ms` | 200 | Time budget per play decision |
| `prefilter_threshold` | 6 | Min `evaluate_for_trump` to consider a suit |
| `max_bid_value` | 12 (=120) | Cap on bid level (DD overestimates 130+) |
| `evaluate_capot` | true | Whether to consider capot bids |

Diagnostic binary uses higher budgets (20 dets, 2000ms bid, 500ms play) for quality.

## Belief System

### No hard constraints from bids
Originally, positive bids added a hard constraint requiring `evaluate_for_trump(hand, suit) >= threshold`. This was **removed** because eval_beliefs measurement showed it rejected reality ~72% of the time against NN bidders — the NN bidding strategy doesn't follow heuristic hand evaluation patterns. Soft weights alone bias sampling correctly without rejecting valid hands.

### Hard constraints (from play only, 0% false exclusion rate)
- **Void inference**: didn't follow suit → void in that suit
- **Trump void**: didn't trump when forced (opponent winning, no prior trump on trick) → void in trump. Correctly handles **"ne pisse pas"**: when an opponent has already trumped and the player can't overtrump, discarding is legal — we apply a **trump ceiling** instead of void.
- **Trump ceiling**: undertrumped → no stronger trump than the best trump on the trick. Only applied when player was forced (opponent winning or following trump lead).

### Soft weights (for passes, bids, play)
- **Bidder**: J ×10, 9 ×6, A ×4, 10 ×3, other trump ×2.5
- **Passer**: J ×0.5, 9 ×0.6 in all suits. If bid on table: bid suit ×0.7, bid suit J ×0.25, 9 ×0.3
- **Coinche**: J ×3, 9 ×2.5 in bid suit, side aces ×2
- **Play**: lead trump → boost remaining trump ×1.8; lead ace → boost 10/K for player, reduce for others; lead low → reduce A/10/K for player, boost for others; lead 10 → boost A ×2; lead K → boost A ×1.8; cut with strong trump → boost trump ×1.4; discard → reduce A/10 for player, boost for others

### Determinization
`determinize_weighted()` biased by soft weights, falls back to `determinize_greedy()`. No retry loop when no hard constraints (the common case). With `parallel` feature: DD solves run in parallel (rayon), each thread gets its own TT buffer.

## Bid EV Formula

For each candidate suit × value:
```
EV_bid(suit, value, team) = mean over determinizations:
    team_pts = DD_solve(suit) for team
    if team_pts >= contract:  team_pts + contract   (made)
    else:                    -contract               (failed)
```

Pass EV accounts for existing bid on table (opponent or partner contract).

**Risk margin**: only bid if `EV_bid - EV_pass > margin`. Margin scales with bid level:
`margin = 50 + (bid_value - 8) × 10` → 50 for bid 80, 90 for bid 120.

Capot requires margin > 200. Coinche requires margin > 50.

---

## Experiment Log

### v0 — Initial implementation (2025-04-04)

Hard pass constraints (J+9 rejection, eval threshold) + hard bid constraints. Sequential DD solves.

**Diagnostic (50 games vs improved_v2 + rule_play):**

| Metric | Value |
|--------|-------|
| Bid coverage | 48.5% |
| Play coverage | 21.7% |
| Acceptance rate | 7.3% |
| Contract success | 30.6% |
| NS advantage | negative |
| Avg bid value | 115 |
| Avg bid time | 2300ms |

**Problem:** beliefs way too restrictive — rejecting reality 50% of the time.

### v1 — Relaxed constraints + parallel + tuning (2025-04-04)

Changes:
1. **Removed hard pass constraints** — passes are soft-only (weight adjustments, no rejection)
2. **Lowered bid thresholds** ~30% (80: 10→7, 100: 17→12, 120: 23→18)
3. **Boosted bidder soft weights** (J: 5→10, 9: 3→6, A: 2→4, 10: 1.5→3)
4. **Parallelized DD solves** with rayon (`#[cfg(feature = "parallel")]`)
5. **Added scaled risk margin** (base 50 + 10/level) to compensate DD overestimation
6. **Capped max bid at 120** — DD overestimates badly at 130+ (0% real success)
7. **Reduced default budgets** for arena speed (500ms bid, 200ms play, 10 dets)

**Diagnostic (50 games vs improved_v2 + rule_play):**

| Metric | v0 | v1 |
|--------|----|----|
| Bid coverage | 48.5% | **89.4%** |
| Play coverage | 21.7% | **85.0%** |
| Acceptance rate | 7.3% | **28.6%** |
| Contract success | 30.6% | **63.2%** |
| NS advantage | negative | **+59** |
| Avg bid value | 115 | **100** |
| Avg bid time | 2300ms | **1284ms** |

Bid breakdown (all 80-120 range, 61-71% success):

| Bid | Count | Made |
|-----|-------|------|
| 80 | 18 | 61% |
| 90 | 7 | 71% |
| 100 | 17 | 65% |
| 110 | 15 | 67% |
| 120 | 16 | 69% |

**Arena H2H (10×2 matches vs improved_isdd, 10 dets, 500ms bid):**
- **bis_dd 35% vs improved_isdd 65%** (7-13, avg margin -290)
- Dir 1 (BisDd=NS): 3-7 | Dir 2 (BisDd=EW): 6-4
- Wall: 474s (2.5 matches/min)

### v2 — Belief quality audit + constraint fix (2025-04-04)

Built `eval_beliefs` binary to measure belief quality against ground truth (known hands). Played 200 deals with NN bots (bid_v2 + doudou50), tracked CardBeliefs and BeliefState at every play decision.

**Critical finding:** BeliefState hard bid constraints rejected reality 72% of the time. The `evaluate_for_trump` thresholds don't match NN bidding behavior — the NN bids based on learned Q-values, not heuristic hand evaluation.

Changes:
1. **Removed hard bid constraints** from BeliefState — soft weights only (rejection rate 72% → 0%)
2. **Simplified `determinize()`** — no retry loop when no hard constraints (was 500 attempts)
3. **Improved play inference** (both CardBeliefs and BeliefState):
   - Bidirectional signals: lead ace boosts 10/K for player AND reduces for others
   - New inferences: lead 10 → boost A, lead K → boost A, cut with strong trump → boost trump
   - Low lead → reduce A/10/K for player (was only boosting for others)
   - Discard → reduce A/10 for player in discarded suit
4. **Fixed "ne pisse pas" bug** in hard constraints: when an opponent already trumped and the player can't overtrump, the player may legally discard (not trump). Old code wrongly inferred "void in trump" → now applies a trump ceiling instead. Also fixed trump ceiling to not apply when partner is master (voluntary undertrump is valid strategy).
5. **Evaluated belief_v3.bin NN**: catastrophically bad (log(p) = -2.1 play, -7.2 bid vs -1.1 uniform). Was trained on different bots — useless for NN bidder/player patterns.

**eval_beliefs results (500 deals, before → after all fixes):**

| Metric | Before | After |
|--------|--------|-------|
| CB log(p) | -1.0515 | **-1.0209** |
| CB false exclusions | 0.215% | **0.000%** |
| Ground truth reachable | ~99% | **100.0%** |
| CB placement accuracy | ~36% | **40.1%** |
| BS constraint rejections | 72.2% | **0%** |

**Arena (200×2 matches, seed 42):**

| Matchup | Win% |
|---------|------|
| nn_v2_isdd vs doudou50 | 52% (first 200), 47.8% (next 400) |
| nn_v2_isdd_no_belief vs doudou50 | 47% |
| nn_v2_isdd is #1 in overall leaderboard (61.1%) | |

### v3 — Bid Belief NN (2026-04-04)

Trained a neural network to predict card locations from auction history, replacing the weak heuristic bid soft weights in BeliefState.

**Architecture:** 108→256→256→96 MLP (LayerNorm+ReLU), 3-class output (left/partner/right per card).

**Training:**
- Data: 500K deals × bid_v2 auctions → 14.2M samples (4 observers × ~7 bid steps/deal)
- 24× suit augmentation, cosine LR with warmup, count regularization
- Format: COLVBB01 (108-float bid obs + 32-byte target + mask)
- 30 epochs, val_loss 1.0283 (baseline uniform = 1.099), val_acc 42.7%

**Integration:** `BeliefState::apply_nn_bid_beliefs()` runs the NN once after bidding, replacing heuristic bid weights. Play heuristic weights multiply on top during the play phase: `NN_bid_prior × play_heuristic`.

**eval_beliefs results (500 deals, bid_v2 + doudou50):**

| Metric | CB | BS (heuristic) | BS+NN |
|--------|-----|----------------|-------|
| Play log(p) | -1.0209 | -1.0564 | **-0.9565** |
| Trick 0 log(p) | -1.0745 | -1.1218 | **-1.0086** |
| Trick 4 log(p) | -0.9729 | -0.9995 | **-0.9088** |

BS+NN is the best belief system at every trick. The NN bid priors carry through the entire game, providing ~0.06-0.07 log(p) improvement over the best heuristic at every data point.

**Files:**
- `belief_obs.rs`: `write_bid_belief_obs()` — 108-float bid observation from any observer's perspective
- `belief_state.rs`: `apply_nn_bid_weights_raw()`, `apply_nn_bid_beliefs()` — NN integration
- `belief_net.rs`: obs_dim=108 auto-detection added
- `bin/gen_bid_belief_data.rs`: training data generator (COLVBB01 format)
- `bin/train_belief_net.rs`: COLVBB01 + bid augmentation support
- `bin/eval_beliefs.rs`: `--bid-belief` flag for NN evaluation
- Model: `models/bid_belief_v4.bin` (477KB)

**Arena integration:** Pending. BisDdAgent uses BeliefState and would naturally benefit via `apply_nn_bid_beliefs()`, but arena.rs doesn't yet wire the bid_belief model. SmartIsDd uses CardBeliefs (not BeliefState) and would need a different integration path.

### Ideas for v4

**Arena integration** (priority):
- [ ] Wire `bid_belief` model into BisDdAgent via arena TOML `[belief] bid_model = "..."`
- [ ] Wire into SmartIsDd (requires CardBeliefs → BeliefState migration or NN weight injection)

**Belief NN improvements:**
- [ ] Larger architecture (512 hidden, 3 layers) for more capacity
- [ ] Play-phase belief NN (using bid NN output as input feature)
- [ ] Hypothesis-based bid inference (discrete hand profiles)

**Speed:**
- [ ] Pre-generate determinizations in batch
- [ ] Profile: determinize vs solve time breakdown
