# Bis-DD: DD-Based Unified Bidding & Playing Agent

A pure-heuristic agent that uses the DD (double-dummy) solver for **both** bidding and playing decisions, with no trained neural network. Beliefs about other players' hands are built from auction and play observations, then used to filter/weight determinizations before solving.

## Concept

Instead of training a bid NN or play NN, sample possible card distributions consistent with what we've observed, solve them with the DD solver, and pick the action that maximizes expected value (EV).

```
Observe bid/play actions
        │
  BeliefState accumulates:
    - Hard constraints (bid: evaluate_for_trump ≥ threshold)
    - Soft weights (J/9 boosts for bidders, reductions for passers, void inference)
        │
  determinize_weighted() → candidate world
        │
  check_constraints() → reject if hard constraints violated
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

### Hard constraints (only for positive bids)
When a player bids suit X at value V, we require `evaluate_for_trump(hand, X) >= threshold(V)`.

Thresholds (relaxed to accommodate different bidding strategies):

| Bid | Threshold | | Bid | Threshold |
|-----|-----------|---|-----|-----------|
| 80  | 7 | | 120 | 18 |
| 90  | 10 | | 130 | 20 |
| 100 | 12 | | 140 | 23 |
| 110 | 15 | | 150+ | 26-29 |

### Soft weights (for passes, bids, play)
- **Bidder**: J ×10, 9 ×6, A ×4, 10 ×3, other trump ×2.5
- **Passer**: J ×0.5, 9 ×0.6 in all suits. If bid on table: bid suit ×0.7, bid suit J ×0.25, 9 ×0.3
- **Coinche**: J ×3, 9 ×2.5 in bid suit, side aces ×2
- **Play**: void inference (hard), trump ceiling (hard), leader/cut inference (soft)

### Determinization
Hybrid weighted + rejection: `determinize_weighted()` biased by soft weights, then `check_constraints()` for hard bid constraints. Falls back to `determinize_greedy()` after 500 failed attempts.

With `parallel` feature: DD solves run in parallel (rayon), each thread gets its own TT buffer.

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

### Ideas for v2

**Bidding:**
- [ ] Tune risk margin (maybe too conservative at base 50?)
- [ ] Try `max_bid_value = 13` (130) with higher margin
- [ ] Partner response: when partner bid, lower the margin for same suit overbids
- [ ] Calibrate `bid_value_to_threshold` empirically from game data
- [ ] Try discounting DD points instead of EV margin (e.g., team_pts × 0.85)

**Beliefs:**
- [ ] Calibrate soft weight multipliers from actual game statistics
- [ ] Track belief coverage per bid level (are 80-bids well-calibrated but 120 not?)
- [ ] Acceptance rate is still 28% — more aggressive weighting could help

**Play:**
- [ ] Profile play time breakdown (determinize vs solve)
- [ ] Try more dets in early tricks (more uncertainty) vs fewer in late tricks
- [ ] Compare play quality vs Smart IS-DD on same deals

**Speed:**
- [ ] Pre-generate determinizations in batch (amortize RNG overhead)
- [ ] Share TT buffer across dets when not parallel (currently each gets 2MB)
- [ ] Profile: how much time in determinize vs solve?

**Arena:**
- [ ] Compare vs nn_v2_dmc35, nn_dmc35 (NN-based bots)
- [ ] Compare vs heuristic-only bots to isolate bid vs play contribution
- [ ] Create a mixed bot: bis_dd bid + smart_isdd play (or vice versa)
