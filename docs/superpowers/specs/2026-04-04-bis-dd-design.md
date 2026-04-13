# Bis-DD: Belief-Informed IS-DD Agent

A unified agent that maintains persistent beliefs from bidding through play, using rejection-filtered determinization and DD solving for all decisions.

## Motivation

Current bidding strategies are either heuristic (fast but shallow) or NN-based (strong but opaque, requires training). The DD solver gives exact evaluations but `DdBidder` uses uniform sampling without exploiting auction information. Meanwhile, `SmartIsMctsSearch` tracks beliefs during play but doesn't use DD for bidding.

Bis-DD bridges these gaps: sample card distributions consistent with the observed auction, solve them with DD, and pick the action that maximizes expected match points. The same belief state persists into the play phase, feeding IS-DD with richer constraints than any current approach.

## Architecture Overview

Two new components:

```
BeliefState (belief/belief_state.rs)
  Accumulates constraints from every observed action.
  Produces belief-filtered determinizations.
      │
      ▼
BisDdAgent (bid/bis_dd.rs)
  Uses BeliefState + DD solver for both bid and play decisions.
  Single agent from deal start to deal end.
```

## Component 1: BeliefState

### Data Structure

```rust
pub struct BeliefState {
    observer: u8,
    observer_hand: CardSet,

    // Soft weights for biased sampling (multiplicative, like CardBeliefs)
    soft_weights: [[f32; 32]; 4],

    // Hard constraints accumulated from observed actions
    constraints: Vec<ActionConstraint>,

    // Void tracking (updated during play)
    voids: [u8; 4],
}
```

### ActionConstraint

```rust
pub struct ActionConstraint {
    player: u8,
    kind: ConstraintKind,
}

pub enum ConstraintKind {
    /// Player bid `suit` at level implying evaluate_for_trump >= min_score.
    Bid { suit: Suit, min_score: u16 },

    /// Player passed. Context determines how restrictive the constraint is.
    Pass {
        /// Minimum bid value they would have needed to overbid.
        min_overbid_value: u8,
        /// What position in the auction (0-3). Position 3 (after 2 passes)
        /// makes a pass very informative since players almost always bid there.
        auction_position: u8,
        /// Whether partner had already bid (pass without support).
        partner_had_bid: bool,
        /// The active bid suit, if any (pass = can't contest this suit).
        active_suit: Option<Suit>,
    },
}
```

### Constraint Rules

**From a bid (suit S, value V):**
- Hard: `evaluate_for_trump(hand, S) >= threshold(V)`
  - threshold(80) = 10, threshold(90) = 12, threshold(100) = 14, etc.
- Soft weights: boost J_S (5x), 9_S (3x), A_S (2x), other trumps (1.5x)

**From a pass — opening (no bid yet):**
- Hard: player does NOT have J+9 in any single suit (from BID_RULES.md: this combination almost always triggers a bid)
- Hard: `max(evaluate_for_trump(hand, s) for s in 0..4) < 10`
- Soft weights: reduce J/9 in all suits (0.6x)
- Exception: position 3 pass is extremely informative (score < 8 in all suits)

**From a pass — after opponent bid S at V:**
- Hard: `evaluate_for_trump(hand, best_suit) < threshold(V + 10)` for all suits (can't overbid)
- Soft weights: reduce trump honors in all suits (0.7x)
- Weaker than opening pass (could have a decent hand, just not enough to overbid)

**From a pass — after partner bid S:**
- Hard: `evaluate_for_trump(hand, S) < threshold(V + 10)` (can't support/raise)
- Soft: reduce honors in partner's suit (0.7x), other suits neutral
- Weakest pass constraint (not supporting partner doesn't mean weak overall)

**From a coinche (opponent's contract S at V):**
- Soft weights: boost J_S (3x), 9_S (2.5x), side aces (2x) — defensive holding
- No hard constraint (coinche is a judgment call, hard to filter precisely)

### Determinization

Hybrid approach: biased generation + hard rejection.

```rust
impl BeliefState {
    /// Generate a determinization consistent with all accumulated beliefs.
    pub fn determinize(
        &self,
        state: &GameState,
        rng: &mut impl Rng,
    ) -> Option<GameState> {
        // Max attempts before giving up
        for _ in 0..500 {
            // Step 1: Generate candidate via weighted sampling
            let candidate = determinize_weighted(
                state, self.observer, &self.soft_weights, rng,
            )?;

            // Step 2: Check all hard constraints
            if self.check_constraints(&candidate.hands) {
                return Some(candidate);
            }
        }
        // Fallback: return unfiltered determinization
        determinize_greedy(state, self.observer, rng)
    }

    fn check_constraints(&self, hands: &[CardSet; 4]) -> bool {
        self.constraints.iter().all(|c| c.is_satisfied(hands))
    }
}
```

**Constraint checking for each kind:**

```rust
impl ActionConstraint {
    fn is_satisfied(&self, hands: &[CardSet; 4]) -> bool {
        let hand = hands[self.player as usize];
        match &self.kind {
            ConstraintKind::Bid { suit, min_score } => {
                evaluate_for_trump(hand, *suit) >= *min_score
            }
            ConstraintKind::Pass {
                min_overbid_value,
                auction_position,
                partner_had_bid,
                active_suit,
            } => {
                // Opening pass: no J+9 in any suit, best score < 10
                if *min_overbid_value == 0 {
                    let has_j9 = (0..4u8).any(|s| {
                        let bits = suit_bits(hand, Suit::from_u8(s));
                        (bits & (1 << 3) != 0) && (bits & (1 << 2) != 0) // J + 9
                    });
                    if has_j9 { return false; }

                    let threshold = if *auction_position >= 2 { 8 } else { 10 };
                    (0..4u8).all(|s|
                        evaluate_for_trump(hand, Suit::from_u8(s)) < threshold
                    )
                } else {
                    // Pass after a bid: can't overbid in any suit
                    let overbid_threshold = bid_value_to_threshold(*min_overbid_value);
                    if *partner_had_bid {
                        // Only check the partner's suit (can't support)
                        if let Some(suit) = active_suit {
                            evaluate_for_trump(hand, *suit) < overbid_threshold
                        } else {
                            true
                        }
                    } else {
                        // Check all suits
                        (0..4u8).all(|s|
                            evaluate_for_trump(hand, Suit::from_u8(s)) < overbid_threshold
                        )
                    }
                }
            }
        }
    }
}
```

**`bid_value_to_threshold` mapping:**

| Bid value (encoded) | Actual bid | eval threshold |
|---------------------|-----------|----------------|
| 8 | 80 | 10 |
| 9 | 90 | 12 |
| 10 | 100 | 14 |
| 11 | 110 | 17 |
| 12 | 120 | 20 |
| 13 | 130 | 23 |

Derived from `score_to_bid_value` inverse in `eval_helpers.rs`.

### Lifecycle

```rust
impl BeliefState {
    pub fn new(observer: u8, observer_hand: CardSet) -> Self { ... }

    /// Record a bid action. Adds constraint + adjusts soft weights.
    pub fn record_bid(&mut self, player: u8, action: u8, state: &GameState) { ... }

    /// Record a play action. Updates voids, trump ceilings, soft weights.
    pub fn record_play(&mut self, player: u8, card: Card, state: &GameState) { ... }

    /// Generate a belief-consistent determinization.
    pub fn determinize(&self, state: &GameState, rng: &mut impl Rng) -> Option<GameState> { ... }
}
```

Play-phase inference reuses the same logic as `CardBeliefs::infer_play`:
- Hard: void marking (didn't follow suit → void in that suit)
- Hard: trump ceiling (couldn't overtrump → no higher trump)
- Soft: led trump → likely has more; led ace → likely has 10/K; cut with low trump → lacks stronger

## Component 2: BisDdAgent

### Data Structure

```rust
pub struct BisDdAgent {
    belief: BeliefState,
    tt_buf: Vec<u64>,       // Reused across all DD solves
    config: BisDdConfig,
    rng: StdRng,            // Seeded RNG for deterministic replay
}

pub struct BisDdConfig {
    pub min_dets: u32,              // Minimum determinizations before considering stop (default 20)
    pub time_budget_ms: u32,        // Time budget per decision (default 2000ms for bid, 500ms for play)
    pub prefilter_threshold: u16,   // Min eval score to consider a suit (default 6)
    pub evaluate_capot: bool,       // Whether to evaluate capot bids (default true)
}
```

**Time-bounded loop:** run at least `min_dets` determinizations, then keep going until `time_budget_ms` is exhausted. Mid-game DD solves are ~13ms (vs ~77ms from full deal), so the same time budget yields many more samples later in the game — automatically adapting precision where it's cheapest.

```
```

### Public API

```rust
impl BisDdAgent {
    pub fn new(config: BisDdConfig, seed: u64) -> Self { ... }

    /// Start a new deal. Resets beliefs.
    pub fn init_deal(&mut self, observer: u8, hand: CardSet) { ... }

    /// Record another player's action (bid or play).
    pub fn observe(&mut self, player: u8, action: u8, state: &GameState) { ... }

    /// Choose a bid action. Returns encoded bid (0=PASS, 1-40=bids, 41=COINCHE, 42=SURCOINCHE).
    pub fn decide_bid(&mut self, state: &GameState) -> u8 { ... }

    /// Choose a play action. Returns card index 0-31.
    pub fn decide_play(&mut self, state: &GameState) -> u8 { ... }
}
```

### decide_bid Algorithm

```
fn decide_bid(&mut self, state: &GameState) -> u8 {
    let player = state.current_player;
    let team = GameState::player_team(player);
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();

    // 1. Identify candidate suits (prefilter)
    let candidates: Vec<u8> = (0..4u8)
        .filter(|&s| evaluate_for_trump(hand, Suit::from_u8(s)) >= self.config.prefilter_threshold)
        .collect();

    // 2. Determine what we need to solve
    let mut suits_to_solve: Vec<u8> = candidates.clone();
    // Add opponent's suit if they bid (needed for pass/coinche evaluation)
    if state.last_bid_value > 0 {
        let opp_suit = state.last_bid_suit;
        if !suits_to_solve.contains(&opp_suit) {
            suits_to_solve.push(opp_suit);
        }
    }

    // 3. Generate determinizations (parallel if feature enabled)
    let dets: Vec<GameState> = (0..self.config.num_dets * 3)  // oversample for rejections
        .filter_map(|_| self.belief.determinize(state, &mut self.rng))
        .take(self.config.num_dets as usize)
        .collect();

    // 4. Solve DD for each det × each suit
    // results[det_idx][suit_idx] = ns_points
    let results: Vec<[u8; 4]> = dets.iter().map(|det| {
        let mut pts = [0u8; 4];
        for &s in &suits_to_solve {
            pts[s as usize] = solve_for_trump_reuse_tt(
                det.hands, state.dealer, s, &mut self.tt_buf
            )[0];
        }
        pts
    }).collect();

    // 5. Evaluate each candidate action

    let mut best_action = BID_PASS;
    let mut best_ev = f32::NEG_INFINITY;

    // Evaluate PASS (or "let them play")
    let ev_pass = self.evaluate_pass(&results, state, team);
    if ev_pass > best_ev {
        best_ev = ev_pass;
        best_action = BID_PASS;
    }

    // Evaluate each suit × value combination
    for &suit in &candidates {
        for bid_value in self.legal_bid_values(suit, state, legal) {
            let ev = self.evaluate_bid(&results, suit, bid_value, team);
            if ev > best_ev {
                best_ev = ev;
                best_action = encode_bid(bid_value, suit);
            }
        }
    }

    // Evaluate CAPOT if enabled
    if self.config.evaluate_capot {
        for &suit in &candidates {
            let ev = self.evaluate_capot(&results, suit, team);
            if ev > best_ev && legal & (1u64 << encode_capot(suit)) != 0 {
                best_ev = ev;
                best_action = encode_capot(suit);
            }
        }
    }

    // Evaluate COINCHE if legal
    if legal & (1u64 << 41) != 0 {
        let ev = self.evaluate_coinche(&results, state, team);
        if ev > best_ev {
            best_ev = ev;
            best_action = 41; // COINCHE
        }
    }

    best_action
}
```

### Scoring Functions

All scoring follows FFB rules (section 10.2): "points faits + points demandes".

```rust
/// Expected value of bidding `bid_value` in `suit`.
fn evaluate_bid(&self, results: &[[u8; 4]], suit: u8, bid_value: u8, team: u8) -> f32 {
    let contract = bid_value as i32 * 10; // 80, 90, ..., 160
    let n = results.len() as f32;
    let mut total = 0.0f32;

    for r in results {
        let ns_pts = r[suit as usize] as i32;
        let team_pts = if team == 0 { ns_pts } else { 162 - ns_pts };

        if team_pts >= contract {
            // Contract made: score = points_faits + points_demandes
            total += (team_pts + contract) as f32;
        } else {
            // Contract failed: opponent scores points_demandes
            total -= contract as f32;
        }
    }

    total / n
}

/// Expected value of passing (letting current contract stand).
fn evaluate_pass(&self, results: &[[u8; 4]], state: &GameState, team: u8) -> f32 {
    // If no active bid, pass EV = 0 (void deal, or leads to void deal)
    if state.last_bid_value == 0 {
        return 0.0;
    }

    let bid_suit = state.last_bid_suit;
    let bidder_team = GameState::player_team(state.last_bidder);
    let contract = state.last_bid_value as i32 * 10;
    // coinche_state: 0=none, 1=coinche, 2=surcoinche
    let coinche_mult = match state.coinche_state { 0 => 1, 1 => 2, _ => 4 };

    let n = results.len() as f32;
    let mut total = 0.0f32;

    for r in results {
        let ns_pts = r[bid_suit as usize] as i32;
        let bidder_pts = if bidder_team == 0 { ns_pts } else { 162 - ns_pts };

        if bidder_pts >= contract {
            // Contract made: bidder_team scores (pts_faits + demandes) * coinche
            let score = (bidder_pts + contract) * coinche_mult;
            if bidder_team == team {
                total += score as f32;   // Partner's contract succeeds
            } else {
                total -= score as f32;   // Opponent's contract succeeds
            }
        } else {
            // Contract failed: defending team scores demandes * coinche
            let score = contract * coinche_mult;
            if bidder_team == team {
                total -= score as f32;   // Partner's contract fails
            } else {
                total += score as f32;   // Opponent's contract fails
            }
        }
    }

    total / n
}

/// Expected value of coinching the opponent's contract.
/// Only callable when opponent has the current bid and coinche_state == 0.
fn evaluate_coinche(&self, results: &[[u8; 4]], state: &GameState, team: u8) -> f32 {
    let bid_suit = state.last_bid_suit;
    let opp_team = 1 - team; // opponent is the bidder
    let contract = state.last_bid_value as i32 * 10;
    let coinche_mult = 2; // coinche doubles

    let n = results.len() as f32;
    let mut total = 0.0f32;

    for r in results {
        let ns_pts = r[bid_suit as usize] as i32;
        let opp_pts = if opp_team == 0 { ns_pts } else { 162 - ns_pts };

        if opp_pts >= contract {
            total -= ((opp_pts + contract) * coinche_mult) as f32;
        } else {
            total += (contract * coinche_mult) as f32;
        }
    }

    total / n
}

/// Expected value of bidding capot in `suit`.
fn evaluate_capot(&self, results: &[[u8; 4]], suit: u8, team: u8) -> f32 {
    let n = results.len() as f32;
    let mut total = 0.0f32;

    for r in results {
        let ns_pts = r[suit as usize] as i32;
        let team_pts = if team == 0 { ns_pts } else { 162 - ns_pts };

        // Capot requires ALL tricks (252 pts with dix de der = 100)
        if team_pts >= 252 {
            total += (252 + 250) as f32; // pts_faits + capot_value
        } else {
            total -= 250.0; // Failed capot
        }
    }

    total / n
}
```

### decide_play Algorithm

Reuses IS-DD logic with BeliefState-filtered determinizations:

```rust
fn decide_play(&mut self, state: &GameState) -> u8 {
    let player = state.current_player;
    let team = GameState::player_team(player);
    let legal = state.legal_actions();

    // Generate determinizations
    let dets: Vec<GameState> = (0..self.config.num_dets * 3)
        .filter_map(|_| self.belief.determinize(state, &mut self.rng))
        .take(self.config.num_dets as usize)
        .collect();

    // Solve each determinization
    let mut score_sum = [0i64; 32];
    let mut score_count = [0u32; 32];

    for det in &dets {
        let scores = solve_with_scores(det, &mut self.tt_buf);
        for i in 0..scores.count {
            let (card, ns_pts) = scores.scores[i];
            let team_pts = if team == 0 { ns_pts } else { 162 - ns_pts as i16 };
            score_sum[card as usize] += team_pts as i64;
            score_count[card as usize] += 1;
        }
    }

    // Pick best card
    let mut best_card = 0u8;
    let mut best_avg = f32::NEG_INFINITY;

    for card in 0..32u8 {
        if legal & (1u64 << card) == 0 { continue; }
        if score_count[card as usize] == 0 { continue; }

        let avg = score_sum[card as usize] as f32 / score_count[card as usize] as f32;
        if avg > best_avg {
            best_avg = avg;
            best_card = card;
        }
    }

    best_card
}
```

## File Layout

**New files:**
- `colver-core/src/belief/belief_state.rs` — BeliefState struct, constraints, determinization
- `colver-core/src/bid/bis_dd.rs` — BisDdAgent, scoring functions, bid/play decision logic

**Modified files:**
- `colver-core/src/belief/mod.rs` — add `pub mod belief_state;`
- `colver-core/src/bid/mod.rs` — add `pub mod bis_dd;`
- `colver-core/src/bid/bid_eval/mod.rs` — add `BisDd` variant to `BidFunction` enum
- `colver-core/src/bin/arena.rs` — parse `strategy = "bis_dd"` and `method = "bis_dd"` in bot TOML

**Untouched:**
- `CardBeliefs` (existing bots depend on it)
- `SmartIsMctsSearch` (remains independent)
- Web frontend (arena-first)

## Testing Strategy

### Unit Tests

**BeliefState:**
- `record_bid` adds correct constraints and adjusts soft weights
- `determinize` produces hands consistent with bid constraints (bidder has eval >= threshold)
- `determinize` produces hands consistent with pass constraints (passer has no J+9 combo, eval < threshold)
- Play-phase inferences: void marking, trump ceiling carry over correctly
- Soft weights: bidder's trump honor weights are boosted, passer's are reduced

**BisDdAgent scoring:**
- `evaluate_bid`: known DD results → verify expected match points calculation
- `evaluate_pass`: opponent contract → verify chute/reussite scoring
- `evaluate_coinche`: verify doubling multiplier applied correctly
- `evaluate_capot`: verify 252-point threshold and scoring

**Integration:**
- Full deal lifecycle: init → bid phase → play phase → verify beliefs accumulated correctly
- Edge cases: void deal (4 passes), surcoinche, capot

### Arena Evaluation

Primary benchmark: H2H vs `nn_v2_isdd_no_belief` (current champion).

```bash
# Create arena/bots/bis_dd.toml
cargo run --bin arena --release -- h2h bis_dd nn_v2_isdd_no_belief --matches 200
```

Secondary benchmarks:
- vs `nn_v2_dmc35` (NN bid + DMC play)
- vs `nn_isdd` (NN bid + IS-DD + belief)

### Performance Targets

| Operation | Target | Budget breakdown |
|-----------|--------|-----------------|
| Bid decision (opening) | < 2s | 20 dets × 4 suits × ~15ms/solve = 1.2s |
| Bid decision (response) | < 1s | 20 dets × 2 suits × ~15ms/solve = 0.6s |
| Coinche decision | < 0.5s | 20 dets × 1 suit × ~15ms/solve = 0.3s |
| Play decision | < 0.5s | 20 dets × ~15ms/solve = 0.3s |
| Determinization (with rejection) | < 1ms | ~50 attempts × ~0.5us = 25us |

## Design Decisions Log

1. **Rejection sampling over weighted-only**: Benchmarks show 50% acceptance at threshold>=10 with uniform sampling, 77% with 3x bias. Both are fast enough (<1ms for 20 samples). Hard rejection gives exact constraint satisfaction; soft weights improve acceptance rate.

2. **No explicit pass constraints on multiple players simultaneously**: Benchmarked at 0.0% acceptance rate. Instead, bid constraints on the bidder naturally weaken other players in that suit. Pass constraints are applied individually and softly.

3. **Pass heuristics from BID_RULES.md**: Opening pass = no J+9 in any suit + max eval < 10. Response pass = can't overbid. These are soft enough to maintain reasonable acceptance rates when combined.

4. **BeliefState as separate component (not merged into CardBeliefs)**: Allows independent testing, doesn't break existing bots, can be adopted by other strategies later.

5. **Scoring uses FFB "points faits + demandes"**: Not just "can we make the contract" but "what's the expected match point gain/loss". This naturally handles the tradeoff between bidding higher vs letting opponents fail.

6. **Capot evaluation included**: DD solver can detect 8-trick sweeps. Capot is rare but worth +502 points when it hits.
