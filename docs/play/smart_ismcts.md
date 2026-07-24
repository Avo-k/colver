# Smart IS-MCTS: Belief-Weighted Information Set MCTS

This document explains how the Smart IS-MCTS agent reasons about hidden information in Belote Contree.

## The Problem

In Belote Contree, each player sees only their own 8 cards. The naive IS-MCTS agent handles this by sampling random "determinized worlds" — possible assignments of the 24 unknown cards to the 3 other players — and running standard MCTS on each. But these worlds are sampled uniformly, ignoring all the information leaked by bidding and play. A player who bid 80 Hearts probably has the Jack of Hearts. A player who didn't follow suit is definitely void in that suit. The smart agent exploits these signals.

## Architecture

```
                    GameState (56 bytes, Copy, unchanged)
                         |
                    CardBeliefs
                   weights[4][32]  (512 bytes)
                    /          \
          Hard constraints    Soft constraints
          (definitive)        (probabilistic)
                    \          /
               normalized_weights()
                         |
               determinize_weighted()
                         |
              Standard MCTS on each world
                         |
              Aggregate visit counts → best action
```

`GameState` stays untouched at 56 bytes. All belief state lives in `CardBeliefs`, a `[[f32; 32]; 4]` weight matrix where `weights[player][card]` represents the unnormalized probability that `player` holds `card`. The matrix starts uniform (1.0 for all eligible player-card pairs) and is updated after every action in the deal.

## Belief Updates

### Hard Constraints (weight = 0.0, definitive)

These are 100% certain deductions from the rules of Belote Contree. When triggered, the weight is set to zero — the card is impossible for that player.

**1. Suit void.** If a player doesn't follow the led suit, they have zero cards in that suit. All 8 ranks of that suit are zeroed for that player.

**2. Trump void from discard.** If a player can't follow suit AND plays a non-trump card AND their partner is not currently winning the trick, the rules require them to trump if they have any. So they must be void in trump too. Both the led suit and trump suit are zeroed.

**3. Trump ceiling.** If a player plays trump but doesn't overtrump the highest trump on the table, they have no trump stronger than that table trump. Specifically, for each rank with `TRUMP_STRENGTH[rank] > TRUMP_STRENGTH[table_best]`, weight is zeroed. Example: if the table has the Ace of trump (strength 5) and a player undertrumps, they cannot have the 9 (strength 6) or Jack (strength 7).

**4. Played cards.** Every card that has been played is zeroed for all players.

**5. Observer's hand.** The observer knows their own hand exactly. Their cards have weight 1.0; all other players have weight 0.0 for those cards.

### Soft Constraints (multiplicative factors, probabilistic)

These assume competent play. Weights are multiplied by factors > 1.0 (more likely) or < 1.0 (less likely). They are only applied when `use_soft_inference = true`.

#### Bidding Inference

| Action | What it implies | Weight adjustments |
|---|---|---|
| **Bid 80 in suit S** | Likely has strong trump | J of S: x5.0, 9 of S: x3.0, A of S: x2.0 |
| **Higher bids (90-160)** | Stronger hand | J scales 5.0→12.0, 9 scales 3.0→8.0, A scales 2.0→5.0 |
| **Capot (250)** | Very strong hand | J: x15, 9: x10, A: x8, side Aces: x4 |
| **Pass** | Lacks strong trump in every suit | J: x0.6, 9: x0.7 (all four suits) |
| **Coinche** | Strong defense against opponent's trump | J of their trump: x3.0, 9: x2.5, side Aces: x2.0 |
| **Surcoinche** | Very confident in declared trump | All trump: x2.0, J: x3.0 total, 9: x3.0 total |

The bid level scaling is linear: for a bid of value V (where V/10 = 8 for 80, 9 for 90, ..., 16 for 160), the factor for the Jack is `5.0 + (V/10 - 8) * 0.875`. This means bidding 160 makes the Jack ~12x more likely for that player.

#### Play Inference

| Observation | What it implies | Weight adjustments |
|---|---|---|
| **Led Ace of suit X** | Strong holding in X | Player: 10 of X x2.0, K of X x1.5 |
| **Led trump** | Drawing trumps, strong trump holding | Player: remaining trump x1.5 |
| **Led low card (7/8)** | Weak in that suit | Others: A x1.2, 10 x1.2 |
| **Cut with low trump (7/8/Q)** | Likely lacks stronger trump | Others: J x1.3, 9 x1.3 |
| **Discarded non-trump non-lead** | Shedding a weak suit | Others: A x1.2, 10 x1.2 of discarded suit |

## Weighted Determinization

The `determinize_weighted()` function in `determinize.rs` samples a complete hand assignment biased by belief weights:

1. **Collect unknown cards** (not in observer's hand, not played).
2. **Sort by constraint tightness** — cards with fewer eligible players are assigned first. This reduces the chance of painting yourself into a corner.
3. **For each card**, pick a player with probability proportional to `weight[player][card] * remaining_slots[player]`. The `remaining_slots` factor ensures players who still need more cards are favored.
4. **Retry** up to 50 times if the assignment fails (e.g., a player ends up with the wrong number of cards).
5. **Fallback** to `determinize_greedy()` (uniform, void-respecting) if weighted sampling fails.

## Search Flow

```rust
// Initialize at deal start (one instance per player)
let mut search = SmartIsMctsSearch::new();
search.init_deal(&state, observer_player, true);

// Game loop — ALL players' actions must be recorded
loop {
    let state_before = state;
    let action = if my_turn {
        search.search(&state, &config, &mut rng)
    } else {
        opponent_action()
    };
    search.record_action(&state_before, player, action);
    state.step(action);
}
```

Each call to `search()`:
1. Computes `normalized_weights()` from the current belief state
2. Samples `D` determinized worlds via `determinize_weighted()`
3. Runs `I` MCTS iterations on each world (reusing the arena-based `MctsSearch`)
4. Aggregates root visit counts across all D worlds
5. Returns the most-visited action

Both teammates should each have their own `SmartIsMctsSearch` instance (different observer, different known hand), but both must observe all actions to keep their beliefs consistent.

## Configuration

```rust
SmartIsMctsConfig {
    determinizations: 20,     // worlds to sample
    iterations_per_det: 50,   // MCTS iterations per world
    exploration: sqrt(2),     // UCB1 exploration constant
    use_soft_inference: true, // enable probabilistic inference
}
```

Total search budget = `determinizations * iterations_per_det`. Default is 20 x 50 = 1000 iterations per decision.

## Bidding Strategy (`bid/bid_eval/`)

Smart IS-MCTS is paired with `smart_bid`, a convention-based bidding strategy that uses J/9 signaling — the same conventions human Contrée players use. This creates a natural synergy: the bidding leaks information that the belief model can exploit.

### Convention Summary

- **Opening**: J+9 → 80-100 (scaled by hand strength). J XOR 9 + 3 trumps → 80 (signals missing honor to partner). 2+ aces → 80 ("aux as").
- **Partner response**: On partner's 80, respond 90 if holding the missing J/9 honor. On partner's 90+ (meaning they have J+9), PASS — don't escalate.
- **Overcall**: J+9 in a different suit with score ≥ 14 → bid up to 100 max. Never overcall above 100.
- **Coinche**: J+9 in opponent's trump, 4+ trumps in their suit, or 3+ trumps + side ace on high bids (≥120).

### Design Principles

1. **One-shot communication**: Partner responses are limited to a single raise (80→90). No escalation spirals.
2. **Conservative overcalls**: Capped at 100. If opponents bid 100+, let them have it rather than start a bidding war.
3. **Achievable contracts**: Avg bid ~88, with 78% achievable even against perfect-info defense (vs 72% for score-based heuristic at avg bid ~117).

### Synergy with Beliefs

The bidding conventions directly feed into `CardBeliefs`:
- A player who bids 80 Hearts gets Jack-of-Hearts weight boosted 5x (soft inference)
- A partner who responds 90 to an 80 bid reveals they hold the missing J or 9
- A player who passes gets J/9 weights reduced across all suits

This creates a feedback loop: structured bidding → better beliefs → better determinization → better play.

## What's Not Modeled

- **Card counting** beyond void tracking (e.g., "only 2 hearts remain unplayed")
- **Opponent modeling** (adjusting beliefs based on opponent skill level)
- **Memory across deals** (each deal starts with fresh beliefs)
- **Belote/rebelote signals** (holding Q+K of trump)

These are potential areas for future improvement.
