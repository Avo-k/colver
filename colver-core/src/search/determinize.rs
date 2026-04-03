#[cfg(feature = "rand")]
use rand::seq::SliceRandom;
#[cfg(feature = "rand")]
use rand::Rng;

use crate::card::*;
use crate::state::GameState;

/// Create a determinized state from the perspective of `observer`.
///
/// Redistributes unknown cards among other players, respecting:
/// - Correct card count per player
/// - Known void constraints
/// - Observer's hand remains unchanged
///
/// Uses rejection sampling with constraint-aware shuffling.
#[cfg(feature = "rand")]
pub fn determinize(state: &GameState, observer: u8, rng: &mut impl Rng) -> Option<GameState> {
    let mut new_state = *state;

    // Cards known to the observer: their own hand + all played cards
    let known = state.hands[observer as usize] | state.played_cards;
    let unknown_set = ALL_CARDS ^ known;

    if unknown_set == 0 {
        return Some(new_state); // Nothing to redistribute
    }

    // Collect unknown cards
    let mut unknown_cards: Vec<Card> = Vec::with_capacity(24);
    for card in CardIter(unknown_set) {
        unknown_cards.push(card);
    }

    // How many cards each other player should have
    let mut target_counts = [0u8; 4];
    for p in 0..4u8 {
        target_counts[p as usize] = card_count(state.hands[p as usize]) as u8;
    }

    // Attempt redistribution with rejection sampling
    for _attempt in 0..100 {
        unknown_cards.shuffle(rng);

        let mut hands = [0u32; 4];
        hands[observer as usize] = state.hands[observer as usize];

        let mut idx = 0;
        let mut valid = true;

        for p in 0..4u8 {
            if p == observer {
                continue;
            }
            let count = target_counts[p as usize] as usize;
            let void_mask = state.voids[p as usize];

            for _ in 0..count {
                if idx >= unknown_cards.len() {
                    valid = false;
                    break;
                }
                let card = unknown_cards[idx];
                idx += 1;

                // Check void constraint
                let suit = card_suit_u8(card);
                if void_mask & (1 << suit) != 0 {
                    valid = false;
                    break;
                }

                hands[p as usize] |= card_to_bit(card);
            }
            if !valid {
                break;
            }
        }

        if valid {
            new_state.hands = hands;
            return Some(new_state);
        }
    }

    None // Failed after max attempts
}

/// Constraint-aware determinization that avoids rejection sampling.
/// Uses a greedy approach: assign cards to players, respecting voids.
#[cfg(feature = "rand")]
pub fn determinize_greedy(
    state: &GameState,
    observer: u8,
    rng: &mut impl Rng,
) -> Option<GameState> {
    let mut new_state = *state;
    let known = state.hands[observer as usize] | state.played_cards;
    let unknown_set = ALL_CARDS ^ known;

    if unknown_set == 0 {
        return Some(new_state);
    }

    // Group unknown cards by suit
    let mut suit_cards: [Vec<Card>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for card in CardIter(unknown_set) {
        suit_cards[card_suit_u8(card) as usize].push(card);
    }

    // Shuffle within each suit
    for suit in &mut suit_cards {
        suit.shuffle(rng);
    }

    let mut hands = [0u32; 4];
    hands[observer as usize] = state.hands[observer as usize];

    let mut remaining = [0u8; 4]; // cards still needed per player
    for p in 0..4u8 {
        remaining[p as usize] = card_count(state.hands[p as usize]) as u8;
    }
    remaining[observer as usize] = 0; // observer's hand is fixed

    // For each suit, distribute cards to players who are not void in that suit
    // This is a simplified approach — for complex void patterns, may need backtracking
    let mut unassigned: Vec<Card> = Vec::new();

    for suit_idx in 0..4u8 {
        let cards = &suit_cards[suit_idx as usize];
        let mut eligible: Vec<u8> = Vec::new();

        for p in 0..4u8 {
            if p == observer {
                continue;
            }
            if remaining[p as usize] > 0 && (state.voids[p as usize] & (1 << suit_idx)) == 0 {
                eligible.push(p);
            }
        }

        for &card in cards {
            if eligible.is_empty() {
                unassigned.push(card);
                continue;
            }

            // Pick a random eligible player weighted by remaining count
            let total_remaining: u16 = eligible
                .iter()
                .map(|&p| remaining[p as usize] as u16)
                .sum();
            if total_remaining == 0 {
                unassigned.push(card);
                continue;
            }

            let mut r = rng.gen_range(0..total_remaining);
            let mut chosen = eligible[0];
            for &p in &eligible {
                let rem = remaining[p as usize] as u16;
                if r < rem {
                    chosen = p;
                    break;
                }
                r -= rem;
            }

            hands[chosen as usize] |= card_to_bit(card);
            remaining[chosen as usize] -= 1;

            // Remove player from eligible if they're full
            if remaining[chosen as usize] == 0 {
                eligible.retain(|&p| p != chosen);
            }
        }
    }

    // Handle any unassigned cards (players who are void in all remaining suits?)
    if !unassigned.is_empty() {
        return None; // Constraints are unsatisfiable
    }

    // Verify all players got the right number of cards
    for p in 0..4u8 {
        if card_count(hands[p as usize]) as u8 != card_count(state.hands[p as usize]) as u8 {
            return None;
        }
    }

    new_state.hands = hands;
    Some(new_state)
}

/// Weighted determinization that biases card assignment by belief weights.
///
/// For each unknown card, chooses a player with probability proportional to
/// `weights[player][card] * remaining_slots[player]`. Cards are processed
/// in order of constraint tightness (fewest eligible players first).
///
/// Falls back after `max_retries` failed attempts.
#[cfg(feature = "rand")]
pub fn determinize_weighted(
    state: &GameState,
    observer: u8,
    weights: &[[f32; 32]; 4],
    rng: &mut impl Rng,
) -> Option<GameState> {
    let known = state.hands[observer as usize] | state.played_cards;
    let unknown_set = ALL_CARDS ^ known;

    if unknown_set == 0 {
        return Some(*state);
    }

    // Collect unknown cards with their constraint tightness
    let mut unknown_cards: Vec<(Card, u8)> = Vec::with_capacity(24);
    for card in CardIter(unknown_set) {
        let suit_idx = card_suit_u8(card);
        let mut eligible = 0u8;
        for p in 0..4u8 {
            if p == observer {
                continue;
            }
            if state.voids[p as usize] & (1 << suit_idx) == 0
                && weights[p as usize][card as usize] > 0.0
            {
                eligible += 1;
            }
        }
        unknown_cards.push((card, eligible));
    }

    // Sort by tightness: fewest eligible players first
    unknown_cards.sort_unstable_by_key(|&(_, e)| e);

    let mut target_counts = [0u8; 4];
    for p in 0..4u8 {
        target_counts[p as usize] = card_count(state.hands[p as usize]) as u8;
    }
    target_counts[observer as usize] = 0;

    for _attempt in 0..50 {
        let mut hands = [0u32; 4];
        hands[observer as usize] = state.hands[observer as usize];
        let mut remaining = target_counts;
        let mut valid = true;

        for &(card, _) in &unknown_cards {
            let suit_idx = card_suit_u8(card);

            // Compute weighted probabilities for each eligible player
            let mut probs = [0.0f32; 4];
            let mut total = 0.0f32;

            for p in 0..4u8 {
                if p == observer || remaining[p as usize] == 0 {
                    continue;
                }
                if state.voids[p as usize] & (1 << suit_idx) != 0 {
                    continue;
                }
                let w = weights[p as usize][card as usize];
                if w <= 0.0 {
                    continue;
                }
                let prob = w * remaining[p as usize] as f32;
                probs[p as usize] = prob;
                total += prob;
            }

            if total <= 0.0 {
                valid = false;
                break;
            }

            // Sample a player
            let mut r = rng.gen::<f32>() * total;
            let mut chosen = 255u8;
            for p in 0..4u8 {
                if probs[p as usize] > 0.0 {
                    r -= probs[p as usize];
                    if r <= 0.0 {
                        chosen = p;
                        break;
                    }
                }
            }
            if chosen == 255 {
                // Floating point edge case - pick last eligible
                for p in (0..4u8).rev() {
                    if probs[p as usize] > 0.0 {
                        chosen = p;
                        break;
                    }
                }
            }
            if chosen == 255 {
                valid = false;
                break;
            }

            hands[chosen as usize] |= card_to_bit(card);
            remaining[chosen as usize] -= 1;
        }

        if valid && remaining.iter().all(|&r| r == 0) {
            let mut new_state = *state;
            new_state.hands = hands;
            return Some(new_state);
        }
    }

    None
}

#[cfg(all(test, feature = "rand"))]
mod tests {
    use super::*;

    #[test]
    #[ignore]
    fn test_determinize_preserves_observer() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);

        let det = determinize(&state, 0, &mut rng).unwrap();
        assert_eq!(det.hands[0], state.hands[0]);
    }

    #[test]
    #[ignore]
    fn test_determinize_correct_card_counts() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let state = GameState::deal_random(0, &mut rng);
            let det = determinize(&state, 0, &mut rng).unwrap();

            for p in 0..4 {
                assert_eq!(
                    card_count(det.hands[p]),
                    card_count(state.hands[p]),
                    "Player {} card count mismatch",
                    p
                );
            }

            // All cards accounted for
            let all = det.hands[0] | det.hands[1] | det.hands[2] | det.hands[3];
            assert_eq!(all, ALL_CARDS);

            // No overlap
            for i in 0..4 {
                for j in (i + 1)..4 {
                    assert_eq!(det.hands[i] & det.hands[j], 0);
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn test_determinize_respects_voids() {
        let mut rng = rand::thread_rng();
        let mut state = GameState::deal_random(0, &mut rng);

        // Mark player 1 as void in spades
        state.voids[1] = 1 << 0; // void in suit 0 (spades)

        // Need to remove spades from P1 and give to others
        // This is a bit tricky to set up perfectly, so let's just test the
        // determinization output respects the constraint
        for _ in 0..50 {
            if let Some(det) = determinize(&state, 0, &mut rng) {
                // Player 1 should have no spades
                assert_eq!(
                    cards_in_suit(det.hands[1], Suit::Spades),
                    0,
                    "Player 1 should be void in spades after determinization"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn test_determinize_greedy() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let state = GameState::deal_random(0, &mut rng);
            if let Some(det) = determinize_greedy(&state, 0, &mut rng) {
                assert_eq!(det.hands[0], state.hands[0]);
                for p in 0..4 {
                    assert_eq!(card_count(det.hands[p]), card_count(state.hands[p]));
                }
            }
        }
    }

    #[test]
    #[ignore]
    fn test_determinize_weighted_preserves_observer() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);

        // Uniform weights
        let mut weights = [[1.0f32; 32]; 4];
        // Zero out observer's unknown cards
        for card in 0..32 {
            if state.hands[0] & card_to_bit(card) != 0 {
                for p in 1..4 {
                    weights[p][card as usize] = 0.0;
                }
            } else {
                weights[0][card as usize] = 0.0;
            }
        }

        let det = determinize_weighted(&state, 0, &weights, &mut rng).unwrap();
        assert_eq!(det.hands[0], state.hands[0]);
    }

    #[test]
    #[ignore]
    fn test_determinize_weighted_correct_counts() {
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let state = GameState::deal_random(0, &mut rng);

            // Use uniform weights
            let mut weights = [[0.0f32; 32]; 4];
            for card in 0..32u8 {
                let bit = card_to_bit(card);
                if state.played_cards & bit != 0 {
                    continue;
                }
                if state.hands[0] & bit != 0 {
                    weights[0][card as usize] = 1.0;
                } else {
                    for p in 1..4 {
                        let suit_idx = card_suit_u8(card);
                        if state.voids[p] & (1 << suit_idx) == 0 {
                            weights[p][card as usize] = 1.0;
                        }
                    }
                }
            }

            if let Some(det) = determinize_weighted(&state, 0, &weights, &mut rng) {
                for p in 0..4 {
                    assert_eq!(
                        card_count(det.hands[p]),
                        card_count(state.hands[p]),
                        "Player {} card count mismatch",
                        p
                    );
                }
                // All cards accounted for
                let all = det.hands[0] | det.hands[1] | det.hands[2] | det.hands[3];
                assert_eq!(all, ALL_CARDS);
            }
        }
    }

    #[test]
    #[ignore]
    fn test_determinize_weighted_respects_voids() {
        let mut rng = rand::thread_rng();
        let state = {
            let mut s = GameState::deal_random(0, &mut rng);
            s.voids[1] = 1 << 0; // P1 void in spades
            s
        };

        let mut weights = [[0.0f32; 32]; 4];
        for card in 0..32u8 {
            let bit = card_to_bit(card);
            if state.hands[0] & bit != 0 {
                weights[0][card as usize] = 1.0;
            } else {
                for p in 1..4 {
                    let suit_idx = card_suit_u8(card);
                    if state.voids[p] & (1 << suit_idx) == 0 {
                        weights[p][card as usize] = 1.0;
                    }
                }
            }
        }

        for _ in 0..50 {
            if let Some(det) = determinize_weighted(&state, 0, &weights, &mut rng) {
                assert_eq!(
                    cards_in_suit(det.hands[1], Suit::Spades),
                    0,
                    "Player 1 should be void in spades"
                );
            }
        }
    }

    #[test]
    #[ignore]
    fn test_determinize_weighted_bias() {
        let mut rng = rand::thread_rng();

        // Simple setup: 3 unknown cards, 3 players with 1 card each
        // Give high weight for card X to player 1
        let state = GameState::deal_random(0, &mut rng);
        // We can't easily control the exact setup, but let's verify
        // the function runs and produces valid results with biased weights
        let mut weights = [[0.0f32; 32]; 4];
        for card in 0..32u8 {
            let bit = card_to_bit(card);
            if state.hands[0] & bit != 0 {
                weights[0][card as usize] = 1.0;
            } else {
                // Give P1 much higher weight for hearts
                let suit_idx = card_suit_u8(card);
                for p in 1..4 {
                    if state.voids[p] & (1 << suit_idx) == 0 {
                        if p == 1 && suit_idx == 1 {
                            weights[p][card as usize] = 10.0; // heavily favor P1 for hearts
                        } else {
                            weights[p][card as usize] = 1.0;
                        }
                    }
                }
            }
        }

        let mut p1_hearts_total = 0u32;
        let trials = 200;
        for _ in 0..trials {
            if let Some(det) = determinize_weighted(&state, 0, &weights, &mut rng) {
                p1_hearts_total += card_count(cards_in_suit(det.hands[1], Suit::Hearts));
            }
        }

        // P1 should get more hearts on average than uniform (uniform ~2.67 of 8)
        let avg = p1_hearts_total as f64 / trials as f64;
        assert!(
            avg > 3.0,
            "P1 should get more hearts with biased weights, got avg {}",
            avg
        );
    }
}
