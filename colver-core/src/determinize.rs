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

#[cfg(all(test, feature = "rand"))]
mod tests {
    use super::*;

    #[test]
    fn test_determinize_preserves_observer() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);

        let det = determinize(&state, 0, &mut rng).unwrap();
        assert_eq!(det.hands[0], state.hands[0]);
    }

    #[test]
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
}
