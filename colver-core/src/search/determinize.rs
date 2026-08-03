#[cfg(feature = "rand")]
use rand::seq::SliceRandom;
#[cfg(feature = "rand")]
use rand::Rng;

use crate::card::*;
use crate::play::{belote_facts, BeloteFacts};
use crate::state::GameState;

/// Cartes que la belote place d'office chez un siège caché, et ce qu'il reste à
/// tirer une fois qu'elles sont posées.
///
/// Une carte forcée n'est pas une préférence : elle sort du tirage et le siège
/// concerné a une place de moins. Les trois déterminisations partagent ce
/// pré-calcul pour ne pas se contredire.
#[cfg(feature = "rand")]
struct ForcedCards {
    facts: BeloteFacts,
    /// `hands[p]` de départ : cartes déjà attribuées avant tout tirage.
    forced: [CardSet; 4],
    /// Cartes encore à répartir.
    unknown: CardSet,
    /// Places restantes par siège (0 pour l'observateur).
    remaining: [u8; 4],
}

#[cfg(feature = "rand")]
impl ForcedCards {
    fn new(state: &GameState, observer: u8, unknown_set: CardSet) -> Self {
        let facts = belote_facts(state);
        let mut fc = ForcedCards {
            facts,
            forced: [0; 4],
            unknown: unknown_set,
            remaining: [0; 4],
        };
        for p in 0..4usize {
            fc.remaining[p] = card_count(state.hands[p]) as u8;
        }
        fc.remaining[observer as usize] = 0;
        if facts.is_empty() {
            return fc;
        }
        for p in 0..4usize {
            if p == observer as usize {
                continue;
            }
            let held = facts.held[p] & unknown_set;
            if held == 0 {
                continue;
            }
            fc.forced[p] = held;
            fc.unknown &= !held;
            fc.remaining[p] = fc.remaining[p].saturating_sub(card_count(held) as u8);
        }
        fc
    }

    /// `p` peut-il recevoir `card` ?
    #[inline]
    fn allows(&self, p: u8, card: Card) -> bool {
        self.facts.banned[p as usize] & card_to_bit(card) == 0
    }
}

/// Create a determinized state from the perspective of `observer`.
///
/// Redistributes unknown cards among other players, respecting:
/// - Correct card count per player
/// - Known void constraints
/// - The belote announcement (see [`belote_facts`])
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

    let fc = ForcedCards::new(state, observer, unknown_set);

    // Collect unknown cards
    let mut unknown_cards: Vec<Card> = Vec::with_capacity(24);
    for card in CardIter(fc.unknown) {
        unknown_cards.push(card);
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
            hands[p as usize] |= fc.forced[p as usize];
            let count = fc.remaining[p as usize] as usize;
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
                if void_mask & (1 << suit) != 0 || !fc.allows(p, card) {
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
/// Uses a greedy approach: assign cards to players, respecting voids and the
/// belote announcement (see [`belote_facts`]).
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

    let fc = ForcedCards::new(state, observer, unknown_set);

    // Group unknown cards by suit
    let mut suit_cards: [Vec<Card>; 4] = [Vec::new(), Vec::new(), Vec::new(), Vec::new()];
    for card in CardIter(fc.unknown) {
        suit_cards[card_suit_u8(card) as usize].push(card);
    }

    // Shuffle within each suit
    for suit in &mut suit_cards {
        suit.shuffle(rng);
    }

    let mut hands = fc.forced;
    hands[observer as usize] = state.hands[observer as usize];

    let mut remaining = fc.remaining; // cards still needed per player

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

            // Pick a random eligible player weighted by remaining count.
            // A card the belote excludes for a seat is skipped here, not
            // corrected afterwards: the draw must not be able to produce it.
            let total_remaining: u16 = eligible
                .iter()
                .filter(|&&p| fc.allows(p, card))
                .map(|&p| remaining[p as usize] as u16)
                .sum();
            if total_remaining == 0 {
                unassigned.push(card);
                continue;
            }

            let mut r = rng.gen_range(0..total_remaining);
            let mut chosen = u8::MAX;
            for &p in &eligible {
                if !fc.allows(p, card) {
                    continue;
                }
                let rem = remaining[p as usize] as u16;
                if r < rem {
                    chosen = p;
                    break;
                }
                r -= rem;
            }
            debug_assert_ne!(chosen, u8::MAX, "total_remaining > 0 garantit un tirage");
            if chosen == u8::MAX {
                unassigned.push(card);
                continue;
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
///
/// The belote announcement is applied on top of `weights`, whatever they say:
/// a caller passing NN beliefs must not be able to hand a card to a seat the
/// rules have excluded.
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

    let fc = ForcedCards::new(state, observer, unknown_set);

    // Applying the facts to a copy keeps the rest of the loop untouched: a
    // forced card ends up with a single eligible seat, which the tightness sort
    // then places first.
    let mut owned_weights;
    let weights = if fc.facts.is_empty() {
        weights
    } else {
        owned_weights = *weights;
        for p in 0..4usize {
            for card in CardIter(fc.facts.banned[p]) {
                owned_weights[p][card as usize] = 0.0;
            }
        }
        &owned_weights
    };

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
    use crate::state::{Contract, Phase};

    /// Atout = pique : Dame = 4, Roi = 5.
    const QS: Card = 4;
    const KS: Card = 5;

    fn cards(list: &[Card]) -> CardSet {
        list.iter().fold(0, |acc, &c| acc | card_to_bit(c))
    }

    /// Le siège 1 entame atout ; à lui de tenir (ou non) l'autre honneur.
    fn state_after_king_of_trump(hands: [CardSet; 4]) -> GameState {
        let mut state = GameState::new(0, hands);
        state.phase = Phase::Playing;
        state.contract = Contract { trump: 0, value: 8, team: 0, coinche: 0 };
        state.trick_lead = 1;
        state.current_player = 1;
        crate::play::apply_play(&mut state, KS);
        state
    }

    /// Poids uniformes sur les cartes inconnues, comme un appelant sans croyances.
    fn flat_weights(state: &GameState, observer: u8) -> [[f32; 32]; 4] {
        let mut w = [[0.0f32; 32]; 4];
        for card in 0..32u8 {
            let bit = card_to_bit(card);
            if state.played_cards & bit != 0 {
                continue;
            }
            if state.hands[observer as usize] & bit != 0 {
                w[observer as usize][card as usize] = 1.0;
                continue;
            }
            for p in 0..4u8 {
                if p != observer && state.voids[p as usize] & (1 << card_suit_u8(card)) == 0 {
                    w[p as usize][card as usize] = 1.0;
                }
            }
        }
        w
    }

    #[test]
    fn every_determinizer_places_an_announced_belote_at_its_announcer() {
        let state = state_after_king_of_trump([
            cards(&[0, 1, 2, 3, 6, 7, 14, 15]),
            cards(&[QS, KS, 8, 9, 10, 11, 12, 13]),
            cards(&[16, 17, 18, 19, 20, 21, 22, 23]),
            cards(&[24, 25, 26, 27, 28, 29, 30, 31]),
        ]);
        assert_eq!(state.belote[1], 1, "le siège 1 a annoncé");

        let mut rng = rand::thread_rng();
        let weights = flat_weights(&state, 0);
        let mut drawn = 0;
        for _ in 0..200 {
            for (name, world) in [
                ("determinize", determinize(&state, 0, &mut rng)),
                ("greedy", determinize_greedy(&state, 0, &mut rng)),
                ("weighted", determinize_weighted(&state, 0, &weights, &mut rng)),
            ] {
                let Some(world) = world else { continue };
                drawn += 1;
                assert_ne!(
                    world.hands[1] & card_to_bit(QS),
                    0,
                    "{name} a déplacé une Dame d'atout annoncée"
                );
            }
        }
        assert!(drawn > 100, "trop peu de mondes tirés ({drawn})");
    }

    #[test]
    fn every_determinizer_keeps_the_unannounced_honour_away() {
        let state = state_after_king_of_trump([
            cards(&[0, 1, 2, 3, 6, 7, 15, 23]),
            cards(&[KS, 8, 9, 10, 11, 12, 13, 14]),
            cards(&[QS, 16, 17, 18, 19, 20, 21, 22]),
            cards(&[24, 25, 26, 27, 28, 29, 30, 31]),
        ]);
        assert_eq!(state.belote, [0, 0], "Roi d'atout posé sans annonce");

        let mut rng = rand::thread_rng();
        let weights = flat_weights(&state, 0);
        let mut drawn = 0;
        for _ in 0..200 {
            for (name, world) in [
                ("determinize", determinize(&state, 0, &mut rng)),
                ("greedy", determinize_greedy(&state, 0, &mut rng)),
                ("weighted", determinize_weighted(&state, 0, &weights, &mut rng)),
            ] {
                let Some(world) = world else { continue };
                drawn += 1;
                assert_eq!(
                    world.hands[1] & card_to_bit(QS),
                    0,
                    "{name} a donné la Dame à un siège qui n'a pas annoncé"
                );
            }
        }
        assert!(drawn > 100, "trop peu de mondes tirés ({drawn})");
    }

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
