use crate::bid_eval::evaluate_for_trump;
use crate::card::*;

/// Maps an encoded bid value (value/10, e.g. 8=80, 9=90, ..., 16=160)
/// to the minimum `evaluate_for_trump` score that would justify that bid.
/// This is the inverse of `score_to_bid_value` from eval_helpers.
pub fn bid_value_to_threshold(bid_value: u8) -> u16 {
    match bid_value {
        8 => 10,  // 80  requires score >= 10
        9 => 14,  // 90  requires score >= 14
        10 => 17, // 100 requires score >= 17
        11 => 20, // 110 requires score >= 20
        12 => 23, // 120 requires score >= 23
        13 => 26, // 130 requires score >= 26
        14 => 29, // 140
        15 => 32, // 150
        16 => 35, // 160
        25 => 35, // capot — same as 160
        _ => 10,  // fallback
    }
}

/// What kind of constraint an observed action imposes on a player's hand.
#[derive(Clone, Debug)]
pub enum ConstraintKind {
    /// Player bid `suit` at a level implying `evaluate_for_trump >= min_score`.
    Bid {
        suit: Suit,
        min_score: u16,
    },
    /// Player passed. The constraint logic depends on context:
    /// - Opening pass (min_overbid_value == 0): reject hands with J+9 in any suit,
    ///   or any suit scoring >= threshold (10 normally, 8 if position >= 2).
    /// - Pass after a bid with partner bidding: only check active_suit < threshold.
    /// - Pass after a bid without partner: check ALL suits < threshold.
    Pass {
        /// The minimum bid value (encoded /10) that would have been needed to overbid.
        /// 0 means this was an opening pass (no bid on the table yet).
        min_overbid_value: u8,
        /// Seat position in the auction (0-based from dealer).
        auction_position: u8,
        /// Whether this player's partner had already placed a bid.
        partner_had_bid: bool,
        /// The suit of the current active bid (only meaningful when min_overbid_value > 0).
        active_suit: Suit,
    },
}

/// A constraint derived from an observed action, tied to a specific player.
#[derive(Clone, Debug)]
pub struct ActionConstraint {
    pub player: u8,
    pub kind: ConstraintKind,
}

impl ActionConstraint {
    /// Check whether the given hands satisfy this constraint.
    ///
    /// Returns `true` if the player's hand is consistent with having taken
    /// the observed action.
    pub fn is_satisfied(&self, hands: &[CardSet; 4]) -> bool {
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
                if *min_overbid_value == 0 {
                    // Opening pass: reject if any suit has J+9 combo
                    for suit_idx in 0..4u8 {
                        let suit = Suit::from_u8(suit_idx);
                        let bits = suit_bits(hand, suit);
                        // J = rank 3, 9 = rank 2
                        if bits & (1 << 3) != 0 && bits & (1 << 2) != 0 {
                            return false;
                        }
                    }
                    // Reject if any suit has evaluate_for_trump >= threshold
                    let threshold = if *auction_position >= 2 { 8 } else { 10 };
                    for suit_idx in 0..4u8 {
                        let suit = Suit::from_u8(suit_idx);
                        if evaluate_for_trump(hand, suit) >= threshold {
                            return false;
                        }
                    }
                    true
                } else if *partner_had_bid {
                    // Pass after partner bid: only check active suit
                    let threshold = bid_value_to_threshold(*min_overbid_value);
                    evaluate_for_trump(hand, *active_suit) < threshold
                } else {
                    // Pass after opponent bid (no partner bid): check ALL suits
                    let threshold = bid_value_to_threshold(*min_overbid_value);
                    for suit_idx in 0..4u8 {
                        let suit = Suit::from_u8(suit_idx);
                        if evaluate_for_trump(hand, suit) >= threshold {
                            return false;
                        }
                    }
                    true
                }
            }
        }
    }
}

/// Tracks belief weights and action constraints for informed determinization.
///
/// `BeliefState` accumulates constraints from observed bids and plays,
/// then uses them to filter/weight sampled hands during determinization.
pub struct BeliefState {
    /// The player whose perspective we are modeling from.
    pub observer: u8,
    /// The observer's known hand.
    pub observer_hand: CardSet,
    /// Per-player per-card soft probability weights.
    /// `soft_weights[player][card]` = unnormalized weight for that player holding that card.
    pub soft_weights: [[f32; 32]; 4],
    /// Accumulated constraints from observed actions.
    constraints: Vec<ActionConstraint>,
}

impl BeliefState {
    /// Create a new BeliefState for the given observer.
    ///
    /// - Observer's cards get weight 1.0 only for the observer, 0.0 for others.
    /// - Unknown cards (not in observer's hand) get weight 1.0 for all non-observer players.
    pub fn new(observer: u8, observer_hand: CardSet) -> Self {
        let mut soft_weights = [[0.0f32; 32]; 4];

        for card in 0..32u8 {
            let bit = card_to_bit(card);
            if observer_hand & bit != 0 {
                // Observer holds this card
                soft_weights[observer as usize][card as usize] = 1.0;
            } else {
                // Unknown card: equal weight for all non-observer players
                for p in 0..4u8 {
                    if p != observer {
                        soft_weights[p as usize][card as usize] = 1.0;
                    }
                }
            }
        }

        BeliefState {
            observer,
            observer_hand,
            soft_weights,
            constraints: Vec::new(),
        }
    }

    /// Check whether all accumulated constraints are satisfied by the given hands.
    pub fn check_constraints(&self, hands: &[CardSet; 4]) -> bool {
        self.constraints.iter().all(|c| c.is_satisfied(hands))
    }

    /// Return a slice of all accumulated constraints.
    pub fn constraints(&self) -> &[ActionConstraint] {
        &self.constraints
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Helper: build a CardSet from a slice of card indices.
    fn hand_from_cards(cards: &[u8]) -> CardSet {
        let mut h: CardSet = 0;
        for &c in cards {
            h |= card_to_bit(c);
        }
        h
    }

    #[test]
    fn test_bid_constraint_satisfied() {
        // Strong spade hand: J♠ (card 3), 9♠ (card 2), A♠ (card 7)
        // Plus some filler cards in other suits
        let hand = hand_from_cards(&[3, 2, 7, 8, 16, 24, 25, 26]);
        let hands = [hand, 0, 0, 0];

        let constraint = ActionConstraint {
            player: 0,
            kind: ConstraintKind::Bid {
                suit: Suit::Spades,
                min_score: 10,
            },
        };

        assert!(
            constraint.is_satisfied(&hands),
            "J+9+A in spades should satisfy bid constraint with min_score=10, actual score={}",
            evaluate_for_trump(hand, Suit::Spades)
        );
    }

    #[test]
    fn test_bid_constraint_not_satisfied() {
        // Weak hand: only 7♠ (card 0) and 8♠ (card 1) plus filler
        let hand = hand_from_cards(&[0, 1, 8, 9, 16, 17, 24, 25]);
        let hands = [hand, 0, 0, 0];

        let constraint = ActionConstraint {
            player: 0,
            kind: ConstraintKind::Bid {
                suit: Suit::Spades,
                min_score: 10,
            },
        };

        assert!(
            !constraint.is_satisfied(&hands),
            "7+8 in spades should not satisfy bid constraint with min_score=10, actual score={}",
            evaluate_for_trump(hand, Suit::Spades)
        );
    }

    #[test]
    fn test_opening_pass_rejects_j9() {
        // Hand with J♠ (card 3) + 9♠ (card 2) — should be inconsistent with opening pass
        let hand = hand_from_cards(&[3, 2, 8, 9, 16, 17, 24, 25]);
        let hands = [hand, 0, 0, 0];

        let constraint = ActionConstraint {
            player: 0,
            kind: ConstraintKind::Pass {
                min_overbid_value: 0,
                auction_position: 0,
                partner_had_bid: false,
                active_suit: Suit::Spades, // irrelevant for opening pass
            },
        };

        assert!(
            !constraint.is_satisfied(&hands),
            "Hand with J♠+9♠ should be rejected for opening pass"
        );
    }

    #[test]
    fn test_opening_pass_accepts_weak_hand() {
        // Weak hand of 7s and 8s: cards 0,1,8,9,16,17,24,25
        let hand = hand_from_cards(&[0, 1, 8, 9, 16, 17, 24, 25]);
        let hands = [hand, 0, 0, 0];

        let constraint = ActionConstraint {
            player: 0,
            kind: ConstraintKind::Pass {
                min_overbid_value: 0,
                auction_position: 0,
                partner_had_bid: false,
                active_suit: Suit::Spades,
            },
        };

        assert!(
            constraint.is_satisfied(&hands),
            "Weak hand of 7s and 8s should be consistent with opening pass"
        );
    }

    #[test]
    fn test_belief_state_new() {
        // Observer = player 0, hand = first 8 spade cards (cards 0..7)
        let observer_hand = hand_from_cards(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let bs = BeliefState::new(0, observer_hand);

        // Observer's cards: weight 1.0 for observer, 0.0 for others
        for card in 0..8u8 {
            assert_eq!(bs.soft_weights[0][card as usize], 1.0,
                "Observer should have weight 1.0 for own card {}", card);
            for p in 1..4 {
                assert_eq!(bs.soft_weights[p][card as usize], 0.0,
                    "Non-observer player {} should have weight 0.0 for observer's card {}", p, card);
            }
        }

        // Unknown cards: weight 0.0 for observer, 1.0 for all others
        for card in 8..32u8 {
            assert_eq!(bs.soft_weights[0][card as usize], 0.0,
                "Observer should have weight 0.0 for unknown card {}", card);
            for p in 1..4 {
                assert_eq!(bs.soft_weights[p][card as usize], 1.0,
                    "Non-observer player {} should have weight 1.0 for unknown card {}", p, card);
            }
        }

        // No constraints initially
        assert!(bs.constraints().is_empty());
    }
}
