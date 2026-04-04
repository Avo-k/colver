use crate::bid_eval::evaluate_for_trump;
use crate::bidding::{decode_bid, BID_COINCHE, BID_PASS, BID_SURCOINCHE};
use crate::card::*;
use crate::determinize::{determinize_greedy, determinize_weighted};
use crate::state::GameState;
use rand::Rng;

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

    // ---- Helpers ----

    /// Multiply a player's soft weight for a card by `factor`,
    /// but only if the player is not the observer and the card is not in the observer's hand.
    fn apply_factor(&mut self, player: u8, card: Card, factor: f32) {
        if player == self.observer {
            return;
        }
        if self.observer_hand & card_to_bit(card) != 0 {
            return;
        }
        self.soft_weights[player as usize][card as usize] *= factor;
    }

    /// Set weight 0 for all 8 cards of a suit for a player (void).
    fn mark_void(&mut self, player: u8, suit_idx: u8) {
        let base = (suit_idx as usize) * 8;
        for rank in 0..8 {
            self.soft_weights[player as usize][base + rank] = 0.0;
        }
    }

    /// Multiply all card weights of a suit for a player by `factor`.
    fn apply_suit_factor(&mut self, player: u8, suit_idx: u8, factor: f32) {
        let base = (suit_idx as usize) * 8;
        for rank in 0..8 {
            let card = (base + rank) as Card;
            self.apply_factor(player, card, factor);
        }
    }

    /// Zero out trump cards stronger than the played card for a player.
    fn apply_trump_ceiling(&mut self, player: u8, trump_suit: u8, best_rank_on_table: u8) {
        let higher_mask = HIGHER_TRUMP_MASK[best_rank_on_table as usize];
        let base = (trump_suit as usize) * 8;
        for rank in 0..8u8 {
            if higher_mask & (1 << rank) != 0 {
                self.soft_weights[player as usize][base + rank as usize] = 0.0;
            }
        }
    }

    /// Check if the player's partner is currently winning the trick (before this player plays).
    fn partner_is_master(&self, state: &GameState, player: u8) -> bool {
        if state.trick_count < 2 {
            return false;
        }

        let partner = GameState::partner(player);
        let lead = state.trick_lead;
        let lead_card = state.current_trick[lead as usize];
        let lead_suit = card_suit(lead_card);
        let trump_suit = state.contract.trump_suit();

        let mut best_seat = lead;
        let mut has_trump = false;
        let mut best_trump_strength = 0u8;
        let mut best_lead_rank = card_rank(lead_card);
        let mut best_lead_seat = lead;

        if lead_suit == trump_suit {
            has_trump = true;
            best_trump_strength = TRUMP_STRENGTH[card_rank(lead_card) as usize];
            best_seat = lead;
        }

        for i in 1..state.trick_count {
            let seat = (lead + i) % 4;
            let card = state.current_trick[seat as usize];
            if card == EMPTY {
                continue;
            }
            let suit = card_suit(card);

            if suit == trump_suit {
                let s = TRUMP_STRENGTH[card_rank(card) as usize];
                if !has_trump || s > best_trump_strength {
                    best_trump_strength = s;
                    best_seat = seat;
                    has_trump = true;
                }
            } else if suit == lead_suit && !has_trump {
                let r = card_rank(card);
                if r > best_lead_rank {
                    best_lead_rank = r;
                    best_lead_seat = seat;
                }
            }
        }

        if !has_trump {
            best_seat = best_lead_seat;
        }

        best_seat == partner
    }

    /// Find the best trump rank currently on the trick.
    fn best_trump_rank_on_trick(&self, state: &GameState, trump_suit: u8) -> Option<u8> {
        let trump = Suit::from_u8(trump_suit);
        let mut best: Option<u8> = None;
        let mut best_strength = 0u8;

        for i in 0..state.trick_count {
            let seat = (state.trick_lead + i) % 4;
            let card = state.current_trick[seat as usize];
            if card == EMPTY {
                continue;
            }
            if card_suit(card) == trump {
                let rank = card_rank(card);
                let strength = TRUMP_STRENGTH[rank as usize];
                if best.is_none() || strength > best_strength {
                    best_strength = strength;
                    best = Some(rank);
                }
            }
        }

        best
    }

    // ---- Task 2: record_bid ----

    /// Record a bidding action and update beliefs accordingly.
    ///
    /// `state` is the state BEFORE the action was applied.
    pub fn record_bid(&mut self, player: u8, action: u8, state: &GameState) {
        if player == self.observer {
            return;
        }

        if action == BID_PASS {
            self.record_pass(player, state);
        } else if action == BID_COINCHE {
            self.record_coinche(player, state);
        } else if action == BID_SURCOINCHE {
            // No constraint for surcoinche
        } else {
            // Positive bid (actions 1-40)
            self.record_positive_bid(player, action);
        }
    }

    fn record_pass(&mut self, player: u8, state: &GameState) {
        // Build Pass constraint context
        let min_overbid_value = if state.last_bid_value > 0 {
            state.last_bid_value + 1
        } else {
            0
        };
        let partner_had_bid =
            state.last_bid_value > 0 && state.last_bidder == GameState::partner(player);
        let active_suit = if state.last_bid_value > 0 {
            Suit::from_u8(state.last_bid_suit)
        } else {
            Suit::Spades // irrelevant for opening pass
        };
        let auction_position = state.consecutive_passes;

        self.constraints.push(ActionConstraint {
            player,
            kind: ConstraintKind::Pass {
                min_overbid_value,
                auction_position,
                partner_had_bid,
                active_suit,
            },
        });

        // Soft: reduce J/9 weights for passer in all suits
        for suit_idx in 0..4u8 {
            let jack = make_card(Suit::from_u8(suit_idx), 3);
            let nine = make_card(Suit::from_u8(suit_idx), 2);
            self.apply_factor(player, jack, 0.6);
            self.apply_factor(player, nine, 0.7);
        }
    }

    fn record_positive_bid(&mut self, player: u8, action: u8) {
        let (bid_value, suit_idx) = decode_bid(action);
        let suit = Suit::from_u8(suit_idx);
        let min_score = bid_value_to_threshold(bid_value);

        self.constraints.push(ActionConstraint {
            player,
            kind: ConstraintKind::Bid { suit, min_score },
        });

        // Soft: boost trump card weights
        let jack = make_card(suit, 3);
        let nine = make_card(suit, 2);
        let ace = make_card(suit, 7);
        let ten = make_card(suit, 6);
        self.apply_factor(player, jack, 5.0);
        self.apply_factor(player, nine, 3.0);
        self.apply_factor(player, ace, 2.0);
        self.apply_factor(player, ten, 1.5);
        // Other trump cards (7, 8, Q, K) get 1.5x
        for rank in [0u8, 1, 4, 5] {
            let card = make_card(suit, rank);
            self.apply_factor(player, card, 1.5);
        }
    }

    fn record_coinche(&mut self, player: u8, state: &GameState) {
        // Boost opponent's trump J/9 and side aces (soft only, no hard constraint)
        let bid_suit = state.last_bid_suit;
        let jack = make_card(Suit::from_u8(bid_suit), 3);
        let nine = make_card(Suit::from_u8(bid_suit), 2);
        self.apply_factor(player, jack, 3.0);
        self.apply_factor(player, nine, 2.5);

        // Side aces
        for suit_idx in 0..4u8 {
            if suit_idx != bid_suit {
                let ace = make_card(Suit::from_u8(suit_idx), 7);
                self.apply_factor(player, ace, 2.0);
            }
        }
    }

    // ---- Task 3: record_play ----

    /// Record a play action and update beliefs accordingly.
    ///
    /// `state` is the state BEFORE the action was applied.
    pub fn record_play(&mut self, player: u8, card: Card, state: &GameState) {
        // Mark played card as weight 0 for all players
        for p in 0..4 {
            self.soft_weights[p][card as usize] = 0.0;
        }

        if player == self.observer {
            return;
        }

        let card_s = card_suit_u8(card);
        let trump_suit = state.contract.trump;

        // === Hard constraints from following rules ===
        if state.trick_count > 0 {
            // Not the leader — check if followed suit
            let lead_card = state.current_trick[state.trick_lead as usize];
            let lead_suit_idx = card_suit(lead_card) as u8;

            if card_s != lead_suit_idx {
                // Didn't follow lead suit -> void in lead suit
                self.mark_void(player, lead_suit_idx);

                if card_s != trump_suit {
                    // Didn't follow AND didn't trump
                    // If partner is NOT winning -> player is void in trump too
                    if !self.partner_is_master(state, player) {
                        self.mark_void(player, trump_suit);
                    }
                }

                // Trump undertrump: played trump but couldn't overtrump
                if card_s == trump_suit {
                    let best_trump_rank = self.best_trump_rank_on_trick(state, trump_suit);
                    if let Some(best_rank) = best_trump_rank {
                        let played_strength = TRUMP_STRENGTH[card_rank(card) as usize];
                        let best_strength = TRUMP_STRENGTH[best_rank as usize];
                        if played_strength < best_strength {
                            self.apply_trump_ceiling(player, trump_suit, best_rank);
                        }
                    }
                }
            } else if lead_suit_idx == trump_suit {
                // Following trump suit — check overtrump constraint
                let best_trump_rank = self.best_trump_rank_on_trick(state, trump_suit);
                if let Some(best_rank) = best_trump_rank {
                    let played_strength = TRUMP_STRENGTH[card_rank(card) as usize];
                    let best_strength = TRUMP_STRENGTH[best_rank as usize];
                    if played_strength < best_strength {
                        self.apply_trump_ceiling(player, trump_suit, best_rank);
                    }
                }
            }
        }

        // === Soft constraints ===
        if state.trick_count == 0 {
            // This player is the leader
            let played_rank = card_rank(card);

            if card_s == trump_suit {
                // Led trump -> boost this player's trump weights
                self.apply_suit_factor(player, trump_suit, 1.5);
            } else if played_rank == 7 {
                // Led ace -> boost 10/K of that suit for this player
                let ten = make_card(Suit::from_u8(card_s), 6);
                let king = make_card(Suit::from_u8(card_s), 5);
                self.apply_factor(player, ten, 2.0);
                self.apply_factor(player, king, 1.5);
            }
        } else {
            // Not the leader
            let lead_card = state.current_trick[state.trick_lead as usize];
            let lead_suit_idx = card_suit(lead_card) as u8;

            if card_s != lead_suit_idx && card_s == trump_suit {
                // Cut with trump
                let played_strength = TRUMP_STRENGTH[card_rank(card) as usize];
                if played_strength <= 2 {
                    // Cut with low trump (7/8/Q) -> boost J/9 for other players
                    for p in 0..4u8 {
                        if p == self.observer || p == player {
                            continue;
                        }
                        let jack = make_card(Suit::from_u8(trump_suit), 3);
                        let nine = make_card(Suit::from_u8(trump_suit), 2);
                        self.apply_factor(p, jack, 1.3);
                        self.apply_factor(p, nine, 1.3);
                    }
                }
            }
        }
    }

    // ---- Task 4: determinize ----

    /// Produce a belief-consistent determinization of the current state.
    ///
    /// Uses weighted sampling biased by soft weights, then checks hard constraints.
    /// Falls back to greedy determinization after 500 failed attempts.
    pub fn determinize(&self, state: &GameState, rng: &mut impl Rng) -> Option<GameState> {
        let has_constraints = !self.constraints.is_empty();

        for _ in 0..500 {
            // Try weighted sampling first, fall back to greedy
            let candidate = determinize_weighted(state, self.observer, &self.soft_weights, rng)
                .or_else(|| determinize_greedy(state, self.observer, rng));

            let candidate = match candidate {
                Some(c) => c,
                None => continue,
            };

            if !has_constraints || self.check_constraints(&candidate.hands) {
                return Some(candidate);
            }
        }

        // Fallback: greedy without constraint checking
        determinize_greedy(state, self.observer, rng)
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

    #[test]
    fn test_record_bid_adds_constraint() {
        // Observer = P0 with spades (cards 0..7)
        let observer_hand = hand_from_cards(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut bs = BeliefState::new(0, observer_hand);

        // Create a game state where P1 is about to bid
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);

        // P1 bids 80 Hearts: encode_bid(8, 1) = action 2
        let action = crate::bidding::encode_bid(8, 1);
        bs.record_bid(1, action, &state);

        assert_eq!(bs.constraints().len(), 1);
        match &bs.constraints()[0].kind {
            ConstraintKind::Bid { suit, min_score } => {
                assert_eq!(*suit, Suit::Hearts);
                assert_eq!(*min_score, bid_value_to_threshold(8));
            }
            _ => panic!("Expected Bid constraint"),
        }
    }

    #[test]
    fn test_record_bid_boosts_weights() {
        // Observer = P0 with spades (cards 0..7)
        let observer_hand = hand_from_cards(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut bs = BeliefState::new(0, observer_hand);

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);

        // J♥ = card 11 (suit 1, rank 3), 9♥ = card 10 (suit 1, rank 2)
        let jack_h = make_card(Suit::Hearts, 3) as usize;
        let nine_h = make_card(Suit::Hearts, 2) as usize;

        let before_j = bs.soft_weights[1][jack_h];
        let before_9 = bs.soft_weights[1][nine_h];

        // P1 bids 80 Hearts
        let action = crate::bidding::encode_bid(8, 1);
        bs.record_bid(1, action, &state);

        assert!(
            bs.soft_weights[1][jack_h] > before_j,
            "Jack weight should increase after bid: {} -> {}",
            before_j, bs.soft_weights[1][jack_h]
        );
        assert!(
            bs.soft_weights[1][nine_h] > before_9,
            "Nine weight should increase after bid: {} -> {}",
            before_9, bs.soft_weights[1][nine_h]
        );
    }

    #[test]
    fn test_record_pass_adds_constraint() {
        let observer_hand = hand_from_cards(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut bs = BeliefState::new(0, observer_hand);

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);

        // P1 passes (opening pass)
        bs.record_bid(1, BID_PASS, &state);

        assert_eq!(bs.constraints().len(), 1);
        match &bs.constraints()[0].kind {
            ConstraintKind::Pass {
                min_overbid_value,
                partner_had_bid,
                ..
            } => {
                assert_eq!(*min_overbid_value, 0, "Opening pass should have min_overbid_value=0");
                assert!(!partner_had_bid, "No partner bid in opening pass");
            }
            _ => panic!("Expected Pass constraint"),
        }

        // Verify soft weights for J/9 decreased
        let jack_h = make_card(Suit::Hearts, 3) as usize;
        assert!(
            bs.soft_weights[1][jack_h] < 1.0,
            "J♥ weight for P1 should decrease after pass: {}",
            bs.soft_weights[1][jack_h]
        );
    }

    #[test]
    fn test_record_bid_observer_ignored() {
        let observer_hand = hand_from_cards(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut bs = BeliefState::new(0, observer_hand);

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);

        // Observer (P0) bids — should be ignored
        let action = crate::bidding::encode_bid(8, 0);
        bs.record_bid(0, action, &state);

        assert!(bs.constraints().is_empty(), "Observer's bid should not add constraints");
    }

    #[test]
    #[ignore] // uses RNG
    fn test_determinize_respects_bid_constraint() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        // Observer = P0 with spades
        let observer_hand = hand_from_cards(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let mut bs = BeliefState::new(0, observer_hand);

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);

        // P1 bids 80 Hearts
        let action = crate::bidding::encode_bid(8, 1);
        bs.record_bid(1, action, &state);

        let threshold = bid_value_to_threshold(8);
        let mut rng = StdRng::seed_from_u64(42);
        let mut pass_count = 0;
        let trials = 50;

        for _ in 0..trials {
            if let Some(det) = bs.determinize(&state, &mut rng) {
                let p1_hand = det.hands[1];
                if evaluate_for_trump(p1_hand, Suit::Hearts) >= threshold {
                    pass_count += 1;
                }
            }
        }

        // Most determinizations should satisfy the bid constraint
        assert!(
            pass_count > trials / 2,
            "Expected most determinizations to satisfy bid constraint, got {}/{}",
            pass_count, trials
        );
    }

    #[test]
    #[ignore] // uses RNG
    fn test_determinize_without_constraints() {
        use rand::SeedableRng;
        use rand::rngs::StdRng;

        let observer_hand = hand_from_cards(&[0, 1, 2, 3, 4, 5, 6, 7]);
        let bs = BeliefState::new(0, observer_hand);

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);

        let mut rng = StdRng::seed_from_u64(42);

        // No constraints → should always succeed
        for _ in 0..50 {
            let result = bs.determinize(&state, &mut rng);
            assert!(result.is_some(), "Determinize should succeed without constraints");
        }
    }
}
