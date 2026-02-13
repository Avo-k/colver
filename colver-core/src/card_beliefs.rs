use crate::bidding::{decode_bid, BID_COINCHE, BID_PASS, BID_SURCOINCHE};
use crate::card::*;
use crate::state::{GameState, Phase};

/// Tracks per-card per-player probability weights for informed determinization.
///
/// After every action (by any player), beliefs are updated with hard constraints
/// (definitive deductions from the rules) and soft constraints (probabilistic
/// inferences from competent play).
pub struct CardBeliefs {
    /// `weights[player][card]` = unnormalized probability that `player` holds `card`.
    /// 0.0 means impossible, higher = more likely.
    weights: [[f32; 32]; 4],
    /// The observer player (whose hand is fully known).
    observer: u8,
    /// Whether to apply soft (probabilistic) inference.
    pub use_soft_inference: bool,
}

impl CardBeliefs {
    /// Initialize beliefs from the current game state for a given observer.
    ///
    /// The observer's hand is known exactly. Played cards are eliminated.
    /// Other cards start with uniform weight 1.0 for all non-observer players
    /// who are not void in that card's suit.
    pub fn new(state: &GameState, observer: u8) -> Self {
        let mut beliefs = CardBeliefs {
            weights: [[0.0; 32]; 4],
            observer,
            use_soft_inference: true,
        };

        let observer_hand = state.hands[observer as usize];

        for card in 0..32u8 {
            let bit = card_to_bit(card);
            let suit_idx = card_suit_u8(card);

            if state.played_cards & bit != 0 {
                // Card already played - weight 0 for everyone
                continue;
            }

            if observer_hand & bit != 0 {
                // Observer has this card
                beliefs.weights[observer as usize][card as usize] = 1.0;
                continue;
            }

            // Unknown card - distribute among non-observer players
            for p in 0..4u8 {
                if p == observer {
                    continue;
                }
                // Check void constraint
                if state.voids[p as usize] & (1 << suit_idx) != 0 {
                    continue;
                }
                beliefs.weights[p as usize][card as usize] = 1.0;
            }
        }

        beliefs
    }

    /// Record an action and update beliefs accordingly.
    ///
    /// `state_before` is the state BEFORE the action was applied.
    /// This allows us to inspect trick state, lead suit, etc.
    pub fn record_action(&mut self, state_before: &GameState, player: u8, action: u8) {
        match state_before.phase {
            Phase::Bidding => self.infer_bid(player, action, state_before),
            Phase::Playing => self.infer_play(state_before, player, action),
            Phase::Done => {}
        }
    }

    /// Mark a card as played (weight 0 for all players).
    fn mark_played(&mut self, card: u8) {
        for p in 0..4 {
            self.weights[p][card as usize] = 0.0;
        }
    }

    /// Mark a player as void in a suit (weight 0 for all cards of that suit).
    fn mark_void(&mut self, player: u8, suit_idx: u8) {
        let base = (suit_idx as usize) * 8;
        for rank in 0..8 {
            self.weights[player as usize][base + rank] = 0.0;
        }
    }

    /// Apply a multiplicative factor to a specific card for a specific player.
    fn apply_factor(&mut self, player: u8, card: u8, factor: f32) {
        self.weights[player as usize][card as usize] *= factor;
    }

    /// Apply a multiplicative factor to all cards of a suit for a specific player.
    fn apply_suit_factor(&mut self, player: u8, suit_idx: u8, factor: f32) {
        let base = (suit_idx as usize) * 8;
        for rank in 0..8 {
            self.weights[player as usize][base + rank] *= factor;
        }
    }

    /// Apply trump ceiling: player has no trump stronger than the given rank.
    fn apply_trump_ceiling(&mut self, player: u8, trump_suit: u8, best_rank_on_table: u8) {
        let higher_mask = HIGHER_TRUMP_MASK[best_rank_on_table as usize];
        let base = (trump_suit as usize) * 8;
        for rank in 0..8u8 {
            if higher_mask & (1 << rank) != 0 {
                self.weights[player as usize][base + rank as usize] = 0.0;
            }
        }
    }

    // ---- Bidding inference ----

    fn infer_bid(&mut self, player: u8, action: u8, _state: &GameState) {
        if player == self.observer {
            return; // We know our own hand
        }

        match action {
            BID_PASS => {
                if !self.use_soft_inference {
                    return;
                }
                // Pass suggests lacking strong trump in any suit
                for suit_idx in 0..4u8 {
                    let jack = make_card(Suit::from_u8(suit_idx), 3); // J
                    let nine = make_card(Suit::from_u8(suit_idx), 2); // 9
                    self.apply_factor(player, jack, 0.6);
                    self.apply_factor(player, nine, 0.7);
                }
            }
            BID_COINCHE => {
                if !self.use_soft_inference {
                    return;
                }
                // Coinche suggests strong defense: J/9 of opponent's trump, side Aces
                // We need to know which suit was bid - use _state
                let bid_suit = _state.last_bid_suit;
                let jack = make_card(Suit::from_u8(bid_suit), 3);
                let nine = make_card(Suit::from_u8(bid_suit), 2);
                self.apply_factor(player, jack, 3.0);
                self.apply_factor(player, nine, 2.5);
                // Side Aces
                for suit_idx in 0..4u8 {
                    if suit_idx != bid_suit {
                        let ace = make_card(Suit::from_u8(suit_idx), 7);
                        self.apply_factor(player, ace, 2.0);
                    }
                }
            }
            BID_SURCOINCHE => {
                if !self.use_soft_inference {
                    return;
                }
                // Surcoinche: very strong trump holding
                let bid_suit = _state.last_bid_suit;
                // Boost all trump cards
                self.apply_suit_factor(player, bid_suit, 2.0);
                // Extra boost for J and 9
                let jack = make_card(Suit::from_u8(bid_suit), 3);
                let nine = make_card(Suit::from_u8(bid_suit), 2);
                self.apply_factor(player, jack, 1.5); // total ~3.0
                self.apply_factor(player, nine, 1.5); // total ~3.0
            }
            _ => {
                // Regular bid (1-40)
                if !self.use_soft_inference {
                    return;
                }
                let (value, suit_idx) = decode_bid(action);
                let jack = make_card(Suit::from_u8(suit_idx), 3);
                let nine = make_card(Suit::from_u8(suit_idx), 2);
                let ace = make_card(Suit::from_u8(suit_idx), 7);

                if value == 25 {
                    // Capot: very strong hand
                    self.apply_factor(player, jack, 15.0);
                    self.apply_factor(player, nine, 10.0);
                    self.apply_factor(player, ace, 8.0);
                    // Side Aces
                    for s in 0..4u8 {
                        if s != suit_idx {
                            let side_ace = make_card(Suit::from_u8(s), 7);
                            self.apply_factor(player, side_ace, 4.0);
                        }
                    }
                } else {
                    // Scale factors by bid level: 80=base, each +10 scales up
                    let level = ((value as f32) - 8.0).max(0.0); // 0 for 80, 1 for 90, ..., 8 for 160
                    let j_factor = 5.0 + level * 0.875; // 5.0 at 80, up to ~12.0 at 160
                    let n_factor = 3.0 + level * 0.625; // 3.0 at 80, up to ~8.0 at 160
                    let a_factor = 2.0 + level * 0.375; // 2.0 at 80, up to ~5.0 at 160
                    self.apply_factor(player, jack, j_factor);
                    self.apply_factor(player, nine, n_factor);
                    self.apply_factor(player, ace, a_factor);
                }
            }
        }
    }

    // ---- Play inference ----

    fn infer_play(&mut self, state: &GameState, player: u8, card: u8) {
        // Mark the card as played
        self.mark_played(card);

        // If observer played, we already know their hand - nothing to infer
        if player == self.observer {
            return;
        }

        // Set this player's weight for the played card to 0 (already done above)
        // But also confirm: this player DID have this card, so set others' weight to 0
        // (already done by mark_played)

        let card_s = card_suit_u8(card);

        // === Hard constraints from following rules ===
        if state.trick_count > 0 {
            // Not the leader - check if they followed suit
            let lead_card = state.current_trick[state.trick_lead as usize];
            let lead_suit = card_suit(lead_card);
            let lead_suit_idx = lead_suit as u8;
            let trump_suit = state.contract.trump;

            if card_s != lead_suit_idx {
                // Didn't follow lead suit -> void in lead suit (hard constraint)
                self.mark_void(player, lead_suit_idx);

                // Check for trump void inference
                if card_s != trump_suit {
                    // Player didn't follow AND didn't play trump.
                    // If partner is NOT master, rules require trumping if possible.
                    // So if partner isn't winning -> player is void in trump too.
                    if !self.partner_is_master_before_play(state, player) {
                        self.mark_void(player, trump_suit);
                    }
                }

                // Trump ceiling: if player played trump but couldn't overtrump
                if card_s == trump_suit {
                    // Player cut with trump - check if they could have overtrumped
                    let best_trump_rank = self.best_trump_rank_on_trick(state, trump_suit);
                    if let Some(best_rank) = best_trump_rank {
                        let played_rank = card_rank(card);
                        let played_strength = TRUMP_STRENGTH[played_rank as usize];
                        let best_strength = TRUMP_STRENGTH[best_rank as usize];
                        if played_strength < best_strength {
                            // Player couldn't overtrump -> no stronger trump
                            self.apply_trump_ceiling(player, trump_suit, best_rank);
                        }
                    }
                }
            } else if lead_suit_idx == trump_suit {
                // Following trump suit - check overtrump constraint
                let best_trump_rank = self.best_trump_rank_on_trick(state, trump_suit);
                if let Some(best_rank) = best_trump_rank {
                    let played_rank = card_rank(card);
                    let played_strength = TRUMP_STRENGTH[played_rank as usize];
                    let best_strength = TRUMP_STRENGTH[best_rank as usize];
                    if played_strength < best_strength {
                        // Couldn't overtrump -> no stronger trump
                        self.apply_trump_ceiling(player, trump_suit, best_rank);
                    }
                }
            }
        }

        // === Soft constraints ===
        if !self.use_soft_inference {
            return;
        }

        let trump_suit = state.contract.trump;

        if state.trick_count == 0 {
            // This player is the leader
            let played_rank = card_rank(card);

            if card_s == trump_suit {
                // Led trump (drawing trumps) -> good trump holding
                for p in 0..4u8 {
                    if p == self.observer || p == player {
                        continue;
                    }
                    // Others more likely to have remaining trump
                    self.apply_suit_factor(p, trump_suit, 1.0); // no change for others
                }
                // Player likely has more trump
                self.apply_suit_factor(player, trump_suit, 1.5);
            } else if played_rank == 7 {
                // Led Ace -> strong in that suit
                for p in 0..4u8 {
                    if p == self.observer || p == player {
                        continue;
                    }
                    let ten = make_card(Suit::from_u8(card_s), 6);
                    let king = make_card(Suit::from_u8(card_s), 5);
                    self.apply_factor(p, ten, 1.0); // neutral for others
                    self.apply_factor(p, king, 1.0);
                }
                // Player likely has 10 and K of that suit
                let ten = make_card(Suit::from_u8(card_s), 6);
                let king = make_card(Suit::from_u8(card_s), 5);
                self.apply_factor(player, ten, 2.0);
                self.apply_factor(player, king, 1.5);
            } else if played_rank <= 1 {
                // Led low card (7 or 8) -> weak in that suit
                for p in 0..4u8 {
                    if p == self.observer || p == player {
                        continue;
                    }
                    let ace = make_card(Suit::from_u8(card_s), 7);
                    let ten = make_card(Suit::from_u8(card_s), 6);
                    self.apply_factor(p, ace, 1.2);
                    self.apply_factor(p, ten, 1.2);
                }
            }
        } else {
            // Not the leader
            let lead_card = state.current_trick[state.trick_lead as usize];
            let lead_suit_idx = card_suit(lead_card) as u8;

            if card_s != lead_suit_idx && card_s == trump_suit {
                // Cut with trump
                let played_rank = card_rank(card);
                let played_strength = TRUMP_STRENGTH[played_rank as usize];
                if played_strength <= 2 {
                    // Cut with low trump (7/8/Q) -> likely lacks stronger trump
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
            } else if card_s != lead_suit_idx && card_s != trump_suit {
                // Discarded (non-trump, non-lead) -> shedding weak suit
                for p in 0..4u8 {
                    if p == self.observer || p == player {
                        continue;
                    }
                    let ace = make_card(Suit::from_u8(card_s), 7);
                    let ten = make_card(Suit::from_u8(card_s), 6);
                    self.apply_factor(p, ace, 1.2);
                    self.apply_factor(p, ten, 1.2);
                }
            }
        }
    }

    /// Check if player's partner is currently winning the trick (before player plays).
    fn partner_is_master_before_play(&self, state: &GameState, player: u8) -> bool {
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

    /// Get normalized weights suitable for weighted determinization.
    ///
    /// Returns `weights[player][card]` normalized so that for each unknown card,
    /// the sum across eligible players equals 1.0.
    pub fn normalized_weights(&self) -> [[f32; 32]; 4] {
        let mut result = self.weights;

        // Normalize per-card: for each card, sum weights across players, divide
        for card in 0..32 {
            let mut sum = 0.0f32;
            for p in 0..4 {
                sum += result[p][card];
            }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for p in 0..4 {
                    result[p][card] *= inv;
                }
            }
        }

        result
    }

    /// Get raw (unnormalized) weights.
    pub fn raw_weights(&self) -> &[[f32; 32]; 4] {
        &self.weights
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_state() -> GameState {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        GameState::new(0, hands)
    }

    #[test]
    fn test_initial_beliefs_observer_known() {
        let state = make_test_state();
        let beliefs = CardBeliefs::new(&state, 0);

        // Observer (P0) has spades (cards 0-7)
        for card in 0..8u8 {
            assert_eq!(beliefs.weights[0][card as usize], 1.0);
            // Other players should have 0 for observer's cards
            for p in 1..4 {
                assert_eq!(beliefs.weights[p][card as usize], 0.0);
            }
        }
    }

    #[test]
    fn test_initial_beliefs_unknown_cards() {
        let state = make_test_state();
        let beliefs = CardBeliefs::new(&state, 0);

        // Cards 8-31 are unknown to P0
        // Each should have weight 1.0 for P1, P2, P3 (no void constraints)
        for card in 8..32u8 {
            assert_eq!(beliefs.weights[0][card as usize], 0.0);
            for p in 1..4 {
                assert_eq!(
                    beliefs.weights[p][card as usize], 1.0,
                    "P{} should have weight 1.0 for card {}",
                    p, card
                );
            }
        }
    }

    #[test]
    fn test_void_constraint_in_initial() {
        let mut state = make_test_state();
        // P1 is known void in spades
        state.voids[1] = 1 << 0;

        let beliefs = CardBeliefs::new(&state, 0);

        // P1 should have 0 weight for all spade cards (0-7)
        // But P0 already has them, so it's moot for 0-7
        // For cards 8-15 (hearts), P1 should still be eligible
        for card in 8..16u8 {
            assert_eq!(beliefs.weights[1][card as usize], 1.0);
        }
    }

    #[test]
    fn test_mark_played_zeros_all() {
        let state = make_test_state();
        let mut beliefs = CardBeliefs::new(&state, 0);

        // Mark card 10 as played
        beliefs.mark_played(10);
        for p in 0..4 {
            assert_eq!(beliefs.weights[p][10], 0.0);
        }
    }

    #[test]
    fn test_mark_void_zeros_suit() {
        let state = make_test_state();
        let mut beliefs = CardBeliefs::new(&state, 0);

        // Mark P2 void in hearts (suit 1, cards 8-15)
        beliefs.mark_void(2, 1);
        for card in 8..16u8 {
            assert_eq!(beliefs.weights[2][card as usize], 0.0);
        }
        // Diamonds (suit 2) should be unaffected
        for card in 16..24u8 {
            assert_eq!(beliefs.weights[2][card as usize], 1.0);
        }
    }

    #[test]
    fn test_trump_ceiling() {
        let state = make_test_state();
        let mut beliefs = CardBeliefs::new(&state, 0);

        // P1 couldn't overtrump Ace of trump (suit 1, rank 7)
        // Ace strength = 5, so 9 (str=6) and J (str=7) should be zeroed
        beliefs.apply_trump_ceiling(1, 1, 7); // suit 1 = hearts, rank 7 = Ace

        // 9 of hearts = card 10 (suit 1, rank 2)
        assert_eq!(beliefs.weights[1][10], 0.0);
        // J of hearts = card 11 (suit 1, rank 3)
        assert_eq!(beliefs.weights[1][11], 0.0);
        // Queen of hearts = card 12 should still be non-zero (Q str=2 < A str=5)
        assert_eq!(beliefs.weights[1][12], 1.0);
    }

    #[test]
    fn test_normalized_weights_sum_to_one() {
        let state = make_test_state();
        let beliefs = CardBeliefs::new(&state, 0);
        let norm = beliefs.normalized_weights();

        for card in 0..32 {
            let sum: f32 = (0..4).map(|p| norm[p][card]).sum();
            if sum > 0.0 {
                assert!(
                    (sum - 1.0).abs() < 1e-5,
                    "Card {} weights don't sum to 1.0: {}",
                    card,
                    sum
                );
            }
        }
    }

    #[test]
    fn test_bid_inference_increases_trump_weight() {
        let mut state = make_test_state();
        state.current_player = 1; // P1 is bidding

        let mut beliefs = CardBeliefs::new(&state, 0);

        // P1 bids 80 Spades (action = encode_bid(8, 0) = 1)
        // Spades are in P0's hand, so test with hearts instead
        let jack_h = make_card(Suit::Hearts, 3) as usize; // card 11
        let nine_h = make_card(Suit::Hearts, 2) as usize; // card 10

        let before_j = beliefs.weights[1][jack_h];
        let before_9 = beliefs.weights[1][nine_h];

        // P1 bids 80 Hearts (suit_idx=1): action = 0*4 + 1 + 1 = 2
        beliefs.infer_bid(1, 2, &state);

        assert!(
            beliefs.weights[1][jack_h] > before_j,
            "Jack weight should increase after bid"
        );
        assert!(
            beliefs.weights[1][nine_h] > before_9,
            "Nine weight should increase after bid"
        );
    }

    #[test]
    fn test_pass_decreases_trump_weight() {
        let state = make_test_state();
        let mut beliefs = CardBeliefs::new(&state, 0);

        let jack_h = make_card(Suit::Hearts, 3) as usize;
        let before = beliefs.weights[1][jack_h];

        beliefs.infer_bid(1, BID_PASS, &state);

        assert!(
            beliefs.weights[1][jack_h] < before,
            "Jack weight should decrease after pass"
        );
    }

    #[test]
    fn test_play_marks_void() {
        use crate::bidding;
        let mut state = make_test_state();
        // Setup: go through bidding to get to playing phase
        // P1 bids 80 Hearts
        bidding::apply_bid(&mut state, bidding::encode_bid(8, 1));
        // P2, P3, P0 pass
        bidding::apply_bid(&mut state, BID_PASS);
        bidding::apply_bid(&mut state, BID_PASS);
        bidding::apply_bid(&mut state, BID_PASS);

        assert_eq!(state.phase, Phase::Playing);

        // Create beliefs from P0's perspective
        let mut beliefs = CardBeliefs::new(&state, 0);

        // P1 leads AH (card 15)
        state.step(15); // P1 plays AH

        // P2 is next, plays 7D (card 16) - doesn't follow hearts
        let state_before2 = state;
        beliefs.record_action(&state_before2, 2, 16);

        // P2 should now be void in hearts (suit 1)
        for rank in 0..8u8 {
            let card = make_card(Suit::Hearts, rank) as usize;
            assert_eq!(
                beliefs.weights[2][card], 0.0,
                "P2 should be void in hearts after not following, card={}",
                card
            );
        }
    }
}
