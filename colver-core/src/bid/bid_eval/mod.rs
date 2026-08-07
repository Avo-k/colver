// Strategy submodules
mod eval_helpers;
mod smart;
mod parametric;
mod improved;

// Re-export shared helpers
pub use eval_helpers::{
    evaluate_for_trump, best_trump, heuristic_bid, quality_ok,
};

// Re-export strategy functions
pub use smart::smart_bid;
pub use parametric::{parametric_bid, BidParams};
pub use improved::{
    improved_bid, improved_v2_bid, improved_v2_configurable_bid, improved_v3_bid, V2Config,
};

use crate::state::*;

/// Which bidding function to use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BidFunction {
    Heuristic,
    Smart,
    Improved,
    ImprovedV2,
    ImprovedV3,
    Maxi,
    /// DD-based bidding ("bid à DD") — uses solver + determinization.
    /// Creates a temporary DdBidder per call (requires `rand` feature).
    #[cfg(feature = "rand")]
    BidADd,
}

impl BidFunction {
    /// Dispatch to the appropriate bid function.
    pub fn bid(self, state: &GameState) -> u8 {
        match self {
            BidFunction::Heuristic => heuristic_bid(state),
            BidFunction::Smart => smart_bid(state),
            BidFunction::Improved => improved_bid(state),
            BidFunction::ImprovedV2 => improved_v2_bid(state),
            BidFunction::ImprovedV3 => improved_v3_bid(state),
            BidFunction::Maxi => crate::maxi::maxi_bid(state),
            #[cfg(feature = "rand")]
            BidFunction::BidADd => {
                let mut bidder =
                    crate::dd_bid::DdBidder::new(crate::dd_bid::DdBidConfig::default());
                let mut rng = rand::thread_rng();
                bidder.bid(state, &mut rng)
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bidding::{self, BID_COINCHE, BID_PASS};
    use crate::card::*;

    /// Helper: create a hand from card specs like ("JS", "9S", "AH", ...)
    fn make_hand(cards: &[&str]) -> CardSet {
        let mut set: CardSet = 0;
        for &name in cards {
            let (rank, suit) = parse_card(name);
            set |= card_to_bit(make_card(suit, rank));
        }
        set
    }

    fn parse_card(name: &str) -> (u8, Suit) {
        let name = name.trim();
        let suit_char = name.chars().last().unwrap();
        let rank_str = &name[..name.len() - 1];

        let rank = match rank_str {
            "7" => 0,
            "8" => 1,
            "9" => 2,
            "J" => 3,
            "Q" => 4,
            "K" => 5,
            "10" => 6,
            "A" => 7,
            _ => panic!("Unknown rank: {}", rank_str),
        };

        let suit = match suit_char {
            'S' => Suit::Spades,
            'H' => Suit::Hearts,
            'D' => Suit::Diamonds,
            'C' => Suit::Clubs,
            _ => panic!("Unknown suit: {}", suit_char),
        };

        (rank, suit)
    }

    #[test]
    fn test_strong_hand_high_score() {
        // J+9+A+10 of Spades: 8+6+4+3 = 21, +4 length bonus (4 trumps-2)*2 = 25
        let hand = make_hand(&["JS", "9S", "AS", "10S", "7H", "8H", "7D", "7C"]);
        let score = evaluate_for_trump(hand, Suit::Spades);
        assert!(
            score >= 20,
            "Strong trump hand should score >= 20, got {}",
            score
        );
    }

    #[test]
    fn test_weak_hand_low_score() {
        // No J/9/A anywhere - just small cards
        let hand = make_hand(&["7S", "8S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let (_, score) = best_trump(hand);
        assert!(
            score < 10,
            "Weak hand should score < 10, got {}",
            score
        );
    }

    #[test]
    fn test_heuristic_bid_pass_on_weak() {
        let hand = make_hand(&["7S", "8S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        // Distribute remaining cards to other players (8 each)
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(
            0,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        let action = heuristic_bid(&state);
        assert_eq!(action, BID_PASS, "Weak hand should PASS");
    }

    #[test]
    fn test_heuristic_bid_bids_on_strong() {
        // J+9+A of Spades + some other cards
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "7H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // Set dealer=3 so player 0 is first bidder
        let state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        assert_eq!(state.current_player, 0);
        let action = heuristic_bid(&state);
        assert_ne!(action, BID_PASS, "Strong hand should not PASS");
    }

    #[test]
    fn test_heuristic_bid_cant_overbid() {
        // Good hand but opponent already bid 150
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "7H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        // Player 0 bids 150 Spades
        state.step(bidding::encode_bid(15, 0)); // P0: 150S
        // Now P1 needs to overbid 150 - heuristic can't go that high
        let action = heuristic_bid(&state);
        // Heuristic never bids above 130, so should PASS
        assert_eq!(action, BID_PASS, "Should PASS when can't overbid 150");
    }

    #[test]
    fn test_heuristic_bid_after_coinche_passes() {
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "7H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        // P0 bids 80S, P1 coinches. Now P2 acts.
        state.step(bidding::encode_bid(8, 0)); // P0: 80S
        state.step(BID_COINCHE); // P1: coinche
        // P2 is current_player. After coinche, only surcoinche or pass are legal.
        // Heuristic never surcoinches.
        let action = heuristic_bid(&state);
        assert_eq!(action, BID_PASS, "Should PASS after coinche (heuristic never surcoinches)");
    }

    #[test]
    fn test_smart_bid_opening_j9() {
        // J+9 of Hearts + side cards → should open 90
        let hand = make_hand(&["JH", "9H", "8H", "7S", "8S", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        let action = smart_bid(&state);
        assert_ne!(action, BID_PASS, "J+9 hand should open");
        if action != BID_PASS && action <= 40 {
            let (val, suit) = bidding::decode_bid(action);
            assert_eq!(suit, 1, "Should bid Hearts");
            assert_eq!(val, 8, "J+9 alone should bid 80");
        }
    }

    #[test]
    fn test_smart_bid_partner_response() {
        // Partner bid 80 Hearts, I have 9H → should respond 90H
        let hand = make_hand(&["9H", "KH", "AS", "10S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        // P0 bids 80H, P1 passes, P2 is partner of P0
        state.step(bidding::encode_bid(8, 1)); // P0: 80H
        state.step(BID_PASS); // P1: pass
        // Now P2 acts (partner of P0)
        assert_eq!(state.current_player, 2);
        // We need P2 to have the response hand
        let mut state2 = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state2.step(bidding::encode_bid(8, 1)); // P0: 80H
        state2.step(BID_PASS); // P1: pass
        assert_eq!(state2.current_player, 2);
        let action = smart_bid(&state2);
        assert_ne!(action, BID_PASS, "Should respond to partner's 80");
    }

    #[test]
    fn test_smart_bid_coinche() {
        // Opponent bid Hearts, I have JH+9H → should COINCHE
        let hand = make_hand(&["JH", "9H", "7S", "8S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // P0 has our hand, P1 bids Hearts, we are P2 (opponent of P1)
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        // P0 passes
        state.step(BID_PASS);
        // P1 bids 80H
        state.step(bidding::encode_bid(8, 1));
        // Now P2 acts
        assert_eq!(state.current_player, 2);
        let action = smart_bid(&state);
        assert_eq!(action, BID_COINCHE, "Should coinche when holding opponent's J+9");
    }

    #[test]
    fn test_legal_action_validation() {
        // After opponent's 150, we shouldn't return an illegal action
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "QS", "8S", "7S"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        state.step(bidding::encode_bid(15, 0)); // P0: 150S
        // P1 acts
        let action = heuristic_bid(&state);
        let legal = state.legal_actions();
        assert!(
            legal & (1u64 << action) != 0,
            "Heuristic bid returned illegal action {}",
            action
        );
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_heuristic_bid_random_deals_complete() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..100 {
            let mut state = GameState::deal_random(0, &mut rng);

            // Use heuristic bidding
            while state.phase == Phase::Bidding && !state.is_terminal() {
                let action = heuristic_bid(&state);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Heuristic bid {} is illegal! Legal mask: {:b}",
                    action,
                    legal
                );
                state.step(action);
            }

            // After bidding, should be Playing or Done (void deal from 4 passes)
            assert!(
                state.phase == Phase::Playing || state.phase == Phase::Done,
                "After heuristic bidding, phase should be Playing or Done, got {:?}",
                state.phase
            );

            // If playing, finish with random plays
            if state.phase == Phase::Playing {
                while !state.is_terminal() {
                    let legal = state.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    let action = crate::rollout::select_nth_bit(legal, idx);
                    state.step(action);
                }
            }

            assert!(state.is_terminal());
        }
    }

    // ---------------------------------------------------------------
    // improved_bid tests
    // ---------------------------------------------------------------

    #[test]
    fn test_improved_bid_opens_on_strong() {
        // J+9+A of Spades → strong hand, should bid 100+ Spades
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "7H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        let action = improved_bid(&state);
        assert_ne!(action, BID_PASS, "Strong hand should not PASS");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 0, "Should bid Spades");
        assert!(val >= 10, "J+9+A should bid at least 100, got {}", val * 10);
    }

    #[test]
    fn test_improved_bid_passes_on_weak() {
        // All 7s and 8s → should PASS
        let hand = make_hand(&["7S", "8S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        let action = improved_bid(&state);
        assert_eq!(action, BID_PASS, "Weak hand should PASS");
    }

    #[test]
    fn test_improved_bid_quality_gate() {
        // K-Q of Spades but no J/9/A/10, only 2 cards → no quality
        let hand = make_hand(&["KS", "QS", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        let action = improved_bid(&state);
        assert_eq!(action, BID_PASS, "Should PASS without quality (K-Q doubleton)");
    }

    #[test]
    fn test_improved_bid_partner_raise() {
        // Partner bid 80H, I have 9H+AH+side ace → score=13 in Hearts, should raise to 90
        let hand = make_hand(&["9H", "AH", "AS", "10S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // P2 has our hand, P0 opens 80H, P1 passes
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state.step(bidding::encode_bid(8, 1)); // P0: 80H
        state.step(BID_PASS); // P1: pass
        assert_eq!(state.current_player, 2);
        let action = improved_bid(&state);
        assert_ne!(action, BID_PASS, "Should raise partner's 80H");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should raise in partner's suit (Hearts)");
        assert!(val >= 9, "Should raise to at least 90");
    }

    #[test]
    fn test_improved_bid_overcall() {
        // Opponent bid 80S, I have J+A of Hearts (score ~15) → bid Hearts
        let hand = make_hand(&["JH", "AH", "10H", "8H", "7S", "8S", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state.step(BID_PASS); // P0: pass
        state.step(bidding::encode_bid(8, 0)); // P1: 80S
        assert_eq!(state.current_player, 2);
        let action = improved_bid(&state);
        assert_ne!(action, BID_PASS, "Should overcall with strong Hearts");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should bid Hearts");
        assert!(val >= 9, "Should bid at least 90H");
    }

    #[test]
    fn test_improved_bid_coinche() {
        // Opponent bid 80S, I have J+9 of Spades → COINCHE
        let hand = make_hand(&["JS", "9S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state.step(BID_PASS); // P0: pass
        state.step(bidding::encode_bid(8, 0)); // P1: 80S
        assert_eq!(state.current_player, 2);
        let action = improved_bid(&state);
        assert_eq!(action, BID_COINCHE, "Should coinche with J+9 in opponent's suit");
    }

    #[test]
    fn test_improved_bid_never_above_120_opening() {
        // Even with a monster hand, opening caps at 120
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "QS", "8S", "AH"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        let action = improved_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, _suit) = bidding::decode_bid(action);
        assert!(val <= 12, "Opening should cap at 120, got {}", val * 10);
    }

    #[test]
    fn test_improved_bid_partner_cap_130() {
        // Partner bid 100H, I have J+9+A of Hearts → raise but cap at 130
        let hand = make_hand(&["JH", "9H", "AH", "10H", "KH", "AS", "AD", "AC"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state.step(bidding::encode_bid(10, 1)); // P0: 100H
        state.step(BID_PASS); // P1: pass
        assert_eq!(state.current_player, 2);
        let action = improved_bid(&state);
        if action != BID_PASS {
            let (val, _suit) = bidding::decode_bid(action);
            assert!(val <= 13, "Partner response should cap at 130, got {}", val * 10);
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_improved_bid_random_deals_complete() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..200 {
            let mut state = GameState::deal_random(0, &mut rng);

            while state.phase == Phase::Bidding && !state.is_terminal() {
                let action = improved_bid(&state);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Improved bid {} is illegal! Legal mask: {:b}",
                    action,
                    legal
                );
                state.step(action);
            }

            assert!(
                state.phase == Phase::Playing || state.phase == Phase::Done,
                "After improved bidding, phase should be Playing or Done, got {:?}",
                state.phase
            );

            if state.phase == Phase::Playing {
                while !state.is_terminal() {
                    let legal = state.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    let action = crate::rollout::select_nth_bit(legal, idx);
                    state.step(action);
                }
            }

            assert!(state.is_terminal());
        }
    }

    // ---------------------------------------------------------------
    // improved_v2_bid tests
    // ---------------------------------------------------------------

    /// Helper: set up a state where player at `seat` is the current bidder.
    /// dealer is set so that seat is the first bidder (dealer = seat-1 mod 4).
    fn v2_state_for_seat(hand: CardSet, seat: u8) -> GameState {
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let dealer = (seat + 3) % 4;
        let mut hands = [0u32; 4];
        hands[seat as usize] = hand;
        let mut j = 0;
        for i in 0..4 {
            if i != seat as usize {
                hands[i] = other_hands[j];
                j += 1;
            }
        }
        GameState::new(dealer, hands)
    }

    #[test]
    fn test_v2_opens_on_strong() {
        // J+9+A of Spades → should bid, same as improved
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "7H", "7D", "7C"]);
        let state = v2_state_for_seat(hand, 0);
        let action = improved_v2_bid(&state);
        assert_ne!(action, BID_PASS, "Strong hand should not PASS");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 0, "Should bid Spades");
        assert!(val >= 10, "J+9+A should bid at least 100");
    }

    #[test]
    fn test_v2_passes_on_weak() {
        let hand = make_hand(&["7S", "8S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let state = v2_state_for_seat(hand, 0);
        let action = improved_v2_bid(&state);
        assert_eq!(action, BID_PASS, "Weak hand should PASS");
    }

    #[test]
    fn test_v2_4th_position_passes_marginal() {
        // Score ~12 in Spades (A+10 = 4+3=7 trump, +3 side ace = ~13 from eval),
        // but only A+10 (no J or 9) — 4th position requires J or 9
        let hand = make_hand(&["AS", "10S", "AH", "7H", "8H", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // Set up P3 in 4th position: dealer=3, first bidder=0
        // P0 passes, P1 passes, P2 passes → P3 is 4th (position=3)
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], other_hands[2], hand]);
        state.step(BID_PASS); // P0
        state.step(BID_PASS); // P1
        state.step(BID_PASS); // P2
        assert_eq!(state.current_player, 3);
        let action = improved_v2_bid(&state);
        assert_eq!(action, BID_PASS, "4th position without J or 9 should PASS");
    }

    #[test]
    fn test_v2_4th_position_bids_with_jack() {
        // J + trumps in 4th position — should bid if score >= 15
        let hand = make_hand(&["JS", "AS", "10S", "KS", "7H", "8H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], other_hands[2], hand]);
        state.step(BID_PASS); // P0
        state.step(BID_PASS); // P1
        state.step(BID_PASS); // P2
        assert_eq!(state.current_player, 3);
        let action = improved_v2_bid(&state);
        assert_ne!(action, BID_PASS, "4th position with J and good score should bid");
    }

    #[test]
    fn test_v2_4th_position_passes_low_score() {
        // J but low score (< 15) in 4th position → PASS
        let hand = make_hand(&["JS", "7S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], other_hands[2], hand]);
        state.step(BID_PASS);
        state.step(BID_PASS);
        state.step(BID_PASS);
        assert_eq!(state.current_player, 3);
        let action = improved_v2_bid(&state);
        assert_eq!(action, BID_PASS, "4th position with J but low score should PASS");
    }

    #[test]
    fn test_v2_coinche_theoreme3() {
        // Opponent bid 80H, I have 0 Hearts + 3 aces → COINCHE
        let hand = make_hand(&["AS", "AD", "AC", "KS", "QS", "7S", "8S", "7D"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        state.step(BID_PASS); // P0
        state.step(bidding::encode_bid(8, 1)); // P1: 80H
        assert_eq!(state.current_player, 2);
        let action = improved_v2_bid(&state);
        assert_eq!(action, BID_COINCHE, "Should coinche with 0 trumps + 3 aces (Théorème 3)");
    }

    #[test]
    fn test_v2_no_coinche_theoreme3_with_trumps() {
        // Opponent bid 80H, I have 1 Heart + 3 aces → NOT Théorème 3
        let hand = make_hand(&["AS", "AD", "AC", "7H", "KS", "QS", "7S", "8S"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        state.step(BID_PASS); // P0
        state.step(bidding::encode_bid(8, 1)); // P1: 80H
        assert_eq!(state.current_player, 2);
        let action = improved_v2_bid(&state);
        // Should NOT coinche via Théorème 3 (has a trump), but might via 4+ trump rule
        // With only 1 trump and 3 aces, this should just overcall or pass
        assert_ne!(action, BID_COINCHE, "Should not Théorème 3 coinche with a trump card");
    }

    #[test]
    fn test_v2_respond_jack_complement() {
        // Partner bid 80H, I have JH + side aces → response bonus should boost
        let hand = make_hand(&["JH", "8H", "AS", "AD", "7S", "7D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // P2 has hand, partner P0 bids
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        state.step(bidding::encode_bid(8, 1)); // P0: 80H
        state.step(BID_PASS); // P1
        assert_eq!(state.current_player, 2);
        let action = improved_v2_bid(&state);
        assert_ne!(action, BID_PASS, "Should raise with J complement + side aces");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should raise in partner's suit");
        assert!(val >= 9, "Should raise to at least 90");
    }

    #[test]
    fn test_v2_respond_misfit() {
        // Partner bid 80H, I have 0 Hearts + low cards → misfit, should PASS
        let hand = make_hand(&["7S", "8S", "KD", "QD", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        state.step(bidding::encode_bid(8, 1)); // P0: 80H
        state.step(BID_PASS); // P1
        assert_eq!(state.current_player, 2);
        let action = improved_v2_bid(&state);
        // With 0 trumps and -3 misfit penalty, unlikely to raise
        // Might change suit if alt score is strong enough, but with weak hand should PASS
        assert_eq!(action, BID_PASS, "Should PASS with misfit (0 trumps in partner suit)");
    }

    #[test]
    fn test_v2_respond_nine_complement() {
        // Partner bid 80H, I have 9H+KH (2 trumps) + side ace → +2 nine bonus + +2 side ace + trumps
        // Uses V2Config::full() which has nine complement bonus enabled.
        let hand = make_hand(&["9H", "KH", "AS", "10S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        state.step(bidding::encode_bid(8, 1)); // P0: 80H
        state.step(BID_PASS); // P1
        assert_eq!(state.current_player, 2);
        let action = improved_v2_configurable_bid(&state, &V2Config::full());
        assert_ne!(action, BID_PASS, "Should raise with 9 complement");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should raise in Hearts");
        assert!(val >= 9, "Should raise to at least 90");
    }

    #[test]
    fn test_v2_lead_bonus_tips_opening() {
        // J alone in Spades scores 8. +1 singleton = 9. Below 10 → PASS normally.
        // With lead bonus +2 (V2Config::full()) → 11 → opens 80.
        // Uses V2Config::full() which has lead_bonus=2 (defensive_lead1 only has +1).
        let hand = make_hand(&["JS", "7H", "8H", "KD", "QD", "7D", "7C", "8C"]);
        let score = evaluate_for_trump(hand, Suit::Spades);
        assert!(score < 10, "Precondition: score {} should be < 10 for this test", score);

        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);

        // dealer=3 → first_bidder=0, lead=0 (P0 has lead!)
        let state_lead = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        assert_eq!(state_lead.current_player, 0);
        let action_lead = improved_v2_configurable_bid(&state_lead, &V2Config::full());
        assert_ne!(action_lead, BID_PASS, "Lead bonus +2 should tip borderline hand to open");

        // improved_bid (no lead bonus) should PASS on the same hand.
        let state_improved = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        let action_improved = improved_bid(&state_improved);
        assert_eq!(action_improved, BID_PASS, "Without lead bonus, improved_bid should PASS");
    }

    #[test]
    fn test_v2_opening_cap_120() {
        // Monster hand still caps at 120
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "QS", "8S", "AH"]);
        let state = v2_state_for_seat(hand, 0);
        let action = improved_v2_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, _) = bidding::decode_bid(action);
        assert!(val <= 12, "Opening should cap at 120, got {}", val * 10);
    }

    #[test]
    fn test_v2_respond_cap_130() {
        // Partner bid 100H, I have monster → cap at 130
        let hand = make_hand(&["JH", "9H", "AH", "10H", "KH", "AS", "AD", "AC"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        state.step(bidding::encode_bid(10, 1)); // P0: 100H
        state.step(BID_PASS); // P1
        assert_eq!(state.current_player, 2);
        let action = improved_v2_bid(&state);
        if action != BID_PASS {
            let (val, _) = bidding::decode_bid(action);
            assert!(val <= 13, "Partner response should cap at 130, got {}", val * 10);
        }
    }

    #[test]
    fn test_v2_coinche_j9_still_works() {
        // Existing J+9 coinche should still trigger
        let hand = make_hand(&["JS", "9S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        state.step(BID_PASS); // P0
        state.step(bidding::encode_bid(8, 0)); // P1: 80S
        assert_eq!(state.current_player, 2);
        let action = improved_v2_bid(&state);
        assert_eq!(action, BID_COINCHE, "J+9 coinche should still work");
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_v2_random_deals_complete() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..500 {
            let mut state = GameState::deal_random(rng.gen_range(0..4), &mut rng);

            while state.phase == Phase::Bidding && !state.is_terminal() {
                let action = improved_v2_bid(&state);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "ImprovedV2 bid {} is illegal! Legal mask: {:b}",
                    action,
                    legal
                );
                state.step(action);
            }

            assert!(
                state.phase == Phase::Playing || state.phase == Phase::Done,
                "After V2 bidding, phase should be Playing or Done, got {:?}",
                state.phase
            );

            if state.phase == Phase::Playing {
                while !state.is_terminal() {
                    let legal = state.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    let action = crate::rollout::select_nth_bit(legal, idx);
                    state.step(action);
                }
            }

            assert!(state.is_terminal());
        }
    }

    /// Distribute remaining cards equally to n players.
    fn distribute_remaining(remaining: CardSet, n: usize) -> Vec<CardSet> {
        let mut hands = vec![0u32; n];
        let mut bits = remaining;
        let mut idx = 0;
        while bits != 0 {
            let card = bits.trailing_zeros();
            hands[idx % n] |= 1 << card;
            idx += 1;
            bits &= bits - 1;
        }
        hands
    }
}
