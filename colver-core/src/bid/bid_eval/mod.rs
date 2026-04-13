// Strategy submodules
mod eval_helpers;
mod smart;
mod parametric;
mod roro;
mod improved;
mod petit_bide;
mod moelleux;

// Re-export shared helpers
pub use eval_helpers::{
    evaluate_for_trump, best_trump, heuristic_bid, quality_ok,
};

// Re-export strategy functions
pub use smart::smart_bid;
pub use parametric::{parametric_bid, BidParams};
pub use roro::roro_bid;
pub use improved::{
    improved_bid, improved_v2_bid, improved_v2_configurable_bid, improved_v3_bid, V2Config,
};
pub use petit_bide::{petit_bide_bid, petit_bide_tricks, petit_bide_score};
pub use moelleux::moelleux_bid;

use crate::state::*;

/// Which bidding function to use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BidFunction {
    Heuristic,
    Smart,
    Improved,
    ImprovedV2,
    ImprovedV3,
    Roro,
    PetitBide,
    Moelleux,
    Maxi,
    /// DD-based bidding ("bid à DD") — uses solver + determinization.
    /// Creates a temporary DdBidder per call (requires `rand` feature).
    #[cfg(feature = "rand")]
    BidADd,
    /// Placeholder for BisDd — actual decisions handled by stateful BisDdAgent
    /// in arena (this variant exists so build_agent can parse the strategy).
    #[cfg(feature = "rand")]
    BisDd,
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
            BidFunction::Roro => roro_bid(state),
            BidFunction::PetitBide => petit_bide_bid(state),
            BidFunction::Moelleux => moelleux_bid(state),
            BidFunction::Maxi => crate::maxi::maxi_bid(state),
            #[cfg(feature = "rand")]
            BidFunction::BidADd => {
                let mut bidder =
                    crate::dd_bid::DdBidder::new(crate::dd_bid::DdBidConfig::default());
                let mut rng = rand::thread_rng();
                bidder.bid(state, &mut rng)
            }
            #[cfg(feature = "rand")]
            BidFunction::BisDd => {
                // Standalone mode: no persistent beliefs (each call is independent)
                let mut agent = crate::bis_dd::BisDdAgent::new(
                    crate::bis_dd::BisDdConfig::default(),
                    rand::random(),
                );
                agent.init_deal(state.current_player, state.hands[state.current_player as usize]);
                agent.decide(state)
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

    // ---------------------------------------------------------------
    // roro_bid tests
    // ---------------------------------------------------------------

    /// Helper: set up a state where player 0 is the current bidder (dealer=3).
    fn roro_state(hand: CardSet) -> GameState {
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]])
    }

    #[test]
    fn test_roro_opening_80_aux_as() {
        // 1st position, 2 aces, no J/9 → 80 "aux as"
        let hand = make_hand(&["AH", "AD", "7S", "8S", "7H", "8H", "7D", "7C"]);
        let state = roro_state(hand);
        assert_eq!(state.current_player, 0);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "2 aces should open 80 aux as");
        let (val, _suit) = bidding::decode_bid(action);
        assert_eq!(val, 8, "Should open at 80");
    }

    #[test]
    fn test_roro_opening_80_aux_as_no_one_ace() {
        // 1st position, only 1 ace → no 80 aux as, PASS
        let hand = make_hand(&["AH", "7S", "8S", "7H", "8H", "7D", "8D", "7C"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        assert_eq!(action, BID_PASS, "1 ace should not open 80 aux as");
    }

    #[test]
    fn test_roro_opening_80_walou_3rd_position() {
        // 3rd position: walou with long suit
        let hand = make_hand(&["10S", "KS", "QS", "8S", "7H", "8H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // dealer=3, so P0=1st, P1=2nd, P2=3rd
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        // P0 passes, P1 passes → P2 is 3rd position
        state.step(BID_PASS);
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "3rd position walou should open 80");
        let (val, _) = bidding::decode_bid(action);
        assert_eq!(val, 8, "Walou should be 80");
    }

    #[test]
    fn test_roro_opening_90_jack_3rd_ace() {
        // J 3rd + 1 ace → 90
        let hand = make_hand(&["JH", "KH", "8H", "AS", "7S", "7D", "8D", "7C"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "J 3rd + ace should open 90");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(val, 9, "Should open at 90");
        assert_eq!(suit, 1, "Should bid Hearts");
    }

    #[test]
    fn test_roro_opening_90_jack_4th() {
        // J 4th → 90
        let hand = make_hand(&["JH", "KH", "QH", "8H", "7S", "7D", "8D", "7C"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "J 4th should open 90");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(val, 9, "Should open at 90");
        assert_eq!(suit, 1, "Should bid Hearts");
    }

    #[test]
    fn test_roro_opening_90_nine_5th() {
        // 9 5th → 90
        let hand = make_hand(&["9H", "KH", "QH", "8H", "7H", "7S", "7D", "7C"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "9 5th should open 90");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(val, 9, "Should open at 90");
        assert_eq!(suit, 1, "Should bid Hearts");
    }

    #[test]
    fn test_roro_never_nine_3rd_ace_at_90() {
        // 9 3rd + 1 ace → should NOT open 90 (this is a coinche hand!)
        let hand = make_hand(&["9H", "KH", "8H", "AS", "7S", "7D", "8D", "7C"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        // Should either pass or bid something other than 90 Hearts
        if action != BID_PASS {
            let (val, suit) = bidding::decode_bid(action);
            assert!(
                !(val == 9 && suit == 1),
                "Should NEVER open 9 3rd + ace at 90 (this is a coinche hand!)"
            );
        }
    }

    #[test]
    fn test_roro_opening_100_j9_4th() {
        // J9 4th → 100
        let hand = make_hand(&["JH", "9H", "KH", "8H", "7S", "7D", "8D", "7C"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "J9 4th should open 100");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(val, 10, "Should open at 100");
        assert_eq!(suit, 1, "Should bid Hearts");
    }

    #[test]
    fn test_roro_opening_100_j9_3rd_ace() {
        // J9 3rd + side ace → 100
        let hand = make_hand(&["JH", "9H", "8H", "AS", "7S", "7D", "8D", "7C"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "J9 3rd + ace should open 100");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(val, 10, "Should open at 100");
        assert_eq!(suit, 1, "Should bid Hearts");
    }

    #[test]
    fn test_roro_opening_110_first_position_only() {
        // J9A 4th, no voids, 1st position → 110
        let hand = make_hand(&["JH", "9H", "AH", "8H", "AS", "7D", "8D", "7C"]);
        let state = roro_state(hand);
        assert_eq!(state.current_player, 0);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should bid Hearts");
        assert_eq!(val, 11, "J9A 4th 1st position should be 110");
    }

    #[test]
    fn test_roro_opening_110_not_in_2nd_position() {
        // Same hand in 2nd position → should NOT get 110
        let hand = make_hand(&["JH", "9H", "AH", "8H", "AS", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], hand, other_hands[1], other_hands[2]]);
        state.step(BID_PASS); // P0 passes, now P1 is 2nd position
        assert_eq!(state.current_player, 1);
        let action = roro_bid(&state);
        // Should still bid (100 at least), but not 110
        if action != BID_PASS {
            let (val, _) = bidding::decode_bid(action);
            assert_ne!(val, 11, "110 should only be available in 1st position");
        }
    }

    #[test]
    fn test_roro_opening_120_tricolore() {
        // J9 in trump + void + max 2 losers → 120
        // JH 9H KH + AS AD + void in clubs
        let hand = make_hand(&["JH", "9H", "KH", "AS", "AD", "10S", "KD", "QD"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, _suit) = bidding::decode_bid(action);
        // Should open high (at least 100, possibly 120)
        assert!(val >= 10, "Strong hand should bid at least 100");
    }

    #[test]
    fn test_roro_opening_highest_first() {
        // Hand that qualifies for both 100 and 90 → should pick 100
        let hand = make_hand(&["JH", "9H", "8H", "AS", "7S", "7D", "8D", "7C"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, _) = bidding::decode_bid(action);
        // J9 3rd + 1 ace qualifies for 100
        assert_eq!(val, 10, "Should open at highest possible level (100)");
    }

    #[test]
    fn test_roro_opening_4th_position_strong_only() {
        // 4th position with weak hand → PASS
        let hand = make_hand(&["10H", "KH", "QH", "8H", "7S", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], other_hands[2], hand]);
        state.step(BID_PASS);
        state.step(BID_PASS);
        state.step(BID_PASS);
        assert_eq!(state.current_player, 3);
        let action = roro_bid(&state);
        assert_eq!(action, BID_PASS, "4th position with weak hand should PASS");
    }

    #[test]
    fn test_roro_respond_on_80_aux_as_support() {
        // Partner (P0, 1st pos) opened 80H. I (P2) have V second → 90 (V second support)
        let hand_p2 = make_hand(&["JH", "7H", "7S", "8S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(8, 1)); // P0: 80H (1st pos = aux as)
        state.step(BID_PASS); // P1: pass
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        // V second = interesting support → 90
        assert_ne!(action, BID_PASS, "V second should raise to 90");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should raise in Hearts");
        assert_eq!(val, 9, "V second → 90");
    }

    #[test]
    fn test_roro_respond_on_80_aux_as_minimum_pass() {
        // Partner (P0, 1st pos) opened 80H. I (P2) have 9 sec (only) → pass (minimum)
        let hand_p2 = make_hand(&["9H", "7H", "7S", "8S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(8, 1)); // P0: 80H (1st pos = aux as)
        state.step(BID_PASS); // P1: pass
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        // 9 second = minimum support → pass is acceptable
        // (9 second with just 2 trumps doesn't hit any raise condition: no J, 9 trump_count=2)
        assert_eq!(action, BID_PASS, "9 second only → minimum support → pass");
    }

    #[test]
    fn test_roro_respond_on_80_aux_as_strong_support() {
        // Partner (P0, 1st pos) opened 80H. I (P2) have V 3rd → 100
        let hand_p2 = make_hand(&["JH", "KH", "8H", "7S", "8S", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(8, 1)); // P0: 80H (aux as)
        state.step(BID_PASS); // P1
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "V 3rd should raise partner's 80");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should respond in Hearts");
        assert_eq!(val, 10, "V 3rd → 100");
    }

    #[test]
    fn test_roro_respond_on_80_aux_as_change_color() {
        // Partner (P0, 1st pos) opened 80H. I (P2) have nothing in Hearts but J 3rd Spades
        let hand_p2 = make_hand(&["JS", "KS", "8S", "7D", "8D", "7C", "8C", "QC"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(8, 1)); // P0: 80H
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        // No Hearts → must change color
        assert_ne!(action, BID_PASS, "Must change color when nothing in partner's suit");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 0, "Should bid Spades (best alternative)");
        assert!(val >= 9, "Change color should be at least 90");
    }

    #[test]
    fn test_roro_respond_on_90_complement() {
        // Partner bid 90H. I have V + 1 side ace → 110 (100 base + 10 for ace)
        let hand_p2 = make_hand(&["JH", "7H", "AS", "7S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(9, 1)); // P0: 90H
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "V complement should raise 90");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should respond in Hearts");
        assert_eq!(val, 11, "V + 1 ace → 110");
    }

    #[test]
    fn test_roro_respond_on_90_complement_no_ace() {
        // Partner bid 90H. I have V, no side ace → 100
        let hand_p2 = make_hand(&["JH", "7H", "10S", "7S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(9, 1)); // P0: 90H
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "V complement should raise 90");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should respond in Hearts");
        assert_eq!(val, 10, "V complement, no ace → 100");
    }

    #[test]
    fn test_roro_respond_on_90_complement_with_ace() {
        // Partner bid 90H. I have V + 1 side ace → 110
        let hand_p2 = make_hand(&["JH", "7H", "AS", "10S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(9, 1));
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1);
        assert_eq!(val, 11, "V + 1 ace → 110");
    }

    #[test]
    fn test_roro_respond_on_90_no_complement() {
        // Partner bid 90H. I have no V, no 9, only 1 trump → pass
        let hand_p2 = make_hand(&["KH", "AS", "10S", "7S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(9, 1));
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_eq!(action, BID_PASS, "No complement should pass on 90");
    }

    #[test]
    fn test_roro_respond_on_100_aces() {
        // Partner bid 100H. I have 1 ace + 1 trump → 110
        let hand_p2 = make_hand(&["8H", "AS", "10S", "7S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(10, 1)); // P0: 100H
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1);
        assert_eq!(val, 11, "1 ace → 110");
    }

    #[test]
    fn test_roro_respond_on_100_two_aces() {
        // Partner bid 100H. I have 2 aces + trump → 120
        let hand_p2 = make_hand(&["8H", "AS", "AD", "10S", "7S", "7D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(10, 1));
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1);
        assert_eq!(val, 12, "2 aces → 120");
    }

    #[test]
    fn test_roro_respond_on_100_no_trump_penalty() {
        // Partner bid 100H. I have 1 ace but 0 trumps → pass (penalty)
        let hand_p2 = make_hand(&["AS", "10S", "KS", "7S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(10, 1));
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        // 0 trump + 1 ace → pass
        assert_eq!(action, BID_PASS, "0 trump + 1 ace → pass on 100");
    }

    #[test]
    fn test_roro_respond_on_110_master_tricks() {
        // Partner bid 110H. I have AS + AD → 2 master tricks → 130
        let hand_p2 = make_hand(&["AS", "AD", "7S", "8S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand_p2;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand_p2, other_hands[2]],
        );
        state.step(bidding::encode_bid(11, 1)); // P0: 110H
        state.step(BID_PASS);
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1);
        assert_eq!(val, 12, "2 master aces → 110+20=120 (capped)");
    }

    #[test]
    fn test_roro_coinche_j9() {
        // Opponent bid 80S, I have JS+9S → COINCHE
        let hand = make_hand(&["JS", "9S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state.step(BID_PASS);
        state.step(bidding::encode_bid(8, 0)); // P1: 80S
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_eq!(action, BID_COINCHE, "J+9 in opponent's suit → coinche");
    }

    #[test]
    fn test_roro_coinche_theoreme_3() {
        // Opponent bid 80S, I have 0 trumps + 3 aces → coinche (théorème 3)
        let hand = make_hand(&["AH", "AD", "AC", "10H", "KH", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state.step(BID_PASS);
        state.step(bidding::encode_bid(8, 0)); // P1: 80S
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_eq!(action, BID_COINCHE, "0 trumps + 3 aces → théorème 3 coinche");
    }

    #[test]
    fn test_roro_no_coinche_above_110_for_4_trumps() {
        // Opponent bid 110S, I have 4 trumps + side ace → should NOT coinche (> 110)
        let hand = make_hand(&["KS", "QS", "8S", "7S", "AH", "7H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state.step(BID_PASS);
        state.step(bidding::encode_bid(11, 0)); // P1: 110S
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_COINCHE, "Should NOT coinche 4 trumps+ace above 110");
    }

    #[test]
    fn test_roro_intervention_light() {
        // Opponent bid 80S. I have J 3rd Hearts → light intervention (+10 = 90)
        let hand = make_hand(&["JH", "KH", "8H", "7S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state.step(BID_PASS);
        state.step(bidding::encode_bid(8, 0)); // P1: 80S
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS, "Should intervene with J 3rd");
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should bid Hearts");
        assert_eq!(val, 9, "+10 intervention → 90");
    }

    #[test]
    fn test_roro_intervention_barre() {
        // Opponent bid 80S. I have J9 3rd Hearts → "la barre" (+20 = 100)
        let hand = make_hand(&["JH", "9H", "8H", "7S", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [other_hands[0], other_hands[1], hand, other_hands[2]],
        );
        state.step(BID_PASS);
        state.step(bidding::encode_bid(8, 0)); // P1: 80S
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(suit, 1, "Should bid Hearts");
        assert_eq!(val, 10, "La barre → +20 = 100");
    }

    #[test]
    fn test_roro_after_coinche_passes() {
        // After coinche, roro should always pass
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "7H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(
            3,
            [hand, other_hands[0], other_hands[1], other_hands[2]],
        );
        state.step(bidding::encode_bid(8, 0)); // P0: 80S
        state.step(BID_COINCHE); // P1: coinche
        // P2 acts after coinche
        assert_eq!(state.current_player, 2);
        let action = roro_bid(&state);
        assert_eq!(action, BID_PASS, "Should pass after coinche");
    }

    #[test]
    fn test_roro_weak_hand_passes() {
        // All 7s and 8s → PASS in any position
        let hand = make_hand(&["7S", "8S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let state = roro_state(hand);
        let action = roro_bid(&state);
        assert_eq!(action, BID_PASS, "All 7/8 hand should PASS");
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_roro_bid_random_deals_complete() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..200 {
            let mut state = GameState::deal_random(0, &mut rng);

            while state.phase == Phase::Bidding && !state.is_terminal() {
                let action = roro_bid(&state);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Roro bid {} is illegal! Legal mask: {:b}, phase: {:?}, player: {}, last_bid: {}, coinche: {}",
                    action,
                    legal,
                    state.phase,
                    state.current_player,
                    state.last_bid_value,
                    state.coinche_state,
                );
                state.step(action);
            }

            assert!(
                state.phase == Phase::Playing || state.phase == Phase::Done,
                "After roro bidding, phase should be Playing or Done, got {:?}",
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
    // PetitBide tests
    // ---------------------------------------------------------------

    #[test]
    fn test_petit_bide_tricks_jack_only() {
        // Jack of Spades only trump → 1 trick, +10 bonus
        let hand = make_hand(&["JS", "7H", "8H", "AH", "7D", "8D", "7C", "8C"]);
        let (tricks, bonus) = petit_bide_tricks(hand, Suit::Spades);
        assert_eq!(tricks, 2, "Jack(1) + side ace AH(1) = 2 tricks"); // J=1, AH=1
        assert_eq!(bonus, 10, "Jack bonus = +10");
    }

    #[test]
    fn test_petit_bide_tricks_j9_3rd() {
        // J+9+8 of Spades + side ace → 3 tricks from trump + 1 from ace
        let hand = make_hand(&["JS", "9S", "8S", "AH", "7H", "7D", "8D", "7C"]);
        let (tricks, bonus) = petit_bide_tricks(hand, Suit::Spades);
        // J=1, 9 2nd=1, 3rd trump=1, side ace=1 = 4 tricks
        assert_eq!(tricks, 4);
        assert_eq!(bonus, 10); // Jack bonus only
    }

    #[test]
    fn test_petit_bide_tricks_side_ten_second() {
        // 10H + KH (2 hearts) → 10 "2nd" = 1 trick, -5 bonus
        let hand = make_hand(&["JS", "9S", "10H", "KH", "7D", "8D", "7C", "8C"]);
        let (tricks, bonus) = petit_bide_tricks(hand, Suit::Spades);
        // J=1, 9 2nd=1, 10H 2nd=1 = 3 tricks
        assert_eq!(tricks, 3);
        assert_eq!(bonus, 10 - 5); // J(+10) + 10H(-5)
    }

    #[test]
    fn test_petit_bide_score_strong_hand() {
        // J+9+A+10 of Spades + side ace → many tricks
        let hand = make_hand(&["JS", "9S", "AS", "10S", "AH", "7H", "7D", "7C"]);
        let score = petit_bide_score(hand, Suit::Spades);
        // J=1, 9 2nd=1, 3rd trump=1, 4th trump=1, side AH=1 = 5 tricks
        // bonus: J(+10) = 10. AS/10S are trumps, not side cards.
        // Score = 5*20 + 10 = 110
        assert!(score >= 100, "Strong hand score should be >= 100, got {}", score);
    }

    #[test]
    fn test_petit_bide_score_weak_hand() {
        let hand = make_hand(&["7S", "8S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let score = petit_bide_score(hand, Suit::Spades);
        assert!(score < 80, "Weak hand should score < 80, got {}", score);
    }

    #[test]
    fn test_petit_bide_bid_opens_strong() {
        let hand = make_hand(&["JS", "9S", "8S", "AH", "10H", "KH", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        let action = petit_bide_bid(&state);
        assert_ne!(action, BID_PASS, "Strong PetitBide hand should bid");
    }

    #[test]
    fn test_petit_bide_bid_passes_weak() {
        let hand = make_hand(&["7S", "8S", "7H", "8H", "7D", "8D", "7C", "8C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        let action = petit_bide_bid(&state);
        assert_eq!(action, BID_PASS, "Weak hand should PASS");
    }

    #[test]
    fn test_petit_bide_4th_position_needs_100() {
        // Moderate hand that scores ~80-90 — should pass in 4th position
        let hand = make_hand(&["JS", "8S", "7S", "AH", "7H", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // dealer=0, first bidder=1. P1,P2,P3 pass, then P0 is 4th position
        let mut state = GameState::new(0, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        state.step(BID_PASS); // P1
        state.step(BID_PASS); // P2
        state.step(BID_PASS); // P3
        assert_eq!(state.current_player, 0);
        let score = petit_bide_score(hand, Suit::Spades);
        if score < 100 {
            let action = petit_bide_bid(&state);
            assert_eq!(action, BID_PASS, "4th position should PASS with score {} < 100", score);
        }
    }

    #[test]
    fn test_petit_bide_opening_cap_120() {
        // Monster hand — should be capped at 120
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "QS", "AH", "AD"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        let action = petit_bide_bid(&state);
        assert_ne!(action, BID_PASS);
        let (val, _suit) = bidding::decode_bid(action);
        assert!(val <= 12, "Opening should be capped at 120, got {}", val * 10);
    }

    #[test]
    fn test_petit_bide_response_jack_boost() {
        // Partner bid 80H, I have JH → +20 response
        let hand = make_hand(&["JH", "7H", "7S", "8S", "7D", "8D", "7C", "8C"]);
        let score = petit_bide::petit_bide_response_score(hand, Suit::Hearts);
        assert!(score >= 20, "Jack in partner suit should give +20, got {}", score);
    }

    #[test]
    fn test_petit_bide_response_zero_trumps_penalty() {
        // No hearts at all → -10 penalty
        let hand = make_hand(&["7S", "8S", "9S", "7D", "8D", "9D", "7C", "8C"]);
        let score = petit_bide::petit_bide_response_score(hand, Suit::Hearts);
        assert!(score < 0, "Zero trumps should penalize, got {}", score);
    }

    #[test]
    fn test_petit_bide_intervention_overbids() {
        // Opponent bid 80H, I have strong spades
        let hand = make_hand(&["JS", "9S", "8S", "AS", "AH", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        state.step(bidding::encode_bid(8, 1)); // P0: 80H
        // P1 should intervene
        assert_eq!(state.current_player, 1);
        // Put our hand at P1
        let mut state2 = GameState::new(3, [other_hands[0], hand, other_hands[1], other_hands[2]]);
        state2.step(bidding::encode_bid(8, 1)); // P0: 80H
        let action = petit_bide_bid(&state2);
        assert_ne!(action, BID_PASS, "Should intervene with strong hand");
    }

    #[test]
    fn test_petit_bide_no_quality_gate_without_j9() {
        // Ace + 10 + K of Spades but no J/9 → should PASS (no opening without J/9)
        let hand = make_hand(&["AS", "10S", "KS", "QS", "7H", "8H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        let action = petit_bide_bid(&state);
        assert_eq!(action, BID_PASS, "Should PASS without J or 9");
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_petit_bide_random_deals_complete() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let mut state = GameState::deal_random(0, &mut rng);
            while state.phase == Phase::Bidding && !state.is_terminal() {
                let action = petit_bide_bid(&state);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "PetitBide returned illegal action {} (legal: {:b})",
                    action, legal
                );
                state.step(action);
            }
            assert!(state.phase == Phase::Playing || state.phase == Phase::Done);
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
    // Moelleux tests
    // ---------------------------------------------------------------

    #[test]
    fn test_moelleux_aux_as_1st_position() {
        // 2 aces, 1st position → should bid 80 in non-ace suit
        let hand = make_hand(&["AH", "AD", "JS", "9S", "8S", "7H", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // dealer=3, first bidder=0 (1st position)
        let state = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        // Note: petit_bide_score in Spades with J+9+8 should be >= 90
        // But moelleux first checks above-80 bids, so it might bid 90+ in spades
        // Let's test with a hand that has 2 aces but weak everywhere
        let hand2 = make_hand(&["AH", "AD", "7S", "8S", "KH", "7D", "7C", "8C"]);
        let remaining2 = ALL_CARDS & !hand2;
        let other_hands2 = distribute_remaining(remaining2, 3);
        let state2 = GameState::new(3, [hand2, other_hands2[0], other_hands2[1], other_hands2[2]]);
        let action = moelleux_bid(&state2);
        if action != BID_PASS && action <= 40 {
            let (val, suit) = bidding::decode_bid(action);
            if val == 8 {
                // If bid 80, should be in a non-ace suit
                let suit_bits_val = suit_bits(hand2, Suit::from_u8(suit));
                // In the moelleux convention, the suit should NOT have an ace
                // (unless all suits have aces)
                let suit_has_ace = suit_bits_val & (1 << 7) != 0;
                // Only enforce if there exists a non-ace suit
                let has_non_ace_suit = (0..4u8).any(|s| {
                    suit_bits(hand2, Suit::from_u8(s)) & (1 << 7) == 0
                        && suit_bits(hand2, Suit::from_u8(s)).count_ones() > 0
                });
                if has_non_ace_suit {
                    assert!(!suit_has_ace, "Moelleux 80 should be in non-ace suit, got {:?}", Suit::from_u8(suit));
                }
            }
        }
    }

    #[test]
    fn test_moelleux_petit_jeu_3rd_position() {
        // 3rd position: 3+ trumps with J + side ace
        let hand = make_hand(&["JS", "8S", "7S", "AH", "7H", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // dealer=3, first bidder=0. P0 passes, P1 passes, now P2 is 3rd
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        state.step(BID_PASS); // P0
        state.step(BID_PASS); // P1
        assert_eq!(state.current_player, 2);
        let action = moelleux_bid(&state);
        if action != BID_PASS && action <= 40 {
            let (val, suit) = bidding::decode_bid(action);
            assert_eq!(val, 8, "3rd position petit jeu should bid 80, got {}", val * 10);
            assert_eq!(suit, 0, "Should bid Spades (strongest trump suit)");
        }
    }

    #[test]
    fn test_moelleux_4th_position_iso() {
        // 4th position: only open if PetitBide score >= 120
        let hand = make_hand(&["JS", "8S", "7S", "AH", "7H", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let mut state = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        state.step(BID_PASS); // P0 (1st)
        state.step(BID_PASS); // P1
        state.step(BID_PASS); // P2
        // P3 was already dealt, but P0 is actually 1st with dealer=3
        // Let's redo: dealer=0, first=1, P1/P2/P3 pass, P0 is 4th
        let mut state2 = GameState::new(0, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        state2.step(BID_PASS); // P1
        state2.step(BID_PASS); // P2
        state2.step(BID_PASS); // P3
        assert_eq!(state2.current_player, 0);
        let best_score = (0..4u8)
            .map(|s| petit_bide_score(hand, Suit::from_u8(s)))
            .max()
            .unwrap();
        if best_score < 120 {
            let action = moelleux_bid(&state2);
            assert_eq!(action, BID_PASS, "4th position ISO: should PASS with score {} < 120", best_score);
        }
    }

    #[test]
    fn test_moelleux_above_80_uses_petit_bide() {
        // Strong hand → should bid above 80 using PetitBide
        let hand = make_hand(&["JS", "9S", "AS", "10S", "KS", "AH", "7D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        let state = GameState::new(3, [hand, other_hands[0], other_hands[1], other_hands[2]]);
        let action = moelleux_bid(&state);
        assert_ne!(action, BID_PASS, "Strong hand should bid");
        let (val, _) = bidding::decode_bid(action);
        assert!(val >= 9, "Strong hand should bid above 80, got {}", val * 10);
    }

    #[test]
    fn test_moelleux_respond_aux_as_with_trumps() {
        // Partner opened "aux as" 80 in Clubs (1st/2nd position)
        // Partner has 2 aces NOT in clubs
        // I have J+9 of Spades → should change color to Spades with ace bonus
        let hand = make_hand(&["JS", "9S", "8S", "7H", "8H", "7D", "8D", "7C"]);
        let remaining = ALL_CARDS & !hand;
        let other_hands = distribute_remaining(remaining, 3);
        // dealer=3, P0 is 1st position. P0 bids 80C, P1 passes, P2 responds
        let mut state = GameState::new(3, [other_hands[0], other_hands[1], hand, other_hands[2]]);
        state.step(bidding::encode_bid(8, 3)); // P0: 80 Clubs (aux as)
        state.step(BID_PASS); // P1
        assert_eq!(state.current_player, 2);
        let action = moelleux_bid(&state);
        // Should bid Spades with ace bonus making it stronger
        if action != BID_PASS && action <= 40 {
            let (val, suit) = bidding::decode_bid(action);
            assert_eq!(suit, 0, "Should change to Spades");
            assert!(val >= 9, "With ace bonus should bid >= 90, got {}", val * 10);
        }
    }

    #[test]
    fn test_find_non_ace_suit() {
        // AH, AD → non-ace suits are Spades, Clubs
        let hand = make_hand(&["AH", "AD", "JS", "9S", "8S", "7H", "7D", "7C"]);
        let suit = moelleux::find_non_ace_suit(hand);
        let bits = suit_bits(hand, Suit::from_u8(suit));
        let has_ace = bits & (1 << 7) != 0;
        assert!(!has_ace, "Should return non-ace suit, got {:?} which has ace", Suit::from_u8(suit));
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_moelleux_random_deals_complete() {
        use rand::Rng;
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let mut state = GameState::deal_random(0, &mut rng);
            while state.phase == Phase::Bidding && !state.is_terminal() {
                let action = moelleux_bid(&state);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Moelleux returned illegal action {} (legal: {:b})",
                    action, legal
                );
                state.step(action);
            }
            assert!(state.phase == Phase::Playing || state.phase == Phase::Done);
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
