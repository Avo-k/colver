use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::state::*;

/// Weights for trump card evaluation (indexed by rank: 7,8,9,J,Q,K,10,A).
const TRUMP_EVAL: [u16; 8] = [0, 0, 6, 8, 1, 1, 3, 4];

/// Evaluate hand strength assuming `trump` is the trump suit.
/// Returns score 0-40+. All bitwise, ~50 ops.
pub fn evaluate_for_trump(hand: CardSet, trump: Suit) -> u16 {
    let mut score: u16 = 0;

    // Trump suit evaluation
    let trump_bits = suit_bits(hand, trump);
    let trump_count = trump_bits.count_ones() as u16;

    // Trump honor values via lookup
    let mut b = trump_bits;
    while b != 0 {
        let rank = b.trailing_zeros() as usize;
        score += TRUMP_EVAL[rank];
        b &= b - 1;
    }

    // Trump length bonus: max(0, count - 2) * 2
    if trump_count > 2 {
        score += (trump_count - 2) * 2;
    }

    // Side suits (3 non-trump suits)
    for suit_idx in 0..4u8 {
        if suit_idx == trump as u8 {
            continue;
        }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        let count = bits.count_ones();

        // Ace = +3
        if bits & (1 << 7) != 0 {
            score += 3;
        }

        // Void = +3, singleton = +1
        if count == 0 {
            score += 3;
        } else if count == 1 {
            score += 1;
        }
    }

    score
}

/// Find the best trump suit and its score.
pub fn best_trump(hand: CardSet) -> (Suit, u16) {
    let mut best_suit = Suit::Spades;
    let mut best_score = 0u16;

    for &suit in &ALL_SUITS {
        let score = evaluate_for_trump(hand, suit);
        if score > best_score {
            best_score = score;
            best_suit = suit;
        }
    }

    (best_suit, best_score)
}

/// Score-to-bid-value threshold table.
/// Returns the bid value encoded (value/10), or 0 for PASS.
fn score_to_bid_value(score: u16) -> u8 {
    if score < 10 {
        0 // PASS
    } else if score < 14 {
        8 // 80
    } else if score < 17 {
        9 // 90
    } else if score < 20 {
        10 // 100
    } else if score < 23 {
        11 // 110
    } else if score < 26 {
        12 // 120
    } else {
        13 // 130
    }
}

/// Fast deterministic bid. ~200 ops, suitable for rollouts (millions/sec).
/// Only reads own hand. Never coinches or surcoinches.
pub fn heuristic_bid(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];

    // Evaluate all 4 suits
    let mut scores = [0u16; 4];
    for suit_idx in 0..4u8 {
        scores[suit_idx as usize] = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
    }

    // If partner is last bidder, boost partner's suit by +3
    if state.last_bid_value > 0 {
        let partner = GameState::partner(player);
        if state.last_bidder == partner {
            scores[state.last_bid_suit as usize] += 3;
        }
    }

    // Find best suit and score
    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for i in 0..4u8 {
        if scores[i as usize] > best_score {
            best_score = scores[i as usize];
            best_suit = i;
        }
    }

    // Map score to bid value
    let bid_value = score_to_bid_value(best_score);
    if bid_value == 0 {
        return BID_PASS;
    }

    // Can't overbid? → PASS
    if bid_value <= state.last_bid_value {
        return BID_PASS;
    }

    // Encode and validate
    let action = bidding::encode_bid(bid_value, best_suit);
    let legal = state.legal_actions();
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

/// Detailed suit evaluation for smart bidder.
struct SuitEval {
    has_jack: bool,
    has_nine: bool,
    trump_count: u32,
    score: u16,
}

fn evaluate_suit(hand: CardSet, suit: Suit) -> SuitEval {
    let bits = suit_bits(hand, suit);
    let trump_count = bits.count_ones();
    SuitEval {
        has_jack: bits & (1 << 3) != 0, // Rank::Jack = 3
        has_nine: bits & (1 << 2) != 0,  // Rank::Nine = 2
        trump_count,
        score: evaluate_for_trump(hand, suit),
    }
}

/// Count side aces (aces in non-trump suits).
fn count_side_aces(hand: CardSet, trump: Suit) -> u32 {
    let mut count = 0u32;
    for suit_idx in 0..4u8 {
        if suit_idx == trump as u8 {
            continue;
        }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        if bits & (1 << 7) != 0 {
            count += 1;
        }
    }
    count
}

/// Convention-based bid using human Belote Contree strategy.
/// More nuanced than heuristic, still deterministic.
pub fn smart_bid(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: only consider coinche/surcoinche logic or pass
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Check if we should coinche (opponent bid, not yet coinched)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);

        if bidder_team != my_team {
            // Opponent made last bid - consider coinche
            let their_suit = Suit::from_u8(state.last_bid_suit);
            let my_their_suit = evaluate_suit(hand, their_suit);

            if my_their_suit.has_jack && my_their_suit.has_nine {
                // I hold their key trump cards
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            if my_their_suit.trump_count >= 4 {
                // Trump exhaustion
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            // High bids (120+) with 3+ trumps + side ace → likely to fail, punish
            if state.last_bid_value >= 12
                && my_their_suit.trump_count >= 3
                && count_side_aces(hand, their_suit) >= 1
            {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
        }
    }

    // No bid yet: opening
    if state.last_bid_value == 0 {
        return smart_opening(hand, &legal);
    }

    // Partner made last bid: respond
    if state.last_bidder == partner {
        return smart_respond(state, hand, &legal);
    }

    // Opponent made last bid: overcall
    smart_overcall(state, hand, &legal)
}

fn smart_opening(hand: CardSet, legal: &u64) -> u8 {
    // Evaluate all suits
    let evals: [SuitEval; 4] = [
        evaluate_suit(hand, Suit::Spades),
        evaluate_suit(hand, Suit::Hearts),
        evaluate_suit(hand, Suit::Diamonds),
        evaluate_suit(hand, Suit::Clubs),
    ];

    // Find best suit with J or 9
    let mut best_idx: Option<usize> = None;
    let mut best_score = 0u16;

    for i in 0..4 {
        if (evals[i].has_jack || evals[i].has_nine) && evals[i].score > best_score {
            best_score = evals[i].score;
            best_idx = Some(i);
        }
    }

    if let Some(idx) = best_idx {
        let eval = &evals[idx];
        let side_aces = count_side_aces(hand, Suit::from_u8(idx as u8));

        let bid_value = if eval.has_jack && eval.has_nine {
            // Has "le 34"
            if side_aces >= 2 || eval.trump_count >= 4 {
                10 // 100
            } else if side_aces >= 1 {
                9 // 90
            } else {
                8 // 80
            }
        } else {
            // J XOR 9 + 2 other trumps → 80 (signals missing the other)
            if eval.trump_count >= 3 {
                8 // 80
            } else {
                0 // not enough support
            }
        };

        if bid_value > 0 {
            let action = bidding::encode_bid(bid_value, idx as u8);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    // Fallback: 2+ aces but no J/9 → 80 in best-supported suit ("aux as")
    let total_aces = (0..4u8)
        .filter(|&i| suit_bits(hand, Suit::from_u8(i)) & (1 << 7) != 0)
        .count();

    if total_aces >= 2 {
        // Find best suit by score
        let (best_suit, _) = best_trump(hand);
        let action = bidding::encode_bid(8, best_suit as u8);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    BID_PASS
}

fn smart_respond(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value;
    let my_eval = evaluate_suit(hand, partner_suit);

    if partner_value == 8 {
        // Partner bid 80: they signaled J XOR 9. Respond 90 if I have the missing honor.
        if my_eval.has_jack || my_eval.has_nine {
            let action = bidding::encode_bid(9, state.last_bid_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }
    // Partner bid 90+: they have J+9, don't escalate. PASS.

    BID_PASS
}

fn smart_overcall(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    // Only overcall if opponent bid < 100; above that, let them have it
    if state.last_bid_value >= 10 {
        return BID_PASS;
    }

    // I have J+9 in another suit with real strength → bid my suit, capped at 100
    for suit_idx in 0..4u8 {
        if suit_idx == state.last_bid_suit {
            continue;
        }
        let eval = evaluate_suit(hand, Suit::from_u8(suit_idx));
        if eval.has_jack && eval.has_nine && eval.score >= 14 {
            // Bid my suit at last_bid_value + 1, minimum 80, capped at 100
            let min_value = if state.last_bid_value + 1 < 8 {
                8
            } else {
                state.last_bid_value + 1
            };
            if min_value <= 10 {
                let action = bidding::encode_bid(min_value, suit_idx);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
        }
    }

    BID_PASS
}

#[cfg(test)]
mod tests {
    use super::*;

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
