use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::state::*;
use super::eval_helpers::{evaluate_suit, count_side_aces, best_trump};

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
    let evals: [super::eval_helpers::SuitEval; 4] = [
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
