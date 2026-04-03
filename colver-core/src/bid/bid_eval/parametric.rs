use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::state::*;
use super::eval_helpers::{evaluate_for_trump, evaluate_suit, count_side_aces, quality_ok};

// ---------------------------------------------------------------------------
// Parametric bidder: configurable thresholds and caps
// ---------------------------------------------------------------------------

/// Configurable bidding parameters for strategy sweeps.
#[derive(Clone, Copy, Debug)]
pub struct BidParams {
    pub name: &'static str,
    /// Minimum score to bid at each level: [80, 90, 100, 110, 120, 130].
    /// Use u16::MAX to disable a level.
    pub thresholds: [u16; 6],
    /// Max bid value on opening (encoded: 8=80, ..., 13=130).
    pub opening_cap: u8,
    /// Max bid value on overcall.
    pub overcall_cap: u8,
    /// Max bid value on partner response.
    pub response_cap: u8,
    /// Minimum score to overcall.
    pub overcall_min_score: u16,
    /// Whether to apply quality gate (J/9/A/10 or 3+ cards).
    pub quality_gate: bool,
}

impl BidParams {
    pub fn ultra_conservative() -> Self {
        BidParams {
            name: "ultra_con",
            thresholds: [12, 18, 24, 30, u16::MAX, u16::MAX],
            opening_cap: 10,  // 100
            overcall_cap: 9,  // 90
            response_cap: 11, // 110
            overcall_min_score: 18,
            quality_gate: true,
        }
    }

    pub fn conservative() -> Self {
        // ≈ current improved_bid
        BidParams {
            name: "conserv",
            thresholds: [10, 15, 20, 25, u16::MAX, u16::MAX],
            opening_cap: 11,  // 110
            overcall_cap: 11, // 110
            response_cap: 12, // 120
            overcall_min_score: 14,
            quality_gate: true,
        }
    }

    pub fn moderate() -> Self {
        BidParams {
            name: "moderate",
            thresholds: [10, 14, 18, 22, 26, u16::MAX],
            opening_cap: 12,  // 120
            overcall_cap: 11, // 110
            response_cap: 12, // 120
            overcall_min_score: 14,
            quality_gate: true,
        }
    }

    pub fn balanced() -> Self {
        BidParams {
            name: "balanced",
            thresholds: [10, 13, 17, 20, 25, u16::MAX],
            opening_cap: 12,  // 120
            overcall_cap: 12, // 120
            response_cap: 13, // 130
            overcall_min_score: 13,
            quality_gate: true,
        }
    }

    pub fn aggressive() -> Self {
        BidParams {
            name: "aggress",
            thresholds: [10, 14, 17, 20, 23, 26],
            opening_cap: 13,  // 130
            overcall_cap: 12, // 120
            response_cap: 13, // 130
            overcall_min_score: 12,
            quality_gate: true,
        }
    }

    pub fn very_aggressive() -> Self {
        // ≈ heuristic_bid with quality gate
        BidParams {
            name: "v_aggr",
            thresholds: [10, 14, 17, 20, 23, 26],
            opening_cap: 13,  // 130
            overcall_cap: 13, // 130
            response_cap: 13, // 130
            overcall_min_score: 10,
            quality_gate: false,
        }
    }

    /// All presets from conservative to aggressive.
    pub fn all_presets() -> Vec<BidParams> {
        vec![
            Self::ultra_conservative(),
            Self::conservative(),
            Self::moderate(),
            Self::balanced(),
            Self::aggressive(),
            Self::very_aggressive(),
        ]
    }

    /// Fine-tune presets: 12 systematic variations around balanced.
    pub fn fine_tune_presets() -> Vec<BidParams> {
        let b = Self::balanced();
        vec![
            // 0: baseline
            b,
            // 1: all thresholds +1
            BidParams { name: "thr_tight", thresholds: [11, 14, 18, 22, 26, u16::MAX], ..b },
            // 2: all thresholds -1
            BidParams { name: "thr_loose", thresholds: [9, 12, 16, 20, 24, u16::MAX], ..b },
            // 3: 80-threshold -2
            BidParams { name: "lo80", thresholds: [8, 13, 17, 21, 25, u16::MAX], ..b },
            // 4: 90-threshold -1
            BidParams { name: "lo90", thresholds: [10, 12, 17, 21, 25, u16::MAX], ..b },
            // 5: 90-threshold +2
            BidParams { name: "hi90", thresholds: [10, 15, 17, 21, 25, u16::MAX], ..b },
            // 6: 110-threshold +1 (old balanced value)
            BidParams { name: "hi110", thresholds: [10, 13, 17, 21, 25, u16::MAX], ..b },
            // 7: lower open+over cap to 110
            BidParams { name: "cap_110", opening_cap: 11, overcall_cap: 11, ..b },
            // 8: raise open+over cap to 130
            BidParams { name: "cap_130", opening_cap: 13, overcall_cap: 13, ..b },
            // 9: lower response cap to 120
            BidParams { name: "resp_120", response_cap: 12, ..b },
            // 10: quality gate off
            BidParams { name: "no_qg", quality_gate: false, ..b },
            // 11: lower overcall min score
            BidParams { name: "oc_min11", overcall_min_score: 11, ..b },
        ]
    }
}

/// Score→bid-value using configurable thresholds.
fn parametric_bid_value(score: u16, thresholds: &[u16; 6]) -> u8 {
    // thresholds[0..6] = min score for 80, 90, 100, 110, 120, 130
    let bid_values = [8u8, 9, 10, 11, 12, 13];
    let mut result = 0u8;
    for i in 0..6 {
        if score >= thresholds[i] {
            result = bid_values[i];
        } else {
            break;
        }
    }
    result
}

/// Parametric bidder with configurable strategy.
pub fn parametric_bid(state: &GameState, params: &BidParams) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: always PASS
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Coinche check (opponent bid, not yet coinched)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);
        if bidder_team != my_team {
            let their_suit = Suit::from_u8(state.last_bid_suit);
            let my_eval = evaluate_suit(hand, their_suit);
            if my_eval.has_jack && my_eval.has_nine {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            if my_eval.trump_count >= 4 && count_side_aces(hand, their_suit) >= 1 {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return parametric_opening(hand, &legal, params);
    }

    // Partner response
    if state.last_bidder == partner {
        return parametric_respond(state, hand, &legal, params);
    }

    // Overcall
    parametric_overcall(state, hand, &legal, params)
}

fn parametric_opening(hand: CardSet, legal: &u64, params: &BidParams) -> u8 {
    let mut scores = [0u16; 4];
    for suit_idx in 0..4u8 {
        scores[suit_idx as usize] = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
    }

    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for i in 0..4u8 {
        if scores[i as usize] > best_score {
            best_score = scores[i as usize];
            best_suit = i;
        }
    }

    if params.quality_gate && !quality_ok(hand, Suit::from_u8(best_suit)) {
        return BID_PASS;
    }

    let mut bid_value = parametric_bid_value(best_score, &params.thresholds);
    if bid_value == 0 {
        return BID_PASS;
    }
    if bid_value > params.opening_cap {
        bid_value = params.opening_cap;
    }

    let action = bidding::encode_bid(bid_value, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

fn parametric_respond(state: &GameState, hand: CardSet, legal: &u64, params: &BidParams) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value;
    let my_score = evaluate_for_trump(hand, partner_suit);

    // Don't push above response cap - 1 (partner already committed)
    if partner_value >= params.response_cap {
        return BID_PASS;
    }

    // Support raise in partner's suit
    let target_value = parametric_bid_value(my_score, &params.thresholds);
    // Raise to at least partner_value + 1, at most response_cap
    if target_value > partner_value {
        let raise_value = target_value.min(params.response_cap);
        if raise_value > partner_value {
            let action = bidding::encode_bid(raise_value, state.last_bid_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    // Alternative suit bid
    let mut alt_best_suit = 0u8;
    let mut alt_best_score = 0u16;
    for suit_idx in 0..4u8 {
        if suit_idx == state.last_bid_suit {
            continue;
        }
        let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
        if score > alt_best_score {
            alt_best_score = score;
            alt_best_suit = suit_idx;
        }
    }

    if alt_best_score >= 16
        && (!params.quality_gate || quality_ok(hand, Suit::from_u8(alt_best_suit)))
    {
        let mut alt_value = parametric_bid_value(alt_best_score, &params.thresholds);
        if alt_value > params.opening_cap {
            alt_value = params.opening_cap;
        }
        if alt_value > partner_value {
            let action = bidding::encode_bid(alt_value, alt_best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}

fn parametric_overcall(state: &GameState, hand: CardSet, legal: &u64, params: &BidParams) -> u8 {
    // Don't compete above overcall cap
    if state.last_bid_value >= params.overcall_cap {
        return BID_PASS;
    }

    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for suit_idx in 0..4u8 {
        if suit_idx == state.last_bid_suit {
            continue;
        }
        let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
        if score > best_score {
            best_score = score;
            best_suit = suit_idx;
        }
    }

    if best_score >= params.overcall_min_score
        && (!params.quality_gate || quality_ok(hand, Suit::from_u8(best_suit)))
    {
        let mut bid_value = parametric_bid_value(best_score, &params.thresholds);
        if bid_value > params.overcall_cap {
            bid_value = params.overcall_cap;
        }
        if bid_value > state.last_bid_value {
            let action = bidding::encode_bid(bid_value, best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}
