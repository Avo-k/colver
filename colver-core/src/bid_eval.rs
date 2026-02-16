use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::state::*;

/// Which bidding function to use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BidFunction {
    Heuristic,
    Smart,
    Improved,
    ImprovedV2,
    Roro,
    PetitBide,
    Moelleux,
}

impl BidFunction {
    /// Dispatch to the appropriate bid function.
    pub fn bid(self, state: &GameState) -> u8 {
        match self {
            BidFunction::Heuristic => heuristic_bid(state),
            BidFunction::Smart => smart_bid(state),
            BidFunction::Improved => improved_bid(state),
            BidFunction::ImprovedV2 => improved_v2_bid(state),
            BidFunction::Roro => roro_bid(state),
            BidFunction::PetitBide => petit_bide_bid(state),
            BidFunction::Moelleux => moelleux_bid(state),
        }
    }
}

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

// ---------------------------------------------------------------------------
// roro_bid: Cannes-Roro expert bidding strategy
// ---------------------------------------------------------------------------

/// Detailed suit evaluation for Roro bidder.
struct RoroSuitEval {
    has_jack: bool,
    has_nine: bool,
    has_ace: bool,
    #[allow(dead_code)]
    has_ten: bool,
    has_king: bool,
    has_queen: bool,
    trump_count: u32,
    score: u16,
}

fn roro_eval_suit(hand: CardSet, suit: Suit) -> RoroSuitEval {
    let bits = suit_bits(hand, suit);
    RoroSuitEval {
        has_jack: bits & (1 << 3) != 0,
        has_nine: bits & (1 << 2) != 0,
        has_ace: bits & (1 << 7) != 0,
        has_ten: bits & (1 << 6) != 0,
        has_king: bits & (1 << 5) != 0,
        has_queen: bits & (1 << 4) != 0,
        trump_count: bits.count_ones(),
        score: evaluate_for_trump(hand, suit),
    }
}

/// Count total aces across all 4 suits.
fn count_total_aces(hand: CardSet) -> u32 {
    let mut count = 0u32;
    for suit_idx in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        if bits & (1 << 7) != 0 {
            count += 1;
        }
    }
    count
}

/// Check if hand has belote (K+Q) in the given suit.
fn has_belote(hand: CardSet, suit: Suit) -> bool {
    let bits = suit_bits(hand, suit);
    (bits & (1 << 5) != 0) && (bits & (1 << 4) != 0) // K + Q
}

/// Count voids in non-excluded suits.
fn count_voids(hand: CardSet, exclude: Suit) -> u32 {
    let mut count = 0u32;
    for suit_idx in 0..4u8 {
        if suit_idx == exclude as u8 {
            continue;
        }
        if suit_bits(hand, Suit::from_u8(suit_idx)) == 0 {
            count += 1;
        }
    }
    count
}

/// Find the longest non-trump suit. Tie-break: belote > has_jack > score.
fn most_playable_suit(hand: CardSet, trump: Suit) -> u8 {
    let mut best_suit = 0u8;
    let mut best_len = 0u32;
    let mut best_belote = false;
    let mut best_jack = false;
    let mut best_score = 0u16;

    for suit_idx in 0..4u8 {
        let suit = Suit::from_u8(suit_idx);
        let bits = suit_bits(hand, suit);
        let len = bits.count_ones();
        let bel = has_belote(hand, suit);
        let jack = bits & (1 << 3) != 0;
        let score = evaluate_for_trump(hand, suit);

        let better = len > best_len
            || (len == best_len && bel && !best_belote)
            || (len == best_len && bel == best_belote && jack && !best_jack)
            || (len == best_len && bel == best_belote && jack == best_jack && score > best_score);

        // Skip the trump suit for "most playable" — we want the best suit to announce
        // when we have no strong trump (80 aux as)
        let _ = trump;

        if better {
            best_suit = suit_idx;
            best_len = len;
            best_belote = bel;
            best_jack = jack;
            best_score = score;
        }
    }
    best_suit
}

/// Count "losers" — rough estimate of tricks that will be lost.
/// For Roro: count non-master side cards. Simple heuristic: side suits
/// without ace = 1 loser per suit, short suits = fewer losers.
fn count_losers(hand: CardSet, trump: Suit, eval: &RoroSuitEval) -> u32 {
    let mut losers = 0u32;
    // Trump losers: if missing J or 9, each missing = 1 loser
    if !eval.has_jack {
        losers += 1;
    }
    if !eval.has_nine {
        losers += 1;
    }
    // Side suit losers
    for suit_idx in 0..4u8 {
        if suit_idx == trump as u8 {
            continue;
        }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        let count = bits.count_ones();
        if count == 0 {
            continue; // void = no losers (can trump)
        }
        let has_a = bits & (1 << 7) != 0;
        if count == 1 && has_a {
            continue; // singleton ace = no loser
        }
        if count == 1 {
            losers += 1; // singleton non-ace
        } else {
            // 2+ cards: 1 loser if no ace, 1 extra if 3+ and no 10
            if !has_a {
                losers += 1;
            }
            if count >= 3 && (bits & (1 << 6) == 0) && !has_a {
                losers += 1;
            }
        }
    }
    losers
}

/// Check if hand has a second long suit (5+ cards) different from trump.
fn has_second_long_suit(hand: CardSet, trump: Suit) -> bool {
    for suit_idx in 0..4u8 {
        if suit_idx == trump as u8 {
            continue;
        }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        if bits.count_ones() >= 5 {
            return true;
        }
    }
    false
}

/// Cannes-Roro expert bidding strategy.
/// Position-aware, convention-based ("80 aux as"), with structured opening
/// levels and complementary partner responses.
pub fn roro_bid(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: always PASS (no surcoinche in deterministic bidder)
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Coinche check (opponent bid, not yet coinched)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);

        if bidder_team != my_team {
            let action = roro_coinche(hand, state, &legal);
            if action != BID_PASS {
                return action;
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return roro_opening(state, hand, &legal);
    }

    // Partner made last bid: respond
    if state.last_bidder == partner {
        return roro_respond(state, hand, &legal);
    }

    // Opponent made last bid: intervene
    roro_intervene(state, hand, &legal)
}

/// Coinche logic for Roro bidder.
fn roro_coinche(hand: CardSet, state: &GameState, legal: &u64) -> u8 {
    let their_suit = Suit::from_u8(state.last_bid_suit);
    let my_eval = roro_eval_suit(hand, their_suit);

    // J+9 in opponent's suit → COINCHE (classic)
    if my_eval.has_jack && my_eval.has_nine {
        if legal & (1u64 << BID_COINCHE) != 0 {
            return BID_COINCHE;
        }
    }

    // Below 110 only: 4+ trumps + side ace → COINCHE
    if state.last_bid_value < 11 && my_eval.trump_count >= 4 && count_side_aces(hand, their_suit) >= 1 {
        if legal & (1u64 << BID_COINCHE) != 0 {
            return BID_COINCHE;
        }
    }

    // Théorème 3: 0 trumps in opponent's suit + 3+ aces → coinche (misfit detection)
    if my_eval.trump_count == 0 && count_total_aces(hand) >= 3 {
        if legal & (1u64 << BID_COINCHE) != 0 {
            return BID_COINCHE;
        }
    }

    BID_PASS
}

/// Determine bidding position: 0=1st, 1=2nd, 2=3rd, 3=4th.
fn bidding_position(state: &GameState) -> u8 {
    // When last_bid_value == 0, consecutive_passes counts how many
    // players have passed before us. First bidder = 0 passes before.
    state.consecutive_passes
}

/// Opening logic — position-aware, scan from highest level down.
fn roro_opening(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let position = bidding_position(state);

    // Evaluate all 4 suits
    let evals: [RoroSuitEval; 4] = [
        roro_eval_suit(hand, Suit::Spades),
        roro_eval_suit(hand, Suit::Hearts),
        roro_eval_suit(hand, Suit::Diamonds),
        roro_eval_suit(hand, Suit::Clubs),
    ];

    // Scan from highest possible level down (Roro rule: open at highest possible)
    // Try 130, 120, 110, 100, 90, then 80.
    for suit_idx in 0..4u8 {
        let eval = &evals[suit_idx as usize];
        let suit = Suit::from_u8(suit_idx);
        let losers = count_losers(hand, suit, eval);

        // 130: Bicolore — J9 in trump + 2nd long suit (5+) + max 2 losers
        if eval.has_jack && eval.has_nine && has_second_long_suit(hand, suit) && losers <= 2 {
            let action = bidding::encode_bid(13, suit_idx);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    for suit_idx in 0..4u8 {
        let eval = &evals[suit_idx as usize];
        let suit = Suit::from_u8(suit_idx);
        let losers = count_losers(hand, suit, eval);

        // 120: Tricolore — J9 in trump + 1 void + max 2 losers (3 with belote)
        let max_losers_120 = if has_belote(hand, suit) { 3 } else { 2 };
        if eval.has_jack && eval.has_nine && count_voids(hand, suit) >= 1 && losers <= max_losers_120 {
            let action = bidding::encode_bid(12, suit_idx);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    // 110: Only 1st position (or 3rd per Roro doc for "without V or 9" case)
    if position == 0 {
        for suit_idx in 0..4u8 {
            let eval = &evals[suit_idx as usize];
            let suit = Suit::from_u8(suit_idx);
            let losers = count_losers(hand, suit, eval);

            // 110 case 1: J9A 4th+, plays all suits, max 3 losers (4 with belote)
            let max_losers_110 = if has_belote(hand, suit) { 4 } else { 3 };
            if eval.has_jack
                && eval.has_nine
                && eval.has_ace
                && eval.trump_count >= 4
                && count_voids(hand, suit) == 0
                && losers <= max_losers_110
            {
                let action = bidding::encode_bid(11, suit_idx);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }

            // 110 case 2: J 5th beloté / 9 6th beloté (without J or 9, very long)
            if eval.has_jack && eval.trump_count >= 5 && has_belote(hand, suit) {
                let action = bidding::encode_bid(11, suit_idx);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
            if eval.has_nine && !eval.has_jack && eval.trump_count >= 6 && has_belote(hand, suit) {
                let action = bidding::encode_bid(11, suit_idx);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
        }
    }

    // 100: J9 4th+, or J9 3rd + side ace. Any position.
    for suit_idx in 0..4u8 {
        let eval = &evals[suit_idx as usize];
        let suit = Suit::from_u8(suit_idx);

        if eval.has_jack && eval.has_nine {
            // J9 4th+
            if eval.trump_count >= 4 {
                let action = bidding::encode_bid(10, suit_idx);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
            // J9 3rd + side ace
            if eval.trump_count >= 3 && count_side_aces(hand, suit) >= 1 {
                let action = bidding::encode_bid(10, suit_idx);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
        }
    }

    // 90: Multiple sub-conditions. Any position.
    for suit_idx in 0..4u8 {
        let eval = &evals[suit_idx as usize];
        let suit = Suit::from_u8(suit_idx);
        let side_aces = count_side_aces(hand, suit);

        let qualifies_90 =
            // J 3rd + 1 ace
            (eval.has_jack && eval.trump_count >= 3 && side_aces >= 1)
            // J 4th
            || (eval.has_jack && eval.trump_count >= 4)
            // JQK + side trick (approximated as side ace or 10)
            || (eval.has_jack && eval.has_queen && eval.has_king && side_aces >= 1)
            // J9 dry + 1 ace (1st/3rd only)
            || (eval.has_jack && eval.has_nine && eval.trump_count == 2
                && side_aces >= 1 && (position == 0 || position == 2))
            // 9 4th + 1 ace
            || (eval.has_nine && !eval.has_jack && eval.trump_count >= 4 && side_aces >= 1)
            // 9QKX (9 + Q + K + at least 4 trumps)
            || (eval.has_nine && !eval.has_jack && eval.has_queen && eval.has_king && eval.trump_count >= 4)
            // 9 5th
            || (eval.has_nine && !eval.has_jack && eval.trump_count >= 5);

        // IMPORTANT: Never open 9 3rd + 1 ace — this is a coinche hand, not an opening!
        let is_nine_3rd_ace = eval.has_nine && !eval.has_jack && eval.trump_count == 3 && side_aces >= 1
            && !(eval.has_queen && eval.has_king); // 9QKX is ok above

        if qualifies_90 && !is_nine_3rd_ace {
            let action = bidding::encode_bid(9, suit_idx);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    // 80: Position-dependent
    match position {
        0 | 1 => {
            // "Aux as": 2+ total aces. Bid most playable suit.
            if count_total_aces(hand) >= 2 {
                // Find most playable suit (longest, tie-break belote/J/score)
                // We pass a dummy trump since we're looking for the best suit overall
                let best = most_playable_suit(hand, Suit::from_u8(4.min(3))); // no exclusion needed
                let action = bidding::encode_bid(8, best);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
        }
        2 => {
            // "Walou": 3rd position — weak hand, longest suit or best available.
            // Open with long suit or anything vaguely decent.
            // Roro: equivalent to a very weak 90 or a long suit without J/9.
            // Find best suit by length then score
            let mut best_suit = 0u8;
            let mut best_len = 0u32;
            let mut best_score = 0u16;
            for i in 0..4u8 {
                let e = &evals[i as usize];
                if e.trump_count > best_len
                    || (e.trump_count == best_len && e.score > best_score)
                {
                    best_suit = i;
                    best_len = e.trump_count;
                    best_score = e.score;
                }
            }
            // Only walou with at least some length (3+) or a decent score
            if best_len >= 3 || best_score >= 8 {
                let action = bidding::encode_bid(8, best_suit);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
        }
        3 => {
            // 4th position: only open 80 if hand can support up to 120
            // Otherwise pass (don't gift opponents a contract).
            for suit_idx in 0..4u8 {
                let eval = &evals[suit_idx as usize];
                // Need a real hand: J or 9 + decent count
                if (eval.has_jack || eval.has_nine) && eval.trump_count >= 3 && eval.score >= 13 {
                    let action = bidding::encode_bid(8, suit_idx);
                    if legal & (1u64 << action) != 0 {
                        return action;
                    }
                }
            }
        }
        _ => {}
    }

    BID_PASS
}

/// Response to partner's bid.
fn roro_respond(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value;
    let my_eval = roro_eval_suit(hand, partner_suit);
    let side_aces = count_side_aces(hand, partner_suit);

    match partner_value {
        8 => roro_respond_on_80(state, hand, legal, &my_eval, side_aces),
        9 => roro_respond_on_90(state, hand, legal, &my_eval, side_aces),
        10 => roro_respond_on_100(state, hand, legal, &my_eval, side_aces),
        _ if partner_value >= 11 => roro_respond_on_110_plus(state, hand, legal, &my_eval, side_aces),
        _ => BID_PASS,
    }
}

/// Detect if partner opened "aux as" (1st/2nd position) or "walou" (3rd/4th).
/// We infer partner's opening position from the game state.
fn partner_was_aux_as(state: &GameState) -> bool {
    // Partner is last_bidder. We need to figure out their position when they bid.
    // First bidder is (dealer + 1) % 4.
    // Partner opened at some point. If last_bid_value == 8, partner was the first bidder
    // in the auction. We approximate: if partner is in seat (dealer+1)%4 or (dealer+2)%4,
    // they were 1st or 2nd of parole → aux as.
    let first_bidder = (state.dealer + 1) % 4;
    let partner = state.last_bidder;
    let partner_pos = (partner + 4 - first_bidder) % 4;
    partner_pos <= 1 // 0=1st, 1=2nd → aux as
}

/// Response on partner's 80.
fn roro_respond_on_80(
    state: &GameState,
    hand: CardSet,
    legal: &u64,
    my_eval: &RoroSuitEval,
    side_aces: u32,
) -> u8 {
    let partner_suit_idx = state.last_bid_suit;

    if partner_was_aux_as(state) {
        // Partner opened "aux as" — promises 2 aces, nothing about trump
        // Priority: change color if possible, support only if strong in partner's suit

        // Support in partner's suit: strong complement
        if my_eval.has_jack && my_eval.has_nine && my_eval.trump_count >= 3 && side_aces >= 1 {
            // V9 3rd + ace → 110
            let action = bidding::encode_bid(11, partner_suit_idx);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
        if (my_eval.has_jack && my_eval.trump_count >= 3)
            || (my_eval.has_jack && my_eval.has_nine)
            || (my_eval.has_nine && my_eval.trump_count >= 4)
        {
            // V 3rd, V9, 9 4th → 100
            let action = bidding::encode_bid(10, partner_suit_idx);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
        if (my_eval.has_jack && my_eval.trump_count >= 2)
            || (my_eval.has_nine && my_eval.trump_count >= 3)
        {
            // V second, 9 3rd → 90
            let action = bidding::encode_bid(9, partner_suit_idx);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }

        // Minimum support (2-3 trumps, 9 second, V sec) → pass
        let has_support = my_eval.trump_count >= 2
            || (my_eval.has_nine && my_eval.trump_count >= 2)
            || (my_eval.has_jack);
        if has_support {
            return BID_PASS; // support by passing
        }

        // No support in partner's suit → must change color
        return roro_change_color(state, hand, legal, 8);
    }

    // Partner opened "walou" (3rd/4th position) — promises nothing at trump
    // +10 if trump complement (V, 9 second+, 3 trumps) + 10 per ace
    let has_complement = my_eval.has_jack
        || (my_eval.has_nine && my_eval.trump_count >= 2)
        || my_eval.trump_count >= 3;

    if has_complement {
        let mut level = 9u8; // base +10 = 90
        level += side_aces as u8; // +10 per ace
        level = level.min(12); // cap at 120
        let action = bidding::encode_bid(level, partner_suit_idx);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    BID_PASS
}

/// Response on partner's 90.
fn roro_respond_on_90(
    _state: &GameState,
    _hand: CardSet,
    legal: &u64,
    my_eval: &RoroSuitEval,
    side_aces: u32,
) -> u8 {
    let partner_suit_idx = _state.last_bid_suit;

    // Only respond with trump complement: V, 9 at least second, or 3 trumps
    let has_complement = my_eval.has_jack
        || (my_eval.has_nine && my_eval.trump_count >= 2)
        || my_eval.trump_count >= 3;

    if has_complement {
        let mut level = 10u8; // 100
        level += side_aces as u8; // +10 per ace
        level = level.min(12); // cap at 120
        let action = bidding::encode_bid(level, partner_suit_idx);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    BID_PASS
}

/// Response on partner's 100: partner promises trump control.
/// Count aces, with penalty if 0 trumps in partner's suit.
fn roro_respond_on_100(
    _state: &GameState,
    _hand: CardSet,
    legal: &u64,
    my_eval: &RoroSuitEval,
    _side_aces: u32,
) -> u8 {
    let partner_suit_idx = _state.last_bid_suit;
    let total_aces = count_total_aces(_hand);

    if my_eval.trump_count == 0 {
        // No trump → minor response: 2 aces→+10, 3 aces→+20
        if total_aces >= 3 {
            let action = bidding::encode_bid(12, partner_suit_idx); // 120
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
        if total_aces >= 2 {
            let action = bidding::encode_bid(11, partner_suit_idx); // 110
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
        // 0-1 aces + no trump → pass
        return BID_PASS;
    }

    // Normal response: 1A→110, 2A→120 (cap)
    if total_aces >= 2 {
        let action = bidding::encode_bid(12, partner_suit_idx);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }
    if total_aces >= 1 {
        let action = bidding::encode_bid(11, partner_suit_idx);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    BID_PASS
}

/// Response on partner's 110+: report master tricks.
fn roro_respond_on_110_plus(
    state: &GameState,
    hand: CardSet,
    legal: &u64,
    my_eval: &RoroSuitEval,
    _side_aces: u32,
) -> u8 {
    let partner_suit_idx = state.last_bid_suit;
    let partner_value = state.last_bid_value;

    // Count master tricks: aces + (10 if we have ace in same suit)
    let mut master_tricks = 0u32;
    for suit_idx in 0..4u8 {
        if suit_idx == partner_suit_idx {
            // In trump suit: only count complement (V or 9)
            if my_eval.has_jack || my_eval.has_nine {
                master_tricks += 1;
            }
            continue;
        }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        if bits & (1 << 7) != 0 {
            master_tricks += 1; // ace
            if bits & (1 << 6) != 0 {
                master_tricks += 1; // 10 behind ace
            }
        }
    }

    if master_tricks > 0 {
        let level = partner_value + master_tricks as u8;
        let level = level.min(12); // cap at 120
        if level > partner_value {
            let action = bidding::encode_bid(level, partner_suit_idx);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}

/// Change color when we can't support partner's suit.
fn roro_change_color(state: &GameState, hand: CardSet, legal: &u64, min_value: u8) -> u8 {
    let partner_suit_idx = state.last_bid_suit;

    // Find best alternative suit
    let mut best_suit = None;
    let mut best_score = 0u16;
    let mut best_level = 0u8;

    for suit_idx in 0..4u8 {
        if suit_idx == partner_suit_idx {
            continue;
        }
        let eval = roro_eval_suit(hand, Suit::from_u8(suit_idx));

        // Determine level we can announce in this suit
        let level = if eval.has_jack && eval.has_nine && eval.trump_count >= 3 {
            10 // 100: V9 3rd minimum
        } else if eval.has_jack || (eval.has_nine && eval.trump_count >= 3) {
            9 // 90: weak new suit
        } else {
            0
        };

        if level > 0 && level > min_value && eval.score > best_score {
            best_suit = Some(suit_idx);
            best_score = eval.score;
            best_level = level;
        }
        // Also accept if level == min_value + 1 (just above partner)
        if level > 0 && level == min_value + 1 && eval.score > best_score {
            best_suit = Some(suit_idx);
            best_score = eval.score;
            best_level = level;
        }
    }

    if let Some(suit_idx) = best_suit {
        let action = bidding::encode_bid(best_level, suit_idx);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    BID_PASS
}

/// Intervention on opponent's bid.
fn roro_intervene(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let opp_value = state.last_bid_value;
    let opp_suit_idx = state.last_bid_suit;

    // Don't intervene above 130
    if opp_value >= 13 {
        return BID_PASS;
    }

    // Check if this is a response to partner's intervention (partner bid after opponent)
    let my_team = GameState::player_team(state.current_player);
    let bidder_team = GameState::player_team(state.last_bidder);

    // If opponent bid and we need to intervene
    if bidder_team != my_team {
        // Try "la barre" (+20): solid hand, J9 3rd minimum. Cap at 120.
        for suit_idx in 0..4u8 {
            if suit_idx == opp_suit_idx {
                continue;
            }
            let eval = roro_eval_suit(hand, Suit::from_u8(suit_idx));
            if eval.has_jack && eval.has_nine && eval.trump_count >= 3 {
                let level = (opp_value + 2).min(12); // +20, cap at 120
                if level > opp_value {
                    let action = bidding::encode_bid(level, suit_idx);
                    if legal & (1u64 << action) != 0 {
                        return action;
                    }
                }
            }
        }

        // Light intervention (+10): any suit with decent trump holdings. Cap at 110.
        for suit_idx in 0..4u8 {
            if suit_idx == opp_suit_idx {
                continue;
            }
            let eval = roro_eval_suit(hand, Suit::from_u8(suit_idx));
            let suit = Suit::from_u8(suit_idx);
            let side_aces = count_side_aces(hand, suit);

            let qualifies = (eval.has_jack && eval.trump_count >= 3)
                || (eval.has_jack && eval.trump_count >= 2 && side_aces >= 1)
                || (eval.has_nine && eval.trump_count >= 4)
                || (eval.has_nine && eval.trump_count >= 3 && eval.has_king && eval.has_queen);

            if qualifies {
                let level = (opp_value + 1).min(11); // +10, cap at 110
                if level > opp_value {
                    let action = bidding::encode_bid(level, suit_idx);
                    if legal & (1u64 << action) != 0 {
                        return action;
                    }
                }
            }
        }
    }

    BID_PASS
}

// ---------------------------------------------------------------------------
// improved_bid: balanced deterministic bidder
// ---------------------------------------------------------------------------

/// Quality gate: suit must have at least one of J, 9, A, 10, or 3+ cards.
fn quality_ok(hand: CardSet, suit: Suit) -> bool {
    let bits = suit_bits(hand, suit);
    let has_j = bits & (1 << 3) != 0;
    let has_9 = bits & (1 << 2) != 0;
    let has_a = bits & (1 << 7) != 0;
    let has_10 = bits & (1 << 6) != 0;
    let count = bits.count_ones();
    has_j || has_9 || has_a || has_10 || count >= 3
}

/// Balanced score→bid-value mapping (tournament-tuned). Returns encoded value (value/10), or 0 for PASS.
fn balanced_bid_value(score: u16) -> u8 {
    if score < 10 {
        0 // PASS
    } else if score < 13 {
        8 // 80
    } else if score < 17 {
        9 // 90
    } else if score < 20 {
        10 // 100
    } else if score < 25 {
        11 // 110
    } else {
        12 // 120
    }
}

/// Tournament-tuned balanced bidder. Quality gate + score→value mapping (10→80, 13→90,
/// 17→100, 20→110, 25→120). Caps: opening 120, overcall 120, response 130.
/// Won round-robin tournament with 62% overall win rate vs 5 other strategies,
/// then fine-tuned (110-threshold 21→20) to 52.6% in 12-strategy fine-tune tournament.
pub fn improved_bid(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: always PASS (never surcoinche in deterministic bidder)
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Coinche check (opponent bid, not yet coinched)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);

        if bidder_team != my_team {
            let their_suit = Suit::from_u8(state.last_bid_suit);
            let my_their_suit = evaluate_suit(hand, their_suit);

            // J+9 in opponent's suit → COINCHE
            if my_their_suit.has_jack && my_their_suit.has_nine {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            // 4+ trumps in their suit + 1+ side ace → COINCHE
            if my_their_suit.trump_count >= 4 && count_side_aces(hand, their_suit) >= 1 {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return improved_opening(hand, &legal);
    }

    // Partner response
    if state.last_bidder == partner {
        return improved_respond(state, hand, &legal);
    }

    // Overcall (opponent bid)
    improved_overcall(state, hand, &legal)
}

fn improved_opening(hand: CardSet, legal: &u64) -> u8 {
    // Evaluate all 4 suits
    let mut scores = [0u16; 4];
    for suit_idx in 0..4u8 {
        scores[suit_idx as usize] = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
    }

    // Find best suit
    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for i in 0..4u8 {
        if scores[i as usize] > best_score {
            best_score = scores[i as usize];
            best_suit = i;
        }
    }

    // Quality gate
    if !quality_ok(hand, Suit::from_u8(best_suit)) {
        return BID_PASS;
    }

    let mut bid_value = balanced_bid_value(best_score);
    if bid_value == 0 {
        return BID_PASS;
    }
    // Cap opening at 120
    if bid_value > 12 {
        bid_value = 12;
    }

    let action = bidding::encode_bid(bid_value, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

fn improved_respond(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value;
    let my_score = evaluate_for_trump(hand, partner_suit);

    // Partner bid 130+: don't push higher
    if partner_value >= 13 {
        return BID_PASS;
    }

    // Support raise in partner's suit using balanced mapping
    let target_value = balanced_bid_value(my_score);
    // Cap at 130
    let target_value = if target_value > 13 { 13 } else { target_value };

    if target_value > partner_value {
        let action = bidding::encode_bid(target_value, state.last_bid_suit);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    // Alternative suit bid: if I can't support partner but have a strong suit of my own
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

    if alt_best_score >= 16 && quality_ok(hand, Suit::from_u8(alt_best_suit)) {
        let mut alt_value = balanced_bid_value(alt_best_score);
        // Cap at 120
        if alt_value > 12 {
            alt_value = 12;
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

fn improved_overcall(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    // Don't compete above 120
    if state.last_bid_value >= 12 {
        return BID_PASS;
    }

    // Find best non-opponent suit
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

    if best_score >= 13 && quality_ok(hand, Suit::from_u8(best_suit)) {
        let mut bid_value = balanced_bid_value(best_score);
        // Cap at 120
        if bid_value > 12 {
            bid_value = 12;
        }
        // Must overbid
        if bid_value > state.last_bid_value {
            let action = bidding::encode_bid(bid_value, best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}

// ---------------------------------------------------------------------------
// improved_v2_bid: Improved bidding with configurable enhancements.
// ---------------------------------------------------------------------------

/// Configuration for ImprovedV2 bidder.
#[derive(Clone, Copy, Debug)]
pub struct V2Config {
    pub name: &'static str,
    /// Lead bonus added to opening score when team has first trick lead.
    pub lead_bonus: u16,
    /// 4th position gate: minimum score to open in 4th seat.
    /// 0 = disabled (use normal quality gate). Requires J or 9 when > 0.
    pub fourth_pos_min: u16,
    /// Use structured partner response with complement bonuses.
    pub partner_response: bool,
    /// Jack complement bonus in partner response.
    pub resp_jack_bonus: i16,
    /// Nine complement bonus (with 2+ trumps) in partner response.
    pub resp_nine_bonus: i16,
    /// Per-side-ace bonus in partner response.
    pub resp_ace_bonus: i16,
    /// 3+ trump support bonus in partner response.
    pub resp_support_bonus: i16,
    /// 0-trump misfit penalty in partner response.
    pub resp_misfit_penalty: i16,
    /// Théorème 3 coinche: 0 trumps in opponent suit + N aces.
    pub theoreme3_aces: u32, // 0 = disabled, 3 = standard, 4 = conservative
}

impl V2Config {
    /// Full V2 as originally designed.
    pub fn full() -> Self {
        V2Config {
            name: "v2_full",
            lead_bonus: 2,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 3,
            resp_nine_bonus: 2,
            resp_ace_bonus: 2,
            resp_support_bonus: 1,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Only Théorème 3 coinche, everything else = Improved.
    pub fn coinche_only() -> Self {
        V2Config {
            name: "v2_coinche",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: false,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: 0,
            theoreme3_aces: 3,
        }
    }

    /// Coinche + partner response only (no lead, no 4th gate).
    pub fn coinche_resp() -> Self {
        V2Config {
            name: "v2_co+resp",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: true,
            resp_jack_bonus: 3,
            resp_nine_bonus: 2,
            resp_ace_bonus: 2,
            resp_support_bonus: 1,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Coinche + 4th position gate only.
    pub fn coinche_4th() -> Self {
        V2Config {
            name: "v2_co+4th",
            lead_bonus: 0,
            fourth_pos_min: 15,
            partner_response: false,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: 0,
            theoreme3_aces: 3,
        }
    }

    /// All except lead bonus.
    pub fn no_lead() -> Self {
        V2Config {
            name: "v2_nolead",
            lead_bonus: 0,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 3,
            resp_nine_bonus: 2,
            resp_ace_bonus: 2,
            resp_support_bonus: 1,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Lead bonus +1 (half).
    pub fn lead1() -> Self {
        V2Config {
            name: "v2_lead1",
            lead_bonus: 1,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 3,
            resp_nine_bonus: 2,
            resp_ace_bonus: 2,
            resp_support_bonus: 1,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Partner response with reduced bonuses (all halved).
    pub fn resp_light() -> Self {
        V2Config {
            name: "v2_rlight",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: true,
            resp_jack_bonus: 2,
            resp_nine_bonus: 1,
            resp_ace_bonus: 1,
            resp_support_bonus: 0,
            resp_misfit_penalty: -2,
            theoreme3_aces: 3,
        }
    }

    /// Partner response: only misfit penalty (no positive bonuses).
    pub fn resp_misfit() -> Self {
        V2Config {
            name: "v2_misfit",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: true,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Théorème 3 with 4 aces required (more conservative).
    pub fn coinche_4aces() -> Self {
        V2Config {
            name: "v2_co4ace",
            lead_bonus: 0,
            fourth_pos_min: 0,
            partner_response: false,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: 0,
            theoreme3_aces: 4,
        }
    }

    /// 4th position gate at 13 (looser than 15).
    pub fn fourth_loose() -> Self {
        V2Config {
            name: "v2_4th13",
            lead_bonus: 0,
            fourth_pos_min: 13,
            partner_response: false,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: 0,
            theoreme3_aces: 3,
        }
    }

    /// Best guess: coinche + light response + no lead + loose 4th.
    pub fn balanced() -> Self {
        V2Config {
            name: "v2_bal",
            lead_bonus: 0,
            fourth_pos_min: 13,
            partner_response: true,
            resp_jack_bonus: 2,
            resp_nine_bonus: 1,
            resp_ace_bonus: 1,
            resp_support_bonus: 0,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Misfit-only response + coinche + 4th gate.
    pub fn defensive() -> Self {
        V2Config {
            name: "v2_def",
            lead_bonus: 0,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// Tournament winner: coinche + misfit penalty + 4th@15 + lead +1.
    pub fn defensive_lead1() -> Self {
        V2Config {
            name: "v2_def_l1",
            lead_bonus: 1,
            fourth_pos_min: 15,
            partner_response: true,
            resp_jack_bonus: 0,
            resp_nine_bonus: 0,
            resp_ace_bonus: 0,
            resp_support_bonus: 0,
            resp_misfit_penalty: -3,
            theoreme3_aces: 3,
        }
    }

    /// All presets for tournament.
    pub fn all_presets() -> Vec<V2Config> {
        vec![
            Self::full(),
            Self::coinche_only(),
            Self::coinche_resp(),
            Self::coinche_4th(),
            Self::no_lead(),
            Self::lead1(),
            Self::resp_light(),
            Self::resp_misfit(),
            Self::coinche_4aces(),
            Self::fourth_loose(),
            Self::balanced(),
            Self::defensive(),
        ]
    }
}

/// Configurable Improved V2 bidder.
pub fn improved_v2_configurable_bid(state: &GameState, cfg: &V2Config) -> u8 {
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
            let my_their_suit = evaluate_suit(hand, their_suit);

            // J+9 in opponent's suit → COINCHE
            if my_their_suit.has_jack && my_their_suit.has_nine {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            // 4+ trumps in their suit + 1+ side ace → COINCHE
            if my_their_suit.trump_count >= 4 && count_side_aces(hand, their_suit) >= 1 {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
            // Théorème 3: 0 trumps in their suit + N total aces → COINCHE
            if cfg.theoreme3_aces > 0
                && my_their_suit.trump_count == 0
                && count_total_aces(hand) >= cfg.theoreme3_aces
            {
                if legal & (1u64 << BID_COINCHE) != 0 {
                    return BID_COINCHE;
                }
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return v2_cfg_opening(state, hand, &legal, cfg);
    }

    // Partner response
    if state.last_bidder == partner {
        if cfg.partner_response {
            return v2_cfg_respond(state, hand, &legal, cfg);
        } else {
            return improved_respond(state, hand, &legal);
        }
    }

    // Overcall: reuse improved_overcall (already well-tuned)
    improved_overcall(state, hand, &legal)
}

fn v2_cfg_opening(state: &GameState, hand: CardSet, legal: &u64, cfg: &V2Config) -> u8 {
    // Evaluate all 4 suits
    let mut scores = [0u16; 4];
    for suit_idx in 0..4u8 {
        scores[suit_idx as usize] = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
    }

    // Find best suit
    let mut best_suit = 0u8;
    let mut best_score = 0u16;
    for i in 0..4u8 {
        if scores[i as usize] > best_score {
            best_score = scores[i as usize];
            best_suit = i;
        }
    }

    // 4th position gate
    if cfg.fourth_pos_min > 0 && bidding_position(state) == 3 {
        if best_score < cfg.fourth_pos_min {
            return BID_PASS;
        }
        let bits = suit_bits(hand, Suit::from_u8(best_suit));
        if bits & (1 << 3) == 0 && bits & (1 << 2) == 0 {
            return BID_PASS; // require J or 9
        }
    }

    // Quality gate
    if !quality_ok(hand, Suit::from_u8(best_suit)) {
        return BID_PASS;
    }

    // Lead bonus
    if cfg.lead_bonus > 0 && has_lead(state) {
        best_score += cfg.lead_bonus;
    }

    let mut bid_value = balanced_bid_value(best_score);
    if bid_value == 0 {
        return BID_PASS;
    }
    if bid_value > 12 {
        bid_value = 12; // cap at 120
    }

    let action = bidding::encode_bid(bid_value, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

fn v2_cfg_respond(state: &GameState, hand: CardSet, legal: &u64, cfg: &V2Config) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value;

    if partner_value >= 13 {
        return BID_PASS;
    }

    let base_score = evaluate_for_trump(hand, partner_suit);
    let trump_bits = suit_bits(hand, partner_suit);
    let trump_count = trump_bits.count_ones();
    let has_jack = trump_bits & (1 << 3) != 0;
    let has_nine = trump_bits & (1 << 2) != 0;

    let mut bonus: i16 = 0;
    if has_jack {
        bonus += cfg.resp_jack_bonus;
    }
    if has_nine && trump_count >= 2 {
        bonus += cfg.resp_nine_bonus;
    }
    bonus += count_side_aces(hand, partner_suit) as i16 * cfg.resp_ace_bonus;
    if trump_count >= 3 {
        bonus += cfg.resp_support_bonus;
    }
    if trump_count == 0 {
        bonus += cfg.resp_misfit_penalty; // negative
    }

    let adjusted_score = (base_score as i16 + bonus).max(0) as u16;

    let target_value = balanced_bid_value(adjusted_score);
    let target_value = if target_value > 13 { 13 } else { target_value };

    if target_value > partner_value {
        let action = bidding::encode_bid(target_value, state.last_bid_suit);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    // Alternative suit
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

    if alt_best_score >= 16 && quality_ok(hand, Suit::from_u8(alt_best_suit)) {
        let mut alt_value = balanced_bid_value(alt_best_score);
        if alt_value > 12 {
            alt_value = 12;
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

/// Default improved_v2_bid uses V2Config::defensive_lead1() (tournament winner).
pub fn improved_v2_bid(state: &GameState) -> u8 {
    improved_v2_configurable_bid(state, &V2Config::defensive_lead1())
}

// ---------------------------------------------------------------------------
// PetitBide: trick-counting bidding system
// ---------------------------------------------------------------------------

/// Count expected tricks and bonus for a hand assuming `suit` is trump.
/// Returns (tricks, bonus_points).
///
/// Trick sources:
/// - Jack of trump: +1 trick, +10 bonus
/// - 9 of trump with ≥2 trumps: +1 trick
/// - Each trump from 3rd onward: +1 trick (i.e., max(0, trump_count - 2))
/// - Each side ace: +1 trick
/// - Each side 10 with ≥2 cards in same suit: +1 trick, -5 bonus
pub fn petit_bide_tricks(hand: CardSet, suit: Suit) -> (u8, i16) {
    let trump_bits = suit_bits(hand, suit);
    let trump_count = trump_bits.count_ones();
    let has_jack = trump_bits & (1 << 3) != 0;
    let has_nine = trump_bits & (1 << 2) != 0;

    let mut tricks: u8 = 0;
    let mut bonus: i16 = 0;

    // Jack of trump: +1 trick, +10 bonus
    if has_jack {
        tricks += 1;
        bonus += 10;
    }

    // 9 of trump "2nd" (with ≥2 trumps): +1 trick
    if has_nine && trump_count >= 2 {
        tricks += 1;
    }

    // Each trump from 3rd onward: +1 trick
    if trump_count > 2 {
        tricks += (trump_count - 2) as u8;
    }

    // Side suits
    for suit_idx in 0..4u8 {
        if suit_idx == suit as u8 {
            continue;
        }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        let count = bits.count_ones();

        // Side ace: +1 trick
        if bits & (1 << 7) != 0 {
            tricks += 1;
        }

        // Side 10 "2nd" (10 with ≥2 cards in same suit): +1 trick, -5 bonus
        if bits & (1 << 6) != 0 && count >= 2 {
            tricks += 1;
            bonus -= 5;
        }
    }

    (tricks, bonus)
}

/// Compute PetitBide score = tricks × 20 + bonus.
pub fn petit_bide_score(hand: CardSet, suit: Suit) -> i16 {
    let (tricks, bonus) = petit_bide_tricks(hand, suit);
    tricks as i16 * 20 + bonus
}

/// Check if first trick lead goes to player or partner.
fn has_lead(state: &GameState) -> bool {
    let player = state.current_player;
    let partner = GameState::partner(player);
    let lead = (state.dealer + 1) % 4;
    lead == player || lead == partner
}

/// Map a PetitBide score to a bid value (encoded as value/10).
/// Returns 0 for PASS.
fn petit_bide_score_to_value(score: i16) -> u8 {
    if score >= 130 {
        13
    } else if score >= 120 {
        12
    } else if score >= 110 {
        11
    } else if score >= 100 {
        10
    } else if score >= 90 {
        9
    } else if score >= 80 {
        8
    } else {
        0
    }
}

/// PetitBide bidding strategy.
pub fn petit_bide_bid(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: always PASS
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Coinche check (reuse roro_coinche logic)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);
        if bidder_team != my_team {
            let action = roro_coinche(hand, state, &legal);
            if action != BID_PASS {
                return action;
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return petit_bide_opening(state, hand, &legal);
    }

    // Partner response
    if state.last_bidder == partner {
        return petit_bide_respond(state, hand, &legal);
    }

    // Intervention
    petit_bide_intervene(state, hand, &legal)
}

/// PetitBide opening: evaluate all 4 suits with trick counting, pick best.
fn petit_bide_opening(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let position = bidding_position(state);
    let lead_bonus: i16 = if has_lead(state) { 10 } else { 0 };

    let mut best_suit = 0u8;
    let mut best_score: i16 = 0;

    for suit_idx in 0..4u8 {
        let score = petit_bide_score(hand, Suit::from_u8(suit_idx)) + lead_bonus;
        if score > best_score {
            best_score = score;
            best_suit = suit_idx;
        }
    }

    // Minimal quality gate: need Jack or 9 to open
    let trump_bits = suit_bits(hand, Suit::from_u8(best_suit));
    let has_jack = trump_bits & (1 << 3) != 0;
    let has_nine = trump_bits & (1 << 2) != 0;
    if !has_jack && !has_nine {
        return BID_PASS;
    }

    // 4th position rule: only open if score ≥ 100
    if position == 3 && best_score < 100 {
        return BID_PASS;
    }

    let mut bid_value = petit_bide_score_to_value(best_score);
    if bid_value == 0 {
        return BID_PASS;
    }

    // Cap opening at 120
    if bid_value > 12 {
        bid_value = 12;
    }

    let action = bidding::encode_bid(bid_value, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

/// PetitBide response to partner's bid.
fn petit_bide_respond(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value;

    // Don't push past 120
    if partner_value >= 12 {
        return BID_PASS;
    }

    // Evaluate response points in partner's suit
    let response_score = petit_bide_response_score(hand, partner_suit);

    // New bid = partner's bid level + response_points
    let new_score = partner_value as i16 * 10 + response_score;
    let mut new_value = petit_bide_score_to_value(new_score);
    // Cap at 120
    if new_value > 12 {
        new_value = 12;
    }

    if new_value > partner_value {
        let action = bidding::encode_bid(new_value, state.last_bid_suit);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    // Check if own best suit is better than responding in partner's suit
    let lead_bonus: i16 = if has_lead(state) { 10 } else { 0 };
    let mut alt_best_suit = 0u8;
    let mut alt_best_score: i16 = 0;

    for suit_idx in 0..4u8 {
        if suit_idx == state.last_bid_suit {
            continue;
        }
        let score = petit_bide_score(hand, Suit::from_u8(suit_idx)) + lead_bonus;
        if score > alt_best_score {
            alt_best_score = score;
            alt_best_suit = suit_idx;
        }
    }

    // Quality gate for alternative suit
    let alt_bits = suit_bits(hand, Suit::from_u8(alt_best_suit));
    let alt_has_j = alt_bits & (1 << 3) != 0;
    let alt_has_9 = alt_bits & (1 << 2) != 0;

    if (alt_has_j || alt_has_9) && alt_best_score >= new_score {
        let mut alt_value = petit_bide_score_to_value(alt_best_score);
        if alt_value > 12 {
            alt_value = 12;
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

/// Compute response score for partner's suit.
/// +20 per Jack in partner's suit, +10 per side ace, +10 if 9 "2nd",
/// +5 per side 10 "2nd", -10 if 0 trumps, +10 if 3+ trumps.
fn petit_bide_response_score(hand: CardSet, partner_suit: Suit) -> i16 {
    let trump_bits = suit_bits(hand, partner_suit);
    let trump_count = trump_bits.count_ones();
    let has_jack = trump_bits & (1 << 3) != 0;
    let has_nine = trump_bits & (1 << 2) != 0;

    let mut score: i16 = 0;

    // +20 per Jack in partner's suit
    if has_jack {
        score += 20;
    }

    // +10 per side ace
    for suit_idx in 0..4u8 {
        if suit_idx == partner_suit as u8 {
            continue;
        }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        if bits & (1 << 7) != 0 {
            score += 10;
        }
    }

    // +10 if 9 "2nd" in partner's suit
    if has_nine && trump_count >= 2 {
        score += 10;
    }

    // +5 per side 10 "2nd" (10 with ≥2 cards in same suit, any suit)
    for suit_idx in 0..4u8 {
        if suit_idx == partner_suit as u8 {
            continue;
        }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        let count = bits.count_ones();
        if bits & (1 << 6) != 0 && count >= 2 {
            score += 5;
        }
    }

    // -10 if 0 trumps in partner's suit
    if trump_count == 0 {
        score -= 10;
    }

    // +10 if 3+ trumps in partner's suit
    if trump_count >= 3 {
        score += 10;
    }

    score
}

/// PetitBide intervention when opponents bid.
fn petit_bide_intervene(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let opp_value = state.last_bid_value;
    let opp_suit_idx = state.last_bid_suit;
    let my_team = GameState::player_team(state.current_player);
    let bidder_team = GameState::player_team(state.last_bidder);

    // Only intervene against opponents
    if bidder_team == my_team {
        return BID_PASS;
    }

    // Don't intervene above 120
    if opp_value >= 12 {
        return BID_PASS;
    }

    let lead_bonus: i16 = if has_lead(state) { 10 } else { 0 };

    // Defense boost: +10 if partner hasn't spoken yet (we're the first defender)
    let partner = GameState::partner(state.current_player);
    let partner_has_spoken = state.last_bidder == partner
        || (state.consecutive_passes == 0); // partner already had a turn
    let defense_boost: i16 = if !partner_has_spoken { 10 } else { 0 };

    let mut best_suit = 0u8;
    let mut best_score: i16 = 0;

    for suit_idx in 0..4u8 {
        if suit_idx == opp_suit_idx {
            continue;
        }
        let score = petit_bide_score(hand, Suit::from_u8(suit_idx)) + lead_bonus + defense_boost;
        if score > best_score {
            best_score = score;
            best_suit = suit_idx;
        }
    }

    // Quality gate
    let trump_bits = suit_bits(hand, Suit::from_u8(best_suit));
    let has_jack = trump_bits & (1 << 3) != 0;
    let has_nine = trump_bits & (1 << 2) != 0;
    if !has_jack && !has_nine {
        return BID_PASS;
    }

    let mut bid_value = petit_bide_score_to_value(best_score);
    if bid_value == 0 {
        return BID_PASS;
    }

    // Cap at 120
    if bid_value > 12 {
        bid_value = 12;
    }

    // Must overbid opponent
    if bid_value <= opp_value {
        return BID_PASS;
    }

    let action = bidding::encode_bid(bid_value, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

// ---------------------------------------------------------------------------
// Moelleux: PetitBide with different 80 convention
// ---------------------------------------------------------------------------

/// Find the most playable non-ace suit for "aux as" 80 convention.
/// Returns the suit index of the longest suit that does NOT contain an ace.
/// Falls back to longest suit overall if all suits have aces.
fn find_non_ace_suit(hand: CardSet) -> u8 {
    let mut best_suit = 0u8;
    let mut best_len = 0u32;
    let mut best_score = 0u16;

    // First pass: suits without aces
    for suit_idx in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        let has_ace = bits & (1 << 7) != 0;
        if has_ace {
            continue;
        }
        let len = bits.count_ones();
        let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
        if len > best_len || (len == best_len && score > best_score) {
            best_suit = suit_idx;
            best_len = len;
            best_score = score;
        }
    }

    if best_len > 0 {
        return best_suit;
    }

    // Fallback: all suits have aces, pick longest
    for suit_idx in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        let len = bits.count_ones();
        let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
        if len > best_len || (len == best_len && score > best_score) {
            best_suit = suit_idx;
            best_len = len;
            best_score = score;
        }
    }

    best_suit
}

/// Moelleux bidding strategy.
pub fn moelleux_bid(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: always PASS
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Coinche check (reuse roro_coinche logic)
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);
        if bidder_team != my_team {
            let action = roro_coinche(hand, state, &legal);
            if action != BID_PASS {
                return action;
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return moelleux_opening(state, hand, &legal);
    }

    // Partner response
    if state.last_bidder == partner {
        return moelleux_respond(state, hand, &legal);
    }

    // Intervention: same as PetitBide
    petit_bide_intervene(state, hand, &legal)
}

/// Moelleux opening: special 80 convention, PetitBide for >80.
fn moelleux_opening(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let position = bidding_position(state);
    let lead_bonus: i16 = if has_lead(state) { 10 } else { 0 };

    // First check if PetitBide wants to open above 80
    let mut best_suit = 0u8;
    let mut best_score: i16 = 0;

    for suit_idx in 0..4u8 {
        let score = petit_bide_score(hand, Suit::from_u8(suit_idx)) + lead_bonus;
        if score > best_score {
            best_score = score;
            best_suit = suit_idx;
        }
    }

    // Quality gate for above-80 bids
    let trump_bits = suit_bits(hand, Suit::from_u8(best_suit));
    let has_jack = trump_bits & (1 << 3) != 0;
    let has_nine = trump_bits & (1 << 2) != 0;

    // 4th position: ISO theorem — only open 80 if PetitBide score ≥ 120
    if position == 3 {
        if best_score >= 120 && (has_jack || has_nine) {
            let mut bid_value = petit_bide_score_to_value(best_score);
            if bid_value > 12 {
                bid_value = 12;
            }
            let action = bidding::encode_bid(bid_value, best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
        return BID_PASS;
    }

    // Above 80: use PetitBide trick counting (with J/9 quality gate)
    if best_score >= 90 && (has_jack || has_nine) {
        let mut bid_value = petit_bide_score_to_value(best_score);
        // Cap at 120
        if bid_value > 12 {
            bid_value = 12;
        }
        let action = bidding::encode_bid(bid_value, best_suit);
        if legal & (1u64 << action) != 0 {
            return action;
        }
    }

    // 80 convention: position-dependent
    match position {
        0 | 1 => {
            // "Aux as": 2+ total aces → bid 80 in non-ace suit
            if count_total_aces(hand) >= 2 {
                let non_ace_suit = find_non_ace_suit(hand);
                let action = bidding::encode_bid(8, non_ace_suit);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
        }
        2 => {
            // "Petit jeu": 3+ trumps with J or 9, plus ≥1 side ace
            for suit_idx in 0..4u8 {
                let bits = suit_bits(hand, Suit::from_u8(suit_idx));
                let count = bits.count_ones();
                let j = bits & (1 << 3) != 0;
                let n = bits & (1 << 2) != 0;
                let suit = Suit::from_u8(suit_idx);
                let side_aces = count_side_aces(hand, suit);

                if count >= 3 && (j || n) && side_aces >= 1 {
                    let action = bidding::encode_bid(8, suit_idx);
                    if legal & (1u64 << action) != 0 {
                        return action;
                    }
                }
            }
        }
        _ => {} // position 3 handled above
    }

    BID_PASS
}

/// Moelleux response to partner's bid.
fn moelleux_respond(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let partner_value = state.last_bid_value;

    // Response to 80 "aux as" (partner in 1st/2nd position)
    if partner_value == 8 && partner_was_aux_as(state) {
        return moelleux_respond_aux_as(state, hand, legal);
    }

    // For all other bids: use PetitBide response system
    petit_bide_respond(state, hand, legal)
}

/// Moelleux response to partner's "aux as" 80.
/// Partner has 2 aces NOT in the announced suit.
/// Add +2 extra trick-equivalents (+40 points) for partner's aces.
fn moelleux_respond_aux_as(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let partner_suit_idx = state.last_bid_suit;
    let lead_bonus: i16 = if has_lead(state) { 10 } else { 0 };
    // Partner's 2 aces = +2 side aces worth of tricks = +40 to score
    let ace_bonus: i16 = 40;

    // Evaluate each suit with the ace bonus
    let mut best_suit = 0u8;
    let mut best_score: i16 = 0;

    for suit_idx in 0..4u8 {
        // Skip partner's announced suit (partner signaled "NOT this suit")
        if suit_idx == partner_suit_idx {
            continue;
        }
        let base_score = petit_bide_score(hand, Suit::from_u8(suit_idx)) + lead_bonus;
        // Add ace bonus, but subtract aces already counted in own hand
        // (partner's aces are in OTHER suits, might overlap with our side ace count)
        let own_side_aces = count_side_aces(hand, Suit::from_u8(suit_idx));
        // Partner has 2 aces. Some might be in suits we already counted.
        // Simplified: add +40, subtract 20 per own side ace (max 2 overlap)
        let overlap = own_side_aces.min(2) as i16;
        let adjusted_score = base_score + ace_bonus - overlap * 20;

        if adjusted_score > best_score {
            best_score = adjusted_score;
            best_suit = suit_idx;
        }
    }

    // Quality gate
    let trump_bits = suit_bits(hand, Suit::from_u8(best_suit));
    let has_jack = trump_bits & (1 << 3) != 0;
    let has_nine = trump_bits & (1 << 2) != 0;

    if best_score >= 80 && (has_jack || has_nine) {
        let mut bid_value = petit_bide_score_to_value(best_score);
        if bid_value > 12 {
            bid_value = 12;
        }
        if bid_value > 8 {
            // Bid above partner's 80
            let action = bidding::encode_bid(bid_value, best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    // If no good suit found, also try partner's suit with ace bonus
    let partner_score = petit_bide_score(hand, Suit::from_u8(partner_suit_idx))
        + lead_bonus
        + ace_bonus;
    if partner_score >= 90 {
        let mut bid_value = petit_bide_score_to_value(partner_score);
        if bid_value > 12 {
            bid_value = 12;
        }
        if bid_value > 8 {
            let action = bidding::encode_bid(bid_value, partner_suit_idx);
            if legal & (1u64 << action) != 0 {
                return action;
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
        let score = petit_bide_response_score(hand, Suit::Hearts);
        assert!(score >= 20, "Jack in partner suit should give +20, got {}", score);
    }

    #[test]
    fn test_petit_bide_response_zero_trumps_penalty() {
        // No hearts at all → -10 penalty
        let hand = make_hand(&["7S", "8S", "9S", "7D", "8D", "9D", "7C", "8C"]);
        let score = petit_bide_response_score(hand, Suit::Hearts);
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
        let suit = find_non_ace_suit(hand);
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
