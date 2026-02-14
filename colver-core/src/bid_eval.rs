use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::state::*;

/// Which bidding function to use.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum BidFunction {
    Heuristic,
    Smart,
    Improved,
}

impl BidFunction {
    /// Dispatch to the appropriate bid function.
    pub fn bid(self, state: &GameState) -> u8 {
        match self {
            BidFunction::Heuristic => heuristic_bid(state),
            BidFunction::Smart => smart_bid(state),
            BidFunction::Improved => improved_bid(state),
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
            thresholds: [10, 13, 17, 21, 25, u16::MAX],
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
            // 6: 110-threshold -1
            BidParams { name: "lo110", thresholds: [10, 13, 17, 20, 25, u16::MAX], ..b },
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
    } else if score < 21 {
        10 // 100
    } else if score < 25 {
        11 // 110
    } else {
        12 // 120
    }
}

/// Tournament-tuned balanced bidder. Quality gate + score→value mapping (10→80, 13→90,
/// 17→100, 21→110, 25→120). Caps: opening 120, overcall 120, response 130.
/// Won round-robin tournament with 62% overall win rate vs 5 other strategies.
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
