use crate::bidding::{self, BID_PASS};
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
pub(crate) fn score_to_bid_value(score: u16) -> u8 {
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
pub(crate) struct SuitEval {
    pub has_jack: bool,
    pub has_nine: bool,
    pub trump_count: u32,
    pub score: u16,
}

pub(crate) fn evaluate_suit(hand: CardSet, suit: Suit) -> SuitEval {
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
pub(crate) fn count_side_aces(hand: CardSet, trump: Suit) -> u32 {
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

/// Quality gate: suit must have at least one of J, 9, A, 10, or 3+ cards.
pub fn quality_ok(hand: CardSet, suit: Suit) -> bool {
    let bits = suit_bits(hand, suit);
    let has_j = bits & (1 << 3) != 0;
    let has_9 = bits & (1 << 2) != 0;
    let has_a = bits & (1 << 7) != 0;
    let has_10 = bits & (1 << 6) != 0;
    let count = bits.count_ones();
    has_j || has_9 || has_a || has_10 || count >= 3
}

/// Check if first trick lead goes to player or partner.
pub(crate) fn has_lead(state: &GameState) -> bool {
    let player = state.current_player;
    let partner = GameState::partner(player);
    let lead = (state.dealer + 1) % 4;
    lead == player || lead == partner
}
