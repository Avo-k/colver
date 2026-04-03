use crate::bidding::{self, BID_PASS};
use crate::card::*;
use crate::state::*;
use super::eval_helpers::has_lead;
use super::roro::{roro_coinche, bidding_position};

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

/// Map a PetitBide score to a bid value (encoded as value/10).
/// Returns 0 for PASS.
pub(super) fn petit_bide_score_to_value(score: i16) -> u8 {
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
pub(super) fn petit_bide_respond(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
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
pub(super) fn petit_bide_response_score(hand: CardSet, partner_suit: Suit) -> i16 {
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
pub(super) fn petit_bide_intervene(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
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
