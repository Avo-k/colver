use crate::bidding::{self, BID_PASS};
use crate::card::*;
use crate::state::*;
use super::eval_helpers::{evaluate_for_trump, count_side_aces, has_lead};
use super::roro::{roro_coinche, count_total_aces, bidding_position};
use super::petit_bide::{petit_bide_score, petit_bide_respond, petit_bide_intervene, petit_bide_score_to_value};

// ---------------------------------------------------------------------------
// Moelleux: PetitBide with different 80 convention
// ---------------------------------------------------------------------------

/// Find the most playable non-ace suit for "aux as" 80 convention.
/// Returns the suit index of the longest suit that does NOT contain an ace.
/// Falls back to longest suit overall if all suits have aces.
pub(super) fn find_non_ace_suit(hand: CardSet) -> u8 {
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

/// Detect if partner opened "aux as" (1st/2nd position).
fn partner_was_aux_as(state: &GameState) -> bool {
    let first_bidder = (state.dealer + 1) % 4;
    let partner = state.last_bidder;
    let partner_pos = (partner + 4 - first_bidder) % 4;
    partner_pos <= 1
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
