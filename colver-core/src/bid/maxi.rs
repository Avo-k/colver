//! Maxi bot: convention-linked bidding + play strategy.
//!
//! Based on a human expert's system that couples bidding conventions with
//! card play decisions. Bidding has 4 phases: opening classification,
//! partner response, suit change, and competitive. Play decisions use
//! knowledge from the bidding (who is strong, contract size) to choose leads.

use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::state::*;

// ---- Suit evaluation ----

struct MaxiSuitEval {
    has_jack: bool,
    has_nine: bool,
    has_ace: bool,
    has_ten: bool,
    has_king: bool,
    has_queen: bool,
    trump_count: u32,
    score: u16,
    has_belote: bool,
}

fn maxi_eval_suit(hand: CardSet, suit: Suit) -> MaxiSuitEval {
    let bits = suit_bits(hand, suit);
    MaxiSuitEval {
        has_jack: bits & (1 << 3) != 0,
        has_nine: bits & (1 << 2) != 0,
        has_ace: bits & (1 << 7) != 0,
        has_ten: bits & (1 << 6) != 0,
        has_king: bits & (1 << 5) != 0,
        has_queen: bits & (1 << 4) != 0,
        trump_count: bits.count_ones(),
        score: crate::bid_eval::evaluate_for_trump(hand, suit),
        has_belote: (bits & (1 << 5) != 0) && (bits & (1 << 4) != 0),
    }
}

/// Count side aces (aces in non-trump suits).
fn count_side_aces(hand: CardSet, trump: Suit) -> u32 {
    let mut count = 0u32;
    for suit_idx in 0..4u8 {
        if suit_idx == trump as u8 {
            continue;
        }
        if suit_bits(hand, Suit::from_u8(suit_idx)) & (1 << 7) != 0 {
            count += 1;
        }
    }
    count
}

/// Count total aces across all 4 suits.
fn count_total_aces(hand: CardSet) -> u32 {
    let mut count = 0u32;
    for suit_idx in 0..4u8 {
        if suit_bits(hand, Suit::from_u8(suit_idx)) & (1 << 7) != 0 {
            count += 1;
        }
    }
    count
}

/// Count losers for a given trump suit.
fn count_losers(hand: CardSet, trump: Suit, eval: &MaxiSuitEval) -> u32 {
    let mut losers = 0u32;
    // Trump losers: each missing J/9 = 1 loser
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
            continue; // void = can trump
        }
        let has_a = bits & (1 << 7) != 0;
        if count == 1 && has_a {
            continue; // singleton ace = no loser
        }
        if count == 1 {
            losers += 1;
        } else {
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

/// Bidding position: 0 = first to speak, 1 = second, etc.
fn bidding_position(state: &GameState) -> u8 {
    state.consecutive_passes
}

/// Check if our team has the first trick lead.
#[allow(dead_code)]
fn has_lead(state: &GameState) -> bool {
    let player = state.current_player;
    let partner = GameState::partner(player);
    let lead = (state.dealer + 1) % 4;
    lead == player || lead == partner
}

// ============================================================
// BIDDING
// ============================================================

/// Main Maxi bidding entry point.
pub fn maxi_bid(state: &GameState) -> u8 {
    debug_assert_eq!(state.phase, Phase::Bidding);

    let player = state.current_player;
    let hand = state.hands[player as usize];
    let legal = state.legal_actions();
    let partner = GameState::partner(player);

    // After coinche: always pass
    if state.coinche_state > 0 {
        return BID_PASS;
    }

    // Coinche check
    if state.last_bid_value > 0 && state.coinche_state == 0 {
        let bidder_team = GameState::player_team(state.last_bidder);
        let my_team = GameState::player_team(player);
        if bidder_team != my_team {
            let action = maxi_coinche(hand, state, &legal);
            if action != BID_PASS {
                return action;
            }
        }
    }

    // Opening: no bid yet
    if state.last_bid_value == 0 {
        return maxi_opening(state, hand, &legal);
    }

    // Partner made last bid: respond
    if state.last_bidder == partner {
        return maxi_respond(state, hand, &legal);
    }

    // Opponent made last bid: compete
    maxi_compete(state, hand, &legal)
}

// ---- Coinche ----

fn maxi_coinche(hand: CardSet, state: &GameState, legal: &u64) -> u8 {
    if legal & (1u64 << BID_COINCHE) == 0 {
        return BID_PASS;
    }
    let their_suit = Suit::from_u8(state.last_bid_suit);
    let eval = maxi_eval_suit(hand, their_suit);

    // J+9 in their suit
    if eval.has_jack && eval.has_nine {
        return BID_COINCHE;
    }
    // 4+ trumps + side ace
    if eval.trump_count >= 4 && count_side_aces(hand, their_suit) >= 1 {
        return BID_COINCHE;
    }
    // Théorème 3: 0 trumps in their suit + 3 aces
    if eval.trump_count == 0 && count_total_aces(hand) >= 3 {
        return BID_COINCHE;
    }
    BID_PASS
}

// ---- Opening (Phase 1) ----

/// Classify suit into Cases A(80), B(90), C(100), D(110+).
/// Returns bid value encoded (8=80, 9=90, etc.) or 0 for not openable.
fn classify_suit(hand: CardSet, suit: Suit, eval: &MaxiSuitEval, position: u8) -> u8 {
    let side_aces = count_side_aces(hand, suit);

    // Case D: 110+ — requires J+9
    if eval.has_jack && eval.has_nine {
        let losers = count_losers(hand, suit, eval);
        if losers <= 1 {
            return 13; // 130
        }
        if losers <= 2 {
            return 12; // 120
        }
        if losers <= 3 {
            // 110 — only if have side strength
            if side_aces >= 1 || eval.trump_count >= 4 {
                return 11; // 110
            }
        }
    }

    // Case C: 100 — requires 4+ cards
    if eval.trump_count >= 4 {
        // J or 9 + (A or 10) + 2 fillers
        if (eval.has_jack || eval.has_nine) && (eval.has_ace || eval.has_ten) {
            return 10; // 100
        }
        // J/9 + A + 10 + filler + ext ace
        if (eval.has_jack || eval.has_nine) && eval.has_ace && eval.has_ten && side_aces >= 1 {
            return 10;
        }
        // 5-6 card suit without J/9 + ext ace + irregular
        if eval.trump_count >= 5 && side_aces >= 1 {
            return 10;
        }
    }

    // Case B: 90 — J+9 guaranteed (already checked D above, so losers > 3)
    if eval.has_jack && eval.has_nine {
        // J+9 + small cards or +A
        if eval.trump_count >= 3 || side_aces >= 1 {
            return 9; // 90
        }
        // J+9 doubleton: only open 90 with ext ace
        if side_aces >= 1 {
            return 9;
        }
    }

    // Case A: 80 — weak hold
    if eval.trump_count >= 3 {
        // J + (A or 10) + filler
        if eval.has_jack && (eval.has_ace || eval.has_ten) {
            if side_aces >= 1 || position >= 2 {
                return 8; // 80
            }
            // No ext ace in early position -> consider passing
            return 0;
        }
        // 9 + A + filler
        if eval.has_nine && eval.has_ace {
            if side_aces >= 1 || position >= 2 {
                return 8;
            }
            return 0;
        }
        // A + 10 + (K/Q) + filler (4+ cards)
        if eval.has_ace && eval.has_ten && (eval.has_king || eval.has_queen) && eval.trump_count >= 4 {
            if side_aces >= 2 || position >= 3 {
                return 8;
            }
            return 0;
        }
    }

    // Rare case: J + K + Q (belote)
    if eval.has_jack && eval.has_belote && eval.trump_count >= 3 {
        return 8;
    }

    // 5-card suit A/10 + K + Q + 8/7
    if eval.trump_count >= 5 && eval.has_ace && eval.has_ten && eval.has_king && eval.has_queen {
        return 8;
    }

    // J+9 only + ext ace (last seat)
    if eval.has_jack && eval.has_nine && eval.trump_count == 2 && side_aces >= 1 && position >= 3 {
        return 8;
    }

    0 // not openable
}

fn maxi_opening(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let position = bidding_position(state);

    let evals: [MaxiSuitEval; 4] = [
        maxi_eval_suit(hand, Suit::Spades),
        maxi_eval_suit(hand, Suit::Hearts),
        maxi_eval_suit(hand, Suit::Diamonds),
        maxi_eval_suit(hand, Suit::Clubs),
    ];

    // Classify each suit and find the best opening
    let mut best_suit = 0u8;
    let mut best_value = 0u8;
    let mut best_score = 0u16;

    for suit_idx in 0..4u8 {
        let eval = &evals[suit_idx as usize];
        if eval.trump_count < 2 {
            continue; // skip very short suits
        }
        let value = classify_suit(hand, Suit::from_u8(suit_idx), eval, position);
        if value == 0 {
            continue;
        }
        // Higher classification wins; tie-break by evaluate_for_trump score
        if value > best_value || (value == best_value && eval.score > best_score) {
            best_value = value;
            best_suit = suit_idx;
            best_score = eval.score;
        }
    }

    // Belote adjustment: +20 virtual (bump one level)
    if best_value > 0 && evals[best_suit as usize].has_belote {
        // Never open 80 with 9+belote only (no J)
        let eval = &evals[best_suit as usize];
        let is_nine_belote_only = !eval.has_jack && eval.has_nine && eval.has_belote;
        if !is_nine_belote_only && best_value < 13 {
            // Bump one level (80->90, 90->100, etc.) but cap at 130
            best_value += 1;
        }
    }

    if best_value == 0 {
        return BID_PASS;
    }

    let action = bidding::encode_bid(best_value, best_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

// ---- Response (Phase 2) ----

fn maxi_respond(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let partner_suit = Suit::from_u8(state.last_bid_suit);
    let partner_value = state.last_bid_value; // encoded: 8=80, 9=90, etc.
    let my_eval = maxi_eval_suit(hand, partner_suit);
    let side_aces = count_side_aces(hand, partner_suit);

    let raise_score = if partner_value <= 8 {
        // Partner opens 80
        maxi_response_80(hand, partner_suit, &my_eval, side_aces)
    } else if partner_value == 9 {
        // Partner opens 90
        maxi_response_90(hand, partner_suit, &my_eval, side_aces)
    } else {
        // Partner opens 100+
        maxi_response_100plus(hand, partner_suit, &my_eval, side_aces)
    };

    if raise_score <= 0 {
        // Check suit change: my best suit might be better
        return maxi_suit_change(state, hand, legal, partner_suit, partner_value);
    }

    // Target value = partner_value + raise_score/10
    let raise_levels = (raise_score / 10) as u8;
    let target_value = (partner_value + raise_levels).min(13); // cap at 130

    // Must overbid current contract
    if target_value <= state.last_bid_value {
        return BID_PASS;
    }

    let action = bidding::encode_bid(target_value, state.last_bid_suit);
    if legal & (1u64 << action) != 0 {
        action
    } else {
        BID_PASS
    }
}

/// Response scoring when partner opens 80.
fn maxi_response_80(
    _hand: CardSet,
    _trump: Suit,
    eval: &MaxiSuitEval,
    side_aces: u32,
) -> i16 {
    let mut score: i16 = 0;

    // +20 for J of trump
    if eval.has_jack {
        score += 20;
    }
    // +10 for 9 + another trump
    if eval.has_nine && eval.trump_count >= 2 {
        score += 10;
    }
    // +10 per ext ace (only if 2+ trumps)
    if eval.trump_count >= 2 {
        score += (side_aces as i16) * 10;
    }
    // +10 for 3+ trumps
    if eval.trump_count >= 3 {
        score += 10;
    }

    // Special: 4-5 small trumps + ext ace
    if eval.trump_count >= 4
        && !eval.has_jack
        && !eval.has_nine
        && side_aces >= 1
    {
        score = score.max(20); // at least +20
        if eval.has_belote {
            score = score.max(30);
        }
    }

    // Subtract
    if eval.trump_count == 1 {
        score -= 10; // singleton trump
    }
    if eval.trump_count == 0 {
        score -= 10; // void in trump
    }

    score
}

/// Response scoring when partner opens 90 (assume partner has J+9).
fn maxi_response_90(
    _hand: CardSet,
    _trump: Suit,
    eval: &MaxiSuitEval,
    side_aces: u32,
) -> i16 {
    // Must hold ≥1 trump
    if eval.trump_count == 0 {
        return -10;
    }

    let mut score: i16 = 0;

    // Same scoring as 80 response but partner already has J+9
    if eval.has_jack {
        score += 20;
    }
    if eval.has_nine && eval.trump_count >= 2 {
        score += 10;
    }
    if eval.trump_count >= 2 {
        score += (side_aces as i16) * 10;
    }
    if eval.trump_count >= 3 {
        score += 10;
    }

    if eval.trump_count == 1 {
        score -= 10;
    }

    // Don't give fake distribution bonuses unless belote
    if eval.trump_count >= 4 && !eval.has_jack && !eval.has_nine && !eval.has_belote {
        score = score.min(10);
    }

    score
}

/// Response scoring when partner opens 100+ — announce keys.
fn maxi_response_100plus(
    hand: CardSet,
    trump: Suit,
    eval: &MaxiSuitEval,
    side_aces: u32,
) -> i16 {
    let mut keys: i16 = 0;

    // Only announce full keys if holding J or 9 of trump
    let has_trump_honor = eval.has_jack || eval.has_nine;

    if has_trump_honor {
        // Each ext ace = 1 key
        keys += side_aces as i16;
        // A+10 in same non-trump suit = 1 extra key
        for suit_idx in 0..4u8 {
            if suit_idx == trump as u8 {
                continue;
            }
            let bits = suit_bits(hand, Suit::from_u8(suit_idx));
            if (bits & (1 << 7) != 0) && (bits & (1 << 6) != 0) {
                keys += 1; // A+10 in same suit
            }
        }
    } else {
        // Only announce ext aces (conservative)
        keys += side_aces as i16;
    }

    // After 120/130: conservative, don't overfeed
    // Just return raw key count * 10
    keys * 10
}

// ---- Suit change (Phase 3) ----

fn maxi_suit_change(
    state: &GameState,
    hand: CardSet,
    legal: &u64,
    partner_suit: Suit,
    partner_value: u8,
) -> u8 {
    // Evaluate my best suit
    let mut best_suit = partner_suit as u8;
    let mut best_value = 0u8;
    let mut best_score = 0u16;
    let position = bidding_position(state);

    for suit_idx in 0..4u8 {
        if suit_idx == partner_suit as u8 {
            continue;
        }
        let eval = maxi_eval_suit(hand, Suit::from_u8(suit_idx));
        if eval.trump_count < 2 {
            continue;
        }
        let value = classify_suit(hand, Suit::from_u8(suit_idx), &eval, position);
        if value > best_value || (value == best_value && eval.score > best_score) {
            best_value = value;
            best_suit = suit_idx;
            best_score = eval.score;
        }
    }

    // Change suit if my best is stronger than raising partner
    if best_value > partner_value && best_suit != partner_suit as u8 {
        // Must overbid current contract
        let target = best_value.max(state.last_bid_value + 1);
        if target <= 16 {
            let action = bidding::encode_bid(target, best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }
    }

    BID_PASS
}

// ---- Competitive bidding (Phase 4) ----

fn maxi_compete(state: &GameState, hand: CardSet, legal: &u64) -> u8 {
    let player = state.current_player;
    let partner = GameState::partner(player);
    let position = bidding_position(state);

    // Check if our team has already bid
    let our_team = GameState::player_team(player);
    let their_team = GameState::player_team(state.last_bidder);

    // Opponents bid — try to compete
    if their_team != our_team {
        // Evaluate my hand for best suit
        let mut best_suit = 0u8;
        let mut best_value = 0u8;
        let mut best_score = 0u16;

        for suit_idx in 0..4u8 {
            let eval = maxi_eval_suit(hand, Suit::from_u8(suit_idx));
            if eval.trump_count < 2 {
                continue;
            }
            let value = classify_suit(hand, Suit::from_u8(suit_idx), &eval, position);
            if value > best_value || (value == best_value && eval.score > best_score) {
                best_value = value;
                best_suit = suit_idx;
                best_score = eval.score;
            }
        }

        if best_value == 0 {
            return BID_PASS;
        }

        // Force +10 above opponent's bid (once per partnership)
        let target = state.last_bid_value + 1;
        if target <= best_value + 1 && target <= 13 {
            let action = bidding::encode_bid(target, best_suit);
            if legal & (1u64 << action) != 0 {
                return action;
            }
        }

        // If opponents jumped too high, pass unless belote
        let eval = maxi_eval_suit(hand, Suit::from_u8(best_suit));
        if eval.has_belote {
            // Belote +20 virtual
            let target = (best_value + 2).min(13);
            if target > state.last_bid_value {
                let action = bidding::encode_bid(target, best_suit);
                if legal & (1u64 << action) != 0 {
                    return action;
                }
            }
        }
    }

    let _ = partner;
    BID_PASS
}

// ============================================================
// CARD PLAY
// ============================================================

/// Maxi play action — convention-linked card play.
/// Uses perfect information (all hands visible) like heuristic_play_action.
pub fn maxi_play_action(state: &GameState) -> u8 {
    let legal = crate::play::legal_plays(state) as u64;
    let legal32 = legal as CardSet;
    let count = legal32.count_ones();

    // Forced move
    if count == 1 {
        return legal32.trailing_zeros() as u8;
    }

    let player = state.current_player;
    let trump_suit = state.contract.trump_suit();

    if state.trick_count == 0 {
        // LEADING — use convention-aware lead logic
        return maxi_lead(state, player, trump_suit, legal32);
    }

    // FOLLOWING — delegate to heuristic (legal play rules constrain most decisions)
    crate::rollout::heuristic_play_action(state)
}

/// Convention-aware lead logic.
fn maxi_lead(state: &GameState, player: u8, trump_suit: Suit, legal: CardSet) -> u8 {
    let taker_team = state.contract.team;
    let my_team = GameState::player_team(player);
    let ct = state.contract.contract_type();

    if my_team == taker_team {
        maxi_attack_lead(state, player, trump_suit, ct, legal)
    } else {
        maxi_defense_lead(state, player, trump_suit, ct, legal)
    }
}

/// Attack lead: our team won the contract.
fn maxi_attack_lead(
    state: &GameState,
    player: u8,
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
) -> u8 {
    let partner = player ^ 2;
    let contract_value = state.contract.point_value();
    let my_trumps = cards_in_suit(legal, trump_suit);
    let opp_trumps = count_opponent_trumps(state, player, trump_suit);

    // Determine who opened (who is "strong")
    // The taker (last_bidder or their partner) is strong
    let taker = state.contract.team;
    let partner_is_strong = GameState::player_team(partner) == taker
        && state.contract.team == GameState::player_team(partner);

    // Belote parity: if leading trump with K+Q, lead Q first if odd count, K first if even
    if my_trumps != 0 {
        let my_trump_bits = suit_bits(legal, trump_suit);
        let has_k = my_trump_bits & (1 << 5) != 0;
        let has_q = my_trump_bits & (1 << 4) != 0;
        if has_k && has_q {
            let total_trumps = suit_bits(state.hands[player as usize], trump_suit).count_ones();
            let card = if total_trumps % 2 == 1 {
                // Odd count: lead Q first
                make_card(trump_suit, 4) // Queen
            } else {
                // Even count: lead K first
                make_card(trump_suit, 5) // King
            };
            if legal & (1u32 << card) != 0 {
                return card;
            }
        }
    }

    // If opponents still have trump, draw them out
    if my_trumps != 0 && opp_trumps > 0 {
        // Check if partner opened weak (80/100 → partner may lack J+9)
        let partner_opened_weak = contract_value <= 100;

        if partner_opened_weak && partner_is_strong {
            // Partner is strong but opened weak — be careful
            let my_trump_bits = suit_bits(legal, trump_suit);
            let has_j = my_trump_bits & (1 << 3) != 0;

            if has_j {
                // I hold J: lead J if singleton trump, else lead small trump
                if my_trumps.count_ones() == 1 {
                    return highest_trump_in_set(my_trumps, trump_suit);
                }
                // Lead small trump toward partner's 9
                return lowest_trump_in_set(my_trumps, trump_suit);
            } else {
                // Don't hold J: lead trump only from 7/8/Q
                let safe_trumps = my_trump_bits & 0b00010011; // bits 0(7), 1(8), 4(Q)
                if safe_trumps != 0 {
                    let card = lowest_trump_in_set(
                        (safe_trumps as u32) << (trump_suit as u32 * 8),
                        trump_suit,
                    );
                    if legal & (1u32 << card) != 0 {
                        return card;
                    }
                }
                // Otherwise lead small side suit
                return lead_smallest_side_suit(state, player, trump_suit, ct, legal);
            }
        }

        // Normal case: lead highest safe trump to draw
        let best = highest_trump_in_set(my_trumps, trump_suit);
        if is_safe_lead(state, player, best) {
            return best;
        }
        // If not safe, lead lowest trump
        return lowest_trump_in_set(my_trumps, trump_suit);
    }

    // No trump to lead — branch by contract size
    if contract_value < 120 {
        // Small contract: neutral lead (avoid exposing strength)
        lead_neutral(state, player, trump_suit, ct, legal)
    } else {
        // Big contract: master card lead (help partner discard losers)
        lead_master(state, player, trump_suit, ct, legal)
    }
}

/// Defense lead: opponents won the contract.
fn maxi_defense_lead(
    state: &GameState,
    player: u8,
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
) -> u8 {
    let my_trumps = cards_in_suit(legal, trump_suit);
    let my_trump_count = my_trumps.count_ones();
    let contract_value = state.contract.point_value();

    // Case 1: 1-2 small trumps → lead singleton for ruff
    if my_trump_count <= 2 && my_trump_count >= 1 {
        // Look for singleton non-trump (not A or 10)
        for &suit in &ALL_SUITS {
            if suit == trump_suit {
                continue;
            }
            let in_suit = cards_in_suit(legal, suit);
            if in_suit.count_ones() == 1 {
                let card = in_suit.trailing_zeros() as Card;
                let rank = card_rank(card);
                // Not A(7) or 10(6)
                if rank != 7 && rank != 6 {
                    return card;
                }
            }
        }
    }

    // Case 2: 3+ trumps → lead long suit (prefer with ace on top)
    if my_trump_count >= 3 {
        let mut best_card = EMPTY;
        let mut best_len = 0u32;
        let mut best_has_ace = false;

        for &suit in &ALL_SUITS {
            if suit == trump_suit {
                continue;
            }
            let in_suit = cards_in_suit(legal, suit);
            let len = in_suit.count_ones();
            if len == 0 {
                continue;
            }
            let has_ace = in_suit & (1u32 << (suit as u32 * 8 + 7)) != 0;
            if len > best_len || (len == best_len && has_ace && !best_has_ace) {
                best_len = len;
                best_has_ace = has_ace;
                if has_ace {
                    best_card = (suit as u8) * 8 + 7; // Ace
                } else {
                    best_card = highest_plain_in_suit(in_suit, suit);
                }
            }
        }
        if best_card != EMPTY && legal & (1u32 << best_card) != 0 {
            return best_card;
        }
    }

    // Case 3: rare trump lead — 1 trump + no good side suit + weak declarer (80/100)
    if my_trump_count == 1 && contract_value <= 100 {
        // Check if no good side suit to lead
        let mut has_good_side = false;
        for &suit in &ALL_SUITS {
            if suit == trump_suit {
                continue;
            }
            let in_suit = cards_in_suit(legal, suit);
            if in_suit.count_ones() >= 2 {
                has_good_side = true;
                break;
            }
        }
        if !has_good_side && my_trumps != 0 {
            return lowest_trump_in_set(my_trumps, trump_suit);
        }
    }

    // Fallback: any safe master lead, then shortest suit
    for &suit in &ALL_SUITS {
        if suit == trump_suit {
            continue;
        }
        let in_suit = cards_in_suit(legal, suit);
        if in_suit == 0 {
            continue;
        }
        let high = highest_plain_in_suit(in_suit, suit);
        if high != EMPTY && is_safe_lead(state, player, high) {
            return high;
        }
    }

    // Lead from shortest non-trump suit
    lead_smallest_side_suit(state, player, trump_suit, ct, legal)
}

// ---- Lead helpers ----

/// Lead from the smallest non-trump suit (cheapest card).
fn lead_smallest_side_suit(
    _state: &GameState,
    _player: u8,
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
) -> u8 {
    let mut best_card = EMPTY;
    let mut best_len = u32::MAX;

    for &suit in &ALL_SUITS {
        if suit == trump_suit {
            continue;
        }
        let in_suit = cards_in_suit(legal, suit);
        let len = in_suit.count_ones();
        if len > 0 && len < best_len {
            best_len = len;
            best_card = lowest_point_card(in_suit, ct);
        }
    }
    if best_card != EMPTY {
        return best_card;
    }
    // Only trump left
    lowest_point_card(legal, ct)
}

/// Neutral lead: prefer non-master cards, avoid exposing strong suits.
fn lead_neutral(
    state: &GameState,
    player: u8,
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
) -> u8 {
    let _ = (state, player);
    // Basic finesse: when leading non-trump with A+Q (missing K), play Q first
    for &suit in &ALL_SUITS {
        if suit == trump_suit {
            continue;
        }
        let in_suit = cards_in_suit(legal, suit);
        if in_suit == 0 {
            continue;
        }
        let bits = suit_bits(in_suit, suit);
        let has_a = bits & (1 << 7) != 0;
        let has_q = bits & (1 << 4) != 0;
        let has_k = bits & (1 << 5) != 0;
        // A+Q without K: finesse — lead Q
        if has_a && has_q && !has_k && in_suit.count_ones() >= 2 {
            return make_card(suit, 4); // Queen
        }
    }

    // Lead from longest non-trump suit, cheapest card (avoid exposing)
    let mut best_card = EMPTY;
    let mut best_len = 0u32;

    for &suit in &ALL_SUITS {
        if suit == trump_suit {
            continue;
        }
        let in_suit = cards_in_suit(legal, suit);
        let len = in_suit.count_ones();
        if len > best_len {
            best_len = len;
            best_card = lowest_point_card(in_suit, ct);
        }
    }
    if best_card != EMPTY {
        return best_card;
    }
    lowest_point_card(legal, ct)
}

/// Master lead: lead highest safe card to help partner discard losers.
fn lead_master(
    state: &GameState,
    player: u8,
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
) -> u8 {
    // Try to lead a safe master in any non-trump suit
    for &suit in &ALL_SUITS {
        if suit == trump_suit {
            continue;
        }
        let in_suit = cards_in_suit(legal, suit);
        if in_suit == 0 {
            continue;
        }
        let high = highest_plain_in_suit(in_suit, suit);
        if high != EMPTY && is_safe_lead(state, player, high) {
            return high;
        }
    }
    // Fallback: neutral lead
    lead_neutral(state, player, trump_suit, ct, legal)
}

// ---- Reused helpers from rollout.rs (made pub(crate) there) ----

/// Check if the trick has been trumped.
#[allow(dead_code)]
fn trick_is_trumped(state: &GameState) -> bool {
    let trump_suit = state.contract.trump_suit();
    for i in 0..state.trick_count {
        let seat = (state.trick_lead + i) % 4;
        let card = state.current_trick[seat as usize];
        if card != EMPTY && card_suit(card) == trump_suit {
            return true;
        }
    }
    false
}

/// Check if leading `card` is guaranteed to win the trick.
fn is_safe_lead(state: &GameState, player: u8, card: Card) -> bool {
    let card_s = card_suit(card);
    let trump_suit = state.contract.trump_suit();
    let card_r = card_rank(card);

    for i in 1..=3u8 {
        let opp = (player + i) % 4;
        if opp == (player ^ 2) {
            continue;
        }
        let opp_hand = state.hands[opp as usize];
        let opp_in_suit = cards_in_suit(opp_hand, card_s);
        if card_s == trump_suit {
            if opp_in_suit != 0 {
                let opp_best = highest_trump_in_set(opp_in_suit, trump_suit);
                if TRUMP_STRENGTH[card_rank(opp_best) as usize]
                    > TRUMP_STRENGTH[card_r as usize]
                {
                    return false;
                }
            }
        } else {
            if opp_in_suit != 0 {
                let opp_high = highest_plain_in_suit(opp_in_suit, card_s);
                if card_rank(opp_high) > card_r {
                    return false;
                }
            } else {
                let opp_trump = cards_in_suit(opp_hand, trump_suit);
                if opp_trump != 0 {
                    return false;
                }
            }
        }
    }
    true
}

/// Count total trump cards held by both opponents.
fn count_opponent_trumps(state: &GameState, player: u8, trump_suit: Suit) -> u32 {
    let opp1 = (player + 1) % 4;
    let opp2 = (player + 3) % 4;
    let t1 = cards_in_suit(state.hands[opp1 as usize], trump_suit);
    let t2 = cards_in_suit(state.hands[opp2 as usize], trump_suit);
    t1.count_ones() + t2.count_ones()
}

/// Make a card from suit and rank.
#[inline]
fn make_card(suit: Suit, rank: u8) -> Card {
    suit as u8 * 8 + rank
}

// ============================================================
// TESTS
// ============================================================

#[cfg(all(test, feature = "rand"))]
mod tests {
    use super::*;

    #[test]
    fn test_maxi_bid_always_legal() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let mut state = GameState::deal_random(0, &mut rng);
            while state.phase == Phase::Bidding && !state.is_terminal() {
                let legal = state.legal_actions();
                let action = maxi_bid(&state);
                assert!(
                    legal & (1u64 << action) != 0,
                    "Maxi bid returned illegal action {} (legal={:064b})",
                    action,
                    legal
                );
                state.step(action);
            }
        }
    }

    #[test]
    fn test_maxi_play_always_legal() {
        let mut rng = rand::thread_rng();
        for _ in 0..500 {
            let mut state = GameState::deal_random(0, &mut rng);
            // Use maxi bids
            while state.phase == Phase::Bidding && !state.is_terminal() {
                state.step(maxi_bid(&state));
            }
            if state.is_terminal() {
                continue;
            }
            // Verify every maxi play action is legal
            while !state.is_terminal() {
                let legal = crate::play::legal_plays(&state) as u64;
                let action = maxi_play_action(&state);
                assert!(
                    legal & (1u64 << action) != 0,
                    "Maxi play returned illegal action {} (legal={:032b})",
                    action,
                    legal as u32
                );
                state.step(action);
            }
        }
    }

    #[test]
    fn test_maxi_rollout_completes() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let mut state = GameState::deal_random(0, &mut rng);
            // Use maxi for both bid and play
            while !state.is_terminal() {
                let action = if state.phase == Phase::Bidding {
                    maxi_bid(&state)
                } else {
                    maxi_play_action(&state)
                };
                state.step(action);
            }
            let total = state.points[0] as u16 + state.points[1] as u16;
            assert!(
                total == 162 || total == 252 || total == 0,
                "Total points {} is unexpected",
                total
            );
        }
    }

    #[test]
    fn test_classify_case_a_jack_ace() {
        // Hand with J+A+7 in spades (3 cards) + ext ace → should classify as 80
        let hand = (1u32 << 3) | (1u32 << 7) | (1u32 << 0) // J+A+7 of spades
            | (1u32 << 15) // Ace of hearts (ext ace)
            | (1u32 << 17) | (1u32 << 18) | (1u32 << 19) | (1u32 << 20); // fillers
        let eval = maxi_eval_suit(hand, Suit::Spades);
        let value = classify_suit(hand, Suit::Spades, &eval, 0);
        assert!(value >= 8, "J+A+filler with ext ace should open at least 80, got {}", value * 10);
    }

    #[test]
    fn test_classify_case_b_jack_nine() {
        // J+9+8 in spades + ext ace → should be at least 90
        let hand = (1u32 << 3) | (1u32 << 2) | (1u32 << 1) // J+9+8 of spades
            | (1u32 << 15) // Ace of hearts
            | (1u32 << 17) | (1u32 << 18) | (1u32 << 19) | (1u32 << 20);
        let eval = maxi_eval_suit(hand, Suit::Spades);
        let value = classify_suit(hand, Suit::Spades, &eval, 0);
        // J+9 → at least 90, likely higher due to Case D checks
        assert!(value >= 9, "J+9+filler should open at least 90, got {}", value * 10);
    }

    #[test]
    fn test_classify_case_d_strong() {
        // J+9+A+10 in spades + 2 ext aces → should be 120+
        let hand = (1u32 << 3) | (1u32 << 2) | (1u32 << 7) | (1u32 << 6) // J+9+A+10 spades
            | (1u32 << 15) // Ace of hearts
            | (1u32 << 23) // Ace of diamonds
            | (1u32 << 25) | (1u32 << 26); // fillers
        let eval = maxi_eval_suit(hand, Suit::Spades);
        let value = classify_suit(hand, Suit::Spades, &eval, 0);
        assert!(value >= 12, "J+9+A+10 + 2 ext aces should open 120+, got {}", value * 10);
    }

    #[test]
    fn test_response_80_with_jack() {
        // Partner opens 80 in spades, I have J + ext ace + 2 trumps
        let hand = (1u32 << 3) | (1u32 << 0) // J+7 of spades
            | (1u32 << 15) // Ace of hearts
            | (1u32 << 17) | (1u32 << 18) | (1u32 << 19) | (1u32 << 20) | (1u32 << 21);
        let eval = maxi_eval_suit(hand, Suit::Spades);
        let side_aces = count_side_aces(hand, Suit::Spades);
        let score = maxi_response_80(hand, Suit::Spades, &eval, side_aces);
        // +20 (J) + 10 (ext ace with 2 trumps) = 30
        assert!(score >= 20, "Should raise at least 20 with J of trump, got {}", score);
    }

    #[test]
    fn test_response_80_void() {
        // Partner opens 80 in spades, I have no spades
        let hand = (1u32 << 15) | (1u32 << 14) | (1u32 << 13) | (1u32 << 12)
            | (1u32 << 23) | (1u32 << 22) | (1u32 << 21) | (1u32 << 20);
        let eval = maxi_eval_suit(hand, Suit::Spades);
        let side_aces = count_side_aces(hand, Suit::Spades);
        let score = maxi_response_80(hand, Suit::Spades, &eval, side_aces);
        assert!(score <= 0, "Should not raise with void in trump, got {}", score);
    }

    #[test]
    fn test_coinche_j9() {
        // Opponent bids spades, I have J+9 of spades → coinche
        let mut state = GameState::deal_random(0, &mut rand::thread_rng());
        state.phase = Phase::Bidding;
        state.last_bid_value = 8;
        state.last_bid_suit = 0; // spades
        state.last_bidder = 1; // opponent
        state.coinche_state = 0;
        state.current_player = 0;

        // Give player 0 a hand with J+9 of spades
        let hand = (1u32 << 3) | (1u32 << 2) | (1u32 << 0) // J+9+7 spades
            | (1u32 << 15) | (1u32 << 14) | (1u32 << 13)
            | (1u32 << 23) | (1u32 << 22);
        state.hands[0] = hand;

        let legal = state.legal_actions();
        let action = maxi_coinche(hand, &state, &legal);
        assert_eq!(action, BID_COINCHE, "Should coinche with J+9 in their suit");
    }
}
