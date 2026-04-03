use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::state::*;
use super::eval_helpers::{evaluate_for_trump, count_side_aces};

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
pub(crate) fn count_total_aces(hand: CardSet) -> u32 {
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
pub(crate) fn roro_coinche(hand: CardSet, state: &GameState, legal: &u64) -> u8 {
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
pub(crate) fn bidding_position(state: &GameState) -> u8 {
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
