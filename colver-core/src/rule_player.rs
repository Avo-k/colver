//! Rule-based card player with hidden information only.
//!
//! Unlike `rollout::heuristic_play_action` which sees all 4 hands,
//! this player only uses information available to a real human:
//! - Own hand
//! - Cards on the current trick
//! - All previously played cards (trick history)
//! - Known voids (from game engine tracking)
//! - Contract details
//!
//! Design principles:
//! - No reading `state.hands[other_player]`
//! - Decisions based on card counting and probability

use crate::card::*;
use crate::play::partner_is_master;
use crate::state::GameState;

/// Pick a card to play using only hidden-information rules.
pub fn rule_play_action(state: &GameState) -> u8 {
    let legal = crate::play::legal_plays(state) as u64;
    let legal32 = legal as CardSet;
    let count = legal32.count_ones();

    if count == 1 {
        return legal32.trailing_zeros() as u8;
    }

    let player = state.current_player;
    let trump_suit = state.contract.trump_suit();
    let ct = state.contract.contract_type();

    // Cards not in our hand and not yet played = held by others
    let my_hand = state.hands[player as usize];
    let outstanding = ALL_CARDS & !my_hand & !state.played_cards;

    if state.trick_count == 0 {
        return rule_lead(state, player, trump_suit, ct, legal32, outstanding);
    }

    let lead_card = state.current_trick[state.trick_lead as usize];
    let lead_suit = card_suit(lead_card);
    let my_in_lead = cards_in_suit(legal32, lead_suit);

    if my_in_lead != 0 {
        // FOLLOWING SUIT
        return rule_follow_suit(state, player, trump_suit, ct, my_in_lead, lead_suit, outstanding);
    }

    // CAN'T FOLLOW SUIT — must trump or discard
    rule_cant_follow(state, player, trump_suit, ct, legal32, outstanding)
}

// ── Leading ──────────────────────────────────────────────────────────

fn rule_lead(
    state: &GameState,
    player: u8,
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
    outstanding: CardSet,
) -> u8 {
    let taker_team = state.contract.team;
    let my_team = GameState::player_team(player);
    let my_trumps = cards_in_suit(legal, trump_suit);
    let out_trumps = cards_in_suit(outstanding, trump_suit);

    if my_team == taker_team {
        return rule_lead_taker(trump_suit, ct, legal, outstanding, my_trumps, out_trumps);
    }
    rule_lead_defender(trump_suit, ct, legal, outstanding, my_trumps)
}

fn rule_lead_taker(
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
    outstanding: CardSet,
    my_trumps: CardSet,
    out_trumps: CardSet,
) -> u8 {
    // 1. Draw trumps if we hold the master trump and opponents still have some
    if my_trumps != 0 && out_trumps != 0 {
        let best = highest_trump_in_set(my_trumps, trump_suit);
        if is_master_trump(best, trump_suit, outstanding) {
            return best;
        }
    }

    // 2. Cash master side-suit cards (guaranteed winners)
    for &suit in &ALL_SUITS {
        if suit == trump_suit { continue; }
        let in_suit = cards_in_suit(legal, suit);
        if in_suit == 0 { continue; }
        let high = highest_plain_in_suit(in_suit, suit);
        if is_master_plain(high, suit, outstanding) {
            return high;
        }
    }

    // 3. If no more outstanding trumps, cash everything from longest suit
    if out_trumps == 0 {
        let mut best_card = EMPTY;
        let mut best_len = 0;
        for &suit in &ALL_SUITS {
            if suit == trump_suit { continue; }
            let in_suit = cards_in_suit(legal, suit);
            let len = in_suit.count_ones();
            if len > best_len {
                best_len = len;
                best_card = highest_point_card(in_suit, ct);
            }
        }
        if best_card != EMPTY { return best_card; }
    }

    // 4. Lead from shortest non-trump to set up voids for trumping
    let mut best_card = EMPTY;
    let mut best_len = u32::MAX;
    for &suit in &ALL_SUITS {
        if suit == trump_suit { continue; }
        let in_suit = cards_in_suit(legal, suit);
        let len = in_suit.count_ones();
        if len > 0 && len < best_len {
            best_len = len;
            best_card = lowest_point_card(in_suit, ct);
        }
    }
    if best_card != EMPTY { return best_card; }

    // 5. Only trump left
    lowest_point_card(legal, ct)
}

fn rule_lead_defender(
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
    outstanding: CardSet,
    my_trumps: CardSet,
) -> u8 {
    // 1. Lead master side-suit cards (safe winners)
    for &suit in &ALL_SUITS {
        if suit == trump_suit { continue; }
        let in_suit = cards_in_suit(legal, suit);
        if in_suit == 0 { continue; }
        let high = highest_plain_in_suit(in_suit, suit);
        if is_master_plain(high, suit, outstanding) {
            return high;
        }
    }

    // 2. Lead from LONGEST non-trump suit — force taker to ruff
    //    This depletes their trump supply, which is the defender's main strategy.
    let mut best_card = EMPTY;
    let mut best_len = 0;
    for &suit in &ALL_SUITS {
        if suit == trump_suit { continue; }
        let in_suit = cards_in_suit(legal, suit);
        let len = in_suit.count_ones();
        if len > best_len {
            best_len = len;
            best_card = lowest_point_card(in_suit, ct);
        }
    }
    if best_card != EMPTY { return best_card; }

    // 3. Only trumps left — lead lowest
    if my_trumps != 0 {
        return lowest_trump_in_set(my_trumps, trump_suit);
    }
    lowest_point_card(legal, ct)
}

// ── Following suit ───────────────────────────────────────────────────

fn rule_follow_suit(
    state: &GameState,
    _player: u8,
    trump_suit: Suit,
    ct: ContractType,
    my_in_lead: CardSet,
    lead_suit: Suit,
    outstanding: CardSet,
) -> u8 {
    if lead_suit == trump_suit {
        // Following trump: play cheapest legal (overtrump rules handled by legal_plays)
        return lowest_trump_in_set(my_in_lead, trump_suit);
    }

    // Is the trick already trumped?
    let trumped = trick_is_trumped(state, trump_suit);

    if partner_is_master(state) {
        if trumped || is_last_to_play(state) {
            // Partner winning and trick is decided (trumped or we're last) → give points
            return highest_point_card(my_in_lead, ct);
        }
        // Partner winning but opponent still to play — play low, don't waste points
        return lowest_point_card(my_in_lead, ct);
    }

    if trumped {
        // Opponent trumped, we can only follow suit — play lowest
        return lowest_point_card(my_in_lead, ct);
    }

    // Opponent winning in lead suit — try to beat them
    let best_rank = best_plain_rank_on_trick(state, lead_suit);
    let winner = min_winning_plain(my_in_lead, lead_suit, best_rank);
    if winner != EMPTY {
        // Can beat: but is it worth it? If we're last, yes. If not, still try.
        return winner;
    }

    // Can't beat — play lowest
    lowest_point_card(my_in_lead, ct)
}

// ── Can't follow suit ────────────────────────────────────────────────

fn rule_cant_follow(
    state: &GameState,
    _player: u8,
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
    _outstanding: CardSet,
) -> u8 {
    let my_trumps = cards_in_suit(legal, trump_suit);

    if partner_is_master(state) {
        if !opponents_might_beat(state, trump_suit) {
            // Partner winning, opponents unlikely to beat → give points
            return highest_point_card(legal, ct);
        }
        // Partner winning but opponents might beat → play low
        return lowest_point_card(legal, ct);
    }

    // Opponent winning — try to trump
    if my_trumps != 0 {
        // Trump with cheapest trump
        return lowest_trump_in_set(my_trumps, trump_suit);
    }

    // No trump — discard lowest point card
    lowest_point_card(legal, ct)
}

// ── Fair-information helpers ─────────────────────────────────────────

/// Is this trump card the highest outstanding trump?
fn is_master_trump(card: Card, trump_suit: Suit, outstanding: CardSet) -> bool {
    let out_trump = cards_in_suit(outstanding, trump_suit);
    if out_trump == 0 {
        return true; // no outstanding trumps
    }
    let best_out = highest_trump_in_set(out_trump, trump_suit);
    TRUMP_STRENGTH[card_rank(card) as usize] > TRUMP_STRENGTH[card_rank(best_out) as usize]
}

/// Is this plain card the highest outstanding card in its suit?
fn is_master_plain(card: Card, suit: Suit, outstanding: CardSet) -> bool {
    let out_suit = cards_in_suit(outstanding, suit);
    if out_suit == 0 {
        return true; // no outstanding cards in this suit
    }
    let best_out = highest_plain_in_suit(out_suit, suit);
    card_rank(card) > card_rank(best_out)
}

/// Has the trick been trumped?
fn trick_is_trumped(state: &GameState, trump_suit: Suit) -> bool {
    for i in 0..state.trick_count {
        let seat = (state.trick_lead + i) % 4;
        let card = state.current_trick[seat as usize];
        if card != EMPTY && card_suit(card) == trump_suit {
            return true;
        }
    }
    false
}

/// Best plain rank on the current trick in the lead suit.
fn best_plain_rank_on_trick(state: &GameState, lead_suit: Suit) -> u8 {
    let mut best = 0u8;
    for i in 0..state.trick_count {
        let seat = (state.trick_lead + i) % 4;
        let card = state.current_trick[seat as usize];
        if card != EMPTY && card_suit(card) == lead_suit {
            let r = card_rank(card);
            if r > best {
                best = r;
            }
        }
    }
    best
}

/// Are we the last player to act on this trick?
fn is_last_to_play(state: &GameState) -> bool {
    state.trick_count == 3
}

/// Without seeing opponent hands, estimate if opponents after us might beat partner.
/// Uses known voids: if the only remaining opponent is void in lead suit and
/// has trump, they can ruff.
fn opponents_might_beat(state: &GameState, trump_suit: Suit) -> bool {
    let player = state.current_player;
    let partner = player ^ 2;
    let lead_card = state.current_trick[state.trick_lead as usize];
    let lead_suit = card_suit(lead_card);

    // Check remaining opponents (those who haven't played yet)
    for offset in 1..=(3 - state.trick_count) {
        let seat = (player + offset) % 4;
        if seat == partner {
            continue;
        }
        // If this opponent is known void in lead suit, they might trump
        if lead_suit != trump_suit && (state.voids[seat as usize] & (1 << lead_suit as u8)) != 0 {
            // Opponent void in lead suit — they could trump unless also void in trump
            if (state.voids[seat as usize] & (1 << trump_suit as u8)) == 0 {
                return true; // might have trump
            }
        }
        // If not void in lead suit, they might have a higher card
        if (state.voids[seat as usize] & (1 << lead_suit as u8)) == 0 {
            return true; // might have higher card in lead suit
        }
    }
    false
}
