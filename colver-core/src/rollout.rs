#[cfg(feature = "rand")]
use rand::Rng;

use crate::card::*;
use crate::play::partner_is_master;
use crate::state::{GameState, Phase};

/// Play the game to completion with random legal moves. State is mutated in place.
/// Returns rewards for both teams.
#[cfg(feature = "rand")]
pub fn rollout_random(state: &mut GameState, rng: &mut impl Rng) -> [f32; 2] {
    while !state.is_terminal() {
        let legal = state.legal_actions();
        debug_assert!(legal != 0);
        let count = legal.count_ones();
        let idx = rng.gen_range(0..count);
        let action = select_nth_bit(legal, idx);
        state.step(action);
    }
    state.rewards()
}

/// Run N independent rollouts from the same starting state.
/// Returns average rewards for both teams.
#[cfg(feature = "rand")]
pub fn rollout_batch(state: &GameState, n: u32, rng: &mut impl Rng) -> [f32; 2] {
    let mut total = [0.0f32; 2];
    for _ in 0..n {
        let mut s = *state; // Copy! ~56 bytes memcpy
        let r = rollout_random(&mut s, rng);
        total[0] += r[0];
        total[1] += r[1];
    }
    [total[0] / n as f32, total[1] / n as f32]
}

/// Play the game to completion using heuristic bidding and random plays.
/// State is mutated in place. Returns rewards for both teams.
#[cfg(feature = "rand")]
pub fn rollout_heuristic_bid(state: &mut GameState, rng: &mut impl Rng) -> [f32; 2] {
    while !state.is_terminal() {
        let action = if state.phase == Phase::Bidding {
            crate::bid_eval::heuristic_bid(state)
        } else {
            let legal = state.legal_actions();
            debug_assert!(legal != 0);
            let count = legal.count_ones();
            let idx = rng.gen_range(0..count);
            select_nth_bit(legal, idx)
        };
        state.step(action);
    }
    state.rewards()
}

/// Select the nth set bit from a u64.
#[inline]
pub fn select_nth_bit(mask: u64, mut n: u32) -> u8 {
    let mut remaining = mask;
    loop {
        debug_assert!(remaining != 0, "n exceeds number of set bits");
        let bit = remaining.trailing_zeros() as u8;
        if n == 0 {
            return bit;
        }
        n -= 1;
        remaining &= remaining - 1;
    }
}

// ---- Heuristic play helpers (determinized rollout: all hands visible) ----

/// Check if the trick has been trumped (any trump card on the table).
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

/// Find the highest plain rank currently winning in the lead suit on the trick.
fn best_plain_rank_on_trick(state: &GameState) -> u8 {
    let lead_card = state.current_trick[state.trick_lead as usize];
    let lead_suit = card_suit(lead_card);
    let mut best_rank = card_rank(lead_card);
    for i in 1..state.trick_count {
        let seat = (state.trick_lead + i) % 4;
        let card = state.current_trick[seat as usize];
        if card != EMPTY && card_suit(card) == lead_suit {
            let r = card_rank(card);
            if r > best_rank {
                best_rank = r;
            }
        }
    }
    best_rank
}

/// In a determinized rollout (all hands visible), check if leading `card`
/// is guaranteed to win the trick.
fn is_safe_lead(state: &GameState, player: u8, card: Card) -> bool {
    let card_s = card_suit(card);
    let trump_suit = state.contract.trump_suit();
    let card_r = card_rank(card);

    for i in 1..=3u8 {
        let opp = (player + i) % 4;
        // Skip partner
        if opp == (player ^ 2) {
            continue;
        }
        let opp_hand = state.hands[opp as usize];
        let opp_in_suit = cards_in_suit(opp_hand, card_s);
        if card_s == trump_suit {
            // We're leading trump — check if opponent has higher trump
            if opp_in_suit != 0 {
                let opp_best = highest_trump_in_set(opp_in_suit, trump_suit);
                if TRUMP_STRENGTH[card_rank(opp_best) as usize]
                    > TRUMP_STRENGTH[card_r as usize]
                {
                    return false;
                }
            }
        } else {
            // Leading non-trump
            if opp_in_suit != 0 {
                // Opponent has cards in our suit — check if they can beat us (plain ordering)
                let opp_high = highest_plain_in_suit(opp_in_suit, card_s);
                if card_rank(opp_high) > card_r {
                    return false;
                }
            } else {
                // Opponent void in our suit — can they cut with trump?
                let opp_trump = cards_in_suit(opp_hand, trump_suit);
                if opp_trump != 0 {
                    return false;
                }
            }
        }
    }
    true
}

/// After partner_is_master() returns true, check whether any opponent
/// who acts AFTER the current player in this trick can beat partner's card.
fn opponents_after_can_beat(state: &GameState) -> bool {
    let player = state.current_player;
    let trump_suit = state.contract.trump_suit();
    let lead_card = state.current_trick[state.trick_lead as usize];
    let lead_suit = card_suit(lead_card);

    // Find current winning card/seat
    let partner = player ^ 2;
    let partner_card = state.current_trick[partner as usize];
    let partner_suit = card_suit(partner_card);

    // Players who haven't acted yet (after current player)
    for offset in 1..=(4 - state.trick_count) {
        let seat = (player + offset) % 4;
        if seat == partner {
            continue;
        }
        let opp_hand = state.hands[seat as usize];

        if partner_suit == trump_suit {
            // Partner played trump — opponent needs higher trump
            let opp_trump = cards_in_suit(opp_hand, trump_suit);
            if opp_trump != 0 {
                let opp_best = highest_trump_in_set(opp_trump, trump_suit);
                if TRUMP_STRENGTH[card_rank(opp_best) as usize]
                    > TRUMP_STRENGTH[card_rank(partner_card) as usize]
                {
                    return true;
                }
            }
        } else {
            // Partner played lead suit (non-trump)
            let opp_in_lead = cards_in_suit(opp_hand, lead_suit);
            if opp_in_lead != 0 {
                let opp_high = highest_plain_in_suit(opp_in_lead, lead_suit);
                if card_rank(opp_high) > card_rank(partner_card) && !trick_is_trumped(state) {
                    return true;
                }
            } else {
                // Opponent void in lead suit — can cut with trump
                let opp_trump = cards_in_suit(opp_hand, trump_suit);
                if opp_trump != 0 {
                    return true;
                }
            }
        }
    }
    false
}

/// Count total trump cards held by both opponents.
fn count_opponent_trumps(state: &GameState, player: u8, trump_suit: Suit) -> u32 {
    let opp1 = (player + 1) % 4;
    let opp2 = (player + 3) % 4;
    let t1 = cards_in_suit(state.hands[opp1 as usize], trump_suit);
    let t2 = cards_in_suit(state.hands[opp2 as usize], trump_suit);
    t1.count_ones() + t2.count_ones()
}

/// Heuristic play action for determinized rollouts (all hands visible).
/// Picks from legal_plays() only, no allocations.
pub fn heuristic_play_action(state: &GameState) -> u8 {
    let legal = crate::play::legal_plays(state) as u64;
    let legal32 = legal as CardSet;
    let count = legal32.count_ones();

    // Forced move
    if count == 1 {
        return legal32.trailing_zeros() as u8;
    }

    let player = state.current_player;
    let trump_suit = state.contract.trump_suit();
    let ct = state.contract.contract_type();

    if state.trick_count == 0 {
        // LEADING
        return heuristic_lead(state, player, trump_suit, ct, legal32);
    }

    let lead_card = state.current_trick[state.trick_lead as usize];
    let lead_suit = card_suit(lead_card);
    let my_in_lead = cards_in_suit(legal32, lead_suit);

    if my_in_lead != 0 {
        // FOLLOWING SUIT
        if lead_suit == trump_suit {
            // Following trump — legal_plays enforces overtrump; play cheapest legal trump
            return lowest_trump_in_set(legal32, trump_suit);
        }
        // Following non-trump
        if trick_is_trumped(state) {
            return lowest_point_card(my_in_lead, ct);
        }
        if partner_is_master(state) {
            if !opponents_after_can_beat(state) {
                return highest_point_card(my_in_lead, ct);
            }
            return lowest_point_card(my_in_lead, ct);
        }
        // Opponent winning — try to beat with minimum card
        let best_rank = best_plain_rank_on_trick(state);
        let winner = min_winning_plain(my_in_lead, lead_suit, best_rank);
        if winner != EMPTY {
            return winner;
        }
        return lowest_point_card(my_in_lead, ct);
    }

    // CAN'T FOLLOW SUIT
    if partner_is_master(state) {
        if !opponents_after_can_beat(state) {
            return highest_point_card(legal32, ct);
        }
        return lowest_point_card(legal32, ct);
    }

    let my_trumps = cards_in_suit(legal32, trump_suit);
    if my_trumps == legal32 {
        // Only have trump — cheapest
        return lowest_trump_in_set(legal32, trump_suit);
    }
    if my_trumps != 0 {
        return lowest_trump_in_set(my_trumps, trump_suit);
    }
    // No trump — discard cheapest
    lowest_point_card(legal32, ct)
}

/// Heuristic for leading (trick_count == 0).
fn heuristic_lead(
    state: &GameState,
    player: u8,
    trump_suit: Suit,
    ct: ContractType,
    legal: CardSet,
) -> u8 {
    let taker_team = state.contract.team;
    let my_team = GameState::player_team(player);
    let my_trumps = cards_in_suit(legal, trump_suit);

    // 1. Taker team + have trump + opponents have trump → lead highest safe trump
    if my_team == taker_team && my_trumps != 0 {
        let opp_trumps = count_opponent_trumps(state, player, trump_suit);
        if opp_trumps > 0 {
            let best = highest_trump_in_set(my_trumps, trump_suit);
            if is_safe_lead(state, player, best) {
                return best;
            }
        }
    }

    // 2. For each non-trump suit, find safe lead with highest card
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

    // 3. Defender: lead any safe master side-suit card
    if my_team != taker_team {
        for &suit in &ALL_SUITS {
            if suit == trump_suit {
                continue;
            }
            let in_suit = cards_in_suit(legal, suit);
            if in_suit == 0 {
                continue;
            }
            // Try all cards in suit from highest
            let mut remaining = in_suit;
            while remaining != 0 {
                let top = (31 - remaining.leading_zeros()) as Card;
                if is_safe_lead(state, player, top) {
                    return top;
                }
                remaining &= !(1u32 << top);
            }
        }
    }

    // 4. Lead lowest-point card from shortest non-trump suit (set up void)
    let mut best_card = EMPTY;
    let mut best_suit_len = u32::MAX;
    for &suit in &ALL_SUITS {
        if suit == trump_suit {
            continue;
        }
        let in_suit = cards_in_suit(legal, suit);
        let len = in_suit.count_ones();
        if len > 0 && len < best_suit_len {
            best_suit_len = len;
            best_card = lowest_point_card(in_suit, ct);
        }
    }
    if best_card != EMPTY {
        return best_card;
    }

    // 5. Fallback: lowest point card overall
    lowest_point_card(legal, ct)
}

/// Play the game to completion using heuristic bidding + heuristic play.
/// State is mutated in place. Returns rewards for both teams.
#[cfg(feature = "rand")]
pub fn rollout_heuristic_play(state: &mut GameState, _rng: &mut impl Rng) -> [f32; 2] {
    while !state.is_terminal() {
        let action = if state.phase == Phase::Bidding {
            crate::bid_eval::heuristic_bid(state)
        } else {
            heuristic_play_action(state)
        };
        state.step(action);
    }
    state.rewards()
}

/// Lightweight struct for RAVE info from one rollout.
/// Only tracks playing-phase actions (cards 0-31).
pub struct RolloutRaveInfo {
    /// Bitmask of cards played by each team.
    pub cards_by_team: [u32; 2],
    /// Bitmask of cards where legal_count == 1 (forced).
    pub forced_cards: u32,
}

impl RolloutRaveInfo {
    pub fn new() -> Self {
        RolloutRaveInfo {
            cards_by_team: [0; 2],
            forced_cards: 0,
        }
    }
}

/// Heuristic bids + heuristic play, recording RAVE info.
#[cfg(feature = "rand")]
pub fn rollout_heuristic_play_with_rave(
    state: &mut GameState,
    rave_info: &mut RolloutRaveInfo,
    _rng: &mut impl Rng,
) -> [f32; 2] {
    while !state.is_terminal() {
        if state.phase == Phase::Bidding {
            let action = crate::bid_eval::heuristic_bid(state);
            state.step(action);
        } else {
            let legal = crate::play::legal_plays(state) as u64;
            let legal32 = legal as CardSet;
            let count = legal32.count_ones();
            let action = heuristic_play_action(state);
            let team = GameState::player_team(state.current_player) as usize;
            rave_info.cards_by_team[team] |= 1u32 << action;
            if count == 1 {
                rave_info.forced_cards |= 1u32 << action;
            }
            state.step(action);
        }
    }
    state.rewards()
}

#[cfg(all(test, feature = "rand"))]
mod tests {
    use super::*;

    #[test]
    fn test_rollout_random_completes() {
        let mut rng = rand::thread_rng();
        for _ in 0..100 {
            let mut state = GameState::deal_random(0, &mut rng);
            let _rewards = rollout_random(&mut state, &mut rng);
            assert!(state.is_terminal());
        }
    }

    #[test]
    fn test_rollout_batch() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);
        let avg = rollout_batch(&state, 100, &mut rng);
        // Rewards should be finite
        assert!(avg[0].is_finite());
        assert!(avg[1].is_finite());
    }

    #[test]
    fn test_rollout_preserves_original() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);
        let original = state;
        let _avg = rollout_batch(&state, 10, &mut rng);
        // Original state should be unchanged (it was passed by reference to batch)
        assert_eq!(state.phase, original.phase);
        assert_eq!(state.hands, original.hands);
    }

    #[test]
    fn test_select_nth_bit() {
        assert_eq!(select_nth_bit(0b1010, 0), 1);
        assert_eq!(select_nth_bit(0b1010, 1), 3);
        assert_eq!(select_nth_bit(0b100001, 0), 0);
        assert_eq!(select_nth_bit(0b100001, 1), 5);
    }

    #[test]
    fn test_heuristic_play_completes() {
        let mut rng = rand::thread_rng();
        for _ in 0..1000 {
            let mut state = GameState::deal_random(0, &mut rng);
            let _rewards = rollout_heuristic_play(&mut state, &mut rng);
            assert!(state.is_terminal());
            // Total card points should be 152 (before dix de der)
            let total = state.points[0] as u16 + state.points[1] as u16;
            // With dix de der: 162 or 252 (capot)
            assert!(
                total == 162 || total == 252,
                "Total points {} is neither 162 nor 252",
                total
            );
        }
    }

    #[test]
    fn test_heuristic_play_legal() {
        let mut rng = rand::thread_rng();
        for _ in 0..500 {
            let mut state = GameState::deal_random(0, &mut rng);
            // Use heuristic bids first
            while state.phase == Phase::Bidding && !state.is_terminal() {
                state.step(crate::bid_eval::heuristic_bid(&state));
            }
            if state.is_terminal() {
                continue;
            }
            // Now verify every heuristic play action is legal
            while !state.is_terminal() {
                let legal = crate::play::legal_plays(&state) as u64;
                let action = heuristic_play_action(&state);
                assert!(
                    legal & (1u64 << action) != 0,
                    "Heuristic play returned illegal action {} (legal={:032b})",
                    action,
                    legal
                );
                state.step(action);
            }
        }
    }

    #[test]
    fn test_heuristic_beats_random() {
        let mut rng = rand::thread_rng();
        let mut ns_wins = 0u32;
        let mut total = 0u32;

        for game in 0..500 {
            let dealer = (game % 4) as u8;
            let mut state = GameState::deal_random(dealer, &mut rng);

            // Heuristic bids for everyone
            while state.phase == Phase::Bidding && !state.is_terminal() {
                state.step(crate::bid_eval::heuristic_bid(&state));
            }
            if state.is_terminal() {
                continue;
            }

            // NS uses heuristic play, EW uses random play
            while !state.is_terminal() {
                let player = state.current_player;
                let action = if player & 1 == 0 {
                    // NS team: heuristic play
                    heuristic_play_action(&state)
                } else {
                    // EW team: random play
                    let legal = state.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    select_nth_bit(legal, idx)
                };
                state.step(action);
            }

            let score = state.deal_score();
            if score.scores[0] > score.scores[1] {
                ns_wins += 1;
            }
            total += 1;
        }

        let win_rate = ns_wins as f64 / total as f64;
        assert!(
            win_rate > 0.55,
            "Heuristic should beat random >55%, got {:.1}% ({}/{})",
            win_rate * 100.0,
            ns_wins,
            total
        );
    }

    #[test]
    fn test_heuristic_play_with_rave_completes() {
        let mut rng = rand::thread_rng();
        for _ in 0..200 {
            let mut state = GameState::deal_random(0, &mut rng);
            let mut rave_info = RolloutRaveInfo::new();
            let _rewards = rollout_heuristic_play_with_rave(&mut state, &mut rave_info, &mut rng);
            assert!(state.is_terminal());
            // Verify RAVE info is populated
            // Total cards played by both teams should be 32
            let total_cards = rave_info.cards_by_team[0].count_ones()
                + rave_info.cards_by_team[1].count_ones();
            // Some cards might be in void deals (bidding passes)
            if total_cards > 0 {
                assert_eq!(total_cards, 32, "All 32 cards should be tracked");
                // No overlap between teams
                assert_eq!(
                    rave_info.cards_by_team[0] & rave_info.cards_by_team[1],
                    0,
                    "Teams shouldn't play same cards"
                );
            }
        }
    }
}
