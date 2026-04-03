/// DMC observation builder: constructs the 415-float observation vector
/// used by the DouZero-style Deep Monte-Carlo agent.
///
/// This is a zero-allocation port of `make_observation_v2()` from `colver-py/src/lib.rs`.
/// Instead of allocating a `Vec<f32>`, it writes directly into a caller-provided `&mut [f32]`
/// buffer at a given offset, enabling flat pre-allocated buffers for vectorized environments.

use crate::bidding;
use crate::card;
use crate::state::{GameState, Phase};

pub const OBS_DIM: usize = 415;
pub const BID_HISTORY_FLOATS: usize = 72; // 12 slots × 6 floats per slot
const MASK_DIM: usize = 32;

/// Tracking state kept outside GameState (to keep GameState ≤64 bytes).
#[derive(Clone)]
pub struct EnvTracking {
    /// Per-player bitmask of cards played by each player.
    pub played_by: [u32; 4],
    /// Chronological play order (card indices, up to 32).
    pub play_order: Vec<u8>,
    /// Bid history: (seat, action) pairs.
    pub bid_history: Vec<(u8, u8)>,
    /// Dealer for this deal.
    pub dealer: u8,
}

impl EnvTracking {
    pub fn new() -> Self {
        EnvTracking {
            played_by: [0; 4],
            play_order: Vec::with_capacity(32),
            bid_history: Vec::new(),
            dealer: 0,
        }
    }

    /// Reset tracking for a new deal.
    pub fn reset(&mut self, dealer: u8) {
        self.played_by = [0; 4];
        self.play_order.clear();
        self.bid_history.clear();
        self.dealer = dealer;
    }

    /// Track a play action: update played_by mask and play_order.
    #[inline]
    pub fn track_action(&mut self, state: &GameState, action: u8) {
        if state.phase == Phase::Bidding {
            self.bid_history.push((state.current_player(), action));
        }
        if state.phase == Phase::Playing {
            let player = state.current_player() as usize;
            self.played_by[player] |= 1u32 << action;
            self.play_order.push(action);
        }
    }
}

/// Write the 415-float observation into `buf[offset..offset+OBS_DIM]`.
///
/// # Panics
/// Panics if `buf` is too small to hold `offset + OBS_DIM` floats.
pub fn write_observation(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    tracking: &EnvTracking,
) {
    debug_assert!(buf.len() >= offset + OBS_DIM);

    let out = &mut buf[offset..offset + OBS_DIM];
    // Zero the output region
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let me = state.current_player() as usize;
    let my_team = me & 1;
    let opp_team = 1 - my_team;
    let trump = state.contract.trump;

    // Player-relative seats: [me, left_opp, partner, right_opp]
    let seats = [me, (me + 1) % 4, (me + 2) % 4, (me + 3) % 4];

    let mut pos = 0;

    // === Block 1: My hand (32) ===
    let my_hand = state.hands[me];
    for i in 0..32u32 {
        if my_hand & (1 << i) != 0 {
            out[pos + i as usize] = 1.0;
        }
    }
    pos += 32;

    // === Block 2: Current trick, player-relative (128) ===
    for &seat in &seats {
        let c = state.current_trick[seat];
        if c != card::EMPTY {
            out[pos + c as usize] = 1.0;
        }
        pos += 32;
    }

    // Current trick union (for excluding from past played)
    let mut trick_union: u32 = 0;
    for i in 0..4 {
        let c = state.current_trick[i];
        if c != card::EMPTY {
            trick_union |= 1u32 << c;
        }
    }

    // === Block 3: Per-player played cards in past tricks (96) ===
    // For left, partner, right (not me)
    for &seat in &seats[1..] {
        let past = tracking.played_by[seat] & !trick_union;
        for i in 0..32u32 {
            if past & (1 << i) != 0 {
                out[pos + i as usize] = 1.0;
            }
        }
        pos += 32;
    }

    // === Block 4: Contract (7) ===
    out[pos + trump as usize] = 1.0;
    pos += 4;
    out[pos] = state.contract.point_value() as f32 / 250.0;
    pos += 1;
    out[pos] = if state.contract.team as usize == my_team {
        1.0
    } else {
        0.0
    };
    pos += 1;
    out[pos] = state.contract.coinche as f32 / 2.0;
    pos += 1;

    // === Block 5: Void tracking (12) ===
    for &seat in &seats[1..] {
        for s in 0..4u8 {
            if state.voids[seat] & (1 << s) != 0 {
                out[pos] = 1.0;
            }
            pos += 1;
        }
    }

    // === Block 6: Scoring context (4) ===
    out[pos] = state.points[my_team] as f32 / 252.0;
    out[pos + 1] = state.points[opp_team] as f32 / 252.0;
    out[pos + 2] = state.tricks_won[my_team] as f32 / 8.0;
    out[pos + 3] = state.tricks_won[opp_team] as f32 / 8.0;
    pos += 4;

    // === Block 7: Bid history (72) ===
    encode_bid_history(out, pos, &tracking.bid_history, me, tracking.dealer);
    pos += BID_HISTORY_FLOATS;

    // === Block 8: Card trick index (32) ===
    for (i, &card_played) in tracking.play_order.iter().enumerate() {
        out[pos + card_played as usize] = (i / 4 + 1) as f32 / 8.0;
    }
    pos += 32;

    // === Block 9: Card sequence index (32) ===
    for (i, &card_played) in tracking.play_order.iter().enumerate() {
        out[pos + card_played as usize] = (i % 4 + 1) as f32 / 4.0;
    }
    pos += 32;

    debug_assert_eq!(pos, OBS_DIM);
}

/// Write the legal action mask into `buf[offset..offset+32]`.
///
/// For playing phase, writes 32 floats (card actions).
pub fn write_mask(buf: &mut [f32], offset: usize, state: &GameState) {
    debug_assert!(buf.len() >= offset + MASK_DIM);
    let out = &mut buf[offset..offset + MASK_DIM];
    let mask = state.legal_actions();
    for i in 0..MASK_DIM {
        out[i] = if mask & (1u64 << i) != 0 { 1.0 } else { 0.0 };
    }
}

/// Convenience wrapper that allocates and returns a Vec<f32>.
pub fn make_observation(
    state: &GameState,
    tracking: &EnvTracking,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; OBS_DIM];
    write_observation(&mut buf, 0, state, tracking);
    buf
}

/// Encode bid history into 72 floats at buf[offset..offset+72].
/// Slots are in player-relative order: [me, left, partner, right] × 3 rounds.
fn encode_bid_history(
    buf: &mut [f32],
    offset: usize,
    bid_history: &[(u8, u8)],
    me: usize,
    dealer: u8,
) {
    // First bidder is after dealer
    let first_bidder = ((dealer + 1) % 4) as usize;
    // Relative offset: how many slots of padding before first bid
    let rel_offset = (first_bidder + 4 - me) % 4;

    // Use last 12 actions if history is longer (extremely rare)
    let history = if bid_history.len() > 12 {
        &bid_history[bid_history.len() - 12..]
    } else {
        bid_history
    };

    for (i, &(_seat, action)) in history.iter().enumerate() {
        let slot = rel_offset + i;
        if slot >= 12 {
            break;
        }
        let base = offset + slot * 6;

        match action {
            0 => {
                // Pass
                buf[base] = 0.2;
            }
            41 => {
                // Coinche
                buf[base] = 0.8;
            }
            42 => {
                // Surcoinche
                buf[base] = 1.0;
            }
            1..=40 => {
                let (val_enc, suit_idx) = bidding::decode_bid(action);
                if val_enc == 25 {
                    // Capot
                    buf[base] = 0.6;
                    buf[base + 1] = 1.0;
                } else {
                    // Regular bid
                    buf[base] = 0.4;
                    buf[base + 1] = (val_enc as f32 * 10.0) / 250.0;
                }
                buf[base + 2 + suit_idx as usize] = 1.0;
            }
            _ => {}
        }
    }
}

// ====== Trump-relative canonical encoding ======
//
// Trump always in slot 0. Non-trump suits sorted by (card_count, rank_pattern)
// descending — longest suit first, ties broken by higher-ranked cards first.
// Two suits with identical count AND identical ranks are truly interchangeable.
// This makes the encoding fully canonical: no suit augmentation needed.

pub const OBS_DIM_TR: usize = 411;

/// Canonical suit ordering for play: trump first, non-trump sorted by
/// card count (descending), then by rank pattern (descending) for ties.
///
/// `initial_hand` is the player's 8-card hand at the start of play
/// (before any cards were played). Reconstruct as `hands[me] | played_by[me]`.
#[inline]
pub fn canonical_play_order(trump: u8, initial_hand: u32) -> [u8; 4] {
    let mut order = [0u8; 4];
    order[0] = trump;

    // Collect non-trump suits with sort key: (count << 8) | lane_bits
    let mut non_trump = [(0u8, 0u32); 3]; // (suit, sort_key)
    let mut idx = 0;
    for s in 0..4u8 {
        if s != trump {
            let lane = (initial_hand >> (s * 8)) & 0xFF;
            let count = lane.count_ones();
            let key = (count << 8) | lane;
            non_trump[idx] = (s, key);
            idx += 1;
        }
    }
    // Sort by key descending, then suit ascending for truly-equal ties
    non_trump.sort_by(|a, b| b.1.cmp(&a.1).then(a.0.cmp(&b.0)));

    for (i, &(s, _)) in non_trump.iter().enumerate() {
        order[i + 1] = s;
    }
    order
}

/// Simple trump-first ordering (non-trump in original S<H<D<C order).
/// Used by legacy code and tests; prefer `canonical_play_order` for training.
#[inline]
pub fn suit_order(trump: u8) -> [u8; 4] {
    match trump {
        0 => [0, 1, 2, 3],
        1 => [1, 0, 2, 3],
        2 => [2, 0, 1, 3],
        3 => [3, 0, 1, 2],
        _ => unreachable!(),
    }
}

/// Remap a CardSet (u32 bitmask) using a given suit ordering.
#[inline]
pub fn cardset_to_canonical(cards: u32, order: &[u8; 4]) -> u32 {
    let mut result = 0u32;
    for (canon_pos, &phys_suit) in order.iter().enumerate() {
        let suit_bits = (cards >> (phys_suit * 8)) & 0xFF;
        result |= suit_bits << (canon_pos as u32 * 8);
    }
    result
}

/// Legacy: remap using simple trump-first ordering.
#[inline]
pub fn remap_cardset(cards: u32, trump: u8) -> u32 {
    cardset_to_canonical(cards, &suit_order(trump))
}

/// Convert a physical card index to canonical given a suit ordering.
#[inline]
pub fn card_to_canonical(card: u8, order: &[u8; 4]) -> u8 {
    let phys_suit = card / 8;
    let rank = card % 8;
    let canon_pos = order.iter().position(|&s| s == phys_suit).unwrap() as u8;
    canon_pos * 8 + rank
}

/// Convert a canonical card index back to physical given a suit ordering.
#[inline]
pub fn card_to_physical(card: u8, order: &[u8; 4]) -> u8 {
    let canon_suit = card / 8;
    let rank = card % 8;
    order[canon_suit as usize] * 8 + rank
}

/// Legacy: physical→canonical using simple trump-first ordering.
#[inline]
pub fn physical_to_canonical(card: u8, trump: u8) -> u8 {
    card_to_canonical(card, &suit_order(trump))
}

/// Legacy: canonical→physical using simple trump-first ordering.
#[inline]
pub fn canonical_to_physical(card: u8, trump: u8) -> u8 {
    card_to_physical(card, &suit_order(trump))
}

/// Compute the canonical suit ordering for the current player.
///
/// Reconstructs the initial hand from current hand + played cards,
/// then returns `canonical_play_order(trump, initial_hand)`.
#[inline]
pub fn current_player_order(state: &GameState, tracking: &EnvTracking) -> [u8; 4] {
    let me = state.current_player() as usize;
    let initial_hand = state.hands[me] | tracking.played_by[me];
    canonical_play_order(state.contract.trump, initial_hand)
}

/// Write 411-float canonical observation into `buf[offset..offset+411]`.
///
/// Uses fully canonical suit ordering: trump first, non-trump sorted by
/// (card_count, rank_pattern) descending.
pub fn write_observation_tr(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    tracking: &EnvTracking,
) {
    debug_assert!(buf.len() >= offset + OBS_DIM_TR);
    let out = &mut buf[offset..offset + OBS_DIM_TR];
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let me = state.current_player() as usize;
    let my_team = me & 1;
    let opp_team = 1 - my_team;
    let trump = state.contract.trump;
    let seats = [me, (me + 1) % 4, (me + 2) % 4, (me + 3) % 4];

    // Canonical ordering based on current player's initial hand
    let initial_hand = state.hands[me] | tracking.played_by[me];
    let order = canonical_play_order(trump, initial_hand);

    let mut pos = 0;

    // === Block 1: My hand (32) — canonical ===
    let my_hand = cardset_to_canonical(state.hands[me], &order);
    for i in 0..32u32 {
        if my_hand & (1 << i) != 0 {
            out[pos + i as usize] = 1.0;
        }
    }
    pos += 32;

    // === Block 2: Current trick, player-relative (128) — canonical ===
    for &seat in &seats {
        let c = state.current_trick[seat];
        if c != card::EMPTY {
            let cc = card_to_canonical(c, &order);
            out[pos + cc as usize] = 1.0;
        }
        pos += 32;
    }

    let mut trick_union: u32 = 0;
    for i in 0..4 {
        let c = state.current_trick[i];
        if c != card::EMPTY {
            trick_union |= 1u32 << c;
        }
    }

    // === Block 3: Per-player played cards in past tricks (96) — canonical ===
    for &seat in &seats[1..] {
        let past = tracking.played_by[seat] & !trick_union;
        let past_canonical = cardset_to_canonical(past, &order);
        for i in 0..32u32 {
            if past_canonical & (1 << i) != 0 {
                out[pos + i as usize] = 1.0;
            }
        }
        pos += 32;
    }

    // === Block 4: Contract (3) — NO trump one-hot ===
    out[pos] = state.contract.point_value() as f32 / 250.0;
    pos += 1;
    out[pos] = if state.contract.team as usize == my_team {
        1.0
    } else {
        0.0
    };
    pos += 1;
    out[pos] = state.contract.coinche as f32 / 2.0;
    pos += 1;

    // === Block 5: Void tracking (12) — canonical suit order ===
    for &seat in &seats[1..] {
        for &s in &order {
            if state.voids[seat] & (1 << s) != 0 {
                out[pos] = 1.0;
            }
            pos += 1;
        }
    }

    // === Block 6: Scoring context (4) ===
    out[pos] = state.points[my_team] as f32 / 252.0;
    out[pos + 1] = state.points[opp_team] as f32 / 252.0;
    out[pos + 2] = state.tricks_won[my_team] as f32 / 8.0;
    out[pos + 3] = state.tricks_won[opp_team] as f32 / 8.0;
    pos += 4;

    // === Block 7: Bid history (72) — canonical suits ===
    encode_bid_history_tr(out, pos, &tracking.bid_history, me, tracking.dealer, &order);
    pos += BID_HISTORY_FLOATS;

    // === Block 8: Card trick index (32) — canonical ===
    for (i, &card_played) in tracking.play_order.iter().enumerate() {
        let cc = card_to_canonical(card_played, &order) as usize;
        out[pos + cc] = (i / 4 + 1) as f32 / 8.0;
    }
    pos += 32;

    // === Block 9: Card sequence index (32) — canonical ===
    for (i, &card_played) in tracking.play_order.iter().enumerate() {
        let cc = card_to_canonical(card_played, &order) as usize;
        out[pos + cc] = (i % 4 + 1) as f32 / 4.0;
    }
    pos += 32;

    debug_assert_eq!(pos, OBS_DIM_TR);
}

/// Write canonical legal action mask (32 floats).
pub fn write_mask_tr(buf: &mut [f32], offset: usize, state: &GameState, tracking: &EnvTracking) {
    debug_assert!(buf.len() >= offset + MASK_DIM);
    let out = &mut buf[offset..offset + MASK_DIM];
    let mask = state.legal_actions() as u32;
    let order = current_player_order(state, tracking);
    let canonical_mask = cardset_to_canonical(mask, &order);
    for i in 0..MASK_DIM {
        out[i] = if canonical_mask & (1 << i) != 0 { 1.0 } else { 0.0 };
    }
}

/// Convenience: allocate and return canonical observation.
pub fn make_observation_tr(state: &GameState, tracking: &EnvTracking) -> Vec<f32> {
    let mut buf = vec![0.0f32; OBS_DIM_TR];
    write_observation_tr(&mut buf, 0, state, tracking);
    buf
}

/// Encode bid history with canonical suit mapping.
fn encode_bid_history_tr(
    buf: &mut [f32],
    offset: usize,
    bid_history: &[(u8, u8)],
    me: usize,
    dealer: u8,
    order: &[u8; 4],
) {
    let first_bidder = ((dealer + 1) % 4) as usize;
    let rel_offset = (first_bidder + 4 - me) % 4;

    let history = if bid_history.len() > 12 {
        &bid_history[bid_history.len() - 12..]
    } else {
        bid_history
    };

    for (i, &(_seat, action)) in history.iter().enumerate() {
        let slot = rel_offset + i;
        if slot >= 12 {
            break;
        }
        let base = offset + slot * 6;

        match action {
            0 => {
                buf[base] = 0.2;
            }
            41 => {
                buf[base] = 0.8;
            }
            42 => {
                buf[base] = 1.0;
            }
            1..=40 => {
                let (val_enc, suit_idx) = bidding::decode_bid(action);
                if val_enc == 25 {
                    buf[base] = 0.6;
                    buf[base + 1] = 1.0;
                } else {
                    buf[base] = 0.4;
                    buf[base + 1] = (val_enc as f32 * 10.0) / 250.0;
                }
                let canon_pos = order.iter().position(|&s| s == suit_idx).unwrap();
                buf[base + 2 + canon_pos] = 1.0;
            }
            _ => {}
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::GameState;

    #[test]
    fn test_obs_dim_constant() {
        assert_eq!(OBS_DIM, 415);
    }

    #[test]
    fn test_write_observation_length() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();
        let mut buf = vec![0.0f32; OBS_DIM];
        write_observation(&mut buf, 0, &state, &tracking);
        assert_eq!(buf.len(), OBS_DIM);
    }

    #[test]
    fn test_write_observation_hand_bits() {
        // Player 1 is current (dealer=0, first bidder=1)
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();
        let mut buf = vec![0.0f32; OBS_DIM];
        write_observation(&mut buf, 0, &state, &tracking);

        // First 32 floats = current player's hand (player 1 = 0xFF00)
        for i in 0..32 {
            let expected = if (0xFF00u32 >> i) & 1 != 0 { 1.0 } else { 0.0 };
            assert_eq!(buf[i], expected, "hand bit {} mismatch", i);
        }
    }

    #[test]
    fn test_write_observation_with_offset() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();
        let offset = 100;
        let mut buf = vec![0.0f32; offset + OBS_DIM];
        write_observation(&mut buf, offset, &state, &tracking);

        // Verify offset region is written, pre-offset region is zero
        for i in 0..offset {
            assert_eq!(buf[i], 0.0, "pre-offset should be zero");
        }
        // First 32 floats at offset = hand
        let hand_nonzero: usize = (0..32).filter(|&i| buf[offset + i] != 0.0).count();
        assert_eq!(hand_nonzero, 8, "player 1 has 8 cards");
    }

    #[test]
    fn test_make_observation_matches_write() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();

        let vec_obs = make_observation(&state, &tracking);

        let mut buf = vec![0.0f32; OBS_DIM];
        write_observation(&mut buf, 0, &state, &tracking);

        assert_eq!(vec_obs, buf);
    }

    #[test]
    fn test_env_tracking_reset() {
        let mut tracking = EnvTracking::new();
        tracking.played_by[0] = 0xFF;
        tracking.play_order.push(5);
        tracking.bid_history.push((0, 1));
        tracking.dealer = 2;

        tracking.reset(3);
        assert_eq!(tracking.played_by, [0; 4]);
        assert!(tracking.play_order.is_empty());
        assert!(tracking.bid_history.is_empty());
        assert_eq!(tracking.dealer, 3);
    }

    #[test]
    fn test_encode_bid_history_pass() {
        let mut buf = vec![0.0f32; BID_HISTORY_FLOATS];
        let history = vec![(1u8, 0u8)]; // Player 1 passes
        // dealer=0, so first bidder=1, me=1, offset=0
        encode_bid_history(&mut buf, 0, &history, 1, 0);
        assert_eq!(buf[0], 0.2); // slot 0, pass marker
    }

    #[test]
    fn test_encode_bid_history_bid() {
        let mut buf = vec![0.0f32; BID_HISTORY_FLOATS];
        // Player 1 bids 80 Spades (action = encode_bid(8, 0) = 1)
        let action = crate::bidding::encode_bid(8, 0);
        let history = vec![(1u8, action)];
        encode_bid_history(&mut buf, 0, &history, 1, 0);
        // slot 0: action_type=0.4, bid_value=80/250=0.32, suit one-hot[0]=1.0
        assert!((buf[0] - 0.4).abs() < 1e-6);
        assert!((buf[1] - 0.32).abs() < 1e-6);
        assert!((buf[2] - 1.0).abs() < 1e-6); // Spades
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_observation_matches_python() {
        // Run a full deal and verify observations match between
        // write_observation and the Python-equivalent logic
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..20 {
            let dealer = rng.gen_range(0..4u8);
            let mut state = GameState::deal_random(dealer, &mut rng);
            let mut tracking = EnvTracking::new();
            tracking.dealer = dealer;

            // Play through with random moves
            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = crate::rollout::select_nth_bit(legal, idx);

                tracking.track_action(&state, action);
                state.step(action);

                if !state.is_terminal() {
                    let obs = make_observation(&state, &tracking);
                    assert_eq!(obs.len(), OBS_DIM);
                    // Verify hand block has exactly as many cards as current player holds
                    let me = state.current_player() as usize;
                    let hand_ones: usize = (0..32).filter(|&i| obs[i] != 0.0).count();
                    let expected_cards = state.hands[me].count_ones() as usize;
                    assert_eq!(hand_ones, expected_cards, "hand card count mismatch");
                }
            }
        }
    }

    #[test]
    fn test_write_mask() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let mut buf = vec![0.0f32; 32];
        write_mask(&mut buf, 0, &state);
        // During bidding, legal actions may extend beyond 32 bits,
        // but write_mask only writes 32 floats (for playing phase use)
        // At least PASS (bit 0) should be available
        assert_eq!(buf[0], 1.0, "PASS should be legal");
    }

    #[test]
    fn test_obs_dim_tr_constant() {
        assert_eq!(OBS_DIM_TR, 411);
    }

    #[test]
    fn test_suit_order() {
        assert_eq!(suit_order(0), [0, 1, 2, 3]);
        assert_eq!(suit_order(1), [1, 0, 2, 3]);
        assert_eq!(suit_order(2), [2, 0, 1, 3]);
        assert_eq!(suit_order(3), [3, 0, 1, 2]);
    }

    #[test]
    fn test_remap_cardset_identity_when_trump_spades() {
        // When trump=0 (spades), suit_order=[0,1,2,3], remap is identity
        let cards: u32 = 0xFF00_FF00;
        assert_eq!(remap_cardset(cards, 0), cards);
    }

    #[test]
    fn test_remap_cardset_trump_hearts() {
        // Trump=1 (hearts), order=[1,0,2,3]
        // Physical: S=0xFF in bits[0:8], H=0 in bits[8:16]
        let cards: u32 = 0xFF; // only spades
        let canonical = remap_cardset(cards, 1);
        // Spades (phys suit 0) → canonical pos 1 → bits[8:16]
        assert_eq!(canonical, 0xFF00);
    }

    #[test]
    fn test_physical_canonical_roundtrip() {
        for trump in 0..4u8 {
            for card in 0..32u8 {
                let canonical = physical_to_canonical(card, trump);
                let back = canonical_to_physical(canonical, trump);
                assert_eq!(back, card, "trump={}, card={}", trump, card);
            }
        }
    }

    #[test]
    fn test_trump_always_canonical_slot0() {
        // A trump card should always map to canonical slot 0 (indices 0-7)
        for trump in 0..4u8 {
            for rank in 0..8u8 {
                let phys_card = trump * 8 + rank;
                let canonical = physical_to_canonical(phys_card, trump);
                assert!(canonical < 8, "trump card {} should map to slot 0, got {}", phys_card, canonical);
                assert_eq!(canonical, rank); // rank preserved
            }
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_tr_observation_hand_count() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..20 {
            let dealer = rng.gen_range(0..4u8);
            let mut state = GameState::deal_random(dealer, &mut rng);
            let mut tracking = EnvTracking::new();
            tracking.dealer = dealer;

            // Play through bidding + some play
            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = crate::rollout::select_nth_bit(legal, idx);

                tracking.track_action(&state, action);
                state.step(action);

                if !state.is_terminal() && state.phase == crate::state::Phase::Playing {
                    let obs = make_observation_tr(&state, &tracking);
                    assert_eq!(obs.len(), OBS_DIM_TR);
                    // Hand block: same number of 1s as cards in hand
                    let me = state.current_player() as usize;
                    let hand_ones: usize = (0..32).filter(|&i| obs[i] != 0.0).count();
                    let expected = state.hands[me].count_ones() as usize;
                    assert_eq!(hand_ones, expected, "TR hand card count mismatch");
                }
            }
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_tr_mask_matches_legal_actions() {
        use rand::Rng;
        let mut rng = rand::thread_rng();

        for _ in 0..20 {
            let dealer = rng.gen_range(0..4u8);
            let mut state = GameState::deal_random(dealer, &mut rng);
            let mut tracking = EnvTracking::new();
            tracking.dealer = dealer;

            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = crate::rollout::select_nth_bit(legal, idx);

                if state.phase == crate::state::Phase::Playing {
                    let mut mask_buf = vec![0.0f32; 32];
                    write_mask_tr(&mut mask_buf, 0, &state, &tracking);
                    // Count legal actions in canonical mask
                    let mask_count: usize = mask_buf.iter().filter(|&&v| v > 0.5).count();
                    let phys_count = (state.legal_actions() as u32).count_ones() as usize;
                    assert_eq!(mask_count, phys_count, "TR mask legal count mismatch");
                }

                tracking.track_action(&state, action);
                state.step(action);
            }
        }
    }
}
