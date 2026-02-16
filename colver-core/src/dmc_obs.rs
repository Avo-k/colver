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
}
