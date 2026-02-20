/// Bidding observation builder: constructs the 114-float observation vector
/// for the NN bidding model.
///
/// Zero-allocation pattern: writes into caller-provided buffer at offset.
///
/// Layout (114 floats):
///   Block 1: Hand (32)        — binary card presence
///   Block 2: Bid history (72) — 12 slots × 6 floats, player-relative
///   Block 3: Position (4)     — one-hot dealer-relative seat
///   Block 4: Auction state (6) — bid_value/160, suit one-hot(4), coinche/2

use crate::bidding;
use crate::state::{GameState, Phase};

pub const BID_OBS_DIM: usize = 114;
pub const BID_MASK_DIM: usize = 43;

/// Write the 114-float bidding observation into `buf[offset..offset+BID_OBS_DIM]`.
pub fn write_bid_observation(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    bid_history: &[(u8, u8)], // (seat, action) pairs
) {
    debug_assert!(buf.len() >= offset + BID_OBS_DIM);
    debug_assert_eq!(state.phase, Phase::Bidding);

    let out = &mut buf[offset..offset + BID_OBS_DIM];
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let me = state.current_player() as usize;
    let mut pos = 0;

    // === Block 1: My hand (32) ===
    let my_hand = state.hands[me];
    for i in 0..32u32 {
        if my_hand & (1 << i) != 0 {
            out[pos + i as usize] = 1.0;
        }
    }
    pos += 32;

    // === Block 2: Bid history (72) ===
    encode_bid_history(out, pos, bid_history, me, state.dealer);
    pos += 72;

    // === Block 3: Dealer-relative position (4) ===
    let rel_pos = (me + 4 - state.dealer as usize) % 4;
    out[pos + rel_pos] = 1.0;
    pos += 4;

    // === Block 4: Auction state (6) ===
    // Current highest bid value normalized
    out[pos] = state.last_bid_value as f32 * 10.0 / 160.0;
    pos += 1;
    // Suit of highest bid (one-hot, only if there is a bid)
    if state.last_bid_value > 0 {
        out[pos + state.last_bid_suit as usize] = 1.0;
    }
    pos += 4;
    // Coinche state
    out[pos] = state.coinche_state as f32 / 2.0;
    pos += 1;

    debug_assert_eq!(pos, BID_OBS_DIM);
}

/// Write the 43-float legal bid mask into `buf[offset..offset+43]`.
pub fn write_bid_mask(buf: &mut [f32], offset: usize, state: &GameState) {
    debug_assert!(buf.len() >= offset + BID_MASK_DIM);
    let out = &mut buf[offset..offset + BID_MASK_DIM];
    let mask = state.legal_actions();
    for i in 0..BID_MASK_DIM {
        out[i] = if mask & (1u64 << i) != 0 { 1.0 } else { 0.0 };
    }
}

/// Encode bid history into 72 floats at buf[offset..offset+72].
/// Reuses the same encoding as dmc_obs.rs for consistency.
fn encode_bid_history(
    buf: &mut [f32],
    offset: usize,
    bid_history: &[(u8, u8)],
    me: usize,
    dealer: u8,
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
                buf[base + 2 + suit_idx as usize] = 1.0;
            }
            _ => {}
        }
    }
}

/// Convenience: allocate and return a Vec<f32> observation.
pub fn make_bid_observation(
    state: &GameState,
    bid_history: &[(u8, u8)],
) -> Vec<f32> {
    let mut buf = vec![0.0f32; BID_OBS_DIM];
    write_bid_observation(&mut buf, 0, state, bid_history);
    buf
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::GameState;

    #[test]
    fn test_bid_obs_dim() {
        assert_eq!(BID_OBS_DIM, 114);
        assert_eq!(BID_MASK_DIM, 43);
    }

    #[test]
    fn test_write_bid_observation_length() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let mut buf = vec![0.0f32; BID_OBS_DIM];
        write_bid_observation(&mut buf, 0, &state, &[]);
        assert_eq!(buf.len(), BID_OBS_DIM);
    }

    #[test]
    fn test_hand_encoding() {
        // Dealer=0, first bidder=1, so current_player=1
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let obs = make_bid_observation(&state, &[]);

        // Player 1 has hand 0xFF00 (bits 8-15)
        for i in 0..32 {
            let expected = if (0xFF00u32 >> i) & 1 != 0 { 1.0 } else { 0.0 };
            assert_eq!(obs[i], expected, "hand bit {} mismatch", i);
        }
    }

    #[test]
    fn test_position_encoding() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let obs = make_bid_observation(&state, &[]);

        // Dealer=0, current_player=1 → dealer-relative position = (1-0)%4 = 1
        let pos_offset = 32 + 72; // after hand + bid_history
        assert_eq!(obs[pos_offset + 0], 0.0);
        assert_eq!(obs[pos_offset + 1], 1.0); // position 1
        assert_eq!(obs[pos_offset + 2], 0.0);
        assert_eq!(obs[pos_offset + 3], 0.0);
    }

    #[test]
    fn test_auction_state_empty() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let obs = make_bid_observation(&state, &[]);

        // No bids yet: bid_value=0, suit one-hot all zero, coinche=0
        let auc_offset = 32 + 72 + 4;
        assert_eq!(obs[auc_offset], 0.0); // bid value
        assert_eq!(obs[auc_offset + 1], 0.0); // suit 0
        assert_eq!(obs[auc_offset + 2], 0.0); // suit 1
        assert_eq!(obs[auc_offset + 3], 0.0); // suit 2
        assert_eq!(obs[auc_offset + 4], 0.0); // suit 3
        assert_eq!(obs[auc_offset + 5], 0.0); // coinche
    }

    #[test]
    fn test_auction_state_after_bid() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let mut state = GameState::new(0, hands);
        let mut bid_history = Vec::new();

        // Player 1 bids 80 Hearts (action = encode_bid(8, 1))
        let action = crate::bidding::encode_bid(8, 1);
        bid_history.push((1, action));
        state.step(action);

        // Now current_player=2
        let obs = make_bid_observation(&state, &bid_history);

        let auc_offset = 32 + 72 + 4;
        assert!((obs[auc_offset] - 80.0 / 160.0).abs() < 1e-6); // bid_value = 80/160 = 0.5
        assert_eq!(obs[auc_offset + 1], 0.0); // not spades
        assert_eq!(obs[auc_offset + 2], 1.0); // hearts
        assert_eq!(obs[auc_offset + 3], 0.0);
        assert_eq!(obs[auc_offset + 4], 0.0);
        assert_eq!(obs[auc_offset + 5], 0.0); // no coinche
    }

    #[test]
    fn test_with_offset() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let offset = 50;
        let mut buf = vec![0.0f32; offset + BID_OBS_DIM];
        write_bid_observation(&mut buf, offset, &state, &[]);

        for i in 0..offset {
            assert_eq!(buf[i], 0.0, "pre-offset should be zero");
        }

        let hand_nonzero: usize = (0..32).filter(|&i| buf[offset + i] != 0.0).count();
        assert_eq!(hand_nonzero, 8);
    }

    #[test]
    fn test_bid_mask() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let mut buf = vec![0.0f32; BID_MASK_DIM];
        write_bid_mask(&mut buf, 0, &state);

        // PASS should always be legal
        assert_eq!(buf[0], 1.0);
        // All 36 regular bids should be legal (no prior bid)
        for i in 1..=36 {
            assert_eq!(buf[i], 1.0, "bid action {} should be legal", i);
        }
        // Capots should be legal
        for i in 37..=40 {
            assert_eq!(buf[i], 1.0, "capot action {} should be legal", i);
        }
        // Coinche/surcoinche should not be legal
        assert_eq!(buf[41], 0.0);
        assert_eq!(buf[42], 0.0);
    }

    #[test]
    fn test_bid_history_encoding() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);

        // Simulate: player 1 passes
        let bid_history = vec![(1u8, 0u8)];
        let obs = make_bid_observation(&state, &bid_history);

        // dealer=0, first_bidder=1, me=1 (current_player)
        // rel_offset = (1 + 4 - 1) % 4 = 0
        // slot 0 should have pass marker 0.2
        let bh_offset = 32;
        assert_eq!(obs[bh_offset], 0.2);
    }
}
