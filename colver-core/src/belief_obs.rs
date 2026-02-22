/// Belief observation builder: constructs the 330-float observation vector
/// for the NN belief network (card location prediction).
///
/// Zero-allocation pattern: writes into caller-provided buffer at offset.
/// Reuses `EnvTracking` from `dmc_obs.rs` for play/bid history tracking.
///
/// Layout (330 floats):
///   Block 1:  Own hand (32)                — binary card presence
///   Block 2:  Per-player played cards (128) — 4 × 32, player-relative [me, left, partner, right], includes current trick
///   Block 3:  Card trick index (32)        — (trick_number+1)/8.0 per played card, 0 if unplayed
///   Block 4:  Card position-in-trick (32)  — (position_in_trick+1)/4.0 per played card, 0 if unplayed
///   Block 5:  Bid history (72)             — 12 slots × 6 floats, player-relative
///   Block 6:  Contract (8)                 — trump one-hot(4) + bid_value/250(1) + taker team one-hot(2) + coinche/2(1)
///   Block 7:  Known voids (12)             — 3 hidden players × 4 suits (player-relative)
///   Block 8:  Scoring context (4)          — my_pts/252, opp_pts/252, my_tricks/8, opp_tricks/8
///   Block 9:  Dealer-relative position (4) — one-hot seat
///   Block 10: Current trick lead suit (4)  — one-hot (zero if no current trick)
///   Block 11: Trick progress (2)           — trick_number/8, cards_in_current_trick/4

use crate::bidding;
use crate::card;
use crate::dmc_obs::EnvTracking;
use crate::state::{GameState, Phase};

pub const BELIEF_OBS_DIM: usize = 330;

/// Write the 330-float belief observation into `buf[offset..offset+BELIEF_OBS_DIM]`.
///
/// `observer` is the player whose perspective we encode (may differ from current_player
/// since we want to predict card locations from any player's view at any game point).
///
/// # Panics
/// Panics if `buf` is too small to hold `offset + BELIEF_OBS_DIM` floats.
pub fn write_belief_observation(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    tracking: &EnvTracking,
    observer: u8,
) {
    debug_assert!(buf.len() >= offset + BELIEF_OBS_DIM);

    let out = &mut buf[offset..offset + BELIEF_OBS_DIM];
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let me = observer as usize;
    let my_team = me & 1;
    let opp_team = 1 - my_team;

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

    // === Block 2: Per-player played cards including current trick (128) ===
    // Current trick union
    let mut trick_cards = [0u32; 4];
    for i in 0..4 {
        let c = state.current_trick[i];
        if c != card::EMPTY {
            trick_cards[i] = 1u32 << c;
        }
    }

    for &seat in &seats {
        // Past played cards + current trick card for this seat
        let played = tracking.played_by[seat] | trick_cards[seat];
        for i in 0..32u32 {
            if played & (1 << i) != 0 {
                out[pos + i as usize] = 1.0;
            }
        }
        pos += 32;
    }

    // === Block 3: Card trick index (32) ===
    // For played cards in past tricks, use play_order to determine trick number
    for (i, &card_played) in tracking.play_order.iter().enumerate() {
        out[pos + card_played as usize] = (i / 4 + 1) as f32 / 8.0;
    }
    // For current trick cards, use current trick number (tricks completed so far + 1)
    let current_trick_num = (tracking.play_order.len() / 4) + 1;
    for i in 0..4 {
        let c = state.current_trick[i];
        if c != card::EMPTY {
            out[pos + c as usize] = current_trick_num as f32 / 8.0;
        }
    }
    pos += 32;

    // === Block 4: Card position-in-trick (32) ===
    for (i, &card_played) in tracking.play_order.iter().enumerate() {
        out[pos + card_played as usize] = (i % 4 + 1) as f32 / 4.0;
    }
    // Current trick positions
    let trick_lead = state.trick_lead as usize;
    for j in 0..4usize {
        let seat = (trick_lead + j) % 4;
        let c = state.current_trick[seat];
        if c != card::EMPTY {
            out[pos + c as usize] = (j + 1) as f32 / 4.0;
        }
    }
    pos += 32;

    // === Block 5: Bid history (72) ===
    encode_bid_history(out, pos, &tracking.bid_history, me, tracking.dealer);
    pos += 72;

    // === Block 6: Contract (8) ===
    if state.phase != Phase::Bidding {
        let trump = state.contract.trump as usize;
        out[pos + trump] = 1.0;
        // pos + 4: bid_value / 250
        out[pos + 4] = state.contract.point_value() as f32 / 250.0;
        // pos + 5..6: taker team one-hot (player-relative: 0=my team, 1=opp team)
        let taker_team = state.contract.team as usize;
        if taker_team == my_team {
            out[pos + 5] = 1.0;
        } else {
            out[pos + 6] = 1.0;
        }
        // pos + 7: coinche / 2
        out[pos + 7] = state.contract.coinche as f32 / 2.0;
    }
    pos += 8;

    // === Block 7: Known voids (12) ===
    // 3 hidden players (left, partner, right) × 4 suits
    for &seat in &seats[1..] {
        for s in 0..4u8 {
            if state.voids[seat] & (1 << s) != 0 {
                out[pos] = 1.0;
            }
            pos += 1;
        }
    }

    // === Block 8: Scoring context (4) ===
    out[pos] = state.points[my_team] as f32 / 252.0;
    out[pos + 1] = state.points[opp_team] as f32 / 252.0;
    out[pos + 2] = state.tricks_won[my_team] as f32 / 8.0;
    out[pos + 3] = state.tricks_won[opp_team] as f32 / 8.0;
    pos += 4;

    // === Block 9: Dealer-relative position (4) ===
    let rel_pos = (me + 4 - state.dealer as usize) % 4;
    out[pos + rel_pos] = 1.0;
    pos += 4;

    // === Block 10: Current trick lead suit (4) ===
    if state.trick_count > 0 {
        // Find the lead card to determine suit
        let lead_card = state.current_trick[state.trick_lead as usize];
        if lead_card != card::EMPTY {
            let suit = (lead_card / 8) as usize;
            out[pos + suit] = 1.0;
        }
    }
    pos += 4;

    // === Block 11: Trick progress (2) ===
    let completed_tricks = tracking.play_order.len() / 4;
    out[pos] = completed_tricks as f32 / 8.0;
    out[pos + 1] = state.trick_count as f32 / 4.0;
    pos += 2;

    debug_assert_eq!(pos, BELIEF_OBS_DIM);
}

/// Convenience wrapper that allocates and returns a Vec<f32>.
pub fn make_belief_observation(
    state: &GameState,
    tracking: &EnvTracking,
    observer: u8,
) -> Vec<f32> {
    let mut buf = vec![0.0f32; BELIEF_OBS_DIM];
    write_belief_observation(&mut buf, 0, state, tracking, observer);
    buf
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dmc_obs::EnvTracking;
    use crate::state::GameState;

    #[test]
    fn test_belief_obs_dim() {
        assert_eq!(BELIEF_OBS_DIM, 330);
    }

    #[test]
    fn test_write_belief_observation_length() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();
        let mut buf = vec![0.0f32; BELIEF_OBS_DIM];
        write_belief_observation(&mut buf, 0, &state, &tracking, 1);
        assert_eq!(buf.len(), BELIEF_OBS_DIM);
    }

    #[test]
    fn test_hand_encoding() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();
        let obs = make_belief_observation(&state, &tracking, 1);

        // Observer=1, hand=0xFF00 (bits 8-15)
        for i in 0..32 {
            let expected = if (0xFF00u32 >> i) & 1 != 0 { 1.0 } else { 0.0 };
            assert_eq!(obs[i], expected, "hand bit {} mismatch", i);
        }
    }

    #[test]
    fn test_observer_independent_of_current_player() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();

        // Observer=0 should see hand 0xFF
        let obs0 = make_belief_observation(&state, &tracking, 0);
        let hand0_ones: usize = (0..32).filter(|&i| obs0[i] != 0.0).count();
        assert_eq!(hand0_ones, 8);
        assert_eq!(obs0[0], 1.0); // bit 0 of 0xFF

        // Observer=2 should see hand 0xFF_0000
        let obs2 = make_belief_observation(&state, &tracking, 2);
        let hand2_ones: usize = (0..32).filter(|&i| obs2[i] != 0.0).count();
        assert_eq!(hand2_ones, 8);
        assert_eq!(obs2[16], 1.0); // bit 16 of 0xFF_0000
    }

    #[test]
    fn test_with_offset() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();
        let offset = 50;
        let mut buf = vec![0.0f32; offset + BELIEF_OBS_DIM];
        write_belief_observation(&mut buf, offset, &state, &tracking, 0);

        for i in 0..offset {
            assert_eq!(buf[i], 0.0, "pre-offset should be zero");
        }
        let hand_nonzero: usize = (0..32).filter(|&i| buf[offset + i] != 0.0).count();
        assert_eq!(hand_nonzero, 8);
    }

    #[test]
    fn test_dealer_relative_position() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();

        // Observer=2, dealer=0 → rel_pos = (2+4-0)%4 = 2
        let obs = make_belief_observation(&state, &tracking, 2);
        let pos_offset = 32 + 128 + 32 + 32 + 72 + 8 + 12 + 4; // = 320
        assert_eq!(obs[pos_offset + 0], 0.0);
        assert_eq!(obs[pos_offset + 1], 0.0);
        assert_eq!(obs[pos_offset + 2], 1.0); // position 2
        assert_eq!(obs[pos_offset + 3], 0.0);
    }

    #[test]
    fn test_trick_progress_initial() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let tracking = EnvTracking::new();
        let obs = make_belief_observation(&state, &tracking, 0);

        // Last 2 floats: trick_number/8=0, trick_count/4=0
        assert_eq!(obs[BELIEF_OBS_DIM - 2], 0.0);
        assert_eq!(obs[BELIEF_OBS_DIM - 1], 0.0);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_observation_during_play() {
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

                if state.phase == Phase::Playing {
                    // Build observation from current player's perspective
                    let observer = state.current_player();
                    let obs = make_belief_observation(&state, &tracking, observer);
                    assert_eq!(obs.len(), BELIEF_OBS_DIM);

                    // Hand bits should match current player's hand
                    let hand = state.hands[observer as usize];
                    let hand_ones: usize = (0..32).filter(|&i| obs[i] != 0.0).count();
                    let expected = hand.count_ones() as usize;
                    assert_eq!(hand_ones, expected, "hand card count mismatch");
                }

                tracking.track_action(&state, action);
                state.step(action);
            }
        }
    }
}
