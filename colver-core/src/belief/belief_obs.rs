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

/// V2 observation dimension: 32 + 32 + 32 + 32 + 72 + 8 + 96 = 304.
/// Replaces V1's per-player played cards (128) + voids (12) + blocks 8-11 (14)
/// with compact played-by (32) + hard constraints (96).
pub const BELIEF_OBS_DIM_V2: usize = 304;

/// V3 observation dimension: V2 (304) + per-card lead suit (32) + per-trick winner (32) + suit failure counts (12) = 380.
/// Extends V2 with temporal features for richer play-history encoding.
pub const BELIEF_OBS_DIM_V3: usize = 380;

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

/// Write the 304-float V2 belief observation into `buf[offset..offset+BELIEF_OBS_DIM_V2]`.
///
/// V2 layout (304 floats):
///   Block 1 [0:32]:    Own hand (32) — binary card presence
///   Block 2 [32:64]:   Card played-by (32) — 0=unplayed, 0.25=me, 0.5=left, 0.75=partner, 1.0=right
///   Block 3 [64:96]:   Card trick index (32) — (trick_number+1)/8.0, 0 if unplayed
///   Block 4 [96:128]:  Card position-in-trick (32) — (pos+1)/4.0, 0 if unplayed
///   Block 5 [128:200]: Bid history (72) — 12 slots × 6 floats, player-relative
///   Block 6 [200:208]: Contract (8) — trump one-hot + value + taker + coinche
///   Block 7 [208:304]: Hard constraints (96) — 3 hidden players × 32 cards, 1.0=impossible
///
/// `hard_constraints` is a pre-computed 96-float array from the caller
/// (built by `TrumpCeilingTracker::compute_hard_constraints`).
pub fn write_belief_observation_v2(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    tracking: &EnvTracking,
    observer: u8,
    hard_constraints: &[f32; 96],
) {
    debug_assert!(buf.len() >= offset + BELIEF_OBS_DIM_V2);

    let out = &mut buf[offset..offset + BELIEF_OBS_DIM_V2];
    for v in out.iter_mut() {
        *v = 0.0;
    }

    let me = observer as usize;
    let my_team = me & 1;

    // Player-relative seats: [me, left_opp, partner, right_opp]
    let seats = [me, (me + 1) % 4, (me + 2) % 4, (me + 3) % 4];

    // Player-relative encoding values for played-by block
    // seat_val[absolute_player] = float value for that player relative to observer
    let mut seat_val = [0.0f32; 4];
    seat_val[seats[0]] = 0.25; // me
    seat_val[seats[1]] = 0.50; // left
    seat_val[seats[2]] = 0.75; // partner
    seat_val[seats[3]] = 1.00; // right

    let mut pos = 0;

    // === Block 1: My hand (32) ===
    let my_hand = state.hands[me];
    for i in 0..32u32 {
        if my_hand & (1 << i) != 0 {
            out[pos + i as usize] = 1.0;
        }
    }
    pos += 32;

    // === Block 2: Card played-by (32) ===
    // For each card, encode which player-relative seat played it.
    // 0.0 = unplayed (still in someone's hand), 0.25/0.5/0.75/1.0 = me/left/partner/right.
    // Covers both past tricks and current trick.
    for seat in 0..4usize {
        let played = tracking.played_by[seat];
        for i in 0..32u32 {
            if played & (1 << i) != 0 {
                out[pos + i as usize] = seat_val[seat];
            }
        }
    }
    // Current trick cards
    for seat in 0..4usize {
        let c = state.current_trick[seat];
        if c != card::EMPTY {
            out[pos + c as usize] = seat_val[seat];
        }
    }
    pos += 32;

    // === Block 3: Card trick index (32) ===
    for (i, &card_played) in tracking.play_order.iter().enumerate() {
        out[pos + card_played as usize] = (i / 4 + 1) as f32 / 8.0;
    }
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
        out[pos + 4] = state.contract.point_value() as f32 / 250.0;
        let taker_team = state.contract.team as usize;
        if taker_team == my_team {
            out[pos + 5] = 1.0;
        } else {
            out[pos + 6] = 1.0;
        }
        out[pos + 7] = state.contract.coinche as f32 / 2.0;
    }
    pos += 8;

    // === Block 7: Hard constraints (96) ===
    // 3 hidden players (left, partner, right) × 32 cards.
    // 1.0 = impossible (card in observer's hand, already played, void suit, trump ceiling).
    out[pos..pos + 96].copy_from_slice(hard_constraints);
    pos += 96;

    debug_assert_eq!(pos, BELIEF_OBS_DIM_V2);
}

/// Write the 380-float V3 belief observation into `buf[offset..offset+BELIEF_OBS_DIM_V3]`.
///
/// V3 layout: V2 (304 floats) + 3 new temporal blocks:
///   Block 8 [304:336]: Per-card lead suit — `(lead_suit + 1) / 5.0` for each played card, 0 if unplayed.
///   Block 9 [336:368]: Per-trick winner — 8 × 4 one-hot, relative seat that won each completed trick.
///   Block 10 [368:380]: Suit failure counts — 3 hidden players × 4 suits, `count / 8.0`.
///
/// `trick_leads`: lead suit index (0-3) for each completed trick.
/// `trick_winners`: absolute seat (0-3) that won each completed trick.
/// `suit_fail_counts`: `[3][4]` — for each hidden player (left, partner, right), count of times
///   they played a non-lead suit when that suit was led.
pub fn write_belief_observation_v3(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    tracking: &EnvTracking,
    observer: u8,
    hard_constraints: &[f32; 96],
    trick_leads: &[u8],
    trick_winners: &[u8],
    suit_fail_counts: &[[u8; 4]; 3],
) {
    debug_assert!(buf.len() >= offset + BELIEF_OBS_DIM_V3);

    // Write V2 base (first 304 floats)
    write_belief_observation_v2(buf, offset, state, tracking, observer, hard_constraints);

    let out = &mut buf[offset..offset + BELIEF_OBS_DIM_V3];
    let mut pos = 304;

    // === Block 8: Per-card lead suit (32) ===
    // For each played card, encode the lead suit of its trick as (lead_suit + 1) / 5.0.
    // Uses play_order to map each card to its trick index, then looks up trick_leads.
    for (i, &card_played) in tracking.play_order.iter().enumerate() {
        let trick_idx = i / 4;
        if trick_idx < trick_leads.len() {
            out[pos + card_played as usize] = (trick_leads[trick_idx] as f32 + 1.0) / 5.0;
        }
    }
    // Current trick cards: use current trick's lead suit
    if state.trick_count > 0 {
        let lead_card = state.current_trick[state.trick_lead as usize];
        if lead_card != card::EMPTY {
            let lead_suit = card::card_suit(lead_card) as u8;
            let current_trick_idx = tracking.play_order.len() / 4; // completed tricks count
            let val = (lead_suit as f32 + 1.0) / 5.0;
            // Encode all cards currently in the trick
            for seat in 0..4 {
                let c = state.current_trick[seat];
                if c != card::EMPTY {
                    out[pos + c as usize] = val;
                }
            }
            // Also set lead suit for trick_leads lookup consistency
            let _ = current_trick_idx; // used indirectly above
        }
    }
    pos += 32;

    // === Block 9: Per-trick winner (32) = 8 tricks × 4 seats (one-hot, relative) ===
    for (t, &winner_abs) in trick_winners.iter().enumerate() {
        if t >= 8 { break; }
        // Convert absolute winner to relative seat
        let winner_rel = ((winner_abs + 4 - observer) % 4) as usize;
        out[pos + t * 4 + winner_rel] = 1.0;
    }
    pos += 32;

    // === Block 10: Suit failure counts (12) = 3 hidden players × 4 suits ===
    // suit_fail_counts[i][s] = times hidden player i played non-lead when suit s was led
    for i in 0..3 {
        for s in 0..4 {
            out[pos + i * 4 + s] = suit_fail_counts[i][s] as f32 / 8.0;
        }
    }
    pos += 12;

    debug_assert_eq!(pos, BELIEF_OBS_DIM_V3);
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
