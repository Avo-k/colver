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

pub const BID_OBS_DIM: usize = 108;
pub const BID_OBS_DIM_SCORE_AWARE: usize = 110;
/// v5: 5 derived score features instead of 2 raw scores.
/// Layout: base 108 + my/2000 + opp/2000 + win_prob + leader_remaining/2000 + diff/2000
pub const BID_OBS_DIM_SCORE_AWARE_V2: usize = 113;
/// v6: v2 features + 4 self-belote bits (one per suit).
/// Layout: 113 + [self_has_QK_of_suit_0..3].
pub const BID_OBS_DIM_SCORE_AWARE_V3: usize = 117;
/// v7: v3 features + 4 per-suit trump scores + 2 auction-conditioned reductions.
/// Layout: 117 + [ts_suit_0..3] + [opp_best_other_ts, opp_second_other_ts].
pub const BID_OBS_DIM_V7: usize = 123;
pub const BID_MASK_DIM: usize = 43;

/// `evaluate_for_trump` tops out just under this; used to keep the obs in [0, 1].
const TS_SCALE: f32 = 35.0;

/// Calibrated match win probability: σ(1.7 × Δ / (R_sum^0.8 + 340))
/// Fitted from 10k full matches. Mirrors `bid_train_env::win_probability`.
#[inline]
pub fn win_probability(s_me: f32, s_opp: f32) -> f32 {
    let r_sum = (2000.0 - s_me) + (2000.0 - s_opp);
    let denom = r_sum.max(1.0).powf(0.8) + 340.0;
    let x = 1.7 * (s_me - s_opp) / denom;
    1.0 / (1.0 + (-x).exp())
}

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

    debug_assert_eq!(pos, BID_OBS_DIM);
}

/// Write the 110-float score-aware bidding observation.
/// Same as write_bid_observation but appends 2 floats: my_score/2000, opp_score/2000.
pub fn write_bid_observation_score_aware(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    bid_history: &[(u8, u8)],
    my_score: i32,
    opp_score: i32,
) {
    debug_assert!(buf.len() >= offset + BID_OBS_DIM_SCORE_AWARE);

    // Write the base 108-dim observation
    write_bid_observation(buf, offset, state, bid_history);

    // Append match scores (2 floats)
    buf[offset + BID_OBS_DIM] = (my_score as f32 / 2000.0).clamp(0.0, 1.0);
    buf[offset + BID_OBS_DIM + 1] = (opp_score as f32 / 2000.0).clamp(0.0, 1.0);
}

/// v2 score-aware obs (113-dim). Appends 5 derived features instead of 2 raw scores:
///   [108] my_score / 2000
///   [109] opp_score / 2000
///   [110] win_probability(my, opp) — calibrated sigmoid
///   [111] (2000 − max(my, opp)) / 2000 — distance from end for the leader
///   [112] (my − opp) / 2000 in [-1, 1]
pub fn write_bid_observation_score_aware_v2(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    bid_history: &[(u8, u8)],
    my_score: i32,
    opp_score: i32,
) {
    debug_assert!(buf.len() >= offset + BID_OBS_DIM_SCORE_AWARE_V2);

    write_bid_observation(buf, offset, state, bid_history);

    let s_me = my_score as f32;
    let s_opp = opp_score as f32;

    buf[offset + BID_OBS_DIM] = (s_me / 2000.0).clamp(0.0, 1.0);
    buf[offset + BID_OBS_DIM + 1] = (s_opp / 2000.0).clamp(0.0, 1.0);
    buf[offset + BID_OBS_DIM + 2] = win_probability(s_me, s_opp);
    let leader = s_me.max(s_opp);
    buf[offset + BID_OBS_DIM + 3] = ((2000.0 - leader) / 2000.0).clamp(0.0, 1.0);
    buf[offset + BID_OBS_DIM + 4] = ((s_me - s_opp) / 2000.0).clamp(-1.0, 1.0);
}

/// v3 score-aware obs (117-dim). v2 layout + 4 self-belote bits (one per suit).
/// Bit i = 1.0 iff the current player holds both Q and K of suit i in hand.
/// Belote in trump gives the declarer team +20, so exposing it at bid time lets
/// the model shift its aggression curve by ~one bid step on those 11% of deals.
pub fn write_bid_observation_score_aware_v3(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    bid_history: &[(u8, u8)],
    my_score: i32,
    opp_score: i32,
) {
    debug_assert!(buf.len() >= offset + BID_OBS_DIM_SCORE_AWARE_V3);

    write_bid_observation_score_aware_v2(buf, offset, state, bid_history, my_score, opp_score);

    let me = state.current_player() as usize;
    let hand = state.hands[me];
    for suit in 0..4u32 {
        // Q of suit = rank 4 (bit suit*8 + 4), K = rank 5.
        let qk_mask = (1u32 << (suit * 8 + 4)) | (1u32 << (suit * 8 + 5));
        let has_belote = (hand & qk_mask) == qk_mask;
        buf[offset + BID_OBS_DIM_SCORE_AWARE_V2 + suit as usize] =
            if has_belote { 1.0 } else { 0.0 };
    }
}

/// v7 obs (123-dim). v3 layout + 6 floats:
///   [117:121] `evaluate_for_trump(hand, s) / 35` for each suit — **suit-indexed**,
///             so it moves under a renaming like the hand block does.
///   [121]     `opp_best_other_ts`   — best of those, *excluding* the suit an
///   [122]     `opp_second_other_ts`   opponent currently holds the contract in.
///
/// The per-suit block is the net's own concept made explicit: the hidden-layer probe
/// found four parallel "suit quality detectors" in v5's last layer rather than a single
/// aggregate ([interpretability/probe_morning_report.md]).
///
/// The two reductions are the part that is *not* a restatement of the hand bitmap. They
/// are a max over suits **minus one named by the auction** — a hand × bid-history
/// interaction. Both are invariant under renaming (hand and excluded suit move
/// together), so they sit outside the permuted region by construction.
///
/// When no opponent holds the bid — opening, or our own side is contracting — nothing is
/// excluded and the two floats are the best and second best over all four suits.
pub fn write_bid_observation_v7(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    bid_history: &[(u8, u8)],
    my_score: i32,
    opp_score: i32,
) {
    debug_assert!(buf.len() >= offset + BID_OBS_DIM_V7);

    write_bid_observation_score_aware_v3(buf, offset, state, bid_history, my_score, opp_score);

    let me = state.current_player();
    let hand = state.hands[me as usize];

    let mut ts = [0.0f32; 4];
    for suit in 0..4u8 {
        ts[suit as usize] =
            crate::bid_eval::evaluate_for_trump(hand, crate::card::Suit::from_u8(suit)) as f32;
    }

    // Teams are the low bit of the seat, so an opponent is a seat of the other parity.
    let opp_holds_bid =
        state.last_bid_value > 0 && (state.last_bidder ^ me) & 1 == 1;
    let excluded = if opp_holds_bid {
        state.last_bid_suit as usize
    } else {
        usize::MAX
    };

    let (mut best, mut second) = (0.0f32, 0.0f32);
    for (suit, &v) in ts.iter().enumerate() {
        if suit == excluded {
            continue;
        }
        if v > best {
            second = best;
            best = v;
        } else if v > second {
            second = v;
        }
    }

    let base = offset + BID_OBS_DIM_SCORE_AWARE_V3;
    for (suit, &v) in ts.iter().enumerate() {
        buf[base + suit] = (v / TS_SCALE).clamp(0.0, 1.0);
    }
    buf[base + 4] = (best / TS_SCALE).clamp(0.0, 1.0);
    buf[base + 5] = (second / TS_SCALE).clamp(0.0, 1.0);
}

/// Write whichever observation layout `obs_dim` names. One place, so a new
/// consumer cannot silently pick a different dispatch than the trainer's.
pub fn write_bid_observation_dim(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    bid_history: &[(u8, u8)],
    my_score: i32,
    opp_score: i32,
    obs_dim: usize,
) {
    match obs_dim {
        BID_OBS_DIM_V7 => {
            write_bid_observation_v7(buf, offset, state, bid_history, my_score, opp_score)
        }
        BID_OBS_DIM_SCORE_AWARE_V3 => write_bid_observation_score_aware_v3(
            buf, offset, state, bid_history, my_score, opp_score,
        ),
        BID_OBS_DIM_SCORE_AWARE_V2 => write_bid_observation_score_aware_v2(
            buf, offset, state, bid_history, my_score, opp_score,
        ),
        BID_OBS_DIM_SCORE_AWARE => write_bid_observation_score_aware(
            buf, offset, state, bid_history, my_score, opp_score,
        ),
        _ => write_bid_observation(buf, offset, state, bid_history),
    }
}

/// Write the observation in **canonical suit order** and return `order`, the
/// mapping `order[canonical] = physical` needed to bring an action back out.
///
/// ## Why
///
/// Two hands identical up to renaming the suits are the same bidding problem, but
/// nothing in an MLP fed raw card bits enforces that. Measured on v6: **24.6% of
/// opening bids flip** under a suit renaming, because the symmetry noise is 8.8×
/// the top1−top2 margin (`bid_v7_plan.md` §1.1). Canonicalising divides the
/// effective input space by ~22 and makes the equivariance exact by construction
/// rather than something training has to rediscover 24 times.
///
/// ## Contract for callers
///
/// The returned `order` is not optional bookkeeping. A model trained on this layout
/// answers in canonical action space, so the caller **must**:
///
/// 1. map the legal mask into canonical space before selecting
///    (`permute_bid_mask_u64(legal, &perm_from_order(&order))`), and
/// 2. map the chosen action back with `permute_bid_action(a, &order)`.
///
/// Skip either and the model still returns a legal-looking bid — in the wrong suit.
/// This is the bid-side twin of the `cardset_to_canonical` / `card_to_physical`
/// warning that governs the 411-dim play observation.
pub fn write_bid_observation_canonical(
    buf: &mut [f32],
    offset: usize,
    state: &GameState,
    bid_history: &[(u8, u8)],
    my_score: i32,
    opp_score: i32,
    obs_dim: usize,
) -> [u8; 4] {
    write_bid_observation_dim(buf, offset, state, bid_history, my_score, opp_score, obs_dim);
    let me = state.current_player() as usize;
    let order = crate::suit_perm::canonical_bid_order(state.hands[me], bid_history);
    let perm = crate::suit_perm::perm_from_order(&order);
    crate::suit_perm::permute_bid_obs_dim(&mut buf[offset..offset + obs_dim], obs_dim, &perm);
    order
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
        assert_eq!(BID_OBS_DIM, 108);
        assert_eq!(BID_OBS_DIM_SCORE_AWARE, 110);
        assert_eq!(BID_OBS_DIM_SCORE_AWARE_V2, 113);
        assert_eq!(BID_OBS_DIM_SCORE_AWARE_V3, 117);
        assert_eq!(BID_MASK_DIM, 43);
    }

    #[test]
    fn test_v3_belote_features() {
        // Current player (first bidder after dealer=0 → seat 1) holds Q♥+K♥
        // plus 6 club fillers that deliberately avoid Q♣ and K♣ (ranks 4/5).
        let mut seat1_hand: u32 = 0;
        seat1_hand |= 1 << (1 * 8 + 4); // Q♥
        seat1_hand |= 1 << (1 * 8 + 5); // K♥
        // 6 club fillers: 7♣(24) 8♣(25) 9♣(26) J♣(27) 10♣(30) A♣(31). No Q♣ or K♣.
        seat1_hand |= (1 << 24) | (1 << 25) | (1 << 26) | (1 << 27) | (1 << 30) | (1 << 31);
        assert_eq!(seat1_hand.count_ones(), 8);
        let hands = [0u32, seat1_hand, 0u32, 0u32];
        let state = GameState::new(0, hands);
        assert_eq!(state.current_player(), 1);

        let mut buf = vec![0.0f32; BID_OBS_DIM_SCORE_AWARE_V3];
        write_bid_observation_score_aware_v3(&mut buf, 0, &state, &[], 0, 0);

        // Belote bits: only Hearts (suit 1) should be 1.0.
        assert_eq!(buf[BID_OBS_DIM_SCORE_AWARE_V2], 0.0);       // spades
        assert_eq!(buf[BID_OBS_DIM_SCORE_AWARE_V2 + 1], 1.0);   // hearts ← belote
        assert_eq!(buf[BID_OBS_DIM_SCORE_AWARE_V2 + 2], 0.0);   // diamonds
        assert_eq!(buf[BID_OBS_DIM_SCORE_AWARE_V2 + 3], 0.0);   // clubs
    }

    #[test]
    fn test_v2_features_layout() {
        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let mut buf = vec![0.0f32; BID_OBS_DIM_SCORE_AWARE_V2];
        write_bid_observation_score_aware_v2(&mut buf, 0, &state, &[], 1500, 800);

        // base 108-dim should be identical to standard obs
        let mut ref_buf = vec![0.0f32; BID_OBS_DIM];
        write_bid_observation(&mut ref_buf, 0, &state, &[]);
        for i in 0..BID_OBS_DIM {
            assert_eq!(buf[i], ref_buf[i]);
        }

        assert!((buf[108] - 0.75).abs() < 1e-6);   // 1500/2000
        assert!((buf[109] - 0.40).abs() < 1e-6);   // 800/2000
        assert!(buf[110] > 0.5 && buf[110] < 1.0); // win_prob > 0.5 (we're ahead)
        assert!((buf[111] - 0.25).abs() < 1e-6);   // (2000-1500)/2000
        assert!((buf[112] - 0.35).abs() < 1e-6);   // (1500-800)/2000
    }

    /// Deal 32 cards deterministically from a seed, so a test can build a real
    /// auction without pulling in `rand`.
    fn deal(seed: u64) -> [u32; 4] {
        let mut cards: Vec<u8> = (0..32).collect();
        let mut s = seed | 1;
        for i in (1..32).rev() {
            s = s.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
            let j = (s >> 33) as usize % (i + 1);
            cards.swap(i, j);
        }
        let mut hands = [0u32; 4];
        for (i, &c) in cards.iter().enumerate() {
            hands[i / 8] |= 1 << c;
        }
        hands
    }

    /// The property §3.1 buys: renaming the suits of an entire position — the four
    /// hands *and* every action already bid — must leave the canonical observation
    /// bit-identical. That is what lets one training sample teach all 24 relabelings.
    #[test]
    fn canonical_bid_obs_is_invariant_under_suit_renaming() {
        use crate::suit_perm::{permute_bid_action, ALL_PERMS};

        for seed in 0..40u64 {
            let hands = deal(seed);
            for perm in ALL_PERMS.iter() {
                // The same deal with the suits renamed.
                let mut permuted = [0u32; 4];
                for (seat, &h) in hands.iter().enumerate() {
                    for s in 0..4u32 {
                        let lane = (h >> (s * 8)) & 0xFF;
                        permuted[seat] |= lane << (perm[s as usize] as u32 * 8);
                    }
                }

                // An auction that names suits, so the history block is exercised too.
                let script = [
                    crate::bidding::encode_bid(10, 1), // 100♥
                    0,                                 // pass
                    crate::bidding::encode_bid(12, 3), // 120♣
                ];

                let mut a = GameState::new(0, hands);
                let mut b = GameState::new(0, permuted);
                let (mut ha, mut hb) = (Vec::new(), Vec::new());
                for &act in script.iter() {
                    let pa = act;
                    let pb = permute_bid_action(act, perm);
                    assert!(a.legal_actions() & (1u64 << pa) != 0);
                    assert!(b.legal_actions() & (1u64 << pb) != 0);
                    ha.push((a.current_player(), pa));
                    hb.push((b.current_player(), pb));
                    a.step(pa);
                    b.step(pb);
                }

                // Every width, so a new suit-indexed tail cannot be added without
                // being wired into `permute_bid_obs_dim`.
                let mut order_a = [0u8; 4];
                let mut order_b = [0u8; 4];
                for dim in [BID_OBS_DIM_SCORE_AWARE_V3, BID_OBS_DIM_V7] {
                    let mut oa = vec![0.0f32; dim];
                    let mut ob = vec![0.0f32; dim];
                    order_a = write_bid_observation_canonical(&mut oa, 0, &a, &ha, 700, 400, dim);
                    order_b = write_bid_observation_canonical(&mut ob, 0, &b, &hb, 700, 400, dim);
                    assert_eq!(
                        oa, ob,
                        "seed {seed}, perm {perm:?}, dim {dim}: obs differs after renaming"
                    );
                }

                // The two decode a canonical bid back to the *same* suit — up to a
                // tie. When two suits hold identical lanes the sort breaks the tie by
                // physical index, which renaming changes, so A and B can pick the
                // other member of the pair. That is not an error: the obs is
                // bit-identical precisely because the hand cannot tell them apart.
                let me = a.current_player() as usize;
                let hb = b.hands[me];
                let lane = |s: u8| (hb >> (s * 8)) & 0xFF;
                for c in 0..4usize {
                    let via_a = perm[order_a[c] as usize];
                    let direct = order_b[c];
                    assert_eq!(
                        lane(via_a),
                        lane(direct),
                        "seed {seed}, perm {perm:?}: canonical slot {c} decodes to \
                         suits that are not interchangeable ({via_a} vs {direct})"
                    );
                }
            }
        }
    }

    /// The v7 tail excludes the suit an **opponent** is contracting in, and only that
    /// case. The per-suit scores restate the hand, so the exclusion is the one thing in
    /// §3.4 the net could not already read off `obs[0..32]` — if it silently degraded to
    /// "best over all four", v7 would be v6 plus six redundant floats.
    #[test]
    fn v7_tail_excludes_only_the_opponents_suit() {
        let ts = |state: &GameState| {
            let mut o = vec![0.0f32; BID_OBS_DIM_V7];
            write_bid_observation_v7(&mut o, 0, state, &[], 0, 0);
            let base = BID_OBS_DIM_SCORE_AWARE_V3;
            (
                [o[base], o[base + 1], o[base + 2], o[base + 3]],
                o[base + 4],
                o[base + 5],
            )
        };

        for seed in 0..48u64 {
            let hands = deal(seed);
            let opening = GameState::new(0, hands);
            let (per_suit, best, second) = ts(&opening);

            // Nobody has bid: no exclusion, so the two reductions are just the top two.
            let mut sorted = per_suit;
            sorted.sort_by(|a, b| b.partial_cmp(a).unwrap());
            assert_eq!(best, sorted[0], "seed {seed}: opening best");
            assert_eq!(second, sorted[1], "seed {seed}: opening second");

            // The opener bids 100♥; the next seat to speak is an opponent, so ♥ drops out.
            let opener = opening.current_player();
            let mut opp = opening;
            opp.step(crate::bidding::encode_bid(10, 1));
            assert_eq!((opp.current_player() ^ opener) & 1, 1, "next seat is an opponent");
            let (per_suit_o, best_o, second_o) = ts(&opp);
            let mut others: Vec<f32> = (0..4).filter(|&s| s != 1).map(|s| per_suit_o[s]).collect();
            others.sort_by(|a, b| b.partial_cmp(a).unwrap());
            assert_eq!(best_o, others[0], "seed {seed}: ♥ should be excluded");
            assert_eq!(second_o, others[1], "seed {seed}: ♥ should be excluded");

            // One more pass and the speaker is the opener's *partner* — same suit on the
            // table, but it is now their own side's, so nothing is excluded.
            let mut partner = opp;
            partner.step(0);
            assert_eq!(partner.current_player(), opener ^ 2, "speaker is the opener's partner");
            let (per_suit_p, best_p, second_p) = ts(&partner);
            let mut all = per_suit_p;
            all.sort_by(|a, b| b.partial_cmp(a).unwrap());
            assert_eq!(best_p, all[0], "seed {seed}: partner's suit must not be excluded");
            assert_eq!(second_p, all[1], "seed {seed}: partner's suit must not be excluded");
        }
    }

    /// `order` and `perm` are inverses, and the canonical→physical→canonical round
    /// trip is the identity on all 43 actions. Confusing the two is *the* canonical
    /// obs bug: everything stays legal and the model answers about another suit.
    #[test]
    fn canonical_order_and_perm_are_inverse() {
        use crate::suit_perm::{canonical_bid_order, perm_from_order, permute_bid_action};
        for seed in 0..64u64 {
            let hand = deal(seed)[0];
            let order = canonical_bid_order(hand, &[]);
            let perm = perm_from_order(&order);
            for s in 0..4usize {
                assert_eq!(perm[order[s] as usize] as usize, s);
            }
            for a in 0..43u8 {
                assert_eq!(permute_bid_action(permute_bid_action(a, &order), &perm), a);
            }
            // Canonical lanes are sorted by (count, pattern), descending.
            let key = |s: u8| {
                let lane = (hand >> (s * 8)) & 0xFF;
                (lane.count_ones() << 8) | lane
            };
            for i in 0..3 {
                assert!(key(order[i]) >= key(order[i + 1]), "order not sorted for {hand:#x}");
            }
        }
    }

    /// The minimal case that broke the first implementation, kept as an anchor.
    ///
    /// My ♠ and ♥ lanes are identical, so the hand alone cannot order them and the
    /// sort falls to a tie-break. An opponent has bid one of the two. Renaming ♠↔♥
    /// moves that bid to the other suit, so a tie-break on physical index sends the
    /// mention to a different canonical slot and the two positions — which are the
    /// same problem — stop canonicalising to the same observation.
    #[test]
    fn canonical_bid_obs_tie_broken_by_the_auction() {
        use crate::suit_perm::canonical_bid_order;

        let lane: u32 = 0b1000_1010; // A, J, 8 — three cards
        // ♠ and ♥ identical, ♦ holds two more so the hand has 8 cards.
        let hand = lane | (lane << 8) | (0b0000_0011u32 << 16);
        assert_eq!(hand.count_ones(), 8);

        let swap = |h: u32| ((h & 0xFF) << 8) | ((h >> 8) & 0xFF) | (h & 0xFFFF_0000);
        assert_eq!(swap(hand), hand, "the hand itself is symmetric under ♠↔♥");

        let dim = BID_OBS_DIM_SCORE_AWARE_V3;
        let bid_s = crate::bidding::encode_bid(10, 0); // 100♠
        let bid_h = crate::bidding::encode_bid(10, 1); // 100♥

        // Same hand, same dealer; the opponent names ♠ in one line and ♥ in the other.
        let mut out = Vec::new();
        for &(opening, other) in &[(bid_s, bid_h), (bid_h, bid_s)] {
            let mut st = GameState::new(2, [hand, 0, 0, 0]); // dealer 2 → seat 3 speaks
            let hist = vec![(st.current_player(), opening)];
            st.step(opening);
            assert_eq!(st.current_player(), 0, "we speak next");

            let order = canonical_bid_order(hand, &hist);
            // The suit that was actually bid must land in the same canonical slot
            // in both lines — that is the whole point of the auction tie-break.
            let bid_suit = crate::bidding::decode_bid(opening).1;
            let slot = order.iter().position(|&s| s == bid_suit).unwrap();
            let mut obs = vec![0.0f32; dim];
            write_bid_observation_canonical(&mut obs, 0, &st, &hist, 0, 0, dim);
            out.push((slot, obs, other));
        }
        assert_eq!(out[0].0, out[1].0, "the bid suit lands in different canonical slots");
        assert_eq!(out[0].1, out[1].1, "the two symmetric positions differ");
    }

    #[test]
    fn test_win_probability_symmetry() {
        let p_ahead = win_probability(1500.0, 800.0);
        let p_behind = win_probability(800.0, 1500.0);
        assert!((p_ahead + p_behind - 1.0).abs() < 1e-5);
        assert!(p_ahead > 0.5);
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
