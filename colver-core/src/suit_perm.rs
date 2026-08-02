/// Suit permutation utilities for data augmentation.
///
/// All 4 suits in Belote are interchangeable within the observation encoding,
/// so any game can be replayed with suits relabeled (4! = 24 permutations),
/// giving 24x data diversity for free.

/// All 24 permutations of 4 suits.
/// `perm[s]` = which output suit lane suit `s` maps to.
///
/// First 6 entries fix slot 0 (for trump-relative augmentation).
pub const ALL_PERMS: [[u8; 4]; 24] = [
    // --- 6 perms that fix slot 0 (for trump-relative encoding) ---
    [0, 1, 2, 3],
    [0, 1, 3, 2],
    [0, 2, 1, 3],
    [0, 2, 3, 1],
    [0, 3, 1, 2],
    [0, 3, 2, 1],
    [1, 0, 2, 3],
    [1, 0, 3, 2],
    [1, 2, 0, 3],
    [1, 2, 3, 0],
    [1, 3, 0, 2],
    [1, 3, 2, 0],
    [2, 0, 1, 3],
    [2, 0, 3, 1],
    [2, 1, 0, 3],
    [2, 1, 3, 0],
    [2, 3, 0, 1],
    [2, 3, 1, 0],
    [3, 0, 1, 2],
    [3, 0, 2, 1],
    [3, 1, 0, 2],
    [3, 1, 2, 0],
    [3, 2, 0, 1],
    [3, 2, 1, 0],
];

/// Permute a 32-float card-indexed block by swapping 8-float suit lanes.
///
/// Card layout: [Spades(8), Hearts(8), Diamonds(8), Clubs(8)].
/// After permutation, suit `s` data moves to lane `perm[s]`.
#[inline]
pub fn permute_card_block(block: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(block.len() >= 32);
    let mut tmp = [0.0f32; 32];
    tmp.copy_from_slice(&block[..32]);
    for s in 0..4 {
        let dst = perm[s] as usize;
        block[dst * 8..(dst + 1) * 8].copy_from_slice(&tmp[s * 8..(s + 1) * 8]);
    }
}

/// Permute a 4-float suit one-hot by swapping positions.
#[inline]
pub fn permute_suit_onehot(block: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(block.len() >= 4);
    let tmp = [block[0], block[1], block[2], block[3]];
    for s in 0..4 {
        block[perm[s] as usize] = tmp[s];
    }
}

/// Permute a 330-float V1 belief observation in-place.
pub fn permute_belief_obs_v1(obs: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(obs.len() >= 330);

    // Block 1 [0:32]: own hand
    permute_card_block(&mut obs[0..32], perm);

    // Block 2 [32:160]: 4x per-player played cards
    for i in 0..4 {
        let start = 32 + i * 32;
        permute_card_block(&mut obs[start..start + 32], perm);
    }

    // Block 3 [160:192]: card trick index
    permute_card_block(&mut obs[160..192], perm);

    // Block 4 [192:224]: card position-in-trick
    permute_card_block(&mut obs[192..224], perm);

    // Block 5 [224:296]: bid history, 12 slots x 6 floats
    // Suit one-hot at offset +2..+6 per slot
    for slot in 0..12 {
        let base = 224 + slot * 6;
        permute_suit_onehot(&mut obs[base + 2..base + 6], perm);
    }

    // Block 6 [296:304]: contract, trump one-hot at [296:300]
    permute_suit_onehot(&mut obs[296..300], perm);

    // Block 7 [304:316]: known voids, 3x suit one-hot (4 floats each)
    permute_suit_onehot(&mut obs[304..308], perm);
    permute_suit_onehot(&mut obs[308..312], perm);
    permute_suit_onehot(&mut obs[312..316], perm);

    // Block 8 [316:320]: scoring context -- no permutation
    // Block 9 [320:324]: dealer-relative position -- no permutation

    // Block 10 [324:328]: current trick lead suit one-hot
    permute_suit_onehot(&mut obs[324..328], perm);

    // Block 11 [328:330]: trick progress -- no permutation
}

/// Permute a 304-float V2 belief observation in-place.
///
/// V2 layout:
///   Block 1 [0:32]:    own hand — card block
///   Block 2 [32:64]:   card played-by — card block
///   Block 3 [64:96]:   card trick index — card block
///   Block 4 [96:128]:  card position-in-trick — card block
///   Block 5 [128:200]: bid history — 12 slots × 6, suit one-hot at +2..+6
///   Block 6 [200:208]: contract — trump one-hot at [200:204]
///   Block 7 [208:304]: hard constraints — 3 × card block (32 each)
pub fn permute_belief_obs_v2(obs: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(obs.len() >= 304);

    // Blocks 1-4 [0:128]: 4 card-indexed blocks
    permute_card_block(&mut obs[0..32], perm);
    permute_card_block(&mut obs[32..64], perm);
    permute_card_block(&mut obs[64..96], perm);
    permute_card_block(&mut obs[96..128], perm);

    // Block 5 [128:200]: bid history, suit one-hot at +2..+6 per slot
    for slot in 0..12 {
        let base = 128 + slot * 6;
        permute_suit_onehot(&mut obs[base + 2..base + 6], perm);
    }

    // Block 6 [200:204]: trump suit one-hot
    permute_suit_onehot(&mut obs[200..204], perm);

    // Block 7 [208:304]: hard constraints, 3 × card block (32 each)
    permute_card_block(&mut obs[208..240], perm);
    permute_card_block(&mut obs[240..272], perm);
    permute_card_block(&mut obs[272..304], perm);
}

/// Permute a 380-float V3 belief observation in-place.
///
/// V3 layout: V2 (304) + 3 new blocks:
///   Block 8 [304:336]: per-card lead suit — card block with encoded suit values
///   Block 9 [336:368]: per-trick winner — 8 × 4, player-relative (no suit permutation needed)
///   Block 10 [368:380]: suit failure counts — 3 × 4 suit-indexed values
pub fn permute_belief_obs_v3(obs: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(obs.len() >= 380);

    // First 304: V2 permutation
    permute_belief_obs_v2(obs, perm);

    // Block 8 [304:336]: per-card lead suit values
    // First permute card positions (which card slots to swap)
    permute_card_block(&mut obs[304..336], perm);
    // Then remap the lead suit encoding: val = (old_suit + 1) / 5.0
    // → old_suit = round(val * 5) - 1 → new_suit = perm[old_suit] → new_val = (new_suit + 1) / 5.0
    for i in 304..336 {
        let val = obs[i];
        if val > 0.0 {
            let old_suit = (val * 5.0).round() as usize - 1;
            if old_suit < 4 {
                let new_suit = perm[old_suit] as usize;
                obs[i] = (new_suit as f32 + 1.0) / 5.0;
            }
        }
    }

    // Block 9 [336:368]: per-trick winner one-hot — player-relative, no suit permutation needed

    // Block 10 [368:380]: suit failure counts — 3 groups of 4 suit-indexed values
    permute_suit_onehot(&mut obs[368..372], perm);
    permute_suit_onehot(&mut obs[372..376], perm);
    permute_suit_onehot(&mut obs[376..380], perm);
}

/// Permute a 32-element target array (card -> player mapping) by swapping suit lanes.
pub fn permute_target(target: &mut [u8; 32], perm: &[u8; 4]) {
    let mut tmp = [0u8; 32];
    tmp.copy_from_slice(target);
    for s in 0..4 {
        let dst = perm[s] as usize;
        for r in 0..8 {
            target[dst * 8 + r] = tmp[s * 8 + r];
        }
    }
}

/// Permute a 32-bit card mask by swapping 8-bit suit lanes.
pub fn permute_mask(mask: u32, perm: &[u8; 4]) -> u32 {
    let mut result = 0u32;
    for s in 0..4u32 {
        let lane = (mask >> (s * 8)) & 0xFF;
        result |= lane << (perm[s as usize] as u32 * 8);
    }
    result
}

/// Permute a 415-float DMC play observation in-place.
///
/// Layout (see `dmc_obs.rs`):
///   [0:32]    My hand — card block
///   [32:160]  Current trick (4×32) — card blocks
///   [160:256] Per-player played (3×32) — card blocks
///   [256:260] Contract trump — suit one-hot
///   [260:263] bid_value, is_my_team, coinche — no perm
///   [263:275] Void tracking — 3×4 suit-indexed
///   [275:279] Scoring context — no perm
///   [279:351] Bid history — 12 slots × 6, suit one-hot at +2..+6
///   [351:383] Card trick index — card block
///   [383:415] Card sequence index — card block
pub fn permute_dmc_obs(obs: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(obs.len() >= 415);

    // Block 1 [0:32]: My hand
    permute_card_block(&mut obs[0..32], perm);

    // Block 2 [32:160]: Current trick — 4 card blocks
    for i in 0..4 {
        let start = 32 + i * 32;
        permute_card_block(&mut obs[start..start + 32], perm);
    }

    // Block 3 [160:256]: Per-player played — 3 card blocks
    for i in 0..3 {
        let start = 160 + i * 32;
        permute_card_block(&mut obs[start..start + 32], perm);
    }

    // Block 4 [256:260]: Contract trump suit one-hot
    permute_suit_onehot(&mut obs[256..260], perm);
    // [260:263]: bid_value, is_my_team, coinche — unchanged

    // Block 5 [263:275]: Void tracking — 3 groups of 4 suit-indexed
    permute_suit_onehot(&mut obs[263..267], perm);
    permute_suit_onehot(&mut obs[267..271], perm);
    permute_suit_onehot(&mut obs[271..275], perm);

    // Block 6 [275:279]: Scoring context — unchanged

    // Block 7 [279:351]: Bid history — 12 slots × 6
    for slot in 0..12 {
        let base = 279 + slot * 6;
        permute_suit_onehot(&mut obs[base + 2..base + 6], perm);
    }

    // Block 8 [351:383]: Card trick index — card block
    permute_card_block(&mut obs[351..383], perm);

    // Block 9 [383:415]: Card sequence index — card block
    permute_card_block(&mut obs[383..415], perm);
}

/// Permute a 114-float bid observation in-place.
///
/// Layout (see `bid_obs.rs`):
///   [0:32]    My hand — card block
///   [32:104]  Bid history — 12 slots × 6, suit one-hot at +2..+6
///   [104:108] Dealer-relative position — no perm
///   [108]     Bid value — no perm
///   [109:113] Bid suit one-hot — suit one-hot
///   [113]     Coinche state — no perm
pub fn permute_bid_obs(obs: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(obs.len() >= 108);

    // Block 1 [0:32]: My hand
    permute_card_block(&mut obs[0..32], perm);

    // Block 2 [32:104]: Bid history — 12 slots × 6
    for slot in 0..12 {
        let base = 32 + slot * 6;
        permute_suit_onehot(&mut obs[base + 2..base + 6], perm);
    }

    // Block 3 [104:108]: Position — unchanged
}

/// Permute the suit-dependent parts of a bid observation of any known width.
///
/// `permute_bid_obs` only covers the base 108. The score-aware tails differ:
/// [108..113] are match-score scalars and suit-invariant, but the v3 belote bits
/// at [113..117] are one per suit and **must** move with the rest. Forgetting them
/// is silent — the obs stays well-formed and merely lies about which suit carries
/// the belote.
pub fn permute_bid_obs_dim(obs: &mut [f32], obs_dim: usize, perm: &[u8; 4]) {
    debug_assert!(obs.len() >= obs_dim);
    permute_bid_obs(&mut obs[..crate::bid_obs::BID_OBS_DIM], perm);
    // Every suit-indexed block past the base obs, in layout order. The two v7 tails
    // ([121], [122]) are reductions over suits and are invariant, so they stay put.
    if obs_dim >= crate::bid_obs::BID_OBS_DIM_SCORE_AWARE_V3 {
        let off = crate::bid_obs::BID_OBS_DIM_SCORE_AWARE_V2; // belote bits
        permute_suit_onehot(&mut obs[off..off + 4], perm);
    }
    if obs_dim >= crate::bid_obs::BID_OBS_DIM_V7 {
        let off = crate::bid_obs::BID_OBS_DIM_SCORE_AWARE_V3; // per-suit trump scores
        permute_suit_onehot(&mut obs[off..off + 4], perm);
    }
}

/// Canonical suit ordering for the **bidding** observation — `order[canon] = phys`.
///
/// Unlike [`crate::dmc_obs::canonical_play_order`] there is no trump to anchor slot 0:
/// a bid is made before any trump exists. Suits are therefore sorted by (card count,
/// rank pattern) descending — a pure function of the hand.
///
/// Anchoring the *primary* key on the hand rather than on the auction is deliberate:
/// an auction-anchored order would change under the observer's feet as opponents bid,
/// so one hand would take different canonical forms at different points of one auction.
///
/// ## Why the tie-break reads the auction
///
/// 7.5% of hands have two suits with identical lane bits (cf. `hand_class`). Breaking
/// that tie by physical suit index looks harmless — the two lanes are equal, so the
/// hand block comes out the same either way — and it is **wrong**, because the
/// observation contains more than the hand. If the auction has named one of the two
/// tied suits, renaming the deal moves that mention to the other member of the pair and
/// the two positions no longer canonicalise to the same thing. Caught by
/// `canonical_bid_obs_is_invariant_under_suit_renaming`, not by inspection.
///
/// So ties fall back to the auction, through keys renaming cannot touch: the highest
/// value bid in the suit, then the earliest slot it appeared at. Physical index remains
/// the last resort — and when it is reached the two suits are genuinely
/// indistinguishable in the *whole* observation (equal lanes, equal auction footprint,
/// and Q/K live inside the lane so the belote bits match too), which makes either
/// choice produce a bit-identical obs.
pub fn canonical_bid_order(hand: u32, bid_history: &[(u8, u8)]) -> [u8; 4] {
    // Same window the observation encodes, so the tie-break sees what the net sees.
    let history = if bid_history.len() > 12 {
        &bid_history[bid_history.len() - 12..]
    } else {
        bid_history
    };
    let mut top = [0u32; 4]; // highest value bid in the suit
    let mut first = [u32::MAX; 4]; // earliest slot it was named at
    for (i, &(_seat, action)) in history.iter().enumerate() {
        if !(1..=40).contains(&action) {
            continue;
        }
        let (val, suit) = crate::bidding::decode_bid(action);
        let s = suit as usize;
        top[s] = top[s].max(val as u32);
        first[s] = first[s].min(i as u32);
    }

    let mut suits = [(0u8, 0u32, 0u32, 0u32); 4];
    for s in 0..4usize {
        let lane = (hand >> (s * 8)) & 0xFF;
        suits[s] = (
            s as u8,
            (lane.count_ones() << 8) | lane,
            top[s],
            // Earliest-first, expressed so that plain descending order works.
            if first[s] == u32::MAX { 0 } else { 13 - first[s] },
        );
    }
    suits.sort_by(|a, b| {
        b.1.cmp(&a.1)
            .then(b.2.cmp(&a.2))
            .then(b.3.cmp(&a.3))
            .then(a.0.cmp(&b.0))
    });
    [suits[0].0, suits[1].0, suits[2].0, suits[3].0]
}

/// Invert an ordering. `order[canon] = phys` ⇒ `perm[phys] = canon`.
///
/// The two are not interchangeable and mixing them up is the classic canonical-obs
/// bug: the observation still looks legal, the model just answers about a different
/// suit. Use `perm` to push an observation *into* canonical space, and `order`
/// itself to bring an action back out.
#[inline]
pub fn perm_from_order(order: &[u8; 4]) -> [u8; 4] {
    let mut perm = [0u8; 4];
    for (canon, &phys) in order.iter().enumerate() {
        perm[phys as usize] = canon as u8;
    }
    perm
}

/// Permute a 43-bit legal-bid mask (the `u64` shape `legal_actions` returns).
#[inline]
pub fn permute_bid_mask_u64(mask: u64, perm: &[u8; 4]) -> u64 {
    let mut out = 0u64;
    for a in 0..43u8 {
        if mask & (1u64 << a) != 0 {
            out |= 1u64 << permute_bid_action(a, perm);
        }
    }
    out
}

/// Permute a card index (0-31) action by suit remapping.
/// Card layout: suit = card/8, rank = card%8.
#[inline]
pub fn permute_play_action(action: u8, perm: &[u8; 4]) -> u8 {
    let suit = action / 8;
    let rank = action % 8;
    perm[suit as usize] * 8 + rank
}

/// Permute a 32-float play mask (card mask) by swapping 8-float suit lanes.
#[inline]
pub fn permute_play_mask_f32(mask: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(mask.len() >= 32);
    let mut tmp = [0.0f32; 32];
    tmp.copy_from_slice(&mask[..32]);
    for s in 0..4 {
        let dst = perm[s] as usize;
        mask[dst * 8..(dst + 1) * 8].copy_from_slice(&tmp[s * 8..(s + 1) * 8]);
    }
}

/// Permute a bid action (0-42) by remapping suit indices.
///
/// Action encoding:
///   0 = PASS, 1-36 = value_idx×4 + suit_idx + 1, 37-40 = capot×suit,
///   41 = COINCHE, 42 = SURCOINCHE.
#[inline]
pub fn permute_bid_action(action: u8, perm: &[u8; 4]) -> u8 {
    match action {
        0 | 41 | 42 => action,
        1..=36 => {
            let idx = action - 1;
            let value_idx = idx / 4;
            let suit_idx = idx % 4;
            value_idx * 4 + perm[suit_idx as usize] + 1
        }
        37..=40 => {
            let suit_idx = action - 37;
            37 + perm[suit_idx as usize]
        }
        _ => action,
    }
}

/// Permute a 43-float bid mask by remapping suit positions within each value group.
///
/// Layout: [PASS] [80×4suits] [90×4suits] ... [160×4suits] [capot×4suits] [COINCHE] [SURCOINCHE]
#[inline]
pub fn permute_bid_mask_f32(mask: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(mask.len() >= 43);
    // 9 regular value groups (actions 1-36) + 1 capot group (37-40) = 10 groups of 4
    for group in 0..10 {
        let base = if group < 9 { 1 + group * 4 } else { 37 };
        permute_suit_onehot(&mut mask[base..base + 4], perm);
    }
    // mask[0] PASS, mask[41] COINCHE, mask[42] SURCOINCHE — unchanged
}

/// Apply suit augmentation to a batch of DMC play samples in-place.
/// Each sample gets a random permutation from the 24 possible.
///
/// - obs_data: flat batch×415
/// - mask_data: flat batch×32
/// - actions: batch
/// Permute a 411-float trump-relative DMC observation in-place.
///
/// Layout (trump already at slot 0):
///   [0:32]    My hand — card block
///   [32:160]  Current trick — 4×32 card blocks
///   [160:256] Past played — 3×32 card blocks
///   [256:259] Contract (value, team, coinche) — no trump one-hot
///   [259:271] Voids — 3 groups of 4 suit-indexed
///   [271:275] Scores — unchanged
///   [275:347] Bid history — 12 slots × 6, suit one-hot at +2..+6
///   [347:379] Card trick index — card block
///   [379:411] Card sequence index — card block
pub fn permute_dmc_obs_tr(obs: &mut [f32], perm: &[u8; 4]) {
    debug_assert!(obs.len() >= 411);

    // Block 1 [0:32]: My hand
    permute_card_block(&mut obs[0..32], perm);

    // Block 2 [32:160]: Current trick — 4 card blocks
    for i in 0..4 {
        let start = 32 + i * 32;
        permute_card_block(&mut obs[start..start + 32], perm);
    }

    // Block 3 [160:256]: Per-player played — 3 card blocks
    for i in 0..3 {
        let start = 160 + i * 32;
        permute_card_block(&mut obs[start..start + 32], perm);
    }

    // Block 4 [256:259]: Contract (value, team, coinche) — NO trump one-hot, unchanged

    // Block 5 [259:271]: Void tracking — 3 groups of 4 suit-indexed
    permute_suit_onehot(&mut obs[259..263], perm);
    permute_suit_onehot(&mut obs[263..267], perm);
    permute_suit_onehot(&mut obs[267..271], perm);

    // Block 6 [271:275]: Scoring context — unchanged

    // Block 7 [275:347]: Bid history — 12 slots × 6
    for slot in 0..12 {
        let base = 275 + slot * 6;
        permute_suit_onehot(&mut obs[base + 2..base + 6], perm);
    }

    // Block 8 [347:379]: Card trick index — card block
    permute_card_block(&mut obs[347..379], perm);

    // Block 9 [379:411]: Card sequence index — card block
    permute_card_block(&mut obs[379..411], perm);
}

/// Apply non-trump suit augmentation to a batch of trump-relative play samples.
/// Uses only the 6 permutations that fix slot 0 (trump stays at position 0).
///
/// - obs_data: flat batch×411
/// - mask_data: flat batch×32
/// - actions: batch (canonical card indices)
pub fn augment_play_batch_tr(
    obs_data: &mut [f32],
    mask_data: &mut [f32],
    actions: &mut [u8],
    rng: &mut impl rand::Rng,
) {
    let batch = actions.len();
    for i in 0..batch {
        let perm_idx = rng.gen_range(0..6usize);
        if perm_idx == 0 {
            continue; // identity permutation, skip
        }
        let perm = &ALL_PERMS[perm_idx]; // first 6 entries fix slot 0
        let obs_start = i * 411;
        permute_dmc_obs_tr(&mut obs_data[obs_start..obs_start + 411], perm);
        let mask_start = i * 32;
        permute_play_mask_f32(&mut mask_data[mask_start..mask_start + 32], perm);
        actions[i] = permute_play_action(actions[i], perm);
    }
}

pub fn augment_play_batch(
    obs_data: &mut [f32],
    mask_data: &mut [f32],
    actions: &mut [u8],
    rng: &mut impl rand::Rng,
) {
    let batch = actions.len();
    for i in 0..batch {
        let perm_idx = rng.gen_range(0..24usize);
        if perm_idx == 0 {
            continue; // identity permutation, skip
        }
        let perm = &ALL_PERMS[perm_idx];
        let obs_start = i * 415;
        permute_dmc_obs(&mut obs_data[obs_start..obs_start + 415], perm);
        let mask_start = i * 32;
        permute_play_mask_f32(&mut mask_data[mask_start..mask_start + 32], perm);
        actions[i] = permute_play_action(actions[i], perm);
    }
}

/// Apply suit augmentation to a batch of bid samples in-place.
///
/// - obs_data: flat batch×114
/// - mask_data: flat batch×43
/// - actions: batch
pub fn augment_bid_batch(
    obs_data: &mut [f32],
    mask_data: &mut [f32],
    actions: &mut [u8],
    rng: &mut impl rand::Rng,
) {
    augment_bid_batch_with_obs_dim(obs_data, mask_data, actions, crate::bid_obs::BID_OBS_DIM, rng);
}

/// Suit augmentation for bid batches with configurable obs_dim.
/// Permutes the first BID_OBS_DIM (108) elements (hand + bid history + position).
/// Score-aware extras at [108..113] are suit-invariant; the v3 belote bits at
/// [113..117] ARE suit-dependent and are permuted accordingly.
pub fn augment_bid_batch_with_obs_dim(
    obs_data: &mut [f32],
    mask_data: &mut [f32],
    actions: &mut [u8],
    obs_dim: usize,
    rng: &mut impl rand::Rng,
) {
    let batch = actions.len();
    for i in 0..batch {
        let perm_idx = rng.gen_range(0..24usize);
        if perm_idx == 0 {
            continue;
        }
        let perm = &ALL_PERMS[perm_idx];
        let obs_start = i * obs_dim;
        permute_bid_obs_dim(&mut obs_data[obs_start..obs_start + obs_dim], obs_dim, perm);
        let mask_start = i * 43;
        permute_bid_mask_f32(&mut mask_data[mask_start..mask_start + 43], perm);
        actions[i] = permute_bid_action(actions[i], perm);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_identity_perm_card_block() {
        let perm = [0, 1, 2, 3];
        let mut block = [0.0f32; 32];
        for i in 0..32 {
            block[i] = i as f32;
        }
        let original = block;
        permute_card_block(&mut block, &perm);
        assert_eq!(block, original);
    }

    #[test]
    fn test_swap_roundtrip_card_block() {
        let perm = [1, 0, 3, 2]; // swap S<->H, D<->C
        let mut block = [0.0f32; 32];
        for i in 0..8 {
            block[i] = 1.0; // spades
        }
        let original = block;

        permute_card_block(&mut block, &perm);
        // Spades data should now be in Hearts lane
        for i in 0..8 {
            assert_eq!(block[i], 0.0, "spades lane should be empty");
            assert_eq!(block[8 + i], 1.0, "hearts lane should have spades data");
        }

        // Apply same perm again to get back
        permute_card_block(&mut block, &perm);
        assert_eq!(block, original);
    }

    #[test]
    fn test_suit_onehot_perm() {
        let perm = [2, 3, 0, 1]; // S->D, H->C, D->S, C->H
        let mut onehot = [1.0, 0.0, 0.0, 0.0]; // Spades
        permute_suit_onehot(&mut onehot, &perm);
        assert_eq!(onehot, [0.0, 0.0, 1.0, 0.0]); // Diamonds
    }

    #[test]
    fn test_all_24_perms_distinct() {
        let mut seen = std::collections::HashSet::new();
        for perm in &ALL_PERMS {
            assert!(seen.insert(*perm), "duplicate permutation: {:?}", perm);
        }
        assert_eq!(seen.len(), 24);
    }

    #[test]
    fn test_identity_v1_no_change() {
        let perm = [0, 1, 2, 3];
        let mut obs = vec![0.5f32; 330];
        let original = obs.clone();
        permute_belief_obs_v1(&mut obs, &perm);
        assert_eq!(obs, original);
    }

    #[test]
    fn test_identity_v2_no_change() {
        let perm = [0, 1, 2, 3];
        let mut obs = vec![0.5f32; 304];
        let original = obs.clone();
        permute_belief_obs_v2(&mut obs, &perm);
        assert_eq!(obs, original);
    }

    #[test]
    fn test_permute_target() {
        let mut target = [0u8; 32];
        // Player 1 for all spades, player 2 for all hearts
        for r in 0..8 {
            target[r] = 1;
            target[8 + r] = 2;
        }

        let perm = [1, 0, 2, 3]; // swap S<->H
        permute_target(&mut target, &perm);

        for r in 0..8 {
            assert_eq!(target[r], 2, "spades lane should now have hearts data");
            assert_eq!(target[8 + r], 1, "hearts lane should now have spades data");
        }
    }

    #[test]
    fn test_permute_mask() {
        let mask = 0xFFu32; // all spades
        let perm = [2, 1, 0, 3]; // S->D
        let result = permute_mask(mask, &perm);
        assert_eq!(result, 0xFF_0000); // all diamonds
    }

    #[test]
    fn test_permute_mask_identity() {
        let mask = 0xDEAD_BEEFu32;
        let perm = [0, 1, 2, 3];
        assert_eq!(permute_mask(mask, &perm), mask);
    }

    #[test]
    fn test_permute_mask_roundtrip() {
        let mask = 0xAB_CD_EF_12u32;
        let perm = [1, 0, 3, 2]; // swap S<->H, D<->C
        let once = permute_mask(mask, &perm);
        let twice = permute_mask(once, &perm);
        assert_eq!(twice, mask);
    }

    #[test]
    fn test_v1_perm_preserves_non_suit_blocks() {
        // Blocks that aren't suit-indexed should be unchanged by any permutation
        let perm = [3, 2, 1, 0]; // reverse all suits
        let mut obs = vec![0.0f32; 330];

        // Set scoring context (block 8, [316:320]) to known values
        obs[316] = 0.1;
        obs[317] = 0.2;
        obs[318] = 0.3;
        obs[319] = 0.4;

        // Set dealer pos (block 9, [320:324])
        obs[322] = 1.0;

        // Set trick progress (block 11, [328:330])
        obs[328] = 0.5;
        obs[329] = 0.75;

        permute_belief_obs_v1(&mut obs, &perm);

        assert_eq!(obs[316], 0.1);
        assert_eq!(obs[317], 0.2);
        assert_eq!(obs[318], 0.3);
        assert_eq!(obs[319], 0.4);
        assert_eq!(obs[322], 1.0);
        assert_eq!(obs[328], 0.5);
        assert_eq!(obs[329], 0.75);
    }

    #[test]
    fn test_v2_perm_preserves_non_suit_blocks() {
        let perm = [3, 2, 1, 0];
        let mut obs = vec![0.0f32; 304];

        // Contract non-trump fields [204:208]: bid_value, taker, coinche
        obs[204] = 0.32; // bid_value / 250
        obs[205] = 1.0;  // taker team
        obs[206] = 0.0;
        obs[207] = 0.5;  // coinche

        permute_belief_obs_v2(&mut obs, &perm);

        assert_eq!(obs[204], 0.32);
        assert_eq!(obs[205], 1.0);
        assert_eq!(obs[206], 0.0);
        assert_eq!(obs[207], 0.5);
    }

    #[test]
    fn test_dmc_obs_identity() {
        let perm = [0, 1, 2, 3];
        let mut obs = vec![0.5f32; 415];
        let original = obs.clone();
        permute_dmc_obs(&mut obs, &perm);
        assert_eq!(obs, original);
    }

    #[test]
    fn test_dmc_obs_preserves_non_suit_blocks() {
        let perm = [3, 2, 1, 0];
        let mut obs = vec![0.0f32; 415];
        // Scoring context [275:279]
        obs[275] = 0.1;
        obs[276] = 0.2;
        obs[277] = 0.3;
        obs[278] = 0.4;
        // bid_value [260], is_my_team [261], coinche [262]
        obs[260] = 0.32;
        obs[261] = 1.0;
        obs[262] = 0.5;
        permute_dmc_obs(&mut obs, &perm);
        assert_eq!(obs[275], 0.1);
        assert_eq!(obs[276], 0.2);
        assert_eq!(obs[260], 0.32);
        assert_eq!(obs[261], 1.0);
        assert_eq!(obs[262], 0.5);
    }

    #[test]
    fn test_dmc_obs_hand_permutation() {
        let perm = [1, 0, 2, 3]; // swap S<->H
        let mut obs = vec![0.0f32; 415];
        // Set spades (bits 0-7) to 1.0 in hand block
        for i in 0..8 {
            obs[i] = 1.0;
        }
        permute_dmc_obs(&mut obs, &perm);
        // Spades data should now be in hearts lane (bits 8-15)
        for i in 0..8 {
            assert_eq!(obs[i], 0.0, "spades lane should be empty");
            assert_eq!(obs[8 + i], 1.0, "hearts lane should have data");
        }
    }

    #[test]
    fn test_bid_obs_identity() {
        let perm = [0, 1, 2, 3];
        let mut obs = vec![0.5f32; 108];
        let original = obs.clone();
        permute_bid_obs(&mut obs, &perm);
        assert_eq!(obs, original);
    }

    #[test]
    fn test_bid_obs_preserves_position() {
        let perm = [2, 3, 0, 1];
        let mut obs = vec![0.0f32; 108];
        // Position [104:108]
        obs[105] = 1.0;
        permute_bid_obs(&mut obs, &perm);
        assert_eq!(obs[105], 1.0);
    }

    #[test]
    fn test_play_action_roundtrip() {
        let perm = [1, 0, 3, 2]; // swap S<->H, D<->C
        for action in 0..32u8 {
            let once = permute_play_action(action, &perm);
            let twice = permute_play_action(once, &perm);
            assert_eq!(twice, action, "roundtrip failed for action {}", action);
        }
    }

    #[test]
    fn test_bid_action_identity() {
        let perm = [0, 1, 2, 3];
        for action in 0..43u8 {
            assert_eq!(permute_bid_action(action, &perm), action);
        }
    }

    #[test]
    fn test_bid_action_roundtrip() {
        let perm = [1, 0, 3, 2];
        for action in 0..43u8 {
            let once = permute_bid_action(action, &perm);
            let twice = permute_bid_action(once, &perm);
            assert_eq!(twice, action, "roundtrip failed for bid action {}", action);
        }
    }

    #[test]
    fn test_bid_action_pass_coinche_unchanged() {
        for &perm in &ALL_PERMS {
            assert_eq!(permute_bid_action(0, &perm), 0);  // PASS
            assert_eq!(permute_bid_action(41, &perm), 41); // COINCHE
            assert_eq!(permute_bid_action(42, &perm), 42); // SURCOINCHE
        }
    }

    #[test]
    fn test_bid_mask_identity() {
        let perm = [0, 1, 2, 3];
        let mut mask = vec![0.0f32; 43];
        mask[0] = 1.0; // PASS
        mask[1] = 1.0; // 80S
        mask[41] = 1.0; // COINCHE
        let original = mask.clone();
        permute_bid_mask_f32(&mut mask, &perm);
        assert_eq!(mask, original);
    }

    #[test]
    fn test_bid_mask_swap() {
        let perm = [1, 0, 2, 3]; // swap S<->H
        let mut mask = vec![0.0f32; 43];
        mask[1] = 1.0; // 80S
        permute_bid_mask_f32(&mut mask, &perm);
        assert_eq!(mask[1], 0.0); // 80S should be gone
        assert_eq!(mask[2], 1.0); // 80H should be set
    }

    #[test]
    fn test_play_mask_f32_roundtrip() {
        let perm = [2, 3, 0, 1]; // S->D, H->C, D->S, C->H
        let mut mask = [0.0f32; 32];
        mask[0] = 1.0; // first spade
        mask[8] = 1.0; // first heart
        let original = mask;
        permute_play_mask_f32(&mut mask, &perm);
        // Spade→Diamond, Heart→Club
        assert_eq!(mask[16], 1.0); // first diamond
        assert_eq!(mask[24], 1.0); // first club
        // Apply again
        permute_play_mask_f32(&mut mask, &perm);
        assert_eq!(mask, original);
    }
}
