/// Suit permutation utilities for data augmentation.
///
/// All 4 suits in Belote are interchangeable within the observation encoding,
/// so any game can be replayed with suits relabeled (4! = 24 permutations),
/// giving 24x data diversity for free.

/// All 24 permutations of 4 suits.
/// `perm[s]` = which output suit lane suit `s` maps to.
pub const ALL_PERMS: [[u8; 4]; 24] = [
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
}
