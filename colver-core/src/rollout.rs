#[cfg(feature = "rand")]
use rand::Rng;

use crate::state::GameState;

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
}
