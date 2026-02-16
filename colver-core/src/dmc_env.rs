/// Training environments for DMC: single env + vectorized env.
///
/// Port of VecEnv stepping logic from `colver-py/src/lib.rs`.
/// Owns pre-allocated observation and mask buffers so the training loop
/// can create GPU tensors with a single memcpy.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bid_eval;
use crate::dmc_obs::{self, EnvTracking, OBS_DIM};
use crate::rollout;
use crate::state::{GameState, Phase};

const MASK_DIM: usize = 32;

/// A single training environment wrapping a Belote Contrée deal.
pub struct TrainingEnv {
    pub state: GameState,
    pub tracking: EnvTracking,
}

impl TrainingEnv {
    pub fn new(rng: &mut impl Rng) -> Self {
        let dealer = rng.gen_range(0..4u8);
        let state = GameState::deal_random(dealer, rng);
        let mut tracking = EnvTracking::new();
        tracking.dealer = dealer;
        TrainingEnv { state, tracking }
    }

    /// Reset with a new random deal.
    pub fn reset(&mut self, rng: &mut impl Rng) {
        let dealer = rng.gen_range(0..4u8);
        self.state = GameState::deal_random(dealer, rng);
        self.tracking.reset(dealer);
    }

    /// Step the environment. Returns (done, ns_outcome, ew_outcome).
    /// Outcomes are only meaningful when done=true:
    /// 1.0/0.0 for win/loss, 0.5/0.5 for void/tie.
    pub fn step(&mut self, action: u8) -> (bool, f32, f32) {
        self.tracking.track_action(&self.state, action);
        self.state.step(action);

        let done = self.state.is_terminal();
        if done {
            let r = self.state.rewards();
            let (ns, ew) = if r[0] == 0.0 && r[1] == 0.0 {
                (0.5, 0.5) // void deal
            } else if r[0] > r[1] {
                (1.0, 0.0)
            } else if r[1] > r[0] {
                (0.0, 1.0)
            } else {
                (0.5, 0.5) // tie
            };
            (true, ns, ew)
        } else {
            (false, 0.0, 0.0)
        }
    }

    /// Write observation into buffer at offset.
    #[inline]
    pub fn write_obs(&self, buf: &mut [f32], offset: usize) {
        dmc_obs::write_observation(buf, offset, &self.state, &self.tracking);
    }

    /// Write legal action mask into buffer at offset.
    #[inline]
    pub fn write_mask(&self, buf: &mut [f32], offset: usize) {
        dmc_obs::write_mask(buf, offset, &self.state);
    }
}

/// Vectorized training environments with pre-allocated flat buffers.
pub struct VecTrainingEnv {
    pub envs: Vec<TrainingEnv>,
    /// Flat observation buffer: n_envs * OBS_DIM.
    pub obs_buf: Vec<f32>,
    /// Flat mask buffer: n_envs * MASK_DIM.
    pub mask_buf: Vec<f32>,
    /// Per-env bidding strategy: 0=improved, 1-6=BidParams presets, 7=heuristic.
    bid_strategies: Vec<u8>,
    pub rng: StdRng,
}

impl VecTrainingEnv {
    pub fn new(n_envs: usize) -> Self {
        let mut rng = StdRng::from_entropy();
        let envs: Vec<TrainingEnv> = (0..n_envs).map(|_| TrainingEnv::new(&mut rng)).collect();
        let obs_buf = vec![0.0f32; n_envs * OBS_DIM];
        let mask_buf = vec![0.0f32; n_envs * MASK_DIM];
        let bid_strategies = vec![0u8; n_envs];

        let mut vec_env = VecTrainingEnv {
            envs,
            obs_buf,
            mask_buf,
            bid_strategies,
            rng,
        };
        vec_env.refresh_observations();
        vec_env
    }

    /// Create with a specific seed for reproducibility.
    pub fn new_with_seed(n_envs: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let envs: Vec<TrainingEnv> = (0..n_envs).map(|_| TrainingEnv::new(&mut rng)).collect();
        let obs_buf = vec![0.0f32; n_envs * OBS_DIM];
        let mask_buf = vec![0.0f32; n_envs * MASK_DIM];
        let bid_strategies = vec![0u8; n_envs];

        let mut vec_env = VecTrainingEnv {
            envs,
            obs_buf,
            mask_buf,
            bid_strategies,
            rng,
        };
        vec_env.refresh_observations();
        vec_env
    }

    /// Number of environments.
    #[inline]
    pub fn n_envs(&self) -> usize {
        self.envs.len()
    }

    /// Set bidding strategies per environment.
    pub fn set_bid_strategies(&mut self, strategies: &[u8]) {
        assert_eq!(strategies.len(), self.envs.len());
        self.bid_strategies.copy_from_slice(strategies);
    }

    /// Get current player for each environment.
    pub fn current_players(&self) -> Vec<u8> {
        self.envs.iter().map(|e| e.state.current_player()).collect()
    }

    /// Get current phase for each environment.
    pub fn phases(&self) -> Vec<Phase> {
        self.envs.iter().map(|e| e.state.phase).collect()
    }

    /// Get bid action for each environment using its assigned strategy.
    pub fn bid_actions(&self) -> Vec<u8> {
        let presets = bid_eval::BidParams::all_presets();
        self.envs
            .iter()
            .enumerate()
            .map(|(i, e)| {
                if e.state.phase != Phase::Bidding {
                    return 0;
                }
                match self.bid_strategies[i] {
                    0 => bid_eval::improved_v2_bid(&e.state),
                    idx @ 1..=6 => bid_eval::parametric_bid(&e.state, &presets[(idx - 1) as usize]),
                    7 => bid_eval::heuristic_bid(&e.state),
                    _ => bid_eval::improved_v2_bid(&e.state),
                }
            })
            .collect()
    }

    /// Step all environments. Auto-resets terminals.
    /// Returns (dones, outcomes) where outcomes[i] = (ns_outcome, ew_outcome).
    pub fn step_all(&mut self, actions: &[u8]) -> (Vec<bool>, Vec<(f32, f32)>) {
        let n = self.envs.len();
        assert_eq!(actions.len(), n);

        let mut dones = Vec::with_capacity(n);
        let mut outcomes = Vec::with_capacity(n);

        for (i, &action) in actions.iter().enumerate() {
            let (done, ns, ew) = self.envs[i].step(action);
            dones.push(done);
            outcomes.push((ns, ew));

            if done {
                self.envs[i].reset(&mut self.rng);
            }
        }

        self.refresh_observations();
        (dones, outcomes)
    }

    /// Refresh all observation and mask buffers.
    pub fn refresh_observations(&mut self) {
        let n = self.envs.len();
        for i in 0..n {
            self.envs[i].write_obs(&mut self.obs_buf, i * OBS_DIM);
            self.envs[i].write_mask(&mut self.mask_buf, i * MASK_DIM);
        }
    }

    /// Get observation slice for environment i.
    #[inline]
    pub fn obs_slice(&self, i: usize) -> &[f32] {
        &self.obs_buf[i * OBS_DIM..(i + 1) * OBS_DIM]
    }

    /// Get mask slice for environment i.
    #[inline]
    pub fn mask_slice(&self, i: usize) -> &[f32] {
        &self.mask_buf[i * MASK_DIM..(i + 1) * MASK_DIM]
    }

    /// Get legal action mask as u32 for environment i (for DmcNet inference).
    #[inline]
    pub fn legal_mask_u32(&self, i: usize) -> u32 {
        self.envs[i].state.legal_actions() as u32
    }

    /// Get a random legal action for environment i.
    pub fn random_action(&mut self, i: usize) -> u8 {
        let mask = self.envs[i].state.legal_actions();
        let count = mask.count_ones();
        let n = self.rng.gen_range(0..count);
        rollout::select_nth_bit(mask, n)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_training_env_new() {
        let mut rng = StdRng::seed_from_u64(42);
        let env = TrainingEnv::new(&mut rng);
        assert_eq!(env.state.phase, Phase::Bidding);
    }

    #[test]
    fn test_training_env_reset() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = TrainingEnv::new(&mut rng);
        let old_hands = env.state.hands;
        env.reset(&mut rng);
        // Hands should be different (with overwhelming probability)
        assert_ne!(env.state.hands, old_hands);
    }

    #[test]
    fn test_training_env_play_to_completion() {
        let mut rng = StdRng::seed_from_u64(42);
        let mut env = TrainingEnv::new(&mut rng);

        // Play through with random moves
        while !env.state.is_terminal() {
            let mask = env.state.legal_actions();
            let count = mask.count_ones();
            let idx = rng.gen_range(0..count);
            let action = rollout::select_nth_bit(mask, idx);
            let (done, ns, ew) = env.step(action);
            if done {
                // Outcomes should be meaningful
                assert!(ns + ew > 0.0 || (ns == 0.5 && ew == 0.5));
                break;
            }
        }
    }

    #[test]
    fn test_vec_env_new() {
        let vec_env = VecTrainingEnv::new_with_seed(4, 42);
        assert_eq!(vec_env.n_envs(), 4);
        assert_eq!(vec_env.obs_buf.len(), 4 * OBS_DIM);
        assert_eq!(vec_env.mask_buf.len(), 4 * MASK_DIM);
    }

    #[test]
    fn test_vec_env_step_all() {
        let mut vec_env = VecTrainingEnv::new_with_seed(4, 42);

        // Get random actions for all envs
        let mut actions = Vec::new();
        for i in 0..4 {
            let mask = vec_env.envs[i].state.legal_actions();
            let count = mask.count_ones();
            let idx = vec_env.rng.gen_range(0..count);
            actions.push(rollout::select_nth_bit(mask, idx));
        }

        let (dones, outcomes) = vec_env.step_all(&actions);
        assert_eq!(dones.len(), 4);
        assert_eq!(outcomes.len(), 4);
    }

    #[test]
    fn test_vec_env_play_through() {
        let mut vec_env = VecTrainingEnv::new_with_seed(8, 123);

        // Play 200 steps — should auto-reset multiple times
        for _ in 0..200 {
            let n = vec_env.n_envs();
            let mut actions = Vec::with_capacity(n);
            for i in 0..n {
                let env = &vec_env.envs[i];
                if env.state.phase == Phase::Bidding {
                    actions.push(bid_eval::improved_v2_bid(&env.state));
                } else {
                    let mask = env.state.legal_actions();
                    let count = mask.count_ones();
                    let idx = vec_env.rng.gen_range(0..count);
                    actions.push(rollout::select_nth_bit(mask, idx));
                }
            }
            let (_dones, _outcomes) = vec_env.step_all(&actions);

            // Verify obs buf is populated
            for i in 0..n {
                let obs = vec_env.obs_slice(i);
                assert_eq!(obs.len(), OBS_DIM);
            }
        }
    }

    #[test]
    fn test_vec_env_bid_actions() {
        let vec_env = VecTrainingEnv::new_with_seed(4, 42);
        let bid_actions = vec_env.bid_actions();
        assert_eq!(bid_actions.len(), 4);
        // All envs start in bidding phase, so all actions should be valid
        for (i, &action) in bid_actions.iter().enumerate() {
            let legal = vec_env.envs[i].state.legal_actions();
            assert!(
                legal & (1u64 << action) != 0,
                "bid action {} illegal for env {}",
                action,
                i
            );
        }
    }

    #[test]
    fn test_vec_env_bid_strategies() {
        let mut vec_env = VecTrainingEnv::new_with_seed(4, 42);
        vec_env.set_bid_strategies(&[0, 7, 3, 5]); // improved, heuristic, moderate, aggressive
        let bid_actions = vec_env.bid_actions();
        assert_eq!(bid_actions.len(), 4);
    }
}
