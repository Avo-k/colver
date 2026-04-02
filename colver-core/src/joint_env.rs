/// Joint training environment: exposes both bidding and playing phases to the
/// training loop so that both a bid NN and a play NN can be trained together.
///
/// Unlike `VecTrainingEnv` (which handles bidding internally via heuristics),
/// this environment returns control to the caller during bidding, allowing the
/// training loop to choose between the training bid NN and heuristic opponents.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::belief_net::BeliefNet;
use crate::belief_obs::{self, BELIEF_OBS_DIM};
use crate::bid_eval;
use crate::bid_net::BidNet;
use crate::bid_obs::{self, BID_MASK_DIM, BID_OBS_DIM};
use crate::card;
use crate::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use crate::rollout;
use crate::scoring::compute_deal_score;
use crate::state::{GameState, Phase};

const PLAY_MASK_DIM: usize = 32;

/// Belief prediction dimension: 32 cards × 3 player slots (left/partner/right).
pub const BELIEF_PRED_DIM: usize = 96;

/// A single joint training environment.
pub struct JointEnv {
    pub state: GameState,
    pub tracking: EnvTracking,
}

impl JointEnv {
    pub fn new(rng: &mut impl Rng) -> Self {
        let dealer = rng.gen_range(0..4u8);
        let state = GameState::deal_random(dealer, rng);
        let mut tracking = EnvTracking::new();
        tracking.dealer = dealer;
        JointEnv { state, tracking }
    }

    pub fn reset(&mut self, rng: &mut impl Rng) {
        let dealer = rng.gen_range(0..4u8);
        self.state = GameState::deal_random(dealer, rng);
        self.tracking.reset(dealer);
    }

    /// Step the environment. Returns (done, ns_reward, ew_reward).
    /// Rewards are normalized score difference: (my_score - opp_score) / 500.
    pub fn step(&mut self, action: u8) -> (bool, f32, f32) {
        self.tracking.track_action(&self.state, action);
        self.state.step(action);

        let done = self.state.is_terminal();
        if done {
            let score = compute_deal_score(&self.state);
            let ns_reward = (score.scores[0] - score.scores[1]) as f32 / 500.0;
            (true, ns_reward, -ns_reward)
        } else {
            (false, 0.0, 0.0)
        }
    }

    #[inline]
    pub fn write_play_obs(&self, buf: &mut [f32], offset: usize) {
        dmc_obs::write_observation_tr(buf, offset, &self.state, &self.tracking);
    }

    #[inline]
    pub fn write_play_mask(&self, buf: &mut [f32], offset: usize) {
        dmc_obs::write_mask_tr(buf, offset, &self.state, &self.tracking);
    }

    #[inline]
    pub fn write_bid_obs(&self, buf: &mut [f32], offset: usize) {
        bid_obs::write_bid_observation(
            buf,
            offset,
            &self.state,
            &self.tracking.bid_history,
        );
    }

    #[inline]
    pub fn write_bid_mask(&self, buf: &mut [f32], offset: usize) {
        bid_obs::write_bid_mask(buf, offset, &self.state);
    }
}

/// Vectorized joint training environment with flat observation buffers.
pub struct VecJointEnv {
    pub envs: Vec<JointEnv>,
    /// Flat play observation buffer: n_envs × OBS_DIM_TR (411, trump-relative).
    pub play_obs_buf: Vec<f32>,
    /// Flat play mask buffer: n_envs × 32.
    pub play_mask_buf: Vec<f32>,
    /// Flat bid observation buffer: n_envs × BID_OBS_DIM (114).
    pub bid_obs_buf: Vec<f32>,
    /// Flat bid mask buffer: n_envs × BID_MASK_DIM (43).
    pub bid_mask_buf: Vec<f32>,
    /// Optional CPU bid NN for heuristic opponent bidding.
    bid_net: Option<BidNet>,
    /// Optional belief net for card location prediction (frozen, CPU inference).
    belief_net: Option<BeliefNet>,
    /// Flat belief prediction buffer: n_envs × BELIEF_PRED_DIM (96).
    /// Contains softmax probabilities in canonical card ordering.
    belief_pred_buf: Vec<f32>,
    /// Scratch buffer for building belief observations (reused across envs).
    belief_obs_scratch: Vec<f32>,
    pub rng: StdRng,
}

impl VecJointEnv {
    pub fn new(n_envs: usize, seed: u64) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let envs: Vec<_> = (0..n_envs).map(|_| JointEnv::new(&mut rng)).collect();

        let mut env = VecJointEnv {
            envs,
            play_obs_buf: vec![0.0f32; n_envs * OBS_DIM_TR],
            play_mask_buf: vec![0.0f32; n_envs * PLAY_MASK_DIM],
            bid_obs_buf: vec![0.0f32; n_envs * BID_OBS_DIM],
            bid_mask_buf: vec![0.0f32; n_envs * BID_MASK_DIM],
            bid_net: None,
            belief_net: None,
            belief_pred_buf: vec![0.0f32; n_envs * BELIEF_PRED_DIM],
            belief_obs_scratch: vec![0.0f32; BELIEF_OBS_DIM],
            rng,
        };
        env.refresh_all();
        env
    }

    #[inline]
    pub fn n_envs(&self) -> usize {
        self.envs.len()
    }

    /// Load a pre-trained bid NN (used for opponent bidding).
    pub fn load_bid_net(&mut self, path: &str) -> std::io::Result<()> {
        self.bid_net = Some(BidNet::load(path)?);
        Ok(())
    }

    /// Load a frozen belief net for card location prediction.
    pub fn load_belief_net(&mut self, path: &str) -> std::io::Result<()> {
        self.belief_net = Some(BeliefNet::load(path)?);
        Ok(())
    }

    /// Whether a belief net is loaded.
    pub fn has_belief_net(&self) -> bool {
        self.belief_net.is_some()
    }

    /// Get belief prediction slice for env i (96 floats in canonical card ordering).
    #[inline]
    pub fn belief_pred_slice(&self, i: usize) -> &[f32] {
        &self.belief_pred_buf[i * BELIEF_PRED_DIM..(i + 1) * BELIEF_PRED_DIM]
    }

    /// Refresh belief predictions for specified envs only.
    /// Runs the belief net on V1 obs, applies softmax, remaps to canonical ordering.
    pub fn refresh_belief_preds_for(&mut self, indices: &[usize]) {
        let belief_net = match self.belief_net.as_mut() {
            Some(net) => net as *mut BeliefNet,
            None => return,
        };
        // SAFETY: we need mutable access to belief_net (scratch buffers) and self.envs
        // simultaneously. The belief_net only modifies its internal scratch buffers.
        let belief_net = unsafe { &mut *belief_net };

        for &i in indices {
            let pred_start = i * BELIEF_PRED_DIM;
            if self.envs[i].state.phase != Phase::Playing {
                for v in &mut self.belief_pred_buf[pred_start..pred_start + BELIEF_PRED_DIM] {
                    *v = 0.0;
                }
                continue;
            }

            let observer = self.envs[i].state.current_player();
            let state = &self.envs[i].state;
            let tracking = &self.envs[i].tracking;

            // Build V1 belief observation
            for v in self.belief_obs_scratch.iter_mut() {
                *v = 0.0;
            }
            belief_obs::write_belief_observation(
                &mut self.belief_obs_scratch, 0, state, tracking, observer,
            );

            // Run belief net inference
            let logits = belief_net.evaluate(&self.belief_obs_scratch);
            let num_classes = belief_net.num_classes();

            // Compute softmax probabilities per card, output in canonical order
            let order = dmc_obs::current_player_order(state, tracking);

            // Known cards (observer hand + played + current trick)
            let observer_hand = state.hands[observer as usize];
            let mut played = state.played_cards;
            for ci in 0..4 {
                let c = state.current_trick[ci];
                if c != card::EMPTY {
                    played |= 1u32 << c;
                }
            }
            let known = observer_hand | played;

            for card_phys in 0..32u8 {
                let card_canon = dmc_obs::card_to_canonical(card_phys, &order);
                let canon_base = card_canon as usize * 3;

                if known & (1u32 << card_phys) != 0 {
                    // Known card: zero probabilities
                    self.belief_pred_buf[pred_start + canon_base] = 0.0;
                    self.belief_pred_buf[pred_start + canon_base + 1] = 0.0;
                    self.belief_pred_buf[pred_start + canon_base + 2] = 0.0;
                    continue;
                }

                // Extract 3-class probabilities (left/partner/right)
                let (p_left, p_partner, p_right) = if num_classes == 3 {
                    let base = card_phys as usize * 3;
                    let max_l = logits[base].max(logits[base + 1]).max(logits[base + 2]);
                    let e0 = (logits[base] - max_l).exp();
                    let e1 = (logits[base + 1] - max_l).exp();
                    let e2 = (logits[base + 2] - max_l).exp();
                    let s = e0 + e1 + e2;
                    (e0 / s, e1 / s, e2 / s)
                } else {
                    // 4-class: softmax, zero observer slot, renormalize
                    let base = card_phys as usize * 4;
                    let max_l = logits[base].max(logits[base + 1]).max(logits[base + 2]).max(logits[base + 3]);
                    let e0 = (logits[base] - max_l).exp(); // observer
                    let e1 = (logits[base + 1] - max_l).exp(); // left
                    let e2 = (logits[base + 2] - max_l).exp(); // partner
                    let e3 = (logits[base + 3] - max_l).exp(); // right
                    let s = e1 + e2 + e3; // exclude observer
                    let _ = e0;
                    (e1 / s, e2 / s, e3 / s)
                };

                // Write in canonical card order: [left, partner, right]
                self.belief_pred_buf[pred_start + canon_base] = p_left;
                self.belief_pred_buf[pred_start + canon_base + 1] = p_partner;
                self.belief_pred_buf[pred_start + canon_base + 2] = p_right;
            }
        }
    }

    /// Set bid NN from flat weights (for bid pool).
    pub fn set_bid_net_from_floats(
        &mut self,
        floats: &[f32],
        hidden: usize,
    ) -> std::io::Result<()> {
        self.bid_net = Some(BidNet::from_floats(floats, hidden, BID_OBS_DIM, true)?);
        Ok(())
    }

    pub fn phases(&self) -> Vec<Phase> {
        self.envs.iter().map(|e| e.state.phase).collect()
    }

    pub fn current_players(&self) -> Vec<u8> {
        self.envs.iter().map(|e| e.state.current_player()).collect()
    }

    /// Refresh play obs/mask for all envs (call after stepping).
    pub fn refresh_play(&mut self) {
        for i in 0..self.envs.len() {
            self.envs[i].write_play_obs(&mut self.play_obs_buf, i * OBS_DIM_TR);
            self.envs[i].write_play_mask(&mut self.play_mask_buf, i * PLAY_MASK_DIM);
        }
    }

    /// Refresh bid obs/mask for all envs (call after stepping).
    pub fn refresh_bid(&mut self) {
        for i in 0..self.envs.len() {
            if self.envs[i].state.phase == Phase::Bidding {
                self.envs[i].write_bid_obs(&mut self.bid_obs_buf, i * BID_OBS_DIM);
                self.envs[i].write_bid_mask(&mut self.bid_mask_buf, i * BID_MASK_DIM);
            }
        }
    }

    /// Refresh all buffers.
    pub fn refresh_all(&mut self) {
        self.refresh_play();
        self.refresh_bid();
    }

    // --- Observation/mask slices ---

    #[inline]
    pub fn play_obs_slice(&self, i: usize) -> &[f32] {
        &self.play_obs_buf[i * OBS_DIM_TR..(i + 1) * OBS_DIM_TR]
    }

    #[inline]
    pub fn play_mask_slice(&self, i: usize) -> &[f32] {
        &self.play_mask_buf[i * PLAY_MASK_DIM..(i + 1) * PLAY_MASK_DIM]
    }

    #[inline]
    pub fn bid_obs_slice(&self, i: usize) -> &[f32] {
        &self.bid_obs_buf[i * BID_OBS_DIM..(i + 1) * BID_OBS_DIM]
    }

    #[inline]
    pub fn bid_mask_slice(&self, i: usize) -> &[f32] {
        &self.bid_mask_buf[i * BID_MASK_DIM..(i + 1) * BID_MASK_DIM]
    }

    /// Get heuristic bid action for env i (for opponent diversity).
    pub fn heuristic_bid(&self, i: usize, strategy: u8) -> u8 {
        match strategy {
            0 => bid_eval::improved_v2_bid(&self.envs[i].state),
            idx @ 1..=6 => {
                let presets = bid_eval::BidParams::all_presets();
                bid_eval::parametric_bid(&self.envs[i].state, &presets[(idx - 1) as usize])
            }
            7 => bid_eval::heuristic_bid(&self.envs[i].state),
            _ => bid_eval::improved_v2_bid(&self.envs[i].state),
        }
    }

    /// Get NN bid action for env i using the loaded bid_net (greedy).
    pub fn nn_bid(&mut self, i: usize) -> u8 {
        let legal = self.envs[i].state.legal_actions();
        let obs_start = i * BID_OBS_DIM;
        let obs_end = obs_start + BID_OBS_DIM;
        if let Some(ref mut net) = self.bid_net {
            net.best_action_fast(&self.bid_obs_buf[obs_start..obs_end], legal)
        } else {
            bid_eval::improved_v2_bid(&self.envs[i].state) // fallback
        }
    }

    /// Get a random legal action for env i.
    pub fn random_action(&mut self, i: usize) -> u8 {
        let mask = self.envs[i].state.legal_actions();
        let count = mask.count_ones();
        let n = self.rng.gen_range(0..count);
        rollout::select_nth_bit(mask, n)
    }

    /// Step all environments with given actions. Auto-resets terminals.
    /// Returns (dones, rewards) where rewards[i] = (ns_reward, ew_reward).
    pub fn step_all(&mut self, actions: &[u8]) -> (Vec<bool>, Vec<(f32, f32)>) {
        let n = self.envs.len();
        assert_eq!(actions.len(), n);

        let mut dones = Vec::with_capacity(n);
        let mut rewards = Vec::with_capacity(n);

        for (i, &action) in actions.iter().enumerate() {
            let (done, ns_r, ew_r) = self.envs[i].step(action);
            dones.push(done);
            rewards.push((ns_r, ew_r));

            if done {
                self.envs[i].reset(&mut self.rng);
            }
        }

        self.refresh_all();
        (dones, rewards)
    }
}
