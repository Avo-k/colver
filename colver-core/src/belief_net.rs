/// BeliefNet: pure Rust inference for the NN belief network (card location prediction).
///
/// Architecture (standard): 330 → 512 (LN+ReLU) → 512 (LN+ReLU) → 128
///
/// Output: 128 raw logits = 32 cards × 4 player slots.
/// Caller applies per-card softmax + hard constraint masking via `belief_to_weights()`.
///
/// Weight file layout (contiguous little-endian f32):
///   For each of 2 hidden layers:
///     W: in_dim × H (row-major), b: H, gamma: H, beta: H
///   Output layer:
///     W: H × 128 (row-major), b: 128

use crate::card;
use crate::state::GameState;

const NUM_OUTPUTS: usize = 128; // 32 cards × 4 players
const NUM_LAYERS: usize = 2;
const LN_EPS: f32 = 1e-5;
const DEFAULT_HIDDEN: usize = 512;

pub struct BeliefNet {
    w: [Vec<f32>; NUM_LAYERS],
    b: [Vec<f32>; NUM_LAYERS],
    gamma: [Vec<f32>; NUM_LAYERS],
    beta: [Vec<f32>; NUM_LAYERS],
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    obs_dim: usize,
    hidden: usize,
    in_dims: [usize; NUM_LAYERS],
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
}

impl BeliefNet {
    /// Load weights from a raw binary file with default hidden size (512).
    pub fn load(path: &str) -> std::io::Result<Self> {
        Self::load_with_hidden(path, DEFAULT_HIDDEN)
    }

    /// Load weights with custom hidden size.
    pub fn load_with_hidden(path: &str, hidden: usize) -> std::io::Result<Self> {
        let data = std::fs::read(path)?;
        if data.len() % 4 != 0 {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "weight file size not a multiple of 4",
            ));
        }

        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        let h = hidden;
        let total = floats.len();

        // Fixed part: layer1 (H*H + 3*H) + layer0 b/gamma/beta (3*H) + output (H*128 + 128)
        let trunk_fixed = h * h + 3 * h + 3 * h;
        let output_tail = h * NUM_OUTPUTS + NUM_OUTPUTS;
        let fixed = trunk_fixed + output_tail;

        if total > fixed && (total - fixed) % h == 0 {
            let obs_dim = (total - fixed) / h;
            return Self::from_floats(&floats, hidden, obs_dim);
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cannot infer obs_dim: {} floats, hidden={} (expected layout doesn't fit)",
                total, h,
            ),
        ))
    }

    /// Construct from a flat array of f32 weights.
    pub fn from_floats(floats: &[f32], hidden: usize, obs_dim: usize) -> std::io::Result<Self> {
        let in_dims = [obs_dim, hidden];

        let mut expected = 0;
        for &in_dim in &in_dims {
            expected += in_dim * hidden + hidden + hidden + hidden; // W + b + gamma + beta
        }
        expected += hidden * NUM_OUTPUTS + NUM_OUTPUTS; // output W + b

        if floats.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "weight file has {} floats, expected {} for obs_dim={}, hidden={}",
                    floats.len(), expected, obs_dim, hidden,
                ),
            ));
        }

        let mut offset = 0;
        let mut w = [Vec::new(), Vec::new()];
        let mut b = [Vec::new(), Vec::new()];
        let mut gamma = [Vec::new(), Vec::new()];
        let mut beta = [Vec::new(), Vec::new()];

        for layer in 0..NUM_LAYERS {
            let in_dim = in_dims[layer];
            let w_size = in_dim * hidden;
            w[layer] = floats[offset..offset + w_size].to_vec();
            offset += w_size;
            b[layer] = floats[offset..offset + hidden].to_vec();
            offset += hidden;
            gamma[layer] = floats[offset..offset + hidden].to_vec();
            offset += hidden;
            beta[layer] = floats[offset..offset + hidden].to_vec();
            offset += hidden;
        }

        let w_out = floats[offset..offset + hidden * NUM_OUTPUTS].to_vec();
        offset += hidden * NUM_OUTPUTS;
        let b_out = floats[offset..offset + NUM_OUTPUTS].to_vec();
        debug_assert_eq!(offset + NUM_OUTPUTS, floats.len());

        Ok(BeliefNet {
            w, b, gamma, beta,
            w_out,
            b_out,
            obs_dim,
            hidden,
            in_dims,
            scratch_a: vec![0.0; hidden],
            scratch_b: vec![0.0; hidden],
        })
    }

    /// Forward pass: compute 128 raw logits (32 cards × 4 player slots).
    #[inline]
    pub fn evaluate(&mut self, obs: &[f32]) -> [f32; NUM_OUTPUTS] {
        debug_assert_eq!(obs.len(), self.obs_dim);

        // Layer 0: scratch_a = ReLU(LN(W0 * obs + b0))
        linear(&self.w[0], &self.b[0], obs, &mut self.scratch_a, self.in_dims[0], self.hidden);
        layer_norm(&mut self.scratch_a, &self.gamma[0], &self.beta[0], self.hidden);
        relu(&mut self.scratch_a);

        // Layer 1: scratch_b = ReLU(LN(W1 * scratch_a + b1))
        linear(&self.w[1], &self.b[1], &self.scratch_a, &mut self.scratch_b, self.hidden, self.hidden);
        layer_norm(&mut self.scratch_b, &self.gamma[1], &self.beta[1], self.hidden);
        relu(&mut self.scratch_b);

        // Output: 128 logits
        let mut logits = [0.0f32; NUM_OUTPUTS];
        for i in 0..NUM_OUTPUTS {
            let row_start = i * self.hidden;
            let mut sum = self.b_out[i];
            for j in 0..self.hidden {
                sum += self.w_out[row_start + j] * self.scratch_b[j];
            }
            logits[i] = sum;
        }

        logits
    }

    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }
}

/// Convert raw 128 logits → [[f32; 32]; 4] normalized weights.
///
/// Logit slots are player-relative: 0=me, 1=left, 2=partner, 3=right.
/// For each unknown card, applies softmax over 4 relative slots, zeros the
/// observer slot (rel=0), renormalizes, then remaps to absolute player indices.
///
/// Output layout: `weights[player][card]` = probability card is in player's hand.
/// Player indices are absolute (0-3).
pub fn belief_to_weights(
    logits: &[f32; NUM_OUTPUTS],
    state: &GameState,
    observer: u8,
) -> [[f32; 32]; 4] {
    let mut weights = [[0.0f32; 32]; 4];

    // Relative slot → absolute player mapping
    let abs_players = [
        observer as usize,
        ((observer + 1) % 4) as usize,
        ((observer + 2) % 4) as usize,
        ((observer + 3) % 4) as usize,
    ];

    // Known cards: observer's hand + all played cards (including current trick)
    let observer_hand = state.hands[observer as usize];
    let mut played = state.played_cards;
    for i in 0..4 {
        let c = state.current_trick[i];
        if c != card::EMPTY {
            played |= 1u32 << c;
        }
    }
    let known = observer_hand | played;

    for card_idx in 0..32u32 {
        if known & (1 << card_idx) != 0 {
            // Known card: zero weight for all players
            continue;
        }

        // Per-card softmax over 4 relative slots
        let base = card_idx as usize * 4;
        let mut max_logit = f32::NEG_INFINITY;
        for rel in 0..4 {
            if logits[base + rel] > max_logit {
                max_logit = logits[base + rel];
            }
        }

        let mut exp_sum = 0.0f32;
        let mut exps = [0.0f32; 4];
        for rel in 0..4 {
            exps[rel] = (logits[base + rel] - max_logit).exp();
            exp_sum += exps[rel];
        }

        // Map relative softmax to absolute players, zeroing observer (rel=0)
        for rel in 0..4 {
            let abs_p = abs_players[rel];
            if rel == 0 {
                // Observer can't hold unknown cards (they'd be in our hand)
                weights[abs_p][card_idx as usize] = 0.0;
            } else {
                weights[abs_p][card_idx as usize] = exps[rel] / exp_sum;
            }
        }

        // Renormalize after zeroing observer
        let mut total = 0.0f32;
        for p in 0..4 {
            total += weights[p][card_idx as usize];
        }
        if total > 0.0 {
            for p in 0..4 {
                weights[p][card_idx as usize] /= total;
            }
        }
    }

    weights
}

#[inline]
fn linear(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32], in_dim: usize, out_dim: usize) {
    for i in 0..out_dim {
        let row_start = i * in_dim;
        let mut sum = b[i];
        for j in 0..in_dim {
            sum += w[row_start + j] * x[j];
        }
        out[i] = sum;
    }
}

#[inline]
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], dim: usize) {
    let mut mean = 0.0f32;
    for i in 0..dim {
        mean += x[i];
    }
    mean /= dim as f32;

    let mut var = 0.0f32;
    for i in 0..dim {
        let d = x[i] - mean;
        var += d * d;
    }
    var /= dim as f32;

    let inv_std = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..dim {
        x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

#[inline]
fn relu(x: &mut [f32]) {
    for v in x.iter_mut() {
        if *v < 0.0 {
            *v = 0.0;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief_obs::BELIEF_OBS_DIM;

    fn build_test_weights(obs_dim: usize, hidden: usize) -> Vec<f32> {
        let mut floats = Vec::new();

        // Layer 0: W, b, gamma, beta
        floats.extend(vec![0.0f32; obs_dim * hidden]);
        floats.extend(vec![0.0f32; hidden]); // b
        floats.extend(vec![1.0f32; hidden]); // gamma
        floats.extend(vec![0.0f32; hidden]); // beta

        // Layer 1: identity
        let mut w1 = vec![0.0f32; hidden * hidden];
        for i in 0..hidden {
            w1[i * hidden + i] = 1.0;
        }
        floats.extend(w1);
        floats.extend(vec![0.0f32; hidden]); // b
        floats.extend(vec![1.0f32; hidden]); // gamma
        floats.extend(vec![0.0f32; hidden]); // beta

        // Output: W, b
        floats.extend(vec![0.0f32; hidden * NUM_OUTPUTS]);
        floats.extend(vec![0.0f32; NUM_OUTPUTS]);

        floats
    }

    #[test]
    fn test_from_floats_tiny() {
        let hidden = 4;
        let floats = build_test_weights(BELIEF_OBS_DIM, hidden);
        let mut net = BeliefNet::from_floats(&floats, hidden, BELIEF_OBS_DIM).unwrap();
        assert_eq!(net.obs_dim(), BELIEF_OBS_DIM);
        assert_eq!(net.hidden(), hidden);

        let obs = vec![0.0f32; BELIEF_OBS_DIM];
        let logits = net.evaluate(&obs);
        // With zero weights, all logits should be ~0
        for &v in &logits {
            assert!(v.abs() < 1e-4, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_evaluate_output_shape() {
        let hidden = 2;
        let floats = build_test_weights(BELIEF_OBS_DIM, hidden);
        let mut net = BeliefNet::from_floats(&floats, hidden, BELIEF_OBS_DIM).unwrap();

        let obs = vec![0.5f32; BELIEF_OBS_DIM];
        let logits = net.evaluate(&obs);
        assert_eq!(logits.len(), 128);
    }

    #[test]
    fn test_belief_to_weights_uniform() {
        // With uniform logits (all zero), should get equal weights for non-observer players
        let logits = [0.0f32; NUM_OUTPUTS];

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);

        // Observer=0, hand=0xFF
        // For card 8 (in player 1's hand): unknown to observer 0
        let weights = belief_to_weights(&logits, &state, 0);

        // Card 0 is in observer's hand → known → all zero
        for p in 0..4 {
            assert_eq!(weights[p][0], 0.0, "known card should have 0 weight");
        }

        // Card 8 (unknown to observer 0): should have ~equal weight for players 1,2,3
        assert_eq!(weights[0][8], 0.0, "observer can't hold unknown card");
        let expected = 1.0 / 3.0;
        assert!((weights[1][8] - expected).abs() < 1e-5);
        assert!((weights[2][8] - expected).abs() < 1e-5);
        assert!((weights[3][8] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_belief_to_weights_biased() {
        // With biased logits, should get skewed weights
        let mut logits = [0.0f32; NUM_OUTPUTS];

        // Card 8: strongly favor player 1
        logits[8 * 4 + 0] = -10.0; // observer (will be zeroed anyway)
        logits[8 * 4 + 1] = 10.0;  // player 1 (high)
        logits[8 * 4 + 2] = 0.0;   // player 2
        logits[8 * 4 + 3] = 0.0;   // player 3

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let weights = belief_to_weights(&logits, &state, 0);

        // Player 1 should have much higher weight than 2,3 for card 8
        assert!(weights[1][8] > 0.9, "player 1 should have >90% weight, got {}", weights[1][8]);
        assert!(weights[2][8] < 0.05);
        assert!(weights[3][8] < 0.05);
    }

    #[test]
    fn test_belief_to_weights_sum_to_one() {
        let logits = [0.5f32; NUM_OUTPUTS]; // arbitrary uniform logits

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let weights = belief_to_weights(&logits, &state, 0);

        // For each unknown card, weights should sum to 1.0
        let known = state.hands[0] | state.played_cards;
        for card_idx in 0..32 {
            if known & (1 << card_idx) != 0 {
                continue;
            }
            let sum: f32 = (0..4).map(|p| weights[p][card_idx]).sum();
            assert!((sum - 1.0).abs() < 1e-5, "card {} weights sum to {}", card_idx, sum);
        }
    }

    #[test]
    fn test_belief_to_weights_relative_remapping() {
        // Verify that relative logit slots are correctly mapped to absolute players
        // when observer != 0.
        let mut logits = [0.0f32; NUM_OUTPUTS];

        // Observer=2, so relative slots map as:
        //   slot 0 (me)      → abs player 2
        //   slot 1 (left)    → abs player 3
        //   slot 2 (partner) → abs player 0
        //   slot 3 (right)   → abs player 1
        //
        // Card 8 (in player 1's hand, unknown to observer 2):
        // Set high logit for slot 3 (= right = abs player 1)
        logits[8 * 4 + 0] = -10.0; // me (zeroed anyway)
        logits[8 * 4 + 1] = 0.0;   // left  (abs 3)
        logits[8 * 4 + 2] = 0.0;   // partner (abs 0)
        logits[8 * 4 + 3] = 10.0;  // right (abs 1) — high

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let weights = belief_to_weights(&logits, &state, 2);

        // Card 16 is in observer 2's hand → known → all zero
        for p in 0..4 {
            assert_eq!(weights[p][16], 0.0, "observer's card should have 0 weight");
        }

        // Card 8: abs player 1 (slot 3 = right) should dominate
        assert_eq!(weights[2][8], 0.0, "observer can't hold unknown card");
        assert!(weights[1][8] > 0.9, "abs player 1 should have >90% weight, got {}", weights[1][8]);
        assert!(weights[0][8] < 0.05, "abs player 0 weight too high: {}", weights[0][8]);
        assert!(weights[3][8] < 0.05, "abs player 3 weight too high: {}", weights[3][8]);
    }

    #[test]
    fn test_belief_to_weights_uniform_nonzero_observer() {
        // Uniform logits with observer=3: each non-observer should get ~1/3
        let logits = [0.0f32; NUM_OUTPUTS];

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let weights = belief_to_weights(&logits, &state, 3);

        // Card 0 (in player 0's hand, unknown to observer 3)
        assert_eq!(weights[3][0], 0.0, "observer can't hold unknown card");
        let expected = 1.0 / 3.0;
        assert!((weights[0][0] - expected).abs() < 1e-5);
        assert!((weights[1][0] - expected).abs() < 1e-5);
        assert!((weights[2][0] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_invalid_floats_len() {
        let floats = vec![0.0f32; 100]; // Too small
        let result = BeliefNet::from_floats(&floats, 4, BELIEF_OBS_DIM);
        assert!(result.is_err());
    }
}
