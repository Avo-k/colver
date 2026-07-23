/// BeliefNet: pure Rust inference for the NN belief network (card location prediction).
///
/// Architecture (standard): 330 → 512 (LN+ReLU) → 512 (LN+ReLU) → output
///
/// V3 (3-class): 96 logits = 32 cards × 3 player slots (left/partner/right).
/// Legacy (4-class): 128 logits = 32 cards × 4 player slots.
/// Auto-detected from weight file size.
///
/// Caller applies per-card softmax via `belief_to_weights()`.
///
/// Weight file layout (contiguous little-endian f32):
///   For each of 2 hidden layers:
///     W: in_dim × H (row-major), b: H, gamma: H, beta: H
///   Output layer:
///     W: H × num_outputs (row-major), b: num_outputs

use crate::card;
use crate::state::GameState;

use crate::nn_kernels::{self, dot};

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
    num_classes: usize, // 3 (V3: left/partner/right) or 4 (legacy: me/left/partner/right)
    num_outputs: usize, // 32 * num_classes
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
    /// Auto-detects num_classes (3 or 4) from weight file size.
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

        // Trunk fixed part: layer1 (H*H + 3*H) + layer0 b/gamma/beta (3*H)
        let trunk_fixed = h * h + 3 * h + 3 * h;

        // Try both 3-class (V3) and 4-class (legacy), prefer the one matching a known obs_dim
        use crate::belief_obs::{BELIEF_OBS_DIM, BELIEF_OBS_DIM_V2, BELIEF_OBS_DIM_V3, BID_BELIEF_OBS_DIM};
        let known_dims = [BELIEF_OBS_DIM, BELIEF_OBS_DIM_V2, BELIEF_OBS_DIM_V3, BID_BELIEF_OBS_DIM];

        let mut candidates: Vec<(usize, usize)> = Vec::new(); // (nc, obs_dim)
        for &nc in &[3usize, 4usize] {
            let num_out = 32 * nc;
            let output_tail = h * num_out + num_out;
            let fixed = trunk_fixed + output_tail;

            if total > fixed && (total - fixed) % h == 0 {
                let obs_dim = (total - fixed) / h;
                candidates.push((nc, obs_dim));
            }
        }

        // Prefer candidate matching a known obs_dim; among ties prefer 3-class (new format)
        if let Some(&(nc, obs_dim)) = candidates
            .iter()
            .find(|(_, od)| known_dims.contains(od))
            .or(candidates.first())
        {
            return Self::from_floats_with_classes(&floats, hidden, obs_dim, nc);
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cannot infer obs_dim: {} floats, hidden={} (expected layout doesn't fit)",
                total, h,
            ),
        ))
    }

    /// Construct from a flat array of f32 weights (legacy 4-class).
    pub fn from_floats(floats: &[f32], hidden: usize, obs_dim: usize) -> std::io::Result<Self> {
        Self::from_floats_with_classes(floats, hidden, obs_dim, 4)
    }

    /// Construct from a flat array of f32 weights with explicit num_classes.
    pub fn from_floats_with_classes(
        floats: &[f32],
        hidden: usize,
        obs_dim: usize,
        num_classes: usize,
    ) -> std::io::Result<Self> {
        let num_outputs = 32 * num_classes;
        let in_dims = [obs_dim, hidden];

        let mut expected = 0;
        for &in_dim in &in_dims {
            expected += in_dim * hidden + hidden + hidden + hidden; // W + b + gamma + beta
        }
        expected += hidden * num_outputs + num_outputs; // output W + b

        if floats.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "weight file has {} floats, expected {} for obs_dim={}, hidden={}, num_classes={}",
                    floats.len(), expected, obs_dim, hidden, num_classes,
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

        let w_out = floats[offset..offset + hidden * num_outputs].to_vec();
        offset += hidden * num_outputs;
        let b_out = floats[offset..offset + num_outputs].to_vec();
        debug_assert_eq!(offset + num_outputs, floats.len());

        Ok(BeliefNet {
            w, b, gamma, beta,
            w_out,
            b_out,
            obs_dim,
            hidden,
            num_classes,
            num_outputs,
            in_dims,
            scratch_a: vec![0.0; hidden],
            scratch_b: vec![0.0; hidden],
        })
    }

    /// Forward pass: compute raw logits (32 cards × num_classes player slots).
    ///
    /// Returns `[f32; 128]`:
    /// - 3-class: fills `[0..96]` with native layout (card*3 + class), rest zeroed.
    /// - 4-class: fills all 128 as before.
    #[inline]
    pub fn evaluate(&mut self, obs: &[f32]) -> [f32; 128] {
        debug_assert_eq!(obs.len(), self.obs_dim);

        // Layer 0: scratch_a = ReLU(LN(W0 * obs + b0))
        linear(&self.w[0], &self.b[0], obs, &mut self.scratch_a, self.in_dims[0], self.hidden);
        layer_norm(&mut self.scratch_a, &self.gamma[0], &self.beta[0], self.hidden);
        relu(&mut self.scratch_a);

        // Layer 1: scratch_b = ReLU(LN(W1 * scratch_a + b1))
        linear(&self.w[1], &self.b[1], &self.scratch_a, &mut self.scratch_b, self.hidden, self.hidden);
        layer_norm(&mut self.scratch_b, &self.gamma[1], &self.beta[1], self.hidden);
        relu(&mut self.scratch_b);

        // Output: num_outputs logits
        let mut logits = [0.0f32; 128];
        let trunk = &self.scratch_b[..self.hidden];
        for i in 0..self.num_outputs {
            logits[i] =
                self.b_out[i] + dot(&self.w_out[i * self.hidden..(i + 1) * self.hidden], trunk);
        }

        logits
    }

    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn num_classes(&self) -> usize {
        self.num_classes
    }
}

/// Convert raw logits → [[f32; 32]; 4] normalized weights.
///
/// For `num_classes=3` (V3): logit slots are 0=left, 1=partner, 2=right.
/// For `num_classes=4` (legacy): logit slots are 0=me, 1=left, 2=partner, 3=right;
///   observer slot (rel=0) is zeroed and renormalized.
///
/// Output layout: `weights[player][card]` = probability card is in player's hand.
/// Player indices are absolute (0-3).
pub fn belief_to_weights(
    logits: &[f32; 128],
    num_classes: usize,
    state: &GameState,
    observer: u8,
) -> [[f32; 32]; 4] {
    let mut weights = [[0.0f32; 32]; 4];

    // Absolute players for non-observer slots: left, partner, right
    let abs_left = ((observer + 1) % 4) as usize;
    let abs_partner = ((observer + 2) % 4) as usize;
    let abs_right = ((observer + 3) % 4) as usize;

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
            continue;
        }

        if num_classes == 3 {
            // V3: 3 slots = left/partner/right, softmax directly
            let base = card_idx as usize * 3;
            let mut max_logit = f32::NEG_INFINITY;
            for s in 0..3 {
                if logits[base + s] > max_logit {
                    max_logit = logits[base + s];
                }
            }
            let mut exp_sum = 0.0f32;
            let mut exps = [0.0f32; 3];
            for s in 0..3 {
                exps[s] = (logits[base + s] - max_logit).exp();
                exp_sum += exps[s];
            }
            weights[abs_left][card_idx as usize] = exps[0] / exp_sum;
            weights[abs_partner][card_idx as usize] = exps[1] / exp_sum;
            weights[abs_right][card_idx as usize] = exps[2] / exp_sum;
        } else {
            // Legacy 4-class: softmax over 4 slots, zero observer, renormalize
            let base = card_idx as usize * 4;
            let abs_players = [
                observer as usize, abs_left, abs_partner, abs_right,
            ];
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
            for rel in 0..4 {
                let abs_p = abs_players[rel];
                if rel == 0 {
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
    }

    weights
}

#[inline]
fn linear(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32], in_dim: usize, out_dim: usize) {
    nn_kernels::linear(w, b, x, out, in_dim, out_dim);
}

#[inline]
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], dim: usize) {
    nn_kernels::layer_norm(x, gamma, beta, dim, LN_EPS);
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
        build_test_weights_nc(obs_dim, hidden, 3)
    }

    fn build_test_weights_nc(obs_dim: usize, hidden: usize, num_classes: usize) -> Vec<f32> {
        let num_outputs = 32 * num_classes;
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
        floats.extend(vec![0.0f32; hidden * num_outputs]);
        floats.extend(vec![0.0f32; num_outputs]);

        floats
    }

    #[test]
    fn test_from_floats_tiny() {
        let hidden = 4;
        let floats = build_test_weights(BELIEF_OBS_DIM, hidden);
        let mut net = BeliefNet::from_floats_with_classes(&floats, hidden, BELIEF_OBS_DIM, 3).unwrap();
        assert_eq!(net.obs_dim(), BELIEF_OBS_DIM);
        assert_eq!(net.hidden(), hidden);
        assert_eq!(net.num_classes(), 3);

        let obs = vec![0.0f32; BELIEF_OBS_DIM];
        let logits = net.evaluate(&obs);
        // With zero weights, first 96 logits should be ~0
        for &v in &logits[..96] {
            assert!(v.abs() < 1e-4, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_evaluate_output_shape() {
        let hidden = 2;
        let floats = build_test_weights(BELIEF_OBS_DIM, hidden);
        let mut net = BeliefNet::from_floats_with_classes(&floats, hidden, BELIEF_OBS_DIM, 3).unwrap();

        let obs = vec![0.5f32; BELIEF_OBS_DIM];
        let logits = net.evaluate(&obs);
        assert_eq!(logits.len(), 128);
    }

    #[test]
    fn test_belief_to_weights_3class_uniform() {
        // 3-class: uniform logits → equal weights for left/partner/right
        let logits = [0.0f32; 128];

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let weights = belief_to_weights(&logits, 3, &state, 0);

        // Card 0 is in observer's hand → known → all zero
        for p in 0..4 {
            assert_eq!(weights[p][0], 0.0, "known card should have 0 weight");
        }

        // Card 8 (unknown to observer 0): ~1/3 each for players 1,2,3
        assert_eq!(weights[0][8], 0.0, "observer can't hold unknown card");
        let expected = 1.0 / 3.0;
        assert!((weights[1][8] - expected).abs() < 1e-5);
        assert!((weights[2][8] - expected).abs() < 1e-5);
        assert!((weights[3][8] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_belief_to_weights_3class_biased() {
        // 3-class: biased logits for card 8
        let mut logits = [0.0f32; 128];

        // Card 8: 3-class layout: base=8*3=24, slots 0=left, 1=partner, 2=right
        // Observer=0: left=P1, partner=P2, right=P3
        logits[8 * 3 + 0] = 10.0; // left (P1) — high
        logits[8 * 3 + 1] = 0.0;  // partner (P2)
        logits[8 * 3 + 2] = 0.0;  // right (P3)

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let weights = belief_to_weights(&logits, 3, &state, 0);

        assert!(weights[1][8] > 0.9, "P1 should have >90% weight, got {}", weights[1][8]);
        assert!(weights[2][8] < 0.05);
        assert!(weights[3][8] < 0.05);
    }

    #[test]
    fn test_belief_to_weights_3class_sum_to_one() {
        let logits = [0.5f32; 128];

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let weights = belief_to_weights(&logits, 3, &state, 0);

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
    fn test_belief_to_weights_3class_nonzero_observer() {
        // Observer=2: left=P3, partner=P0, right=P1
        let mut logits = [0.0f32; 128];

        // Card 8: high logit for slot 2 (right = P1)
        logits[8 * 3 + 0] = 0.0;   // left (P3)
        logits[8 * 3 + 1] = 0.0;   // partner (P0)
        logits[8 * 3 + 2] = 10.0;  // right (P1) — high

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let weights = belief_to_weights(&logits, 3, &state, 2);

        // Card 16 in observer 2's hand → known
        for p in 0..4 {
            assert_eq!(weights[p][16], 0.0);
        }

        // Card 8: P1 (right) should dominate
        assert_eq!(weights[2][8], 0.0, "observer can't hold unknown card");
        assert!(weights[1][8] > 0.9, "P1 should have >90%, got {}", weights[1][8]);
    }

    #[test]
    fn test_belief_to_weights_4class_legacy() {
        // Legacy 4-class: uniform → 1/3 each for non-observer
        let logits = [0.0f32; 128];

        let hands = [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000];
        let state = GameState::new(0, hands);
        let weights = belief_to_weights(&logits, 4, &state, 0);

        assert_eq!(weights[0][8], 0.0);
        let expected = 1.0 / 3.0;
        assert!((weights[1][8] - expected).abs() < 1e-5);
        assert!((weights[2][8] - expected).abs() < 1e-5);
        assert!((weights[3][8] - expected).abs() < 1e-5);
    }

    #[test]
    fn test_auto_detect_3class() {
        let hidden = 4;
        let floats = build_test_weights_nc(BELIEF_OBS_DIM, hidden, 3);
        // load_with_hidden should auto-detect 3-class
        let tmp = "/tmp/test_belief_3class.bin";
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(tmp, &bytes).unwrap();
        let net = BeliefNet::load_with_hidden(tmp, hidden).unwrap();
        assert_eq!(net.num_classes(), 3);
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_auto_detect_4class() {
        let hidden = 4;
        let floats = build_test_weights_nc(BELIEF_OBS_DIM, hidden, 4);
        let tmp = "/tmp/test_belief_4class.bin";
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(tmp, &bytes).unwrap();
        let net = BeliefNet::load_with_hidden(tmp, hidden).unwrap();
        assert_eq!(net.num_classes(), 4);
        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_invalid_floats_len() {
        let floats = vec![0.0f32; 100]; // Too small
        let result = BeliefNet::from_floats(&floats, 4, BELIEF_OBS_DIM);
        assert!(result.is_err());
    }

    #[test]
    fn test_bid_belief_net_obs_dim_108() {
        use crate::belief_obs::BID_BELIEF_OBS_DIM;

        // Create a minimal weight buffer for obs_dim=108, hidden=256, 3-class
        let obs_dim = BID_BELIEF_OBS_DIM;
        assert_eq!(obs_dim, 108);
        let hidden = 256usize;
        let num_classes = 3usize;

        let floats = build_test_weights_nc(obs_dim, hidden, num_classes);
        let mut net =
            BeliefNet::from_floats_with_classes(&floats, hidden, obs_dim, num_classes).unwrap();

        assert_eq!(net.obs_dim(), obs_dim);
        assert_eq!(net.hidden(), hidden);
        assert_eq!(net.num_classes(), num_classes);

        // Forward pass should not panic
        let obs = vec![0.0f32; obs_dim];
        let logits = net.evaluate(&obs);
        for i in 0..96 {
            assert!(!logits[i].is_nan());
        }
    }

    #[test]
    fn test_bid_belief_net_auto_detect_108() {
        use crate::belief_obs::BID_BELIEF_OBS_DIM;

        // Verify auto-detection from weight file picks up obs_dim=108
        let hidden = 4;
        let floats = build_test_weights_nc(BID_BELIEF_OBS_DIM, hidden, 3);
        let tmp = "/tmp/test_belief_bid_108.bin";
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(tmp, &bytes).unwrap();
        let net = BeliefNet::load_with_hidden(tmp, hidden).unwrap();
        assert_eq!(net.obs_dim(), BID_BELIEF_OBS_DIM);
        assert_eq!(net.num_classes(), 3);
        std::fs::remove_file(tmp).ok();
    }
}
