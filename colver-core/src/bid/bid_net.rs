/// BidNet: pure Rust inference for the NN bidding model.
///
/// Architecture (dueling): obs_dim → [H (LN+ReLU)]×N
///                            → Value head: H → 1 (V)
///                            → Advantage head: H → 43 (A)
///                            → Q = V + (A - mean(A))
///
/// Architecture (standard): obs_dim → [H (LN+ReLU)]×N → 43
///
/// where H = hidden size, N = number of layers, LN = LayerNorm.
/// Layer count auto-detected from weight file size.
///
/// Weight file layout (per layer): W: in_dim × H, b: H, gamma: H, beta: H

use crate::nn_kernels::{self, dot};

const NUM_ACTIONS: usize = 43;
const LN_EPS: f32 = 1e-5;

pub struct BidNet {
    w: Vec<Vec<f32>>,
    b: Vec<Vec<f32>>,
    gamma: Vec<Vec<f32>>,
    beta: Vec<Vec<f32>>,
    // Standard output (dueling=false)
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    // Dueling heads (dueling=true)
    dueling: bool,
    w_value: Vec<f32>,
    b_value: f32,
    w_adv: Vec<f32>,
    b_adv: Vec<f32>,
    // Dimensions
    obs_dim: usize,
    hidden: usize,
    layers: usize,
    in_dims: Vec<usize>,
    // Scratch buffers
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
}

impl BidNet {
    /// Load weights from a raw binary file with default hidden size (256).
    pub fn load(path: &str) -> std::io::Result<Self> {
        // Try common hidden sizes for auto-detection
        for &h in &[256, 512, 1024] {
            if let Ok(net) = Self::load_with_hidden(path, h) {
                return Ok(net);
            }
        }
        Self::load_with_hidden(path, 256) // final attempt for error message
    }

    /// Load weights with custom hidden size. Architecture auto-detected.
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
        let standard_tail = h * NUM_ACTIONS + NUM_ACTIONS;
        let dueling_tail = h + 1 + h * NUM_ACTIONS + NUM_ACTIONS;
        let known_dims = [108, 110, 113, 114];

        // Collect all valid (layers, dueling, obs_dim) candidates
        let mut candidates = Vec::new();
        for layers in 2..=4 {
            let trunk_fixed = (layers - 1) * (h * h + 3 * h) + 3 * h;

            for &(tail, dueling) in &[(dueling_tail, true), (standard_tail, false)] {
                let fixed = trunk_fixed + tail;
                if total > fixed && (total - fixed) % h == 0 {
                    let obs_dim = (total - fixed) / h;
                    if obs_dim > 0 && obs_dim <= 500 {
                        candidates.push((layers, dueling, obs_dim));
                    }
                }
            }
        }

        // Prefer: known obs_dim first, then more layers, then dueling
        if let Some(&(layers, dueling, obs_dim)) = candidates.iter()
            .filter(|&&(_, _, od)| known_dims.contains(&od))
            .max_by_key(|&&(l, d, _)| (l, d as usize))
        {
            return Self::from_floats_with_layers(&floats, hidden, obs_dim, dueling, layers);
        }
        // Fallback: any valid candidate, prefer more layers
        if let Some(&(layers, dueling, obs_dim)) = candidates.iter()
            .max_by_key(|&&(l, d, _)| (l, d as usize))
        {
            return Self::from_floats_with_layers(&floats, hidden, obs_dim, dueling, layers);
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cannot infer architecture: {} floats, hidden={} (tried 2-4 layers, standard+dueling)",
                total, h,
            ),
        ))
    }

    /// Construct from flat weights (default 2 layers for backward compat).
    pub fn from_floats(floats: &[f32], hidden: usize, obs_dim: usize, dueling: bool) -> std::io::Result<Self> {
        Self::from_floats_with_layers(floats, hidden, obs_dim, dueling, 2)
    }

    /// Construct from flat weights with explicit layer count.
    pub fn from_floats_with_layers(
        floats: &[f32],
        hidden: usize,
        obs_dim: usize,
        dueling: bool,
        layers: usize,
    ) -> std::io::Result<Self> {
        let mut in_dims = vec![obs_dim];
        for _ in 1..layers {
            in_dims.push(hidden);
        }

        let mut expected = 0;
        for &in_dim in &in_dims {
            expected += in_dim * hidden + hidden + hidden + hidden;
        }
        if dueling {
            expected += hidden + 1;
            expected += hidden * NUM_ACTIONS + NUM_ACTIONS;
        } else {
            expected += hidden * NUM_ACTIONS + NUM_ACTIONS;
        }

        if floats.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "weight file has {} floats, expected {} for obs_dim={}, hidden={}, layers={}, dueling={}",
                    floats.len(), expected, obs_dim, hidden, layers, dueling,
                ),
            ));
        }

        let mut offset = 0;
        let mut w = Vec::with_capacity(layers);
        let mut b = Vec::with_capacity(layers);
        let mut gamma = Vec::with_capacity(layers);
        let mut beta = Vec::with_capacity(layers);

        for layer in 0..layers {
            let in_dim = in_dims[layer];
            let w_size = in_dim * hidden;
            w.push(floats[offset..offset + w_size].to_vec());
            offset += w_size;
            b.push(floats[offset..offset + hidden].to_vec());
            offset += hidden;
            gamma.push(floats[offset..offset + hidden].to_vec());
            offset += hidden;
            beta.push(floats[offset..offset + hidden].to_vec());
            offset += hidden;
        }

        if dueling {
            let w_value = floats[offset..offset + hidden].to_vec();
            offset += hidden;
            let b_value = floats[offset];
            offset += 1;
            let w_adv = floats[offset..offset + hidden * NUM_ACTIONS].to_vec();
            offset += hidden * NUM_ACTIONS;
            let b_adv = floats[offset..offset + NUM_ACTIONS].to_vec();
            offset += NUM_ACTIONS;
            debug_assert_eq!(offset, floats.len());

            Ok(BidNet {
                w, b, gamma, beta,
                w_out: Vec::new(),
                b_out: Vec::new(),
                dueling: true,
                w_value,
                b_value,
                w_adv,
                b_adv,
                obs_dim,
                hidden,
                layers,
                in_dims,
                scratch_a: vec![0.0; hidden],
                scratch_b: vec![0.0; hidden],
            })
        } else {
            let w_out = floats[offset..offset + hidden * NUM_ACTIONS].to_vec();
            offset += hidden * NUM_ACTIONS;
            let b_out = floats[offset..offset + NUM_ACTIONS].to_vec();
            debug_assert_eq!(offset + NUM_ACTIONS, floats.len());

            Ok(BidNet {
                w, b, gamma, beta,
                w_out,
                b_out,
                dueling: false,
                w_value: Vec::new(),
                b_value: 0.0,
                w_adv: Vec::new(),
                b_adv: Vec::new(),
                obs_dim,
                hidden,
                layers,
                in_dims,
                scratch_a: vec![0.0; hidden],
                scratch_b: vec![0.0; hidden],
            })
        }
    }

    /// Forward pass: compute Q-values for all 43 bid actions.
    #[inline]
    pub fn evaluate(&mut self, obs: &[f32]) -> [f32; NUM_ACTIONS] {
        debug_assert_eq!(obs.len(), self.obs_dim);

        // Layer 0: obs → scratch_a
        linear(&self.w[0], &self.b[0], obs, &mut self.scratch_a, self.in_dims[0], self.hidden);
        layer_norm(&mut self.scratch_a, &self.gamma[0], &self.beta[0], self.hidden);
        relu(&mut self.scratch_a);

        // Remaining layers alternate between scratch buffers
        for layer in 1..self.layers {
            if layer % 2 == 1 {
                // scratch_a → scratch_b
                linear(&self.w[layer], &self.b[layer], &self.scratch_a, &mut self.scratch_b, self.hidden, self.hidden);
                layer_norm(&mut self.scratch_b, &self.gamma[layer], &self.beta[layer], self.hidden);
                relu(&mut self.scratch_b);
            } else {
                // scratch_b → scratch_a
                linear(&self.w[layer], &self.b[layer], &self.scratch_b, &mut self.scratch_a, self.hidden, self.hidden);
                layer_norm(&mut self.scratch_a, &self.gamma[layer], &self.beta[layer], self.hidden);
                relu(&mut self.scratch_a);
            }
        }

        // Final output is in scratch_b if layers is even, scratch_a if odd
        let trunk_out = if self.layers % 2 == 0 { &self.scratch_b } else { &self.scratch_a };

        if self.dueling {
            let trunk = &trunk_out[..self.hidden];
            let v = self.b_value + dot(&self.w_value[..self.hidden], trunk);

            let mut q = [0.0f32; NUM_ACTIONS];
            let mut adv_sum = 0.0f32;
            for i in 0..NUM_ACTIONS {
                let a = self.b_adv[i]
                    + dot(&self.w_adv[i * self.hidden..(i + 1) * self.hidden], trunk);
                q[i] = a;
                adv_sum += a;
            }

            let adv_mean = adv_sum / NUM_ACTIONS as f32;
            for i in 0..NUM_ACTIONS {
                q[i] = v + q[i] - adv_mean;
            }
            q
        } else {
            let mut q = [0.0f32; NUM_ACTIONS];
            linear(&self.w_out, &self.b_out, trunk_out, &mut q, self.hidden, NUM_ACTIONS);
            q
        }
    }

    /// Pick best legal action given a u64 legal mask.
    pub fn best_action(&mut self, obs: &[f32], legal_mask: u64) -> (u8, Vec<(u8, f32)>) {
        let q = self.evaluate(obs);

        let mut best_action = 0u8;
        let mut best_q = f32::NEG_INFINITY;
        let mut legal_q = Vec::new();

        let mut mask = legal_mask;
        while mask != 0 {
            let bit = mask.trailing_zeros() as u8;
            if (bit as usize) < NUM_ACTIONS {
                let q_val = q[bit as usize];
                legal_q.push((bit, q_val));
                if q_val > best_q {
                    best_q = q_val;
                    best_action = bit;
                }
            }
            mask &= mask - 1;
        }

        (best_action, legal_q)
    }

    /// Pick best legal action (no allocation — returns action only).
    pub fn best_action_fast(&mut self, obs: &[f32], legal_mask: u64) -> u8 {
        let q = self.evaluate(obs);
        let mut best_action = 0u8;
        let mut best_q = f32::NEG_INFINITY;
        let mut mask = legal_mask;
        while mask != 0 {
            let bit = mask.trailing_zeros() as u8;
            if (bit as usize) < NUM_ACTIONS {
                let q_val = q[bit as usize];
                if q_val > best_q {
                    best_q = q_val;
                    best_action = bit;
                }
            }
            mask &= mask - 1;
        }
        best_action
    }

    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    pub fn hidden(&self) -> usize {
        self.hidden
    }

    pub fn layers(&self) -> usize {
        self.layers
    }

    pub fn is_dueling(&self) -> bool {
        self.dueling
    }
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

    const TEST_OBS_DIM: usize = 114;

    fn build_standard_weights(obs_dim: usize, hidden: usize, layers: usize) -> Vec<f32> {
        let mut floats = Vec::new();

        // Layer 0
        floats.extend(vec![0.0f32; obs_dim * hidden]);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Layers 1+: identity
        for _ in 1..layers {
            let mut w = vec![0.0f32; hidden * hidden];
            for i in 0..hidden {
                w[i * hidden + i] = 1.0;
            }
            floats.extend(w);
            floats.extend(vec![0.0f32; hidden]);
            floats.extend(vec![1.0f32; hidden]);
            floats.extend(vec![0.0f32; hidden]);
        }

        // Output
        floats.extend(vec![0.0f32; hidden * NUM_ACTIONS]);
        floats.extend(vec![0.0f32; NUM_ACTIONS]);

        floats
    }

    fn build_dueling_weights(obs_dim: usize, hidden: usize, layers: usize) -> Vec<f32> {
        let mut floats = Vec::new();

        // Layer 0
        floats.extend(vec![0.0f32; obs_dim * hidden]);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Layers 1+: identity
        for _ in 1..layers {
            let mut w = vec![0.0f32; hidden * hidden];
            for i in 0..hidden {
                w[i * hidden + i] = 1.0;
            }
            floats.extend(w);
            floats.extend(vec![0.0f32; hidden]);
            floats.extend(vec![1.0f32; hidden]);
            floats.extend(vec![0.0f32; hidden]);
        }

        // Value head
        floats.extend(vec![0.0f32; hidden]);
        floats.push(0.0);

        // Advantage head
        floats.extend(vec![0.0f32; hidden * NUM_ACTIONS]);
        floats.extend(vec![0.0f32; NUM_ACTIONS]);

        floats
    }

    #[test]
    fn test_standard_2layer() {
        let hidden = 2;
        let floats = build_standard_weights(TEST_OBS_DIM, hidden, 2);
        let mut net = BidNet::from_floats(&floats, hidden, TEST_OBS_DIM, false).unwrap();
        assert!(!net.is_dueling());
        assert_eq!(net.obs_dim(), TEST_OBS_DIM);
        assert_eq!(net.layers(), 2);

        let obs = vec![0.0f32; TEST_OBS_DIM];
        let q = net.evaluate(&obs);
        for &v in &q {
            assert!(v.abs() < 1e-4, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dueling_2layer() {
        let hidden = 4;
        let floats = build_dueling_weights(TEST_OBS_DIM, hidden, 2);
        let mut net = BidNet::from_floats(&floats, hidden, TEST_OBS_DIM, true).unwrap();
        assert!(net.is_dueling());
        assert_eq!(net.layers(), 2);

        let obs = vec![0.0f32; TEST_OBS_DIM];
        let q = net.evaluate(&obs);
        for &v in &q {
            assert!(v.abs() < 1e-4, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_3layer_dueling() {
        let hidden = 4;
        let floats = build_dueling_weights(TEST_OBS_DIM, hidden, 3);
        let mut net = BidNet::from_floats_with_layers(&floats, hidden, TEST_OBS_DIM, true, 3).unwrap();
        assert!(net.is_dueling());
        assert_eq!(net.layers(), 3);

        let obs = vec![0.0f32; TEST_OBS_DIM];
        let q = net.evaluate(&obs);
        for &v in &q {
            assert!(v.abs() < 1e-4, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_autodetect_2layer() {
        let hidden = 4;
        let floats = build_dueling_weights(TEST_OBS_DIM, hidden, 2);

        // Write to temp file
        let tmp = "/tmp/test_bid_autodetect_2.bin";
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(tmp, bytes).unwrap();

        let net = BidNet::load_with_hidden(tmp, hidden).unwrap();
        assert_eq!(net.layers(), 2);
        assert!(net.is_dueling());

        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_autodetect_3layer() {
        let hidden = 4;
        let floats = build_dueling_weights(TEST_OBS_DIM, hidden, 3);

        let tmp = "/tmp/test_bid_autodetect_3.bin";
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(tmp, bytes).unwrap();

        let net = BidNet::load_with_hidden(tmp, hidden).unwrap();
        assert_eq!(net.layers(), 3);
        assert!(net.is_dueling());

        std::fs::remove_file(tmp).ok();
    }

    #[test]
    fn test_dueling_v_plus_a() {
        let hidden = 2;
        let obs_dim = 4;
        let mut floats = Vec::new();

        // Layer 0: pass through first 2 inputs
        let mut w0 = vec![0.0f32; obs_dim * hidden];
        w0[0] = 1.0;
        w0[obs_dim + 1] = 1.0;
        floats.extend(w0);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Layer 1: identity
        floats.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Value head: V = 5.0
        floats.extend(vec![0.0f32; hidden]);
        floats.push(5.0);

        // Advantage head: A[0] = 2.0, A[1] = -1.0, rest 0
        floats.extend(vec![0.0f32; hidden * NUM_ACTIONS]);
        let mut b_adv = vec![0.0f32; NUM_ACTIONS];
        b_adv[0] = 2.0;
        b_adv[1] = -1.0;
        floats.extend(b_adv);

        let mut net = BidNet::from_floats(&floats, hidden, obs_dim, true).unwrap();
        let obs = vec![0.0f32; obs_dim];
        let q = net.evaluate(&obs);

        let adv_mean = (2.0 - 1.0) / NUM_ACTIONS as f32;
        let expected_q0 = 5.0 + 2.0 - adv_mean;
        let expected_q1 = 5.0 + (-1.0) - adv_mean;
        let expected_q2 = 5.0 + 0.0 - adv_mean;

        assert!((q[0] - expected_q0).abs() < 1e-4);
        assert!((q[1] - expected_q1).abs() < 1e-4);
        assert!((q[2] - expected_q2).abs() < 1e-4);
    }

    #[test]
    fn test_best_action() {
        let hidden = 2;
        let mut floats = Vec::new();

        // Layer 0
        let mut w0 = vec![0.0f32; TEST_OBS_DIM * hidden];
        w0[0] = 5.0;
        floats.extend(w0);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Layer 1: identity
        floats.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Output: action 5 gets neuron 0 strongly
        let mut w_out = vec![0.0f32; hidden * NUM_ACTIONS];
        w_out[5 * hidden] = 10.0;
        floats.extend(w_out);
        floats.extend(vec![0.0f32; NUM_ACTIONS]);

        let mut net = BidNet::from_floats(&floats, hidden, TEST_OBS_DIM, false).unwrap();

        let mut obs = vec![0.0f32; TEST_OBS_DIM];
        obs[0] = 1.0;

        // Legal: pass(0), bid 80s(1), action 5, action 10
        let legal_mask: u64 = (1 << 0) | (1 << 1) | (1 << 5) | (1 << 10);
        let (best, legal_q) = net.best_action(&obs, legal_mask);
        assert_eq!(best, 5);
        assert_eq!(legal_q.len(), 4);
    }
}
