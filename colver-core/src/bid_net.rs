/// BidNet: pure Rust inference for the NN bidding model.
///
/// Architecture (dueling): obs_dim → H (LN+ReLU) → H (LN+ReLU)
///                            → Value head: H → 1 (V)
///                            → Advantage head: H → 43 (A)
///                            → Q = V + (A - mean(A))
///
/// Architecture (standard): obs_dim → H (LN+ReLU) → H (LN+ReLU) → 43
///
/// where H = hidden size (default 256), LN = LayerNorm.
///
/// Weight file layout — dueling (little-endian f32):
///   For each of 2 hidden layers:
///     W: in_dim × H (row-major), b: H, gamma: H, beta: H
///   Value head: W_v: H, b_v: 1
///   Advantage head: W_a: H × 43, b_a: 43
///
/// Weight file layout — standard:
///   For each of 2 hidden layers:
///     W: in_dim × H (row-major), b: H, gamma: H, beta: H
///   Output: W: H × 43, b: 43

const NUM_ACTIONS: usize = 43;
const NUM_LAYERS: usize = 2;
const LN_EPS: f32 = 1e-5;

pub struct BidNet {
    w: [Vec<f32>; NUM_LAYERS],
    b: [Vec<f32>; NUM_LAYERS],
    gamma: [Vec<f32>; NUM_LAYERS],
    beta: [Vec<f32>; NUM_LAYERS],
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
    in_dims: [usize; NUM_LAYERS],
    // Scratch buffers
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
}

impl BidNet {
    /// Load weights from a raw binary file with default hidden size (256).
    pub fn load(path: &str) -> std::io::Result<Self> {
        Self::load_with_hidden(path, 256)
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

        // trunk_fixed = layer1 (H*H + 3*H) + layer0 b/gamma/beta (3*H)
        let trunk_fixed = h * h + 3 * h + 3 * h;
        let standard_tail = h * NUM_ACTIONS + NUM_ACTIONS;
        let dueling_tail = h + 1 + h * NUM_ACTIONS + NUM_ACTIONS;

        // Try standard first
        let standard_fixed = trunk_fixed + standard_tail;
        if total > standard_fixed && (total - standard_fixed) % h == 0 {
            let obs_dim = (total - standard_fixed) / h;
            // Check dueling too
            let dueling_fixed = trunk_fixed + dueling_tail;
            if total > dueling_fixed && (total - dueling_fixed) % h == 0 {
                let dueling_obs = (total - dueling_fixed) / h;
                let known = [114];
                if known.contains(&obs_dim) && !known.contains(&dueling_obs) {
                    return Self::from_floats(&floats, hidden, obs_dim, false);
                } else if !known.contains(&obs_dim) && known.contains(&dueling_obs) {
                    return Self::from_floats(&floats, hidden, dueling_obs, true);
                }
                return Self::from_floats(&floats, hidden, obs_dim, false);
            }
            return Self::from_floats(&floats, hidden, obs_dim, false);
        }

        // Try dueling
        let dueling_fixed = trunk_fixed + dueling_tail;
        if total > dueling_fixed && (total - dueling_fixed) % h == 0 {
            let obs_dim = (total - dueling_fixed) / h;
            return Self::from_floats(&floats, hidden, obs_dim, true);
        }

        Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!(
                "cannot infer obs_dim: {} floats, hidden={} (neither standard nor dueling fits)",
                total, h,
            ),
        ))
    }

    /// Construct from a flat array of f32 weights.
    pub fn from_floats(floats: &[f32], hidden: usize, obs_dim: usize, dueling: bool) -> std::io::Result<Self> {
        let in_dims = [obs_dim, hidden];

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
                    "weight file has {} floats, expected {} for obs_dim={}, hidden={}, dueling={}",
                    floats.len(), expected, obs_dim, hidden, dueling,
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

        // Layer 0: scratch_a = ReLU(LN(W0 * obs + b0))
        linear(&self.w[0], &self.b[0], obs, &mut self.scratch_a, self.in_dims[0], self.hidden);
        layer_norm(&mut self.scratch_a, &self.gamma[0], &self.beta[0], self.hidden);
        relu(&mut self.scratch_a);

        // Layer 1: scratch_b = ReLU(LN(W1 * scratch_a + b1))
        linear(&self.w[1], &self.b[1], &self.scratch_a, &mut self.scratch_b, self.hidden, self.hidden);
        layer_norm(&mut self.scratch_b, &self.gamma[1], &self.beta[1], self.hidden);
        relu(&mut self.scratch_b);

        if self.dueling {
            // Value head
            let mut v = self.b_value;
            for j in 0..self.hidden {
                v += self.w_value[j] * self.scratch_b[j];
            }

            // Advantage head
            let mut q = [0.0f32; NUM_ACTIONS];
            let mut adv_sum = 0.0f32;
            for i in 0..NUM_ACTIONS {
                let row_start = i * self.hidden;
                let mut a = self.b_adv[i];
                for j in 0..self.hidden {
                    a += self.w_adv[row_start + j] * self.scratch_b[j];
                }
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
            for i in 0..NUM_ACTIONS {
                let row_start = i * self.hidden;
                let mut sum = self.b_out[i];
                for j in 0..self.hidden {
                    sum += self.w_out[row_start + j] * self.scratch_b[j];
                }
                q[i] = sum;
            }
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

    pub fn is_dueling(&self) -> bool {
        self.dueling
    }
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

    const TEST_OBS_DIM: usize = 114;

    fn build_standard_weights(obs_dim: usize, hidden: usize) -> Vec<f32> {
        let mut floats = Vec::new();

        // Layer 0
        floats.extend(vec![0.0f32; obs_dim * hidden]);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Layer 1: identity
        let mut w1 = vec![0.0f32; hidden * hidden];
        for i in 0..hidden {
            w1[i * hidden + i] = 1.0;
        }
        floats.extend(w1);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Output
        floats.extend(vec![0.0f32; hidden * NUM_ACTIONS]);
        floats.extend(vec![0.0f32; NUM_ACTIONS]);

        floats
    }

    fn build_dueling_weights(obs_dim: usize, hidden: usize) -> Vec<f32> {
        let mut floats = Vec::new();

        // Layer 0
        floats.extend(vec![0.0f32; obs_dim * hidden]);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Layer 1: identity
        let mut w1 = vec![0.0f32; hidden * hidden];
        for i in 0..hidden {
            w1[i * hidden + i] = 1.0;
        }
        floats.extend(w1);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Value head
        floats.extend(vec![0.0f32; hidden]);
        floats.push(0.0);

        // Advantage head
        floats.extend(vec![0.0f32; hidden * NUM_ACTIONS]);
        floats.extend(vec![0.0f32; NUM_ACTIONS]);

        floats
    }

    #[test]
    fn test_standard_tiny() {
        let hidden = 2;
        let floats = build_standard_weights(TEST_OBS_DIM, hidden);
        let mut net = BidNet::from_floats(&floats, hidden, TEST_OBS_DIM, false).unwrap();
        assert!(!net.is_dueling());
        assert_eq!(net.obs_dim(), TEST_OBS_DIM);

        let obs = vec![0.0f32; TEST_OBS_DIM];
        let q = net.evaluate(&obs);
        for &v in &q {
            assert!(v.abs() < 1e-4, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dueling_tiny() {
        let hidden = 4;
        let floats = build_dueling_weights(TEST_OBS_DIM, hidden);
        let mut net = BidNet::from_floats(&floats, hidden, TEST_OBS_DIM, true).unwrap();
        assert!(net.is_dueling());

        let obs = vec![0.0f32; TEST_OBS_DIM];
        let q = net.evaluate(&obs);
        for &v in &q {
            assert!(v.abs() < 1e-4, "expected ~0, got {}", v);
        }
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
