/// DMC Q-Network: pure Rust inference for the DouZero-style Deep Monte-Carlo agent.
///
/// Architecture (standard): obs_dim → H (LN+ReLU) → H (LN+ReLU) → H (LN+ReLU) → 32
/// Architecture (dueling):  obs_dim → H (LN+ReLU) → H (LN+ReLU) → H (LN+ReLU)
///                            → Value head: H → 1 (V)
///                            → Advantage head: H → 32 (A)
///                            → Q = V + (A - mean(A))
///
/// where H = hidden size (default 1024), LN = LayerNorm.
/// obs_dim is auto-detected from weight file size (372, 415, or 444).
///
/// No external dependencies — loads raw f32 binary weights exported from PyTorch.
/// Uses scratch buffers for zero-allocation forward pass.
///
/// Weight file layout — standard (contiguous little-endian f32):
///   For each of 3 hidden layers:
///     W: in_dim × H (row-major), b: H, gamma: H, beta: H
///   Final output layer:
///     W: H × 32 (row-major), b: 32
///
/// Weight file layout — dueling:
///   For each of 3 hidden layers:
///     W: in_dim × H (row-major), b: H, gamma: H, beta: H
///   Value head:
///     W_v: H × 1 (row-major), b_v: 1
///   Advantage head:
///     W_a: H × 32 (row-major), b_a: 32

const NUM_ACTIONS: usize = 32;
const LN_EPS: f32 = 1e-5;

pub struct DmcNet {
    // 3 hidden layers: Linear + LayerNorm
    w: [Vec<f32>; 3],
    b: [Vec<f32>; 3],
    gamma: [Vec<f32>; 3],
    beta: [Vec<f32>; 3],
    // Standard output layer (used when dueling=false)
    w_out: Vec<f32>,
    b_out: Vec<f32>,
    // Dueling heads (used when dueling=true)
    dueling: bool,
    w_value: Vec<f32>,   // H × 1
    b_value: f32,
    w_adv: Vec<f32>,     // H × 32
    b_adv: Vec<f32>,     // 32
    // Dimensions
    obs_dim: usize,
    hidden: usize,
    in_dims: [usize; 3], // [obs_dim, hidden, hidden]
    // Scratch buffers (avoid allocations in hot loop)
    scratch_a: Vec<f32>,
    scratch_b: Vec<f32>,
}

impl DmcNet {
    /// Load weights from a raw binary file with default hidden size (1024).
    pub fn load(path: &str) -> std::io::Result<Self> {
        Self::load_with_hidden(path, 1024)
    }

    /// Load weights from a raw binary file with custom hidden size.
    /// The obs_dim and dueling mode are auto-detected from the weight file size.
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

        // Trunk floats = obs_dim*H + 3*H (layer 0) + 2*(H*H + 3*H) (layers 1-2)
        // Standard tail = H*32 + 32
        // Dueling tail = H + 1 + H*32 + 32
        let trunk_fixed = 2 * (h * h + 3 * h) + 3 * h; // layers 1-2 + layer 0 b/gamma/beta
        let standard_tail = h * NUM_ACTIONS + NUM_ACTIONS;
        let dueling_tail = h + 1 + h * NUM_ACTIONS + NUM_ACTIONS;

        // Try standard first
        let standard_fixed = trunk_fixed + standard_tail;
        if total > standard_fixed && (total - standard_fixed) % h == 0 {
            let obs_dim = (total - standard_fixed) / h;
            // Also check if dueling fits
            let dueling_fixed = trunk_fixed + dueling_tail;
            if total > dueling_fixed && (total - dueling_fixed) % h == 0 {
                let dueling_obs_dim = (total - dueling_fixed) / h;
                // Both fit — disambiguate: if standard obs_dim is a known value, prefer it
                let known_dims = [372, 415, 444];
                if known_dims.contains(&obs_dim) && !known_dims.contains(&dueling_obs_dim) {
                    return Self::from_floats(&floats, hidden, obs_dim, false);
                } else if !known_dims.contains(&obs_dim) && known_dims.contains(&dueling_obs_dim) {
                    return Self::from_floats(&floats, hidden, dueling_obs_dim, true);
                }
                // Both known or both unknown — prefer standard (backward compat)
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
                "cannot infer obs_dim: {} floats, hidden={} (neither standard nor dueling layout fits)",
                total, h,
            ),
        ))
    }

    /// Construct from a flat array of f32 weights.
    pub fn from_floats(floats: &[f32], hidden: usize, obs_dim: usize, dueling: bool) -> std::io::Result<Self> {
        let in_dims = [obs_dim, hidden, hidden];

        // Calculate expected size
        let mut expected = 0;
        for &in_dim in &in_dims {
            expected += in_dim * hidden + hidden + hidden + hidden; // W + b + gamma + beta
        }
        if dueling {
            expected += hidden + 1; // value head: W (H×1) + b (1)
            expected += hidden * NUM_ACTIONS + NUM_ACTIONS; // advantage head: W (H×32) + b (32)
        } else {
            expected += hidden * NUM_ACTIONS + NUM_ACTIONS; // output W + b
        }

        if floats.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "weight file has {} floats, expected {} for hidden={}, dueling={}",
                    floats.len(),
                    expected,
                    hidden,
                    dueling,
                ),
            ));
        }

        let mut offset = 0;
        let mut w = [Vec::new(), Vec::new(), Vec::new()];
        let mut b = [Vec::new(), Vec::new(), Vec::new()];
        let mut gamma = [Vec::new(), Vec::new(), Vec::new()];
        let mut beta = [Vec::new(), Vec::new(), Vec::new()];

        for layer in 0..3 {
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
            // Value head: W_v (H×1), b_v (1)
            let w_value = floats[offset..offset + hidden].to_vec();
            offset += hidden;
            let b_value = floats[offset];
            offset += 1;

            // Advantage head: W_a (H×32), b_a (32)
            let w_adv = floats[offset..offset + hidden * NUM_ACTIONS].to_vec();
            offset += hidden * NUM_ACTIONS;
            let b_adv = floats[offset..offset + NUM_ACTIONS].to_vec();
            offset += NUM_ACTIONS;
            debug_assert_eq!(offset, floats.len());

            Ok(DmcNet {
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
            let w_out_size = hidden * NUM_ACTIONS;
            let w_out = floats[offset..offset + w_out_size].to_vec();
            offset += w_out_size;
            let b_out = floats[offset..offset + NUM_ACTIONS].to_vec();

            Ok(DmcNet {
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

    /// Forward pass: compute Q-values for all 32 card actions.
    ///
    /// Uses internal scratch buffers — not thread-safe (use separate instances per thread).
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

        // Layer 2: scratch_a = ReLU(LN(W2 * scratch_b + b2))
        linear(&self.w[2], &self.b[2], &self.scratch_b, &mut self.scratch_a, self.hidden, self.hidden);
        layer_norm(&mut self.scratch_a, &self.gamma[2], &self.beta[2], self.hidden);
        relu(&mut self.scratch_a);

        if self.dueling {
            // Value head: V = W_v * trunk + b_v (scalar)
            let mut v = self.b_value;
            for j in 0..self.hidden {
                v += self.w_value[j] * self.scratch_a[j];
            }

            // Advantage head: A[i] = W_a[i] * trunk + b_a[i]
            let mut q = [0.0f32; NUM_ACTIONS];
            let mut adv_sum = 0.0f32;
            for i in 0..NUM_ACTIONS {
                let row_start = i * self.hidden;
                let mut a = self.b_adv[i];
                for j in 0..self.hidden {
                    a += self.w_adv[row_start + j] * self.scratch_a[j];
                }
                q[i] = a;
                adv_sum += a;
            }

            // Q = V + (A - mean(A))
            let adv_mean = adv_sum / NUM_ACTIONS as f32;
            for i in 0..NUM_ACTIONS {
                q[i] = v + q[i] - adv_mean;
            }

            q
        } else {
            // Standard: q = W_out * scratch_a + b_out
            let mut q = [0.0f32; NUM_ACTIONS];
            for i in 0..NUM_ACTIONS {
                let row_start = i * self.hidden;
                let mut sum = self.b_out[i];
                for j in 0..self.hidden {
                    sum += self.w_out[row_start + j] * self.scratch_a[j];
                }
                q[i] = sum;
            }
            q
        }
    }

    /// Pick the best legal action given a legal action bitmask (u32, bit i = card i legal).
    /// Returns (best_action, q_values_for_legal_actions).
    pub fn best_action(&mut self, obs: &[f32], legal_mask: u32) -> (u8, Vec<(u8, f32)>) {
        let q = self.evaluate(obs);

        let mut best_action = 0u8;
        let mut best_q = f32::NEG_INFINITY;
        let mut legal_q = Vec::new();

        let mut mask = legal_mask;
        while mask != 0 {
            let bit = mask.trailing_zeros() as u8;
            let q_val = q[bit as usize];
            legal_q.push((bit, q_val));
            if q_val > best_q {
                best_q = q_val;
                best_action = bit;
            }
            mask &= mask - 1;
        }

        (best_action, legal_q)
    }

    /// Return the observation dimensionality this network expects.
    pub fn obs_dim(&self) -> usize {
        self.obs_dim
    }

    /// Return whether this is a dueling network.
    pub fn is_dueling(&self) -> bool {
        self.dueling
    }

    /// Return hidden size.
    pub fn hidden(&self) -> usize {
        self.hidden
    }
}

/// Compute out = W * x + b (no activation).
/// W is row-major: W[i * in_dim + j] = weight from input j to output i.
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

/// Apply LayerNorm in-place: x = gamma * (x - mean) / sqrt(var + eps) + beta
#[inline]
fn layer_norm(x: &mut [f32], gamma: &[f32], beta: &[f32], dim: usize) {
    // Compute mean
    let mut mean = 0.0f32;
    for i in 0..dim {
        mean += x[i];
    }
    mean /= dim as f32;

    // Compute variance
    let mut var = 0.0f32;
    for i in 0..dim {
        let d = x[i] - mean;
        var += d * d;
    }
    var /= dim as f32;

    // Normalize: x = gamma * (x - mean) / sqrt(var + eps) + beta
    let inv_std = 1.0 / (var + LN_EPS).sqrt();
    for i in 0..dim {
        x[i] = gamma[i] * (x[i] - mean) * inv_std + beta[i];
    }
}

/// Apply ReLU in-place.
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

    const TEST_OBS_DIM: usize = 372;

    /// Build standard (non-dueling) weight vector for a tiny network.
    fn build_standard_weights(obs_dim: usize, hidden: usize) -> Vec<f32> {
        let mut floats = Vec::new();

        // Layer 0: W, b, gamma, beta
        floats.extend(vec![0.0f32; obs_dim * hidden]);
        floats.extend(vec![0.0f32; hidden]); // b
        floats.extend(vec![1.0f32; hidden]); // gamma
        floats.extend(vec![0.0f32; hidden]); // beta

        // Layers 1-2: identity-ish
        for _ in 0..2 {
            let mut w = vec![0.0f32; hidden * hidden];
            for i in 0..hidden {
                w[i * hidden + i] = 1.0;
            }
            floats.extend(w);
            floats.extend(vec![0.0f32; hidden]);
            floats.extend(vec![1.0f32; hidden]);
            floats.extend(vec![0.0f32; hidden]);
        }

        // Output: W (hidden×32), b (32)
        floats.extend(vec![0.0f32; hidden * NUM_ACTIONS]);
        floats.extend(vec![0.0f32; NUM_ACTIONS]);

        floats
    }

    /// Build dueling weight vector for a tiny network.
    fn build_dueling_weights(obs_dim: usize, hidden: usize) -> Vec<f32> {
        let mut floats = Vec::new();

        // Layer 0
        floats.extend(vec![0.0f32; obs_dim * hidden]);
        floats.extend(vec![0.0f32; hidden]);
        floats.extend(vec![1.0f32; hidden]);
        floats.extend(vec![0.0f32; hidden]);

        // Layers 1-2: identity
        for _ in 0..2 {
            let mut w = vec![0.0f32; hidden * hidden];
            for i in 0..hidden {
                w[i * hidden + i] = 1.0;
            }
            floats.extend(w);
            floats.extend(vec![0.0f32; hidden]);
            floats.extend(vec![1.0f32; hidden]);
            floats.extend(vec![0.0f32; hidden]);
        }

        // Value head: W_v (hidden), b_v (1)
        floats.extend(vec![0.0f32; hidden]);
        floats.push(0.0);

        // Advantage head: W_a (hidden×32), b_a (32)
        floats.extend(vec![0.0f32; hidden * NUM_ACTIONS]);
        floats.extend(vec![0.0f32; NUM_ACTIONS]);

        floats
    }

    #[test]
    fn test_layer_norm() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![1.0; 4];
        let beta = vec![0.0; 4];
        layer_norm(&mut x, &gamma, &beta, 4);

        // mean = 2.5, var = 1.25, inv_std = 1/sqrt(1.25 + 1e-5)
        let mean = 2.5f32;
        let var = 1.25f32;
        let inv_std = 1.0 / (var + LN_EPS).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| (v - mean) * inv_std)
            .collect();

        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {}, expected {}", a, b);
        }
    }

    #[test]
    fn test_layer_norm_with_params() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let gamma = vec![2.0; 4];
        let beta = vec![1.0; 4];
        layer_norm(&mut x, &gamma, &beta, 4);

        let mean = 2.5f32;
        let var = 1.25f32;
        let inv_std = 1.0 / (var + LN_EPS).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|&v| 2.0 * (v - mean) * inv_std + 1.0)
            .collect();

        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-5, "got {}, expected {}", a, b);
        }
    }

    #[test]
    fn test_linear() {
        let x = [1.0, 2.0, 3.0];
        let w = [
            1.0, 0.0, 0.0, // row 0: 1*1 + 0*2 + 0*3 = 1
            0.0, 1.0, 0.0, // row 1: 0*1 + 1*2 + 0*3 = 2
        ];
        let b = [0.5, -0.5];
        let mut out = [0.0f32; 2];
        linear(&w, &b, &x, &mut out, 3, 2);
        assert!((out[0] - 1.5).abs() < 1e-6);
        assert!((out[1] - 1.5).abs() < 1e-6);
    }

    #[test]
    fn test_relu() {
        let mut x = vec![-2.0, -1.0, 0.0, 1.0, 2.0];
        relu(&mut x);
        assert_eq!(x, vec![0.0, 0.0, 0.0, 1.0, 2.0]);
    }

    #[test]
    fn test_dmc_net_tiny() {
        let hidden = 2;
        let mut floats = Vec::new();

        // Layer 0: W (TEST_OBS_DIM×2), b (2), gamma (2), beta (2)
        let mut w0 = vec![0.0f32; TEST_OBS_DIM * hidden];
        w0[0] = 1.0;
        w0[TEST_OBS_DIM + 1] = 1.0;
        floats.extend_from_slice(&w0);
        floats.extend_from_slice(&[0.0, 0.0]);
        floats.extend_from_slice(&[1.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);

        // Layer 1: identity
        floats.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);
        floats.extend_from_slice(&[1.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);

        // Layer 2: identity
        floats.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);
        floats.extend_from_slice(&[1.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);

        // Output: W (2×32), b (32)
        let mut w_out = vec![0.0f32; hidden * NUM_ACTIONS];
        w_out[0] = 1.0; // action 0 = neuron 0
        w_out[1 * hidden + 1] = 1.0; // action 1 = neuron 1
        floats.extend_from_slice(&w_out);
        floats.extend_from_slice(&vec![0.0f32; NUM_ACTIONS]);

        let mut net = DmcNet::from_floats(&floats, hidden, TEST_OBS_DIM, false).unwrap();
        assert!(!net.is_dueling());

        let obs = vec![0.0f32; TEST_OBS_DIM];
        let q = net.evaluate(&obs);
        for &v in &q {
            assert!(v.abs() < 1e-4, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_real_model_zeros_obs() {
        // Cross-check against PyTorch reference values for all-zeros observation.
        // Skip if model file doesn't exist.
        let path = "models/dmc_final.bin";
        let full_path = if std::path::Path::new(path).exists() {
            path.to_string()
        } else {
            let p = format!("../{}", path);
            if !std::path::Path::new(&p).exists() {
                eprintln!("Skipping test_real_model_zeros_obs: model file not found");
                return;
            }
            p
        };

        let mut net = DmcNet::load(&full_path).expect("Failed to load model");
        let obs = vec![0.0f32; net.obs_dim()];
        let q = net.evaluate(&obs);

        // Reference PyTorch values for all-zeros obs (from cross-check):
        let pytorch_ref = [
            0.944901, 0.797158, 0.327786, 0.974957, 0.440011, 0.496235, 0.781434, 0.811141,
            0.587593, 0.359501, 0.897357, 0.607040, 0.097179, 0.372778, 0.311100, 0.524696,
            0.343981, 0.248739, 0.212704, 0.418073, 0.505762, 0.339129, 0.251999, 0.154897,
            0.461956, 0.606996, 0.596551, 0.541014, 0.628931, 0.585381, 0.683913, 0.387073,
        ];

        let mut max_diff = 0.0f32;
        for i in 0..NUM_ACTIONS {
            let diff = (q[i] - pytorch_ref[i]).abs();
            if diff > max_diff {
                max_diff = diff;
            }
            assert!(
                diff < 0.001,
                "Q[{}]: rust={:.6}, pytorch={:.6}, diff={:.6}",
                i, q[i], pytorch_ref[i], diff,
            );
        }
        eprintln!("Max difference from PyTorch: {:.8}", max_diff);
    }

    #[test]
    fn test_best_action() {
        let hidden = 2;
        let mut floats = Vec::new();

        // Layer 0
        let mut w0 = vec![0.0f32; TEST_OBS_DIM * hidden];
        w0[0] = 5.0;
        floats.extend_from_slice(&w0);
        floats.extend_from_slice(&[0.0, 0.0]);
        floats.extend_from_slice(&[1.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);

        // Layer 1: identity
        floats.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);
        floats.extend_from_slice(&[1.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);

        // Layer 2: identity
        floats.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);
        floats.extend_from_slice(&[1.0, 1.0]);
        floats.extend_from_slice(&[0.0, 0.0]);

        // Output: action 5 gets neuron 0 strongly
        let mut w_out = vec![0.0f32; hidden * NUM_ACTIONS];
        w_out[5 * hidden + 0] = 10.0;
        floats.extend_from_slice(&w_out);
        floats.extend_from_slice(&vec![0.0f32; NUM_ACTIONS]);

        let mut net = DmcNet::from_floats(&floats, hidden, TEST_OBS_DIM, false).unwrap();

        let mut obs = vec![0.0f32; TEST_OBS_DIM];
        obs[0] = 1.0;

        let legal_mask: u32 = (1 << 0) | (1 << 3) | (1 << 5) | (1 << 7);
        let (best, legal_q) = net.best_action(&obs, legal_mask);
        assert_eq!(best, 5, "expected action 5 to be best");
        assert_eq!(legal_q.len(), 4);
    }

    #[test]
    fn test_obs_dim_getter() {
        let hidden = 2;
        let obs_dim = 444;
        let floats = build_standard_weights(obs_dim, hidden);
        let net = DmcNet::from_floats(&floats, hidden, obs_dim, false).unwrap();
        assert_eq!(net.obs_dim(), 444);
        assert!(!net.is_dueling());
    }

    #[test]
    fn test_dueling_net_tiny() {
        let hidden = 4;
        let obs_dim = 8;
        let floats = build_dueling_weights(obs_dim, hidden);
        let mut net = DmcNet::from_floats(&floats, hidden, obs_dim, true).unwrap();
        assert!(net.is_dueling());
        assert_eq!(net.obs_dim(), obs_dim);

        // All-zero input: trunk outputs zero, V=0, A=0, Q=0
        let obs = vec![0.0f32; obs_dim];
        let q = net.evaluate(&obs);
        for &v in &q {
            assert!(v.abs() < 1e-4, "expected ~0, got {}", v);
        }
    }

    #[test]
    fn test_dueling_v_plus_a() {
        // Verify Q = V + (A - mean(A)) with known weights.
        let hidden = 2;
        let obs_dim = 4;
        let mut floats = Vec::new();

        // Layer 0: identity-ish (just pass through first 2 inputs)
        let mut w0 = vec![0.0f32; obs_dim * hidden];
        w0[0] = 1.0; // neuron 0 = input 0
        w0[obs_dim + 1] = 1.0; // neuron 1 = input 1
        floats.extend(w0);
        floats.extend(vec![0.0f32; hidden]); // b
        floats.extend(vec![1.0f32; hidden]); // gamma
        floats.extend(vec![0.0f32; hidden]); // beta

        // Layers 1-2: identity
        for _ in 0..2 {
            floats.extend_from_slice(&[1.0, 0.0, 0.0, 1.0]);
            floats.extend(vec![0.0f32; hidden]);
            floats.extend(vec![1.0f32; hidden]);
            floats.extend(vec![0.0f32; hidden]);
        }

        // Value head: V = 5.0 (constant)
        floats.extend(vec![0.0f32; hidden]); // W_v = 0
        floats.push(5.0); // b_v = 5.0

        // Advantage head: A[0] = 2.0, A[1] = -1.0, rest = 0.0
        let mut w_adv = vec![0.0f32; hidden * NUM_ACTIONS];
        // All zeros — advantages come from bias
        floats.extend(w_adv);
        let mut b_adv = vec![0.0f32; NUM_ACTIONS];
        b_adv[0] = 2.0;
        b_adv[1] = -1.0;
        floats.extend(b_adv);

        let mut net = DmcNet::from_floats(&floats, hidden, obs_dim, true).unwrap();
        assert!(net.is_dueling());

        let obs = vec![0.0f32; obs_dim];
        let q = net.evaluate(&obs);

        // A = [2, -1, 0, 0, ..., 0], mean(A) = (2-1)/32 = 1/32 ≈ 0.03125
        let adv_mean = (2.0 - 1.0) / 32.0;
        let expected_q0 = 5.0 + 2.0 - adv_mean;
        let expected_q1 = 5.0 + (-1.0) - adv_mean;
        let expected_q2 = 5.0 + 0.0 - adv_mean;

        assert!((q[0] - expected_q0).abs() < 1e-4, "Q[0]: got {}, expected {}", q[0], expected_q0);
        assert!((q[1] - expected_q1).abs() < 1e-4, "Q[1]: got {}, expected {}", q[1], expected_q1);
        assert!((q[2] - expected_q2).abs() < 1e-4, "Q[2]: got {}, expected {}", q[2], expected_q2);
    }

    #[test]
    fn test_standard_backward_compat() {
        // Existing standard weights should still load correctly
        let hidden = 2;
        let floats = build_standard_weights(TEST_OBS_DIM, hidden);
        let net = DmcNet::from_floats(&floats, hidden, TEST_OBS_DIM, false).unwrap();
        assert!(!net.is_dueling());
        assert_eq!(net.obs_dim(), TEST_OBS_DIM);
    }
}
