use crate::features::FEATURE_DIM;

/// A simple MLP value network: 278 → 256 (ReLU) → 256 (ReLU) → 1 (Sigmoid).
///
/// Loads raw f32 binary weights. Forward pass uses scratch buffers to avoid allocations
/// in the hot loop. ~137K parameters.
///
/// Weight file layout (contiguous f32, little-endian):
///   W1: FEATURE_DIM × H1 (row-major)
///   b1: H1
///   W2: H1 × H2 (row-major)
///   b2: H2
///   W3: H2 × 1 (row-major)
///   b3: 1
pub struct ValueNet {
    w1: Vec<f32>,
    b1: Vec<f32>,
    w2: Vec<f32>,
    b2: Vec<f32>,
    w3: Vec<f32>,
    b3: Vec<f32>,
    h1: usize,
    h2: usize,
    // Scratch buffers for forward pass (avoid allocations)
    scratch1: Vec<f32>,
    scratch2: Vec<f32>,
}

impl ValueNet {
    /// Load weights from a raw binary file.
    ///
    /// Format: all weights as little-endian f32, concatenated in order:
    /// W1 (FEATURE_DIM × h1), b1 (h1), W2 (h1 × h2), b2 (h2), W3 (h2 × 1), b3 (1).
    ///
    /// The hidden dimensions are inferred from the file size.
    pub fn load(path: &str) -> std::io::Result<Self> {
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

        Self::from_floats(&floats, 256, 256)
    }

    /// Load weights from a raw binary file with custom hidden dimensions.
    pub fn load_with_dims(path: &str, h1: usize, h2: usize) -> std::io::Result<Self> {
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

        Self::from_floats(&floats, h1, h2)
    }

    /// Construct from a flat array of f32 weights.
    fn from_floats(floats: &[f32], h1: usize, h2: usize) -> std::io::Result<Self> {
        let expected = FEATURE_DIM * h1 + h1 + h1 * h2 + h2 + h2 + 1;
        if floats.len() != expected {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                format!(
                    "weight file has {} floats, expected {} for {}×{}→{}×{}→1",
                    floats.len(),
                    expected,
                    FEATURE_DIM,
                    h1,
                    h1,
                    h2
                ),
            ));
        }

        let mut offset = 0;
        let w1 = floats[offset..offset + FEATURE_DIM * h1].to_vec();
        offset += FEATURE_DIM * h1;
        let b1 = floats[offset..offset + h1].to_vec();
        offset += h1;
        let w2 = floats[offset..offset + h1 * h2].to_vec();
        offset += h1 * h2;
        let b2 = floats[offset..offset + h2].to_vec();
        offset += h2;
        let w3 = floats[offset..offset + h2].to_vec();
        offset += h2;
        let b3 = floats[offset..offset + 1].to_vec();

        Ok(ValueNet {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            h1,
            h2,
            scratch1: vec![0.0; h1],
            scratch2: vec![0.0; h2],
        })
    }

    /// Create a ValueNet with given weights (for testing).
    pub fn from_weights(
        w1: Vec<f32>,
        b1: Vec<f32>,
        w2: Vec<f32>,
        b2: Vec<f32>,
        w3: Vec<f32>,
        b3: Vec<f32>,
        h1: usize,
        h2: usize,
    ) -> Self {
        debug_assert_eq!(w1.len(), FEATURE_DIM * h1);
        debug_assert_eq!(b1.len(), h1);
        debug_assert_eq!(w2.len(), h1 * h2);
        debug_assert_eq!(b2.len(), h2);
        debug_assert_eq!(w3.len(), h2);
        debug_assert_eq!(b3.len(), 1);
        ValueNet {
            w1,
            b1,
            w2,
            b2,
            w3,
            b3,
            h1,
            h2,
            scratch1: vec![0.0; h1],
            scratch2: vec![0.0; h2],
        }
    }

    /// Evaluate the value function. Returns P(team 0 wins this deal) in [0, 1].
    ///
    /// Uses internal scratch buffers — not thread-safe (use separate instances per thread).
    #[inline]
    pub fn evaluate(&mut self, features: &[f32; FEATURE_DIM]) -> f32 {
        // Layer 1: scratch1 = ReLU(W1 * features + b1)
        linear_relu(
            features,
            &self.w1,
            &self.b1,
            &mut self.scratch1,
            FEATURE_DIM,
            self.h1,
        );

        // Layer 2: scratch2 = ReLU(W2 * scratch1 + b2)
        linear_relu(
            &self.scratch1,
            &self.w2,
            &self.b2,
            &mut self.scratch2,
            self.h1,
            self.h2,
        );

        // Layer 3: output = sigmoid(W3 * scratch2 + b3)
        let mut sum = self.b3[0];
        for j in 0..self.h2 {
            sum += self.w3[j] * self.scratch2[j];
        }
        sigmoid(sum)
    }
}

/// Compute out = ReLU(W * x + b).
/// W is row-major: W[i * in_dim + j] = weight from input j to output i.
#[inline]
fn linear_relu(x: &[f32], w: &[f32], b: &[f32], out: &mut [f32], in_dim: usize, out_dim: usize) {
    for i in 0..out_dim {
        let row_start = i * in_dim;
        let mut sum = b[i];
        // Manual loop — the compiler will auto-vectorize this with SIMD
        for j in 0..in_dim {
            sum += w[row_start + j] * x[j];
        }
        out[i] = if sum > 0.0 { sum } else { 0.0 };
    }
}

#[inline]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_linear_relu() {
        // 2→3 linear + relu
        let x = [1.0, 2.0];
        let w = [
            1.0, 0.0, // row 0: 1*1 + 0*2 = 1
            0.0, 1.0, // row 1: 0*1 + 1*2 = 2
            -1.0, -1.0, // row 2: -1 + -2 = -3 → ReLU → 0
        ];
        let b = [0.0, 0.0, 0.0];
        let mut out = [0.0; 3];
        linear_relu(&x, &w, &b, &mut out, 2, 3);
        assert!((out[0] - 1.0).abs() < 1e-6);
        assert!((out[1] - 2.0).abs() < 1e-6);
        assert!((out[2] - 0.0).abs() < 1e-6); // ReLU clamps negative
    }

    #[test]
    fn test_value_net_known_weights() {
        // Tiny network: 278 → 2 → 2 → 1
        let h1 = 2;
        let h2 = 2;

        // W1: 278×2 — all zeros except first row
        let mut w1 = vec![0.0f32; FEATURE_DIM * h1];
        w1[0] = 1.0; // neuron 0 reads input 0
        w1[FEATURE_DIM + 1] = 1.0; // neuron 1 reads input 1

        let b1 = vec![0.0; h1];

        // W2: 2×2 — identity
        let w2 = vec![1.0, 0.0, 0.0, 1.0];
        let b2 = vec![0.0; h2];

        // W3: 2×1
        let w3 = vec![1.0, -1.0];
        let b3 = vec![0.0];

        let mut net = ValueNet::from_weights(w1, b1, w2, b2, w3, b3, h1, h2);

        // Input: all zeros → output = sigmoid(0) = 0.5
        let features = [0.0f32; FEATURE_DIM];
        let p = net.evaluate(&features);
        assert!((p - 0.5).abs() < 1e-6, "all-zeros: expected 0.5, got {}", p);

        // Input: feature[0] = 1.0 → layer1 = [1, 0] → layer2 = [1, 0] → out = sigmoid(1*1 + (-1)*0) = sigmoid(1)
        let mut features = [0.0f32; FEATURE_DIM];
        features[0] = 1.0;
        let p = net.evaluate(&features);
        let expected = sigmoid(1.0);
        assert!(
            (p - expected).abs() < 1e-5,
            "feature[0]=1: expected {}, got {}",
            expected,
            p
        );
    }
}
