/// Candle-based belief network for supervised training.
///
/// Architecture:
///   330 → FC(512) → LN → ReLU → FC(512) → LN → ReLU → FC(128)
///
/// Output: 128 logits = 32 cards × 4 player slots.
/// Loss: per-card cross-entropy, masked to unknown cards only.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{self, linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::belief_obs::BELIEF_OBS_DIM;

const NUM_OUTPUTS: usize = 128; // 32 cards × 4 players

/// Manual LayerNorm using basic tensor ops.
struct ManualLayerNorm {
    weight: Tensor,
    bias: Tensor,
    eps: f64,
}

impl ManualLayerNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.0))?;
        Ok(ManualLayerNorm { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let mean = x.mean_keepdim(D::Minus1)?;
        let centered = x.broadcast_sub(&mean)?;
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?;
        let std = (var + self.eps)?.sqrt()?;
        let normed = centered.broadcast_div(&std)?;
        normed.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

/// Belief network: 2-layer trunk + linear output head.
pub struct BeliefQNet {
    trunk_fc: [Linear; 2],
    trunk_ln: [ManualLayerNorm; 2],
    output_head: Linear,
}

impl BeliefQNet {
    pub fn new(hidden: usize, vb: VarBuilder) -> Result<Self> {
        let trunk_fc = [
            linear(BELIEF_OBS_DIM, hidden, vb.pp("trunk.0"))?,
            linear(hidden, hidden, vb.pp("trunk.1"))?,
        ];
        let trunk_ln = [
            ManualLayerNorm::new(hidden, 1e-5, vb.pp("trunk_ln.0"))?,
            ManualLayerNorm::new(hidden, 1e-5, vb.pp("trunk_ln.1"))?,
        ];
        let output_head = linear(hidden, NUM_OUTPUTS, vb.pp("output_head"))?;

        Ok(BeliefQNet {
            trunk_fc,
            trunk_ln,
            output_head,
        })
    }

    /// Forward pass: obs (batch, BELIEF_OBS_DIM) → logits (batch, 128).
    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let mut x = obs.clone();
        for i in 0..2 {
            x = self.trunk_fc[i].forward(&x)?;
            x = self.trunk_ln[i].forward(&x)?;
            x = x.relu()?;
        }
        self.output_head.forward(&x)
    }
}

/// Training wrapper: VarMap, optimizer, model.
pub struct BeliefTrainer {
    pub net: BeliefQNet,
    pub varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    hidden: usize,
}

impl BeliefTrainer {
    pub fn new(hidden: usize, lr: f64, weight_decay: f64, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BeliefQNet::new(hidden, vb)?;

        let adamw_params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        };
        let optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

        Ok(BeliefTrainer {
            net,
            varmap,
            optimizer,
            device,
            hidden,
        })
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Single training step with masked cross-entropy loss.
    ///
    /// - obs: flat f32 slice (batch × BELIEF_OBS_DIM)
    /// - targets: u8 slice (batch × 32) — ground truth player index per card
    /// - masks: u32 slice (batch,) — unknown card bitmask
    ///
    /// Returns average CE loss.
    pub fn train_step(
        &mut self,
        obs: &[f32],
        targets: &[u8],
        masks: &[u32],
    ) -> Result<f32> {
        let batch_size = masks.len();
        let device = &self.device;

        // Create obs tensor: (batch, BELIEF_OBS_DIM)
        let obs_t = Tensor::from_slice(obs, (batch_size, BELIEF_OBS_DIM), device)?;

        // Forward pass: (batch, 128)
        let logits = self.net.forward(&obs_t)?;

        // Reshape logits to (batch, 32, 4) for per-card softmax
        let logits_3d = logits.reshape((batch_size, 32, 4))?;

        // Build target tensor: (batch, 32) with player indices as u32
        let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
        let targets_t = Tensor::from_slice(&targets_u32, (batch_size, 32), device)?;

        // Build mask tensor: (batch, 32) with 1.0 for unknown cards
        let mut mask_flat = vec![0.0f32; batch_size * 32];
        for i in 0..batch_size {
            for c in 0..32 {
                if masks[i] & (1u32 << c) != 0 {
                    mask_flat[i * 32 + c] = 1.0;
                }
            }
        }
        let mask_t = Tensor::from_slice(&mask_flat, (batch_size, 32), device)?;

        // Per-card log-softmax: (batch, 32, 4)
        let log_probs = candle_nn::ops::log_softmax(&logits_3d, D::Minus1)?;

        // Gather log-probs for target class: (batch, 32, 1) → (batch, 32)
        let targets_3d = targets_t.unsqueeze(D::Minus1)?;
        let target_log_probs = log_probs.gather(&targets_3d, D::Minus1)?.squeeze(D::Minus1)?;

        // Masked negative log-likelihood: -sum(mask * log_prob) / sum(mask)
        let neg_log_probs = target_log_probs.neg()?;
        let masked_loss = (&neg_log_probs * &mask_t)?;

        let loss_sum = masked_loss.sum_all()?;
        let mask_sum = mask_t.sum_all()?;
        let loss = (&loss_sum / &mask_sum)?;

        // Backward + optimizer step
        self.optimizer.backward_step(&loss)?;

        let loss_val: f32 = loss.detach().to_vec0()?;
        Ok(loss_val)
    }

    /// Compute validation loss without gradient updates.
    pub fn eval_loss(
        &self,
        obs: &[f32],
        targets: &[u8],
        masks: &[u32],
    ) -> Result<f32> {
        let batch_size = masks.len();
        let device = &self.device;

        let obs_t = Tensor::from_slice(obs, (batch_size, BELIEF_OBS_DIM), device)?;
        let logits = self.net.forward(&obs_t)?;
        let logits_3d = logits.reshape((batch_size, 32, 4))?;

        let targets_u32: Vec<u32> = targets.iter().map(|&t| t as u32).collect();
        let targets_t = Tensor::from_slice(&targets_u32, (batch_size, 32), device)?;

        let mut mask_flat = vec![0.0f32; batch_size * 32];
        for i in 0..batch_size {
            for c in 0..32 {
                if masks[i] & (1u32 << c) != 0 {
                    mask_flat[i * 32 + c] = 1.0;
                }
            }
        }
        let mask_t = Tensor::from_slice(&mask_flat, (batch_size, 32), device)?;

        let log_probs = candle_nn::ops::log_softmax(&logits_3d, D::Minus1)?;
        let targets_3d = targets_t.unsqueeze(D::Minus1)?;
        let target_log_probs = log_probs.gather(&targets_3d, D::Minus1)?.squeeze(D::Minus1)?;

        let neg_log_probs = target_log_probs.neg()?;
        let masked_loss = (&neg_log_probs * &mask_t)?;

        let loss_sum = masked_loss.sum_all()?;
        let mask_sum = mask_t.sum_all()?;
        let loss = (&loss_sum / &mask_sum)?;

        let loss_val: f32 = loss.detach().to_vec0()?;
        Ok(loss_val)
    }

    /// Compute per-card accuracy on a batch.
    /// Returns (correct_unknown, total_unknown).
    pub fn eval_accuracy(
        &self,
        obs: &[f32],
        targets: &[u8],
        masks: &[u32],
    ) -> Result<(u64, u64)> {
        let batch_size = masks.len();
        let device = &self.device;

        let obs_t = Tensor::from_slice(obs, (batch_size, BELIEF_OBS_DIM), device)?;
        let logits = self.net.forward(&obs_t)?;
        let logits_3d = logits.reshape((batch_size, 32, 4))?;

        // Argmax over player axis → predicted player per card (batch, 32)
        let preds = logits_3d.argmax(D::Minus1)?.to_vec2::<u32>()?;

        let mut correct = 0u64;
        let mut total = 0u64;
        for i in 0..batch_size {
            for c in 0..32 {
                if masks[i] & (1u32 << c) != 0 {
                    total += 1;
                    if preds[i][c] == targets[i * 32 + c] as u32 {
                        correct += 1;
                    }
                }
            }
        }

        Ok((correct, total))
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        Ok(())
    }

    /// Export weights as raw f32 binary compatible with BeliefNet CPU inference.
    ///
    /// Layout: 2× (W + b + gamma + beta) + output(W + b)
    pub fn export_binary(&self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let data = self.varmap.data().lock().unwrap();
        let mut floats: Vec<f32> = Vec::new();
        let in_dims = [BELIEF_OBS_DIM, self.hidden];

        for i in 0..2 {
            let w: Vec<f32> = data
                .get(&format!("trunk.{}.weight", i))
                .ok_or_else(|| format!("missing trunk.{}.weight", i))?
                .flatten_all()?
                .to_vec1()?;
            let b: Vec<f32> = data
                .get(&format!("trunk.{}.bias", i))
                .ok_or_else(|| format!("missing trunk.{}.bias", i))?
                .flatten_all()?
                .to_vec1()?;
            let gamma: Vec<f32> = data
                .get(&format!("trunk_ln.{}.weight", i))
                .ok_or_else(|| format!("missing trunk_ln.{}.weight", i))?
                .flatten_all()?
                .to_vec1()?;
            let beta: Vec<f32> = data
                .get(&format!("trunk_ln.{}.bias", i))
                .ok_or_else(|| format!("missing trunk_ln.{}.bias", i))?
                .flatten_all()?
                .to_vec1()?;

            assert_eq!(w.len(), self.hidden * in_dims[i]);
            assert_eq!(b.len(), self.hidden);
            assert_eq!(gamma.len(), self.hidden);
            assert_eq!(beta.len(), self.hidden);

            floats.extend(&w);
            floats.extend(&b);
            floats.extend(&gamma);
            floats.extend(&beta);
        }

        // Output head
        let wo: Vec<f32> = data
            .get("output_head.weight")
            .ok_or("missing output_head.weight")?
            .flatten_all()?
            .to_vec1()?;
        let bo: Vec<f32> = data
            .get("output_head.bias")
            .ok_or("missing output_head.bias")?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(wo.len(), self.hidden * NUM_OUTPUTS);
        assert_eq!(bo.len(), NUM_OUTPUTS);
        floats.extend(&wo);
        floats.extend(&bo);

        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_model_creation() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BeliefQNet::new(64, vb)?;

        let obs = Tensor::zeros((2, BELIEF_OBS_DIM), DType::F32, &device)?;
        let logits = net.forward(&obs)?;
        assert_eq!(logits.dims(), &[2, NUM_OUTPUTS]);
        Ok(())
    }

    #[test]
    fn test_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BeliefQNet::new(32, vb)?;

        let batch_size = 16;
        let obs = Tensor::randn(0f32, 1f32, (batch_size, BELIEF_OBS_DIM), &device)?;
        let logits = net.forward(&obs)?;
        assert_eq!(logits.dims(), &[batch_size, NUM_OUTPUTS]);
        Ok(())
    }

    #[test]
    fn test_trainer_train_step() -> Result<()> {
        let device = Device::Cpu;
        let mut trainer = BeliefTrainer::new(32, 1e-3, 0.0, device)?;

        let batch = 4;
        let obs = vec![0.0f32; batch * BELIEF_OBS_DIM];
        // Each card assigned to player 1
        let targets = vec![1u8; batch * 32];
        // All cards unknown
        let masks = vec![0xFFFF_FFFFu32; batch];

        let loss = trainer.train_step(&obs, &targets, &masks)?;
        assert!(loss.is_finite(), "loss should be finite: {}", loss);
        // CE loss for uniform 4-class ≈ ln(4) ≈ 1.386
        assert!(loss > 0.5 && loss < 3.0, "initial loss should be ~1.386, got {}", loss);
        Ok(())
    }

    #[test]
    fn test_eval_accuracy() -> Result<()> {
        let device = Device::Cpu;
        let trainer = BeliefTrainer::new(32, 1e-3, 0.0, device)?;

        let batch = 4;
        let obs = vec![0.0f32; batch * BELIEF_OBS_DIM];
        let targets = vec![0u8; batch * 32]; // all player 0
        let masks = vec![0xFFFF_FFFFu32; batch];

        let (correct, total) = trainer.eval_accuracy(&obs, &targets, &masks)?;
        assert_eq!(total, batch as u64 * 32);
        // With random initialization, accuracy should be ~25% (random guess)
        let acc = correct as f64 / total as f64;
        assert!(acc >= 0.0 && acc <= 1.0);
        Ok(())
    }

    #[test]
    fn test_export_and_load() -> Result<()> {
        let device = Device::Cpu;
        let trainer = BeliefTrainer::new(32, 1e-3, 0.0, device)?;

        let tmp = "/tmp/test_belief_net.bin";
        trainer.export_binary(tmp).unwrap();

        // Load with BeliefNet
        let mut net = crate::belief_net::BeliefNet::load_with_hidden(tmp, 32).unwrap();
        assert_eq!(net.obs_dim(), BELIEF_OBS_DIM);

        // Verify forward pass
        let obs = vec![0.0f32; BELIEF_OBS_DIM];
        let logits = net.evaluate(&obs);
        assert_eq!(logits.len(), NUM_OUTPUTS);
        for &v in &logits {
            assert!(v.is_finite());
        }

        // Compare with candle forward pass
        let obs_t = Tensor::zeros((1, BELIEF_OBS_DIM), DType::F32, trainer.device())?;
        let logits_candle = trainer.net.forward(&obs_t)?;
        let logits_candle_vec: Vec<f32> = logits_candle.flatten_all()?.to_vec1()?;

        for i in 0..NUM_OUTPUTS {
            let diff = (logits[i] - logits_candle_vec[i]).abs();
            assert!(
                diff < 1e-4,
                "logit[{}]: rust={:.6}, candle={:.6}, diff={:.6}",
                i, logits[i], logits_candle_vec[i], diff,
            );
        }

        std::fs::remove_file(tmp).ok();
        Ok(())
    }
}
