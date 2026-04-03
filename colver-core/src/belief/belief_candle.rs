/// Candle-based belief network for supervised training.
///
/// Architecture:
///   330 → FC(512) → LN → ReLU → FC(512) → LN → ReLU → FC(96)
///
/// Output: 96 logits = 32 cards × 3 player slots (left/partner/right, observer excluded).
/// Loss: per-card cross-entropy, masked to unknown cards only.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{self, linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

const NUM_CLASSES: usize = 3;   // left, partner, right (observer excluded)
const NUM_OUTPUTS: usize = 32 * NUM_CLASSES; // 96

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
    pub fn new(obs_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let trunk_fc = [
            linear(obs_dim, hidden, vb.pp("trunk.0"))?,
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

    /// Forward pass: obs (batch, BELIEF_OBS_DIM) → logits (batch, NUM_OUTPUTS).
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
    obs_dim: usize,
    hidden: usize,
    lr: f64,
    count_reg_weight: f32,
}

impl BeliefTrainer {
    pub fn new(obs_dim: usize, hidden: usize, lr: f64, weight_decay: f64, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BeliefQNet::new(obs_dim, hidden, vb)?;

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
            obs_dim,
            hidden,
            lr,
            count_reg_weight: 0.0,
        })
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        self.optimizer.set_learning_rate(lr);
    }

    pub fn current_lr(&self) -> f64 {
        self.lr
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

        // Create obs tensor: (batch, obs_dim)
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), device)?;

        // Forward pass: (batch, 128)
        let logits = self.net.forward(&obs_t)?;

        // Reshape logits to (batch, 32, NUM_CLASSES) for per-card softmax
        let logits_3d = logits.reshape((batch_size, 32, NUM_CLASSES))?;

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

        // Per-card log-softmax: (batch, 32, NUM_CLASSES)
        let log_probs = candle_nn::ops::log_softmax(&logits_3d, D::Minus1)?;

        // Gather log-probs for target class: (batch, 32, 1) → (batch, 32)
        let targets_3d = targets_t.unsqueeze(D::Minus1)?;
        let target_log_probs = log_probs.gather(&targets_3d, D::Minus1)?.squeeze(D::Minus1)?;

        // Masked negative log-likelihood: -sum(mask * log_prob) / sum(mask)
        let neg_log_probs = target_log_probs.neg()?;
        let masked_loss = (&neg_log_probs * &mask_t)?;

        let loss_sum = masked_loss.sum_all()?;
        let mask_sum = mask_t.sum_all()?;
        let ce_loss = (&loss_sum / &mask_sum)?;

        // Add count regularization if enabled
        let loss = if self.count_reg_weight > 0.0 {
            let count_loss = count_regularization(&logits, targets, masks, batch_size, device)?;
            (&ce_loss + (&count_loss * (self.count_reg_weight as f64))?)?
        } else {
            ce_loss
        };

        // Backward + optimizer step
        self.optimizer.backward_step(&loss)?;

        let loss_val: f32 = loss.detach().to_vec0()?;
        Ok(loss_val)
    }

    pub fn set_count_reg(&mut self, weight: f32) {
        self.count_reg_weight = weight;
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

        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), device)?;
        let logits = self.net.forward(&obs_t)?;
        let logits_3d = logits.reshape((batch_size, 32, NUM_CLASSES))?;

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

        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), device)?;
        let logits = self.net.forward(&obs_t)?;
        let logits_3d = logits.reshape((batch_size, 32, NUM_CLASSES))?;

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
        let in_dims = [self.obs_dim, self.hidden];

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

// ---------------------------------------------------------------------------
// Variant D: Variable-depth MLP
// ---------------------------------------------------------------------------

/// Variable-depth MLP: configurable number of hidden layers.
/// Same forward() -> (batch, NUM_OUTPUTS) contract as BeliefQNet.
pub struct BeliefVarMlp {
    layers_fc: Vec<Linear>,
    layers_ln: Vec<ManualLayerNorm>,
    output_head: Linear,
}

impl BeliefVarMlp {
    pub fn new(obs_dim: usize, hidden: usize, num_layers: usize, vb: VarBuilder) -> Result<Self> {
        assert!(num_layers >= 1, "need at least 1 hidden layer");
        let mut layers_fc = Vec::with_capacity(num_layers);
        let mut layers_ln = Vec::with_capacity(num_layers);

        for i in 0..num_layers {
            let in_dim = if i == 0 { obs_dim } else { hidden };
            layers_fc.push(linear(in_dim, hidden, vb.pp(format!("trunk.{}", i)))?);
            layers_ln.push(ManualLayerNorm::new(hidden, 1e-5, vb.pp(format!("trunk_ln.{}", i)))?);
        }

        let output_head = linear(hidden, NUM_OUTPUTS, vb.pp("output_head"))?;

        Ok(BeliefVarMlp { layers_fc, layers_ln, output_head })
    }

    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let mut x = obs.clone();
        for i in 0..self.layers_fc.len() {
            x = self.layers_fc[i].forward(&x)?;
            x = self.layers_ln[i].forward(&x)?;
            x = x.relu()?;
        }
        self.output_head.forward(&x)
    }
}

/// Training wrapper for BeliefVarMlp.
pub struct BeliefVarMlpTrainer {
    pub net: BeliefVarMlp,
    pub varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    obs_dim: usize,
    hidden: usize,
    num_layers: usize,
    lr: f64,
    count_reg_weight: f32,
}

impl BeliefVarMlpTrainer {
    pub fn new(
        obs_dim: usize,
        hidden: usize,
        num_layers: usize,
        lr: f64,
        weight_decay: f64,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BeliefVarMlp::new(obs_dim, hidden, num_layers, vb)?;

        let adamw_params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        };
        let optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

        Ok(BeliefVarMlpTrainer {
            net, varmap, optimizer, device, obs_dim, hidden, num_layers, lr,
            count_reg_weight: 0.0,
        })
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        self.optimizer.set_learning_rate(lr);
    }

    pub fn current_lr(&self) -> f64 {
        self.lr
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    pub fn train_step(&mut self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<f32> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        let ce_loss = masked_cross_entropy(&logits, targets, masks, batch_size, &self.device)?;
        let loss = if self.count_reg_weight > 0.0 {
            let cr = count_regularization(&logits, targets, masks, batch_size, &self.device)?;
            (&ce_loss + (&cr * (self.count_reg_weight as f64))?)?
        } else { ce_loss };
        self.optimizer.backward_step(&loss)?;
        loss.detach().to_vec0()
    }

    pub fn set_count_reg(&mut self, weight: f32) { self.count_reg_weight = weight; }

    pub fn eval_loss(&self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<f32> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        let loss = masked_cross_entropy(&logits, targets, masks, batch_size, &self.device)?;
        loss.detach().to_vec0()
    }

    pub fn eval_accuracy(&self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<(u64, u64)> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        compute_accuracy(&logits, targets, masks, batch_size)
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn export_binary(&self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let data = self.varmap.data().lock().unwrap();
        let mut floats: Vec<f32> = Vec::new();

        for i in 0..self.num_layers {
            let in_dim = if i == 0 { self.obs_dim } else { self.hidden };
            let w: Vec<f32> = data
                .get(&format!("trunk.{}.weight", i))
                .ok_or_else(|| format!("missing trunk.{}.weight", i))?
                .flatten_all()?.to_vec1()?;
            let b: Vec<f32> = data
                .get(&format!("trunk.{}.bias", i))
                .ok_or_else(|| format!("missing trunk.{}.bias", i))?
                .flatten_all()?.to_vec1()?;
            let gamma: Vec<f32> = data
                .get(&format!("trunk_ln.{}.weight", i))
                .ok_or_else(|| format!("missing trunk_ln.{}.weight", i))?
                .flatten_all()?.to_vec1()?;
            let beta: Vec<f32> = data
                .get(&format!("trunk_ln.{}.bias", i))
                .ok_or_else(|| format!("missing trunk_ln.{}.bias", i))?
                .flatten_all()?.to_vec1()?;

            assert_eq!(w.len(), self.hidden * in_dim);
            assert_eq!(b.len(), self.hidden);
            floats.extend(&w);
            floats.extend(&b);
            floats.extend(&gamma);
            floats.extend(&beta);
        }

        let wo: Vec<f32> = data.get("output_head.weight").ok_or("missing output_head.weight")?
            .flatten_all()?.to_vec1()?;
        let bo: Vec<f32> = data.get("output_head.bias").ok_or("missing output_head.bias")?
            .flatten_all()?.to_vec1()?;
        floats.extend(&wo);
        floats.extend(&bo);

        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Variant E: Per-Suit Weight-Shared Network
// ---------------------------------------------------------------------------

/// Per-suit weight-shared architecture.
/// Each suit is processed by the same MLP, combined with a global context.
/// Suit-equivariant by construction — no augmentation needed.
pub struct BeliefSuitNet {
    suit_fc1: Linear,
    suit_ln1: ManualLayerNorm,
    suit_fc2: Linear,
    suit_ln2: ManualLayerNorm,
    global_fc1: Linear,
    global_ln1: ManualLayerNorm,
    global_fc2: Linear,
    global_ln2: ManualLayerNorm,
    output_head: Linear,
    suit_hidden: usize,
    global_hidden: usize,
}

impl BeliefSuitNet {
    /// `suit_hidden` and `global_hidden` control the width.
    pub fn new(suit_hidden: usize, global_hidden: usize, vb: VarBuilder) -> Result<Self> {
        // Per-suit input: hand(8) + played_by(8) + trick_idx(8) + pos(8)
        //                 + hc_left(8) + hc_partner(8) + hc_right(8) + is_trump(1) = 57
        let suit_in = 57;
        let suit_fc1 = linear(suit_in, suit_hidden, vb.pp("suit.0"))?;
        let suit_ln1 = ManualLayerNorm::new(suit_hidden, 1e-5, vb.pp("suit_ln.0"))?;
        let suit_fc2 = linear(suit_hidden, suit_hidden, vb.pp("suit.1"))?;
        let suit_ln2 = ManualLayerNorm::new(suit_hidden, 1e-5, vb.pp("suit_ln.1"))?;

        // Global input: bid_history(72) + contract(8) = 80
        let global_in = 80;
        let global_fc1 = linear(global_in, global_hidden, vb.pp("global.0"))?;
        let global_ln1 = ManualLayerNorm::new(global_hidden, 1e-5, vb.pp("global_ln.0"))?;
        let global_fc2 = linear(global_hidden, global_hidden, vb.pp("global.1"))?;
        let global_ln2 = ManualLayerNorm::new(global_hidden, 1e-5, vb.pp("global_ln.1"))?;

        // Output: per-suit combined → 8 cards × NUM_CLASSES players = 24
        let combined_dim = suit_hidden + global_hidden;
        let output_head = linear(combined_dim, 8 * NUM_CLASSES, vb.pp("output_head"))?;

        Ok(BeliefSuitNet {
            suit_fc1, suit_ln1, suit_fc2, suit_ln2,
            global_fc1, global_ln1, global_fc2, global_ln2,
            output_head, suit_hidden, global_hidden,
        })
    }

    /// Forward pass: V2 obs (batch, 304) → logits (batch, NUM_OUTPUTS).
    ///
    /// Extracts per-suit features from V2 layout:
    ///   Block 1 [0:32]:    hand — 8 per suit
    ///   Block 2 [32:64]:   played-by — 8 per suit
    ///   Block 3 [64:96]:   trick index — 8 per suit
    ///   Block 4 [96:128]:  position-in-trick — 8 per suit
    ///   Block 5 [128:200]: bid history (global)
    ///   Block 6 [200:208]: contract (global, trump one-hot indicates which suit)
    ///   Block 7 [208:304]: hard constraints — 3 groups of 32, 8 per suit per group
    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let batch_size = obs.dims()[0];
        let device = obs.device();

        // Extract global features: bid_history[128:200] + contract[200:208] = 80
        // contiguous() needed because narrow creates a non-contiguous view
        let global_input = obs.narrow(1, 128, 80)?.contiguous()?;
        let mut g = self.global_fc1.forward(&global_input)?;
        g = self.global_ln1.forward(&g)?;
        g = g.relu()?;
        g = self.global_fc2.forward(&g)?;
        g = self.global_ln2.forward(&g)?;
        g = g.relu()?; // (batch, global_hidden)

        // Trump suit from contract one-hot [200:204]
        let trump_onehot = obs.narrow(1, 200, 4)?; // (batch, 4)

        // Process each suit and collect outputs
        let mut suit_outputs = Vec::with_capacity(4);

        for s in 0..4usize {
            // Gather per-suit slices:
            //   hand:      obs[s*8 .. s*8+8]         (8)
            //   played_by: obs[32+s*8 .. 32+s*8+8]   (8)
            //   trick_idx: obs[64+s*8 .. 64+s*8+8]   (8)
            //   pos:       obs[96+s*8 .. 96+s*8+8]   (8)
            //   hc_left:   obs[208+s*8 .. 208+s*8+8] (8)
            //   hc_ptnr:   obs[240+s*8 .. 240+s*8+8] (8)
            //   hc_right:  obs[272+s*8 .. 272+s*8+8] (8)
            //   is_trump:  trump_onehot[:, s:s+1]    (1)

            let hand_s = obs.narrow(1, s * 8, 8)?;
            let played_s = obs.narrow(1, 32 + s * 8, 8)?;
            let trick_s = obs.narrow(1, 64 + s * 8, 8)?;
            let pos_s = obs.narrow(1, 96 + s * 8, 8)?;
            let hc_left_s = obs.narrow(1, 208 + s * 8, 8)?;
            let hc_ptnr_s = obs.narrow(1, 240 + s * 8, 8)?;
            let hc_right_s = obs.narrow(1, 272 + s * 8, 8)?;
            let is_trump_s = trump_onehot.narrow(1, s, 1)?;

            // Concatenate: (batch, 57)
            let suit_input = Tensor::cat(
                &[hand_s, played_s, trick_s, pos_s, hc_left_s, hc_ptnr_s, hc_right_s, is_trump_s],
                1,
            )?;

            // Shared suit MLP
            let mut h = self.suit_fc1.forward(&suit_input)?;
            h = self.suit_ln1.forward(&h)?;
            h = h.relu()?;
            h = self.suit_fc2.forward(&h)?;
            h = self.suit_ln2.forward(&h)?;
            h = h.relu()?; // (batch, suit_hidden)

            // Combine with global: (batch, suit_hidden + global_hidden)
            let combined = Tensor::cat(&[h, g.clone()], 1)?;

            // Output head: (batch, 8*NUM_CLASSES) = 8 cards × NUM_CLASSES players
            let out = self.output_head.forward(&combined)?;
            suit_outputs.push(out);
        }

        // Concatenate suit outputs: (batch, NUM_OUTPUTS)
        Tensor::cat(&suit_outputs, 1)
    }
}

/// Training wrapper for BeliefSuitNet.
pub struct BeliefSuitNetTrainer {
    pub net: BeliefSuitNet,
    pub varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    suit_hidden: usize,
    global_hidden: usize,
    lr: f64,
    count_reg_weight: f32,
}

impl BeliefSuitNetTrainer {
    pub fn new(
        suit_hidden: usize,
        global_hidden: usize,
        lr: f64,
        weight_decay: f64,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BeliefSuitNet::new(suit_hidden, global_hidden, vb)?;

        let adamw_params = ParamsAdamW {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay,
        };
        let optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

        Ok(BeliefSuitNetTrainer {
            net, varmap, optimizer, device, suit_hidden, global_hidden, lr,
            count_reg_weight: 0.0,
        })
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        self.optimizer.set_learning_rate(lr);
    }

    pub fn current_lr(&self) -> f64 { self.lr }
    pub fn device(&self) -> &Device { &self.device }

    pub fn train_step(&mut self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<f32> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, 304), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        let ce_loss = masked_cross_entropy(&logits, targets, masks, batch_size, &self.device)?;
        let loss = if self.count_reg_weight > 0.0 {
            let cr = count_regularization(&logits, targets, masks, batch_size, &self.device)?;
            (&ce_loss + (&cr * (self.count_reg_weight as f64))?)?
        } else { ce_loss };
        self.optimizer.backward_step(&loss)?;
        loss.detach().to_vec0()
    }

    pub fn set_count_reg(&mut self, weight: f32) { self.count_reg_weight = weight; }

    pub fn eval_loss(&self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<f32> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, 304), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        let loss = masked_cross_entropy(&logits, targets, masks, batch_size, &self.device)?;
        loss.detach().to_vec0()
    }

    pub fn eval_accuracy(&self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<(u64, u64)> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, 304), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        compute_accuracy(&logits, targets, masks, batch_size)
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn export_binary(&self, _path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // SuitNet binary export not compatible with BeliefNet CPU inference
        // (different architecture). Save safetensors checkpoints instead.
        Err("SuitNet binary export not supported — use safetensors checkpoints".into())
    }
}

// ---------------------------------------------------------------------------
// Variant B: Card-Level Cross-Attention
// ---------------------------------------------------------------------------

/// Card-level cross-attention architecture.
/// Global encoder processes full obs → 256-dim. Per-card: learned embedding (64) +
/// global context → self-attention over 32 card tokens → per-card output (4 logits).
pub struct BeliefCrossAttnNet {
    global_fc: Linear,
    global_ln: ManualLayerNorm,
    card_embeddings: Tensor,  // (32, 64) learnable
    card_proj: Linear,        // (320 → 64)
    // Self-attention: QKV projection
    qkv_proj: Linear,        // (64 → 192) = 3 × 4_heads × 16
    out_proj: Linear,         // (64 → 64)
    attn_ln: ManualLayerNorm, // post-attention layer norm
    card_output: Linear,      // (64 → 4) per card
}

impl BeliefCrossAttnNet {
    pub fn new(obs_dim: usize, vb: VarBuilder) -> Result<Self> {
        let global_dim = 256;
        let card_dim = 64;

        let global_fc = linear(obs_dim, global_dim, vb.pp("global_fc"))?;
        let global_ln = ManualLayerNorm::new(global_dim, 1e-5, vb.pp("global_ln"))?;

        // Learnable card embeddings
        let card_embeddings = vb.get_with_hints(
            (32, card_dim),
            "card_embed",
            candle_nn::Init::Randn { mean: 0.0, stdev: 0.02 },
        )?;

        // card_embed(64) + global(256) = 320 → 64
        let card_proj = linear(card_dim + global_dim, card_dim, vb.pp("card_proj"))?;

        // QKV for self-attention: 64 → 192 (Q=64, K=64, V=64)
        let qkv_proj = linear(card_dim, card_dim * 3, vb.pp("qkv"))?;
        let out_proj = linear(card_dim, card_dim, vb.pp("out_proj"))?;
        let attn_ln = ManualLayerNorm::new(card_dim, 1e-5, vb.pp("attn_ln"))?;

        // Per-card output: 64 → NUM_CLASSES (player probabilities)
        let card_output = linear(card_dim, NUM_CLASSES, vb.pp("card_output"))?;

        Ok(BeliefCrossAttnNet {
            global_fc, global_ln, card_embeddings, card_proj,
            qkv_proj, out_proj, attn_ln, card_output,
        })
    }

    /// Forward: obs (batch, obs_dim) → logits (batch, NUM_OUTPUTS).
    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let batch_size = obs.dims()[0];
        let card_dim = 64usize;
        let n_heads = 4usize;
        let head_dim = card_dim / n_heads; // 16

        // Global encoding: (batch, 256)
        let mut global = self.global_fc.forward(obs)?;
        global = self.global_ln.forward(&global)?;
        global = global.relu()?;

        // Expand card embeddings to batch: (32, 64) → (batch, 32, 64)
        let card_emb = self.card_embeddings.unsqueeze(0)?.broadcast_as((batch_size, 32, card_dim))?;

        // Expand global to per-card: (batch, 256) → (batch, 32, 256)
        let global_expanded = global.unsqueeze(1)?.broadcast_as((batch_size, 32, 256))?;

        // Concatenate: (batch, 32, 320)
        let card_input = Tensor::cat(&[card_emb.contiguous()?, global_expanded.contiguous()?], 2)?;

        // Project: (batch, 32, 64) + ReLU
        let tokens = self.card_proj.forward(&card_input)?.relu()?;

        // Self-attention: QKV projection → (batch, 32, 192)
        let qkv = self.qkv_proj.forward(&tokens)?;
        // Split into Q, K, V: each (batch, 32, 64)
        let q = qkv.narrow(2, 0, card_dim)?;
        let k = qkv.narrow(2, card_dim, card_dim)?;
        let v = qkv.narrow(2, card_dim * 2, card_dim)?;

        // Reshape for multi-head: (batch, 32, 4, 16) → (batch, 4, 32, 16)
        // contiguous() needed after permute for matmul
        let q = q.reshape((batch_size, 32, n_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let k = k.reshape((batch_size, 32, n_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;
        let v = v.reshape((batch_size, 32, n_heads, head_dim))?.permute((0, 2, 1, 3))?.contiguous()?;

        // Attention scores: (batch, 4, 32, 32)
        let scale = (head_dim as f64).sqrt();
        let scores = (q.matmul(&k.transpose(2, 3)?)? / scale)?;
        let attn_weights = candle_nn::ops::softmax(&scores, D::Minus1)?;

        // Weighted values: (batch, 4, 32, 16)
        let attn_out = attn_weights.matmul(&v)?;

        // Reshape back: (batch, 4, 32, 16) → (batch, 32, 64)
        let attn_out = attn_out.permute((0, 2, 1, 3))?.reshape((batch_size, 32, card_dim))?;

        // Output projection + residual + layer norm
        let projected = self.out_proj.forward(&attn_out)?;
        let residual = (&tokens + &projected)?;
        let normed = self.attn_ln.forward(&residual)?;

        // Per-card output: (batch, 32, 64) → (batch, 32, NUM_CLASSES)
        let card_logits = self.card_output.forward(&normed)?;

        // Reshape to (batch, NUM_OUTPUTS)
        card_logits.reshape((batch_size, NUM_OUTPUTS))
    }
}

/// Training wrapper for BeliefCrossAttnNet.
pub struct BeliefCrossAttnTrainer {
    pub net: BeliefCrossAttnNet,
    pub varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    obs_dim: usize,
    lr: f64,
    count_reg_weight: f32,
}

impl BeliefCrossAttnTrainer {
    pub fn new(obs_dim: usize, lr: f64, weight_decay: f64, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BeliefCrossAttnNet::new(obs_dim, vb)?;

        let adamw_params = ParamsAdamW {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay,
        };
        let optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

        Ok(BeliefCrossAttnTrainer { net, varmap, optimizer, device, obs_dim, lr, count_reg_weight: 0.0 })
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        self.optimizer.set_learning_rate(lr);
    }

    pub fn current_lr(&self) -> f64 { self.lr }
    pub fn device(&self) -> &Device { &self.device }

    pub fn train_step(&mut self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<f32> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        let ce_loss = masked_cross_entropy(&logits, targets, masks, batch_size, &self.device)?;
        let loss = if self.count_reg_weight > 0.0 {
            let cr = count_regularization(&logits, targets, masks, batch_size, &self.device)?;
            (&ce_loss + (&cr * (self.count_reg_weight as f64))?)?
        } else { ce_loss };
        self.optimizer.backward_step(&loss)?;
        loss.detach().to_vec0()
    }

    pub fn set_count_reg(&mut self, weight: f32) { self.count_reg_weight = weight; }

    pub fn eval_loss(&self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<f32> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        let loss = masked_cross_entropy(&logits, targets, masks, batch_size, &self.device)?;
        loss.detach().to_vec0()
    }

    pub fn eval_accuracy(&self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<(u64, u64)> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        compute_accuracy(&logits, targets, masks, batch_size)
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn export_binary(&self, _path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        Err("CrossAttnNet binary export not supported — use safetensors checkpoints".into())
    }
}

// ---------------------------------------------------------------------------
// Variant C: Auxiliary Losses
// ---------------------------------------------------------------------------

/// Same architecture as BeliefQNet but with auxiliary prediction heads.
/// Extra outputs: trick winner prediction (8×4) and void prediction (3×4).
pub struct BeliefAuxNet {
    trunk_fc: [Linear; 2],
    trunk_ln: [ManualLayerNorm; 2],
    output_head: Linear,
    trick_winner_head: Linear,  // hidden → 32 (8 tricks × 4 seats)
    void_head: Linear,          // hidden → 12 (3 hidden players × 4 suits)
}

impl BeliefAuxNet {
    pub fn new(obs_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        let trunk_fc = [
            linear(obs_dim, hidden, vb.pp("trunk.0"))?,
            linear(hidden, hidden, vb.pp("trunk.1"))?,
        ];
        let trunk_ln = [
            ManualLayerNorm::new(hidden, 1e-5, vb.pp("trunk_ln.0"))?,
            ManualLayerNorm::new(hidden, 1e-5, vb.pp("trunk_ln.1"))?,
        ];
        let output_head = linear(hidden, NUM_OUTPUTS, vb.pp("output_head"))?;
        let trick_winner_head = linear(hidden, 32, vb.pp("trick_winner_head"))?;
        let void_head = linear(hidden, 12, vb.pp("void_head"))?;

        Ok(BeliefAuxNet {
            trunk_fc, trunk_ln, output_head, trick_winner_head, void_head,
        })
    }

    /// Forward: returns (main_logits, trunk_hidden) for flexibility.
    fn forward_trunk(&self, obs: &Tensor) -> Result<(Tensor, Tensor)> {
        let mut x = obs.clone();
        for i in 0..2 {
            x = self.trunk_fc[i].forward(&x)?;
            x = self.trunk_ln[i].forward(&x)?;
            x = x.relu()?;
        }
        let logits = self.output_head.forward(&x)?;
        Ok((logits, x))
    }

    /// Forward: main logits only (compatible with standard evaluation).
    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let (logits, _) = self.forward_trunk(obs)?;
        Ok(logits)
    }

    /// Forward with auxiliary outputs.
    /// Returns (main_logits, trick_winner_logits, void_logits).
    pub fn forward_with_aux(&self, obs: &Tensor) -> Result<(Tensor, Tensor, Tensor)> {
        let (logits, hidden) = self.forward_trunk(obs)?;
        let trick_logits = self.trick_winner_head.forward(&hidden)?;
        let void_logits = self.void_head.forward(&hidden)?;
        Ok((logits, trick_logits, void_logits))
    }
}

/// Training wrapper for BeliefAuxNet.
pub struct BeliefAuxTrainer {
    pub net: BeliefAuxNet,
    pub varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    obs_dim: usize,
    hidden: usize,
    lr: f64,
    count_reg_weight: f32,
}

impl BeliefAuxTrainer {
    pub fn new(obs_dim: usize, hidden: usize, lr: f64, weight_decay: f64, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BeliefAuxNet::new(obs_dim, hidden, vb)?;

        let adamw_params = ParamsAdamW {
            lr, beta1: 0.9, beta2: 0.999, eps: 1e-8, weight_decay,
        };
        let optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

        Ok(BeliefAuxTrainer { net, varmap, optimizer, device, obs_dim, hidden, lr, count_reg_weight: 0.0 })
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.lr = lr;
        self.optimizer.set_learning_rate(lr);
    }

    pub fn current_lr(&self) -> f64 { self.lr }
    pub fn device(&self) -> &Device { &self.device }

    /// Train step with auxiliary losses.
    ///
    /// `aux_targets`: (trick_winners [batch × 8], void_labels [batch × 12])
    /// `aux_weight`: weighting for auxiliary losses (decays over training)
    pub fn train_step_with_aux(
        &mut self,
        obs: &[f32],
        targets: &[u8],
        masks: &[u32],
        trick_winner_targets: &[u8],  // batch × 8, 0-3 or 0xFF for unknown
        void_targets: &[f32],         // batch × 12, 0.0 or 1.0
        aux_weight: f32,
    ) -> Result<f32> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let (main_logits, trick_logits, void_logits) = self.net.forward_with_aux(&obs_t)?;

        // Main loss: masked cross-entropy (same as standard)
        let main_loss = masked_cross_entropy(&main_logits, targets, masks, batch_size, &self.device)?;

        let total_loss = if aux_weight > 0.0 {
            // Trick winner loss: 4-class CE per completed trick, masked for incomplete
            let trick_logits_3d = trick_logits.reshape((batch_size, 8, 4))?;
            let mut trick_loss_sum = 0.0f32;
            let mut trick_count = 0u32;

            // Build mask for completed tricks
            let mut tw_mask = vec![0.0f32; batch_size * 8];
            let mut tw_targets_u32 = vec![0u32; batch_size * 8];
            for i in 0..batch_size {
                for t in 0..8 {
                    let target_val = trick_winner_targets[i * 8 + t];
                    if target_val < 4 {
                        tw_mask[i * 8 + t] = 1.0;
                        tw_targets_u32[i * 8 + t] = target_val as u32;
                        trick_count += 1;
                    }
                }
            }

            let trick_loss = if trick_count > 0 {
                let tw_targets_t = Tensor::from_slice(&tw_targets_u32, (batch_size, 8), &self.device)?;
                let tw_mask_t = Tensor::from_slice(&tw_mask, (batch_size, 8), &self.device)?;
                let log_probs = candle_nn::ops::log_softmax(&trick_logits_3d, D::Minus1)?;
                let targets_3d = tw_targets_t.unsqueeze(D::Minus1)?;
                let target_lp = log_probs.gather(&targets_3d, D::Minus1)?.squeeze(D::Minus1)?;
                let masked = (target_lp.neg()? * &tw_mask_t)?;
                let sum = masked.sum_all()?;
                let count = tw_mask_t.sum_all()?;
                (&sum / &count)?
            } else {
                Tensor::new(0.0f32, &self.device)?
            };

            // Void loss: binary CE per player × suit
            let void_targets_t = Tensor::from_slice(void_targets, (batch_size, 12), &self.device)?;
            // Manual sigmoid: 1/(1+exp(-x)) — candle CUDA lacks sigmoid kernel
            let void_probs = (void_logits.neg()?.exp()? + 1.0)?.recip()?;
            let void_probs_clamped = void_probs.clamp(1e-7, 1.0 - 1e-7)?;
            let one_minus = (1.0 - &void_probs_clamped)?;
            let one_minus_clamped = one_minus.clamp(1e-7, 1.0 - 1e-7)?;
            let one_minus_targets = (1.0 - &void_targets_t)?;
            let term1 = (&void_targets_t * void_probs_clamped.log()?)?;
            let term2 = (&one_minus_targets * one_minus_clamped.log()?)?;
            let bce = (&term1 + &term2)?;
            let void_loss = bce.neg()?.mean_all()?;

            // Total: main + weight × (trick + void)
            let aux_loss = (&trick_loss + &void_loss)?;
            let aux_scaled = (&aux_loss * (aux_weight as f64))?;
            (&main_loss + &aux_scaled)?
        } else {
            main_loss
        };

        self.optimizer.backward_step(&total_loss)?;
        total_loss.detach().to_vec0()
    }

    /// Standard train step (no aux loss).
    pub fn train_step(&mut self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<f32> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        let ce_loss = masked_cross_entropy(&logits, targets, masks, batch_size, &self.device)?;
        let loss = if self.count_reg_weight > 0.0 {
            let cr = count_regularization(&logits, targets, masks, batch_size, &self.device)?;
            (&ce_loss + (&cr * (self.count_reg_weight as f64))?)?
        } else { ce_loss };
        self.optimizer.backward_step(&loss)?;
        loss.detach().to_vec0()
    }

    pub fn set_count_reg(&mut self, weight: f32) { self.count_reg_weight = weight; }

    pub fn eval_loss(&self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<f32> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        let loss = masked_cross_entropy(&logits, targets, masks, batch_size, &self.device)?;
        loss.detach().to_vec0()
    }

    pub fn eval_accuracy(&self, obs: &[f32], targets: &[u8], masks: &[u32]) -> Result<(u64, u64)> {
        let batch_size = masks.len();
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let logits = self.net.forward(&obs_t)?;
        compute_accuracy(&logits, targets, masks, batch_size)
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn export_binary(&self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        // Export only the main trunk + output head (compatible with BeliefNet CPU inference)
        let data = self.varmap.data().lock().unwrap();
        let mut floats: Vec<f32> = Vec::new();
        let in_dims = [self.obs_dim, self.hidden];

        for i in 0..2 {
            let w: Vec<f32> = data.get(&format!("trunk.{}.weight", i))
                .ok_or_else(|| format!("missing trunk.{}.weight", i))?.flatten_all()?.to_vec1()?;
            let b: Vec<f32> = data.get(&format!("trunk.{}.bias", i))
                .ok_or_else(|| format!("missing trunk.{}.bias", i))?.flatten_all()?.to_vec1()?;
            let gamma: Vec<f32> = data.get(&format!("trunk_ln.{}.weight", i))
                .ok_or_else(|| format!("missing trunk_ln.{}.weight", i))?.flatten_all()?.to_vec1()?;
            let beta: Vec<f32> = data.get(&format!("trunk_ln.{}.bias", i))
                .ok_or_else(|| format!("missing trunk_ln.{}.bias", i))?.flatten_all()?.to_vec1()?;

            assert_eq!(w.len(), self.hidden * in_dims[i]);
            floats.extend(&w);
            floats.extend(&b);
            floats.extend(&gamma);
            floats.extend(&beta);
        }

        let wo: Vec<f32> = data.get("output_head.weight").ok_or("missing output_head.weight")?
            .flatten_all()?.to_vec1()?;
        let bo: Vec<f32> = data.get("output_head.bias").ok_or("missing output_head.bias")?
            .flatten_all()?.to_vec1()?;
        floats.extend(&wo);
        floats.extend(&bo);

        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;
        Ok(())
    }
}

// ---------------------------------------------------------------------------
// Shared helpers for masked cross-entropy loss and accuracy
// ---------------------------------------------------------------------------

/// Compute masked cross-entropy loss from logits (batch, NUM_OUTPUTS).
fn masked_cross_entropy(
    logits: &Tensor,
    targets: &[u8],
    masks: &[u32],
    batch_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let logits_3d = logits.reshape((batch_size, 32, NUM_CLASSES))?;

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
    &loss_sum / &mask_sum
}

/// Card count regularization loss.
///
/// After per-card softmax, the predicted count of cards for each player should
/// match the actual count. For unknown cards only (masked), we compute:
///   predicted_count[player] = sum(softmax_prob[card][player]) for card in unknown
///   true_count[player] = number of cards that player actually holds (from targets)
///   loss = mean_over_players_and_batch( (predicted - true)^2 )
///
/// This is differentiable through the softmax and encourages realistic card distributions.
fn count_regularization(
    logits: &Tensor,
    targets: &[u8],
    masks: &[u32],
    batch_size: usize,
    device: &Device,
) -> Result<Tensor> {
    let logits_3d = logits.reshape((batch_size, 32, NUM_CLASSES))?;
    let probs = candle_nn::ops::softmax(&logits_3d, D::Minus1)?; // (batch, 32, 4)

    // Build mask: (batch, 32, 1) — broadcast over players
    let mut mask_flat = vec![0.0f32; batch_size * 32];
    for i in 0..batch_size {
        for c in 0..32 {
            if masks[i] & (1u32 << c) != 0 {
                mask_flat[i * 32 + c] = 1.0;
            }
        }
    }
    let mask_t = Tensor::from_slice(&mask_flat, (batch_size, 32, 1), device)?
        .broadcast_as((batch_size, 32, NUM_CLASSES))?.contiguous()?;

    // Masked probs: zero out known cards so they don't contribute to count
    let masked_probs = (&probs * &mask_t)?; // (batch, 32, NUM_CLASSES)

    // Predicted count per player: sum over cards dim
    let pred_counts = masked_probs.sum(1)?; // (batch, NUM_CLASSES)

    // True count per player from targets (only unknown cards)
    let mut true_counts = vec![0.0f32; batch_size * NUM_CLASSES];
    for i in 0..batch_size {
        for c in 0..32 {
            if masks[i] & (1u32 << c) != 0 {
                let player = targets[i * 32 + c] as usize;
                if player < NUM_CLASSES {
                    true_counts[i * NUM_CLASSES + player] += 1.0;
                }
            }
        }
    }
    let true_counts_t = Tensor::from_slice(&true_counts, (batch_size, NUM_CLASSES), device)?;

    // MSE over (batch, NUM_CLASSES)
    let diff = (&pred_counts - &true_counts_t)?;
    let sq = (&diff * &diff)?;
    sq.mean_all()
}

/// Compute per-card accuracy from logits (batch, NUM_OUTPUTS).
fn compute_accuracy(
    logits: &Tensor,
    targets: &[u8],
    masks: &[u32],
    batch_size: usize,
) -> Result<(u64, u64)> {
    let logits_3d = logits.reshape((batch_size, 32, NUM_CLASSES))?;
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

#[cfg(test)]
mod tests {
    use super::*;
    use crate::belief_obs::BELIEF_OBS_DIM;

    #[test]
    fn test_model_creation() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BeliefQNet::new(BELIEF_OBS_DIM, 64, vb)?;

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
        let net = BeliefQNet::new(BELIEF_OBS_DIM, 32, vb)?;

        let batch_size = 16;
        let obs = Tensor::randn(0f32, 1f32, (batch_size, BELIEF_OBS_DIM), &device)?;
        let logits = net.forward(&obs)?;
        assert_eq!(logits.dims(), &[batch_size, NUM_OUTPUTS]);
        Ok(())
    }

    #[test]
    fn test_trainer_train_step() -> Result<()> {
        let device = Device::Cpu;
        let mut trainer = BeliefTrainer::new(BELIEF_OBS_DIM, 32, 1e-3, 0.0, device)?;

        let batch = 4;
        let obs = vec![0.0f32; batch * BELIEF_OBS_DIM];
        let targets = vec![1u8; batch * 32];
        let masks = vec![0xFFFF_FFFFu32; batch];

        let loss = trainer.train_step(&obs, &targets, &masks)?;
        assert!(loss.is_finite(), "loss should be finite: {}", loss);
        assert!(loss > 0.5 && loss < 3.0, "initial loss should be ~1.099, got {}", loss);
        Ok(())
    }

    #[test]
    fn test_eval_accuracy() -> Result<()> {
        let device = Device::Cpu;
        let trainer = BeliefTrainer::new(BELIEF_OBS_DIM, 32, 1e-3, 0.0, device)?;

        let batch = 4;
        let obs = vec![0.0f32; batch * BELIEF_OBS_DIM];
        let targets = vec![0u8; batch * 32];
        let masks = vec![0xFFFF_FFFFu32; batch];

        let (correct, total) = trainer.eval_accuracy(&obs, &targets, &masks)?;
        assert_eq!(total, batch as u64 * 32);
        let acc = correct as f64 / total as f64;
        assert!(acc >= 0.0 && acc <= 1.0);
        Ok(())
    }

    #[test]
    fn test_export_and_load() -> Result<()> {
        let device = Device::Cpu;
        let trainer = BeliefTrainer::new(BELIEF_OBS_DIM, 32, 1e-3, 0.0, device)?;

        let tmp = "/tmp/test_belief_net.bin";
        trainer.export_binary(tmp).unwrap();

        let mut net = crate::belief_net::BeliefNet::load_with_hidden(tmp, 32).unwrap();
        assert_eq!(net.obs_dim(), BELIEF_OBS_DIM);

        let obs = vec![0.0f32; BELIEF_OBS_DIM];
        let logits = net.evaluate(&obs);
        assert_eq!(logits.len(), NUM_OUTPUTS);
        for &v in &logits {
            assert!(v.is_finite());
        }

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

    #[test]
    fn test_v2_obs_dim() -> Result<()> {
        use crate::belief_obs::BELIEF_OBS_DIM_V2;
        let device = Device::Cpu;
        let mut trainer = BeliefTrainer::new(BELIEF_OBS_DIM_V2, 32, 1e-3, 0.0, device)?;

        let batch = 2;
        let obs = vec![0.0f32; batch * BELIEF_OBS_DIM_V2];
        let targets = vec![0u8; batch * 32];
        let masks = vec![0xFFFF_FFFFu32; batch];

        let loss = trainer.train_step(&obs, &targets, &masks)?;
        assert!(loss.is_finite());
        Ok(())
    }
}
