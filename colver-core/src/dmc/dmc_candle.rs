/// Candle-based Dueling Q-Network for DMC training.
///
/// Architecture:
///   415 → FC(1024) → LN → ReLU → FC(1024) → LN → ReLU → FC(1024) → LN → ReLU
///     → Value head: FC(1) → V(s)
///     → Advantage head: FC(32) → A(s,a)
///     → Q(s,a) = V(s) + A(s,a) - mean(A)

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{self, linear, AdamW, Linear, Module,
                Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::dmc_obs::OBS_DIM;

const NUM_ACTIONS: usize = 32;

/// Manual LayerNorm using basic tensor ops (candle's built-in lacks CUDA kernel).
struct ManualLayerNorm {
    weight: Tensor, // gamma
    bias: Tensor,   // beta
    eps: f64,
}

impl ManualLayerNorm {
    fn new(size: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(size, "weight", candle_nn::Init::Const(1.0))?;
        let bias = vb.get_with_hints(size, "bias", candle_nn::Init::Const(0.0))?;
        Ok(ManualLayerNorm { weight, bias, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        // x: (batch, hidden)
        let mean = x.mean_keepdim(D::Minus1)?;        // (batch, 1)
        let centered = x.broadcast_sub(&mean)?;        // (batch, hidden)
        let var = centered.sqr()?.mean_keepdim(D::Minus1)?; // (batch, 1)
        let std = (var + self.eps)?.sqrt()?;           // (batch, 1)
        let normed = centered.broadcast_div(&std)?;    // (batch, hidden)
        normed.broadcast_mul(&self.weight)?.broadcast_add(&self.bias)
    }
}

/// Dueling Q-Network: shared trunk + separate value/advantage heads.
///
/// When `residual` is true, layers 1+ use skip connections:
///   x_new = ReLU(LN(FC(x)) + x)
/// Same weights — only the forward pass changes.
pub struct DuelingQNet {
    trunk_fc: [Linear; 3],
    trunk_ln: [ManualLayerNorm; 3],
    value_head: Linear,
    advantage_head: Linear,
    pub obs_dim: usize,
    pub residual: bool,
}

impl DuelingQNet {
    /// Create a new network with random initialization.
    pub fn new(hidden: usize, vb: VarBuilder) -> Result<Self> {
        Self::with_obs_dim(OBS_DIM, hidden, vb)
    }

    /// Create with explicit obs_dim (e.g. 411 for trump-relative encoding).
    pub fn with_obs_dim(obs_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Self::build(obs_dim, hidden, false, vb)
    }

    /// Create with explicit obs_dim and residual skip connections.
    pub fn with_residual(obs_dim: usize, hidden: usize, vb: VarBuilder) -> Result<Self> {
        Self::build(obs_dim, hidden, true, vb)
    }

    fn build(obs_dim: usize, hidden: usize, residual: bool, vb: VarBuilder) -> Result<Self> {
        let trunk_fc = [
            linear(obs_dim, hidden, vb.pp("trunk.0"))?,
            linear(hidden, hidden, vb.pp("trunk.1"))?,
            linear(hidden, hidden, vb.pp("trunk.2"))?,
        ];
        let trunk_ln = [
            ManualLayerNorm::new(hidden, 1e-5, vb.pp("trunk_ln.0"))?,
            ManualLayerNorm::new(hidden, 1e-5, vb.pp("trunk_ln.1"))?,
            ManualLayerNorm::new(hidden, 1e-5, vb.pp("trunk_ln.2"))?,
        ];
        let value_head = linear(hidden, 1, vb.pp("value_head"))?;
        let advantage_head = linear(hidden, NUM_ACTIONS, vb.pp("advantage_head"))?;

        Ok(DuelingQNet {
            trunk_fc,
            trunk_ln,
            value_head,
            advantage_head,
            obs_dim,
            residual,
        })
    }

    /// Forward pass: obs (batch, OBS_DIM) → Q (batch, 32).
    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        // Layer 0: input projection (no skip — different dims)
        let mut x = self.trunk_fc[0].forward(obs)?;
        x = self.trunk_ln[0].forward(&x)?;
        x = x.relu()?;

        // Layers 1-2: with optional residual skip connections
        for i in 1..3 {
            if self.residual {
                let residual = x.clone();
                x = self.trunk_fc[i].forward(&x)?;
                x = self.trunk_ln[i].forward(&x)?;
                x = (x + residual)?.relu()?;
            } else {
                x = self.trunk_fc[i].forward(&x)?;
                x = self.trunk_ln[i].forward(&x)?;
                x = x.relu()?;
            }
        }

        // Value head: (batch, H) → (batch, 1)
        let v = self.value_head.forward(&x)?;

        // Advantage head: (batch, H) → (batch, 32)
        let a = self.advantage_head.forward(&x)?;

        // Q = V + (A - mean(A))
        let a_mean = a.mean_keepdim(D::Minus1)?;
        let a_centered = a.broadcast_sub(&a_mean)?;
        let q = a_centered.broadcast_add(&v)?;

        Ok(q)
    }

    /// Select actions: ε-greedy on GPU, returns action indices (batch,).
    /// mask: (batch, 32) with 1.0 for legal actions.
    pub fn act(
        &self,
        obs: &Tensor,
        mask: &Tensor,
        epsilon: f32,
        rng: &mut impl rand::Rng,
    ) -> Result<Vec<u8>> {
        let batch_size = obs.dim(0)?;
        let q = self.forward(obs)?;

        // Mask illegal actions with large negative value
        let neg_inf = Tensor::full(-1e9f32, q.shape(), q.device())?;
        let mask_u8 = mask.to_dtype(DType::U8)?;
        let q_masked = mask_u8.where_cond(&q, &neg_inf)?;

        // Get greedy actions
        let greedy = q_masked.argmax(D::Minus1)?.to_vec1::<u32>()?;

        // ε-greedy with legal action sampling
        let mask_data: Vec<Vec<f32>> = mask.to_vec2()?;
        let mut actions = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            if rng.gen::<f32>() < epsilon {
                // Random legal action
                let legal: Vec<u8> = mask_data[i]
                    .iter()
                    .enumerate()
                    .filter(|(_, &v)| v > 0.5)
                    .map(|(j, _)| j as u8)
                    .collect();
                let idx = rng.gen_range(0..legal.len());
                actions.push(legal[idx]);
            } else {
                actions.push(greedy[i] as u8);
            }
        }

        Ok(actions)
    }
}

/// Training wrapper: owns the VarMap, optimizer, and model.
pub struct DuelingTrainer {
    pub net: DuelingQNet,
    pub varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    hidden: usize,
    obs_dim: usize,
}

impl DuelingTrainer {
    /// Create a new trainer with random initialization (default OBS_DIM=415).
    pub fn new(hidden: usize, lr: f64, weight_decay: f64, device: Device) -> Result<Self> {
        Self::with_obs_dim(OBS_DIM, hidden, lr, weight_decay, device)
    }

    /// Create with explicit obs_dim (e.g. 411 for trump-relative encoding).
    pub fn with_obs_dim(obs_dim: usize, hidden: usize, lr: f64, weight_decay: f64, device: Device) -> Result<Self> {
        Self::build(obs_dim, hidden, false, lr, weight_decay, device)
    }

    /// Create with explicit obs_dim and residual skip connections.
    pub fn with_residual(obs_dim: usize, hidden: usize, lr: f64, weight_decay: f64, device: Device) -> Result<Self> {
        Self::build(obs_dim, hidden, true, lr, weight_decay, device)
    }

    fn build(obs_dim: usize, hidden: usize, residual: bool, lr: f64, weight_decay: f64, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = DuelingQNet::build(obs_dim, hidden, residual, vb)?;

        let adamw_params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        };
        let optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

        Ok(DuelingTrainer {
            net,
            varmap,
            optimizer,
            device,
            hidden,
            obs_dim,
        })
    }

    /// Set learning rate.
    pub fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Single training step. Returns (loss_value, td_errors).
    ///
    /// - obs: flat f32 slice (batch * OBS_DIM)
    /// - masks: flat f32 slice (batch * 32)
    /// - actions: u8 slice (batch,)
    /// - returns: f32 slice (batch,) — MC returns (win=1.0, loss=0.0)
    /// - weights: f32 slice (batch,) — PER importance sampling weights
    pub fn train_step(
        &mut self,
        obs: &[f32],
        _masks: &[f32],
        actions: &[u8],
        returns: &[f32],
        weights: &[f32],
    ) -> Result<(f32, Vec<f32>)> {
        let batch_size = actions.len();
        let device = &self.device;

        // Create tensors on device
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), device)?;
        let returns_t = Tensor::from_slice(returns, batch_size, device)?;
        let weights_t = Tensor::from_slice(weights, batch_size, device)?;

        // Forward pass
        let q_all = self.net.forward(&obs_t)?; // (batch, 32)

        // Gather Q-values for taken actions
        let actions_i64: Vec<u32> = actions.iter().map(|&a| a as u32).collect();
        let actions_t = Tensor::from_slice(&actions_i64, (batch_size, 1), device)?;
        let q_taken = q_all.gather(&actions_t, D::Minus1)?.squeeze(D::Minus1)?; // (batch,)

        // TD errors = Q(s,a) - return (for PER priority updates)
        let td_errors_t = (&q_taken - &returns_t)?;
        let td_errors: Vec<f32> = td_errors_t.detach().to_vec1()?;

        // Weighted MSE loss: mean(weights * (Q(s,a) - return)^2)
        let sq_errors = td_errors_t.sqr()?;
        let weighted = (&sq_errors * &weights_t)?;
        let loss = weighted.mean_all()?;

        // Backward + optimizer step
        self.optimizer.backward_step(&loss)?;

        let loss_val: f32 = loss.detach().to_vec0()?;
        Ok((loss_val, td_errors))
    }

    /// Get Q-values for a batch of observations (no grad).
    pub fn q_values(&self, obs: &[f32], batch_size: usize) -> Result<Vec<f32>> {
        let obs_t = Tensor::from_slice(obs, (batch_size, self.obs_dim), &self.device)?;
        let q = self.net.forward(&obs_t)?;
        q.flatten_all()?.to_vec1()
    }

    /// Save checkpoint as safetensors.
    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    /// Load checkpoint from safetensors.
    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        Ok(())
    }

    /// Export weights as raw f32 binary compatible with DmcNet (dueling format).
    ///
    /// Layout: 3× (W + b + gamma + beta) + value(W + b) + advantage(W + b)
    pub fn export_binary(&self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let data = self.varmap.data().lock().unwrap();
        let mut floats: Vec<f32> = Vec::new();
        let in_dims = [self.obs_dim, self.hidden, self.hidden];

        for i in 0..3 {
            // Weight: Linear stores weight as (out_dim, in_dim), we need row-major (out_dim, in_dim)
            let w_key = format!("trunk.{}.weight", i);
            let b_key = format!("trunk.{}.bias", i);
            let gamma_key = format!("trunk_ln.{}.weight", i);
            let beta_key = format!("trunk_ln.{}.bias", i);

            let w = data.get(&w_key).ok_or_else(|| format!("missing {}", w_key))?;
            let b = data.get(&b_key).ok_or_else(|| format!("missing {}", b_key))?;
            let gamma = data.get(&gamma_key).ok_or_else(|| format!("missing {}", gamma_key))?;
            let beta = data.get(&beta_key).ok_or_else(|| format!("missing {}", beta_key))?;

            // candle Linear weight is (out_dim, in_dim), same as our row-major layout
            let w_data: Vec<f32> = w.flatten_all()?.to_vec1()?;
            let b_data: Vec<f32> = b.flatten_all()?.to_vec1()?;
            let gamma_data: Vec<f32> = gamma.flatten_all()?.to_vec1()?;
            let beta_data: Vec<f32> = beta.flatten_all()?.to_vec1()?;

            assert_eq!(w_data.len(), self.hidden * in_dims[i]);
            assert_eq!(b_data.len(), self.hidden);
            assert_eq!(gamma_data.len(), self.hidden);
            assert_eq!(beta_data.len(), self.hidden);

            floats.extend(&w_data);
            floats.extend(&b_data);
            floats.extend(&gamma_data);
            floats.extend(&beta_data);
        }

        // Value head: (H, 1) → flatten to H, plus bias 1
        let wv = data.get("value_head.weight").ok_or("missing value_head.weight")?;
        let bv = data.get("value_head.bias").ok_or("missing value_head.bias")?;
        let wv_data: Vec<f32> = wv.flatten_all()?.to_vec1()?;
        let bv_data: Vec<f32> = bv.flatten_all()?.to_vec1()?;
        assert_eq!(wv_data.len(), self.hidden);
        assert_eq!(bv_data.len(), 1);
        floats.extend(&wv_data);
        floats.extend(&bv_data);

        // Advantage head: (32, H)
        let wa = data.get("advantage_head.weight").ok_or("missing advantage_head.weight")?;
        let ba = data.get("advantage_head.bias").ok_or("missing advantage_head.bias")?;
        let wa_data: Vec<f32> = wa.flatten_all()?.to_vec1()?;
        let ba_data: Vec<f32> = ba.flatten_all()?.to_vec1()?;
        assert_eq!(wa_data.len(), self.hidden * NUM_ACTIONS);
        assert_eq!(ba_data.len(), NUM_ACTIONS);
        floats.extend(&wa_data);
        floats.extend(&ba_data);

        // Write raw f32 binary
        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;

        Ok(())
    }

    /// Extract flat weight vector for DmcNet CPU inference (opponent pool).
    pub fn snapshot_weights(&self) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error>> {
        let data = self.varmap.data().lock().unwrap();
        let mut floats: Vec<f32> = Vec::new();

        for i in 0..3 {
            let w: Vec<f32> = data.get(&format!("trunk.{}.weight", i)).unwrap().flatten_all()?.to_vec1()?;
            let b: Vec<f32> = data.get(&format!("trunk.{}.bias", i)).unwrap().flatten_all()?.to_vec1()?;
            let gamma: Vec<f32> = data.get(&format!("trunk_ln.{}.weight", i)).unwrap().flatten_all()?.to_vec1()?;
            let beta: Vec<f32> = data.get(&format!("trunk_ln.{}.bias", i)).unwrap().flatten_all()?.to_vec1()?;
            floats.extend(w);
            floats.extend(b);
            floats.extend(gamma);
            floats.extend(beta);
        }

        // Value head
        let wv: Vec<f32> = data.get("value_head.weight").unwrap().flatten_all()?.to_vec1()?;
        let bv: Vec<f32> = data.get("value_head.bias").unwrap().flatten_all()?.to_vec1()?;
        floats.extend(wv);
        floats.extend(bv);

        // Advantage head
        let wa: Vec<f32> = data.get("advantage_head.weight").unwrap().flatten_all()?.to_vec1()?;
        let ba: Vec<f32> = data.get("advantage_head.bias").unwrap().flatten_all()?.to_vec1()?;
        floats.extend(wa);
        floats.extend(ba);

        Ok(floats)
    }

    /// Get device reference.
    pub fn device(&self) -> &Device {
        &self.device
    }
}

/// GPU-based inference net for opponent pool (no optimizer, no grad).
///
/// Loaded from flat weight snapshots. All pool envs batch through this
/// single net on GPU instead of per-env CPU DmcNet inference.
pub struct PoolNet {
    net: DuelingQNet,
    varmap: VarMap,
    device: Device,
    hidden: usize,
    obs_dim: usize,
}

impl PoolNet {
    /// Create a new PoolNet (random init, call `load_weights` before use).
    pub fn new(hidden: usize, device: &Device) -> Result<Self> {
        Self::with_obs_dim(OBS_DIM, hidden, device)
    }

    /// Create with explicit obs_dim.
    pub fn with_obs_dim(obs_dim: usize, hidden: usize, device: &Device) -> Result<Self> {
        Self::build_pool(obs_dim, hidden, false, device)
    }

    /// Create with explicit obs_dim and residual skip connections.
    pub fn with_residual(obs_dim: usize, hidden: usize, device: &Device) -> Result<Self> {
        Self::build_pool(obs_dim, hidden, true, device)
    }

    fn build_pool(obs_dim: usize, hidden: usize, residual: bool, device: &Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, device);
        let net = DuelingQNet::build(obs_dim, hidden, residual, vb)?;
        Ok(PoolNet { net, varmap, device: device.clone(), hidden, obs_dim })
    }

    /// Load weights from flat f32 vector (same format as `snapshot_weights` output).
    pub fn load_weights(&self, weights: &[f32]) -> Result<()> {
        let data = self.varmap.data().lock().unwrap();
        let mut offset = 0;
        let in_dims = [self.obs_dim, self.hidden, self.hidden];

        for i in 0..3 {
            let w_size = self.hidden * in_dims[i];
            let w_t = Tensor::from_slice(&weights[offset..offset + w_size], (self.hidden, in_dims[i]), &self.device)?;
            offset += w_size;
            data.get(&format!("trunk.{}.weight", i)).unwrap().set(&w_t)?;

            let b_t = Tensor::from_slice(&weights[offset..offset + self.hidden], self.hidden, &self.device)?;
            offset += self.hidden;
            data.get(&format!("trunk.{}.bias", i)).unwrap().set(&b_t)?;

            let g_t = Tensor::from_slice(&weights[offset..offset + self.hidden], self.hidden, &self.device)?;
            offset += self.hidden;
            data.get(&format!("trunk_ln.{}.weight", i)).unwrap().set(&g_t)?;

            let bt_t = Tensor::from_slice(&weights[offset..offset + self.hidden], self.hidden, &self.device)?;
            offset += self.hidden;
            data.get(&format!("trunk_ln.{}.bias", i)).unwrap().set(&bt_t)?;
        }

        // Value head: (1, hidden)
        let wv = Tensor::from_slice(&weights[offset..offset + self.hidden], (1, self.hidden), &self.device)?;
        offset += self.hidden;
        data.get("value_head.weight").unwrap().set(&wv)?;

        let bv = Tensor::from_slice(&weights[offset..offset + 1], 1, &self.device)?;
        offset += 1;
        data.get("value_head.bias").unwrap().set(&bv)?;

        // Advantage head: (32, hidden)
        let wa_size = NUM_ACTIONS * self.hidden;
        let wa = Tensor::from_slice(&weights[offset..offset + wa_size], (NUM_ACTIONS, self.hidden), &self.device)?;
        offset += wa_size;
        data.get("advantage_head.weight").unwrap().set(&wa)?;

        let ba = Tensor::from_slice(&weights[offset..offset + NUM_ACTIONS], NUM_ACTIONS, &self.device)?;
        let _ = offset + NUM_ACTIONS; // consumed
        data.get("advantage_head.bias").unwrap().set(&ba)?;

        Ok(())
    }

    /// Greedy action selection on GPU (no exploration).
    /// obs: (batch, OBS_DIM), mask: (batch, 32) → Vec<u8> actions.
    pub fn act_greedy(&self, obs: &Tensor, mask: &Tensor) -> Result<Vec<u8>> {
        let q = self.net.forward(obs)?;
        let neg_inf = Tensor::full(-1e9f32, q.shape(), q.device())?;
        let mask_u8 = mask.to_dtype(DType::U8)?;
        let q_masked = mask_u8.where_cond(&q, &neg_inf)?;
        let greedy = q_masked.argmax(D::Minus1)?.to_vec1::<u32>()?;
        Ok(greedy.into_iter().map(|a| a as u8).collect())
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
        let net = DuelingQNet::new(64, vb)?;

        // Verify forward pass with dummy input
        let obs = Tensor::zeros((2, OBS_DIM), DType::F32, &device)?;
        let q = net.forward(&obs)?;
        assert_eq!(q.dims(), &[2, NUM_ACTIONS]);
        Ok(())
    }

    #[test]
    fn test_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = DuelingQNet::new(32, vb)?;

        let batch_size = 16;
        let obs = Tensor::randn(0.0f32, 1.0, (batch_size, OBS_DIM), &device)?;
        let q = net.forward(&obs)?;
        assert_eq!(q.dims(), &[batch_size, NUM_ACTIONS]);
        Ok(())
    }

    #[test]
    fn test_trainer_train_step() -> Result<()> {
        let device = Device::Cpu;
        let mut trainer = DuelingTrainer::new(32, 1e-3, 0.0, device)?;

        let batch = 4;
        let obs = vec![0.0f32; batch * OBS_DIM];
        let masks = vec![1.0f32; batch * 32];
        let actions = vec![0u8, 5, 10, 15];
        let returns = vec![1.0f32, 0.0, 1.0, 0.5];
        let weights = vec![1.0f32; batch];

        let (loss, td_errors) = trainer.train_step(&obs, &masks, &actions, &returns, &weights)?;
        assert!(loss.is_finite(), "loss should be finite: {}", loss);
        assert_eq!(td_errors.len(), batch);
        Ok(())
    }

    #[test]
    fn test_trainer_export_roundtrip() -> Result<()> {
        let device = Device::Cpu;
        let trainer = DuelingTrainer::new(32, 1e-3, 0.0, device)?;

        // Export to flat weights
        let weights = trainer.snapshot_weights().unwrap();

        // Load into DmcNet
        let mut net = crate::dmc_net::DmcNet::from_floats(&weights, 32, OBS_DIM, true).unwrap();
        assert!(net.is_dueling());
        assert_eq!(net.obs_dim(), OBS_DIM);

        // Verify forward pass gives same results
        let obs = vec![0.0f32; OBS_DIM];
        let q_rust = net.evaluate(&obs);

        let obs_t = Tensor::zeros((1, OBS_DIM), DType::F32, trainer.device())?;
        let q_candle = trainer.net.forward(&obs_t)?;
        let q_candle_vec: Vec<f32> = q_candle.flatten_all()?.to_vec1()?;

        for i in 0..NUM_ACTIONS {
            let diff = (q_rust[i] - q_candle_vec[i]).abs();
            assert!(
                diff < 1e-4,
                "Q[{}]: rust={:.6}, candle={:.6}, diff={:.6}",
                i, q_rust[i], q_candle_vec[i], diff,
            );
        }
        Ok(())
    }

    #[test]
    fn test_act_epsilon_greedy() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = DuelingQNet::new(32, vb)?;

        let batch = 8;
        let obs = Tensor::zeros((batch, OBS_DIM), DType::F32, &device)?;
        // All actions legal
        let mask = Tensor::ones((batch, NUM_ACTIONS), DType::F32, &device)?;

        let mut rng = rand::thread_rng();
        let actions = net.act(&obs, &mask, 0.0, &mut rng)?; // greedy
        assert_eq!(actions.len(), batch);
        for &a in &actions {
            assert!(a < NUM_ACTIONS as u8);
        }

        let actions = net.act(&obs, &mask, 1.0, &mut rng)?; // fully random
        assert_eq!(actions.len(), batch);
        Ok(())
    }
}
