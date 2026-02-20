/// Candle-based Dueling Q-Network for NN bidding training.
///
/// Architecture:
///   114 → FC(256) → LN → ReLU → FC(256) → LN → ReLU
///     → Value head: FC(1) → V(s)
///     → Advantage head: FC(43) → A(s,a)
///     → Q(s,a) = V(s) + A(s,a) - mean(A)

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{self, linear, AdamW, Linear, Module, Optimizer, ParamsAdamW, VarBuilder, VarMap};

use crate::bid_obs::BID_OBS_DIM;

const NUM_ACTIONS: usize = 43;

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

/// Dueling Q-Network for bidding: 2-layer trunk + value/advantage heads.
pub struct BiddingQNet {
    trunk_fc: [Linear; 2],
    trunk_ln: [ManualLayerNorm; 2],
    value_head: Linear,
    advantage_head: Linear,
}

impl BiddingQNet {
    pub fn new(hidden: usize, vb: VarBuilder) -> Result<Self> {
        let trunk_fc = [
            linear(BID_OBS_DIM, hidden, vb.pp("trunk.0"))?,
            linear(hidden, hidden, vb.pp("trunk.1"))?,
        ];
        let trunk_ln = [
            ManualLayerNorm::new(hidden, 1e-5, vb.pp("trunk_ln.0"))?,
            ManualLayerNorm::new(hidden, 1e-5, vb.pp("trunk_ln.1"))?,
        ];
        let value_head = linear(hidden, 1, vb.pp("value_head"))?;
        let advantage_head = linear(hidden, NUM_ACTIONS, vb.pp("advantage_head"))?;

        Ok(BiddingQNet {
            trunk_fc,
            trunk_ln,
            value_head,
            advantage_head,
        })
    }

    /// Forward pass: obs (batch, BID_OBS_DIM) → Q (batch, 43).
    pub fn forward(&self, obs: &Tensor) -> Result<Tensor> {
        let mut x = obs.clone();
        for i in 0..2 {
            x = self.trunk_fc[i].forward(&x)?;
            x = self.trunk_ln[i].forward(&x)?;
            x = x.relu()?;
        }

        let v = self.value_head.forward(&x)?;
        let a = self.advantage_head.forward(&x)?;

        let a_mean = a.mean_keepdim(D::Minus1)?;
        let a_centered = a.broadcast_sub(&a_mean)?;
        let q = a_centered.broadcast_add(&v)?;

        Ok(q)
    }

    /// ε-greedy action selection with legal mask.
    pub fn act(
        &self,
        obs: &Tensor,
        mask: &Tensor,
        epsilon: f32,
        rng: &mut impl rand::Rng,
    ) -> Result<Vec<u8>> {
        let batch_size = obs.dim(0)?;
        let q = self.forward(obs)?;

        let neg_inf = Tensor::full(-1e9f32, q.shape(), q.device())?;
        let mask_u8 = mask.to_dtype(DType::U8)?;
        let q_masked = mask_u8.where_cond(&q, &neg_inf)?;

        let greedy = q_masked.argmax(D::Minus1)?.to_vec1::<u32>()?;

        let mask_data: Vec<Vec<f32>> = mask.to_vec2()?;
        let mut actions = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            if rng.gen::<f32>() < epsilon {
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

/// Training wrapper: VarMap, optimizer, model.
pub struct BiddingTrainer {
    pub net: BiddingQNet,
    pub varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    hidden: usize,
}

impl BiddingTrainer {
    pub fn new(hidden: usize, lr: f64, weight_decay: f64, device: Device) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BiddingQNet::new(hidden, vb)?;

        let adamw_params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.999,
            eps: 1e-8,
            weight_decay,
        };
        let optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

        Ok(BiddingTrainer {
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

    /// Single training step. Returns (loss_value, td_errors).
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

        let obs_t = Tensor::from_slice(obs, (batch_size, BID_OBS_DIM), device)?;
        let returns_t = Tensor::from_slice(returns, batch_size, device)?;
        let weights_t = Tensor::from_slice(weights, batch_size, device)?;

        let q_all = self.net.forward(&obs_t)?;

        let actions_u32: Vec<u32> = actions.iter().map(|&a| a as u32).collect();
        let actions_t = Tensor::from_slice(&actions_u32, (batch_size, 1), device)?;
        let q_taken = q_all.gather(&actions_t, D::Minus1)?.squeeze(D::Minus1)?;

        let td_errors_t = (&q_taken - &returns_t)?;
        let td_errors: Vec<f32> = td_errors_t.detach().to_vec1()?;

        let sq_errors = td_errors_t.sqr()?;
        let weighted = (&sq_errors * &weights_t)?;
        let loss = weighted.mean_all()?;

        self.optimizer.backward_step(&loss)?;

        let loss_val: f32 = loss.detach().to_vec0()?;
        Ok((loss_val, td_errors))
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        Ok(())
    }

    /// Export weights as raw f32 binary compatible with BidNet (dueling format).
    pub fn export_binary(&self, path: &str) -> std::result::Result<(), Box<dyn std::error::Error>> {
        let data = self.varmap.data().lock().unwrap();
        let mut floats: Vec<f32> = Vec::new();
        let in_dims = [BID_OBS_DIM, self.hidden];

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

        // Value head
        let wv: Vec<f32> = data
            .get("value_head.weight")
            .ok_or("missing value_head.weight")?
            .flatten_all()?
            .to_vec1()?;
        let bv: Vec<f32> = data
            .get("value_head.bias")
            .ok_or("missing value_head.bias")?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(wv.len(), self.hidden);
        assert_eq!(bv.len(), 1);
        floats.extend(&wv);
        floats.extend(&bv);

        // Advantage head
        let wa: Vec<f32> = data
            .get("advantage_head.weight")
            .ok_or("missing advantage_head.weight")?
            .flatten_all()?
            .to_vec1()?;
        let ba: Vec<f32> = data
            .get("advantage_head.bias")
            .ok_or("missing advantage_head.bias")?
            .flatten_all()?
            .to_vec1()?;
        assert_eq!(wa.len(), self.hidden * NUM_ACTIONS);
        assert_eq!(ba.len(), NUM_ACTIONS);
        floats.extend(&wa);
        floats.extend(&ba);

        let bytes: Vec<u8> = floats.iter().flat_map(|f| f.to_le_bytes()).collect();
        std::fs::write(path, bytes)?;

        Ok(())
    }

    /// Extract flat weight vector for BidNet CPU inference.
    pub fn snapshot_weights(&self) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error>> {
        let data = self.varmap.data().lock().unwrap();
        let mut floats: Vec<f32> = Vec::new();

        for i in 0..2 {
            let w: Vec<f32> = data
                .get(&format!("trunk.{}.weight", i))
                .unwrap()
                .flatten_all()?
                .to_vec1()?;
            let b: Vec<f32> = data
                .get(&format!("trunk.{}.bias", i))
                .unwrap()
                .flatten_all()?
                .to_vec1()?;
            let gamma: Vec<f32> = data
                .get(&format!("trunk_ln.{}.weight", i))
                .unwrap()
                .flatten_all()?
                .to_vec1()?;
            let beta: Vec<f32> = data
                .get(&format!("trunk_ln.{}.bias", i))
                .unwrap()
                .flatten_all()?
                .to_vec1()?;
            floats.extend(w);
            floats.extend(b);
            floats.extend(gamma);
            floats.extend(beta);
        }

        let wv: Vec<f32> = data
            .get("value_head.weight")
            .unwrap()
            .flatten_all()?
            .to_vec1()?;
        let bv: Vec<f32> = data
            .get("value_head.bias")
            .unwrap()
            .flatten_all()?
            .to_vec1()?;
        floats.extend(wv);
        floats.extend(bv);

        let wa: Vec<f32> = data
            .get("advantage_head.weight")
            .unwrap()
            .flatten_all()?
            .to_vec1()?;
        let ba: Vec<f32> = data
            .get("advantage_head.bias")
            .unwrap()
            .flatten_all()?
            .to_vec1()?;
        floats.extend(wa);
        floats.extend(ba);

        Ok(floats)
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
        let net = BiddingQNet::new(64, vb)?;

        let obs = Tensor::zeros((2, BID_OBS_DIM), DType::F32, &device)?;
        let q = net.forward(&obs)?;
        assert_eq!(q.dims(), &[2, NUM_ACTIONS]);
        Ok(())
    }

    #[test]
    fn test_forward_shape() -> Result<()> {
        let device = Device::Cpu;
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BiddingQNet::new(256, vb)?;

        let obs = Tensor::randn(0f32, 1f32, (8, BID_OBS_DIM), &device)?;
        let q = net.forward(&obs)?;
        assert_eq!(q.dims(), &[8, NUM_ACTIONS]);
        Ok(())
    }

    #[test]
    fn test_trainer_train_step() -> Result<()> {
        let device = Device::Cpu;
        let mut trainer = BiddingTrainer::new(32, 1e-3, 0.0, device)?;

        let batch = 4;
        let obs = vec![0.5f32; batch * BID_OBS_DIM];
        let masks = vec![1.0f32; batch * NUM_ACTIONS];
        let actions = vec![0u8, 1, 5, 10];
        let returns = vec![0.5f32, -0.3, 0.2, 0.0];
        let weights = vec![1.0f32; batch];

        let (loss, td_errors) = trainer.train_step(&obs, &masks, &actions, &returns, &weights)?;
        assert!(loss.is_finite());
        assert_eq!(td_errors.len(), batch);
        Ok(())
    }

    #[test]
    fn test_export_and_load_binary() -> Result<()> {
        let device = Device::Cpu;
        let trainer = BiddingTrainer::new(32, 1e-3, 0.0, device)?;

        let tmp = "/tmp/test_bid_net.bin";
        trainer.export_binary(tmp).unwrap();

        // Load with BidNet
        let mut net = crate::bid_net::BidNet::load_with_hidden(tmp, 32).unwrap();
        assert!(net.is_dueling());
        assert_eq!(net.obs_dim(), BID_OBS_DIM);

        let obs = vec![0.0f32; BID_OBS_DIM];
        let q = net.evaluate(&obs);
        assert_eq!(q.len(), 43);
        for &v in &q {
            assert!(v.is_finite());
        }

        std::fs::remove_file(tmp).ok();
        Ok(())
    }
}
