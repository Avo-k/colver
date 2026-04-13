/// Bumblebid transformer implemented with tch-rs (libtorch backend).
///
/// Same architecture as bumblebid_candle.rs but using PyTorch's C++ backend
/// for ~3-4x faster GPU training.

use tch::{nn, nn::Module, nn::OptimizerConfig, Device, Kind, Tensor};

use crate::bid_obs::BID_OBS_DIM;

const NUM_ACTIONS: usize = 43;
const MAX_SEQ_LEN: usize = 34;

// Token IDs (must match model.py / bumblebid_candle.rs)
const P_NONE: i64 = 0;
const P_CLS: i64 = 1;
const P_POS0: i64 = 2;
const P_RANK0: i64 = 6;
const P_VAL0: i64 = 14;
const P_CAPOT: i64 = 23;
const P_PASS: i64 = 24;
const P_COINCHE: i64 = 25;
const P_SURCOINCHE: i64 = 26;
const NUM_PRIMARY: i64 = 27;

const S_NULL: i64 = 4;
const NUM_SUITS: i64 = 5;

// ---------------------------------------------------------------------------
// Convert 108-dim obs batch to token tensors
// ---------------------------------------------------------------------------
fn obs_batch_to_tokens(obs: &[f32], batch_size: usize, device: Device) -> (Tensor, Tensor) {
    let mut primary = vec![0i64; batch_size * MAX_SEQ_LEN];
    let mut suits = vec![S_NULL; batch_size * MAX_SEQ_LEN];

    for b in 0..batch_size {
        let obs_b = &obs[b * BID_OBS_DIM..(b + 1) * BID_OBS_DIM];
        let base = b * MAX_SEQ_LEN;

        // Hand cards
        let mut cards = Vec::with_capacity(8);
        for bit in 0..32u32 {
            if obs_b[bit as usize] > 0.5 {
                cards.push((bit % 8, bit / 8));
            }
        }
        cards.sort_by_key(|&(r, s)| s * 8 + r);

        let pos = (0..4i64).find(|&p| obs_b[104 + p as usize] > 0.5).unwrap_or(0);

        primary[base] = P_CLS;
        primary[base + 1] = P_POS0 + pos;
        for (j, &(rank, suit)) in cards.iter().enumerate().take(8) {
            primary[base + 2 + j] = P_RANK0 + rank as i64;
            suits[base + 2 + j] = suit as i64;
        }

        let mut tok_pos = 10;
        for slot in 0..12 {
            if tok_pos + 2 > MAX_SEQ_LEN {
                break;
            }
            let sb = 32 + slot * 6;
            let tf = obs_b[sb];
            if tf < 0.1 {
                continue;
            }
            let (p_tok, s_tok) = if tf < 0.3 {
                (P_PASS, S_NULL)
            } else if tf < 0.5 {
                let ve = (obs_b[sb + 1] * 250.0 / 10.0).round() as i64;
                let vi = (ve - 8).clamp(0, 8);
                let si = (0..4i64)
                    .max_by(|&a, &b| {
                        obs_b[sb + 2 + a as usize]
                            .partial_cmp(&obs_b[sb + 2 + b as usize])
                            .unwrap()
                    })
                    .unwrap();
                (P_VAL0 + vi, si)
            } else if tf < 0.7 {
                let si = (0..4i64)
                    .max_by(|&a, &b| {
                        obs_b[sb + 2 + a as usize]
                            .partial_cmp(&obs_b[sb + 2 + b as usize])
                            .unwrap()
                    })
                    .unwrap();
                (P_CAPOT, si)
            } else if tf < 0.9 {
                (P_COINCHE, S_NULL)
            } else {
                (P_SURCOINCHE, S_NULL)
            };

            primary[base + tok_pos] = p_tok;
            suits[base + tok_pos] = s_tok;
            primary[base + tok_pos + 1] = P_NONE;
            suits[base + tok_pos + 1] = if s_tok < 4 { s_tok } else { S_NULL };
            tok_pos += 2;
        }
    }

    let p_t = Tensor::from_slice(&primary)
        .reshape([batch_size as i64, MAX_SEQ_LEN as i64])
        .to(device);
    let s_t = Tensor::from_slice(&suits)
        .reshape([batch_size as i64, MAX_SEQ_LEN as i64])
        .to(device);
    (p_t, s_t)
}

// ---------------------------------------------------------------------------
// Model
// ---------------------------------------------------------------------------

#[derive(Debug)]
struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    fn new(vs: &nn::Path, dim: i64, eps: f64) -> Self {
        let weight = vs.ones("weight", &[dim]);
        RMSNorm { weight, eps }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let rms = x
            .pow_tensor_scalar(2)
            .mean_dim(-1, true, Kind::Float)
            .f_add_scalar(self.eps)
            .unwrap()
            .sqrt();
        (x / &rms) * &self.weight
    }
}

#[derive(Debug)]
struct GeGLU {
    w_gate: nn::Linear,
    w_up: nn::Linear,
    w_down: nn::Linear,
}

impl GeGLU {
    fn new(vs: &nn::Path, d_model: i64, d_ff: i64) -> Self {
        let no_bias = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        GeGLU {
            w_gate: nn::linear(vs / "w_gate", d_model, d_ff, no_bias),
            w_up: nn::linear(vs / "w_up", d_model, d_ff, no_bias),
            w_down: nn::linear(vs / "w_down", d_ff, d_model, no_bias),
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let gate = self.w_gate.forward(x).gelu("none");
        let up = self.w_up.forward(x);
        self.w_down.forward(&(gate * up))
    }
}

#[derive(Debug)]
struct TransformerBlock {
    attn_norm: RMSNorm,
    qkv_proj: nn::Linear,
    out_proj: nn::Linear,
    ffn_norm: RMSNorm,
    ffn: GeGLU,
    n_heads: i64,
    head_dim: i64,
}

impl TransformerBlock {
    fn new(vs: &nn::Path, d_model: i64, n_heads: i64, d_ff: i64) -> Self {
        let no_bias = nn::LinearConfig {
            bias: false,
            ..Default::default()
        };
        TransformerBlock {
            attn_norm: RMSNorm::new(&(vs / "attn_norm"), d_model, 1e-6),
            qkv_proj: nn::linear(vs / "qkv_proj", d_model, 3 * d_model, no_bias),
            out_proj: nn::linear(vs / "out_proj", d_model, d_model, no_bias),
            ffn_norm: RMSNorm::new(&(vs / "ffn_norm"), d_model, 1e-6),
            ffn: GeGLU::new(&(vs / "ffn"), d_model, d_ff),
            n_heads,
            head_dim: d_model / n_heads,
        }
    }

    fn forward(&self, x: &Tensor) -> Tensor {
        let (b, l, _d) = x.size3().unwrap();
        let d = self.n_heads * self.head_dim;

        let h = self.attn_norm.forward(x);
        let qkv = self.qkv_proj.forward(&h);
        let q = qkv.narrow(2, 0, d).reshape([b, l, self.n_heads, self.head_dim]).permute([0, 2, 1, 3]);
        let k = qkv.narrow(2, d, d).reshape([b, l, self.n_heads, self.head_dim]).permute([0, 2, 1, 3]);
        let v = qkv.narrow(2, 2 * d, d).reshape([b, l, self.n_heads, self.head_dim]).permute([0, 2, 1, 3]);

        // Manual scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(-2, -1)) / scale;
        let attn = scores.softmax(-1, Kind::Float);
        let h = attn.matmul(&v);
        let h = h.permute([0, 2, 1, 3]).contiguous().reshape([b, l, d]);
        let h = self.out_proj.forward(&h);

        let x = x + h;
        let ffn_out = self.ffn.forward(&self.ffn_norm.forward(&x));
        x + ffn_out
    }
}

#[derive(Debug)]
pub struct BumblebidNet {
    primary_emb: nn::Embedding,
    suit_emb: nn::Embedding,
    pos_emb: nn::Embedding,
    layers: Vec<TransformerBlock>,
    out_norm: RMSNorm,
    value_head: nn::Linear,
    advantage_head: nn::Linear,
    pub d_model: i64,
    pub n_layers: usize,
    pub n_heads: i64,
    device: Device,
}

impl BumblebidNet {
    pub fn new(vs: &nn::Path, d_model: i64, n_layers: usize, n_heads: i64) -> Self {
        let d_ff = (2 * 4 * d_model) / 3;
        let emb_config = Default::default();
        let no_bias = nn::LinearConfig { bias: false, ..Default::default() };

        let primary_emb = nn::embedding(vs / "primary_emb", NUM_PRIMARY, d_model, emb_config);
        let suit_emb = nn::embedding(vs / "suit_emb", NUM_SUITS, d_model, emb_config);
        let pos_emb = nn::embedding(vs / "pos_emb", MAX_SEQ_LEN as i64, d_model, emb_config);

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(TransformerBlock::new(
                &(vs / format!("layers_{}", i)),
                d_model, n_heads, d_ff,
            ));
        }

        let out_norm = RMSNorm::new(&(vs / "out_norm"), d_model, 1e-6);
        let value_head = nn::linear(vs / "value_head", d_model, 1, no_bias);
        let advantage_head = nn::linear(vs / "advantage_head", d_model, NUM_ACTIONS as i64, no_bias);

        let device = vs.device();
        BumblebidNet {
            primary_emb, suit_emb, pos_emb, layers, out_norm,
            value_head, advantage_head, d_model, n_layers, n_heads, device,
        }
    }

    fn forward_tokens(&self, primary_ids: &Tensor, suit_ids: &Tensor) -> Tensor {
        let (_b, l) = primary_ids.size2().unwrap();
        let positions = Tensor::arange(l, (Kind::Int64, self.device));

        let mut x = self.primary_emb.forward(primary_ids)
            + self.suit_emb.forward(suit_ids)
            + self.pos_emb.forward(&positions);

        for layer in &self.layers {
            x = layer.forward(&x);
        }

        let cls = x.select(1, 0); // CLS token
        let normed = self.out_norm.forward(&cls);

        let v = self.value_head.forward(&normed);
        let a = self.advantage_head.forward(&normed);
        let a_mean = a.mean_dim(-1, true, Kind::Float);
        &v + &a - &a_mean
    }

    pub fn forward_obs(&self, obs: &[f32], batch_size: usize) -> Tensor {
        let (p_t, s_t) = obs_batch_to_tokens(obs, batch_size, self.device);
        self.forward_tokens(&p_t, &s_t)
    }

    pub fn act(&self, obs: &[f32], masks: &[f32], batch_size: usize, epsilon: f32, rng: &mut impl rand::Rng) -> Vec<u8> {
        let q = tch::no_grad(|| self.forward_obs(obs, batch_size));
        let masks_t = Tensor::from_slice(masks)
            .reshape([batch_size as i64, NUM_ACTIONS as i64])
            .to(self.device);
        let neg_inf = Tensor::full(q.size().as_slice(), -1e9f64, (Kind::Float, self.device));
        let cond = masks_t.gt(0.5);
        let q_masked = cond.where_self(&q, &neg_inf);
        let greedy: Vec<i64> = q_masked.argmax(-1, false).to(Device::Cpu).try_into().unwrap();

        let mut actions = Vec::with_capacity(batch_size);
        for i in 0..batch_size {
            if rng.gen::<f32>() < epsilon {
                let legal: Vec<u8> = (0..NUM_ACTIONS)
                    .filter(|&j| masks[i * NUM_ACTIONS + j] > 0.5)
                    .map(|j| j as u8)
                    .collect();
                let idx = rng.gen_range(0..legal.len());
                actions.push(legal[idx]);
            } else {
                actions.push(greedy[i] as u8);
            }
        }
        actions
    }
}

// ---------------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------------
pub struct BumblebidTchTrainer {
    pub net: BumblebidNet,
    pub vs: nn::VarStore,
    optimizer: nn::Optimizer,
    device: Device,
}

impl BumblebidTchTrainer {
    pub fn new(
        d_model: i64,
        n_layers: usize,
        n_heads: i64,
        lr: f64,
        _weight_decay: f64,
        device: Device,
    ) -> Self {
        let vs = nn::VarStore::new(device);
        let net = BumblebidNet::new(&vs.root(), d_model, n_layers, n_heads);
        let optimizer = nn::AdamW::default().build(&vs, lr).unwrap();

        BumblebidTchTrainer { net, vs, optimizer, device }
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_lr(lr);
    }

    pub fn device(&self) -> Device {
        self.device
    }

    pub fn train_step(
        &mut self,
        obs: &[f32],
        _masks: &[f32],
        actions: &[u8],
        returns: &[f32],
        weights: &[f32],
    ) -> (f32, Vec<f32>) {
        let batch_size = actions.len();

        let q_all = self.net.forward_obs(obs, batch_size);

        let actions_t = Tensor::from_slice(
            &actions.iter().map(|&a| a as i64).collect::<Vec<_>>(),
        )
        .to(self.device)
        .unsqueeze(1);
        let q_taken = q_all.gather(1, &actions_t, false).squeeze_dim(1);

        let returns_t = Tensor::from_slice(returns).to(self.device);
        let weights_t = Tensor::from_slice(weights).to(self.device);

        let td_errors = &q_taken - &returns_t;
        let td_vec: Vec<f32> = td_errors.detach().to(Device::Cpu).try_into().unwrap();

        let loss = (&td_errors.pow_tensor_scalar(2) * &weights_t).mean(Kind::Float);
        self.optimizer.backward_step(&loss);

        let loss_val: f64 = loss.double_value(&[]);
        (loss_val as f32, td_vec)
    }

    pub fn save_checkpoint(&self, path: &str) {
        self.vs.save(path).expect("Failed to save checkpoint");
    }

    pub fn load_checkpoint(&mut self, path: &str) {
        self.vs.load(path).expect("Failed to load checkpoint");
    }

    pub fn snapshot_weights(&self) -> Vec<f32> {
        let mut all = Vec::new();
        for (_, tensor) in self.vs.variables() {
            let v: Vec<f32> = tensor.detach().to(Device::Cpu).try_into().unwrap();
            all.extend(v);
        }
        all
    }
}
