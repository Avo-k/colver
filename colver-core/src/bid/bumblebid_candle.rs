/// Bumblebid: Encoder-only transformer for bidding, implemented in candle.
///
/// Takes the same 108-dim bid observation as the MLP (BiddingQNet), but
/// internally converts to token sequences and processes with transformer layers.
///
/// This allows plugging into the exact same training infrastructure:
/// same PER, same env, same augmentation, same train_step signature.
///
/// Architecture (matching scripts/bumblebid/model.py):
///   Token sequence: [CLS] [POS_x] [card_1..card_8] [bid_tok bid_tok] ...
///   Each token: primary_emb(id) + suit_emb(suit) + pos_emb(pos)
///   Transformer blocks: pre-norm RMSNorm, GeGLU FFN, multi-head attention
///   Output: CLS → RMSNorm → Linear → 43 Q-values (dueling: V + A)

use candle_core::{DType, Device, IndexOp, Result, Shape, Tensor, D};
use candle_nn::{self, embedding, linear, AdamW, Embedding, Linear, Module, Optimizer, ParamsAdamW,
                VarBuilder, VarMap};

use crate::bid_obs::BID_OBS_DIM;

const NUM_ACTIONS: usize = 43;
const MAX_SEQ_LEN: usize = 34;

// Token IDs (must match model.py)
const P_NONE: i64 = 0;
const P_CLS: i64 = 1;
const P_POS0: i64 = 2;
const P_RANK0: i64 = 6;
const P_VAL0: i64 = 14;
const P_CAPOT: i64 = 23;
const P_PASS: i64 = 24;
const P_COINCHE: i64 = 25;
const P_SURCOINCHE: i64 = 26;
const NUM_PRIMARY: usize = 27;

const S_NULL: i64 = 4;
const NUM_SUITS: usize = 5;

// ---------------------------------------------------------------------------
// Convert 108-dim obs batch to token sequences (CPU)
// ---------------------------------------------------------------------------

/// Convert a batch of 108-dim bid observations to (primary_ids, suit_ids, seq_lens).
/// Returns arrays ready to be turned into tensors.
fn obs_batch_to_tokens(
    obs: &[f32],
    batch_size: usize,
) -> (Vec<i64>, Vec<i64>, Vec<i64>) {
    let mut primary = vec![0i64; batch_size * MAX_SEQ_LEN];
    let mut suits = vec![S_NULL; batch_size * MAX_SEQ_LEN];
    let mut seq_lens = vec![10i64; batch_size]; // minimum: CLS + POS + 8 cards

    for b in 0..batch_size {
        let obs_b = &obs[b * BID_OBS_DIM..(b + 1) * BID_OBS_DIM];
        let p_base = b * MAX_SEQ_LEN;

        // Extract hand: obs[0:32] → 8 cards as (rank, suit)
        let mut cards = Vec::with_capacity(8);
        for bit in 0..32u32 {
            if obs_b[bit as usize] > 0.5 {
                let rank = (bit % 8) as i64;
                let suit = (bit / 8) as i64;
                cards.push((rank, suit));
            }
        }
        cards.sort_by_key(|&(r, s)| s * 8 + r);

        // Extract position: obs[104:108] one-hot
        let pos = (0..4i64).find(|&p| obs_b[104 + p as usize] > 0.5).unwrap_or(0);

        // Build hand tokens
        primary[p_base] = P_CLS;
        suits[p_base] = S_NULL;
        primary[p_base + 1] = P_POS0 + pos;
        suits[p_base + 1] = S_NULL;

        for (j, &(rank, suit)) in cards.iter().enumerate().take(8) {
            primary[p_base + 2 + j] = P_RANK0 + rank;
            suits[p_base + 2 + j] = suit;
        }
        // If hand has < 8 cards (shouldn't happen), pad with zeros
        for j in cards.len()..8 {
            primary[p_base + 2 + j] = P_NONE;
            suits[p_base + 2 + j] = S_NULL;
        }

        // Extract bid history: obs[32:104] = 12 slots × 6 floats
        let mut tok_pos = 10; // after hand tokens
        for slot in 0..12 {
            if tok_pos + 2 > MAX_SEQ_LEN {
                break;
            }
            let base = 32 + slot * 6;
            let type_flag = obs_b[base];
            if type_flag < 0.1 {
                continue; // empty slot
            }

            let (p_tok, s_tok) = if type_flag < 0.3 {
                // PASS
                (P_PASS, S_NULL)
            } else if type_flag < 0.5 {
                // Regular bid
                let val_scaled = obs_b[base + 1];
                let val_enc = (val_scaled * 250.0 / 10.0).round() as i64; // 8-16
                let val_idx = (val_enc - 8).clamp(0, 8);
                let suit = (0..4i64)
                    .find(|&s| obs_b[base + 2 + s as usize] > 0.5)
                    .unwrap_or(0);
                (P_VAL0 + val_idx, suit)
            } else if type_flag < 0.7 {
                // CAPOT
                let suit = (0..4i64)
                    .find(|&s| obs_b[base + 2 + s as usize] > 0.5)
                    .unwrap_or(0);
                (P_CAPOT, suit)
            } else if type_flag < 0.9 {
                (P_COINCHE, S_NULL)
            } else {
                (P_SURCOINCHE, S_NULL)
            };

            // First token of bid pair: the action
            primary[p_base + tok_pos] = p_tok;
            suits[p_base + tok_pos] = s_tok;
            // Second token: P_NONE with suit (or S_NULL for pass/coinche)
            primary[p_base + tok_pos + 1] = P_NONE;
            suits[p_base + tok_pos + 1] = if s_tok < 4 { s_tok } else { S_NULL };
            tok_pos += 2;
        }

        seq_lens[b] = tok_pos as i64;
    }

    (primary, suits, seq_lens)
}

// ---------------------------------------------------------------------------
// Transformer building blocks
// ---------------------------------------------------------------------------

struct RMSNorm {
    weight: Tensor,
    eps: f64,
}

impl RMSNorm {
    fn new(dim: usize, eps: f64, vb: VarBuilder) -> Result<Self> {
        let weight = vb.get_with_hints(dim, "weight", candle_nn::Init::Const(1.0))?;
        Ok(RMSNorm { weight, eps })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let rms = x.sqr()?.mean_keepdim(D::Minus1)?.broadcast_add(
            &Tensor::full(self.eps as f32, x.shape(), x.device())?,
        )?.sqrt()?;
        let normed = x.broadcast_div(&rms)?;
        normed.broadcast_mul(&self.weight)
    }
}

struct GeGLU {
    w_gate: Linear,
    w_up: Linear,
    w_down: Linear,
}

impl GeGLU {
    fn new(d_model: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(GeGLU {
            w_gate: linear(d_model, d_ff, vb.pp("w_gate"))?,
            w_up: linear(d_model, d_ff, vb.pp("w_up"))?,
            w_down: linear(d_ff, d_model, vb.pp("w_down"))?,
        })
    }

    fn forward(&self, x: &Tensor) -> Result<Tensor> {
        let gate = self.w_gate.forward(x)?.gelu()?;
        let up = self.w_up.forward(x)?;
        self.w_down.forward(&(gate * up)?)
    }
}

struct TransformerBlock {
    attn_norm: RMSNorm,
    qkv_proj: Linear,
    out_proj: Linear,
    ffn_norm: RMSNorm,
    ffn: GeGLU,
    n_heads: usize,
    head_dim: usize,
}

impl TransformerBlock {
    fn new(d_model: usize, n_heads: usize, d_ff: usize, vb: VarBuilder) -> Result<Self> {
        Ok(TransformerBlock {
            attn_norm: RMSNorm::new(d_model, 1e-6, vb.pp("attn_norm"))?,
            qkv_proj: linear(d_model, 3 * d_model, vb.pp("qkv_proj"))?,
            out_proj: linear(d_model, d_model, vb.pp("out_proj"))?,
            ffn_norm: RMSNorm::new(d_model, 1e-6, vb.pp("ffn_norm"))?,
            ffn: GeGLU::new(d_model, d_ff, vb.pp("ffn"))?,
            n_heads,
            head_dim: d_model / n_heads,
        })
    }

    fn forward(&self, x: &Tensor, pad_mask: Option<&Tensor>) -> Result<Tensor> {
        let (b, l, _d) = x.dims3()?;
        let h = self.attn_norm.forward(x)?;

        // QKV projection → [B, L, 3*D]
        let qkv = self.qkv_proj.forward(&h)?;
        let d = self.n_heads * self.head_dim;
        let q = qkv.narrow(2, 0, d)?;
        let k = qkv.narrow(2, d, d)?;
        let v = qkv.narrow(2, 2 * d, d)?;

        // Reshape to [B, H, L, head_dim] — contiguous() needed for matmul
        let q = q.reshape((b, l, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, l, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, l, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        // Scaled dot-product attention
        let scale = (self.head_dim as f64).sqrt();
        let scores = q.matmul(&k.transpose(2, 3)?.contiguous()?)?.broadcast_div(
            &Tensor::full(scale as f32, &[1], x.device())?,
        )?;

        // Apply padding mask: [B, L] bool → [B, 1, 1, L]
        let scores = if let Some(mask) = pad_mask {
            let mask_4d = mask
                .unsqueeze(1)?
                .unsqueeze(2)?
                .to_dtype(DType::F32)?;
            let neg_inf = Tensor::full(-1e9f32, scores.shape(), scores.device())?;
            let keep = mask_4d.broadcast_mul(
                &Tensor::full(-1e9f32, &[1], scores.device())?,
            )?;
            scores.broadcast_add(&keep)?
        } else {
            scores
        };

        // Manual softmax (candle 0.9 softmax_last_dim lacks CUDA kernel)
        let max_scores = scores.max_keepdim(D::Minus1)?;
        let exp_scores = scores.broadcast_sub(&max_scores)?.exp()?;
        let sum_exp = exp_scores.sum_keepdim(D::Minus1)?;
        let attn = exp_scores.broadcast_div(&sum_exp)?;
        let h = attn.matmul(&v)?;
        let h = h.transpose(1, 2)?.contiguous()?.reshape((b, l, d))?;
        let h = self.out_proj.forward(&h)?;

        // Residual connections
        let x = (x + h)?;
        let ffn_out = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        &x + ffn_out
    }
}

// ---------------------------------------------------------------------------
// Main model
// ---------------------------------------------------------------------------

pub struct BumblebidNet {
    primary_emb: Embedding,
    suit_emb: Embedding,
    pos_emb: Embedding,
    layers: Vec<TransformerBlock>,
    out_norm: RMSNorm,
    value_head: Linear,
    advantage_head: Linear,
    d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

impl BumblebidNet {
    pub fn new(
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        vb: VarBuilder,
    ) -> Result<Self> {
        let d_ff = (2 * 4 * d_model) / 3; // GeGLU intermediate

        let primary_emb = embedding(NUM_PRIMARY, d_model, vb.pp("primary_emb"))?;
        let suit_emb = embedding(NUM_SUITS, d_model, vb.pp("suit_emb"))?;
        let pos_emb = embedding(MAX_SEQ_LEN, d_model, vb.pp("pos_emb"))?;

        let mut layers = Vec::with_capacity(n_layers);
        for i in 0..n_layers {
            layers.push(TransformerBlock::new(
                d_model, n_heads, d_ff,
                vb.pp(format!("layers.{}", i)),
            )?);
        }

        let out_norm = RMSNorm::new(d_model, 1e-6, vb.pp("out_norm"))?;
        let value_head = linear(d_model, 1, vb.pp("value_head"))?;
        let advantage_head = linear(d_model, NUM_ACTIONS, vb.pp("advantage_head"))?;

        Ok(BumblebidNet {
            primary_emb,
            suit_emb,
            pos_emb,
            layers,
            out_norm,
            value_head,
            advantage_head,
            d_model,
            n_layers,
            n_heads,
        })
    }

    /// Forward on pre-tokenized input.
    fn forward_tokens(
        &self,
        primary_ids: &Tensor,  // [B, L] i64
        suit_ids: &Tensor,     // [B, L] i64
        pad_mask: Option<&Tensor>, // [B, L] f32: 1.0 = padding
    ) -> Result<Tensor> {
        let (_b, l) = primary_ids.dims2()?;
        let device = primary_ids.device();

        let positions = Tensor::arange(0i64, l as i64, device)?;
        let pos_emb = self.pos_emb.forward(&positions)?.unsqueeze(0)?; // [1, L, D]

        let x = self.primary_emb.forward(primary_ids)?
            .broadcast_add(&self.suit_emb.forward(suit_ids)?)?
            .broadcast_add(&pos_emb)?;

        let mut x = x;
        for layer in &self.layers {
            x = layer.forward(&x, pad_mask)?;
        }

        // CLS token (position 0)
        let cls = x.i((.., 0, ..))?;
        let normed = self.out_norm.forward(&cls)?;

        // Dueling: V + A - mean(A)
        let v = self.value_head.forward(&normed)?;
        let a = self.advantage_head.forward(&normed)?;
        let a_mean = a.mean_keepdim(D::Minus1)?;
        a.broadcast_sub(&a_mean)?.broadcast_add(&v)
    }

    /// Forward from 108-dim obs batch (converts to tokens internally).
    pub fn forward_obs(&self, obs: &Tensor) -> Result<Tensor> {
        let batch_size = obs.dim(0)?;
        let device = obs.device();

        // Convert to CPU for tokenization
        let obs_data: Vec<f32> = obs.to_vec2::<f32>()?.into_iter().flatten().collect();
        let (prim, suit, slens) = obs_batch_to_tokens(&obs_data, batch_size);

        // Find max seq_len and bucket
        let max_len = *slens.iter().max().unwrap_or(&10) as usize;
        let seq_len = [10, 16, 24, 34]
            .iter()
            .copied()
            .find(|&b| b >= max_len)
            .unwrap_or(MAX_SEQ_LEN);

        // Truncate to seq_len and move to GPU
        let prim_trunc: Vec<i64> = (0..batch_size)
            .flat_map(|b| prim[b * MAX_SEQ_LEN..b * MAX_SEQ_LEN + seq_len].iter().copied())
            .collect();
        let suit_trunc: Vec<i64> = (0..batch_size)
            .flat_map(|b| suit[b * MAX_SEQ_LEN..b * MAX_SEQ_LEN + seq_len].iter().copied())
            .collect();

        let prim_t = Tensor::from_slice(&prim_trunc, (batch_size, seq_len), device)?;
        let suit_t = Tensor::from_slice(&suit_trunc, (batch_size, seq_len), device)?;

        // Build padding mask: 1.0 where position >= seq_len for that sample
        let mut pad_data = vec![0.0f32; batch_size * seq_len];
        for b in 0..batch_size {
            let sl = slens[b] as usize;
            for p in sl..seq_len {
                pad_data[b * seq_len + p] = 1.0;
            }
        }
        let pad_mask = Tensor::from_slice(&pad_data, (batch_size, seq_len), device)?;

        self.forward_tokens(&prim_t, &suit_t, Some(&pad_mask))
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
        let q = self.forward_obs(obs)?;

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

// ---------------------------------------------------------------------------
// Trainer wrapper (matches BiddingTrainer interface)
// ---------------------------------------------------------------------------

pub struct BumblebidTrainer {
    pub net: BumblebidNet,
    pub varmap: VarMap,
    optimizer: AdamW,
    device: Device,
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
}

impl BumblebidTrainer {
    pub fn new(
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        lr: f64,
        weight_decay: f64,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = BumblebidNet::new(d_model, n_layers, n_heads, vb)?;

        let adamw_params = ParamsAdamW {
            lr,
            beta1: 0.9,
            beta2: 0.98,
            eps: 1e-8,
            weight_decay,
        };
        let optimizer = AdamW::new(varmap.all_vars(), adamw_params)?;

        Ok(BumblebidTrainer {
            net,
            varmap,
            optimizer,
            device,
            d_model,
            n_layers,
            n_heads,
        })
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }

    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Single training step — same interface as BiddingTrainer.
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

        let q_all = self.net.forward_obs(&obs_t)?;

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

    /// Snapshot all weights as a flat f32 vector (for pool diversity).
    pub fn snapshot_weights(&self) -> Result<Vec<f32>> {
        let data = self.varmap.data().lock().unwrap();
        let mut all = Vec::new();
        for (_name, tensor) in data.iter() {
            let v: Vec<f32> = tensor.flatten_all()?.to_vec1()?;
            all.extend(v);
        }
        Ok(all)
    }
}
