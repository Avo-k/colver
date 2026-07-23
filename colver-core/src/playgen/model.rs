//! Causal transformer for playgen, implemented in candle (feature `dmc_train`).
//!
//! Decoder-only: pre-norm RMSNorm, GeGLU FFN, multi-head attention with a
//! causal mask. Input tokens carry 4 embedding channels (primary, suit, actor,
//! segment) plus learned absolute positions. A 32-way card head is applied at
//! every position; the trainer reads logits at ACT-token positions and applies
//! the observer-visible legal mask before the softmax.

use candle_core::{DType, Device, Result, Tensor, D};
use candle_nn::{embedding, linear, AdamW, Embedding, Linear, Module, Optimizer, ParamsAdamW,
                VarBuilder, VarMap};

use super::tokens::{
    MAX_SEQ_LEN, MAX_SEQ_LEN_V2, NUM_ACTOR, NUM_BID_ACTIONS, NUM_CARD_ACTIONS, NUM_PRIMARY,
    NUM_SEG, NUM_SUIT,
};

/// Architecture config. V1: `bid_head = false`, `max_seq_len = MAX_SEQ_LEN`.
/// V2 (physical suits + auction targets): `bid_head = true`,
/// `max_seq_len = MAX_SEQ_LEN_V2`.
#[derive(Clone, Copy)]
pub struct PlaygenConfig {
    pub d_model: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    pub bid_head: bool,
    pub max_seq_len: usize,
}

impl PlaygenConfig {
    pub fn v1(d_model: usize, n_layers: usize, n_heads: usize) -> Self {
        PlaygenConfig { d_model, n_layers, n_heads, bid_head: false, max_seq_len: MAX_SEQ_LEN }
    }

    pub fn v2(d_model: usize, n_layers: usize, n_heads: usize) -> Self {
        PlaygenConfig { d_model, n_layers, n_heads, bid_head: true, max_seq_len: MAX_SEQ_LEN_V2 }
    }
}

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
        let rms = x
            .sqr()?
            .mean_keepdim(D::Minus1)?
            .broadcast_add(&Tensor::full(self.eps as f32, x.shape(), x.device())?)?
            .sqrt()?;
        x.broadcast_div(&rms)?.broadcast_mul(&self.weight)
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

    /// `attn_bias`: additive [1, 1, L, L] mask (0 = attend, -1e9 = blocked).
    fn forward(&self, x: &Tensor, attn_bias: &Tensor) -> Result<Tensor> {
        let (b, l, _d) = x.dims3()?;
        let h = self.attn_norm.forward(x)?;

        let qkv = self.qkv_proj.forward(&h)?;
        let d = self.n_heads * self.head_dim;
        let q = qkv.narrow(2, 0, d)?;
        let k = qkv.narrow(2, d, d)?;
        let v = qkv.narrow(2, 2 * d, d)?;

        let q = q.reshape((b, l, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let k = k.reshape((b, l, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;
        let v = v.reshape((b, l, self.n_heads, self.head_dim))?.transpose(1, 2)?.contiguous()?;

        let scale = (self.head_dim as f64).sqrt();
        let scores = q
            .matmul(&k.transpose(2, 3)?.contiguous()?)?
            .broadcast_div(&Tensor::full(scale as f32, &[1], x.device())?)?;
        let scores = scores.broadcast_add(attn_bias)?;

        // Manual softmax (candle 0.9 softmax_last_dim lacks a CUDA kernel)
        let max_scores = scores.max_keepdim(D::Minus1)?;
        let exp_scores = scores.broadcast_sub(&max_scores)?.exp()?;
        let sum_exp = exp_scores.sum_keepdim(D::Minus1)?;
        let attn = exp_scores.broadcast_div(&sum_exp)?;

        let h = attn.matmul(&v)?;
        let h = h.transpose(1, 2)?.contiguous()?.reshape((b, l, d))?;
        let h = self.out_proj.forward(&h)?;

        let x = (x + h)?;
        let ffn_out = self.ffn.forward(&self.ffn_norm.forward(&x)?)?;
        &x + ffn_out
    }
}

pub struct PlaygenNet {
    primary_emb: Embedding,
    suit_emb: Embedding,
    actor_emb: Embedding,
    seg_emb: Embedding,
    pos_emb: Embedding,
    layers: Vec<TransformerBlock>,
    out_norm: RMSNorm,
    head: Linear,
    bid_head: Option<Linear>,
}

impl PlaygenNet {
    pub fn new(d_model: usize, n_layers: usize, n_heads: usize, vb: VarBuilder) -> Result<Self> {
        Self::with_config(PlaygenConfig::v1(d_model, n_layers, n_heads), vb)
    }

    pub fn with_config(cfg: PlaygenConfig, vb: VarBuilder) -> Result<Self> {
        let d_model = cfg.d_model;
        let d_ff = (2 * 4 * d_model) / 3;
        let mut layers = Vec::with_capacity(cfg.n_layers);
        for i in 0..cfg.n_layers {
            layers.push(TransformerBlock::new(
                d_model, cfg.n_heads, d_ff,
                vb.pp(format!("layers.{}", i)),
            )?);
        }
        let bid_head = if cfg.bid_head {
            Some(linear(d_model, NUM_BID_ACTIONS, vb.pp("bid_head"))?)
        } else {
            None
        };
        Ok(PlaygenNet {
            primary_emb: embedding(NUM_PRIMARY, d_model, vb.pp("primary_emb"))?,
            suit_emb: embedding(NUM_SUIT, d_model, vb.pp("suit_emb"))?,
            actor_emb: embedding(NUM_ACTOR, d_model, vb.pp("actor_emb"))?,
            seg_emb: embedding(NUM_SEG, d_model, vb.pp("seg_emb"))?,
            pos_emb: embedding(cfg.max_seq_len, d_model, vb.pp("pos_emb"))?,
            layers,
            out_norm: RMSNorm::new(d_model, 1e-6, vb.pp("out_norm"))?,
            head: linear(d_model, NUM_CARD_ACTIONS, vb.pp("head"))?,
            bid_head,
        })
    }

    /// Backbone forward: 4× [B, L] i64 token tensors → [B, L, d] normed hidden.
    pub fn forward_hidden(
        &self,
        primary: &Tensor,
        suit: &Tensor,
        actor: &Tensor,
        segment: &Tensor,
        attn_bias: &Tensor,
    ) -> Result<Tensor> {
        let (_b, l) = primary.dims2()?;
        let device = primary.device();

        let positions = Tensor::arange(0i64, l as i64, device)?;
        let pos = self.pos_emb.forward(&positions)?.unsqueeze(0)?;

        let mut x = self
            .primary_emb
            .forward(primary)?
            .broadcast_add(&self.suit_emb.forward(suit)?)?
            .broadcast_add(&self.actor_emb.forward(actor)?)?
            .broadcast_add(&self.seg_emb.forward(segment)?)?
            .broadcast_add(&pos)?;

        for layer in &self.layers {
            x = layer.forward(&x, attn_bias)?;
        }

        self.out_norm.forward(&x)
    }

    /// Forward: 4× [B, L] i64 token tensors → [B, L, 32] card logits.
    pub fn forward(
        &self,
        primary: &Tensor,
        suit: &Tensor,
        actor: &Tensor,
        segment: &Tensor,
        attn_bias: &Tensor,
    ) -> Result<Tensor> {
        let normed = self.forward_hidden(primary, suit, actor, segment, attn_bias)?;
        self.head.forward(&normed)
    }
}

/// Build the additive causal mask [1, 1, L, L] once per batch length.
pub fn causal_bias(l: usize, device: &Device) -> Result<Tensor> {
    let mut data = vec![0.0f32; l * l];
    for q in 0..l {
        for k in (q + 1)..l {
            data[q * l + k] = -1e9;
        }
    }
    Tensor::from_slice(&data, (1, 1, l, l), device)
}

/// A prepared (padded) batch, ready for the GPU.
pub struct PlaygenBatch {
    pub primary: Vec<i64>,
    pub suit: Vec<i64>,
    pub actor: Vec<i64>,
    pub segment: Vec<i64>,
    pub batch_size: usize,
    pub seq_len: usize,
    /// Flat indices b * seq_len + pos of play ACT tokens.
    pub pred_idx: Vec<u32>,
    pub targets: Vec<u32>,
    /// Legality masks, 1.0 = legal, [n_preds * 32].
    pub mask: Vec<f32>,
    /// Flat indices of bid ACT tokens (v2; empty for v1 batches).
    pub bid_pred_idx: Vec<u32>,
    pub bid_targets: Vec<u32>,
    /// Bid legality masks, 1.0 = legal, [n_bid_preds * 43].
    pub bid_mask: Vec<f32>,
}

pub struct EvalStats {
    pub loss_sum: f64,
    pub correct: usize,
    pub n: usize,
    /// Per-prediction NLL (same order as batch play predictions).
    pub nll: Vec<f32>,
    pub bid_loss_sum: f64,
    pub bid_correct: usize,
    pub bid_n: usize,
    pub bid_nll: Vec<f32>,
}

pub struct PlaygenTrainer {
    pub net: PlaygenNet,
    pub varmap: VarMap,
    optimizer: AdamW,
    pub device: Device,
}

impl PlaygenTrainer {
    pub fn new(
        d_model: usize,
        n_layers: usize,
        n_heads: usize,
        lr: f64,
        weight_decay: f64,
        device: Device,
    ) -> Result<Self> {
        Self::with_config(PlaygenConfig::v1(d_model, n_layers, n_heads), lr, weight_decay, device)
    }

    pub fn with_config(
        cfg: PlaygenConfig,
        lr: f64,
        weight_decay: f64,
        device: Device,
    ) -> Result<Self> {
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &device);
        let net = PlaygenNet::with_config(cfg, vb)?;
        let optimizer = AdamW::new(
            varmap.all_vars(),
            ParamsAdamW { lr, beta1: 0.9, beta2: 0.98, eps: 1e-8, weight_decay },
        )?;
        Ok(PlaygenTrainer { net, varmap, optimizer, device })
    }

    pub fn num_params(&self) -> usize {
        self.varmap
            .all_vars()
            .iter()
            .map(|v| v.as_tensor().elem_count())
            .sum()
    }

    pub fn set_lr(&mut self, lr: f64) {
        self.optimizer.set_learning_rate(lr);
    }

    /// Masked-CE at selected positions. Returns (per-pred NLL [N], argmax hits).
    fn masked_nll(
        &self,
        flat_hidden: &Tensor,
        head: &Linear,
        pred_idx: &[u32],
        targets: &[u32],
        mask: &[f32],
        n_actions: usize,
    ) -> Result<(Tensor, usize)> {
        let device = &self.device;
        let n = targets.len();

        let pred_idx_t = Tensor::from_slice(pred_idx, n, device)?;
        let at_preds = head.forward(&flat_hidden.index_select(&pred_idx_t, 0)?)?; // [N, A]

        // Apply legality mask: illegal → -1e9
        let mask_t = Tensor::from_slice(mask, (n, n_actions), device)?;
        let bias_mask = ((mask_t - 1.0)? * 1e9)?;
        let masked = (at_preds + bias_mask)?;

        // log_softmax (manual)
        let max_l = masked.max_keepdim(D::Minus1)?;
        let shifted = masked.broadcast_sub(&max_l)?;
        let lse = shifted.exp()?.sum_keepdim(D::Minus1)?.log()?;
        let logp = shifted.broadcast_sub(&lse)?; // [N, A]

        let targets_t = Tensor::from_slice(targets, (n, 1), device)?;
        let nll = logp.gather(&targets_t, D::Minus1)?.squeeze(D::Minus1)?.neg()?; // [N]

        let argmax: Vec<u32> = masked.argmax(D::Minus1)?.to_vec1()?;
        let correct = argmax.iter().zip(targets.iter()).filter(|(a, t)| a == t).count();

        Ok((nll, correct))
    }

    /// Forward with both heads. Returns (play_nll, play_correct, bid_nll, bid_correct).
    fn forward_loss(
        &self,
        batch: &PlaygenBatch,
    ) -> Result<(Option<Tensor>, usize, Option<Tensor>, usize)> {
        let device = &self.device;
        let (b, l) = (batch.batch_size, batch.seq_len);

        let primary = Tensor::from_slice(&batch.primary, (b, l), device)?;
        let suit = Tensor::from_slice(&batch.suit, (b, l), device)?;
        let actor = Tensor::from_slice(&batch.actor, (b, l), device)?;
        let segment = Tensor::from_slice(&batch.segment, (b, l), device)?;
        let bias = causal_bias(l, device)?;

        let hidden = self.net.forward_hidden(&primary, &suit, &actor, &segment, &bias)?;
        let d = hidden.dim(D::Minus1)?;
        let flat = hidden.reshape((b * l, d))?;

        let (play_nll, play_correct) = if batch.targets.is_empty() {
            (None, 0)
        } else {
            let (nll, c) = self.masked_nll(
                &flat, &self.net.head,
                &batch.pred_idx, &batch.targets, &batch.mask, NUM_CARD_ACTIONS,
            )?;
            (Some(nll), c)
        };

        let (bid_nll, bid_correct) = if batch.bid_targets.is_empty() {
            (None, 0)
        } else {
            let bid_head = self
                .net
                .bid_head
                .as_ref()
                .expect("batch has bid targets but model has no bid head");
            let (nll, c) = self.masked_nll(
                &flat, bid_head,
                &batch.bid_pred_idx, &batch.bid_targets, &batch.bid_mask, NUM_BID_ACTIONS,
            )?;
            (Some(nll), c)
        };

        Ok((play_nll, play_correct, bid_nll, bid_correct))
    }

    /// One optimizer step; loss = mean over all predictions (play + bid).
    /// Returns (mean loss, play accuracy, n_play_preds, bid accuracy, n_bid_preds).
    pub fn train_step(&mut self, batch: &PlaygenBatch) -> Result<(f32, f32, usize, f32, usize)> {
        let n_play = batch.targets.len();
        let n_bid = batch.bid_targets.len();
        let (play_nll, play_c, bid_nll, bid_c) = self.forward_loss(batch)?;
        let all_nll = match (&play_nll, &bid_nll) {
            (Some(p), Some(b)) => Tensor::cat(&[p, b], 0)?,
            (Some(p), None) => p.clone(),
            (None, Some(b)) => b.clone(),
            (None, None) => panic!("empty batch"),
        };
        let loss = all_nll.mean_all()?;
        self.optimizer.backward_step(&loss)?;
        let loss_val: f32 = loss.detach().to_vec0()?;
        Ok((
            loss_val,
            play_c as f32 / n_play.max(1) as f32,
            n_play,
            bid_c as f32 / n_bid.max(1) as f32,
            n_bid,
        ))
    }

    /// Eval without gradient. Returns per-prediction NLLs for breakdowns.
    pub fn eval_step(&self, batch: &PlaygenBatch) -> Result<EvalStats> {
        let (play_nll, play_c, bid_nll, bid_c) = self.forward_loss(batch)?;
        let nll_v: Vec<f32> = match &play_nll {
            Some(t) => t.detach().to_vec1()?,
            None => Vec::new(),
        };
        let bid_nll_v: Vec<f32> = match &bid_nll {
            Some(t) => t.detach().to_vec1()?,
            None => Vec::new(),
        };
        Ok(EvalStats {
            loss_sum: nll_v.iter().map(|&x| x as f64).sum(),
            correct: play_c,
            n: nll_v.len(),
            nll: nll_v,
            bid_loss_sum: bid_nll_v.iter().map(|&x| x as f64).sum(),
            bid_correct: bid_c,
            bid_n: bid_nll_v.len(),
            bid_nll: bid_nll_v,
        })
    }

    pub fn save_checkpoint(&self, path: &str) -> Result<()> {
        self.varmap.save(path)?;
        Ok(())
    }

    pub fn load_checkpoint(&mut self, path: &str) -> Result<()> {
        self.varmap.load(path)?;
        Ok(())
    }
}
