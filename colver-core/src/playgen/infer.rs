//! Pure-Rust playgen inference: autoregressive world sampling for IS-DD.
//!
//! Loads flat-f32 weights (COLVPG01, exported by `export_playgen`) and runs the
//! causal transformer on CPU with an incremental per-deal KV cache. Sampling a
//! continuation to the end of the deal yields a hidden-hand assignment — a
//! determinized world drawn from the model's posterior.
//!
//! Leak-safety: every mask used during generation is observer-visible
//! (true legal set for the observer's own plays; unseen ∩ hard constraints for
//! hidden players) — identical semantics to training (`tokens.rs`).

use std::io;
use std::sync::Arc;

use crate::card::{self, card_rank, card_suit_u8, Card, HIGHER_TRUMP_MASK, SUIT_MASK,
                  TRUMP_STRENGTH};
use crate::state::{Contract, GameState, Phase};
use crate::suit_perm::permute_mask;
use crate::trick::trick_winner;

use super::tokens::{
    canonical_trump_perm, identity_perm, A_NULL, MAX_BID_ENTRIES_V2, MAX_BID_TOKENS, MAX_SEQ_LEN,
    MAX_SEQ_LEN_V2, NUM_BID_ACTIONS, P_ACT0, P_BOS, P_CAPOT, P_COINCHE, P_OBSPOS0, P_PASS,
    P_RANK0, P_SURCOINCHE, P_VAL0, SEG_BID, SEG_HEADER, SEG_PLAY, S_NULL,
};

const MAGIC: &[u8; 8] = b"COLVPG01";
const MAGIC_V2: &[u8; 8] = b"COLVPG02";
/// Upper bound on sequence length across model versions (buffer sizing).
const SEQ_BUF: usize = MAX_SEQ_LEN_V2;
const NUM_PRIMARY: usize = super::tokens::NUM_PRIMARY;
const NUM_SUIT: usize = super::tokens::NUM_SUIT;
const NUM_ACTOR: usize = super::tokens::NUM_ACTOR;
const NUM_SEG: usize = super::tokens::NUM_SEG;

// ---------------------------------------------------------------------------
// Model weights
// ---------------------------------------------------------------------------

struct Block {
    attn_norm: Vec<f32>,
    qkv_w: Vec<f32>, // [3d, d]
    qkv_b: Vec<f32>,
    out_w: Vec<f32>, // [d, d]
    out_b: Vec<f32>,
    ffn_norm: Vec<f32>,
    gate_w: Vec<f32>, // [dff, d]
    gate_b: Vec<f32>,
    up_w: Vec<f32>, // [dff, d]
    up_b: Vec<f32>,
    down_w: Vec<f32>, // [d, dff]
    down_b: Vec<f32>,
}

pub struct PlaygenModel {
    pub d: usize,
    pub n_layers: usize,
    pub n_heads: usize,
    /// V2: physical suits, auction as prediction target (COLVPG02).
    pub v2: bool,
    /// Sequence-length capacity (98 for v1, 122 for v2).
    pub max_seq_len: usize,
    dff: usize,
    primary_emb: Vec<f32>,
    suit_emb: Vec<f32>,
    actor_emb: Vec<f32>,
    seg_emb: Vec<f32>,
    pos_emb: Vec<f32>,
    blocks: Vec<Block>,
    out_norm: Vec<f32>,
    head_w: Vec<f32>, // [32, d]
    head_b: Vec<f32>,
    bid_head_w: Vec<f32>, // [43, d] (v2 only, else empty)
    bid_head_b: Vec<f32>,
}

struct Reader<'a> {
    data: &'a [f32],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> io::Result<Vec<f32>> {
        if self.pos + n > self.data.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "weight file truncated"));
        }
        let v = self.data[self.pos..self.pos + n].to_vec();
        self.pos += n;
        Ok(v)
    }
}

impl PlaygenModel {
    pub fn load(path: &str) -> io::Result<Self> {
        let bytes = std::fs::read(path)?;
        if bytes.len() < 20 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "weight file too short"));
        }
        let v2 = match &bytes[..8] {
            m if m == MAGIC => false,
            m if m == MAGIC_V2 => true,
            _ => {
                return Err(io::Error::new(io::ErrorKind::InvalidData, "bad COLVPG magic"));
            }
        };
        let max_seq_len = if v2 { MAX_SEQ_LEN_V2 } else { MAX_SEQ_LEN };
        let d = u32::from_le_bytes(bytes[8..12].try_into().unwrap()) as usize;
        let n_layers = u32::from_le_bytes(bytes[12..16].try_into().unwrap()) as usize;
        let n_heads = u32::from_le_bytes(bytes[16..20].try_into().unwrap()) as usize;
        let dff = (2 * 4 * d) / 3;

        let floats: Vec<f32> = bytes[20..]
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
            .collect();
        let mut r = Reader { data: &floats, pos: 0 };

        let primary_emb = r.take(NUM_PRIMARY * d)?;
        let suit_emb = r.take(NUM_SUIT * d)?;
        let actor_emb = r.take(NUM_ACTOR * d)?;
        let seg_emb = r.take(NUM_SEG * d)?;
        let pos_emb = r.take(max_seq_len * d)?;

        let mut blocks = Vec::with_capacity(n_layers);
        for _ in 0..n_layers {
            blocks.push(Block {
                attn_norm: r.take(d)?,
                qkv_w: r.take(3 * d * d)?,
                qkv_b: r.take(3 * d)?,
                out_w: r.take(d * d)?,
                out_b: r.take(d)?,
                ffn_norm: r.take(d)?,
                gate_w: r.take(dff * d)?,
                gate_b: r.take(dff)?,
                up_w: r.take(dff * d)?,
                up_b: r.take(dff)?,
                down_w: r.take(d * dff)?,
                down_b: r.take(d)?,
            });
        }
        let out_norm = r.take(d)?;
        let head_w = r.take(32 * d)?;
        let head_b = r.take(32)?;
        let (bid_head_w, bid_head_b) = if v2 {
            (r.take(NUM_BID_ACTIONS * d)?, r.take(NUM_BID_ACTIONS)?)
        } else {
            (Vec::new(), Vec::new())
        };
        if r.pos != floats.len() {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "trailing weight data"));
        }

        Ok(PlaygenModel {
            d, n_layers, n_heads, v2, max_seq_len, dff,
            primary_emb, suit_emb, actor_emb, seg_emb, pos_emb,
            blocks, out_norm, head_w, head_b, bid_head_w, bid_head_b,
        })
    }
}

// ---------------------------------------------------------------------------
// Forward pass with KV cache
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct KvCache {
    /// Per layer: keys and values, packed [t * d + h * head_dim + i].
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    pub len: usize,
}

impl KvCache {
    pub fn new(model: &PlaygenModel) -> Self {
        KvCache {
            k: vec![Vec::with_capacity(model.max_seq_len * model.d); model.n_layers],
            v: vec![Vec::with_capacity(model.max_seq_len * model.d); model.n_layers],
            len: 0,
        }
    }

    /// Roll back to a previous length (cheap undo of generated tokens).
    pub fn truncate(&mut self, len: usize, d: usize) {
        for l in 0..self.k.len() {
            self.k[l].truncate(len * d);
            self.v[l].truncate(len * d);
        }
        self.len = len;
    }
}

/// Manually-reassociated dot product: 8 independent accumulators break the
/// FP dependency chain so LLVM can vectorize (it won't reassociate float sums
/// on its own without fast-math).
#[inline]
fn dot8(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    // 4 independent 8-wide accumulator groups: keeps 4 vector FMA chains in
    // flight (hides FMA latency), each group maps to one SIMD register.
    let mut acc = [[0.0f32; 8]; 4];
    let chunks = a.len() / 32;
    for c in 0..chunks {
        let ac = &a[c * 32..c * 32 + 32];
        let bc = &b[c * 32..c * 32 + 32];
        for g in 0..4 {
            for j in 0..8 {
                acc[g][j] += ac[g * 8 + j] * bc[g * 8 + j];
            }
        }
    }
    let mut tail = 0.0f32;
    for i in chunks * 32..a.len() {
        tail += a[i] * b[i];
    }
    let mut total = tail;
    for g in 0..4 {
        total += (acc[g][0] + acc[g][1])
            + (acc[g][2] + acc[g][3])
            + (acc[g][4] + acc[g][5])
            + (acc[g][6] + acc[g][7]);
    }
    total
}

/// Batched matmul: x [K, n_in] → out [K, n_out]. Streams each weight row once
/// and reuses it across the K lanes (weight traffic amortized).
fn matmul_batch(
    w: &[f32],
    b: &[f32],
    x: &[f32],
    k_lanes: usize,
    n_in: usize,
    n_out: usize,
    out: &mut [f32],
) {
    for o in 0..n_out {
        let row = &w[o * n_in..(o + 1) * n_in];
        let bo = b[o];
        for k in 0..k_lanes {
            let xk = &x[k * n_in..(k + 1) * n_in];
            out[k * n_out + o] = dot8(row, xk) + bo;
        }
    }
}

/// Batched KV cache: `K` lanes advancing in lockstep. Per layer, keys/values
/// are stored in fixed-stride per-lane blocks `[k][MAX_SEQ_LEN][d]` so each
/// lane's attention scan over time steps stays contiguous.
pub struct KvCacheBatch {
    k: Vec<Vec<f32>>,
    v: Vec<Vec<f32>>,
    n_lanes: usize,
    pub len: usize,
}

impl KvCacheBatch {
    /// Replicate a single-stream prefix cache into `n_lanes` lockstep lanes.
    pub fn from_prefix(model: &PlaygenModel, prefix: &KvCache, n_lanes: usize) -> Self {
        let d = model.d;
        let lane_stride = model.max_seq_len * d;
        let plen = prefix.len;
        let mut k = Vec::with_capacity(model.n_layers);
        let mut v = Vec::with_capacity(model.n_layers);
        for l in 0..model.n_layers {
            let mut kl = vec![0.0f32; n_lanes * lane_stride];
            let mut vl = vec![0.0f32; n_lanes * lane_stride];
            for lane in 0..n_lanes {
                kl[lane * lane_stride..lane * lane_stride + plen * d]
                    .copy_from_slice(&prefix.k[l][..plen * d]);
                vl[lane * lane_stride..lane * lane_stride + plen * d]
                    .copy_from_slice(&prefix.v[l][..plen * d]);
            }
            k.push(kl);
            v.push(vl);
        }
        KvCacheBatch { k, v, n_lanes, len: plen }
    }
}

impl PlaygenModel {
    /// Lockstep forward: process one token per lane (`toks.len() == n_lanes`),
    /// all at sequence position `pos`. Returns final hidden states [K * d].
    pub fn forward_tokens_batch(
        &self,
        cache: &mut KvCacheBatch,
        toks: &[Tok],
        pos: usize,
    ) -> Vec<f32> {
        let d = self.d;
        let kl_n = cache.n_lanes;
        debug_assert_eq!(toks.len(), kl_n);
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(cache.len, pos);
        let hd = d / self.n_heads;
        let lane_stride = self.max_seq_len * d;

        let mut x = vec![0.0f32; kl_n * d];
        for (k, tok) in toks.iter().enumerate() {
            let xk = &mut x[k * d..(k + 1) * d];
            for i in 0..d {
                xk[i] = self.primary_emb[tok.primary as usize * d + i]
                    + self.suit_emb[tok.suit as usize * d + i]
                    + self.actor_emb[tok.actor as usize * d + i]
                    + self.seg_emb[tok.segment as usize * d + i]
                    + self.pos_emb[pos * d + i];
            }
        }

        let mut normed = vec![0.0f32; kl_n * d];
        let mut qkv = vec![0.0f32; kl_n * 3 * d];
        let mut attn_out = vec![0.0f32; kl_n * d];
        let mut proj = vec![0.0f32; kl_n * d];
        let mut gate = vec![0.0f32; kl_n * self.dff];
        let mut up = vec![0.0f32; kl_n * self.dff];
        let mut ffn_out = vec![0.0f32; kl_n * d];

        for (l, blk) in self.blocks.iter().enumerate() {
            // --- Attention ---
            for k in 0..kl_n {
                rmsnorm(&x[k * d..(k + 1) * d], &blk.attn_norm, &mut normed[k * d..(k + 1) * d]);
            }
            matmul_batch(&blk.qkv_w, &blk.qkv_b, &normed, kl_n, d, 3 * d, &mut qkv);

            let t_len = pos + 1;
            let scale = 1.0 / (hd as f32).sqrt();
            for k in 0..kl_n {
                let q = &qkv[k * 3 * d..k * 3 * d + d];
                let k_new = &qkv[k * 3 * d + d..k * 3 * d + 2 * d];
                let v_new = &qkv[k * 3 * d + 2 * d..k * 3 * d + 3 * d];
                let base = k * lane_stride + pos * d;
                cache.k[l][base..base + d].copy_from_slice(k_new);
                cache.v[l][base..base + d].copy_from_slice(v_new);

                for h in 0..self.n_heads {
                    let q_h = &q[h * hd..(h + 1) * hd];
                    let mut scores = [0.0f32; SEQ_BUF];
                    let mut max_s = f32::NEG_INFINITY;
                    for t in 0..t_len {
                        let kt = &cache.k[l]
                            [k * lane_stride + t * d + h * hd..k * lane_stride + t * d + (h + 1) * hd];
                        let mut s = 0.0f32;
                        for i in 0..hd {
                            s += q_h[i] * kt[i];
                        }
                        let s = s * scale;
                        scores[t] = s;
                        if s > max_s {
                            max_s = s;
                        }
                    }
                    let mut denom = 0.0f32;
                    for t in 0..t_len {
                        scores[t] = (scores[t] - max_s).exp();
                        denom += scores[t];
                    }
                    let inv_denom = 1.0 / denom;
                    let out_h = &mut attn_out[k * d + h * hd..k * d + (h + 1) * hd];
                    out_h.fill(0.0);
                    for t in 0..t_len {
                        let w = scores[t] * inv_denom;
                        let vt = &cache.v[l]
                            [k * lane_stride + t * d + h * hd..k * lane_stride + t * d + (h + 1) * hd];
                        for i in 0..hd {
                            out_h[i] += w * vt[i];
                        }
                    }
                }
            }
            matmul_batch(&blk.out_w, &blk.out_b, &attn_out, kl_n, d, d, &mut proj);
            for i in 0..kl_n * d {
                x[i] += proj[i];
            }

            // --- FFN ---
            for k in 0..kl_n {
                rmsnorm(&x[k * d..(k + 1) * d], &blk.ffn_norm, &mut normed[k * d..(k + 1) * d]);
            }
            matmul_batch(&blk.gate_w, &blk.gate_b, &normed, kl_n, d, self.dff, &mut gate);
            matmul_batch(&blk.up_w, &blk.up_b, &normed, kl_n, d, self.dff, &mut up);
            for i in 0..kl_n * self.dff {
                gate[i] = gelu(gate[i]) * up[i];
            }
            matmul_batch(&blk.down_w, &blk.down_b, &gate, kl_n, self.dff, d, &mut ffn_out);
            for i in 0..kl_n * d {
                x[i] += ffn_out[i];
            }
        }

        cache.len += 1;
        x
    }
}

fn matvec(w: &[f32], b: &[f32], x: &[f32], out: &mut [f32]) {
    let n_in = x.len();
    for (o, out_val) in out.iter_mut().enumerate() {
        let row = &w[o * n_in..(o + 1) * n_in];
        *out_val = dot8(row, x) + b[o];
    }
}

fn rmsnorm(x: &[f32], w: &[f32], out: &mut [f32]) {
    let mean_sq: f32 = x.iter().map(|v| v * v).sum::<f32>() / x.len() as f32;
    let inv = 1.0 / (mean_sq + 1e-6).sqrt();
    for i in 0..x.len() {
        out[i] = x[i] * inv * w[i];
    }
}

/// Tanh-approximation GELU (matches candle's `gelu()`).
fn gelu(x: f32) -> f32 {
    0.5 * x * (1.0 + ((0.797_884_56) * (x + 0.044715 * x * x * x)).tanh())
}

/// One token of the 4-channel input.
#[derive(Clone, Copy, Debug)]
pub struct Tok {
    pub primary: u8,
    pub suit: u8,
    pub actor: u8,
    pub segment: u8,
}

impl PlaygenModel {
    /// Process one token through all layers, appending K/V to the cache.
    /// Returns the final-layer hidden state (pre out_norm).
    pub fn forward_token(&self, cache: &mut KvCache, tok: Tok, pos: usize) -> Vec<f32> {
        let d = self.d;
        let hd = d / self.n_heads;
        debug_assert!(pos < self.max_seq_len);
        debug_assert_eq!(cache.len, pos);

        let mut x = vec![0.0f32; d];
        for i in 0..d {
            x[i] = self.primary_emb[tok.primary as usize * d + i]
                + self.suit_emb[tok.suit as usize * d + i]
                + self.actor_emb[tok.actor as usize * d + i]
                + self.seg_emb[tok.segment as usize * d + i]
                + self.pos_emb[pos * d + i];
        }

        let mut normed = vec![0.0f32; d];
        let mut qkv = vec![0.0f32; 3 * d];
        let mut attn_out = vec![0.0f32; d];
        let mut proj = vec![0.0f32; d];
        let mut gate = vec![0.0f32; self.dff];
        let mut up = vec![0.0f32; self.dff];
        let mut ffn_out = vec![0.0f32; d];

        for (l, blk) in self.blocks.iter().enumerate() {
            // --- Attention ---
            rmsnorm(&x, &blk.attn_norm, &mut normed);
            matvec(&blk.qkv_w, &blk.qkv_b, &normed, &mut qkv);
            let (q, kv) = qkv.split_at(d);
            let (k_new, v_new) = kv.split_at(d);
            cache.k[l].extend_from_slice(k_new);
            cache.v[l].extend_from_slice(v_new);
            let t_len = pos + 1;

            let scale = 1.0 / (hd as f32).sqrt();
            for h in 0..self.n_heads {
                let q_h = &q[h * hd..(h + 1) * hd];
                // Scores over all cached positions
                let mut scores = [0.0f32; SEQ_BUF];
                let mut max_s = f32::NEG_INFINITY;
                for t in 0..t_len {
                    let k_t = &cache.k[l][t * d + h * hd..t * d + (h + 1) * hd];
                    let mut s = 0.0f32;
                    for i in 0..hd {
                        s += q_h[i] * k_t[i];
                    }
                    let s = s * scale;
                    scores[t] = s;
                    if s > max_s {
                        max_s = s;
                    }
                }
                let mut denom = 0.0f32;
                for t in 0..t_len {
                    scores[t] = (scores[t] - max_s).exp();
                    denom += scores[t];
                }
                let inv_denom = 1.0 / denom;
                let out_h = &mut attn_out[h * hd..(h + 1) * hd];
                out_h.fill(0.0);
                for t in 0..t_len {
                    let w = scores[t] * inv_denom;
                    let v_t = &cache.v[l][t * d + h * hd..t * d + (h + 1) * hd];
                    for i in 0..hd {
                        out_h[i] += w * v_t[i];
                    }
                }
            }
            matvec(&blk.out_w, &blk.out_b, &attn_out, &mut proj);
            for i in 0..d {
                x[i] += proj[i];
            }

            // --- FFN ---
            rmsnorm(&x, &blk.ffn_norm, &mut normed);
            matvec(&blk.gate_w, &blk.gate_b, &normed, &mut gate);
            matvec(&blk.up_w, &blk.up_b, &normed, &mut up);
            for i in 0..self.dff {
                gate[i] = gelu(gate[i]) * up[i];
            }
            matvec(&blk.down_w, &blk.down_b, &gate, &mut ffn_out);
            for i in 0..d {
                x[i] += ffn_out[i];
            }
        }

        cache.len += 1;
        x
    }

    /// Card logits from a final-layer hidden state.
    pub fn logits(&self, hidden: &[f32]) -> [f32; 32] {
        let mut normed = vec![0.0f32; self.d];
        rmsnorm(hidden, &self.out_norm, &mut normed);
        let mut out = [0.0f32; 32];
        let mut buf = vec![0.0f32; 32];
        matvec(&self.head_w, &self.head_b, &normed, &mut buf);
        out.copy_from_slice(&buf);
        out
    }

    /// Bid logits from a final-layer hidden state (v2 models only).
    pub fn bid_logits(&self, hidden: &[f32]) -> [f32; NUM_BID_ACTIONS] {
        debug_assert!(self.v2, "bid_logits requires a v2 model");
        let mut normed = vec![0.0f32; self.d];
        rmsnorm(hidden, &self.out_norm, &mut normed);
        let mut out = [0.0f32; NUM_BID_ACTIONS];
        let mut buf = vec![0.0f32; NUM_BID_ACTIONS];
        matvec(&self.bid_head_w, &self.bid_head_b, &normed, &mut buf);
        out.copy_from_slice(&buf);
        out
    }
}

// ---------------------------------------------------------------------------
// Light game mechanics for generation (public info only)
// ---------------------------------------------------------------------------

/// Public-information trick context during generation.
#[derive(Clone)]
struct GenState {
    contract: Contract,
    trick_cards: [Card; 4], // by seat, EMPTY if not played
    trick_lead: u8,
    trick_count: u8,
    current: u8,
    plays_done: u8,
    played: u32,
    remaining: [u8; 4],
    /// Suit void bitmask per seat (observed + deduced, physical suits).
    voids: [u8; 4],
    /// Impossible trump ranks per seat.
    ceiling: [u8; 4],
}

/// Best trump rank on a (partial) trick.
fn best_trump_rank_in(trick: &[Card; 4], lead: u8, count: u8, trump: u8) -> Option<u8> {
    let mut best: Option<u8> = None;
    let mut best_strength = 0u8;
    for i in 0..count {
        let seat = (lead + i) % 4;
        let c = trick[seat as usize];
        if c == card::EMPTY {
            continue;
        }
        if card_suit_u8(c) == trump {
            let rank = card_rank(c);
            let s = TRUMP_STRENGTH[rank as usize];
            if best.is_none() || s > best_strength {
                best_strength = s;
                best = Some(rank);
            }
        }
    }
    best
}

/// Seat currently winning a (partial) trick.
fn winner_so_far_in(trick: &[Card; 4], lead: u8, count: u8, contract: &Contract) -> Option<u8> {
    if count == 0 {
        return None;
    }
    let trump = contract.trump;
    let lead_card = trick[lead as usize];
    let lead_suit = card_suit_u8(lead_card);
    let mut best_seat = lead;
    let mut best_is_trump = lead_suit == trump;
    let mut best_key = if best_is_trump {
        TRUMP_STRENGTH[card_rank(lead_card) as usize]
    } else {
        card_rank(lead_card)
    };
    for i in 1..count {
        let seat = (lead + i) % 4;
        let c = trick[seat as usize];
        if c == card::EMPTY {
            continue;
        }
        let suit = card_suit_u8(c);
        if suit == trump {
            let k = TRUMP_STRENGTH[card_rank(c) as usize];
            if !best_is_trump || k > best_key {
                best_is_trump = true;
                best_key = k;
                best_seat = seat;
            }
        } else if !best_is_trump && suit == lead_suit {
            let k = card_rank(c);
            if k > best_key {
                best_key = k;
                best_seat = seat;
            }
        }
    }
    Some(best_seat)
}

fn partner_is_master_in(trick: &[Card; 4], lead: u8, count: u8, contract: &Contract, player: u8) -> bool {
    if count < 2 {
        return false;
    }
    winner_so_far_in(trick, lead, count, contract) == Some(player ^ 2)
}

/// Deduced (void_mask, ceiling_mask) additions implied by `player` playing `c`
/// on the given (partial) trick. Mirrors the fixed `TrumpCeilingTracker` and
/// the engine's void tracking — observer-visible facts only.
fn deduce_play_constraints(
    trick: &[Card; 4],
    lead: u8,
    count: u8,
    contract: &Contract,
    player: u8,
    c: Card,
) -> (u8, u8) {
    let mut voids = 0u8;
    let mut ceiling = 0u8;
    if count == 0 {
        return (voids, ceiling);
    }
    let trump = contract.trump;
    let card_s = card_suit_u8(c);
    let lead_card = trick[lead as usize];
    let lead_suit = card_suit_u8(lead_card);

    if card_s != lead_suit {
        // Didn't follow → void in lead suit.
        voids |= 1 << lead_suit;

        if card_s != trump {
            if !partner_is_master_in(trick, lead, count, contract, player) {
                if let Some(best_rank) = best_trump_rank_in(trick, lead, count, trump) {
                    // "Ne pisse pas": only a ceiling.
                    ceiling |= HIGHER_TRUMP_MASK[best_rank as usize];
                } else {
                    voids |= 1 << trump;
                }
            }
        } else if !partner_is_master_in(trick, lead, count, contract, player) {
            if let Some(best_rank) = best_trump_rank_in(trick, lead, count, trump) {
                if TRUMP_STRENGTH[card_rank(c) as usize] < TRUMP_STRENGTH[best_rank as usize] {
                    ceiling |= HIGHER_TRUMP_MASK[best_rank as usize];
                }
            }
        }
    } else if lead_suit == trump {
        // Following trump lead: must overtrump if possible.
        if let Some(best_rank) = best_trump_rank_in(trick, lead, count, trump) {
            if TRUMP_STRENGTH[card_rank(c) as usize] < TRUMP_STRENGTH[best_rank as usize] {
                ceiling |= HIGHER_TRUMP_MASK[best_rank as usize];
            }
        }
    }
    (voids, ceiling)
}

impl GenState {
    fn best_trump_rank(&self) -> Option<u8> {
        best_trump_rank_in(&self.trick_cards, self.trick_lead, self.trick_count, self.contract.trump)
    }

    fn partner_is_master(&self, player: u8) -> bool {
        partner_is_master_in(&self.trick_cards, self.trick_lead, self.trick_count, &self.contract, player)
    }

    /// Update deduced constraints from a play, then advance the trick.
    fn step(&mut self, player: u8, c: Card) {
        let (dv, dc) = deduce_play_constraints(
            &self.trick_cards,
            self.trick_lead,
            self.trick_count,
            &self.contract,
            player,
            c,
        );
        self.voids[player as usize] |= dv;
        self.ceiling[player as usize] |= dc;

        // Place card, advance
        self.trick_cards[player as usize] = c;
        self.trick_count += 1;
        self.played |= card::card_to_bit(c);
        self.remaining[player as usize] -= 1;
        self.plays_done += 1;

        if self.trick_count == 4 {
            let winner = trick_winner(&self.trick_cards, self.trick_lead, &self.contract);
            self.trick_cards = [card::EMPTY; 4];
            self.trick_lead = winner;
            self.trick_count = 0;
            self.current = winner;
        } else {
            self.current = (self.current + 1) % 4;
        }
    }

    /// Legal plays for a hand fully known to us (the observer's own turns).
    /// Mirrors `play.rs::legal_plays` on public trick context.
    fn legal_for_hand(&self, hand: u32, player: u8) -> u32 {
        if self.trick_count == 0 {
            return hand;
        }
        let trump = self.contract.trump;
        let lead_card = self.trick_cards[self.trick_lead as usize];
        let lead_suit = card_suit_u8(lead_card);
        let in_lead = hand & SUIT_MASK[lead_suit as usize];

        if lead_suit == trump {
            if in_lead != 0 {
                if let Some(br) = self.best_trump_rank() {
                    let higher = in_lead & (HIGHER_TRUMP_MASK[br as usize] as u32) << (trump * 8);
                    if higher != 0 {
                        return higher;
                    }
                }
                return in_lead;
            }
            return hand;
        }

        if in_lead != 0 {
            return in_lead;
        }
        let in_trump = hand & SUIT_MASK[trump as usize];

        if self.partner_is_master(player) {
            let non_trump = hand & !SUIT_MASK[trump as usize];
            if non_trump != 0 {
                return hand;
            }
            // Only trumps left. If partner cut, must overtrump if possible.
            let partner_card = self.trick_cards[(player ^ 2) as usize];
            if partner_card != card::EMPTY && card_suit_u8(partner_card) == trump {
                if let Some(br) = self.best_trump_rank() {
                    let higher =
                        in_trump & (HIGHER_TRUMP_MASK[br as usize] as u32) << (trump * 8);
                    if higher != 0 {
                        return higher;
                    }
                }
                return in_trump;
            }
            return hand;
        }

        if in_trump != 0 {
            if let Some(br) = self.best_trump_rank() {
                let higher = in_trump & (HIGHER_TRUMP_MASK[br as usize] as u32) << (trump * 8);
                if higher != 0 {
                    return higher;
                }
                // "Ne pisse pas": can't overtrump → discard or undertrump.
                let non_trump = hand & !SUIT_MASK[trump as usize];
                if non_trump != 0 {
                    return in_trump | non_trump;
                }
                return in_trump;
            }
            return in_trump;
        }
        hand
    }
}

// ---------------------------------------------------------------------------
// Sampler
// ---------------------------------------------------------------------------

/// Per-deal playgen world sampler. Feed actions via `record_action`, then call
/// `generate_world` at decision points to draw determinized worlds.
pub struct PlaygenSampler {
    model: Arc<PlaygenModel>,
    observer: u8,
    dealer: u8,
    observer_initial_hand: u32,
    /// Physical → canonical suit permutation (set when contract is known).
    perm: Option<[u8; 4]>,
    /// Buffered bid actions (bidder, action) until the contract is known.
    bids: Vec<(u8, u8)>,
    /// Tokens not yet run through the model.
    pending: Vec<Tok>,
    cache: KvCache,
    /// Deduced constraints from the observed prefix (trump voids not tracked
    /// by the engine, trump ceilings).
    prefix_voids: [u8; 4],
    prefix_ceiling: [u8; 4],
    /// V2: number of auction actions recorded so far.
    bid_entries: usize,
    /// V2: auction exceeded MAX_BID_ENTRIES_V2 — playgen disabled for the deal.
    dead: bool,
}

impl PlaygenSampler {
    pub fn new(model: Arc<PlaygenModel>) -> Self {
        let cache = KvCache::new(&model);
        PlaygenSampler {
            model,
            observer: 0,
            dealer: 0,
            observer_initial_hand: 0,
            perm: None,
            bids: Vec::new(),
            pending: Vec::new(),
            cache,
            prefix_voids: [0; 4],
            prefix_ceiling: [0; 4],
            bid_entries: 0,
            dead: false,
        }
    }

    pub fn init_deal(&mut self, state: &GameState, observer: u8) {
        self.observer = observer;
        self.dealer = state.dealer;
        self.observer_initial_hand = state.hands[observer as usize];
        self.perm = None;
        self.bids.clear();
        self.pending.clear();
        self.cache = KvCache::new(&self.model);
        self.prefix_voids = [0; 4];
        self.prefix_ceiling = [0; 4];
        self.bid_entries = 0;
        self.dead = false;

        if self.model.v2 {
            // V2: physical suits — the header needs no trump, emit it now.
            self.perm = Some(identity_perm());
            self.pending.push(Tok {
                primary: P_BOS, suit: S_NULL, actor: A_NULL, segment: SEG_HEADER,
            });
            let obs_pos = (observer + 4 - self.dealer) % 4;
            self.pending.push(Tok {
                primary: P_OBSPOS0 + obs_pos,
                suit: S_NULL,
                actor: A_NULL,
                segment: SEG_HEADER,
            });
            let mut hand_cards: Vec<u8> = (0..32u8)
                .filter(|&c| self.observer_initial_hand & (1 << c) != 0)
                .collect();
            hand_cards.sort_unstable();
            for c in hand_cards {
                self.pending.push(Tok {
                    primary: P_RANK0 + c % 8,
                    suit: c / 8,
                    actor: 0,
                    segment: SEG_HEADER,
                });
            }
        }
    }

    fn rel(&self, seat: u8) -> u8 {
        (seat + 4 - self.observer) % 4
    }

    fn push_card_tok(&mut self, canon_card: u8, rel_actor: u8) {
        self.pending.push(Tok {
            primary: P_RANK0 + canon_card % 8,
            suit: canon_card / 8,
            actor: rel_actor,
            segment: SEG_PLAY,
        });
    }

    /// Flush header + hand + bid tokens once the contract (trump) is known.
    fn flush_prefix(&mut self, trump: u8) {
        let perm = canonical_trump_perm(trump);
        self.perm = Some(perm);

        self.pending.push(Tok { primary: P_BOS, suit: S_NULL, actor: A_NULL, segment: SEG_HEADER });
        let obs_pos = (self.observer + 4 - self.dealer) % 4;
        self.pending.push(Tok {
            primary: P_OBSPOS0 + obs_pos,
            suit: S_NULL,
            actor: A_NULL,
            segment: SEG_HEADER,
        });

        let mut hand_cards: Vec<u8> = (0..32u8)
            .filter(|&c| self.observer_initial_hand & (1 << c) != 0)
            .map(|c| perm[(c / 8) as usize] * 8 + c % 8)
            .collect();
        hand_cards.sort_unstable();
        for c in hand_cards {
            self.pending.push(Tok {
                primary: P_RANK0 + c % 8,
                suit: c / 8,
                actor: 0,
                segment: SEG_HEADER,
            });
        }

        let skip = self.bids.len().saturating_sub(MAX_BID_TOKENS);
        let bids: Vec<(u8, u8)> = self.bids[skip..].to_vec();
        for (bidder, action) in bids {
            let (p_tok, phys_suit) = bid_token(action);
            let s_tok = if phys_suit == 255 { S_NULL } else { perm[phys_suit as usize] };
            self.pending.push(Tok {
                primary: p_tok,
                suit: s_tok,
                actor: self.rel(bidder),
                segment: SEG_BID,
            });
        }
    }

    /// Record an action (any player). `state_before` = state before the action.
    pub fn record_action(&mut self, state_before: &GameState, player: u8, action: u8) {
        match state_before.phase {
            Phase::Bidding => {
                if self.model.v2 {
                    if self.bid_entries >= MAX_BID_ENTRIES_V2 {
                        self.dead = true;
                        return;
                    }
                    self.bid_entries += 1;
                    let rel = self.rel(player);
                    self.pending.push(Tok {
                        primary: P_ACT0 + rel,
                        suit: S_NULL,
                        actor: rel,
                        segment: SEG_BID,
                    });
                    let (p_tok, phys_suit) = bid_token(action);
                    self.pending.push(Tok {
                        primary: p_tok,
                        suit: if phys_suit == 255 { S_NULL } else { phys_suit },
                        actor: rel,
                        segment: SEG_BID,
                    });
                } else {
                    self.bids.push((player, action));
                }
            }
            Phase::Playing => {
                if self.perm.is_none() {
                    self.flush_prefix(state_before.contract.trump);
                }
                let perm = self.perm.unwrap();
                let rel = self.rel(player);
                self.pending.push(Tok {
                    primary: P_ACT0 + rel,
                    suit: S_NULL,
                    actor: rel,
                    segment: SEG_PLAY,
                });
                let canon = perm[(action / 8) as usize] * 8 + action % 8;
                self.push_card_tok(canon, rel);

                // Track deduced constraints from the observed play.
                let (dv, dc) = deduce_play_constraints(
                    &state_before.current_trick,
                    state_before.trick_lead,
                    state_before.trick_count,
                    &state_before.contract,
                    player,
                    action,
                );
                self.prefix_voids[player as usize] |= dv;
                self.prefix_ceiling[player as usize] |= dc;
            }
            Phase::Done => {}
        }
    }

    /// Run pending prefix tokens through the model (incremental).
    fn sync_cache(&mut self) {
        if self.pending.is_empty() {
            return;
        }
        let toks = std::mem::take(&mut self.pending);
        for tok in toks {
            let pos = self.cache.len;
            if pos >= self.model.max_seq_len {
                break; // safety; cannot happen for legal games
            }
            self.model.forward_token(&mut self.cache, tok, pos);
        }
    }

    /// Sample one determinized world from the current position.
    ///
    /// Returns hidden+observer hands consistent with all observer-visible
    /// constraints, or `None` on repeated dead-ends.
    pub fn generate_world(
        &mut self,
        state: &GameState,
        temperature: f32,
        rng: &mut impl rand::Rng,
    ) -> Option<[u32; 4]> {
        if self.perm.is_none() || self.dead {
            // Contract not known yet (shouldn't happen in play phase),
            // or over-long auction disabled playgen for this deal.
            return None;
        }
        self.sync_cache();
        let perm = self.perm.unwrap();
        let observer = self.observer;

        // Current trick cards count as played/known.
        let mut trick_mask = 0u32;
        for i in 0..4 {
            let c = state.current_trick[i];
            if c != card::EMPTY {
                trick_mask |= card::card_to_bit(c);
            }
        }

        let base = GenState {
            contract: state.contract,
            trick_cards: state.current_trick,
            trick_lead: state.trick_lead,
            trick_count: state.trick_count,
            current: state.current_player(),
            plays_done: (card::card_count(state.played_cards | trick_mask)) as u8,
            played: state.played_cards | trick_mask,
            remaining: [
                card::card_count(state.hands[0]) as u8,
                card::card_count(state.hands[1]) as u8,
                card::card_count(state.hands[2]) as u8,
                card::card_count(state.hands[3]) as u8,
            ],
            voids: [
                state.voids[0] | self.prefix_voids[0],
                state.voids[1] | self.prefix_voids[1],
                state.voids[2] | self.prefix_voids[2],
                state.voids[3] | self.prefix_voids[3],
            ],
            ceiling: self.prefix_ceiling,
        };

        let observer_hand_now = state.hands[observer as usize];
        let base_len = self.cache.len;

        'attempt: for _ in 0..4 {
            let mut gen = base.clone();
            let mut cache = self.cache.clone();
            let mut assigned = [0u32; 4];
            let mut obs_remaining = observer_hand_now;

            while gen.plays_done < 32 {
                let actor = gen.current;
                let rel = self.rel(actor);

                // ACT query token
                let pos = cache.len;
                if pos + 1 >= self.model.max_seq_len {
                    break 'attempt;
                }
                let hidden = self.model.forward_token(
                    &mut cache,
                    Tok { primary: P_ACT0 + rel, suit: S_NULL, actor: rel, segment: SEG_PLAY },
                    pos,
                );

                // Observer-visible mask (physical space)
                let mask_phys = if actor == observer {
                    gen.legal_for_hand(obs_remaining, actor)
                } else {
                    let unseen = card::ALL_CARDS & !self.observer_initial_hand & !gen.played;
                    let mut m = 0u32;
                    for c in 0..32u8 {
                        let bit = 1u32 << c;
                        if unseen & bit == 0 {
                            continue;
                        }
                        let suit = c / 8;
                        if gen.voids[actor as usize] & (1 << suit) != 0 {
                            continue;
                        }
                        if suit == gen.contract.trump
                            && gen.ceiling[actor as usize] & (1 << (c % 8)) != 0
                        {
                            continue;
                        }
                        m |= bit;
                    }
                    m
                };
                if mask_phys == 0 || gen.remaining[actor as usize] == 0 {
                    continue 'attempt; // dead end → restart
                }
                let mask_canon = permute_mask(mask_phys, &perm);

                // Sample card from masked softmax
                let logits = self.model.logits(&hidden);
                let canon_card = sample_masked(&logits, mask_canon, temperature, rng);
                // canonical → physical: perm maps phys→canon, invert
                let phys_card = {
                    let cs = canon_card / 8;
                    let rank = canon_card % 8;
                    let mut ps = 0u8;
                    for s in 0..4u8 {
                        if perm[s as usize] == cs {
                            ps = s;
                            break;
                        }
                    }
                    ps * 8 + rank
                };

                // CARD token
                let pos = cache.len;
                self.model.forward_token(
                    &mut cache,
                    Tok {
                        primary: P_RANK0 + canon_card % 8,
                        suit: canon_card / 8,
                        actor: rel,
                        segment: SEG_PLAY,
                    },
                    pos,
                );

                assigned[actor as usize] |= 1u32 << phys_card;
                if actor == observer {
                    obs_remaining &= !(1u32 << phys_card);
                }
                gen.step(actor, phys_card);
            }

            // Success: extract hands at the CURRENT position.
            let mut hands = assigned;
            hands[observer as usize] = observer_hand_now;
            // Sanity: counts must match
            for p in 0..4usize {
                if card::card_count(hands[p]) != card::card_count(state.hands[p]) {
                    continue 'attempt;
                }
            }
            debug_assert_eq!(self.cache.len, base_len);
            return Some(hands);
        }
        None
    }

    /// Sample up to `n_worlds` determinized worlds in lockstep (batched
    /// forward — weights streamed once per token-step for all lanes).
    /// Lanes that hit a dead-end are dropped; the result may be shorter than
    /// requested (possibly empty).
    pub fn generate_worlds_batch(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl rand::Rng,
    ) -> Vec<[u32; 4]> {
        self.generate_worlds_batch_scored(state, n_worlds, temperature, rng)
            .into_iter()
            .map(|(w, _)| w)
            .collect()
    }

    /// Scored variant of [`generate_worlds_batch`]: each world carries the
    /// cumulative log-probability (masked softmax at temperature 1) of the
    /// hidden-actor cards sampled along its continuation.
    pub fn generate_worlds_batch_scored(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl rand::Rng,
    ) -> Vec<([u32; 4], WorldLogp)> {
        if self.perm.is_none() || self.dead || n_worlds == 0 {
            return Vec::new();
        }
        self.sync_cache();
        let perm = self.perm.unwrap();
        let observer = self.observer;

        let mut trick_mask = 0u32;
        for i in 0..4 {
            let c = state.current_trick[i];
            if c != card::EMPTY {
                trick_mask |= card::card_to_bit(c);
            }
        }

        let base = GenState {
            contract: state.contract,
            trick_cards: state.current_trick,
            trick_lead: state.trick_lead,
            trick_count: state.trick_count,
            current: state.current_player(),
            plays_done: (card::card_count(state.played_cards | trick_mask)) as u8,
            played: state.played_cards | trick_mask,
            remaining: [
                card::card_count(state.hands[0]) as u8,
                card::card_count(state.hands[1]) as u8,
                card::card_count(state.hands[2]) as u8,
                card::card_count(state.hands[3]) as u8,
            ],
            voids: [
                state.voids[0] | self.prefix_voids[0],
                state.voids[1] | self.prefix_voids[1],
                state.voids[2] | self.prefix_voids[2],
                state.voids[3] | self.prefix_voids[3],
            ],
            ceiling: self.prefix_ceiling,
        };
        let observer_hand_now = state.hands[observer as usize];
        let steps = 32 - base.plays_done as usize;
        if self.cache.len + 2 * steps > self.model.max_seq_len {
            return Vec::new(); // cannot happen for legal games
        }

        let mut cache = KvCacheBatch::from_prefix(&self.model, &self.cache, n_worlds);
        let mut gens: Vec<GenState> = vec![base; n_worlds];
        let mut assigned = vec![[0u32; 4]; n_worlds];
        let mut obs_remaining = vec![observer_hand_now; n_worlds];
        let mut alive = vec![true; n_worlds];
        let mut logps = vec![WorldLogp::default(); n_worlds];
        let mut act_toks = vec![Tok { primary: P_ACT0, suit: S_NULL, actor: 0, segment: SEG_PLAY }; n_worlds];
        let mut card_toks = act_toks.clone();

        for step_i in 0..steps {
            // ACT query tokens (dead lanes: padded with a valid token, ignored)
            for k in 0..n_worlds {
                let rel = if alive[k] { self.rel(gens[k].current) } else { 0 };
                act_toks[k] =
                    Tok { primary: P_ACT0 + rel, suit: S_NULL, actor: rel, segment: SEG_PLAY };
            }
            let pos = cache.len;
            let hidden = self.model.forward_tokens_batch(&mut cache, &act_toks, pos);

            for k in 0..n_worlds {
                if !alive[k] {
                    card_toks[k] = act_toks[k];
                    continue;
                }
                let actor = gens[k].current;
                let rel = self.rel(actor);
                let mask_phys = if actor == observer {
                    gens[k].legal_for_hand(obs_remaining[k], actor)
                } else {
                    let unseen = card::ALL_CARDS & !self.observer_initial_hand & !gens[k].played;
                    let mut m = 0u32;
                    for c in 0..32u8 {
                        let bit = 1u32 << c;
                        if unseen & bit == 0 {
                            continue;
                        }
                        let suit = c / 8;
                        if gens[k].voids[actor as usize] & (1 << suit) != 0 {
                            continue;
                        }
                        if suit == gens[k].contract.trump
                            && gens[k].ceiling[actor as usize] & (1 << (c % 8)) != 0
                        {
                            continue;
                        }
                        m |= bit;
                    }
                    m
                };
                if mask_phys == 0 || gens[k].remaining[actor as usize] == 0 {
                    alive[k] = false;
                    card_toks[k] = act_toks[k];
                    continue;
                }
                let mask_canon = permute_mask(mask_phys, &perm);
                let logits = self.model.logits(&hidden[k * self.model.d..(k + 1) * self.model.d]);
                let canon_card = sample_masked(&logits, mask_canon, temperature, rng);
                if actor != observer {
                    let lp = masked_logp(&logits, mask_canon as u64, canon_card);
                    logps[k].sum += lp;
                    logps[k].n += 1;
                    if step_i * 2 < steps {
                        logps[k].half_sum += lp;
                        logps[k].half_n += 1;
                    }
                }
                let phys_card = {
                    let cs = canon_card / 8;
                    let rank = canon_card % 8;
                    let mut ps = 0u8;
                    for s in 0..4u8 {
                        if perm[s as usize] == cs {
                            ps = s;
                            break;
                        }
                    }
                    ps * 8 + rank
                };
                card_toks[k] = Tok {
                    primary: P_RANK0 + canon_card % 8,
                    suit: canon_card / 8,
                    actor: rel,
                    segment: SEG_PLAY,
                };
                assigned[k][actor as usize] |= 1u32 << phys_card;
                if actor == observer {
                    obs_remaining[k] &= !(1u32 << phys_card);
                }
                gens[k].step(actor, phys_card);
            }

            let pos = cache.len;
            self.model.forward_tokens_batch(&mut cache, &card_toks, pos);
        }

        let mut worlds = Vec::with_capacity(n_worlds);
        'lane: for k in 0..n_worlds {
            if !alive[k] {
                continue;
            }
            let mut hands = assigned[k];
            hands[observer as usize] = observer_hand_now;
            for p in 0..4usize {
                if card::card_count(hands[p]) != card::card_count(state.hands[p]) {
                    continue 'lane;
                }
            }
            worlds.push((hands, logps[k]));
        }
        worlds
    }
}

/// Cumulative log-probability of a world's sampled play continuation
/// (hidden actors only, masked softmax at temperature 1).
#[derive(Clone, Copy, Debug, Default)]
pub struct WorldLogp {
    /// Sum over the full continuation.
    pub sum: f32,
    pub n: u32,
    /// Sum over the first half of the generation steps (early-pruning signal).
    pub half_sum: f32,
    pub half_n: u32,
}

/// Cumulative log-probability of a mid-auction deal sample: auction
/// completion (all seats) and hidden-actor playout, separately.
#[derive(Clone, Copy, Debug, Default)]
pub struct AuctionLogp {
    pub bid_sum: f32,
    pub bid_n: u32,
    pub play_sum: f32,
    pub play_n: u32,
}

/// log p of `action` under the masked softmax of `logits` at temperature 1.
fn masked_logp(logits: &[f32], mask: u64, action: u8) -> f32 {
    let mut max_l = f32::NEG_INFINITY;
    for c in 0..logits.len() {
        if mask & (1u64 << c) != 0 && logits[c] > max_l {
            max_l = logits[c];
        }
    }
    let mut denom = 0.0f32;
    for c in 0..logits.len() {
        if mask & (1u64 << c) != 0 {
            denom += (logits[c] - max_l).exp();
        }
    }
    logits[action as usize] - max_l - denom.ln()
}

impl PlaygenSampler {
    /// Bid-policy logits for the current player (v2 models only): 43-way
    /// distribution given the observer-visible prefix. The observer must be
    /// the current player (its hand is in the prefix). Cache is left intact.
    pub fn bid_policy(&mut self, state: &GameState) -> Option<[f32; NUM_BID_ACTIONS]> {
        if !self.model.v2 || self.dead || state.phase != Phase::Bidding {
            return None;
        }
        self.sync_cache();
        let rel = self.rel(state.current_player());
        let pos = self.cache.len;
        if pos >= self.model.max_seq_len {
            return None;
        }
        let hidden = self.model.forward_token(
            &mut self.cache,
            Tok { primary: P_ACT0 + rel, suit: S_NULL, actor: rel, segment: SEG_BID },
            pos,
        );
        self.cache.truncate(pos, self.model.d);
        Some(self.model.bid_logits(&hidden))
    }

    /// Sample determinized deals from a mid-auction position (v2 models only).
    ///
    /// Completes the auction with the bid head (masked to the public legal bid
    /// set), then plays the deal out with the card head; the play assignment
    /// reveals the hidden hands. Returns full 8-card hands per seat.
    /// Generated continuations that end in a void deal (4 passes) are retried.
    pub fn generate_deals_from_auction(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl rand::Rng,
    ) -> Vec<[u32; 4]> {
        self.generate_deals_from_auction_scored(state, n_worlds, temperature, rng)
            .into_iter()
            .map(|(w, _)| w)
            .collect()
    }

    /// Scored variant of [`generate_deals_from_auction`]: each deal carries
    /// the cumulative log-probability of its sampled auction completion and
    /// hidden-actor playout.
    pub fn generate_deals_from_auction_scored(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl rand::Rng,
    ) -> Vec<([u32; 4], AuctionLogp)> {
        if !self.model.v2 || self.dead || state.phase != Phase::Bidding || n_worlds == 0 {
            return Vec::new();
        }
        self.sync_cache();
        let observer = self.observer;
        let observer_hand = self.observer_initial_hand;
        let mut worlds = Vec::with_capacity(n_worlds);

        'world: for _ in 0..n_worlds {
            for _attempt in 0..4 {
                let mut cache = self.cache.clone();
                let mut alp = AuctionLogp::default();
                // Public auction state machine: bid legality never reads hands,
                // so a clone of the (dummy-hand) state is safe to step.
                let mut sim = *state;
                let mut bid_entries = self.bid_entries;
                let mut ok = true;

                while sim.phase == Phase::Bidding {
                    if bid_entries >= MAX_BID_ENTRIES_V2
                        || cache.len + 2 >= self.model.max_seq_len
                    {
                        ok = false;
                        break;
                    }
                    let bidder = sim.current_player();
                    let rel = self.rel(bidder);
                    let pos = cache.len;
                    let hidden = self.model.forward_token(
                        &mut cache,
                        Tok { primary: P_ACT0 + rel, suit: S_NULL, actor: rel, segment: SEG_BID },
                        pos,
                    );
                    let logits = self.model.bid_logits(&hidden);
                    let action =
                        sample_bid_masked(&logits, sim.legal_actions(), temperature, rng);
                    alp.bid_sum += masked_logp(&logits, sim.legal_actions(), action);
                    alp.bid_n += 1;
                    let (p_tok, phys_suit) = bid_token(action);
                    let pos = cache.len;
                    self.model.forward_token(
                        &mut cache,
                        Tok {
                            primary: p_tok,
                            suit: if phys_suit == 255 { S_NULL } else { phys_suit },
                            actor: rel,
                            segment: SEG_BID,
                        },
                        pos,
                    );
                    sim.step(action);
                    bid_entries += 1;
                }
                if !ok || sim.phase != Phase::Playing {
                    continue; // over-long or void auction → retry
                }

                // Play out the deal, assigning hidden cards as they are played.
                let mut gen = GenState {
                    contract: sim.contract,
                    trick_cards: [card::EMPTY; 4],
                    trick_lead: sim.current_player(),
                    trick_count: 0,
                    current: sim.current_player(),
                    plays_done: 0,
                    played: 0,
                    remaining: [8; 4],
                    voids: [0; 4],
                    ceiling: [0; 4],
                };
                let mut assigned = [0u32; 4];
                let mut obs_remaining = observer_hand;
                let mut dead_end = false;

                while gen.plays_done < 32 {
                    let actor = gen.current;
                    let rel = self.rel(actor);
                    let pos = cache.len;
                    if pos + 1 >= self.model.max_seq_len {
                        dead_end = true;
                        break;
                    }
                    let hidden = self.model.forward_token(
                        &mut cache,
                        Tok { primary: P_ACT0 + rel, suit: S_NULL, actor: rel, segment: SEG_PLAY },
                        pos,
                    );
                    // Same semantics as `generate_world`: assignment happens at
                    // play time, so `gen.played` already covers every card
                    // assigned to any seat.
                    let mask = if actor == observer {
                        gen.legal_for_hand(obs_remaining, actor)
                    } else {
                        let unseen = card::ALL_CARDS & !observer_hand & !gen.played;
                        let mut m = 0u32;
                        for c in 0..32u8 {
                            let bit = 1u32 << c;
                            if unseen & bit == 0 {
                                continue;
                            }
                            let suit = c / 8;
                            if gen.voids[actor as usize] & (1 << suit) != 0 {
                                continue;
                            }
                            if suit == gen.contract.trump
                                && gen.ceiling[actor as usize] & (1 << (c % 8)) != 0
                            {
                                continue;
                            }
                            m |= bit;
                        }
                        m
                    };
                    if mask == 0 || gen.remaining[actor as usize] == 0 {
                        dead_end = true;
                        break;
                    }
                    let logits = self.model.logits(&hidden);
                    let c = sample_masked(&logits, mask, temperature, rng);
                    if actor != observer {
                        alp.play_sum += masked_logp(&logits, mask as u64, c);
                        alp.play_n += 1;
                    }
                    let pos = cache.len;
                    self.model.forward_token(
                        &mut cache,
                        Tok {
                            primary: P_RANK0 + c % 8,
                            suit: c / 8,
                            actor: rel,
                            segment: SEG_PLAY,
                        },
                        pos,
                    );
                    assigned[actor as usize] |= 1u32 << c;
                    if actor == observer {
                        obs_remaining &= !(1u32 << c);
                    }
                    gen.step(actor, c);
                }
                if dead_end {
                    continue;
                }

                let mut hands = assigned;
                hands[observer as usize] = observer_hand;
                let mut valid = card::card_count(hands[observer as usize]) == 8;
                let mut all = 0u32;
                for p in 0..4usize {
                    valid &= card::card_count(hands[p]) == 8 && (all & hands[p]) == 0;
                    all |= hands[p];
                }
                if valid {
                    worlds.push((hands, alp));
                    continue 'world;
                }
            }
            // 4 failed attempts for this world slot → give up on it.
        }
        worlds
    }
}

/// Masked softmax sampling over the 43 bid actions.
pub fn sample_bid_masked(
    logits: &[f32; NUM_BID_ACTIONS],
    mask: u64,
    temperature: f32,
    rng: &mut impl rand::Rng,
) -> u8 {
    let t = temperature.max(1e-3);
    let mut max_l = f32::NEG_INFINITY;
    for c in 0..NUM_BID_ACTIONS {
        if mask & (1u64 << c) != 0 && logits[c] > max_l {
            max_l = logits[c];
        }
    }
    let mut probs = [0.0f32; NUM_BID_ACTIONS];
    let mut total = 0.0f32;
    for c in 0..NUM_BID_ACTIONS {
        if mask & (1u64 << c) != 0 {
            let p = ((logits[c] - max_l) / t).exp();
            probs[c] = p;
            total += p;
        }
    }
    let mut r = rng.gen::<f32>() * total;
    for c in 0..NUM_BID_ACTIONS {
        if probs[c] > 0.0 {
            r -= probs[c];
            if r <= 0.0 {
                return c as u8;
            }
        }
    }
    (0..NUM_BID_ACTIONS as u8)
        .rev()
        .find(|&c| mask & (1u64 << c) != 0)
        .unwrap_or(0)
}

fn bid_token(action: u8) -> (u8, u8) {
    match action {
        0 => (P_PASS, 255),
        1..=36 => {
            let value_idx = (action - 1) / 4;
            let suit = (action - 1) % 4;
            (P_VAL0 + value_idx, suit)
        }
        37..=40 => (P_CAPOT, action - 37),
        41 => (P_COINCHE, 255),
        _ => (P_SURCOINCHE, 255),
    }
}

pub fn sample_masked(
    logits: &[f32; 32],
    mask: u32,
    temperature: f32,
    rng: &mut impl rand::Rng,
) -> u8 {
    let t = temperature.max(1e-3);
    let mut max_l = f32::NEG_INFINITY;
    for c in 0..32 {
        if mask & (1 << c) != 0 && logits[c] > max_l {
            max_l = logits[c];
        }
    }
    let mut probs = [0.0f32; 32];
    let mut total = 0.0f32;
    for c in 0..32 {
        if mask & (1 << c) != 0 {
            let p = ((logits[c] - max_l) / t).exp();
            probs[c] = p;
            total += p;
        }
    }
    let mut r = rng.gen::<f32>() * total;
    for c in 0..32 {
        if probs[c] > 0.0 {
            r -= probs[c];
            if r <= 0.0 {
                return c as u8;
            }
        }
    }
    // Fallback: last legal
    (0..32).rev().find(|&c| mask & (1 << c) != 0).unwrap_or(0) as u8
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::game_replay::GameReplay;
    use crate::playgen::tokens::tokenize_replay;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    fn final_trump(replay: &GameReplay) -> Option<u8> {
        let mut state = GameState::new(replay.dealer, replay.hands);
        for &a in &replay.actions {
            if state.phase == Phase::Playing {
                return Some(state.contract.trump);
            }
            state.step(a);
        }
        None
    }

    /// Teacher-forcing accuracy of the pure-Rust forward vs training eval.
    /// Run: COLVER_PLAYGEN_BIN=<abs .bin> COLVER_GAMES=<abs games.bin> \
    ///   cargo test -p colver-core --release playgen_forward_accuracy -- --ignored --nocapture
    #[test]
    #[ignore]
    fn playgen_forward_accuracy() {
        let model_path = std::env::var("COLVER_PLAYGEN_BIN").expect("set COLVER_PLAYGEN_BIN");
        let games_path = std::env::var("COLVER_GAMES").expect("set COLVER_GAMES");
        let model = PlaygenModel::load(&model_path).expect("load model");
        let replays = GameReplay::load_all(&games_path).expect("load games");

        let n_games: usize = std::env::var("COLVER_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(500);

        let mut correct = 0usize;
        let mut total = 0usize;
        let mut nll_sum = 0.0f64;
        let t0 = std::time::Instant::now();

        for (gi, replay) in replays.iter().take(n_games).enumerate() {
            let Some(trump) = final_trump(replay) else { continue };
            let observer = (gi % 4) as u8;
            let perm = canonical_trump_perm(trump);
            let Some(s) = tokenize_replay(replay, observer, &perm) else { continue };

            let mut cache = KvCache::new(&model);
            let mut pred_i = 0usize;
            for j in 0..s.primary.len() {
                let tok = Tok {
                    primary: s.primary[j],
                    suit: s.suit[j],
                    actor: s.actor[j],
                    segment: s.segment[j],
                };
                let hidden = model.forward_token(&mut cache, tok, j);
                if pred_i < s.pred_pos.len() && s.pred_pos[pred_i] as usize == j {
                    let logits = model.logits(&hidden);
                    let mask = s.masks[pred_i];
                    let target = s.targets[pred_i];
                    // masked argmax + NLL
                    let mut best = 255u8;
                    let mut best_l = f32::NEG_INFINITY;
                    let mut max_l = f32::NEG_INFINITY;
                    for c in 0..32u8 {
                        if mask & (1 << c) != 0 {
                            if logits[c as usize] > best_l {
                                best_l = logits[c as usize];
                                best = c;
                            }
                            if logits[c as usize] > max_l {
                                max_l = logits[c as usize];
                            }
                        }
                    }
                    let mut denom = 0.0f64;
                    for c in 0..32u8 {
                        if mask & (1 << c) != 0 {
                            denom += ((logits[c as usize] - max_l) as f64).exp();
                        }
                    }
                    let logp = (logits[target as usize] - max_l) as f64 - denom.ln();
                    nll_sum -= logp;
                    if best == target {
                        correct += 1;
                    }
                    total += 1;
                    pred_i += 1;
                }
            }
        }
        println!(
            "teacher-forcing: {} preds, acc {:.4}, nll {:.4}, {:.1}s",
            total,
            correct as f64 / total as f64,
            nll_sum / total as f64,
            t0.elapsed().as_secs_f64()
        );
        assert!(total > 1000);
    }

    /// Teacher-forcing accuracy of a V2 model (COLVPG02): replays real games
    /// through the pure-Rust forward and reports play AND bid head metrics.
    /// Must match the candle trainer's eval on the same checkpoint.
    /// Run: COLVER_PLAYGEN_BIN=... COLVER_GAMES=... cargo test -p colver-core \
    ///   --release playgen_forward_accuracy_v2 -- --ignored --nocapture
    #[test]
    #[ignore]
    fn playgen_forward_accuracy_v2() {
        use crate::playgen::tokens::tokenize_replay_v2;
        let model_path = std::env::var("COLVER_PLAYGEN_BIN").expect("set COLVER_PLAYGEN_BIN");
        let games_path = std::env::var("COLVER_GAMES").expect("set COLVER_GAMES");
        let model = PlaygenModel::load(&model_path).expect("load model");
        assert!(model.v2, "playgen_forward_accuracy_v2 needs a COLVPG02 model");
        let replays = GameReplay::load_all(&games_path).expect("load games");
        let n_games: usize = std::env::var("COLVER_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(200);

        let t0 = std::time::Instant::now();
        let (mut correct, mut total, mut nll_sum) = (0usize, 0usize, 0.0f64);
        let (mut bid_correct, mut bid_total, mut bid_nll_sum) = (0usize, 0usize, 0.0f64);

        for (gi, replay) in replays.iter().take(n_games).enumerate() {
            let observer = (gi % 4) as u8;
            let Some(s) = tokenize_replay_v2(replay, observer, &identity_perm()) else {
                continue;
            };
            let mut cache = KvCache::new(&model);
            let mut pred_i = 0usize;
            let mut bid_i = 0usize;
            for j in 0..s.primary.len() {
                let tok = Tok {
                    primary: s.primary[j],
                    suit: s.suit[j],
                    actor: s.actor[j],
                    segment: s.segment[j],
                };
                let hidden = model.forward_token(&mut cache, tok, j);
                if pred_i < s.pred_pos.len() && s.pred_pos[pred_i] as usize == j {
                    let logits = model.logits(&hidden);
                    let mask = s.masks[pred_i];
                    let target = s.targets[pred_i];
                    let mut best = 255u8;
                    let mut best_l = f32::NEG_INFINITY;
                    let mut max_l = f32::NEG_INFINITY;
                    for c in 0..32u8 {
                        if mask & (1 << c) != 0 {
                            if logits[c as usize] > best_l {
                                best_l = logits[c as usize];
                                best = c;
                            }
                            if logits[c as usize] > max_l {
                                max_l = logits[c as usize];
                            }
                        }
                    }
                    let mut denom = 0.0f64;
                    for c in 0..32u8 {
                        if mask & (1 << c) != 0 {
                            denom += ((logits[c as usize] - max_l) as f64).exp();
                        }
                    }
                    nll_sum -= (logits[target as usize] - max_l) as f64 - denom.ln();
                    if best == target {
                        correct += 1;
                    }
                    total += 1;
                    pred_i += 1;
                }
                if bid_i < s.bid_pred_pos.len() && s.bid_pred_pos[bid_i] as usize == j {
                    let logits = model.bid_logits(&hidden);
                    let mask = s.bid_masks[bid_i];
                    let target = s.bid_targets[bid_i];
                    let mut best = 255u8;
                    let mut best_l = f32::NEG_INFINITY;
                    let mut max_l = f32::NEG_INFINITY;
                    for c in 0..NUM_BID_ACTIONS as u8 {
                        if mask & (1u64 << c) != 0 {
                            if logits[c as usize] > best_l {
                                best_l = logits[c as usize];
                                best = c;
                            }
                            if logits[c as usize] > max_l {
                                max_l = logits[c as usize];
                            }
                        }
                    }
                    let mut denom = 0.0f64;
                    for c in 0..NUM_BID_ACTIONS as u8 {
                        if mask & (1u64 << c) != 0 {
                            denom += ((logits[c as usize] - max_l) as f64).exp();
                        }
                    }
                    bid_nll_sum -= (logits[target as usize] - max_l) as f64 - denom.ln();
                    if best == target {
                        bid_correct += 1;
                    }
                    bid_total += 1;
                    bid_i += 1;
                }
            }
        }
        println!(
            "teacher-forcing v2: play {} preds acc {:.4} nll {:.4} | bid {} preds acc {:.4} nll {:.4} | {:.1}s",
            total,
            correct as f64 / total.max(1) as f64,
            nll_sum / total.max(1) as f64,
            bid_total,
            bid_correct as f64 / bid_total.max(1) as f64,
            bid_nll_sum / bid_total.max(1) as f64,
            t0.elapsed().as_secs_f64()
        );
        assert!(total > 1000 && bid_total > 100);
    }

    /// Mid-auction deal sampling (v2): validity + speed on real games.
    /// Run: COLVER_PLAYGEN_BIN=... COLVER_GAMES=... cargo test -p colver-core \
    ///   --release playgen_auction_deals -- --ignored --nocapture
    #[test]
    #[ignore]
    fn playgen_auction_deals() {
        let model_path = std::env::var("COLVER_PLAYGEN_BIN").expect("set COLVER_PLAYGEN_BIN");
        let games_path = std::env::var("COLVER_GAMES").expect("set COLVER_GAMES");
        let model = Arc::new(PlaygenModel::load(&model_path).expect("load model"));
        assert!(model.v2, "needs a COLVPG02 model");
        let replays = GameReplay::load_all(&games_path).expect("load games");
        let mut rng = StdRng::seed_from_u64(11);
        let n_games = 20usize;
        let per_point = 5usize;

        let mut sampler = PlaygenSampler::new(model);
        let (mut generated, mut missing, mut exact) = (0usize, 0usize, 0usize);
        let mut policy_checked = 0usize;
        let mut gen_time = 0.0f64;

        for (gi, replay) in replays.iter().take(n_games).enumerate() {
            let observer = (gi % 4) as u8;
            let mut state = GameState::new(replay.dealer, replay.hands);
            sampler.init_deal(&state, observer);

            // Record half the auction, then sample deals mid-auction.
            let n_bids = replay
                .actions
                .iter()
                .scan(GameState::new(replay.dealer, replay.hands), |s, &a| {
                    let bidding = s.phase == Phase::Bidding;
                    s.step(a);
                    Some(bidding)
                })
                .take_while(|&b| b)
                .count();
            let stop_at = (n_bids / 2).max(1);

            for (i, &a) in replay.actions.iter().enumerate() {
                if i == stop_at {
                    break;
                }
                sampler.record_action(&state, state.current_player(), a);
                state.step(a);
            }
            assert_eq!(state.phase, Phase::Bidding);

            // Bid policy sanity: finite logits at the query point.
            if let Some(logits) = sampler.bid_policy(&state) {
                assert!(logits.iter().all(|l| l.is_finite()));
                policy_checked += 1;
            }

            let t0 = std::time::Instant::now();
            let worlds = sampler.generate_deals_from_auction(&state, per_point, 1.0, &mut rng);
            gen_time += t0.elapsed().as_secs_f64();
            missing += per_point - worlds.len();
            for hands in &worlds {
                let mut all = 0u32;
                for p in 0..4usize {
                    assert_eq!(card::card_count(hands[p]), 8, "8 cards per seat");
                    assert_eq!(all & hands[p], 0, "no overlap");
                    all |= hands[p];
                }
                assert_eq!(all, card::ALL_CARDS);
                assert_eq!(hands[observer as usize], replay.hands[observer as usize]);
                if *hands == replay.hands {
                    exact += 1;
                }
                generated += 1;
            }
        }
        println!(
            "auction deals: {} ok, {} missing, {} exact-true, {:.1} ms/deal, {} policies checked",
            generated,
            missing,
            exact,
            gen_time * 1000.0 / generated.max(1) as f64,
            policy_checked,
        );
        assert!(generated > n_games * per_point / 2);
        assert_eq!(policy_checked, n_games);
    }

    /// World generation validity + speed on real games.
    /// Run: COLVER_PLAYGEN_BIN=... COLVER_GAMES=... cargo test -p colver-core \
    ///   --release playgen_generate_worlds -- --ignored --nocapture
    #[test]
    #[ignore]
    fn playgen_generate_worlds() {
        let model_path = std::env::var("COLVER_PLAYGEN_BIN").expect("set COLVER_PLAYGEN_BIN");
        let games_path = std::env::var("COLVER_GAMES").expect("set COLVER_GAMES");
        let model = Arc::new(PlaygenModel::load(&model_path).expect("load model"));
        let replays = GameReplay::load_all(&games_path).expect("load games");
        let mut rng = StdRng::seed_from_u64(123);

        let n_games: usize = std::env::var("COLVER_N")
            .ok()
            .and_then(|s| s.parse().ok())
            .unwrap_or(50);
        let worlds_per_point = 5usize;

        let mut sampler = PlaygenSampler::new(model);
        let mut generated = 0usize;
        let mut failed = 0usize;
        let mut exact_world = 0usize;
        let mut gen_time = 0.0f64;
        let mut gen_tokens = 0usize;

        for (gi, replay) in replays.iter().take(n_games).enumerate() {
            let observer = (gi % 4) as u8;
            let mut state = GameState::new(replay.dealer, replay.hands);
            sampler.init_deal(&state, observer);

            // Stop points: after 8, 16, 24 plays
            let mut plays_seen = 0usize;
            for &a in &replay.actions {
                if state.phase == Phase::Playing {
                    if state.current_player() == observer
                        && (plays_seen == 8 || plays_seen == 17 || plays_seen == 26)
                    {
                        // hmm decision points must be at observer's turn; approximate:
                    }
                    plays_seen += 1;
                }
                sampler.record_action(&state, state.current_player(), a);
                state.step(a);
                if state.phase == Phase::Playing && (plays_seen == 8 || plays_seen == 16 || plays_seen == 24)
                {
                    let t0 = std::time::Instant::now();
                    for _ in 0..worlds_per_point {
                        match sampler.generate_world(&state, 1.0, &mut rng) {
                            Some(hands) => {
                                generated += 1;
                                // Validity checks
                                let mut all = 0u32;
                                for p in 0..4usize {
                                    assert_eq!(
                                        card::card_count(hands[p]),
                                        card::card_count(state.hands[p]),
                                        "count mismatch p{}", p
                                    );
                                    assert_eq!(all & hands[p], 0, "overlap");
                                    all |= hands[p];
                                }
                                assert_eq!(
                                    all,
                                    card::ALL_CARDS & !state.played_cards & !{
                                        let mut tm = 0u32;
                                        for i in 0..4 {
                                            let c = state.current_trick[i];
                                            if c != card::EMPTY { tm |= card::card_to_bit(c); }
                                        }
                                        tm
                                    },
                                    "world must cover exactly the unplayed cards"
                                );
                                assert_eq!(hands[observer as usize], state.hands[observer as usize]);
                                if hands == state.hands {
                                    exact_world += 1;
                                }
                                gen_tokens += 2 * (32 - plays_seen);
                            }
                            None => failed += 1,
                        }
                    }
                    gen_time += t0.elapsed().as_secs_f64();
                }
            }
        }
        println!(
            "worlds: {} ok, {} failed ({:.1}%), {} exact-true, {:.2} ms/world, {:.0} tokens/s",
            generated,
            failed,
            failed as f64 / (generated + failed).max(1) as f64 * 100.0,
            exact_world,
            gen_time * 1000.0 / generated.max(1) as f64,
            gen_tokens as f64 / gen_time.max(1e-9)
        );
        assert!(generated > 0);
    }
}

#[cfg(test)]
mod batch_tests {
    use super::*;
    use crate::game_replay::GameReplay;
    use rand::rngs::StdRng;
    use rand::SeedableRng;

    /// Batched forward must match the single-stream forward numerically.
    /// Run: COLVER_PLAYGEN_BIN=... cargo test -p colver-core --release \
    ///   playgen_batch_matches_single -- --ignored --nocapture
    #[test]
    #[ignore]
    fn playgen_batch_matches_single() {
        let model_path = std::env::var("COLVER_PLAYGEN_BIN").expect("set COLVER_PLAYGEN_BIN");
        let model = PlaygenModel::load(&model_path).expect("load model");

        // Arbitrary token sequence (valid vocab ids)
        let toks: Vec<Tok> = (0..40)
            .map(|i| Tok {
                primary: (1 + (i * 7) % 30) as u8,
                suit: (i % 5) as u8,
                actor: (i % 5) as u8,
                segment: (i % 3) as u8,
            })
            .collect();

        // Single-stream reference
        let mut single = KvCache::new(&model);
        let mut ref_hidden = Vec::new();
        for (pos, tok) in toks.iter().enumerate() {
            ref_hidden = model.forward_token(&mut single, *tok, pos);
        }
        let ref_logits = model.logits(&ref_hidden);

        // Prefix of 10 via single stream, then 30 lockstep steps on 4 identical lanes
        let mut prefix = KvCache::new(&model);
        for (pos, tok) in toks.iter().take(10).enumerate() {
            model.forward_token(&mut prefix, *tok, pos);
        }
        let lanes = 4;
        let mut batch = KvCacheBatch::from_prefix(&model, &prefix, lanes);
        let mut hidden = Vec::new();
        for (i, tok) in toks.iter().skip(10).enumerate() {
            let batch_toks = vec![*tok; lanes];
            hidden = model.forward_tokens_batch(&mut batch, &batch_toks, 10 + i);
        }
        for k in 0..lanes {
            let logits_k = model.logits(&hidden[k * model.d..(k + 1) * model.d]);
            for c in 0..32 {
                assert!(
                    (logits_k[c] - ref_logits[c]).abs() < 1e-3,
                    "lane {} logit {} mismatch: {} vs {}",
                    k, c, logits_k[c], ref_logits[c]
                );
            }
        }
        println!("batch forward matches single-stream (4 lanes, 40 tokens)");
    }

    /// Batched world sampling: validity + speed vs sequential.
    /// Run: COLVER_PLAYGEN_BIN=... COLVER_GAMES=... cargo test -p colver-core \
    ///   --release playgen_batch_worlds -- --ignored --nocapture
    #[test]
    #[ignore]
    fn playgen_batch_worlds() {
        let model_path = std::env::var("COLVER_PLAYGEN_BIN").expect("set COLVER_PLAYGEN_BIN");
        let games_path = std::env::var("COLVER_GAMES").expect("set COLVER_GAMES");
        let model = Arc::new(PlaygenModel::load(&model_path).expect("load model"));
        let replays = GameReplay::load_all(&games_path).expect("load games");
        let mut rng = StdRng::seed_from_u64(7);
        let n_games: usize = 30;
        let k = 16usize;

        let mut sampler = PlaygenSampler::new(model);
        let mut batch_worlds = 0usize;
        let mut batch_time = 0.0f64;
        let mut seq_time = 0.0f64;
        let mut seq_worlds = 0usize;

        for (gi, replay) in replays.iter().take(n_games).enumerate() {
            let observer = (gi % 4) as u8;
            let mut state = GameState::new(replay.dealer, replay.hands);
            sampler.init_deal(&state, observer);
            let mut plays = 0usize;
            for &a in &replay.actions {
                let in_play = state.phase == Phase::Playing;
                sampler.record_action(&state, state.current_player(), a);
                state.step(a);
                if in_play {
                    plays += 1;
                    if plays == 12 {
                        let t0 = std::time::Instant::now();
                        let worlds = sampler.generate_worlds_batch(&state, k, 1.0, &mut rng);
                        batch_time += t0.elapsed().as_secs_f64();
                        for hands in &worlds {
                            let mut all = 0u32;
                            for p in 0..4usize {
                                assert_eq!(
                                    card::card_count(hands[p]),
                                    card::card_count(state.hands[p])
                                );
                                assert_eq!(all & hands[p], 0);
                                all |= hands[p];
                            }
                            assert_eq!(hands[observer as usize], state.hands[observer as usize]);
                        }
                        batch_worlds += worlds.len();

                        let t1 = std::time::Instant::now();
                        for _ in 0..k {
                            if sampler.generate_world(&state, 1.0, &mut rng).is_some() {
                                seq_worlds += 1;
                            }
                        }
                        seq_time += t1.elapsed().as_secs_f64();
                    }
                }
            }
        }
        println!(
            "batch: {} worlds in {:.2}s ({:.1} ms/world) | sequential: {} worlds in {:.2}s ({:.1} ms/world) | speedup {:.1}x",
            batch_worlds,
            batch_time,
            batch_time * 1000.0 / batch_worlds.max(1) as f64,
            seq_worlds,
            seq_time,
            seq_time * 1000.0 / seq_worlds.max(1) as f64,
            seq_time / batch_time.max(1e-9) * (batch_worlds as f64 / seq_worlds.max(1) as f64),
        );
        assert!(batch_worlds > n_games * k / 2, "too many dead lanes");
    }
}
