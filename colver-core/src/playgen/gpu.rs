//! GPU (candle CUDA) batched playgen inference — mid-auction deal sampling at
//! large batch (N lanes lockstep, ~6 ms/pas quasi indépendant du batch sur
//! une 4090 pour le modèle v2 10.7M).
//!
//! Même sémantique que `infer.rs::auction_batch_round` (mêmes masques
//! observer-visibles, même machine à états publique), seul le forward change.
//! Les lanes désynchronisées (longueurs d'enchère différentes) sont gérées en
//! « lockstep paddé » : chaque pas, chaque lane appende un token dans son KV
//! cache ; les tokens factices des lanes inactives sont exclus de l'attention
//! future par un masque additif par lane, et la position *logique* (embedding
//! de position) avance indépendamment de la position physique dans le cache.

use candle_core::{Device, Result, Tensor, D};
use rand::Rng;

use crate::card;
use crate::state::{GameState, Phase};

use crate::suit_perm::permute_mask;

use super::infer::{
    bid_token, masked_logp, sample_bid_masked, sample_masked, AuctionLogp, GenState,
    PlaygenModel, PlaygenSampler, Tok, WorldLogp,
};
use super::tokens::{
    MAX_BID_ENTRIES_V2, NUM_BID_ACTIONS, P_ACT0, P_RANK0, SEG_BID, SEG_PLAY, S_NULL,
};

struct GpuBlock {
    attn_norm: Tensor, // [d]
    qkv_w_t: Tensor,   // [d, 3d]
    qkv_b: Tensor,
    out_w_t: Tensor, // [d, d]
    out_b: Tensor,
    ffn_norm: Tensor,
    gate_w_t: Tensor, // [d, dff]
    gate_b: Tensor,
    up_w_t: Tensor,
    up_b: Tensor,
    down_w_t: Tensor, // [dff, d]
    down_b: Tensor,
}

/// Playgen weights resident on a CUDA device, with batched decode.
pub struct GpuPlaygen {
    device: Device,
    d: usize,
    n_heads: usize,
    hd: usize,
    v2: bool,
    max_seq_len: usize,
    primary_emb: Tensor,
    suit_emb: Tensor,
    actor_emb: Tensor,
    seg_emb: Tensor,
    pos_emb: Tensor,
    blocks: Vec<GpuBlock>,
    out_norm: Tensor,
    head_w_t: Tensor, // [d, 32]
    head_b: Tensor,
    bid_head_w_t: Tensor, // [d, 43]
    bid_head_b: Tensor,
}

fn t2(dev: &Device, data: &[f32], rows: usize, cols: usize) -> Result<Tensor> {
    Tensor::from_slice(data, (rows, cols), dev)
}

fn t1(dev: &Device, data: &[f32]) -> Result<Tensor> {
    Tensor::from_slice(data, data.len(), dev)
}

fn t2t(dev: &Device, data: &[f32], rows: usize, cols: usize) -> Result<Tensor> {
    t2(dev, data, rows, cols)?.t()?.contiguous()
}

impl GpuPlaygen {
    pub fn new(m: &PlaygenModel, device: Device) -> Result<Self> {
        assert!(m.v2, "GpuPlaygen auction sampling requires a v2 (COLVPG02) model");
        let d = m.d;
        let blocks = m
            .blocks
            .iter()
            .map(|b| {
                Ok(GpuBlock {
                    attn_norm: t1(&device, &b.attn_norm)?,
                    qkv_w_t: t2t(&device, &b.qkv_w, 3 * d, d)?,
                    qkv_b: t1(&device, &b.qkv_b)?,
                    out_w_t: t2t(&device, &b.out_w, d, d)?,
                    out_b: t1(&device, &b.out_b)?,
                    ffn_norm: t1(&device, &b.ffn_norm)?,
                    gate_w_t: t2t(&device, &b.gate_w, m.dff, d)?,
                    gate_b: t1(&device, &b.gate_b)?,
                    up_w_t: t2t(&device, &b.up_w, m.dff, d)?,
                    up_b: t1(&device, &b.up_b)?,
                    down_w_t: t2t(&device, &b.down_w, d, m.dff)?,
                    down_b: t1(&device, &b.down_b)?,
                })
            })
            .collect::<Result<Vec<_>>>()?;
        Ok(GpuPlaygen {
            d,
            n_heads: m.n_heads,
            hd: d / m.n_heads,
            v2: m.v2,
            max_seq_len: m.max_seq_len,
            primary_emb: t2(&device, &m.primary_emb, m.primary_emb.len() / d, d)?,
            suit_emb: t2(&device, &m.suit_emb, m.suit_emb.len() / d, d)?,
            actor_emb: t2(&device, &m.actor_emb, m.actor_emb.len() / d, d)?,
            seg_emb: t2(&device, &m.seg_emb, m.seg_emb.len() / d, d)?,
            pos_emb: t2(&device, &m.pos_emb, m.max_seq_len, d)?,
            blocks,
            out_norm: t1(&device, &m.out_norm)?,
            head_w_t: t2t(&device, &m.head_w, 32, d)?,
            head_b: t1(&device, &m.head_b)?,
            bid_head_w_t: t2t(&device, &m.bid_head_w, NUM_BID_ACTIONS, d)?,
            bid_head_b: t1(&device, &m.bid_head_b)?,
            device,
        })
    }

    fn rmsnorm(&self, x: &Tensor, w: &Tensor) -> Result<Tensor> {
        let ms = x.sqr()?.mean_keepdim(D::Minus1)?;
        let rms = (ms + 1e-6f64)?.sqrt()?;
        x.broadcast_div(&rms)?.broadcast_mul(w)
    }

    /// Embed one token per lane at per-lane logical positions.
    fn embed(&self, toks: &[Tok], pos_ids: &[u32]) -> Result<Tensor> {
        let b = toks.len();
        let prim: Vec<u32> = toks.iter().map(|t| t.primary as u32).collect();
        let suit: Vec<u32> = toks.iter().map(|t| t.suit as u32).collect();
        let actor: Vec<u32> = toks.iter().map(|t| t.actor as u32).collect();
        let seg: Vec<u32> = toks.iter().map(|t| t.segment as u32).collect();
        let e = self
            .primary_emb
            .index_select(&Tensor::from_slice(&prim, b, &self.device)?, 0)?;
        let e = (e + self
            .suit_emb
            .index_select(&Tensor::from_slice(&suit, b, &self.device)?, 0)?)?;
        let e = (e + self
            .actor_emb
            .index_select(&Tensor::from_slice(&actor, b, &self.device)?, 0)?)?;
        let e = (e + self
            .seg_emb
            .index_select(&Tensor::from_slice(&seg, b, &self.device)?, 0)?)?;
        e + self
            .pos_emb
            .index_select(&Tensor::from_slice(pos_ids, b, &self.device)?, 0)?
    }

    /// One decode step for B lanes.
    /// `caches`: per layer (k, v) as [B, H, T, hd] (None au premier pas).
    /// `mask`: masque additif persistant [B, 1, 1, T] (0 valide / -1e9 factice)
    /// couvrant les T positions déjà en cache ; la colonne du token courant est
    /// toujours considérée valide pour ce pas (pas de ligne toute -inf → pas de
    /// NaN), sa validité future étant gérée par l'appelant.
    fn forward_step(
        &self,
        x: &Tensor,
        caches: &mut [Option<(Tensor, Tensor)>],
        mask: Option<&Tensor>,
    ) -> Result<Tensor> {
        let (b, _) = x.dims2()?;
        let mut x = x.clone();
        let scale = 1.0 / (self.hd as f64).sqrt();
        for (l, blk) in self.blocks.iter().enumerate() {
            let normed = self.rmsnorm(&x, &blk.attn_norm)?;
            let qkv = normed.matmul(&blk.qkv_w_t)?.broadcast_add(&blk.qkv_b)?;
            let shape = (b, self.n_heads, 1, self.hd);
            let q = qkv.narrow(1, 0, self.d)?.reshape(shape)?;
            let k = qkv.narrow(1, self.d, self.d)?.reshape(shape)?;
            let v = qkv.narrow(1, 2 * self.d, self.d)?.reshape(shape)?;
            let (k_cache, v_cache) = match caches[l].take() {
                Some((kc, vc)) => (
                    Tensor::cat(&[&kc, &k], 2)?,
                    Tensor::cat(&[&vc, &v], 2)?,
                ),
                None => (k, v),
            };
            let mut scores = (q.matmul(&k_cache.transpose(2, 3)?.contiguous()?)? * scale)?;
            if let Some(m) = mask {
                // m couvre T-1 positions (avant append) ; la colonne du token
                // courant est valide → pad d'une colonne de zéros.
                let t_now = scores.dim(3)?;
                let t_m = m.dim(3)?;
                if t_m + 1 == t_now {
                    let zero = Tensor::zeros((b, 1, 1, 1), m.dtype(), &self.device)?;
                    let full = Tensor::cat(&[m, &zero], 3)?;
                    scores = scores.broadcast_add(&full)?;
                } else {
                    scores = scores.broadcast_add(m)?;
                }
            }
            let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
            let ctx = probs.matmul(&v_cache)?.reshape((b, self.d))?;
            caches[l] = Some((k_cache, v_cache));
            let attn = ctx.matmul(&blk.out_w_t)?.broadcast_add(&blk.out_b)?;
            x = (x + attn)?;

            let normed = self.rmsnorm(&x, &blk.ffn_norm)?;
            let gate = normed
                .matmul(&blk.gate_w_t)?
                .broadcast_add(&blk.gate_b)?
                .gelu()?;
            let up = normed.matmul(&blk.up_w_t)?.broadcast_add(&blk.up_b)?;
            let ffn = (gate * up)?
                .matmul(&blk.down_w_t)?
                .broadcast_add(&blk.down_b)?;
            x = (x + ffn)?;
        }
        Ok(x)
    }

    fn card_logits(&self, hidden: &Tensor) -> Result<Vec<Vec<f32>>> {
        let normed = self.rmsnorm(hidden, &self.out_norm)?;
        normed
            .matmul(&self.head_w_t)?
            .broadcast_add(&self.head_b)?
            .to_vec2::<f32>()
    }

    fn bid_logits(&self, hidden: &Tensor) -> Result<Vec<Vec<f32>>> {
        let normed = self.rmsnorm(hidden, &self.out_norm)?;
        normed
            .matmul(&self.bid_head_w_t)?
            .broadcast_add(&self.bid_head_b)?
            .to_vec2::<f32>()
    }

    /// Sample deals from a mid-auction position — GPU equivalent of
    /// `PlaygenSampler::generate_deals_from_auction_scored`.
    ///
    /// `prefix`: observed-prefix tokens (`PlaygenSampler::prefix_tokens`).
    /// Retries: up to 4 lockstep rounds on the missing count.
    #[allow(clippy::too_many_arguments)]
    pub fn generate_deals_from_auction_scored(
        &self,
        prefix: &[Tok],
        state: &GameState,
        observer: u8,
        observer_hand: u32,
        bid_entries0: usize,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Result<Vec<([u32; 4], AuctionLogp)>> {
        if !self.v2 || state.phase != Phase::Bidding || n_worlds == 0 {
            return Ok(Vec::new());
        }
        let mut worlds = Vec::with_capacity(n_worlds);
        for _round in 0..4 {
            let missing = n_worlds - worlds.len();
            if missing == 0 {
                break;
            }
            self.auction_round(
                prefix, state, observer, observer_hand, bid_entries0, missing, temperature,
                rng, &mut worlds,
            )?;
        }
        Ok(worlds)
    }

    /// Sample worlds from a mid-play position — GPU equivalent of
    /// `PlaygenSampler::generate_worlds_batch_scored`.
    ///
    /// Simpler than the auction path: every lane replays the same number of
    /// remaining cards, so all lanes stay at the same logical position and no
    /// per-lane position tracking is needed. Dead lanes keep appending dummy
    /// tokens (kept uniform) and are excluded from future attention by the
    /// additive mask; their outputs are never read.
    pub fn generate_worlds_scored(
        &self,
        sampler: &PlaygenSampler,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Result<Vec<([u32; 4], WorldLogp)>> {
        if n_worlds == 0 {
            return Ok(Vec::new());
        }
        let spec = match sampler.play_gen_spec(state) {
            Some(s) => s,
            None => return Ok(Vec::new()),
        };
        let observer = spec.observer;
        let rel = |seat: u8| (seat + 4 - observer) % 4;
        let prefix = sampler.prefix_tokens();

        // ---- Prefill B=1 then replicate the cache across lanes ----
        let mut caches1: Vec<Option<(Tensor, Tensor)>> = vec![None; self.blocks.len()];
        for (pos, tok) in prefix.iter().enumerate() {
            let x = self.embed(std::slice::from_ref(tok), &[pos as u32])?;
            self.forward_step(&x, &mut caches1, None)?;
        }
        let plen = prefix.len();
        let mut caches: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(self.blocks.len());
        for c in &caches1 {
            match c {
                Some((k, v)) => {
                    let k = k.expand((n_worlds, self.n_heads, plen, self.hd))?.contiguous()?;
                    let v = v.expand((n_worlds, self.n_heads, plen, self.hd))?.contiguous()?;
                    caches.push(Some((k, v)));
                }
                None => caches.push(None),
            }
        }
        let mut mask =
            Tensor::zeros((n_worlds, 1, 1, plen), candle_core::DType::F32, &self.device)?;

        let mut gens = vec![spec.base; n_worlds];
        let mut assigned = vec![[0u32; 4]; n_worlds];
        let mut obs_remaining = vec![spec.observer_hand_now; n_worlds];
        let mut alive = vec![true; n_worlds];
        let mut logps = vec![WorldLogp::default(); n_worlds];
        let dummy = Tok { primary: P_ACT0, suit: S_NULL, actor: 0, segment: SEG_PLAY };
        let mut act_toks = vec![dummy; n_worlds];
        let mut card_toks = vec![dummy; n_worlds];
        let neg = -1e9f32;
        let mut pos = plen;

        for step_i in 0..spec.steps {
            for k in 0..n_worlds {
                let r = if alive[k] { rel(gens[k].current) } else { 0 };
                act_toks[k] =
                    Tok { primary: P_ACT0 + r, suit: S_NULL, actor: r, segment: SEG_PLAY };
            }
            let pos_ids = vec![pos as u32; n_worlds];
            let x = self.embed(&act_toks, &pos_ids)?;
            let hidden = self.forward_step(&x, &mut caches, Some(&mask))?;
            let col: Vec<f32> = alive.iter().map(|&a| if a { 0.0 } else { neg }).collect();
            let col_t = Tensor::from_slice(&col, (n_worlds, 1, 1, 1), &self.device)?;
            mask = Tensor::cat(&[&mask, &col_t], 3)?;
            pos += 1;
            let card_lg = self.card_logits(&hidden)?;

            for k in 0..n_worlds {
                if !alive[k] {
                    card_toks[k] = act_toks[k];
                    continue;
                }
                let actor = gens[k].current;
                let r = rel(actor);
                let mask_phys = if actor == observer {
                    gens[k].legal_for_hand(obs_remaining[k], actor)
                } else {
                    let unseen =
                        card::ALL_CARDS & !spec.observer_initial_hand & !gens[k].played;
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
                let mask_canon = permute_mask(mask_phys, &spec.perm);
                let mut logits = [0.0f32; 32];
                logits.copy_from_slice(&card_lg[k]);
                let canon_card = sample_masked(&logits, mask_canon, temperature, rng);
                if actor != observer {
                    let lp = masked_logp(&logits, mask_canon as u64, canon_card);
                    logps[k].sum += lp;
                    logps[k].n += 1;
                    if step_i * 2 < spec.steps {
                        logps[k].half_sum += lp;
                        logps[k].half_n += 1;
                    }
                }
                let phys_card = {
                    let cs = canon_card / 8;
                    let rank = canon_card % 8;
                    let mut ps = 0u8;
                    for s in 0..4u8 {
                        if spec.perm[s as usize] == cs {
                            ps = s;
                            break;
                        }
                    }
                    ps * 8 + rank
                };
                card_toks[k] = Tok {
                    primary: P_RANK0 + canon_card % 8,
                    suit: canon_card / 8,
                    actor: r,
                    segment: SEG_PLAY,
                };
                assigned[k][actor as usize] |= 1u32 << phys_card;
                if actor == observer {
                    obs_remaining[k] &= !(1u32 << phys_card);
                }
                gens[k].step(actor, phys_card);
            }

            let pos_ids = vec![pos as u32; n_worlds];
            let x = self.embed(&card_toks, &pos_ids)?;
            self.forward_step(&x, &mut caches, Some(&mask))?;
            let col: Vec<f32> = alive.iter().map(|&a| if a { 0.0 } else { neg }).collect();
            let col_t = Tensor::from_slice(&col, (n_worlds, 1, 1, 1), &self.device)?;
            mask = Tensor::cat(&[&mask, &col_t], 3)?;
            pos += 1;
        }

        let mut worlds = Vec::with_capacity(n_worlds);
        'lane: for k in 0..n_worlds {
            if !alive[k] {
                continue;
            }
            let mut hands = assigned[k];
            hands[observer as usize] = spec.observer_hand_now;
            for p in 0..4usize {
                if card::card_count(hands[p]) != card::card_count(state.hands[p]) {
                    continue 'lane;
                }
            }
            worlds.push((hands, logps[k]));
        }
        Ok(worlds)
    }

    #[allow(clippy::too_many_arguments)]
    fn auction_round(
        &self,
        prefix: &[Tok],
        state: &GameState,
        observer: u8,
        observer_hand: u32,
        bid_entries0: usize,
        n_lanes: usize,
        temperature: f32,
        rng: &mut impl Rng,
        out: &mut Vec<([u32; 4], AuctionLogp)>,
    ) -> Result<()> {
        const BIDDING: u8 = 0;
        const PLAYING: u8 = 1;
        const DONE: u8 = 2;
        const DEAD: u8 = 3;
        let rel = |seat: u8| (seat + 4 - observer) % 4;

        // ---- Prefill B=1 puis réplication du cache sur n_lanes ----
        let mut caches1: Vec<Option<(Tensor, Tensor)>> = vec![None; self.blocks.len()];
        for (pos, tok) in prefix.iter().enumerate() {
            let x = self.embed(std::slice::from_ref(tok), &[pos as u32])?;
            self.forward_step(&x, &mut caches1, None)?;
        }
        let plen = prefix.len();
        let mut caches: Vec<Option<(Tensor, Tensor)>> = Vec::with_capacity(self.blocks.len());
        for c in &caches1 {
            match c {
                Some((k, v)) => {
                    let k = k
                        .expand((n_lanes, self.n_heads, plen, self.hd))?
                        .contiguous()?;
                    let v = v
                        .expand((n_lanes, self.n_heads, plen, self.hd))?
                        .contiguous()?;
                    caches.push(Some((k, v)));
                }
                None => caches.push(None),
            }
        }
        // Masque persistant : préfixe valide partout.
        let mut mask = Tensor::zeros(
            (n_lanes, 1, 1, plen),
            candle_core::DType::F32,
            &self.device,
        )?;

        // ---- États par lane (identique au chemin CPU) ----
        let mut phases = vec![BIDDING; n_lanes];
        let mut sims = vec![*state; n_lanes];
        let mut bid_entries = vec![bid_entries0; n_lanes];
        let placeholder = GenState {
            contract: state.contract,
            trick_cards: [card::EMPTY; 4],
            trick_lead: 0,
            trick_count: 0,
            current: 0,
            plays_done: 0,
            played: 0,
            remaining: [8; 4],
            voids: [0; 4],
            ceiling: [0; 4],
        };
        let mut gens = vec![placeholder; n_lanes];
        let mut assigned = vec![[0u32; 4]; n_lanes];
        let mut obs_remaining = vec![observer_hand; n_lanes];
        let mut alps = vec![AuctionLogp::default(); n_lanes];
        let mut lens = vec![plen; n_lanes]; // positions logiques
        let dummy = Tok { primary: P_ACT0, suit: S_NULL, actor: 0, segment: SEG_BID };
        let mut act_toks = vec![dummy; n_lanes];
        let mut action_toks = vec![dummy; n_lanes];
        let mut act_active = vec![false; n_lanes];
        let mut action_active = vec![false; n_lanes];
        let neg = -1e9f32;

        // 2 tokens par entrée d'enchère + 2 par carte ; borne large.
        let max_steps = 2 * (MAX_BID_ENTRIES_V2 + 34);
        for _step in 0..max_steps {
            // 1) tokens ACT
            let mut any = false;
            for k in 0..n_lanes {
                act_active[k] = false;
                match phases[k] {
                    BIDDING => {
                        if bid_entries[k] >= MAX_BID_ENTRIES_V2
                            || lens[k] + 2 >= self.max_seq_len
                        {
                            phases[k] = DEAD;
                            continue;
                        }
                        let r = rel(sims[k].current_player());
                        act_toks[k] =
                            Tok { primary: P_ACT0 + r, suit: S_NULL, actor: r, segment: SEG_BID };
                        act_active[k] = true;
                        any = true;
                    }
                    PLAYING => {
                        if lens[k] + 2 >= self.max_seq_len {
                            phases[k] = DEAD;
                            continue;
                        }
                        let r = rel(gens[k].current);
                        act_toks[k] =
                            Tok { primary: P_ACT0 + r, suit: S_NULL, actor: r, segment: SEG_PLAY };
                        act_active[k] = true;
                        any = true;
                    }
                    _ => {}
                }
            }
            if !any {
                break;
            }
            let pos_ids: Vec<u32> = (0..n_lanes)
                .map(|k| if act_active[k] { lens[k] as u32 } else { 0 })
                .collect();
            let x = self.embed(&act_toks, &pos_ids)?;
            let hidden = self.forward_step(&x, &mut caches, Some(&mask))?;
            // Validité future de la colonne appendée
            let col: Vec<f32> = act_active.iter().map(|&a| if a { 0.0 } else { neg }).collect();
            let col_t = Tensor::from_slice(&col, (n_lanes, 1, 1, 1), &self.device)?;
            mask = Tensor::cat(&[&mask, &col_t], 3)?;

            // Les deux têtes en un seul download chacune.
            let card_lg = self.card_logits(&hidden)?;
            let bid_lg = self.bid_logits(&hidden)?;

            // 2) échantillonnage par lane → tokens action
            for k in 0..n_lanes {
                action_active[k] = false;
                if !act_active[k] {
                    continue;
                }
                lens[k] += 1;
                match phases[k] {
                    BIDDING => {
                        let r = rel(sims[k].current_player());
                        let mut logits = [0.0f32; NUM_BID_ACTIONS];
                        logits.copy_from_slice(&bid_lg[k]);
                        let legal = sims[k].legal_actions();
                        let action = sample_bid_masked(&logits, legal, temperature, rng);
                        alps[k].bid_sum += masked_logp(&logits, legal, action);
                        alps[k].bid_n += 1;
                        let (p_tok, phys_suit) = bid_token(action);
                        action_toks[k] = Tok {
                            primary: p_tok,
                            suit: if phys_suit == 255 { S_NULL } else { phys_suit },
                            actor: r,
                            segment: SEG_BID,
                        };
                        sims[k].step(action);
                        bid_entries[k] += 1;
                        match sims[k].phase {
                            Phase::Playing => {
                                let lead = sims[k].current_player();
                                gens[k] = GenState {
                                    contract: sims[k].contract,
                                    trick_cards: [card::EMPTY; 4],
                                    trick_lead: lead,
                                    trick_count: 0,
                                    current: lead,
                                    plays_done: 0,
                                    played: 0,
                                    remaining: [8; 4],
                                    voids: [0; 4],
                                    ceiling: [0; 4],
                                };
                                phases[k] = PLAYING;
                                action_active[k] = true;
                            }
                            Phase::Bidding => action_active[k] = true,
                            Phase::Done => phases[k] = DEAD, // donne blanche
                        }
                    }
                    PLAYING => {
                        let actor = gens[k].current;
                        let r = rel(actor);
                        let m = if actor == observer {
                            gens[k].legal_for_hand(obs_remaining[k], actor)
                        } else {
                            let unseen = card::ALL_CARDS & !observer_hand & !gens[k].played;
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
                        if m == 0 || gens[k].remaining[actor as usize] == 0 {
                            phases[k] = DEAD;
                            continue;
                        }
                        let mut logits = [0.0f32; 32];
                        logits.copy_from_slice(&card_lg[k]);
                        let c = sample_masked(&logits, m, temperature, rng);
                        if actor != observer {
                            alps[k].play_sum += masked_logp(&logits, m as u64, c);
                            alps[k].play_n += 1;
                        }
                        action_toks[k] = Tok {
                            primary: P_RANK0 + c % 8,
                            suit: c / 8,
                            actor: r,
                            segment: SEG_PLAY,
                        };
                        assigned[k][actor as usize] |= 1u32 << c;
                        if actor == observer {
                            obs_remaining[k] &= !(1u32 << c);
                        }
                        gens[k].step(actor, c);
                        if gens[k].plays_done == 32 {
                            phases[k] = DONE; // dernier token CARD jamais lu
                        } else {
                            action_active[k] = true;
                        }
                    }
                    _ => {}
                }
            }

            let any_action = action_active.iter().any(|&a| a);
            if any_action {
                let pos_ids: Vec<u32> = (0..n_lanes)
                    .map(|k| if action_active[k] { lens[k] as u32 } else { 0 })
                    .collect();
                let x = self.embed(&action_toks, &pos_ids)?;
                self.forward_step(&x, &mut caches, Some(&mask))?;
                let col: Vec<f32> =
                    action_active.iter().map(|&a| if a { 0.0 } else { neg }).collect();
                let col_t = Tensor::from_slice(&col, (n_lanes, 1, 1, 1), &self.device)?;
                mask = Tensor::cat(&[&mask, &col_t], 3)?;
                for k in 0..n_lanes {
                    if action_active[k] {
                        lens[k] += 1;
                    }
                }
            }
        }

        'lane: for k in 0..n_lanes {
            if phases[k] != DONE {
                continue;
            }
            let mut hands = assigned[k];
            hands[observer as usize] = observer_hand;
            let mut all = 0u32;
            for p in 0..4usize {
                if card::card_count(hands[p]) != 8 || (all & hands[p]) != 0 {
                    continue 'lane;
                }
                all |= hands[p];
            }
            out.push((hands, alps[k]));
        }
        Ok(())
    }
}
