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
    PlayGenSpec, PlaygenModel, PlaygenSampler, Tok, WorldLogp,
};
use super::tokens::{
    MAX_BID_ENTRIES_V2, NUM_BID_ACTIONS, P_ACT0, P_RANK0, SEG_BID, SEG_PLAY, S_NULL,
};

/// One position in a [`GpuPlaygen::generate_worlds_multi`] batch.
///
/// Holds borrows rather than owned state so the caller — typically an HTTP
/// handler that has just replayed the deal — hands over what it already has.
pub struct WorldBatchItem<'a> {
    pub sampler: &'a PlaygenSampler,
    pub state: &'a GameState,
    pub n_worlds: usize,
    /// Per item, deliberately: a batch mixes unrelated callers, and letting one
    /// of them set the sampling temperature for the others would silently
    /// change a distribution the caller never asked to change.
    pub temperature: f32,
}

/// Profileur opt-in du décodage (`COLVER_PLAYGEN_PROFILE=1`).
///
/// Les lancements CUDA sont asynchrones, donc chronométrer une phase sans
/// synchroniser mesure le temps de *soumission*, pas celui du calcul — et
/// attribue tout le coût à la première opération qui, elle, synchronise
/// (ici `to_vec2` dans `card_logits`). Chaque phase est donc suivie d'un
/// `device.synchronize()`. Ça fausse légèrement le total à la hausse en
/// supprimant le recouvrement, mais c'est la seule façon de savoir *où* passe
/// le temps plutôt que de le deviner.
/// Ventilation *interne* à `forward_step`, en nanosecondes. Statique parce que
/// `forward_step` est appelé depuis trois chemins qui n'ont pas de `Profile`
/// sous la main ; l'écriture n'a lieu que si le profilage est armé.
pub(crate) static FWD_NS: [std::sync::atomic::AtomicU64; 5] = [
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
    std::sync::atomic::AtomicU64::new(0),
];
const FWD_LABELS: [&str; 5] = ["qkv", "cat cache KV", "attention", "proj sortie", "FFN"];

#[derive(Default)]
struct Profile {
    on: bool,
    embed: f64,
    forward: f64,
    logits: f64,
    sample: f64,
    mask_cat: f64,
    steps: u64,
}

impl Profile {
    fn new() -> Self {
        Profile {
            on: std::env::var("COLVER_PLAYGEN_PROFILE").map(|v| v != "0").unwrap_or(false),
            ..Default::default()
        }
    }

    /// Chronomètre `f`, en synchronisant le device avant de rendre la main.
    fn lap<T>(&mut self, dev: &Device, slot: usize, f: impl FnOnce() -> T) -> T {
        if !self.on {
            return f();
        }
        let t0 = std::time::Instant::now();
        let out = f();
        let _ = dev.synchronize();
        let dt = t0.elapsed().as_secs_f64() * 1e3;
        match slot {
            0 => self.embed += dt,
            1 => self.forward += dt,
            2 => self.logits += dt,
            3 => self.sample += dt,
            _ => self.mask_cat += dt,
        }
        out
    }

    fn report(&self, lanes: usize, prefill_ms: f64) {
        if !self.on {
            return;
        }
        let tot = self.embed + self.forward + self.logits + self.sample + self.mask_cat;
        eprintln!(
            "[playgen] {lanes} lanes, {} pas de décodage | prefill {prefill_ms:.0} ms\n\
             [playgen]   embed {:.0} ms ({:.0}%)  forward {:.0} ms ({:.0}%)  \
             logits+sync {:.0} ms ({:.0}%)  échantillonnage {:.0} ms ({:.0}%)  \
             cat masque {:.0} ms ({:.0}%)\n\
             [playgen]   total décodage {tot:.0} ms, {:.2} ms/pas",
            self.steps,
            self.embed, self.embed / tot * 100.0,
            self.forward, self.forward / tot * 100.0,
            self.logits, self.logits / tot * 100.0,
            self.sample, self.sample / tot * 100.0,
            self.mask_cat, self.mask_cat / tot * 100.0,
            tot / self.steps.max(1) as f64,
        );
        let inner: Vec<f64> = FWD_NS
            .iter()
            .map(|a| a.load(std::sync::atomic::Ordering::Relaxed) as f64 / 1e6)
            .collect();
        let isum: f64 = inner.iter().sum();
        if isum > 0.0 {
            eprint!("[playgen]   dont, dans forward_step :");
            for (lbl, ms) in FWD_LABELS.iter().zip(inner.iter()) {
                eprint!("  {lbl} {ms:.0} ms ({:.0}%)", ms / isum * 100.0);
            }
            eprintln!();
        }
    }
}

/// Cache KV d'une couche, **à capacité fixe**.
///
/// Deux choix, tous deux là pour supprimer une recopie intégrale du cache à
/// chaque pas de décodage — ce que le profilage a chiffré à ~75 % du temps :
///
/// - **capacité fixe** : le cache est alloué une fois à la longueur maximale de
///   la séquence, et chaque pas y écrit son jeton *en place* (`slice_set`) au
///   lieu de réallouer par `Tensor::cat`. Les créneaux pas encore écrits sont
///   masqués à -1e9, dont l'exponentielle vaut exactement 0 en f32 : le softmax
///   est donc inchangé, et attendre sur toute la capacité plutôt que sur un
///   `narrow` garde chaque tenseur contigu — ce qui est précisément ce que
///   `slice_set` exige.
/// - **K déjà transposé** (`[B, H, hd, CAP]`) : l'attention le consommait via
///   `transpose(2, 3).contiguous()`, qui matérialisait une copie transposée de
///   tout le cache à chaque pas.
///
/// Le chemin CPU (`KvCacheBatch`) faisait déjà les deux ; c'est la version GPU
/// qui avait dérivé.
struct KvSlot {
    /// `[B, H, hd, CAP]` — transposé.
    kt: Tensor,
    /// `[B, H, CAP, hd]`.
    v: Tensor,
}

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

    /// One decode step for B lanes against a fixed-capacity KV cache.
    ///
    /// `pos` is the slot this step's k/v are written into; `mask` is the
    /// persistent additive mask over the **whole** capacity (`[B, 1, 1, CAP]`,
    /// 0 valid / -1e9 not). The caller must have marked column `pos` valid for
    /// every lane before calling — a row that is entirely -1e9 would make the
    /// softmax produce NaN.
    fn forward_step(
        &self,
        x: &Tensor,
        caches: &mut [KvSlot],
        pos: usize,
        mask: &Tensor,
    ) -> Result<Tensor> {
        let (b, _) = x.dims2()?;
        let mut x = x.clone();
        let scale = 1.0 / (self.hd as f64).sqrt();
        let prof = std::env::var("COLVER_PLAYGEN_PROFILE").map(|v| v != "0").unwrap_or(false);
        macro_rules! lap {
            ($slot:expr, $e:expr) => {{
                if prof {
                    let t = std::time::Instant::now();
                    let r = $e;
                    let _ = self.device.synchronize();
                    FWD_NS[$slot].fetch_add(
                        t.elapsed().as_nanos() as u64,
                        std::sync::atomic::Ordering::Relaxed,
                    );
                    r
                } else {
                    $e
                }
            }};
        }
        for (l, blk) in self.blocks.iter().enumerate() {
            let normed = self.rmsnorm(&x, &blk.attn_norm)?;
            let qkv = lap!(0, normed.matmul(&blk.qkv_w_t)?.broadcast_add(&blk.qkv_b))?;
            let shape = (b, self.n_heads, 1, self.hd);
            let q = qkv.narrow(1, 0, self.d)?.reshape(shape)?;
            let k = qkv.narrow(1, self.d, self.d)?.reshape(shape)?;
            let v = qkv.narrow(1, 2 * self.d, self.d)?.reshape(shape)?;

            lap!(1, {
                let slot = &caches[l];
                slot.kt.slice_set(&k.transpose(2, 3)?.contiguous()?, 3, pos)?;
                slot.v.slice_set(&v, 2, pos)
            })?;

            let ctx = lap!(2, {
                let slot = &caches[l];
                let scores = (q.matmul(&slot.kt)? * scale)?.broadcast_add(mask)?;
                let probs = candle_nn::ops::softmax(&scores, D::Minus1)?;
                probs.matmul(&slot.v)?.reshape((b, self.d))
            })?;

            let attn = lap!(3, ctx.matmul(&blk.out_w_t)?.broadcast_add(&blk.out_b))?;
            x = (x + attn)?;

            let normed = self.rmsnorm(&x, &blk.ffn_norm)?;
            let ffn = lap!(4, {
                let gate = normed
                    .matmul(&blk.gate_w_t)?
                    .broadcast_add(&blk.gate_b)?
                    .gelu()?;
                let up = normed.matmul(&blk.up_w_t)?.broadcast_add(&blk.up_b)?;
                (gate * up)?
                    .matmul(&blk.down_w_t)?
                    .broadcast_add(&blk.down_b)
            })?;
            x = (x + ffn)?;
        }
        Ok(x)
    }

    /// Allocate an empty fixed-capacity cache for `b` lanes.
    fn new_kv(&self, b: usize, cap: usize) -> Result<Vec<KvSlot>> {
        let dt = candle_core::DType::F32;
        (0..self.blocks.len())
            .map(|_| {
                Ok(KvSlot {
                    kt: Tensor::zeros((b, self.n_heads, self.hd, cap), dt, &self.device)?,
                    v: Tensor::zeros((b, self.n_heads, cap, self.hd), dt, &self.device)?,
                })
            })
            .collect()
    }

    /// Fan a prefill cache out to one lane per world, `idx` giving each output
    /// lane's source lane.
    fn select_kv(&self, src: &[KvSlot], idx: &Tensor) -> Result<Vec<KvSlot>> {
        src.iter()
            .map(|s| {
                Ok(KvSlot {
                    kt: s.kt.index_select(idx, 0)?.contiguous()?,
                    v: s.v.index_select(idx, 0)?.contiguous()?,
                })
            })
            .collect()
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

    /// Sample worlds for **several unrelated positions in one GPU batch**.
    ///
    /// ## Why this exists
    ///
    /// A `/play_worlds` call costs ~220 ms whether it returns 1 world or 256:
    /// the whole cost is the sequential token loop (prefill + 2 steps per card),
    /// and each step is kernel-launch bound, not FLOP bound. So a caller like
    /// IS-DD — which wants 20 worlds per decision, from a different position
    /// every time — runs the GPU at ~8% of its achievable throughput, and no
    /// amount of client concurrency fixes that against a serial server.
    ///
    /// Batching *within* one position is what [`generate_worlds_scored`] already
    /// does. This batches *across* positions, which is the axis that was missing.
    ///
    /// ## How the positions share a batch despite differing
    ///
    /// Two things differ per position: the prefix length, and the number of
    /// cards left to generate. Both are handled the way [`auction_round`]
    /// already handles desynchronized auction lanes — "lockstep paddé":
    ///
    /// - **Prefill** runs `max(prefix_len)` steps at batch K, with each lane's
    ///   prefix **right-aligned** so every lane finishes its prefix on the same
    ///   step. Padding steps emit a dummy token whose KV entry is then excluded
    ///   from all future attention by the additive mask. The *logical* position
    ///   fed to the position embedding is the token's index within its own
    ///   sequence, which is why `embed` takes per-lane `pos_ids`.
    /// - **Decode** runs `max(steps)` rounds over `sum(n_worlds)` lanes. A lane
    ///   that has finished its own position stops being active and is masked out
    ///   exactly like a dead one, but is *kept* in the result.
    ///
    /// Between the two phases the K prefix lanes are fanned out to
    /// `sum(n_worlds)` decode lanes with an `index_select` on the batch axis —
    /// the multi-position analogue of the `expand` the single-position path does.
    ///
    /// ## Equivalence
    ///
    /// With one item this is bit-identical to [`generate_worlds_scored`] for the
    /// same seed: K=1 means no padding, so the prefill is the same B=1 loop, and
    /// the per-lane sampling order is unchanged. `gpu_multi_matches_single`
    /// pins that.
    ///
    /// Returns one vector per item, in input order. An item whose position
    /// cannot generate (no perm, dead sampler, not in play phase) yields an
    /// empty vector rather than an error — same contract as the single path.
    pub fn generate_worlds_multi(
        &self,
        items: &[WorldBatchItem],
        rng: &mut impl Rng,
    ) -> Result<Vec<Vec<([u32; 4], WorldLogp)>>> {
        let mut out: Vec<Vec<([u32; 4], WorldLogp)>> = vec![Vec::new(); items.len()];

        // ---- Keep only the items that can actually generate ----
        struct Active<'a> {
            item: usize,
            spec: PlayGenSpec,
            prefix: Vec<Tok>,
            n: usize,
            state: &'a GameState,
            temperature: f32,
        }
        let mut act: Vec<Active> = Vec::with_capacity(items.len());
        for (i, it) in items.iter().enumerate() {
            if it.n_worlds == 0 || it.state.phase != Phase::Playing {
                continue;
            }
            if let Some(spec) = it.sampler.play_gen_spec(it.state) {
                act.push(Active {
                    item: i,
                    spec,
                    prefix: it.sampler.prefix_tokens(),
                    n: it.n_worlds,
                    state: it.state,
                    temperature: it.temperature,
                });
            }
        }
        if act.is_empty() {
            return Ok(out);
        }

        let k_lanes = act.len();
        let dummy = Tok { primary: P_ACT0, suit: S_NULL, actor: 0, segment: SEG_PLAY };
        let neg = -1e9f32;

        // ══ Phase 1 — batched prefill, prefixes right-aligned ══
        let plen: Vec<usize> = act.iter().map(|a| a.prefix.len()).collect();
        let lmax = *plen.iter().max().expect("act non-empty");
        let pad: Vec<usize> = plen.iter().map(|&l| lmax - l).collect();

        let mut prof = Profile::new();
        let t_prefill = std::time::Instant::now();

        // Capacité du cache : préfixe le plus long + deux jetons par carte
        // restante. Allouée une fois, écrite en place ensuite.
        let steps_max = act.iter().map(|a| a.spec.steps).max().expect("act non-empty");
        let cap = lmax + 2 * steps_max;

        let mut caches = self.new_kv(k_lanes, cap)?;
        // Masque hôte, réuploadé à chaque pas : quelques centaines de Ko, sans
        // commune mesure avec les Go que la réallocation du cache déplaçait.
        let mut mh = vec![neg; k_lanes * cap];
        let mut toks = vec![dummy; k_lanes];
        let mut pos_ids = vec![0u32; k_lanes];

        for t in 0..lmax {
            for j in 0..k_lanes {
                let real = t >= pad[j];
                if real {
                    let idx = t - pad[j];
                    toks[j] = act[j].prefix[idx];
                    pos_ids[j] = idx as u32;
                } else {
                    toks[j] = dummy;
                    pos_ids[j] = 0;
                }
                // La colonne courante est valide pour *toutes* les lanes le
                // temps du pas : une ligne entièrement à -1e9 rendrait NaN.
                mh[j * cap + t] = 0.0;
            }
            let mask = Tensor::from_slice(&mh, (k_lanes, 1, 1, cap), &self.device)?;
            let x = self.embed(&toks, &pos_ids)?;
            self.forward_step(&x, &mut caches, t, &mask)?;
            for j in 0..k_lanes {
                mh[j * cap + t] = if t >= pad[j] { 0.0 } else { neg };
            }
        }

        if prof.on {
            let _ = self.device.synchronize();
        }
        let prefill_ms = t_prefill.elapsed().as_secs_f64() * 1e3;

        // ══ Phase 2 — fan K prefix lanes out to sum(n) decode lanes ══
        let mut lane_of: Vec<u32> = Vec::new();
        for (j, a) in act.iter().enumerate() {
            lane_of.extend(std::iter::repeat(j as u32).take(a.n));
        }
        let m_lanes = lane_of.len();
        let sel = Tensor::from_slice(&lane_of, m_lanes, &self.device)?;
        let mut caches = self.select_kv(&caches, &sel)?;
        let mut mh_dec = vec![neg; m_lanes * cap];
        for (m, &j) in lane_of.iter().enumerate() {
            let src = j as usize * cap;
            mh_dec[m * cap..m * cap + cap].copy_from_slice(&mh[src..src + cap]);
        }

        // ══ Phase 3 — decode ══
        let mut gens: Vec<GenState> = Vec::with_capacity(m_lanes);
        let mut obs_remaining: Vec<u32> = Vec::with_capacity(m_lanes);
        let mut pos: Vec<u32> = Vec::with_capacity(m_lanes);
        for &j in &lane_of {
            let a = &act[j as usize];
            gens.push(a.spec.base.clone());
            obs_remaining.push(a.spec.observer_hand_now);
            pos.push(plen[j as usize] as u32);
        }
        let mut assigned = vec![[0u32; 4]; m_lanes];
        let mut alive = vec![true; m_lanes];
        let mut logps = vec![WorldLogp::default(); m_lanes];
        let mut act_toks = vec![dummy; m_lanes];
        let mut card_toks = vec![dummy; m_lanes];
        let mut mpos = vec![0u32; m_lanes];
        let mut mcol = vec![0f32; m_lanes];

        // A lane is *active* while it is alive and has not yet reached its own
        // position's step count. Finished lanes are masked out like dead ones
        // but are still harvested at the end.
        let active_at = |step_i: usize, m: usize, alive: &[bool], lane_of: &[u32], act: &[Active]| {
            alive[m] && step_i < act[lane_of[m] as usize].spec.steps
        };

        for step_i in 0..steps_max {
            // --- ACT query token ---
            for m in 0..m_lanes {
                let on = active_at(step_i, m, &alive, &lane_of, &act);
                if on {
                    let a = &act[lane_of[m] as usize];
                    let r = (gens[m].current + 4 - a.spec.observer) % 4;
                    act_toks[m] =
                        Tok { primary: P_ACT0 + r, suit: S_NULL, actor: r, segment: SEG_PLAY };
                    mpos[m] = pos[m];
                } else {
                    act_toks[m] = dummy;
                    mpos[m] = 0;
                }
                mcol[m] = if on { 0.0 } else { neg };
            }
            prof.steps += 1;
            let t_slot = lmax + 2 * step_i;
            for m in 0..m_lanes {
                mh_dec[m * cap + t_slot] = 0.0;
            }
            let mask = prof.lap(&self.device, 4, || {
                Tensor::from_slice(&mh_dec, (m_lanes, 1, 1, cap), &self.device)
            })?;
            let x = prof.lap(&self.device, 0, || self.embed(&act_toks, &mpos))?;
            let hidden = prof.lap(&self.device, 1, || {
                self.forward_step(&x, &mut caches, t_slot, &mask)
            })?;
            for m in 0..m_lanes {
                mh_dec[m * cap + t_slot] = mcol[m];
            }
            let card_lg = prof.lap(&self.device, 2, || self.card_logits(&hidden))?;
            let t_sample = std::time::Instant::now();

            // --- sample one card per active lane ---
            for m in 0..m_lanes {
                if !active_at(step_i, m, &alive, &lane_of, &act) {
                    card_toks[m] = act_toks[m];
                    continue;
                }
                let a = &act[lane_of[m] as usize];
                let observer = a.spec.observer;
                let actor = gens[m].current;
                let r = (actor + 4 - observer) % 4;

                let mask_phys = if actor == observer {
                    gens[m].legal_for_hand(obs_remaining[m], actor)
                } else {
                    let unseen =
                        card::ALL_CARDS & !a.spec.observer_initial_hand & !gens[m].played;
                    gens[m].hidden_mask(actor, unseen)
                };
                if mask_phys == 0 || gens[m].remaining[actor as usize] == 0 {
                    alive[m] = false;
                    card_toks[m] = act_toks[m];
                    continue;
                }

                let mask_canon = permute_mask(mask_phys, &a.spec.perm);
                let mut logits = [0.0f32; 32];
                logits.copy_from_slice(&card_lg[m]);
                let canon_card = sample_masked(&logits, mask_canon, a.temperature, rng);
                if actor != observer {
                    let lp = masked_logp(&logits, mask_canon as u64, canon_card);
                    logps[m].sum += lp;
                    logps[m].n += 1;
                    if step_i * 2 < a.spec.steps {
                        logps[m].half_sum += lp;
                        logps[m].half_n += 1;
                    }
                }
                let phys_card = {
                    let cs = canon_card / 8;
                    let rank = canon_card % 8;
                    let mut ps = 0u8;
                    for s in 0..4u8 {
                        if a.spec.perm[s as usize] == cs {
                            ps = s;
                            break;
                        }
                    }
                    ps * 8 + rank
                };
                card_toks[m] = Tok {
                    primary: P_RANK0 + canon_card % 8,
                    suit: canon_card / 8,
                    actor: r,
                    segment: SEG_PLAY,
                };
                assigned[m][actor as usize] |= 1u32 << phys_card;
                if actor == observer {
                    obs_remaining[m] &= !(1u32 << phys_card);
                }
                gens[m].step(actor, phys_card);
            }

            if prof.on {
                prof.sample += t_sample.elapsed().as_secs_f64() * 1e3;
            }

            // --- card token ---
            for m in 0..m_lanes {
                let on = active_at(step_i, m, &alive, &lane_of, &act);
                if on {
                    mpos[m] = pos[m] + 1;
                    pos[m] += 2;
                } else {
                    mpos[m] = 0;
                }
                mcol[m] = if on { 0.0 } else { neg };
            }
            prof.steps += 1;
            let t_slot = lmax + 2 * step_i + 1;
            for m in 0..m_lanes {
                mh_dec[m * cap + t_slot] = 0.0;
            }
            let mask = prof.lap(&self.device, 4, || {
                Tensor::from_slice(&mh_dec, (m_lanes, 1, 1, cap), &self.device)
            })?;
            let x = prof.lap(&self.device, 0, || self.embed(&card_toks, &mpos))?;
            prof.lap(&self.device, 1, || self.forward_step(&x, &mut caches, t_slot, &mask))?;
            for m in 0..m_lanes {
                mh_dec[m * cap + t_slot] = mcol[m];
            }
        }
        prof.report(m_lanes, prefill_ms);

        // ══ Phase 4 — harvest, per item ══
        for m in 0..m_lanes {
            if !alive[m] {
                continue;
            }
            let a = &act[lane_of[m] as usize];
            let mut hands = assigned[m];
            hands[a.spec.observer as usize] = a.spec.observer_hand_now;
            if (0..4).any(|p| card::card_count(hands[p]) != card::card_count(a.state.hands[p])) {
                continue;
            }
            out[a.item].push((hands, logps[m]));
        }
        Ok(out)
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
        let plen = prefix.len();
        let cap = plen + 2 * spec.steps;
        let mut caches1 = self.new_kv(1, cap)?;
        let mut mh1 = vec![-1e9f32; cap];
        for (pos, tok) in prefix.iter().enumerate() {
            mh1[pos] = 0.0;
            let mask1 = Tensor::from_slice(&mh1, (1, 1, 1, cap), &self.device)?;
            let x = self.embed(std::slice::from_ref(tok), &[pos as u32])?;
            self.forward_step(&x, &mut caches1, pos, &mask1)?;
        }
        let sel = Tensor::from_slice(&vec![0u32; n_worlds], n_worlds, &self.device)?;
        let mut caches = self.select_kv(&caches1, &sel)?;
        let mut mh = vec![-1e9f32; n_worlds * cap];
        for k in 0..n_worlds {
            mh[k * cap..k * cap + plen].fill(0.0);
        }

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
            let t_slot = plen + 2 * step_i;
            for k in 0..n_worlds {
                mh[k * cap + t_slot] = 0.0;
            }
            let mask = Tensor::from_slice(&mh, (n_worlds, 1, 1, cap), &self.device)?;
            let x = self.embed(&act_toks, &pos_ids)?;
            let hidden = self.forward_step(&x, &mut caches, t_slot, &mask)?;
            for k in 0..n_worlds {
                mh[k * cap + t_slot] = if alive[k] { 0.0 } else { neg };
            }
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
                    gens[k].hidden_mask(actor, unseen)
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
            let t_slot = plen + 2 * step_i + 1;
            for k in 0..n_worlds {
                mh[k * cap + t_slot] = 0.0;
            }
            let mask = Tensor::from_slice(&mh, (n_worlds, 1, 1, cap), &self.device)?;
            let x = self.embed(&card_toks, &pos_ids)?;
            self.forward_step(&x, &mut caches, t_slot, &mask)?;
            for k in 0..n_worlds {
                mh[k * cap + t_slot] = if alive[k] { 0.0 } else { neg };
            }
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
        //
        // Les lanes se désynchronisent (les enchères n'ont pas la même
        // longueur), donc chacune avance à sa *position logique* propre — mais
        // toutes écrivent au même *créneau physique* du cache, un par pas. La
        // capacité est donc bornée par `max_seq_len`, la seule borne dure sur
        // le nombre de jetons qu'une lane peut émettre.
        let plen = prefix.len();
        let cap = self.max_seq_len;
        let mut caches1 = self.new_kv(1, cap)?;
        let mut mh1 = vec![-1e9f32; cap];
        for (pos, tok) in prefix.iter().enumerate() {
            mh1[pos] = 0.0;
            let mask1 = Tensor::from_slice(&mh1, (1, 1, 1, cap), &self.device)?;
            let x = self.embed(std::slice::from_ref(tok), &[pos as u32])?;
            self.forward_step(&x, &mut caches1, pos, &mask1)?;
        }
        let sel = Tensor::from_slice(&vec![0u32; n_lanes], n_lanes, &self.device)?;
        let mut caches = self.select_kv(&caches1, &sel)?;
        let mut mh = vec![-1e9f32; n_lanes * cap];
        for k in 0..n_lanes {
            mh[k * cap..k * cap + plen].fill(0.0);
        }
        // Créneau physique du prochain jeton, commun à toutes les lanes.
        let mut slot = plen;

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
            // Aucune carte posée à l'entame : la belote ne peut rien avoir dit.
            banned: [0; 4],
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
            for k in 0..n_lanes {
                mh[k * cap + slot] = 0.0;
            }
            let mask = Tensor::from_slice(&mh, (n_lanes, 1, 1, cap), &self.device)?;
            let x = self.embed(&act_toks, &pos_ids)?;
            let hidden = self.forward_step(&x, &mut caches, slot, &mask)?;
            // Validité future du créneau qu'on vient d'écrire
            for k in 0..n_lanes {
                mh[k * cap + slot] = if act_active[k] { 0.0 } else { neg };
            }
            slot += 1;

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
                                    banned: [0; 4], // enchère finie, rien de posé
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
                            gens[k].hidden_mask(actor, unseen)
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
                for k in 0..n_lanes {
                    mh[k * cap + slot] = 0.0;
                }
                let mask = Tensor::from_slice(&mh, (n_lanes, 1, 1, cap), &self.device)?;
                let x = self.embed(&action_toks, &pos_ids)?;
                self.forward_step(&x, &mut caches, slot, &mask)?;
                for k in 0..n_lanes {
                    mh[k * cap + slot] = if action_active[k] { 0.0 } else { neg };
                }
                slot += 1;
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
