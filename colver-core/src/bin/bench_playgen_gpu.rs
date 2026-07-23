//! GPU vs CPU inference bench for the playgen transformer.
//!
//! Mesure le débit de génération (pattern décodage token-par-token avec KV
//! cache, sync host à chaque pas pour lire les logits — le pattern réel du
//! sampling de mondes) sur CUDA (candle) contre le forward lockstep CPU pur
//! (`forward_tokens_batch`), à plusieurs tailles de batch.
//!
//! Un « monde » ≈ 64 pas de décodage (2 tokens × 32 cartes) après un préfixe.
//!
//! Usage:
//!   cargo run -p colver-core --bin bench_playgen_gpu --features dmc_train \
//!     --release -- --playgen models/playgen_v2/playgen_v2_half.bin \
//!     --batches 1,8,32,128,512 --steps 64 --prefix 58

use candle_core::{Device, Tensor, D};

use colver_core::playgen::infer::{KvCache, KvCacheBatch, PlaygenModel, Tok};

struct GpuBlock {
    attn_norm: Tensor, // [d]
    qkv_w_t: Tensor,   // [d, 3d]
    qkv_b: Tensor,     // [3d]
    out_w_t: Tensor,   // [d, d]
    out_b: Tensor,
    ffn_norm: Tensor,
    gate_w_t: Tensor, // [d, dff]
    gate_b: Tensor,
    up_w_t: Tensor,
    up_b: Tensor,
    down_w_t: Tensor, // [dff, d]
    down_b: Tensor,
}

struct GpuModel {
    d: usize,
    n_heads: usize,
    hd: usize,
    primary_emb: Tensor, // [NUM_PRIMARY, d]
    suit_emb: Tensor,
    actor_emb: Tensor,
    seg_emb: Tensor,
    pos_emb: Tensor, // [T, d]
    blocks: Vec<GpuBlock>,
    out_norm: Tensor,
    head_w_t: Tensor, // [d, 32]
    head_b: Tensor,
}

fn t2(dev: &Device, data: &[f32], rows: usize, cols: usize) -> Tensor {
    Tensor::from_slice(data, (rows, cols), dev).unwrap()
}

fn t1(dev: &Device, data: &[f32]) -> Tensor {
    Tensor::from_slice(data, data.len(), dev).unwrap()
}

impl GpuModel {
    fn from_flat(m: &PlaygenModel, dev: &Device) -> Self {
        let d = m.d;
        let blocks = m
            .blocks
            .iter()
            .map(|b| GpuBlock {
                attn_norm: t1(dev, &b.attn_norm),
                qkv_w_t: t2(dev, &b.qkv_w, 3 * d, d).t().unwrap().contiguous().unwrap(),
                qkv_b: t1(dev, &b.qkv_b),
                out_w_t: t2(dev, &b.out_w, d, d).t().unwrap().contiguous().unwrap(),
                out_b: t1(dev, &b.out_b),
                ffn_norm: t1(dev, &b.ffn_norm),
                gate_w_t: t2(dev, &b.gate_w, m.dff, d).t().unwrap().contiguous().unwrap(),
                gate_b: t1(dev, &b.gate_b),
                up_w_t: t2(dev, &b.up_w, m.dff, d).t().unwrap().contiguous().unwrap(),
                up_b: t1(dev, &b.up_b),
                down_w_t: t2(dev, &b.down_w, d, m.dff).t().unwrap().contiguous().unwrap(),
                down_b: t1(dev, &b.down_b),
            })
            .collect();
        GpuModel {
            d,
            n_heads: m.n_heads,
            hd: d / m.n_heads,
            primary_emb: t2(dev, &m.primary_emb, m.primary_emb.len() / d, d),
            suit_emb: t2(dev, &m.suit_emb, m.suit_emb.len() / d, d),
            actor_emb: t2(dev, &m.actor_emb, m.actor_emb.len() / d, d),
            seg_emb: t2(dev, &m.seg_emb, m.seg_emb.len() / d, d),
            pos_emb: t2(dev, &m.pos_emb, m.max_seq_len, d),
            blocks,
            out_norm: t1(dev, &m.out_norm),
            head_w_t: t2(dev, &m.head_w, 32, d).t().unwrap().contiguous().unwrap(),
            head_b: t1(dev, &m.head_b),
        }
    }

    fn rmsnorm(&self, x: &Tensor, w: &Tensor) -> Tensor {
        let ms = x.sqr().unwrap().mean_keepdim(D::Minus1).unwrap();
        let rms = (ms + 1e-6f64).unwrap().sqrt().unwrap();
        x.broadcast_div(&rms).unwrap().broadcast_mul(w).unwrap()
    }

    /// Embed one token per lane: ids are [B] u32 per channel, pos scalar.
    fn embed(&self, prim: &Tensor, suit: &Tensor, actor: &Tensor, seg: &Tensor, pos: usize) -> Tensor {
        let e = self.primary_emb.index_select(prim, 0).unwrap();
        let e = (e + self.suit_emb.index_select(suit, 0).unwrap()).unwrap();
        let e = (e + self.actor_emb.index_select(actor, 0).unwrap()).unwrap();
        let e = (e + self.seg_emb.index_select(seg, 0).unwrap()).unwrap();
        e.broadcast_add(&self.pos_emb.narrow(0, pos, 1).unwrap()).unwrap()
    }

    /// One decode step for B lanes. caches: per layer (k, v) as [B, H, T, hd].
    fn forward_step(&self, x: &Tensor, caches: &mut Vec<Option<(Tensor, Tensor)>>) -> Tensor {
        let (b, _) = x.dims2().unwrap();
        let mut x = x.clone();
        let scale = 1.0 / (self.hd as f64).sqrt();
        for (l, blk) in self.blocks.iter().enumerate() {
            let normed = self.rmsnorm(&x, &blk.attn_norm);
            let qkv = normed.matmul(&blk.qkv_w_t).unwrap().broadcast_add(&blk.qkv_b).unwrap();
            let q = qkv.narrow(1, 0, self.d).unwrap();
            let k = qkv.narrow(1, self.d, self.d).unwrap();
            let v = qkv.narrow(1, 2 * self.d, self.d).unwrap();
            // [B, d] → [B, H, 1, hd]
            let shape = (b, self.n_heads, 1, self.hd);
            let q = q.reshape(shape).unwrap();
            let k = k.reshape(shape).unwrap();
            let v = v.reshape(shape).unwrap();
            let (k_cache, v_cache) = match caches[l].take() {
                Some((kc, vc)) => (
                    Tensor::cat(&[&kc, &k], 2).unwrap(),
                    Tensor::cat(&[&vc, &v], 2).unwrap(),
                ),
                None => (k, v),
            };
            // scores [B, H, 1, T]
            let scores = (q
                .matmul(&k_cache.transpose(2, 3).unwrap().contiguous().unwrap())
                .unwrap()
                * scale)
                .unwrap();
            let probs = candle_nn::ops::softmax(&scores, D::Minus1).unwrap();
            let ctx = probs.matmul(&v_cache).unwrap(); // [B, H, 1, hd]
            let ctx = ctx.reshape((b, self.d)).unwrap();
            caches[l] = Some((k_cache, v_cache));
            let attn = ctx.matmul(&blk.out_w_t).unwrap().broadcast_add(&blk.out_b).unwrap();
            x = (x + attn).unwrap();

            let normed = self.rmsnorm(&x, &blk.ffn_norm);
            let gate = normed
                .matmul(&blk.gate_w_t)
                .unwrap()
                .broadcast_add(&blk.gate_b)
                .unwrap()
                .gelu()
                .unwrap();
            let up = normed.matmul(&blk.up_w_t).unwrap().broadcast_add(&blk.up_b).unwrap();
            let ffn = (gate * up)
                .unwrap()
                .matmul(&blk.down_w_t)
                .unwrap()
                .broadcast_add(&blk.down_b)
                .unwrap();
            x = (x + ffn).unwrap();
        }
        x
    }

    fn logits(&self, hidden: &Tensor) -> Vec<Vec<f32>> {
        let normed = self.rmsnorm(hidden, &self.out_norm);
        normed
            .matmul(&self.head_w_t)
            .unwrap()
            .broadcast_add(&self.head_b)
            .unwrap()
            .to_vec2::<f32>()
            .unwrap()
    }
}

/// Pseudo-random valid token stream (same for CPU and GPU runs).
fn tok_stream(n: usize) -> Vec<Tok> {
    (0..n)
        .map(|i| Tok {
            primary: (1 + (i * 7) % 30) as u8,
            suit: (i % 5) as u8,
            actor: (i % 5) as u8,
            segment: (i % 3) as u8,
        })
        .collect()
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut playgen_path = String::from("models/playgen_v2/playgen_v2_half.bin");
    let mut batches: Vec<usize> = vec![1, 8, 32, 128, 512];
    let mut steps = 64usize;
    let mut prefix_len = 58usize;
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--playgen" => { playgen_path = args[i + 1].clone(); i += 2; }
            "--batches" => {
                batches = args[i + 1].split(',').map(|s| s.parse().unwrap()).collect();
                i += 2;
            }
            "--steps" => { steps = args[i + 1].parse().unwrap(); i += 2; }
            "--prefix" => { prefix_len = args[i + 1].parse().unwrap(); i += 2; }
            other => { eprintln!("unknown arg {}", other); std::process::exit(1); }
        }
    }
    assert!(prefix_len + steps <= 122, "prefix + steps exceeds v2 max_seq_len");

    let model = PlaygenModel::load(&playgen_path).expect("load playgen model");
    println!(
        "bench_playgen_gpu — d={} L={} H={} ({:.1}M params), prefix {} + {} pas de décodage",
        model.d,
        model.n_layers,
        model.n_heads,
        (model.blocks.len() * (3 * model.d * model.d + model.d * model.d + 3 * model.d * model.dff))
            as f64
            / 1e6,
        prefix_len,
        steps
    );
    let toks = tok_stream(prefix_len + steps);

    // ------------------------------------------------------------------
    // CPU reference: pure-Rust lockstep forward (prefix single-stream once,
    // then B lanes decode). Mirror of generate_worlds_batch's compute.
    // ------------------------------------------------------------------
    println!("\n== CPU (forward_tokens_batch, 1 thread) ==");
    let mut prefix_cache = KvCache::new(&model);
    for (pos, tok) in toks.iter().take(prefix_len).enumerate() {
        model.forward_token(&mut prefix_cache, *tok, pos);
    }
    for &b in &batches {
        if b > 128 {
            println!("  B={:4}: skipped (CPU trop lent)", b);
            continue;
        }
        let active = vec![true; b];
        let t0 = std::time::Instant::now();
        let mut cache = KvCacheBatch::from_prefix(&model, &prefix_cache, b);
        let mut sink = 0.0f32;
        for tok in toks.iter().skip(prefix_len) {
            let batch_toks = vec![*tok; b];
            let hidden = model.forward_tokens_batch(&mut cache, &batch_toks, &active);
            // Host-side logits read per step, like real sampling.
            for k in 0..b {
                sink += model.logits(&hidden[k * model.d..(k + 1) * model.d])[0];
            }
        }
        let dt = t0.elapsed().as_secs_f64();
        report("CPU", b, steps, dt, sink);
    }

    // ------------------------------------------------------------------
    // GPU: candle CUDA, KV-cache decode with per-step host sync.
    // ------------------------------------------------------------------
    let dev = Device::new_cuda(0).expect("CUDA device");
    let gpu = GpuModel::from_flat(&model, &dev);
    println!("\n== GPU (candle CUDA, sync host à chaque pas) ==");
    for &b in &batches {
        // Token ids per channel for the whole stream (same tokens each lane).
        let ids = |f: fn(&Tok) -> u8, tok: &Tok| vec![f(tok) as u32; b];

        // Warmup + timed run
        for timed in [false, true] {
            let mut caches: Vec<Option<(Tensor, Tensor)>> = vec![None; model.n_layers];
            let t0 = std::time::Instant::now();
            // Prefill token by token (simple; not counted in decode timing)
            for (pos, tok) in toks.iter().take(prefix_len).enumerate() {
                let prim = Tensor::from_slice(&ids(|t| t.primary, tok), b, &dev).unwrap();
                let suit = Tensor::from_slice(&ids(|t| t.suit, tok), b, &dev).unwrap();
                let actor = Tensor::from_slice(&ids(|t| t.actor, tok), b, &dev).unwrap();
                let seg = Tensor::from_slice(&ids(|t| t.segment, tok), b, &dev).unwrap();
                let x = gpu.embed(&prim, &suit, &actor, &seg, pos);
                gpu.forward_step(&x, &mut caches);
            }
            let prefill_dt = t0.elapsed().as_secs_f64();
            let t1 = std::time::Instant::now();
            let mut sink = 0.0f32;
            for (si, tok) in toks.iter().skip(prefix_len).enumerate() {
                let pos = prefix_len + si;
                let prim = Tensor::from_slice(&ids(|t| t.primary, tok), b, &dev).unwrap();
                let suit = Tensor::from_slice(&ids(|t| t.suit, tok), b, &dev).unwrap();
                let actor = Tensor::from_slice(&ids(|t| t.actor, tok), b, &dev).unwrap();
                let seg = Tensor::from_slice(&ids(|t| t.segment, tok), b, &dev).unwrap();
                let x = gpu.embed(&prim, &suit, &actor, &seg, pos);
                let hidden = gpu.forward_step(&x, &mut caches);
                let lg = gpu.logits(&hidden); // host sync
                sink += lg[0][0];
            }
            let dt = t1.elapsed().as_secs_f64();
            if timed {
                report("GPU", b, steps, dt, sink);
                println!("           (prefill {} tokens: {:.1} ms)", prefix_len, prefill_dt * 1e3);
            }
        }
    }
}

fn report(dev: &str, b: usize, steps: usize, dt: f64, sink: f32) {
    let ms_step = dt * 1e3 / steps as f64;
    // Un monde = `steps` pas de décodage ; B mondes générés par run.
    let worlds_per_s = b as f64 / dt;
    println!(
        "  {} B={:4}: {:7.2} ms/pas  {:8.1} mondes-équiv/s   (sink {:.1})",
        dev, b, ms_step, worlds_per_s, sink
    );
}
