//! Distillation with **auction-continuation** targets (COLVQL03).
//!
//! The previous attempt ([train_bid_distill.rs]) scored a bid as "my team takes this
//! contract and plays it". That is not the value of *making* the bid: the auction goes
//! on, partner may raise, opponents may overcall, and a cheap bad bid gets overcalled
//! rather than played. Worse, PASS was anchored to v6's own Q while bids were scored in
//! DD points — two different yardsticks, and the optimistic one always won. The bidder
//! came out over-aggressive and lost the arena at 47.3%.
//!
//! Here every candidate action is evaluated the same way:
//!   force the action → let the real policy (v6, all four seats) finish the auction
//!   inside the sampled world → score whatever contract results.
//!
//! PASS goes through exactly that path, so it is measured with the same instrument as
//! bidding. No anchor and no re-centring: the quantity is now internally consistent.
//!
//! The worlds come from playgen conditioned on the auction prefix, so the continuation
//! sees plausible hidden hands. Variety across worlds comes from the hands, not from
//! randomising the policy — v6 stays deterministic, and in one world partner holds a
//! long suit and raises while in another he passes.
//!
//! Usage:
//!   cargo run --bin train_bid_cont --features dmc_train --release -- \
//!     --labels data/bid_labels3/shard_local.ql --labels data/bid_labels3/shard_remote.ql \
//!     --games data/training/labelcorpus_120k.bin --out-dir models/bid_v8_cont --epochs 6

use std::fs;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_candle::BiddingTrainer;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{
    write_bid_observation_score_aware_v3, BID_MASK_DIM, BID_OBS_DIM_SCORE_AWARE_V3,
};
use colver_core::bid_train_env::score_aware_reward;
use colver_core::game_replay::GameReplay;
use colver_core::scoring::compute_deal_score;
use colver_core::state::{GameState, Phase};

use std::sync::atomic::AtomicUsize;
static CONT_N: AtomicUsize = AtomicUsize::new(0);
static CONT_VOID: AtomicUsize = AtomicUsize::new(0);
static CONT_VAL: AtomicUsize = AtomicUsize::new(0);

const MAGIC: &[u8; 8] = b"COLVQL03";
const OBS: usize = BID_OBS_DIM_SCORE_AWARE_V3;

struct Label {
    game_idx: u32,
    prefix_len: u16,
    observer: u8,
    hands: Vec<[u32; 4]>,
    pts: Vec<[u8; 4]>,
}

fn load_labels(path: &str) -> Vec<Label> {
    let data = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(data.len() >= 16, "{path}: too short");
    assert_eq!(&data[..8], MAGIC, "{path}: bad magic (need COLVQL03)");
    let mut out = Vec::new();
    let mut p = 16;
    while p + 8 <= data.len() {
        let game_idx = u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
        let prefix_len = u16::from_le_bytes(data[p + 4..p + 6].try_into().unwrap());
        let observer = data[p + 6];
        let n = data[p + 7] as usize;
        p += 8;
        if n == 0 || p + n * 20 > data.len() {
            break; // truncated tail from an interrupted writer
        }
        let mut hands = Vec::with_capacity(n);
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            let mut h = [0u32; 4];
            for (k, slot) in h.iter_mut().enumerate() {
                *slot = u32::from_le_bytes(data[p + k * 4..p + k * 4 + 4].try_into().unwrap());
            }
            p += 16;
            pts.push([data[p], data[p + 1], data[p + 2], data[p + 3]]);
            p += 4;
            hands.push(h);
        }
        out.push(Label { game_idx, prefix_len, observer, hands, pts });
    }
    out
}

/// Which team holds Q+K of `trump` in these hands (0 = nobody, else team+1).
fn belote_team(hands: &[u32; 4], trump: u8) -> u8 {
    let qk = (1u32 << (trump as u32 * 8 + 4)) | (1u32 << (trump as u32 * 8 + 5));
    for (p, h) in hands.iter().enumerate() {
        if h & qk == qk {
            return 1 + (p as u8 & 1);
        }
    }
    0
}

/// Score a finished auction inside one world, from `my_team`'s point of view.
#[allow(clippy::too_many_arguments)]
fn settle(
    state: &GameState,
    world_hands: &[u32; 4],
    world_pts: &[u8; 4],
    my_team: usize,
    cum: [i32; 2],
    scale: f32,
    clip: f32,
    calib: &[[u16; 64]],
    qidx: usize,
) -> f32 {
    // Void deal: nobody scores, but the match state still advances.
    if state.contract.value == 0 {
        return 0.0;
    }
    let trump = state.contract.trump;
    // DD assumes perfect play and flatters whoever declares — which pushes *both*
    // "I take it" and "they take it" in the direction that makes bidding look good, so
    // the bias does not cancel between branches. Draw a real-play outcome from
    // P(isdd | dd) instead. `qidx` is fixed per world, so every candidate action is
    // compared against the same realisation of play quality (common random numbers).
    let ns_pts = calib[world_pts[trump as usize] as usize][qidx].min(252) as u8;
    let ew_pts = if ns_pts == 252 || ns_pts == 0 { 252 - ns_pts } else { 162 - ns_pts };

    let taker = state.contract.team as usize;
    let defense = 1 - taker;
    let taker_pts = if taker == 0 { ns_pts } else { ew_pts };
    let defense_pts = if defense == 0 { ns_pts } else { ew_pts };

    let (tt, dt) = if defense_pts == 0 {
        (8u8, 0u8)
    } else if taker_pts == 0 {
        (0u8, 8u8)
    } else {
        let total = taker_pts as u16 + defense_pts as u16;
        let f = taker_pts as f32 / total as f32;
        let t = (f * 8.0).round().clamp(1.0, 7.0) as u8;
        (t, 8 - t)
    };

    let mut terminal = GameState::new(0, [0; 4]);
    terminal.phase = Phase::Done;
    terminal.contract = state.contract;
    terminal.points[taker] = taker_pts;
    terminal.points[defense] = defense_pts;
    terminal.tricks_won[taker] = tt;
    terminal.tricks_won[defense] = dt;
    let bt = belote_team(world_hands, trump);
    if bt > 0 {
        terminal.belote[(bt - 1) as usize] = 2;
    }

    let ds = compute_deal_score(&terminal);
    let mut r = score_aware_reward(
        cum[my_team] as f32,
        cum[1 - my_team] as f32,
        ds.scores[my_team],
        ds.scores[1 - my_team],
        scale,
    );
    if clip > 0.0 {
        r = r.clamp(-clip, clip);
    }
    r
}

/// Run the auction to its end with `net` on every seat, starting from `state`.
fn finish_auction(
    state: &mut GameState,
    hist: &mut Vec<(u8, u8)>,
    net: &mut BidNet,
    cum: [i32; 2],
    buf: &mut [f32],
) {
    let mut guard = 0;
    while state.phase == Phase::Bidding {
        guard += 1;
        if guard > 40 {
            break; // auctions are bounded; this only trips on a malformed state
        }
        let seat = state.current_player();
        let team = (seat & 1) as usize;
        write_bid_observation_score_aware_v3(buf, 0, state, hist, cum[team], cum[1 - team]);
        let q = net.evaluate(buf);
        let legal = state.legal_actions();
        let mut best = 0u8;
        let mut bv = f32::NEG_INFINITY;
        for a in 0..BID_MASK_DIM {
            if legal & (1u64 << a) != 0 && q[a] > bv {
                bv = q[a];
                best = a as u8;
            }
        }
        hist.push((seat, best));
        state.step(best);
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut label_paths: Vec<String> = Vec::new();
    let mut games_path = String::from("data/training/labelcorpus_120k.bin");
    let mut init = String::from("models/bid_v6_isdd_resume/bid_nn_final.safetensors");
    let mut frozen = String::from("models/bid_v6_isdd_resume/bid_nn_final.bin");
    let mut out_dir = String::from("models/bid_v8_cont");
    let mut epochs: usize = 6;
    let mut batch: usize = 512;
    let mut lr: f64 = 5e-5;
    let mut hidden: usize = 512;
    let mut layers: usize = 3;
    let mut scale: f32 = 3.0;
    let mut clip: f32 = 1.0;
    let mut seed: u64 = 777;
    let mut val_frac: f64 = 0.05;
    let mut top_k: usize = 12;
    let mut calib_path = String::new();
    let mut neutral_score = false;
    let mut threads: usize = 0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--labels" => { i += 1; label_paths.push(args[i].clone()); }
            "--games" => { i += 1; games_path = args[i].clone(); }
            "--init" => { i += 1; init = args[i].clone(); }
            "--frozen" => { i += 1; frozen = args[i].clone(); }
            "--out-dir" => { i += 1; out_dir = args[i].clone(); }
            "--epochs" => { i += 1; epochs = args[i].parse().unwrap(); }
            "--batch" => { i += 1; batch = args[i].parse().unwrap(); }
            "--lr" => { i += 1; lr = args[i].parse().unwrap(); }
            "--hidden" => { i += 1; hidden = args[i].parse().unwrap(); }
            "--layers" => { i += 1; layers = args[i].parse().unwrap(); }
            "--scale" => { i += 1; scale = args[i].parse().unwrap(); }
            "--clip" => { i += 1; clip = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--val-frac" => { i += 1; val_frac = args[i].parse().unwrap(); }
            "--top-k" => { i += 1; top_k = args[i].parse().unwrap(); }
            "--calib" => { i += 1; calib_path = args[i].clone(); }
            "--neutral-score" => { neutral_score = true; }
            "--threads" => { i += 1; threads = args[i].parse().unwrap(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }
    assert!(!label_paths.is_empty(), "need at least one --labels shard");
    fs::create_dir_all(&out_dir).ok();
    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }

    println!("Loading corpus {games_path}...");
    let replays = Arc::new(GameReplay::load_all(&games_path).expect("load corpus"));
    let mut labels = Vec::new();
    for p in &label_paths {
        let l = load_labels(p);
        println!("  {p}: {} records", l.len());
        labels.extend(l);
    }
    println!("  {} label records total", labels.len());
    assert!(!labels.is_empty());

    // Quantile table: identity when absent.
    let calib: Arc<Vec<[u16; 64]>> = Arc::new(if calib_path.is_empty() {
        (0..253u16).map(|v| [v; 64]).collect()
    } else {
        let txt = fs::read_to_string(&calib_path).expect("read calib");
        let mut t: Vec<[u16; 64]> = (0..253u16).map(|v| [v; 64]).collect();
        for line in txt.lines() {
            if line.starts_with('#') { continue; }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 66 { continue; }
            let dd: usize = f[0].parse().unwrap();
            if dd > 252 { continue; }
            for q in 0..64 { t[dd][q] = f[2 + q].parse().unwrap(); }
        }
        println!("  calibration: dd=120 -> p10 {} p50 {} p90 {}", t[120][6], t[120][32], t[120][57]);
        t
    });

    println!("\nBuilding continuation targets (top-{top_k} candidates + PASS)...");
    let t0 = Instant::now();

    use rayon::prelude::*;
    let built: Vec<(Vec<f32>, [f32; BID_MASK_DIM], [f32; BID_MASK_DIM], usize, usize)> = labels
        .par_iter()
        .enumerate()
        .filter_map(|(li, lab)| {
            let r = &replays[lab.game_idx as usize];
            let mut net = BidNet::load(&frozen).ok()?;
            let mut rng = StdRng::seed_from_u64(seed ^ (li as u64).wrapping_mul(0x9E37_79B9));
            let mut obs = vec![0.0f32; OBS];

            // Real position: this is what the net sees at inference time.
            let real0 = GameState::new(r.dealer, r.hands);
            let mut real = real0;
            let mut hist: Vec<(u8, u8)> = Vec::with_capacity(12);
            for &a in r.actions.iter().take(lab.prefix_len as usize) {
                hist.push((real.current_player(), a));
                real.step(a);
            }
            if real.phase != Phase::Bidding || real.current_player() != lab.observer {
                return None;
            }
            let my_team = (lab.observer & 1) as usize;
            let legal = real.legal_actions();

            // One match context per label, used for both the continuation and the reward
            // so the two are consistent.
            // v6 was trained with match simulation: both scores climb together toward
            // 2000. Sampling them independently and uniformly puts a quarter of the mass
            // in "I am hopelessly behind", where aggression is correct — which would
            // teach systematic over-bidding all by itself. Draw a match progress and a
            // differential instead.
            let progress: f64 = rng.gen_range(0.0..1.0);
            let lead = (progress * 1950.0) as i32;
            let diff = (rng.gen_range(0.0..1.0f64).powf(1.6) * 600.0) as i32;
            let (ns_cum, ew_cum) = if neutral_score {
                // Diagnostic only: the corpus was generated at 0-0, so this is the
                // apples-to-apples setting for comparing contract levels.
                (0, 0)
            } else if rng.gen_bool(0.5) {
                (lead, (lead - diff).max(0))
            } else {
                ((lead - diff).max(0), lead)
            };
            let cum = [ns_cum, ew_cum];
            let my_cum = cum[my_team];
            let opp_cum = cum[1 - my_team];
            write_bid_observation_score_aware_v3(&mut obs, 0, &real, &hist, my_cum, opp_cum);

            // Candidate set: PASS is always in, plus the top-k bids v6 rates highest.
            // Sharpening the actions v6 would never consider buys nothing and the
            // continuation is the expensive part.
            let q6 = net.evaluate(&obs);
            let mut cands: Vec<usize> = (0..BID_MASK_DIM).filter(|&a| legal & (1u64 << a) != 0).collect();
            if top_k > 0 && cands.len() > top_k + 1 {
                cands.sort_by(|&a, &b| q6[b].partial_cmp(&q6[a]).unwrap());
                cands.truncate(top_k);
                if !cands.contains(&0) {
                    cands.push(0); // PASS
                }
            }

            // One play-quality draw per world, shared by every candidate action.
            let qidx: Vec<usize> = (0..lab.hands.len()).map(|_| rng.gen_range(0..64)).collect();

            let mut target = [0.0f32; BID_MASK_DIM];
            let mut mask = [0.0f32; BID_MASK_DIM];
            let mut cbuf = vec![0.0f32; OBS];
            for &a in &cands {
                let mut acc = 0.0f32;
                for (w, wh) in lab.hands.iter().enumerate() {
                    // Same auction prefix, but this world's hidden hands.
                    let mut st = GameState::new(r.dealer, *wh);
                    let mut h2: Vec<(u8, u8)> = Vec::with_capacity(12);
                    for &pa in r.actions.iter().take(lab.prefix_len as usize) {
                        h2.push((st.current_player(), pa));
                        st.step(pa);
                    }
                    if st.phase != Phase::Bidding {
                        continue;
                    }
                    h2.push((st.current_player(), a as u8));
                    st.step(a as u8);
                    finish_auction(&mut st, &mut h2, &mut net, cum, &mut cbuf);
                    CONT_N.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if st.contract.value == 0 {
                        CONT_VOID.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    } else {
                        CONT_VAL.fetch_add(st.contract.point_value() as usize, std::sync::atomic::Ordering::Relaxed);
                    }
                    acc += settle(&st, wh, &lab.pts[w], my_team, cum, scale, clip, &calib, qidx[w]);
                }
                target[a] = acc / lab.hands.len() as f32;
                mask[a] = 1.0;
            }
            Some((obs, target, mask, cands.len(), li))
        })
        .collect();

    let n_samples = built.len();
    let avg_c: f64 = built.iter().map(|b| b.3 as f64).sum::<f64>() / n_samples as f64;
    println!(
        "  {} samples in {:.1}s ({:.1} candidates/position avg)",
        n_samples,
        t0.elapsed().as_secs_f64(),
        avg_c
    );

    // ── Calibration report: does the continuation target agree with v6 on the
    // bid/pass balance? The previous run's failure showed up exactly here.
    {
        let mut fb = BidNet::load(&frozen).unwrap();
        let (mut tp, mut vp) = (0usize, 0usize);
        for (obs, target, mask, _, _) in built.iter() {
            let q = fb.evaluate(obs);
            let (mut bt, mut bq) = (99usize, 99usize);
            let (mut btv, mut bqv) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for a in 0..BID_MASK_DIM {
                if mask[a] < 0.5 { continue; }
                if target[a] > btv { btv = target[a]; bt = a; }
                if q[a] > bqv { bqv = q[a]; bq = a; }
            }
            if bt == 0 { tp += 1; }
            if bq == 0 { vp += 1; }
        }
        println!(
            "\n  argmax is PASS: continuation target {:.1}%   v6 {:.1}%",
            tp as f64 / n_samples as f64 * 100.0,
            vp as f64 / n_samples as f64 * 100.0
        );
    }

    // ── Auction sanity: do v6-inside-a-playgen-world auctions look like real ones?
    // If the sampled hands were off-distribution the continuations would be nonsense
    // and every target with them.
    {
        let mut real_vals: Vec<f64> = Vec::new();
        let mut real_void = 0usize;
        for r in replays.iter().take(20000) {
            let mut st = GameState::new(r.dealer, r.hands);
            for &a in &r.actions {
                if st.phase != Phase::Bidding { break; }
                st.step(a);
            }
            if st.contract.value == 0 { real_void += 1; } else { real_vals.push(st.contract.point_value() as f64); }
        }
        let rm = real_vals.iter().sum::<f64>() / real_vals.len().max(1) as f64;
        println!(
            "\n  real corpus auctions: {:.1}% void, mean contract {:.1}",
            real_void as f64 / 20000.0 * 100.0, rm
        );
        println!(
            "  continuation auctions: {:.1}% void, mean contract {:.1}",
            CONT_VOID.load(std::sync::atomic::Ordering::Relaxed) as f64
                / CONT_N.load(std::sync::atomic::Ordering::Relaxed).max(1) as f64 * 100.0,
            CONT_VAL.load(std::sync::atomic::Ordering::Relaxed) as f64
                / (CONT_N.load(std::sync::atomic::Ordering::Relaxed)
                    - CONT_VOID.load(std::sync::atomic::Ordering::Relaxed)).max(1) as f64
        );
    }

    let mut xs: Vec<f32> = Vec::with_capacity(n_samples * OBS);
    let mut ys: Vec<f32> = Vec::with_capacity(n_samples * BID_MASK_DIM);
    let mut ms: Vec<f32> = Vec::with_capacity(n_samples * BID_MASK_DIM);
    for (obs, t, m, _, _) in &built {
        xs.extend_from_slice(obs);
        ys.extend_from_slice(t);
        ms.extend_from_slice(m);
    }
    drop(built);

    let n_val = ((n_samples as f64) * val_frac) as usize;
    let n_train = n_samples - n_val;
    println!("  train {n_train} / val {n_val}");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("  device: {device:?}");
    let mut trainer =
        BiddingTrainer::with_layers_and_obs(layers, hidden, OBS, lr, 0.0, device.clone())
            .expect("build trainer");
    trainer.load_checkpoint(&init).expect("load v6 checkpoint");

    let val_x = Tensor::from_slice(&xs[n_train * OBS..], (n_val, OBS), &device).unwrap();
    let val_y = Tensor::from_slice(&ys[n_train * BID_MASK_DIM..], (n_val, BID_MASK_DIM), &device).unwrap();
    let val_m = Tensor::from_slice(&ms[n_train * BID_MASK_DIM..], (n_val, BID_MASK_DIM), &device).unwrap();

    let masked_mse = |q: &Tensor, y: &Tensor, m: &Tensor| -> candle_core::Result<Tensor> {
        let d = (q - y)?;
        let sq = (&d * &d)?;
        ((&sq * m)?.sum_all()? / m.sum_all()?)
    };
    let policy = |q: &Tensor| -> Vec<usize> {
        let qv: Vec<Vec<f32>> = q.to_vec2().unwrap();
        let mv: Vec<Vec<f32>> = val_m.to_vec2().unwrap();
        qv.iter().zip(mv.iter()).map(|(row, m)| {
            let (mut b, mut bv) = (0usize, f32::NEG_INFINITY);
            for a in 0..BID_MASK_DIM {
                if m[a] > 0.5 && row[a] > bv { bv = row[a]; b = a; }
            }
            b
        }).collect()
    };
    let agree = |a: &[usize], b: &[usize]| -> f64 {
        a.iter().zip(b.iter()).filter(|(x, y)| x == y).count() as f64 / a.len() as f64 * 100.0
    };

    let tgt_pol = policy(&val_y);
    let v6_pol = policy(&trainer.net.forward(&val_x).unwrap());
    {
        let q = trainer.net.forward(&val_x).unwrap();
        let l = masked_mse(&q, &val_y, &val_m).unwrap().to_vec0::<f32>().unwrap();
        println!("\n  val MSE, v6 as-is: {l:.6}");
        println!("  v6 agrees with continuation targets on {:.1}% of val positions", agree(&v6_pol, &tgt_pol));
    }

    let mut order: Vec<usize> = (0..n_train).collect();
    let mut rng = StdRng::seed_from_u64(seed ^ 0xABCD);
    println!("\nTraining {epochs} epochs, batch {batch}, lr {lr}");
    for ep in 0..epochs {
        for k in (1..order.len()).rev() {
            let j = rng.gen_range(0..=k);
            order.swap(k, j);
        }
        let t = Instant::now();
        let (mut run, mut nb) = (0.0f64, 0usize);
        let mut bx = vec![0.0f32; batch * OBS];
        let mut by = vec![0.0f32; batch * BID_MASK_DIM];
        let mut bm = vec![0.0f32; batch * BID_MASK_DIM];
        for chunk in order.chunks(batch) {
            let b = chunk.len();
            for (i, &idx) in chunk.iter().enumerate() {
                bx[i * OBS..(i + 1) * OBS].copy_from_slice(&xs[idx * OBS..(idx + 1) * OBS]);
                by[i * BID_MASK_DIM..(i + 1) * BID_MASK_DIM]
                    .copy_from_slice(&ys[idx * BID_MASK_DIM..(idx + 1) * BID_MASK_DIM]);
                bm[i * BID_MASK_DIM..(i + 1) * BID_MASK_DIM]
                    .copy_from_slice(&ms[idx * BID_MASK_DIM..(idx + 1) * BID_MASK_DIM]);
            }
            let x = Tensor::from_slice(&bx[..b * OBS], (b, OBS), &device).unwrap();
            let y = Tensor::from_slice(&by[..b * BID_MASK_DIM], (b, BID_MASK_DIM), &device).unwrap();
            let m = Tensor::from_slice(&bm[..b * BID_MASK_DIM], (b, BID_MASK_DIM), &device).unwrap();
            let q = trainer.net.forward(&x).unwrap();
            let loss = masked_mse(&q, &y, &m).unwrap();
            trainer.backward_step(&loss).unwrap();
            run += loss.to_vec0::<f32>().unwrap() as f64;
            nb += 1;
        }
        let q = trainer.net.forward(&val_x).unwrap();
        let vl = masked_mse(&q, &val_y, &val_m).unwrap().to_vec0::<f32>().unwrap();
        let p = policy(&q);
        println!(
            "  epoch {:>2}  train {:.6}  val {:.6}  | matches target {:.1}% | differs from v6 {:.1}%  ({:.1}s)",
            ep + 1, run / nb as f64, vl, agree(&p, &tgt_pol), 100.0 - agree(&p, &v6_pol),
            t.elapsed().as_secs_f64()
        );
        trainer.export_binary(&format!("{out_dir}/bid_nn_ep{}.bin", ep + 1)).unwrap();
        trainer.save_checkpoint(&format!("{out_dir}/bid_nn_ep{}.safetensors", ep + 1)).unwrap();
    }
    println!("\nWrote {out_dir}/bid_nn_ep*.bin");
}
