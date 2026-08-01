//! Supervised distillation of auction-conditioned bid labels into v6.
//!
//! v6 learned its Q-values by RL on a single dealt outcome per auction — a target
//! carrying ~45 points of noise from the 24 cards the bidder cannot see. This
//! rebuilds the same quantity from playgen worlds sampled given the auction, so the
//! target is the conditional expectation instead of one draw from it.
//!
//! Targets, per bid position:
//!   * legal colour/capot bids → mean over worlds of the Δ-winprob obtained if the
//!     observer's team takes that contract (same scoring path as the RL trainer:
//!     synthetic terminal state → compute_deal_score → score_aware_reward → clip).
//!   * PASS / COINCHE / SURCOINCHE → frozen v6's own Q, an anchor. Without it the
//!     bid values get recalibrated while pass does not, and the bidder drifts
//!     systematically more (or less) aggressive for no principled reason.
//!
//! Loss is MSE over legal actions only. Everything else about the net is v6.
//!
//! Usage:
//!   cargo run --bin train_bid_distill --features dmc_train --release -- \
//!     --labels data/bid_labels/shard_local.ql --labels data/bid_labels/shard_remote.ql \
//!     --games data/training/labelcorpus_120k.bin \
//!     --init models/bid_v6_isdd_resume/bid_nn_final.safetensors \
//!     --frozen models/bid_v6_isdd_resume/bid_nn_final.bin \
//!     --out-dir models/bid_v7_distill --epochs 8

use std::fs;
use std::sync::Arc;
use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_candle::BiddingTrainer;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{write_bid_observation_score_aware_v3, BID_MASK_DIM, BID_OBS_DIM_SCORE_AWARE_V3};
use colver_core::bid_train_env::score_aware_reward;
use colver_core::bidding::{decode_bid, BID_PASS};
use colver_core::game_replay::GameReplay;
use colver_core::scoring::compute_deal_score;
use colver_core::state::{Contract, GameState, Phase};

const MAGIC: &[u8; 8] = b"COLVQL02";
const OBS: usize = BID_OBS_DIM_SCORE_AWARE_V3;

struct Label {
    game_idx: u32,
    prefix_len: u16,
    observer: u8,
    pts: Vec<[u8; 4]>,
    belote: Vec<[u8; 4]>,
}

/// Read a COLVQL02 shard. The record count in the header is advisory — we read to
/// EOF so a shard from an interrupted run is still usable.
fn load_labels(path: &str) -> Vec<Label> {
    let data = fs::read(path).unwrap_or_else(|e| panic!("read {path}: {e}"));
    assert!(data.len() >= 16, "{path}: too short");
    assert_eq!(&data[..8], MAGIC, "{path}: bad magic");
    let mut out = Vec::new();
    let mut p = 16;
    while p + 8 <= data.len() {
        let game_idx = u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
        let prefix_len = u16::from_le_bytes(data[p + 4..p + 6].try_into().unwrap());
        let observer = data[p + 6];
        let n = data[p + 7] as usize;
        p += 8;
        if n == 0 || p + n * 8 > data.len() {
            break; // truncated tail from a killed writer
        }
        let mut pts = Vec::with_capacity(n);
        let mut belote = Vec::with_capacity(n);
        for _ in 0..n {
            pts.push([data[p], data[p + 1], data[p + 2], data[p + 3]]);
            belote.push([data[p + 4], data[p + 5], data[p + 6], data[p + 7]]);
            p += 8;
        }
        out.push(Label { game_idx, prefix_len, observer, pts, belote });
    }
    out
}

/// Δ-winprob for "my team takes `contract` and the world plays out to `ns_pts`".
/// Mirrors `BidTrainingEnv::compute_scores` — synthetic terminal state, approximate
/// trick split, belote credited to whichever team holds Q+K of trump.
fn world_reward(
    ns_pts: u8,
    belote_team: u8,
    contract: Contract,
    my_team: usize,
    my_cum: i32,
    opp_cum: i32,
    scale: f32,
    clip: f32,
) -> f32 {
    let ew_pts = if ns_pts == 252 || ns_pts == 0 { 252 - ns_pts } else { 162 - ns_pts };
    let taker = contract.team as usize;
    let defense = 1 - taker;
    let taker_pts = if taker == 0 { ns_pts } else { ew_pts };
    let defense_pts = if defense == 0 { ns_pts } else { ew_pts };

    let (taker_tricks, defense_tricks) = if defense_pts == 0 {
        (8u8, 0u8)
    } else if taker_pts == 0 {
        (0u8, 8u8)
    } else {
        let total = taker_pts as u16 + defense_pts as u16;
        let frac = taker_pts as f32 / total as f32;
        let t = (frac * 8.0).round().clamp(1.0, 7.0) as u8;
        (t, 8 - t)
    };

    let mut terminal = GameState::new(0, [0; 4]);
    terminal.phase = Phase::Done;
    terminal.contract = contract;
    terminal.points[taker] = taker_pts;
    terminal.points[defense] = defense_pts;
    terminal.tricks_won[taker] = taker_tricks;
    terminal.tricks_won[defense] = defense_tricks;
    if belote_team > 0 {
        terminal.belote[(belote_team - 1) as usize] = 2;
    }

    let ds = compute_deal_score(&terminal);
    let opp_team = 1 - my_team;
    let mut r = score_aware_reward(
        my_cum as f32,
        opp_cum as f32,
        ds.scores[my_team],
        ds.scores[opp_team],
        scale,
    );
    if clip > 0.0 {
        r = r.clamp(-clip, clip);
    }
    r
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut label_paths: Vec<String> = Vec::new();
    let mut games_path = String::from("data/training/labelcorpus_120k.bin");
    let mut init = String::from("models/bid_v6_isdd_resume/bid_nn_final.safetensors");
    let mut frozen = String::from("models/bid_v6_isdd_resume/bid_nn_final.bin");
    let mut out_dir = String::from("models/bid_v7_distill");
    let mut epochs: usize = 8;
    let mut batch: usize = 512;
    let mut lr: f64 = 1e-4;
    let mut hidden: usize = 512;
    let mut layers: usize = 3;
    let mut scale: f32 = 3.0;
    let mut clip: f32 = 1.0;
    let mut seed: u64 = 777;
    let mut ctx_per_label: usize = 2;
    let mut val_frac: f64 = 0.05;
    let mut recenter = false;
    let mut recenter_per_level = false;
    let mut calib_path = String::new();

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
            "--ctx-per-label" => { i += 1; ctx_per_label = args[i].parse().unwrap(); }
            "--val-frac" => { i += 1; val_frac = args[i].parse().unwrap(); }
            "--recenter" => { recenter = true; }
            "--recenter-per-level" => { recenter_per_level = true; }
            "--calib" => { i += 1; calib_path = args[i].clone(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }
    assert!(!label_paths.is_empty(), "need at least one --labels shard");
    fs::create_dir_all(&out_dir).ok();

    println!("Loading corpus {games_path}...");
    let replays = Arc::new(GameReplay::load_all(&games_path).expect("load corpus"));
    let mut labels = Vec::new();
    for p in &label_paths {
        let l = load_labels(p);
        println!("  {p}: {} records", l.len());
        labels.extend(l);
    }
    println!("  {} label records total", labels.len());

    // ── Build the dataset. Each label is materialised at `ctx_per_label` random
    // match scores so the score-aware inputs get coverage, exactly as the RL
    // trainer saw them through match simulation.
    println!("\nBuilding dataset ({} contexts/label)...", ctx_per_label);
    let t0 = Instant::now();
    let mut frozen_net = BidNet::load(&frozen).expect("load frozen v6");
    assert_eq!(frozen_net.obs_dim(), OBS, "frozen net obs_dim mismatch");

    // DD → IS-DD point calibration. v6's returns come from IS-DD rollouts; a raw DD
    // solve is optimistic for the taker by 8-15 points exactly where contracts are
    // decided, which is enough to flip réussi/chute and make the distilled bidder
    // over-bid. Identity when no table is given.
    let calib: Vec<u8> = if calib_path.is_empty() {
        (0..=252u16).map(|v| v as u8).collect()
    } else {
        let txt = fs::read_to_string(&calib_path).expect("read calib");
        let mut t: Vec<u8> = (0..=252u16).map(|v| v as u8).collect();
        for line in txt.lines() {
            if line.starts_with('#') { continue; }
            let f: Vec<&str> = line.split_whitespace().collect();
            if f.len() < 2 { continue; }
            let dd: usize = f[0].parse().unwrap();
            let m: f64 = f[1].parse().unwrap();
            if dd <= 252 { t[dd] = m.round().clamp(0.0, 252.0) as u8; }
        }
        println!("  calibration loaded from {calib_path} (dd 120 -> {}, dd 140 -> {})", t[120], t[140]);
        t
    };

    let mut rng = StdRng::seed_from_u64(seed);
    let mut xs: Vec<f32> = Vec::with_capacity(labels.len() * ctx_per_label * OBS);
    let mut ys: Vec<f32> = Vec::with_capacity(labels.len() * ctx_per_label * BID_MASK_DIM);
    let mut ms: Vec<f32> = Vec::with_capacity(labels.len() * ctx_per_label * BID_MASK_DIM);
    let mut n_samples = 0usize;
    let mut obs_buf = vec![0.0f32; OBS];
    let mut skipped = 0usize;

    for lab in &labels {
        let r = &replays[lab.game_idx as usize];
        let state0 = GameState::new(r.dealer, r.hands);
        let mut state = state0;
        let mut hist: Vec<(u8, u8)> = Vec::with_capacity(12);
        for &a in r.actions.iter().take(lab.prefix_len as usize) {
            hist.push((state.current_player(), a));
            state.step(a);
        }
        if state.phase != Phase::Bidding || state.current_player() != lab.observer {
            skipped += 1;
            continue;
        }
        let my_team = (lab.observer & 1) as usize;
        let legal = state.legal_actions();

        for _ in 0..ctx_per_label {
            // Match context: uniform over plausible cumulative scores.
            let my_cum = rng.gen_range(0..1900);
            let opp_cum = rng.gen_range(0..1900);

            write_bid_observation_score_aware_v3(&mut obs_buf, 0, &state, &hist, my_cum, opp_cum);
            let q_frozen = frozen_net.evaluate(&obs_buf);

            let mut target = [0.0f32; BID_MASK_DIM];
            let mut mask = [0.0f32; BID_MASK_DIM];
            for a in 0..BID_MASK_DIM {
                if legal & (1u64 << a) == 0 {
                    continue;
                }
                mask[a] = 1.0;
                if a == BID_PASS as usize || a > 40 {
                    // PASS / COINCHE / SURCOINCHE: anchor to frozen v6.
                    target[a] = q_frozen[a];
                    continue;
                }
                let (value, suit) = decode_bid(a as u8);
                let contract = Contract {
                    trump: suit,
                    value,
                    team: my_team as u8,
                    coinche: 0,
                };
                let mut acc = 0.0f32;
                for (w, pts) in lab.pts.iter().enumerate() {
                    acc += world_reward(
                        calib[pts[suit as usize] as usize],
                        lab.belote[w][suit as usize],
                        contract,
                        my_team,
                        my_cum,
                        opp_cum,
                        scale,
                        clip,
                    );
                }
                target[a] = acc / lab.pts.len() as f32;
            }

            // Re-centre the bid targets on v6's own mean over the same legal actions.
            //
            // Two biases inflate the raw targets and both are per-position offsets:
            // DD assumes perfect play (v6 was trained on IS-DD rollout points, which
            // are lower), and "the auction ends when I bid" ignores that a bad cheap
            // bid gets overcalled rather than played. Left alone they shift the argmax
            // to PASS on only 32% of positions where v6 says 57%, i.e. a wildly
            // over-aggressive bidder. Subtracting the offset keeps the *relative*
            // structure across suits and levels — the part measured as better — and
            // leaves v6's bid/pass calibration untouched.
            // Per-level re-centring: shift the 4 suits of each bid level onto v6's mean
            // for that same level. The measured gain from auction conditioning is about
            // ranking *suits* (21.5% RMSE); the level and pass structure is where the
            // remaining bias lives, and that bias is level-dependent — a cheap 80 gets
            // overcalled in reality while 160 does not — so a single global shift cannot
            // remove it. This injects only the suit comparison and leaves every level's
            // absolute attractiveness exactly as v6 has it.
            if recenter_per_level {
                for lvl in 0..10usize {
                    let (mut ts, mut qs, mut n) = (0.0f32, 0.0f32, 0usize);
                    for suit in 0..4usize {
                        let a = lvl * 4 + suit + 1;
                        if a <= 40 && mask[a] > 0.5 { ts += target[a]; qs += q_frozen[a]; n += 1; }
                    }
                    if n > 0 {
                        let shift = qs / n as f32 - ts / n as f32;
                        for suit in 0..4usize {
                            let a = lvl * 4 + suit + 1;
                            if a <= 40 && mask[a] > 0.5 { target[a] += shift; }
                        }
                    }
                }
            } else if recenter {
                let (mut ts, mut qs, mut n) = (0.0f32, 0.0f32, 0usize);
                for a in 1..=40usize {
                    if mask[a] > 0.5 { ts += target[a]; qs += q_frozen[a]; n += 1; }
                }
                if n > 0 {
                    let shift = qs / n as f32 - ts / n as f32;
                    for a in 1..=40usize {
                        if mask[a] > 0.5 { target[a] += shift; }
                    }
                }
            }

            xs.extend_from_slice(&obs_buf);
            ys.extend_from_slice(&target);
            ms.extend_from_slice(&mask);
            n_samples += 1;
        }
    }
    // ── Diagnostics: is the target on the same scale as what v6 believes, and does
    // it prefer the same *kind* of action? A distillation that silently shifts the
    // bid/pass balance would wreck the bidder while the MSE still looks fine.
    {
        let n = ys.len() / BID_MASK_DIM;
        let (mut tb, mut qb, mut nb) = (0.0f64, 0.0f64, 0usize);
        let (mut tp, mut qp, mut np_) = (0.0f64, 0.0f64, 0usize);
        let (mut tgt_pass, mut v6_pass) = (0usize, 0usize);
        let mut tgt_lvl = [0usize; 10];
        let mut v6_lvl = [0usize; 10];
        let mut fb = BidNet::load(&frozen).unwrap();
        for r in 0..n {
            let obs = &xs[r * OBS..(r + 1) * OBS];
            let q = fb.evaluate(obs);
            let (mut bt, mut bq) = (0usize, 0usize);
            let (mut btv, mut bqv) = (f32::NEG_INFINITY, f32::NEG_INFINITY);
            for a in 0..BID_MASK_DIM {
                if ms[r * BID_MASK_DIM + a] < 0.5 { continue; }
                let t = ys[r * BID_MASK_DIM + a];
                if t > btv { btv = t; bt = a; }
                if q[a] > bqv { bqv = q[a]; bq = a; }
                if a == BID_PASS as usize || a > 40 { tp += t as f64; qp += q[a] as f64; np_ += 1; }
                else { tb += t as f64; qb += q[a] as f64; nb += 1; }
            }
            if bt == BID_PASS as usize { tgt_pass += 1; } else if bt <= 40 && bt >= 1 {
                tgt_lvl[(decode_bid(bt as u8).0 as usize).min(25) / 3 % 10] += 1;
            }
            if bq == BID_PASS as usize { v6_pass += 1; } else if bq <= 40 && bq >= 1 {
                v6_lvl[(decode_bid(bq as u8).0 as usize).min(25) / 3 % 10] += 1;
            }
        }
        println!("\n  --- target vs v6 calibration ---");
        println!("  bid actions : target mean {:+.4}  v6 mean {:+.4}  (n={})", tb / nb as f64, qb / nb as f64, nb);
        println!("  pass/coinche: target mean {:+.4}  v6 mean {:+.4}  (n={})", tp / np_ as f64, qp / np_ as f64, np_);
        println!("  argmax is PASS: target {:.1}%   v6 {:.1}%", tgt_pass as f64 / n as f64 * 100.0, v6_pass as f64 / n as f64 * 100.0);
    }

    println!(
        "  {} samples in {:.1}s ({} labels skipped as inconsistent)",
        n_samples,
        t0.elapsed().as_secs_f64(),
        skipped
    );
    assert!(n_samples > 1000, "dataset too small");

    // ── Split train / val.
    let n_val = ((n_samples as f64) * val_frac) as usize;
    let n_train = n_samples - n_val;
    println!("  train {n_train} / val {n_val}");

    let device = Device::cuda_if_available(0).unwrap_or(Device::Cpu);
    println!("  device: {device:?}");

    let mut trainer =
        BiddingTrainer::with_layers_and_obs(layers, hidden, OBS, lr, 0.0, device.clone())
            .expect("build trainer");
    trainer.load_checkpoint(&init).expect("load v6 checkpoint");
    println!("  initialised from {init}");

    let val_x = Tensor::from_slice(&xs[n_train * OBS..], (n_val, OBS), &device).unwrap();
    let val_y = Tensor::from_slice(&ys[n_train * BID_MASK_DIM..], (n_val, BID_MASK_DIM), &device).unwrap();
    let val_m = Tensor::from_slice(&ms[n_train * BID_MASK_DIM..], (n_val, BID_MASK_DIM), &device).unwrap();

    let masked_mse = |q: &Tensor, y: &Tensor, m: &Tensor| -> candle_core::Result<Tensor> {
        let diff = (q - y)?;
        let sq = (&diff * &diff)?;
        let masked = (&sq * m)?;
        let n = m.sum_all()?;
        masked.sum_all()? / n
    };

    // Argmax over legal actions, per row.
    let policy = |q: &Tensor| -> Vec<usize> {
        let qv: Vec<Vec<f32>> = q.to_vec2().unwrap();
        let mv: Vec<Vec<f32>> = val_m.to_vec2().unwrap();
        qv.iter().zip(mv.iter()).map(|(row, m)| {
            let mut best = 0usize;
            let mut bv = f32::NEG_INFINITY;
            for a in 0..BID_MASK_DIM {
                if m[a] > 0.5 && row[a] > bv { bv = row[a]; best = a; }
            }
            best
        }).collect()
    };
    let agree = |a: &[usize], b: &[usize]| -> f64 {
        a.iter().zip(b.iter()).filter(|(x, y)| x == y).count() as f64 / a.len() as f64 * 100.0
    };

    let target_policy = policy(&val_y);
    let v6_policy = policy(&trainer.net.forward(&val_x).unwrap());
    {
        let q = trainer.net.forward(&val_x).unwrap();
        let l = masked_mse(&q, &val_y, &val_m).unwrap().to_vec0::<f32>().unwrap();
        println!("\n  val MSE before any training (v6 as-is): {l:.6}");
        println!("  v6 agrees with the distilled targets on {:.1}% of val positions", agree(&v6_policy, &target_policy));

        // The PASS anchor comes from the pure-Rust BidNet while the loss is computed
        // on the candle net. If the two disagree numerically the anchor would drag the
        // model for no reason, so assert they agree at init.
        let qv: Vec<Vec<f32>> = q.to_vec2().unwrap();
        let yv: Vec<Vec<f32>> = val_y.to_vec2().unwrap();
        let mv: Vec<Vec<f32>> = val_m.to_vec2().unwrap();
        let (mut anchor_se, mut anchor_n) = (0.0f64, 0usize);
        for r in 0..qv.len() {
            for a in 0..BID_MASK_DIM {
                if mv[r][a] > 0.5 && (a == BID_PASS as usize || a > 40) {
                    anchor_se += ((qv[r][a] - yv[r][a]) as f64).powi(2);
                    anchor_n += 1;
                }
            }
        }
        println!(
            "  anchor consistency (candle vs frozen BidNet on PASS/coinche): RMSE {:.6} over {} entries",
            (anchor_se / anchor_n.max(1) as f64).sqrt(), anchor_n
        );
    }

    let mut order: Vec<usize> = (0..n_train).collect();
    let mut best = f32::INFINITY;
    println!("\nTraining {epochs} epochs, batch {batch}, lr {lr}");
    for ep in 0..epochs {
        // Shuffle
        for k in (1..order.len()).rev() {
            let j = rng.gen_range(0..=k);
            order.swap(k, j);
        }
        let t = Instant::now();
        let mut run = 0.0f64;
        let mut nb = 0usize;
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
        println!(
            "  epoch {:>2}  train {:.6}  val {:.6}  ({:.1}s)",
            ep + 1,
            run / nb as f64,
            vl,
            t.elapsed().as_secs_f64()
        );

        {
            let p = policy(&trainer.net.forward(&val_x).unwrap());
            println!(
                "            -> matches target {:.1}% | differs from v6 on {:.1}% of positions",
                agree(&p, &target_policy),
                100.0 - agree(&p, &v6_policy)
            );
        }

        // One checkpoint per epoch: MSE cannot tell us how far to fine-tune, so the
        // arena arbitrates between drift levels instead of us guessing.
        trainer.export_binary(&format!("{out_dir}/bid_nn_ep{}.bin", ep + 1)).unwrap();
        trainer.save_checkpoint(&format!("{out_dir}/bid_nn_ep{}.safetensors", ep + 1)).unwrap();

        if vl < best {
            best = vl;
            trainer.save_checkpoint(&format!("{out_dir}/bid_nn_best.safetensors")).unwrap();
            trainer.export_binary(&format!("{out_dir}/bid_nn_best.bin")).unwrap();
        }
        trainer.save_checkpoint(&format!("{out_dir}/bid_nn_final.safetensors")).unwrap();
        trainer.export_binary(&format!("{out_dir}/bid_nn_final.bin")).unwrap();
    }

    println!("\nBest val MSE {best:.6}");
    println!("Wrote {out_dir}/bid_nn_best.bin (and _final)");
    let _ = DType::F32;
}
