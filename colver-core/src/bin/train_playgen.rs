//! Offline supervised training of the playgen causal transformer.
//!
//! Teacher forcing on pre-generated self-play games (COLVGM01 replays):
//! predict every played card given the observer-visible prefix, masked to the
//! observer-visible legal set. Pure supervised CE — no env in the loop.
//!
//! Usage:
//!   cargo run -p colver-core --bin train_playgen --features dmc_train --release -- \
//!     --games data/training/playgen_games_1M.bin \
//!     --steps 60000 --batch-size 512 --d-model 256 --layers 4 --heads 8 \
//!     --save-dir models/playgen

use std::time::Instant;

use candle_core::Device;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::game_replay::GameReplay;
use colver_core::playgen::model::{PlaygenBatch, PlaygenTrainer};
use colver_core::playgen::tokens::{
    canonical_trump_perm, random_trump_perm, tokenize_replay, PlaygenSample, NUM_CARD_ACTIONS,
};
use colver_core::state::{GameState, Phase};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut games_path = String::from("data/training/playgen_games.bin");
    let mut steps: usize = 60_000;
    let mut batch_size: usize = 512;
    let mut d_model: usize = 256;
    let mut n_layers: usize = 4;
    let mut n_heads: usize = 8;
    let mut lr: f64 = 3e-4;
    let mut warmup: usize = 1_000;
    let mut weight_decay: f64 = 0.01;
    let mut save_dir = String::from("models/playgen");
    let mut save_freq: usize = 10_000;
    let mut eval_freq: usize = 2_000;
    let mut log_freq: usize = 100;
    let mut val_games: usize = 2_000;
    let mut eval_batches: usize = 0; // 0 = full val set
    let mut seed: u64 = 42;
    let mut resume: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => { games_path = args[i + 1].clone(); i += 2; }
            "--steps" => { steps = args[i + 1].parse().unwrap(); i += 2; }
            "--batch-size" => { batch_size = args[i + 1].parse().unwrap(); i += 2; }
            "--d-model" => { d_model = args[i + 1].parse().unwrap(); i += 2; }
            "--layers" => { n_layers = args[i + 1].parse().unwrap(); i += 2; }
            "--heads" => { n_heads = args[i + 1].parse().unwrap(); i += 2; }
            "--lr" => { lr = args[i + 1].parse().unwrap(); i += 2; }
            "--warmup" => { warmup = args[i + 1].parse().unwrap(); i += 2; }
            "--weight-decay" => { weight_decay = args[i + 1].parse().unwrap(); i += 2; }
            "--save-dir" => { save_dir = args[i + 1].clone(); i += 2; }
            "--save-freq" => { save_freq = args[i + 1].parse().unwrap(); i += 2; }
            "--eval-freq" => { eval_freq = args[i + 1].parse().unwrap(); i += 2; }
            "--log-freq" => { log_freq = args[i + 1].parse().unwrap(); i += 2; }
            "--val-games" => { val_games = args[i + 1].parse().unwrap(); i += 2; }
            "--eval-batches" => { eval_batches = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--resume" => { resume = Some(args[i + 1].clone()); i += 2; }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).expect("CUDA device creation failed")
    } else {
        Device::Cpu
    };
    println!("=== Playgen Training ===");
    println!("Device:     {}", match device { Device::Cpu => "CPU", _ => "CUDA" });
    println!("Games:      {}", games_path);
    println!("Model:      d={} L={} H={}", d_model, n_layers, n_heads);
    println!("Steps:      {} (batch {}, lr {}, warmup {})", steps, batch_size, lr, warmup);

    // ---- Load replays ----
    let t0 = Instant::now();
    let mut replays = GameReplay::load_all(&games_path).expect("failed to load games file");
    let total = replays.len();
    replays.retain(|r| r.actions.len() >= 36); // drop void deals
    println!(
        "Loaded {} games ({} playable, {} void) in {:.1}s",
        total, replays.len(), total - replays.len(), t0.elapsed().as_secs_f64(),
    );
    assert!(replays.len() > val_games + 1000, "not enough games");

    // Precompute final trump per replay (perm must map trump -> 0)
    let trumps: Vec<u8> = replays.iter().map(final_trump).collect();

    let n_train = replays.len() - val_games;
    println!("Train: {} games, val: {} games", n_train, val_games);

    std::fs::create_dir_all(&save_dir).ok();

    let mut trainer =
        PlaygenTrainer::new(d_model, n_layers, n_heads, lr, weight_decay, device.clone())
            .expect("trainer init failed");
    println!("Parameters: {:.2}M", trainer.num_params() as f64 / 1e6);

    if let Some(path) = resume {
        trainer.load_checkpoint(&path).expect("resume failed");
        println!("Resumed from {}", path);
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let start = Instant::now();
    let mut window_loss = 0.0f64;
    let mut window_acc = 0.0f64;
    let mut window_n = 0usize;
    let mut window_t = Instant::now();

    for step in 1..=steps {
        // LR warmup
        if step <= warmup {
            trainer.set_lr(lr * step as f64 / warmup as f64);
        }

        // ---- Build batch (CPU) ----
        let mut samples = Vec::with_capacity(batch_size);
        while samples.len() < batch_size {
            let gi = rng.gen_range(0..n_train);
            let observer = rng.gen_range(0..4u8);
            let perm = random_trump_perm(trumps[gi], &mut rng);
            if let Some(s) = tokenize_replay(&replays[gi], observer, &perm) {
                samples.push(s);
            }
        }
        let batch = build_batch(&samples);

        let (loss, acc, n) = trainer.train_step(&batch).expect("train step failed");
        window_loss += loss as f64 * n as f64;
        window_acc += acc as f64 * n as f64;
        window_n += n;

        if step % log_freq == 0 {
            let dt = window_t.elapsed().as_secs_f64();
            println!(
                "[{:>7}] loss {:.4}  acc {:.3}  {:.1} steps/s  ({:.0} preds/s)  elapsed {:.0}s",
                step,
                window_loss / window_n as f64,
                window_acc / window_n as f64,
                log_freq as f64 / dt,
                window_n as f64 / dt,
                start.elapsed().as_secs_f64(),
            );
            window_loss = 0.0;
            window_acc = 0.0;
            window_n = 0;
            window_t = Instant::now();
        }

        if step % eval_freq == 0 || step == steps {
            run_eval(&trainer, &replays[n_train..], &trumps[n_train..], batch_size, eval_batches);
            window_t = Instant::now();
        }

        if step % save_freq == 0 || step == steps {
            let path = format!("{}/playgen_{}.safetensors", save_dir, step);
            trainer.save_checkpoint(&path).expect("checkpoint save failed");
            let final_path = format!("{}/playgen_latest.safetensors", save_dir);
            trainer.save_checkpoint(&final_path).ok();
            println!("Saved {}", path);
        }
    }

    let path = format!("{}/playgen_final.safetensors", save_dir);
    trainer.save_checkpoint(&path).expect("final save failed");
    println!("Done in {:.0}s. Final checkpoint: {}", start.elapsed().as_secs_f64(), path);
}

fn final_trump(replay: &GameReplay) -> u8 {
    let mut state = GameState::new(replay.dealer, replay.hands);
    for &a in &replay.actions {
        if state.phase == Phase::Playing {
            break;
        }
        state.step(a);
    }
    state.contract.trump
}

fn build_batch(samples: &[PlaygenSample]) -> PlaygenBatch {
    let b = samples.len();
    let l = samples.iter().map(|s| s.len()).max().unwrap_or(1);

    let mut primary = vec![0i64; b * l];
    let mut suit = vec![4i64; b * l]; // S_NULL
    let mut actor = vec![4i64; b * l]; // A_NULL
    let mut segment = vec![0i64; b * l];
    let n_preds: usize = samples.iter().map(|s| s.targets.len()).sum();
    let mut pred_idx = Vec::with_capacity(n_preds);
    let mut targets = Vec::with_capacity(n_preds);
    let mut mask = vec![0.0f32; n_preds * NUM_CARD_ACTIONS];

    let mut p = 0usize;
    for (bi, s) in samples.iter().enumerate() {
        let base = bi * l;
        for j in 0..s.len() {
            primary[base + j] = s.primary[j] as i64;
            suit[base + j] = s.suit[j] as i64;
            actor[base + j] = s.actor[j] as i64;
            segment[base + j] = s.segment[j] as i64;
        }
        for k in 0..s.targets.len() {
            pred_idx.push((base + s.pred_pos[k] as usize) as u32);
            targets.push(s.targets[k] as u32);
            let m = s.masks[k];
            for c in 0..NUM_CARD_ACTIONS {
                if m & (1 << c) != 0 {
                    mask[p * NUM_CARD_ACTIONS + c] = 1.0;
                }
            }
            p += 1;
        }
    }

    PlaygenBatch {
        primary,
        suit,
        actor,
        segment,
        batch_size: b,
        seq_len: l,
        pred_idx,
        targets,
        mask,
    }
}

fn run_eval(
    trainer: &PlaygenTrainer,
    val_replays: &[GameReplay],
    val_trumps: &[u8],
    batch_size: usize,
    eval_batches: usize,
) {
    let t0 = Instant::now();
    let mut loss_sum = 0.0f64;
    let mut correct = 0usize;
    let mut n = 0usize;
    let mut hidden_loss = 0.0f64;
    let mut hidden_n = 0usize;
    let mut trick_loss = [0.0f64; 8];
    let mut trick_n = [0usize; 8];

    let mut batch_count = 0;
    for (chunk_idx, chunk) in val_replays.chunks(batch_size).enumerate() {
        if eval_batches > 0 && batch_count >= eval_batches {
            break;
        }
        let mut samples = Vec::with_capacity(chunk.len());
        for (j, r) in chunk.iter().enumerate() {
            let gi = chunk_idx * batch_size + j;
            let observer = (gi % 4) as u8;
            let perm = canonical_trump_perm(val_trumps[gi]);
            if let Some(s) = tokenize_replay(r, observer, &perm) {
                samples.push(s);
            }
        }
        if samples.is_empty() {
            continue;
        }
        let batch = build_batch(&samples);
        let stats = trainer.eval_step(&batch).expect("eval step failed");
        loss_sum += stats.loss_sum;
        correct += stats.correct;
        n += stats.n;

        // Breakdowns
        let mut p = 0usize;
        for s in &samples {
            for k in 0..s.targets.len() {
                let nll = stats.nll[p] as f64;
                if s.hidden_actor[k] {
                    hidden_loss += nll;
                    hidden_n += 1;
                }
                let t = s.trick_idx[k] as usize;
                trick_loss[t] += nll;
                trick_n[t] += 1;
                p += 1;
            }
        }
        batch_count += 1;
    }

    let per_trick: Vec<String> = (0..8)
        .map(|t| {
            if trick_n[t] > 0 {
                format!("{:.2}", trick_loss[t] / trick_n[t] as f64)
            } else {
                "-".into()
            }
        })
        .collect();

    println!(
        "  [eval] loss {:.4}  acc {:.3}  hidden-loss {:.4}  per-trick [{}]  ({} preds, {:.1}s)",
        loss_sum / n as f64,
        correct as f64 / n as f64,
        hidden_loss / hidden_n.max(1) as f64,
        per_trick.join(" "),
        n,
        t0.elapsed().as_secs_f64(),
    );
}
