/// Bumblebid training with tch-rs (PyTorch backend) for faster GPU.
///
/// Same training loop as train_bid_nn --transformer but using tch-rs
/// instead of candle for ~3-4x faster GPU training.
///
/// Usage:
///   LIBTORCH_USE_PYTORCH=1 cargo run -p colver-core --bin train_bid_tch --features tch_train --release -- \
///     --d-model 256 --layers 2 --n-heads 8 --steps 5000000

use std::time::Instant;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid::bumblebid_tch::BumblebidTchTrainer;
use colver_core::bid_eval;
use colver_core::bid_obs::BID_OBS_DIM;
use colver_core::suit_perm;
use colver_core::bid_train_env::{BidReplayBuffer, DealPool, RewardMode, VecBidEnv};
use colver_core::rollout;
use colver_core::state::GameState;

const NUM_ACTIONS: usize = 43;

/// Opponent bidding strategy.
#[derive(Clone, Copy, PartialEq, Eq)]
enum OpponentMode { SelfPlay, ImprovedV2, Aggressive, Conservative, Random }

const OPP_WEIGHT_IMPROVED: f32 = 8.0;
const OPP_WEIGHT_AGGRESSIVE: f32 = 8.0;
const OPP_WEIGHT_CONSERVATIVE: f32 = 8.0;
const OPP_WEIGHT_RANDOM: f32 = 16.0;

fn pick_opponent_mode(step: usize, total: usize, div_start: f32, div_end: f32, rng: &mut impl Rng) -> OpponentMode {
    let progress = (step as f32 / total as f32).min(1.0);
    let div_ratio = div_start + (div_end - div_start) * progress;
    if rng.gen::<f32>() >= div_ratio { return OpponentMode::SelfPlay; }
    let total_w = OPP_WEIGHT_IMPROVED + OPP_WEIGHT_AGGRESSIVE + OPP_WEIGHT_CONSERVATIVE + OPP_WEIGHT_RANDOM;
    let r = rng.gen::<f32>() * total_w;
    if r < OPP_WEIGHT_IMPROVED { OpponentMode::ImprovedV2 }
    else if r < OPP_WEIGHT_IMPROVED + OPP_WEIGHT_AGGRESSIVE { OpponentMode::Aggressive }
    else if r < OPP_WEIGHT_IMPROVED + OPP_WEIGHT_AGGRESSIVE + OPP_WEIGHT_CONSERVATIVE { OpponentMode::Conservative }
    else { OpponentMode::Random }
}

fn opponent_action(state: &GameState, mode: OpponentMode, rng: &mut impl Rng) -> u8 {
    match mode {
        OpponentMode::SelfPlay => unreachable!(),
        OpponentMode::ImprovedV2 => bid_eval::improved_v2_bid(state),
        OpponentMode::Aggressive => bid_eval::parametric_bid(state, &bid_eval::BidParams::very_aggressive()),
        OpponentMode::Conservative => bid_eval::parametric_bid(state, &bid_eval::BidParams::ultra_conservative()),
        OpponentMode::Random => {
            let mask = state.legal_actions();
            let count = mask.count_ones();
            rollout::select_nth_bit(mask, rng.gen_range(0..count))
        }
    }
}

#[derive(Parser)]
#[command(name = "train_bid_tch", about = "Bumblebid training with tch-rs")]
struct Args {
    #[arg(long, default_value_t = 64)] num_envs: usize,
    #[arg(long, default_value_t = 5_000_000)] steps: usize,
    #[arg(long, default_value_t = 256)] batch_size: usize,
    #[arg(long, default_value_t = 3e-4)] lr: f64,
    #[arg(long, default_value_t = 256)] d_model: i64,
    #[arg(long, default_value_t = 2)] layers: usize,
    #[arg(long, default_value_t = 8)] n_heads: i64,
    #[arg(long, default_value_t = 0.3)] eps_start: f32,
    #[arg(long, default_value_t = 0.02)] eps_end: f32,
    #[arg(long, default_value_t = 3_000_000)] eps_decay_steps: usize,
    #[arg(long, default_value_t = 500_000)] buffer_size: usize,
    #[arg(long, default_value_t = 5_000)] min_buffer: usize,
    #[arg(long, default_value_t = 4)] train_freq: usize,
    #[arg(long, default_value_t = 1_000_000)] eval_freq: usize,
    #[arg(long, default_value_t = 1_000_000)] save_freq: usize,
    #[arg(long, default_value = "models/bumblebid")] save_dir: String,
    #[arg(long)] resume: Option<String>,
    #[arg(long, default_value_t = 42)] seed: u64,
    #[arg(long, default_value_t = 0.6)] per_alpha: f64,
    #[arg(long, default_value_t = 0.4)] per_beta_start: f64,
    #[arg(long, default_value_t = 1.0)] per_beta_end: f64,
    #[arg(long, default_value_t = 0.40)] diversity_start: f32,
    #[arg(long, default_value_t = 0.15)] diversity_end: f32,
    #[arg(long, default_value = "data/deals/archive/dd_5M_enriched.bin")] pool_file: String,
    #[arg(long, default_value = "dd")] reward: String,
}

fn main() {
    let args = Args::parse();

    let device = if tch::Cuda::is_available() {
        tch::Device::Cuda(0)
    } else {
        eprintln!("WARNING: CUDA not available, using CPU");
        tch::Device::Cpu
    };

    println!("=== Bumblebid Training (tch-rs) ===");
    println!("Device: {:?}", device);
    println!("Model: d={} L={} H={}", args.d_model, args.layers, args.n_heads);
    println!("Steps: {}, Batch: {}, Envs: {}", args.steps, args.batch_size, args.num_envs);
    println!("eps: {:.2}->{:.2} over {}", args.eps_start, args.eps_end, args.eps_decay_steps);

    let mut trainer = BumblebidTchTrainer::new(
        args.d_model, args.layers, args.n_heads, args.lr, 0.0, device,
    );
    let n_params: i64 = trainer.vs.trainable_variables().iter().map(|t| t.numel() as i64).sum();
    println!("Params: {}", n_params);

    if let Some(ref path) = args.resume {
        trainer.load_checkpoint(path);
        println!("Resumed from {}", path);
    }

    let mut replay_buffer = BidReplayBuffer::new(args.buffer_size, args.per_alpha);

    let reward_mode = if args.reward == "dd" {
        RewardMode::DdOnly
    } else if args.reward == "real" {
        RewardMode::RealOnly
    } else if args.reward.starts_with("blend:") {
        RewardMode::Blend(args.reward[6..].parse().expect("Bad blend alpha"))
    } else {
        panic!("Unknown reward mode '{}'", args.reward);
    };
    println!("Reward: {:?}", reward_mode);

    // Load deal pool
    let pool = if args.pool_file.contains("enriched") || matches!(reward_mode, RewardMode::RealOnly | RewardMode::Blend(_)) {
        match DealPool::load_enriched(&args.pool_file) {
            Ok(p) => p,
            Err(_) => DealPool::load_or_generate(&args.pool_file, 1_000_000, args.seed + 100),
        }
    } else {
        DealPool::load_or_generate(&args.pool_file, 1_000_000, args.seed + 100)
    };
    println!("Pool: {} deals", pool.len());

    let mut vec_env = VecBidEnv::new_with_pool_and_mode(args.num_envs, args.seed, &pool, reward_mode);
    let mut rng = StdRng::seed_from_u64(args.seed + 1);
    let n = args.num_envs;

    let mut opp_modes: Vec<OpponentMode> = (0..n)
        .map(|_| pick_opponent_mode(0, args.steps, args.diversity_start, args.diversity_end, &mut rng))
        .collect();
    let mut nn_teams: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2u8)).collect();

    let mut total_transitions = 0usize;
    let mut total_episodes = 0usize;
    let mut total_loss = 0.0f64;
    let mut loss_count = 0usize;
    let mut total_void_deals = 0usize;
    let step_start = Instant::now();
    let mut last_log_time = Instant::now();
    let mut last_log_step = 0usize;

    println!("\n{:>10} | {:>5} | {:>5} | {:>7} | {:>8} | {:>8} | {:>6} | {:>7}",
        "Step", "Eps", "Beta", "Buffer", "Loss", "Episodes", "Voids", "Steps/s");
    println!("{}", "-".repeat(80));

    for step in 0..args.steps {
        let progress = ((step as f64) / args.eps_decay_steps as f64).min(1.0);
        let eps = args.eps_start + (args.eps_end - args.eps_start) * progress as f32;
        let beta_progress = ((step as f64) / args.steps as f64).min(1.0);
        let beta = args.per_beta_start + (args.per_beta_end - args.per_beta_start) * beta_progress;

        // Collect actions
        let obs_flat = &vec_env.obs_buf;
        let mask_flat = &vec_env.mask_buf;

        let mut actions = trainer.net.act(obs_flat, mask_flat, n, eps, &mut rng);

        for i in 0..n {
            if opp_modes[i] != OpponentMode::SelfPlay {
                let player = vec_env.envs[i].state.current_player();
                let team = GameState::player_team(player);
                if team != nn_teams[i] {
                    actions[i] = opponent_action(&vec_env.envs[i].state, opp_modes[i], &mut rng);
                }
            }
        }

        // Step envs
        for i in 0..n {
            if let Some(transitions) = vec_env.step_env_pooled(i, actions[i], &pool) {
                let is_void = transitions.iter().all(|(_, _, _, r, _)| *r == 0.0);
                if is_void { total_void_deals += 1; }
                let is_self_play = opp_modes[i] == OpponentMode::SelfPlay;
                for (obs, mask, action, reward, team) in &transitions {
                    if is_self_play || *team == nn_teams[i] {
                        replay_buffer.push(obs, mask, *action, *reward);
                        total_transitions += 1;
                    }
                }
                total_episodes += 1;
                opp_modes[i] = pick_opponent_mode(step, args.steps, args.diversity_start, args.diversity_end, &mut rng);
                nn_teams[i] = rng.gen_range(0..2u8);
            }
        }

        // Train
        if replay_buffer.size() >= args.min_buffer && step % args.train_freq == 0 {
            let mut sample = replay_buffer.sample(args.batch_size, beta, &mut rng);
            suit_perm::augment_bid_batch(
                &mut sample.obs_data, &mut sample.mask_data, &mut sample.actions, &mut rng,
            );
            let (loss, td_errors) = trainer.train_step(
                &sample.obs_data, &sample.mask_data, &sample.actions,
                &sample.returns, &sample.weights,
            );
            replay_buffer.update_priorities(&sample.indices, &td_errors);
            total_loss += loss as f64;
            loss_count += 1;
        }

        // Log
        if (step + 1) % 1_000 == 0 {
            let elapsed = last_log_time.elapsed().as_secs_f64();
            let sps = (step + 1 - last_log_step) as f64 / elapsed.max(1e-6);
            let avg_loss = if loss_count > 0 { total_loss / loss_count as f64 } else { 0.0 };
            println!("{:>10} | {:>5.3} | {:>5.3} | {:>7} | {:>8.5} | {:>8} | {:>6} | {:>7.0}",
                step + 1, eps, beta, replay_buffer.size(), avg_loss,
                total_episodes, total_void_deals, sps);
            total_loss = 0.0;
            loss_count = 0;
            last_log_time = Instant::now();
            last_log_step = step + 1;
        }

        // Save
        if (step + 1) % args.save_freq == 0 {
            std::fs::create_dir_all(&args.save_dir).ok();
            let path = format!("{}/bb_tch_{}.pt", args.save_dir, step + 1);
            trainer.save_checkpoint(&path);
            println!("  [SAVE] {}", path);
        }
    }

    std::fs::create_dir_all(&args.save_dir).ok();
    let final_path = format!("{}/bb_tch_final.pt", args.save_dir);
    trainer.save_checkpoint(&final_path);
    println!("\nDone in {:.0}s. Episodes: {}, Transitions: {}, Voids: {}",
        step_start.elapsed().as_secs_f64(), total_episodes, total_transitions, total_void_deals);
}
