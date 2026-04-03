/// Pure Rust DMC training binary using Candle + Dueling DQN.
///
/// Features:
/// - VecEnv with auto-reset and diverse bidding strategies
/// - NN bidding (BidNet) with annealing from mixed to NN-dominant
/// - Dueling Q-network (Candle GPU) with ε-greedy exploration
/// - Prioritized Experience Replay (PER)
/// - Opponent diversity: self-play, pool (CPU DmcNet), random
/// - Inline evaluation: match play vs random, frozen checkpoint, IS-DD
/// - Checkpoint saving (safetensors + binary export)

use std::time::Instant;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use candle_core::Device;

use colver_core::bid_net::BidNet;
use colver_core::dmc_candle::{DuelingTrainer, PoolNet};
use colver_core::dmc_env::VecTrainingEnv;
use colver_core::dmc_eval::{self, EvalConfig, EvalResult};
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::OBS_DIM;
use colver_core::dmc_replay::PrioritizedReplayBuffer;
use colver_core::rollout;
use colver_core::state::{GameState, Phase};

const NUM_ACTIONS: usize = 32;

#[derive(Parser)]
#[command(name = "train_dmc", about = "Pure Rust DMC training with Candle + Dueling DQN")]
struct Args {
    #[arg(long, default_value_t = 256)]
    num_envs: usize,
    #[arg(long, default_value_t = 20_000_000)]
    steps: usize,
    #[arg(long, default_value_t = 1024)]
    batch_size: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 1024)]
    hidden: usize,
    #[arg(long, default_value_t = 0.25)]
    eps_start: f32,
    #[arg(long, default_value_t = 0.01)]
    eps_end: f32,
    #[arg(long, default_value_t = 8_000_000)]
    eps_decay_steps: usize,
    #[arg(long, default_value_t = 2_000_000)]
    buffer_size: usize,
    #[arg(long, default_value_t = 10_000)]
    min_buffer: usize,
    #[arg(long, default_value_t = 4)]
    train_freq: usize,
    #[arg(long, default_value_t = 1_000_000)]
    eval_freq: usize,
    #[arg(long, default_value_t = 100)]
    eval_random_matches: usize,
    #[arg(long, default_value_t = 30)]
    eval_isdd_matches: usize,
    #[arg(long, default_value_t = 20)]
    eval_isdd_time_ms: u32,
    #[arg(long, default_value_t = 500_000)]
    save_freq: usize,
    #[arg(long, default_value = "models")]
    save_dir: String,
    #[arg(long)]
    resume: Option<String>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0.6)]
    per_alpha: f64,
    #[arg(long, default_value_t = 0.4)]
    per_beta_start: f64,
    #[arg(long, default_value_t = 1.0)]
    per_beta_end: f64,
    #[arg(long, default_value_t = 0.2)]
    pool_frac: f32,
    #[arg(long, default_value_t = 0.1)]
    random_frac: f32,
    #[arg(long, default_value_t = 500_000)]
    pool_save_freq: usize,
    #[arg(long, default_value_t = 10)]
    pool_size: usize,
    /// Path to a frozen .bin model to evaluate against (e.g. DouDou35).
    #[arg(long)]
    eval_checkpoint: Option<String>,
    /// Number of matches to play against the frozen checkpoint.
    #[arg(long, default_value_t = 50)]
    eval_checkpoint_matches: usize,
    /// Offset added to step counter for logging and checkpoint filenames (for resumed runs).
    #[arg(long, default_value_t = 0)]
    step_offset: usize,
    /// Path to NN bid model (bid_nn_final.bin). Enables strategy 8 = NN bid.
    #[arg(long)]
    bid_model: Option<String>,
    /// NN bid fraction at start of training.
    #[arg(long, default_value_t = 0.75)]
    nn_bid_start: f32,
    /// NN bid fraction at end of annealing.
    #[arg(long, default_value_t = 0.95)]
    nn_bid_end: f32,
    /// Steps over which to anneal NN bid fraction.
    #[arg(long, default_value_t = 20_000_000)]
    nn_bid_anneal_steps: usize,
}

/// Opponent type per environment.
const OPP_SELF: u8 = 0;
const OPP_POOL: u8 = 1;
const OPP_RANDOM: u8 = 2;

/// Episode buffer: stores transitions for a single deal.
struct EpisodeTransition {
    obs: Vec<f32>,   // OBS_DIM floats
    mask: Vec<f32>,  // 32 floats
    action: u8,
    team: u8,        // 0=NS, 1=EW
}

/// Pick a bid strategy for one team at the current step, given NN bid annealing params.
fn pick_bid_strategy(
    rng: &mut StdRng,
    step: usize,
    has_bid_model: bool,
    nn_bid_start: f32,
    nn_bid_end: f32,
    nn_bid_anneal_steps: usize,
) -> u8 {
    if !has_bid_model {
        return rng.gen_range(0..8u8); // strategies 0-7 only
    }
    let progress = (step as f32 / nn_bid_anneal_steps as f32).min(1.0);
    let nn_frac = nn_bid_start + (nn_bid_end - nn_bid_start) * progress;
    let roll: f32 = rng.gen();
    if roll < nn_frac {
        8 // NN bid
    } else {
        // Distribute among heuristic bidders:
        // ~40% improved_v2 (0), ~30% heuristic (7), ~30% BidParams presets (1-6)
        let r: f32 = rng.gen();
        if r < 0.4 {
            0 // improved_v2
        } else if r < 0.7 {
            7 // heuristic
        } else {
            rng.gen_range(1..=6u8) // BidParams presets
        }
    }
}

/// Evaluate the Q-network vs various baselines using duplicate matching.
fn evaluate(
    trainer: &DuelingTrainer,
    hidden: usize,
    random_matches: usize,
    checkpoint_matches: usize,
    isdd_matches: usize,
    isdd_time_ms: u32,
    baseline_net: &mut Option<DmcNet>,
    eval_bid_net: &mut Option<BidNet>,
) -> EvalResult {
    // Export current weights for CPU inference
    let weights = match trainer.snapshot_weights() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to snapshot weights: {}", e);
            return EvalResult { rand_wr: 0.0, ckpt_wr: 0.0, isdd_wr: 0.0, elapsed: 0.0 };
        }
    };
    let mut q_net = match DmcNet::from_floats(&weights, hidden, OBS_DIM, true) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to load eval net: {}", e);
            return EvalResult { rand_wr: 0.0, ckpt_wr: 0.0, isdd_wr: 0.0, elapsed: 0.0 };
        }
    };

    let config = EvalConfig {
        random_matches,
        checkpoint_matches,
        isdd_matches,
        isdd_time_ms,
    };

    dmc_eval::run_eval(&mut q_net, baseline_net, eval_bid_net, &config)
}

fn main() {
    let args = Args::parse();

    // Select device
    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).expect("CUDA device creation failed")
    } else {
        eprintln!("WARNING: CUDA not available, using CPU (training will be slow)");
        Device::Cpu
    };
    let device_name = match &device {
        Device::Cpu => "CPU".to_string(),
        Device::Cuda(_) => "CUDA".to_string(),
        _ => "Other".to_string(),
    };
    println!("Device: {}", device_name);
    println!("Envs: {}, Steps: {}, LR: {}, Hidden: {}", args.num_envs, args.steps, args.lr, args.hidden);
    println!("Training: batch={}, freq={}", args.batch_size, args.train_freq);
    println!("PER: alpha={}, beta={}->{}", args.per_alpha, args.per_beta_start, args.per_beta_end);
    println!("Opponents: {:.0}% self, {:.0}% pool, {:.0}% random",
        (1.0 - args.pool_frac - args.random_frac) * 100.0,
        args.pool_frac * 100.0,
        args.random_frac * 100.0);

    // Load bid model
    let has_bid_model = if let Some(ref path) = args.bid_model {
        println!("Bid model: {}", path);
        true
    } else {
        println!("Bid model: none (using improved_v2 for all bidding)");
        false
    };
    if has_bid_model {
        println!("Bid annealing: {:.0}% -> {:.0}% NN bid over {}M steps",
            args.nn_bid_start * 100.0, args.nn_bid_end * 100.0,
            args.nn_bid_anneal_steps / 1_000_000);
    }

    // Print eval config
    let mut eval_parts = Vec::new();
    if args.eval_random_matches > 0 {
        eval_parts.push(format!("{} rand", args.eval_random_matches));
    }
    if args.eval_checkpoint.is_some() && args.eval_checkpoint_matches > 0 {
        eval_parts.push(format!("{} ckpt", args.eval_checkpoint_matches));
    }
    if args.eval_isdd_matches > 0 {
        eval_parts.push(format!("{} isdd ({}ms/move)", args.eval_isdd_matches, args.eval_isdd_time_ms));
    }
    println!("Eval (every {}): {}", args.eval_freq, eval_parts.join(" + "));

    if args.step_offset > 0 {
        println!("Step offset: {} (steps will display as {}..{})",
            args.step_offset, args.step_offset + 1, args.step_offset + args.steps);
    }

    // Initialize trainer
    let mut trainer = DuelingTrainer::new(args.hidden, args.lr, 0.0, device)
        .expect("Failed to create trainer");

    if let Some(ref path) = args.resume {
        trainer.load_checkpoint(path).expect("Failed to load checkpoint");
        println!("Resumed from {}", path);
    }

    // Load frozen checkpoint for evaluation baseline
    let mut eval_baseline: Option<DmcNet> = args.eval_checkpoint.as_ref().map(|path| {
        let net = DmcNet::load(path).unwrap_or_else(|e| panic!("Failed to load eval checkpoint {}: {}", path, e));
        println!("Eval baseline: {} (obs_dim={}, dueling={})", path, net.obs_dim(), net.is_dueling());
        net
    });

    // Load bid model for eval (separate instance to avoid borrow conflicts with training env)
    let mut eval_bid_net: Option<BidNet> = args.bid_model.as_ref().and_then(|path| {
        BidNet::load(path).ok()
    });

    // Initialize replay buffer
    let mut replay_buffer = PrioritizedReplayBuffer::new(args.buffer_size, args.per_alpha);

    // Initialize environments
    let mut vec_env = VecTrainingEnv::new_with_seed(args.num_envs, args.seed);
    let mut rng = StdRng::seed_from_u64(args.seed);

    // Load bid model into VecTrainingEnv for training bidding
    if let Some(ref path) = args.bid_model {
        if let Err(e) = vec_env.load_bid_model(path) {
            eprintln!("WARNING: Failed to load bid model for training: {}", e);
        }
    }

    // Initialize per-team bid strategies
    let mut bid_strategies: Vec<(u8, u8)> = (0..args.num_envs)
        .map(|_| {
            let ns = pick_bid_strategy(&mut rng, 0, has_bid_model, args.nn_bid_start, args.nn_bid_end, args.nn_bid_anneal_steps);
            let ew = pick_bid_strategy(&mut rng, 0, has_bid_model, args.nn_bid_start, args.nn_bid_end, args.nn_bid_anneal_steps);
            (ns, ew)
        })
        .collect();
    vec_env.set_bid_strategies_per_team(&bid_strategies);

    // Opponent tracking
    let mut opp_type = vec![OPP_SELF; args.num_envs];
    let mut opp_team = vec![0u8; args.num_envs];

    // Opponent pool (GPU PoolNet for batched inference)
    let mut pool_weights: Vec<Vec<f32>> = Vec::new();
    let mut pool_net: Option<PoolNet> = None;

    // Episode buffers
    let mut episode_bufs: Vec<Vec<EpisodeTransition>> = (0..args.num_envs)
        .map(|_| Vec::with_capacity(32))
        .collect();

    // Stats
    let mut _total_transitions = 0usize;
    let mut total_episodes = 0usize;
    let mut total_loss = 0.0f64;
    let mut loss_count = 0usize;
    let step_start = Instant::now();
    let mut last_log_time = Instant::now();
    let mut last_log_step = 0usize;

    println!("\n{:>10} | {:>5} | {:>5} | {:>7} | {:>8} | {:>8} | {:>7}",
        "Step", "Eps", "Beta", "Buffer", "Loss", "Episodes", "Steps/s");
    println!("{}", "-".repeat(72));

    for step in 0..args.steps {
        // Epsilon schedule
        let progress = ((step as f64) / args.eps_decay_steps as f64).min(1.0);
        let eps = args.eps_start + (args.eps_end - args.eps_start) * progress as f32;

        // Beta schedule for PER
        let beta_progress = ((step as f64) / args.steps as f64).min(1.0);
        let beta = args.per_beta_start + (args.per_beta_end - args.per_beta_start) * beta_progress;

        let n = args.num_envs;
        let phases = vec_env.phases();
        let players = vec_env.current_players();

        let mut actions = vec![0u8; n];

        // --- Bidding envs: use bid strategies ---
        let bid_actions = vec_env.bid_actions();
        for i in 0..n {
            if phases[i] == Phase::Bidding {
                actions[i] = bid_actions[i];
            }
        }

        // --- Playing envs: decide per opponent type ---
        let mut play_indices = Vec::new();
        for i in 0..n {
            if phases[i] == Phase::Playing {
                play_indices.push(i);
            }
        }

        if !play_indices.is_empty() {
            // Separate envs by: self-play (Q-net), pool (DmcNet), random
            let mut self_indices = Vec::new();
            let mut pool_indices = Vec::new();
            let mut random_indices = Vec::new();

            for &i in &play_indices {
                let team = GameState::player_team(players[i]);
                if opp_type[i] != OPP_SELF && team == opp_team[i] {
                    // Opponent's turn
                    match opp_type[i] {
                        OPP_POOL => pool_indices.push(i),
                        OPP_RANDOM => random_indices.push(i),
                        _ => self_indices.push(i),
                    }
                } else {
                    self_indices.push(i);
                }
            }

            // Self-play: GPU batch Q-network + ε-greedy
            if !self_indices.is_empty() {
                let batch = self_indices.len();
                let mut obs_flat = vec![0.0f32; batch * OBS_DIM];
                let mut mask_flat = vec![0.0f32; batch * NUM_ACTIONS];
                for (j, &i) in self_indices.iter().enumerate() {
                    obs_flat[j * OBS_DIM..(j + 1) * OBS_DIM]
                        .copy_from_slice(vec_env.obs_slice(i));
                    mask_flat[j * NUM_ACTIONS..(j + 1) * NUM_ACTIONS]
                        .copy_from_slice(vec_env.mask_slice(i));
                }

                // GPU forward pass for batch
                let q_actions = match trainer.net.act(
                    &candle_core::Tensor::from_slice(&obs_flat, (batch, OBS_DIM), trainer.device()).unwrap(),
                    &candle_core::Tensor::from_slice(&mask_flat, (batch, NUM_ACTIONS), trainer.device()).unwrap(),
                    eps,
                    &mut rng,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("GPU forward failed: {}", e);
                        // Fallback to random
                        self_indices.iter().map(|&i| {
                            let mask = vec_env.envs[i].state.legal_actions();
                            let count = mask.count_ones();
                            let idx = rng.gen_range(0..count);
                            rollout::select_nth_bit(mask, idx)
                        }).collect()
                    }
                };

                for (j, &i) in self_indices.iter().enumerate() {
                    actions[i] = q_actions[j];
                }
            }

            // Pool opponent: GPU batched greedy
            if !pool_indices.is_empty() {
                if let Some(ref pnet) = pool_net {
                    let pbatch = pool_indices.len();
                    let mut pool_obs = vec![0.0f32; pbatch * OBS_DIM];
                    let mut pool_mask = vec![0.0f32; pbatch * NUM_ACTIONS];
                    for (j, &i) in pool_indices.iter().enumerate() {
                        pool_obs[j * OBS_DIM..(j + 1) * OBS_DIM]
                            .copy_from_slice(vec_env.obs_slice(i));
                        pool_mask[j * NUM_ACTIONS..(j + 1) * NUM_ACTIONS]
                            .copy_from_slice(vec_env.mask_slice(i));
                    }
                    let pool_obs_t = candle_core::Tensor::from_slice(&pool_obs, (pbatch, OBS_DIM), trainer.device()).unwrap();
                    let pool_mask_t = candle_core::Tensor::from_slice(&pool_mask, (pbatch, NUM_ACTIONS), trainer.device()).unwrap();
                    match pnet.act_greedy(&pool_obs_t, &pool_mask_t) {
                        Ok(pa) => {
                            for (j, &i) in pool_indices.iter().enumerate() {
                                actions[i] = pa[j];
                            }
                        }
                        Err(_) => {
                            for &i in &pool_indices {
                                actions[i] = vec_env.random_action(i);
                            }
                        }
                    }
                } else {
                    for &i in &pool_indices {
                        actions[i] = vec_env.random_action(i);
                    }
                }
            }

            // Random opponent
            for &i in &random_indices {
                actions[i] = vec_env.random_action(i);
            }

            // Store transitions for play-phase envs (only non-forced moves)
            for &i in &play_indices {
                let mask = vec_env.mask_slice(i);
                let n_legal: usize = mask.iter().filter(|&&v| v > 0.5).count();
                if n_legal > 1 {
                    let team = GameState::player_team(players[i]);
                    episode_bufs[i].push(EpisodeTransition {
                        obs: vec_env.obs_slice(i).to_vec(),
                        mask: mask.to_vec(),
                        action: actions[i],
                        team,
                    });
                }
            }
        }

        // --- Step all envs ---
        let (dones, outcomes) = vec_env.step_all(&actions);

        // --- Flush completed episodes ---
        for i in 0..n {
            if dones[i] {
                let buf = std::mem::replace(&mut episode_bufs[i], Vec::with_capacity(32));
                if !buf.is_empty() {
                    let (ns_outcome, ew_outcome) = outcomes[i];
                    for trans in &buf {
                        let ret = if trans.team == 0 { ns_outcome } else { ew_outcome };
                        replay_buffer.push(&trans.obs, &trans.mask, trans.action, ret);
                    }
                    _total_transitions += buf.len();
                }
                total_episodes += 1;

                // Re-randomize bid strategy per team with annealing
                let ns_strat = pick_bid_strategy(&mut rng, step, has_bid_model, args.nn_bid_start, args.nn_bid_end, args.nn_bid_anneal_steps);
                let ew_strat = pick_bid_strategy(&mut rng, step, has_bid_model, args.nn_bid_start, args.nn_bid_end, args.nn_bid_anneal_steps);
                bid_strategies[i] = (ns_strat, ew_strat);

                // Assign opponent type for new deal
                let roll: f32 = rng.gen();
                if roll < args.random_frac {
                    opp_type[i] = OPP_RANDOM;
                    opp_team[i] = rng.gen_range(0..2);
                } else if roll < args.random_frac + args.pool_frac && pool_net.is_some() {
                    opp_type[i] = OPP_POOL;
                    opp_team[i] = rng.gen_range(0..2);
                } else {
                    opp_type[i] = OPP_SELF;
                }
            }
        }
        if dones.iter().any(|&d| d) {
            vec_env.set_bid_strategies_per_team(&bid_strategies);
        }

        // --- Train ---
        if replay_buffer.size() >= args.min_buffer && step % args.train_freq == 0 {
            let sample = replay_buffer.sample(args.batch_size, beta, &mut rng);

            match trainer.train_step(
                &sample.obs_data,
                &sample.mask_data,
                &sample.actions,
                &sample.returns,
                &sample.weights,
            ) {
                Ok((loss, td_errors)) => {
                    replay_buffer.update_priorities(&sample.indices, &td_errors);
                    total_loss += loss as f64;
                    loss_count += 1;
                }
                Err(e) => {
                    eprintln!("Training step failed: {}", e);
                }
            }
        }

        // --- Save to opponent pool ---
        if (step + 1) % args.pool_save_freq == 0 && args.pool_frac > 0.0 {
            if let Ok(weights) = trainer.snapshot_weights() {
                pool_weights.push(weights);
                if pool_weights.len() > args.pool_size {
                    pool_weights.remove(0);
                }
                // Load a random pool model onto GPU for batched inference
                let idx = rng.gen_range(0..pool_weights.len());
                match &pool_net {
                    Some(pnet) => {
                        pnet.load_weights(&pool_weights[idx]).ok();
                    }
                    None => {
                        if let Ok(pnet) = PoolNet::new(args.hidden, trainer.device()) {
                            pnet.load_weights(&pool_weights[idx]).ok();
                            pool_net = Some(pnet);
                        }
                    }
                }
                println!("  [POOL] Saved model to pool (size: {})", pool_weights.len());
            }
        }

        // --- Logging ---
        if (step + 1) % 10_000 == 0 {
            let elapsed = last_log_time.elapsed().as_secs_f64();
            let steps_done = step + 1 - last_log_step;
            let sps = steps_done as f64 / elapsed.max(1e-6);
            let avg_loss = if loss_count > 0 { total_loss / loss_count as f64 } else { 0.0 };
            println!(
                "{:>10} | {:>5.3} | {:>5.3} | {:>7} | {:>8.4} | {:>8} | {:>7.0}",
                step + 1 + args.step_offset, eps, beta, replay_buffer.size(), avg_loss, total_episodes, sps
            );
            total_loss = 0.0;
            loss_count = 0;
            last_log_time = Instant::now();
            last_log_step = step + 1;
        }

        // --- Evaluate ---
        if (step + 1) % args.eval_freq == 0 {
            let er = evaluate(
                &trainer, args.hidden,
                args.eval_random_matches,
                args.eval_checkpoint_matches,
                args.eval_isdd_matches,
                args.eval_isdd_time_ms,
                &mut eval_baseline,
                &mut eval_bid_net,
            );
            let mut parts = Vec::new();
            if args.eval_random_matches > 0 {
                parts.push(format!("rand {:.0}%", er.rand_wr * 100.0));
            }
            if eval_baseline.is_some() && args.eval_checkpoint_matches > 0 {
                parts.push(format!("ckpt {:.0}%", er.ckpt_wr * 100.0));
            }
            if args.eval_isdd_matches > 0 {
                parts.push(format!("isdd {:.0}%", er.isdd_wr * 100.0));
            }
            // Log current NN bid fraction
            if has_bid_model {
                let nn_progress = (step as f32 / args.nn_bid_anneal_steps as f32).min(1.0);
                let nn_frac = args.nn_bid_start + (args.nn_bid_end - args.nn_bid_start) * nn_progress;
                parts.push(format!("nn_bid {:.0}%", nn_frac * 100.0));
            }
            println!("  [EVAL] {} ({:.0}s)", parts.join(" | "), er.elapsed);
        }

        // --- Save checkpoint ---
        if (step + 1) % args.save_freq == 0 {
            std::fs::create_dir_all(&args.save_dir).ok();
            let st_path = format!("{}/dmc_{}.safetensors", args.save_dir, step + 1 + args.step_offset);
            let bin_path = format!("{}/dmc_{}.bin", args.save_dir, step + 1 + args.step_offset);
            if let Err(e) = trainer.save_checkpoint(&st_path) {
                eprintln!("Failed to save safetensors: {}", e);
            }
            if let Err(e) = trainer.export_binary(&bin_path) {
                eprintln!("Failed to export binary: {}", e);
            }
            // Also save as latest
            let latest_st = format!("{}/dmc_latest.safetensors", args.save_dir);
            let latest_bin = format!("{}/dmc_latest.bin", args.save_dir);
            trainer.save_checkpoint(&latest_st).ok();
            trainer.export_binary(&latest_bin).ok();
            println!("  [SAVE] {}", st_path);
        }
    }

    // Final eval and save
    println!("\n--- Final Evaluation ---");
    let er = evaluate(
        &trainer, args.hidden,
        args.eval_random_matches,
        args.eval_checkpoint_matches,
        args.eval_isdd_matches,
        args.eval_isdd_time_ms,
        &mut eval_baseline,
        &mut eval_bid_net,
    );
    if args.eval_random_matches > 0 {
        println!("Matches vs random: {:.1}%", er.rand_wr * 100.0);
    }
    if eval_baseline.is_some() && args.eval_checkpoint_matches > 0 {
        println!("Matches vs checkpoint: {:.1}%", er.ckpt_wr * 100.0);
    }
    if args.eval_isdd_matches > 0 {
        println!("Matches vs IS-DD: {:.1}%", er.isdd_wr * 100.0);
    }
    println!("Eval time: {:.0}s", er.elapsed);
    println!("Total training time: {:.0}s", step_start.elapsed().as_secs_f64());

    std::fs::create_dir_all(&args.save_dir).ok();
    let final_st = format!("{}/dmc_final.safetensors", args.save_dir);
    let final_bin = format!("{}/dmc_final_dueling.bin", args.save_dir);
    trainer.save_checkpoint(&final_st).ok();
    trainer.export_binary(&final_bin).ok();
    println!("Saved final model to {} and {}", final_st, final_bin);
}
