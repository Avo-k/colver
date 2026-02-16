/// Pure Rust DMC training binary using Candle + Dueling DQN.
///
/// Replicates the functionality of `scripts/train_dmc.py`:
/// - VecEnv with auto-reset and diverse bidding strategies
/// - Dueling Q-network (Candle GPU) with ε-greedy exploration
/// - Prioritized Experience Replay (PER)
/// - Opponent diversity: self-play, pool (CPU DmcNet), random
/// - Inline evaluation: deal win rate + match play vs random/naive/smart IS-MCTS
/// - Checkpoint saving (safetensors + binary export)

use std::time::Instant;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use candle_core::Device;

use colver_core::bid_eval;
use colver_core::dmc_candle::{DuelingTrainer, PoolNet};
use colver_core::dmc_env::VecTrainingEnv;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::OBS_DIM;
use colver_core::dmc_replay::PrioritizedReplayBuffer;
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};

const NUM_ACTIONS: usize = 32;
const NUM_BID_STRATEGIES: u8 = 8;

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
    #[arg(long, default_value_t = 100_000)]
    eval_freq: usize,
    #[arg(long, default_value_t = 100)]
    eval_random_matches: usize,
    #[arg(long, default_value_t = 10)]
    eval_naive_matches: usize,
    #[arg(long, default_value_t = 10)]
    eval_smart_matches: usize,
    #[arg(long, default_value_t = 20)]
    eval_time_ms: u32,
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

/// Evaluate the Q-network vs various baselines.
fn evaluate(
    trainer: &DuelingTrainer,
    hidden: usize,
    random_matches: usize,
    naive_matches: usize,
    smart_matches: usize,
    time_ms: u32,
) -> (f64, f64, f64, f64, f64) {
    let start = Instant::now();

    // Export current weights for CPU inference
    let weights = match trainer.snapshot_weights() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to snapshot weights: {}", e);
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }
    };
    let mut q_net = match DmcNet::from_floats(&weights, hidden, OBS_DIM, true) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to load eval net: {}", e);
            return (0.0, 0.0, 0.0, 0.0, 0.0);
        }
    };

    let mut rng = StdRng::seed_from_u64(12345);

    // 1. Deal-level vs random (both sides, 100 per side)
    let mut deal_wins = 0;
    for q_team in 0..2u8 {
        for _ in 0..100 {
            let won = play_deal_eval(&mut q_net, q_team, "random", 0, &mut rng);
            if won { deal_wins += 1; }
        }
    }
    let deal_wr = deal_wins as f64 / 200.0;

    // 2. Match play vs random
    let rand_wr = if random_matches > 0 {
        let mut wins = 0;
        let per_side = random_matches / 2;
        for q_team in 0..2u8 {
            for _ in 0..per_side {
                if play_match_eval(&mut q_net, q_team, "random", 0, &mut rng) {
                    wins += 1;
                }
            }
        }
        wins as f64 / random_matches as f64
    } else {
        0.0
    };

    // 3. Match play vs naive IS-MCTS
    let naive_wr = if naive_matches > 0 {
        let mut wins = 0;
        let per_side = naive_matches / 2;
        for q_team in 0..2u8 {
            for _ in 0..per_side {
                if play_match_eval(&mut q_net, q_team, "naive", time_ms, &mut rng) {
                    wins += 1;
                }
            }
        }
        wins as f64 / naive_matches as f64
    } else {
        0.0
    };

    // 4. Match play vs smart IS-MCTS
    let smart_wr = if smart_matches > 0 {
        let mut wins = 0;
        let per_side = smart_matches / 2;
        for q_team in 0..2u8 {
            for _ in 0..per_side {
                if play_match_eval(&mut q_net, q_team, "smart", time_ms, &mut rng) {
                    wins += 1;
                }
            }
        }
        wins as f64 / smart_matches as f64
    } else {
        0.0
    };

    let elapsed = start.elapsed().as_secs_f64();
    (deal_wr, rand_wr, naive_wr, smart_wr, elapsed)
}

/// Play a single deal for evaluation. Returns true if Q-team wins.
fn play_deal_eval(
    q_net: &mut DmcNet,
    q_team: u8,
    baseline: &str,
    time_ms: u32,
    rng: &mut StdRng,
) -> bool {
    let dealer = rng.gen_range(0..4u8);
    let mut state = GameState::deal_random(dealer, rng);
    let mut tracking = colver_core::dmc_obs::EnvTracking::new();
    tracking.dealer = dealer;

    let use_smart = baseline == "smart";
    let mut naive_search = NaiveIsMctsSearch::new();
    let mut smart_searches: Option<[SmartIsMctsSearch; 4]> = if use_smart {
        let mut s = [
            SmartIsMctsSearch::new(), SmartIsMctsSearch::new(),
            SmartIsMctsSearch::new(), SmartIsMctsSearch::new(),
        ];
        for (p, search) in s.iter_mut().enumerate() {
            search.init_deal(&state, p as u8, true);
        }
        Some(s)
    } else {
        None
    };

    while !state.is_terminal() {
        let player = state.current_player();
        let team = GameState::player_team(player);

        let action = if state.phase == Phase::Bidding {
            bid_eval::improved_bid(&state)
        } else if team == q_team {
            // Q-network action
            let obs = colver_core::dmc_obs::make_observation(&state, &tracking);
            let legal_mask = state.legal_actions() as u32;
            let (best, _) = q_net.best_action(&obs, legal_mask);
            best
        } else {
            // Baseline action
            match baseline {
                "random" => {
                    let mask = state.legal_actions();
                    let count = mask.count_ones();
                    let idx = rng.gen_range(0..count);
                    rollout::select_nth_bit(mask, idx)
                }
                "naive" => {
                    let config = NaiveIsMctsConfig {
                        time_limit_ms: Some(time_ms),
                        ..Default::default()
                    };
                    naive_search.search(&state, &config, rng)
                }
                "smart" => {
                    let config = SmartIsMctsConfig {
                        time_limit_ms: Some(time_ms),
                        ..Default::default()
                    };
                    let searches = smart_searches.as_mut().unwrap();
                    searches[player as usize].search(&state, &config, rng)
                }
                _ => unreachable!(),
            }
        };

        // Record action for Smart IS-MCTS beliefs
        if let Some(ref mut searches) = smart_searches {
            for search in searches.iter_mut() {
                search.record_action(&state, player, action);
            }
        }

        tracking.track_action(&state, action);
        state.step(action);
    }

    let rewards = state.rewards();
    rewards[q_team as usize] > rewards[1 - q_team as usize]
}

/// Play a match to 2000 for evaluation.
fn play_match_eval(
    q_net: &mut DmcNet,
    q_team: u8,
    baseline: &str,
    time_ms: u32,
    rng: &mut StdRng,
) -> bool {
    let mut q_total = 0.0f32;
    let mut opp_total = 0.0f32;
    for _ in 0..50 {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, rng);
        let mut tracking = colver_core::dmc_obs::EnvTracking::new();
        tracking.dealer = dealer;

        let use_smart = baseline == "smart";
        let mut naive_search = NaiveIsMctsSearch::new();
        let mut smart_searches: Option<[SmartIsMctsSearch; 4]> = if use_smart {
            let mut s = [
                SmartIsMctsSearch::new(), SmartIsMctsSearch::new(),
                SmartIsMctsSearch::new(), SmartIsMctsSearch::new(),
            ];
            for (p, search) in s.iter_mut().enumerate() {
                search.init_deal(&state, p as u8, true);
            }
            Some(s)
        } else {
            None
        };

        while !state.is_terminal() {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if state.phase == Phase::Bidding {
                bid_eval::improved_bid(&state)
            } else if team == q_team {
                let obs = colver_core::dmc_obs::make_observation(&state, &tracking);
                let legal_mask = state.legal_actions() as u32;
                let (best, _) = q_net.best_action(&obs, legal_mask);
                best
            } else {
                match baseline {
                    "random" => {
                        let mask = state.legal_actions();
                        let count = mask.count_ones();
                        let idx = rng.gen_range(0..count);
                        rollout::select_nth_bit(mask, idx)
                    }
                    "naive" => {
                        let config = NaiveIsMctsConfig {
                            time_limit_ms: Some(time_ms),
                            ..Default::default()
                        };
                        naive_search.search(&state, &config, rng)
                    }
                    "smart" => {
                        let config = SmartIsMctsConfig {
                            time_limit_ms: Some(time_ms),
                            ..Default::default()
                        };
                        let searches = smart_searches.as_mut().unwrap();
                        searches[player as usize].search(&state, &config, rng)
                    }
                    _ => unreachable!(),
                }
            };

            if let Some(ref mut searches) = smart_searches {
                for search in searches.iter_mut() {
                    search.record_action(&state, player, action);
                }
            }
            tracking.track_action(&state, action);
            state.step(action);
        }

        let rewards = state.rewards();
        q_total += rewards[q_team as usize];
        opp_total += rewards[1 - q_team as usize];
        if q_total >= 2000.0 || opp_total >= 2000.0 {
            break;
        }
    }
    q_total >= 2000.0
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

    // Initialize trainer
    let mut trainer = DuelingTrainer::new(args.hidden, args.lr, 0.0, device)
        .expect("Failed to create trainer");

    if let Some(ref path) = args.resume {
        trainer.load_checkpoint(path).expect("Failed to load checkpoint");
        println!("Resumed from {}", path);
    }

    // Initialize replay buffer
    let mut replay_buffer = PrioritizedReplayBuffer::new(args.buffer_size, args.per_alpha);

    // Initialize environments
    let mut vec_env = VecTrainingEnv::new_with_seed(args.num_envs, args.seed);
    let mut rng = StdRng::seed_from_u64(args.seed);

    // Randomize bid strategies
    let bid_strategies: Vec<u8> = (0..args.num_envs).map(|_| rng.gen_range(0..NUM_BID_STRATEGIES)).collect();
    vec_env.set_bid_strategies(&bid_strategies);
    let mut bid_strategies = bid_strategies;

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

                // Re-randomize bid strategy
                bid_strategies[i] = rng.gen_range(0..NUM_BID_STRATEGIES);

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
            vec_env.set_bid_strategies(&bid_strategies);
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
                step + 1, eps, beta, replay_buffer.size(), avg_loss, total_episodes, sps
            );
            total_loss = 0.0;
            loss_count = 0;
            last_log_time = Instant::now();
            last_log_step = step + 1;
        }

        // --- Evaluate ---
        if (step + 1) % args.eval_freq == 0 {
            let (deal_wr, rand_wr, naive_wr, smart_wr, eval_time) = evaluate(
                &trainer, args.hidden,
                args.eval_random_matches,
                args.eval_naive_matches,
                args.eval_smart_matches,
                args.eval_time_ms,
            );
            let mut parts = vec![format!("deals {:.0}%", deal_wr * 100.0)];
            if args.eval_random_matches > 0 {
                parts.push(format!("rand {:.0}%", rand_wr * 100.0));
            }
            if args.eval_naive_matches > 0 {
                parts.push(format!("naive {:.0}%", naive_wr * 100.0));
            }
            if args.eval_smart_matches > 0 {
                parts.push(format!("smart {:.0}%", smart_wr * 100.0));
            }
            println!("  [EVAL] {} ({:.0}s)", parts.join(" | "), eval_time);
        }

        // --- Save checkpoint ---
        if (step + 1) % args.save_freq == 0 {
            std::fs::create_dir_all(&args.save_dir).ok();
            let st_path = format!("{}/dmc_{}.safetensors", args.save_dir, step + 1);
            let bin_path = format!("{}/dmc_{}.bin", args.save_dir, step + 1);
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
    let (deal_wr, rand_wr, naive_wr, smart_wr, eval_time) = evaluate(
        &trainer, args.hidden,
        args.eval_random_matches,
        args.eval_naive_matches,
        args.eval_smart_matches,
        args.eval_time_ms,
    );
    println!("Deals vs random: {:.1}%", deal_wr * 100.0);
    if args.eval_random_matches > 0 {
        println!("Matches vs random: {:.1}%", rand_wr * 100.0);
    }
    if args.eval_naive_matches > 0 {
        println!("Matches vs naive IS-MCTS: {:.1}%", naive_wr * 100.0);
    }
    if args.eval_smart_matches > 0 {
        println!("Matches vs smart IS-MCTS: {:.1}%", smart_wr * 100.0);
    }
    println!("Eval time: {:.0}s", eval_time);
    println!("Total training time: {:.0}s", step_start.elapsed().as_secs_f64());

    std::fs::create_dir_all(&args.save_dir).ok();
    let final_st = format!("{}/dmc_final.safetensors", args.save_dir);
    let final_bin = format!("{}/dmc_final_dueling.bin", args.save_dir);
    trainer.save_checkpoint(&final_st).ok();
    trainer.export_binary(&final_bin).ok();
    println!("Saved final model to {} and {}", final_st, final_bin);
}
