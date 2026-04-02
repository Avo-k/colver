/// Pure Rust NN bidding training binary using Candle + Dueling DQN.
///
/// Trains a bidding Q-network using DD oracle for reward computation.
/// Each env DD-solves all 4 suits at deal start (~52ms), then runs
/// bidding in microseconds. Reward = (my_team_score - opp_score) / 500.
///
/// Opponent diversity: configurable mix of self-play, improved_v2,
/// aggressive, conservative, and random bidding opponents. The non-self-play
/// ratio anneals linearly from --diversity-start to --diversity-end.
///
/// Usage:
///   cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- --num-envs 64 --steps 5000000

use std::time::Instant;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use candle_core::Device;

use colver_core::bid_candle::BiddingTrainer;
use colver_core::bid_eval;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::BID_OBS_DIM;
use colver_core::suit_perm;
use colver_core::bid_train_env::{BidReplayBuffer, DealPool, VecBidEnv};
use colver_core::rollout;
use colver_core::state::GameState;

const NUM_ACTIONS: usize = 43;

/// Opponent bidding strategy for non-self-play environments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OpponentMode {
    /// Both teams use the NN (pure self-play).
    SelfPlay,
    /// Opponent uses improved_v2 (the default balanced bidder).
    ImprovedV2,
    /// Opponent uses aggressive bidding (low thresholds, high caps, no quality gate).
    Aggressive,
    /// Opponent uses conservative bidding (high thresholds, low caps).
    Conservative,
    /// Opponent bids randomly among legal actions.
    Random,
}

/// Within non-self-play, relative weights for each opponent type.
/// These stay constant; only the total non-self-play ratio anneals.
const OPP_WEIGHT_IMPROVED: f32 = 8.0;
const OPP_WEIGHT_AGGRESSIVE: f32 = 8.0;
const OPP_WEIGHT_CONSERVATIVE: f32 = 8.0;
const OPP_WEIGHT_RANDOM: f32 = 16.0;
const OPP_WEIGHT_TOTAL: f32 =
    OPP_WEIGHT_IMPROVED + OPP_WEIGHT_AGGRESSIVE + OPP_WEIGHT_CONSERVATIVE + OPP_WEIGHT_RANDOM;

/// Pick an opponent mode based on current training progress.
fn pick_opponent_mode(
    step: usize,
    total_steps: usize,
    diversity_start: f32,
    diversity_end: f32,
    rng: &mut impl Rng,
) -> OpponentMode {
    let progress = (step as f32 / total_steps as f32).min(1.0);
    let non_self_play = diversity_start + (diversity_end - diversity_start) * progress;

    let r: f32 = rng.gen();
    if r >= non_self_play {
        return OpponentMode::SelfPlay;
    }

    // Within non-self-play, distribute by relative weights
    let inner_r: f32 = rng.gen::<f32>() * OPP_WEIGHT_TOTAL;
    if inner_r < OPP_WEIGHT_IMPROVED {
        OpponentMode::ImprovedV2
    } else if inner_r < OPP_WEIGHT_IMPROVED + OPP_WEIGHT_AGGRESSIVE {
        OpponentMode::Aggressive
    } else if inner_r < OPP_WEIGHT_IMPROVED + OPP_WEIGHT_AGGRESSIVE + OPP_WEIGHT_CONSERVATIVE {
        OpponentMode::Conservative
    } else {
        OpponentMode::Random
    }
}

/// Get action from a fixed bidding strategy.
fn opponent_action(state: &GameState, mode: OpponentMode, rng: &mut impl Rng) -> u8 {
    match mode {
        OpponentMode::SelfPlay => unreachable!("SelfPlay has no fixed action"),
        OpponentMode::ImprovedV2 => bid_eval::improved_v2_bid(state),
        OpponentMode::Aggressive => {
            bid_eval::parametric_bid(state, &bid_eval::BidParams::very_aggressive())
        }
        OpponentMode::Conservative => {
            bid_eval::parametric_bid(state, &bid_eval::BidParams::ultra_conservative())
        }
        OpponentMode::Random => {
            let mask = state.legal_actions();
            let count = mask.count_ones();
            let idx = rng.gen_range(0..count);
            rollout::select_nth_bit(mask, idx)
        }
    }
}

#[derive(Parser)]
#[command(
    name = "train_bid_nn",
    about = "NN bidding training with DD oracle + Dueling DQN"
)]
struct Args {
    #[arg(long, default_value_t = 64)]
    num_envs: usize,
    #[arg(long, default_value_t = 5_000_000)]
    steps: usize,
    #[arg(long, default_value_t = 512)]
    batch_size: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 256)]
    hidden: usize,
    #[arg(long, default_value_t = 2)]
    layers: usize,
    #[arg(long, default_value_t = 0.3)]
    eps_start: f32,
    #[arg(long, default_value_t = 0.02)]
    eps_end: f32,
    #[arg(long, default_value_t = 3_000_000)]
    eps_decay_steps: usize,
    #[arg(long, default_value_t = 500_000)]
    buffer_size: usize,
    #[arg(long, default_value_t = 5_000)]
    min_buffer: usize,
    #[arg(long, default_value_t = 4)]
    train_freq: usize,
    #[arg(long, default_value_t = 50_000)]
    eval_freq: usize,
    #[arg(long, default_value_t = 200)]
    eval_matches: usize,
    #[arg(long, default_value_t = 200_000)]
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
    /// Initial non-self-play ratio (0.0-1.0). Default 0.40.
    #[arg(long, default_value_t = 0.40)]
    diversity_start: f32,
    /// Final non-self-play ratio (0.0-1.0). Default 0.15.
    #[arg(long, default_value_t = 0.15)]
    diversity_end: f32,
    /// Number of deals to pre-solve with DD.
    #[arg(long, default_value_t = 1_000_000)]
    pool_size: usize,
    /// Path to save/load deal pool (auto-generates if missing).
    #[arg(long, default_value = "data/dd_pool_1M.bin")]
    pool_file: String,
}

/// Evaluate NN bidding vs improved_v2 bidding.
/// Both use DD oracle for card play (same scoring).
/// Returns (nn_wins, total_deals, avg_margin).
fn evaluate(trainer: &BiddingTrainer, hidden: usize, layers: usize, num_matches: usize) -> (usize, usize, f64) {
    let weights = match trainer.snapshot_weights() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to snapshot weights: {}", e);
            return (0, 0, 0.0);
        }
    };
    let mut bid_net = match BidNet::from_floats_with_layers(&weights, hidden, BID_OBS_DIM, true, layers) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to load eval net: {}", e);
            return (0, 0, 0.0);
        }
    };

    let mut rng = StdRng::seed_from_u64(12345);
    let mut nn_wins = 0usize;
    let mut total_margin = 0i64;
    let mut total_deals = 0usize;

    for match_idx in 0..num_matches {
        // Alternate sides: NN plays NS for even matches, EW for odd
        let nn_team: u8 = (match_idx % 2) as u8;

        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut bid_history: Vec<(u8, u8)> = Vec::new();

        // DD-solve all 4 suits for this deal
        let mut tt_buf = colver_core::solver::new_tt_buffer();
        let mut dd_pts = [0u8; 4];
        for suit in 0..4u8 {
            let result = colver_core::solver::solve_for_trump_reuse_tt(
                state.hands,
                state.dealer,
                suit,
                &mut tt_buf,
            );
            dd_pts[suit as usize] = result[0];
        }

        // Run bidding
        while state.phase == colver_core::state::Phase::Bidding {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if team == nn_team {
                // NN bid
                let obs = colver_core::bid_obs::make_bid_observation(&state, &bid_history);
                let legal = state.legal_actions();
                let (best, _) = bid_net.best_action(&obs, legal);
                best
            } else {
                // Baseline: improved_v2
                colver_core::bid_eval::improved_v2_bid(&state)
            };

            bid_history.push((player, action));
            state.step(action);
        }

        // Score using DD
        let (ns_score, ew_score) = compute_dd_scores(&state, &dd_pts);
        let nn_score = if nn_team == 0 { ns_score } else { ew_score };
        let opp_score = if nn_team == 0 { ew_score } else { ns_score };

        if nn_score > opp_score {
            nn_wins += 1;
        }
        total_margin += (nn_score - opp_score) as i64;
        total_deals += 1;
    }

    let avg_margin = if total_deals > 0 {
        total_margin as f64 / total_deals as f64
    } else {
        0.0
    };

    (nn_wins, total_deals, avg_margin)
}

/// Compute deal scores from DD results (same logic as BidTrainingEnv).
fn compute_dd_scores(state: &GameState, dd_pts: &[u8; 4]) -> (i16, i16) {
    if state.contract.value == 0 {
        return (0, 0);
    }

    let trump = state.contract.trump;
    let ns_dd_pts = dd_pts[trump as usize];
    let ew_dd_pts = if ns_dd_pts == 252 || ns_dd_pts == 0 {
        252 - ns_dd_pts
    } else {
        162 - ns_dd_pts
    };

    let taker = state.contract.team as usize;
    let defense = 1 - taker;

    let taker_pts = if taker == 0 { ns_dd_pts } else { ew_dd_pts };
    let defense_pts = if defense == 0 { ns_dd_pts } else { ew_dd_pts };

    let (taker_tricks, defense_tricks) = if defense_pts == 0 {
        (8u8, 0u8)
    } else if taker_pts == 0 {
        (0u8, 8u8)
    } else {
        let total_pts = taker_pts as u16 + defense_pts as u16;
        let taker_frac = taker_pts as f32 / total_pts as f32;
        let t = (taker_frac * 8.0).round().max(1.0).min(7.0) as u8;
        (t, 8 - t)
    };

    let mut terminal = GameState::new(0, [0; 4]);
    terminal.phase = colver_core::state::Phase::Done;
    terminal.contract = state.contract;
    terminal.points[taker] = taker_pts;
    terminal.points[defense] = defense_pts;
    terminal.tricks_won[taker] = taker_tricks;
    terminal.tricks_won[defense] = defense_tricks;
    terminal.belote = [0; 2];

    let score = colver_core::scoring::compute_deal_score(&terminal);
    (score.scores[0], score.scores[1])
}

fn main() {
    let args = Args::parse();

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).expect("CUDA device creation failed")
    } else {
        eprintln!("WARNING: CUDA not available, using CPU");
        Device::Cpu
    };
    let device_name = match &device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        _ => "Other",
    };

    println!("=== NN Bidding Training (DD Oracle) ===");
    println!("Device: {}", device_name);
    println!(
        "Envs: {}, Steps: {}, LR: {}, Hidden: {}, Layers: {}",
        args.num_envs, args.steps, args.lr, args.hidden, args.layers
    );
    println!(
        "Training: batch={}, freq={}, buffer={}",
        args.batch_size, args.train_freq, args.buffer_size
    );
    println!(
        "PER: alpha={}, beta={}->{}",
        args.per_alpha, args.per_beta_start, args.per_beta_end
    );
    println!(
        "Epsilon: {}->{}  over {} steps",
        args.eps_start, args.eps_end, args.eps_decay_steps
    );
    println!(
        "Diversity: {:.0}% -> {:.0}% non-self-play (improved_v2 + aggressive + conservative + random)",
        args.diversity_start * 100.0,
        args.diversity_end * 100.0
    );

    // Initialize trainer
    let mut trainer = BiddingTrainer::with_layers(args.layers, args.hidden, args.lr, 0.0, device)
        .expect("Failed to create trainer");

    if let Some(ref path) = args.resume {
        trainer
            .load_checkpoint(path)
            .expect("Failed to load checkpoint");
        println!("Resumed from {}", path);
    }

    // Initialize replay buffer
    let mut replay_buffer = BidReplayBuffer::new(args.buffer_size, args.per_alpha);

    // Phase 1: Load or generate deal pool
    println!(
        "\n--- Phase 1: {} DD-solved deals (file: {}) ---",
        args.pool_size, args.pool_file
    );
    // Ensure data directory exists
    if let Some(parent) = std::path::Path::new(&args.pool_file).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let pool_start = Instant::now();
    let pool = DealPool::load_or_generate(&args.pool_file, args.pool_size, args.seed + 100);
    println!(
        "Deal pool ready: {} deals in {:.1}s",
        pool.len(),
        pool_start.elapsed().as_secs_f64()
    );

    // Phase 2: Initialize envs from pool (instant)
    println!("\n--- Phase 2: Training ---");
    let mut vec_env = VecBidEnv::new_with_pool(args.num_envs, args.seed, &pool);

    let mut rng = StdRng::seed_from_u64(args.seed + 1);

    // Per-env opponent tracking
    let n = args.num_envs;
    let mut opp_modes: Vec<OpponentMode> = (0..n)
        .map(|_| {
            pick_opponent_mode(
                0,
                args.steps,
                args.diversity_start,
                args.diversity_end,
                &mut rng,
            )
        })
        .collect();
    let mut nn_teams: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2u8)).collect();

    // Stats
    let mut total_transitions = 0usize;
    let mut total_episodes = 0usize;
    let mut total_loss = 0.0f64;
    let mut loss_count = 0usize;
    let mut total_void_deals = 0usize;
    let mut self_play_episodes = 0usize;
    let mut diverse_episodes = 0usize;
    let step_start = Instant::now();
    let mut last_log_time = Instant::now();
    let mut last_log_step = 0usize;

    println!(
        "\n{:>10} | {:>5} | {:>5} | {:>7} | {:>8} | {:>8} | {:>6} | {:>5} | {:>7}",
        "Step", "Eps", "Beta", "Buffer", "Loss", "Episodes", "Voids", "Div%", "Steps/s"
    );
    println!("{}", "-".repeat(88));

    for step in 0..args.steps {
        // Epsilon schedule
        let progress = ((step as f64) / args.eps_decay_steps as f64).min(1.0);
        let eps = args.eps_start + (args.eps_end - args.eps_start) * progress as f32;

        // Beta schedule
        let beta_progress = ((step as f64) / args.steps as f64).min(1.0);
        let beta =
            args.per_beta_start + (args.per_beta_end - args.per_beta_start) * beta_progress;

        // --- Collect actions for all envs ---
        // GPU batch forward pass for all envs (even opponent-controlled ones, for simplicity)
        let obs_flat = &vec_env.obs_buf;
        let mask_flat = &vec_env.mask_buf;

        let nn_actions = match trainer.net.act(
            &candle_core::Tensor::from_slice(obs_flat, (n, BID_OBS_DIM), trainer.device())
                .unwrap(),
            &candle_core::Tensor::from_slice(mask_flat, (n, NUM_ACTIONS), trainer.device())
                .unwrap(),
            eps,
            &mut rng,
        ) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("GPU forward failed: {}", e);
                (0..n).map(|i| vec_env.random_action(i)).collect()
            }
        };

        // --- Determine final actions: NN or opponent ---
        let mut actions = nn_actions;
        for i in 0..n {
            if opp_modes[i] != OpponentMode::SelfPlay {
                let player = vec_env.envs[i].state.current_player();
                let team = GameState::player_team(player);
                if team != nn_teams[i] {
                    // Opponent's turn — use fixed strategy
                    actions[i] = opponent_action(&vec_env.envs[i].state, opp_modes[i], &mut rng);
                }
            }
        }

        // --- Step all envs, flush completed episodes ---
        for i in 0..n {
            if let Some(transitions) = vec_env.step_env_pooled(i, actions[i], &pool) {
                let is_void = transitions.iter().all(|(_, _, _, r, _)| *r == 0.0);
                if is_void {
                    total_void_deals += 1;
                }

                let is_self_play = opp_modes[i] == OpponentMode::SelfPlay;

                for (obs, mask, action, reward, team) in &transitions {
                    // In self-play: add all transitions
                    // In diverse mode: only add NN team's transitions
                    if is_self_play || *team == nn_teams[i] {
                        replay_buffer.push(obs, mask, *action, *reward);
                        total_transitions += 1;
                    }
                }

                if is_self_play {
                    self_play_episodes += 1;
                } else {
                    diverse_episodes += 1;
                }
                total_episodes += 1;

                // Pick new opponent mode and team for next episode
                opp_modes[i] = pick_opponent_mode(
                    step,
                    args.steps,
                    args.diversity_start,
                    args.diversity_end,
                    &mut rng,
                );
                nn_teams[i] = rng.gen_range(0..2u8);
            }
        }

        // --- Train ---
        if replay_buffer.size() >= args.min_buffer && step % args.train_freq == 0 {
            let mut sample = replay_buffer.sample(args.batch_size, beta, &mut rng);

            // 24× suit augmentation: random permutation per sample
            suit_perm::augment_bid_batch(
                &mut sample.obs_data,
                &mut sample.mask_data,
                &mut sample.actions,
                &mut rng,
            );

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

        // --- Logging ---
        if (step + 1) % 1_000 == 0 {
            let elapsed = last_log_time.elapsed().as_secs_f64();
            let steps_done = step + 1 - last_log_step;
            let sps = steps_done as f64 / elapsed.max(1e-6);
            let avg_loss = if loss_count > 0 {
                total_loss / loss_count as f64
            } else {
                0.0
            };
            let div_pct = if total_episodes > 0 {
                diverse_episodes as f64 / total_episodes as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "{:>10} | {:>5.3} | {:>5.3} | {:>7} | {:>8.5} | {:>8} | {:>6} | {:>4.0}% | {:>7.0}",
                step + 1,
                eps,
                beta,
                replay_buffer.size(),
                avg_loss,
                total_episodes,
                total_void_deals,
                div_pct,
                sps
            );
            total_loss = 0.0;
            loss_count = 0;
            last_log_time = Instant::now();
            last_log_step = step + 1;
        }

        // --- Evaluate ---
        if (step + 1) % args.eval_freq == 0 {
            let eval_start = Instant::now();
            let (wins, total, margin) = evaluate(&trainer, args.hidden, args.layers, args.eval_matches);
            let wr = if total > 0 {
                wins as f64 / total as f64
            } else {
                0.0
            };
            println!(
                "  [EVAL] vs improved_v2: {:.1}% ({}/{}) margin={:+.0}  ({:.0}s)",
                wr * 100.0,
                wins,
                total,
                margin,
                eval_start.elapsed().as_secs_f64()
            );
        }

        // --- Save checkpoint ---
        if (step + 1) % args.save_freq == 0 {
            std::fs::create_dir_all(&args.save_dir).ok();
            let st_path = format!("{}/bid_nn_{}.safetensors", args.save_dir, step + 1);
            let bin_path = format!("{}/bid_nn_{}.bin", args.save_dir, step + 1);
            if let Err(e) = trainer.save_checkpoint(&st_path) {
                eprintln!("Failed to save safetensors: {}", e);
            }
            if let Err(e) = trainer.export_binary(&bin_path) {
                eprintln!("Failed to export binary: {}", e);
            }
            // Also save as latest
            let latest_st = format!("{}/bid_nn_latest.safetensors", args.save_dir);
            let latest_bin = format!("{}/bid_nn_latest.bin", args.save_dir);
            trainer.save_checkpoint(&latest_st).ok();
            trainer.export_binary(&latest_bin).ok();
            println!("  [SAVE] {}", st_path);
        }
    }

    // Final eval and save
    println!("\n--- Final Evaluation ---");
    let eval_start = Instant::now();
    let (wins, total, margin) = evaluate(&trainer, args.hidden, args.layers, args.eval_matches);
    let wr = if total > 0 {
        wins as f64 / total as f64
    } else {
        0.0
    };
    println!(
        "vs improved_v2: {:.1}% ({}/{}) margin={:+.0}",
        wr * 100.0, wins, total, margin
    );
    println!(
        "Self-play: {} episodes, Diverse: {} episodes ({:.0}%)",
        self_play_episodes,
        diverse_episodes,
        if total_episodes > 0 {
            diverse_episodes as f64 / total_episodes as f64 * 100.0
        } else {
            0.0
        }
    );
    println!("Eval time: {:.0}s", eval_start.elapsed().as_secs_f64());
    println!(
        "Total training time: {:.0}s",
        step_start.elapsed().as_secs_f64()
    );
    println!(
        "Total transitions: {}, episodes: {}, void deals: {}",
        total_transitions, total_episodes, total_void_deals
    );

    std::fs::create_dir_all(&args.save_dir).ok();
    let final_st = format!("{}/bid_nn_final.safetensors", args.save_dir);
    let final_bin = format!("{}/bid_nn_final.bin", args.save_dir);
    trainer.save_checkpoint(&final_st).ok();
    trainer.export_binary(&final_bin).ok();
    println!("Saved final model to {} and {}", final_st, final_bin);
}
