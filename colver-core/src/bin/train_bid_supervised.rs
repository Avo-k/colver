//! Full-information supervised bid training.
//!
//! Instead of DQN (learn from one action per episode), compute the reward
//! for ALL possible bids on each deal and train via regression on all 43 outputs.
//!
//! For each deal in the enriched pool:
//!   - For each suit (4) × value (9) + capot (4) = 40 bid actions:
//!     compute score(contract=bid, pts=dd_pts[suit]) → target reward
//!   - PASS = 0 reward
//!   - COINCHE/SURCOINCHE = masked out (no context)
//!
//! 43× more signal per sample than DQN.
//!
//! Usage:
//!   cargo run --bin train_bid_supervised --features dmc_train --release -- [options]

use std::time::Instant;

use candle_core::{DType, Device, Tensor, D};
use candle_nn::{VarBuilder, VarMap};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand::seq::SliceRandom;

use clap::Parser;

use colver_core::bid_candle::{BiddingTrainer, BiddingQNet};
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{self, BID_OBS_DIM, BID_MASK_DIM};
use colver_core::bid_train_env::{DealPool, RewardMode};
use colver_core::scoring::{compute_deal_score, DealScore};
use colver_core::state::{GameState, Phase, Contract};
use colver_core::suit_perm;

const NUM_ACTIONS: usize = 43;
/// Bid values: index 0=80, 1=90, ..., 8=160
const BID_VALUES: [u8; 9] = [8, 9, 10, 11, 12, 13, 14, 15, 16];

#[derive(Parser)]
#[command(name = "train_bid_supervised")]
struct Args {
    #[arg(long, default_value_t = 5_000_000)]
    steps: usize,
    #[arg(long, default_value_t = 512)]
    batch_size: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 512)]
    hidden: usize,
    #[arg(long, default_value_t = 3)]
    layers: usize,
    #[arg(long, default_value_t = 50_000)]
    eval_freq: usize,
    #[arg(long, default_value_t = 200)]
    eval_matches: usize,
    #[arg(long, default_value_t = 1_000_000)]
    save_freq: usize,
    #[arg(long, default_value = "models/bid_v3_supervised")]
    save_dir: String,
    #[arg(long)]
    resume: Option<String>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value = "data/deals/archive/dd_pool_enriched_1M.bin")]
    pool_file: String,
    /// Reward source: "dd" or "real" or "blend:0.7"
    #[arg(long, default_value = "dd")]
    reward: String,
}

fn main() {
    let args = Args::parse();

    let device = Device::cuda_if_available(0).expect("No CUDA device");
    println!("=== Supervised Bid Training (Full-Information) ===");
    println!("Device: {:?}", device);
    println!("Hidden: {}, Layers: {}, LR: {}", args.hidden, args.layers, args.lr);
    println!("Steps: {}, Batch: {}", args.steps, args.batch_size);

    // Parse reward mode
    let reward_mode = if args.reward == "dd" {
        RewardMode::DdOnly
    } else if args.reward == "real" {
        RewardMode::RealOnly
    } else if args.reward.starts_with("blend:") {
        let alpha: f32 = args.reward[6..].parse().unwrap();
        RewardMode::Blend(alpha)
    } else {
        panic!("Unknown reward mode '{}'", args.reward);
    };
    println!("Reward: {:?}", reward_mode);

    // Load pool
    let pool = match DealPool::load_enriched(&args.pool_file) {
        Ok(p) => p,
        Err(_) => DealPool::load(&args.pool_file).expect("Failed to load pool"),
    };
    println!("Pool: {} deals from {}", pool.len(), args.pool_file);

    // Precompute target rewards for all deals
    println!("\nPrecomputing target rewards for all deals...");
    let precompute_start = Instant::now();
    let targets = precompute_all_targets(&pool, reward_mode);
    println!(
        "  Done: {} deals × 43 actions in {:.1}s",
        targets.len(),
        precompute_start.elapsed().as_secs_f64()
    );

    // Create trainer
    let mut trainer = BiddingTrainer::with_layers(args.layers, args.hidden, args.lr, 0.0, device.clone())
        .expect("Failed to create trainer");

    if let Some(ref path) = args.resume {
        trainer.load_checkpoint(path).expect("Failed to load checkpoint");
        println!("Resumed from {}", path);
    }

    // Ensure save dir exists
    std::fs::create_dir_all(&args.save_dir).ok();

    let mut rng = StdRng::seed_from_u64(args.seed);
    let n_deals = pool.len();

    println!("\n--- Training ---");
    println!(
        "      Step |     Loss |  Eval W% | Steps/s"
    );
    println!("{}", "-".repeat(50));

    let start = Instant::now();
    let mut total_loss = 0.0f64;
    let mut loss_count = 0usize;

    // Shuffle indices for sampling
    let mut indices: Vec<usize> = (0..n_deals).collect();

    for step in 0..args.steps {
        // Reshuffle every epoch
        if step % n_deals == 0 {
            indices.shuffle(&mut rng);
        }

        // Sample a batch of deals
        let batch_start = (step * args.batch_size) % n_deals;
        let mut obs_flat = vec![0.0f32; args.batch_size * BID_OBS_DIM];
        let mut target_flat = vec![0.0f32; args.batch_size * NUM_ACTIONS];
        let mut mask_flat = vec![0.0f32; args.batch_size * NUM_ACTIONS];

        for b in 0..args.batch_size {
            let deal_idx = indices[(batch_start + b) % n_deals];
            let deal = pool.get(deal_idx);

            // Write observation for first bidder (no bid history)
            let state = GameState::new(deal.dealer, deal.hands);
            let empty_history: Vec<(u8, u8)> = Vec::new();
            bid_obs::write_bid_observation(
                &mut obs_flat, b * BID_OBS_DIM, &state, &empty_history,
            );

            // Copy target rewards
            let t = &targets[deal_idx];
            target_flat[b * NUM_ACTIONS..(b + 1) * NUM_ACTIONS].copy_from_slice(t);

            // Mask: all bid actions + pass are valid for first bidder
            // (no coinche/surcoinche since no prior bid)
            for a in 0..41 {
                mask_flat[b * NUM_ACTIONS + a] = 1.0;
            }
            // coinche/surcoinche masked out
            mask_flat[b * NUM_ACTIONS + 41] = 0.0;
            mask_flat[b * NUM_ACTIONS + 42] = 0.0;

            // 24× suit augmentation: randomly permute
            let perm_idx = rng.gen_range(0..24usize);
            let perm = &suit_perm::ALL_PERMS[perm_idx];
            suit_perm::permute_bid_obs(
                &mut obs_flat[b * BID_OBS_DIM..(b + 1) * BID_OBS_DIM],
                perm,
            );
            // Also permute targets and masks
            permute_targets(
                &mut target_flat[b * NUM_ACTIONS..(b + 1) * NUM_ACTIONS],
                perm,
            );
            permute_targets(
                &mut mask_flat[b * NUM_ACTIONS..(b + 1) * NUM_ACTIONS],
                perm,
            );
        }

        // GPU forward + backward on all 43 outputs
        let loss = train_step_supervised(
            &mut trainer, &obs_flat, &target_flat, &mask_flat,
            args.batch_size, &device,
        );
        total_loss += loss as f64;
        loss_count += 1;

        // Logging
        if (step + 1) % 1000 == 0 {
            let avg_loss = total_loss / loss_count as f64;
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (step + 1) as f64 / elapsed;
            eprintln!(
                "{:>10} | {:>8.5} |          | {:>7.0}",
                step + 1, avg_loss, rate,
            );
            total_loss = 0.0;
            loss_count = 0;
        }

        // Eval
        if (step + 1) % args.eval_freq == 0 {
            let weights = trainer.snapshot_weights().unwrap();
            let mut eval_net = BidNet::from_floats_with_layers(
                &weights, args.hidden, BID_OBS_DIM, true, args.layers,
            ).unwrap();
            let (wins, total, margin) = evaluate(&mut eval_net, args.eval_matches);
            let win_pct = wins as f64 / total as f64 * 100.0;
            eprintln!(
                "  [EVAL] vs improved_v2: {:.1}% ({}/{}) margin={:+}",
                win_pct, wins, total, margin,
            );
        }

        // Save
        if (step + 1) % args.save_freq == 0 {
            let path_st = format!("{}/bid_nn_{}.safetensors", args.save_dir, step + 1);
            trainer.save_checkpoint(&path_st).ok();
            let path_bin = format!("{}/bid_nn_{}.bin", args.save_dir, step + 1);
            trainer.export_binary(&path_bin).ok();
            // Also save as latest
            let latest_st = format!("{}/bid_nn_latest.safetensors", args.save_dir);
            trainer.save_checkpoint(&latest_st).ok();
            let latest_bin = format!("{}/bid_nn_latest.bin", args.save_dir);
            trainer.export_binary(&latest_bin).ok();
        }
    }

    // Save final
    let final_st = format!("{}/bid_nn_final.safetensors", args.save_dir);
    trainer.save_checkpoint(&final_st).ok();
    let final_bin = format!("{}/bid_nn_final.bin", args.save_dir);
    trainer.export_binary(&final_bin).ok();

    let elapsed = start.elapsed().as_secs_f64();
    println!("\n\nDone: {} steps in {:.1}s ({:.0} steps/s)", args.steps, elapsed, args.steps as f64 / elapsed);
    println!("Final model saved to {}", args.save_dir);
}

/// Precompute target rewards for all 43 actions on each deal.
/// Returns Vec of [f32; 43] per deal.
fn precompute_all_targets(pool: &DealPool, reward_mode: RewardMode) -> Vec<[f32; NUM_ACTIONS]> {
    let n = pool.len();
    let mut all_targets = Vec::with_capacity(n);

    for i in 0..n {
        let deal = pool.get(i);
        let mut targets = [0.0f32; NUM_ACTIONS];

        // Get NS points per suit based on reward mode
        let ns_pts: [u8; 4] = match reward_mode {
            RewardMode::DdOnly => deal.dd_pts,
            RewardMode::RealOnly => deal.real_pts.unwrap_or(deal.dd_pts),
            RewardMode::Blend(alpha) => {
                if let Some(real) = deal.real_pts {
                    let mut blended = [0u8; 4];
                    for s in 0..4 {
                        let v = alpha * deal.dd_pts[s] as f32
                            + (1.0 - alpha) * real[s] as f32;
                        blended[s] = v.round().max(0.0).min(252.0) as u8;
                    }
                    blended
                } else {
                    deal.dd_pts
                }
            }
        };

        // First bidder is (dealer + 1) % 4
        let bidder = (deal.dealer + 1) % 4;
        let bidder_team = GameState::player_team(bidder); // 0=NS, 1=EW

        // Action 0: PASS → reward 0 (void deal)
        targets[0] = 0.0;

        // Actions 1-36: regular bids (value_idx × 4 + suit_idx + 1)
        for value_idx in 0..9u8 {
            for suit_idx in 0..4u8 {
                let action = (value_idx * 4 + suit_idx + 1) as usize;
                let contract_value = BID_VALUES[value_idx as usize]; // encoded value (8=80, etc.)
                let (ns_score, ew_score) = score_hypothetical_bid(
                    ns_pts[suit_idx as usize],
                    suit_idx,
                    contract_value,
                    false, // not capot
                    bidder_team,
                );
                targets[action] = (ns_score - ew_score) as f32 / 500.0;
                if bidder_team == 1 {
                    targets[action] = -targets[action]; // EW perspective
                }
            }
        }

        // Actions 37-40: capot per suit
        for suit_idx in 0..4u8 {
            let action = (37 + suit_idx) as usize;
            let (ns_score, ew_score) = score_hypothetical_bid(
                ns_pts[suit_idx as usize],
                suit_idx,
                25, // capot
                true,
                bidder_team,
            );
            targets[action] = (ns_score - ew_score) as f32 / 500.0;
            if bidder_team == 1 {
                targets[action] = -targets[action];
            }
        }

        // Actions 41-42: coinche/surcoinche → 0 (masked out during training)
        targets[41] = 0.0;
        targets[42] = 0.0;

        all_targets.push(targets);
    }

    all_targets
}

/// Compute match score for a hypothetical bid.
fn score_hypothetical_bid(
    ns_pts: u8,
    trump: u8,
    contract_value: u8,
    is_capot: bool,
    taker_team: u8,
) -> (i16, i16) {
    let taker = taker_team as usize;
    let defense = 1 - taker;

    let taker_pts = if taker == 0 { ns_pts } else {
        if ns_pts == 252 || ns_pts == 0 { 252 - ns_pts } else { 162 - ns_pts }
    };
    let defense_pts = if defense == 0 { ns_pts } else {
        if ns_pts == 252 || ns_pts == 0 { 252 - ns_pts } else { 162 - ns_pts }
    };

    let (taker_tricks, defense_tricks) = if defense_pts == 0 {
        (8u8, 0u8)
    } else if taker_pts == 0 {
        (0u8, 8u8)
    } else {
        let total = taker_pts as f32 + defense_pts as f32;
        let t = (taker_pts as f32 / total * 8.0).round().max(1.0).min(7.0) as u8;
        (t, 8 - t)
    };

    let mut terminal = GameState::new(0, [0; 4]);
    terminal.phase = Phase::Done;
    terminal.contract = Contract {
        trump,
        value: contract_value,
        team: taker_team,
        coinche: 0,
    };
    terminal.points[taker] = taker_pts;
    terminal.points[defense] = defense_pts;
    terminal.tricks_won[taker] = taker_tricks;
    terminal.tricks_won[defense] = defense_tricks;
    terminal.belote = [0; 2];

    let score = compute_deal_score(&terminal);
    (score.scores[0], score.scores[1])
}

/// Supervised training step: MSE loss on all 43 outputs (masked).
fn train_step_supervised(
    trainer: &mut BiddingTrainer,
    obs: &[f32],
    targets: &[f32],
    mask: &[f32],
    batch_size: usize,
    device: &Device,
) -> f32 {
    let obs_t = Tensor::from_slice(obs, (batch_size, BID_OBS_DIM), device).unwrap();
    let targets_t = Tensor::from_slice(targets, (batch_size, NUM_ACTIONS), device).unwrap();
    let mask_t = Tensor::from_slice(mask, (batch_size, NUM_ACTIONS), device).unwrap();

    let q_all = trainer.net.forward(&obs_t).unwrap();
    let errors = (&q_all - &targets_t).unwrap();
    let sq_errors = errors.sqr().unwrap();
    let masked = (&sq_errors * &mask_t).unwrap();
    // Average over valid actions only
    let loss = masked.sum_all().unwrap();
    let n_valid = mask_t.sum_all().unwrap();
    let loss = (&loss / &n_valid).unwrap();

    trainer.backward_step(&loss).unwrap();
    loss.detach().to_vec0::<f32>().unwrap()
}

/// Permute target/mask array for 43 bid actions according to suit permutation.
fn permute_targets(buf: &mut [f32], perm: &[u8; 4]) {
    debug_assert_eq!(buf.len(), NUM_ACTIONS);

    // Save originals
    let orig: [f32; NUM_ACTIONS] = {
        let mut arr = [0.0f32; NUM_ACTIONS];
        arr.copy_from_slice(buf);
        arr
    };

    // Action 0 (PASS) stays
    // Actions 1-36: action = value_idx * 4 + suit_idx + 1
    for value_idx in 0..9u8 {
        for suit_idx in 0..4u8 {
            let old_action = (value_idx * 4 + suit_idx + 1) as usize;
            let new_suit = perm[suit_idx as usize];
            let new_action = (value_idx * 4 + new_suit + 1) as usize;
            buf[new_action] = orig[old_action];
        }
    }
    // Actions 37-40: capot per suit
    for suit_idx in 0..4u8 {
        let old_action = (37 + suit_idx) as usize;
        let new_suit = perm[suit_idx as usize];
        let new_action = (37 + new_suit) as usize;
        buf[new_action] = orig[old_action];
    }
    // 41, 42 stay
}

/// Evaluate NN bidding vs improved_v2 (same as train_bid_nn eval).
fn evaluate(bid_net: &mut BidNet, num_matches: usize) -> (usize, usize, i64) {
    use colver_core::bid_eval;
    use colver_core::solver;

    let mut rng = StdRng::seed_from_u64(12345);
    let mut nn_wins = 0usize;
    let mut total_margin = 0i64;

    for match_idx in 0..num_matches {
        let nn_team: u8 = (match_idx % 2) as u8;
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut bid_history: Vec<(u8, u8)> = Vec::new();
        let mut obs_buf = vec![0.0f32; BID_OBS_DIM];

        while state.phase == Phase::Bidding {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if team == nn_team {
                bid_obs::write_bid_observation(&mut obs_buf, 0, &state, &bid_history);
                let mask = state.legal_actions();
                bid_net.best_action_fast(&obs_buf, mask)
            } else {
                bid_eval::improved_v2_bid(&state)
            };

            bid_history.push((player, action));
            state.step(action);
        }

        if state.contract.value == 0 {
            continue;
        }

        // Score with DD
        let mut tt = solver::new_tt_buffer();
        let trump = state.contract.trump;
        let dd_result = solver::solve_for_trump_reuse_tt(state.hands, dealer, trump, &mut tt);
        let ns_pts = dd_result[0];

        let taker = state.contract.team as usize;
        let defense = 1 - taker;
        let taker_pts = if taker == 0 { ns_pts } else {
            if ns_pts == 252 || ns_pts == 0 { 252 - ns_pts } else { 162 - ns_pts }
        };
        let defense_pts = if defense == 0 { ns_pts } else {
            if ns_pts == 252 || ns_pts == 0 { 252 - ns_pts } else { 162 - ns_pts }
        };

        let (taker_tricks, defense_tricks) = if defense_pts == 0 {
            (8u8, 0u8)
        } else if taker_pts == 0 {
            (0u8, 8u8)
        } else {
            let total = taker_pts as f32 + defense_pts as f32;
            let t = (taker_pts as f32 / total * 8.0).round().max(1.0).min(7.0) as u8;
            (t, 8 - t)
        };

        let mut terminal = GameState::new(0, [0; 4]);
        terminal.phase = Phase::Done;
        terminal.contract = state.contract;
        terminal.points[taker] = taker_pts;
        terminal.points[defense] = defense_pts;
        terminal.tricks_won[taker] = taker_tricks;
        terminal.tricks_won[defense] = defense_tricks;
        terminal.belote = [0; 2];

        let score = compute_deal_score(&terminal);
        let nn_score = score.scores[nn_team as usize];
        let opp_score = score.scores[1 - nn_team as usize];

        if nn_score > opp_score {
            nn_wins += 1;
        }
        total_margin += (nn_score - opp_score) as i64;
    }

    (nn_wins, num_matches, total_margin)
}
