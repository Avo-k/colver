//! GPU-batched penalty sweep: test bid penalty values by playing full matches.
//!
//! All bots share the same bid NN + play NN, differing only in bid penalty.
//! GPU batches both bid and play inference for massive throughput.
//!
//! Usage:
//!   cargo run --bin penalty_sweep --release --features dmc_train -- [options]
//!
//! Options:
//!   --matches N        Matches per penalty pair per direction (default: 500)
//!   --bid-model PATH   Bid model (default: models/bid_v2/bid_nn_final.bin)
//!   --play-model PATH  Play model (default: models/play_v2/play_final.bin)
//!   --penalties LIST   Comma-separated penalty values (default: 0.0,0.05,0.10,0.15,0.20,0.30)
//!   --seed N           RNG seed (default: 42)

use std::time::Instant;

use candle_core::{DType, Device, Tensor, D};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_candle::BiddingTrainer;
use colver_core::bid_obs::{self, BID_OBS_DIM, BID_MASK_DIM};
use colver_core::dmc_candle::PoolNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use colver_core::scoring::compute_deal_score;
use colver_core::state::{GameState, Phase};

const MATCH_TARGET: i32 = 2000;
const NUM_PLAY_ACTIONS: usize = 32;
const NUM_BID_ACTIONS: usize = 43;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut num_matches: usize = 500;
    let mut bid_model_path = String::from("models/bid_v2/bid_nn_final.bin");
    let mut play_model_path = String::from("models/play_v2/play_final.bin");
    let mut penalties: Vec<f32> = vec![0.0, 0.05, 0.10, 0.15, 0.20, 0.30];
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--matches" => { i += 1; num_matches = args[i].parse().unwrap(); }
            "--bid-model" => { i += 1; bid_model_path = args[i].clone(); }
            "--play-model" => { i += 1; play_model_path = args[i].clone(); }
            "--penalties" => {
                i += 1;
                penalties = args[i].split(',').map(|s| s.parse().unwrap()).collect();
            }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            _ => { eprintln!("Unknown arg: {}", args[i]); }
        }
        i += 1;
    }

    let device = Device::cuda_if_available(0).expect("No CUDA device");

    // Load bid NN onto GPU
    eprintln!("Loading bid model {}...", bid_model_path);
    let mut bid_trainer = BiddingTrainer::with_layers(3, 512, 1e-4, 0.0, device.clone()).unwrap();
    // Load weights from .bin file
    let bid_bytes = std::fs::read(&bid_model_path).expect("Failed to read bid model");
    let bid_floats: Vec<f32> = bid_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    // Load via varmap - need to use the snapshot format
    // Actually, BiddingTrainer uses safetensors for checkpoints. Let's load .safetensors
    let safetensors_path = bid_model_path.replace(".bin", ".safetensors");
    bid_trainer.varmap.load(&safetensors_path)
        .unwrap_or_else(|e| panic!("Failed to load bid weights from {}: {}", safetensors_path, e));
    eprintln!("  Bid NN loaded ({} params)", bid_floats.len());

    // Load play NN onto GPU
    eprintln!("Loading play model {}...", play_model_path);
    let play_net = PoolNet::with_residual(OBS_DIM_TR, 1024, &device).unwrap();
    let play_bytes = std::fs::read(&play_model_path).expect("Failed to read play model");
    let play_floats: Vec<f32> = play_bytes.chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    play_net.load_weights(&play_floats).unwrap();
    eprintln!("  Play NN loaded ({} params)", play_floats.len());

    let n_penalties = penalties.len();
    let n_pairs = n_penalties * (n_penalties - 1) / 2;
    let total_matches = n_pairs * num_matches * 2; // both directions

    println!("=== GPU-Batched Penalty Sweep ===");
    println!("Penalties: {:?}", penalties);
    println!("Matches/pair/dir: {}", num_matches);
    println!("Total pairs: {}, total matches: {}", n_pairs, total_matches);
    println!("Device: {:?}", device);
    println!();

    // Results matrix: wins[i][j] = how many times penalty[i] beat penalty[j]
    let mut wins = vec![vec![0u32; n_penalties]; n_penalties];
    let mut margins = vec![vec![0i64; n_penalties]; n_penalties];
    let mut games_played = vec![vec![0u32; n_penalties]; n_penalties];

    let start = Instant::now();

    // For each pair of penalties, run matches
    for pi in 0..n_penalties {
        for pj in (pi + 1)..n_penalties {
            let pen_ns = penalties[pi];
            let pen_ew = penalties[pj];

            // Run num_matches in each direction
            for direction in 0..2 {
                let (ns_pen, ew_pen, ns_idx, ew_idx) = if direction == 0 {
                    (pen_ns, pen_ew, pi, pj)
                } else {
                    (pen_ew, pen_ns, pj, pi)
                };

                let (ns_wins, ew_wins, ns_margin) = run_matches_batched(
                    &bid_trainer,
                    &play_net,
                    &device,
                    ns_pen,
                    ew_pen,
                    num_matches,
                    seed + (pi * n_penalties + pj) as u64 * 1000 + direction as u64,
                );

                wins[ns_idx][ew_idx] += ns_wins;
                wins[ew_idx][ns_idx] += ew_wins;
                margins[ns_idx][ew_idx] += ns_margin;
                margins[ew_idx][ns_idx] -= ns_margin;
                games_played[ns_idx][ew_idx] += num_matches as u32;
                games_played[ew_idx][ns_idx] += num_matches as u32;
            }

            let elapsed = start.elapsed().as_secs_f64();
            let pair_done = (pi * (2 * n_penalties - pi - 1) / 2 + (pj - pi));
            eprintln!(
                "  Pair {}/{}: pen {:.2} vs {:.2} done ({:.0}s elapsed)",
                pair_done, n_pairs, pen_ns, pen_ew, elapsed
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    // Print results
    println!("\n{}", "=".repeat(80));
    println!("WIN MATRIX (row win% vs column)");
    println!("{}", "=".repeat(80));

    // Header
    print!("{:<10}", "Penalty");
    for p in &penalties {
        print!(" {:>8}", format!("{:.2}", p));
    }
    println!(" {:>8}", "TOTAL");
    println!("{}", "-".repeat(10 + 9 * (n_penalties + 1)));

    for i in 0..n_penalties {
        print!("{:<10}", format!("{:.2}", penalties[i]));
        let mut total_wins = 0u32;
        let mut total_games = 0u32;
        for j in 0..n_penalties {
            if i == j {
                print!(" {:>8}", "-");
            } else {
                let w = wins[i][j];
                let g = games_played[i][j];
                let pct = if g > 0 { w as f64 / g as f64 * 100.0 } else { 0.0 };
                print!(" {:>7.1}%", pct);
                total_wins += w;
                total_games += g;
            }
        }
        let total_pct = if total_games > 0 {
            total_wins as f64 / total_games as f64 * 100.0
        } else {
            0.0
        };
        println!(" {:>7.1}%", total_pct);
    }

    // Rankings
    println!("\n{}", "=".repeat(60));
    println!("RANKINGS");
    println!("{}", "=".repeat(60));

    let mut rankings: Vec<(usize, f64, i64)> = (0..n_penalties)
        .map(|i| {
            let total_games: u32 = (0..n_penalties)
                .filter(|&j| j != i)
                .map(|j| games_played[i][j])
                .sum();
            let total_wins: u32 = (0..n_penalties)
                .filter(|&j| j != i)
                .map(|j| wins[i][j])
                .sum();
            let total_margin: i64 = (0..n_penalties)
                .filter(|&j| j != i)
                .map(|j| margins[i][j])
                .sum();
            let win_pct = if total_games > 0 {
                total_wins as f64 / total_games as f64 * 100.0
            } else {
                0.0
            };
            (i, win_pct, total_margin)
        })
        .collect();

    rankings.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, win_pct, margin)) in rankings.iter().enumerate() {
        println!(
            "  {}. penalty={:.2}  win {:.1}%  margin {:+}",
            rank + 1,
            penalties[*idx],
            win_pct,
            margin,
        );
    }

    println!("\nWall time: {:.1}s ({:.0} matches/s)", elapsed, total_matches as f64 / elapsed);
}

/// Run `num_matches` matches with NS using `ns_penalty` and EW using `ew_penalty`.
/// Returns (ns_wins, ew_wins, ns_total_margin).
fn run_matches_batched(
    bid_trainer: &BiddingTrainer,
    play_net: &PoolNet,
    device: &Device,
    ns_penalty: f32,
    ew_penalty: f32,
    num_matches: usize,
    seed: u64,
) -> (u32, u32, i64) {
    let mut rng = StdRng::seed_from_u64(seed);
    let mut ns_wins = 0u32;
    let mut ew_wins = 0u32;
    let mut ns_total_margin = 0i64;

    // Run matches in batches
    let batch = num_matches.min(256);
    let mut match_idx = 0;

    while match_idx < num_matches {
        let this_batch = (num_matches - match_idx).min(batch);

        let (nw, ew, nm) = run_match_batch(
            bid_trainer, play_net, device,
            ns_penalty, ew_penalty,
            this_batch, &mut rng,
        );
        ns_wins += nw;
        ew_wins += ew;
        ns_total_margin += nm;
        match_idx += this_batch;
    }

    (ns_wins, ew_wins, ns_total_margin)
}

/// Run a batch of matches concurrently, using GPU for all inference.
fn run_match_batch(
    bid_trainer: &BiddingTrainer,
    play_net: &PoolNet,
    device: &Device,
    ns_penalty: f32,
    ew_penalty: f32,
    batch: usize,
    rng: &mut StdRng,
) -> (u32, u32, i64) {
    // Each match tracks cumulative scores
    let mut ns_scores = vec![0i32; batch];
    let mut ew_scores = vec![0i32; batch];
    let mut finished = vec![false; batch];
    let mut ns_wins = 0u32;
    let mut ew_wins = 0u32;
    let mut ns_margin = 0i64;

    // Play deals until all matches finish
    loop {
        // Count active matches
        let active: Vec<usize> = (0..batch).filter(|i| !finished[*i]).collect();
        if active.is_empty() {
            break;
        }

        // Deal cards for all active matches
        let mut states: Vec<GameState> = Vec::with_capacity(active.len());
        let mut trackings: Vec<EnvTracking> = Vec::with_capacity(active.len());

        for _ in &active {
            let dealer = rng.gen_range(0..4u8);
            let state = GameState::deal_random(dealer, rng);
            let mut tracking = EnvTracking::new();
            tracking.dealer = dealer;
            states.push(state);
            trackings.push(tracking);
        }

        // === BIDDING PHASE ===
        let mut bid_obs_flat = vec![0.0f32; active.len() * BID_OBS_DIM];
        let mut bid_mask_flat = vec![0.0f32; active.len() * BID_MASK_DIM];

        loop {
            // Check if all done bidding
            let still_bidding: Vec<usize> = (0..active.len())
                .filter(|&i| states[i].phase == Phase::Bidding)
                .collect();
            if still_bidding.is_empty() {
                break;
            }

            // Collect obs for bidding games
            for &i in &still_bidding {
                bid_obs::write_bid_observation(
                    &mut bid_obs_flat, i * BID_OBS_DIM,
                    &states[i], &trackings[i].bid_history,
                );
                bid_obs::write_bid_mask(
                    &mut bid_mask_flat, i * BID_MASK_DIM,
                    &states[i],
                );
            }

            // GPU batch forward for ALL active bidding games
            let n_bid = still_bidding.len();
            let mut batch_obs = vec![0.0f32; n_bid * BID_OBS_DIM];
            let mut batch_mask = vec![0.0f32; n_bid * BID_MASK_DIM];
            for (j, &i) in still_bidding.iter().enumerate() {
                batch_obs[j * BID_OBS_DIM..(j + 1) * BID_OBS_DIM]
                    .copy_from_slice(&bid_obs_flat[i * BID_OBS_DIM..(i + 1) * BID_OBS_DIM]);
                batch_mask[j * BID_MASK_DIM..(j + 1) * BID_MASK_DIM]
                    .copy_from_slice(&bid_mask_flat[i * BID_MASK_DIM..(i + 1) * BID_MASK_DIM]);
            }

            let obs_t = Tensor::from_slice(&batch_obs, (n_bid, BID_OBS_DIM), device).unwrap();
            let q_all = bid_trainer.net.forward(&obs_t).unwrap();
            let q_vals: Vec<f32> = q_all.to_vec2::<f32>().unwrap().into_iter().flatten().collect();

            // Select actions with per-game penalty
            for (j, &i) in still_bidding.iter().enumerate() {
                let player = states[i].current_player();
                let is_ns = player == 0 || player == 2;
                let penalty = if is_ns { ns_penalty } else { ew_penalty };

                let q = &q_vals[j * NUM_BID_ACTIONS..(j + 1) * NUM_BID_ACTIONS];
                let mask = &batch_mask[j * BID_MASK_DIM..(j + 1) * BID_MASK_DIM];

                let action = select_bid_with_penalty(q, mask, penalty);
                trackings[i].track_action(&states[i], action);
                states[i].step(action);
            }
        }

        // === PLAYING PHASE ===
        let playing: Vec<usize> = (0..active.len())
            .filter(|&i| states[i].phase == Phase::Playing)
            .collect();

        if !playing.is_empty() {
            // Play 32 steps (all games finish at the same time)
            let mut play_obs_flat = vec![0.0f32; playing.len() * OBS_DIM_TR];
            let mut play_mask_flat = vec![0.0f32; playing.len() * NUM_PLAY_ACTIONS];
            let mut orders: Vec<[u8; 4]> = vec![[0; 4]; playing.len()];

            for _step in 0..32 {
                let still_playing: Vec<usize> = (0..playing.len())
                    .filter(|&j| states[playing[j]].phase == Phase::Playing)
                    .collect();
                if still_playing.is_empty() {
                    break;
                }

                let n_play = still_playing.len();
                let mut batch_obs = vec![0.0f32; n_play * OBS_DIM_TR];
                let mut batch_mask = vec![0.0f32; n_play * NUM_PLAY_ACTIONS];
                let mut batch_orders: Vec<[u8; 4]> = vec![[0; 4]; n_play];

                for (k, &j) in still_playing.iter().enumerate() {
                    let gi = playing[j];
                    dmc_obs::write_observation_tr(
                        &mut batch_obs, k * OBS_DIM_TR,
                        &states[gi], &trackings[gi],
                    );
                    let order = dmc_obs::current_player_order(&states[gi], &trackings[gi]);
                    batch_orders[k] = order;
                    let canonical_mask = dmc_obs::cardset_to_canonical(
                        states[gi].legal_actions() as u32, &order,
                    );
                    for b in 0..NUM_PLAY_ACTIONS {
                        batch_mask[k * NUM_PLAY_ACTIONS + b] =
                            if canonical_mask & (1 << b) != 0 { 1.0 } else { 0.0 };
                    }
                }

                let obs_t = Tensor::from_slice(&batch_obs, (n_play, OBS_DIM_TR), device).unwrap();
                let mask_t = Tensor::from_slice(&batch_mask, (n_play, NUM_PLAY_ACTIONS), device).unwrap();
                let actions = play_net.act_greedy(&obs_t, &mask_t).unwrap();

                for (k, &j) in still_playing.iter().enumerate() {
                    let gi = playing[j];
                    let physical = dmc_obs::card_to_physical(actions[k], &batch_orders[k]);
                    trackings[gi].track_action(&states[gi], physical);
                    states[gi].step(physical);
                }
            }
        }

        // Score all completed deals
        for (local_idx, &match_idx) in active.iter().enumerate() {
            if finished[match_idx] {
                continue;
            }

            let state = &states[local_idx];
            if state.phase == Phase::Done {
                if state.contract.value > 0 {
                    let score = compute_deal_score(state);
                    ns_scores[match_idx] += score.scores[0] as i32;
                    ew_scores[match_idx] += score.scores[1] as i32;
                }
                // else: void deal, no points

                // Check if match is over
                if ns_scores[match_idx] >= MATCH_TARGET || ew_scores[match_idx] >= MATCH_TARGET {
                    finished[match_idx] = true;
                    if ns_scores[match_idx] > ew_scores[match_idx] {
                        ns_wins += 1;
                    } else {
                        ew_wins += 1;
                    }
                    ns_margin += (ns_scores[match_idx] - ew_scores[match_idx]) as i64;
                }
            }
        }
    }

    (ns_wins, ew_wins, ns_margin)
}

/// Select best bid action from Q-values with penalty applied.
fn select_bid_with_penalty(q: &[f32], mask: &[f32], penalty: f32) -> u8 {
    let mut best_action = 0u8;
    let mut best_q = f32::NEG_INFINITY;

    for a in 0..NUM_BID_ACTIONS {
        if mask[a] < 0.5 {
            continue;
        }
        let pen = bid_penalty_for_action(a as u8, penalty);
        let q_adj = q[a] - pen;
        if q_adj > best_q {
            best_q = q_adj;
            best_action = a as u8;
        }
    }
    best_action
}

/// Compute penalty for a bid action.
fn bid_penalty_for_action(action: u8, penalty: f32) -> f32 {
    if action == 0 { return 0.0; }             // PASS
    if action >= 41 { return penalty * 0.5; }   // COINCHE/SURCOINCHE
    if action >= 37 { return penalty * 2.5; }   // CAPOT

    let value_idx = (action - 1) / 4;
    let level = value_idx as f32 / 8.0; // 0.0 at 80, 1.0 at 160
    penalty * level
}
