/// Evaluate a trained belief network.
///
/// Two modes:
/// A. Offline accuracy test: load model + validation data, report accuracy/CE/calibration.
/// B. Online match play: IS-DD + NN beliefs vs IS-DD + heuristic CardBeliefs.
///
/// Usage:
///   # Offline eval
///   cargo run -p colver-core --bin belief_eval --release -- \
///     --model models/belief_net.bin --data data/belief_train.bin --mode offline
///
///   # Online match play
///   cargo run -p colver-core --bin belief_eval --release -- \
///     --model models/belief_net.bin --bid-model models/bid_nn_final.bin \
///     --mode match --matches 100 --time-ms 20

use std::io::Read;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::belief_net::BeliefNet;
use colver_core::belief_obs::BELIEF_OBS_DIM;
use colver_core::bid_eval;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::state::{GameState, Phase};

const MAGIC: &[u8; 8] = b"COLVBL01";

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::from("models/belief_net.bin");
    let mut data_path = String::new();
    let mut bid_model_path = String::new();
    let mut mode = String::from("offline");
    let mut matches: usize = 100;
    let mut time_ms: u32 = 20;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model_path = args[i + 1].clone(); i += 2; }
            "--data" => { data_path = args[i + 1].clone(); i += 2; }
            "--bid-model" => { bid_model_path = args[i + 1].clone(); i += 2; }
            "--mode" => { mode = args[i + 1].clone(); i += 2; }
            "--matches" => { matches = args[i + 1].parse().unwrap(); i += 2; }
            "--time-ms" => { time_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    match mode.as_str() {
        "offline" => run_offline_eval(&model_path, &data_path),
        "match" => run_match_eval(&model_path, &bid_model_path, matches, time_ms, seed),
        other => {
            eprintln!("Unknown mode: {} (expected 'offline' or 'match')", other);
            std::process::exit(1);
        }
    }
}

fn run_offline_eval(model_path: &str, data_path: &str) {
    if data_path.is_empty() {
        eprintln!("--data is required for offline mode");
        std::process::exit(1);
    }

    println!("=== Offline Belief Evaluation ===");
    println!("Model: {}", model_path);
    println!("Data:  {}", data_path);

    let mut net = BeliefNet::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });
    println!("Model loaded (obs_dim={}, hidden={})", net.obs_dim(), net.hidden());

    // Load data
    let mut file = std::fs::File::open(data_path).unwrap();
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, MAGIC);

    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4).unwrap();
    let obs_dim = u32::from_le_bytes(buf4) as usize;
    assert_eq!(obs_dim, BELIEF_OBS_DIM);

    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8).unwrap();
    let num_samples = u64::from_le_bytes(buf8) as usize;
    println!("Evaluating on {} samples...", num_samples);

    let sample_bytes = BELIEF_OBS_DIM * 4 + 32 + 4;
    let mut raw = vec![0u8; sample_bytes];

    let mut correct = 0u64;
    let mut total = 0u64;
    let mut ce_sum = 0.0f64;
    let mut ce_count = 0u64;

    // Per-trick accuracy (8 tricks, but we track by completed tricks 0-7)
    let mut trick_correct = [0u64; 8];
    let mut trick_total = [0u64; 8];

    // Calibration: 10 bins of predicted confidence
    let mut cal_correct = [0u64; 10];
    let mut cal_total = [0u64; 10];

    let start = Instant::now();
    let max_eval = num_samples.min(200_000); // cap to avoid excessive time

    for sample_idx in 0..max_eval {
        file.read_exact(&mut raw).unwrap();

        // Parse obs
        let mut obs = [0.0f32; BELIEF_OBS_DIM];
        for j in 0..BELIEF_OBS_DIM {
            let off = j * 4;
            obs[j] = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
        }

        // Parse target
        let target_off = BELIEF_OBS_DIM * 4;
        let target = &raw[target_off..target_off + 32];

        // Parse mask
        let mask_off = target_off + 32;
        let mask = u32::from_le_bytes([
            raw[mask_off], raw[mask_off + 1], raw[mask_off + 2], raw[mask_off + 3],
        ]);

        // Forward pass
        let logits = net.evaluate(&obs);

        // Trick number from obs (Block 11, offset 328)
        let trick_num = (obs[BELIEF_OBS_DIM - 2] * 8.0).round() as usize;
        let trick_idx = trick_num.min(7);

        // Per-card evaluation
        for c in 0..32u32 {
            if mask & (1 << c) == 0 {
                continue;
            }

            let base = c as usize * 4;
            let true_player = target[c as usize] as usize;

            // Softmax
            let mut max_l = f32::NEG_INFINITY;
            for p in 0..4 {
                if logits[base + p] > max_l {
                    max_l = logits[base + p];
                }
            }
            let mut exp_sum = 0.0f32;
            let mut exps = [0.0f32; 4];
            for p in 0..4 {
                exps[p] = (logits[base + p] - max_l).exp();
                exp_sum += exps[p];
            }
            let probs: Vec<f32> = exps.iter().map(|&e| e / exp_sum).collect();

            // Accuracy (argmax)
            let mut pred = 0;
            for p in 1..4 {
                if probs[p] > probs[pred] {
                    pred = p;
                }
            }
            if pred == true_player {
                correct += 1;
                trick_correct[trick_idx] += 1;
            }
            total += 1;
            trick_total[trick_idx] += 1;

            // Cross-entropy
            let true_prob = probs[true_player].max(1e-10);
            ce_sum += -(true_prob as f64).ln();
            ce_count += 1;

            // Calibration
            let confidence = probs[pred];
            let bin = ((confidence * 10.0) as usize).min(9);
            cal_total[bin] += 1;
            if pred == true_player {
                cal_correct[bin] += 1;
            }
        }

        if (sample_idx + 1) % 50_000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let acc = correct as f64 / total as f64 * 100.0;
            let ce = ce_sum / ce_count as f64;
            println!("  [{}/{}] acc={:.2}% CE={:.4} ({:.1}s)", sample_idx + 1, max_eval, acc, ce, elapsed);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== Results ===");
    println!("Samples evaluated: {}", max_eval);
    println!("Per-card accuracy: {:.2}% ({}/{})", correct as f64 / total as f64 * 100.0, correct, total);
    println!("Cross-entropy:     {:.4}", ce_sum / ce_count as f64);
    println!("Random baseline:   {:.2}% (33.3%)", 100.0 / 3.0);
    println!("Elapsed:           {:.1}s", elapsed);

    println!("\n--- Accuracy by trick number ---");
    for t in 0..8 {
        if trick_total[t] > 0 {
            let acc = trick_correct[t] as f64 / trick_total[t] as f64 * 100.0;
            println!("  Trick {}: {:.1}% ({}/{})", t, acc, trick_correct[t], trick_total[t]);
        }
    }

    println!("\n--- Calibration ---");
    println!("  {:>12} {:>10} {:>10} {:>8}", "Confidence", "Correct", "Total", "Actual%");
    for bin in 0..10 {
        if cal_total[bin] > 0 {
            let actual = cal_correct[bin] as f64 / cal_total[bin] as f64 * 100.0;
            println!(
                "  {:>4.0}-{:>4.0}%  {:>10} {:>10} {:>7.1}%",
                bin as f64 * 10.0, (bin + 1) as f64 * 10.0,
                cal_correct[bin], cal_total[bin], actual,
            );
        }
    }
}

fn run_match_eval(
    model_path: &str,
    bid_model_path: &str,
    num_matches: usize,
    time_ms: u32,
    seed: u64,
) {
    println!("=== Online Match Evaluation ===");
    println!("Belief model: {}", model_path);
    println!("Bid model:    {}", bid_model_path);
    println!("Matches:      {} (duplicate pairs = {})", num_matches, num_matches / 2);
    println!("Time/move:    {}ms", time_ms);

    // Load bid model
    let mut bid_net = if !bid_model_path.is_empty() {
        match BidNet::load(bid_model_path) {
            Ok(net) => {
                println!("Bid model loaded");
                Some(net)
            }
            Err(e) => {
                println!("Bid model not found ({}), using improved_v2", e);
                None
            }
        }
    } else {
        None
    };

    // Duplicate matching: each pair uses same seed, teams swapped
    let num_pairs = num_matches / 2;
    let mut nn_wins = 0u32;
    let mut heuristic_wins = 0u32;
    let mut nn_margin_sum = 0.0f64;

    let start = Instant::now();

    for pair in 0..num_pairs {
        let pair_seed = seed + pair as u64;

        // Match A: NN beliefs = team 0, heuristic = team 1
        let mut rng_a = StdRng::seed_from_u64(pair_seed);
        let (nn_pts_a, heur_pts_a) = play_match(
            model_path, &mut bid_net, time_ms, 0, &mut rng_a,
        );
        if nn_pts_a >= 2000.0 {
            nn_wins += 1;
        } else {
            heuristic_wins += 1;
        }
        nn_margin_sum += (nn_pts_a - heur_pts_a) as f64;

        // Match B: NN beliefs = team 1, heuristic = team 0
        let mut rng_b = StdRng::seed_from_u64(pair_seed);
        let (nn_pts_b, heur_pts_b) = play_match(
            model_path, &mut bid_net, time_ms, 1, &mut rng_b,
        );
        if nn_pts_b >= 2000.0 {
            nn_wins += 1;
        } else {
            heuristic_wins += 1;
        }
        nn_margin_sum += (nn_pts_b - heur_pts_b) as f64;

        if (pair + 1) % 10 == 0 || pair + 1 == num_pairs {
            let total = (pair + 1) * 2;
            let wr = nn_wins as f64 / total as f64 * 100.0;
            let margin = nn_margin_sum / total as f64;
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "  [{}/{}] NN: {}W-{}L ({:.1}%) margin={:+.0} ({:.1}s)",
                total, num_pairs * 2, nn_wins, heuristic_wins, wr, margin, elapsed,
            );
        }
    }

    let total_matches = num_pairs * 2;
    let elapsed = start.elapsed().as_secs_f64();
    println!("\n=== Results ===");
    println!(
        "NN beliefs:       {}W-{}L ({:.1}%)",
        nn_wins, heuristic_wins,
        nn_wins as f64 / total_matches as f64 * 100.0,
    );
    println!(
        "Avg margin:       {:+.0}",
        nn_margin_sum / total_matches as f64,
    );
    println!("Elapsed:          {:.1}s ({:.1}s/match)", elapsed, elapsed / total_matches as f64);
}

/// Play a match to 2000 between NN-belief IS-DD and heuristic IS-DD.
/// Returns (nn_team_points, heuristic_team_points).
fn play_match(
    belief_model_path: &str,
    bid_net: &mut Option<BidNet>,
    time_ms: u32,
    nn_team: u8,
    rng: &mut StdRng,
) -> (f32, f32) {
    let nn_config = IsDdConfig {
        time_limit_ms: Some(time_ms),
        use_nn_beliefs: true,
        ..Default::default()
    };
    let heur_config = IsDdConfig {
        time_limit_ms: Some(time_ms),
        use_nn_beliefs: false,
        ..Default::default()
    };

    // Create searches for all 4 players
    let mut searches: Vec<IsDdSearch> = (0..4).map(|_| IsDdSearch::new()).collect();

    // Load belief net for NN team players
    for p in 0..4u8 {
        if GameState::player_team(p) == nn_team {
            searches[p as usize].load_belief_net(belief_model_path).unwrap_or_else(|e| {
                eprintln!("Failed to load belief net: {}", e);
                std::process::exit(1);
            });
        }
    }

    let mut nn_total = 0.0f32;
    let mut heur_total = 0.0f32;

    for _ in 0..50 {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, rng);

        // Init all searches
        for p in 0..4u8 {
            searches[p as usize].init_deal(&state, p, true);
        }

        while !state.is_terminal() {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if state.phase == Phase::Bidding {
                bid_action(&state, bid_net)
            } else if team == nn_team {
                searches[player as usize].search(&state, &nn_config, rng)
            } else {
                searches[player as usize].search(&state, &heur_config, rng)
            };

            // Record for all players
            for s in searches.iter_mut() {
                s.record_action(&state, player, action);
            }
            state.step(action);
        }

        let rewards = state.rewards();
        nn_total += rewards[nn_team as usize];
        heur_total += rewards[1 - nn_team as usize];

        if nn_total >= 2000.0 || heur_total >= 2000.0 {
            break;
        }
    }

    (nn_total, heur_total)
}

fn bid_action(state: &GameState, bid_net: &mut Option<BidNet>) -> u8 {
    if let Some(ref mut net) = bid_net {
        // We don't track bid history here, use empty for simplicity
        // (the IS-DD searches handle their own bidding)
        let obs = bid_obs::make_bid_observation(state, &[]);
        let legal = state.legal_actions();
        net.best_action_fast(&obs, legal)
    } else {
        bid_eval::improved_v2_bid(state)
    }
}
