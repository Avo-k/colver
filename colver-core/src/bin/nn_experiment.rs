/// NN value function evaluation experiment.
///
/// Tests:
/// 1. Accuracy: Load held-out data, compare NN predictions vs actual outcomes
/// 2. Strength: Play matches — Smart IS-MCTS + NN vs Smart IS-MCTS + heuristic rollouts
/// 3. Speed: Measure NN eval speed vs rollout speed
///
/// Usage:
///   cargo run --bin nn_experiment --release --features nn -- <model_path> <num_matches> [--time-limit-ms N] [--data PATH]
use std::time::Instant;

use colver_core::bid_eval::BidFunction;
use colver_core::features::{extract_features, FEATURE_DIM};
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};
use colver_core::value_net::ValueNet;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const MATCH_TARGET: i32 = 2000;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: nn_experiment <model_path> <num_matches> [--time-limit-ms N] [--data PATH]");
        std::process::exit(1);
    }

    let model_path = &args[1];
    let num_matches: u32 = args[2].parse().expect("Invalid num_matches");

    let mut time_limit_ms: u32 = 15;
    let mut data_path: Option<String> = None;

    let mut i = 3;
    while i < args.len() {
        match args[i].as_str() {
            "--time-limit-ms" => {
                i += 1;
                time_limit_ms = args[i].parse().expect("Invalid time limit");
            }
            "--data" => {
                i += 1;
                data_path = Some(args[i].clone());
            }
            _ => {
                eprintln!("Unknown argument: {}", args[i]);
                std::process::exit(1);
            }
        }
        i += 1;
    }

    println!("Loading model from: {}", model_path);
    let mut value_net = ValueNet::load(model_path).expect("Failed to load model");

    // --- Test 1: Accuracy on held-out data ---
    if let Some(ref path) = data_path {
        println!("\n=== Accuracy Test ===");
        accuracy_test(&mut value_net, path);
    }

    // --- Test 2: Speed test ---
    println!("\n=== Speed Test ===");
    speed_test(&mut value_net);

    // --- Test 3: Strength test ---
    println!("\n=== Strength Test: NN vs Rollout ({} matches, {}ms/move) ===", num_matches, time_limit_ms);
    strength_test(&mut value_net, num_matches, time_limit_ms);
}

fn accuracy_test(value_net: &mut ValueNet, data_path: &str) {
    let data = std::fs::read(data_path).expect("Cannot read data file");
    let num_records = u32::from_le_bytes(data[0..4].try_into().unwrap()) as usize;
    let feature_dim = u32::from_le_bytes(data[4..8].try_into().unwrap()) as usize;
    assert_eq!(feature_dim, FEATURE_DIM, "Feature dim mismatch");

    let record_size = (FEATURE_DIM + 1) * 4;
    let mut correct = 0u32;
    let mut total = 0u32;
    let mut total_bce = 0.0f64;

    for r in 0..num_records {
        let offset = 8 + r * record_size;
        let mut features = [0.0f32; FEATURE_DIM];
        for j in 0..FEATURE_DIM {
            let start = offset + j * 4;
            features[j] = f32::from_le_bytes(data[start..start + 4].try_into().unwrap());
        }
        let label_offset = offset + FEATURE_DIM * 4;
        let label = f32::from_le_bytes(data[label_offset..label_offset + 4].try_into().unwrap());

        let pred = value_net.evaluate(&features);

        // Binary accuracy
        let pred_class: f32 = if pred >= 0.5 { 1.0 } else { 0.0 };
        let label_class: f32 = if label >= 0.5 { 1.0 } else { 0.0 };
        if (pred_class - label_class).abs() < 0.01 {
            correct += 1;
        }

        // BCE loss
        let p = pred.clamp(1e-7, 1.0 - 1e-7) as f64;
        let l = label as f64;
        total_bce += -(l * p.ln() + (1.0 - l) * (1.0 - p).ln());

        total += 1;
    }

    let accuracy = correct as f64 / total as f64 * 100.0;
    let avg_bce = total_bce / total as f64;
    println!(
        "  Records: {}, Accuracy: {:.1}%, Avg BCE: {:.4}",
        total, accuracy, avg_bce
    );
}

fn speed_test(value_net: &mut ValueNet) {
    let mut rng = StdRng::seed_from_u64(123);

    // Generate some random playing states
    let mut states = Vec::new();
    for _ in 0..100 {
        let mut state = GameState::deal_random(0, &mut rng);
        // Run through bidding
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let action = colver_core::bid_eval::improved_bid(&state);
            state.step(action);
        }
        if !state.is_terminal() {
            states.push(state);
        }
    }

    if states.is_empty() {
        println!("  No valid states for speed test");
        return;
    }

    // NN eval speed
    let n_evals = 100_000;
    let mut feature_buf = [0.0f32; FEATURE_DIM];
    let start = Instant::now();
    for i in 0..n_evals {
        let state = &states[i % states.len()];
        extract_features(state, &mut feature_buf);
        let _p = value_net.evaluate(&feature_buf);
    }
    let nn_elapsed = start.elapsed();
    let nn_per_sec = n_evals as f64 / nn_elapsed.as_secs_f64();
    let nn_us = nn_elapsed.as_micros() as f64 / n_evals as f64;

    println!(
        "  NN eval: {:.0}/sec ({:.1}μs each), {} evals in {:.1}ms",
        nn_per_sec,
        nn_us,
        n_evals,
        nn_elapsed.as_secs_f64() * 1000.0
    );

    // Rollout speed (for comparison)
    let n_rollouts = 100_000;
    let start = Instant::now();
    for i in 0..n_rollouts {
        let mut s = states[i % states.len()]; // Copy
        colver_core::rollout::rollout_heuristic_play(&mut s, &mut rng);
    }
    let rollout_elapsed = start.elapsed();
    let rollout_per_sec = n_rollouts as f64 / rollout_elapsed.as_secs_f64();
    let rollout_us = rollout_elapsed.as_micros() as f64 / n_rollouts as f64;

    println!(
        "  Rollout: {:.0}/sec ({:.1}μs each), {} rollouts in {:.1}ms",
        rollout_per_sec,
        rollout_us,
        n_rollouts,
        rollout_elapsed.as_secs_f64() * 1000.0
    );

    println!(
        "  Speed ratio: NN is {:.1}x {} than rollout",
        if nn_us < rollout_us {
            rollout_us / nn_us
        } else {
            nn_us / rollout_us
        },
        if nn_us < rollout_us { "faster" } else { "slower" }
    );
}

fn strength_test(value_net: &mut ValueNet, num_matches: u32, time_limit_ms: u32) {
    let mut rng = StdRng::seed_from_u64(42);

    let nn_config = SmartIsMctsConfig {
        determinizations: 100, // high, time-limited
        iterations_per_det: 50,
        time_limit_ms: Some(time_limit_ms),
        use_soft_inference: true,
        bid_function: BidFunction::Improved,
        ..Default::default()
    };

    let rollout_config = SmartIsMctsConfig {
        determinizations: 100,
        iterations_per_det: 50,
        time_limit_ms: Some(time_limit_ms),
        use_soft_inference: true,
        bid_function: BidFunction::Improved,
        ..Default::default()
    };

    let mut nn_wins = 0u32;
    let mut rollout_wins = 0u32;
    let mut margins: Vec<i32> = Vec::new();

    let start = Instant::now();

    for m in 0..num_matches {
        // Alternate sides: even matches = NN plays NS, odd = NN plays EW
        let nn_is_ns = m % 2 == 0;

        let mut ns_cumulative: i32 = 0;
        let mut ew_cumulative: i32 = 0;
        let mut dealer: u8 = rng.gen_range(0..4);

        while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
            let mut state = GameState::deal_random(dealer, &mut rng);

            // Create search instances
            let mut nn_searches: [SmartIsMctsSearch; 2] =
                [SmartIsMctsSearch::new(), SmartIsMctsSearch::new()];
            let mut rollout_searches: [SmartIsMctsSearch; 2] =
                [SmartIsMctsSearch::new(), SmartIsMctsSearch::new()];

            let (nn_players, rollout_players) = if nn_is_ns {
                ([0u8, 2u8], [1u8, 3u8])
            } else {
                ([1u8, 3u8], [0u8, 2u8])
            };

            for (i, &p) in nn_players.iter().enumerate() {
                nn_searches[i].init_deal(&state, p, true);
            }
            for (i, &p) in rollout_players.iter().enumerate() {
                rollout_searches[i].init_deal(&state, p, true);
            }

            while !state.is_terminal() {
                let player = state.current_player();
                let state_before = state;

                let action = if nn_players.contains(&player) {
                    let idx = nn_players.iter().position(|&p| p == player).unwrap();
                    nn_searches[idx].search_with_nn(&state, &nn_config, value_net, &mut rng)
                } else {
                    let idx = rollout_players.iter().position(|&p| p == player).unwrap();
                    rollout_searches[idx].search(&state, &rollout_config, &mut rng)
                };

                for s in &mut nn_searches {
                    s.record_action(&state_before, player, action);
                }
                for s in &mut rollout_searches {
                    s.record_action(&state_before, player, action);
                }
                state.step(action);
            }

            let score = state.deal_score();
            if score.scores[0] != 0 || score.scores[1] != 0 {
                ns_cumulative += score.scores[0] as i32;
                ew_cumulative += score.scores[1] as i32;
            }
            dealer = (dealer + 3) % 4;
        }

        let nn_score = if nn_is_ns { ns_cumulative } else { ew_cumulative };
        let rollout_score = if nn_is_ns { ew_cumulative } else { ns_cumulative };
        let margin = nn_score - rollout_score;
        margins.push(margin);

        if nn_score >= rollout_score {
            nn_wins += 1;
        } else {
            rollout_wins += 1;
        }

        if (m + 1) % 10 == 0 || m + 1 == num_matches {
            let elapsed = start.elapsed().as_secs_f64();
            let avg_margin: f64 = margins.iter().map(|&m| m as f64).sum::<f64>() / margins.len() as f64;
            println!(
                "  Match {}/{}: NN {}-{} Rollout, avg margin {:.0}, {:.1}s elapsed",
                m + 1,
                num_matches,
                nn_wins,
                rollout_wins,
                avg_margin,
                elapsed
            );
        }
    }

    let total = num_matches;
    let nn_wr = nn_wins as f64 / total as f64 * 100.0;
    let avg_margin: f64 = margins.iter().map(|&m| m as f64).sum::<f64>() / margins.len() as f64;

    println!("\n--- Results ---");
    println!("  NN wins: {} ({:.1}%)", nn_wins, nn_wr);
    println!("  Rollout wins: {} ({:.1}%)", rollout_wins, 100.0 - nn_wr);
    println!("  Avg margin (NN perspective): {:.0}", avg_margin);
    println!("  Total time: {:.1}s", start.elapsed().as_secs_f64());
}
