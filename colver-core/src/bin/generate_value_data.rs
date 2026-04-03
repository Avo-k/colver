/// Data generation for NN value function training.
///
/// Runs self-play deals, records play-phase positions with features + outcome labels.
///
/// Binary output format:
///   [u32 num_records] [u32 feature_dim] [records...]
///   Each record: FEATURE_DIM × f32 features + 1 × f32 label
///
/// Usage:
///   cargo run --bin generate_value_data --release --features nn -- <num_deals> <output_path> [--fast]
///
/// --fast: Use heuristic play instead of IS-MCTS (much faster, lower quality data)
use std::io::Write;

use colver_core::bid_eval::improved_bid;
use colver_core::features::{extract_features, FEATURE_DIM};
use colver_core::scoring::compute_deal_score;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};

use rand::rngs::StdRng;
use rand::SeedableRng;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 3 {
        eprintln!("Usage: generate_value_data <num_deals> <output_path> [--fast]");
        std::process::exit(1);
    }

    let num_deals: u32 = args[1].parse().expect("Invalid num_deals");
    let output_path = &args[2];
    let fast_mode = args.iter().any(|a| a == "--fast");

    eprintln!(
        "Generating data: {} deals, output: {}, mode: {}",
        num_deals,
        output_path,
        if fast_mode { "fast (heuristic)" } else { "IS-MCTS" }
    );

    let mut rng = StdRng::seed_from_u64(42);
    let mut all_records: Vec<([f32; FEATURE_DIM], f32)> = Vec::new();

    let ismcts_config = SmartIsMctsConfig {
        determinizations: 20,
        iterations_per_det: 50,
        time_limit_ms: None,
        ..Default::default()
    };

    let start = std::time::Instant::now();
    let mut deals_completed = 0u32;
    let mut void_deals = 0u32;

    for deal_idx in 0..num_deals {
        let dealer = (deal_idx % 4) as u8;
        let mut state = GameState::deal_random(dealer, &mut rng);

        // Bidding phase: use improved_bid for all players
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let action = improved_bid(&state);
            state.step(action);
        }

        if state.is_terminal() {
            void_deals += 1;
            deals_completed += 1;
            if deals_completed % 1000 == 0 {
                let elapsed = start.elapsed().as_secs_f64();
                eprintln!(
                    "  {} deals ({:.0}/sec), {} records, {} void",
                    deals_completed,
                    deals_completed as f64 / elapsed,
                    all_records.len(),
                    void_deals
                );
            }
            continue;
        }

        // Play phase: collect positions
        let mut positions: Vec<[f32; FEATURE_DIM]> = Vec::new();

        if fast_mode {
            // Fast mode: heuristic play, extract features at each step
            while !state.is_terminal() {
                debug_assert_eq!(state.phase, Phase::Playing);
                let mut feature_buf = [0.0f32; FEATURE_DIM];
                extract_features(&state, &mut feature_buf);
                positions.push(feature_buf);

                // Heuristic play makes a deterministic choice
                let action = colver_core::rollout::heuristic_play_action(&state);
                state.step(action);
            }
        } else {
            // IS-MCTS mode: use Smart IS-MCTS for all players
            let mut searches: [SmartIsMctsSearch; 4] = [
                SmartIsMctsSearch::new(),
                SmartIsMctsSearch::new(),
                SmartIsMctsSearch::new(),
                SmartIsMctsSearch::new(),
            ];
            for p in 0..4 {
                searches[p].init_deal(&state, p as u8, true);
            }

            while !state.is_terminal() {
                debug_assert_eq!(state.phase, Phase::Playing);
                let mut feature_buf = [0.0f32; FEATURE_DIM];
                extract_features(&state, &mut feature_buf);
                positions.push(feature_buf);

                let player = state.current_player() as usize;
                let state_before = state;
                let action = searches[player].search(&state, &ismcts_config, &mut rng);

                for s in &mut searches {
                    s.record_action(&state_before, player as u8, action);
                }
                state.step(action);
            }
        }

        // Label all positions with outcome
        debug_assert!(state.is_terminal());
        let label = if state.contract.value == 0 {
            0.5 // void deal — shouldn't happen since we checked above
        } else {
            let score = compute_deal_score(&state);
            if score.scores[0] > score.scores[1] {
                1.0
            } else if score.scores[1] > score.scores[0] {
                0.0
            } else {
                0.5
            }
        };

        for features in positions {
            all_records.push((features, label));
        }

        deals_completed += 1;
        if deals_completed % 100 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            eprintln!(
                "  {} deals ({:.1}/sec), {} records, {} void",
                deals_completed,
                deals_completed as f64 / elapsed,
                all_records.len(),
                void_deals
            );
        }
    }

    let elapsed = start.elapsed();
    eprintln!(
        "\nDone: {} deals in {:.1}s ({:.1}/sec), {} records, {} void deals",
        deals_completed,
        elapsed.as_secs_f64(),
        deals_completed as f64 / elapsed.as_secs_f64(),
        all_records.len(),
        void_deals
    );

    // Write binary output
    let mut file = std::fs::File::create(output_path).expect("Cannot create output file");
    let num_records = all_records.len() as u32;
    let feature_dim = FEATURE_DIM as u32;

    file.write_all(&num_records.to_le_bytes()).unwrap();
    file.write_all(&feature_dim.to_le_bytes()).unwrap();

    for (features, label) in &all_records {
        for &f in features.iter() {
            file.write_all(&f.to_le_bytes()).unwrap();
        }
        file.write_all(&label.to_le_bytes()).unwrap();
    }

    eprintln!(
        "Wrote {} records ({} bytes) to {}",
        num_records,
        8 + num_records as u64 * (FEATURE_DIM as u64 + 1) * 4,
        output_path
    );
}
