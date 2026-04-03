/// Generate training data for the belief network (card location prediction).
///
/// Plays N games using DMC Q-net for card play and NN bid for bidding.
/// At each play step, records:
///   - Belief observation from current player's perspective (330 floats)
///   - Ground truth target: which player holds each card (u8 × 32)
///   - Unknown mask: bitmask of cards not in observer's hand and not yet played
///
/// Binary format (COLVBL01):
///   Header: magic [u8; 8] + obs_dim: u32 + num_samples: u64
///   Per sample: obs [f32; 330] + target [u8; 32] + unknown_mask: u32
///
/// With `--features parallel`, uses all CPU cores (each thread loads its own
/// model copies). ~10-20x speedup on multi-core machines.
///
/// Usage:
///   cargo run -p colver-core --bin generate_belief_data --release --features parallel -- \
///     --dmc-model models/dmc_final.bin \
///     --bid-model models/bid_nn_final.bin \
///     --games 500000 --output data/belief/belief_train.bin

use std::io::Write;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::belief_obs::{self, BELIEF_OBS_DIM};
use colver_core::bid_eval;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking};
use colver_core::card;
use colver_core::state::{GameState, Phase};

const MAGIC: &[u8; 8] = b"COLVBL01";

/// Per-thread collected samples.
struct ChunkResult {
    obs: Vec<f32>,
    targets: Vec<u8>,
    masks: Vec<u32>,
    num_samples: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut dmc_model_path = String::from("models/dmc_final.bin");
    let mut bid_model_path = String::from("models/bid_nn_final.bin");
    let mut num_games: u64 = 50_000;
    let mut output_path = String::from("data/belief/belief_train.bin");
    let mut seed: u64 = 42;
    let mut num_threads: usize = 0; // 0 = auto-detect

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dmc-model" => { dmc_model_path = args[i + 1].clone(); i += 2; }
            "--bid-model" => { bid_model_path = args[i + 1].clone(); i += 2; }
            "--games" => { num_games = args[i + 1].parse().unwrap(); i += 2; }
            "--output" => { output_path = args[i + 1].clone(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--threads" => { num_threads = args[i + 1].parse().unwrap(); i += 2; }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    // Verify models can load before spawning threads
    {
        let dmc_net = DmcNet::load(&dmc_model_path).unwrap_or_else(|e| {
            eprintln!("Failed to load DMC model: {}", e);
            std::process::exit(1);
        });
        println!("DMC model loaded (obs_dim={}, hidden={})", dmc_net.obs_dim(), dmc_net.hidden());
    }
    {
        match BidNet::load(&bid_model_path) {
            Ok(net) => println!("Bid model loaded (obs_dim={}, dueling={})", net.obs_dim(), net.is_dueling()),
            Err(e) => println!("Bid model not found ({}), using improved_v2", e),
        }
    }

    println!("=== Belief Data Generation ===");
    println!("DMC model:  {}", dmc_model_path);
    println!("Bid model:  {}", bid_model_path);
    println!("Games:      {}", num_games);
    println!("Output:     {}", output_path);
    println!("Seed:       {}", seed);

    let start = Instant::now();

    #[cfg(feature = "parallel")]
    let results = {
        use rayon::prelude::*;

        let n_threads = if num_threads > 0 {
            num_threads
        } else {
            rayon::current_num_threads()
        };

        if num_threads > 0 {
            rayon::ThreadPoolBuilder::new()
                .num_threads(num_threads)
                .build_global()
                .ok();
        }

        println!("Threads:    {}", n_threads);

        // Split games across threads
        let games_per_thread = num_games / n_threads as u64;
        let remainder = num_games % n_threads as u64;

        let chunks: Vec<(u64, u64)> = (0..n_threads)
            .map(|t| {
                let count = games_per_thread + if (t as u64) < remainder { 1 } else { 0 };
                let thread_seed = seed.wrapping_add(t as u64 * 1_000_000);
                (count, thread_seed)
            })
            .collect();

        let dmc_path = &dmc_model_path;
        let bid_path = &bid_model_path;

        let results: Vec<ChunkResult> = chunks
            .into_par_iter()
            .enumerate()
            .map(|(thread_id, (count, thread_seed))| {
                generate_chunk(dmc_path, bid_path, count, thread_seed, thread_id, n_threads)
            })
            .collect();

        results
    };

    #[cfg(not(feature = "parallel"))]
    let results = {
        println!("Threads:    1 (enable --features parallel for multi-threaded)");
        let result = generate_chunk(
            &dmc_model_path, &bid_model_path, num_games, seed, 0, 1,
        );
        vec![result]
    };

    // Merge results
    let total_samples: u64 = results.iter().map(|r| r.num_samples).sum();
    let total_obs_len: usize = results.iter().map(|r| r.obs.len()).sum();
    let total_targets_len: usize = results.iter().map(|r| r.targets.len()).sum();

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\nGeneration complete: {} games, {} samples in {:.1}s ({:.0} games/s)",
        num_games, total_samples, elapsed, num_games as f64 / elapsed,
    );
    println!(
        "Avg samples/game: {:.1}",
        total_samples as f64 / num_games as f64,
    );

    // Write binary file
    println!("Writing to {}...", output_path);
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    // Pre-allocate the full output buffer to avoid per-sample allocation
    let total_bytes = 8 + 4 + 8 + total_samples as usize * (BELIEF_OBS_DIM * 4 + 32 + 4);
    let mut out_buf = Vec::with_capacity(total_bytes);

    // Header
    out_buf.extend_from_slice(MAGIC);
    out_buf.extend_from_slice(&(BELIEF_OBS_DIM as u32).to_le_bytes());
    out_buf.extend_from_slice(&total_samples.to_le_bytes());

    // Samples — write chunk by chunk
    for result in &results {
        for i in 0..result.num_samples as usize {
            let obs_start = i * BELIEF_OBS_DIM;
            let obs_end = obs_start + BELIEF_OBS_DIM;
            for &f in &result.obs[obs_start..obs_end] {
                out_buf.extend_from_slice(&f.to_le_bytes());
            }
            let target_start = i * 32;
            out_buf.extend_from_slice(&result.targets[target_start..target_start + 32]);
            out_buf.extend_from_slice(&result.masks[i].to_le_bytes());
        }
    }

    std::fs::write(&output_path, &out_buf).unwrap();

    let file_size = out_buf.len();
    println!(
        "Written {} ({:.1} MB, {:.0} bytes/sample)",
        output_path,
        file_size as f64 / (1024.0 * 1024.0),
        file_size as f64 / total_samples as f64,
    );
}

fn generate_chunk(
    dmc_model_path: &str,
    bid_model_path: &str,
    num_games: u64,
    seed: u64,
    thread_id: usize,
    num_threads: usize,
) -> ChunkResult {
    // Each thread loads its own model copies
    let mut dmc_net = DmcNet::load(dmc_model_path).unwrap();
    let mut bid_net = BidNet::load(bid_model_path).ok();

    let mut rng = StdRng::seed_from_u64(seed);

    // Pre-allocate estimate: ~30 samples/game
    let est_samples = (num_games as usize) * 31;
    let mut all_obs: Vec<f32> = Vec::with_capacity(est_samples * BELIEF_OBS_DIM);
    let mut all_targets: Vec<u8> = Vec::with_capacity(est_samples * 32);
    let mut all_masks: Vec<u32> = Vec::with_capacity(est_samples);
    let mut total_samples: u64 = 0;

    let mut obs_buf = vec![0.0f32; BELIEF_OBS_DIM];
    let mut dmc_obs_buf = vec![0.0f32; dmc_obs::OBS_DIM];

    let report_interval = if num_threads == 1 { 10_000u64 } else { 50_000 };
    let start = Instant::now();

    for game_idx in 0..num_games {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut tracking = EnvTracking::new();
        tracking.dealer = dealer;

        let true_hands = state.hands;

        while !state.is_terminal() {
            let player = state.current_player();

            if state.phase == Phase::Playing {
                let observer = player;

                belief_obs::write_belief_observation(&mut obs_buf, 0, &state, &tracking, observer);

                // Targets use player-relative IDs: 0=me, 1=left, 2=partner, 3=right
                let mut target = [0u8; 32];
                for p in 0..4u8 {
                    for c in 0..32u8 {
                        if true_hands[p as usize] & (1u32 << c) != 0 {
                            let rel_p = (p + 4 - observer) % 4;
                            target[c as usize] = rel_p;
                        }
                    }
                }

                let observer_hand = state.hands[observer as usize];
                let mut played = state.played_cards;
                for j in 0..4 {
                    let c = state.current_trick[j];
                    if c != card::EMPTY {
                        played |= 1u32 << c;
                    }
                }
                let unknown_mask = !observer_hand & !played;

                if unknown_mask != 0 {
                    all_obs.extend_from_slice(&obs_buf);
                    all_targets.extend_from_slice(&target);
                    all_masks.push(unknown_mask);
                    total_samples += 1;
                }
            }

            let action = if state.phase == Phase::Bidding {
                bid_action(&state, &tracking.bid_history, &mut bid_net)
            } else {
                dmc_action(&state, &tracking, &mut dmc_net, &mut dmc_obs_buf)
            };

            tracking.track_action(&state, action);
            state.step(action);
        }

        if thread_id == 0 && ((game_idx + 1) % report_interval == 0 || game_idx + 1 == num_games) {
            let elapsed = start.elapsed().as_secs_f64();
            let total_games_est = (game_idx + 1) as f64 * num_threads as f64;
            let games_per_sec = total_games_est / elapsed;
            let total_samples_est = total_samples as f64 * num_threads as f64;
            println!(
                "[~{:.0}k/{}k] ~{:.0}k samples ({:.0} games/s, {:.1}s)",
                total_games_est / 1000.0,
                num_games as f64 * num_threads as f64 / 1000.0,
                total_samples_est / 1000.0,
                games_per_sec,
                elapsed,
            );
        }
    }

    ChunkResult {
        obs: all_obs,
        targets: all_targets,
        masks: all_masks,
        num_samples: total_samples,
    }
}

fn bid_action(state: &GameState, bid_history: &[(u8, u8)], bid_net: &mut Option<BidNet>) -> u8 {
    if let Some(ref mut net) = bid_net {
        let obs = bid_obs::make_bid_observation(state, bid_history);
        let legal = state.legal_actions();
        net.best_action_fast(&obs, legal)
    } else {
        bid_eval::improved_v2_bid(state)
    }
}

fn dmc_action(
    state: &GameState,
    tracking: &EnvTracking,
    dmc_net: &mut DmcNet,
    obs_buf: &mut [f32],
) -> u8 {
    dmc_obs::write_observation(obs_buf, 0, state, tracking);
    let legal_mask = state.legal_actions() as u32;
    let (best, _) = dmc_net.best_action(obs_buf, legal_mask);
    best
}
