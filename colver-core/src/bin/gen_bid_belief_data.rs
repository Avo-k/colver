/// Generate training data for the bid belief network (card location from auction).
///
/// Deals random hands, runs bid_v2 (NN bidder) auctions, records one sample
/// per observer per bid step. No play model needed — bidding only.
///
/// Binary format (COLVBB01):
///   Header: magic [u8; 8] + obs_dim: u32 + num_samples: u64
///   Per sample: obs [f32; 108] + target [u8; 32] + unknown_mask: u32
///
/// Usage:
///   cargo run -p colver-core --bin gen_bid_belief_data --release --features parallel -- \
///     --bid-model models/bid_v2/bid_nn_final.bin --bid-hidden 512 \
///     --deals 500000 --output data/belief/bid_belief_train.bin

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::belief_obs::{self, BID_BELIEF_OBS_DIM};
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::state::{GameState, Phase};

const MAGIC: &[u8; 8] = b"COLVBB01";

struct ChunkResult {
    obs: Vec<f32>,
    targets: Vec<u8>,
    masks: Vec<u32>,
    num_samples: u64,
    num_deals: u64,
    void_deals: u64,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut bid_model_path = String::from("models/bid_v2/bid_nn_final.bin");
    let mut bid_hidden: usize = 512;
    let mut num_deals: u64 = 500_000;
    let mut output_path = String::from("data/belief/bid_belief_train.bin");
    let mut seed: u64 = 42;
    let mut num_threads: usize = 0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bid-model" => { bid_model_path = args[i + 1].clone(); i += 2; }
            "--bid-hidden" => { bid_hidden = args[i + 1].parse().unwrap(); i += 2; }
            "--deals" => { num_deals = args[i + 1].parse().unwrap(); i += 2; }
            "--output" => { output_path = args[i + 1].clone(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--threads" => { num_threads = args[i + 1].parse().unwrap(); i += 2; }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    // Verify model loads
    {
        let net = BidNet::load_with_hidden(&bid_model_path, bid_hidden)
            .unwrap_or_else(|e| {
                eprintln!("Failed to load bid model: {}", e);
                std::process::exit(1);
            });
        println!("Bid model loaded (obs_dim={}, dueling={})", net.obs_dim(), net.is_dueling());
    }

    println!("=== Bid Belief Data Generation ===");
    println!("Bid model:  {} (hidden={})", bid_model_path, bid_hidden);
    println!("Deals:      {}", num_deals);
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

        let deals_per_thread = num_deals / n_threads as u64;
        let remainder = num_deals % n_threads as u64;

        let chunks: Vec<(u64, u64)> = (0..n_threads)
            .map(|t| {
                let count = deals_per_thread + if (t as u64) < remainder { 1 } else { 0 };
                let thread_seed = seed.wrapping_add(t as u64 * 1_000_000);
                (count, thread_seed)
            })
            .collect();

        let bid_path = &bid_model_path;

        let results: Vec<ChunkResult> = chunks
            .into_par_iter()
            .enumerate()
            .map(|(thread_id, (count, thread_seed))| {
                generate_chunk(bid_path, bid_hidden, count, thread_seed, thread_id, n_threads)
            })
            .collect();

        results
    };

    #[cfg(not(feature = "parallel"))]
    let results = {
        println!("Threads:    1 (enable --features parallel for multi-threaded)");
        let result = generate_chunk(
            &bid_model_path, bid_hidden, num_deals, seed, 0, 1,
        );
        vec![result]
    };

    // Merge results
    let total_samples: u64 = results.iter().map(|r| r.num_samples).sum();
    let total_deals: u64 = results.iter().map(|r| r.num_deals).sum();
    let void_deals: u64 = results.iter().map(|r| r.void_deals).sum();

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\nGeneration complete: {} deals ({} void), {} samples in {:.1}s ({:.0} deals/s)",
        total_deals, void_deals, total_samples, elapsed, total_deals as f64 / elapsed,
    );
    println!(
        "Avg samples/deal: {:.1} (excl. void: {:.1})",
        total_samples as f64 / total_deals as f64,
        total_samples as f64 / (total_deals - void_deals).max(1) as f64,
    );

    // Write binary file
    println!("Writing to {}...", output_path);
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let sample_bytes = BID_BELIEF_OBS_DIM * 4 + 32 + 4; // obs + target + mask
    let total_bytes = 8 + 4 + 8 + total_samples as usize * sample_bytes;
    let mut out_buf = Vec::with_capacity(total_bytes);

    // Header
    out_buf.extend_from_slice(MAGIC);
    out_buf.extend_from_slice(&(BID_BELIEF_OBS_DIM as u32).to_le_bytes());
    out_buf.extend_from_slice(&total_samples.to_le_bytes());

    // Samples
    for result in &results {
        for i in 0..result.num_samples as usize {
            let obs_start = i * BID_BELIEF_OBS_DIM;
            let obs_end = obs_start + BID_BELIEF_OBS_DIM;
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
        file_size as f64 / total_samples.max(1) as f64,
    );
}

fn generate_chunk(
    bid_model_path: &str,
    bid_hidden: usize,
    num_deals: u64,
    seed: u64,
    thread_id: usize,
    num_threads: usize,
) -> ChunkResult {
    let mut bid_net = BidNet::load_with_hidden(bid_model_path, bid_hidden)
        .unwrap();

    let mut rng = StdRng::seed_from_u64(seed);

    // ~25 samples per deal (5 bid steps x 4 observers, minus void deals)
    let est_samples = (num_deals as usize) * 25;
    let mut all_obs: Vec<f32> = Vec::with_capacity(est_samples * BID_BELIEF_OBS_DIM);
    let mut all_targets: Vec<u8> = Vec::with_capacity(est_samples * 32);
    let mut all_masks: Vec<u32> = Vec::with_capacity(est_samples);
    let mut total_samples: u64 = 0;
    let mut void_deals: u64 = 0;

    let mut obs_buf = [0.0f32; BID_BELIEF_OBS_DIM];

    let report_interval = if num_threads == 1 { 50_000u64 } else { 200_000 };
    let start = Instant::now();

    for deal_idx in 0..num_deals {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, &mut rng);
        let true_hands = state.hands;

        let mut bid_history: Vec<(u8, u8)> = Vec::with_capacity(12);

        // Run bidding phase, recording at each step
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let player = state.current_player();

            // Record one sample per observer at this bid step
            for observer in 0..4u8 {
                write_bid_belief_sample(
                    &state, &bid_history, observer, &true_hands,
                    &mut obs_buf, &mut all_obs, &mut all_targets, &mut all_masks,
                );
                total_samples += 1;
            }

            // Take bid action
            let obs = bid_obs::make_bid_observation(&state, &bid_history);
            let legal = state.legal_actions();
            let action = bid_net.best_action_fast(&obs, legal);

            bid_history.push((player, action));
            state.step(action);
        }

        // Check if void deal (no contract set)
        if state.contract.value == 0 {
            void_deals += 1;
        }

        // Progress report
        if thread_id == 0 && ((deal_idx + 1) % report_interval == 0 || deal_idx + 1 == num_deals) {
            let elapsed = start.elapsed().as_secs_f64();
            let total_deals_est = (deal_idx + 1) as f64 * num_threads as f64;
            let deals_per_sec = total_deals_est / elapsed;
            let total_samples_est = total_samples as f64 * num_threads as f64;
            eprintln!(
                "[~{:.0}k/{:.0}k] ~{:.0}k samples ({:.0} deals/s, {:.1}s)",
                total_deals_est / 1000.0,
                num_deals as f64 * num_threads as f64 / 1000.0,
                total_samples_est / 1000.0,
                deals_per_sec,
                elapsed,
            );
        }
    }

    ChunkResult {
        obs: all_obs,
        targets: all_targets,
        masks: all_masks,
        num_samples: total_samples,
        num_deals,
        void_deals,
    }
}

fn write_bid_belief_sample(
    state: &GameState,
    bid_history: &[(u8, u8)],
    observer: u8,
    true_hands: &[u32; 4],
    obs_buf: &mut [f32; BID_BELIEF_OBS_DIM],
    all_obs: &mut Vec<f32>,
    all_targets: &mut Vec<u8>,
    all_masks: &mut Vec<u32>,
) {
    belief_obs::write_bid_belief_obs(obs_buf, 0, state, bid_history, observer);

    // Target: 3-class indices (0=left, 1=partner, 2=right)
    // Observer's own cards get 0 (masked out anyway)
    let mut target = [0u8; 32];
    for p in 0..4u8 {
        for c in 0..32u8 {
            if true_hands[p as usize] & (1u32 << c) != 0 {
                let rel_p = (p + 4 - observer) % 4;
                target[c as usize] = if rel_p == 0 { 0 } else { rel_p - 1 };
            }
        }
    }

    // Unknown mask: all cards not in observer's hand (always 24 cards during bidding)
    let observer_hand = state.hands[observer as usize];
    let unknown_mask = !observer_hand;

    all_obs.extend_from_slice(obs_buf);
    all_targets.extend_from_slice(&target);
    all_masks.push(unknown_mask);
}
