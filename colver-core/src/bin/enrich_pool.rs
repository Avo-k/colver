//! Enrich a DD pool with real DouDou50 rollout points using GPU-batched inference.
//!
//! For each deal × 4 suits, plays out the game with DouDou50 on GPU
//! and records the actual NS points alongside the DD-optimal points.
//!
//! Output format: "COLVDR01" + count + per-deal (dealer[1] + hands[16] + dd_pts[4] + real_pts[4]) = 25B/deal
//!
//! Usage:
//!   cargo run --bin enrich_pool --release --features dmc_train -- [options]
//!
//! Options:
//!   --pool PATH        Input pool file (default: data/pools/dd_2.5M.bin)
//!   --output PATH      Output enriched pool (default: data/pools/dd_pool_enriched.bin)
//!   --deals N          Number of deals to enrich (default: 100000)
//!   --batch N          GPU batch size (default: 4096)
//!   --model PATH       DMC model (default: models/play_v2/play_final.bin)
//!   --seed N           RNG seed for deal sampling (default: 42)

use std::time::Instant;

use candle_core::{DType, Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_train_env::DealPool;
use colver_core::dmc_candle::PoolNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use colver_core::state::{GameState, Phase};

const NUM_ACTIONS: usize = 32;
const MASK_DIM: usize = 32;

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut pool_path = String::from("data/pools/dd_2.5M.bin");
    let mut output_path = String::from("data/pools/dd_pool_enriched.bin");
    let mut num_deals: usize = 100_000;
    let mut batch_size: usize = 4096;
    let mut model_path = String::from("models/play_v2/play_final.bin");
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--output" => { i += 1; output_path = args[i].clone(); }
            "--deals" => { i += 1; num_deals = args[i].parse().unwrap(); }
            "--batch" => { i += 1; batch_size = args[i].parse().unwrap(); }
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            _ => { eprintln!("Unknown arg: {}", args[i]); }
        }
        i += 1;
    }

    // Load pool
    eprintln!("Loading pool from {}...", pool_path);
    let pool = DealPool::load(&pool_path).expect("Failed to load pool");
    eprintln!("  Pool has {} deals", pool.len());

    // Sample deals
    let mut rng = StdRng::seed_from_u64(seed);
    let sampled: Vec<_> = (0..num_deals).map(|_| {
        let deal = pool.sample(&mut rng);
        (deal.dealer, deal.hands, deal.dd_pts)
    }).collect();

    // Load GPU model
    let device = Device::cuda_if_available(0).expect("No CUDA device");
    eprintln!("Device: {:?}", device);
    eprintln!("Loading model {}...", model_path);

    let net = PoolNet::with_residual(OBS_DIM_TR, 1024, &device)
        .expect("Failed to create PoolNet");

    // Load weights from .bin file
    let weight_bytes = std::fs::read(&model_path).expect("Failed to read model");
    let weights: Vec<f32> = weight_bytes
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    net.load_weights(&weights).expect("Failed to load weights");

    eprintln!("Model loaded ({} params)", weights.len());

    // Expand deals into games: each deal × 4 suits = 4 games
    // We'll process games in GPU batches
    let total_games = num_deals * 4;
    eprintln!(
        "\nEnriching {} deals ({} games) in batches of {}...",
        num_deals, total_games, batch_size
    );

    let start = Instant::now();

    // Pre-allocate result array: real_pts[deal_idx][suit]
    let mut real_pts: Vec<[u8; 4]> = vec![[0u8; 4]; num_deals];

    // Process in batches of `batch_size` games
    // Each game = (deal_idx, suit)
    let mut game_idx = 0;
    let mut games_done = 0;

    while game_idx < total_games {
        let batch_end = (game_idx + batch_size).min(total_games);
        let this_batch = batch_end - game_idx;

        // Initialize game states and tracking for this batch
        let mut states: Vec<GameState> = Vec::with_capacity(this_batch);
        let mut trackings: Vec<EnvTracking> = Vec::with_capacity(this_batch);
        let mut game_meta: Vec<(usize, u8)> = Vec::with_capacity(this_batch); // (deal_idx, suit)

        for g in game_idx..batch_end {
            let deal_idx = g / 4;
            let suit = (g % 4) as u8;
            let (dealer, hands, _dd_pts) = sampled[deal_idx];

            let mut state = GameState::setup_dd(dealer, hands, suit);
            state.contract.trump = suit;
            state.contract.value = 8; // 80
            state.contract.team = 0; // NS
            state.contract.coinche = 0;

            let mut tracking = EnvTracking::new();
            tracking.dealer = dealer;
            // Fake bid history for obs generation
            let bidder = (dealer + 1) % 4;
            let bid_action = 0 * 4 + suit + 1;
            tracking.bid_history.push((bidder, bid_action));
            tracking.bid_history.push(((bidder + 1) % 4, 0));
            tracking.bid_history.push(((bidder + 2) % 4, 0));
            tracking.bid_history.push(((bidder + 3) % 4, 0));

            states.push(state);
            trackings.push(tracking);
            game_meta.push((deal_idx, suit));
        }

        // Play 32 steps (all games finish at the same time)
        let mut obs_flat = vec![0.0f32; this_batch * OBS_DIM_TR];
        let mut mask_flat = vec![0.0f32; this_batch * NUM_ACTIONS];
        // Store canonical suit orderings per game for action conversion
        let mut orders: Vec<[u8; 4]> = vec![[0; 4]; this_batch];

        for _step in 0..32 {
            // Collect observations and masks
            for i in 0..this_batch {
                if states[i].phase != Phase::Playing {
                    // Game already done (shouldn't happen in normal flow)
                    continue;
                }
                let obs_offset = i * OBS_DIM_TR;
                let mask_offset = i * NUM_ACTIONS;

                dmc_obs::write_observation_tr(
                    &mut obs_flat, obs_offset, &states[i], &trackings[i]
                );

                let order = dmc_obs::current_player_order(&states[i], &trackings[i]);
                orders[i] = order;
                let canonical_mask = dmc_obs::cardset_to_canonical(
                    states[i].legal_actions() as u32, &order
                );
                for j in 0..NUM_ACTIONS {
                    mask_flat[mask_offset + j] = if canonical_mask & (1 << j) != 0 {
                        1.0
                    } else {
                        0.0
                    };
                }
            }

            // GPU batch inference
            let obs_tensor = Tensor::from_slice(
                &obs_flat, (this_batch, OBS_DIM_TR), &device
            ).unwrap();
            let mask_tensor = Tensor::from_slice(
                &mask_flat, (this_batch, NUM_ACTIONS), &device
            ).unwrap();
            let actions = net.act_greedy(&obs_tensor, &mask_tensor).unwrap();

            // Step each game
            for i in 0..this_batch {
                if states[i].phase != Phase::Playing {
                    continue;
                }
                // Convert canonical action back to physical card
                let physical_action = dmc_obs::card_to_physical(actions[i], &orders[i]);
                trackings[i].track_action(&states[i], physical_action);
                states[i].step(physical_action);
            }
        }

        // Collect results
        for i in 0..this_batch {
            let (deal_idx, suit) = game_meta[i];
            real_pts[deal_idx][suit as usize] = states[i].points[0]; // NS points
        }

        games_done += this_batch;
        game_idx = batch_end;

        if games_done % (batch_size * 4) == 0 || game_idx >= total_games {
            let elapsed = start.elapsed().as_secs_f64();
            let deals_done = games_done / 4;
            let rate = deals_done as f64 / elapsed;
            let eta = if game_idx < total_games {
                (total_games - game_idx) as f64 / 4.0 / rate
            } else {
                0.0
            };
            eprintln!(
                "  {}/{} deals ({:.0}/s) {:.1}s elapsed, ETA {:.0}s",
                deals_done, num_deals, rate, elapsed, eta,
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "\nDone: {} deals enriched in {:.1}s ({:.0} deals/s, {:.0} games/s)",
        num_deals,
        elapsed,
        num_deals as f64 / elapsed,
        total_games as f64 / elapsed,
    );

    // Quick calibration summary
    print_summary(&sampled, &real_pts);

    // Save enriched pool
    save_enriched_pool(&output_path, &sampled, &real_pts);
    let size_mb = (num_deals * 25 + 16) as f64 / 1_048_576.0;
    eprintln!("Saved enriched pool to {} ({:.1}MB)", output_path, size_mb);
}

fn print_summary(
    deals: &[(u8, [u32; 4], [u8; 4])],
    real_pts: &[[u8; 4]],
) {
    let mut dd_sum = 0u64;
    let mut real_sum = 0u64;
    let mut count = 0u64;

    // Contract success rates
    let thresholds = [80u8, 90, 100, 110, 120, 130];
    let mut dd_above = [0u64; 6];
    let mut real_also_above = [0u64; 6];

    for (i, (_dealer, _hands, dd_pts)) in deals.iter().enumerate() {
        for suit in 0..4 {
            let dd = dd_pts[suit] as u64;
            let real = real_pts[i][suit] as u64;
            dd_sum += dd;
            real_sum += real;
            count += 1;

            for (t, &thr) in thresholds.iter().enumerate() {
                if dd >= thr as u64 {
                    dd_above[t] += 1;
                    if real >= thr as u64 {
                        real_also_above[t] += 1;
                    }
                }
            }
        }
    }

    let dd_mean = dd_sum as f64 / count as f64;
    let real_mean = real_sum as f64 / count as f64;

    println!("\n=== Calibration Summary ===");
    println!("Samples:  {}", count);
    println!("DD mean:  {:.1}", dd_mean);
    println!("Real mean: {:.1}", real_mean);
    println!("Ratio:    {:.4}", real_mean / dd_mean);
    println!();

    println!("Contract Success: P(real >= thr | DD >= thr)");
    println!("{:<10} {:>10} {:>12}", "Threshold", "DD count", "P(success)");
    println!("{}", "-".repeat(35));
    for (t, &thr) in thresholds.iter().enumerate() {
        if dd_above[t] > 0 {
            let pct = real_also_above[t] as f64 / dd_above[t] as f64 * 100.0;
            println!("  >= {:<6} {:>8}   {:>9.1}%", thr, dd_above[t], pct);
        }
    }
}

fn save_enriched_pool(
    path: &str,
    deals: &[(u8, [u32; 4], [u8; 4])],
    real_pts: &[[u8; 4]],
) {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path).unwrap());
    f.write_all(b"COLVDR01").unwrap(); // Different magic for enriched pool
    f.write_all(&(deals.len() as u64).to_le_bytes()).unwrap();
    for (i, (dealer, hands, dd_pts)) in deals.iter().enumerate() {
        f.write_all(&[*dealer]).unwrap();
        for &h in hands {
            f.write_all(&h.to_le_bytes()).unwrap();
        }
        f.write_all(dd_pts).unwrap();
        f.write_all(&real_pts[i]).unwrap();
    }
    f.flush().unwrap();
}
