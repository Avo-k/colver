//! DD vs Real Play Calibration: compare DD-optimal points with actual points
//! achieved by DouDou50 (or other DMC models) playing out deals.
//!
//! For each deal × each trump suit:
//!   - DD says NS gets `dd_pts` points (perfect play both sides)
//!   - DouDou50 plays all 4 seats → actual NS points
//!
//! Outputs calibration curve: how much does DD overestimate?
//!
//! Usage:
//!   cargo run --bin dd_vs_real --release -- [options]
//!
//! Options:
//!   --deals N          Number of deals (default: 2000)
//!   --model PATH       DMC model path (default: models/play_v2/play_final.bin)
//!   --pool PATH        Load deals from pool file instead of generating fresh
//!   --seed N           RNG seed (default: 42)
//!   --threads N        Parallelism (default: auto)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::card::*;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM, OBS_DIM_TR};
use colver_core::solver::{new_tt_buffer, solve_for_trump_reuse_tt};
use colver_core::state::{GameState, Phase};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut num_deals: usize = 2000;
    let mut model_path = String::from("models/play_v2/play_final.bin");
    let mut pool_path: Option<String> = None;
    let mut seed: u64 = 42;
    let mut num_threads: usize = std::thread::available_parallelism()
        .map(|p| p.get())
        .unwrap_or(4);

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--deals" => { i += 1; num_deals = args[i].parse().unwrap(); }
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--pool" => { i += 1; pool_path = Some(args[i].clone()); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--threads" => { i += 1; num_threads = args[i].parse().unwrap(); }
            _ => { eprintln!("Unknown arg: {}", args[i]); }
        }
        i += 1;
    }

    // Load model to detect obs_dim
    let net = DmcNet::load(&model_path).expect("Failed to load DMC model");
    let obs_dim = net.obs_dim();
    let is_canonical = obs_dim == OBS_DIM_TR;
    let residual = is_canonical; // DouDou50 uses residual
    drop(net);

    println!("=== DD vs Real Play Calibration ===");
    println!("Deals:   {}", num_deals);
    println!("Model:   {} (obs_dim={}, canonical={}, residual={})", model_path, obs_dim, is_canonical, residual);
    println!("Threads: {}", num_threads);
    println!("Seed:    {}", seed);
    println!();

    // Load or generate deals
    let deals: Vec<(u8, [u32; 4])> = if let Some(ref path) = pool_path {
        eprintln!("Loading deals from pool {}...", path);
        let pool = colver_core::bid_train_env::DealPool::load(path)
            .expect("Failed to load pool");
        let mut rng = StdRng::seed_from_u64(seed);
        use rand::Rng;
        (0..num_deals)
            .map(|_| {
                let deal = pool.sample(&mut rng);
                (deal.dealer, deal.hands)
            })
            .collect()
    } else {
        eprintln!("Generating {} random deals...", num_deals);
        let mut rng = StdRng::seed_from_u64(seed);
        (0..num_deals)
            .map(|_| {
                use rand::Rng;
                let dealer = rng.gen_range(0..4u8);
                let state = GameState::deal_random(dealer, &mut rng);
                (dealer, state.hands)
            })
            .collect()
    };

    let start = Instant::now();
    let progress = AtomicUsize::new(0);

    // Each result: (dd_ns_pts, real_ns_pts, suit)
    let mut all_results: Vec<(u8, u8, u8)> = Vec::with_capacity(num_deals * 4);

    // Process in parallel chunks
    let chunk_size = (num_deals + num_threads - 1) / num_threads;
    let chunks: Vec<_> = deals.chunks(chunk_size).collect();

    let thread_results: Vec<Vec<(u8, u8, u8)>> = std::thread::scope(|s| {
        let mut handles = Vec::new();

        for (_t, chunk) in chunks.iter().enumerate() {
            let model_path_ref = &model_path;
            let progress_ref = &progress;
            let start_ref = &start;

            handles.push(s.spawn(move || {
                let mut net = DmcNet::load(model_path_ref).expect("Failed to load model in thread");
                if residual {
                    net.set_residual(true);
                }
                let mut tt_buf = new_tt_buffer();
                let mut obs_buf = vec![0.0f32; obs_dim];
                let mut results = Vec::with_capacity(chunk.len() * 4);

                for &(dealer, hands) in *chunk {
                    // DD solve all 4 suits
                    for suit in 0..4u8 {
                        let dd_result = solve_for_trump_reuse_tt(hands, dealer, suit, &mut tt_buf);
                        let dd_ns_pts = dd_result[0]; // NS points with perfect play

                        // Play out with DMC model
                        let real_ns_pts = play_deal_with_dmc(
                            dealer, hands, suit,
                            &mut net, &mut obs_buf, is_canonical,
                        );

                        results.push((dd_ns_pts, real_ns_pts, suit));
                    }

                    let done = progress_ref.fetch_add(1, Ordering::Relaxed) + 1;
                    if done % 500 == 0 {
                        let elapsed = start_ref.elapsed().as_secs_f64();
                        let rate = done as f64 / elapsed;
                        eprintln!(
                            "  {}/{} deals ({:.0}/s, ETA {:.0}s)",
                            done, num_deals, rate, (num_deals - done) as f64 / rate
                        );
                    }
                }

                results
            }));
        }

        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    for r in thread_results {
        all_results.extend(r);
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\nCompleted {} deals ({} suit evaluations) in {:.1}s ({:.1} deals/s)\n",
        num_deals,
        all_results.len(),
        elapsed,
        num_deals as f64 / elapsed
    );

    // === Analysis ===
    print_calibration_analysis(&all_results);
}

/// Play a full deal with DMC model controlling all 4 players.
/// Returns actual NS points.
fn play_deal_with_dmc(
    dealer: u8,
    hands: [u32; 4],
    trump: u8,
    net: &mut DmcNet,
    obs_buf: &mut [f32],
    is_canonical: bool,
) -> u8 {
    // Set up play phase directly (skip bidding)
    let mut state = GameState::setup_dd(dealer, hands, trump);
    // Set a realistic contract (80 in this suit, NS taker)
    state.contract.trump = trump;
    state.contract.value = 8; // 80 points
    state.contract.team = 0; // NS
    state.contract.coinche = 0;

    let mut tracking = EnvTracking::new();
    tracking.dealer = dealer;
    // Add a fake bid history entry so obs generation doesn't break
    // Bid: player (dealer+1)%4 bids 80 in `trump`, then 3 passes
    let bidder = (dealer + 1) % 4;
    let bid_action = 0 * 4 + trump + 1; // value_idx=0 (80), suit_idx=trump
    tracking.bid_history.push((bidder, bid_action));
    tracking.bid_history.push(((bidder + 1) % 4, 0)); // pass
    tracking.bid_history.push(((bidder + 2) % 4, 0)); // pass
    tracking.bid_history.push(((bidder + 3) % 4, 0)); // pass

    while state.phase == Phase::Playing {
        let action = if is_canonical {
            dmc_obs::write_observation_tr(obs_buf, 0, &state, &tracking);
            let order = dmc_obs::current_player_order(&state, &tracking);
            let canonical_mask =
                dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
            let (canonical_best, _) = net.best_action(obs_buf, canonical_mask);
            dmc_obs::card_to_physical(canonical_best, &order)
        } else {
            dmc_obs::write_observation(obs_buf, 0, &state, &tracking);
            let legal_mask = state.legal_actions() as u32;
            let (action, _) = net.best_action(obs_buf, legal_mask);
            action
        };

        tracking.track_action(&state, action);
        state.step(action);
    }

    state.points[0] // NS points
}

fn print_calibration_analysis(results: &[(u8, u8, u8)]) {
    let n = results.len();

    // Overall stats
    let dd_mean = results.iter().map(|(dd, _, _)| *dd as f64).sum::<f64>() / n as f64;
    let real_mean = results.iter().map(|(_, real, _)| *real as f64).sum::<f64>() / n as f64;
    let ratio = real_mean / dd_mean;

    println!("=== Overall Calibration ===");
    println!("DD mean NS pts:   {:.1}", dd_mean);
    println!("Real mean NS pts: {:.1}", real_mean);
    println!("Ratio (real/DD):  {:.4}", ratio);
    println!("Avg overestimate: {:.1} pts ({:.1}%)", dd_mean - real_mean, (1.0 - ratio) * 100.0);
    println!();

    // Bucket by DD points
    println!("{}", "=".repeat(100));
    println!("DD Points → Real Points (bucketed)");
    println!("{}", "=".repeat(100));
    println!(
        "{:<12} {:>6} {:>8} {:>8} {:>8} {:>8} | {:>7} {:>7} {:>7} {:>7}",
        "DD Range", "Count", "DD Mean", "Real Mn", "Ratio", "Δ",
        "≥80%", "≥90%", "≥100%", "≥110%"
    );
    println!("{}", "-".repeat(100));

    let buckets: &[(u8, u8, &str)] = &[
        (0, 40, "0-39"),
        (40, 60, "40-59"),
        (60, 70, "60-69"),
        (70, 80, "70-79"),
        (80, 90, "80-89"),
        (90, 100, "90-99"),
        (100, 110, "100-109"),
        (110, 120, "110-119"),
        (120, 130, "120-129"),
        (130, 142, "130-141"),
        (142, 152, "142-151"),
        (152, 162, "152-161"),
        (162, 253, "162+"),
    ];

    for &(lo, hi, label) in buckets {
        let bucket: Vec<_> = results
            .iter()
            .filter(|(dd, _, _)| *dd >= lo && *dd < hi)
            .collect();

        if bucket.is_empty() {
            continue;
        }

        let count = bucket.len();
        let dd_avg = bucket.iter().map(|(dd, _, _)| *dd as f64).sum::<f64>() / count as f64;
        let real_avg = bucket.iter().map(|(_, real, _)| *real as f64).sum::<f64>() / count as f64;
        let ratio = if dd_avg > 0.0 { real_avg / dd_avg } else { 0.0 };
        let delta = real_avg - dd_avg;

        // Success rates: what fraction of real results actually achieved threshold?
        let pct_above = |threshold: u8| -> f64 {
            bucket.iter().filter(|(_, real, _)| *real >= threshold).count() as f64
                / count as f64
                * 100.0
        };

        println!(
            "{:<12} {:>6} {:>8.1} {:>8.1} {:>8.3} {:>+8.1} | {:>6.1}% {:>6.1}% {:>6.1}% {:>6.1}%",
            label, count, dd_avg, real_avg, ratio, delta,
            pct_above(80), pct_above(90), pct_above(100), pct_above(110),
        );
    }

    // Contract success analysis: if DD says we make N, how often do we actually make N?
    println!("\n{}", "=".repeat(80));
    println!("Contract Success: P(real ≥ threshold | DD ≥ threshold)");
    println!("{}", "=".repeat(80));
    println!(
        "{:<12} {:>8} {:>8} {:>10} {:>12}",
        "Threshold", "DD≥thr", "Real≥thr", "P(success)", "Gap"
    );
    println!("{}", "-".repeat(55));

    for threshold in [80u8, 90, 100, 110, 120, 130, 140, 152] {
        let dd_above: Vec<_> = results
            .iter()
            .filter(|(dd, _, _)| *dd >= threshold)
            .collect();

        if dd_above.is_empty() {
            continue;
        }

        let also_real_above = dd_above
            .iter()
            .filter(|(_, real, _)| *real >= threshold)
            .count();

        let dd_pct = dd_above.len() as f64 / results.len() as f64 * 100.0;
        let success_pct = also_real_above as f64 / dd_above.len() as f64 * 100.0;
        let real_pct = results.iter().filter(|(_, real, _)| *real >= threshold).count() as f64
            / results.len() as f64 * 100.0;

        println!(
            "≥{:<10} {:>7.1}% {:>7.1}% {:>9.1}% {:>+11.1}%",
            threshold,
            dd_pct,
            real_pct,
            success_pct,
            success_pct - 100.0,
        );
    }

    // Per-suit breakdown
    println!("\n{}", "=".repeat(60));
    println!("Per-Suit Calibration");
    println!("{}", "=".repeat(60));
    let suit_names = ["Spades", "Hearts", "Diamonds", "Clubs"];
    for suit in 0..4u8 {
        let suit_results: Vec<_> = results.iter().filter(|(_, _, s)| *s == suit).collect();
        let dd_avg = suit_results.iter().map(|(dd, _, _)| *dd as f64).sum::<f64>()
            / suit_results.len() as f64;
        let real_avg = suit_results.iter().map(|(_, real, _)| *real as f64).sum::<f64>()
            / suit_results.len() as f64;
        println!(
            "  {:<10}: DD={:.1}, Real={:.1}, Ratio={:.3}, Δ={:+.1}",
            suit_names[suit as usize],
            dd_avg,
            real_avg,
            real_avg / dd_avg,
            real_avg - dd_avg,
        );
    }

    // Scatter data for plotting (optional)
    println!("\n{}", "=".repeat(60));
    println!("Linear fit: real = a × DD + b");
    println!("{}", "=".repeat(60));

    // Simple linear regression
    let n_f = n as f64;
    let sum_x: f64 = results.iter().map(|(dd, _, _)| *dd as f64).sum();
    let sum_y: f64 = results.iter().map(|(_, real, _)| *real as f64).sum();
    let sum_xx: f64 = results.iter().map(|(dd, _, _)| (*dd as f64).powi(2)).sum();
    let sum_xy: f64 = results
        .iter()
        .map(|(dd, real, _)| *dd as f64 * *real as f64)
        .sum();

    let a = (n_f * sum_xy - sum_x * sum_y) / (n_f * sum_xx - sum_x * sum_x);
    let b = (sum_y - a * sum_x) / n_f;

    // R²
    let ss_res: f64 = results
        .iter()
        .map(|(dd, real, _)| {
            let pred = a * *dd as f64 + b;
            (*real as f64 - pred).powi(2)
        })
        .sum();
    let mean_y = sum_y / n_f;
    let ss_tot: f64 = results
        .iter()
        .map(|(_, real, _)| (*real as f64 - mean_y).powi(2))
        .sum();
    let r_sq = 1.0 - ss_res / ss_tot;

    println!("  real ≈ {:.4} × DD + {:.2}", a, b);
    println!("  R² = {:.4}", r_sq);
    println!("  → At DD=80:  predicted real = {:.1}", a * 80.0 + b);
    println!("  → At DD=100: predicted real = {:.1}", a * 100.0 + b);
    println!("  → At DD=120: predicted real = {:.1}", a * 120.0 + b);
    println!("  → At DD=140: predicted real = {:.1}", a * 140.0 + b);

    // RMSE
    let rmse = (ss_res / n_f).sqrt();
    println!("  RMSE = {:.1} pts", rmse);
}
