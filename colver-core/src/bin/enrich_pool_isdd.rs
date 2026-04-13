/// Enrich a DD pool with IS-DD play points (no belief, no GPU).
///
/// For each deal × 4 suits, plays out with IS-DD (smart_is_dd, no belief)
/// and records actual NS points. Parallelized with rayon.
///
/// Usage:
///   cargo run --bin enrich_pool_isdd --release --features parallel -- [options]
///
/// Options:
///   --pool PATH        Input pool (default: data/pools/dd_2.5M.bin)
///   --output PATH      Output enriched pool (default: data/pools/dd_pool_enriched_isdd.bin)
///   --deals N          Number of deals (default: 1000)
///   --time-ms N        IS-DD time per move (default: 50)
///   --dets N           Determinizations (default: 20)
///   --seed N           RNG seed (default: 42)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_train_env::DealPool;
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::state::{GameState, Phase};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut pool_path = String::from("data/pools/dd_5M.bin");
    let mut output_path = String::from("data/pools/dd_pool_enriched_isdd.bin");
    let mut num_deals: usize = 1000;
    let mut time_ms: u32 = 50;
    let mut dets: u32 = 20;
    let mut seed: u64 = 42;
    let mut offset: usize = 0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--output" => { i += 1; output_path = args[i].clone(); }
            "--deals" => { i += 1; num_deals = args[i].parse().unwrap(); }
            "--time-ms" => { i += 1; time_ms = args[i].parse().unwrap(); }
            "--dets" => { i += 1; dets = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--offset" => { i += 1; offset = args[i].parse().unwrap(); }
            _ => { eprintln!("Unknown arg: {}", args[i]); }
        }
        i += 1;
    }

    // Load pool
    eprintln!("Loading pool from {}...", pool_path);
    let pool = DealPool::load(&pool_path).expect("Failed to load pool");
    eprintln!("  Pool has {} deals", pool.len());

    // Take deals sequentially from offset
    let end = (offset + num_deals).min(pool.len());
    let actual = end - offset;
    if actual < num_deals {
        eprintln!("  WARNING: only {} deals available from offset {}", actual, offset);
    }
    let num_deals = actual;
    let sampled: Vec<_> = (offset..end).map(|idx| {
        let deal = pool.get(idx);
        (deal.dealer, deal.hands, deal.dd_pts)
    }).collect();
    eprintln!("  Taking deals [{}, {}) ({} deals)", offset, end, num_deals);

    let config = IsDdConfig {
        determinizations: dets,
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };

    let total_games = num_deals * 4;
    eprintln!(
        "\nEnriching {} deals ({} games) with IS-DD ({}ms, {} dets)...",
        num_deals, total_games, time_ms, dets
    );

    let start = Instant::now();
    let progress = AtomicUsize::new(0);

    // Parallelize over individual games (deal × suit) for better load balancing.
    // Each game is independent; 4× more work items than deals for rayon to schedule.
    let game_results: Vec<(usize, u8, u8)> = {
        use rayon::prelude::*;

        // Build flat list of (deal_idx, suit)
        let games: Vec<(usize, u8)> = (0..num_deals)
            .flat_map(|d| (0..4u8).map(move |s| (d, s)))
            .collect();

        games.par_iter().map(|&(deal_idx, suit)| {
            let (dealer, hands, _dd_pts) = sampled[deal_idx];
            let mut rng = StdRng::seed_from_u64(seed + deal_idx as u64 * 100 + suit as u64);
            let mut state = GameState::setup_dd(dealer, hands, suit);
            let mut search = IsDdSearch::new();

            while state.phase == Phase::Playing {
                let action = search.search(&state, &config, &mut rng);
                state.step(action);
            }

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 400 == 0 || done == total_games {
                let elapsed = start.elapsed().as_secs_f64();
                let deals_done = done / 4;
                let rate = deals_done as f64 / elapsed;
                let eta = (num_deals - deals_done) as f64 / rate;
                eprintln!(
                    "  {}/{} deals ({:.1}/s) {:.1}s elapsed, ETA {:.0}s",
                    deals_done, num_deals, rate, elapsed, eta
                );
            }

            (deal_idx, suit, state.points[0])
        }).collect()
    };

    // Reassemble into per-deal [u8; 4]
    let mut real_pts: Vec<[u8; 4]> = vec![[0u8; 4]; num_deals];
    for (deal_idx, suit, pts) in &game_results {
        real_pts[*deal_idx][*suit as usize] = *pts;
    }

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "\nDone: {} deals enriched in {:.1}s ({:.1} deals/s, {:.1} games/s)",
        num_deals,
        elapsed,
        num_deals as f64 / elapsed,
        total_games as f64 / elapsed,
    );

    // Quick calibration summary
    print_summary(&sampled, &real_pts);

    // Save enriched pool (legacy COLVDR01 format)
    save_enriched_pool(&output_path, &sampled, &real_pts);
    let size_mb = (num_deals * 25 + 16) as f64 / 1_048_576.0;
    eprintln!("Saved enriched pool to {} ({:.1}MB)", output_path, size_mb);

    // Also save as COLVSC01 score file
    let sc_path = output_path.replace(".bin", ".sc");
    DealPool::save_scores("isdd", offset, &real_pts, &sc_path)
        .expect("Failed to save score file");
    eprintln!("Saved score file to {}", sc_path);
}

fn print_summary(
    deals: &[(u8, [u32; 4], [u8; 4])],
    real_pts: &[[u8; 4]],
) {
    let mut dd_sum = 0u64;
    let mut real_sum = 0u64;
    let mut count = 0u64;

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
    f.write_all(b"COLVDR01").unwrap();
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
