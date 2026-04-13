/// Enrich a DD pool with mixed-team play: one team uses DMC (DouDou50),
/// the other uses IS-DD (no belief). Parallelized with rayon.
///
/// Usage:
///   cargo run --bin enrich_pool_mixed --release --features parallel -- [options]
///
/// Options:
///   --pool PATH        Input pool (default: data/pools/dd_2.5M.bin)
///   --output PATH      Output enriched pool
///   --deals N          Number of deals (default: 1000)
///   --time-ms N        IS-DD time per move (default: 50)
///   --dets N           IS-DD determinizations (default: 20)
///   --model PATH       DMC model (default: models/play_v2/play_final.bin)
///   --ns-method STR    NS play method: "dmc" or "isdd" (default: dmc)
///   --seed N           RNG seed (default: 42)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::bid_train_env::DealPool;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::state::{GameState, Phase};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut pool_path = String::from("data/pools/dd_2.5M.bin");
    let mut output_path = String::from("data/pools/dd_100k_mixed.bin");
    let mut num_deals: usize = 1000;
    let mut time_ms: u32 = 50;
    let mut dets: u32 = 20;
    let mut model_path = String::from("models/play_v2/play_final.bin");
    let mut ns_method = String::from("dmc"); // "dmc" or "isdd"
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--output" => { i += 1; output_path = args[i].clone(); }
            "--deals" => { i += 1; num_deals = args[i].parse().unwrap(); }
            "--time-ms" => { i += 1; time_ms = args[i].parse().unwrap(); }
            "--dets" => { i += 1; dets = args[i].parse().unwrap(); }
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--ns-method" => { i += 1; ns_method = args[i].clone(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            _ => { eprintln!("Unknown arg: {}", args[i]); }
        }
        i += 1;
    }

    let ns_is_dmc = ns_method == "dmc";
    eprintln!("NS method: {}  EW method: {}",
        if ns_is_dmc { "DMC" } else { "IS-DD" },
        if ns_is_dmc { "IS-DD" } else { "DMC" });

    // Load pool
    eprintln!("Loading pool from {}...", pool_path);
    let pool = DealPool::load(&pool_path).expect("Failed to load pool");
    eprintln!("  Pool has {} deals", pool.len());

    // Sample deals (same seed as enrich_pool_isdd for matching)
    let mut rng = StdRng::seed_from_u64(seed);
    let sampled: Vec<_> = (0..num_deals).map(|_| {
        let deal = pool.sample(&mut rng);
        (deal.dealer, deal.hands, deal.dd_pts)
    }).collect();

    // Load DMC weights once (will clone per thread)
    let dmc_weights = std::fs::read(&model_path).expect("Failed to read DMC model");
    let dmc_floats: Vec<f32> = dmc_weights
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    eprintln!("DMC model: {} ({} params)", model_path, dmc_floats.len());

    let isdd_config = IsDdConfig {
        determinizations: dets,
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };

    let total_games = num_deals * 4;
    eprintln!(
        "\nEnriching {} deals ({} games), NS={} EW={}...",
        num_deals, total_games,
        if ns_is_dmc { "DMC" } else { "IS-DD" },
        if ns_is_dmc { "IS-DD" } else { "DMC" },
    );

    let start = Instant::now();
    let progress = AtomicUsize::new(0);

    let real_pts: Vec<[u8; 4]> = {
        use rayon::prelude::*;
        sampled.par_iter().enumerate().map(|(deal_idx, &(dealer, hands, _dd_pts))| {
            let mut pts = [0u8; 4];
            let mut rng = StdRng::seed_from_u64(seed + deal_idx as u64 * 100);

            // Each thread gets its own DmcNet and IsDdSearch
            let mut dmc_net = DmcNet::load(&model_path).expect("Failed to load DMC");
            let mut isdd_search = IsDdSearch::new();
            let mut obs_buf = vec![0.0f32; OBS_DIM_TR];

            for suit in 0..4u8 {
                let mut state = GameState::setup_dd(dealer, hands, suit);

                // Set up tracking for DMC obs
                let mut tracking = EnvTracking::new();
                tracking.dealer = dealer;
                let bidder = (dealer + 1) % 4;
                let bid_action = 0 * 4 + suit + 1;
                tracking.bid_history.push((bidder, bid_action));
                tracking.bid_history.push(((bidder + 1) % 4, 0));
                tracking.bid_history.push(((bidder + 2) % 4, 0));
                tracking.bid_history.push(((bidder + 3) % 4, 0));

                while state.phase == Phase::Playing {
                    let player = state.current_player();
                    let is_ns = player == 0 || player == 2;
                    let use_dmc = (is_ns && ns_is_dmc) || (!is_ns && !ns_is_dmc);

                    let action = if use_dmc {
                        // DMC: canonical obs + greedy action
                        dmc_obs::write_observation_tr(&mut obs_buf, 0, &state, &tracking);
                        let order = dmc_obs::current_player_order(&state, &tracking);
                        let canonical_mask = dmc_obs::cardset_to_canonical(
                            state.legal_actions() as u32, &order
                        );
                        let (canonical_best, _) = dmc_net.best_action(&obs_buf, canonical_mask);
                        dmc_obs::card_to_physical(canonical_best, &order)
                    } else {
                        // IS-DD
                        isdd_search.search(&state, &isdd_config, &mut rng)
                    };

                    tracking.track_action(&state, action);
                    state.step(action);
                }

                pts[suit as usize] = state.points[0]; // NS points
            }

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 100 == 0 || done == num_deals {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                let eta = (num_deals - done) as f64 / rate;
                eprintln!(
                    "  {}/{} deals ({:.1}/s) {:.1}s elapsed, ETA {:.0}s",
                    done, num_deals, rate, elapsed, eta
                );
            }

            pts
        }).collect()
    };

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!(
        "\nDone: {} deals enriched in {:.1}s ({:.1} deals/s)",
        num_deals, elapsed, num_deals as f64 / elapsed,
    );

    // Summary
    print_summary(&sampled, &real_pts);

    // Save
    save_enriched_pool(&output_path, &sampled, &real_pts);
    let size_mb = (num_deals * 25 + 16) as f64 / 1_048_576.0;
    eprintln!("Saved to {} ({:.1}MB)", output_path, size_mb);
}

fn print_summary(
    deals: &[(u8, [u32; 4], [u8; 4])],
    real_pts: &[[u8; 4]],
) {
    let mut dd_sum = 0u64;
    let mut real_sum = 0u64;
    let mut count = 0u64;

    for (i, (_dealer, _hands, dd_pts)) in deals.iter().enumerate() {
        for suit in 0..4 {
            dd_sum += dd_pts[suit] as u64;
            real_sum += real_pts[i][suit] as u64;
            count += 1;
        }
    }

    let dd_mean = dd_sum as f64 / count as f64;
    let real_mean = real_sum as f64 / count as f64;

    println!("\n=== Summary ({} samples) ===", count);
    println!("DD mean:   {:.1}", dd_mean);
    println!("Real mean: {:.1}", real_mean);
    println!("Ratio:     {:.4}", real_mean / dd_mean);
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
