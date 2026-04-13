/// Play N deals × 4 suits with both DMC and IS-DD, save full replays.
///
/// Each deal is played twice (once per method), so we can analyze:
/// - When do the methods play identically?
/// - Which tricks diverge?
/// - Where does each method win points?
///
/// Output: two COLVGM01 files (dmc.bin and isdd.bin) + a metadata JSON with
/// deal_idx, suit, method, dd_pts, realized_pts mapping.
///
/// Usage:
///   cargo run --bin replay_dmc_vs_isdd --release --features parallel -- [options]
///
/// Options:
///   --pool PATH        Input pool (default: data/pools/dd_2.5M.bin)
///   --deals N          Number of deals (default: 1000)
///   --output DIR       Output directory (default: data/replays/dmc_vs_isdd)
///   --model PATH       DMC model (default: models/play_v2/play_final.bin)
///   --time-ms N        IS-DD time per move (default: 50)
///   --dets N           IS-DD determinizations (default: 20)
///   --seed N           RNG seed (default: 42)

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Mutex;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::bid_train_env::DealPool;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use colver_core::game_replay::GameReplay;
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::state::{GameState, Phase};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut pool_path = String::from("data/pools/dd_2.5M.bin");
    let mut num_deals: usize = 1000;
    let mut output_dir = String::from("data/replays/dmc_vs_isdd");
    let mut model_path = String::from("models/play_v2/play_final.bin");
    let mut time_ms: u32 = 50;
    let mut dets: u32 = 20;
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--deals" => { i += 1; num_deals = args[i].parse().unwrap(); }
            "--output" => { i += 1; output_dir = args[i].clone(); }
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--time-ms" => { i += 1; time_ms = args[i].parse().unwrap(); }
            "--dets" => { i += 1; dets = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            _ => { eprintln!("Unknown arg: {}", args[i]); }
        }
        i += 1;
    }

    std::fs::create_dir_all(&output_dir).unwrap();

    eprintln!("Loading pool from {}...", pool_path);
    let pool = DealPool::load(&pool_path).expect("Failed to load pool");
    eprintln!("  Pool has {} deals", pool.len());

    let mut rng = StdRng::seed_from_u64(seed);
    let sampled: Vec<_> = (0..num_deals).map(|_| {
        let deal = pool.sample(&mut rng);
        (deal.dealer, deal.hands, deal.dd_pts)
    }).collect();

    let isdd_config = IsDdConfig {
        determinizations: dets,
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };

    let total_games = num_deals * 4;
    eprintln!("Playing {} deals × 4 suits = {} games × 2 methods", num_deals, total_games);

    let start = Instant::now();
    let progress = AtomicUsize::new(0);

    // Results: for each (deal_idx, suit), we'll have
    //   dmc_replay, dmc_pts
    //   isdd_replay, isdd_pts
    // Store in parallel-safe mutex containers
    let results: Mutex<Vec<GameResult>> = Mutex::new(Vec::with_capacity(total_games));

    use rayon::prelude::*;

    // Each (deal, suit) is independent → parallelize over flat indices
    let work: Vec<(usize, u8)> = (0..num_deals).flat_map(|di| (0..4u8).map(move |s| (di, s))).collect();

    work.par_iter().for_each(|&(deal_idx, suit)| {
        let (dealer, hands, dd_pts) = sampled[deal_idx];

        // DMC replay
        let (dmc_actions, dmc_pts) = play_dmc(dealer, hands, suit, &model_path);

        // IS-DD replay
        let mut rng = StdRng::seed_from_u64(seed + (deal_idx as u64) * 100 + (suit as u64));
        let (isdd_actions, isdd_pts) = play_isdd(dealer, hands, suit, &isdd_config, &mut rng);

        results.lock().unwrap().push(GameResult {
            deal_idx,
            suit,
            dealer,
            hands,
            dd_pts: dd_pts[suit as usize],
            dmc_actions,
            dmc_pts,
            isdd_actions,
            isdd_pts,
        });

        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
        if done % 50 == 0 || done == total_games {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed;
            let eta = (total_games - done) as f64 / rate;
            eprintln!("  {}/{} games ({:.1}/s) {:.0}s elapsed, ETA {:.0}s",
                done, total_games, rate, elapsed, eta);
        }
    });

    let mut results = results.into_inner().unwrap();
    results.sort_by_key(|r| (r.deal_idx, r.suit));

    let elapsed = start.elapsed().as_secs_f64();
    eprintln!("\nDone: {} games in {:.1}s ({:.1} games/s)",
        total_games, elapsed, total_games as f64 / elapsed);

    // Save replays
    let dmc_replays: Vec<GameReplay> = results.iter().map(|r| GameReplay {
        dealer: r.dealer,
        hands: r.hands,
        actions: r.dmc_actions.clone(),
    }).collect();
    let isdd_replays: Vec<GameReplay> = results.iter().map(|r| GameReplay {
        dealer: r.dealer,
        hands: r.hands,
        actions: r.isdd_actions.clone(),
    }).collect();

    let dmc_path = format!("{}/dmc.bin", output_dir);
    let isdd_path = format!("{}/isdd.bin", output_dir);
    GameReplay::write_all(&dmc_path, &dmc_replays).unwrap();
    GameReplay::write_all(&isdd_path, &isdd_replays).unwrap();
    eprintln!("Saved replays to {} and {}", dmc_path, isdd_path);

    // Save metadata CSV
    let meta_path = format!("{}/metadata.csv", output_dir);
    let mut meta = String::from("game_idx,deal_idx,suit,dealer,dd_pts,dmc_pts,isdd_pts,diff_dd_dmc,diff_dd_isdd\n");
    for (i, r) in results.iter().enumerate() {
        meta.push_str(&format!("{},{},{},{},{},{},{},{},{}\n",
            i, r.deal_idx, r.suit, r.dealer, r.dd_pts, r.dmc_pts, r.isdd_pts,
            (r.dd_pts as i32) - (r.dmc_pts as i32),
            (r.dd_pts as i32) - (r.isdd_pts as i32)));
    }
    std::fs::write(&meta_path, meta).unwrap();
    eprintln!("Saved metadata to {}", meta_path);

    // Quick summary
    let n = results.len();
    let dmc_mae: f64 = results.iter().map(|r| (r.dd_pts as i32 - r.dmc_pts as i32).abs() as f64).sum::<f64>() / n as f64;
    let isdd_mae: f64 = results.iter().map(|r| (r.dd_pts as i32 - r.isdd_pts as i32).abs() as f64).sum::<f64>() / n as f64;
    let identical = results.iter().filter(|r| r.dmc_actions == r.isdd_actions).count();

    println!("\n=== Summary ===");
    println!("Games:            {}", n);
    println!("DMC  MAE vs DD:   {:.2}", dmc_mae);
    println!("ISDD MAE vs DD:   {:.2}", isdd_mae);
    println!("Identical plays:  {} ({:.1}%)", identical, 100.0 * identical as f64 / n as f64);
}

struct GameResult {
    deal_idx: usize,
    suit: u8,
    dealer: u8,
    hands: [u32; 4],
    dd_pts: u8,
    dmc_actions: Vec<u8>,
    dmc_pts: u8,
    isdd_actions: Vec<u8>,
    isdd_pts: u8,
}

fn play_dmc(dealer: u8, hands: [u32; 4], suit: u8, model_path: &str) -> (Vec<u8>, u8) {
    let mut state = GameState::setup_dd(dealer, hands, suit);

    let mut tracking = EnvTracking::new();
    tracking.dealer = dealer;
    let bidder = (dealer + 1) % 4;
    let bid_action = 0 * 4 + suit + 1;
    tracking.bid_history.push((bidder, bid_action));
    tracking.bid_history.push(((bidder + 1) % 4, 0));
    tracking.bid_history.push(((bidder + 2) % 4, 0));
    tracking.bid_history.push(((bidder + 3) % 4, 0));

    let mut dmc_net = DmcNet::load(model_path).expect("Failed to load DMC");
    dmc_net.set_residual(true);
    let mut obs_buf = vec![0.0f32; OBS_DIM_TR];
    let mut actions = Vec::with_capacity(32);

    while state.phase == Phase::Playing {
        dmc_obs::write_observation_tr(&mut obs_buf, 0, &state, &tracking);
        let order = dmc_obs::current_player_order(&state, &tracking);
        let canonical_mask = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
        let (canonical_best, _) = dmc_net.best_action(&obs_buf, canonical_mask);
        let action = dmc_obs::card_to_physical(canonical_best, &order);

        tracking.track_action(&state, action);
        state.step(action);
        actions.push(action);
    }

    (actions, state.points[0])
}

fn play_isdd(dealer: u8, hands: [u32; 4], suit: u8, config: &IsDdConfig, rng: &mut StdRng) -> (Vec<u8>, u8) {
    let mut state = GameState::setup_dd(dealer, hands, suit);
    let mut search = IsDdSearch::new();
    let mut actions = Vec::with_capacity(32);

    while state.phase == Phase::Playing {
        let action = search.search(&state, config, rng);
        state.step(action);
        actions.push(action);
    }

    (actions, state.points[0])
}
