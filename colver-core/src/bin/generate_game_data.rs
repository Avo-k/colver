/// Generate game replay data for offline extraction pipelines.
///
/// Plays N games using DMC Q-net for card play and NN bid for bidding,
/// storing compact replays (dealer + hands + action sequence, ~62 bytes/game).
///
/// Binary format: COLVGM01 (see `game_replay.rs` for spec).
///
/// Usage:
///   cargo run -p colver-core --bin generate_game_data --release --features parallel -- \
///     --dmc-model models/dmc_final.bin \
///     --bid-model models/bid_nn_final.bin \
///     --games 500000 --output data/games.bin --threads 8

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_eval;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{self, BID_OBS_DIM, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2,
                           BID_OBS_DIM_SCORE_AWARE_V3};
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use colver_core::game_replay::GameReplay;
use colver_core::state::{GameState, Phase};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut dmc_model_path = String::from("models/dmc_final.bin");
    let mut bid_model_path = String::from("models/bid_nn_final.bin");
    let mut num_games: u64 = 50_000;
    let mut output_path = String::from("data/games.bin");
    let mut seed: u64 = 42;
    let mut num_threads: usize = 0;
    let mut score_ns: i32 = 0;
    let mut score_ew: i32 = 0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dmc-model" => { dmc_model_path = args[i + 1].clone(); i += 2; }
            "--bid-model" => { bid_model_path = args[i + 1].clone(); i += 2; }
            "--games" => { num_games = args[i + 1].parse().unwrap(); i += 2; }
            "--output" => { output_path = args[i + 1].clone(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--threads" => { num_threads = args[i + 1].parse().unwrap(); i += 2; }
            "--score-ns" => { score_ns = args[i + 1].parse().unwrap(); i += 2; }
            "--score-ew" => { score_ew = args[i + 1].parse().unwrap(); i += 2; }
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

    println!("=== Game Replay Generation ===");
    println!("DMC model:  {}", dmc_model_path);
    println!("Bid model:  {}", bid_model_path);
    println!("Games:      {}", num_games);
    println!("Output:     {}", output_path);
    println!("Seed:       {}", seed);
    println!("Scores:     {}-{} (obs des encheres score-aware)", score_ns, score_ew);

    let start = Instant::now();

    #[cfg(feature = "parallel")]
    let all_replays = {
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

        let results: Vec<Vec<GameReplay>> = chunks
            .into_par_iter()
            .enumerate()
            .map(|(thread_id, (count, thread_seed))| {
                generate_chunk(dmc_path, bid_path, count, thread_seed, thread_id, n_threads, (score_ns, score_ew))
            })
            .collect();

        let mut all: Vec<GameReplay> = Vec::with_capacity(num_games as usize);
        for chunk in results {
            all.extend(chunk);
        }
        all
    };

    #[cfg(not(feature = "parallel"))]
    let all_replays = {
        println!("Threads:    1 (enable --features parallel for multi-threaded)");
        generate_chunk(&dmc_model_path, &bid_model_path, num_games, seed, 0, 1, (score_ns, score_ew))
    };

    let elapsed = start.elapsed().as_secs_f64();
    let total_actions: usize = all_replays.iter().map(|r| r.actions.len()).sum();
    let void_deals = all_replays.iter().filter(|r| r.actions.len() <= 4).count();
    println!(
        "\nGeneration complete: {} games in {:.1}s ({:.0} games/s)",
        all_replays.len(), elapsed, all_replays.len() as f64 / elapsed,
    );
    println!(
        "Avg actions/game: {:.1}, void deals: {} ({:.1}%)",
        total_actions as f64 / all_replays.len() as f64,
        void_deals,
        void_deals as f64 / all_replays.len() as f64 * 100.0,
    );

    // Write
    println!("Writing to {}...", output_path);
    GameReplay::write_all(&output_path, &all_replays).unwrap();

    let file_size = std::fs::metadata(&output_path).unwrap().len();
    println!(
        "Written {} ({:.1} MB, {:.1} bytes/game)",
        output_path,
        file_size as f64 / (1024.0 * 1024.0),
        file_size as f64 / all_replays.len() as f64,
    );
}

fn generate_chunk(
    dmc_model_path: &str,
    bid_model_path: &str,
    num_games: u64,
    seed: u64,
    thread_id: usize,
    num_threads: usize,
    scores: (i32, i32),
) -> Vec<GameReplay> {
    let mut dmc_net = DmcNet::load(dmc_model_path).unwrap();
    if dmc_net.obs_dim() == OBS_DIM_TR {
        // Canonical-obs models (DouDou50 lineage) are ResNets.
        dmc_net.set_residual(true);
    }
    let mut bid_net = BidNet::load(bid_model_path).ok();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut replays = Vec::with_capacity(num_games as usize);

    let mut dmc_obs_buf = vec![0.0f32; dmc_obs::OBS_DIM.max(OBS_DIM_TR)];

    let report_interval = if num_threads == 1 { 10_000u64 } else { 50_000 };
    let start = Instant::now();

    for game_idx in 0..num_games {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, &mut rng);
        let hands = state.hands;
        let mut tracking = EnvTracking::new();
        tracking.dealer = dealer;
        let mut actions = Vec::with_capacity(48);

        while !state.is_terminal() {
            let action = if state.phase == Phase::Bidding {
                bid_action(&state, &tracking, &mut bid_net, scores)
            } else {
                dmc_action(&state, &tracking, &mut dmc_net, &mut dmc_obs_buf)
            };

            actions.push(action);
            tracking.track_action(&state, action);
            state.step(action);
        }

        replays.push(GameReplay { dealer, hands, actions });

        if thread_id == 0 && ((game_idx + 1) % report_interval == 0 || game_idx + 1 == num_games) {
            let elapsed = start.elapsed().as_secs_f64();
            let total_games_est = (game_idx + 1) as f64 * num_threads as f64;
            let games_per_sec = total_games_est / elapsed;
            println!(
                "[~{:.0}k/{}k] ({:.0} games/s, {:.1}s)",
                total_games_est / 1000.0,
                num_games as f64 * num_threads as f64 / 1000.0,
                games_per_sec,
                elapsed,
            );
        }
    }

    replays
}

fn bid_action(
    state: &GameState,
    tracking: &EnvTracking,
    bid_net: &mut Option<BidNet>,
    scores: (i32, i32),
) -> u8 {
    if let Some(ref mut net) = bid_net {
        let legal = state.legal_actions();
        let obs_dim = net.obs_dim();
        if obs_dim > BID_OBS_DIM {
            // Score-aware model (bid v4/v5/v6). Par défaut 0-0 (donnes isolées) ;
            // `--score-ns/--score-ew` fabrique un corpus d'enchères de **milieu de
            // partie**, ce que playgen n'a jamais vu et ne peut pas déduire — le
            // score n'est pas dans son flux de tokens.
            //
            // L'obs prend `my_score, opp_score` **relatifs à l'annonceur**, pas des
            // scores de camp : passer (ns, ew) tel quel ferait croire aux quatre
            // sièges qu'ils sont dans le même camp. Épinglé par le contrôle de
            // symétrie — (1500,300) et (300,1500) doivent donner le même résultat
            // une fois moyennés sur les quatre observateurs, et ne le donnaient pas.
            let (s_ns, s_ew) = if GameState::player_team(state.current_player()) == 0 {
                scores
            } else {
                (scores.1, scores.0)
            };
            let mut buf = [0.0f32; BID_OBS_DIM_SCORE_AWARE_V3];
            if obs_dim == BID_OBS_DIM_SCORE_AWARE_V3 {
                bid_obs::write_bid_observation_score_aware_v3(
                    &mut buf, 0, state, &tracking.bid_history, s_ns, s_ew,
                );
            } else if obs_dim == BID_OBS_DIM_SCORE_AWARE_V2 {
                bid_obs::write_bid_observation_score_aware_v2(
                    &mut buf, 0, state, &tracking.bid_history, s_ns, s_ew,
                );
            } else {
                debug_assert_eq!(obs_dim, BID_OBS_DIM_SCORE_AWARE);
                bid_obs::write_bid_observation_score_aware(
                    &mut buf, 0, state, &tracking.bid_history, s_ns, s_ew,
                );
            }
            net.best_action_fast(&buf[..obs_dim], legal)
        } else {
            let obs = bid_obs::make_bid_observation(state, &tracking.bid_history);
            net.best_action_fast(&obs, legal)
        }
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
    if dmc_net.obs_dim() == OBS_DIM_TR {
        // Canonical obs (DouDou50): canonical mask in, physical card out.
        dmc_obs::write_observation_tr(obs_buf, 0, state, tracking);
        let order = dmc_obs::current_player_order(state, tracking);
        let canonical_mask = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
        let (canonical_best, _) = dmc_net.best_action(obs_buf, canonical_mask);
        dmc_obs::card_to_physical(canonical_best, &order)
    } else {
        dmc_obs::write_observation(obs_buf, 0, state, tracking);
        let legal_mask = state.legal_actions() as u32;
        let (best, _) = dmc_net.best_action(obs_buf, legal_mask);
        best
    }
}
