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
/// Usage:
///   cargo run -p colver-core --bin generate_belief_data --release -- \
///     --dmc-model models/dmc_final.bin \
///     --bid-model models/bid_nn_final.bin \
///     --games 50000 --output data/belief_train.bin

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

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut dmc_model_path = String::from("models/dmc_final.bin");
    let mut bid_model_path = String::from("models/bid_nn_final.bin");
    let mut num_games: u64 = 50_000;
    let mut output_path = String::from("data/belief_train.bin");
    let mut seed: u64 = 42;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--dmc-model" => { dmc_model_path = args[i + 1].clone(); i += 2; }
            "--bid-model" => { bid_model_path = args[i + 1].clone(); i += 2; }
            "--games" => { num_games = args[i + 1].parse().unwrap(); i += 2; }
            "--output" => { output_path = args[i + 1].clone(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    println!("=== Belief Data Generation ===");
    println!("DMC model:  {}", dmc_model_path);
    println!("Bid model:  {}", bid_model_path);
    println!("Games:      {}", num_games);
    println!("Output:     {}", output_path);
    println!("Seed:       {}", seed);

    // Load models
    let mut dmc_net = DmcNet::load(&dmc_model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load DMC model: {}", e);
        std::process::exit(1);
    });
    println!("DMC model loaded (obs_dim={}, hidden={})", dmc_net.obs_dim(), dmc_net.hidden());

    let mut bid_net = match BidNet::load(&bid_model_path) {
        Ok(net) => {
            println!("Bid model loaded (obs_dim={}, dueling={})", net.obs_dim(), net.is_dueling());
            Some(net)
        }
        Err(e) => {
            println!("Bid model not found ({}), using improved_v2", e);
            None
        }
    };

    let mut rng = StdRng::seed_from_u64(seed);

    // Collect samples in memory
    let mut all_obs: Vec<f32> = Vec::new();
    let mut all_targets: Vec<u8> = Vec::new();
    let mut all_masks: Vec<u32> = Vec::new();
    let mut total_samples: u64 = 0;

    let start = Instant::now();
    let mut obs_buf = vec![0.0f32; BELIEF_OBS_DIM];
    let mut dmc_obs_buf = vec![0.0f32; dmc_obs::OBS_DIM];

    for game_idx in 0..num_games {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut tracking = EnvTracking::new();
        tracking.dealer = dealer;

        // Save ground truth hands at deal start
        let true_hands = state.hands;

        // Play through the game
        while !state.is_terminal() {
            let player = state.current_player();

            // Record observation + target at each play step BEFORE the action
            if state.phase == Phase::Playing {
                let observer = player;

                // Write belief observation
                write_belief_obs(&mut obs_buf, &state, &tracking, observer);

                // Build target: for each card, which player holds it?
                let mut target = [0u8; 32];
                for p in 0..4u8 {
                    for c in 0..32u8 {
                        if true_hands[p as usize] & (1u32 << c) != 0 {
                            target[c as usize] = p;
                        }
                    }
                }
                // For played cards and current trick cards, target is arbitrary
                // (they'll be masked out during training)

                // Build unknown mask: cards not in observer's hand and not yet played
                let observer_hand = state.hands[observer as usize];
                let mut played = state.played_cards;
                for j in 0..4 {
                    let c = state.current_trick[j];
                    if c != card::EMPTY {
                        played |= 1u32 << c;
                    }
                }
                let unknown_mask = !observer_hand & !played;

                // Only record if there are unknown cards (always true mid-game)
                if unknown_mask != 0 {
                    all_obs.extend_from_slice(&obs_buf);
                    all_targets.extend_from_slice(&target);
                    all_masks.push(unknown_mask);
                    total_samples += 1;
                }
            }

            // Choose action
            let action = if state.phase == Phase::Bidding {
                bid_action(&state, &tracking.bid_history, &mut bid_net)
            } else {
                dmc_action(&state, &tracking, &mut dmc_net, &mut dmc_obs_buf, &mut rng)
            };

            tracking.track_action(&state, action);
            state.step(action);
        }

        if (game_idx + 1) % 10_000 == 0 || game_idx + 1 == num_games {
            let elapsed = start.elapsed().as_secs_f64();
            let games_per_sec = (game_idx + 1) as f64 / elapsed;
            println!(
                "[{}/{}] samples={} ({:.0} games/s, {:.1}s)",
                game_idx + 1, num_games, total_samples, games_per_sec, elapsed,
            );
        }
    }

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
    // Create parent directories if needed
    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut file = std::fs::File::create(&output_path).unwrap();

    // Header
    file.write_all(MAGIC).unwrap();
    file.write_all(&(BELIEF_OBS_DIM as u32).to_le_bytes()).unwrap();
    file.write_all(&total_samples.to_le_bytes()).unwrap();

    // Samples
    for i in 0..total_samples as usize {
        let obs_start = i * BELIEF_OBS_DIM;
        let obs_end = obs_start + BELIEF_OBS_DIM;
        let obs_bytes: Vec<u8> = all_obs[obs_start..obs_end]
            .iter()
            .flat_map(|f| f.to_le_bytes())
            .collect();
        file.write_all(&obs_bytes).unwrap();

        let target_start = i * 32;
        file.write_all(&all_targets[target_start..target_start + 32]).unwrap();

        file.write_all(&all_masks[i].to_le_bytes()).unwrap();
    }

    let file_size = std::fs::metadata(&output_path).unwrap().len();
    println!(
        "Written {} ({:.1} MB, {:.0} bytes/sample)",
        output_path,
        file_size as f64 / (1024.0 * 1024.0),
        file_size as f64 / total_samples as f64,
    );
}

fn write_belief_obs(buf: &mut [f32], state: &GameState, tracking: &EnvTracking, observer: u8) {
    belief_obs::write_belief_observation(buf, 0, state, tracking, observer);
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
    _rng: &mut impl Rng,
) -> u8 {
    dmc_obs::write_observation(obs_buf, 0, state, tracking);
    let legal_mask = state.legal_actions() as u32;
    let (best, _) = dmc_net.best_action(obs_buf, legal_mask);
    best
}
