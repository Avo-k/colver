/// Evaluate a trained belief network.
///
/// Three modes:
/// A. Offline accuracy test: load model + validation data, report accuracy/CE/calibration.
/// B. Online match play: IS-DD + NN beliefs vs IS-DD + heuristic CardBeliefs.
/// C. Diagnose: print concrete per-card predictions on sample game positions.
///
/// Usage:
///   # Offline eval
///   cargo run -p colver-core --bin belief_eval --release -- \
///     --model models/belief_net.bin --data data/belief/belief_train.bin --mode offline
///
///   # Online match play
///   cargo run -p colver-core --bin belief_eval --release -- \
///     --model models/belief_net.bin --bid-model models/bid_nn_final.bin \
///     --mode match --matches 100 --time-ms 20
///
///   # Diagnose (concrete predictions)
///   cargo run -p colver-core --bin belief_eval --release -- \
///     --model models/belief_net.bin --replays data/training/games_500k.bin --mode diagnose

use std::io::Read;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::belief_net::{self, BeliefNet};
use colver_core::belief_obs::{self, BELIEF_OBS_DIM, BELIEF_OBS_DIM_V2, BELIEF_OBS_DIM_V3};
use colver_core::bid_eval;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::card;
use colver_core::card_beliefs::CardBeliefs;
use colver_core::dmc_obs::EnvTracking;
use colver_core::game_replay::GameReplay;
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::state::{GameState, Phase};

const MAGIC: &[u8; 8] = b"COLVBL01";

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut model_path = String::from("models/belief_net.bin");
    let mut data_path = String::new();
    let mut replays_path = String::new();
    let mut bid_model_path = String::new();
    let mut mode = String::from("offline");
    let mut matches: usize = 100;
    let mut time_ms: u32 = 20;
    let mut seed: u64 = 42;
    let mut num_games: usize = 5;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model_path = args[i + 1].clone(); i += 2; }
            "--data" => { data_path = args[i + 1].clone(); i += 2; }
            "--replays" => { replays_path = args[i + 1].clone(); i += 2; }
            "--bid-model" => { bid_model_path = args[i + 1].clone(); i += 2; }
            "--mode" => { mode = args[i + 1].clone(); i += 2; }
            "--matches" => { matches = args[i + 1].parse().unwrap(); i += 2; }
            "--time-ms" => { time_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--games" => { num_games = args[i + 1].parse().unwrap(); i += 2; }
            other => {
                eprintln!("Unknown argument: {}", other);
                std::process::exit(1);
            }
        }
    }

    match mode.as_str() {
        "offline" => run_offline_eval(&model_path, &data_path),
        "match" => run_match_eval(&model_path, &bid_model_path, matches, time_ms, seed),
        "diagnose" => run_diagnose(&model_path, &replays_path, num_games, seed),
        "scenario" => run_scenario_test(&model_path),
        "per_trick" => run_per_trick_eval(&model_path, &replays_path, num_games),
        "ablation" => run_ablation(&model_path, &replays_path, num_games),
        "ensemble" => run_ensemble_eval(&model_path, &replays_path, num_games),
        "bidding" => run_bidding_eval(&model_path, &bid_model_path, num_games, seed),
        other => {
            eprintln!("Unknown mode: {} (expected 'offline', 'match', 'diagnose', 'scenario', 'per_trick', 'ablation', 'ensemble', or 'bidding')", other);
            std::process::exit(1);
        }
    }
}

fn run_offline_eval(model_path: &str, data_path: &str) {
    if data_path.is_empty() {
        eprintln!("--data is required for offline mode");
        std::process::exit(1);
    }

    println!("=== Offline Belief Evaluation ===");
    println!("Model: {}", model_path);
    println!("Data:  {}", data_path);

    let mut net = BeliefNet::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });
    println!("Model loaded (obs_dim={}, hidden={})", net.obs_dim(), net.hidden());

    // Load data
    let mut file = std::fs::File::open(data_path).unwrap();
    let mut magic = [0u8; 8];
    file.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, MAGIC);

    let mut buf4 = [0u8; 4];
    file.read_exact(&mut buf4).unwrap();
    let obs_dim = u32::from_le_bytes(buf4) as usize;
    assert_eq!(obs_dim, BELIEF_OBS_DIM);

    let mut buf8 = [0u8; 8];
    file.read_exact(&mut buf8).unwrap();
    let num_samples = u64::from_le_bytes(buf8) as usize;
    println!("Evaluating on {} samples...", num_samples);

    let sample_bytes = BELIEF_OBS_DIM * 4 + 32 + 4;
    let mut raw = vec![0u8; sample_bytes];

    let mut correct = 0u64;
    let mut total = 0u64;
    let mut ce_sum = 0.0f64;
    let mut ce_count = 0u64;

    // Per-trick accuracy (8 tricks, but we track by completed tricks 0-7)
    let mut trick_correct = [0u64; 8];
    let mut trick_total = [0u64; 8];

    // Calibration: 10 bins of predicted confidence
    let mut cal_correct = [0u64; 10];
    let mut cal_total = [0u64; 10];

    let start = Instant::now();
    let max_eval = num_samples.min(200_000); // cap to avoid excessive time

    for sample_idx in 0..max_eval {
        file.read_exact(&mut raw).unwrap();

        // Parse obs
        let mut obs = [0.0f32; BELIEF_OBS_DIM];
        for j in 0..BELIEF_OBS_DIM {
            let off = j * 4;
            obs[j] = f32::from_le_bytes([raw[off], raw[off + 1], raw[off + 2], raw[off + 3]]);
        }

        // Parse target
        let target_off = BELIEF_OBS_DIM * 4;
        let target = &raw[target_off..target_off + 32];

        // Parse mask
        let mask_off = target_off + 32;
        let mask = u32::from_le_bytes([
            raw[mask_off], raw[mask_off + 1], raw[mask_off + 2], raw[mask_off + 3],
        ]);

        // Forward pass
        let logits = net.evaluate(&obs);

        // Trick number from obs (Block 11, offset 328)
        let trick_num = (obs[BELIEF_OBS_DIM - 2] * 8.0).round() as usize;
        let trick_idx = trick_num.min(7);

        // Per-card evaluation
        let nc = net.num_classes();
        for c in 0..32u32 {
            if mask & (1 << c) == 0 {
                continue;
            }

            let base = c as usize * nc;
            // Old data format uses 4-class targets (0=me,1=left,2=partner,3=right)
            // Remap to 3-class if needed
            let raw_target = target[c as usize] as usize;
            let true_player = if nc == 3 && raw_target > 0 { raw_target - 1 } else { raw_target };

            // Softmax
            let mut max_l = f32::NEG_INFINITY;
            for p in 0..nc {
                if logits[base + p] > max_l {
                    max_l = logits[base + p];
                }
            }
            let mut exp_sum = 0.0f32;
            let mut exps = vec![0.0f32; nc];
            for p in 0..nc {
                exps[p] = (logits[base + p] - max_l).exp();
                exp_sum += exps[p];
            }
            let probs: Vec<f32> = exps.iter().map(|&e| e / exp_sum).collect();

            // Accuracy (argmax)
            let mut pred = 0;
            for p in 1..nc {
                if probs[p] > probs[pred] {
                    pred = p;
                }
            }
            if pred == true_player {
                correct += 1;
                trick_correct[trick_idx] += 1;
            }
            total += 1;
            trick_total[trick_idx] += 1;

            // Cross-entropy
            let true_prob = probs[true_player].max(1e-10);
            ce_sum += -(true_prob as f64).ln();
            ce_count += 1;

            // Calibration
            let confidence = probs[pred];
            let bin = ((confidence * 10.0) as usize).min(9);
            cal_total[bin] += 1;
            if pred == true_player {
                cal_correct[bin] += 1;
            }
        }

        if (sample_idx + 1) % 50_000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let acc = correct as f64 / total as f64 * 100.0;
            let ce = ce_sum / ce_count as f64;
            println!("  [{}/{}] acc={:.2}% CE={:.4} ({:.1}s)", sample_idx + 1, max_eval, acc, ce, elapsed);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== Results ===");
    println!("Samples evaluated: {}", max_eval);
    println!("Per-card accuracy: {:.2}% ({}/{})", correct as f64 / total as f64 * 100.0, correct, total);
    println!("Cross-entropy:     {:.4}", ce_sum / ce_count as f64);
    println!("Random baseline:   {:.2}% (33.3%)", 100.0 / 3.0);
    println!("Elapsed:           {:.1}s", elapsed);

    println!("\n--- Accuracy by trick number ---");
    for t in 0..8 {
        if trick_total[t] > 0 {
            let acc = trick_correct[t] as f64 / trick_total[t] as f64 * 100.0;
            println!("  Trick {}: {:.1}% ({}/{})", t, acc, trick_correct[t], trick_total[t]);
        }
    }

    println!("\n--- Calibration ---");
    println!("  {:>12} {:>10} {:>10} {:>8}", "Confidence", "Correct", "Total", "Actual%");
    for bin in 0..10 {
        if cal_total[bin] > 0 {
            let actual = cal_correct[bin] as f64 / cal_total[bin] as f64 * 100.0;
            println!(
                "  {:>4.0}-{:>4.0}%  {:>10} {:>10} {:>7.1}%",
                bin as f64 * 10.0, (bin + 1) as f64 * 10.0,
                cal_correct[bin], cal_total[bin], actual,
            );
        }
    }
}

fn run_match_eval(
    model_path: &str,
    bid_model_path: &str,
    num_matches: usize,
    time_ms: u32,
    seed: u64,
) {
    println!("=== Online Match Evaluation ===");
    println!("Belief model: {}", model_path);
    println!("Bid model:    {}", bid_model_path);
    println!("Matches:      {} (duplicate pairs = {})", num_matches, num_matches / 2);
    println!("Time/move:    {}ms", time_ms);

    // Load bid model
    let mut bid_net = if !bid_model_path.is_empty() {
        match BidNet::load(bid_model_path) {
            Ok(net) => {
                println!("Bid model loaded");
                Some(net)
            }
            Err(e) => {
                println!("Bid model not found ({}), using improved_v2", e);
                None
            }
        }
    } else {
        None
    };

    // Duplicate matching: each pair uses same seed, teams swapped
    let num_pairs = num_matches / 2;
    let mut nn_wins = 0u32;
    let mut heuristic_wins = 0u32;
    let mut nn_margin_sum = 0.0f64;

    let start = Instant::now();

    for pair in 0..num_pairs {
        let pair_seed = seed + pair as u64;

        // Match A: NN beliefs = team 0, heuristic = team 1
        let mut rng_a = StdRng::seed_from_u64(pair_seed);
        let (nn_pts_a, heur_pts_a) = play_match(
            model_path, &mut bid_net, time_ms, 0, &mut rng_a,
        );
        if nn_pts_a >= 2000.0 {
            nn_wins += 1;
        } else {
            heuristic_wins += 1;
        }
        nn_margin_sum += (nn_pts_a - heur_pts_a) as f64;

        // Match B: NN beliefs = team 1, heuristic = team 0
        let mut rng_b = StdRng::seed_from_u64(pair_seed);
        let (nn_pts_b, heur_pts_b) = play_match(
            model_path, &mut bid_net, time_ms, 1, &mut rng_b,
        );
        if nn_pts_b >= 2000.0 {
            nn_wins += 1;
        } else {
            heuristic_wins += 1;
        }
        nn_margin_sum += (nn_pts_b - heur_pts_b) as f64;

        if (pair + 1) % 10 == 0 || pair + 1 == num_pairs {
            let total = (pair + 1) * 2;
            let wr = nn_wins as f64 / total as f64 * 100.0;
            let margin = nn_margin_sum / total as f64;
            let elapsed = start.elapsed().as_secs_f64();
            println!(
                "  [{}/{}] NN: {}W-{}L ({:.1}%) margin={:+.0} ({:.1}s)",
                total, num_pairs * 2, nn_wins, heuristic_wins, wr, margin, elapsed,
            );
        }
    }

    let total_matches = num_pairs * 2;
    let elapsed = start.elapsed().as_secs_f64();
    println!("\n=== Results ===");
    println!(
        "NN beliefs:       {}W-{}L ({:.1}%)",
        nn_wins, heuristic_wins,
        nn_wins as f64 / total_matches as f64 * 100.0,
    );
    println!(
        "Avg margin:       {:+.0}",
        nn_margin_sum / total_matches as f64,
    );
    println!("Elapsed:          {:.1}s ({:.1}s/match)", elapsed, elapsed / total_matches as f64);
}

/// Play a match to 2000 between NN-belief IS-DD and heuristic IS-DD.
/// Returns (nn_team_points, heuristic_team_points).
fn play_match(
    belief_model_path: &str,
    bid_net: &mut Option<BidNet>,
    time_ms: u32,
    nn_team: u8,
    rng: &mut StdRng,
) -> (f32, f32) {
    let nn_config = IsDdConfig {
        time_limit_ms: Some(time_ms),
        use_nn_beliefs: true,
        use_hard_constraints: false,
        ..Default::default()
    };
    let heur_config = IsDdConfig {
        time_limit_ms: Some(time_ms),
        use_nn_beliefs: false,
        ..Default::default()
    };

    // Create searches for all 4 players
    let mut searches: Vec<IsDdSearch> = (0..4).map(|_| IsDdSearch::new()).collect();

    // Load belief net for NN team players
    for p in 0..4u8 {
        if GameState::player_team(p) == nn_team {
            searches[p as usize].load_belief_net(belief_model_path).unwrap_or_else(|e| {
                eprintln!("Failed to load belief net: {}", e);
                std::process::exit(1);
            });
        }
    }

    let mut nn_total = 0.0f32;
    let mut heur_total = 0.0f32;

    for _ in 0..100 {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, rng);

        // Init all searches
        for p in 0..4u8 {
            searches[p as usize].init_deal(&state, p, true);
        }

        while !state.is_terminal() {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if state.phase == Phase::Bidding {
                bid_action(&state, bid_net)
            } else if team == nn_team {
                searches[player as usize].search(&state, &nn_config, rng)
            } else {
                searches[player as usize].search(&state, &heur_config, rng)
            };

            // Record for all players
            for s in searches.iter_mut() {
                s.record_action(&state, player, action);
            }
            state.step(action);
        }

        let rewards = state.rewards();
        nn_total += rewards[nn_team as usize];
        heur_total += rewards[1 - nn_team as usize];

        if nn_total >= 2000.0 || heur_total >= 2000.0 {
            break;
        }
    }

    (nn_total, heur_total)
}

fn bid_action(state: &GameState, bid_net: &mut Option<BidNet>) -> u8 {
    if let Some(ref mut net) = bid_net {
        // We don't track bid history here, use empty for simplicity
        // (the IS-DD searches handle their own bidding)
        let obs = bid_obs::make_bid_observation(state, &[]);
        let legal = state.legal_actions();
        net.best_action_fast(&obs, legal)
    } else {
        bid_eval::improved_v2_bid(state)
    }
}

fn run_diagnose(model_path: &str, replays_path: &str, num_games: usize, seed: u64) {
    if replays_path.is_empty() {
        eprintln!("--replays is required for diagnose mode");
        std::process::exit(1);
    }

    println!("=== Belief Network Diagnosis ===");
    println!("Model:   {}", model_path);
    println!("Replays: {}", replays_path);
    println!("Games:   {}", num_games);

    let mut net = BeliefNet::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });
    println!("Model loaded (obs_dim={}, hidden={})\n", net.obs_dim(), net.hidden());

    let replays = GameReplay::load_all(replays_path).unwrap_or_else(|e| {
        eprintln!("Failed to load replays: {}", e);
        std::process::exit(1);
    });

    let player_names = ["North", "East", "South", "West"];
    let rel_names = ["Me", "Left", "Prtner", "Right"];
    let suit_symbols = ["♠", "♥", "♦", "♣"];

    let _rng = StdRng::seed_from_u64(seed);
    let step = replays.len().max(1) / num_games.max(1);

    let mut total_correct = 0u64;
    let mut total_unknown = 0u64;

    for game_idx in 0..num_games {
        let replay_idx = (game_idx * step) % replays.len();
        let replay = &replays[replay_idx];

        let mut state = GameState::new(replay.dealer, replay.hands);
        let mut tracking = EnvTracking::new();
        tracking.dealer = replay.dealer;
        let true_hands = replay.hands;

        println!("{}", "=".repeat(70));
        println!(
            "Game {} (replay #{}), dealer={}",
            game_idx + 1, replay_idx, player_names[replay.dealer as usize],
        );

        // Print initial hands
        println!("\nDealt hands:");
        for p in 0..4u8 {
            println!("  {:>5}: {}", player_names[p as usize], card::cardset_str(true_hands[p as usize]));
        }

        // Play through bidding, recording each bid
        println!("\nBidding:");
        let mut action_idx = 0;
        while action_idx < replay.actions.len() && state.phase == Phase::Bidding {
            let action = replay.actions[action_idx];
            let player = state.current_player();
            let action_str = format_bid_action(action);
            println!("  {:>5}: {}", player_names[player as usize], action_str);
            tracking.track_action(&state, action);
            state.step(action);
            action_idx += 1;
        }

        if state.is_terminal() || state.phase != Phase::Playing {
            println!("  → Void deal\n");
            continue;
        }

        // Print contract
        let trump_suit = state.contract.trump as usize;
        let trump_name = suit_symbols[trump_suit];
        let bid_value = state.contract.point_value();
        let taker_team = if state.contract.team == 0 { "NS" } else { "EW" };
        let coinche_str = match state.contract.coinche {
            0 => "",
            1 => " coinchée",
            _ => " surcoinchée",
        };
        println!(
            "  → Contract: {}{} by {}{}",
            bid_value, trump_name, taker_team, coinche_str,
        );

        // Play tricks and print them
        println!("\nPlay:");
        let mut trick_num = 1;
        let mut trick_cards: Vec<(u8, u8)> = Vec::new(); // (player, card)

        // Play until target position (after 2 complete tricks)
        let target_play_steps = 8;
        let mut play_steps = 0;
        while action_idx < replay.actions.len() && play_steps < target_play_steps {
            let action = replay.actions[action_idx];
            let player = state.current_player();

            trick_cards.push((player, action));

            if trick_cards.len() == 4 || state.trick_count == 3 {
                // About to complete a trick (or this is the 4th card)
            }

            tracking.track_action(&state, action);
            state.step(action);
            action_idx += 1;
            play_steps += 1;

            // Check if trick just completed (play_order length is multiple of 4)
            if tracking.play_order.len() % 4 == 0 && !trick_cards.is_empty() {
                let cards_str: Vec<String> = trick_cards.iter().map(|(p, c)| {
                    format!("{}({})", card::card_name(*c), player_names[*p as usize])
                }).collect();
                println!("  Trick {}: {}", trick_num, cards_str.join(", "));
                trick_num += 1;
                trick_cards.clear();
            }
        }

        // Print any partial trick in progress
        if !trick_cards.is_empty() {
            let cards_str: Vec<String> = trick_cards.iter().map(|(p, c)| {
                format!("{}({})", card::card_name(*c), player_names[*p as usize])
            }).collect();
            println!("  Trick {} (in progress): {}", trick_num, cards_str.join(", "));
        }

        if state.is_terminal() {
            continue;
        }

        let observer = state.current_player();
        let completed_tricks = tracking.play_order.len() / 4;

        println!(
            "\n--- Position: {}'s turn to play (trick {}) ---",
            player_names[observer as usize], completed_tricks + 1,
        );

        // Print remaining hands
        println!("\nRemaining hands:");
        let mut in_trick = 0u32;
        for j in 0..4 {
            let c = state.current_trick[j];
            if c != card::EMPTY {
                in_trick |= 1u32 << c;
            }
        }
        for p in 0..4u8 {
            let remaining = true_hands[p as usize] & !state.played_cards & !in_trick;
            let marker = if p == observer { " ← observer" } else { "" };
            println!(
                "  {:>5}: {}{}",
                player_names[p as usize],
                card::cardset_str(remaining),
                marker,
            );
        }

        // Run belief net
        let logits = eval_belief(&mut net, &state, &tracking, observer);

        // Compute masks
        let observer_hand = state.hands[observer as usize];
        let mut played = state.played_cards;
        for j in 0..4 {
            let c = state.current_trick[j];
            if c != card::EMPTY {
                played |= 1u32 << c;
            }
        }
        let unknown_mask = !observer_hand & !played;

        // Print ALL 32 cards grouped by suit
        println!("\nAll cards (from {}'s perspective):", player_names[observer as usize]);
        println!(
            "  {:>4}  {:>8}  {:>8}  {:>6} {:>6} {:>6} {:>6}",
            "Card", "Status", "True", "Me", "Left", "Prtner", "Right",
        );
        println!("  {}", "-".repeat(62));

        let mut game_correct = 0u32;
        let mut game_total = 0u32;

        for suit in 0..4u8 {
            for rank in (0..8u8).rev() {
                let c = suit * 8 + rank;

                // Find true holder (relative)
                let mut true_abs = 0u8;
                for p in 0..4u8 {
                    if true_hands[p as usize] & (1u32 << c) != 0 {
                        true_abs = p;
                        break;
                    }
                }
                let true_rel = (true_abs + 4 - observer) % 4;
                let true_str = rel_names[true_rel as usize];

                let is_trump = suit == state.contract.trump;
                let trump_mark = if is_trump { "*" } else { " " };

                if observer_hand & (1u32 << c) != 0 {
                    // In my hand
                    println!(
                        "  {:>3}{} {:>8}  {:>8}",
                        card::card_name(c), trump_mark, "MY HAND", true_str,
                    );
                } else if played & (1u32 << c) != 0 {
                    // Already played
                    // Find who played it
                    let mut played_by_str = "?";
                    for p in 0..4usize {
                        if tracking.played_by[p] & (1u32 << c) != 0 {
                            played_by_str = rel_names[(p as u8 + 4 - observer) as usize % 4];
                            break;
                        }
                    }
                    // Check current trick
                    for j in 0..4usize {
                        if state.current_trick[j] == c {
                            played_by_str = rel_names[(j as u8 + 4 - observer) as usize % 4];
                            break;
                        }
                    }
                    println!(
                        "  {:>3}{} {:>8}  {:>8}",
                        card::card_name(c), trump_mark,
                        format!("played"), played_by_str,
                    );
                } else {
                    // Unknown — show prediction
                    let nc = net.num_classes();
                    let probs = card_probs(&logits, c as usize, nc);

                    // Argmax over non-observer slots (1..4 in 4-class display)
                    let mut pred_rel = 1usize;
                    for p in 2..4 {
                        if probs[p] > probs[pred_rel] {
                            pred_rel = p;
                        }
                    }

                    let correct = pred_rel as u8 == true_rel;
                    if correct {
                        game_correct += 1;
                        total_correct += 1;
                    }
                    game_total += 1;
                    total_unknown += 1;

                    let mark = if correct { "✓" } else { "✗" };
                    println!(
                        "  {:>3}{} {:>8}  {:>8}  {:>5.0}% {:>5.0}% {:>5.0}% {:>5.0}%  {}",
                        card::card_name(c), trump_mark,
                        "?",
                        true_str,
                        probs[0] * 100.0,
                        probs[1] * 100.0,
                        probs[2] * 100.0,
                        probs[3] * 100.0,
                        mark,
                    );
                }
            }
            if suit < 3 {
                println!("  {}", "-".repeat(62));
            }
        }

        println!(
            "\n  Unknown card accuracy: {}/{} ({:.0}%)\n",
            game_correct, game_total,
            game_correct as f64 / game_total.max(1) as f64 * 100.0,
        );
    }

    println!("{}", "=".repeat(70));
    println!(
        "Overall: {}/{} ({:.1}%)",
        total_correct, total_unknown,
        total_correct as f64 / total_unknown.max(1) as f64 * 100.0,
    );
}

/// Test the belief model on hand-crafted scenarios to check what it has learned.
fn run_scenario_test(model_path: &str) {
    println!("=== Belief Model Scenario Tests ===");
    println!("Model: {}\n", model_path);

    let mut net = BeliefNet::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });

    // Card indices: suit*8 + rank. Ranks: 7=0, 8=1, 9=2, J=3, Q=4, K=5, 10=6, A=7
    // Suits: Spades=0, Hearts=1, Diamonds=2, Clubs=3
    // Trump strength: J(7) > 9(6) > A(5) > 10(4) > K(3) > Q(2) > 8(1) > 7(0)

    scenario_trump_ceiling(&mut net);
    scenario_void_detection(&mut net);
    scenario_bidding_signal(&mut net);
}

/// Helper: compute per-card softmax from logits.
/// Returns [f32; 4] with probabilities. For 3-class models, slot 0 (observer) is 0.
fn card_probs(logits: &[f32; 128], card: usize, num_classes: usize) -> [f32; 4] {
    let base = card * num_classes;
    let mut max_l = f32::NEG_INFINITY;
    for p in 0..num_classes {
        if logits[base + p] > max_l { max_l = logits[base + p]; }
    }
    let mut raw = [0.0f32; 4];
    let mut sum = 0.0f32;
    for p in 0..num_classes {
        let e = (logits[base + p] - max_l).exp();
        raw[p] = e;
        sum += e;
    }
    for p in 0..num_classes { raw[p] /= sum; }

    if num_classes == 3 {
        // Remap: raw[0]=left, raw[1]=partner, raw[2]=right → [observer=0, left, partner, right]
        [0.0, raw[0], raw[1], raw[2]]
    } else {
        [raw[0], raw[1], raw[2], raw[3]]
    }
}

fn print_card_prediction(name: &str, probs: &[f32; 4], rel_names: &[&str; 4], highlight_rel: Option<usize>, expected_low: bool) {
    let mark = if let Some(h) = highlight_rel {
        if expected_low && probs[h] < 0.15 { "  ✓ learned!" }
        else if expected_low && probs[h] < 0.25 { "  ~ partial" }
        else if !expected_low && probs[h] > 0.40 { "  ✓ learned!" }
        else { "  ✗" }
    } else { "" };
    println!(
        "  {:>5}: {:>6}={:>4.0}%  {:>6}={:>4.0}%  {:>6}={:>4.0}%  {:>6}={:>4.0}%{}",
        name,
        rel_names[0], probs[0] * 100.0,
        rel_names[1], probs[1] * 100.0,
        rel_names[2], probs[2] * 100.0,
        rel_names[3], probs[3] * 100.0,
        mark,
    );
}

/// Build observation and evaluate, handling V1 (330-dim) and V2 (304-dim) models.
/// For V2, computes hard constraints from state + optional TrumpCeilingTracker.
fn eval_belief(
    net: &mut BeliefNet,
    state: &GameState,
    tracking: &EnvTracking,
    observer: u8,
) -> [f32; 128] {
    eval_belief_with_tracker(net, state, tracking, observer, None)
}

/// Like eval_belief but with optional TrumpCeilingTracker for full hard constraints.
fn eval_belief_with_tracker(
    net: &mut BeliefNet,
    state: &GameState,
    tracking: &EnvTracking,
    observer: u8,
    tracker: Option<&colver_core::game_replay::TrumpCeilingTracker>,
) -> [f32; 128] {
    if net.obs_dim() == BELIEF_OBS_DIM_V2 {
        let hard_constraints = if let Some(t) = tracker {
            t.compute_hard_constraints(state, observer)
        } else {
            build_hard_constraints_from_state(state, observer)
        };
        let mut obs_buf = vec![0.0f32; BELIEF_OBS_DIM_V2];
        belief_obs::write_belief_observation_v2(&mut obs_buf, 0, state, tracking, observer, &hard_constraints);
        net.evaluate(&obs_buf)
    } else {
        let mut obs_buf = vec![0.0f32; BELIEF_OBS_DIM];
        belief_obs::write_belief_observation(&mut obs_buf, 0, state, tracking, observer);
        net.evaluate(&obs_buf)
    }
}

/// Build hard constraints from GameState for V2 eval.
/// Uses state.voids for void constraints. No trump ceiling (would need TrumpCeilingTracker).
/// For full trump ceiling support, use TrumpCeilingTracker from game_replay.
fn build_hard_constraints_from_state(state: &GameState, observer: u8) -> [f32; 96] {
    let mut hc = [0.0f32; 96];
    let seats = [
        ((observer as usize + 1) % 4),
        ((observer as usize + 2) % 4),
        ((observer as usize + 3) % 4),
    ];
    let observer_hand = state.hands[observer as usize];
    let mut played = state.played_cards;
    for j in 0..4 {
        let c = state.current_trick[j];
        if c != card::EMPTY {
            played |= 1u32 << c;
        }
    }
    let known = observer_hand | played;

    for (i, &seat) in seats.iter().enumerate() {
        for card_idx in 0..32u32 {
            let offset = i * 32 + card_idx as usize;
            let suit = (card_idx / 8) as u8;

            if known & (1 << card_idx) != 0 {
                hc[offset] = 1.0;
                continue;
            }
            if state.voids[seat] & (1 << suit) != 0 {
                hc[offset] = 1.0;
            }
        }
    }
    hc
}

/// Scenario 1: Trump ceiling.
/// Trump=♠. P0 leads K♠, P1 plays 8♠ (can't overtrump).
/// Observer=P2. P1 should NOT have J♠/9♠/A♠/10♠ (all stronger than K♠).
fn scenario_trump_ceiling(net: &mut BeliefNet) {
    println!("{}", "=".repeat(70));
    println!("SCENARIO 1: Trump Ceiling");
    println!("Trump=♠. P0 leads K♠ (strength 3). P1 plays 8♠ (strength 1).");
    println!("P1 couldn't overtrump → P1 should NOT have J♠/9♠/A♠/10♠.");
    println!("Observer: P2 (South)");
    println!();

    // Hands: 8 cards each, all 32 cards covered
    // P0: K♠ Q♠ | A♥ K♥ | 7♦ 8♦ | 7♣ 8♣
    // P1: 8♠ 7♠ | J♥ 9♥ | A♦ K♦ | 9♣ Q♣
    // P2: J♠ A♠ | Q♥ 8♥ | 10♦ 9♦ | A♣ 10♣
    // P3: 10♠ 9♠ | 10♥ 7♥ | J♦ Q♦ | K♣ J♣
    let hands: [u32; 4] = [
        (1<<5)|(1<<4)|(1<<15)|(1<<13)|(1<<16)|(1<<17)|(1<<24)|(1<<25),  // P0
        (1<<1)|(1<<0)|(1<<11)|(1<<10)|(1<<23)|(1<<21)|(1<<26)|(1<<28),  // P1
        (1<<3)|(1<<7)|(1<<12)|(1<<9)|(1<<22)|(1<<18)|(1<<31)|(1<<30),   // P2
        (1<<6)|(1<<2)|(1<<14)|(1<<8)|(1<<19)|(1<<20)|(1<<29)|(1<<27),   // P3
    ];

    let dealer = 3u8; // P0 bids first, leads first trick
    let mut state = GameState::new(dealer, hands);
    let mut tracking = EnvTracking::new();
    tracking.dealer = dealer;

    // Bidding: P0 bids 80♠ (action=1), P1/P2/P3 pass
    let mut tracker = colver_core::game_replay::TrumpCeilingTracker::new();
    for &action in &[1u8, 0, 0, 0] {
        tracking.track_action(&state, action);
        state.step(action);
    }
    assert_eq!(state.phase, Phase::Playing);

    // P0 leads K♠ (card 5)
    let player = state.current_player();
    tracker.record_play(&state, player, 5);
    tracking.track_action(&state, 5);
    state.step(5);

    // P1 plays 8♠ (card 1) — can't overtrump
    let player = state.current_player();
    tracker.record_play(&state, player, 1);
    tracking.track_action(&state, 1);
    state.step(1);

    // Now P2's turn (observer)
    assert_eq!(state.current_player(), 2);
    let observer = 2u8;

    let logits = eval_belief_with_tracker(net, &state, &tracking, observer, Some(&tracker));

    // Observer=P2: relative 0=P2(me), 1=P3(left), 2=P0(partner), 3=P1(right)
    let rel_names = ["Me", "Left", "Prtner", "Right"];

    println!("Spade predictions (P1 = 'Right'):");
    println!("  Should be LOW for Right: J♠(3), 9♠(2), A♠(7), 10♠(6)");
    println!("  (K♠ and 8♠ are on the current trick)\n");

    let spade_cards = [
        (7, "A♠"), (6, "10♠"), (3, "J♠"), (2, "9♠"),  // P1 shouldn't have these
        (4, "Q♠"), (0, "7♠"),  // Q♠ in P0's hand (known), 7♠ in P1's hand
    ];

    for &(card_idx, name) in &spade_cards {
        let probs = card_probs(&logits, card_idx, net.num_classes());
        let is_constraint_card = card_idx == 7 || card_idx == 6 || card_idx == 3 || card_idx == 2;
        print_card_prediction(name, &probs, &rel_names, if is_constraint_card { Some(3) } else { None }, true);
    }
    println!();
}

/// Scenario 2: Void detection.
/// Trump=♥. P0 leads A♦, P1 plays 7♣ (doesn't follow diamonds).
/// Observer=P2. P1 is void in diamonds AND trump (didn't cut).
fn scenario_void_detection(net: &mut BeliefNet) {
    println!("{}", "=".repeat(70));
    println!("SCENARIO 2: Void Detection (void in suit + void in trump)");
    println!("Trump=♥. P0 leads A♦. P1 plays 7♣ (discards, doesn't follow, doesn't cut).");
    println!("P1 is void in ♦ AND void in ♥ (trump).");
    println!("Observer: P2 (South)");
    println!();

    // Hands designed so P1 has no diamonds and no hearts
    // P0: 7♠ 8♠ | 7♥ 8♥ | A♦ K♦ | 7♣ 8♣
    // P1: 9♠ J♠ | (no ♥) | (no ♦) | 9♣ Q♣ K♣ J♣ A♣ 10♣ (6 clubs + 2 spades)
    // Wait, that's 8 cards for P1 but only 2 spades + 6 clubs = 8. Good.
    // P2: Q♠ K♠ | 9♥ J♥ | 10♦ 9♦ | (no clubs)
    // Wait, that's only 6. Need 8 cards each.
    // Let me redo:
    // P0: 7♠ 8♠ | 7♥ 8♥ | A♦ K♦ 10♦ J♦ | (no clubs) = 8
    // P1: 9♠ J♠ Q♠ K♠ | (no ♥) | (no ♦) | 7♣ 8♣ 9♣ Q♣ = 8
    // P2: 10♠ A♠ | 9♥ J♥ | 9♦ Q♦ | K♣ J♣ = 8
    // P3: 7♠... wait P0 already has 7♠.
    // Let me be more careful.

    // Spades (0-7): 7♠=0, 8♠=1, 9♠=2, J♠=3, Q♠=4, K♠=5, 10♠=6, A♠=7
    // Hearts (8-15): 7♥=8, 8♥=9, 9♥=10, J♥=11, Q♥=12, K♥=13, 10♥=14, A♥=15
    // Diamonds (16-23): 7♦=16, 8♦=17, 9♦=18, J♦=19, Q♦=20, K♦=21, 10♦=22, A♦=23
    // Clubs (24-31): 7♣=24, 8♣=25, 9♣=26, J♣=27, Q♣=28, K♣=29, 10♣=30, A♣=31

    // P0: 7♠(0) 8♠(1) | 7♥(8) 8♥(9) | A♦(23) K♦(21) 10♦(22) J♦(19) = 8
    // P1: 9♠(2) J♠(3) Q♠(4) K♠(5) | (no hearts) | (no diamonds) | 7♣(24) 8♣(25) 9♣(26) Q♣(28) = 8
    // P2: 10♠(6) A♠(7) | 9♥(10) J♥(11) | 9♦(18) Q♦(20) | K♣(29) J♣(27) = 8
    // P3: (no spades) | Q♥(12) K♥(13) 10♥(14) A♥(15) | 7♦(16) 8♦(17) | 10♣(30) A♣(31) = 8

    let hands: [u32; 4] = [
        (1<<0)|(1<<1)|(1<<8)|(1<<9)|(1<<23)|(1<<21)|(1<<22)|(1<<19),     // P0
        (1<<2)|(1<<3)|(1<<4)|(1<<5)|(1<<24)|(1<<25)|(1<<26)|(1<<28),     // P1
        (1<<6)|(1<<7)|(1<<10)|(1<<11)|(1<<18)|(1<<20)|(1<<29)|(1<<27),   // P2
        (1<<12)|(1<<13)|(1<<14)|(1<<15)|(1<<16)|(1<<17)|(1<<30)|(1<<31), // P3
    ];

    let dealer = 3u8;
    let mut state = GameState::new(dealer, hands);
    let mut tracking = EnvTracking::new();
    tracking.dealer = dealer;

    // Bidding: P0 bids 80♥ (suit 1, action = 0*4 + 1 + 1 = 2), others pass
    for &action in &[2u8, 0, 0, 0] {
        tracking.track_action(&state, action);
        state.step(action);
    }
    assert_eq!(state.phase, Phase::Playing);

    // P0 leads A♦ (card 23)
    let mut tracker = colver_core::game_replay::TrumpCeilingTracker::new();
    let player = state.current_player();
    tracker.record_play(&state, player, 23);
    tracking.track_action(&state, 23);
    state.step(23);

    // P1 plays 7♣ (card 24) — discards, doesn't follow ♦, doesn't cut with ♥
    let player = state.current_player();
    tracker.record_play(&state, player, 24);
    tracking.track_action(&state, 24);
    state.step(24);

    // P2's turn (observer)
    assert_eq!(state.current_player(), 2);
    let observer = 2u8;

    let logits = eval_belief_with_tracker(net, &state, &tracking, observer, Some(&tracker));

    let rel_names = ["Me", "Left", "Prtner", "Right"];

    println!("Diamond predictions (P1 = 'Right', should be ~0% for all ♦):");
    let diamond_cards = [(16, "7♦"), (17, "8♦"), (18, "9♦"), (19, "J♦"), (20, "Q♦"), (21, "K♦"), (22, "10♦")];
    for &(card_idx, name) in &diamond_cards {
        let probs = card_probs(&logits, card_idx, net.num_classes());
        print_card_prediction(name, &probs, &rel_names, Some(3), true);
    }

    println!("\nHeart (trump) predictions (P1 = 'Right', should be ~0% for all ♥):");
    let heart_cards = [(8, "7♥"), (9, "8♥"), (10, "9♥"), (11, "J♥"), (12, "Q♥"), (13, "K♥"), (14, "10♥"), (15, "A♥")];
    for &(card_idx, name) in &heart_cards {
        let probs = card_probs(&logits, card_idx, net.num_classes());
        print_card_prediction(name, &probs, &rel_names, Some(3), true);
    }
    println!();
}

/// Scenario 3: Bidding signal.
/// P1 bids 80♠. Does the model give P1 higher prob for J♠ and 9♠?
/// Test BEFORE any cards are played (start of play phase).
fn scenario_bidding_signal(net: &mut BeliefNet) {
    println!("{}", "=".repeat(70));
    println!("SCENARIO 3: Bidding Signal");
    println!("P1 bids 80♠, others pass. Does the model assign higher prob for P1 having J♠/9♠?");
    println!("Observer: P2 (South). Testing at start of play phase.");
    println!();

    // P0: 7♦ 8♦ 9♦ J♦ Q♦ K♦ 10♦ A♦ (all diamonds)
    // P1: J♠ 9♠ A♠ 10♠ | 7♥ 8♥ 9♥ J♥ (strong spades + some hearts)
    // P2: Q♠ K♠ 7♠ 8♠ | Q♥ K♥ 10♥ A♥ (observer)
    // P3: 7♣ 8♣ 9♣ J♣ Q♣ K♣ 10♣ A♣ (all clubs)
    let hands: [u32; 4] = [
        0x00FF_0000, // P0: all diamonds (bits 16-23)
        (1<<3)|(1<<2)|(1<<7)|(1<<6)|(1<<8)|(1<<9)|(1<<10)|(1<<11),  // P1: J♠9♠A♠10♠ + 7♥8♥9♥J♥
        (1<<4)|(1<<5)|(1<<0)|(1<<1)|(1<<12)|(1<<13)|(1<<14)|(1<<15), // P2: Q♠K♠7♠8♠ + Q♥K♥10♥A♥
        0xFF00_0000, // P3: all clubs (bits 24-31)
    ];

    let dealer = 0u8; // P1 bids first
    let mut state = GameState::new(dealer, hands);
    let mut tracking = EnvTracking::new();
    tracking.dealer = dealer;

    // Bidding: P1 bids 80♠ (action=1), P2/P3/P0 pass
    for &action in &[1u8, 0, 0, 0] {
        tracking.track_action(&state, action);
        state.step(action);
    }
    assert_eq!(state.phase, Phase::Playing);

    // No cards played yet. Observer=P2.
    let observer = 2u8;
    // P1 leads (first player after dealer)
    assert_eq!(state.current_player(), 1);

    // Build obs from P2's perspective even though it's P1's turn
    let logits = eval_belief(net, &state, &tracking, observer);

    // Observer=P2: relative 0=P2(me), 1=P3(left), 2=P0(partner), 3=P1(right)
    let rel_names = ["Me", "Left", "Prtner", "Right"];

    println!("Spade predictions (P1 = 'Right', bid 80♠ → should have higher J♠/9♠):");
    let spade_cards = [(3, "J♠"), (2, "9♠"), (7, "A♠"), (6, "10♠")];
    for &(card_idx, name) in &spade_cards {
        let probs = card_probs(&logits, card_idx, net.num_classes());
        print_card_prediction(name, &probs, &rel_names, Some(3), false);
    }

    // Compare: random suit cards where no signal exists
    println!("\nDiamond predictions (no signal — baseline):");
    let diamond_cards = [(19, "J♦"), (18, "9♦"), (23, "A♦"), (22, "10♦")];
    for &(card_idx, name) in &diamond_cards {
        let probs = card_probs(&logits, card_idx, net.num_classes());
        print_card_prediction(name, &probs, &rel_names, None, false);
    }
    println!();
}

/// Per-trick accuracy evaluation: replay many games, evaluate at every play step,
/// report accuracy grouped by trick number. Answers: "does the model get better as
/// it sees more tricks?"
///
/// Supports V1 (330-dim), V2 (304-dim), and V3 (380-dim) models automatically.
fn run_per_trick_eval(model_path: &str, replays_path: &str, num_games: usize) {
    if replays_path.is_empty() {
        eprintln!("--replays is required for per_trick mode");
        std::process::exit(1);
    }

    println!("=== Per-Trick Accuracy Evaluation ===");
    println!("Model:   {}", model_path);
    println!("Replays: {}", replays_path);
    println!("Games:   {}", num_games);

    let mut net = BeliefNet::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });
    let obs_dim = net.obs_dim();
    let version_str = match obs_dim {
        BELIEF_OBS_DIM => "V1 (330)",
        BELIEF_OBS_DIM_V2 => "V2 (304)",
        BELIEF_OBS_DIM_V3 => "V3 (380)",
        d => { eprintln!("Unknown obs_dim={}", d); std::process::exit(1); }
    };
    println!("Model loaded ({}, hidden={}, {}-class)\n", version_str, net.hidden(), net.num_classes());

    let replays = GameReplay::load_all(replays_path).unwrap_or_else(|e| {
        eprintln!("Failed to load replays: {}", e);
        std::process::exit(1);
    });
    println!("Loaded {} replays", replays.len());

    // Select subset of replays
    let step = replays.len().max(1) / num_games.max(1);
    let subset: Vec<GameReplay> = (0..num_games)
        .map(|i| {
            let r = &replays[(i * step) % replays.len()];
            GameReplay { dealer: r.dealer, hands: r.hands, actions: r.actions.clone() }
        })
        .collect();

    // Extract samples using the correct version's extraction function
    println!("Extracting samples ({})...", version_str);
    let extract_start = Instant::now();
    let samples = match obs_dim {
        BELIEF_OBS_DIM_V3 => colver_core::game_replay::extract_belief_samples_v3(&subset),
        BELIEF_OBS_DIM_V2 => colver_core::game_replay::extract_belief_samples_v2(&subset),
        _ => colver_core::game_replay::extract_belief_samples(&subset),
    };
    println!("Extracted {} samples in {:.1}s\n", samples.len(), extract_start.elapsed().as_secs_f64());

    let start = Instant::now();
    let nc = net.num_classes();

    // Per-trick stats
    let mut correct_by_trick = [0u64; 8];
    let mut total_by_trick = [0u64; 8];
    let mut ce_by_trick = [0.0f64; 8];

    // Fine-grained: trick*4 + pos_in_trick
    let mut correct_by_step = [0u64; 32];
    let mut total_by_step = [0u64; 32];

    for (si, sample) in samples.iter().enumerate() {
        let logits = net.evaluate(&sample.obs);
        let trick_idx = (sample.trick_idx as usize).min(7);
        let step_idx = (trick_idx * 4 + sample.pos_in_trick as usize).min(31);

        for c in 0..32u32 {
            if sample.mask & (1 << c) == 0 { continue; }

            // Extraction always produces 3-class targets: 0=left, 1=partner, 2=right.
            // For 4-class models (0=me, 1=left, 2=partner, 3=right), shift target +1
            // and skip observer slot in argmax.
            let true_rel = if nc == 4 {
                sample.target[c as usize] as usize + 1
            } else {
                sample.target[c as usize] as usize
            };
            let base = c as usize * nc;
            let start_slot = if nc == 4 { 1 } else { 0 }; // skip observer for 4-class

            // Softmax
            let mut max_l = f32::NEG_INFINITY;
            for p in start_slot..nc { if logits[base + p] > max_l { max_l = logits[base + p]; } }
            let mut exp_sum = 0.0f32;
            let mut exps = [0.0f32; 4];
            for p in start_slot..nc {
                exps[p] = (logits[base + p] - max_l).exp();
                exp_sum += exps[p];
            }

            // Argmax over non-observer slots
            let mut pred = start_slot;
            for p in (start_slot + 1)..nc {
                if exps[p] > exps[pred] { pred = p; }
            }

            if pred == true_rel {
                correct_by_trick[trick_idx] += 1;
                correct_by_step[step_idx] += 1;
            }
            total_by_trick[trick_idx] += 1;
            total_by_step[step_idx] += 1;

            // Cross-entropy
            let true_prob = (exps[true_rel] / exp_sum).max(1e-10);
            ce_by_trick[trick_idx] += -(true_prob as f64).ln();
        }

        if (si + 1) % 100_000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let total_correct: u64 = correct_by_trick.iter().sum();
            let total_cards: u64 = total_by_trick.iter().sum();
            let acc = if total_cards > 0 { total_correct as f64 / total_cards as f64 * 100.0 } else { 0.0 };
            println!("  [{}/{}] acc={:.1}% ({:.1}s)", si + 1, samples.len(), acc, elapsed);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total_correct: u64 = correct_by_trick.iter().sum();
    let total_cards: u64 = total_by_trick.iter().sum();
    let overall_acc = total_correct as f64 / total_cards.max(1) as f64 * 100.0;

    println!("\n=== Results ===");
    println!("Games:            {}", num_games);
    println!("Samples:          {}", samples.len());
    println!("Cards predicted:  {}", total_cards);
    println!("Overall accuracy: {:.2}% ({}/{})", overall_acc, total_correct, total_cards);
    println!("Random baseline:  33.33%");
    println!("Eval time:        {:.1}s", elapsed);

    println!("\n--- Accuracy by completed trick ---");
    println!("  {:>7}  {:>8}  {:>10}  {:>10}  {:>7}", "Trick", "Acc%", "Correct", "Total", "CE");
    for t in 0..8 {
        if total_by_trick[t] > 0 {
            let acc = correct_by_trick[t] as f64 / total_by_trick[t] as f64 * 100.0;
            let ce = ce_by_trick[t] / total_by_trick[t] as f64;
            println!("  {:>7}  {:>7.1}%  {:>10}  {:>10}  {:>7.4}", t, acc, correct_by_trick[t], total_by_trick[t], ce);
        }
    }

    println!("\n--- Fine-grained: accuracy by play step (trick.position) ---");
    println!("  {:>10}  {:>8}  {:>10}  {:>10}", "Step", "Acc%", "Correct", "Total");
    for s in 0..32 {
        if total_by_step[s] > 0 {
            let trick = s / 4;
            let pos = s % 4;
            let acc = correct_by_step[s] as f64 / total_by_step[s] as f64 * 100.0;
            println!("  {:>4}.{:<5}  {:>7.1}%  {:>10}  {:>10}", trick, pos, acc, correct_by_step[s], total_by_step[s]);
        }
    }
}

/// Input block ablation: zero each V2 input block and report accuracy drop.
fn run_ablation(model_path: &str, replays_path: &str, num_games: usize) {
    if replays_path.is_empty() {
        eprintln!("--replays is required for ablation mode");
        std::process::exit(1);
    }

    println!("=== Input Block Ablation (V2) ===");
    println!("Model:   {}", model_path);
    println!("Replays: {}", replays_path);
    println!("Games:   {}", num_games);

    let mut net = BeliefNet::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load model: {}", e);
        std::process::exit(1);
    });

    if net.obs_dim() != BELIEF_OBS_DIM_V2 {
        eprintln!("Ablation mode requires a V2 model (obs_dim={}), got obs_dim={}",
                  BELIEF_OBS_DIM_V2, net.obs_dim());
        std::process::exit(1);
    }
    println!("Model loaded (obs_dim={}, hidden={})\n", net.obs_dim(), net.hidden());

    let replays = GameReplay::load_all(replays_path).unwrap_or_else(|e| {
        eprintln!("Failed to load replays: {}", e);
        std::process::exit(1);
    });
    println!("Loaded {} replays", replays.len());

    // Extract V2 samples
    let step = replays.len().max(1) / num_games.max(1);
    let subset: Vec<&GameReplay> = (0..num_games)
        .map(|i| &replays[(i * step) % replays.len()])
        .collect();
    let samples = {
        let owned: Vec<GameReplay> = subset.iter().map(|r| GameReplay {
            dealer: r.dealer,
            hands: r.hands,
            actions: r.actions.clone(),
        }).collect();
        colver_core::game_replay::extract_belief_samples_v2(&owned)
    };
    println!("Extracted {} samples\n", samples.len());

    // Evaluate baseline
    let baseline_acc = eval_samples_accuracy(&mut net, &samples);
    println!("Baseline accuracy: {:.2}%\n", baseline_acc * 100.0);

    // Block definitions for V2
    let blocks: &[(&str, usize, usize)] = &[
        ("Own hand",           0,   32),
        ("Played-by",          32,  64),
        ("Trick index",        64,  96),
        ("Position-in-trick",  96,  128),
        ("Bid history",        128, 200),
        ("Contract",           200, 208),
        ("Hard constraints",   208, 304),
    ];

    println!("{:<22} {:>8} {:>10} {:>10}", "Block", "Acc%", "Delta%", "Rel. Drop%");
    println!("{}", "-".repeat(55));

    for &(name, start, end) in blocks {
        let ablated_acc = eval_samples_ablated(&mut net, &samples, start, end);
        let delta = ablated_acc - baseline_acc;
        let rel_drop = if baseline_acc > 0.0 { delta / baseline_acc * 100.0 } else { 0.0 };
        println!(
            "{:<22} {:>7.2}% {:>+9.2}% {:>+9.1}%",
            format!("[{}:{}] {}", start, end, name),
            ablated_acc * 100.0,
            delta * 100.0,
            rel_drop,
        );
    }
}

/// Compute argmax accuracy over unknown cards for a set of belief samples.
fn eval_samples_accuracy(net: &mut BeliefNet, samples: &[colver_core::game_replay::BeliefSample]) -> f64 {
    let mut correct = 0u64;
    let mut total = 0u64;
    let nc = net.num_classes();

    for sample in samples {
        let logits = net.evaluate(&sample.obs);

        for c in 0..32u32 {
            if sample.mask & (1 << c) == 0 {
                continue;
            }
            let true_rel = sample.target[c as usize] as usize;
            let base = c as usize * nc;

            // Argmax over all nc slots
            let mut pred = 0usize;
            for p in 1..nc {
                if logits[base + p] > logits[base + pred] {
                    pred = p;
                }
            }

            if pred == true_rel {
                correct += 1;
            }
            total += 1;
        }
    }

    if total > 0 { correct as f64 / total as f64 } else { 0.0 }
}

/// Compute accuracy after zeroing obs[start..end] for each sample.
fn eval_samples_ablated(
    net: &mut BeliefNet,
    samples: &[colver_core::game_replay::BeliefSample],
    zero_start: usize,
    zero_end: usize,
) -> f64 {
    let mut correct = 0u64;
    let mut total = 0u64;
    let nc = net.num_classes();

    for sample in samples {
        let mut obs = sample.obs.clone();
        for v in &mut obs[zero_start..zero_end] {
            *v = 0.0;
        }
        let logits = net.evaluate(&obs);

        for c in 0..32u32 {
            if sample.mask & (1 << c) == 0 {
                continue;
            }
            let true_rel = sample.target[c as usize] as usize;
            let base = c as usize * nc;

            let mut pred = 0usize;
            for p in 1..nc {
                if logits[base + p] > logits[base + pred] {
                    pred = p;
                }
            }

            if pred == true_rel {
                correct += 1;
            }
            total += 1;
        }
    }

    if total > 0 { correct as f64 / total as f64 } else { 0.0 }
}

/// Ensemble evaluation: load multiple models, average raw logits, report accuracy.
fn run_ensemble_eval(model_paths: &str, replays_path: &str, num_games: usize) {
    if replays_path.is_empty() {
        eprintln!("--replays is required for ensemble mode");
        std::process::exit(1);
    }
    if model_paths.is_empty() {
        eprintln!("--model is required for ensemble mode (comma-separated paths)");
        std::process::exit(1);
    }

    let paths: Vec<&str> = model_paths.split(',').collect();
    println!("=== Ensemble Evaluation ===");
    println!("Models:  {} models", paths.len());
    for (i, p) in paths.iter().enumerate() {
        println!("  [{}] {}", i, p);
    }
    println!("Replays: {}", replays_path);
    println!("Games:   {}", num_games);

    let mut nets: Vec<BeliefNet> = paths.iter().map(|p| {
        BeliefNet::load(p).unwrap_or_else(|e| {
            eprintln!("Failed to load model {}: {}", p, e);
            std::process::exit(1);
        })
    }).collect();

    // Verify all models have the same obs_dim
    let obs_dim = nets[0].obs_dim();
    for (i, net) in nets.iter().enumerate() {
        if net.obs_dim() != obs_dim {
            eprintln!("Model {} has obs_dim={}, expected {}", paths[i], net.obs_dim(), obs_dim);
            std::process::exit(1);
        }
    }
    println!("All models: obs_dim={}\n", obs_dim);

    let replays = GameReplay::load_all(replays_path).unwrap_or_else(|e| {
        eprintln!("Failed to load replays: {}", e);
        std::process::exit(1);
    });

    // Extract samples matching model version
    let step = replays.len().max(1) / num_games.max(1);
    let subset: Vec<GameReplay> = (0..num_games)
        .map(|i| {
            let r = &replays[(i * step) % replays.len()];
            GameReplay { dealer: r.dealer, hands: r.hands, actions: r.actions.clone() }
        })
        .collect();
    let samples = if obs_dim == BELIEF_OBS_DIM_V2 {
        colver_core::game_replay::extract_belief_samples_v2(&subset)
    } else {
        colver_core::game_replay::extract_belief_samples(&subset)
    };
    println!("Extracted {} samples\n", samples.len());

    // Evaluate individual models and ensemble
    let n_models = nets.len();

    // Individual accuracies
    for (i, net) in nets.iter_mut().enumerate() {
        let acc = eval_samples_accuracy(net, &samples);
        println!("Model [{}]: {:.2}%", i, acc * 100.0);
    }

    // Ensemble: average logits
    let mut correct = 0u64;
    let mut total = 0u64;
    let nc = nets[0].num_classes();
    let n_logits = 32 * nc;

    for sample in &samples {
        let mut avg_logits = [0.0f32; 128];
        for net in nets.iter_mut() {
            let logits = net.evaluate(&sample.obs);
            for j in 0..n_logits {
                avg_logits[j] += logits[j];
            }
        }
        for j in 0..n_logits {
            avg_logits[j] /= n_models as f32;
        }

        for c in 0..32u32 {
            if sample.mask & (1 << c) == 0 {
                continue;
            }
            let true_rel = sample.target[c as usize] as usize;
            let base = c as usize * nc;

            let mut pred = 0usize;
            for p in 1..nc {
                if avg_logits[base + p] > avg_logits[base + pred] {
                    pred = p;
                }
            }

            if pred == true_rel {
                correct += 1;
            }
            total += 1;
        }
    }

    let ensemble_acc = if total > 0 { correct as f64 / total as f64 } else { 0.0 };
    println!("\nEnsemble ({} models): {:.2}%", n_models, ensemble_acc * 100.0);
}

/// Evaluate belief accuracy at the start of play phase (after bidding, before any
/// cards played). Compares NN beliefs vs CardBeliefs heuristic vs uniform random.
///
/// Usage:
///   cargo run -p colver-core --bin belief_eval --release -- \
///     --model models/belief_net.bin --bid-model models/bid_nn_final.bin \
///     --mode bidding --games 5000 --seed 42
fn run_bidding_eval(model_path: &str, bid_model_path: &str, num_deals: usize, seed: u64) {
    println!("=== Bidding-Phase Belief Evaluation ===");
    println!("Belief model: {}", model_path);
    println!("Bid model:    {}", if bid_model_path.is_empty() { "(improved_v2)" } else { bid_model_path });
    println!("Deals:        {}", num_deals);
    println!("Seed:         {}", seed);

    let mut belief_net = BeliefNet::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load belief model: {}", e);
        std::process::exit(1);
    });
    let obs_dim = belief_net.obs_dim();
    let nc = belief_net.num_classes();
    println!("Belief model loaded (obs_dim={}, hidden={}, {}-class)", obs_dim, belief_net.hidden(), nc);

    let mut bid_net = if !bid_model_path.is_empty() {
        match BidNet::load(bid_model_path) {
            Ok(net) => { println!("Bid model loaded"); Some(net) }
            Err(e) => { println!("Bid model not found ({}), using improved_v2", e); None }
        }
    } else {
        None
    };

    let mut rng = StdRng::seed_from_u64(seed);

    // Accumulators: [nn, heuristic_soft, heuristic_hard, random]
    let mut correct = [0u64; 4];
    let mut total = [0u64; 4];
    let mut ce_sum = [0.0f64; 4]; // Cross-entropy sum
    let mut ce_count = [0u64; 4];

    // Per-bidder-position accuracy (0 = opener, 1-3 = subsequent)
    let mut nn_correct_by_pos = [0u64; 4];
    let mut nn_total_by_pos = [0u64; 4];

    // Track how many deals had a contract vs void
    let mut void_deals = 0u32;
    let mut contract_deals = 0u32;

    // NN accuracy when bidder bid vs when bidder passed (for the bid-suit cards)
    let mut nn_correct_bidder_trump = 0u64;
    let mut nn_total_bidder_trump = 0u64;
    let mut nn_correct_passer_trump = 0u64;
    let mut nn_total_passer_trump = 0u64;

    let start = Instant::now();

    for deal_idx in 0..num_deals {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, &mut rng);
        let true_hands = state.hands;

        let mut tracking = EnvTracking::new();
        tracking.dealer = dealer;

        // Run bidding
        while state.phase == Phase::Bidding {
            let action = bid_action(&state, &mut bid_net);
            tracking.track_action(&state, action);
            state.step(action);
        }

        if state.is_terminal() {
            // Void deal (4 passes)
            void_deals += 1;
            continue;
        }
        contract_deals += 1;

        let trump_suit = state.contract.trump;
        let taker_team = state.contract.team;

        // Evaluate from each player's perspective
        for observer in 0..4u8 {
            let observer_hand = state.hands[observer as usize];
            let unknown_mask = !observer_hand; // All 24 cards not in hand

            // --- 1. NN beliefs ---
            let nn_weights = {
                let logits = eval_belief(&mut belief_net, &state, &tracking, observer);
                belief_net::belief_to_weights(&logits, nc, &state, observer)
            };

            // --- 2. CardBeliefs with soft inference ---
            let heur_soft_weights = {
                let mut beliefs = CardBeliefs::new(&GameState::new(dealer, true_hands), observer);
                // Replay bidding actions
                let mut replay_state = GameState::new(dealer, true_hands);
                for &(player, action) in &tracking.bid_history {
                    beliefs.record_action(&replay_state, player, action);
                    replay_state.step(action);
                }
                beliefs.normalized_weights()
            };

            // --- 3. CardBeliefs hard only (no soft inference) ---
            let heur_hard_weights = {
                let mut beliefs = CardBeliefs::new(&GameState::new(dealer, true_hands), observer);
                beliefs.use_soft_inference = false;
                let mut replay_state = GameState::new(dealer, true_hands);
                for &(player, action) in &tracking.bid_history {
                    beliefs.record_action(&replay_state, player, action);
                    replay_state.step(action);
                }
                beliefs.normalized_weights()
            };

            // Evaluate each unknown card
            for c in 0..32u8 {
                if unknown_mask & (1u32 << c) == 0 {
                    continue;
                }

                // Ground truth: which player holds this card?
                let mut true_abs = 0u8;
                for p in 0..4u8 {
                    if true_hands[p as usize] & (1u32 << c) != 0 {
                        true_abs = p;
                        break;
                    }
                }

                // Player-relative mapping: 0=me, 1=left, 2=partner, 3=right
                // For argmax: exclude observer (me)
                let true_rel = ((true_abs + 4 - observer) % 4) as usize;
                debug_assert!(true_rel >= 1 && true_rel <= 3);

                // NN: argmax over non-observer players
                let nn_pred = {
                    let mut best_p = 1usize;
                    let seats = [
                        ((observer + 1) % 4) as usize,
                        ((observer + 2) % 4) as usize,
                        ((observer + 3) % 4) as usize,
                    ];
                    let mut best_w = nn_weights[seats[0]][c as usize];
                    for (i, &seat) in seats.iter().enumerate().skip(1) {
                        if nn_weights[seat][c as usize] > best_w {
                            best_w = nn_weights[seat][c as usize];
                            best_p = i + 1; // relative: 1=left, 2=partner, 3=right
                        }
                    }
                    best_p
                };

                // Heuristic soft: argmax over non-observer
                let heur_soft_pred = {
                    let seats = [
                        ((observer + 1) % 4) as usize,
                        ((observer + 2) % 4) as usize,
                        ((observer + 3) % 4) as usize,
                    ];
                    let mut best_p = 1usize;
                    let mut best_w = heur_soft_weights[seats[0]][c as usize];
                    for (i, &seat) in seats.iter().enumerate().skip(1) {
                        if heur_soft_weights[seat][c as usize] > best_w {
                            best_w = heur_soft_weights[seat][c as usize];
                            best_p = i + 1;
                        }
                    }
                    best_p
                };

                // Heuristic hard: argmax
                let heur_hard_pred = {
                    let seats = [
                        ((observer + 1) % 4) as usize,
                        ((observer + 2) % 4) as usize,
                        ((observer + 3) % 4) as usize,
                    ];
                    let mut best_p = 1usize;
                    let mut best_w = heur_hard_weights[seats[0]][c as usize];
                    for (i, &seat) in seats.iter().enumerate().skip(1) {
                        if heur_hard_weights[seat][c as usize] > best_w {
                            best_w = heur_hard_weights[seat][c as usize];
                            best_p = i + 1;
                        }
                    }
                    best_p
                };

                // Random: always 33.3%
                let random_pred = (rng.gen_range(0..3u8) + 1) as usize;

                // Record accuracy
                let preds = [nn_pred, heur_soft_pred, heur_hard_pred, random_pred];
                for (i, &pred) in preds.iter().enumerate() {
                    if pred == true_rel {
                        correct[i] += 1;
                    }
                    total[i] += 1;
                }

                // Cross-entropy for NN
                {
                    let seats = [
                        ((observer + 1) % 4) as usize,
                        ((observer + 2) % 4) as usize,
                        ((observer + 3) % 4) as usize,
                    ];
                    let true_seat = seats[true_rel - 1];
                    let p_nn = nn_weights[true_seat][c as usize].max(1e-10);
                    ce_sum[0] += -(p_nn as f64).ln();
                    ce_count[0] += 1;

                    let p_heur = heur_soft_weights[true_seat][c as usize].max(1e-10);
                    ce_sum[1] += -(p_heur as f64).ln();
                    ce_count[1] += 1;

                    let p_hard = heur_hard_weights[true_seat][c as usize].max(1e-10);
                    ce_sum[2] += -(p_hard as f64).ln();
                    ce_count[2] += 1;

                    ce_sum[3] += -(1.0f64 / 3.0).ln(); // random
                    ce_count[3] += 1;
                }

                // NN per-position accuracy
                let obs_pos = ((observer as usize) + 4 - dealer as usize) % 4;
                if nn_pred == true_rel {
                    nn_correct_by_pos[obs_pos] += 1;
                }
                nn_total_by_pos[obs_pos] += 1;

                // NN accuracy for trump suit cards: bidder vs non-bidder
                let card_suit = c / 8;
                if card_suit == trump_suit {
                    let card_holder_team = GameState::player_team(true_abs);
                    if card_holder_team == taker_team {
                        // Card belongs to the taker's team
                        if nn_pred == true_rel { nn_correct_bidder_trump += 1; }
                        nn_total_bidder_trump += 1;
                    } else {
                        if nn_pred == true_rel { nn_correct_passer_trump += 1; }
                        nn_total_passer_trump += 1;
                    }
                }
            }
        }

        if (deal_idx + 1) % 500 == 0 || deal_idx + 1 == num_deals {
            let elapsed = start.elapsed().as_secs_f64();
            let nn_acc = correct[0] as f64 / total[0].max(1) as f64 * 100.0;
            let hs_acc = correct[1] as f64 / total[1].max(1) as f64 * 100.0;
            println!(
                "  [{}/{}] NN={:.1}% Heur={:.1}% ({:.1}s)",
                deal_idx + 1, num_deals, nn_acc, hs_acc, elapsed,
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();

    println!("\n=== Results ({} deals, {} void, {} with contract) ===",
        num_deals, void_deals, contract_deals);

    let labels = ["NN Belief", "Heur Soft", "Heur Hard", "Random"];
    println!("\n{:<12} {:>8} {:>12} {:>10}", "Method", "Acc%", "CE", "Cards");
    println!("{}", "-".repeat(48));
    for i in 0..4 {
        let acc = correct[i] as f64 / total[i].max(1) as f64 * 100.0;
        let ce = if ce_count[i] > 0 { ce_sum[i] / ce_count[i] as f64 } else { 0.0 };
        println!(
            "{:<12} {:>7.2}% {:>12.4} {:>10}",
            labels[i], acc, ce, total[i],
        );
    }

    // NN accuracy by observer position relative to dealer
    println!("\n--- NN accuracy by observer position (relative to dealer) ---");
    let pos_labels = ["Opener (dealer+1)", "2nd seat", "3rd seat", "4th seat (dealer)"];
    for i in 0..4 {
        if nn_total_by_pos[i] > 0 {
            let acc = nn_correct_by_pos[i] as f64 / nn_total_by_pos[i] as f64 * 100.0;
            println!("  {:<20}: {:.1}% ({}/{})", pos_labels[i], acc, nn_correct_by_pos[i], nn_total_by_pos[i]);
        }
    }

    // NN accuracy on trump suit: taker team vs defender team
    println!("\n--- NN accuracy on trump suit cards ---");
    if nn_total_bidder_trump > 0 {
        let acc = nn_correct_bidder_trump as f64 / nn_total_bidder_trump as f64 * 100.0;
        println!("  Taker team trump:   {:.1}% ({}/{})", acc, nn_correct_bidder_trump, nn_total_bidder_trump);
    }
    if nn_total_passer_trump > 0 {
        let acc = nn_correct_passer_trump as f64 / nn_total_passer_trump as f64 * 100.0;
        println!("  Defender team trump: {:.1}% ({}/{})", acc, nn_correct_passer_trump, nn_total_passer_trump);
    }

    println!("\nElapsed: {:.1}s ({:.1}ms/deal)", elapsed, elapsed * 1000.0 / num_deals as f64);
}

fn format_bid_action(action: u8) -> String {
    use colver_core::bidding;
    let suit_symbols = ["♠", "♥", "♦", "♣"];
    match action {
        0 => "Pass".to_string(),
        41 => "Coinche".to_string(),
        42 => "Surcoinche".to_string(),
        1..=40 => {
            let (val_enc, suit_idx) = bidding::decode_bid(action);
            if val_enc == 25 {
                format!("Capot {}", suit_symbols[suit_idx as usize])
            } else {
                format!("{}{}", val_enc * 10, suit_symbols[suit_idx as usize])
            }
        }
        _ => format!("?{}", action),
    }
}
