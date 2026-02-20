/// Evaluation binary for NN bidding model.
///
/// Match play: NN bid + DMC/DD play vs improved_v2 + DMC/DD play.
/// Reports win rate, contract take/success rates, margins, coinche stats.
///
/// Usage:
///   cargo run -p colver-core --bin bid_nn_eval --release -- models/bid_nn_final.bin --matches 500

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_eval;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs;
use colver_core::rollout;
use colver_core::solver;
use colver_core::scoring::compute_deal_score;
use colver_core::state::{GameState, Phase};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    if args.len() < 2 {
        eprintln!("Usage: bid_nn_eval <bid_model.bin> [--matches N] [--hidden H] [--dmc-model PATH] [--seed S]");
        std::process::exit(1);
    }

    let bid_model_path = &args[1];
    let mut num_matches = 500;
    let mut hidden = 256;
    let mut dmc_model_path: Option<String> = None;
    let mut seed = 42u64;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--matches" => { num_matches = args[i+1].parse().unwrap(); i += 2; }
            "--hidden" => { hidden = args[i+1].parse().unwrap(); i += 2; }
            "--dmc-model" => { dmc_model_path = Some(args[i+1].clone()); i += 2; }
            "--seed" => { seed = args[i+1].parse().unwrap(); i += 2; }
            _ => { eprintln!("Unknown arg: {}", args[i]); i += 1; }
        }
    }

    // Load bid model
    let mut bid_net = BidNet::load_with_hidden(bid_model_path, hidden)
        .unwrap_or_else(|e| panic!("Failed to load bid model {}: {}", bid_model_path, e));
    println!("Bid model: {} (obs_dim={}, hidden={}, dueling={})",
        bid_model_path, bid_net.obs_dim(), bid_net.hidden(), bid_net.is_dueling());

    // Load DMC model (optional, for card play)
    let mut dmc_net: Option<DmcNet> = dmc_model_path.as_ref().map(|path| {
        let net = DmcNet::load(path)
            .unwrap_or_else(|e| panic!("Failed to load DMC model {}: {}", path, e));
        println!("DMC model: {} (obs_dim={}, dueling={})", path, net.obs_dim(), net.is_dueling());
        net
    });

    let use_dd = dmc_net.is_none();
    if use_dd {
        println!("Card play: DD oracle");
    } else {
        println!("Card play: DMC Q-network");
    }

    println!("\n=== NN Bid vs improved_v2 ({} matches) ===\n", num_matches);

    let mut rng = StdRng::seed_from_u64(seed);
    let start = Instant::now();

    // Stats
    let mut nn_wins = 0usize;
    let mut nn_losses = 0usize;
    let mut draws = 0usize;
    let mut total_margin = 0i64;
    let mut nn_contracts_taken = 0usize;
    let mut nn_contracts_made = 0usize;
    let mut baseline_contracts_taken = 0usize;
    let mut baseline_contracts_made = 0usize;
    let mut nn_coinches = 0usize;
    let mut baseline_coinches = 0usize;
    let mut void_deals = 0usize;

    for match_idx in 0..num_matches {
        let nn_team: u8 = (match_idx % 2) as u8;
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut bid_history: Vec<(u8, u8)> = Vec::new();

        // DD solve all 4 suits for this deal
        let mut tt_buf = solver::new_tt_buffer();
        let mut dd_pts = [0u8; 4];
        for suit in 0..4u8 {
            let result = solver::solve_for_trump_reuse_tt(
                state.hands, state.dealer, suit, &mut tt_buf,
            );
            dd_pts[suit as usize] = result[0];
        }

        // --- Bidding phase ---
        while state.phase == Phase::Bidding {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if team == nn_team {
                let obs = bid_obs::make_bid_observation(&state, &bid_history);
                let legal = state.legal_actions();
                let (best, _) = bid_net.best_action(&obs, legal);
                best
            } else {
                bid_eval::improved_v2_bid(&state)
            };

            // Track coinches
            if action == 41 {
                if team == nn_team { nn_coinches += 1; } else { baseline_coinches += 1; }
            }

            bid_history.push((player, action));
            state.step(action);
        }

        // Check if void deal
        if state.contract.value == 0 {
            void_deals += 1;
            continue;
        }

        // Track contract
        let contract_team = state.contract.team;
        if contract_team == nn_team {
            nn_contracts_taken += 1;
        } else {
            baseline_contracts_taken += 1;
        }

        // --- Card play phase ---
        let (ns_score, ew_score) = if use_dd {
            // DD oracle scoring
            compute_dd_deal_scores(&state, &dd_pts)
        } else {
            // DMC card play
            let dmc = dmc_net.as_mut().unwrap();
            play_with_dmc(&mut state, dmc, &mut rng)
        };

        // Track contract success
        let contract_score = if contract_team == 0 { ns_score } else { ew_score };
        if contract_score > 0 {
            if contract_team == nn_team {
                nn_contracts_made += 1;
            } else {
                baseline_contracts_made += 1;
            }
        }

        // Score
        let nn_score = if nn_team == 0 { ns_score } else { ew_score };
        let opp_score = if nn_team == 0 { ew_score } else { ns_score };

        if nn_score > opp_score {
            nn_wins += 1;
        } else if opp_score > nn_score {
            nn_losses += 1;
        } else {
            draws += 1;
        }
        total_margin += (nn_score - opp_score) as i64;

        // Progress
        if (match_idx + 1) % 100 == 0 {
            let total = nn_wins + nn_losses + draws;
            let wr = nn_wins as f64 / total as f64 * 100.0;
            let avg_margin = total_margin as f64 / total as f64;
            println!("  [{}/{}] WR: {:.1}%  Margin: {:+.0}  Voids: {}",
                match_idx + 1, num_matches, wr, avg_margin, void_deals);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let total = nn_wins + nn_losses + draws;
    let wr = if total > 0 { nn_wins as f64 / total as f64 * 100.0 } else { 0.0 };
    let avg_margin = if total > 0 { total_margin as f64 / total as f64 } else { 0.0 };

    println!("\n=== Results ===");
    println!("Win rate:  {:.1}% ({} W / {} L / {} D)", wr, nn_wins, nn_losses, draws);
    println!("Margin:    {:+.1} avg", avg_margin);
    println!("Void:      {} deals", void_deals);
    println!();
    println!("NN contracts:       {}/{} taken, {}/{} made ({:.0}%)",
        nn_contracts_taken, total, nn_contracts_made, nn_contracts_taken,
        if nn_contracts_taken > 0 { nn_contracts_made as f64 / nn_contracts_taken as f64 * 100.0 } else { 0.0 });
    println!("Baseline contracts: {}/{} taken, {}/{} made ({:.0}%)",
        baseline_contracts_taken, total, baseline_contracts_made, baseline_contracts_taken,
        if baseline_contracts_taken > 0 { baseline_contracts_made as f64 / baseline_contracts_taken as f64 * 100.0 } else { 0.0 });
    println!("NN coinches:        {}", nn_coinches);
    println!("Baseline coinches:  {}", baseline_coinches);
    println!("\nElapsed: {:.1}s ({:.0} deals/s)", elapsed, num_matches as f64 / elapsed);
}

/// Compute deal scores from DD results.
fn compute_dd_deal_scores(state: &GameState, dd_pts: &[u8; 4]) -> (i16, i16) {
    let trump = state.contract.trump;
    let ns_dd_pts = dd_pts[trump as usize];
    let ew_dd_pts = if ns_dd_pts == 252 || ns_dd_pts == 0 {
        252 - ns_dd_pts
    } else {
        162 - ns_dd_pts
    };

    let taker = state.contract.team as usize;
    let defense = 1 - taker;

    let taker_pts = if taker == 0 { ns_dd_pts } else { ew_dd_pts };
    let defense_pts = if defense == 0 { ns_dd_pts } else { ew_dd_pts };

    let (taker_tricks, defense_tricks) = if defense_pts == 0 {
        (8u8, 0u8)
    } else if taker_pts == 0 {
        (0u8, 8u8)
    } else {
        let total_pts = taker_pts as u16 + defense_pts as u16;
        let taker_frac = taker_pts as f32 / total_pts as f32;
        let t = (taker_frac * 8.0).round().max(1.0).min(7.0) as u8;
        (t, 8 - t)
    };

    let mut terminal = GameState::new(0, [0; 4]);
    terminal.phase = Phase::Done;
    terminal.contract = state.contract;
    terminal.points[taker] = taker_pts;
    terminal.points[defense] = defense_pts;
    terminal.tricks_won[taker] = taker_tricks;
    terminal.tricks_won[defense] = defense_tricks;
    terminal.belote = [0; 2];

    let score = compute_deal_score(&terminal);
    (score.scores[0], score.scores[1])
}

/// Play cards using DMC Q-network, return (ns_score, ew_score).
fn play_with_dmc(
    state: &mut GameState,
    dmc_net: &mut DmcNet,
    rng: &mut StdRng,
) -> (i16, i16) {
    let mut tracking = dmc_obs::EnvTracking::new();
    tracking.dealer = state.dealer;

    while !state.is_terminal() {
        let action = if state.phase == Phase::Playing {
            let obs = dmc_obs::make_observation(state, &tracking);
            let obs_slice = &obs[..dmc_net.obs_dim()];
            let legal_mask = state.legal_actions() as u32;
            let (best, _) = dmc_net.best_action(obs_slice, legal_mask);
            best
        } else {
            // Should not happen (bidding already done), but handle gracefully
            let mask = state.legal_actions();
            let count = mask.count_ones();
            let idx = rng.gen_range(0..count);
            rollout::select_nth_bit(mask, idx)
        };

        tracking.track_action(state, action);
        state.step(action);
    }

    let score = compute_deal_score(state);
    (score.scores[0], score.scores[1])
}
