use colver_core::bid_eval::BidFunction;
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::GameState;
use rand::Rng;
use std::time::Instant;

struct MatchResult {
    ns_wins: u32,
    ew_wins: u32,
    draws: u32,
    ns_total_score: i32,
    ew_total_score: i32,
    void_deals: u32,
    total_dets: u64,
    elapsed: std::time::Duration,
}

/// Run Smart IS-MCTS (NS) vs Naive IS-MCTS (EW) with configurable bid functions.
fn run_experiment(
    label: &str,
    n_games: u32,
    ns_smart_bid: bool,
    ew_smart_bid: bool,
    use_beliefs: bool, // if false, NS uses Naive too
    time_ms: u32,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut result = MatchResult {
        ns_wins: 0,
        ew_wins: 0,
        draws: 0,
        ns_total_score: 0,
        ew_total_score: 0,
        void_deals: 0,
        total_dets: 0,
        elapsed: std::time::Duration::ZERO,
    };

    let start = Instant::now();

    let ew_bf = if ew_smart_bid { BidFunction::Smart } else { BidFunction::Heuristic };
    let ns_bf = if ns_smart_bid { BidFunction::Smart } else { BidFunction::Heuristic };

    // EW config (always naive)
    let ew_config = NaiveIsMctsConfig {
        iterations_per_det: 50,
        time_limit_ms: Some(time_ms),
        bid_function: ew_bf,
        ..Default::default()
    };

    if use_beliefs {
        // NS = Smart IS-MCTS
        let ns_config = SmartIsMctsConfig {
            iterations_per_det: 50,
            time_limit_ms: Some(time_ms),
            bid_function: ns_bf,
            ..Default::default()
        };

        let mut search_p0 = SmartIsMctsSearch::new();
        let mut search_p2 = SmartIsMctsSearch::new();
        let mut ew_search = NaiveIsMctsSearch::new();

        for game in 0..n_games {
            let dealer = (game % 4) as u8;
            let mut state = GameState::deal_random(dealer, rng);

            search_p0.init_deal(&state, 0, ns_config.use_soft_inference);
            search_p2.init_deal(&state, 2, ns_config.use_soft_inference);

            while !state.is_terminal() {
                let player = state.current_player();
                let state_before = state;

                let action = match player {
                    0 => search_p0.search(&state, &ns_config, rng),
                    2 => search_p2.search(&state, &ns_config, rng),
                    _ => ew_search.search(&state, &ew_config, rng),
                };

                search_p0.record_action(&state_before, player, action);
                search_p2.record_action(&state_before, player, action);
                state.step(action);
            }

            let score = state.deal_score();
            if score.scores[0] == 0 && score.scores[1] == 0 {
                result.void_deals += 1;
            } else {
                result.ns_total_score += score.scores[0] as i32;
                result.ew_total_score += score.scores[1] as i32;
                if score.scores[0] > score.scores[1] {
                    result.ns_wins += 1;
                } else if score.scores[1] > score.scores[0] {
                    result.ew_wins += 1;
                } else {
                    result.draws += 1;
                }
            }

            if (game + 1) % 10 == 0 {
                let elapsed = start.elapsed();
                let ms_per_game = elapsed.as_millis() as f64 / (game + 1) as f64;
                eprint!(
                    "\r  [{}] {}/{} games ({:.0}ms/game)   ",
                    label,
                    game + 1,
                    n_games,
                    ms_per_game
                );
            }
        }
    } else {
        // NS = Naive IS-MCTS (no beliefs)
        let ns_config = NaiveIsMctsConfig {
            iterations_per_det: 50,
            time_limit_ms: Some(time_ms),
            bid_function: ns_bf,
            ..Default::default()
        };

        let mut ns_search = NaiveIsMctsSearch::new();
        let mut ew_search = NaiveIsMctsSearch::new();

        for game in 0..n_games {
            let dealer = (game % 4) as u8;
            let mut state = GameState::deal_random(dealer, rng);

            while !state.is_terminal() {
                let player = state.current_player();

                let action = match player {
                    0 | 2 => ns_search.search(&state, &ns_config, rng),
                    _ => ew_search.search(&state, &ew_config, rng),
                };

                state.step(action);
            }

            let score = state.deal_score();
            if score.scores[0] == 0 && score.scores[1] == 0 {
                result.void_deals += 1;
            } else {
                result.ns_total_score += score.scores[0] as i32;
                result.ew_total_score += score.scores[1] as i32;
                if score.scores[0] > score.scores[1] {
                    result.ns_wins += 1;
                } else if score.scores[1] > score.scores[0] {
                    result.ew_wins += 1;
                } else {
                    result.draws += 1;
                }
            }

            if (game + 1) % 10 == 0 {
                let elapsed = start.elapsed();
                let ms_per_game = elapsed.as_millis() as f64 / (game + 1) as f64;
                eprint!(
                    "\r  [{}] {}/{} games ({:.0}ms/game)   ",
                    label,
                    game + 1,
                    n_games,
                    ms_per_game
                );
            }
        }
    }

    eprintln!();
    result.elapsed = start.elapsed();

    // Estimate total determinizations from wall time
    // Each det ≈ 50 iters × ~0.001ms/iter ≈ 0.05ms, across 4 players
    result.total_dets = (result.elapsed.as_millis() as f64 / 0.05) as u64;

    result
}

fn print_result(num: u32, label: &str, ns_desc: &str, ew_desc: &str, n_games: u32, r: &MatchResult) {
    let played = n_games - r.void_deals;
    println!("  Experiment {}: {}", num, label);
    println!("    NS: {}", ns_desc);
    println!("    EW: {}", ew_desc);
    println!(
        "    NS wins: {:3} ({:.1}%)  |  EW wins: {:3} ({:.1}%)  |  Draws: {}  |  Void: {}",
        r.ns_wins,
        if played > 0 { r.ns_wins as f64 / played as f64 * 100.0 } else { 0.0 },
        r.ew_wins,
        if played > 0 { r.ew_wins as f64 / played as f64 * 100.0 } else { 0.0 },
        r.draws,
        r.void_deals,
    );
    println!(
        "    Avg score: NS {:.0}  EW {:.0}  (delta: {:+.0})",
        if played > 0 { r.ns_total_score as f64 / played as f64 } else { 0.0 },
        if played > 0 { r.ew_total_score as f64 / played as f64 } else { 0.0 },
        if played > 0 {
            (r.ns_total_score - r.ew_total_score) as f64 / played as f64
        } else {
            0.0
        },
    );
    println!(
        "    Time: {:.1?} ({:.0}ms/game)",
        r.elapsed,
        r.elapsed.as_millis() as f64 / n_games as f64,
    );
    println!();
}

/// CSV line: exp_num,ns_wins,ew_wins,draws,void_deals,ns_total,ew_total,elapsed_ms
fn print_csv(num: u32, r: &MatchResult) {
    println!(
        "{},{},{},{},{},{},{},{}",
        num,
        r.ns_wins,
        r.ew_wins,
        r.draws,
        r.void_deals,
        r.ns_total_score,
        r.ew_total_score,
        r.elapsed.as_millis(),
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    // Usage:
    //   bidding_experiment [n_games] [time_ms] [exp_num]   — run single experiment
    //   bidding_experiment [n_games] [time_ms]             — run all 4 sequentially
    //   bidding_experiment --summary f1 f2 f3 f4           — aggregate 4 CSV result files
    if args.len() >= 2 && args[1] == "--summary" {
        run_summary(&args[2..]);
        return;
    }

    let n_games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let time_ms: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100);
    let exp_num: Option<u32> = args.get(3).and_then(|s| s.parse().ok());

    let mut rng = rand::thread_rng();

    if let Some(exp) = exp_num {
        // Single experiment mode — output CSV for aggregation
        eprintln!(
            "Running experiment {} ({} games, {}ms/move)",
            exp, n_games, time_ms
        );
        let r = run_single(exp, n_games, time_ms, &mut rng);
        print_csv(exp, &r);
    } else {
        // All experiments sequentially
        println!("=== Bidding Strategy Experiment ===");
        println!("Games per experiment: {}", n_games);
        println!("Time per move: {}ms (scaled by cards remaining)", time_ms);
        println!();

        let mut results = Vec::new();
        for exp in 1..=4 {
            let r = run_single(exp, n_games, time_ms, &mut rng);
            results.push(r);
        }

        print_all_results(n_games, &results);
    }
}

fn run_single(exp: u32, n_games: u32, time_ms: u32, rng: &mut impl Rng) -> MatchResult {
    match exp {
        1 => run_experiment("Exp1", n_games, true, false, true, time_ms, rng),
        2 => run_experiment("Exp2", n_games, false, false, true, time_ms, rng),
        3 => run_experiment("Exp3", n_games, true, true, true, time_ms, rng),
        4 => run_experiment("Exp4", n_games, true, false, false, time_ms, rng),
        _ => panic!("Unknown experiment number: {}", exp),
    }
}

fn print_all_results(n_games: u32, results: &[MatchResult]) {
    println!("============================================================");
    println!("                      RESULTS SUMMARY");
    println!("============================================================");
    println!();

    print_result(1, "Default matchup", "Smart IS-MCTS + smart_bid", "Naive IS-MCTS + heuristic_bid", n_games, &results[0]);
    print_result(2, "Beliefs only (no smart bid)", "Smart IS-MCTS + heuristic_bid", "Naive IS-MCTS + heuristic_bid", n_games, &results[1]);
    print_result(3, "Both smart bid (isolate beliefs)", "Smart IS-MCTS + smart_bid", "Naive IS-MCTS + smart_bid", n_games, &results[2]);
    print_result(4, "Pure bidding effect (no beliefs)", "Naive IS-MCTS + smart_bid", "Naive IS-MCTS + heuristic_bid", n_games, &results[3]);

    println!("--- Analysis ---");
    let delta = |r: &MatchResult| -> f64 {
        let played = (r.ns_wins + r.ew_wins + r.draws) as f64;
        if played > 0.0 {
            (r.ns_total_score - r.ew_total_score) as f64 / played
        } else {
            0.0
        }
    };

    let d1 = delta(&results[0]);
    let d2 = delta(&results[1]);
    let d3 = delta(&results[2]);
    let d4 = delta(&results[3]);

    println!("  smart_bid value for Smart agent: Exp1-Exp2 delta diff = {:+.0}", d1 - d2);
    println!("  Pure belief value:               Exp2 NS delta = {:+.0}", d2);
    println!("  Giving Naive smart_bid:          Exp1 vs Exp3 delta diff = {:+.0}", d1 - d3);
    println!("  Pure bidding strategy value:     Exp4 NS delta = {:+.0}", d4);

    let total_time: std::time::Duration = results.iter().map(|r| r.elapsed).sum();
    println!();
    println!("Total time: {:.1?}", total_time);
}

fn run_summary(files: &[String]) {
    if files.len() != 4 {
        eprintln!("Usage: bidding_experiment --summary <exp1.csv> <exp2.csv> <exp3.csv> <exp4.csv>");
        std::process::exit(1);
    }

    let mut results = Vec::new();
    let mut n_games = 0u32;

    for (i, path) in files.iter().enumerate() {
        let content = std::fs::read_to_string(path).unwrap_or_else(|e| {
            eprintln!("Failed to read {}: {}", path, e);
            std::process::exit(1);
        });
        let line = content.trim();
        let parts: Vec<&str> = line.split(',').collect();
        if parts.len() != 8 {
            eprintln!("Bad CSV in {}: expected 8 fields, got {}", path, parts.len());
            std::process::exit(1);
        }

        let r = MatchResult {
            ns_wins: parts[1].parse().unwrap(),
            ew_wins: parts[2].parse().unwrap(),
            draws: parts[3].parse().unwrap(),
            void_deals: parts[4].parse().unwrap(),
            ns_total_score: parts[5].parse().unwrap(),
            ew_total_score: parts[6].parse().unwrap(),
            total_dets: 0,
            elapsed: std::time::Duration::from_millis(parts[7].parse().unwrap()),
        };

        if i == 0 {
            n_games = r.ns_wins + r.ew_wins + r.draws + r.void_deals;
        }
        results.push(r);
    }

    print_all_results(n_games, &results);
}
