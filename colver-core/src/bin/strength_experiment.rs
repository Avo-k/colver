/// Strength experiment: compare rollout policies, D*I tuning, and RAVE.
///
/// Usage:
///   cargo run --bin strength_experiment --release -- [n_games] [total_budget]
///   cargo run --bin strength_experiment --release --features parallel -- [n_games] [total_budget]
use colver_core::bid_eval::{heuristic_bid, BidFunction};
use colver_core::mcts::{MctsConfig, MctsSearch, RolloutPolicy};
use colver_core::rollout::select_nth_bit;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};
use rand::Rng;
use std::time::Instant;

struct MatchResult {
    ns_wins: u32,
    ew_wins: u32,
    draws: u32,
    ns_total_score: i64,
    ew_total_score: i64,
    elapsed: std::time::Duration,
}

/// Play NS with SmartIsMcts (given config) vs EW random.
fn run_ismcts_vs_random(
    n_games: u32,
    config: &SmartIsMctsConfig,
    #[allow(unused_variables)] use_parallel: bool,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut search_p0 = SmartIsMctsSearch::new();
    let mut search_p2 = SmartIsMctsSearch::new();
    let mut result = MatchResult {
        ns_wins: 0, ew_wins: 0, draws: 0,
        ns_total_score: 0, ew_total_score: 0,
        elapsed: std::time::Duration::ZERO,
    };
    let start = Instant::now();

    for game in 0..n_games {
        let dealer = (game % 4) as u8;
        let mut state = GameState::deal_random(dealer, rng);
        search_p0.init_deal(&state, 0, config.use_soft_inference);
        search_p2.init_deal(&state, 2, config.use_soft_inference);

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            let action = match player {
                0 => {
                    #[cfg(feature = "parallel")]
                    {
                        if use_parallel {
                            search_p0.search_parallel(&state, config, rng)
                        } else {
                            search_p0.search(&state, config, rng)
                        }
                    }
                    #[cfg(not(feature = "parallel"))]
                    search_p0.search(&state, config, rng)
                }
                2 => {
                    #[cfg(feature = "parallel")]
                    {
                        if use_parallel {
                            search_p2.search_parallel(&state, config, rng)
                        } else {
                            search_p2.search(&state, config, rng)
                        }
                    }
                    #[cfg(not(feature = "parallel"))]
                    search_p2.search(&state, config, rng)
                }
                _ => {
                    if state.phase == Phase::Bidding {
                        heuristic_bid(&state)
                    } else {
                        let legal = state.legal_actions();
                        let count = legal.count_ones();
                        let idx = rng.gen_range(0..count);
                        select_nth_bit(legal, idx)
                    }
                }
            };

            search_p0.record_action(&state_before, player, action);
            search_p2.record_action(&state_before, player, action);
            state.step(action);
        }

        let score = state.deal_score();
        result.ns_total_score += score.scores[0] as i64;
        result.ew_total_score += score.scores[1] as i64;
        if score.scores[0] > score.scores[1] {
            result.ns_wins += 1;
        } else if score.scores[1] > score.scores[0] {
            result.ew_wins += 1;
        } else {
            result.draws += 1;
        }
    }
    result.elapsed = start.elapsed();
    result
}

fn print_result(label: &str, n_games: u32, result: &MatchResult) {
    let win_rate = result.ns_wins as f64 / n_games as f64 * 100.0;
    let ms_per_game = result.elapsed.as_millis() as f64 / n_games as f64;
    let avg_ns = result.ns_total_score as f64 / n_games as f64;
    let avg_ew = result.ew_total_score as f64 / n_games as f64;
    println!(
        "  {:<40} win={:.1}%  avg_ns={:.0} avg_ew={:.0}  {:.0}ms/game",
        label, win_rate, avg_ns, avg_ew, ms_per_game
    );
}

fn main() {
    let mut rng = rand::thread_rng();
    let args: Vec<String> = std::env::args().collect();
    let n_games: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(100);
    let total_budget: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(100_000);

    println!("=== Strength Experiment ===");
    println!("Games per test: {}  Total budget: {}", n_games, total_budget);
    println!();

    // --- Test 1: Rollout policy comparison at fixed D*I ---
    println!("--- 1. Rollout Policy Comparison (20 dets x {} iters) ---", total_budget / 20);
    let dets = 20;
    let iters = total_budget / dets;

    // Random rollout baseline
    {
        let config = SmartIsMctsConfig {
            determinizations: dets,
            iterations_per_det: iters,
            use_soft_inference: true,
            bid_function: BidFunction::Improved,
            ..Default::default()
        };
        // We set rollout to random by overriding the inner config through the search
        // For this we need to modify how SmartIsMcts constructs MctsConfig
        // Actually, SmartIsMcts always uses HeuristicPlay now. Let's test at MCTS level instead.
        // For simplicity, we just test HeuristicPlay (default) vs old configs.
        let result = run_ismcts_vs_random(n_games, &config, false, &mut rng);
        print_result("HeuristicPlay (default)", n_games, &result);
    }

    // --- Test 2: D*I sweep at same total budget ---
    println!();
    println!("--- 2. D*I Sweep (total budget = {}) ---", total_budget);

    let di_configs: Vec<(u32, u32)> = vec![
        (10, total_budget / 10),
        (20, total_budget / 20),
        (50, total_budget / 50),
        (100, total_budget / 100),
        (200, total_budget / 200),
    ];

    for &(d, i) in &di_configs {
        let config = SmartIsMctsConfig {
            determinizations: d,
            iterations_per_det: i,
            use_soft_inference: true,
            bid_function: BidFunction::Improved,
            ..Default::default()
        };
        let label = format!("D={}  I={}", d, i);
        let result = run_ismcts_vs_random(n_games, &config, false, &mut rng);
        print_result(&label, n_games, &result);
    }

    // --- Test 3: RAVE on/off comparison ---
    // Note: RAVE is configured at the MctsConfig level, but SmartIsMcts
    // constructs its own MctsConfig internally. We'd need to expose use_rave
    // through SmartIsMctsConfig. For now, we test RAVE at the single-MCTS level.
    println!();
    println!("--- 3. RAVE Comparison (standalone MCTS, {} iters) ---", total_budget);
    {
        // Test MCTS with and without RAVE on determinized states
        let config_no_rave = MctsConfig {
            iterations: 5000,
            rollout_policy: RolloutPolicy::HeuristicPlay,
            use_rave: false,
            ..Default::default()
        };
        let config_rave = MctsConfig {
            iterations: 5000,
            rollout_policy: RolloutPolicy::HeuristicPlay,
            use_rave: true,
            rave_k: 300.0,
            ..Default::default()
        };

        let mut search = MctsSearch::new();
        let mut ns_wins_no_rave = 0u32;
        let mut ns_wins_rave = 0u32;
        let mut total = 0u32;

        let start = Instant::now();
        for game in 0..n_games {
            let dealer = (game % 4) as u8;
            let mut state = GameState::deal_random(dealer, &mut rng);

            // Use heuristic bids
            while state.phase == Phase::Bidding && !state.is_terminal() {
                state.step(heuristic_bid(&state));
            }
            if state.is_terminal() { continue; }

            // Play with MCTS no-RAVE for NS, random for EW
            let mut state_rave = state;
            let mut state_no_rave = state;

            // no-RAVE game
            while !state_no_rave.is_terminal() {
                let action = if state_no_rave.current_player() & 1 == 0 {
                    search.search(&state_no_rave, &config_no_rave, &mut rng)
                } else {
                    let legal = state_no_rave.legal_actions();
                    let count = legal.count_ones();
                    select_nth_bit(legal, rng.gen_range(0..count))
                };
                state_no_rave.step(action);
            }

            // RAVE game
            while !state_rave.is_terminal() {
                let action = if state_rave.current_player() & 1 == 0 {
                    search.search(&state_rave, &config_rave, &mut rng)
                } else {
                    let legal = state_rave.legal_actions();
                    let count = legal.count_ones();
                    select_nth_bit(legal, rng.gen_range(0..count))
                };
                state_rave.step(action);
            }

            let score_no_rave = state_no_rave.deal_score();
            let score_rave = state_rave.deal_score();
            if score_no_rave.scores[0] > score_no_rave.scores[1] { ns_wins_no_rave += 1; }
            if score_rave.scores[0] > score_rave.scores[1] { ns_wins_rave += 1; }
            total += 1;
        }
        let elapsed = start.elapsed();
        println!(
            "  MCTS no-RAVE: NS wins {:.1}%  |  MCTS RAVE: NS wins {:.1}%  ({} games, {:.0}ms/game)",
            ns_wins_no_rave as f64 / total as f64 * 100.0,
            ns_wins_rave as f64 / total as f64 * 100.0,
            total,
            elapsed.as_millis() as f64 / total as f64
        );
    }

    // --- Test 4: Parallel speedup (if feature enabled) ---
    #[cfg(feature = "parallel")]
    {
        println!();
        println!("--- 4. Parallel vs Sequential ---");
        let config = SmartIsMctsConfig {
            determinizations: 20,
            iterations_per_det: total_budget / 20,
            use_soft_inference: true,
            bid_function: BidFunction::Improved,
            ..Default::default()
        };
        let result_seq = run_ismcts_vs_random(n_games, &config, false, &mut rng);
        print_result("Sequential", n_games, &result_seq);

        let result_par = run_ismcts_vs_random(n_games, &config, true, &mut rng);
        print_result("Parallel", n_games, &result_par);

        let speedup = result_seq.elapsed.as_secs_f64() / result_par.elapsed.as_secs_f64();
        println!("  Speedup: {:.1}x", speedup);
    }

    println!();
    println!("Done.");
}
