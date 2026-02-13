use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout::select_nth_bit;
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
    elapsed: std::time::Duration,
}

fn run_smart_vs_random(
    n_games: u32,
    config: &SmartIsMctsConfig,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut search_p0 = SmartIsMctsSearch::new();
    let mut search_p2 = SmartIsMctsSearch::new();
    let mut result = MatchResult {
        ns_wins: 0,
        ew_wins: 0,
        draws: 0,
        ns_total_score: 0,
        ew_total_score: 0,
        elapsed: std::time::Duration::ZERO,
    };

    let start = Instant::now();

    for game in 0..n_games {
        let dealer = (game % 4) as u8;
        let mut state = GameState::deal_random(dealer, rng);

        // Initialize beliefs for both NS players
        search_p0.init_deal(&state, 0, config.use_soft_inference);
        search_p2.init_deal(&state, 2, config.use_soft_inference);

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            let action = match player {
                0 => search_p0.search(&state, config, rng),
                2 => search_p2.search(&state, config, rng),
                _ => {
                    // EW: Random
                    let legal = state.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    select_nth_bit(legal, idx)
                }
            };

            // Both searches observe all actions
            search_p0.record_action(&state_before, player, action);
            search_p2.record_action(&state_before, player, action);

            state.step(action);
        }

        let score = state.deal_score();
        result.ns_total_score += score.scores[0] as i32;
        result.ew_total_score += score.scores[1] as i32;

        if score.scores[0] > score.scores[1] {
            result.ns_wins += 1;
        } else if score.scores[1] > score.scores[0] {
            result.ew_wins += 1;
        } else {
            result.draws += 1;
        }

        println!(
            "  Game {:3}: NS={:4} EW={:4} {}",
            game + 1,
            score.scores[0],
            score.scores[1],
            if score.scores[0] > score.scores[1] {
                "NS wins"
            } else if score.scores[1] > score.scores[0] {
                "EW wins"
            } else {
                "Draw"
            }
        );
    }

    result.elapsed = start.elapsed();
    result
}

fn run_smart_vs_naive(
    n_games: u32,
    smart_config: &SmartIsMctsConfig,
    naive_config: &NaiveIsMctsConfig,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut search_p0 = SmartIsMctsSearch::new();
    let mut search_p2 = SmartIsMctsSearch::new();
    let mut naive_search = NaiveIsMctsSearch::new();
    let mut result = MatchResult {
        ns_wins: 0,
        ew_wins: 0,
        draws: 0,
        ns_total_score: 0,
        ew_total_score: 0,
        elapsed: std::time::Duration::ZERO,
    };

    let start = Instant::now();

    for game in 0..n_games {
        let dealer = (game % 4) as u8;
        let mut state = GameState::deal_random(dealer, rng);

        search_p0.init_deal(&state, 0, smart_config.use_soft_inference);
        search_p2.init_deal(&state, 2, smart_config.use_soft_inference);

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            let action = match player {
                0 | 2 => {
                    // NS: Smart IS-MCTS
                    if player == 0 {
                        search_p0.search(&state, smart_config, rng)
                    } else {
                        search_p2.search(&state, smart_config, rng)
                    }
                }
                _ => {
                    // EW: Naive IS-MCTS
                    naive_search.search(&state, naive_config, rng)
                }
            };

            search_p0.record_action(&state_before, player, action);
            search_p2.record_action(&state_before, player, action);

            state.step(action);
        }

        let score = state.deal_score();
        result.ns_total_score += score.scores[0] as i32;
        result.ew_total_score += score.scores[1] as i32;

        if score.scores[0] > score.scores[1] {
            result.ns_wins += 1;
        } else if score.scores[1] > score.scores[0] {
            result.ew_wins += 1;
        } else {
            result.draws += 1;
        }

        println!(
            "  Game {:3}: NS={:4} EW={:4} {}",
            game + 1,
            score.scores[0],
            score.scores[1],
            if score.scores[0] > score.scores[1] {
                "NS wins"
            } else if score.scores[1] > score.scores[0] {
                "EW wins"
            } else {
                "Draw"
            }
        );
    }

    result.elapsed = start.elapsed();
    result
}

fn print_summary(label: &str, ns_label: &str, ew_label: &str, n_games: u32, r: &MatchResult) {
    println!();
    println!("=== {} ({} games) ===", label, n_games);
    println!(
        "  {} wins: {} ({:.1}%)",
        ns_label,
        r.ns_wins,
        r.ns_wins as f64 / n_games as f64 * 100.0
    );
    println!(
        "  {} wins: {} ({:.1}%)",
        ew_label,
        r.ew_wins,
        r.ew_wins as f64 / n_games as f64 * 100.0
    );
    println!("  Draws: {}", r.draws);
    println!(
        "  Avg score: {} {:.0}, {} {:.0}",
        ns_label,
        r.ns_total_score as f64 / n_games as f64,
        ew_label,
        r.ew_total_score as f64 / n_games as f64
    );
    println!(
        "  Time: {:.2?} ({:.0}ms/game)",
        r.elapsed,
        r.elapsed.as_millis() as f64 / n_games as f64
    );
}

fn main() {
    let mut rng = rand::thread_rng();
    let n_games: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let dets = 20u32;
    let iters = 50u32;
    let total = dets * iters;

    // Part 1: Smart IS-MCTS vs Random
    println!("===== Part 1: Smart IS-MCTS (NS) vs Random (EW) =====");
    println!("--- {}D x {}I = {} total iterations ---", dets, iters, total);

    let smart_config = SmartIsMctsConfig {
        determinizations: dets,
        iterations_per_det: iters,
        ..Default::default()
    };

    let result = run_smart_vs_random(n_games, &smart_config, &mut rng);
    print_summary(
        &format!("Smart IS-MCTS({}D x {}I) vs Random", dets, iters),
        "NS (Smart IS-MCTS)",
        "EW (Random)",
        n_games,
        &result,
    );

    // Part 2: Smart IS-MCTS vs Naive IS-MCTS (same budget)
    println!();
    println!("===== Part 2: Smart IS-MCTS (NS) vs Naive IS-MCTS (EW) =====");
    println!("--- Both at {}D x {}I = {} total iterations ---", dets, iters, total);

    let naive_config = NaiveIsMctsConfig {
        determinizations: dets,
        iterations_per_det: iters,
        ..Default::default()
    };

    let result = run_smart_vs_naive(n_games, &smart_config, &naive_config, &mut rng);
    print_summary(
        &format!("Smart({}D x {}I) vs Naive({}D x {}I)", dets, iters, dets, iters),
        "NS (Smart IS-MCTS)",
        "EW (Naive IS-MCTS)",
        n_games,
        &result,
    );
}
