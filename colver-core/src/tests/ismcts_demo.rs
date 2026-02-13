use colver_core::mcts::{MctsConfig, MctsSearch};
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout::select_nth_bit;
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

fn run_ismcts_vs_random(
    n_games: u32,
    config: &NaiveIsMctsConfig,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut search = NaiveIsMctsSearch::new();
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

        while !state.is_terminal() {
            let team = state.current_player() & 1;
            let action = if team == 0 {
                // NS: Naive IS-MCTS
                search.search(&state, config, rng)
            } else {
                // EW: Random
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                select_nth_bit(legal, idx)
            };
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

fn run_ismcts_vs_mcts(
    n_games: u32,
    ismcts_config: &NaiveIsMctsConfig,
    mcts_config: &MctsConfig,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut ismcts_search = NaiveIsMctsSearch::new();
    let mut mcts_search = MctsSearch::new();
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

        while !state.is_terminal() {
            let team = state.current_player() & 1;
            let action = if team == 0 {
                // NS: Naive IS-MCTS
                ismcts_search.search(&state, ismcts_config, rng)
            } else {
                // EW: Perfect-info MCTS
                mcts_search.search(&state, mcts_config, rng)
            };
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

    // Part 1: IS-MCTS vs Random at various budgets
    let budgets: &[(u32, u32)] = &[
        (20, 50),    // 1000 total
        (40, 100),   // 4000 total
        (50, 200),   // 10000 total
    ];

    println!("===== Part 1: Naive IS-MCTS (NS) vs Random (EW) =====");
    for &(dets, iters) in budgets {
        let total = dets * iters;
        println!();
        println!(
            "--- {}D x {}I = {} total iterations ---",
            dets, iters, total
        );

        let config = NaiveIsMctsConfig {
            determinizations: dets,
            iterations_per_det: iters,
            ..Default::default()
        };

        let result = run_ismcts_vs_random(n_games, &config, &mut rng);
        print_summary(
            &format!("IS-MCTS({}D x {}I) vs Random", dets, iters),
            "NS (IS-MCTS)",
            "EW (Random)",
            n_games,
            &result,
        );
    }

    // Part 2: IS-MCTS vs Perfect-Info MCTS (same total budget)
    println!();
    println!("===== Part 2: Naive IS-MCTS (NS) vs Perfect-Info MCTS (EW) =====");
    println!("--- IS-MCTS 40x100=4000 vs MCTS 4000 ---");

    let ismcts_config = NaiveIsMctsConfig {
        determinizations: 40,
        iterations_per_det: 100,
        ..Default::default()
    };
    let mcts_config = MctsConfig {
        iterations: 4000,
        ..Default::default()
    };

    let result = run_ismcts_vs_mcts(n_games, &ismcts_config, &mcts_config, &mut rng);
    print_summary(
        "IS-MCTS(40D x 100I) vs MCTS(4000I)",
        "NS (IS-MCTS)",
        "EW (MCTS)",
        n_games,
        &result,
    );
}
