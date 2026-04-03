use colver_core::bid_eval::heuristic_bid;
use colver_core::rollout::select_nth_bit;
use colver_core::single_tree_ismcts::{SingleTreeIsmctsConfig, SingleTreeIsmctsSearch};
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};
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

/// Single-tree IS-MCTS (NS) vs Smart/Ensemble IS-MCTS (EW).
fn run_single_tree_vs_ensemble(
    n_games: u32,
    st_config: &SingleTreeIsmctsConfig,
    ens_config: &SmartIsMctsConfig,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut st_p0 = SingleTreeIsmctsSearch::new();
    let mut st_p2 = SingleTreeIsmctsSearch::new();
    let mut ens_p1 = SmartIsMctsSearch::new();
    let mut ens_p3 = SmartIsMctsSearch::new();
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

        st_p0.init_deal(&state, 0, st_config.use_soft_inference);
        st_p2.init_deal(&state, 2, st_config.use_soft_inference);
        ens_p1.init_deal(&state, 1, ens_config.use_soft_inference);
        ens_p3.init_deal(&state, 3, ens_config.use_soft_inference);

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            let action = match player {
                0 => st_p0.search(&state, st_config, rng),
                2 => st_p2.search(&state, st_config, rng),
                1 => ens_p1.search(&state, ens_config, rng),
                3 => ens_p3.search(&state, ens_config, rng),
                _ => unreachable!(),
            };

            // All searches observe all actions
            st_p0.advance(&state_before, player, action);
            st_p2.advance(&state_before, player, action);
            ens_p1.record_action(&state_before, player, action);
            ens_p3.record_action(&state_before, player, action);

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

        if (game + 1) % 50 == 0 || game + 1 == n_games {
            println!(
                "  Game {:3}: NS={:4} EW={:4} | Running: NS {}-{}-{} EW",
                game + 1,
                score.scores[0],
                score.scores[1],
                result.ns_wins,
                result.draws,
                result.ew_wins,
            );
        }
    }

    result.elapsed = start.elapsed();
    result
}

/// Single-tree IS-MCTS (NS) vs Random (EW).
fn run_single_tree_vs_random(
    n_games: u32,
    config: &SingleTreeIsmctsConfig,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut st_p0 = SingleTreeIsmctsSearch::new();
    let mut st_p2 = SingleTreeIsmctsSearch::new();
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

        st_p0.init_deal(&state, 0, config.use_soft_inference);
        st_p2.init_deal(&state, 2, config.use_soft_inference);

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            let action = match player {
                0 => st_p0.search(&state, config, rng),
                2 => st_p2.search(&state, config, rng),
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

            st_p0.advance(&state_before, player, action);
            st_p2.advance(&state_before, player, action);

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

        if (game + 1) % 50 == 0 || game + 1 == n_games {
            println!(
                "  Game {:3}: NS={:4} EW={:4} | Running: NS {}-{}-{} EW",
                game + 1,
                score.scores[0],
                score.scores[1],
                result.ns_wins,
                result.draws,
                result.ew_wins,
            );
        }
    }

    result.elapsed = start.elapsed();
    result
}

/// Single-tree with reuse vs Single-tree without reuse.
fn run_reuse_vs_no_reuse(
    n_games: u32,
    iterations: u32,
    rng: &mut impl Rng,
) -> MatchResult {
    let config_reuse = SingleTreeIsmctsConfig {
        iterations,
        reuse_tree: true,
        ..Default::default()
    };
    let config_no_reuse = SingleTreeIsmctsConfig {
        iterations,
        reuse_tree: false,
        ..Default::default()
    };

    let mut reuse_p0 = SingleTreeIsmctsSearch::new();
    let mut reuse_p2 = SingleTreeIsmctsSearch::new();
    let mut fresh_p1 = SingleTreeIsmctsSearch::new();
    let mut fresh_p3 = SingleTreeIsmctsSearch::new();

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

        reuse_p0.init_deal(&state, 0, true);
        reuse_p2.init_deal(&state, 2, true);
        fresh_p1.init_deal(&state, 1, true);
        fresh_p3.init_deal(&state, 3, true);

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            let action = match player {
                0 => reuse_p0.search(&state, &config_reuse, rng),
                2 => reuse_p2.search(&state, &config_reuse, rng),
                1 => fresh_p1.search(&state, &config_no_reuse, rng),
                3 => fresh_p3.search(&state, &config_no_reuse, rng),
                _ => unreachable!(),
            };

            reuse_p0.advance(&state_before, player, action);
            reuse_p2.advance(&state_before, player, action);
            fresh_p1.advance(&state_before, player, action);
            fresh_p3.advance(&state_before, player, action);

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

        if (game + 1) % 50 == 0 || game + 1 == n_games {
            println!(
                "  Game {:3}: NS={:4} EW={:4} | Running: NS {}-{}-{} EW",
                game + 1,
                score.scores[0],
                score.scores[1],
                result.ns_wins,
                result.draws,
                result.ew_wins,
            );
        }
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

    let total_iters: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    // Part 1: Single-tree IS-MCTS vs Random
    println!("===== Part 1: Single-tree IS-MCTS (NS) vs Random (EW) =====");
    println!("--- {} iterations ---", total_iters);

    let st_config = SingleTreeIsmctsConfig {
        iterations: total_iters,
        ..Default::default()
    };

    let result = run_single_tree_vs_random(n_games, &st_config, &mut rng);
    print_summary(
        &format!("Single-tree IS-MCTS({}) vs Random", total_iters),
        "NS (Single-tree)",
        "EW (Random)",
        n_games,
        &result,
    );

    // Part 2: Single-tree IS-MCTS vs Ensemble IS-MCTS (same total budget)
    println!();
    println!("===== Part 2: Single-tree IS-MCTS (NS) vs Ensemble IS-MCTS (EW) =====");
    println!("--- Both at {} total iterations ---", total_iters);

    // Split total budget across determinizations to match single-tree budget
    let ens_dets = 20u32;
    let ens_iters = total_iters / ens_dets;
    let ens_config = SmartIsMctsConfig {
        determinizations: ens_dets,
        iterations_per_det: ens_iters,
        ..Default::default()
    };

    let result = run_single_tree_vs_ensemble(n_games, &st_config, &ens_config, &mut rng);
    print_summary(
        &format!("Single-tree({}) vs Ensemble({}x{})", total_iters, ens_dets, ens_iters),
        "NS (Single-tree)",
        "EW (Ensemble)",
        n_games,
        &result,
    );

    // Part 3: Single-tree with reuse vs without reuse
    println!();
    println!("===== Part 3: Tree Reuse (NS) vs Fresh Tree (EW) =====");
    println!("--- Both single-tree at {} iterations ---", total_iters);

    let result = run_reuse_vs_no_reuse(n_games, total_iters, &mut rng);
    print_summary(
        &format!("Reuse({}) vs Fresh({})", total_iters, total_iters),
        "NS (Reuse)",
        "EW (Fresh)",
        n_games,
        &result,
    );
}
