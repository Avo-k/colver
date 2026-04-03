use colver_core::bid_eval::heuristic_bid;
use colver_core::mcts::{MctsConfig, MctsSearch, RolloutPolicy};
use colver_core::rollout::select_nth_bit;
use colver_core::state::{GameState, Phase};
use rand::Rng;
use std::time::Instant;

fn main() {
    let mut rng = rand::thread_rng();
    let n_games: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let config = MctsConfig {
        rollout_policy: RolloutPolicy::HeuristicPlay,
        ..MctsConfig::default()
    };
    let mut search = MctsSearch::new();

    let mut ns_wins = 0u32;
    let mut ew_wins = 0u32;
    let mut draws = 0u32;
    let mut ns_total_score = 0i32;
    let mut ew_total_score = 0i32;

    let start = Instant::now();

    for game in 0..n_games {
        let dealer = (game % 4) as u8;
        let mut state = GameState::deal_random(dealer, &mut rng);

        while !state.is_terminal() {
            let team = state.current_player() & 1;
            let action = if team == 0 {
                // NS: MCTS (tree-searches bids, heuristic rollouts)
                search.search(&state, &config, &mut rng)
            } else {
                // EW: Heuristic bid, random play
                if state.phase == Phase::Bidding {
                    heuristic_bid(&state)
                } else {
                    let legal = state.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    select_nth_bit(legal, idx)
                }
            };
            state.step(action);
        }

        let score = state.deal_score();
        ns_total_score += score.scores[0] as i32;
        ew_total_score += score.scores[1] as i32;

        if score.scores[0] > score.scores[1] {
            ns_wins += 1;
        } else if score.scores[1] > score.scores[0] {
            ew_wins += 1;
        } else {
            draws += 1;
        }

        println!(
            "Game {:3}: NS={:4} EW={:4} {}",
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

    let elapsed = start.elapsed();

    println!();
    println!("=== Results ({} games) ===", n_games);
    println!(
        "NS (MCTS) wins: {} ({:.1}%)",
        ns_wins,
        ns_wins as f64 / n_games as f64 * 100.0
    );
    println!(
        "EW (Random) wins: {} ({:.1}%)",
        ew_wins,
        ew_wins as f64 / n_games as f64 * 100.0
    );
    println!("Draws: {}", draws);
    println!(
        "NS total: {}, EW total: {}",
        ns_total_score, ew_total_score
    );
    println!(
        "Time: {:.2?} ({:.0}ms/game)",
        elapsed,
        elapsed.as_millis() as f64 / n_games as f64
    );
}
