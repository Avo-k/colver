/// Round-robin bidding tournament: multiple parameterized bidding strategies
/// compete head-to-head in match play (first to 2000 points).
///
/// All players use Naive IS-MCTS for card play; only bidding differs.
/// Each pair plays both directions (A-NS vs B-EW and B-NS vs A-EW).
///
/// Usage:
///   cargo run --bin bid_tournament --release -- [matches_per_matchup] [time_ms] [mode]
///   cargo run --bin bid_tournament --release -- 100 20
///   cargo run --bin bid_tournament --release -- 100 15 fine-tune

use colver_core::bid_eval::{parametric_bid, BidParams};
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::state::{GameState, Phase};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const MATCH_TARGET: i32 = 2000;

struct MatchResult {
    winner: u8, // 0=NS, 1=EW
    ns_final: i32,
    ew_final: i32,
    deals_played: u32,
}

/// Play a single match — both sides use Naive IS-MCTS for play,
/// but bidding is controlled by BidParams.
fn play_match(
    ns_params: &BidParams,
    ew_params: &BidParams,
    time_ms: u32,
    rng: &mut impl Rng,
) -> MatchResult {
    let play_config = NaiveIsMctsConfig {
        iterations_per_det: 50,
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };

    let mut ns_cumulative: i32 = 0;
    let mut ew_cumulative: i32 = 0;
    let mut deals_played: u32 = 0;
    let mut dealer: u8 = rng.gen_range(0..4);

    while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);
        let mut ns_search = NaiveIsMctsSearch::new();
        let mut ew_search = NaiveIsMctsSearch::new();

        while !state.is_terminal() {
            let player = state.current_player();

            let action = if state.phase == Phase::Bidding {
                // Use parametric bidding
                let params = if player == 0 || player == 2 { ns_params } else { ew_params };
                parametric_bid(&state, params)
            } else {
                // Use IS-MCTS for card play
                match player {
                    0 | 2 => ns_search.search(&state, &play_config, rng),
                    _ => ew_search.search(&state, &play_config, rng),
                }
            };
            state.step(action);
        }

        let score = state.deal_score();
        deals_played += 1;

        if !(score.scores[0] == 0 && score.scores[1] == 0) {
            ns_cumulative += score.scores[0] as i32;
            ew_cumulative += score.scores[1] as i32;
        }

        dealer = (dealer + 3) % 4;
    }

    let winner = if ns_cumulative >= MATCH_TARGET && ew_cumulative >= MATCH_TARGET {
        if ns_cumulative >= ew_cumulative { 0 } else { 1 }
    } else if ns_cumulative >= MATCH_TARGET { 0 } else { 1 };

    MatchResult { winner, ns_final: ns_cumulative, ew_final: ew_cumulative, deals_played }
}

/// Result for one directional matchup (A as NS, B as EW).
#[derive(Default, Clone)]
struct MatchupResult {
    n_matches: u32,
    ns_wins: u32,
    ew_wins: u32,
    total_margin: i64, // NS - EW cumulative
}

impl MatchupResult {
    fn merge(&mut self, other: &MatchupResult) {
        self.n_matches += other.n_matches;
        self.ns_wins += other.ns_wins;
        self.ew_wins += other.ew_wins;
        self.total_margin += other.total_margin;
    }
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

/// Run matches for one direction (A as NS, B as EW), parallelized across threads.
fn run_matchup(
    ns_params: &BidParams,
    ew_params: &BidParams,
    n_matches: u32,
    time_ms: u32,
    base_seed: u64,
    progress: &AtomicU32,
) -> MatchupResult {
    let n_threads = num_cpus();
    let per_thread = (n_matches as usize + n_threads - 1) / n_threads;

    let results: Vec<MatchupResult> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let start = t * per_thread;
            let end = ((t + 1) * per_thread).min(n_matches as usize);
            if start >= end { continue; }
            let count = end - start;

            handles.push(s.spawn(move || {
                let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(t as u64 * 7919));
                let mut result = MatchupResult::default();
                for _ in 0..count {
                    let mr = play_match(ns_params, ew_params, time_ms, &mut rng);
                    result.n_matches += 1;
                    if mr.winner == 0 { result.ns_wins += 1; } else { result.ew_wins += 1; }
                    result.total_margin += (mr.ns_final - mr.ew_final) as i64;
                    progress.fetch_add(1, Ordering::Relaxed);
                }
                result
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut combined = MatchupResult::default();
    for r in &results {
        combined.merge(r);
    }
    combined
}

fn main() {
    let n_matches: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(100);

    let time_ms: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20);

    let mode = std::env::args().nth(3).unwrap_or_default();
    let strategies = if mode == "fine-tune" {
        BidParams::fine_tune_presets()
    } else {
        BidParams::all_presets()
    };
    let n = strategies.len();
    let total_matchups = n * (n - 1); // each pair, both directions
    let total_matches = total_matchups as u32 * n_matches;

    println!("=============================================================");
    println!("  BIDDING TOURNAMENT — Round Robin, First to {}", MATCH_TARGET);
    println!("  {} strategies, {} matches per matchup (both dirs)", n, n_matches);
    println!("  {} total matchups, {} total matches", total_matchups, total_matches);
    println!("  {}ms/move, {} CPUs", time_ms, num_cpus());
    println!("=============================================================");
    println!();

    for (i, s) in strategies.iter().enumerate() {
        println!("  [{}] {:<12} thresholds={:?} cap={}/{}/{}  QG={}",
            i, s.name,
            s.thresholds.map(|t| if t == u16::MAX { 99 } else { t }),
            s.opening_cap * 10, s.overcall_cap * 10, s.response_cap * 10,
            if s.quality_gate { "Y" } else { "N" });
    }
    println!();

    let progress = Arc::new(AtomicU32::new(0));
    let start = Instant::now();

    // Win matrix: win_matrix[i][j] = combined wins for i when playing against j
    // (sum of both directions: i-as-NS + i-as-EW)
    let mut win_matrix = vec![vec![0u32; n]; n];
    let mut margin_matrix = vec![vec![0i64; n]; n];
    let mut matches_matrix = vec![vec![0u32; n]; n];

    // Progress monitor
    let progress_clone = progress.clone();
    let monitor = std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let done = progress_clone.load(Ordering::Relaxed);
            if done >= total_matches { break; }
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed * 60.0;
            let eta = if rate > 0.0 { (total_matches - done) as f64 / rate * 60.0 } else { 0.0 };
            eprint!("\r  Progress: {}/{} matches ({:.0}%), {:.0}/min, ETA {:.0}s   ",
                done, total_matches, 100.0 * done as f64 / total_matches as f64, rate, eta);
        }
    });

    // Run all matchups sequentially (each matchup parallelizes internally)
    for i in 0..n {
        for j in 0..n {
            if i == j { continue; }
            // i as NS, j as EW
            let seed = (i * 1000 + j * 100 + 42) as u64;
            let result = run_matchup(
                &strategies[i], &strategies[j],
                n_matches, time_ms, seed, &progress,
            );
            win_matrix[i][j] += result.ns_wins;
            win_matrix[j][i] += result.ew_wins;
            margin_matrix[i][j] += result.total_margin;
            margin_matrix[j][i] -= result.total_margin;
            matches_matrix[i][j] += result.n_matches;
            matches_matrix[j][i] += result.n_matches;
        }
    }

    // Signal monitor to stop
    progress.store(total_matches, Ordering::Relaxed);
    let _ = monitor.join();
    let elapsed = start.elapsed();
    eprintln!();

    // Print results
    println!("=============================================================");
    println!("  WIN MATRIX (row player win% vs column, both directions combined)");
    println!("=============================================================");
    // Header
    print!("  {:>12}", "");
    for s in &strategies {
        print!("  {:>8}", s.name);
    }
    println!("    TOTAL");
    println!("  {}", "-".repeat(12 + 10 * (n + 1) + 8));

    let mut total_wins = vec![0u32; n];
    let mut total_played = vec![0u32; n];

    for i in 0..n {
        print!("  {:>12}", strategies[i].name);
        let mut row_wins = 0u32;
        let mut row_played = 0u32;
        for j in 0..n {
            if i == j {
                print!("      -  ");
            } else {
                let wins = win_matrix[i][j];
                let played = matches_matrix[i][j];
                let pct = 100.0 * wins as f64 / played as f64;
                print!("   {:5.1}% ", pct);
                row_wins += wins;
                row_played += played;
            }
        }
        let total_pct = 100.0 * row_wins as f64 / row_played as f64;
        print!("   {:5.1}%", total_pct);
        println!();
        total_wins[i] = row_wins;
        total_played[i] = row_played;
    }

    // Margin matrix
    println!();
    println!("=============================================================");
    println!("  MARGIN MATRIX (avg point margin for row player vs column)");
    println!("=============================================================");
    print!("  {:>12}", "");
    for s in &strategies {
        print!("  {:>8}", s.name);
    }
    println!();
    println!("  {}", "-".repeat(12 + 10 * n));

    for i in 0..n {
        print!("  {:>12}", strategies[i].name);
        for j in 0..n {
            if i == j {
                print!("      -  ");
            } else {
                let played = matches_matrix[i][j];
                let avg_margin = margin_matrix[i][j] as f64 / played as f64;
                print!("   {:+5.0}  ", avg_margin);
            }
        }
        println!();
    }

    // Rankings
    println!();
    println!("=============================================================");
    println!("  RANKINGS");
    println!("=============================================================");

    let mut ranking: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, 100.0 * total_wins[i] as f64 / total_played[i] as f64))
        .collect();
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, pct)) in ranking.iter().enumerate() {
        let s = &strategies[*idx];
        let avg_margin: f64 = {
            let total_m: i64 = (0..n).filter(|&j| j != *idx).map(|j| margin_matrix[*idx][j]).sum();
            let total_p: u32 = (0..n).filter(|&j| j != *idx).map(|j| matches_matrix[*idx][j]).sum();
            total_m as f64 / total_p as f64
        };
        println!("  {}. {:<12}  win {:.1}%  avg margin {:+.0}  [thresholds={:?} cap={}/{}/{} QG={}]",
            rank + 1, s.name, pct, avg_margin,
            s.thresholds.map(|t| if t == u16::MAX { 99 } else { t }),
            s.opening_cap * 10, s.overcall_cap * 10, s.response_cap * 10,
            if s.quality_gate { "Y" } else { "N" });
    }

    // vs balanced comparison (only in fine-tune mode)
    if mode == "fine-tune" {
        println!();
        println!("=============================================================");
        println!("  VS BALANCED (head-to-head win rate of each variant)");
        println!("=============================================================");
        // balanced is index 0
        for i in 1..n {
            let played = matches_matrix[i][0];
            if played == 0 { continue; }
            let wins = win_matrix[i][0];
            let pct = 100.0 * wins as f64 / played as f64;
            let avg_margin = margin_matrix[i][0] as f64 / played as f64;
            println!("  {:<12} vs balanced: {:5.1}% win  avg margin {:+.0}",
                strategies[i].name, pct, avg_margin);
        }
    }

    println!();
    println!("  Wall: {:.1}s ({} matches, {:.0}/min)",
        elapsed.as_secs_f64(), total_matches,
        total_matches as f64 / elapsed.as_secs_f64() * 60.0);
}
