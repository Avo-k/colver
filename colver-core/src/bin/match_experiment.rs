/// Match experiment: play full matches (first to 2000 points) to evaluate
/// bidding strategies with realistic scoring dynamics.
///
/// All experiments run in parallel using std::thread with inner match parallelism.
///
/// Usage:
///   cargo run --bin match_experiment --release -- [n_matches] [time_ms]
///   cargo run --bin match_experiment --release -- 200 30
use colver_core::bid_eval::BidFunction;
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::GameState;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const MATCH_TARGET: i32 = 2000;

/// Detailed stats for one match.
struct MatchResult {
    winner: u8,
    ns_final: i32,
    ew_final: i32,
    deals_played: u32,
    void_deals: u32,
    // Per-team contract stats
    ns_contracts: u32,
    ns_contracts_made: u32,
    ew_contracts: u32,
    ew_contracts_made: u32,
    // Deal score details
    ns_deal_scores: Vec<i16>,
    ew_deal_scores: Vec<i16>,
    coinches: u32,
}

/// Aggregated stats across all matches in an experiment.
struct ExperimentResult {
    label: String,
    ns_desc: String,
    ew_desc: String,
    n_matches: u32,
    // Match-level
    ns_match_wins: u32,
    ew_match_wins: u32,
    total_deals: u64,
    total_void_deals: u64,
    deal_counts: Vec<u32>,
    margins: Vec<i32>,
    // Contract stats
    total_ns_contracts: u64,
    total_ns_made: u64,
    total_ew_contracts: u64,
    total_ew_made: u64,
    total_coinches: u64,
    // Score distributions
    all_ns_deal_scores: Vec<i16>,
    all_ew_deal_scores: Vec<i16>,
    elapsed: std::time::Duration,
}

impl ExperimentResult {
    fn new(label: &str, ns_desc: &str, ew_desc: &str) -> Self {
        ExperimentResult {
            label: label.to_string(),
            ns_desc: ns_desc.to_string(),
            ew_desc: ew_desc.to_string(),
            n_matches: 0,
            ns_match_wins: 0,
            ew_match_wins: 0,
            total_deals: 0,
            total_void_deals: 0,
            deal_counts: Vec::new(),
            margins: Vec::new(),
            total_ns_contracts: 0,
            total_ns_made: 0,
            total_ew_contracts: 0,
            total_ew_made: 0,
            total_coinches: 0,
            all_ns_deal_scores: Vec::new(),
            all_ew_deal_scores: Vec::new(),
            elapsed: std::time::Duration::ZERO,
        }
    }

    fn add_match(&mut self, mr: MatchResult) {
        self.n_matches += 1;
        if mr.winner == 0 { self.ns_match_wins += 1; } else { self.ew_match_wins += 1; }
        self.total_deals += mr.deals_played as u64;
        self.total_void_deals += mr.void_deals as u64;
        self.deal_counts.push(mr.deals_played);
        self.margins.push(mr.ns_final - mr.ew_final);
        self.total_ns_contracts += mr.ns_contracts as u64;
        self.total_ns_made += mr.ns_contracts_made as u64;
        self.total_ew_contracts += mr.ew_contracts as u64;
        self.total_ew_made += mr.ew_contracts_made as u64;
        self.total_coinches += mr.coinches as u64;
        self.all_ns_deal_scores.extend_from_slice(&mr.ns_deal_scores);
        self.all_ew_deal_scores.extend_from_slice(&mr.ew_deal_scores);
    }

    fn merge(&mut self, other: ExperimentResult) {
        self.n_matches += other.n_matches;
        self.ns_match_wins += other.ns_match_wins;
        self.ew_match_wins += other.ew_match_wins;
        self.total_deals += other.total_deals;
        self.total_void_deals += other.total_void_deals;
        self.deal_counts.extend(other.deal_counts);
        self.margins.extend(other.margins);
        self.total_ns_contracts += other.total_ns_contracts;
        self.total_ns_made += other.total_ns_made;
        self.total_ew_contracts += other.total_ew_contracts;
        self.total_ew_made += other.total_ew_made;
        self.total_coinches += other.total_coinches;
        self.all_ns_deal_scores.extend(other.all_ns_deal_scores);
        self.all_ew_deal_scores.extend(other.all_ew_deal_scores);
    }
}

/// Play a single match (first to 2000) — NS Smart IS-MCTS, EW Naive IS-MCTS.
fn play_match_smart_vs_naive(
    ns_config: &SmartIsMctsConfig,
    ew_config: &NaiveIsMctsConfig,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut ns_cumulative: i32 = 0;
    let mut ew_cumulative: i32 = 0;
    let mut mr = MatchResult {
        winner: 0, ns_final: 0, ew_final: 0,
        deals_played: 0, void_deals: 0,
        ns_contracts: 0, ns_contracts_made: 0,
        ew_contracts: 0, ew_contracts_made: 0,
        ns_deal_scores: Vec::new(), ew_deal_scores: Vec::new(),
        coinches: 0,
    };
    let mut dealer: u8 = rng.gen_range(0..4);

    while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);

        let mut search_p0 = SmartIsMctsSearch::new();
        let mut search_p2 = SmartIsMctsSearch::new();
        let mut ew_search = NaiveIsMctsSearch::new();

        search_p0.init_deal(&state, 0, ns_config.use_soft_inference);
        search_p2.init_deal(&state, 2, ns_config.use_soft_inference);

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            let action = match player {
                0 => search_p0.search(&state, ns_config, rng),
                2 => search_p2.search(&state, ns_config, rng),
                _ => ew_search.search(&state, ew_config, rng),
            };

            search_p0.record_action(&state_before, player, action);
            search_p2.record_action(&state_before, player, action);
            state.step(action);
        }

        let score = state.deal_score();
        mr.deals_played += 1;
        mr.ns_deal_scores.push(score.scores[0]);
        mr.ew_deal_scores.push(score.scores[1]);

        if score.scores[0] == 0 && score.scores[1] == 0 {
            mr.void_deals += 1;
        } else {
            ns_cumulative += score.scores[0] as i32;
            ew_cumulative += score.scores[1] as i32;

            if state.contract.value > 0 {
                if state.contract.coinche > 0 { mr.coinches += 1; }
                if state.contract.team == 0 {
                    mr.ns_contracts += 1;
                    if score.scores[0] > 0 { mr.ns_contracts_made += 1; }
                } else {
                    mr.ew_contracts += 1;
                    if score.scores[1] > 0 { mr.ew_contracts_made += 1; }
                }
            }
        }

        dealer = (dealer + 3) % 4;
    }

    mr.ns_final = ns_cumulative;
    mr.ew_final = ew_cumulative;
    mr.winner = if ns_cumulative >= MATCH_TARGET && ew_cumulative >= MATCH_TARGET {
        if ns_cumulative >= ew_cumulative { 0 } else { 1 }
    } else if ns_cumulative >= MATCH_TARGET { 0 } else { 1 };
    mr
}

/// Play a single match — both sides Naive IS-MCTS.
fn play_match_naive_vs_naive(
    ns_config: &NaiveIsMctsConfig,
    ew_config: &NaiveIsMctsConfig,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut ns_cumulative: i32 = 0;
    let mut ew_cumulative: i32 = 0;
    let mut mr = MatchResult {
        winner: 0, ns_final: 0, ew_final: 0,
        deals_played: 0, void_deals: 0,
        ns_contracts: 0, ns_contracts_made: 0,
        ew_contracts: 0, ew_contracts_made: 0,
        ns_deal_scores: Vec::new(), ew_deal_scores: Vec::new(),
        coinches: 0,
    };
    let mut dealer: u8 = rng.gen_range(0..4);

    while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);
        let mut ns_search = NaiveIsMctsSearch::new();
        let mut ew_search = NaiveIsMctsSearch::new();

        while !state.is_terminal() {
            let player = state.current_player();
            let action = match player {
                0 | 2 => ns_search.search(&state, ns_config, rng),
                _ => ew_search.search(&state, ew_config, rng),
            };
            state.step(action);
        }

        let score = state.deal_score();
        mr.deals_played += 1;
        mr.ns_deal_scores.push(score.scores[0]);
        mr.ew_deal_scores.push(score.scores[1]);

        if score.scores[0] == 0 && score.scores[1] == 0 {
            mr.void_deals += 1;
        } else {
            ns_cumulative += score.scores[0] as i32;
            ew_cumulative += score.scores[1] as i32;

            if state.contract.value > 0 {
                if state.contract.coinche > 0 { mr.coinches += 1; }
                if state.contract.team == 0 {
                    mr.ns_contracts += 1;
                    if score.scores[0] > 0 { mr.ns_contracts_made += 1; }
                } else {
                    mr.ew_contracts += 1;
                    if score.scores[1] > 0 { mr.ew_contracts_made += 1; }
                }
            }
        }

        dealer = (dealer + 3) % 4;
    }

    mr.ns_final = ns_cumulative;
    mr.ew_final = ew_cumulative;
    mr.winner = if ns_cumulative >= MATCH_TARGET && ew_cumulative >= MATCH_TARGET {
        if ns_cumulative >= ew_cumulative { 0 } else { 1 }
    } else if ns_cumulative >= MATCH_TARGET { 0 } else { 1 };
    mr
}

struct ExperimentSpec {
    label: &'static str,
    ns_desc: &'static str,
    ew_desc: &'static str,
    ns_bid: BidFunction,
    ew_bid: BidFunction,
    ns_smart: bool,
}

fn num_cpus() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

fn run_experiment_parallel(
    spec: &ExperimentSpec,
    n_matches: u32,
    time_ms: u32,
    base_seed: u64,
    progress: &AtomicU32,
) -> ExperimentResult {
    let n_threads = num_cpus();
    let matches_per_thread = (n_matches as usize + n_threads - 1) / n_threads;
    let start = Instant::now();

    let thread_results: Vec<ExperimentResult> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let start_idx = t * matches_per_thread;
            let end_idx = ((t + 1) * matches_per_thread).min(n_matches as usize);
            if start_idx >= end_idx { continue; }
            let count = end_idx - start_idx;

            handles.push(s.spawn(move || {
                let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(t as u64 * 9973));
                let mut result = ExperimentResult::new(spec.label, spec.ns_desc, spec.ew_desc);

                if spec.ns_smart {
                    let ns_config = SmartIsMctsConfig {
                        iterations_per_det: 50,
                        time_limit_ms: Some(time_ms),
                        bid_function: spec.ns_bid,
                        ..Default::default()
                    };
                    let ew_config = NaiveIsMctsConfig {
                        iterations_per_det: 50,
                        time_limit_ms: Some(time_ms),
                        bid_function: spec.ew_bid,
                        ..Default::default()
                    };
                    for _ in 0..count {
                        result.add_match(play_match_smart_vs_naive(&ns_config, &ew_config, &mut rng));
                        progress.fetch_add(1, Ordering::Relaxed);
                    }
                } else {
                    let ns_config = NaiveIsMctsConfig {
                        iterations_per_det: 50,
                        time_limit_ms: Some(time_ms),
                        bid_function: spec.ns_bid,
                        ..Default::default()
                    };
                    let ew_config = NaiveIsMctsConfig {
                        iterations_per_det: 50,
                        time_limit_ms: Some(time_ms),
                        bid_function: spec.ew_bid,
                        ..Default::default()
                    };
                    for _ in 0..count {
                        result.add_match(play_match_naive_vs_naive(&ns_config, &ew_config, &mut rng));
                        progress.fetch_add(1, Ordering::Relaxed);
                    }
                }
                result
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut combined = ExperimentResult::new(spec.label, spec.ns_desc, spec.ew_desc);
    for tr in thread_results {
        combined.merge(tr);
    }
    combined.elapsed = start.elapsed();
    combined
}

fn percentile(sorted: &[i16], p: f64) -> i16 {
    let idx = ((sorted.len() as f64 - 1.0) * p) as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn print_experiment(r: &ExperimentResult) {
    let n = r.n_matches as f64;
    let total_deals = r.total_deals as f64;
    let avg_deals = total_deals / n;
    let avg_voids = r.total_void_deals as f64 / n;

    let mut sorted_deals = r.deal_counts.clone();
    sorted_deals.sort();
    let median_deals = sorted_deals[sorted_deals.len() / 2];

    let mut sorted_margins = r.margins.clone();
    sorted_margins.sort();
    let avg_margin = r.margins.iter().sum::<i32>() as f64 / n;
    let median_margin = sorted_margins[sorted_margins.len() / 2];

    // Contract stats
    let ns_contracts = r.total_ns_contracts as f64;
    let ew_contracts = r.total_ew_contracts as f64;
    let total_contracts = ns_contracts + ew_contracts;
    let ns_take_rate = if total_contracts > 0.0 { ns_contracts / total_contracts * 100.0 } else { 0.0 };
    let ns_success_rate = if ns_contracts > 0.0 { r.total_ns_made as f64 / ns_contracts * 100.0 } else { 0.0 };
    let ew_success_rate = if ew_contracts > 0.0 { r.total_ew_made as f64 / ew_contracts * 100.0 } else { 0.0 };

    // Deal score distribution
    let mut ns_scores = r.all_ns_deal_scores.clone();
    let mut ew_scores = r.all_ew_deal_scores.clone();
    ns_scores.sort();
    ew_scores.sort();
    let ns_avg_deal = ns_scores.iter().map(|&x| x as f64).sum::<f64>() / ns_scores.len() as f64;
    let ew_avg_deal = ew_scores.iter().map(|&x| x as f64).sum::<f64>() / ew_scores.len() as f64;

    // Count big scores (>= 300, i.e. coinche/capot results)
    let ns_big = ns_scores.iter().filter(|&&x| x >= 300).count();
    let ew_big = ew_scores.iter().filter(|&&x| x >= 300).count();
    let ns_zero = ns_scores.iter().filter(|&&x| x == 0).count();
    let ew_zero = ew_scores.iter().filter(|&&x| x == 0).count();

    println!();
    println!("  {}", r.label);
    println!("    NS: {}", r.ns_desc);
    println!("    EW: {}", r.ew_desc);
    println!();

    // Match results
    println!("    MATCH RESULTS ({} matches)", r.n_matches);
    println!(
        "      Wins: NS {} ({:.1}%)  |  EW {} ({:.1}%)",
        r.ns_match_wins, r.ns_match_wins as f64 / n * 100.0,
        r.ew_match_wins, r.ew_match_wins as f64 / n * 100.0,
    );
    println!("      Margin: avg {:+.0}, median {:+}", avg_margin, median_margin);
    println!(
        "      Deals/match: avg {:.1}, median {} (void: {:.1}/match)",
        avg_deals, median_deals, avg_voids,
    );

    // Contract stats
    println!();
    println!("    CONTRACTS ({:.0} total across {} deals)", total_contracts, r.total_deals);
    println!(
        "      NS took: {:.0} ({:.0}%), made {:.0} ({:.0}%), failed {:.0} ({:.0}%)",
        ns_contracts, ns_take_rate,
        r.total_ns_made as f64, ns_success_rate,
        ns_contracts - r.total_ns_made as f64, 100.0 - ns_success_rate,
    );
    println!(
        "      EW took: {:.0} ({:.0}%), made {:.0} ({:.0}%), failed {:.0} ({:.0}%)",
        ew_contracts, 100.0 - ns_take_rate,
        r.total_ew_made as f64, ew_success_rate,
        ew_contracts - r.total_ew_made as f64, 100.0 - ew_success_rate,
    );
    println!(
        "      Contracts/match: NS {:.1}, EW {:.1}",
        ns_contracts / n, ew_contracts / n,
    );
    println!("      Coinches: {:.0} ({:.1}% of contracts)", r.total_coinches as f64, r.total_coinches as f64 / total_contracts * 100.0);

    // Deal score distribution
    println!();
    println!("    DEAL SCORES ({} deals)", r.total_deals);
    println!(
        "      NS avg: {:.0}/deal  |  EW avg: {:.0}/deal  |  delta: {:+.0}/deal",
        ns_avg_deal, ew_avg_deal, ns_avg_deal - ew_avg_deal,
    );
    println!(
        "      NS: p10={} p25={} p50={} p75={} p90={}",
        percentile(&ns_scores, 0.10), percentile(&ns_scores, 0.25),
        percentile(&ns_scores, 0.50), percentile(&ns_scores, 0.75),
        percentile(&ns_scores, 0.90),
    );
    println!(
        "      EW: p10={} p25={} p50={} p75={} p90={}",
        percentile(&ew_scores, 0.10), percentile(&ew_scores, 0.25),
        percentile(&ew_scores, 0.50), percentile(&ew_scores, 0.75),
        percentile(&ew_scores, 0.90),
    );
    println!(
        "      Zero deals: NS {} ({:.1}%), EW {} ({:.1}%)",
        ns_zero, ns_zero as f64 / ns_scores.len() as f64 * 100.0,
        ew_zero, ew_zero as f64 / ew_scores.len() as f64 * 100.0,
    );
    println!(
        "      Big deals (>=300): NS {} ({:.1}%), EW {} ({:.1}%)",
        ns_big, ns_big as f64 / ns_scores.len() as f64 * 100.0,
        ew_big, ew_big as f64 / ew_scores.len() as f64 * 100.0,
    );

    println!(
        "    Time: {:.1?} ({:.1}s/match, {:.0}ms/deal)",
        r.elapsed, r.elapsed.as_secs_f64() / n,
        r.elapsed.as_millis() as f64 / total_deals,
    );
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let n_matches: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let time_ms: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50);

    let specs: Vec<ExperimentSpec> = vec![
        ExperimentSpec {
            label: "Exp 1: improved_bid + beliefs vs heuristic_bid",
            ns_desc: "Smart IS-MCTS + improved_bid",
            ew_desc: "Naive IS-MCTS + heuristic_bid",
            ns_bid: BidFunction::Improved, ew_bid: BidFunction::Heuristic, ns_smart: true,
        },
        ExperimentSpec {
            label: "Exp 2: beliefs only (both heuristic_bid)",
            ns_desc: "Smart IS-MCTS + heuristic_bid",
            ew_desc: "Naive IS-MCTS + heuristic_bid",
            ns_bid: BidFunction::Heuristic, ew_bid: BidFunction::Heuristic, ns_smart: true,
        },
        ExperimentSpec {
            label: "Exp 3: pure bidding (improved vs heuristic, both Naive)",
            ns_desc: "Naive IS-MCTS + improved_bid",
            ew_desc: "Naive IS-MCTS + heuristic_bid",
            ns_bid: BidFunction::Improved, ew_bid: BidFunction::Heuristic, ns_smart: false,
        },
        ExperimentSpec {
            label: "Exp 4: baseline (both Naive + heuristic_bid)",
            ns_desc: "Naive IS-MCTS + heuristic_bid",
            ew_desc: "Naive IS-MCTS + heuristic_bid",
            ns_bid: BidFunction::Heuristic, ew_bid: BidFunction::Heuristic, ns_smart: false,
        },
        ExperimentSpec {
            label: "Exp 5: symmetry (both Naive + improved_bid)",
            ns_desc: "Naive IS-MCTS + improved_bid",
            ew_desc: "Naive IS-MCTS + improved_bid",
            ns_bid: BidFunction::Improved, ew_bid: BidFunction::Improved, ns_smart: false,
        },
    ];

    let n_experiments = specs.len();
    let total_matches = n_matches * n_experiments as u32;

    println!("=============================================================");
    println!("  MATCH EXPERIMENT — First to {} points", MATCH_TARGET);
    println!("  {} matches/experiment x {} experiments = {} total",
        n_matches, n_experiments, total_matches);
    println!("  {}ms/move, {} CPUs, {} threads/experiment",
        time_ms, num_cpus(), num_cpus());
    println!("  All experiments + matches run in parallel");
    println!("=============================================================");
    println!();

    let progress = Arc::new(AtomicU32::new(0));
    let mut rng = rand::thread_rng();
    let seeds: Vec<u64> = (0..n_experiments).map(|_| rng.gen()).collect();
    let global_start = Instant::now();

    // Progress monitor
    let progress_clone = Arc::clone(&progress);
    let monitor = std::thread::spawn(move || {
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let done = progress_clone.load(Ordering::Relaxed);
            if done >= total_matches { break; }
            let elapsed = global_start.elapsed();
            let rate = done as f64 / elapsed.as_secs_f64().max(0.001);
            let remaining = (total_matches - done) as f64 / rate.max(0.01);
            eprint!(
                "\r  Progress: {}/{} matches ({:.0}%), {:.0}/min, ETA {:.0}s   ",
                done, total_matches, done as f64 / total_matches as f64 * 100.0,
                rate * 60.0, remaining,
            );
        }
        eprintln!();
    });

    // Run all experiments in parallel, each with inner match parallelism
    let results: Vec<ExperimentResult> = std::thread::scope(|s| {
        let handles: Vec<_> = specs.iter().enumerate().map(|(i, spec)| {
            let progress = &progress;
            let seed = seeds[i];
            s.spawn(move || {
                run_experiment_parallel(spec, n_matches, time_ms, seed, progress)
            })
        }).collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    progress.store(total_matches, Ordering::Relaxed);
    let _ = monitor.join();
    let wall_time = global_start.elapsed();

    // Print results
    println!("=============================================================");
    println!("  RESULTS");
    println!("=============================================================");
    for r in &results {
        print_experiment(r);
    }

    // Summary
    println!();
    println!("=============================================================");
    println!("  SUMMARY");
    println!("=============================================================");
    println!("  {:50} {:>6} {:>8} {:>6} {:>6} {:>6}",
        "", "Win%", "Margin", "NStak", "NSmk%", "EWmk%");
    for r in &results {
        let n = r.n_matches as f64;
        let ns_rate = if r.total_ns_contracts > 0 { r.total_ns_made as f64 / r.total_ns_contracts as f64 * 100.0 } else { 0.0 };
        let ew_rate = if r.total_ew_contracts > 0 { r.total_ew_made as f64 / r.total_ew_contracts as f64 * 100.0 } else { 0.0 };
        let total_c = r.total_ns_contracts + r.total_ew_contracts;
        let ns_take = if total_c > 0 { r.total_ns_contracts as f64 / total_c as f64 * 100.0 } else { 0.0 };
        println!(
            "  {:50} {:5.0}% {:+7.0} {:5.0}% {:5.0}% {:5.0}%",
            r.label,
            r.ns_match_wins as f64 / n * 100.0,
            r.margins.iter().sum::<i32>() as f64 / n,
            ns_take, ns_rate, ew_rate,
        );
    }

    let cpu_time: std::time::Duration = results.iter().map(|r| r.elapsed).sum();
    println!();
    println!(
        "  Wall: {:.1?} (CPU: {:.1?}, speedup: {:.1}x)",
        wall_time, cpu_time, cpu_time.as_secs_f64() / wall_time.as_secs_f64(),
    );
}
