/// V2 bidding fine-tune tournament (round 2): focused variants with DMC + Oracle MCTS.
///
/// Usage:
///   cargo run --bin v2_tournament --release -- [dmc_matches] [oracle_matches] [model_path] [oracle_iters]
///   cargo run --bin v2_tournament --release -- 60 30 models/dmc_16000000.bin 3000

use colver_core::bid_eval::{self, V2Config};
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM};
use colver_core::mcts::{MctsConfig, MctsSearch, RolloutPolicy};
use colver_core::state::{GameState, Phase};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const MATCH_TARGET: i32 = 2000;

// --- Shared DMC weights ---

struct DmcWeights {
    floats: Vec<f32>,
    hidden: usize,
    obs_dim: usize,
    dueling: bool,
}

impl DmcWeights {
    fn load(path: &str) -> std::io::Result<Self> {
        let net = DmcNet::load(path)?;
        let obs_dim = net.obs_dim();
        let hidden = net.hidden();
        let dueling = net.is_dueling();
        drop(net);
        let data = std::fs::read(path)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(DmcWeights { floats, hidden, obs_dim, dueling })
    }

    fn make_net(&self) -> DmcNet {
        DmcNet::from_floats(&self.floats, self.hidden, self.obs_dim, self.dueling).unwrap()
    }
}

// --- Bidder enum ---

#[derive(Clone)]
enum Bidder {
    Improved,
    V2(V2Config),
}

impl Bidder {
    fn bid(&self, state: &GameState) -> u8 {
        match self {
            Bidder::Improved => bid_eval::improved_bid(state),
            Bidder::V2(cfg) => bid_eval::improved_v2_configurable_bid(state, cfg),
        }
    }

    fn name(&self) -> &str {
        match self {
            Bidder::Improved => "Improved",
            Bidder::V2(cfg) => cfg.name,
        }
    }
}

// --- Card play ---

#[derive(Clone, Copy)]
enum PlayMethod {
    Dmc,
    Oracle,
}

// --- Match play ---

struct MatchResult {
    winner: u8,
    ns_final: i32,
    ew_final: i32,
}

fn play_match_dmc(
    ns_bidder: &Bidder,
    ew_bidder: &Bidder,
    weights: &DmcWeights,
    rng: &mut StdRng,
) -> MatchResult {
    let mut net = weights.make_net();
    let mut obs_buf = vec![0.0f32; OBS_DIM];

    let mut ns_cumulative: i32 = 0;
    let mut ew_cumulative: i32 = 0;
    let mut dealer: u8 = rng.gen_range(0..4);

    while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);

        while !state.is_terminal() {
            let player = state.current_player();
            let is_ns = player == 0 || player == 2;
            let state_before = state;

            let action = if state.phase == Phase::Bidding {
                if is_ns { ns_bidder.bid(&state) } else { ew_bidder.bid(&state) }
            } else {
                dmc_obs::write_observation(&mut obs_buf, 0, &state, &tracking);
                let legal_mask = state.legal_actions() as u32;
                let (a, _) = net.best_action(&obs_buf, legal_mask);
                a
            };

            tracking.track_action(&state_before, action);
            state.step(action);
        }

        let score = state.deal_score();
        if !(score.scores[0] == 0 && score.scores[1] == 0) {
            ns_cumulative += score.scores[0] as i32;
            ew_cumulative += score.scores[1] as i32;
        }
        dealer = (dealer + 3) % 4;
    }

    finish_match(ns_cumulative, ew_cumulative)
}

fn play_match_oracle(
    ns_bidder: &Bidder,
    ew_bidder: &Bidder,
    oracle_iters: u32,
    rng: &mut StdRng,
) -> MatchResult {
    let oracle_config = MctsConfig {
        iterations: oracle_iters,
        rollout_policy: RolloutPolicy::HeuristicPlay,
        ..Default::default()
    };
    let mut oracle = MctsSearch::new();

    let mut ns_cumulative: i32 = 0;
    let mut ew_cumulative: i32 = 0;
    let mut dealer: u8 = rng.gen_range(0..4);

    while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);

        while !state.is_terminal() {
            let player = state.current_player();
            let is_ns = player == 0 || player == 2;

            let action = if state.phase == Phase::Bidding {
                if is_ns { ns_bidder.bid(&state) } else { ew_bidder.bid(&state) }
            } else {
                oracle.search(&state, &oracle_config, rng)
            };

            state.step(action);
        }

        let score = state.deal_score();
        if !(score.scores[0] == 0 && score.scores[1] == 0) {
            ns_cumulative += score.scores[0] as i32;
            ew_cumulative += score.scores[1] as i32;
        }
        dealer = (dealer + 3) % 4;
    }

    finish_match(ns_cumulative, ew_cumulative)
}

fn finish_match(ns_cumulative: i32, ew_cumulative: i32) -> MatchResult {
    let winner = if ns_cumulative >= MATCH_TARGET && ew_cumulative >= MATCH_TARGET {
        if ns_cumulative >= ew_cumulative { 0 } else { 1 }
    } else if ns_cumulative >= MATCH_TARGET { 0 } else { 1 };
    MatchResult { winner, ns_final: ns_cumulative, ew_final: ew_cumulative }
}

// --- Matchup runner ---

#[derive(Default, Clone)]
struct MatchupResult {
    n_matches: u32,
    ns_wins: u32,
    total_margin: i64,
}

impl MatchupResult {
    fn merge(&mut self, other: &MatchupResult) {
        self.n_matches += other.n_matches;
        self.ns_wins += other.ns_wins;
        self.total_margin += other.total_margin;
    }
}

fn run_matchup(
    ns: &Bidder,
    ew: &Bidder,
    n_matches: u32,
    play: PlayMethod,
    weights: &DmcWeights,
    oracle_iters: u32,
    n_threads: usize,
    base_seed: u64,
    progress: &AtomicU32,
) -> MatchupResult {
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
                    let mr = match play {
                        PlayMethod::Dmc => play_match_dmc(ns, ew, weights, &mut rng),
                        PlayMethod::Oracle => play_match_oracle(ns, ew, oracle_iters, &mut rng),
                    };
                    result.n_matches += 1;
                    if mr.winner == 0 { result.ns_wins += 1; }
                    result.total_margin += (mr.ns_final - mr.ew_final) as i64;
                    progress.fetch_add(1, Ordering::Relaxed);
                }
                result
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut combined = MatchupResult::default();
    for r in &results { combined.merge(r); }
    combined
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

// --- Round robin for one play method ---

struct RoundRobinResult {
    win_matrix: Vec<Vec<u32>>,
    margin_matrix: Vec<Vec<i64>>,
    matches_matrix: Vec<Vec<u32>>,
}

fn run_round_robin(
    bidders: &[Bidder],
    n_matches: u32,
    play: PlayMethod,
    weights: &DmcWeights,
    oracle_iters: u32,
    n_threads: usize,
    progress: &AtomicU32,
    seed_offset: u64,
) -> RoundRobinResult {
    let n = bidders.len();
    let mut win_matrix = vec![vec![0u32; n]; n];
    let mut margin_matrix = vec![vec![0i64; n]; n];
    let mut matches_matrix = vec![vec![0u32; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let seed1 = seed_offset + (i * 1000 + j * 100 + 42) as u64;
            let r1 = run_matchup(&bidders[i], &bidders[j], n_matches, play, weights, oracle_iters, n_threads, seed1, progress);

            let seed2 = seed_offset + (j * 1000 + i * 100 + 42) as u64;
            let r2 = run_matchup(&bidders[j], &bidders[i], n_matches, play, weights, oracle_iters, n_threads, seed2, progress);

            win_matrix[i][j] += r1.ns_wins + (r2.n_matches - r2.ns_wins);
            win_matrix[j][i] += (r1.n_matches - r1.ns_wins) + r2.ns_wins;
            margin_matrix[i][j] += r1.total_margin - r2.total_margin;
            margin_matrix[j][i] += r2.total_margin - r1.total_margin;
            matches_matrix[i][j] += r1.n_matches + r2.n_matches;
            matches_matrix[j][i] += r1.n_matches + r2.n_matches;
        }
    }

    RoundRobinResult { win_matrix, margin_matrix, matches_matrix }
}

fn print_results(label: &str, bidders: &[Bidder], rr: &RoundRobinResult) {
    let n = bidders.len();

    // Rankings
    let mut total_wins = vec![0u32; n];
    let mut total_played = vec![0u32; n];
    for i in 0..n {
        for j in 0..n {
            if i != j {
                total_wins[i] += rr.win_matrix[i][j];
                total_played[i] += rr.matches_matrix[i][j];
            }
        }
    }

    let mut ranking: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, 100.0 * total_wins[i] as f64 / total_played[i] as f64))
        .collect();
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!("=============================================================");
    println!("  RANKINGS — {}", label);
    println!("=============================================================");

    for (rank, (idx, pct)) in ranking.iter().enumerate() {
        let avg_margin: f64 = {
            let total_m: i64 = (0..n).filter(|&j| j != *idx).map(|j| rr.margin_matrix[*idx][j]).sum();
            let total_p: u32 = (0..n).filter(|&j| j != *idx).map(|j| rr.matches_matrix[*idx][j]).sum();
            total_m as f64 / total_p as f64
        };
        println!("  {:>2}. {:<12}  win {:5.1}%  avg margin {:+6.0}",
            rank + 1, bidders[*idx].name(), pct, avg_margin);
    }

    // Head-to-head vs Improved baseline
    println!();
    println!("  HEAD-TO-HEAD vs Improved ({}):", label);
    for i in 1..n {
        let played = rr.matches_matrix[0][i];
        if played == 0 { continue; }
        let impr_wins = rr.win_matrix[0][i];
        let v2_wins = rr.win_matrix[i][0];
        let margin = rr.margin_matrix[i][0] as f64 / played as f64;
        println!("    {:<12}  {:5.1}% vs Improved {:5.1}%  margin {:+.0}",
            bidders[i].name(),
            100.0 * v2_wins as f64 / played as f64,
            100.0 * impr_wins as f64 / played as f64,
            margin);
    }
    println!();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let dmc_matches: u32 = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(60);
    let oracle_matches: u32 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(30);
    let model_path = args.get(3).map(|s| s.as_str()).unwrap_or("models/dmc_16000000.bin");
    let oracle_iters: u32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(3000);
    let n_threads = default_threads();

    // Load DMC model
    let weights = DmcWeights::load(model_path).unwrap_or_else(|e| {
        eprintln!("Failed to load {}: {}", model_path, e);
        std::process::exit(1);
    });
    println!("Loaded {} (obs_dim={}, hidden={}, dueling={})",
        model_path, weights.obs_dim, weights.hidden, weights.dueling);

    // Focused bidder list: top performers from round 1 + new combos
    let bidders: Vec<Bidder> = vec![
        Bidder::Improved,
        // Round 1 winners
        Bidder::V2(V2Config::no_lead()),       // #1 H2H (71.7% vs Improved)
        Bidder::V2(V2Config::defensive()),      // #2 H2H (66.7% vs Improved)
        Bidder::V2(V2Config::coinche_4aces()),  // #3 H2H (58.3% vs Improved)
        Bidder::V2(V2Config::coinche_only()),   // Simple coinche (53.3% vs Improved)
        Bidder::V2(V2Config::resp_misfit()),    // Misfit only (51.9% overall)
        // New combos from round 1 insights
        Bidder::V2(V2Config {
            name: "v2_co+mis4",
            theoreme3_aces: 4,
            partner_response: true,
            resp_misfit_penalty: -3,
            ..V2Config::coinche_4aces()
        }),
        Bidder::V2(V2Config {
            name: "v2_def_l1",
            lead_bonus: 1,
            ..V2Config::defensive()
        }),
    ];
    let n = bidders.len();
    let dmc_matchups = n * (n - 1);
    let oracle_matchups = n * (n - 1);
    let total_matches = dmc_matchups as u32 * dmc_matches + oracle_matchups as u32 * oracle_matches;

    println!("=============================================================");
    println!("  V2 FINE-TUNE TOURNAMENT R2 — DMC + Oracle MCTS");
    println!("  {} bidders, {}/{}  matches/matchup (DMC/Oracle)", n, dmc_matches, oracle_matches);
    println!("  {} oracle iters, {} threads", oracle_iters, n_threads);
    println!("  {} total matches", total_matches);
    println!("=============================================================\n");

    for (i, b) in bidders.iter().enumerate() {
        println!("  [{:>2}] {}", i, b.name());
    }
    println!();

    let progress = Arc::new(AtomicU32::new(0));
    let start = Instant::now();

    // Progress monitor
    let progress_clone = progress.clone();
    let monitor = std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(15));
            let done = progress_clone.load(Ordering::Relaxed);
            if done >= total_matches { break; }
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed * 60.0;
            let eta = if rate > 0.0 { (total_matches - done) as f64 / rate * 60.0 } else { 0.0 };
            eprint!("\r  Progress: {}/{} ({:.0}%), {:.0}/min, ETA {:.0}s   ",
                done, total_matches, 100.0 * done as f64 / total_matches as f64, rate, eta);
        }
    });

    // --- DMC round robin ---
    println!("  Running DMC round robin ({} matches/matchup)...", dmc_matches);
    let dmc_rr = run_round_robin(&bidders, dmc_matches, PlayMethod::Dmc, &weights, oracle_iters, n_threads, &progress, 0);
    let dmc_elapsed = start.elapsed();
    println!("  DMC done in {:.0}s\n", dmc_elapsed.as_secs_f64());

    // --- Oracle round robin ---
    println!("  Running Oracle MCTS round robin ({} matches/matchup, {} iters)...", oracle_matches, oracle_iters);
    let oracle_rr = run_round_robin(&bidders, oracle_matches, PlayMethod::Oracle, &weights, oracle_iters, n_threads, &progress, 100000);
    let oracle_elapsed = start.elapsed() - dmc_elapsed;
    println!("  Oracle done in {:.0}s\n", oracle_elapsed.as_secs_f64());

    // Signal monitor to stop
    progress.store(total_matches, Ordering::Relaxed);
    let _ = monitor.join();
    eprintln!();

    // --- Print results ---
    print_results("DMC card play", &bidders, &dmc_rr);
    print_results("Oracle MCTS card play", &bidders, &oracle_rr);

    // --- Combined rankings (sum across both play methods) ---
    println!("=============================================================");
    println!("  COMBINED RANKINGS (DMC + Oracle)");
    println!("=============================================================");

    let mut combined_wins = vec![0u32; n];
    let mut combined_played = vec![0u32; n];
    let mut combined_margin = vec![0i64; n];
    let mut combined_margin_played = vec![0u32; n];

    for i in 0..n {
        for j in 0..n {
            if i != j {
                combined_wins[i] += dmc_rr.win_matrix[i][j] + oracle_rr.win_matrix[i][j];
                combined_played[i] += dmc_rr.matches_matrix[i][j] + oracle_rr.matches_matrix[i][j];
                combined_margin[i] += dmc_rr.margin_matrix[i][j] + oracle_rr.margin_matrix[i][j];
                combined_margin_played[i] += dmc_rr.matches_matrix[i][j] + oracle_rr.matches_matrix[i][j];
            }
        }
    }

    let mut ranking: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, 100.0 * combined_wins[i] as f64 / combined_played[i] as f64))
        .collect();
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, pct)) in ranking.iter().enumerate() {
        let avg_margin = combined_margin[*idx] as f64 / combined_margin_played[*idx] as f64;
        println!("  {:>2}. {:<12}  win {:5.1}%  avg margin {:+6.0}",
            rank + 1, bidders[*idx].name(), pct, avg_margin);
    }

    let elapsed = start.elapsed();
    println!();
    println!("  Wall: {:.1}s ({} total matches, DMC {:.0}s + Oracle {:.0}s)",
        elapsed.as_secs_f64(), total_matches,
        dmc_elapsed.as_secs_f64(), oracle_elapsed.as_secs_f64());
}
