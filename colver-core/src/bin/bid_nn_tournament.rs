/// Tournament: NN bid vs improved_v2 across play methods.
///
/// Tests whether the NN bidding advantage holds across different card play engines:
///   - Smart IS-DD (belief-weighted determinization + DD solver)
///   - DD oracle (perfect play, solve_best_card)
///   - DouDou35 / DMC (Q-network play)
///
/// Round-robin match play (first to 2000) with multi-threaded matchup execution.
///
/// Usage:
///   cargo run --bin bid_nn_tournament --release -- models/bid_nn_final.bin \
///     --dmc-model models/dmc_35.bin --matches 100 --time-ms 20 --threads 8

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_eval::BidFunction;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM};
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::solver;
use colver_core::state::{GameState, Phase};

const MATCH_TARGET: i32 = 2000;

// --- Shared weights (read-only after init, each thread makes its own net) ---

struct BidNetWeights {
    floats: Vec<f32>,
    hidden: usize,
    obs_dim: usize,
    dueling: bool,
}

impl BidNetWeights {
    fn load(path: &str) -> std::io::Result<Self> {
        let net = BidNet::load(path)?;
        let obs_dim = net.obs_dim();
        let hidden = net.hidden();
        let dueling = net.is_dueling();
        drop(net);
        let data = std::fs::read(path)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(BidNetWeights { floats, hidden, obs_dim, dueling })
    }

    fn make_net(&self) -> BidNet {
        BidNet::from_floats(&self.floats, self.hidden, self.obs_dim, self.dueling).unwrap()
    }
}

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

// --- Agent definitions ---

#[derive(Clone, Copy)]
enum BidMethod {
    ImprovedV2,
    NnBid,
}

#[derive(Clone, Copy)]
enum PlayMethod {
    SmartIsDd,
    DdOracle,
    Dmc,
}

#[derive(Clone)]
struct Agent {
    name: String,
    bid: BidMethod,
    play: PlayMethod,
}

// --- Match play ---

struct MatchResult {
    winner: u8, // 0=NS, 1=EW
    ns_final: i32,
    ew_final: i32,
}

fn play_match(
    ns_agent: &Agent,
    ew_agent: &Agent,
    time_ms: u32,
    bid_weights: &BidNetWeights,
    dmc_weights: Option<&DmcWeights>,
    rng: &mut StdRng,
) -> MatchResult {
    // Thread-local resources
    let mut bid_net = bid_weights.make_net();
    let mut dmc_net: Option<DmcNet> = dmc_weights.map(|w| w.make_net());
    let mut tt_buf = solver::new_tt_buffer();
    let mut obs_buf = vec![0.0f32; OBS_DIM];

    let dd_config = IsDdConfig {
        determinizations: 20,
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };

    let mut ns_cumulative: i32 = 0;
    let mut ew_cumulative: i32 = 0;
    let mut dealer: u8 = rng.gen_range(0..4);

    while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);
        let mut bid_history: Vec<(u8, u8)> = Vec::new();

        // Per-deal Smart IS-DD search objects (per player: 0,2 for NS; 1,3 for EW)
        let ns_uses_sdd = matches!(ns_agent.play, PlayMethod::SmartIsDd);
        let ew_uses_sdd = matches!(ew_agent.play, PlayMethod::SmartIsDd);
        let mut ns_sdd = [IsDdSearch::new(), IsDdSearch::new()];
        let mut ew_sdd = [IsDdSearch::new(), IsDdSearch::new()];
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);

        if ns_uses_sdd {
            ns_sdd[0].init_deal(&state, 0, true);
            ns_sdd[1].init_deal(&state, 2, true);
        }
        if ew_uses_sdd {
            ew_sdd[0].init_deal(&state, 1, true);
            ew_sdd[1].init_deal(&state, 3, true);
        }

        while !state.is_terminal() {
            let player = state.current_player();
            let is_ns = player == 0 || player == 2;
            let agent = if is_ns { ns_agent } else { ew_agent };
            let state_before = state;

            let action = if state.phase == Phase::Bidding {
                match agent.bid {
                    BidMethod::ImprovedV2 => BidFunction::ImprovedV2.bid(&state),
                    BidMethod::NnBid => {
                        let obs = bid_obs::make_bid_observation(&state, &bid_history);
                        let legal = state.legal_actions();
                        let (best, _) = bid_net.best_action(&obs, legal);
                        best
                    }
                }
            } else {
                match agent.play {
                    PlayMethod::SmartIsDd => {
                        if is_ns {
                            let idx = if player == 0 { 0 } else { 1 };
                            ns_sdd[idx].search(&state, &dd_config, rng)
                        } else {
                            let idx = if player == 1 { 0 } else { 1 };
                            ew_sdd[idx].search(&state, &dd_config, rng)
                        }
                    }
                    PlayMethod::DdOracle => {
                        solver::solve_with_scores(&state, Some(&mut tt_buf)).best_card
                    }
                    PlayMethod::Dmc => {
                        let net = dmc_net.as_mut().unwrap();
                        dmc_obs::write_observation(&mut obs_buf, 0, &state, &tracking);
                        let legal_mask = state.legal_actions() as u32;
                        let (action, _) = net.best_action(&obs_buf, legal_mask);
                        action
                    }
                }
            };

            // Track bid history for NN observation
            if state.phase == Phase::Bidding {
                bid_history.push((player, action));
            }

            // Record action on Smart IS-DD searches
            if ns_uses_sdd {
                ns_sdd[0].record_action(&state_before, player, action);
                ns_sdd[1].record_action(&state_before, player, action);
            }
            if ew_uses_sdd {
                ew_sdd[0].record_action(&state_before, player, action);
                ew_sdd[1].record_action(&state_before, player, action);
            }
            // Track for DMC obs
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

    let winner = if ns_cumulative >= MATCH_TARGET && ew_cumulative >= MATCH_TARGET {
        if ns_cumulative >= ew_cumulative { 0 } else { 1 }
    } else if ns_cumulative >= MATCH_TARGET {
        0
    } else {
        1
    };

    MatchResult {
        winner,
        ns_final: ns_cumulative,
        ew_final: ew_cumulative,
    }
}

// --- Matchup runner ---

#[derive(Default, Clone)]
struct MatchupResult {
    n_matches: u32,
    ns_wins: u32,
    ew_wins: u32,
    total_margin: i64,
}

impl MatchupResult {
    fn merge(&mut self, other: &MatchupResult) {
        self.n_matches += other.n_matches;
        self.ns_wins += other.ns_wins;
        self.ew_wins += other.ew_wins;
        self.total_margin += other.total_margin;
    }
}

fn run_matchup(
    ns_agent: &Agent,
    ew_agent: &Agent,
    n_matches: u32,
    time_ms: u32,
    bid_weights: &BidNetWeights,
    dmc_weights: Option<&DmcWeights>,
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
            if start >= end {
                continue;
            }
            let count = end - start;

            handles.push(s.spawn(move || {
                let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(t as u64 * 7919));
                let mut result = MatchupResult::default();
                for _ in 0..count {
                    let mr = play_match(
                        ns_agent, ew_agent, time_ms, bid_weights, dmc_weights, &mut rng,
                    );
                    result.n_matches += 1;
                    if mr.winner == 0 {
                        result.ns_wins += 1;
                    } else {
                        result.ew_wins += 1;
                    }
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

// --- CLI & main ---

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        eprintln!("Usage: bid_nn_tournament <bid_model.bin> [--dmc-model PATH] [--matches N] [--time-ms T] [--threads T] [--seed S]");
        std::process::exit(1);
    }

    let bid_model_path = &args[1];
    let mut dmc_model_path: Option<String> = None;
    let mut n_matches: u32 = 50;
    let mut time_ms: u32 = 20;
    let mut n_threads = default_threads();
    let mut seed = 42u64;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--dmc-model" => { dmc_model_path = Some(args[i + 1].clone()); i += 2; }
            "--matches" => { n_matches = args[i + 1].parse().unwrap(); i += 2; }
            "--time-ms" => { time_ms = args[i + 1].parse().unwrap(); i += 2; }
            "--threads" => { n_threads = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            _ => { eprintln!("Unknown arg: {}", args[i]); i += 1; }
        }
    }

    // Load models
    let bid_weights = BidNetWeights::load(bid_model_path)
        .unwrap_or_else(|e| panic!("Failed to load bid model {}: {}", bid_model_path, e));
    println!("Bid NN: {} (obs_dim={}, hidden={}, dueling={})",
        bid_model_path, bid_weights.obs_dim, bid_weights.hidden, bid_weights.dueling);

    let dmc_weights: Option<DmcWeights> = dmc_model_path.as_ref().map(|path| {
        let w = DmcWeights::load(path)
            .unwrap_or_else(|e| panic!("Failed to load DMC model {}: {}", path, e));
        println!("DMC:    {} (obs_dim={}, hidden={}, dueling={})",
            path, w.obs_dim, w.hidden, w.dueling);
        w
    });

    // Build agent list
    let mut agents: Vec<Agent> = vec![
        Agent { name: "NN+SDD".into(), bid: BidMethod::NnBid, play: PlayMethod::SmartIsDd },
        Agent { name: "V2+SDD".into(), bid: BidMethod::ImprovedV2, play: PlayMethod::SmartIsDd },
        Agent { name: "NN+DD".into(), bid: BidMethod::NnBid, play: PlayMethod::DdOracle },
        Agent { name: "V2+DD".into(), bid: BidMethod::ImprovedV2, play: PlayMethod::DdOracle },
    ];

    if dmc_weights.is_some() {
        agents.push(Agent { name: "NN+DMC".into(), bid: BidMethod::NnBid, play: PlayMethod::Dmc });
        agents.push(Agent { name: "V2+DMC".into(), bid: BidMethod::ImprovedV2, play: PlayMethod::Dmc });
    }

    let n = agents.len();
    let total_matchups = n * (n - 1); // both directions
    let total_matches = total_matchups as u32 * n_matches;

    println!();
    println!("=============================================================");
    println!("  NN BID TOURNAMENT — Round Robin, First to {}", MATCH_TARGET);
    println!("  {} agents, {} matches/matchup (both dirs)", n, n_matches);
    println!("  {} matchups, {} total matches", total_matchups, total_matches);
    println!("  {}ms/move (IS-DD), {} threads", time_ms, n_threads);
    println!("=============================================================");
    println!();
    for (idx, a) in agents.iter().enumerate() {
        let bid_str = match a.bid {
            BidMethod::ImprovedV2 => "improved_v2",
            BidMethod::NnBid => "NN bidder",
        };
        let play_str = match a.play {
            PlayMethod::SmartIsDd => "Smart IS-DD",
            PlayMethod::DdOracle => "DD oracle",
            PlayMethod::Dmc => "DMC (DouDou35)",
        };
        println!("  [{:>2}] {:<8}  bid={:<12}  play={}", idx, a.name, bid_str, play_str);
    }
    println!();

    let progress = Arc::new(AtomicU32::new(0));
    let start = Instant::now();

    // Matrices
    let mut win_matrix = vec![vec![0u32; n]; n];
    let mut margin_matrix = vec![vec![0i64; n]; n];
    let mut matches_matrix = vec![vec![0u32; n]; n];

    // Progress monitor
    let progress_clone = progress.clone();
    let monitor = std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(15));
            let done = progress_clone.load(Ordering::Relaxed);
            if done >= total_matches {
                break;
            }
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed * 60.0;
            let eta = if rate > 0.0 {
                (total_matches - done) as f64 / rate * 60.0
            } else {
                0.0
            };
            eprint!(
                "\r  Progress: {}/{} matches ({:.0}%), {:.1}/min, ETA {:.0}s   ",
                done, total_matches,
                100.0 * done as f64 / total_matches as f64,
                rate, eta,
            );
        }
    });

    // Run all matchups
    for i in 0..n {
        for j in (i + 1)..n {
            let seed1 = seed.wrapping_add((i * 1000 + j * 100) as u64);
            let r1 = run_matchup(
                &agents[i], &agents[j], n_matches, time_ms,
                &bid_weights, dmc_weights.as_ref(), n_threads, seed1, &progress,
            );

            let seed2 = seed.wrapping_add((j * 1000 + i * 100) as u64);
            let r2 = run_matchup(
                &agents[j], &agents[i], n_matches, time_ms,
                &bid_weights, dmc_weights.as_ref(), n_threads, seed2, &progress,
            );

            // Aggregate both directions
            win_matrix[i][j] += r1.ns_wins + r2.ew_wins;
            win_matrix[j][i] += r1.ew_wins + r2.ns_wins;
            margin_matrix[i][j] += r1.total_margin - r2.total_margin;
            margin_matrix[j][i] += r2.total_margin - r1.total_margin;
            matches_matrix[i][j] += r1.n_matches + r2.n_matches;
            matches_matrix[j][i] += r1.n_matches + r2.n_matches;
        }
    }

    // Signal monitor to stop
    progress.store(total_matches, Ordering::Relaxed);
    let _ = monitor.join();
    let elapsed = start.elapsed();
    eprintln!();

    // --- Print results ---

    // Win matrix
    println!("=============================================================");
    println!("  WIN MATRIX (row win% vs column, both dirs combined)");
    println!("=============================================================");
    print!("  {:>8}", "");
    for a in &agents {
        print!("  {:>8}", a.name);
    }
    println!("   TOTAL");
    println!("  {}", "-".repeat(8 + 10 * (n + 1) + 4));

    let mut total_wins = vec![0u32; n];
    let mut total_played = vec![0u32; n];

    for i in 0..n {
        print!("  {:>8}", agents[i].name);
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
        print!("  {:5.1}%", total_pct);
        println!();
        total_wins[i] = row_wins;
        total_played[i] = row_played;
    }

    // Margin matrix
    println!();
    println!("=============================================================");
    println!("  MARGIN MATRIX (avg point margin, row vs column)");
    println!("=============================================================");
    print!("  {:>8}", "");
    for a in &agents {
        print!("  {:>8}", a.name);
    }
    println!();
    println!("  {}", "-".repeat(8 + 10 * n));

    for i in 0..n {
        print!("  {:>8}", agents[i].name);
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
        let a = &agents[*idx];
        let avg_margin: f64 = {
            let total_m: i64 = (0..n)
                .filter(|&j| j != *idx)
                .map(|j| margin_matrix[*idx][j])
                .sum();
            let total_p: u32 = (0..n)
                .filter(|&j| j != *idx)
                .map(|j| matches_matrix[*idx][j])
                .sum();
            total_m as f64 / total_p as f64
        };
        let bid_str = match a.bid {
            BidMethod::ImprovedV2 => "improved_v2",
            BidMethod::NnBid => "NN",
        };
        let play_str = match a.play {
            PlayMethod::SmartIsDd => "SmartDD",
            PlayMethod::DdOracle => "DD",
            PlayMethod::Dmc => "DMC",
        };
        println!(
            "  {:>2}. {:<8}  win {:5.1}%  margin {:+5.0}  [bid={}, play={}]",
            rank + 1, a.name, pct, avg_margin, bid_str, play_str,
        );
    }

    // --- Key head-to-head: NN vs V2 per play method ---
    println!();
    println!("=============================================================");
    println!("  NN BID vs IMPROVED_V2 — Head-to-Head per Play Method");
    println!("=============================================================");

    let h2h_pairs = [
        ("NN+SDD", "V2+SDD", "Smart IS-DD"),
        ("NN+DD", "V2+DD", "DD Oracle"),
        ("NN+DMC", "V2+DMC", "DMC (DouDou35)"),
    ];

    for (nn_name, v2_name, label) in &h2h_pairs {
        let nn_idx = agents.iter().position(|a| a.name == *nn_name);
        let v2_idx = agents.iter().position(|a| a.name == *v2_name);
        if let (Some(ni), Some(vi)) = (nn_idx, v2_idx) {
            let played = matches_matrix[ni][vi];
            if played == 0 {
                continue;
            }
            let nn_wins = win_matrix[ni][vi];
            let v2_wins = win_matrix[vi][ni];
            let nn_pct = 100.0 * nn_wins as f64 / played as f64;
            let v2_pct = 100.0 * v2_wins as f64 / played as f64;
            let avg_margin = margin_matrix[ni][vi] as f64 / played as f64;
            println!(
                "  {}:  NN {:5.1}% vs V2 {:5.1}%  margin {:+.0}  ({} matches)",
                label, nn_pct, v2_pct, avg_margin, played,
            );
        }
    }

    // --- Play method strength comparison ---
    println!();
    println!("=============================================================");
    println!("  PLAY METHOD STRENGTH (NN bid only)");
    println!("=============================================================");

    let nn_agents = [("NN+SDD", "SmartDD"), ("NN+DD", "DD"), ("NN+DMC", "DMC")];
    for (i, (a_name, _a_label)) in nn_agents.iter().enumerate() {
        for (b_name, _b_label) in nn_agents.iter().skip(i + 1) {
            let a_idx = agents.iter().position(|a| a.name == *a_name);
            let b_idx = agents.iter().position(|a| a.name == *b_name);
            if let (Some(ai), Some(bi)) = (a_idx, b_idx) {
                let played = matches_matrix[ai][bi];
                if played == 0 {
                    continue;
                }
                let a_wins = win_matrix[ai][bi];
                let b_wins = win_matrix[bi][ai];
                let a_pct = 100.0 * a_wins as f64 / played as f64;
                let avg_margin = margin_matrix[ai][bi] as f64 / played as f64;
                println!(
                    "  {} ({:5.1}%) vs {} ({:5.1}%)  margin {:+.0}",
                    a_name, a_pct, b_name, 100.0 * b_wins as f64 / played as f64, avg_margin,
                );
            }
        }
    }

    println!();
    println!(
        "  Wall: {:.1}s ({} matches, {:.1}/min)",
        elapsed.as_secs_f64(),
        total_matches,
        total_matches as f64 / elapsed.as_secs_f64() * 60.0,
    );
}
