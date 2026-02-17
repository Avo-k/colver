/// Maxi strength tournament: round-robin match play (first to 2000) comparing
/// Maxi bot against Heuristique, Random, DMC checkpoints, IS-MCTS, and Oracle.
///
/// Usage:
///   cargo run --bin maxi_tournament --release -- [matches] [time_ms] [--threads N]

use colver_core::bid_eval::BidFunction;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM};
use colver_core::mcts::{MctsConfig, MctsSearch, RolloutPolicy};
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const MATCH_TARGET: i32 = 2000;

// --- Agent definition ---

#[derive(Clone)]
enum CardPlayMethod {
    HeuristicPlay,
    MaxiPlay,
    Random,
    NaiveIsMcts,
    SmartIsMcts,
    Oracle,
    Dmc(usize),
}

#[derive(Clone)]
struct Agent {
    name: String,
    bid_function: BidFunction,
    card_play: CardPlayMethod,
    bid_desc: String,  // human-readable bidding strategy description
    play_desc: String, // human-readable play strategy description
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

// --- Match result ---

struct MatchResult {
    winner: u8,
    ns_final: i32,
    ew_final: i32,
}

// --- Match play ---

fn play_match(
    ns_agent: &Agent,
    ew_agent: &Agent,
    time_ms: u32,
    oracle_iters: u32,
    dmc_models: &[DmcWeights],
    rng: &mut StdRng,
) -> MatchResult {
    let naive_config = NaiveIsMctsConfig {
        iterations_per_det: 50,
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };
    let smart_config = SmartIsMctsConfig {
        iterations_per_det: 50,
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };
    let oracle_config = MctsConfig {
        iterations: oracle_iters,
        rollout_policy: RolloutPolicy::HeuristicPlay,
        ..Default::default()
    };

    let mut dmc_nets: Vec<Option<DmcNet>> = (0..dmc_models.len()).map(|_| None).collect();
    for agent in [ns_agent, ew_agent] {
        if let CardPlayMethod::Dmc(idx) = agent.card_play {
            if dmc_nets[idx].is_none() {
                dmc_nets[idx] = Some(dmc_models[idx].make_net());
            }
        }
    }

    let mut ns_cumulative: i32 = 0;
    let mut ew_cumulative: i32 = 0;
    let mut dealer: u8 = rng.gen_range(0..4);
    let mut obs_buf = vec![0.0f32; OBS_DIM];
    let mut oracle = MctsSearch::new();

    // IS-MCTS search objects
    let mut ns_naive = NaiveIsMctsSearch::new();
    let mut ew_naive = NaiveIsMctsSearch::new();

    let ns_is_smart = matches!(ns_agent.card_play, CardPlayMethod::SmartIsMcts);
    let ew_is_smart = matches!(ew_agent.card_play, CardPlayMethod::SmartIsMcts);

    while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);

        // Per-deal Smart IS-MCTS instances (need re-init per deal for beliefs)
        let mut ns_smart = [SmartIsMctsSearch::new(), SmartIsMctsSearch::new()];
        let mut ew_smart = [SmartIsMctsSearch::new(), SmartIsMctsSearch::new()];
        if ns_is_smart {
            ns_smart[0].init_deal(&state, 0, true);
            ns_smart[1].init_deal(&state, 2, true);
        }
        if ew_is_smart {
            ew_smart[0].init_deal(&state, 1, true);
            ew_smart[1].init_deal(&state, 3, true);
        }

        while !state.is_terminal() {
            let player = state.current_player();
            let is_ns = player == 0 || player == 2;
            let agent = if is_ns { ns_agent } else { ew_agent };
            let state_before = state;

            let action = if state.phase == Phase::Bidding {
                agent.bid_function.bid(&state)
            } else {
                match &agent.card_play {
                    CardPlayMethod::HeuristicPlay => {
                        rollout::heuristic_play_action(&state)
                    }
                    CardPlayMethod::MaxiPlay => {
                        colver_core::maxi::maxi_play_action(&state)
                    }
                    CardPlayMethod::Random => {
                        let legal = state.legal_actions();
                        let count = legal.count_ones();
                        let idx = rng.gen_range(0..count);
                        rollout::select_nth_bit(legal, idx)
                    }
                    CardPlayMethod::NaiveIsMcts => {
                        if is_ns {
                            ns_naive.search(&state, &naive_config, rng)
                        } else {
                            ew_naive.search(&state, &naive_config, rng)
                        }
                    }
                    CardPlayMethod::SmartIsMcts => {
                        if is_ns {
                            let idx = if player == 0 { 0 } else { 1 };
                            ns_smart[idx].search(&state, &smart_config, rng)
                        } else {
                            let idx = if player == 1 { 0 } else { 1 };
                            ew_smart[idx].search(&state, &smart_config, rng)
                        }
                    }
                    CardPlayMethod::Oracle => {
                        oracle.search(&state, &oracle_config, rng)
                    }
                    CardPlayMethod::Dmc(model_idx) => {
                        let net = dmc_nets[*model_idx].as_mut().unwrap();
                        dmc_obs::write_observation(&mut obs_buf, 0, &state, &tracking);
                        let legal_mask = state.legal_actions() as u32;
                        let (action, _) = net.best_action(&obs_buf, legal_mask);
                        action
                    }
                }
            };

            // Record action on Smart searches BEFORE stepping
            if ns_is_smart {
                ns_smart[0].record_action(&state_before, player, action);
                ns_smart[1].record_action(&state_before, player, action);
            }
            if ew_is_smart {
                ew_smart[0].record_action(&state_before, player, action);
                ew_smart[1].record_action(&state_before, player, action);
            }
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

    MatchResult { winner, ns_final: ns_cumulative, ew_final: ew_cumulative }
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
    oracle_iters: u32,
    dmc_models: &[DmcWeights],
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
                    let mr = play_match(ns_agent, ew_agent, time_ms, oracle_iters, dmc_models, &mut rng);
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
    for r in &results { combined.merge(r); }
    combined
}

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut n_matches: u32 = 100;
    let mut time_ms: u32 = 20;
    let mut n_threads = default_threads();
    let mut oracle_iters: u32 = 700;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => { i += 1; n_threads = args[i].parse().unwrap_or(default_threads()); }
            "--oracle-iters" => { i += 1; oracle_iters = args[i].parse().unwrap_or(700); }
            s => {
                if let Ok(v) = s.parse::<u32>() {
                    if i == 1 { n_matches = v; }
                    else if i == 2 { time_ms = v; }
                }
            }
        }
        i += 1;
    }

    // Load DMC models (only 21M and 40M)
    let dmc_paths = [
        ("models/dmc_21000000.bin", "DMC-21M"),
        ("models/dmc_40000000.bin", "DMC-40M"),
    ];

    let mut dmc_models: Vec<DmcWeights> = Vec::new();
    let mut dmc_labels: Vec<&str> = Vec::new();

    for (path, label) in &dmc_paths {
        match DmcWeights::load(path) {
            Ok(w) => {
                println!("  Loaded {} (obs={}, h={}, duel={})", label, w.obs_dim, w.hidden, w.dueling);
                dmc_models.push(w);
                dmc_labels.push(label);
            }
            Err(e) => {
                eprintln!("  Warning: {} not found: {}", path, e);
            }
        }
    }

    // Build agent list
    let mut agents: Vec<Agent> = vec![
        Agent {
            name: "Maxi".into(),
            bid_function: BidFunction::Maxi,
            card_play: CardPlayMethod::MaxiPlay,
            bid_desc: "Maxi (case A-D, response, competitive)".into(),
            play_desc: "Maxi (convention-linked leads)".into(),
        },
        Agent {
            name: "Heuristiq".into(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::HeuristicPlay,
            bid_desc: "ImprovedV2 (tournament-tuned, QG, caps)".into(),
            play_desc: "Heuristic (safe leads, min-win, cheapest)".into(),
        },
        Agent {
            name: "Random".into(),
            bid_function: BidFunction::Heuristic,
            card_play: CardPlayMethod::Random,
            bid_desc: "Heuristic (score-based, no QG)".into(),
            play_desc: "Random (uniform legal)".into(),
        },
        Agent {
            name: "Naive".into(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::NaiveIsMcts,
            bid_desc: "ImprovedV2".into(),
            play_desc: format!("Naive IS-MCTS ({}ms)", time_ms),
        },
        Agent {
            name: "Smart".into(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::SmartIsMcts,
            bid_desc: "ImprovedV2".into(),
            play_desc: format!("Smart IS-MCTS ({}ms, beliefs)", time_ms),
        },
        Agent {
            name: "Oracle".into(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::Oracle,
            bid_desc: "ImprovedV2".into(),
            play_desc: format!("Oracle MCTS ({}it, perfect info)", oracle_iters),
        },
    ];

    // Add DMC agents
    for (idx, label) in dmc_labels.iter().enumerate() {
        agents.push(Agent {
            name: label.to_string(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::Dmc(idx),
            bid_desc: "ImprovedV2".into(),
            play_desc: format!("DMC Q-net (dueling, 1024h)"),
        });
    }

    let n = agents.len();
    let total_matchups = n * (n - 1);
    let total_matches = total_matchups as u32 * n_matches;

    println!();
    println!("================================================================");
    println!("  MAXI STRENGTH TOURNAMENT — Round Robin, First to {}", MATCH_TARGET);
    println!("  {} agents, {} matches/matchup (both dirs)", n, n_matches);
    println!("  {} matchups, {} total matches", total_matchups, total_matches);
    println!("  {}ms/move, {} oracle iters, {} threads", time_ms, oracle_iters, n_threads);
    println!("================================================================");
    println!();
    println!("  {:<10} {:<15} {:<45} {}", "Name", "Play", "Bidding", "Play Detail");
    println!("  {}", "-".repeat(110));
    for a in &agents {
        let play_short = match &a.card_play {
            CardPlayMethod::HeuristicPlay => "Heuristic",
            CardPlayMethod::MaxiPlay => "Maxi",
            CardPlayMethod::Random => "Random",
            CardPlayMethod::NaiveIsMcts => "Naive IS-MCTS",
            CardPlayMethod::SmartIsMcts => "Smart IS-MCTS",
            CardPlayMethod::Oracle => "Oracle MCTS",
            CardPlayMethod::Dmc(_) => "DMC Q-net",
        };
        println!("  {:<10} {:<15} {:<45} {}", a.name, play_short, a.bid_desc, a.play_desc);
    }
    println!();

    let progress = Arc::new(AtomicU32::new(0));
    let start = Instant::now();

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
            if done >= total_matches { break; }
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed * 60.0;
            let eta = if rate > 0.0 { (total_matches - done) as f64 / rate * 60.0 } else { 0.0 };
            eprint!("\r  Progress: {}/{} ({:.0}%), {:.0}/min, ETA {:.0}s   ", done, total_matches,
                100.0 * done as f64 / total_matches as f64, rate, eta);
        }
    });

    // Run all matchups
    for i in 0..n {
        for j in (i + 1)..n {
            let seed1 = (i * 1000 + j * 100 + 42) as u64;
            let r1 = run_matchup(&agents[i], &agents[j], n_matches, time_ms, oracle_iters,
                &dmc_models, n_threads, seed1, &progress);

            let seed2 = (j * 1000 + i * 100 + 42) as u64;
            let r2 = run_matchup(&agents[j], &agents[i], n_matches, time_ms, oracle_iters,
                &dmc_models, n_threads, seed2, &progress);

            win_matrix[i][j] += r1.ns_wins + r2.ew_wins;
            win_matrix[j][i] += r1.ew_wins + r2.ns_wins;
            margin_matrix[i][j] += r1.total_margin - r2.total_margin;
            margin_matrix[j][i] += r2.total_margin - r1.total_margin;
            matches_matrix[i][j] += r1.n_matches + r2.n_matches;
            matches_matrix[j][i] += r1.n_matches + r2.n_matches;
        }
    }

    progress.store(total_matches, Ordering::Relaxed);
    let _ = monitor.join();
    let elapsed = start.elapsed();
    eprintln!();

    // --- Rankings table ---

    println!();
    println!("================================================================");
    println!("  RANKINGS");
    println!("================================================================");

    let mut ranking: Vec<(usize, f64)> = (0..n)
        .map(|i| {
            let w: u32 = (0..n).filter(|&j| j != i).map(|j| win_matrix[i][j]).sum();
            let p: u32 = (0..n).filter(|&j| j != i).map(|j| matches_matrix[i][j]).sum();
            (i, 100.0 * w as f64 / p as f64)
        })
        .collect();
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    println!();
    println!("  {:<4} {:<10} {:>7} {:>8}  {:<15} {:<45}", "#", "Name", "Win%", "Margin", "Play", "Bidding");
    println!("  {}", "-".repeat(100));

    for (rank, (idx, pct)) in ranking.iter().enumerate() {
        let a = &agents[*idx];
        let avg_margin: f64 = {
            let total_m: i64 = (0..n).filter(|&j| j != *idx).map(|j| margin_matrix[*idx][j]).sum();
            let total_p: u32 = (0..n).filter(|&j| j != *idx).map(|j| matches_matrix[*idx][j]).sum();
            total_m as f64 / total_p as f64
        };
        let play_short = match &a.card_play {
            CardPlayMethod::HeuristicPlay => "Heuristic",
            CardPlayMethod::MaxiPlay => "Maxi",
            CardPlayMethod::Random => "Random",
            CardPlayMethod::NaiveIsMcts => "Naive IS-MCTS",
            CardPlayMethod::SmartIsMcts => "Smart IS-MCTS",
            CardPlayMethod::Oracle => "Oracle MCTS",
            CardPlayMethod::Dmc(_) => "DMC Q-net",
        };
        println!(
            "  {:<4} {:<10} {:>5.1}% {:>+7.0}  {:<15} {}",
            format!("{}.", rank + 1), a.name, pct, avg_margin, play_short, a.bid_desc,
        );
    }

    // --- Win matrix ---

    println!();
    println!("================================================================");
    println!("  WIN MATRIX (row win% vs column, both dirs combined)");
    println!("================================================================");

    // Print in ranking order
    let ranked_indices: Vec<usize> = ranking.iter().map(|(idx, _)| *idx).collect();

    print!("  {:>10}", "");
    for &ri in &ranked_indices { print!("  {:>8}", &agents[ri].name[..agents[ri].name.len().min(8)]); }
    println!("  |  TOTAL");
    println!("  {}", "-".repeat(10 + 10 * n + 12));

    for &ri in &ranked_indices {
        print!("  {:>10}", agents[ri].name);
        let mut row_wins = 0u32;
        let mut row_played = 0u32;
        for &rj in &ranked_indices {
            if ri == rj {
                print!("      -  ");
            } else {
                let wins = win_matrix[ri][rj];
                let played = matches_matrix[ri][rj];
                let pct = 100.0 * wins as f64 / played as f64;
                print!("   {:5.1}% ", pct);
                row_wins += wins;
                row_played += played;
            }
        }
        let total_pct = 100.0 * row_wins as f64 / row_played as f64;
        print!("  | {:5.1}%", total_pct);
        println!();
    }

    // --- Margin matrix ---

    println!();
    println!("================================================================");
    println!("  MARGIN MATRIX (avg point margin, row vs column)");
    println!("================================================================");
    print!("  {:>10}", "");
    for &ri in &ranked_indices { print!("  {:>8}", &agents[ri].name[..agents[ri].name.len().min(8)]); }
    println!();
    println!("  {}", "-".repeat(10 + 10 * n));

    for &ri in &ranked_indices {
        print!("  {:>10}", agents[ri].name);
        for &rj in &ranked_indices {
            if ri == rj {
                print!("      -  ");
            } else {
                let played = matches_matrix[ri][rj];
                let avg_margin = margin_matrix[ri][rj] as f64 / played as f64;
                print!("   {:+5.0}  ", avg_margin);
            }
        }
        println!();
    }

    // --- Big cross table: win% (margin) per matchup ---

    println!();
    println!("================================================================");
    println!("  FULL CROSS TABLE: win% [margin] per matchup");
    println!("================================================================");
    println!();

    // Header row
    print!("  {:>10} |", "");
    for &ri in &ranked_indices {
        print!(" {:^15} |", &agents[ri].name);
    }
    println!();
    print!("  {:-<11}+", "");
    for _ in 0..n { print!("{:-<17}+", ""); }
    println!();

    for &ri in &ranked_indices {
        print!("  {:>10} |", agents[ri].name);
        for &rj in &ranked_indices {
            if ri == rj {
                print!("       ---       |");
            } else {
                let wins = win_matrix[ri][rj];
                let played = matches_matrix[ri][rj];
                let pct = 100.0 * wins as f64 / played as f64;
                let avg_margin = margin_matrix[ri][rj] as f64 / played as f64;
                print!(" {:5.1}% [{:+5.0}] |", pct, avg_margin);
            }
        }
        println!();
    }

    println!();
    println!(
        "  Wall: {:.1}s ({} matches, {:.1}/min)",
        elapsed.as_secs_f64(), total_matches,
        total_matches as f64 / elapsed.as_secs_f64() * 60.0,
    );
}
