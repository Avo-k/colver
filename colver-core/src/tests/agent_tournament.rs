/// Agent tournament: round-robin match play (first to 2000) comparing different
/// agent configurations (bidding strategy + card play method).
///
/// Usage:
///   cargo run --bin agent_tournament --release -- [matches] [time_ms] [--threads N] [--dmc]
///   cargo run --bin agent_tournament --release -- 50 20 --threads 8 --dmc

use colver_core::bid_eval::BidFunction;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{EnvTracking, OBS_DIM};
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::mcts::{MctsConfig, MctsSearch, RolloutPolicy};
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
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
    NaiveIsMcts,
    SmartIsMcts,
    Oracle,
    Dmc(usize), // index into shared weights vec
    IsDd,       // IS-DD (naive, no beliefs)
    SmartIsDd,  // IS-DD with belief-weighted determinization
}

#[derive(Clone)]
struct Agent {
    name: String,
    bid_function: BidFunction,
    card_play: CardPlayMethod,
}

/// Shared DMC weights (read-only after init, one per model file).
/// Each thread creates its own DmcNet from these via DmcNet::from_floats().
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

        // Re-read raw floats for thread-local construction
        let data = std::fs::read(path)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();

        Ok(DmcWeights {
            floats,
            hidden,
            obs_dim,
            dueling,
        })
    }

    fn make_net(&self) -> DmcNet {
        DmcNet::from_floats(&self.floats, self.hidden, self.obs_dim, self.dueling).unwrap()
    }
}

// --- Match result ---

struct MatchResult {
    winner: u8, // 0=NS, 1=EW
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
    let dd_config = IsDdConfig {
        determinizations: 20,
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };

    // Pre-create thread-local DmcNet instances for each DMC model used
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

    while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);

        // Per-deal search objects
        let mut ns_naive = NaiveIsMctsSearch::new();
        let mut ew_naive = NaiveIsMctsSearch::new();
        let mut ns_smart = [SmartIsMctsSearch::new(), SmartIsMctsSearch::new()]; // p0, p2
        let mut ew_smart = [SmartIsMctsSearch::new(), SmartIsMctsSearch::new()]; // p1, p3
        let mut ns_dd = IsDdSearch::new();
        let mut ew_dd = IsDdSearch::new();
        let mut ns_smart_dd = [IsDdSearch::new(), IsDdSearch::new()]; // p0, p2
        let mut ew_smart_dd = [IsDdSearch::new(), IsDdSearch::new()]; // p1, p3
        let mut oracle = MctsSearch::new();
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);

        // Init Smart IS-MCTS if needed
        let ns_is_smart = matches!(ns_agent.card_play, CardPlayMethod::SmartIsMcts);
        let ew_is_smart = matches!(ew_agent.card_play, CardPlayMethod::SmartIsMcts);
        if ns_is_smart {
            ns_smart[0].init_deal(&state, 0, true);
            ns_smart[1].init_deal(&state, 2, true);
        }
        if ew_is_smart {
            ew_smart[0].init_deal(&state, 1, true);
            ew_smart[1].init_deal(&state, 3, true);
        }
        // Init Smart IS-DD if needed
        let ns_is_smart_dd = matches!(ns_agent.card_play, CardPlayMethod::SmartIsDd);
        let ew_is_smart_dd = matches!(ew_agent.card_play, CardPlayMethod::SmartIsDd);
        if ns_is_smart_dd {
            ns_smart_dd[0].init_deal(&state, 0, true);
            ns_smart_dd[1].init_deal(&state, 2, true);
        }
        if ew_is_smart_dd {
            ew_smart_dd[0].init_deal(&state, 1, true);
            ew_smart_dd[1].init_deal(&state, 3, true);
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
                    CardPlayMethod::IsDd => {
                        if is_ns {
                            ns_dd.search(&state, &dd_config, rng)
                        } else {
                            ew_dd.search(&state, &dd_config, rng)
                        }
                    }
                    CardPlayMethod::SmartIsDd => {
                        if is_ns {
                            let idx = if player == 0 { 0 } else { 1 };
                            ns_smart_dd[idx].search(&state, &dd_config, rng)
                        } else {
                            let idx = if player == 1 { 0 } else { 1 };
                            ew_smart_dd[idx].search(&state, &dd_config, rng)
                        }
                    }
                }
            };

            // Record action on all Smart searches BEFORE stepping
            if ns_is_smart {
                ns_smart[0].record_action(&state_before, player, action);
                ns_smart[1].record_action(&state_before, player, action);
            }
            if ew_is_smart {
                ew_smart[0].record_action(&state_before, player, action);
                ew_smart[1].record_action(&state_before, player, action);
            }
            if ns_is_smart_dd {
                ns_smart_dd[0].record_action(&state_before, player, action);
                ns_smart_dd[1].record_action(&state_before, player, action);
            }
            if ew_is_smart_dd {
                ew_smart_dd[0].record_action(&state_before, player, action);
                ew_smart_dd[1].record_action(&state_before, player, action);
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

// Need this import for write_observation
use colver_core::dmc_obs;

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
            if start >= end {
                continue;
            }
            let count = end - start;

            handles.push(s.spawn(move || {
                let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(t as u64 * 7919));
                let mut result = MatchupResult::default();
                for _ in 0..count {
                    let mr = play_match(
                        ns_agent, ew_agent, time_ms, oracle_iters, dmc_models, &mut rng,
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

    // Parse positional + flags
    let mut n_matches: u32 = 50;
    let mut time_ms: u32 = 20;
    let mut n_threads = default_threads();
    let mut use_dmc = false;
    let mut use_dd_only = false;
    let mut oracle_iters: u32 = 2000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--threads" => {
                i += 1;
                n_threads = args[i].parse().unwrap_or(default_threads());
            }
            "--dmc" => {
                use_dmc = true;
            }
            "--dd" => {
                use_dd_only = true;
            }
            "--oracle-iters" => {
                i += 1;
                oracle_iters = args[i].parse().unwrap_or(2000);
            }
            s => {
                if let Ok(v) = s.parse::<u32>() {
                    if n_matches == 50 && i == 1 {
                        n_matches = v;
                    } else if time_ms == 20 && i == 2 {
                        time_ms = v;
                    }
                }
            }
        }
        i += 1;
    }

    // Load DMC models
    let mut dmc_models: Vec<DmcWeights> = Vec::new();
    let dmc_paths: Vec<(&str, &str)> = if use_dd_only {
        // For --dd mode, load best available model
        vec![("models/dmc_35.bin", "DouDou35")]
    } else {
        vec![
            ("models/dmc_2000000.bin", "DMC2M"),
            ("models/dmc_6000000.bin", "DMC6M"),
        ]
    };

    if use_dmc || use_dd_only {
        for (path, label) in &dmc_paths {
            match DmcWeights::load(path) {
                Ok(w) => {
                    println!("  Loaded {} (obs_dim={}, hidden={}, dueling={})",
                        label, w.obs_dim, w.hidden, w.dueling);
                    dmc_models.push(w);
                }
                Err(e) => {
                    eprintln!("  Warning: could not load {}: {}", path, e);
                }
            }
        }
    }

    // Build agent list
    let mut agents: Vec<Agent> = vec![
        Agent {
            name: "Naive+Heur".into(),
            bid_function: BidFunction::Heuristic,
            card_play: CardPlayMethod::NaiveIsMcts,
        },
        Agent {
            name: "Naive+Impr".into(),
            bid_function: BidFunction::Improved,
            card_play: CardPlayMethod::NaiveIsMcts,
        },
        Agent {
            name: "Naive+ImV2".into(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::NaiveIsMcts,
        },
        Agent {
            name: "Naive+Roro".into(),
            bid_function: BidFunction::Roro,
            card_play: CardPlayMethod::NaiveIsMcts,
        },
        Agent {
            name: "Smart+Impr".into(),
            bid_function: BidFunction::Improved,
            card_play: CardPlayMethod::SmartIsMcts,
        },
        Agent {
            name: "Smart+ImV2".into(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::SmartIsMcts,
        },
        Agent {
            name: "Smart+Roro".into(),
            bid_function: BidFunction::Roro,
            card_play: CardPlayMethod::SmartIsMcts,
        },
        Agent {
            name: "Naive+PBid".into(),
            bid_function: BidFunction::PetitBide,
            card_play: CardPlayMethod::NaiveIsMcts,
        },
        Agent {
            name: "Naive+Moel".into(),
            bid_function: BidFunction::Moelleux,
            card_play: CardPlayMethod::NaiveIsMcts,
        },
        Agent {
            name: "Smart+PBid".into(),
            bid_function: BidFunction::PetitBide,
            card_play: CardPlayMethod::SmartIsMcts,
        },
        Agent {
            name: "Smart+Moel".into(),
            bid_function: BidFunction::Moelleux,
            card_play: CardPlayMethod::SmartIsMcts,
        },
        Agent {
            name: "Orcl+Impr".into(),
            bid_function: BidFunction::Improved,
            card_play: CardPlayMethod::Oracle,
        },
        Agent {
            name: "Orcl+ImV2".into(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::Oracle,
        },
        Agent {
            name: "Orcl+Roro".into(),
            bid_function: BidFunction::Roro,
            card_play: CardPlayMethod::Oracle,
        },
        Agent {
            name: "IsDd+ImV2".into(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::IsDd,
        },
        Agent {
            name: "SDD+ImV2".into(),
            bid_function: BidFunction::ImprovedV2,
            card_play: CardPlayMethod::SmartIsDd,
        },
    ];

    // If --dd flag, keep only DD-focused agents for quick comparison
    if use_dd_only {
        agents.retain(|a| matches!(a.card_play,
            CardPlayMethod::SmartIsMcts | CardPlayMethod::NaiveIsMcts |
            CardPlayMethod::IsDd | CardPlayMethod::SmartIsDd
        ) && matches!(a.bid_function, BidFunction::ImprovedV2));
    }

    // Add DMC agents if models loaded
    if use_dd_only {
        if !dmc_models.is_empty() {
            agents.push(Agent {
                name: "DMC+ImV2".into(),
                bid_function: BidFunction::ImprovedV2,
                card_play: CardPlayMethod::Dmc(0),
            });
        }
    } else {
        if dmc_models.len() >= 1 {
            agents.push(Agent {
                name: "DMC2M+Impr".into(),
                bid_function: BidFunction::Improved,
                card_play: CardPlayMethod::Dmc(0),
            });
            agents.push(Agent {
                name: "DMC2M+Roro".into(),
                bid_function: BidFunction::Roro,
                card_play: CardPlayMethod::Dmc(0),
            });
        }
        if dmc_models.len() >= 2 {
            agents.push(Agent {
                name: "DMC6M+Impr".into(),
                bid_function: BidFunction::Improved,
                card_play: CardPlayMethod::Dmc(1),
            });
            agents.push(Agent {
                name: "DMC6M+Roro".into(),
                bid_function: BidFunction::Roro,
                card_play: CardPlayMethod::Dmc(1),
            });
        }
    }

    let n = agents.len();
    let total_matchups = n * (n - 1);
    let total_matches = total_matchups as u32 * n_matches;

    println!("=============================================================");
    println!("  AGENT TOURNAMENT — Round Robin, First to {}", MATCH_TARGET);
    println!("  {} agents, {} matches/matchup (both dirs)", n, n_matches);
    println!("  {} matchups, {} total matches", total_matchups, total_matches);
    println!("  {}ms/move, {} oracle iters, {} threads", time_ms, oracle_iters, n_threads);
    if use_dmc {
        println!("  DMC models: {}", dmc_models.len());
    }
    println!("=============================================================");
    println!();

    for (i, a) in agents.iter().enumerate() {
        let play_str = match &a.card_play {
            CardPlayMethod::NaiveIsMcts => "Naive IS-MCTS".to_string(),
            CardPlayMethod::SmartIsMcts => "Smart IS-MCTS".to_string(),
            CardPlayMethod::Oracle => format!("Oracle ({}it)", oracle_iters),
            CardPlayMethod::Dmc(idx) => format!("DMC ({})", dmc_paths.get(*idx).map(|p| p.1).unwrap_or("?")),
            CardPlayMethod::IsDd => "IS-DD (naive)".to_string(),
            CardPlayMethod::SmartIsDd => "Smart IS-DD".to_string(),
        };
        println!("  [{:>2}] {:<12}  bid={:<10}  play={}", i, a.name, format!("{:?}", a.bid_function), play_str);
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
            std::thread::sleep(std::time::Duration::from_secs(10));
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
                "\r  Progress: {}/{} matches ({:.0}%), {:.0}/min, ETA {:.0}s   ",
                done,
                total_matches,
                100.0 * done as f64 / total_matches as f64,
                rate,
                eta,
            );
        }
    });

    // Run all matchups sequentially (each matchup parallelizes internally)
    for i in 0..n {
        for j in (i + 1)..n {
            // Direction 1: i as NS, j as EW
            let seed1 = (i * 1000 + j * 100 + 42) as u64;
            let r1 = run_matchup(
                &agents[i],
                &agents[j],
                n_matches,
                time_ms,
                oracle_iters,
                &dmc_models,
                n_threads,
                seed1,
                &progress,
            );

            // Direction 2: j as NS, i as EW
            let seed2 = (j * 1000 + i * 100 + 42) as u64;
            let r2 = run_matchup(
                &agents[j],
                &agents[i],
                n_matches,
                time_ms,
                oracle_iters,
                &dmc_models,
                n_threads,
                seed2,
                &progress,
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

    println!("=============================================================");
    println!("  WIN MATRIX (row player win% vs column, both dirs combined)");
    println!("=============================================================");
    print!("  {:>12}", "");
    for a in &agents {
        print!("  {:>8}", &a.name[..a.name.len().min(8)]);
    }
    println!("    TOTAL");
    println!("  {}", "-".repeat(12 + 10 * (n + 1) + 8));

    let mut total_wins = vec![0u32; n];
    let mut total_played = vec![0u32; n];

    for i in 0..n {
        print!("  {:>12}", agents[i].name);
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
    println!("  MARGIN MATRIX (avg point margin for row vs column)");
    println!("=============================================================");
    print!("  {:>12}", "");
    for a in &agents {
        print!("  {:>8}", &a.name[..a.name.len().min(8)]);
    }
    println!();
    println!("  {}", "-".repeat(12 + 10 * n));

    for i in 0..n {
        print!("  {:>12}", agents[i].name);
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
        println!(
            "  {:>2}. {:<12}  win {:5.1}%  avg margin {:+5.0}  [bid={:?}, play={:?}]",
            rank + 1,
            a.name,
            pct,
            avg_margin,
            a.bid_function,
            match &a.card_play {
                CardPlayMethod::NaiveIsMcts => "Naive",
                CardPlayMethod::SmartIsMcts => "Smart",
                CardPlayMethod::Oracle => "Oracle",
                CardPlayMethod::Dmc(_) => "DMC",
                CardPlayMethod::IsDd => "IS-DD",
                CardPlayMethod::SmartIsDd => "SmartDD",
            },
        );
    }

    // Head-to-head comparison: same play method, different bidding
    println!();
    println!("=============================================================");
    println!("  BIDDING STRATEGY HEAD-TO-HEAD (same play method)");
    println!("=============================================================");

    let h2h_pairs: Vec<(&str, &str, &str)> = vec![
        ("Naive+Heur", "Naive+Impr", "Naive"),
        ("Naive+Impr", "Naive+ImV2", "Naive"),
        ("Naive+Impr", "Naive+Roro", "Naive"),
        ("Naive+Impr", "Naive+PBid", "Naive"),
        ("Naive+Impr", "Naive+Moel", "Naive"),
        ("Naive+ImV2", "Naive+Roro", "Naive"),
        ("Naive+Roro", "Naive+PBid", "Naive"),
        ("Naive+Roro", "Naive+Moel", "Naive"),
        ("Smart+Impr", "Smart+ImV2", "Smart"),
        ("Smart+Impr", "Smart+Roro", "Smart"),
        ("Smart+Impr", "Smart+PBid", "Smart"),
        ("Smart+Impr", "Smart+Moel", "Smart"),
        ("Smart+ImV2", "Smart+Roro", "Smart"),
        ("Smart+Roro", "Smart+PBid", "Smart"),
        ("Smart+Roro", "Smart+Moel", "Smart"),
    ];

    for (a_name, b_name, label) in &h2h_pairs {
        let a_idx = agents.iter().position(|a| a.name == *a_name);
        let b_idx = agents.iter().position(|a| a.name == *b_name);
        if let (Some(a_idx), Some(b_idx)) = (a_idx, b_idx) {
            let played = matches_matrix[a_idx][b_idx];
            if played == 0 {
                continue;
            }
            let a_wins = win_matrix[a_idx][b_idx];
            let b_wins = win_matrix[b_idx][a_idx];
            let a_pct = 100.0 * a_wins as f64 / played as f64;
            let avg_margin = margin_matrix[a_idx][b_idx] as f64 / played as f64;
            println!(
                "  {} ({:>5.1}%) vs {} ({:>5.1}%)  margin {:+.0}  [{}]",
                agents[a_idx].name,
                a_pct,
                agents[b_idx].name,
                100.0 * b_wins as f64 / played as f64,
                avg_margin,
                label,
            );
        }
    }

    // DMC head-to-head if present
    if dmc_models.len() >= 1 {
        println!();
        println!("=============================================================");
        println!("  DMC HEAD-TO-HEAD");
        println!("=============================================================");
        // Find DMC agent indices
        let dmc_agents: Vec<usize> = (0..n)
            .filter(|&i| matches!(agents[i].card_play, CardPlayMethod::Dmc(_)))
            .collect();
        for &a_idx in &dmc_agents {
            for &b_idx in &dmc_agents {
                if a_idx >= b_idx {
                    continue;
                }
                let played = matches_matrix[a_idx][b_idx];
                if played == 0 {
                    continue;
                }
                let a_wins = win_matrix[a_idx][b_idx];
                let a_pct = 100.0 * a_wins as f64 / played as f64;
                let avg_margin = margin_matrix[a_idx][b_idx] as f64 / played as f64;
                println!(
                    "  {} ({:>5.1}%) vs {} ({:>5.1}%)  margin {:+.0}",
                    agents[a_idx].name,
                    a_pct,
                    agents[b_idx].name,
                    100.0 - a_pct,
                    avg_margin,
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
