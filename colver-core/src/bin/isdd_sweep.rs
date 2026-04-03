/// IS-DD Parameter Sweep Experiment.
///
/// Measures how IS-DD strength varies with different configurations:
///   1. Count-based sweep: vary determinizations (1, 2, 4, 8, 16, 32, 64)
///   2. Time-based sweep: vary time budget (5, 10, 20, 50ms)
///   3. Soft inference comparison: hard-only vs hard+soft for D=8, D=16
///
/// Each config is measured against two baselines per deal:
///   - vs Random: IS-DD as NS, random card play as EW
///   - vs DouDou35 (DMC): IS-DD as NS, DMC model as EW
///
/// Bidding: BidFunction::ImprovedV2 for all 4 players in all configs.
///
/// Usage:
///   cargo run -p colver-core --bin isdd_sweep --release -- \
///     --dmc-model models/dmc_35.bin --deals 200 --threads 8

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_eval::BidFunction;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM};
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::state::{GameState, Phase};

// --- Shared DMC weights (read-only after init, each thread makes its own net) ---

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

// --- Sweep configuration ---

struct SweepConfig {
    label: String,
    dd_config: IsDdConfig,
}

// --- Accumulator (per-thread, then merged) ---

#[derive(Default, Clone)]
struct SweepAccum {
    // vs Random baseline
    rand_ns_pts: i64,
    rand_ew_pts: i64,
    rand_ns_wins: u32,
    rand_deals: u32,
    // vs DMC baseline
    dmc_ns_pts: i64,
    dmc_ew_pts: i64,
    dmc_ns_wins: u32,
    dmc_deals: u32,
    // IS-DD timing (from vs-random run only, to avoid double-counting)
    isdd_ms: f64,
    isdd_calls: u64,
    isdd_dets: u64,
}

impl SweepAccum {
    fn merge(&mut self, other: &SweepAccum) {
        self.rand_ns_pts += other.rand_ns_pts;
        self.rand_ew_pts += other.rand_ew_pts;
        self.rand_ns_wins += other.rand_ns_wins;
        self.rand_deals += other.rand_deals;
        self.dmc_ns_pts += other.dmc_ns_pts;
        self.dmc_ew_pts += other.dmc_ew_pts;
        self.dmc_ns_wins += other.dmc_ns_wins;
        self.dmc_deals += other.dmc_deals;
        self.isdd_ms += other.isdd_ms;
        self.isdd_calls += other.isdd_calls;
        self.isdd_dets += other.isdd_dets;
    }
}

// --- Final result ---

struct SweepResult {
    label: String,
    rand_win_pct: f64,
    rand_avg_ns: f64,
    rand_avg_ew: f64,
    dmc_win_pct: f64,
    dmc_avg_ns: f64,
    dmc_avg_ew: f64,
    ms_per_deal: f64,
    avg_dets: f64,
}

impl SweepResult {
    fn from_accum(label: String, a: &SweepAccum) -> Self {
        let rand_deals = a.rand_deals.max(1) as f64;
        let dmc_deals = a.dmc_deals.max(1) as f64;
        SweepResult {
            label,
            rand_win_pct: 100.0 * a.rand_ns_wins as f64 / rand_deals,
            rand_avg_ns: a.rand_ns_pts as f64 / rand_deals,
            rand_avg_ew: a.rand_ew_pts as f64 / rand_deals,
            dmc_win_pct: 100.0 * a.dmc_ns_wins as f64 / dmc_deals,
            dmc_avg_ns: a.dmc_ns_pts as f64 / dmc_deals,
            dmc_avg_ew: a.dmc_ew_pts as f64 / dmc_deals,
            ms_per_deal: a.isdd_ms / rand_deals,
            avg_dets: if a.isdd_calls > 0 {
                a.isdd_dets as f64 / a.isdd_calls as f64
            } else {
                0.0
            },
        }
    }
}

// --- Helpers ---

fn random_action(legal: u64, rng: &mut impl Rng) -> u8 {
    let count = legal.count_ones();
    let pick = rng.gen_range(0..count);
    let mut mask = legal;
    for _ in 0..pick {
        mask &= mask - 1;
    }
    mask.trailing_zeros() as u8
}

// --- Core per-deal function ---

/// Play one deal twice: once vs random, once vs DMC.
/// Bidding is run once (deterministic, same for both runs).
/// Returns None if the deal is void (all passed).
fn play_deal(
    initial_state: &GameState,
    dd_config: &IsDdConfig,
    dmc_net: &mut DmcNet,
    obs_buf: &mut Vec<f32>,
    rng: &mut StdRng,
) -> Option<SweepAccum> {
    let use_soft = dd_config.use_soft_inference;

    // Initialize IS-DD searches for NS players (p0=N, p2=S).
    // Two independent sets: one for vs-random, one for vs-DMC.
    let mut ns_r = [IsDdSearch::new(), IsDdSearch::new()]; // vs random
    let mut ns_d = [IsDdSearch::new(), IsDdSearch::new()]; // vs DMC
    ns_r[0].init_deal(initial_state, 0, use_soft);
    ns_r[1].init_deal(initial_state, 2, use_soft);
    ns_d[0].init_deal(initial_state, 0, use_soft);
    ns_d[1].init_deal(initial_state, 2, use_soft);

    // --- Bidding phase (same for both runs, record on both IS-DD sets) ---
    let mut bids: Vec<(u8, u8, GameState)> = Vec::new(); // (player, action, state_before)
    let mut state = *initial_state;
    while state.phase == Phase::Bidding {
        let player = state.current_player();
        let state_before = state;
        let action = dd_config.bid_function.bid(&state);
        ns_r[0].record_action(&state_before, player, action);
        ns_r[1].record_action(&state_before, player, action);
        ns_d[0].record_action(&state_before, player, action);
        ns_d[1].record_action(&state_before, player, action);
        bids.push((player, action, state_before));
        state.step(action);
    }

    // Void deal: all players passed
    if state.is_terminal() {
        return None;
    }

    let post_bid_state = state;
    let mut accum = SweepAccum::default();

    // --- Run 1: IS-DD (NS) vs Random (EW), with IS-DD timing ---
    {
        let mut s = post_bid_state;

        while !s.is_terminal() {
            let player = s.current_player();
            let is_ns = player == 0 || player == 2;
            let state_before = s;

            let action = if is_ns {
                let idx = if player == 0 { 0 } else { 1 };
                let t0 = Instant::now();
                let result = ns_r[idx].search_with_stats(&s, dd_config, rng);
                accum.isdd_ms += t0.elapsed().as_secs_f64() * 1000.0;
                accum.isdd_calls += 1;
                accum.isdd_dets += result.determinizations as u64;
                result.best_action
            } else {
                random_action(s.legal_actions(), rng)
            };

            // Record on both NS IS-DD searches (belief tracking requires seeing all actions)
            ns_r[0].record_action(&state_before, player, action);
            ns_r[1].record_action(&state_before, player, action);

            s.step(action);
        }

        let score = s.deal_score();
        accum.rand_ns_pts += score.scores[0] as i64;
        accum.rand_ew_pts += score.scores[1] as i64;
        if score.scores[0] > score.scores[1] {
            accum.rand_ns_wins += 1;
        }
        accum.rand_deals += 1;
    }

    // --- Run 2: IS-DD (NS) vs DMC (EW), with DMC obs tracking ---
    {
        let mut s = post_bid_state;
        let mut tracking = EnvTracking::new();
        tracking.reset(initial_state.dealer);

        // Replay bidding actions into tracking so DMC obs has correct bid history
        for (_player, action, state_before) in &bids {
            tracking.track_action(state_before, *action);
        }

        while !s.is_terminal() {
            let player = s.current_player();
            let is_ns = player == 0 || player == 2;
            let state_before = s;

            let action = if is_ns {
                let idx = if player == 0 { 0 } else { 1 };
                ns_d[idx].search(&s, dd_config, rng)
            } else {
                // DMC card play
                dmc_obs::write_observation(obs_buf, 0, &s, &tracking);
                let legal_mask = s.legal_actions() as u32;
                let (a, _) = dmc_net.best_action(obs_buf, legal_mask);
                a
            };

            ns_d[0].record_action(&state_before, player, action);
            ns_d[1].record_action(&state_before, player, action);
            tracking.track_action(&state_before, action);

            s.step(action);
        }

        let score = s.deal_score();
        accum.dmc_ns_pts += score.scores[0] as i64;
        accum.dmc_ew_pts += score.scores[1] as i64;
        if score.scores[0] > score.scores[1] {
            accum.dmc_ns_wins += 1;
        }
        accum.dmc_deals += 1;
    }

    Some(accum)
}

// --- Multi-threaded sweep runner ---

fn run_sweep(
    config: &SweepConfig,
    n_deals: u32,
    dmc_weights: &DmcWeights,
    n_threads: usize,
    base_seed: u64,
    progress: &AtomicU32,
) -> SweepResult {
    let per_thread = (n_deals as usize + n_threads - 1) / n_threads;

    let accums: Vec<SweepAccum> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let start = t * per_thread;
            let end = ((t + 1) * per_thread).min(n_deals as usize);
            if start >= end {
                continue;
            }
            let count = end - start;

            handles.push(s.spawn(move || {
                let seed = base_seed.wrapping_add(t as u64 * 7919);
                let mut rng = StdRng::seed_from_u64(seed);
                let mut dmc_net = dmc_weights.make_net();
                let mut obs_buf = vec![0.0f32; OBS_DIM];
                let mut local = SweepAccum::default();

                for _ in 0..count {
                    let dealer: u8 = rng.gen_range(0..4);
                    let state = GameState::deal_random(dealer, &mut rng);
                    if let Some(delta) = play_deal(
                        &state,
                        &config.dd_config,
                        &mut dmc_net,
                        &mut obs_buf,
                        &mut rng,
                    ) {
                        local.merge(&delta);
                    }
                    progress.fetch_add(1, Ordering::Relaxed);
                }
                local
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut combined = SweepAccum::default();
    for a in &accums {
        combined.merge(a);
    }
    SweepResult::from_accum(config.label.clone(), &combined)
}

// --- Output formatting ---

fn print_header() {
    println!(
        "  {:<14} | {:<24} | {:<24} | {}",
        "Config", "vs Random", "vs DouDou35", "timing"
    );
    println!(
        "  {:<14} | {:<7} {:>6} {:>6}   | {:<7} {:>6} {:>6}   | {:>8} {:>6}",
        "", "win%", "ns", "ew", "win%", "ns", "ew", "ms/deal", "dets"
    );
    println!("  {}", "-".repeat(90));
}

fn print_row(r: &SweepResult) {
    println!(
        "  {:<14} | {:>6.1}% {:>6.1} {:>6.1}   | {:>6.1}% {:>6.1} {:>6.1}   | {:>8.1} {:>6.1}",
        r.label,
        r.rand_win_pct, r.rand_avg_ns, r.rand_avg_ew,
        r.dmc_win_pct, r.dmc_avg_ns, r.dmc_avg_ew,
        r.ms_per_deal, r.avg_dets,
    );
}

// --- CLI helpers ---

fn get_arg(args: &[String], name: &str) -> Option<String> {
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1).cloned())
}

fn get_arg_or(args: &[String], name: &str, default: &str) -> String {
    get_arg(args, name).unwrap_or_else(|| default.to_string())
}

fn default_threads() -> usize {
    std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4)
}

// --- Main ---

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let dmc_model_path = match get_arg(&args, "--dmc-model") {
        Some(p) => p,
        None => {
            eprintln!("Usage: isdd_sweep --dmc-model <path> [--deals N] [--threads T] [--seed S]");
            eprintln!("Example: isdd_sweep --dmc-model models/dmc_35.bin --deals 200 --threads 8");
            std::process::exit(1);
        }
    };

    let n_deals: u32 = get_arg_or(&args, "--deals", "200").parse().unwrap_or(200);
    let n_threads: usize = get_arg_or(&args, "--threads", &default_threads().to_string())
        .parse()
        .unwrap_or_else(|_| default_threads());
    let seed: u64 = get_arg_or(&args, "--seed", "42").parse().unwrap_or(42);

    // Load DMC model
    let dmc_weights = DmcWeights::load(&dmc_model_path)
        .unwrap_or_else(|e| panic!("Failed to load DMC model {}: {}", dmc_model_path, e));
    println!(
        "Loaded DMC model: {} (obs_dim={}, hidden={}, dueling={})",
        dmc_model_path, dmc_weights.obs_dim, dmc_weights.hidden, dmc_weights.dueling
    );

    println!(
        "\n=== IS-DD Parameter Sweep ({} deals, {} threads) ===\n",
        n_deals, n_threads
    );

    // Progress monitor state (shared across sections)
    let progress = Arc::new(AtomicU32::new(0));

    // --- Section 1: Count-based sweep ---
    let count_configs: Vec<SweepConfig> = [1u32, 2, 4, 8, 16, 32, 64]
        .iter()
        .map(|&d| SweepConfig {
            label: format!("D={}", d),
            dd_config: IsDdConfig {
                determinizations: d,
                time_limit_ms: None,
                use_soft_inference: true,
                ..Default::default()
            },
        })
        .collect();

    // --- Section 2: Time-based sweep ---
    let time_configs: Vec<SweepConfig> = [5u32, 10, 20, 50]
        .iter()
        .map(|&ms| SweepConfig {
            label: format!("{}ms", ms),
            dd_config: IsDdConfig {
                determinizations: 1000,
                time_limit_ms: Some(ms),
                ..Default::default()
            },
        })
        .collect();

    // --- Section 3: Soft inference comparison ---
    let soft_configs: Vec<SweepConfig> = [
        (8u32, false, "D=8  hard"),
        (8, true, "D=8  soft"),
        (16, false, "D=16 hard"),
        (16, true, "D=16 soft"),
    ]
    .iter()
    .map(|&(d, soft, label)| SweepConfig {
        label: label.to_string(),
        dd_config: IsDdConfig {
            determinizations: d,
            use_soft_inference: soft,
            ..Default::default()
        },
    })
    .collect();

    let total_configs =
        count_configs.len() + time_configs.len() + soft_configs.len();
    let total_work = total_configs as u32 * n_deals;

    // Progress monitor thread
    let prog_clone = progress.clone();
    let monitor = std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(15));
            let done = prog_clone.load(Ordering::Relaxed);
            if done >= total_work {
                break;
            }
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed * 60.0;
            let eta = if rate > 0.0 {
                (total_work - done) as f64 / rate * 60.0
            } else {
                0.0
            };
            eprint!(
                "\r  Progress: {}/{} deals ({:.0}%), {:.0}/min, ETA {:.0}s   ",
                done, total_work,
                100.0 * done as f64 / total_work as f64,
                rate,
                eta,
            );
        }
    });

    // Run and print each section
    let mut section_seed = seed;

    // Section 1
    println!("--- Count-based sweep (time_limit=None, soft=on) ---");
    print_header();
    for cfg in &count_configs {
        let r = run_sweep(cfg, n_deals, &dmc_weights, n_threads, section_seed, &progress);
        print_row(&r);
        section_seed = section_seed.wrapping_add(1);
    }
    println!();

    // Section 2
    println!("--- Time-based sweep (det=1000, soft=on) ---");
    print_header();
    for cfg in &time_configs {
        let r = run_sweep(cfg, n_deals, &dmc_weights, n_threads, section_seed, &progress);
        print_row(&r);
        section_seed = section_seed.wrapping_add(1);
    }
    println!();

    // Section 3
    println!("--- Soft inference comparison ---");
    print_header();
    for cfg in &soft_configs {
        let r = run_sweep(cfg, n_deals, &dmc_weights, n_threads, section_seed, &progress);
        print_row(&r);
        section_seed = section_seed.wrapping_add(1);
    }
    println!();

    // Signal monitor to stop
    progress.store(total_work, Ordering::Relaxed);
    let _ = monitor.join();
    eprintln!();
}
