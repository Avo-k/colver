//! Arena: systematic bot comparison framework.
//!
//! Bots are TOML files in `arena/bots/`, results are appended to
//! `arena/results/matches.csv`. The arena itself knows nothing about how a bot
//! plays: it parses each file into an [`AgentSpec`], asks it for four seated
//! [`Player`]s, and runs [`game_loop::play_match`]. Everything that decides
//! *how* a seat plays — which models, which observation layout, where the
//! determinized worlds come from — lives in `colver_core::agent`, so the arena
//! and the web server cannot drift apart the way they did when each carried
//! its own copy of the dispatch.
//!
//! ```text
//!   cargo run --bin arena --release -- list
//!   cargo run --bin arena --release -- h2h bot_a bot_b --matches 200
//!   cargo run --bin arena --release -- round-robin --matches 100 [--bots a,b,c]
//!   cargo run --bin arena --release -- results [--bot name]
//!   cargo run --bin arena --release -- trace bot_a bot_b [--deals N]
//! ```

use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

use colver_core::agent::{AgentSpec, MatchContext, Player};
use colver_core::game_loop::{self, MATCH_TARGET};
use colver_core::state::GameState;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

const RESULTS_PATH: &str = "arena/results/matches.csv";
const BOTS_DIR: &str = "arena/bots";

// ══════════════════════════════════════════════════════════════════════
//  Bot loading
// ══════════════════════════════════════════════════════════════════════

fn load_all_bots() -> Vec<AgentSpec> {
    let dir = match std::fs::read_dir(BOTS_DIR) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {}: {}", BOTS_DIR, e);
            return Vec::new();
        }
    };
    let mut paths: Vec<_> = dir
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("toml"))
        .map(|e| e.path())
        .collect();
    paths.sort();

    let mut bots = Vec::new();
    for path in paths {
        match AgentSpec::from_toml_file(path.to_str().unwrap_or("")) {
            Ok(spec) => bots.push(spec),
            Err(e) => eprintln!("  Warning: {}: {}", path.display(), e),
        }
    }
    bots
}

fn find_bot<'a>(bots: &'a [AgentSpec], name: &str) -> &'a AgentSpec {
    bots.iter().find(|b| b.name == name).unwrap_or_else(|| {
        eprintln!("Bot '{}' not found in {}/", name, BOTS_DIR);
        std::process::exit(1);
    })
}

/// Seat two specs at the table: NS takes 0 and 2, EW takes 1 and 3.
///
/// Each seat is an independent player object with its own models, beliefs and
/// RNG stream — partners must not share state, since each only knows its own
/// hand.
fn seat_players(
    ns: &AgentSpec,
    ew: &AgentSpec,
    seed: u64,
) -> Result<[Box<dyn Player>; 4], String> {
    let build = |spec: &AgentSpec, seat: u8| -> Result<Box<dyn Player>, String> {
        let mut spec = spec.clone();
        spec.seed = seed;
        spec.build(seat).map_err(|e| format!("{}: {}", spec.name, e))
    };
    Ok([
        build(ns, 0)?,
        build(ew, 1)?,
        build(ns, 2)?,
        build(ew, 3)?,
    ])
}


// ══════════════════════════════════════════════════════════════════════
//  Matchup runner (parallel)
// ══════════════════════════════════════════════════════════════════════

#[derive(Default, Clone)]
struct MatchupResult {
    n_matches: u32,
    ns_wins: u32,
    ew_wins: u32,
    total_margin: i64,
    /// Donnes jouées, cumulées. Le coût d'un run se compte en donnes, pas en
    /// matches : un match dure jusqu'à 2000 points, donc un nombre variable de
    /// donnes, et c'est ce nombre-là qu'il faut pour estimer une durée avant
    /// de lancer plusieurs heures de recherche IS-DD.
    n_deals: u64,
}

impl MatchupResult {
    fn merge(&mut self, other: &MatchupResult) {
        self.n_matches += other.n_matches;
        self.ns_wins += other.ns_wins;
        self.ew_wins += other.ew_wins;
        self.total_margin += other.total_margin;
        self.n_deals += other.n_deals;
    }
}

fn run_matchup(
    ns: &AgentSpec,
    ew: &AgentSpec,
    n_matches: u32,
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
                let thread_seed = base_seed.wrapping_add(t as u64 * 7919);
                let mut rng = StdRng::seed_from_u64(thread_seed);
                // Players are built once per thread and reused across matches;
                // `init_deal` is what clears their per-deal state, so this only
                // saves reloading the nets.
                let mut players = match seat_players(ns, ew, thread_seed) {
                    Ok(p) => p,
                    Err(e) => {
                        eprintln!("Error: {}", e);
                        std::process::exit(1);
                    }
                };
                let mut result = MatchupResult::default();
                for _ in 0..count {
                    let dealer = rng.gen_range(0..4);
                    let mr = match game_loop::play_match(&mut players, dealer, &mut rng) {
                        Ok(mr) => mr,
                        Err(e) => {
                            eprintln!("\nMatch aborted: {}", e);
                            std::process::exit(1);
                        }
                    };
                    result.n_matches += 1;
                    if mr.winner == 0 {
                        result.ns_wins += 1;
                    } else {
                        result.ew_wins += 1;
                    }
                    result.total_margin += (mr.ns_final - mr.ew_final) as i64;
                    result.n_deals += mr.deals as u64;
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

/// Run H2H with duplicate matching (both directions, same seeds).
fn run_h2h(
    a: &AgentSpec,
    b: &AgentSpec,
    n_matches: u32,
    n_threads: usize,
    base_seed: u64,
    progress: &AtomicU32,
) -> (MatchupResult, MatchupResult) {
    // Direction 1: A as NS, B as EW.
    let r1 = run_matchup(a, b, n_matches, n_threads, base_seed, progress);
    // Direction 2: sides swapped, so deal luck largely cancels between the two.
    let r2 = run_matchup(b, a, n_matches, n_threads, base_seed.wrapping_add(1_000_000), progress);
    (r1, r2)
}

// ══════════════════════════════════════════════════════════════════════
//  CSV results persistence
// ══════════════════════════════════════════════════════════════════════

fn now_iso() -> String {
    // Simple timestamp without chrono crate
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Approximate: good enough for logging
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    // Days since 1970-01-01
    let (year, month, day) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", year, month, day, hours, minutes, seconds)
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Simple Gregorian calendar calculation
    let mut y = 1970;
    let mut remaining = days;
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining < days_in_year { break; }
        remaining -= days_in_year;
        y += 1;
    }
    let month_days: [u64; 12] = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m = 0;
    while m < 12 && remaining >= month_days[m] {
        remaining -= month_days[m];
        m += 1;
    }
    (y, m as u64 + 1, remaining + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn append_csv_result(
    bot_a: &str, bot_b: &str,
    bid_a: &str, play_a: &str,
    bid_b: &str, play_b: &str,
    r1: &MatchupResult, r2: &MatchupResult,
    seed: u64, wall_secs: f64,
) {
    let results_dir = std::path::Path::new(RESULTS_PATH).parent().unwrap();
    let _ = std::fs::create_dir_all(results_dir);

    let write_header = !std::path::Path::new(RESULTS_PATH).exists();
    let mut file = match std::fs::OpenOptions::new().create(true).append(true).open(RESULTS_PATH) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot write results to {}: {}", RESULTS_PATH, e);
            return;
        }
    };

    use std::io::Write;
    if write_header {
        let _ = writeln!(file, "timestamp,bot_a,bid_a,play_a,bot_b,bid_b,play_b,matches,a_wins,b_wins,win_pct,avg_margin,seed,wall_secs,matches_per_min");
    }

    let ts = now_iso();
    // Aggregate both directions: A wins = r1.ns_wins (A as NS) + r2.ew_wins (A as EW)
    let total_matches = r1.n_matches + r2.n_matches;
    let a_wins = r1.ns_wins + r2.ew_wins;
    let b_wins = r1.ew_wins + r2.ns_wins;
    let a_margin = r1.total_margin - r2.total_margin;
    let win_pct = 100.0 * a_wins as f64 / total_matches as f64;
    let avg_margin = a_margin as f64 / total_matches as f64;
    let matches_per_min = if wall_secs > 0.0 { total_matches as f64 / wall_secs * 60.0 } else { 0.0 };

    let _ = writeln!(file, "{},{},{},{},{},{},{},{},{},{},{:.1},{:+.0},{},{:.1},{:.1}",
        ts, bot_a, bid_a, play_a, bot_b, bid_b, play_b,
        total_matches, a_wins, b_wins, win_pct, avg_margin, seed, wall_secs, matches_per_min);
}

// ══════════════════════════════════════════════════════════════════════
//  Results display
// ══════════════════════════════════════════════════════════════════════

struct AggResult {
    bot_a: String,
    bot_b: String,
    matches: u32,
    a_wins: u32,
    avg_margin: f64,
    wall_secs: f64,
    matches_per_min: f64,
    timestamp: String,
}

fn load_results() -> Vec<AggResult> {
    let content = match std::fs::read_to_string(RESULTS_PATH) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut results = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 { continue; } // skip header
        let cols: Vec<&str> = line.split(',').collect();
        // New format (15 cols): timestamp,bot_a,bid_a,play_a,bot_b,bid_b,play_b,matches,...
        // Old format (10-11 cols): timestamp,bot_a,bot_b,matches,...
        if cols.len() >= 14 {
            // New format
            let wall_secs: f64 = cols[13].parse().unwrap_or(0.0);
            let matches: u32 = cols[7].parse().unwrap_or(0);
            let matches_per_min = if cols.len() > 14 {
                cols[14].parse().unwrap_or(0.0)
            } else if wall_secs > 0.0 {
                matches as f64 / wall_secs * 60.0
            } else { 0.0 };
            results.push(AggResult {
                timestamp: cols[0].to_string(),
                bot_a: cols[1].to_string(),
                bot_b: cols[4].to_string(),
                matches,
                a_wins: cols[8].parse().unwrap_or(0),
                avg_margin: cols[11].parse().unwrap_or(0.0),
                wall_secs,
                matches_per_min,
            });
        } else if cols.len() >= 10 {
            // Old format
            let wall_secs: f64 = cols[9].parse().unwrap_or(0.0);
            let matches: u32 = cols[3].parse().unwrap_or(0);
            let matches_per_min = if cols.len() > 10 {
                cols[10].parse().unwrap_or(0.0)
            } else if wall_secs > 0.0 {
                matches as f64 / wall_secs * 60.0
            } else { 0.0 };
            results.push(AggResult {
                timestamp: cols[0].to_string(),
                bot_a: cols[1].to_string(),
                bot_b: cols[2].to_string(),
                matches,
                a_wins: cols[4].parse().unwrap_or(0),
                avg_margin: cols[7].parse().unwrap_or(0.0),
                wall_secs,
                matches_per_min,
            });
        }
    }
    results
}

fn cmd_results(filter_bot: Option<&str>) {
    let results = load_results();
    if results.is_empty() {
        println!("No results yet. Run some matches first!");
        return;
    }

    // Load bot configs for bid/play labels
    use std::collections::HashMap;
    let bot_configs: HashMap<String, AgentSpec> = load_all_bots()
        .into_iter().map(|b| (b.name.clone(), b)).collect();
    let bot_label = |name: &str| -> (String, String) {
        if let Some(cfg) = bot_configs.get(name) {
            (cfg.bid_label(), cfg.play_label())
        } else {
            ("?".into(), "?".into())
        }
    };

    // Build leaderboard: aggregate all H2H into per-bot stats
    let mut wins: HashMap<String, u32> = HashMap::new();
    let mut played: HashMap<String, u32> = HashMap::new();
    let mut margin: HashMap<String, f64> = HashMap::new();

    for r in &results {
        if let Some(f) = filter_bot {
            if r.bot_a != f && r.bot_b != f { continue; }
        }
        *wins.entry(r.bot_a.clone()).or_default() += r.a_wins;
        *wins.entry(r.bot_b.clone()).or_default() += r.matches - r.a_wins;
        *played.entry(r.bot_a.clone()).or_default() += r.matches;
        *played.entry(r.bot_b.clone()).or_default() += r.matches;
        *margin.entry(r.bot_a.clone()).or_default() += r.avg_margin * r.matches as f64;
        *margin.entry(r.bot_b.clone()).or_default() -= r.avg_margin * r.matches as f64;
    }

    let mut ranking: Vec<_> = played.keys().map(|bot| {
        let w = *wins.get(bot).unwrap_or(&0);
        let p = *played.get(bot).unwrap_or(&1);
        let m = *margin.get(bot).unwrap_or(&0.0);
        (bot.clone(), w, p, 100.0 * w as f64 / p as f64, m / p as f64)
    }).collect();
    ranking.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("=============================================================");
    println!("  ARENA LEADERBOARD");
    if let Some(f) = filter_bot {
        println!("  (filtered: matches involving '{}')", f);
    }
    println!("=============================================================");
    println!("  {:>3} {:<20} {:<22} {:<22} {:>6} {:>8}", "#", "Bot", "Bid", "Play", "Win%", "Margin");
    println!("  {}", "-".repeat(90));

    for (rank, (bot, _w, _p, pct, m)) in ranking.iter().enumerate() {
        let (bid, play) = bot_label(bot);
        println!("  {:>3} {:<20} {:<22} {:<22} {:>5.1}% {:>+7.0}", rank + 1, bot, bid, play, pct, m);
    }

    // Bot speed estimates: for each bot, average matches_per_min across all matchups
    // The slowest bot in a pair determines the speed, so we track per-pair speeds
    // and attribute to both bots.
    let mut bot_speed_sum: HashMap<String, f64> = HashMap::new();
    let mut bot_speed_count: HashMap<String, u32> = HashMap::new();
    for r in &results {
        if r.matches_per_min <= 0.0 { continue; }
        if let Some(f) = filter_bot {
            if r.bot_a != f && r.bot_b != f { continue; }
        }
        // A pair's speed reflects both bots — attribute to each
        *bot_speed_sum.entry(r.bot_a.clone()).or_default() += r.matches_per_min;
        *bot_speed_count.entry(r.bot_a.clone()).or_default() += 1;
        *bot_speed_sum.entry(r.bot_b.clone()).or_default() += r.matches_per_min;
        *bot_speed_count.entry(r.bot_b.clone()).or_default() += 1;
    }

    if !bot_speed_sum.is_empty() {
        println!();
        println!("  BOT SPEEDS (avg matches/min across observed matchups)");
        println!("  {}", "-".repeat(60));
        // Sort by speed descending
        let mut speeds: Vec<_> = bot_speed_sum.keys().map(|bot| {
            let avg = bot_speed_sum[bot] / *bot_speed_count.get(bot).unwrap_or(&1) as f64;
            (bot.clone(), avg)
        }).collect();
        speeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (bot, avg_speed) in &speeds {
            let time_200 = if *avg_speed > 0.0 { 400.0 / avg_speed } else { f64::INFINITY };
            println!("  {:<20} {:>6.0} matches/min   (~{:.0}min for 200-match H2H)",
                bot, avg_speed, time_200);
        }
    }

    // Show recent H2H results
    println!();
    println!("  RECENT MATCHES");
    println!("  {}", "-".repeat(60));
    let start = if results.len() > 20 { results.len() - 20 } else { 0 };
    for r in &results[start..] {
        if let Some(f) = filter_bot {
            if r.bot_a != f && r.bot_b != f { continue; }
        }
        let pct = 100.0 * r.a_wins as f64 / r.matches as f64;
        let speed_str = if r.matches_per_min > 0.0 {
            format!(" {:.0}m/min", r.matches_per_min)
        } else {
            String::new()
        };
        println!("  {} vs {}: {:.1}% ({}/{}) margin {:+.0}{}  [{}]",
            r.bot_a, r.bot_b, pct, r.a_wins, r.matches, r.avg_margin, speed_str, r.timestamp);
    }
}

// ══════════════════════════════════════════════════════════════════════
//  CLI
// ══════════════════════════════════════════════════════════════════════

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  arena list                                 List available bots");
    eprintln!("  arena h2h <bot_a> <bot_b> [--matches N]   Head-to-head comparison");
    eprintln!("  arena round-robin [--matches N] [--bots a,b,c] [--no-save]  Round-robin tournament");
    eprintln!("  arena results [--bot name]                 Show results leaderboard");
    eprintln!("  arena trace <bot_a> <bot_b> [--deals N]   Play same deals with both bots, show diffs");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --matches N     Matches per direction (default 100)");
    eprintln!("  --threads N     Thread count (default auto)");
    eprintln!("  --seed N        Base RNG seed (default 42)");
    eprintln!("  --bots a,b,c    Only include these bots in round-robin");
    eprintln!("  --no-save       Don't persist results to CSV (h2h only)");
    eprintln!("  --deals N       Number of deals to trace (default 50)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let subcmd = args[1].as_str();

    match subcmd {
        "list" => cmd_list(),
        "h2h" => cmd_h2h(&args[2..]),
        "round-robin" => cmd_round_robin(&args[2..]),
        "results" => {
            let filter = parse_flag(&args[2..], "--bot");
            cmd_results(filter.as_deref());
        }
        "trace" => cmd_trace(&args[2..]),
        "--help" | "-h" | "help" => print_usage(),
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
            std::process::exit(1);
        }
    }
}

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
    }
    None
}

fn parse_flag_u32(args: &[String], flag: &str, default: u32) -> u32 {
    parse_flag(args, flag).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn parse_flag_u64(args: &[String], flag: &str, default: u64) -> u64 {
    parse_flag(args, flag).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn cmd_list() {
    let bots = load_all_bots();
    if bots.is_empty() {
        println!("No bots found in {}/", BOTS_DIR);
        return;
    }
    println!("Available bots ({}):", bots.len());
    println!("  {:<20} {:<22} {:<22}", "Name", "Bid", "Play");
    println!("  {}", "-".repeat(66));
    for b in &bots {
        println!("  {:<20} {:<22} {:<22}", b.name, b.bid_label(), b.play_label());
    }
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn cmd_h2h(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: arena h2h <bot_a> <bot_b> [--matches N] [--threads N] [--seed N] [--no-save]");
        std::process::exit(1);
    }

    let bot_a_name = &args[0];
    let bot_b_name = &args[1];
    let rest = &args[2..];
    let n_matches = parse_flag_u32(rest, "--matches", 100);
    let n_threads = parse_flag_u32(rest, "--threads", default_threads() as u32) as usize;
    let seed = parse_flag_u64(rest, "--seed", 42);
    let no_save = has_flag(rest, "--no-save");

    let all_bots = load_all_bots();
    let agent_a = find_bot(&all_bots, bot_a_name);
    let agent_b = find_bot(&all_bots, bot_b_name);

    println!("=============================================================");
    println!("  ARENA H2H: {} vs {}", agent_a.name, agent_b.name);
    println!("  {} matches/direction ({}x2 total), {} threads, seed {}",
        n_matches, n_matches, n_threads, seed);
    println!("=============================================================");
    println!();

    let progress = Arc::new(AtomicU32::new(0));
    let total = n_matches * 2;

    // Progress monitor
    let progress_clone = progress.clone();
    let monitor = std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let done = progress_clone.load(Ordering::Relaxed);
            if done >= total { break; }
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed * 60.0;
            let eta = if rate > 0.0 { (total - done) as f64 / rate * 60.0 } else { 0.0 };
            eprint!("\r  Progress: {}/{} matches ({:.0}%), {:.0}/min, ETA {:.0}s   ",
                done, total, 100.0 * done as f64 / total as f64, rate, eta);
        }
    });

    let start = Instant::now();
    let (r1, r2) = run_h2h(agent_a, agent_b, n_matches, n_threads, seed, &progress);
    let elapsed = start.elapsed();

    progress.store(total, Ordering::Relaxed);
    let _ = monitor.join();
    eprintln!();

    // Aggregate
    let a_wins = r1.ns_wins + r2.ew_wins;
    let b_wins = r1.ew_wins + r2.ns_wins;
    let total_matches = r1.n_matches + r2.n_matches;
    let a_margin = r1.total_margin - r2.total_margin;
    let a_pct = 100.0 * a_wins as f64 / total_matches as f64;
    let avg_margin = a_margin as f64 / total_matches as f64;

    println!("  RESULT: {} {:.1}% vs {} {:.1}%",
        agent_a.name, a_pct, agent_b.name, 100.0 - a_pct);
    println!("    A: {} | {}     B: {} | {}", agent_a.bid_label(), agent_a.play_label(), agent_b.bid_label(), agent_b.play_label());
    println!("  Wins: {} {} — {} {}", agent_a.name, a_wins, agent_b.name, b_wins);
    println!("  Avg margin: {:+.0} (from {}'s perspective)", avg_margin, agent_a.name);
    println!("  Dir 1 ({}=NS): {}-{}", agent_a.name, r1.ns_wins, r1.ew_wins);
    println!("  Dir 2 ({}=NS): {}-{}", agent_b.name, r2.ns_wins, r2.ew_wins);
    let total_deals = r1.n_deals + r2.n_deals;
    println!("  Wall: {:.1}s ({:.1} matches/min) — {} donnes ({:.1}/match, {:.1} donnes/s)",
        elapsed.as_secs_f64(), total_matches as f64 / elapsed.as_secs_f64() * 60.0,
        total_deals, total_deals as f64 / total_matches as f64,
        total_deals as f64 / elapsed.as_secs_f64());

    print_world_telemetry();

    // Persist
    if !no_save {
        append_csv_result(&agent_a.name, &agent_b.name,
            &agent_a.bid_label(), &agent_a.play_label(),
            &agent_b.bid_label(), &agent_b.play_label(),
            &r1, &r2, seed, elapsed.as_secs_f64());
        println!();
        println!("  Results saved to {}", RESULTS_PATH);
    } else {
        println!();
        println!("  (--no-save: results NOT written to CSV)");
    }
}

/// D'où venaient les mondes que les recherches IS-DD ont résolus.
///
/// Silencieux quand aucun bot du run ne joue en IS-DD — un h2h de deux réseaux
/// n'a rien à dire ici. Sinon c'est la réponse chiffrée à « la file playgen
/// s'assèche-t-elle en cours de recherche ? », donc à « le belief net sert-il
/// encore à quelque chose ? ».
fn print_world_telemetry() {
    let s = colver_core::agent::isdd::telemetry::snapshot();
    if s.decisions == 0 {
        return;
    }
    let pct = |x: u64, n: u64| if n == 0 { 0.0 } else { 100.0 * x as f64 / n as f64 };
    let sampled = s.sampled();
    println!();
    println!("  Mondes IS-DD ({} décisions) :", s.decisions);
    println!("    sans échantillonnage (coup forcé / position résolue) : {} ({:.1}%)",
        s.no_sampling, pct(s.no_sampling, s.decisions));
    println!("    échantillonnées : {} — 100% playgen {} ({:.1}%), partielles {} ({:.1}%), sans playgen {} ({:.1}%)",
        sampled,
        s.all_playgen, pct(s.all_playgen, sampled),
        s.partial, pct(s.partial, sampled),
        s.no_playgen, pct(s.no_playgen, sampled));
    println!("    mondes : source {} | belief {} | uniforme {}  → repli {:.2}%",
        s.worlds_injected + s.worlds_playgen, s.worlds_belief, s.worlds_uniform,
        s.fallback_world_pct());

    let lanes = colver_core::agent::isdd::telemetry::lanes();
    if lanes.is_empty() {
        return;
    }
    println!();
    println!("  Par stade de donne (cartes restantes en main) :");
    println!("    {:>6} {:>8} {:>8} {:>7} {:>9} {:>9} {:>8} {:>8} {:>7}",
        "cartes", "décis.", "cherché", "a/r", "demandés", "reçus", "remplis", "résolus", "utilisés");
    for l in &lanes {
        println!("    {:>6} {:>8} {:>8} {:>7} {:>9} {:>9} {:>7.0}% {:>8} {:>6.0}%",
            l.cards_left, l.decisions, l.searched, l.rounds,
            l.requested, l.delivered, l.fill_pct(), l.solved, l.used_pct());
    }
    let delivered: u64 = lanes.iter().map(|l| l.delivered).sum();
    let discarded: u64 = lanes.iter().map(|l| l.discarded).sum();
    if delivered > 0 {
        println!("    → {} mondes reçus, {} jetés sans être résolus ({:.1}%)",
            delivered, discarded, 100.0 * discarded as f64 / delivered as f64);
    }
}

fn cmd_round_robin(args: &[String]) {
    let n_matches = parse_flag_u32(args, "--matches", 100);
    let n_threads = parse_flag_u32(args, "--threads", default_threads() as u32) as usize;
    let seed = parse_flag_u64(args, "--seed", 42);
    let bot_filter = parse_flag(args, "--bots");
    let no_save = has_flag(args, "--no-save");

    let all_bots = load_all_bots();
    let bots: Vec<&AgentSpec> = if let Some(ref filter) = bot_filter {
        let names: Vec<&str> = filter.split(',').collect();
        all_bots.iter().filter(|b| names.contains(&b.name.as_str())).collect()
    } else {
        all_bots.iter().collect()
    };

    if bots.len() < 2 {
        eprintln!("Need at least 2 bots for round-robin. Found {} in {}/", bots.len(), BOTS_DIR);
        std::process::exit(1);
    }

    let agents: Vec<&AgentSpec> = bots.clone();

    let n = agents.len();
    let total_matchups = n * (n - 1) / 2;
    let total_matches = total_matchups as u32 * n_matches * 2; // both directions

    println!("=============================================================");
    println!("  ARENA ROUND-ROBIN — First to {}", MATCH_TARGET);
    println!("  {} bots, {} matches/direction, {} threads", n, n_matches, n_threads);
    println!("  {} matchups, {} total matches", total_matchups, total_matches);
    println!("=============================================================");
    println!();

    for (i, a) in agents.iter().enumerate() {
        println!("  [{:>2}] {:<20} {} | {}", i, a.name, a.bid_label(), a.play_label());
    }
    println!();

    let progress = Arc::new(AtomicU32::new(0));
    let start = Instant::now();

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
            eprint!("\r  Progress: {}/{} ({:.0}%), {:.0}/min, ETA {:.0}s   ",
                done, total_matches, 100.0 * done as f64 / total_matches as f64, rate, eta);
        }
    });

    // Win/margin matrices
    let mut win_matrix = vec![vec![0u32; n]; n];
    let mut margin_matrix = vec![vec![0i64; n]; n];
    let mut matches_matrix = vec![vec![0u32; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let pair_seed = seed.wrapping_add((i * 1000 + j * 100) as u64);
            let pair_start = Instant::now();
            let (r1, r2) = run_h2h(agents[i], agents[j], n_matches, n_threads, pair_seed, &progress);
            let pair_secs = pair_start.elapsed().as_secs_f64();

            // Persist each pair with its own wall time
            if !no_save {
                append_csv_result(&agents[i].name, &agents[j].name,
                    &agents[i].bid_label(), &agents[i].play_label(),
                    &agents[j].bid_label(), &agents[j].play_label(),
                    &r1, &r2, pair_seed, pair_secs);
            }

            // Aggregate into matrices
            win_matrix[i][j] += r1.ns_wins + r2.ew_wins;
            win_matrix[j][i] += r1.ew_wins + r2.ns_wins;
            margin_matrix[i][j] += r1.total_margin - r2.total_margin;
            margin_matrix[j][i] += r2.total_margin - r1.total_margin;
            let total = r1.n_matches + r2.n_matches;
            matches_matrix[i][j] += total;
            matches_matrix[j][i] += total;
        }
    }

    progress.store(total_matches, Ordering::Relaxed);
    let _ = monitor.join();
    let elapsed = start.elapsed();
    eprintln!();

    // Print win matrix
    println!("=============================================================");
    println!("  WIN MATRIX (row win% vs column)");
    println!("=============================================================");
    print!("  {:>16}", "");
    for a in &agents { print!("  {:>8}", &a.name[..a.name.len().min(8)]); }
    println!("    TOTAL");
    println!("  {}", "-".repeat(16 + 10 * (n + 1) + 4));

    let mut total_wins = vec![0u32; n];
    let mut total_played = vec![0u32; n];

    for i in 0..n {
        print!("  {:>16}", agents[i].name);
        let mut row_wins = 0u32;
        let mut row_played = 0u32;
        for j in 0..n {
            if i == j {
                print!("       - ");
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
        let avg_m: f64 = {
            let total_m: i64 = (0..n).filter(|&j| j != *idx).map(|j| margin_matrix[*idx][j]).sum();
            let total_p: u32 = (0..n).filter(|&j| j != *idx).map(|j| matches_matrix[*idx][j]).sum();
            total_m as f64 / total_p as f64
        };
        println!("  {:>2}. {:<20} {:<22} {:<22} win {:5.1}%  margin {:+5.0}",
            rank + 1, agents[*idx].name, agents[*idx].bid_label(), agents[*idx].play_label(), pct, avg_m);
    }

    println!();
    println!("  Wall: {:.1}s ({} matches, {:.1}/min)",
        elapsed.as_secs_f64(), total_matches, total_matches as f64 / elapsed.as_secs_f64() * 60.0);
    if no_save {
        println!("  (--no-save: results NOT written to CSV)");
    } else {
        println!("  Results saved to {}", RESULTS_PATH);
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Trace: play same deals with two bots, compare decisions
// ══════════════════════════════════════════════════════════════════════

use colver_core::card::{card_name, cardset_str};

const SUIT_SYMS: [&str; 4] = ["S", "H", "D", "C"];

fn bid_action_str(action: u8) -> String {
    match action {
        0 => "PASS".to_string(),
        41 => "COINCHE".to_string(),
        42 => "SURCOINCHE".to_string(),
        1..=36 => {
            let idx = action - 1;
            let value_idx = idx / 4;
            let suit_idx = idx % 4;
            let value = (value_idx as u16 + 8) * 10;
            format!("{}{}", value, SUIT_SYMS[suit_idx as usize])
        }
        37..=40 => {
            let suit_idx = action - 37;
            format!("Capot{}", SUIT_SYMS[suit_idx as usize])
        }
        _ => format!("?{}", action),
    }
}

const SEAT_NAMES: [&str; 4] = ["N", "E", "S", "W"];

/// One deal replayed by a given seating, for side-by-side comparison.
struct DealTrace {
    bids: Vec<(u8, u8)>,  // (player, bid action)
    plays: Vec<(u8, u8)>, // (player, card)
    trick_leads: Vec<u8>,
    contract_str: String,
    ns_score: i32,
    ew_score: i32,
    void_deal: bool,
}

/// Play one deal with `players` already seated, recording what happened.
fn trace_deal(state_orig: &GameState, players: &mut [Box<dyn Player>; 4]) -> DealTrace {
    let mut state = *state_orig;
    let mut ctx = MatchContext::new(state.dealer);
    let (score, decisions) = game_loop::play_deal_traced(&mut state, players, &mut ctx)
        .unwrap_or_else(|e| {
            eprintln!("Trace aborted: {}", e);
            std::process::exit(1);
        });

    // Split the decision stream back into auction and play. Bids all precede
    // plays, so the boundary is wherever the first card appears.
    let n_bids = ctx.tracking.bid_history.len();
    let bids: Vec<(u8, u8)> = decisions[..n_bids].iter().map(|(p, d)| (*p, d.action)).collect();
    let plays: Vec<(u8, u8)> = decisions[n_bids..].iter().map(|(p, d)| (*p, d.action)).collect();
    let trick_leads: Vec<u8> = plays.chunks(4).map(|c| c[0].0).collect();

    let void_deal = state.contract.value == 0;
    let contract_str = if void_deal {
        "passed out".to_string()
    } else {
        format!(
            "{}{} by {}",
            state.contract.value,
            SUIT_SYMS[state.contract.trump_suit() as usize],
            if state.contract.team == 0 { "NS" } else { "EW" }
        )
    };

    DealTrace {
        bids,
        plays,
        trick_leads,
        contract_str,
        ns_score: score[0],
        ew_score: score[1],
        void_deal,
    }
}

fn cmd_trace(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: arena trace <bot_a> <bot_b> [--deals N] [--seed N]");
        std::process::exit(1);
    }

    let bot_a_name = &args[0];
    let bot_b_name = &args[1];
    let rest = &args[2..];
    let n_deals = parse_flag_u32(rest, "--deals", 50);
    let seed = parse_flag_u64(rest, "--seed", 42);

    let all_bots = load_all_bots();
    let agent_a = find_bot(&all_bots, bot_a_name);
    let agent_b = find_bot(&all_bots, bot_b_name);

    println!("═══════════════════════════════════════════════════════════════");
    println!("  TRACE: {} vs {}", agent_a.name, agent_b.name);
    println!("  {} deals, seed {}", n_deals, seed);
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut rng = StdRng::seed_from_u64(seed);

    // One seating per direction, reused across deals.
    let mut seating_a = seat_players(agent_a, agent_b, seed)
        .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
    let mut seating_b = seat_players(agent_b, agent_a, seed)
        .unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });

    // Stats
    let mut a_better = 0u32;
    let mut b_better = 0u32;
    let mut same = 0u32;
    let mut void_deals = 0u32;
    let mut a_total_ns: i32 = 0;
    let mut b_total_ns: i32 = 0;

    // Categorize differences
    let mut bid_diffs = 0u32;
    let mut play_diffs = 0u32;

    for deal_idx in 0..n_deals {
        let dealer = (deal_idx % 4) as u8;
        let state = GameState::deal_random(dealer, &mut rng);

        // The same deal played both ways round, so the comparison is about the
        // bots and not about who was dealt the good hand.
        let trace_a = trace_deal(&state, &mut seating_a);
        let trace_b = trace_deal(&state, &mut seating_b);

        if trace_a.void_deal && trace_b.void_deal {
            void_deals += 1;
            continue;
        }

        let a_net = trace_a.ns_score - trace_a.ew_score;
        let b_net = trace_b.ns_score - trace_b.ew_score;
        a_total_ns += a_net as i32;
        b_total_ns += b_net as i32;

        // Check if bids differ
        let bids_same = trace_a.bids.len() == trace_b.bids.len()
            && trace_a.bids.iter().zip(&trace_b.bids).all(|(a, b)| a.1 == b.1);
        if !bids_same { bid_diffs += 1; }

        // Check if plays differ
        let plays_same = trace_a.plays.len() == trace_b.plays.len()
            && trace_a.plays.iter().zip(&trace_b.plays).all(|(a, b)| a.1 == b.1);
        if !plays_same { play_diffs += 1; }

        let score_diff = a_net - b_net;

        if score_diff > 0 {
            a_better += 1;
        } else if score_diff < 0 {
            b_better += 1;
        } else {
            same += 1;
            continue;
        }

        // Print interesting deals (score diff >= 50 points)
        if score_diff.abs() >= 50 {
            let winner_name = if score_diff > 0 { &agent_a.name } else { &agent_b.name };
            println!("─────────────────────────────────────────────────────────");
            println!("Deal #{} (dealer={}) — {} better by {} pts",
                deal_idx, SEAT_NAMES[dealer as usize], winner_name, score_diff.abs());
            println!();

            // Hands
            for p in 0..4 {
                println!("  {} {}: {}", SEAT_NAMES[p],
                    if p % 2 == 0 { "(NS)" } else { "(EW)" },
                    cardset_str(state.hands[p]));
            }
            println!();

            // Bidding comparison
            println!("  Bidding ({}):", agent_a.name);
            print!("    ");
            for (player, action) in &trace_a.bids {
                print!("{}:{} ", SEAT_NAMES[*player as usize], bid_action_str(*action));
            }
            println!(" → {}", trace_a.contract_str);

            println!("  Bidding ({}):", agent_b.name);
            print!("    ");
            for (player, action) in &trace_b.bids {
                print!("{}:{} ", SEAT_NAMES[*player as usize], bid_action_str(*action));
            }
            println!(" → {}", trace_b.contract_str);
            println!();

            // Play comparison (trick by trick)
            let trump_a = if !trace_a.void_deal {
                format!(" ({})", trace_a.contract_str)
            } else { String::new() };
            println!("  {} as NS{}: score NS={} EW={}",
                agent_a.name, trump_a, trace_a.ns_score, trace_a.ew_score);
            for (i, chunk) in trace_a.plays.chunks(4).enumerate() {
                let lead = if i < trace_a.trick_leads.len() {
                    SEAT_NAMES[trace_a.trick_leads[i] as usize]
                } else { "?" };
                print!("    T{} (lead {}): ", i + 1, lead);
                for (player, card) in chunk {
                    print!("{}={} ", SEAT_NAMES[*player as usize], card_name(*card));
                }
                println!();
            }

            println!("  {} as NS: score NS={} EW={}",
                agent_b.name, trace_b.ns_score, trace_b.ew_score);
            for (i, chunk) in trace_b.plays.chunks(4).enumerate() {
                let lead = if i < trace_b.trick_leads.len() {
                    SEAT_NAMES[trace_b.trick_leads[i] as usize]
                } else { "?" };
                print!("    T{} (lead {}): ", i + 1, lead);
                for (player, card) in chunk {
                    print!("{}={} ", SEAT_NAMES[*player as usize], card_name(*card));
                }
                println!();
            }
            println!();
        }
    }

    // Summary
    let played = n_deals - void_deals;
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY ({} deals played, {} void)", played, void_deals);
    println!("  {} better in {} deals ({:.1}%)",
        agent_a.name, a_better, 100.0 * a_better as f64 / played as f64);
    println!("  {} better in {} deals ({:.1}%)",
        agent_b.name, b_better, 100.0 * b_better as f64 / played as f64);
    println!("  Same score: {} deals ({:.1}%)",
        same, 100.0 * same as f64 / played as f64);
    println!("  Avg NS score: {} {:.1}, {} {:.1}",
        agent_a.name, a_total_ns as f64 / played as f64,
        agent_b.name, b_total_ns as f64 / played as f64);
    println!("  Bid differences: {} ({:.0}%)", bid_diffs, 100.0 * bid_diffs as f64 / played as f64);
    println!("  Play differences: {} ({:.0}%)", play_diffs, 100.0 * play_diffs as f64 / played as f64);
    print_world_telemetry();
    println!("═══════════════════════════════════════════════════════════════");
}
