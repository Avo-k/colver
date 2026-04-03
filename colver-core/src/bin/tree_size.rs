use colver_core::bid_eval;
use colver_core::state::{GameState, Phase};
use rand::thread_rng;
use rand::Rng;
use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::time::Instant;

struct TreeStats {
    nodes: u64,
    leaves: u64,
    max_depth: u32,
}

static NODE_COUNTER: AtomicU64 = AtomicU64::new(0);
static STOP: AtomicBool = AtomicBool::new(false);

fn enumerate(state: &GameState, depth_limit: u32) -> TreeStats {
    if STOP.load(Ordering::Relaxed) {
        return TreeStats { nodes: 0, leaves: 0, max_depth: 0 };
    }

    if state.is_terminal() || depth_limit == 0 {
        NODE_COUNTER.fetch_add(1, Ordering::Relaxed);
        return TreeStats { nodes: 1, leaves: 1, max_depth: 0 };
    }

    let mask = state.legal_actions();
    let mut total = TreeStats { nodes: 1, leaves: 0, max_depth: 0 };
    NODE_COUNTER.fetch_add(1, Ordering::Relaxed);

    let mut remaining = mask;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as u8;
        remaining &= remaining - 1;

        let mut child = *state;
        child.step(bit);
        let sub = enumerate(&child, depth_limit - 1);
        total.nodes += sub.nodes;
        total.leaves += sub.leaves;
        total.max_depth = total.max_depth.max(sub.max_depth + 1);
    }

    total
}

/// Sample random paths to estimate tree size (Knuth's method).
/// Each random root-to-leaf path gives an unbiased estimate of tree size.
fn estimate_tree_size(state: &GameState, num_samples: u64, rng: &mut impl Rng) -> (f64, f64) {
    let mut estimates = Vec::new();

    for _ in 0..num_samples {
        let mut s = *state;
        let mut path_weight = 1.0f64;

        while !s.is_terminal() {
            let mask = s.legal_actions();
            let count = mask.count_ones();
            path_weight *= count as f64;

            // Pick a random action
            let idx = rng.gen_range(0..count);
            let action = select_nth_bit(mask, idx);
            s.step(action);
        }

        estimates.push(path_weight);
    }

    let mean = estimates.iter().sum::<f64>() / num_samples as f64;
    let variance = estimates.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / num_samples as f64;
    let stderr = (variance / num_samples as f64).sqrt();
    (mean, stderr)
}

fn select_nth_bit(mask: u64, mut n: u32) -> u8 {
    let mut remaining = mask;
    loop {
        let bit = remaining.trailing_zeros() as u8;
        if n == 0 {
            return bit;
        }
        n -= 1;
        remaining &= remaining - 1;
    }
}

fn deal_with_contract(rng: &mut impl Rng) -> GameState {
    loop {
        let mut s = GameState::deal_random(0, rng);
        while s.phase == Phase::Bidding {
            let action = bid_eval::improved_v2_bid(&s);
            s.step(action);
        }
        if s.phase == Phase::Playing {
            return s;
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_deals: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(10);
    let mode = args.get(2).map(|s| s.as_str()).unwrap_or("estimate");

    let mut rng = thread_rng();

    match mode {
        "exact" => run_exact(num_deals, &mut rng, args.get(3)),
        "estimate" => run_estimate(num_deals, &mut rng),
        "profile" => run_profile(num_deals, &mut rng),
        _ => {
            eprintln!("Usage: tree_size [num_deals] [exact|estimate|profile] [depth_limit]");
            eprintln!("  exact     - Full enumeration (slow! use depth_limit)");
            eprintln!("  estimate  - Knuth random sampling estimate (default)");
            eprintln!("  profile   - Per-ply branching factor profile");
            std::process::exit(1);
        }
    }
}

fn run_estimate(num_deals: usize, rng: &mut impl Rng) {
    let num_samples: u64 = 100_000;

    println!(
        "Estimating play-phase tree sizes for {} deals ({} random paths each)",
        num_deals,
        format_num(num_samples)
    );
    println!("{:-<80}", "");
    println!(
        "{:>5} {:>20} {:>15} {:>12} {:>10}",
        "Deal", "Est. Nodes", "Stderr", "Log10", "Time"
    );
    println!("{:-<80}", "");

    let mut all_estimates = Vec::new();
    let total_start = Instant::now();

    for i in 0..num_deals {
        let state = deal_with_contract(rng);

        let start = Instant::now();
        let (est, stderr) = estimate_tree_size(&state, num_samples, rng);
        let elapsed = start.elapsed();

        println!(
            "{:>5} {:>20.0} {:>15.0} {:>12.1} {:>9.2}s",
            i + 1,
            est,
            stderr,
            est.log10(),
            elapsed.as_secs_f64()
        );

        all_estimates.push(est);
    }

    let total_elapsed = total_start.elapsed();
    println!("{:-<80}", "");
    println!();

    all_estimates.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let mean = all_estimates.iter().sum::<f64>() / num_deals as f64;
    let median = all_estimates[num_deals / 2];

    println!(
        "Summary ({} deals, total {:.1}s):",
        num_deals,
        total_elapsed.as_secs_f64()
    );
    println!("  Mean nodes:   {:.2e} (10^{:.1})", mean, mean.log10());
    println!("  Median nodes: {:.2e} (10^{:.1})", median, median.log10());
    println!(
        "  Min:          {:.2e} (10^{:.1})",
        all_estimates[0],
        all_estimates[0].log10()
    );
    println!(
        "  Max:          {:.2e} (10^{:.1})",
        all_estimates[num_deals - 1],
        all_estimates[num_deals - 1].log10()
    );
}

fn run_exact(num_deals: usize, rng: &mut impl Rng, depth_arg: Option<&String>) {
    let depth_limit: u32 = depth_arg.and_then(|s| s.parse().ok()).unwrap_or(32);

    println!(
        "Enumerating play-phase game trees for {} deals (depth limit={})",
        num_deals, depth_limit
    );
    println!("{:-<80}", "");
    println!(
        "{:>5} {:>15} {:>15} {:>8} {:>10} {:>10}",
        "Deal", "Nodes", "Leaves", "Depth", "Avg BF", "Time"
    );
    println!("{:-<80}", "");

    let mut all_nodes = Vec::new();
    let mut all_leaves = Vec::new();
    let mut all_depths = Vec::new();
    let mut all_bfs = Vec::new();

    let total_start = Instant::now();

    // Progress thread — prints to stderr every 5s
    let progress_handle = std::thread::spawn(|| {
        let mut last = 0u64;
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            if STOP.load(Ordering::Relaxed) {
                break;
            }
            let current = NODE_COUNTER.load(Ordering::Relaxed);
            let rate = (current - last) / 5;
            eprintln!(
                "  ... {} nodes ({}/s)",
                format_num(current),
                format_num(rate)
            );
            last = current;
        }
    });

    for i in 0..num_deals {
        if STOP.load(Ordering::Relaxed) {
            println!("\nInterrupted.");
            break;
        }

        let state = deal_with_contract(rng);

        NODE_COUNTER.store(0, Ordering::Relaxed);
        let start = Instant::now();
        let stats = enumerate(&state, depth_limit);
        let elapsed = start.elapsed();

        let avg_bf = if stats.nodes > stats.leaves {
            (stats.nodes - 1) as f64 / (stats.nodes - stats.leaves) as f64
        } else {
            0.0
        };

        println!(
            "{:>5} {:>15} {:>15} {:>8} {:>10.2} {:>9.1}s",
            i + 1,
            format_num(stats.nodes),
            format_num(stats.leaves),
            stats.max_depth,
            avg_bf,
            elapsed.as_secs_f64()
        );

        all_nodes.push(stats.nodes);
        all_leaves.push(stats.leaves);
        all_depths.push(stats.max_depth);
        all_bfs.push(avg_bf);
    }

    STOP.store(true, Ordering::Relaxed);
    let _ = progress_handle.join();

    if all_nodes.is_empty() {
        return;
    }

    let total_elapsed = total_start.elapsed();
    let n = all_nodes.len() as u64;

    println!("{:-<80}", "");
    println!();
    println!(
        "Summary ({} deals, total {:.1}s):",
        all_nodes.len(),
        total_elapsed.as_secs_f64()
    );
    println!(
        "  Nodes:  min={}, max={}, mean={}",
        format_num(*all_nodes.iter().min().unwrap()),
        format_num(*all_nodes.iter().max().unwrap()),
        format_num(all_nodes.iter().sum::<u64>() / n)
    );
    println!(
        "  Leaves: min={}, max={}, mean={}",
        format_num(*all_leaves.iter().min().unwrap()),
        format_num(*all_leaves.iter().max().unwrap()),
        format_num(all_leaves.iter().sum::<u64>() / n)
    );
    println!(
        "  Depth:  min={}, max={}, mean={:.1}",
        all_depths.iter().min().unwrap(),
        all_depths.iter().max().unwrap(),
        all_depths.iter().sum::<u32>() as f64 / n as f64
    );
    println!(
        "  Avg BF: min={:.2}, max={:.2}, mean={:.2}",
        all_bfs.iter().cloned().reduce(f64::min).unwrap(),
        all_bfs.iter().cloned().reduce(f64::max).unwrap(),
        all_bfs.iter().sum::<f64>() / n as f64
    );
}

fn run_profile(num_deals: usize, rng: &mut impl Rng) {
    let num_paths = 10_000usize;
    let paths_per_deal = (num_paths / num_deals).max(1);

    println!(
        "Branching factor profile across {} deals ({} paths/deal)",
        num_deals, paths_per_deal
    );
    println!();

    let mut ply_counts: Vec<Vec<u32>> = vec![Vec::new(); 32];

    for _ in 0..num_deals {
        let state = deal_with_contract(rng);

        for _ in 0..paths_per_deal {
            let mut s = state;
            let mut ply = 0;
            while !s.is_terminal() {
                let mask = s.legal_actions();
                let count = mask.count_ones();
                ply_counts[ply].push(count);

                let idx = rng.gen_range(0..count);
                let action = select_nth_bit(mask, idx);
                s.step(action);
                ply += 1;
            }
        }
    }

    println!(
        "{:>4} {:>5} {:>7} {:>5} {:>8} {:>10}",
        "Ply", "Min", "Mean", "Max", "Samples", "Trick"
    );
    println!("{:-<55}", "");

    for ply in 0..32 {
        if ply_counts[ply].is_empty() {
            break;
        }
        let samples = &ply_counts[ply];
        let min = *samples.iter().min().unwrap();
        let max = *samples.iter().max().unwrap();
        let mean = samples.iter().sum::<u32>() as f64 / samples.len() as f64;
        let trick = ply / 4 + 1;
        let pos_label = ["lead", "2nd", "3rd", "4th"][ply % 4];

        println!(
            "{:>4} {:>5} {:>7.2} {:>5} {:>8} {:>6} {}",
            ply, min, mean, max,
            samples.len(),
            trick, pos_label
        );
    }
}

fn format_num(n: u64) -> String {
    let s = n.to_string();
    let mut result = String::new();
    for (i, c) in s.chars().rev().enumerate() {
        if i > 0 && i % 3 == 0 {
            result.push(',');
        }
        result.push(c);
    }
    result.chars().rev().collect()
}
