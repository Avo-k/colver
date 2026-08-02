//! Does the transposition table thrash L3 when 32 threads each carry their own?
//!
//! `new_tt_buffer()` is 2 MB (1<<18 entries). Under a full 32-thread fan-out that is a 64 MB
//! working set against ~36 MB of L3 on this part. The solver masks with `tt.len() - 1`, so any
//! power-of-two buffer is legal — this sweeps the size and asks whether the ideal size at one
//! thread is still the ideal size at 32.
//!
//! **Two effects live in one number and have to be separated.** A bigger table collides less,
//! so it *prunes more* and visits fewer nodes — that is thread-independent and shows up in the
//! node count. A bigger table also spills out of cache, so each probe *costs more* — that only
//! shows up per-node, and only under contention. Hence three columns: nodes/solve (exact),
//! ns/node at 1 thread, ns/node at N. The L3 story, if there is one, is the gap between the
//! last two widening as the size grows.
//!
//! The TT is allocated **once per worker**, not once per deal: that is what production does
//! (`solve_*_reuse_tt` + the epoch stamp), and it is also the premise being tested — 32
//! resident tables. Allocating per deal would both measure the wrong thing and charge the
//! large sizes for a calloc the real workload never pays.
//!
//! Sizes are measured in the order given, so **repeating them interleaves the configurations**:
//! `--sizes 14,16,18,14,16,18` is the alternating A/B this repo requires before believing a
//! wall-clock gap, and costs nothing to ask for. The node column needs no such care.
//!
//! Usage:
//!   cargo run --bin bench_tt_size --release --features "parallel solver_stats" -- \
//!     --deals 300 --threads 32 --sizes 12,14,16,18,20,22

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

use colver_core::solver::{self, solve_for_trump_reuse_tt};
use colver_core::state::GameState;

/// One sweep at a fixed thread count. Returns (bits, nodes, ns_per_node, ms_per_solve, checksum).
fn sweep(
    hands: &[[u32; 4]],
    sizes: &[u32],
    threads: usize,
) -> Vec<(u32, u64, f64, f64, u64)> {
    use rayon::prelude::*;

    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(threads)
        .build()
        .unwrap();
    let n_solves = hands.len() * 4;
    let mut out = Vec::new();

    for &bits in sizes {
        let t0 = Instant::now();
        let (checksum, nodes): (u64, u64) = pool.install(|| {
            hands
                .par_iter()
                .map_init(
                    || solver::TtBuf::with_log2_size(bits),
                    |tt, h| {
                        let _ = solver::take_nodes();
                        let mut acc = 0u64;
                        for suit in 0..4u8 {
                            acc += solve_for_trump_reuse_tt(*h, 0, suit, tt)[0] as u64;
                        }
                        (acc, solver::take_nodes())
                    },
                )
                .reduce(|| (0, 0), |a, b| (a.0 + b.0, a.1 + b.1))
        });
        let wall = t0.elapsed().as_secs_f64();
        // Wall time is what the L3 question is about, so it stays wall time — but it is divided
        // by the thread count to give a per-core figure comparable to the 1-thread column.
        let ns_per_node = wall * 1e9 * threads as f64 / nodes as f64;
        let ms_per_solve = wall * 1000.0 * threads as f64 / n_solves as f64;
        out.push((bits, nodes, ns_per_node, ms_per_solve, checksum));
    }
    out
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut deals: usize = 300;
    let mut threads: usize = 32;
    let mut sizes: Vec<u32> = vec![12, 14, 16, 18, 20, 22];
    let mut seed: u64 = 3;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--deals" => { i += 1; deals = args[i].parse().unwrap(); }
            "--threads" => { i += 1; threads = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--sizes" => {
                i += 1;
                sizes = args[i].split(',').map(|s| s.parse().unwrap()).collect();
            }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    if !solver::stats_enabled() {
        eprintln!(
            "REFUSING: without --features solver_stats there is no node count, and a wall-clock \
             difference cannot be attributed to pruning rather than to cache."
        );
        std::process::exit(2);
    }

    // Fixed deal set, identical across every configuration and both thread counts.
    let mut rng = StdRng::seed_from_u64(seed);
    let hands: Vec<[u32; 4]> = (0..deals)
        .map(|_| GameState::deal_random(0, &mut rng).hands)
        .collect();

    println!("{} deals x 4 suits = {} full-deal solves per row\n", deals, deals * 4);

    // The 1-thread pass first: it is the uncontended reference every N-thread number is read
    // against, and running it first means it is the one *least* likely to be perturbed by the
    // other pass. Sequential runs of different configs would be unsafe to compare on time
    // alone — that is exactly why the node column is here.
    eprintln!("pass 1/2: {} sizes at 1 thread...", sizes.len());
    let one = sweep(&hands, &sizes, 1);
    eprintln!("pass 2/2: {} sizes at {} threads...", sizes.len(), threads);
    let many = sweep(&hands, &sizes, threads);

    println!(
        "{:<8} {:>9} {:>13} {:>11} {:>12} {:>12} {:>8} {:>12}",
        "TT bits", "per thread", "nodes/solve", "ms/solve 1T", "ns/node 1T",
        format!("ns/node {}T", threads), "ratio", format!("ms/solve {}T", threads)
    );
    for (i, &bits) in sizes.iter().enumerate() {
        let bytes = (1usize << bits) * 8;
        let size_str = if bytes >= 1 << 20 {
            format!("{} MB", bytes >> 20)
        } else {
            format!("{} KB", bytes >> 10)
        };
        assert_eq!(one[i].1, many[i].1, "node count must not depend on thread count");
        assert_eq!(one[i].4, many[i].4, "value must not depend on thread count");
        println!(
            "{:<8} {:>9} {:>13.0} {:>11.2} {:>12.1} {:>12.1} {:>8.2} {:>12.2}",
            bits,
            size_str,
            many[i].1 as f64 / (deals * 4) as f64,
            one[i].3,
            one[i].2,
            many[i].2,
            many[i].2 / one[i].2,
            many[i].3,
        );
    }

    // A single deal's four solves must give the same value whatever the table size: the TT is
    // a cache, never a source of truth. Guarding this is the whole reason a size sweep is safe.
    let base = one[0].4;
    for r in one.iter().chain(many.iter()) {
        assert_eq!(r.4, base, "TT size {} changed the answer — the table is not sound", r.0);
    }
    println!("\nvalue checksum {base} — identical for every size and both thread counts");
    println!(
        "ratio > 1 means a core gets less done per node under contention than alone: that is \
         the cache-pressure signal, and it should grow with size if the premise holds."
    );
}
