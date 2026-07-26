//! Does the transposition table thrash L3 when 32 threads each carry their own?
//!
//! `new_tt_buffer()` is 2 MB (1<<18 entries). Under a full 32-thread solve fan-out
//! that is a 64 MB working set, well past L3 on most desktop parts. The solver masks
//! with `tt.len() - 1`, so any power-of-two buffer is legal — this sweeps the size
//! at a fixed thread count and reports end-to-end solve throughput.
//!
//! Usage:
//!   cargo run --bin bench_tt_size --release --features parallel -- \
//!     --deals 400 --threads 32 --sizes 14,16,18,20

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

use colver_core::solver::solve_for_trump_reuse_tt;
use colver_core::state::GameState;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut deals: usize = 400;
    let mut threads: usize = 0;
    let mut sizes: Vec<u32> = vec![14, 16, 18, 20];
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

    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }
    let nt = rayon::current_num_threads();

    // Fixed deal set, identical across every configuration.
    let mut rng = StdRng::seed_from_u64(seed);
    let hands: Vec<[u32; 4]> = (0..deals)
        .map(|_| GameState::deal_random(0, &mut rng).hands)
        .collect();

    println!("{} deals x 4 suits = {} solves | {} threads\n", deals, deals * 4, nt);
    println!("{:<10} {:>10} {:>12} {:>14} {:>10}", "TT bits", "TT size", "elapsed", "ms/solve", "vs 2MB");

    use rayon::prelude::*;
    let mut baseline = 0.0f64;
    for &bits in &sizes {
        let n_entries = 1usize << bits;
        let bytes = n_entries * 8;

        let t0 = Instant::now();
        let checksum: u64 = hands
            .par_iter()
            .map(|h| {
                let mut tt = vec![0u64; n_entries];
                let mut acc = 0u64;
                for suit in 0..4u8 {
                    acc += solve_for_trump_reuse_tt(*h, 0, suit, &mut tt)[0] as u64;
                }
                acc
            })
            .sum();
        let el = t0.elapsed().as_secs_f64();
        let per = el * 1000.0 / (deals * 4) as f64;
        if bits == 18 {
            baseline = per;
        }
        let size_str = if bytes >= 1 << 20 {
            format!("{} MB", bytes >> 20)
        } else {
            format!("{} KB", bytes >> 10)
        };
        println!(
            "{:<10} {:>10} {:>11.2}s {:>13.2} {:>9}   (sum {})",
            bits, size_str, el, per,
            if baseline > 0.0 { format!("{:.2}x", baseline / per) } else { "-".into() },
            checksum
        );
    }
    println!("\n(checksum must be identical across rows — the TT never changes the value)");
}
