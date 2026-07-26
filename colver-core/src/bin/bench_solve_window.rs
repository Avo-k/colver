//! Can we exploit the fact that a bid label solves 40 *near-identical* worlds?
//!
//! Every world of one bid position shares the observer's 8 cards, so their DD
//! values cluster. A full-window search `[0, 252]` throws that away and rediscovers
//! the value from scratch each time. Here we seed a narrow window from the running
//! mean of the worlds already solved, and re-search on a wider window only when the
//! search fails high or low.
//!
//! Correctness is asserted, not assumed: every windowed result is compared against
//! the full-window value for the same world.
//!
//! Usage:
//!   cargo run --bin bench_solve_window --release --features parallel -- \
//!     --deals 40 --worlds 40 --delta 20

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use colver_core::bid_train_env::DealPool;
use colver_core::solver::{new_tt_buffer, solve_for_trump_reuse_tt, solve_for_trump_windowed};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut pool_path = String::from("data/deals/base_5M.bin");
    let mut deals: usize = 40;
    let mut worlds: usize = 40;
    let mut delta: i16 = 20;
    let mut seat: u8 = 0;
    let mut seed: u64 = 5;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--deals" => { i += 1; deals = args[i].parse().unwrap(); }
            "--worlds" => { i += 1; worlds = args[i].parse().unwrap(); }
            "--delta" => { i += 1; delta = args[i].parse().unwrap(); }
            "--seat" => { i += 1; seat = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    let pool = DealPool::load(&pool_path).expect("load pool");
    let n = deals.min(pool.len());
    eprintln!("{n} positions x {worlds} worlds x 4 suits | delta={delta}");

    let t_full = AtomicUsize::new(0); // nanos
    let t_win = AtomicUsize::new(0);
    let researches = AtomicUsize::new(0);
    let solves = AtomicUsize::new(0);
    let mismatches = AtomicUsize::new(0);

    use rayon::prelude::*;
    (0..n).into_par_iter().for_each(|idx| {
        let deal = pool.get(idx);
        let mut tt = new_tt_buffer();
        let mut rng = StdRng::seed_from_u64(seed + idx as u64);

        let mine = deal.hands[seat as usize];
        let unseen: Vec<u8> = (0..32u8).filter(|&c| mine & (1 << c) == 0).collect();

        // Build the world set once; both strategies solve the same worlds.
        let mut deck = unseen.clone();
        let mut world_set = Vec::with_capacity(worlds);
        for _ in 0..worlds {
            deck.shuffle(&mut rng);
            let mut hands = [0u32; 4];
            hands[seat as usize] = mine;
            let mut c = 0;
            for s in 0..4u8 {
                if s == seat { continue; }
                for _ in 0..8 {
                    hands[s as usize] |= 1 << deck[c];
                    c += 1;
                }
            }
            world_set.push(hands);
        }

        for suit in 0..4u8 {
            // ---- A: full window every time (what we do today).
            let t0 = Instant::now();
            let mut truth = Vec::with_capacity(worlds);
            for w in &world_set {
                truth.push(solve_for_trump_reuse_tt(*w, deal.dealer, suit, &mut tt)[0] as i16);
            }
            t_full.fetch_add(t0.elapsed().as_nanos() as usize, Ordering::Relaxed);

            // ---- B: narrow window seeded by the running mean, re-search on failure.
            let t1 = Instant::now();
            let mut sum: i32 = 0;
            let mut got = Vec::with_capacity(worlds);
            for (k, w) in world_set.iter().enumerate() {
                let v = if k == 0 {
                    solve_for_trump_reuse_tt(*w, deal.dealer, suit, &mut tt)[0] as i16
                } else {
                    let g = (sum / k as i32) as i16;
                    let (a, b) = ((g - delta).max(0), (g + delta).min(252));
                    let v = solve_for_trump_windowed(*w, deal.dealer, suit, &mut tt, a, b);
                    if v <= a || v >= b {
                        researches.fetch_add(1, Ordering::Relaxed);
                        solve_for_trump_reuse_tt(*w, deal.dealer, suit, &mut tt)[0] as i16
                    } else {
                        v
                    }
                };
                sum += v as i32;
                got.push(v);
                solves.fetch_add(1, Ordering::Relaxed);
            }
            t_win.fetch_add(t1.elapsed().as_nanos() as usize, Ordering::Relaxed);

            for (a, b) in truth.iter().zip(got.iter()) {
                if a != b {
                    mismatches.fetch_add(1, Ordering::Relaxed);
                }
            }
        }
    });

    let full = t_full.load(Ordering::Relaxed) as f64 / 1e9;
    let win = t_win.load(Ordering::Relaxed) as f64 / 1e9;
    let ns = solves.load(Ordering::Relaxed);
    let rs = researches.load(Ordering::Relaxed);
    let mm = mismatches.load(Ordering::Relaxed);

    println!("\n=== Windowed DD over clustered worlds (delta={delta}) ===");
    println!("solves:            {ns}");
    println!("full-window  CPU:  {full:8.1} s   ({:.2} ms/solve)", full * 1000.0 / ns as f64);
    println!("windowed     CPU:  {win:8.1} s   ({:.2} ms/solve)", win * 1000.0 / ns as f64);
    println!("speedup:           {:.2}x", full / win);
    println!("re-search rate:    {:.1}%", rs as f64 / ns as f64 * 100.0);
    println!("VALUE MISMATCHES:  {mm}   <- must be 0");
}
