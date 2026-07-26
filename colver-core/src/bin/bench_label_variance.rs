//! How much of a bid label is determined by the hand the bidder can actually see?
//!
//! Takes real deals from the pool. For each, freezes one seat's 8 cards — the only
//! thing that seat holds at bid time — and redeals the other 24 cards at random,
//! DD-solving each redeal. The spread of that distribution is the noise a bid model
//! swallows when it is trained on the single deal that happened to be dealt.
//!
//! Usage:
//!   cargo run --bin bench_label_variance --release --features parallel -- \
//!     --deals 200 --samples 60

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use colver_core::bid_train_env::DealPool;
use colver_core::solver::{new_tt_buffer, solve_for_trump_reuse_tt};

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut pool_path = String::from("data/deals/base_5M.bin");
    let mut num_deals: usize = 200;
    let mut samples: usize = 60;
    let mut seat: u8 = 0;
    let mut seed: u64 = 7;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--deals" => { i += 1; num_deals = args[i].parse().unwrap(); }
            "--samples" => { i += 1; samples = args[i].parse().unwrap(); }
            "--seat" => { i += 1; seat = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    let pool = DealPool::load(&pool_path).expect("load pool");
    let n = num_deals.min(pool.len());
    eprintln!("{n} deals x {samples} redeals x 4 suits, seat {seat} frozen");

    use rayon::prelude::*;
    // Per deal, per suit: (true dd value, mean over redeals, sd over redeals)
    let rows: Vec<[(f64, f64, f64); 4]> = (0..n)
        .into_par_iter()
        .map(|idx| {
            let deal = pool.get(idx);
            let mut tt = new_tt_buffer();
            let mut rng = StdRng::seed_from_u64(seed + idx as u64);

            let mine = deal.hands[seat as usize];
            let others: Vec<u8> = (0..32u8).filter(|&c| mine & (1 << c) == 0).collect();

            let mut out = [(0.0, 0.0, 0.0); 4];
            for suit in 0..4usize {
                let truth = deal.dd_pts[suit] as f64;
                let mut vals = Vec::with_capacity(samples);
                let mut deck = others.clone();
                for _ in 0..samples {
                    deck.shuffle(&mut rng);
                    let mut hands = [0u32; 4];
                    hands[seat as usize] = mine;
                    let mut k = 0;
                    for s in 0..4u8 {
                        if s == seat { continue; }
                        for _ in 0..8 {
                            hands[s as usize] |= 1 << deck[k];
                            k += 1;
                        }
                    }
                    let pts = solve_for_trump_reuse_tt(hands, deal.dealer, suit as u8, &mut tt);
                    vals.push(pts[0] as f64);
                }
                let mean = vals.iter().sum::<f64>() / vals.len() as f64;
                let var = vals.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / vals.len() as f64;
                out[suit] = (truth, mean, var.sqrt());
            }
            out
        })
        .collect();

    // Aggregate: within-hand sd (noise from unseen cards) vs across-hand sd of the
    // conditional mean (the part the bidder's own hand actually explains).
    let mut within = 0.0f64;
    let mut means = Vec::new();
    let mut truths = Vec::new();
    let mut cnt = 0usize;
    for r in &rows {
        for suit in 0..4 {
            let (truth, mean, sd) = r[suit];
            within += sd * sd;
            means.push(mean);
            truths.push(truth);
            cnt += 1;
        }
    }
    let within_sd = (within / cnt as f64).sqrt();
    let gm = means.iter().sum::<f64>() / cnt as f64;
    let between_sd = (means.iter().map(|m| (m - gm).powi(2)).sum::<f64>() / cnt as f64).sqrt();
    let total_sd = (truths.iter().map(|t| (t - gm).powi(2)).sum::<f64>() / cnt as f64).sqrt();

    println!("\n=== Bid-label variance, seat {seat}, {n} deals x {samples} redeals ===");
    println!("Total sd of the raw label (dd_pts):        {total_sd:6.1} pts");
    println!("  explained by the visible hand (between): {between_sd:6.1} pts");
    println!("  noise from the 24 unseen cards (within): {within_sd:6.1} pts");
    println!(
        "  -> share of label variance the bidder CANNOT see: {:.0}%",
        within_sd.powi(2) / (within_sd.powi(2) + between_sd.powi(2)) * 100.0
    );
    println!(
        "\nAveraging {samples} worlds shrinks the noise to {:.1} pts (sd/sqrt(n)).",
        within_sd / (samples as f64).sqrt()
    );

    // A couple of concrete deals to make it tangible.
    println!("\nFirst 3 deals, per suit: true dd_pts vs mean +/- sd over redeals");
    for (idx, r) in rows.iter().take(3).enumerate() {
        print!("  deal {idx}: ");
        for suit in 0..4 {
            let (truth, mean, sd) = r[suit];
            print!("{}={:3.0} (mu{:5.1} sd{:4.1})  ", "SHDC".as_bytes()[suit] as char, truth, mean, sd);
        }
        println!();
    }
}
