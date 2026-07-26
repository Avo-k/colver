//! Can a cheap heuristic pick which trumps are worth DD-solving?
//!
//! Labelling every bid position for all 4 trumps is the dominant cost of an
//! auction-conditioned label set. `dd_bid.rs` already prefilters candidate suits
//! with `evaluate_for_trump` before spending solves; this measures how much that
//! prefilter actually costs in label quality.
//!
//! Two things are reported per top-k:
//!   recall  — how often the truly best trump survives the filter
//!   EV loss — points given up by taking the best *surviving* suit instead of the
//!             best suit overall (the quantity a bidder actually cares about)
//!
//! Truth is a fresh DD solve of all 4 suits, not the pool's stored dd_pts (stale
//! since the quick_tricks removal).
//!
//! Usage:
//!   cargo run --bin bench_suit_prefilter --release --features parallel -- --deals 4000

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::bid_eval::evaluate_for_trump;
use colver_core::card::Suit;
use colver_core::game_replay::GameReplay;
use colver_core::solver::{new_tt_buffer, solve_for_trump_reuse_tt};
use colver_core::state::GameState;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut games_path = String::from("data/training/heldout_20k_s90210.bin");
    let mut deals: usize = 4000;
    let mut seed: u64 = 21;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => { i += 1; games_path = args[i].clone(); }
            "--deals" => { i += 1; deals = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    let replays = GameReplay::load_all(&games_path).expect("load replays");
    let n = deals.min(replays.len());
    eprintln!("{n} deals x 4 seats x 4 suits, fresh DD solves");

    use rand::Rng;
    use rayon::prelude::*;

    // Per (deal, seat): heuristic ranking + true DD points for all 4 suits.
    let rows: Vec<([u16; 4], [i16; 4])> = (0..n)
        .into_par_iter()
        .flat_map(|idx| {
            let r = &replays[idx];
            let mut tt = new_tt_buffer();
            let mut truth = [0i16; 4];
            for suit in 0..4usize {
                // NS points; convert to "points for the seat's team" below.
                truth[suit] =
                    solve_for_trump_reuse_tt(r.hands, r.dealer, suit as u8, &mut tt)[0] as i16;
            }
            (0..4u8)
                .map(|seat| {
                    let hand = r.hands[seat as usize];
                    let mut heur = [0u16; 4];
                    for suit in 0..4usize {
                        heur[suit] = evaluate_for_trump(hand, Suit::from_u8(suit as u8));
                    }
                    // Team 0 = seats 0,2 read NS points directly; team 1 reads the complement.
                    let mut t = [0i16; 4];
                    for suit in 0..4 {
                        t[suit] = if seat % 2 == 0 {
                            truth[suit]
                        } else {
                            let tot = if truth[suit] == 252 || truth[suit] == 0 { 252 } else { 162 };
                            tot - truth[suit]
                        };
                    }
                    (heur, t)
                })
                .collect::<Vec<_>>()
        })
        .collect();

    println!("\n=== Heuristic suit prefilter vs fresh DD ({} hands) ===", rows.len());
    println!("{:<22} {:>9} {:>12} {:>12}", "candidate set", "recall", "mean EV loss", "P90 EV loss");

    let mut rng = StdRng::seed_from_u64(seed);

    for k in 1..=3usize {
        for eps in [0.0f64, 0.10] {
            let mut hits = 0usize;
            let mut losses: Vec<f64> = Vec::with_capacity(rows.len());
            for (heur, truth) in &rows {
                // Rank suits by heuristic score, descending.
                let mut order: Vec<usize> = (0..4).collect();
                order.sort_by_key(|&s| std::cmp::Reverse(heur[s]));
                let mut cand: Vec<usize> = order[..k].to_vec();
                // Exploration: with prob eps add one uniformly random other suit.
                if eps > 0.0 && rng.gen_bool(eps) {
                    let extra = order[k..][rng.gen_range(0..(4 - k))];
                    cand.push(extra);
                }
                let best_all = (0..4).map(|s| truth[s]).max().unwrap();
                let best_cand = cand.iter().map(|&s| truth[s]).max().unwrap();
                if best_cand == best_all {
                    hits += 1;
                }
                losses.push((best_all - best_cand) as f64);
            }
            let recall = hits as f64 / rows.len() as f64 * 100.0;
            let mean = losses.iter().sum::<f64>() / losses.len() as f64;
            let mut sorted = losses.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let p90 = sorted[(sorted.len() as f64 * 0.90) as usize];
            let label = if eps > 0.0 {
                format!("top-{k} + {:.0}% random", eps * 100.0)
            } else {
                format!("top-{k}")
            };
            println!("{:<22} {:>8.1}% {:>11.1} {:>12.1}", label, recall, mean, p90);
        }
    }
    println!("\nSolve budget: top-1 = 1/4 of the cost, top-2 = 1/2, top-3 = 3/4.");
}
