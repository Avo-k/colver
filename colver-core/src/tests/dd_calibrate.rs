//! DD Calibration Tool: correlate heuristic scores with DD-achievable points.
//!
//! For each random deal, evaluates every player×suit combination:
//! - Heuristic score from `evaluate_for_trump`
//! - DD-optimal team points from the solver (with known hands)
//!
//! Buckets by heuristic score and computes success rates at each bid level.
//! Derives optimal thresholds and compares with current `improved_bid` mapping.
//!
//! Usage: cargo run --bin dd_calibrate --release -- [num_deals]
//! Default: 2000 deals

use std::env;
use std::time::Instant;

use colver_core::bid_eval::evaluate_for_trump;
use colver_core::card::*;
use colver_core::solver::{new_tt_buffer, solve_for_trump_reuse_tt};
use colver_core::state::GameState;

fn main() {
    let num_deals: usize = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(2000);

    println!("DD Calibration Tool");
    println!("===================");
    println!("Deals: {}", num_deals);
    println!();

    let start = Instant::now();
    let mut rng = rand::thread_rng();
    let mut tt = new_tt_buffer();

    // Data: (heuristic_score, team_dd_points, suit_idx, player)
    let mut data: Vec<(u16, u8, u8, u8)> = Vec::with_capacity(num_deals * 16);

    for deal_idx in 0..num_deals {
        let state = GameState::deal_random(0, &mut rng);
        let hands = state.hands;

        for player in 0..4u8 {
            let hand = hands[player as usize];
            let team = GameState::player_team(player);

            for suit_idx in 0..4u8 {
                let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
                let dd_result = solve_for_trump_reuse_tt(hands, 0, suit_idx, &mut tt);

                // Team points with optimal play
                let team_pts = dd_result[team as usize];
                data.push((score, team_pts, suit_idx, player));
            }
        }

        if (deal_idx + 1) % 500 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (deal_idx + 1) as f64 / elapsed;
            println!(
                "  [{}/{}] {:.1} deals/s, {:.0}s elapsed",
                deal_idx + 1,
                num_deals,
                rate,
                elapsed
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\nCompleted {} deals ({} evaluations) in {:.1}s ({:.1} deals/s)",
        num_deals,
        data.len(),
        elapsed,
        num_deals as f64 / elapsed
    );

    // Bucket analysis
    println!("\n{}", "=".repeat(90));
    println!("Score → DD Points Analysis");
    println!("{}", "=".repeat(90));

    // Define score buckets
    let buckets = [
        (0, 5, "0-4"),
        (5, 8, "5-7"),
        (8, 10, "8-9"),
        (10, 13, "10-12"),
        (13, 17, "13-16"),
        (17, 20, "17-19"),
        (20, 25, "20-24"),
        (25, 30, "25-29"),
        (30, 100, "30+"),
    ];

    println!(
        "{:<8} {:>6} {:>7} {:>7} {:>7} | {:>6} {:>6} {:>6} {:>6} {:>6} {:>6}",
        "Score", "Count", "Mean", "Median", "StdDev", "≥80", "≥90", "≥100", "≥110", "≥120", "≥130"
    );
    println!("{}", "-".repeat(90));

    for &(lo, hi, label) in &buckets {
        let mut pts: Vec<u8> = data
            .iter()
            .filter(|(s, _, _, _)| *s >= lo && *s < hi)
            .map(|(_, p, _, _)| *p)
            .collect();

        if pts.is_empty() {
            println!("{:<8} {:>6}", label, 0);
            continue;
        }

        pts.sort();
        let count = pts.len();
        let mean = pts.iter().map(|&p| p as f64).sum::<f64>() / count as f64;
        let median = pts[count / 2] as f64;
        let variance =
            pts.iter().map(|&p| (p as f64 - mean).powi(2)).sum::<f64>() / count as f64;
        let stddev = variance.sqrt();

        let pct = |threshold: u8| -> f64 {
            let above = pts.iter().filter(|&&p| p >= threshold).count();
            above as f64 / count as f64 * 100.0
        };

        println!(
            "{:<8} {:>6} {:>7.1} {:>7.1} {:>7.1} | {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}% {:>5.1}%",
            label,
            count,
            mean,
            median,
            stddev,
            pct(80),
            pct(90),
            pct(100),
            pct(110),
            pct(120),
            pct(130)
        );
    }

    // Optimal thresholds: find minimum score where success rate >= target
    println!("\n{}", "=".repeat(70));
    println!("Optimal Thresholds (minimum score for ≥70% success at bid level)");
    println!("{}", "=".repeat(70));

    let bid_levels = [
        (80u8, "80"),
        (90, "90"),
        (100, "100"),
        (110, "110"),
        (120, "120"),
        (130, "130"),
    ];
    let target_rates = [60.0, 65.0, 70.0, 75.0];

    println!(
        "{:<8} {:>10} {:>10} {:>10} {:>10}",
        "Level", "≥60%", "≥65%", "≥70%", "≥75%"
    );
    println!("{}", "-".repeat(50));

    for &(threshold, label) in &bid_levels {
        print!("{:<8}", label);
        for &target_rate in &target_rates {
            // Find minimum score where success rate >= target
            let mut best_score: Option<u16> = None;
            for score in 0..40u16 {
                let matching: Vec<u8> = data
                    .iter()
                    .filter(|(s, _, _, _)| *s >= score && *s < score + 3)
                    .map(|(_, p, _, _)| *p)
                    .collect();

                if matching.len() < 20 {
                    continue; // not enough data
                }

                let success_rate = matching.iter().filter(|&&p| p >= threshold).count() as f64
                    / matching.len() as f64
                    * 100.0;

                if success_rate >= target_rate {
                    best_score = Some(score);
                    break;
                }
            }

            match best_score {
                Some(s) => print!("{:>10}", s),
                None => print!("{:>10}", "-"),
            }
        }
        println!();
    }

    // Compare with current improved_bid thresholds
    println!("\n{}", "=".repeat(70));
    println!("Current improved_bid Thresholds vs DD-Optimal (70%)");
    println!("{}", "=".repeat(70));

    let current_thresholds: [(u8, u16, &str); 5] = [
        (80, 10, "80→10"),
        (90, 13, "90→13"),
        (100, 17, "100→17"),
        (110, 20, "110→20"),
        (120, 25, "120→25"),
    ];

    println!(
        "{:<12} {:>10} {:>12} {:>12}",
        "Bid Level", "Current", "DD @70%", "Difference"
    );
    println!("{}", "-".repeat(48));

    for &(threshold, current, label) in &current_thresholds {
        // Find DD-optimal at 70%
        let mut dd_optimal: Option<u16> = None;
        for score in 0..40u16 {
            let matching: Vec<u8> = data
                .iter()
                .filter(|(s, _, _, _)| *s >= score && *s < score + 3)
                .map(|(_, p, _, _)| *p)
                .collect();

            if matching.len() < 20 {
                continue;
            }

            let success_rate = matching.iter().filter(|&&p| p >= threshold).count() as f64
                / matching.len() as f64
                * 100.0;

            if success_rate >= 70.0 {
                dd_optimal = Some(score);
                break;
            }
        }

        let dd_str = match dd_optimal {
            Some(s) => format!("{}", s),
            None => "-".to_string(),
        };
        let diff = match dd_optimal {
            Some(s) => format!("{:+}", s as i32 - current as i32),
            None => "-".to_string(),
        };

        println!("{:<12} {:>10} {:>12} {:>12}", label, current, dd_str, diff);
    }

    // Success rate at current thresholds
    println!("\n{}", "=".repeat(70));
    println!("Success Rate of Current Thresholds");
    println!("{}", "=".repeat(70));

    for &(threshold, min_score, label) in &current_thresholds {
        let matching: Vec<u8> = data
            .iter()
            .filter(|(s, _, _, _)| *s >= min_score)
            .map(|(_, p, _, _)| *p)
            .collect();

        if matching.is_empty() {
            println!("{}: no data", label);
            continue;
        }

        let success = matching.iter().filter(|&&p| p >= threshold).count();
        let rate = success as f64 / matching.len() as f64 * 100.0;
        let mean = matching.iter().map(|&p| p as f64).sum::<f64>() / matching.len() as f64;

        println!(
            "{}: {:.1}% success ({}/{}, mean DD={:.1})",
            label,
            rate,
            success,
            matching.len(),
            mean
        );
    }

    // Suit distribution analysis
    println!("\n{}", "=".repeat(70));
    println!("Per-Suit Average DD Points (all deals)");
    println!("{}", "=".repeat(70));

    let suit_names = ["Spades", "Hearts", "Diamonds", "Clubs"];
    for suit_idx in 0..4u8 {
        let pts: Vec<u8> = data
            .iter()
            .filter(|(_, _, s, _)| *s == suit_idx)
            .map(|(_, p, _, _)| *p)
            .collect();

        let mean = pts.iter().map(|&p| p as f64).sum::<f64>() / pts.len() as f64;
        println!("  {}: mean DD = {:.1} pts", suit_names[suit_idx as usize], mean);
    }
}
