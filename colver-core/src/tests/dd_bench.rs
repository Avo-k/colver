use colver_core::solver;
use colver_core::state::GameState;
use rand::SeedableRng;
use std::time::Instant;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_deals: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(1000);

    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

    println!("Double-dummy solver benchmark");
    println!("=============================");
    println!();

    let mut total_solves = 0u64;
    let mut total_check_ok = 0u64;
    let mut total_check_fail = 0u64;
    let mut capot_count = [0u64; 4]; // per suit
    let mut ns_points_sum = [0u64; 4]; // per suit
    let mut solve_times_ms = Vec::new();

    let start = Instant::now();

    for deal_idx in 0..num_deals {
        let state = GameState::deal_random(0, &mut rng);
        let hands = state.hands;

        for trump in 0..4u8 {
            let t0 = Instant::now();
            let result = solver::solve_for_trump(hands, 0, trump);
            solve_times_ms.push(t0.elapsed().as_secs_f64() * 1000.0);

            let total = result[0] as u16 + result[1] as u16;

            if total == 162 || total == 252 {
                total_check_ok += 1;
            } else {
                total_check_fail += 1;
                if total_check_fail <= 5 {
                    eprintln!(
                        "FAIL deal {}, trump {}: NS={} EW={} total={}",
                        deal_idx, trump, result[0], result[1], total
                    );
                }
            }

            if total == 252 {
                capot_count[trump as usize] += 1;
            }

            ns_points_sum[trump as usize] += result[0] as u64;
            total_solves += 1;
        }

        // Progress report every 100 deals
        if (deal_idx + 1) % 100 == 0 {
            let elapsed = start.elapsed();
            let rate = (deal_idx + 1) as f64 / elapsed.as_secs_f64();
            eprint!("\r  {} deals ({:.0} deals/s)...", deal_idx + 1, rate);
        }
    }

    let elapsed = start.elapsed();
    eprintln!();
    println!();
    println!(
        "Solved {} deals x 4 suits = {} solves",
        num_deals, total_solves
    );
    println!(
        "  Total time: {:.2?} ({:.2}ms/deal, {:.2}ms/solve)",
        elapsed,
        elapsed.as_secs_f64() * 1000.0 / num_deals as f64,
        elapsed.as_secs_f64() * 1000.0 / total_solves as f64,
    );
    println!(
        "  Point total check: {}/{} OK",
        total_check_ok, total_solves
    );
    if total_check_fail > 0 {
        println!("  FAILURES: {} !!!", total_check_fail);
    }

    let suit_names = ["Spades", "Hearts", "Diamonds", "Clubs"];
    println!();
    println!("  Per-suit avg NS points:");
    for s in 0..4 {
        let avg = ns_points_sum[s] as f64 / num_deals as f64;
        let capot_pct = capot_count[s] as f64 / num_deals as f64 * 100.0;
        println!(
            "    {:>8}: {:5.1}  (capot: {:.1}%)",
            suit_names[s], avg, capot_pct
        );
    }

    // Timing distribution
    solve_times_ms.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = solve_times_ms.len();
    if n > 0 {
        let sum: f64 = solve_times_ms.iter().sum();
        println!();
        println!("  Solve time distribution:");
        println!("    Min:    {:.2}ms", solve_times_ms[0]);
        println!("    P25:    {:.2}ms", solve_times_ms[n / 4]);
        println!("    Median: {:.2}ms", solve_times_ms[n / 2]);
        println!("    P75:    {:.2}ms", solve_times_ms[3 * n / 4]);
        println!("    P90:    {:.2}ms", solve_times_ms[9 * n / 10]);
        println!("    P95:    {:.2}ms", solve_times_ms[19 * n / 20]);
        println!("    Max:    {:.2}ms", solve_times_ms[n - 1]);

        let top10_start = 9 * n / 10;
        let top10_sum: f64 = solve_times_ms[top10_start..].iter().sum();
        println!(
            "    Top 10% take {:.1}ms ({:.1}% of total)",
            top10_sum,
            top10_sum / sum * 100.0
        );
    }
}
