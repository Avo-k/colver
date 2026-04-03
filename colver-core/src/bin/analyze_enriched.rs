//! Analyze an enriched DD pool (COLVDR01): DD pts vs real pts deltas.
//!
//! Usage: cargo run --bin analyze_enriched --release -- [path]

use std::io::Read;

fn main() {
    let path = std::env::args()
        .nth(1)
        .unwrap_or_else(|| "data/pools/dd_pool_enriched_1M.bin".to_string());

    // Load enriched pool
    let mut f = std::io::BufReader::new(std::fs::File::open(&path).unwrap());
    let mut magic = [0u8; 8];
    f.read_exact(&mut magic).unwrap();
    assert_eq!(&magic, b"COLVDR01", "Not an enriched pool file");

    let mut count_buf = [0u8; 8];
    f.read_exact(&mut count_buf).unwrap();
    let count = u64::from_le_bytes(count_buf) as usize;

    // Read all deals: 25 bytes each (dealer[1] + hands[16] + dd_pts[4] + real_pts[4])
    let mut data: Vec<(u8, u8)> = Vec::with_capacity(count * 4); // (dd, real) per suit

    for _ in 0..count {
        let mut buf = [0u8; 25];
        f.read_exact(&mut buf).unwrap();
        let dd_pts = &buf[17..21];
        let real_pts = &buf[21..25];
        for suit in 0..4 {
            data.push((dd_pts[suit], real_pts[suit]));
        }
    }

    println!("=== Enriched Pool Analysis: {} deals, {} suit evaluations ===\n", count, data.len());

    // Bucketed analysis
    let buckets: &[(u8, u8, &str)] = &[
        (0, 20, "0-19"),
        (20, 40, "20-39"),
        (40, 60, "40-59"),
        (60, 70, "60-69"),
        (70, 80, "70-79"),
        (80, 85, "80-84"),
        (85, 90, "85-89"),
        (90, 95, "90-94"),
        (95, 100, "95-99"),
        (100, 105, "100-104"),
        (105, 110, "105-109"),
        (110, 115, "110-114"),
        (115, 120, "115-119"),
        (120, 125, "120-124"),
        (125, 130, "125-129"),
        (130, 140, "130-139"),
        (140, 152, "140-151"),
        (152, 162, "152-161"),
        (162, 253, "162(capot)"),
    ];

    println!(
        "{:<12} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}",
        "DD Range", "Count", "DD Mean", "Real Mn", "Delta", "Ratio", "StdDev", "Median Δ"
    );
    println!("{}", "-".repeat(78));

    for &(lo, hi, label) in buckets {
        let bucket: Vec<_> = data.iter()
            .filter(|(dd, _)| *dd >= lo && *dd < hi)
            .collect();

        if bucket.is_empty() {
            continue;
        }

        let n = bucket.len() as f64;
        let dd_avg = bucket.iter().map(|(dd, _)| *dd as f64).sum::<f64>() / n;
        let real_avg = bucket.iter().map(|(_, real)| *real as f64).sum::<f64>() / n;
        let delta = real_avg - dd_avg;
        let ratio = if dd_avg > 0.0 { real_avg / dd_avg } else { 0.0 };

        // StdDev of delta
        let deltas: Vec<f64> = bucket.iter().map(|(dd, real)| *real as f64 - *dd as f64).collect();
        let var = deltas.iter().map(|d| (d - delta).powi(2)).sum::<f64>() / n;
        let stddev = var.sqrt();

        // Median delta
        let mut sorted_deltas: Vec<i16> = bucket.iter().map(|(dd, real)| *real as i16 - *dd as i16).collect();
        sorted_deltas.sort();
        let median_delta = sorted_deltas[sorted_deltas.len() / 2];

        println!(
            "{:<12} {:>8} {:>8.1} {:>8.1} {:>+8.1} {:>8.3} {:>8.1} {:>+8}",
            label, bucket.len(), dd_avg, real_avg, delta, ratio, stddev, median_delta
        );
    }

    // Percentile analysis of deltas
    println!("\n=== Delta Distribution (real - DD) ===");
    let mut all_deltas: Vec<i16> = data.iter().map(|(dd, real)| *real as i16 - *dd as i16).collect();
    all_deltas.sort();
    let n = all_deltas.len();
    println!("  P1:   {:+}", all_deltas[n / 100]);
    println!("  P5:   {:+}", all_deltas[n * 5 / 100]);
    println!("  P10:  {:+}", all_deltas[n * 10 / 100]);
    println!("  P25:  {:+}", all_deltas[n * 25 / 100]);
    println!("  P50:  {:+}", all_deltas[n / 2]);
    println!("  P75:  {:+}", all_deltas[n * 75 / 100]);
    println!("  P90:  {:+}", all_deltas[n * 90 / 100]);
    println!("  P95:  {:+}", all_deltas[n * 95 / 100]);
    println!("  P99:  {:+}", all_deltas[n * 99 / 100]);
    println!("  Mean: {:+.1}", all_deltas.iter().map(|d| *d as f64).sum::<f64>() / n as f64);

    // Contract-relevant analysis: for bids at each level,
    // what's the expected delta and variance?
    println!("\n=== Contract-Level View ===");
    println!("If NN bids X in suit where DD=Y, what's the real outcome?");
    println!(
        "{:<10} {:>8} {:>10} {:>10} {:>10} {:>10}",
        "Contract", "Count", "DD Mean", "Real Mean", "Avg Delta", "P(make)"
    );
    println!("{}", "-".repeat(60));

    for threshold in [80u8, 90, 100, 110, 120, 130, 140, 150, 160] {
        // Deals where DD >= threshold (would be bid at this level)
        let qualifying: Vec<_> = data.iter()
            .filter(|(dd, _)| *dd >= threshold)
            .collect();

        if qualifying.is_empty() {
            continue;
        }

        let n = qualifying.len() as f64;
        let dd_avg = qualifying.iter().map(|(dd, _)| *dd as f64).sum::<f64>() / n;
        let real_avg = qualifying.iter().map(|(_, real)| *real as f64).sum::<f64>() / n;
        let delta = real_avg - dd_avg;
        let makes = qualifying.iter().filter(|(_, real)| *real >= threshold).count();
        let make_pct = makes as f64 / n * 100.0;

        println!(
            "≥{:<9} {:>8} {:>10.1} {:>10.1} {:>+10.1} {:>9.1}%",
            threshold, qualifying.len(), dd_avg, real_avg, delta, make_pct
        );
    }

    // "Chute" analysis: when contract fails, how badly?
    println!("\n=== Chute Severity ===");
    println!("When DD says ≥ threshold but real < threshold:");
    println!(
        "{:<10} {:>8} {:>10} {:>10} {:>10}",
        "Contract", "Chutes", "Avg Short", "Med Short", "Worst 5%"
    );
    println!("{}", "-".repeat(52));

    for threshold in [80u8, 90, 100, 110, 120, 130] {
        let chutes: Vec<i16> = data.iter()
            .filter(|(dd, real)| *dd >= threshold && *real < threshold)
            .map(|(_, real)| threshold as i16 - *real as i16)
            .collect();

        if chutes.is_empty() {
            continue;
        }

        let n = chutes.len();
        let avg_short = chutes.iter().map(|s| *s as f64).sum::<f64>() / n as f64;
        let mut sorted = chutes.clone();
        sorted.sort();
        let median = sorted[n / 2];
        let p95 = sorted[n * 95 / 100];

        println!(
            "≥{:<9} {:>8} {:>10.1} {:>10} {:>10}",
            threshold, n, avg_short, median, p95
        );
    }
}
