//! Build the DD → IS-DD point calibration table.
//!
//! v6's reward comes from `scores_isdd_5M.sc` — points actually taken when IS-DD
//! plays the deal — while auction-conditioned labels are cheap DD solves. DD assumes
//! perfect play and is optimistic for the taker, so distilling DD-derived targets into
//! a net trained on IS-DD returns makes it over-bid. This tabulates `E[isdd | dd]`
//! over the 5M pool so the labels can be mapped onto the scale v6 was trained on.
//!
//! Output is a plain text table, one line per DD value 0..=252:
//!   dd_value  count  q0 q1 ... q63
//!
//! The full conditional **distribution** (64 quantiles), not its mean. Contract scoring
//! is a threshold in card points, so replacing a sample by `E[isdd | dd]` collapses the
//! variance the threshold depends on — that is `f(E[Y])` where the label needs `E[f(Y)]`.
//! Callers must draw a quantile, never read a mean.
//!
//! Both inputs predate the quick_tricks fix, so the mapping is a coarse statistical
//! relation rather than ground truth — good enough to test whether the scale mismatch
//! is what breaks the distillation.
//!
//! Usage:
//!   cargo run --bin gen_dd_calibration --release -- \
//!     --pool data/deals/base_5M.bin --scores data/deals/scores_isdd_5M.sc \
//!     --output data/deals/dd_to_isdd.calib

use std::io::Write;

use colver_core::bid_train_env::DealPool;

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut pool_path = String::from("data/deals/base_5M.bin");
    let mut scores_path = String::from("data/deals/scores_isdd_5M.sc");
    let mut output = String::from("data/deals/dd_to_isdd.calib");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--scores" => { i += 1; scores_path = args[i].clone(); }
            "--output" => { i += 1; output = args[i].clone(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    eprintln!("Loading {pool_path}...");
    let mut pool = DealPool::load(&pool_path).expect("load pool");
    eprintln!("  {} deals", pool.len());
    eprintln!("Loading {scores_path}...");
    pool.load_scores(&scores_path).expect("load scores");
    pool.select_score_layer(Some("isdd"));

    let mut hist: Vec<Vec<u32>> = vec![vec![0u32; 253]; 253];
    let mut sum = vec![0f64; 253];
    let mut cnt = vec![0u64; 253];
    let mut covered = 0usize;
    for idx in 0..pool.len() {
        let d = pool.get(idx);
        let Some(real) = d.real_pts else { continue };
        covered += 1;
        for suit in 0..4 {
            let dd = d.dd_pts[suit] as usize;
            sum[dd] += real[suit] as f64;
            cnt[dd] += 1;
            hist[dd][real[suit] as usize] += 1;
        }
    }
    eprintln!("  {covered} deals carry an IS-DD layer");
    assert!(covered > 1000, "score layer covers almost nothing");

    // Smooth sparse bins by falling back to the identity, so rare DD values do not
    // get a wild mean from a handful of samples.
    const NQ: usize = 64;
    let mut f = std::fs::File::create(&output).expect("create output");
    writeln!(f, "# dd_value count q0..q{}", NQ - 1).unwrap();
    let mut shown = Vec::new();
    for dd in 0..=252usize {
        let c = cnt[dd];
        let mut qs = [0u16; NQ];
        if c >= 30 {
            // Walk the histogram once, emitting the value at each quantile cut.
            let mut acc = 0u64;
            let mut qi = 0usize;
            for v in 0..253usize {
                acc += hist[dd][v] as u64;
                while qi < NQ && (acc as f64) >= (qi as f64 + 0.5) / NQ as f64 * c as f64 {
                    qs[qi] = v as u16;
                    qi += 1;
                }
            }
            while qi < NQ { qs[qi] = 252; qi += 1; }
        } else {
            for q in qs.iter_mut() { *q = dd as u16; }
        }
        write!(f, "{dd} {c}").unwrap();
        for q in qs.iter() { write!(f, " {q}").unwrap(); }
        writeln!(f).unwrap();
        if dd % 20 == 0 {
            let m = if c >= 30 { sum[dd] / c as f64 } else { dd as f64 };
            shown.push((dd, m, c, qs[6], qs[32], qs[57]));
        }
    }
    eprintln!("\n  dd -> isdd   mean, and p10 / median / p90 of the conditional");
    for (dd, m, c, p10, p50, p90) in shown {
        eprintln!("  {dd:>4} -> mean {m:6.1}   p10 {p10:3}  p50 {p50:3}  p90 {p90:3}   (n={c})");
    }
    eprintln!("\nWrote {output}");
}
