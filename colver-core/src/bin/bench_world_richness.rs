//! Are playgen's auction-conditioned worlds "richer" than the deal actually held?
//!
//! Continuation auctions run inside sampled worlds settle 12 points higher than the
//! same policy on real deals. Either the sampled hands support stronger contracts, or
//! the continuation code is wrong. This isolates the first: it compares the DD points
//! of the sampled worlds against the DD points of the true deal, for the very same
//! positions, using the points already stored in the labels.
//!
//! `best` = max over the 4 trumps of the points the *observer's team* would take. That
//! is what drives a bid, and a posterior can be well-centred on the mean while still
//! being biased in the max.
//!
//! Usage:
//!   cargo run --bin bench_world_richness --release --features parallel -- \
//!     --labels data/bid_labels3/shard_local.ql --games data/training/labelcorpus_120k.bin

use std::fs;

use colver_core::game_replay::GameReplay;
use colver_core::solver::{new_tt_buffer, solve_for_trump_reuse_tt};

const MAGIC: &[u8; 8] = b"COLVQL03";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut labels_path = String::from("data/bid_labels3/shard_local.ql");
    let mut games_path = String::from("data/training/labelcorpus_120k.bin");
    let mut limit: usize = 4000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--labels" => { i += 1; labels_path = args[i].clone(); }
            "--games" => { i += 1; games_path = args[i].clone(); }
            "--limit" => { i += 1; limit = args[i].parse().unwrap(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    let replays = GameReplay::load_all(&games_path).expect("load corpus");
    let data = fs::read(&labels_path).expect("read labels");
    assert_eq!(&data[..8], MAGIC);

    // Parse just enough records.
    struct Rec { game_idx: u32, observer: u8, pts: Vec<[u8; 4]> }
    let mut recs = Vec::new();
    let mut p = 16;
    while p + 8 <= data.len() && recs.len() < limit {
        let game_idx = u32::from_le_bytes(data[p..p + 4].try_into().unwrap());
        let observer = data[p + 6];
        let n = data[p + 7] as usize;
        p += 8;
        if n == 0 || p + n * 20 > data.len() { break; }
        let mut pts = Vec::with_capacity(n);
        for _ in 0..n {
            p += 16; // hands, not needed here
            pts.push([data[p], data[p + 1], data[p + 2], data[p + 3]]);
            p += 4;
        }
        recs.push(Rec { game_idx, observer, pts });
    }
    eprintln!("{} positions", recs.len());

    use rayon::prelude::*;
    let rows: Vec<(f64, f64, f64, f64)> = recs
        .par_iter()
        .map(|r| {
            let g = &replays[r.game_idx as usize];
            let mut tt = new_tt_buffer();
            let my_team = (r.observer & 1) as usize;
            let mine = |ns: u8| -> f64 {
                let ew = if ns == 252 || ns == 0 { 252 - ns } else { 162 - ns };
                if my_team == 0 { ns as f64 } else { ew as f64 }
            };

            // Truth: DD of the deal actually held.
            let mut truth = [0f64; 4];
            for s in 0..4usize {
                truth[s] = mine(solve_for_trump_reuse_tt(g.hands, g.dealer, s as u8, &mut tt)[0]);
            }
            let t_best = truth.iter().cloned().fold(f64::MIN, f64::max);
            let t_mean = truth.iter().sum::<f64>() / 4.0;

            // Worlds: already-solved DD points.
            let mut w_best = 0.0;
            let mut w_mean = 0.0;
            for row in &r.pts {
                let v: Vec<f64> = (0..4).map(|s| mine(row[s])).collect();
                w_best += v.iter().cloned().fold(f64::MIN, f64::max);
                w_mean += v.iter().sum::<f64>() / 4.0;
            }
            let k = r.pts.len() as f64;
            (t_best, w_best / k, t_mean, w_mean / k)
        })
        .collect();

    let n = rows.len() as f64;
    let tb = rows.iter().map(|r| r.0).sum::<f64>() / n;
    let wb = rows.iter().map(|r| r.1).sum::<f64>() / n;
    let tm = rows.iter().map(|r| r.2).sum::<f64>() / n;
    let wm = rows.iter().map(|r| r.3).sum::<f64>() / n;

    println!("\n=== Playgen world richness vs the true deal ({} positions) ===", rows.len());
    println!("                        true deal   playgen worlds   delta");
    println!("mean over 4 trumps      {tm:9.1}   {wm:14.1}   {:+.1}", wm - tm);
    println!("best of the 4 trumps    {tb:9.1}   {wb:14.1}   {:+.1}", wb - tb);
    println!("\n(points for the observer's team; 'best' is what a bidder acts on)");
}
