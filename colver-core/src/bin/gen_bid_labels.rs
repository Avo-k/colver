//! Generate auction-conditioned bid labels: COLVQL01.
//!
//! For a bid position taken from a COLVGM01 corpus, sample N complete deals from
//! playgen's posterior given the auction prefix, DD-solve each for all 4 trumps,
//! and record the raw per-world NS points.
//!
//! Raw per-world points are stored rather than a mean, because contract scoring is
//! non-linear in card points (réussi/chute is a threshold) — the expected reward of
//! a contract is `E[f(pts)]`, not `f(E[pts])`. Keeping the worlds lets the trainer
//! derive any target (per-contract Δ-winprob, raw points, chute probability) without
//! regenerating. Costs 4 bytes per world; the whole file stays tiny.
//!
//! Record layout (little-endian), after an 8-byte magic + u64 record count:
//!   game_idx u32 | prefix_len u16 | observer u8 | n_worlds u8
//!   then per world: hands[4] u32, ns_pts[4] u8
//!
//! The **hands** are stored, not just the points, so the trainer can replay the rest
//! of the auction inside each world with the real bidding policy. Without them a label
//! can only say "what if my team played this contract", which is not the value of
//! making a bid — the auction continues, partner may raise, opponents may overcall.
//! Belote (+20 for Q+K of trump) is a pure function of the hands and is derived at
//! load time rather than stored.
//!
//! Usage:
//!   cargo run --bin gen_bid_labels --release --features parallel -- \
//!     --games data/training/corpus.bin --output data/bid_labels/shard0.ql \
//!     --offset 0 --deals 25000 --per-deal 3 --worlds 16

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::game_replay::GameReplay;
use colver_core::playgen::infer::{PlaygenModel, PlaygenSampler};
use colver_core::solver::{new_tt_buffer, solve_for_trump_reuse_tt};
use colver_core::state::{GameState, Phase};

const MAGIC: &[u8; 8] = b"COLVQL03";

struct Record {
    game_idx: u32,
    prefix_len: u16,
    observer: u8,
    hands: Vec<[u32; 4]>,
    pts: Vec<[u8; 4]>,
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut games_path = String::from("data/training/heldout_20k_s90210.bin");
    let mut playgen_path = String::from("models/playgen/playgen_v2_final.bin");
    let mut output = String::from("data/bid_labels/labels.ql");
    let mut offset: usize = 0;
    let mut deals: usize = 25_000;
    let mut per_deal: usize = 3;
    let mut worlds: usize = 16;
    let mut temperature: f32 = 1.0;
    let mut seed: u64 = 1234;
    let mut threads: usize = 0;
    let mut chunk: usize = 2000;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => { i += 1; games_path = args[i].clone(); }
            "--playgen" => { i += 1; playgen_path = args[i].clone(); }
            "--output" => { i += 1; output = args[i].clone(); }
            "--offset" => { i += 1; offset = args[i].parse().unwrap(); }
            "--deals" => { i += 1; deals = args[i].parse().unwrap(); }
            "--per-deal" => { i += 1; per_deal = args[i].parse().unwrap(); }
            "--worlds" => { i += 1; worlds = args[i].parse().unwrap(); }
            "--temperature" => { i += 1; temperature = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--threads" => { i += 1; threads = args[i].parse().unwrap(); }
            "--chunk" => { i += 1; chunk = args[i].parse().unwrap(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }
    let nt = rayon::current_num_threads();

    eprintln!("Loading {games_path}...");
    let replays = GameReplay::load_all(&games_path).expect("load replays");
    let end = (offset + deals).min(replays.len());
    if offset >= replays.len() {
        eprintln!("offset {offset} past corpus end ({})", replays.len());
        std::process::exit(1);
    }
    let model = Arc::new(PlaygenModel::load(&playgen_path).expect("load playgen"));
    assert!(model.v2, "auction-conditioned sampling needs a v2 playgen model");
    eprintln!(
        "  corpus {} games, taking [{}, {}) | {} pos/deal | {} worlds | {} threads",
        replays.len(), offset, end, per_deal, worlds, nt
    );

    if let Some(parent) = std::path::Path::new(&output).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let mut out = BufWriter::new(File::create(&output).expect("create output"));
    out.write_all(MAGIC).unwrap();
    out.write_all(&0u64.to_le_bytes()).unwrap(); // record count, patched at the end

    let done = AtomicUsize::new(0);
    let dead = AtomicUsize::new(0);
    let empty = AtomicUsize::new(0);
    let start = Instant::now();
    let mut written: u64 = 0;

    use rayon::prelude::*;

    // Chunked so progress is durable: a crash keeps everything already flushed.
    let mut lo = offset;
    while lo < end {
        let hi = (lo + chunk).min(end);
        let batch: Vec<Record> = (lo..hi)
            .into_par_iter()
            .flat_map(|gi| {
                let r = &replays[gi];
                let mut tt = new_tt_buffer();
                let mut rng = StdRng::seed_from_u64(seed ^ (gi as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15));

                // Enumerate this game's bid positions.
                let state0 = GameState::new(r.dealer, r.hands);
                let mut s = state0;
                let mut bid_len = 0usize;
                for &a in &r.actions {
                    if s.phase != Phase::Bidding { break; }
                    bid_len += 1;
                    s.step(a);
                }
                if bid_len == 0 { return Vec::new(); }

                // Sample distinct cut points (a prefix of 0 is the opening bid).
                let mut cuts: Vec<usize> = (0..bid_len).collect();
                if cuts.len() > per_deal {
                    for k in 0..per_deal {
                        let j = k + rng.gen_range(0..(cuts.len() - k));
                        cuts.swap(k, j);
                    }
                    cuts.truncate(per_deal);
                }

                let mut recs = Vec::with_capacity(cuts.len());
                for cut in cuts {
                    // Observer is whoever is to move at the cut.
                    let mut st = state0;
                    for &a in r.actions.iter().take(cut) { st.step(a); }
                    if st.phase != Phase::Bidding { continue; }
                    let observer = st.current_player();

                    let mut sampler = PlaygenSampler::new(model.clone());
                    sampler.init_deal(&state0, observer);
                    let mut walk = state0;
                    for &a in r.actions.iter().take(cut) {
                        sampler.record_action(&walk, walk.current_player(), a);
                        walk.step(a);
                    }
                    if sampler.is_dead() {
                        dead.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    let ws = sampler.generate_deals_from_auction(&walk, worlds, temperature, &mut rng);
                    if ws.is_empty() {
                        empty.fetch_add(1, Ordering::Relaxed);
                        continue;
                    }

                    let mut pts = Vec::with_capacity(ws.len());
                    for w in &ws {
                        debug_assert_eq!(w[observer as usize], walk.hands[observer as usize]);
                        let mut row = [0u8; 4];
                        for suit in 0..4usize {
                            row[suit] =
                                solve_for_trump_reuse_tt(*w, r.dealer, suit as u8, &mut tt)[0];
                        }
                        pts.push(row);
                    }
                    recs.push(Record {
                        game_idx: gi as u32,
                        prefix_len: cut as u16,
                        observer,
                        hands: ws,
                        pts,
                    });
                }
                done.fetch_add(1, Ordering::Relaxed);
                recs
            })
            .collect();

        for rec in &batch {
            out.write_all(&rec.game_idx.to_le_bytes()).unwrap();
            out.write_all(&rec.prefix_len.to_le_bytes()).unwrap();
            out.write_all(&[rec.observer, rec.pts.len() as u8]).unwrap();
            for (h, row) in rec.hands.iter().zip(rec.pts.iter()) {
                for x in h {
                    out.write_all(&x.to_le_bytes()).unwrap();
                }
                out.write_all(row).unwrap();
            }
            written += 1;
        }
        out.flush().unwrap();

        let d = done.load(Ordering::Relaxed);
        let el = start.elapsed().as_secs_f64();
        let rate = d as f64 / el;
        let remaining = (end - offset).saturating_sub(d);
        eprintln!(
            "  {}/{} deals ({:.1}/s) {} records | {:.0}s elapsed, ETA {:.0} min",
            d, end - offset, rate, written, el, remaining as f64 / rate.max(1e-9) / 60.0
        );
        lo = hi;
    }

    // Patch the record count.
    out.flush().unwrap();
    drop(out);
    use std::io::{Seek, SeekFrom};
    let mut f = std::fs::OpenOptions::new().write(true).open(&output).unwrap();
    f.seek(SeekFrom::Start(8)).unwrap();
    f.write_all(&written.to_le_bytes()).unwrap();

    let el = start.elapsed().as_secs_f64();
    eprintln!(
        "\nDone: {} records from {} deals in {:.0}s | dead {} | empty {}",
        written,
        end - offset,
        el,
        dead.load(Ordering::Relaxed),
        empty.load(Ordering::Relaxed)
    );
    eprintln!("Wrote {output}");
}
