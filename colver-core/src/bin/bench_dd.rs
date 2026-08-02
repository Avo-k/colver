//! DD solver benchmark **and** exactness gate.
//!
//! Why this exists: before 2026-08-02 nothing in the repo measured `solve_with_scores`, the
//! unit of the workload that dominates DD CPU in this project (IS-DD determinization, ~2800
//! core-h for a 5M score layer, against ~180 for `gen_pool`). `dd_bench` measures full-deal
//! `solve_for_trump` only — roughly 6 % of the real cost. And four different documents quote
//! 13.5 / 14.9 / 28 / 77 ms for things all called "a solve".
//!
//! Two jobs, deliberately in one binary so they cannot drift apart:
//!
//!   1. **Benchmark.** Node counts are the primary metric (exact, immune to the P/E-core
//!      scheduling noise of a 13900K under WSL2); wall time is secondary and always reported
//!      with its thread count.
//!   2. **Exactness gate.** Every run writes every per-card value it computed. `diff` compares
//!      two such files. This is the only thing standing between an optimisation and another
//!      `quick_tricks` — which was measured with a harness that was never committed, which is
//!      precisely why its 25 %-wrong verdict could not be re-derived later.
//!
//! The corpus is a **file written once and kept**, never a seed replayed: `gen_pool` hands
//! slot indices out of an `AtomicUsize` to N workers, so no seeded generator in this repo is
//! reproducible across runs or machines.
//!
//! Usage:
//!   cargo run --release --features "parallel solver_stats" --bin bench_dd -- build \
//!       --out data/analysis/dd_corpus_v1.bin
//!   cargo run --release --features "parallel solver_stats" --bin bench_dd -- run \
//!       --corpus data/analysis/dd_corpus_v1.bin --values baseline.vals --json baseline.json
//!   cargo run --release --features "parallel solver_stats" --bin bench_dd -- diff \
//!       --a baseline.vals --b candidate.vals

use std::fs;
use std::io::{self, Write};
use std::time::Instant;

use colver_core::bid_train_env::DealPool;
use colver_core::card::*;
use colver_core::game_replay::GameReplay;
use colver_core::play;
use colver_core::solver;
use colver_core::state::{GameState, Phase};

use rand::rngs::StdRng;
use rand::SeedableRng;

const CORPUS_MAGIC: &[u8; 8] = b"COLVDDC1";
const VALUES_MAGIC: &[u8; 8] = b"COLVDDV1";

#[derive(Clone, Copy, PartialEq, Eq, Debug)]
#[repr(u8)]
enum Shape {
    /// Full 8-card deal, empty trick. The `gen_pool` shape — the regression guard.
    Full = 0,
    /// Real played position, 24..13 cards left.
    Mid = 1,
    /// Real played position, 12..2 cards left. Where the fixed per-solve cost stops being free.
    End = 2,
    /// Determinized worlds of one observer position: the IS-DD unit.
    Worlds = 3,
}

impl Shape {
    fn from_u8(v: u8) -> Shape {
        match v {
            0 => Shape::Full,
            1 => Shape::Mid,
            2 => Shape::End,
            _ => Shape::Worlds,
        }
    }
    fn name(self) -> &'static str {
        match self {
            Shape::Full => "full",
            Shape::Mid => "mid",
            Shape::End => "end",
            Shape::Worlds => "worlds",
        }
    }
}

/// One benchmark position, stored as "deal + the cards played to get here" rather than as a
/// serialized `GameState`: the replay re-validates legality on load, and the file survives any
/// change to the state layout.
#[derive(Clone)]
struct Position {
    shape: Shape,
    dealer: u8,
    trump: u8,
    hands: [CardSet; 4],
    played: Vec<u8>,
}

impl Position {
    /// Rebuild the state, asserting every recorded card was legal when it was played.
    fn rebuild(&self) -> Result<GameState, String> {
        let mut st = GameState::setup_dd(self.dealer, self.hands, self.trump);
        for (i, &c) in self.played.iter().enumerate() {
            if st.is_terminal() {
                return Err(format!("terminal after {i} cards, {} recorded", self.played.len()));
            }
            let legal = play::legal_plays(&st);
            if legal & card_to_bit(c) == 0 {
                return Err(format!("card {} illegal at ply {i}", card_name(c)));
            }
            play::apply_play(&mut st, c);
        }
        if st.is_terminal() {
            return Err("position is terminal".into());
        }
        Ok(st)
    }

    fn cards_left(&self) -> usize {
        32 - self.played.len()
    }
}

// ---------------------------------------------------------------- corpus io

fn write_corpus(path: &str, positions: &[Position]) -> io::Result<()> {
    if let Some(p) = std::path::Path::new(path).parent() {
        fs::create_dir_all(p)?;
    }
    let mut out = Vec::with_capacity(positions.len() * 48);
    out.extend_from_slice(CORPUS_MAGIC);
    out.extend_from_slice(&1u32.to_le_bytes());
    out.extend_from_slice(&(positions.len() as u32).to_le_bytes());
    for p in positions {
        out.push(p.shape as u8);
        out.push(p.dealer);
        out.push(p.trump);
        out.push(p.played.len() as u8);
        for h in &p.hands {
            out.extend_from_slice(&h.to_le_bytes());
        }
        out.extend_from_slice(&p.played);
    }
    fs::write(path, out)
}

fn read_corpus(path: &str) -> io::Result<Vec<Position>> {
    let data = fs::read(path)?;
    if data.len() < 16 || &data[..8] != CORPUS_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad corpus magic"));
    }
    let count = u32::from_le_bytes(data[12..16].try_into().unwrap()) as usize;
    let mut pos = Vec::with_capacity(count);
    let mut o = 16usize;
    for _ in 0..count {
        let shape = Shape::from_u8(data[o]);
        let dealer = data[o + 1];
        let trump = data[o + 2];
        let nplayed = data[o + 3] as usize;
        o += 4;
        let mut hands = [0u32; 4];
        for h in hands.iter_mut() {
            *h = u32::from_le_bytes(data[o..o + 4].try_into().unwrap());
            o += 4;
        }
        let played = data[o..o + nplayed].to_vec();
        o += nplayed;
        pos.push(Position { shape, dealer, trump, hands, played });
    }
    Ok(pos)
}

/// Per-card values for one position, sorted by card so the file is order-independent.
type Vals = Vec<(u8, i16)>;

fn write_values(path: &str, vals: &[Vals]) -> io::Result<()> {
    if let Some(p) = std::path::Path::new(path).parent() {
        fs::create_dir_all(p)?;
    }
    let mut out = Vec::with_capacity(vals.len() * 20);
    out.extend_from_slice(VALUES_MAGIC);
    out.extend_from_slice(&(vals.len() as u32).to_le_bytes());
    for v in vals {
        out.push(v.len() as u8);
        for &(c, s) in v {
            out.push(c);
            out.extend_from_slice(&s.to_le_bytes());
        }
    }
    fs::write(path, out)
}

fn read_values(path: &str) -> io::Result<Vec<Vals>> {
    let data = fs::read(path)?;
    if data.len() < 12 || &data[..8] != VALUES_MAGIC {
        return Err(io::Error::new(io::ErrorKind::InvalidData, "bad values magic"));
    }
    let count = u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize;
    let mut out = Vec::with_capacity(count);
    let mut o = 12usize;
    for _ in 0..count {
        let n = data[o] as usize;
        o += 1;
        let mut v = Vec::with_capacity(n);
        for _ in 0..n {
            let c = data[o];
            let s = i16::from_le_bytes(data[o + 1..o + 3].try_into().unwrap());
            o += 3;
            v.push((c, s));
        }
        out.push(v);
    }
    Ok(out)
}

// ---------------------------------------------------------------- corpus build

/// Full deals from the base pool. Only `hands` + `dealer` are read: the pool's own `dd_pts`
/// predate both the `quick_tricks` removal and the 2026-08-01 legality widening, so they are
/// stale — but a deal distribution cannot go stale.
fn build_full(pool_path: &str, n_deals: usize, out: &mut Vec<Position>) -> Result<(), String> {
    let pool = DealPool::load(pool_path).map_err(|e| format!("{pool_path}: {e}"))?;
    let take = n_deals.min(pool.len());
    for i in 0..take {
        let d = pool.get(i);
        for trump in 0..4u8 {
            out.push(Position {
                shape: Shape::Full,
                dealer: d.dealer,
                trump,
                hands: d.hands,
                played: Vec::new(),
            });
        }
    }
    eprintln!("  full   : {} positions ({take} deals x 4 suits)", take * 4);
    Ok(())
}

/// Mid-game and endgame positions from **real played games** (COLVGM01), per the standing
/// rule that benchmarks use played positions rather than synthetic random ones: a position
/// reached by two bots differs systematically from one reached by random legal play.
fn build_from_games(
    games_path: &str,
    want_per_bucket: usize,
    out: &mut Vec<Position>,
) -> Result<(), String> {
    let replays = GameReplay::load_all(games_path).map_err(|e| format!("{games_path}: {e}"))?;
    // cards_left -> how many collected so far
    let mut got: [usize; 33] = [0; 33];
    let mut n_mid = 0usize;
    let mut n_end = 0usize;

    for r in &replays {
        let mut st = GameState::new(r.dealer, r.hands);
        let mut idx = 0usize;
        while idx < r.actions.len() && st.phase == Phase::Bidding {
            st.step(r.actions[idx]);
            idx += 1;
        }
        if st.phase != Phase::Playing || st.is_terminal() {
            continue; // void deal
        }
        let trump = st.contract.trump;
        let dealer = r.dealer;
        let mut played: Vec<u8> = Vec::new();

        // Replay the card phase, snapshotting one position per cards_left bucket.
        let mut probe = GameState::setup_dd(dealer, r.hands, trump);
        while idx < r.actions.len() && !probe.is_terminal() {
            let card = r.actions[idx];
            idx += 1;
            let legal = play::legal_plays(&probe);
            if legal & card_to_bit(card) == 0 {
                break; // logged game inconsistent with its own hands — skip the rest
            }
            let cards_left = 32 - played.len();
            let bucket_ok = (13..=24).contains(&cards_left) || (2..=12).contains(&cards_left);
            // Only snapshot real decisions: a forced card measures nothing.
            if bucket_ok && legal.count_ones() >= 2 && got[cards_left] < want_per_bucket {
                got[cards_left] += 1;
                let shape = if cards_left >= 13 { Shape::Mid } else { Shape::End };
                if shape == Shape::Mid { n_mid += 1 } else { n_end += 1 }
                out.push(Position {
                    shape,
                    dealer,
                    trump,
                    hands: r.hands,
                    played: played.clone(),
                });
            }
            play::apply_play(&mut probe, card);
            played.push(card);
        }
        if got.iter().skip(2).take(23).all(|&g| g >= want_per_bucket) {
            break;
        }
    }
    eprintln!("  mid    : {n_mid} positions (13..24 cards left, real games)");
    eprintln!("  end    : {n_end} positions (2..12 cards left, real games)");
    Ok(())
}

/// The IS-DD unit: one observer position, N determinized worlds sharing its visible prefix.
/// Uniform determinization, not playgen — a benchmark must not depend on a GPU sidecar being
/// up, and must be byte-reproducible from the corpus file.
fn build_worlds(
    games_path: &str,
    n_positions: usize,
    n_worlds: usize,
    out: &mut Vec<Position>,
) -> Result<(), String> {
    let replays = GameReplay::load_all(games_path).map_err(|e| format!("{games_path}: {e}"))?;
    let mut rng = StdRng::seed_from_u64(0xDDC0_1BEE);
    let mut made = 0usize;

    for (gi, r) in replays.iter().enumerate() {
        if made >= n_positions {
            break;
        }
        let mut st = GameState::new(r.dealer, r.hands);
        let mut idx = 0usize;
        while idx < r.actions.len() && st.phase == Phase::Bidding {
            st.step(r.actions[idx]);
            idx += 1;
        }
        if st.phase != Phase::Playing || st.is_terminal() {
            continue;
        }
        let trump = st.contract.trump;
        // Spread the observer positions over the whole deal: this is what one IS-DD game does.
        let target_plies = 4 + (gi % 7) * 4;
        let mut probe = GameState::setup_dd(r.dealer, r.hands, trump);
        let mut played: Vec<u8> = Vec::new();
        let mut ok = true;
        for _ in 0..target_plies {
            if idx >= r.actions.len() || probe.is_terminal() {
                ok = false;
                break;
            }
            let card = r.actions[idx];
            idx += 1;
            if play::legal_plays(&probe) & card_to_bit(card) == 0 {
                ok = false;
                break;
            }
            play::apply_play(&mut probe, card);
            played.push(card);
        }
        if !ok || probe.is_terminal() || play::legal_plays(&probe).count_ones() < 2 {
            continue;
        }

        let observer = probe.current_player;
        let mut got = 0usize;
        let mut tries = 0usize;
        while got < n_worlds && tries < n_worlds * 20 {
            tries += 1;
            if let Some(world) = colver_core::determinize::determinize(&probe, observer, &mut rng) {
                // `determinize` hands back the position with hidden hands resampled; recover
                // the *initial* hands by giving every seat back the cards it has played.
                let mut init = world.hands;
                let mut s = GameState::setup_dd(r.dealer, r.hands, trump);
                for &c in &played {
                    let seat = s.current_player as usize;
                    init[seat] |= card_to_bit(c);
                    play::apply_play(&mut s, c);
                }
                let cand = Position {
                    shape: Shape::Worlds,
                    dealer: r.dealer,
                    trump,
                    hands: init,
                    played: played.clone(),
                };
                // A world is only usable if it replays: reject rather than store a lie.
                if cand.rebuild().is_ok() {
                    out.push(cand);
                    got += 1;
                }
            }
        }
        if got > 0 {
            made += 1;
        }
    }
    eprintln!("  worlds : {} positions ({made} observer positions x {n_worlds} worlds)", made * n_worlds);
    Ok(())
}

// ---------------------------------------------------------------- run

fn solve_one(p: &Position, tt: &mut solver::TtBuf) -> (Vals, u64, f64) {
    let st = p.rebuild().expect("corpus position must rebuild");
    let _ = solver::take_nodes();
    let t = Instant::now();
    let sc = solver::solve_with_scores(&st, Some(tt));
    let us = t.elapsed().as_secs_f64() * 1e6;
    let nodes = solver::take_nodes();
    let mut v: Vals = sc.scores[..sc.count].to_vec();
    v.sort_unstable();
    (v, nodes, us)
}

fn cmd_run(args: &Args) -> io::Result<()> {
    let positions = read_corpus(&args.corpus)?;
    if !solver::stats_enabled() {
        eprintln!(
            "WARNING: built without --features solver_stats; node counts will be 0 and \
             wall-clock alone cannot separate pruning from core scheduling."
        );
    }
    eprintln!("corpus: {} positions from {}", positions.len(), args.corpus);
    eprintln!(
        "heuristics: {} (ablation switches {})",
        solver::ablation_label(),
        if solver::ablation_enabled() { "compiled in" } else { "compiled out" }
    );

    let threads = args.threads;
    let t_all = Instant::now();

    // With `--repeats N`, keep the FASTEST pass per position. Competing load on a shared
    // machine only ever adds time, so the minimum is the least-disturbed observation; a mean
    // would fold in whatever else the box was doing. Two single runs of the same binary
    // differed by 20 % here, which is larger than most of the wins being measured.
    let results: Vec<(Vals, u64, f64)> = if threads == 1 {
        let mut tt = solver::new_tt_buffer();
        positions.iter().map(|p| solve_one(p, &mut tt)).collect()
    } else {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            rayon::ThreadPoolBuilder::new()
                .num_threads(threads)
                .build()
                .unwrap()
                .install(|| {
                    positions
                        .par_iter()
                        .map_init(solver::new_tt_buffer, |tt, p| solve_one(p, tt))
                        .collect()
                })
        }
        #[cfg(not(feature = "parallel"))]
        {
            panic!("--threads > 1 needs --features parallel");
        }
    };
    // Extra passes, keeping the per-position minimum.
    let mut results = results;
    for _ in 1..args.repeats {
        let again: Vec<(Vals, u64, f64)> = if threads == 1 {
            let mut tt = solver::new_tt_buffer();
            positions.iter().map(|p| solve_one(p, &mut tt)).collect()
        } else {
            #[cfg(feature = "parallel")]
            {
                use rayon::prelude::*;
                rayon::ThreadPoolBuilder::new()
                    .num_threads(threads)
                    .build()
                    .unwrap()
                    .install(|| {
                        positions
                            .par_iter()
                            .map_init(solver::new_tt_buffer, |tt, p| solve_one(p, tt))
                            .collect()
                    })
            }
            #[cfg(not(feature = "parallel"))]
            {
                unreachable!()
            }
        };
        for (i, r) in again.into_iter().enumerate() {
            assert_eq!(r.0, results[i].0, "position {i} changed value between repeats");
            assert_eq!(r.1, results[i].1, "position {i} changed node count between repeats");
            if r.2 < results[i].2 {
                results[i].2 = r.2;
            }
        }
    }
    let wall = t_all.elapsed().as_secs_f64();

    // Per-shape aggregation.
    println!(
        "\n{:>8} {:>7} {:>12} {:>12} {:>12} {:>10}",
        "shape", "n", "nodes(M)", "nodes/pos", "us/pos", "cards_left"
    );
    let mut totals = (0u64, 0f64, 0usize);
    for shape in [Shape::Full, Shape::Mid, Shape::End, Shape::Worlds] {
        let idx: Vec<usize> = (0..positions.len())
            .filter(|&i| positions[i].shape == shape)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let nodes: u64 = idx.iter().map(|&i| results[i].1).sum();
        let us: f64 = idx.iter().map(|&i| results[i].2).sum();
        let cl: f64 =
            idx.iter().map(|&i| positions[i].cards_left() as f64).sum::<f64>() / idx.len() as f64;
        println!(
            "{:>8} {:>7} {:>12.1} {:>12.0} {:>12.1} {:>10.1}",
            shape.name(),
            idx.len(),
            nodes as f64 / 1e6,
            nodes as f64 / idx.len() as f64,
            us / idx.len() as f64,
            cl
        );
        totals.0 += nodes;
        totals.1 += us;
        totals.2 += idx.len();
    }
    println!(
        "{:>8} {:>7} {:>12.1} {:>12.0} {:>12.1}",
        "ALL", totals.2, totals.0 as f64 / 1e6, totals.0 as f64 / totals.2 as f64,
        totals.1 / totals.2 as f64
    );
    println!(
        "\nwall {:.2}s at {} thread(s); summed per-position time {:.2}s",
        wall, threads, totals.1 / 1e6
    );
    if solver::stats_enabled() && totals.1 > 0.0 {
        println!("throughput {:.2} M nodes/s (summed CPU time)", totals.0 as f64 / totals.1);
    }

    // Value checksum: a single number that must be identical across any exact change.
    let checksum: i64 = results
        .iter()
        .flat_map(|(v, _, _)| v.iter())
        .map(|&(c, s)| c as i64 * 1_000_003 + s as i64)
        .sum();
    println!("value checksum: {checksum}");

    let vals: Vec<Vals> = results.iter().map(|(v, _, _)| v.clone()).collect();
    if let Some(path) = &args.values {
        write_values(path, &vals)?;
        println!("values -> {path}");
    }
    if let Some(path) = &args.json {
        let mut f = fs::File::create(path)?;
        writeln!(
            f,
            "{{\"positions\":{},\"nodes\":{},\"cpu_us\":{:.0},\"wall_s\":{:.3},\"threads\":{},\"checksum\":{},\"stats\":{},\"heuristics\":\"{}\"}}",
            totals.2, totals.0, totals.1, wall, threads, checksum, solver::stats_enabled(),
            solver::ablation_label()
        )?;
        println!("summary -> {path}");
    }
    Ok(())
}

// ------------------------------------------------------------------ window oracle

/// **Ceiling** on the whole family of window-seeding ideas: what if the seed were perfect?
///
/// `solve_windowed_reuse_tt` exists so a caller can hand the search a guess at the answer.
/// Every proposal in that family — seed from a sibling world, from a heuristic evaluation,
/// from the previous trick — differs only in *how good the guess is*. So rather than build
/// one and measure it, solve each position twice: once for real, then again with a window
/// centred on the answer we just got. No heuristic can beat the answer itself.
///
/// The δ sweep turns the ceiling into a requirement. δ=1 is the perfect seed; δ=40 is a seed
/// that is merely in the right region. If the saving has already evaporated at δ=20, then the
/// measured fact that 36 % of a hand's worlds deviate by more than 40 points settles it.
///
/// Break-even: a seeder that brackets the true value with probability `p` pays `r` always and
/// a full re-search on a miss, so it is worth building only if `p > r`. A ratio of 0.7 means
/// the guess must be right 70 % of the time just to pay for itself — printed per shape.
///
/// Scope: this is the *root* value, not the per-card table `solve_with_scores` returns. A
/// windowed `solve_with_scores` would need one accurate seed per legal card, which is strictly
/// harder, so the ceiling measured here bounds that case too.
fn cmd_oracle(args: &Args, deltas: &[i16]) -> io::Result<()> {
    let positions = read_corpus(&args.corpus)?;
    if !solver::stats_enabled() {
        eprintln!("REFUSING: the oracle compares node counts; rebuild with --features solver_stats");
        std::process::exit(2);
    }
    eprintln!("corpus: {} positions from {}", positions.len(), args.corpus);
    eprintln!("deltas: {deltas:?}\n");

    let mut tt = solver::new_tt_buffer();
    // [shape][delta] -> summed nodes; column 0 is the full-window reference.
    let mut nodes = vec![vec![0u64; deltas.len() + 1]; 4];
    let mut count = [0usize; 4];

    for p in &positions {
        let st = p.rebuild().expect("corpus position must rebuild");
        let si = p.shape as usize;

        let _ = solver::take_nodes();
        let v = solver::solve_windowed_reuse_tt(&st, &mut tt, 0, 252);
        nodes[si][0] += solver::take_nodes();
        count[si] += 1;

        for (di, &d) in deltas.iter().enumerate() {
            let _ = solver::take_nodes();
            let vd = solver::solve_windowed_reuse_tt(&st, &mut tt, v - d, v + d);
            nodes[si][di + 1] += solver::take_nodes();
            // The window strictly brackets v, so fail-soft owes us the exact value. If this
            // ever fires, the windowed entry point is unsound and no ratio below means anything.
            assert_eq!(vd, v, "windowed solve disagreed inside a bracketing window");
        }
    }

    print!("{:>8} {:>7} {:>12}", "shape", "n", "nodes/pos");
    for d in deltas {
        print!("   +/-{:<3}", d);
    }
    println!("     (fraction of the full-window search that survives)");

    for shape in [Shape::Full, Shape::Mid, Shape::End, Shape::Worlds] {
        let si = shape as usize;
        if count[si] == 0 {
            continue;
        }
        let base = nodes[si][0] as f64;
        print!(
            "{:>8} {:>7} {:>12.0}",
            shape.name(),
            count[si],
            base / count[si] as f64
        );
        for di in 0..deltas.len() {
            print!("  {:>7.3}", nodes[si][di + 1] as f64 / base);
        }
        println!();
    }

    // The decision number. A seeder accurate to +/-d pays the windowed search always and a full
    // re-search whenever it misses, so it only pays for itself if it lands inside +/-d more
    // often than the fraction of the search that window leaves standing. Put the other way:
    // this table is the accuracy a seeder must reach, and it can be read straight against a
    // measured error distribution.
    println!("\nseeder accurate to +/-d must bracket the answer at least this often to break even:");
    print!("{:>8}", "shape");
    for d in deltas {
        print!("   +/-{:<3}", d);
    }
    println!();
    for shape in [Shape::Full, Shape::Mid, Shape::End, Shape::Worlds] {
        let si = shape as usize;
        if count[si] == 0 {
            continue;
        }
        let base = nodes[si][0] as f64;
        print!("{:>8}", shape.name());
        for di in 0..deltas.len() {
            let r = nodes[si][di + 1] as f64 / base;
            if r >= 1.0 {
                print!("  {:>6}", "never");
            } else {
                print!("  {:>5.1}%", r * 100.0);
            }
        }
        println!();
    }
    println!(
        "\n\"never\" = the windowed search is already no cheaper than the full one, so no hit\n\
         rate saves it. Read the +/-40 column against the measured spread of a hand's worlds."
    );
    Ok(())
}

// --------------------------------------------------------------- move-ordering oracle

/// **Ceiling** on move ordering, and specifically on the hard tail.
///
/// "The 10 % hardest solves carry 40 % of the nodes, and a tree 20× the median is a signature
/// of an ordering failure rather than of intrinsic difficulty" has been the standing hypothesis
/// behind every idea to teach the solver about Contrée. It has never been tested, and § 3 does
/// not test it — knowing that the generic heuristics remove 71.5 % says the lever works, not
/// that anything is left in it.
///
/// So: solve once recording the best move at every node, then replay with those moves forced
/// first. Whatever a perfect Contrée-aware rule would achieve, it cannot beat replaying the
/// answer. A third pass, still recording, shows whether the figure has converged.
///
/// The tail split is the actual deliverable. If the hardest decile improves by the same ratio
/// as the median, its trees are big because the positions are hard, and the whole line dies.
/// If it improves far more, the hypothesis holds and the ceiling says by how much.
/// Ply windows for the depth sweep: the first trick, the first two, the first four, then the
/// complement of each so the two halves can be checked for additivity.
const PLY_CUTS: &[(u8, u8)] = &[(0, 1), (0, 2), (0, 4), (0, 8), (0, 16), (1, 32), (4, 32)];

fn cmd_ordering(args: &Args, rates: &[f64]) -> io::Result<()> {
    let positions = read_corpus(&args.corpus)?;
    if !solver::stats_enabled() || !solver::oracle_enabled() {
        eprintln!(
            "REFUSING: needs --features \"solver_stats solver_oracle\" (node counts and the \
             recorded-move map)"
        );
        std::process::exit(2);
    }
    eprintln!("corpus: {} positions from {}", positions.len(), args.corpus);
    eprintln!("note: the map is per-thread and unbounded; a hard full deal can hold ~3 M entries\n");

    // (n0, n1, n2, recorded, ranks, nodes-at-each-coverage) per position.
    struct Row {
        n0: u64,
        n1: u64,
        n2: u64,
        recorded: usize,
        ranks: [u64; 9],
        cov: Vec<u64>,
        depth: Vec<u64>,
        conf: [[[u64; 8]; 8]; 2],
    }
    let solve_one_pos = |p: &Position| -> Row {
        let st = p.rebuild().expect("corpus position must rebuild");
        let mut tt = solver::new_tt_buffer();
        solver::oracle_set_root_ply(&st);

        // The map is keyed per (deal, trump) — carrying it over would feed one position's
        // moves to another and silently invent an oracle that knows the wrong game.
        solver::oracle_clear();
        solver::oracle_set_hint_rate(1.0);

        solver::oracle_set_mode(solver::ORACLE_RECORD);
        let _ = solver::take_nodes();
        let _ = solver::oracle_take_ranks();
        // Both accumulators are thread-local and the positions run on rayon workers, so they
        // have to be drained here — draining them after the join would read an empty main
        // thread and report a confident zero.
        let _ = solver::oracle_take_confusion();
        let v0 = solver::solve_reuse_tt(&st, &mut tt);
        let n0 = solver::take_nodes();
        // The ranks are taken from THIS pass: it is the production search, unhinted, so the
        // histogram describes today's ordering rather than the oracle's.
        let ranks = solver::oracle_take_ranks();
        let conf = solver::oracle_take_confusion();
        let recorded = solver::oracle_len();

        solver::oracle_set_mode(solver::ORACLE_USE_RECORD);
        let _ = solver::take_nodes();
        let v1 = solver::solve_reuse_tt(&st, &mut tt);
        let n1 = solver::take_nodes();

        let _ = solver::take_nodes();
        let v2 = solver::solve_reuse_tt(&st, &mut tt);
        let n2 = solver::take_nodes();

        // Coverage sweep, off the map recorded above. USE only — recording here would let a
        // low-coverage pass enrich the map and flatter the next one.
        solver::oracle_set_mode(solver::ORACLE_USE);
        let mut cov = Vec::with_capacity(rates.len());
        for &r in rates {
            solver::oracle_set_hint_rate(r);
            let _ = solver::take_nodes();
            let vr = solver::solve_reuse_tt(&st, &mut tt);
            cov.push(solver::take_nodes());
            assert_eq!(v0, vr, "partial-coverage ordering changed the answer");
        }
        solver::oracle_set_hint_rate(1.0);

        // Depth sweep: the hint only in the first k plies, then only *after* them. If almost
        // all of the gain sits in the first tricks, a predictor far too slow for a 22 ns node
        // could still be run there and pay for itself.
        let mut depth = Vec::with_capacity(PLY_CUTS.len() * 2);
        for &(lo, hi) in PLY_CUTS {
            solver::oracle_set_ply_window(lo, hi);
            let _ = solver::take_nodes();
            let vd = solver::solve_reuse_tt(&st, &mut tt);
            depth.push(solver::take_nodes());
            assert_eq!(v0, vd, "depth-windowed ordering changed the answer");
        }
        solver::oracle_set_ply_window(0, 32);
        solver::oracle_set_mode(solver::ORACLE_OFF);

        // Ordering changes the shape of the search, never its result. If this fires, the
        // hint is being applied to a position it does not describe.
        assert_eq!(v0, v1, "oracle ordering changed the answer");
        assert_eq!(v0, v2, "iterated oracle ordering changed the answer");
        Row { n0, n1, n2, recorded, ranks, cov, depth, conf }
    };

    let t0 = Instant::now();
    let res: Vec<Row> = if args.threads == 1 {
        positions.iter().map(solve_one_pos).collect()
    } else {
        #[cfg(feature = "parallel")]
        {
            use rayon::prelude::*;
            rayon::ThreadPoolBuilder::new()
                .num_threads(args.threads)
                .build()
                .unwrap()
                .install(|| positions.par_iter().map(solve_one_pos).collect())
        }
        #[cfg(not(feature = "parallel"))]
        {
            positions.iter().map(solve_one_pos).collect()
        }
    };
    eprintln!("{:.1}s at {} thread(s)\n", t0.elapsed().as_secs_f64(), args.threads);

    println!(
        "{:>8} {:>7} {:>13} {:>11} {:>10} {:>10}",
        "shape", "n", "nodes/pos", "recorded", "oracle", "iterated"
    );
    for shape in [Shape::Full, Shape::Mid, Shape::End, Shape::Worlds] {
        let idx: Vec<usize> = (0..positions.len())
            .filter(|&i| positions[i].shape == shape)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let s0: u64 = idx.iter().map(|&i| res[i].n0).sum();
        let s1: u64 = idx.iter().map(|&i| res[i].n1).sum();
        let s2: u64 = idx.iter().map(|&i| res[i].n2).sum();
        let rec: usize = idx.iter().map(|&i| res[i].recorded).sum();
        println!(
            "{:>8} {:>7} {:>13.0} {:>11.0} {:>10.3} {:>10.3}",
            shape.name(),
            idx.len(),
            s0 as f64 / idx.len() as f64,
            rec as f64 / idx.len() as f64,
            s1 as f64 / s0 as f64,
            s2 as f64 / s0 as f64,
        );
    }

    // The tail. Buckets are over the baseline node count, which is the only ranking available
    // before the search runs — the same one a dispatcher would have to use.
    for shape in [Shape::Full, Shape::Worlds] {
        let mut idx: Vec<usize> = (0..positions.len())
            .filter(|&i| positions[i].shape == shape)
            .collect();
        if idx.len() < 20 {
            continue;
        }
        idx.sort_by_key(|&i| std::cmp::Reverse(res[i].n0));
        let total: u64 = idx.iter().map(|&i| res[i].n0).sum();
        let n = idx.len();

        println!("\n{} — by difficulty (baseline nodes), {} positions:", shape.name(), n);
        println!(
            "{:>14} {:>7} {:>13} {:>14} {:>10} {:>10}",
            "bucket", "n", "nodes/pos", "share of nodes", "oracle", "iterated"
        );
        let cuts: [(&str, usize, usize); 4] = [
            ("hardest 10 %", 0, n / 10),
            ("next 15 %", n / 10, n / 4),
            ("middle 50 %", n / 4, n * 3 / 4),
            ("easiest 25 %", n * 3 / 4, n),
        ];
        for (label, lo, hi) in cuts {
            if hi <= lo {
                continue;
            }
            let part = &idx[lo..hi];
            let s0: u64 = part.iter().map(|&i| res[i].n0).sum();
            let s1: u64 = part.iter().map(|&i| res[i].n1).sum();
            let s2: u64 = part.iter().map(|&i| res[i].n2).sum();
            println!(
                "{:>14} {:>7} {:>13.0} {:>13.1}% {:>10.3} {:>10.3}",
                label,
                part.len(),
                s0 as f64 / part.len() as f64,
                100.0 * s0 as f64 / total as f64,
                s1 as f64 / s0 as f64,
                s2 as f64 / s0 as f64,
            );
        }
    }

    // Where today's ordering already stands. Rank 0 = the first move tried caused the cutoff.
    // Caveat worth keeping in mind while reading it: at a cut node several moves may each be
    // good enough, and only the one that fired is credited — so this understates the current
    // ordering rather than flattering it.
    // When today's ordering misses, what does it try instead of what? This is the only table
    // here written in the vocabulary a Contrée-aware rule would use, so it is the one that
    // says whether such a rule is writable at all.
    let mut conf = [[[0u64; 8]; 8]; 2];
    for r in &res {
        for b in 0..2 {
            for g in 0..8 {
                for w in 0..8 {
                    conf[b][g][w] += r.conf[b][g][w];
                }
            }
        }
    }
    for (bi, label) in ["tricks 1-3", "tricks 4-8"].iter().enumerate() {
        let total: u64 = conf[bi].iter().flatten().sum();
        if total == 0 {
            continue;
        }
        println!("\nordering failures, {label} — rows: tried first, cols: should have been");
        print!("{:>12}", "");
        for c in solver::MOVE_CATEGORIES {
            print!("{:>11}", c);
        }
        println!("{:>9}", "row %");
        for (gi, g) in solver::MOVE_CATEGORIES.iter().enumerate() {
            let row: u64 = conf[bi][gi].iter().sum();
            if row == 0 {
                continue;
            }
            print!("{:>12}", g);
            for wi in 0..8 {
                let v = conf[bi][gi][wi];
                if v == 0 {
                    print!("{:>11}", "·");
                } else {
                    print!("{:>10.1}%", 100.0 * v as f64 / total as f64);
                }
            }
            println!("{:>8.1}%", 100.0 * row as f64 / total as f64);
        }
        println!("  ({total} failures counted)");
    }

    println!("\nrank the eventual best move held in today's ordering:");
    println!("{:>8} {:>10} {:>10} {:>10} {:>12}", "shape", "rank 0", "rank 1", "rank 2", "rank >=3");
    for shape in [Shape::Full, Shape::Mid, Shape::End, Shape::Worlds] {
        let idx: Vec<usize> = (0..positions.len())
            .filter(|&i| positions[i].shape == shape)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let mut h = [0u64; 9];
        for &i in &idx {
            for k in 0..9 {
                h[k] += res[i].ranks[k];
            }
        }
        let tot: u64 = h.iter().sum();
        if tot == 0 {
            continue;
        }
        let pc = |v: u64| 100.0 * v as f64 / tot as f64;
        println!(
            "{:>8} {:>9.1}% {:>9.1}% {:>9.1}% {:>11.1}%",
            shape.name(),
            pc(h[0]),
            pc(h[1]),
            pc(h[2]),
            pc(h[3..].iter().sum::<u64>()),
        );
    }

    // What partial coverage buys. This models a rule that fires on a fraction of nodes and is
    // exactly right when it does, falling back to today's ordering elsewhere — so it is an
    // upper bound for any partial rule, and the curve says what coverage is worth chasing.
    if !rates.is_empty() {
        println!("\nfraction of the baseline search surviving a rule that is right on p of nodes:");
        print!("{:>8}", "shape");
        for r in rates {
            print!("{:>9.0}%", r * 100.0);
        }
        println!();
        for shape in [Shape::Full, Shape::Mid, Shape::End, Shape::Worlds] {
            let idx: Vec<usize> = (0..positions.len())
                .filter(|&i| positions[i].shape == shape)
                .collect();
            if idx.is_empty() {
                continue;
            }
            let s0: u64 = idx.iter().map(|&i| res[i].n0).sum();
            print!("{:>8}", shape.name());
            for k in 0..rates.len() {
                let s: u64 = idx.iter().map(|&i| res[i].cov[k]).sum();
                print!("{:>10.3}", s as f64 / s0 as f64);
            }
            println!();
        }
    }

    println!("\nfraction surviving when the perfect hint is restricted to a ply window:");
    print!("{:>8}", "shape");
    for (lo, hi) in PLY_CUTS {
        print!("{:>12}", format!("[{lo},{hi})"));
    }
    println!();
    for shape in [Shape::Full, Shape::Mid, Shape::End, Shape::Worlds] {
        let idx: Vec<usize> = (0..positions.len())
            .filter(|&i| positions[i].shape == shape)
            .collect();
        if idx.is_empty() {
            continue;
        }
        let s0: u64 = idx.iter().map(|&i| res[i].n0).sum();
        print!("{:>8}", shape.name());
        for k in 0..PLY_CUTS.len() {
            let s: u64 = idx.iter().map(|&i| res[i].depth[k]).sum();
            print!("{:>12.3}", s as f64 / s0 as f64);
        }
        println!();
    }

    println!(
        "\n`oracle` / `iterated` are the fraction of the baseline search that survives perfect\n\
         ordering. A tail ratio no better than the median's means those trees are big because\n\
         the positions are hard, and a Contrée-aware ordering rule has nothing to recover."
    );
    Ok(())
}

/// Interleaved A/B of the epoch TT against the old memset-every-solve behaviour.
///
/// Runs both configurations alternately, `repeats` times each, and reports the **minimum**
/// per shape. Minimum rather than mean because the competing noise on a shared machine is
/// strictly additive: the fastest observed run is the one least disturbed. Interleaving is
/// what makes the comparison survive load that drifts over the run — measuring A then B
/// sequentially attributes any drift to the change, which is exactly the mistake that made
/// the first reading of this experiment look like a 20 % regression.
fn cmd_ab(args: &Args) -> io::Result<()> {
    let positions = read_corpus(&args.corpus)?;
    eprintln!(
        "A/B over {} positions, {} interleaved repeats each\n",
        positions.len(),
        args.repeats
    );

    // [config][shape] -> best total microseconds
    let mut best = [[f64::INFINITY; 4]; 2];
    let mut nodes_seen = [[0u64; 4]; 2];
    let mut checksums = [0i64; 2];

    for r in 0..args.repeats {
        for config in 0..2 {
            let legacy = config == 1;
            let mut tt = solver::new_tt_buffer();
            tt.set_legacy_clear(legacy);
            let mut per_shape = [0f64; 4];
            let mut nodes = [0u64; 4];
            let mut sum = 0i64;
            for p in &positions {
                let (v, n, us) = solve_one(p, &mut tt);
                let s = p.shape as usize;
                per_shape[s] += us;
                nodes[s] += n;
                for &(c, sc) in &v {
                    sum += c as i64 * 1_000_003 + sc as i64;
                }
            }
            checksums[config] = sum;
            for s in 0..4 {
                if per_shape[s] < best[config][s] {
                    best[config][s] = per_shape[s];
                }
                nodes_seen[config][s] = nodes[s];
            }
        }
        eprintln!("  repeat {}/{} done", r + 1, args.repeats);
    }

    // Exactness is not optional even here: the two configurations must agree exactly.
    if checksums[0] != checksums[1] {
        println!("CHECKSUM MISMATCH: epoch={} legacy={}", checksums[0], checksums[1]);
        std::process::exit(1);
    }
    println!("value checksum identical in both configurations: {}\n", checksums[0]);

    println!(
        "{:>8} {:>7} {:>12} {:>13} {:>13} {:>9}",
        "shape", "n", "nodes/pos", "epoch us/pos", "clear us/pos", "speedup"
    );
    let mut tot = [0f64; 2];
    for (si, shape) in [Shape::Full, Shape::Mid, Shape::End, Shape::Worlds].iter().enumerate() {
        let n = positions.iter().filter(|p| p.shape == *shape).count();
        if n == 0 {
            continue;
        }
        let e = best[0][si] / n as f64;
        let l = best[1][si] / n as f64;
        tot[0] += best[0][si];
        tot[1] += best[1][si];
        println!(
            "{:>8} {:>7} {:>12} {:>13.1} {:>13.1} {:>8.2}x",
            shape.name(),
            n,
            nodes_seen[0][si] / n.max(1) as u64,
            e,
            l,
            l / e
        );
        if nodes_seen[0][si] != nodes_seen[1][si] {
            println!(
                "    !! node counts differ ({} vs {}) — the change is NOT tree-neutral",
                nodes_seen[0][si], nodes_seen[1][si]
            );
        }
    }
    println!(
        "{:>8} {:>7} {:>12} {:>13.1} {:>13.1} {:>8.2}x",
        "ALL",
        positions.len(),
        "",
        tot[0] / positions.len() as f64,
        tot[1] / positions.len() as f64,
        tot[1] / tot[0]
    );
    Ok(())
}

fn cmd_diff(a: &str, b: &str) -> io::Result<()> {
    let va = read_values(a)?;
    let vb = read_values(b)?;
    if va.len() != vb.len() {
        println!("DIFFERENT CORPUS SIZE: {} vs {}", va.len(), vb.len());
        std::process::exit(2);
    }
    let mut pos_diff = 0usize;
    let mut card_diff = 0usize;
    let mut cards = 0usize;
    let mut worst = 0i16;
    let mut best_card_diff = 0usize;
    let mut shown = 0usize;
    for (i, (x, y)) in va.iter().zip(vb.iter()).enumerate() {
        cards += x.len();
        if x == y {
            continue;
        }
        pos_diff += 1;
        if x.len() == y.len() {
            for (p, q) in x.iter().zip(y.iter()) {
                if p != q {
                    card_diff += 1;
                    worst = worst.max((p.1 - q.1).abs());
                }
            }
            // Did the *best* card change? That is the part a player would feel.
            let bx = x.iter().max_by_key(|c| c.1).unwrap().0;
            let by = y.iter().max_by_key(|c| c.1).unwrap().0;
            if bx != by {
                best_card_diff += 1;
            }
        } else {
            card_diff += x.len().max(y.len());
        }
        if shown < 5 {
            shown += 1;
            println!("position {i}:\n  a={x:?}\n  b={y:?}");
        }
    }
    println!("\npositions      : {} ({pos_diff} differ)", va.len());
    println!("card values    : {cards} ({card_diff} differ)");
    println!("worst |delta|  : {worst} points");
    println!("best card moved: {best_card_diff} positions");
    if pos_diff == 0 {
        println!("\nEXACT MATCH");
    } else {
        println!("\nNOT EXACT — {} of {} positions disagree", pos_diff, va.len());
        std::process::exit(1);
    }
    Ok(())
}

// ---------------------------------------------------------------- cli

struct Args {
    corpus: String,
    values: Option<String>,
    json: Option<String>,
    threads: usize,
    repeats: usize,
    ab: bool,
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let cmd = argv.get(1).map(|s| s.as_str()).unwrap_or("help");

    let mut corpus = "data/analysis/dd_corpus_v1.bin".to_string();
    let mut out = corpus.clone();
    let mut values: Option<String> = None;
    let mut json: Option<String> = None;
    let mut threads = 1usize;
    let mut repeats = 1usize;
    let mut ab = false;
    let mut pool = "data/deals/base_5M.bin".to_string();
    let mut games = "data/training/heldout_20k_s90210.bin".to_string();
    let mut full_deals = 200usize;
    let mut per_bucket = 30usize;
    let mut world_positions = 30usize;
    let mut worlds = 24usize;
    let mut a = String::new();
    let mut b = String::new();
    let mut deltas: Vec<i16> = vec![1, 5, 20, 40];
    let mut hint_rates: Vec<f64> = vec![0.25, 0.5, 0.75, 0.9];

    let mut i = 2;
    while i < argv.len() {
        let k = argv[i].as_str();
        let mut next = || {
            i += 1;
            argv.get(i).cloned().unwrap_or_default()
        };
        match k {
            "--corpus" => corpus = next(),
            "--out" => out = next(),
            "--values" => values = Some(next()),
            "--json" => json = Some(next()),
            "--threads" => threads = next().parse().unwrap(),
            "--repeats" => repeats = next().parse().unwrap(),
            "--ab" => ab = true,
            "--pool" => pool = next(),
            "--games" => games = next(),
            "--full-deals" => full_deals = next().parse().unwrap(),
            "--per-bucket" => per_bucket = next().parse().unwrap(),
            "--world-positions" => world_positions = next().parse().unwrap(),
            "--worlds" => worlds = next().parse().unwrap(),
            "--a" => a = next(),
            "--b" => b = next(),
            "--deltas" => deltas = next().split(',').map(|s| s.parse().unwrap()).collect(),
            "--hint-rates" => {
                hint_rates = next().split(',').map(|s| s.parse().unwrap()).collect()
            }
            other => {
                eprintln!("unknown arg {other}");
                std::process::exit(2);
            }
        }
        i += 1;
    }

    match cmd {
        "build" => {
            let mut positions = Vec::new();
            eprintln!("building corpus:");
            if let Err(e) = build_full(&pool, full_deals, &mut positions) {
                eprintln!("  full: SKIPPED ({e})");
            }
            if let Err(e) = build_from_games(&games, per_bucket, &mut positions) {
                eprintln!("  games: SKIPPED ({e})");
            }
            if let Err(e) = build_worlds(&games, world_positions, worlds, &mut positions) {
                eprintln!("  worlds: SKIPPED ({e})");
            }
            // Every position must rebuild before we write the file.
            let mut bad = 0;
            for (i, p) in positions.iter().enumerate() {
                if let Err(e) = p.rebuild() {
                    if bad < 5 {
                        eprintln!("  position {i} does not rebuild: {e}");
                    }
                    bad += 1;
                }
            }
            if bad > 0 {
                eprintln!("REFUSING to write: {bad} positions do not rebuild");
                std::process::exit(1);
            }
            write_corpus(&out, &positions).expect("write corpus");
            eprintln!("\n{} positions -> {out}", positions.len());
        }
        "run" => {
            let args = Args { corpus, values, json, threads, repeats, ab };
            if args.ab {
                cmd_ab(&args).expect("ab");
            } else {
                cmd_run(&args).expect("run");
            }
        }
        "oracle" => {
            let args = Args { corpus, values, json, threads, repeats, ab };
            cmd_oracle(&args, &deltas).expect("oracle");
        }
        "ordering" => {
            let args = Args { corpus, values, json, threads, repeats, ab };
            cmd_ordering(&args, &hint_rates).expect("ordering");
        }
        "diff" => {
            if a.is_empty() || b.is_empty() {
                eprintln!("diff needs --a and --b");
                std::process::exit(2);
            }
            cmd_diff(&a, &b).expect("diff");
        }
        _ => {
            eprintln!("usage: bench_dd build|run|oracle|ordering|diff  (see the module doc comment)");
            std::process::exit(2);
        }
    }
}
