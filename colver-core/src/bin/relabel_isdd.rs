//! Re-score deals of an existing pool with the **current** IS-DD, and compare
//! the labels head-to-head against a score layer produced by an older one.
//!
//! The question it answers is not "what are the new labels" but **"how stale is
//! `scores_isdd_5M.sc`, and does its staleness reach the bidding decision?"** —
//! which decides whether a full pool regeneration is worth its GPU-weeks.
//!
//! ## Three arms, because three things changed at once
//!
//! `scores_isdd_5M.sc` was produced on 2026-04-23. Since then:
//!
//! 1. **the solver was fixed** (`quick_tricks` removed, 2026-07-23) and the
//!    surcoupe legality widened (2026-08-01) — both change DD values;
//! 2. **world generation moved into the agent** (2026-07-24), so IS-DD now
//!    draws playgen worlds instead of constraint-uniform ones.
//!
//! Comparing today's playgen IS-DD straight against the April file measures
//! all of it at once and attributes none of it. So the arms are:
//!
//! | arm | `--worlds` | `--auction` | isolates |
//! |---|---|---|---|
//! | A | *(the existing `.sc` file)* | | baseline |
//! | B0 | `uniform` | `none` | A→B0 = **code drift** (solver + rules), same protocol |
//! | B | `uniform` | `synthetic` | B0→B = the **auction prefix** playgen forces on us |
//! | C | `sidecar` | `synthetic` | B→C = **playgen worlds**, the thing under test |
//!
//! ## Why arm C needs a synthetic auction at all
//!
//! The old protocol forces each trump with [`GameState::setup_dd`], which lands
//! directly in `Phase::Playing` with an *empty* auction. The sidecar cannot
//! represent that: `/play_worlds` replays `GameState::new` through an action
//! list, so reaching play *requires* an auction — and for playgen v2 the trump
//! is carried only by the bid tokens, so a bid-less prefix leaves the model
//! blind to the trump. Arm C therefore opens 80 in the target suit from
//! dealer+1 and passes three times: the cheapest legal prefix that names the
//! trump. That prefix is a real conditioning signal (it tells playgen dealer+1
//! likes that suit), which is exactly why arm B exists to price it separately.
//!
//! Card points are unaffected by the auction's *scoring* side: the label is
//! `state.points[0]`, NS card points including dix de der, and contract value,
//! taker and multipliers never enter it.
//!
//! ## Mode: count, not time
//!
//! Offline labelling runs in **count mode** (`--time-ms 0`), for two reasons.
//! It is reproducible — a wall-clock deadline makes the label depend on machine
//! load, which is not a property a training target should have. And under a
//! sidecar it is ~6x cheaper: in time mode IS-DD refills `world_batch` (128)
//! worlds per decision, then blows its 20 ms deadline on the first round trip
//! and solves one or two of them. Count mode asks for exactly `--dets`.
//!
//! ## `--threads` : bien plus que le nombre de cœurs, avec le sidecar
//!
//! Contre-intuitif mais mesuré. Avec `--worlds sidecar` les threads passent leur
//! temps **bloqués en HTTP**, pas à calculer : à 32 threads le process n'utilise
//! que ~180 % de CPU sur 3200 % disponibles. Or le sidecar groupe les requêtes
//! qu'il trouve en file, donc peu de requêtes en vol = petits lots = GPU mal
//! amorti. Monter la concurrence remplit les lots :
//!
//! | threads | donnes/s | lots moyens |
//! |---|---|---|
//! | 32 (défaut) | 0,237 | 12,8 req |
//! | 96 | 0,396 | |
//! | **192** | **0,496** | 25,9 req |
//! | 384 | 0,492 | plateau |
//!
//! Avec `--worlds uniform` c'est l'inverse : tout est CPU, le défaut est bon.
//!
//! Usage:
//!   cargo run --bin relabel_isdd --release --features parallel -- \
//!     --deals 1000 --worlds uniform --auction none --dets 20 \
//!     --output data/deals/relabel/b0_uniform_noauction.sc
//!
//!   # avec le sidecar : monter la concurrence
//!   ... --worlds sidecar --auction synthetic --threads 192

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use colver_core::agent::isdd::IsDdPlayer;
use colver_core::agent::{CardPlayer, MatchContext};
use colver_core::bid_train_env::DealPool;
use colver_core::bidding::{encode_bid, BID_PASS};
use colver_core::is_dd::IsDdConfig;
use colver_core::state::{GameState, Phase};
use colver_core::worlds::{SidecarWorldSource, UniformWorldSource, WorldSource};

struct Args {
    pool: String,
    baseline: String,
    output: String,
    name: String,
    deals: usize,
    offset: usize,
    time_ms: u32,
    dets: u32,
    seed: u64,
    worlds: String,
    url: String,
    threads: usize,
    world_batch: usize,
    auction: bool,
    chunk: usize,
    compare_only: String,
    json: String,
}

fn parse_args() -> Args {
    let a: Vec<String> = std::env::args().collect();
    let mut args = Args {
        pool: "data/deals/base_5M.bin".into(),
        baseline: "data/deals/scores_isdd_5M.sc".into(),
        output: String::new(),
        name: String::new(),
        deals: 1000,
        offset: 0,
        time_ms: 0,
        dets: 20,
        seed: 42,
        worlds: "uniform".into(),
        url: std::env::var("COLVER_PLAYGEN_GPU_URL")
            .unwrap_or_else(|_| "http://localhost:8003".into()),
        threads: 0,
        world_batch: 128,
        auction: false,
        chunk: 500,
        compare_only: String::new(),
        json: String::new(),
    };
    let mut i = 1;
    while i < a.len() {
        match a[i].as_str() {
            "--pool" => { i += 1; args.pool = a[i].clone(); }
            "--baseline" => { i += 1; args.baseline = a[i].clone(); }
            "--output" => { i += 1; args.output = a[i].clone(); }
            "--name" => { i += 1; args.name = a[i].clone(); }
            "--deals" => { i += 1; args.deals = a[i].parse().unwrap(); }
            "--offset" => { i += 1; args.offset = a[i].parse().unwrap(); }
            "--time-ms" => { i += 1; args.time_ms = a[i].parse().unwrap(); }
            "--dets" => { i += 1; args.dets = a[i].parse().unwrap(); }
            "--seed" => { i += 1; args.seed = a[i].parse().unwrap(); }
            "--worlds" => { i += 1; args.worlds = a[i].clone(); }
            "--url" => { i += 1; args.url = a[i].clone(); }
            "--threads" => { i += 1; args.threads = a[i].parse().unwrap(); }
            "--world-batch" => { i += 1; args.world_batch = a[i].parse().unwrap(); }
            "--chunk" => { i += 1; args.chunk = a[i].parse().unwrap(); }
            "--compare-only" => { i += 1; args.compare_only = a[i].clone(); }
            "--json" => { i += 1; args.json = a[i].clone(); }
            "--auction" => {
                i += 1;
                args.auction = match a[i].as_str() {
                    "none" => false,
                    "synthetic" => true,
                    o => panic!("--auction takes none|synthetic, got {o}"),
                };
            }
            o => panic!("unknown arg {o}"),
        }
        i += 1;
    }
    if args.name.is_empty() {
        args.name = format!(
            "isdd_{}_{}",
            args.worlds,
            if args.auction { "synth" } else { "noauction" }
        );
    }
    args
}

/// Read a COLVSC01 score layer: magic + name + count + offset + count x [u8; 4].
/// Read here rather than through `DealPool::load_scores` because the comparison
/// needs the raw array, and `DealPool` does not expose its layers.
fn read_score_layer(path: &str) -> std::io::Result<(String, usize, Vec<[u8; 4]>)> {
    use std::io::Read;
    let mut f = std::io::BufReader::new(std::fs::File::open(path)?);

    let mut magic = [0u8; 8];
    f.read_exact(&mut magic)?;
    if &magic != b"COLVSC01" {
        return Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("magic attendu COLVSC01, lu {magic:?}"),
        ));
    }
    let mut u2 = [0u8; 2];
    f.read_exact(&mut u2)?;
    let mut name = vec![0u8; u16::from_le_bytes(u2) as usize];
    f.read_exact(&mut name)?;
    let name = String::from_utf8_lossy(&name).into_owned();

    let mut u4 = [0u8; 4];
    f.read_exact(&mut u4)?;
    let count = u32::from_le_bytes(u4) as usize;
    f.read_exact(&mut u4)?;
    let offset = u32::from_le_bytes(u4) as usize;

    let mut scores = Vec::with_capacity(count);
    for _ in 0..count {
        let mut pts = [0u8; 4];
        f.read_exact(&mut pts)?;
        scores.push(pts);
    }
    Ok((name, offset, scores))
}

fn main() {
    let args = parse_args();

    // Compare two existing layers without scoring anything. Long sidecar runs
    // checkpoint every `--chunk` deals; this is what makes a partial one usable
    // before it finishes.
    if !args.compare_only.is_empty() {
        let (an, ao, asc) = read_score_layer(&args.baseline).expect("baseline");
        let (bn, bo, bsc) = read_score_layer(&args.compare_only).expect("comparand");
        let lo = ao.max(bo);
        let hi = (ao + asc.len()).min(bo + bsc.len()).min(lo + args.deals);
        assert!(hi > lo, "les deux couches ne se recouvrent pas");
        eprintln!("'{an}' [{ao}..] vs '{bn}' [{bo}..] — recouvrement [{lo}, {hi})");
        let old: Vec<[u8; 4]> = (lo..hi).map(|i| asc[i - ao]).collect();
        let new: Vec<[u8; 4]> = (lo..hi).map(|i| bsc[i - bo]).collect();
        let mut a2 = args;
        a2.name = bn;
        compare(&old, &new, &a2);
        return;
    }

    if args.worlds == "sidecar" && !args.auction {
        eprintln!(
            "refus : --worlds sidecar exige --auction synthetic.\n\
             Le sidecar rejoue GameState::new à travers une liste d'actions, donc il ne peut\n\
             pas représenter une position atteinte par setup_dd ; et pour playgen v2 l'atout\n\
             n'est porté que par les jetons d'enchère. Sans enchère, le modèle est aveugle à\n\
             l'atout et les mondes ne valent rien."
        );
        std::process::exit(2);
    }

    if args.threads > 0 {
        rayon::ThreadPoolBuilder::new()
            .num_threads(args.threads)
            .build_global()
            .unwrap();
    }
    let nthreads = rayon::current_num_threads();

    // ── Load base pool and the baseline score layer ────────────────────
    eprintln!("Chargement de {}...", args.pool);
    let pool = DealPool::load(&args.pool).expect("load pool");
    eprintln!("  {} donnes", pool.len());
    let (b_name, b_start, b_scores) =
        read_score_layer(&args.baseline).expect("load baseline scores");
    eprintln!(
        "  référence '{b_name}' : {} scores à l'offset {b_start}",
        b_scores.len()
    );

    let end = (args.offset + args.deals).min(pool.len());
    let n = end.saturating_sub(args.offset);
    assert!(n > 0, "aucune donne dans [{}, {})", args.offset, end);

    // The baseline must actually cover the slice, or the comparison is vacuous.
    let b_end = b_start + b_scores.len();
    assert!(
        args.offset >= b_start && end <= b_end,
        "la couche de référence couvre [{b_start}, {b_end}), pas [{}, {end})",
        args.offset
    );

    let deals: Vec<(u8, [u32; 4])> = (args.offset..end)
        .map(|i| {
            let d = pool.get(i);
            (d.dealer, d.hands)
        })
        .collect();
    let old: Vec<[u8; 4]> = (args.offset..end).map(|i| b_scores[i - b_start]).collect();

    if args.worlds == "sidecar" {
        let probe = SidecarWorldSource::new(args.url.clone(), 1.0, Duration::from_secs(10));
        match probe.health_check() {
            Ok(s) => eprintln!("sidecar {} ok : {}", args.url, s.trim()),
            Err(e) => {
                eprintln!("sidecar injoignable : {e}");
                std::process::exit(1);
            }
        }
    }

    eprintln!(
        "\n{} donnes x 4 couleurs | mondes={} | enchère={} | {} | wbatch={} | {} threads",
        n,
        args.worlds,
        if args.auction { "synthétique" } else { "aucune (setup_dd)" },
        if args.time_ms == 0 {
            format!("mode compte, {} dets", args.dets)
        } else {
            format!("mode temps, {} ms x {} dets", args.time_ms, args.dets)
        },
        args.world_batch,
        nthreads
    );

    // ── Score, in chunks so a long run checkpoints ─────────────────────
    let start = Instant::now();
    let progress = AtomicUsize::new(0);
    let dry = AtomicUsize::new(0);
    let calls = AtomicUsize::new(0);
    let mut new: Vec<[u8; 4]> = Vec::with_capacity(n);

    let mut done_deals = 0usize;
    while done_deals < n {
        let hi = (done_deals + args.chunk).min(n);
        let block = score_block(
            &deals[done_deals..hi],
            done_deals,
            &args,
            &progress,
            &dry,
            &calls,
            n,
            start,
        );
        new.extend(block);
        done_deals = hi;

        if !args.output.is_empty() {
            DealPool::save_scores(&args.name, args.offset, &new, &args.output)
                .expect("save score layer");
            eprintln!(
                "  checkpoint : {} donnes → {} ({:.1} min écoulées)",
                new.len(),
                args.output,
                start.elapsed().as_secs_f64() / 60.0
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    let rate = n as f64 / elapsed;
    eprintln!(
        "\nTerminé : {n} donnes en {:.1} s ({:.3} donnes/s)",
        elapsed, rate
    );
    let nc = calls.load(Ordering::Relaxed);
    let nd = dry.load(Ordering::Relaxed);
    eprintln!(
        "décisions {nc}, source à sec {nd} ({:.2} %)",
        nd as f64 / nc.max(1) as f64 * 100.0
    );
    for (label, k) in [("10k", 10_000.0), ("100k", 100_000.0), ("1M", 1_000_000.0)] {
        eprintln!("  extrapolation {label} : {:.1} h", k / rate / 3600.0);
    }

    compare(&old, &new, &args);
}

#[allow(clippy::too_many_arguments)]
fn score_block(
    deals: &[(u8, [u32; 4])],
    base_idx: usize,
    args: &Args,
    progress: &AtomicUsize,
    dry: &AtomicUsize,
    calls: &AtomicUsize,
    total_deals: usize,
    start: Instant,
) -> Vec<[u8; 4]> {
    use rayon::prelude::*;

    let games: Vec<(usize, u8)> = (0..deals.len())
        .flat_map(|d| (0..4u8).map(move |s| (d, s)))
        .collect();

    let results: Vec<(usize, u8, u8)> = games
        .par_iter()
        .map(|&(d, suit)| {
            let (dealer, hands) = deals[d];
            let game_seed = args.seed + (base_idx + d) as u64 * 100 + suit as u64;

            let make_config = || IsDdConfig {
                determinizations: args.dets,
                time_limit_ms: if args.time_ms == 0 { None } else { Some(args.time_ms) },
                world_batch: args.world_batch,
                ..Default::default()
            };

            // One player per seat: IS-DD is observer-bound, and a single shared
            // instance would be conditioned on a hand three of the seats never saw.
            let mut players: Vec<IsDdPlayer> = (0..4u8)
                .map(|seat| {
                    let p = IsDdPlayer::new(make_config(), seat, game_seed + seat as u64);
                    let src: Box<dyn WorldSource> = if args.worlds == "sidecar" {
                        Box::new(SidecarWorldSource::new(
                            args.url.clone(),
                            1.0,
                            Duration::from_secs(120),
                        ))
                    } else {
                        Box::new(UniformWorldSource)
                    };
                    p.with_world_source(src)
                })
                .collect();

            let mut state = if args.auction {
                let mut s = GameState::new(dealer, hands);
                for p in players.iter_mut() {
                    p.init_deal(&s);
                }
                // dealer+1 opens 80 in `suit`, three passes.
                for &act in [encode_bid(8, suit), BID_PASS, BID_PASS, BID_PASS].iter() {
                    let actor = s.current_player();
                    assert!(
                        s.legal_actions() & (1u64 << act) != 0,
                        "enchère synthétique illégale"
                    );
                    for p in players.iter_mut() {
                        p.observe(&s, actor, act);
                    }
                    s.step(act);
                }
                assert_eq!(s.phase, Phase::Playing);
                assert_eq!(s.contract.trump, suit);
                s
            } else {
                let s = GameState::setup_dd(dealer, hands, suit);
                for p in players.iter_mut() {
                    p.init_deal(&s);
                }
                s
            };

            let ctx = MatchContext::new(dealer);

            while state.phase == Phase::Playing {
                let actor = state.current_player();
                calls.fetch_add(1, Ordering::Relaxed);
                let action = match players[actor as usize].decide(&state, &ctx) {
                    Ok(d) => d.action,
                    Err(e) => {
                        // A world source that fails is not a small perturbation:
                        // substituting the lowest legal card plays a *different,
                        // much weaker* agent for that decision, and the label
                        // would silently stop describing IS-DD. Surface it.
                        let k = dry.fetch_add(1, Ordering::Relaxed);
                        if k < 5 {
                            eprintln!("  échec source de mondes (#{k}) : {e}");
                        }
                        state.legal_actions().trailing_zeros() as u8
                    }
                };
                for p in players.iter_mut() {
                    p.observe(&state, actor, action);
                }
                state.step(action);
            }

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 200 == 0 {
                let el = start.elapsed().as_secs_f64();
                let dd = done as f64 / 4.0;
                eprintln!(
                    "  {:.0}/{} donnes ({:.3}/s) {:.0} s, ETA {:.0} min",
                    dd,
                    total_deals,
                    dd / el,
                    el,
                    (total_deals as f64 - dd) / (dd / el) / 60.0
                );
            }

            (d, suit, state.points[0])
        })
        .collect();

    let mut out = vec![[0u8; 4]; deals.len()];
    for (d, suit, pts) in results {
        out[d][suit as usize] = pts;
    }
    out
}

// ══════════════════════════════════════════════════════════════════════
//  Comparison
// ══════════════════════════════════════════════════════════════════════

fn compare(old: &[[u8; 4]], new: &[[u8; 4]], args: &Args) {
    let pairs: Vec<(f64, f64)> = old
        .iter()
        .zip(new.iter())
        .flat_map(|(o, n)| (0..4).map(move |s| (o[s] as f64, n[s] as f64)))
        .collect();
    let m = pairs.len() as f64;

    let om = pairs.iter().map(|p| p.0).sum::<f64>() / m;
    let nm = pairs.iter().map(|p| p.1).sum::<f64>() / m;

    let diffs: Vec<f64> = pairs.iter().map(|(o, n)| n - o).collect();
    let dm = diffs.iter().sum::<f64>() / m;
    let dsd = (diffs.iter().map(|d| (d - dm).powi(2)).sum::<f64>() / (m - 1.0)).sqrt();
    let se = dsd / m.sqrt();
    let mad = diffs.iter().map(|d| d.abs()).sum::<f64>() / m;
    let rms = (diffs.iter().map(|d| d * d).sum::<f64>() / m).sqrt();

    let osd = (pairs.iter().map(|p| (p.0 - om).powi(2)).sum::<f64>() / m).sqrt();
    let nsd = (pairs.iter().map(|p| (p.1 - nm).powi(2)).sum::<f64>() / m).sqrt();
    let cov = pairs.iter().map(|p| (p.0 - om) * (p.1 - nm)).sum::<f64>() / m;
    let r = cov / (osd * nsd);

    let mut sorted = diffs.clone();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let q = |f: f64| sorted[((m - 1.0) * f) as usize];

    println!("\n=== {} vs {} ===", args.name, args.baseline);
    println!("paires      : {} ({} donnes x 4 couleurs)", pairs.len(), old.len());
    println!("moyenne anc.: {om:.2} pts cartes NS");
    println!("moyenne nouv: {nm:.2}");
    println!("écart moyen : {dm:+.2}  (± {:.2} SE, sd {dsd:.2})", se);
    println!("  significatif : {}", if dm.abs() > 2.0 * se { "oui (|Δ| > 2 SE)" } else { "NON" });
    println!("|Δ| moyen   : {mad:.2}   RMS {rms:.2}");
    println!(
        "quantiles Δ : p1 {:.0}  p10 {:.0}  p25 {:.0}  méd {:.0}  p75 {:.0}  p90 {:.0}  p99 {:.0}",
        q(0.01), q(0.10), q(0.25), q(0.50), q(0.75), q(0.90), q(0.99)
    );
    println!("corrélation : r = {r:.4}   (r² = {:.4})", r * r);
    println!("identiques  : {:.1} %", diffs.iter().filter(|d| **d == 0.0).count() as f64 / m * 100.0);

    // ── What actually reaches a bidding decision ───────────────────────
    // A label only matters through the choice it drives. Two probes:
    // which trump looks best, and whether a contract clears a threshold.
    let mut argmax_same = 0usize;
    for (o, n) in old.iter().zip(new.iter()) {
        let bo = (0..4).max_by_key(|&s| o[s]).unwrap();
        let bn = (0..4).max_by_key(|&s| n[s]).unwrap();
        if bo == bn {
            argmax_same += 1;
        }
    }
    println!(
        "\nmeilleure couleur inchangée : {:.1} % des donnes",
        argmax_same as f64 / old.len() as f64 * 100.0
    );

    if !args.json.is_empty() {
        let mut thr_rows = String::new();
        for (i, thr) in [80.0f64, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0]
            .iter()
            .enumerate()
        {
            let oa = pairs.iter().filter(|p| p.0 >= *thr).count() as f64 / m * 100.0;
            let na = pairs.iter().filter(|p| p.1 >= *thr).count() as f64 / m * 100.0;
            let di = pairs.iter().filter(|p| (p.0 >= *thr) != (p.1 >= *thr)).count() as f64
                / m
                * 100.0;
            if i > 0 {
                thr_rows.push(',');
            }
            thr_rows.push_str(&format!(
                "{{\"threshold\":{thr},\"old_pct\":{oa:.3},\"new_pct\":{na:.3},\"disagree_pct\":{di:.3}}}"
            ));
        }
        let json = format!(
            "{{\"name\":\"{}\",\"baseline\":\"{}\",\"deals\":{},\"pairs\":{},\
             \"old_mean\":{om:.4},\"new_mean\":{nm:.4},\"delta_mean\":{dm:.4},\
             \"delta_se\":{se:.4},\"delta_sd\":{dsd:.4},\"abs_delta_mean\":{mad:.4},\
             \"rms\":{rms:.4},\"pearson_r\":{r:.6},\"identical_pct\":{:.3},\
             \"best_suit_same_pct\":{:.3},\"thresholds\":[{thr_rows}]}}",
            args.name,
            args.baseline,
            old.len(),
            pairs.len(),
            diffs.iter().filter(|d| **d == 0.0).count() as f64 / m * 100.0,
            argmax_same as f64 / old.len() as f64 * 100.0,
        );
        std::fs::write(&args.json, json).expect("write --json");
        eprintln!("stats écrites dans {}", args.json);
    }

    println!("\nfranchissement de seuil (la décision d'annonce) :");
    println!("{:<8} {:>10} {:>10} {:>12}", "seuil", "anc. ≥", "nouv. ≥", "désaccord");
    println!("{}", "-".repeat(44));
    for thr in [80.0, 90.0, 100.0, 110.0, 120.0, 130.0, 140.0, 150.0, 160.0] {
        let oa = pairs.iter().filter(|p| p.0 >= thr).count();
        let na = pairs.iter().filter(|p| p.1 >= thr).count();
        let dis = pairs.iter().filter(|p| (p.0 >= thr) != (p.1 >= thr)).count();
        println!(
            "{:<8.0} {:>9.1}% {:>9.1}% {:>11.1}%",
            thr,
            oa as f64 / m * 100.0,
            na as f64 / m * 100.0,
            dis as f64 / m * 100.0
        );
    }
}
