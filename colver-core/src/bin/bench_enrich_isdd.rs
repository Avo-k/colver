//! Dimension the cost of an IS-DD score layer, uniform worlds vs playgen sidecar.
//!
//! `enrich_pool_isdd` forces each trump with `GameState::setup_dd`, which lands
//! directly in `Phase::Playing` with an empty auction. The playgen sidecar cannot
//! represent that position: `/play_worlds` replays `GameState::new(dealer, hands)`
//! through an action list, so reaching Playing *requires* an auction. Worse, for
//! playgen v2 the trump is carried only by the bid tokens (`perm` is identity),
//! so a bid-less prefix leaves the model blind to the trump.
//!
//! This bench therefore forces the trump with a **synthetic auction** — dealer+1
//! opens 80 in the target suit, three passes — which is the cheapest legal prefix
//! that names the trump. It measures throughput only; see the report for why the
//! resulting labels are not calibration-neutral.
//!
//! Usage:
//!   cargo run --bin bench_enrich_isdd --release --features parallel -- \
//!     --deals 100 --worlds sidecar --url http://localhost:8003 --time-ms 20

use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::agent::isdd::IsDdPlayer;
use colver_core::agent::{CardPlayer, MatchContext};
use colver_core::bid_train_env::DealPool;
use colver_core::bidding::{encode_bid, BID_PASS};
use colver_core::is_dd::IsDdConfig;
use colver_core::state::{GameState, Phase};
use colver_core::worlds::{SidecarWorldSource, UniformWorldSource, WorldSource};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut pool_path = String::from("data/deals/base_5M.bin");
    let mut num_deals: usize = 100;
    let mut time_ms: u32 = 20;
    let mut dets: u32 = 20;
    let mut seed: u64 = 42;
    let mut source = String::from("uniform");
    let mut url = std::env::var("COLVER_PLAYGEN_GPU_URL")
        .unwrap_or_else(|_| "http://localhost:8003".into());
    let mut threads: usize = 0;
    let mut wbatch: usize = 128;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--deals" => { i += 1; num_deals = args[i].parse().unwrap(); }
            "--time-ms" => { i += 1; time_ms = args[i].parse().unwrap(); }
            "--dets" => { i += 1; dets = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            "--worlds" => { i += 1; source = args[i].clone(); }
            "--url" => { i += 1; url = args[i].clone(); }
            "--threads" => { i += 1; threads = args[i].parse().unwrap(); }
            "--world-batch" => { i += 1; wbatch = args[i].parse().unwrap(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    if threads > 0 {
        rayon::ThreadPoolBuilder::new().num_threads(threads).build_global().unwrap();
    }
    let nthreads = rayon::current_num_threads();

    eprintln!("Loading {pool_path}...");
    let pool = DealPool::load(&pool_path).expect("load pool");
    let sampled: Vec<(u8, [u32; 4])> = (0..num_deals.min(pool.len()))
        .map(|idx| { let d = pool.get(idx); (d.dealer, d.hands) })
        .collect();
    let num_deals = sampled.len();

    if source == "sidecar" {
        let probe = SidecarWorldSource::new(url.clone(), 1.0, Duration::from_secs(10));
        match probe.health_check() {
            Ok(s) => eprintln!("sidecar {url} ok: {}", s.trim()),
            Err(e) => { eprintln!("sidecar unreachable: {e}"); std::process::exit(1); }
        }
    }

    let make_config = || IsDdConfig {
        determinizations: dets,
        time_limit_ms: if time_ms == 0 { None } else { Some(time_ms) },
        world_batch: wbatch,
        ..Default::default()
    };

    let total_games = num_deals * 4;
    eprintln!(
        "\n{} deals x 4 suits = {} games | worlds={} | {}ms/{} dets | wbatch={} | {} threads",
        num_deals, total_games, source, time_ms, dets, wbatch, nthreads
    );

    let start = Instant::now();
    let progress = AtomicUsize::new(0);
    let failures = AtomicUsize::new(0);
    let calls = AtomicUsize::new(0);
    let fail_by_trick: Vec<AtomicUsize> = (0..8).map(|_| AtomicUsize::new(0)).collect();
    let fail_by_trick = &fail_by_trick;

    let results: Vec<u8> = {
        use rayon::prelude::*;
        let games: Vec<(usize, u8)> =
            (0..num_deals).flat_map(|d| (0..4u8).map(move |s| (d, s))).collect();

        games
            .par_iter()
            .map(|&(deal_idx, suit)| {
                let (dealer, hands) = sampled[deal_idx];
                let game_seed = seed + deal_idx as u64 * 100 + suit as u64;

                // Four seats, each with its own observer-side world source.
                let mut players: Vec<IsDdPlayer> = (0..4u8)
                    .map(|seat| {
                        let p = IsDdPlayer::new(make_config(), seat, game_seed + seat as u64);
                        let src: Box<dyn WorldSource> = if source == "sidecar" {
                            Box::new(SidecarWorldSource::new(
                                url.clone(), 1.0, Duration::from_secs(30),
                            ))
                        } else {
                            Box::new(UniformWorldSource)
                        };
                        p.with_world_source(src)
                    })
                    .collect();

                let mut state = GameState::new(dealer, hands);
                for p in players.iter_mut() {
                    p.init_deal(&state);
                }

                // Synthetic auction: dealer+1 opens 80 in `suit`, then three passes.
                // Cheapest legal prefix that names the trump to playgen.
                let auction = [encode_bid(8, suit), BID_PASS, BID_PASS, BID_PASS];
                for &a in auction.iter() {
                    let actor = state.current_player();
                    assert!(state.legal_actions() & (1u64 << a) != 0, "illegal synthetic bid");
                    for p in players.iter_mut() {
                        p.observe(&state, actor, a);
                    }
                    state.step(a);
                }
                assert_eq!(state.phase, Phase::Playing, "auction did not reach play");
                assert_eq!(state.contract.trump, suit, "trump mismatch");

                let ctx = MatchContext::new(dealer);
                let mut rng = StdRng::seed_from_u64(game_seed);
                let _ = &mut rng;

                while state.phase == Phase::Playing {
                    let actor = state.current_player();
                    calls.fetch_add(1, Ordering::Relaxed);
                    let action = match players[actor as usize].decide(&state, &ctx) {
                        Ok(d) => d.action,
                        Err(_) => {
                            failures.fetch_add(1, Ordering::Relaxed);
                            let trick = (state.tricks_won[0] + state.tricks_won[1]) as usize;
                            fail_by_trick[trick.min(7)].fetch_add(1, Ordering::Relaxed);
                            let legal = state.legal_actions();
                            legal.trailing_zeros() as u8
                        }
                    };
                    for p in players.iter_mut() {
                        p.observe(&state, actor, action);
                    }
                    state.step(action);
                }

                let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                if done % 20 == 0 || done == total_games {
                    let el = start.elapsed().as_secs_f64();
                    eprintln!(
                        "  {done}/{total_games} games ({:.2} games/s, {:.3} deals/s) {:.1}s",
                        done as f64 / el, done as f64 / 4.0 / el, el
                    );
                }
                state.points[0]
            })
            .collect()
    };

    let elapsed = start.elapsed().as_secs_f64();
    let deals_per_s = num_deals as f64 / elapsed;
    let mean: f64 = results.iter().map(|&p| p as f64).sum::<f64>() / results.len() as f64;

    println!("\n=== {} | {}ms x {} dets | {} threads ===", source, time_ms, dets, nthreads);
    println!("deals:        {num_deals}  ({total_games} games)");
    println!("elapsed:      {elapsed:.1}s");
    println!("throughput:   {deals_per_s:.3} deals/s   ({:.2} games/s)", total_games as f64 / elapsed);
    println!("NS mean pts:  {mean:.1}");
    let nc = calls.load(Ordering::Relaxed);
    let nf = failures.load(Ordering::Relaxed);
    println!("decisions:    {nc}");
    println!("empty-world:  {nf}  ({:.1}% of decisions)", nf as f64 / nc.max(1) as f64 * 100.0);
    print!("  by trick:   ");
    for (t, c) in fail_by_trick.iter().enumerate() {
        print!("T{}={} ", t + 1, c.load(Ordering::Relaxed));
    }
    println!();
    for (label, n) in [("100k", 100_000.0), ("1M", 1_000_000.0), ("5M", 5_000_000.0)] {
        let secs = n / deals_per_s;
        println!("  extrapolated {label}: {:.1} h", secs / 3600.0);
    }
}
