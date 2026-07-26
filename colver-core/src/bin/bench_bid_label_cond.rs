//! How much does the auction narrow a bid label?
//!
//! At a real bid position taken from a held-out COLVGM01 corpus, the seat to move
//! sees its own 8 cards plus the auction so far. Two ways to turn that into a label:
//!
//!   uniform — redeal the 24 unseen cards at random (ignores the auction entirely)
//!   playgen — sample deals from the model's posterior given the auction prefix
//!
//! Both are averaged over `--worlds` samples and DD-solved for all 4 trumps. We
//! score each against the truth: the DD value of the deal that was actually held.
//! The comparison that matters is RMSE against truth — a narrower posterior is
//! only worth having if it is narrower around the right place.
//!
//! Usage:
//!   cargo run --bin bench_bid_label_cond --release --features parallel -- \
//!     --games data/training/heldout_20k_s90210.bin --positions 150 --worlds 40

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::SeedableRng;

use colver_core::game_replay::GameReplay;
use colver_core::playgen::infer::{PlaygenModel, PlaygenSampler};
use colver_core::solver::{new_tt_buffer, solve_for_trump_reuse_tt};
use colver_core::state::{GameState, Phase};

/// Mean and sd of a sample.
fn mean_sd(v: &[f64]) -> (f64, f64) {
    if v.is_empty() {
        return (f64::NAN, f64::NAN);
    }
    let m = v.iter().sum::<f64>() / v.len() as f64;
    let var = v.iter().map(|x| (x - m).powi(2)).sum::<f64>() / v.len() as f64;
    (m, var.sqrt())
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut games_path = String::from("data/training/heldout_20k_s90210.bin");
    let mut playgen_path = String::from("models/playgen/playgen_v2_final.bin");
    let mut positions: usize = 150;
    let mut worlds: usize = 40;
    let mut min_prior: usize = 2;
    let mut temperature: f32 = 1.0;
    let mut seed: u64 = 11;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => { i += 1; games_path = args[i].clone(); }
            "--playgen" => { i += 1; playgen_path = args[i].clone(); }
            "--positions" => { i += 1; positions = args[i].parse().unwrap(); }
            "--worlds" => { i += 1; worlds = args[i].parse().unwrap(); }
            "--min-prior" => { i += 1; min_prior = args[i].parse().unwrap(); }
            "--temperature" => { i += 1; temperature = args[i].parse().unwrap(); }
            "--seed" => { i += 1; seed = args[i].parse().unwrap(); }
            other => panic!("unknown arg {other}"),
        }
        i += 1;
    }

    eprintln!("Loading {games_path}...");
    let replays = GameReplay::load_all(&games_path).expect("load replays");
    eprintln!("  {} games", replays.len());
    let model = Arc::new(PlaygenModel::load(&playgen_path).expect("load playgen"));
    eprintln!("  playgen loaded (v2={})", model.v2);

    // ── Pick all positions FIRST, from a stream nothing under test consumes.
    // (Repo rule: never draw the questions from the stream the samplers use.)
    let mut pick_rng = StdRng::seed_from_u64(seed);
    let mut chosen: Vec<(usize, usize)> = Vec::new(); // (replay idx, prefix len)
    let mut order: Vec<usize> = (0..replays.len()).collect();
    order.shuffle(&mut pick_rng);
    for &ri in &order {
        if chosen.len() >= positions {
            break;
        }
        let r = &replays[ri];
        // Bid positions are the leading actions until the state leaves Bidding.
        let mut state = GameState::new(r.dealer, r.hands);
        let mut bid_len = 0usize;
        for &a in &r.actions {
            if state.phase != Phase::Bidding {
                break;
            }
            bid_len += 1;
            state.step(a);
        }
        // Need at least `min_prior` actions of context, and a position left to ask about.
        if bid_len <= min_prior {
            continue;
        }
        let prefix = min_prior + (pick_rng.gen_range(0..(bid_len - min_prior)));
        chosen.push((ri, prefix));
    }
    eprintln!("  {} bid positions selected\n", chosen.len());

    use rand::Rng;
    use rayon::prelude::*;

    struct Row {
        truth: [f64; 4],
        uni_mean: [f64; 4],
        uni_sd: [f64; 4],
        pg_mean: [f64; 4],
        pg_sd: [f64; 4],
        pg_worlds: usize,
        /// Non-pass actions visible in the prefix — how much the auction has said.
        info: usize,
    }

    let rows: Vec<Row> = chosen
        .par_iter()
        .enumerate()
        .filter_map(|(k, &(ri, prefix))| {
            let r = &replays[ri];
            let mut tt = new_tt_buffer();
            let mut rng = StdRng::seed_from_u64(seed.wrapping_mul(7919) + k as u64);

            // Replay the auction prefix, feeding the playgen sampler as we go.
            let state0 = GameState::new(r.dealer, r.hands);
            let mut state = state0;
            let mut sampler = PlaygenSampler::new(model.clone());
            // Observer is whoever is to move at the cut point; we must know that
            // before init_deal, so walk the prefix once to find it.
            {
                let mut s = state0;
                for &a in r.actions.iter().take(prefix) {
                    s.step(a);
                }
                if s.phase != Phase::Bidding {
                    return None;
                }
                sampler.init_deal(&state0, s.current_player());
            }
            let observer = sampler.observer();
            for &a in r.actions.iter().take(prefix) {
                sampler.record_action(&state, state.current_player(), a);
                state.step(a);
            }
            if state.phase != Phase::Bidding || sampler.is_dead() {
                return None;
            }

            let mine = state.hands[observer as usize];
            let unseen: Vec<u8> = (0..32u8).filter(|&c| mine & (1 << c) == 0).collect();

            // Truth: DD value of the deal actually held.
            let mut truth = [0.0f64; 4];
            for suit in 0..4usize {
                truth[suit] =
                    solve_for_trump_reuse_tt(r.hands, r.dealer, suit as u8, &mut tt)[0] as f64;
            }

            // Uniform: redeal the 24 unseen cards, auction ignored.
            let mut uni: [Vec<f64>; 4] = Default::default();
            let mut deck = unseen.clone();
            for _ in 0..worlds {
                deck.shuffle(&mut rng);
                let mut hands = [0u32; 4];
                hands[observer as usize] = mine;
                let mut c = 0;
                for s in 0..4u8 {
                    if s == observer {
                        continue;
                    }
                    for _ in 0..8 {
                        hands[s as usize] |= 1 << deck[c];
                        c += 1;
                    }
                }
                for suit in 0..4usize {
                    uni[suit].push(
                        solve_for_trump_reuse_tt(hands, r.dealer, suit as u8, &mut tt)[0] as f64,
                    );
                }
            }

            // Playgen: posterior given the auction prefix.
            let pg_deals = sampler.generate_deals_from_auction(&state, worlds, temperature, &mut rng);
            let mut pg: [Vec<f64>; 4] = Default::default();
            for w in &pg_deals {
                debug_assert_eq!(w[observer as usize], mine, "playgen altered the observer hand");
                for suit in 0..4usize {
                    pg[suit].push(
                        solve_for_trump_reuse_tt(*w, r.dealer, suit as u8, &mut tt)[0] as f64,
                    );
                }
            }
            if pg_deals.is_empty() {
                return None;
            }

            let mut row = Row {
                truth,
                uni_mean: [0.0; 4],
                uni_sd: [0.0; 4],
                pg_mean: [0.0; 4],
                pg_sd: [0.0; 4],
                pg_worlds: pg_deals.len(),
                info: r.actions.iter().take(prefix).filter(|&&a| a != 0).count(),
            };
            for suit in 0..4 {
                let (m, s) = mean_sd(&uni[suit]);
                row.uni_mean[suit] = m;
                row.uni_sd[suit] = s;
                let (m, s) = mean_sd(&pg[suit]);
                row.pg_mean[suit] = m;
                row.pg_sd[suit] = s;
            }
            if k % 10 == 0 {
                eprintln!("  position {k}/{} done", chosen.len());
            }
            Some(row)
        })
        .collect();

    // ── Aggregate.
    let n = rows.len();
    let mut uni_sd = 0.0;
    let mut pg_sd = 0.0;
    let mut uni_se = 0.0;
    let mut pg_se = 0.0;
    let mut pg_world_total = 0usize;
    let mut cnt = 0usize;
    for r in &rows {
        pg_world_total += r.pg_worlds;
        for s in 0..4 {
            uni_sd += r.uni_sd[s].powi(2);
            pg_sd += r.pg_sd[s].powi(2);
            uni_se += (r.uni_mean[s] - r.truth[s]).powi(2);
            pg_se += (r.pg_mean[s] - r.truth[s]).powi(2);
            cnt += 1;
        }
    }
    let c = cnt as f64;
    let (uni_sd, pg_sd) = ((uni_sd / c).sqrt(), (pg_sd / c).sqrt());
    let (uni_rmse, pg_rmse) = ((uni_se / c).sqrt(), (pg_se / c).sqrt());

    println!("\n=== Auction-conditioned bid labels ===");
    println!("positions: {n}   worlds/position requested: {worlds}");
    println!("playgen worlds actually returned: {:.1} avg", pg_world_total as f64 / n.max(1) as f64);
    println!();
    println!("                          uniform    playgen");
    println!("posterior spread (sd)    {uni_sd:8.1}  {pg_sd:9.1}  pts");
    println!("RMSE vs true dd_pts      {uni_rmse:8.1}  {pg_rmse:9.1}  pts");
    println!();
    println!(
        "spread reduction from the auction: {:.1}%",
        (1.0 - pg_sd / uni_sd) * 100.0
    );
    println!(
        "RMSE reduction from the auction:   {:.1}%",
        (1.0 - pg_rmse / uni_rmse) * 100.0
    );
    println!(
        "\nFor scale: a single dealt deal (what training uses today) has RMSE {:.1} pts\n\
         against its own conditional mean — that is the noise being averaged out.",
        uni_sd
    );

    // Stratify by how much the auction has actually said.
    println!("\n=== By auction content (non-pass actions visible) ===");
    println!("{:<12} {:>5} {:>10} {:>10} {:>9}", "non-pass", "n", "uni RMSE", "pg RMSE", "gain");
    for (lo, hi, label) in [(0usize, 0usize, "0 (passes)"), (1, 1, "1"), (2, 2, "2"), (3, 99, "3+")] {
        let sel: Vec<&Row> = rows.iter().filter(|r| r.info >= lo && r.info <= hi).collect();
        if sel.is_empty() { continue; }
        let (mut u, mut g) = (0.0f64, 0.0f64);
        let mut k = 0usize;
        for r in &sel {
            for s in 0..4 {
                u += (r.uni_mean[s] - r.truth[s]).powi(2);
                g += (r.pg_mean[s] - r.truth[s]).powi(2);
                k += 1;
            }
        }
        let (u, g) = ((u / k as f64).sqrt(), (g / k as f64).sqrt());
        println!("{:<12} {:>5} {:>10.1} {:>10.1} {:>8.1}%", label, sel.len(), u, g, (1.0 - g / u) * 100.0);
    }
}
