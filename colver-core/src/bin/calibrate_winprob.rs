/// Calibrate match win probability P(win | my_score, opp_score).
///
/// Runs full matches from 0/0 → 2000, logging every intermediate score state.
/// Each deal produces a data point: (ns_cum, ew_cum, eventual_winner).
/// Full replays are saved in COLVMR01 binary format.
///
/// Usage:
///   cargo run --bin calibrate_winprob --release -- --matches 10000

use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{self, BID_OBS_DIM};
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use colver_core::state::{GameState, Phase};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU32, Ordering};
use std::time::Instant;

const MATCH_TARGET: i32 = 2000;

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
    }
    None
}
fn parse_flag_usize(args: &[String], flag: &str, default: usize) -> usize {
    parse_flag(args, flag).and_then(|s| s.parse().ok()).unwrap_or(default)
}
fn parse_flag_u64(args: &[String], flag: &str, default: u64) -> u64 {
    parse_flag(args, flag).and_then(|s| s.parse().ok()).unwrap_or(default)
}

// ══════════════════════════════════════════════════════════════════════
//  Match replay data
// ══════════════════════════════════════════════════════════════════════

struct DealReplay {
    dealer: u8,
    hands: [u32; 4],
    actions: Vec<u8>,
    ns_deal_score: i16,
    ew_deal_score: i16,
}

struct MatchReplay {
    winner: u8,
    ns_final: i32,
    ew_final: i32,
    deals: Vec<DealReplay>,
}

/// Score state after each deal: (ns_cum, ew_cum) BEFORE the deal, plus outcome.
struct ScorePoint {
    ns_before: i32,
    ew_before: i32,
    winner: u8, // eventual match winner (0=NS, 1=EW)
}

/// Write replays in COLVMR01 format (all matches start from 0/0).
fn write_replays(path: &str, replays: &[MatchReplay]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"COLVMR01")?;
    f.write_all(&(replays.len() as u32).to_le_bytes())?;
    for m in replays {
        // ns_start=0, ew_start=0 for all matches
        f.write_all(&0i16.to_le_bytes())?;
        f.write_all(&0i16.to_le_bytes())?;
        f.write_all(&[m.winner])?;
        f.write_all(&(m.ns_final as i16).to_le_bytes())?;
        f.write_all(&(m.ew_final as i16).to_le_bytes())?;
        f.write_all(&[m.deals.len() as u8])?;
        for d in &m.deals {
            f.write_all(&[d.dealer])?;
            for &h in &d.hands {
                f.write_all(&h.to_le_bytes())?;
            }
            f.write_all(&[d.actions.len() as u8])?;
            f.write_all(&d.actions)?;
            f.write_all(&d.ns_deal_score.to_le_bytes())?;
            f.write_all(&d.ew_deal_score.to_le_bytes())?;
        }
    }
    f.flush()?;
    Ok(())
}

// ══════════════════════════════════════════════════════════════════════
//  Match simulation
// ══════════════════════════════════════════════════════════════════════

fn play_one_deal_logged(
    bid_net: &mut BidNet,
    dmc_net: &mut DmcNet,
    rng: &mut StdRng,
    dealer: u8,
    ns_cum: i32,
    ew_cum: i32,
) -> DealReplay {
    let mut state = GameState::deal_random(dealer, rng);
    let hands = state.hands;
    let mut tracking = EnvTracking::new();
    tracking.reset(dealer);
    let mut actions = Vec::with_capacity(44);

    let bid_obs_dim = bid_net.obs_dim();
    let mut bid_obs_buf = vec![0.0f32; bid_obs_dim];
    let mut obs_buf = vec![0.0f32; OBS_DIM_TR];
    let score_aware = bid_obs_dim > BID_OBS_DIM;

    while state.phase == Phase::Bidding {
        if score_aware {
            let player = state.current_player();
            let team = colver_core::state::GameState::player_team(player);
            let (my, opp) = if team == 0 { (ns_cum, ew_cum) } else { (ew_cum, ns_cum) };
            bid_obs::write_bid_observation_score_aware(
                &mut bid_obs_buf, 0, &state, &tracking.bid_history, my, opp,
            );
        } else {
            bid_obs::write_bid_observation(&mut bid_obs_buf, 0, &state, &tracking.bid_history);
        }
        let legal_mask = state.legal_actions();
        let action = bid_net.best_action_fast(&bid_obs_buf, legal_mask);
        actions.push(action);
        tracking.track_action(&state, action);
        state.step(action);
    }

    while !state.is_terminal() {
        dmc_obs::write_observation_tr(&mut obs_buf, 0, &state, &tracking);
        let order = dmc_obs::current_player_order(&state, &tracking);
        let canonical_mask =
            dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
        let (canonical_best, _) = dmc_net.best_action(&obs_buf, canonical_mask as u32);
        let action = dmc_obs::card_to_physical(canonical_best, &order);
        actions.push(action);
        tracking.track_action(&state, action);
        state.step(action);
    }

    let score = state.deal_score();
    DealReplay {
        dealer,
        hands,
        actions,
        ns_deal_score: score.scores[0],
        ew_deal_score: score.scores[1],
    }
}

/// Play a full match from 0/0 and return (replay, score_points).
/// score_points: one entry per deal = (ns_cum_before, ew_cum_before, eventual_winner).
fn play_full_match(
    bid_net: &mut BidNet,
    dmc_net: &mut DmcNet,
    rng: &mut StdRng,
) -> (MatchReplay, Vec<ScorePoint>) {
    let mut ns_cum: i32 = 0;
    let mut ew_cum: i32 = 0;
    let mut dealer: u8 = rng.gen_range(0..4);
    let mut deals = Vec::with_capacity(12);
    let mut snapshots: Vec<(i32, i32)> = Vec::with_capacity(12);

    while ns_cum < MATCH_TARGET && ew_cum < MATCH_TARGET {
        // Record score BEFORE this deal
        snapshots.push((ns_cum, ew_cum));

        let deal = play_one_deal_logged(bid_net, dmc_net, rng, dealer, ns_cum, ew_cum);
        if deal.ns_deal_score != 0 || deal.ew_deal_score != 0 {
            ns_cum += deal.ns_deal_score as i32;
            ew_cum += deal.ew_deal_score as i32;
        }
        dealer = (dealer + 1) % 4;
        deals.push(deal);
    }

    let winner = if ns_cum >= MATCH_TARGET && ew_cum >= MATCH_TARGET {
        if ns_cum >= ew_cum { 0 } else { 1 }
    } else if ns_cum >= MATCH_TARGET { 0 } else { 1 };

    let score_points: Vec<ScorePoint> = snapshots
        .into_iter()
        .map(|(ns, ew)| ScorePoint { ns_before: ns, ew_before: ew, winner })
        .collect();

    let replay = MatchReplay { winner, ns_final: ns_cum, ew_final: ew_cum, deals };
    (replay, score_points)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.iter().any(|a| a == "--help" || a == "-h") {
        eprintln!("Usage: calibrate_winprob [OPTIONS]");
        eprintln!("  --matches N        Number of full matches from 0/0 (default 10000)");
        eprintln!("  --bid-model PATH   (default models/bid_v3_max_20M/bid_nn_final.bin)");
        eprintln!("  --bid-hidden N     (default 512)");
        eprintln!("  --play-model PATH  (default models/play_v2/play_final.bin)");
        eprintln!("  --seed N           (default 42)");
        eprintln!("  --threads N        (default 16)");
        eprintln!("  --output PATH      CSV output (default data/winprob_points.csv)");
        eprintln!("  --replay PATH      Replay output (default data/winprob_replays.bin)");
        std::process::exit(0);
    }

    let num_matches = parse_flag_usize(&args, "--matches", 10_000);
    let bid_model = parse_flag(&args, "--bid-model")
        .unwrap_or_else(|| "models/bid_v3_max_20M/bid_nn_final.bin".into());
    let bid_hidden = parse_flag_usize(&args, "--bid-hidden", 512);
    let play_model = parse_flag(&args, "--play-model")
        .unwrap_or_else(|| "models/play_v2/play_final.bin".into());
    let seed = parse_flag_u64(&args, "--seed", 42);
    let num_threads = parse_flag_usize(&args, "--threads", 16);
    let output_csv = parse_flag(&args, "--output")
        .unwrap_or_else(|| "data/winprob_points.csv".into());
    let output_replay = parse_flag(&args, "--replay")
        .unwrap_or_else(|| "data/winprob_replays.bin".into());

    eprintln!("Loading bid model: {}", bid_model);
    let bn = BidNet::load_with_hidden(&bid_model, bid_hidden).expect("load bid");
    eprintln!("  obs={}, h={}, L={}, dueling={}", bn.obs_dim(), bn.hidden(), bn.layers(), bn.is_dueling());
    eprintln!("Loading play model: {}", play_model);
    let dn = DmcNet::load(&play_model).expect("load play");
    eprintln!("  obs={}, h={}, dueling={}", dn.obs_dim(), dn.hidden(), dn.is_dueling());
    drop(bn); drop(dn);

    eprintln!(
        "\n=== Full Match Calibration ===\n{} matches from 0/0, {} threads",
        num_matches, num_threads,
    );

    let start = Instant::now();
    let progress = AtomicU32::new(0);

    // Run matches in parallel, collect replays + score points
    let (all_replays, all_points): (Vec<MatchReplay>, Vec<ScorePoint>) = std::thread::scope(|s| {
        let chunk_size = (num_matches + num_threads - 1) / num_threads;
        let progress = &progress;

        let handles: Vec<_> = (0..num_threads)
            .map(|tid| {
                let my_count = if tid < num_threads - 1 {
                    chunk_size
                } else {
                    num_matches - chunk_size * tid
                };
                let bid_model = &bid_model;
                let play_model = &play_model;
                s.spawn(move || {
                    let mut bid_net = BidNet::load_with_hidden(bid_model, bid_hidden)
                        .expect("load bid net");
                    let mut dmc_net = DmcNet::load(&play_model).expect("load dmc net");
                    dmc_net.set_residual(true);
                    let mut rng = StdRng::seed_from_u64(seed + tid as u64 * 1_000_000);

                    let mut replays = Vec::with_capacity(my_count);
                    let mut points = Vec::with_capacity(my_count * 10);

                    for _ in 0..my_count {
                        let (replay, pts) = play_full_match(&mut bid_net, &mut dmc_net, &mut rng);
                        points.extend(pts);
                        replays.push(replay);

                        let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
                        if done % 500 == 0 || done == num_matches as u32 {
                            let elapsed = start.elapsed().as_secs_f64();
                            let rate = done as f64 / elapsed;
                            let eta = (num_matches as f64 - done as f64) / rate;
                            eprint!(
                                "\r  {}/{} matches ({:.0}/s, ETA {:.0}s)    ",
                                done, num_matches, rate, eta
                            );
                        }
                    }
                    (replays, points)
                })
            })
            .collect();

        let mut all_replays = Vec::with_capacity(num_matches);
        let mut all_points = Vec::new();
        for h in handles {
            let (r, p) = h.join().unwrap();
            all_replays.extend(r);
            all_points.extend(p);
        }
        (all_replays, all_points)
    });

    let elapsed = start.elapsed().as_secs_f64();
    let total_deals: usize = all_replays.iter().map(|m| m.deals.len()).sum();
    eprintln!(
        "\n\nDone: {} matches, {} deals, {} score points in {:.1}s ({:.0} matches/s)",
        all_replays.len(), total_deals, all_points.len(), elapsed,
        all_replays.len() as f64 / elapsed,
    );

    // Write replays
    if let Some(p) = std::path::Path::new(&output_replay).parent() {
        std::fs::create_dir_all(p).ok();
    }
    write_replays(&output_replay, &all_replays).expect("write replays");
    let file_size = std::fs::metadata(&output_replay).map(|m| m.len()).unwrap_or(0);
    eprintln!(
        "Replays: {} ({:.1}MB)",
        output_replay, file_size as f64 / 1_048_576.0,
    );

    // Write score points CSV: ns_before, ew_before, winner
    if let Some(p) = std::path::Path::new(&output_csv).parent() {
        std::fs::create_dir_all(p).ok();
    }
    let mut csv = String::with_capacity(all_points.len() * 20);
    csv.push_str("ns,ew,winner\n");
    for pt in &all_points {
        csv.push_str(&format!("{},{},{}\n", pt.ns_before, pt.ew_before, pt.winner));
    }
    std::fs::write(&output_csv, &csv).expect("write CSV");
    eprintln!("Score points: {} ({} rows)", output_csv, all_points.len());

    // ══════════════════════════════════════════════════════════════════
    //  Analysis: bin into grid and show win rates
    // ══════════════════════════════════════════════════════════════════

    // Match stats
    let ns_wins = all_replays.iter().filter(|m| m.winner == 0).count();
    let avg_deals = total_deals as f64 / all_replays.len() as f64;
    println!("\n=== Match Stats ===");
    println!("NS wins: {}/{} ({:.1}%)", ns_wins, all_replays.len(),
        ns_wins as f64 / all_replays.len() as f64 * 100.0);
    println!("Avg deals/match: {:.1}", avg_deals);

    // Bin score points into a grid for visualization
    let bin_size = 200;
    let n_bins = (MATCH_TARGET as usize + bin_size - 1) / bin_size; // 10 bins: 0-200, 200-400, ..., 1800-2000
    let mut grid_wins = vec![0u32; n_bins * n_bins];
    let mut grid_total = vec![0u32; n_bins * n_bins];

    for pt in &all_points {
        let i = ((pt.ns_before as usize).min(MATCH_TARGET as usize - 1)) / bin_size;
        let j = ((pt.ew_before as usize).min(MATCH_TARGET as usize - 1)) / bin_size;
        let idx = i * n_bins + j;
        grid_total[idx] += 1;
        if pt.winner == 0 { grid_wins[idx] += 1; }
    }

    println!("\nNS win% (binned {}pts) — rows: NS, cols: EW [count in brackets]", bin_size);
    print!("         ");
    for j in 0..n_bins { print!("{:>10}", format!("{}-{}", j * bin_size, (j + 1) * bin_size)); }
    println!();

    for i in 0..n_bins {
        print!("{:>4}-{:<4}|", i * bin_size, (i + 1) * bin_size);
        for j in 0..n_bins {
            let idx = i * n_bins + j;
            let total = grid_total[idx];
            if total > 0 {
                let wr = grid_wins[idx] as f64 / total as f64 * 100.0;
                print!(" {:>4.0}%[{:<3}]", wr, total);
            } else {
                print!("     —    ");
            }
        }
        println!();
    }

    // ══════════════════════════════════════════════════════════════════
    //  Sigmoid fit on raw score points (logistic regression via grid search)
    // ══════════════════════════════════════════════════════════════════

    println!("\n=== Sigmoid Fits ===");

    // Model 1: σ(k × Δ / max(2000 - min, δ))  [current]
    // Model 2: σ(k × Δ / (4000 - sum + δ))     [sum-based]
    // Model 3: σ(k × Δ / max(R_me, δ₁) + k₂ × Δ / max(R_opp, δ₂))  [two-term]
    // Model 4: σ(k × Δ × (1 + α × max(s_me, s_opp) / 2000) / δ) [score-scaled]

    // Use binned data for faster fitting (weighted by count)
    struct Bin { ns: f64, ew: f64, wr: f64, count: u32 }
    let mut bins: Vec<Bin> = Vec::new();
    for i in 0..n_bins {
        for j in 0..n_bins {
            let idx = i * n_bins + j;
            if grid_total[idx] >= 10 {
                bins.push(Bin {
                    ns: (i as f64 + 0.5) * bin_size as f64,
                    ew: (j as f64 + 0.5) * bin_size as f64,
                    wr: grid_wins[idx] as f64 / grid_total[idx] as f64,
                    count: grid_total[idx],
                });
            }
        }
    }
    let total_weight: f64 = bins.iter().map(|b| b.count as f64).sum();

    fn sigmoid(x: f64) -> f64 { 1.0 / (1.0 + (-x).exp()) }

    // Model 1: current sigmoid
    {
        let mut best = (0.0f64, 0.0f64, f64::MAX);
        for k_10 in 10..80 {
            let k = k_10 as f64 / 10.0;
            for d_i in 1..40 {
                let delta = d_i as f64 * 50.0;
                let mut wmse = 0.0;
                for b in &bins {
                    let denom = (2000.0 - b.ns.min(b.ew)).max(delta);
                    let pred = sigmoid(k * (b.ns - b.ew) / denom);
                    wmse += b.count as f64 * (b.wr - pred).powi(2);
                }
                wmse /= total_weight;
                if wmse < best.2 { best = (k, delta, wmse); }
            }
        }
        println!("Model 1 — σ(k×Δ / max(2000-min, δ))");
        println!("  k={:.1}, δ={:.0}, wMSE={:.6}", best.0, best.1, best.2);
    }

    // Model 2: sum-based denominator
    {
        let mut best = (0.0f64, 0.0f64, f64::MAX);
        for k_10 in 10..80 {
            let k = k_10 as f64 / 10.0;
            for d_i in 1..80 {
                let delta = d_i as f64 * 50.0;
                let mut wmse = 0.0;
                for b in &bins {
                    let denom = (4000.0 - b.ns - b.ew).max(delta);
                    let pred = sigmoid(k * (b.ns - b.ew) / denom);
                    wmse += b.count as f64 * (b.wr - pred).powi(2);
                }
                wmse /= total_weight;
                if wmse < best.2 { best = (k, delta, wmse); }
            }
        }
        println!("Model 2 — σ(k×Δ / max(4000-sum, δ))");
        println!("  k={:.1}, δ={:.0}, wMSE={:.6}", best.0, best.1, best.2);
    }

    // Model 3: product-based  σ(k × Δ × 2000 / max(R_me × R_opp, δ))
    {
        let mut best = (0.0f64, 0.0f64, f64::MAX);
        for k_10 in 1..100 {
            let k = k_10 as f64 / 10.0;
            for d_i in 1..100 {
                let delta = d_i as f64 * 10000.0;
                let mut wmse = 0.0;
                for b in &bins {
                    let r_me = 2000.0 - b.ns;
                    let r_opp = 2000.0 - b.ew;
                    let denom = (r_me * r_opp).max(delta);
                    let pred = sigmoid(k * (b.ns - b.ew) * 2000.0 / denom);
                    wmse += b.count as f64 * (b.wr - pred).powi(2);
                }
                wmse /= total_weight;
                if wmse < best.2 { best = (k, delta, wmse); }
            }
        }
        println!("Model 3 — σ(k×Δ×2000 / max(R_me×R_opp, δ))");
        println!("  k={:.1}, δ={:.0}, wMSE={:.6}", best.0, best.1, best.2);
    }

    // Model 4: σ(k × Δ / (R_sum^α + δ)) where R_sum = (2000-s_me) + (2000-s_opp)
    {
        let mut best = (0.0f64, 0.0f64, 0.0f64, f64::MAX);
        for k_10 in 10..80 {
            let k = k_10 as f64 / 10.0;
            for alpha_10 in 3..15 {
                let alpha = alpha_10 as f64 / 10.0;
                for d_i in 1..40 {
                    let delta = d_i as f64 * 10.0;
                    let mut wmse = 0.0;
                    for b in &bins {
                        let r_sum = (2000.0 - b.ns) + (2000.0 - b.ew);
                        let denom = r_sum.powf(alpha) + delta;
                        let pred = sigmoid(k * (b.ns - b.ew) / denom);
                        wmse += b.count as f64 * (b.wr - pred).powi(2);
                    }
                    wmse /= total_weight;
                    if wmse < best.3 { best = (k, alpha, delta, wmse); }
                }
            }
        }
        println!("Model 4 — σ(k×Δ / (R_sum^α + δ))");
        println!("  k={:.1}, α={:.1}, δ={:.0}, wMSE={:.6}", best.0, best.1, best.2, best.3);
    }

    // Model 5: σ(k × Δ / max(2000-max, δ))  [max-based: distance of leader to 2000]
    {
        let mut best = (0.0f64, 0.0f64, f64::MAX);
        for k_10 in 5..60 {
            let k = k_10 as f64 / 10.0;
            for d_i in 1..40 {
                let delta = d_i as f64 * 25.0;
                let mut wmse = 0.0;
                for b in &bins {
                    let denom = (2000.0 - b.ns.max(b.ew)).max(delta);
                    let pred = sigmoid(k * (b.ns - b.ew) / denom);
                    wmse += b.count as f64 * (b.wr - pred).powi(2);
                }
                wmse /= total_weight;
                if wmse < best.2 { best = (k, delta, wmse); }
            }
        }
        println!("Model 5 — σ(k×Δ / max(2000-max, δ))");
        println!("  k={:.1}, δ={:.0}, wMSE={:.6}", best.0, best.1, best.2);
    }
}
