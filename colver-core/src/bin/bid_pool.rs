/// Run the NN bidder (Bid a Dede, v2) on each deal of a pool and record the
/// resulting contract. Used to filter our IS-DD play dataset to "realistic"
/// (deal, suit) pairs — the contracts that the bidder actually would have
/// taken in real play.
///
/// Output CSV: deal_id, dealer, declarer_seat, trump_suit, value, coinche, passed
///   passed=1 → all 4 players passed (no contract)
///   trump_suit ∈ {0,1,2,3} (S/H/D/C); declarer_seat ∈ {0,1,2,3}
///   value: 8..16 (= 80..160 / 10), or 25 for capot
///
/// Usage:
///   cargo run --bin bid_pool --release --features parallel -- [options]
///
/// Options:
///   --pool PATH       Pool input (default: data/deals/base_5M.bin)
///   --model PATH      Bid NN weights (default: models/bid_v2/bid_nn_final.bin)
///   --hidden N        Hidden size (default: 512 for v2)
///   --output PATH     CSV output (default: data/distill/bid_pool_200k.csv)
///   --deals N         Number of deals (default: 200000)
///   --offset N        Skip first N deals (default: 0)

use std::fs::File;
use std::io::{BufWriter, Write};
use std::sync::atomic::{AtomicUsize, Ordering};
use std::time::Instant;

use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bid_train_env::DealPool;
use colver_core::state::{GameState, Phase};

fn main() {
    let args: Vec<String> = std::env::args().collect();

    let mut pool_path = String::from("data/deals/base_5M.bin");
    let mut model_path = String::from("models/bid_v5_isdd/bid_nn_final.bin");
    let mut output_path = String::from("data/distill/bid_pool_200k.csv");
    let mut hidden: usize = 512;
    let mut num_deals: usize = 200_000;
    let mut offset: usize = 0;
    let mut my_score: i32 = 0;
    let mut opp_score: i32 = 0;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--pool" => { i += 1; pool_path = args[i].clone(); }
            "--model" => { i += 1; model_path = args[i].clone(); }
            "--output" => { i += 1; output_path = args[i].clone(); }
            "--hidden" => { i += 1; hidden = args[i].parse().unwrap(); }
            "--deals" => { i += 1; num_deals = args[i].parse().unwrap(); }
            "--offset" => { i += 1; offset = args[i].parse().unwrap(); }
            "--my-score" => { i += 1; my_score = args[i].parse().unwrap(); }
            "--opp-score" => { i += 1; opp_score = args[i].parse().unwrap(); }
            _ => { eprintln!("Unknown arg: {}", args[i]); std::process::exit(1); }
        }
        i += 1;
    }

    eprintln!("Loading pool {}...", pool_path);
    let pool = DealPool::load(&pool_path).expect("load pool");
    eprintln!("  pool size {}", pool.len());

    let end = (offset + num_deals).min(pool.len());
    let actual = end - offset;
    eprintln!("  taking deals [{}, {}) ({})", offset, end, actual);

    if let Some(parent) = std::path::Path::new(&output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let total = actual;
    let progress = AtomicUsize::new(0);
    let start = Instant::now();

    // Load deals first so we can parallelize
    let deals: Vec<(u8, [u32; 4])> = (offset..end)
        .map(|idx| {
            let d = pool.get(idx);
            (d.dealer, d.hands)
        })
        .collect();

    use rayon::prelude::*;

    // Sniff obs_dim once
    let probe = BidNet::load_with_hidden(&model_path, hidden).expect("load bid net");
    let obs_dim = probe.obs_dim();
    drop(probe);
    eprintln!("  bid net obs_dim: {} (v1=108, v1+score=110, v2+score=113, v3+score=117)", obs_dim);

    let results: Vec<String> = deals
        .par_iter()
        .enumerate()
        .map(|(local_idx, &(dealer, hands))| {
            // Each thread gets its own BidNet (load is cheap-ish; could share via thread_local)
            let mut net = BidNet::load_with_hidden(&model_path, hidden)
                .expect("load bid net");

            let mut state = GameState::new(dealer, hands);
            let mut history: Vec<(u8, u8)> = Vec::with_capacity(8);

            while state.phase == Phase::Bidding {
                let _player = state.current_player();
                let obs: Vec<f32> = match obs_dim {
                    bid_obs::BID_OBS_DIM => bid_obs::make_bid_observation(&state, &history),
                    bid_obs::BID_OBS_DIM_SCORE_AWARE => {
                        let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE];
                        bid_obs::write_bid_observation_score_aware(
                            &mut buf, 0, &state, &history, my_score, opp_score,
                        );
                        buf
                    }
                    bid_obs::BID_OBS_DIM_SCORE_AWARE_V2 => {
                        let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE_V2];
                        bid_obs::write_bid_observation_score_aware_v2(
                            &mut buf, 0, &state, &history, my_score, opp_score,
                        );
                        buf
                    }
                    other => panic!("unsupported bid obs_dim {}", other),
                };
                let legal = state.legal_actions();
                let (action, _) = net.best_action(&obs, legal);
                history.push((state.current_player(), action));
                state.step(action);
            }

            let passed = state.phase == Phase::Done;
            let (suit, value, declarer_seat, coinche) = if passed {
                (-1i32, 0i32, -1i32, 0i32)
            } else {
                (
                    state.contract.trump as i32,
                    state.contract.value as i32,
                    state.last_bidder as i32,
                    state.contract.coinche as i32,
                )
            };

            let done = progress.fetch_add(1, Ordering::Relaxed) + 1;
            if done % 5000 == 0 || done == total {
                let elapsed = start.elapsed().as_secs_f64();
                let rate = done as f64 / elapsed;
                eprintln!("  {}/{} ({:.0}/s)", done, total, rate);
            }

            let deal_id = offset + local_idx;
            let passed_flag = if passed { 1 } else { 0 };
            format!(
                "{},{},{},{},{},{},{}",
                deal_id, dealer, declarer_seat, suit, value, coinche, passed_flag
            )
        })
        .collect();

    let mut f = BufWriter::new(File::create(&output_path).expect("create output"));
    writeln!(f, "deal_id,dealer,declarer_seat,trump_suit,value,coinche,passed").unwrap();
    for line in &results {
        writeln!(f, "{}", line).unwrap();
    }
    f.flush().unwrap();

    eprintln!("\nWrote {} rows to {}", results.len(), output_path);

    // Quick summary
    let mut n_passed = 0;
    let mut suit_counts = [0u64; 4];
    let mut value_counts = std::collections::BTreeMap::new();
    let mut team_counts = [0u64; 2];
    for line in &results {
        let cols: Vec<&str> = line.split(',').collect();
        let suit: i32 = cols[3].parse().unwrap();
        let value: i32 = cols[4].parse().unwrap();
        let dec: i32 = cols[2].parse().unwrap();
        let p: i32 = cols[6].parse().unwrap();
        if p == 1 {
            n_passed += 1;
        } else {
            suit_counts[suit as usize] += 1;
            *value_counts.entry(value).or_insert(0u64) += 1;
            team_counts[(dec as usize) % 2] += 1;
        }
    }
    let n_taken = (results.len() - n_passed) as u64;
    eprintln!("\n--- Summary ---");
    eprintln!("  passed (no contract): {} ({:.1}%)", n_passed, n_passed as f64 / results.len() as f64 * 100.0);
    eprintln!("  taken: {}", n_taken);
    eprintln!("  trump suit (S/H/D/C):  {:?}", suit_counts);
    eprintln!("  declarer team (NS/EW): {:?}", team_counts);
    eprintln!("  contract values:");
    for (v, c) in &value_counts {
        eprintln!("    {}0: {} ({:.1}%)", v, c, *c as f64 / n_taken as f64 * 100.0);
    }
}
