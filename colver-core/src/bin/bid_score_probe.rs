/// Probe a score-aware bid model against a baseline on identical hands/auctions
/// across different match score states.
///
/// For each of N random deals, replays a random partial auction, then queries
/// both models' next bid at a fixed set of match score states to quantify how
/// the score-aware model changes its behavior based on match context.
///
/// Usage:
///   cargo run --bin bid_score_probe --release -- --deals 5000

use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{self, BID_OBS_DIM};
use colver_core::state::{GameState, Phase};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

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

const SCORE_STATES: &[(&str, i32, i32)] = &[
    ("early_even", 200, 200),
    ("mid_even",   1000, 1000),
    ("end_even",   1800, 1800),
    ("end_lead",   1800, 1200),
    ("end_trail",  1200, 1800),
    ("near_win",   1900, 1500),
    ("near_loss",  1500, 1900),
];

fn bid_value(action: u8) -> Option<i32> {
    match action {
        0 | 41 | 42 => None,
        37..=40 => Some(250),
        1..=36 => Some(80 + ((action as i32 - 1) / 4) * 10),
        _ => None,
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let deals = parse_flag_usize(&args, "--deals", 5000);
    let v4_path = parse_flag(&args, "--v4")
        .unwrap_or_else(|| "models/bid_v4_score_aware/bid_nn_16000000.bin".into());
    let v3_path = parse_flag(&args, "--v3")
        .unwrap_or_else(|| "models/bid_v3_max_20M/bid_nn_final.bin".into());
    let seed: u64 = parse_flag_usize(&args, "--seed", 42) as u64;

    eprintln!("Loading v4: {}", v4_path);
    let mut v4 = BidNet::load_with_hidden(&v4_path, 512).expect("load v4");
    eprintln!("  obs_dim={}, hidden={}, layers={}", v4.obs_dim(), v4.hidden(), v4.layers());
    assert!(v4.obs_dim() > BID_OBS_DIM, "v4 must be score-aware (110-dim)");

    eprintln!("Loading v3: {}", v3_path);
    let mut v3 = BidNet::load_with_hidden(&v3_path, 512).expect("load v3");
    eprintln!("  obs_dim={}, hidden={}", v3.obs_dim(), v3.hidden());
    assert_eq!(v3.obs_dim(), BID_OBS_DIM);

    eprintln!("\n=== Probing {} scenarios × {} score states ===\n", deals, SCORE_STATES.len());

    let mut rng = StdRng::seed_from_u64(seed);
    let mut obs_108 = vec![0.0f32; BID_OBS_DIM];
    let mut obs_110 = vec![0.0f32; v4.obs_dim()];

    let n_states = SCORE_STATES.len();
    let mut v4_passes = vec![0usize; n_states];
    let mut v4_bid_values = vec![Vec::<i32>::new(); n_states];
    let mut v4_coinches = vec![0usize; n_states];
    let mut v4_capots = vec![0usize; n_states];
    let mut v4_bid_counts = vec![0usize; n_states];
    let mut v3_passes = 0usize;
    let mut v3_bid_values: Vec<i32> = Vec::new();
    let mut v3_coinches = 0usize;
    let mut v3_capots = 0usize;
    let mut v3_bid_count = 0usize;

    let mut scenarios_with_variation = 0usize;
    let mut total_scenarios = 0usize;
    let mut v4_agrees_v3 = vec![0usize; n_states];
    let mut lead_trail_deltas: Vec<i32> = Vec::new();

    // Additional: track when leading → PASS but trailing → BID, and vice versa
    let mut pass_when_lead_bid_when_trail = 0usize;
    let mut bid_when_lead_pass_when_trail = 0usize;

    for _ in 0..deals {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut bid_history: Vec<(u8, u8)> = Vec::new();

        let prefix_len = rng.gen_range(0..7usize);
        for _ in 0..prefix_len {
            if state.phase != Phase::Bidding { break; }
            let legal = state.legal_actions();
            let candidates: Vec<u8> = (0..43u8)
                .filter(|&a| legal & (1u64 << a) != 0)
                .collect();
            if candidates.is_empty() { break; }
            let action = if candidates.contains(&0) && rng.gen_bool(0.5) {
                0
            } else {
                let low_bids: Vec<u8> = candidates.iter()
                    .filter(|&&a| a >= 1 && a <= 20).copied().collect();
                if !low_bids.is_empty() && rng.gen_bool(0.7) {
                    low_bids[rng.gen_range(0..low_bids.len())]
                } else {
                    candidates[rng.gen_range(0..candidates.len())]
                }
            };
            bid_history.push((state.current_player(), action));
            state.step(action);
        }

        if state.phase != Phase::Bidding { continue; }
        let legal = state.legal_actions();
        if legal == 0 { continue; }

        total_scenarios += 1;

        bid_obs::write_bid_observation(&mut obs_108, 0, &state, &bid_history);
        let (v3_action, _) = v3.best_action(&obs_108, legal);

        match v3_action {
            0 => v3_passes += 1,
            41 | 42 => v3_coinches += 1,
            37..=40 => { v3_capots += 1; v3_bid_count += 1; }
            _ => {
                if let Some(v) = bid_value(v3_action) {
                    v3_bid_values.push(v);
                    v3_bid_count += 1;
                }
            }
        }

        let me = state.current_player() as usize;
        let team_is_ns = me % 2 == 0;
        let mut v4_actions = vec![0u8; n_states];
        for (i, &(_label, ns, ew)) in SCORE_STATES.iter().enumerate() {
            let (my_score, opp_score) = if team_is_ns { (ns, ew) } else { (ew, ns) };
            bid_obs::write_bid_observation_score_aware(
                &mut obs_110, 0, &state, &bid_history, my_score, opp_score,
            );
            let (action, _) = v4.best_action(&obs_110, legal);
            v4_actions[i] = action;

            match action {
                0 => v4_passes[i] += 1,
                41 | 42 => v4_coinches[i] += 1,
                37..=40 => { v4_capots[i] += 1; v4_bid_counts[i] += 1; }
                _ => {
                    if let Some(v) = bid_value(action) {
                        v4_bid_values[i].push(v);
                        v4_bid_counts[i] += 1;
                    }
                }
            }

            if action == v3_action {
                v4_agrees_v3[i] += 1;
            }
        }

        let first = v4_actions[0];
        if v4_actions.iter().any(|&a| a != first) {
            scenarios_with_variation += 1;
        }

        // Lead (state 3 = end_lead 1800/1200) vs Trail (state 4 = end_trail 1200/1800)
        let lead_action = v4_actions[3];
        let trail_action = v4_actions[4];
        if let (Some(l), Some(t)) = (bid_value(lead_action), bid_value(trail_action)) {
            lead_trail_deltas.push(l - t);
        }
        if lead_action == 0 && trail_action != 0 && trail_action != 41 && trail_action != 42 {
            pass_when_lead_bid_when_trail += 1;
        }
        if trail_action == 0 && lead_action != 0 && lead_action != 41 && lead_action != 42 {
            bid_when_lead_pass_when_trail += 1;
        }
    }

    println!("\n=== Overall ===");
    println!("  Scenarios probed: {}", total_scenarios);
    println!("  v4 picks differently across score states: {} ({:.1}%)",
        scenarios_with_variation,
        scenarios_with_variation as f64 / total_scenarios as f64 * 100.0);

    println!("\n=== v3 baseline (no score context) ===");
    let v3_avg_bid = if !v3_bid_values.is_empty() {
        v3_bid_values.iter().sum::<i32>() as f64 / v3_bid_values.len() as f64
    } else { 0.0 };
    println!("  PASS: {} ({:.1}%)  BID: {} (avg {:.1})  CAPOT: {}  COINCHE: {}",
        v3_passes, v3_passes as f64 / total_scenarios as f64 * 100.0,
        v3_bid_count, v3_avg_bid, v3_capots, v3_coinches);

    println!("\n=== v4 score-aware per state ===");
    println!("  {:<18} {:>7} {:>7} {:>9} {:>6} {:>8} {:>8}",
        "state (me/opp)", "pass%", "bid%", "avg_bid", "capot", "coinche", "=v3%");
    println!("  {}", "-".repeat(68));
    for (i, (label, ns, ew)) in SCORE_STATES.iter().enumerate() {
        let pass_pct = v4_passes[i] as f64 / total_scenarios as f64 * 100.0;
        let bid_pct = v4_bid_counts[i] as f64 / total_scenarios as f64 * 100.0;
        let avg = if !v4_bid_values[i].is_empty() {
            v4_bid_values[i].iter().sum::<i32>() as f64 / v4_bid_values[i].len() as f64
        } else { 0.0 };
        let agree = v4_agrees_v3[i] as f64 / total_scenarios as f64 * 100.0;
        println!(
            "  {:<18} {:>6.1}% {:>6.1}% {:>9.1} {:>6} {:>8} {:>7.1}%",
            format!("{} ({}/{})", label, ns, ew),
            pass_pct, bid_pct, avg, v4_capots[i], v4_coinches[i], agree,
        );
    }

    if !lead_trail_deltas.is_empty() {
        let sum: i32 = lead_trail_deltas.iter().sum();
        let avg = sum as f64 / lead_trail_deltas.len() as f64;
        let positive = lead_trail_deltas.iter().filter(|&&d| d > 0).count();
        let negative = lead_trail_deltas.iter().filter(|&&d| d < 0).count();
        let zero = lead_trail_deltas.iter().filter(|&&d| d == 0).count();
        println!("\n=== Lead (1800/1200) vs Trail (1200/1800) — bid value diff ===");
        println!("  Both bid: {} scenarios", lead_trail_deltas.len());
        println!("  Mean Δ (lead − trail): {:+.2} pts (negative = more aggressive when behind)", avg);
        println!("  Lead bids HIGHER: {} ({:.1}%)",
            positive, positive as f64 / lead_trail_deltas.len() as f64 * 100.0);
        println!("  Lead bids LOWER:  {} ({:.1}%)",
            negative, negative as f64 / lead_trail_deltas.len() as f64 * 100.0);
        println!("  Equal:            {} ({:.1}%)",
            zero, zero as f64 / lead_trail_deltas.len() as f64 * 100.0);
    }

    println!("\n=== Pass/bid switches between lead and trail ===");
    println!("  PASS when leading, BID when trailing: {} ({:.2}%)  [plays safe when ahead]",
        pass_when_lead_bid_when_trail,
        pass_when_lead_bid_when_trail as f64 / total_scenarios as f64 * 100.0);
    println!("  BID when leading,  PASS when trailing: {} ({:.2}%)  [surprising]",
        bid_when_lead_pass_when_trail,
        bid_when_lead_pass_when_trail as f64 / total_scenarios as f64 * 100.0);

    println!("\n=== Conservatism: early_even (200/200) vs end_even (1800/1800) ===");
    let early_pass_pct = v4_passes[0] as f64 / total_scenarios as f64 * 100.0;
    let end_pass_pct = v4_passes[2] as f64 / total_scenarios as f64 * 100.0;
    let early_avg = if !v4_bid_values[0].is_empty() {
        v4_bid_values[0].iter().sum::<i32>() as f64 / v4_bid_values[0].len() as f64
    } else { 0.0 };
    let end_avg = if !v4_bid_values[2].is_empty() {
        v4_bid_values[2].iter().sum::<i32>() as f64 / v4_bid_values[2].len() as f64
    } else { 0.0 };
    println!("  PASS rate: {:.1}% → {:.1}% ({:+.1}pp)",
        early_pass_pct, end_pass_pct, end_pass_pct - early_pass_pct);
    println!("  Avg bid:   {:.1} → {:.1} ({:+.1} pts)",
        early_avg, end_avg, end_avg - early_avg);
    println!("  CAPOT:     {} → {}", v4_capots[0], v4_capots[2]);
    println!("  COINCHE:   {} → {}", v4_coinches[0], v4_coinches[2]);
}
