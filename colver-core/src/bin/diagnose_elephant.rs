/// Diagnose elephant memory: compare predictions vs ground truth at each decision point.
///
/// Plays N deals with IS-DD (beliefs + elephant). At each observer decision:
/// - base beliefs log-likelihood of ground truth
/// - elephant evidence log-likelihood
/// - blended log-likelihood
/// - particle survival stats
/// - disagreements: when elephant and base disagree, who's right?
///
/// Usage:
///   cargo run --bin diagnose_elephant --release -- [--deals 500] [--seed 42]

use colver_core::card::{CardIter, ALL_CARDS, EMPTY};
use colver_core::bid_eval::BidFunction;
use colver_core::elephant::blend_with_evidence;
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::rollout::heuristic_play_action;
use colver_core::state::{GameState, Phase};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[derive(Default, Clone)]
struct TrickStats {
    decisions: u32,
    hidden_cards: u32,
    // Log-prob sums (higher = better, max 0).
    lp_base: f64,
    lp_elephant: f64,
    lp_blended: f64,
    // False exclusion: prob < 0.05 for true owner.
    fe_base: u32,
    fe_elephant: u32,
    fe_blended: u32,
    // Disagreements (argmax differs between base and elephant for a card).
    disagree: u32,
    disagree_elephant_right: u32,
    disagree_base_right: u32,
    disagree_neither: u32, // both wrong
    // Particle stats.
    surviving_sum: u64,
    total_sum: u64,
    elephant_available: u32,
}

/// Compute log-prob and false exclusion rate of weights against ground truth.
fn eval_weights(
    weights: &[[f32; 32]; 4],
    unknown: u32,
    truth: &[u32; 4],
    observer: u8,
) -> (f64, u32, u32) {
    // Returns (log_prob_sum, hidden_count, false_exclusion_count)
    let mut lp = 0.0f64;
    let mut count = 0u32;
    let mut fe = 0u32;

    for card in CardIter(unknown) {
        let true_owner = (0..4u8)
            .filter(|&p| p != observer)
            .find(|&p| truth[p as usize] & (1 << card) != 0);
        let true_owner = match true_owner {
            Some(p) => p,
            None => continue,
        };

        let prob = weights[true_owner as usize][card as usize];
        lp += (prob.max(1e-6) as f64).ln();
        count += 1;
        if prob < 0.05 {
            fe += 1;
        }
    }
    (lp, count, fe)
}

/// For each unknown card, find which player has the highest weight.
fn argmax_per_card(weights: &[[f32; 32]; 4], unknown: u32, observer: u8) -> [u8; 32] {
    let mut result = [255u8; 32];
    for card in CardIter(unknown) {
        let mut best_p = 0u8;
        let mut best_w = -1.0f32;
        for p in 0..4u8 {
            if p == observer {
                continue;
            }
            if weights[p as usize][card as usize] > best_w {
                best_w = weights[p as usize][card as usize];
                best_p = p;
            }
        }
        result[card as usize] = best_p;
    }
    result
}

fn main() {
    let mut num_deals = 500u32;
    let mut seed = 42u64;
    let mut smoothing = 0.30f32;
    let mut decay = 0.8f32;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--deals" => { num_deals = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--smoothing" => { smoothing = args[i + 1].parse().unwrap(); i += 2; }
            "--decay" => { decay = args[i + 1].parse().unwrap(); i += 2; }
            _ => { i += 1; }
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);

    let config = IsDdConfig {
        determinizations: 20,
        time_limit_ms: None,
        use_soft_inference: true,
        bid_function: BidFunction::ImprovedV2,
        use_elephant_memory: true,
        elephant_smoothing: smoothing,
        elephant_dominance_penalty: 0.5,
        elephant_use_dominance: false,
        elephant_decay: decay,
        early_termination: true,
        ..Default::default()
    };

    let mut stats = vec![TrickStats::default(); 8];
    let mut total_deals = 0u32;

    eprintln!("Diagnosing elephant: {} deals, smoothing={}, decay={}", num_deals, smoothing, decay);

    for deal_idx in 0..num_deals {
        let dealer = (deal_idx % 4) as u8;
        let mut state = GameState::deal_random(dealer, &mut rng);
        let ground_truth = state.hands;
        let observer = 0u8;

        let mut search = IsDdSearch::new();
        search.init_deal_with_config(&state, observer, &config);

        // Bidding phase.
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;
            let action = config.bid_function.bid(&state);
            search.record_action(&state_before, player, action);
            state.step(action);
        }
        if state.is_terminal() { continue; }
        total_deals += 1;

        // Play phase.
        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;
            let trick_num = (state.tricks_won[0] + state.tricks_won[1]) as usize;
            let trick_idx = trick_num.min(7);

            // Analyze at observer's decision points (skip forced moves).
            if player == observer && state.legal_actions().count_ones() > 1 {
                let s = &mut stats[trick_idx];
                s.decisions += 1;

                // Compute unknown cards.
                let mut played = state.played_cards;
                for ci in 0..4 {
                    let c = state.current_trick[ci];
                    if c != EMPTY { played |= 1u32 << c; }
                }
                let known = state.hands[observer as usize] | played;
                let unknown = ALL_CARDS ^ known;

                // 1. Base beliefs (without elephant).
                if let Some(base) = search.base_belief_weights() {
                    let (lp_b, hidden, fe_b) = eval_weights(&base, unknown, &ground_truth, observer);
                    s.lp_base += lp_b;
                    s.hidden_cards += hidden;
                    s.fe_base += fe_b;

                    // 2. Elephant evidence.
                    let (surviving, total_p) = search.elephant_stats();
                    s.surviving_sum += surviving as u64;
                    s.total_sum += total_p as u64;

                    if let Some(evidence) = search.elephant_evidence(&state) {
                        s.elephant_available += 1;

                        let (lp_e, _, fe_e) = eval_weights(&evidence, unknown, &ground_truth, observer);
                        s.lp_elephant += lp_e;
                        s.fe_elephant += fe_e;

                        // 3. Blended weights.
                        let blended = blend_with_evidence(&base, &evidence, &state, observer, smoothing);
                        let (lp_bl, _, fe_bl) = eval_weights(&blended, unknown, &ground_truth, observer);
                        s.lp_blended += lp_bl;
                        s.fe_blended += fe_bl;

                        // 4. Disagreements.
                        let base_argmax = argmax_per_card(&base, unknown, observer);
                        let eleph_argmax = argmax_per_card(&evidence, unknown, observer);

                        for card in CardIter(unknown) {
                            let bp = base_argmax[card as usize];
                            let ep = eleph_argmax[card as usize];
                            if bp == ep { continue; }

                            s.disagree += 1;
                            let true_owner = (0..4u8)
                                .filter(|&p| p != observer)
                                .find(|&p| ground_truth[p as usize] & (1 << card) != 0);
                            if let Some(tp) = true_owner {
                                if ep == tp && bp != tp {
                                    s.disagree_elephant_right += 1;
                                } else if bp == tp && ep != tp {
                                    s.disagree_base_right += 1;
                                } else {
                                    s.disagree_neither += 1;
                                }
                            }
                        }
                    }
                }
            }

            // Play action: observer uses IS-DD, others use heuristic.
            let action = if player == observer {
                search.search(&state, &config, &mut rng)
            } else {
                heuristic_play_action(&state)
            };

            search.record_action(&state_before, player, action);
            state.step(action);
        }

        if (deal_idx + 1) % 100 == 0 {
            eprint!("\r  {}/{} deals...", deal_idx + 1, num_deals);
        }
    }
    eprintln!("\r  Done: {} deals\n", total_deals);

    // === Print results ===
    println!("=== ELEPHANT MEMORY DIAGNOSTIC (smoothing={}, decay={}) ===\n", smoothing, decay);

    // Table 1: Log-prob per hidden card (higher = better).
    println!("--- Log-prob per hidden card (higher = better, max 0) ---");
    println!("{:>6}  {:>5}  {:>5}  {:>9}  {:>9}  {:>9}  {:>7}",
        "Trick", "Decs", "Avail", "Base", "Elephant", "Blended", "Delta");
    println!("{}", "-".repeat(65));

    let mut tot = TrickStats::default();
    for trick in 0..8 {
        let s = &stats[trick];
        if s.decisions == 0 { continue; }
        let h = s.hidden_cards.max(1) as f64;
        let ea = s.elephant_available.max(1) as f64;
        // For elephant and blended, normalize by elephant-available hidden cards.
        // Approximate: use same hidden_cards count (close enough).
        let lp_b = s.lp_base / h;
        let lp_e = s.lp_elephant / h;
        let lp_bl = s.lp_blended / h;
        let delta = lp_bl - lp_b;
        println!("{:>6}  {:>5}  {:>5}  {:>9.4}  {:>9.4}  {:>9.4}  {:>+7.4}",
            trick, s.decisions, s.elephant_available, lp_b, lp_e, lp_bl, delta);

        // Accumulate totals.
        tot.decisions += s.decisions;
        tot.hidden_cards += s.hidden_cards;
        tot.lp_base += s.lp_base;
        tot.lp_elephant += s.lp_elephant;
        tot.lp_blended += s.lp_blended;
        tot.fe_base += s.fe_base;
        tot.fe_elephant += s.fe_elephant;
        tot.fe_blended += s.fe_blended;
        tot.disagree += s.disagree;
        tot.disagree_elephant_right += s.disagree_elephant_right;
        tot.disagree_base_right += s.disagree_base_right;
        tot.disagree_neither += s.disagree_neither;
        tot.elephant_available += s.elephant_available;
        tot.surviving_sum += s.surviving_sum;
        tot.total_sum += s.total_sum;
    }
    let h = tot.hidden_cards.max(1) as f64;
    println!("{}", "-".repeat(65));
    println!("{:>6}  {:>5}  {:>5}  {:>9.4}  {:>9.4}  {:>9.4}  {:>+7.4}",
        "ALL", tot.decisions, tot.elephant_available,
        tot.lp_base / h, tot.lp_elephant / h, tot.lp_blended / h,
        (tot.lp_blended - tot.lp_base) / h);

    // Table 2: False exclusion rate (prob < 5% for true owner).
    println!("\n--- False exclusion rate (prob < 5%% for true owner, lower = better) ---");
    println!("{:>6}  {:>9}  {:>9}  {:>9}", "Trick", "Base", "Elephant", "Blended");
    println!("{}", "-".repeat(45));
    for trick in 0..8 {
        let s = &stats[trick];
        if s.elephant_available == 0 { continue; }
        let h = s.hidden_cards.max(1) as f64;
        println!("{:>6}  {:>8.1}%  {:>8.1}%  {:>8.1}%",
            trick,
            s.fe_base as f64 / h * 100.0,
            s.fe_elephant as f64 / h * 100.0,
            s.fe_blended as f64 / h * 100.0);
    }
    println!("{}", "-".repeat(45));
    println!("{:>6}  {:>8.1}%  {:>8.1}%  {:>8.1}%",
        "ALL",
        tot.fe_base as f64 / h * 100.0,
        tot.fe_elephant as f64 / h * 100.0,
        tot.fe_blended as f64 / h * 100.0);

    // Table 3: Disagreements.
    println!("\n--- Disagreements (base vs elephant argmax differ) ---");
    println!("Total disagreements: {}", tot.disagree);
    if tot.disagree > 0 {
        let d = tot.disagree as f64;
        println!("  Elephant right: {:>5} ({:.1}%)", tot.disagree_elephant_right, tot.disagree_elephant_right as f64 / d * 100.0);
        println!("  Base right:     {:>5} ({:.1}%)", tot.disagree_base_right, tot.disagree_base_right as f64 / d * 100.0);
        println!("  Neither right:  {:>5} ({:.1}%)", tot.disagree_neither, tot.disagree_neither as f64 / d * 100.0);
    }

    // Table 4: Particle survival.
    println!("\n--- Particle survival by trick ---");
    println!("{:>6}  {:>8}  {:>8}  {:>8}", "Trick", "Survive", "Total", "Rate%");
    println!("{}", "-".repeat(40));
    for trick in 0..8 {
        let s = &stats[trick];
        if s.decisions == 0 { continue; }
        let avg_s = s.surviving_sum as f64 / s.decisions as f64;
        let avg_t = s.total_sum as f64 / s.decisions as f64;
        let rate = if avg_t > 0.0 { avg_s / avg_t * 100.0 } else { 0.0 };
        println!("{:>6}  {:>8.1}  {:>8.1}  {:>7.1}%", trick, avg_s, avg_t, rate);
    }
}
