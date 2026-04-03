/// DD oracle validation of bidding hypotheses.
///
/// Same experiments as bid_nn_probe, but using double-dummy solver as ground truth.
/// For each hand: shuffle remaining cards N times, DD-solve all 4 suits, report
/// success rates at each contract level.
///
/// The Monte Carlo compares NN decisions vs DD ground truth to measure calibration.
///
/// Usage:
///   cargo run -p colver-core --bin bid_dd_probe --release -- [model_path] [--sims N] [--mc N]

use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bidding;
use colver_core::card::*;
use colver_core::solver;
use colver_core::state::GameState;
use rand::prelude::*;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

const SYM: [&str; 4] = ["♠", "♥", "♦", "♣"];
const RNK: [&str; 8] = ["7", "8", "9", "J", "Q", "K", "10", "A"];
const THRESHOLDS: [u8; 7] = [80, 90, 100, 110, 120, 130, 160];

const R7: u8 = 0;
const R8: u8 = 1;
const R9: u8 = 2;
const RJ: u8 = 3;
const RQ: u8 = 4;
const RK: u8 = 5;
const R10: u8 = 6;
const RA: u8 = 7;

const S: u8 = 0;
const H: u8 = 1;
const D: u8 = 2;
const C: u8 = 3;

fn c(suit: u8, rank: u8) -> u8 {
    suit * 8 + rank
}
fn hand_of(cards: &[u8]) -> u32 {
    assert_eq!(cards.len(), 8);
    cards.iter().fold(0u32, |h, &c| h | (1u32 << c))
}

fn pretty(hand: u32) -> String {
    let mut parts = Vec::new();
    for s in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(s));
        if bits == 0 {
            continue;
        }
        let mut ranks = Vec::new();
        for r in (0..8).rev() {
            if bits & (1 << r) != 0 {
                ranks.push(RNK[r]);
            }
        }
        parts.push(format!("{}{}", SYM[s as usize], ranks.join("")));
    }
    parts.join(" ")
}

fn act_str(action: u8) -> String {
    match action {
        0 => "PASS".into(),
        41 => "COINCHE".into(),
        42 => "SURCOINCHE".into(),
        1..=40 => {
            let (val, suit) = bidding::decode_bid(action);
            if val == 25 {
                format!("Capot{}", SYM[suit as usize])
            } else {
                format!("{}{}", val as u16 * 10, SYM[suit as usize])
            }
        }
        _ => format!("?{}", action),
    }
}

// =====================================================================
//  DD simulation engine
// =====================================================================

struct DDResult {
    /// pts[suit] = list of NS points across simulations
    pts: [Vec<u8>; 4],
}

impl DDResult {
    fn new() -> Self {
        DDResult {
            pts: [vec![], vec![], vec![], vec![]],
        }
    }

    fn avg(&self, suit: u8) -> f64 {
        let p = &self.pts[suit as usize];
        if p.is_empty() {
            return 0.0;
        }
        p.iter().map(|&x| x as f64).sum::<f64>() / p.len() as f64
    }

    fn pct_ge(&self, suit: u8, threshold: u8) -> f64 {
        let p = &self.pts[suit as usize];
        if p.is_empty() {
            return 0.0;
        }
        p.iter().filter(|&&x| x >= threshold).count() as f64 / p.len() as f64 * 100.0
    }
}

/// Run DD simulation for a fixed hand (seat 0) with N random opponent distributions.
fn dd_simulate(hand: u32, n_sims: usize, rng: &mut impl Rng) -> DDResult {
    let mut tt_buf = solver::new_tt_buffer();
    let mut result = DDResult::new();
    let remaining: Vec<u8> = (0..32).filter(|&i| hand & (1 << i) == 0).collect();

    for _ in 0..n_sims {
        let mut rem = remaining.clone();
        rem.shuffle(rng);

        let mut hands = [0u32; 4];
        hands[0] = hand; // seat 0 = North (NS team)
        let mut idx = 0;
        for p in [1u8, 2, 3] {
            for _ in 0..8 {
                hands[p as usize] |= 1u32 << rem[idx];
                idx += 1;
            }
        }

        for suit in 0..4u8 {
            let [ns_pts, _] =
                solver::solve_for_trump_reuse_tt(hands, 3, suit, &mut tt_buf);
            result.pts[suit as usize].push(ns_pts);
        }
    }
    result
}

/// Print DD result table for a hand.
fn print_dd_table(dd: &DDResult) {
    print!("  {:>6} {:>7}", "Suit", "AvgPts");
    for &t in &THRESHOLDS {
        print!("  {:>5}", format!("≥{}", t));
    }
    println!();
    println!("  {}", "-".repeat(62));

    for s in 0..4u8 {
        let avg = dd.avg(s);
        let color = if dd.pct_ge(s, 80) > 60.0 {
            "\x1b[32m"
        } else if dd.pct_ge(s, 80) > 30.0 {
            "\x1b[33m"
        } else {
            "\x1b[0m"
        };
        print!("  {:>6} {}{:>7.1}\x1b[0m", SYM[s as usize], color, avg);
        for &t in &THRESHOLDS {
            let pct = dd.pct_ge(s, t);
            let c = if pct > 70.0 {
                "\x1b[32m"
            } else if pct > 40.0 {
                "\x1b[33m"
            } else {
                "\x1b[0m"
            };
            print!("  {}{:>4.0}%\x1b[0m", c, pct);
        }
        println!();
    }
}

fn header(title: &str) {
    println!("\n{}", "=".repeat(90));
    println!("  {}", title);
    println!("{}\n", "=".repeat(90));
}

/// Show DD table + NN decision for a hand.
fn show_hand(
    net: &mut BidNet,
    hand: u32,
    n_sims: usize,
    rng: &mut impl Rng,
    label: &str,
) {
    println!("  {} — {}\n", pretty(hand), label);

    // NN query (position 1, no prior bids)
    let hands = {
        let rem: Vec<u8> = (0..32).filter(|&i| hand & (1 << i) == 0).collect();
        let mut h = [0u32; 4];
        h[0] = hand;
        let mut idx = 0;
        for p in [1u8, 2, 3] {
            for _ in 0..8 {
                h[p as usize] |= 1u32 << rem[idx];
                idx += 1;
            }
        }
        h
    };
    let state = GameState::new(3, hands); // dealer=3 → seat 0 is pos 1
    let obs = bid_obs::make_bid_observation(&state, &[]);
    let legal = state.legal_actions();
    let (best, qvals) = net.best_action(&obs, legal);
    let best_q = qvals
        .iter()
        .find(|(a, _)| *a == best)
        .map(|(_, q)| *q)
        .unwrap_or(0.0);
    println!("  NN decision: {} (Q={:+.3})\n", act_str(best), best_q);

    // DD simulation
    let dd = dd_simulate(hand, n_sims, rng);
    print_dd_table(&dd);
    println!();
}

// =====================================================================
//  Controlled Experiments
// =====================================================================

fn exp_jack(net: &mut BidNet, n_sims: usize, rng: &mut impl Rng) {
    header("EXPERIMENT 1: The Trump Jack (DD ground truth)");
    println!(
        "  Does J really make the difference? DD oracle over {} random deals.\n",
        n_sims
    );

    let side = [c(H, RK), c(H, RQ), c(D, R8), c(D, R7)];

    let cases: [(&str, [u8; 4]); 6] = [
        (
            "J 9 A 10 (monster)",
            [c(S, RJ), c(S, R9), c(S, RA), c(S, R10)],
        ),
        (
            "J 9 A 8 (strong)",
            [c(S, RJ), c(S, R9), c(S, RA), c(S, R8)],
        ),
        (
            "J A 10 8 (J no 9)",
            [c(S, RJ), c(S, RA), c(S, R10), c(S, R8)],
        ),
        (
            "9 A 10 8 (9 no J)",
            [c(S, R9), c(S, RA), c(S, R10), c(S, R8)],
        ),
        (
            "K Q A 10 (no J/9)",
            [c(S, RK), c(S, RQ), c(S, RA), c(S, R10)],
        ),
        (
            "7 8 Q K (garbage)",
            [c(S, R7), c(S, R8), c(S, RQ), c(S, RK)],
        ),
    ];

    for (label, trump) in &cases {
        let mut cards = trump.to_vec();
        cards.extend_from_slice(&side);
        show_hand(net, hand_of(&cards), n_sims, rng, label);
    }
}

fn exp_aux_as(net: &mut BidNet, n_sims: usize, rng: &mut impl Rng) {
    header("EXPERIMENT 2: \"Annoncer aux as\" (DD ground truth)");
    println!("  Can you actually make contracts with aces alone?\n");

    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "4 aces + all 7s",
            vec![
                c(S, RA),
                c(H, RA),
                c(D, RA),
                c(C, RA),
                c(S, R7),
                c(H, R7),
                c(D, R7),
                c(C, R7),
            ],
        ),
        (
            "4 aces + 10♠ K♠ 7♥ 7♦",
            vec![
                c(S, RA),
                c(H, RA),
                c(D, RA),
                c(C, RA),
                c(S, R10),
                c(S, RK),
                c(H, R7),
                c(D, R7),
            ],
        ),
        (
            "A10KQ♠ + A♥ A♦ 7♥ 7♦",
            vec![
                c(S, RA),
                c(S, R10),
                c(S, RK),
                c(S, RQ),
                c(H, RA),
                c(D, RA),
                c(H, R7),
                c(D, R7),
            ],
        ),
        (
            "A10KQ87♠ (6 trump, no J/9) + A♥ 7♦",
            vec![
                c(S, RA),
                c(S, R10),
                c(S, RK),
                c(S, RQ),
                c(S, R8),
                c(S, R7),
                c(H, RA),
                c(D, R7),
            ],
        ),
        (
            "A10KQ987♠ (7 trump, no J) + 7♥",
            vec![
                c(S, RA),
                c(S, R10),
                c(S, RK),
                c(S, RQ),
                c(S, R9),
                c(S, R8),
                c(S, R7),
                c(H, R7),
            ],
        ),
    ];

    for (label, cards) in &cases {
        show_hand(net, hand_of(cards), n_sims, rng, label);
    }
}

fn exp_trump_length(net: &mut BidNet, n_sims: usize, rng: &mut impl Rng) {
    header("EXPERIMENT 3: Trump Length (DD ground truth)");
    println!("  How many trump cards to reliably make contracts?\n");

    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "J♠ A♠ (2 trump) + AKQ10♥ 8♦ 7♦",
            vec![
                c(S, RJ),
                c(S, RA),
                c(H, RA),
                c(H, RK),
                c(H, RQ),
                c(H, R10),
                c(D, R8),
                c(D, R7),
            ],
        ),
        (
            "J♠ A10♠ (3 trump) + AKQ♥ 8♦ 7♦",
            vec![
                c(S, RJ),
                c(S, RA),
                c(S, R10),
                c(H, RA),
                c(H, RK),
                c(H, RQ),
                c(D, R8),
                c(D, R7),
            ],
        ),
        (
            "J9A10♠ (4 trump) + AK♥ 8♦ 7♦",
            vec![
                c(S, RJ),
                c(S, R9),
                c(S, RA),
                c(S, R10),
                c(H, RA),
                c(H, RK),
                c(D, R8),
                c(D, R7),
            ],
        ),
        (
            "J9A10K♠ (5 trump) + A♥ 8♦ 7♦",
            vec![
                c(S, RJ),
                c(S, R9),
                c(S, RA),
                c(S, R10),
                c(S, RK),
                c(H, RA),
                c(D, R8),
                c(D, R7),
            ],
        ),
        (
            "J9A10KQ♠ (6 trump) + A♥ 8♦",
            vec![
                c(S, RJ),
                c(S, R9),
                c(S, RA),
                c(S, R10),
                c(S, RK),
                c(S, RQ),
                c(H, RA),
                c(D, R8),
            ],
        ),
    ];

    for (label, cards) in &cases {
        show_hand(net, hand_of(cards), n_sims, rng, label);
    }
}

fn exp_side_strength(net: &mut BidNet, n_sims: usize, rng: &mut impl Rng) {
    header("EXPERIMENT 4: Side Strength (DD ground truth)");
    println!("  Same J9A♠ trump, vary side cards. Do aces protect against chute?\n");

    let trump = [c(S, RJ), c(S, R9), c(S, RA)];

    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "+ A♥ A♦ A♣ 10♥ 7♦ (3 side aces)",
            vec![c(H, RA), c(D, RA), c(C, RA), c(H, R10), c(D, R7)],
        ),
        (
            "+ A♥ A♦ K♣ Q♣ 7♦ (2 side aces)",
            vec![c(H, RA), c(D, RA), c(C, RK), c(C, RQ), c(D, R7)],
        ),
        (
            "+ A♥ K♦ Q♦ 8♣ 7♣ (1 side ace)",
            vec![c(H, RA), c(D, RK), c(D, RQ), c(C, R8), c(C, R7)],
        ),
        (
            "+ K♥ Q♥ K♦ Q♦ 7♣ (0 aces, KQ×2)",
            vec![c(H, RK), c(H, RQ), c(D, RK), c(D, RQ), c(C, R7)],
        ),
        (
            "+ 7♥ 8♥ 7♦ 8♦ 7♣ (0 aces, garbage)",
            vec![c(H, R7), c(H, R8), c(D, R7), c(D, R8), c(C, R7)],
        ),
    ];

    for (label, side) in &cases {
        let mut cards: Vec<u8> = trump.to_vec();
        cards.extend_from_slice(side);
        show_hand(net, hand_of(&cards), n_sims, rng, label);
    }
}

fn exp_belote(net: &mut BidNet, n_sims: usize, rng: &mut impl Rng) {
    header("EXPERIMENT 5: Belote (K+Q of trump) — DD ground truth");
    println!("  Does the +20 belote bonus show up in DD success rates?\n");

    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "J9♠ + K♠Q♠ (belote!) + A♥ 8♦7♦7♣",
            vec![
                c(S, RJ),
                c(S, R9),
                c(S, RK),
                c(S, RQ),
                c(H, RA),
                c(D, R8),
                c(D, R7),
                c(C, R7),
            ],
        ),
        (
            "J9♠ + K♠8♠ (no belote) + A♥ 8♦7♦7♣",
            vec![
                c(S, RJ),
                c(S, R9),
                c(S, RK),
                c(S, R8),
                c(H, RA),
                c(D, R8),
                c(D, R7),
                c(C, R7),
            ],
        ),
        (
            "J9♠ + Q♠8♠ (no belote) + A♥ 8♦7♦7♣",
            vec![
                c(S, RJ),
                c(S, R9),
                c(S, RQ),
                c(S, R8),
                c(H, RA),
                c(D, R8),
                c(D, R7),
                c(C, R7),
            ],
        ),
        (
            "J9♠ + 8♠7♠ (no belote) + A♥ 8♦7♦7♣",
            vec![
                c(S, RJ),
                c(S, R9),
                c(S, R8),
                c(S, R7),
                c(H, RA),
                c(D, R8),
                c(D, R7),
                c(C, R7),
            ],
        ),
    ];

    for (label, cards) in &cases {
        show_hand(net, hand_of(cards), n_sims, rng, label);
    }
}

// =====================================================================
//  Monte Carlo: NN vs DD comparison
// =====================================================================

fn exp_monte_carlo(net: &mut BidNet, n_deals: usize) {
    header(&format!(
        "MONTE CARLO: NN vs DD comparison ({} deals)",
        n_deals
    ));
    println!("  For each random deal: DD-solve all 4 suits + query NN at position 1.");
    println!("  Compare: does NN bid when DD says contract is makeable?\n");

    let mut rng = StdRng::seed_from_u64(42);
    let mut tt_buf = solver::new_tt_buffer();
    let start = Instant::now();

    // Per-suit stats: [has_j][has_9][count]
    #[derive(Clone)]
    struct Bucket {
        total: u32,
        dd_ge80: u32,
        dd_ge100: u32,
        nn_bids: u32,
        nn_bids_and_dd_ge80: u32,
        sum_dd_pts: u64,
    }
    impl Bucket {
        fn new() -> Self {
            Bucket {
                total: 0,
                dd_ge80: 0,
                dd_ge100: 0,
                nn_bids: 0,
                nn_bids_and_dd_ge80: 0,
                sum_dd_pts: 0,
            }
        }
    }

    let mut stats = vec![vec![vec![Bucket::new(); 9]; 2]; 2]; // [j][9][count]

    // Best-suit stats
    let mut best = vec![vec![vec![Bucket::new(); 9]; 2]; 2];

    // "Aux as" stats
    let mut no_honors_total = 0u32;
    let mut no_honors_dd_ge80 = 0u32; // any suit makes 80
    let mut no_honors_nn_bids = 0u32;

    // Overall NN calibration
    let mut nn_bid_total = 0u32;
    let mut nn_bid_dd_ge80 = 0u32; // NN bid this suit AND DD says ≥80
    let mut nn_bid_dd_ge_contract = 0u32; // NN bid AND DD ≥ actual contract level
    let mut nn_pass_total = 0u32;
    let mut nn_pass_best_dd_ge80 = 0u32; // NN passed but best suit DD ≥80

    for deal_idx in 0..n_deals {
        let state = GameState::deal_random(3, &mut rng);
        let hand = state.hands[0];

        // DD solve all 4 suits
        let mut dd_pts = [0u8; 4];
        for suit in 0..4u8 {
            let [ns, _] = solver::solve_for_trump_reuse_tt(state.hands, 3, suit, &mut tt_buf);
            dd_pts[suit as usize] = ns;
        }

        // NN query
        let obs = bid_obs::make_bid_observation(&state, &[]);
        let legal = state.legal_actions();
        let (nn_action, _qvals) = net.best_action(&obs, legal);
        let nn_bids = nn_action >= 1 && nn_action <= 40;
        let nn_suit = if nn_bids {
            let (val, s) = bidding::decode_bid(nn_action);
            let level = if val == 25 { 250u16 } else { val as u16 * 10 };
            Some((s, level))
        } else {
            None
        };

        // Overall calibration
        if let Some((s, level)) = nn_suit {
            nn_bid_total += 1;
            if dd_pts[s as usize] >= 80 {
                nn_bid_dd_ge80 += 1;
            }
            if dd_pts[s as usize] as u16 >= level {
                nn_bid_dd_ge_contract += 1;
            }
        } else {
            nn_pass_total += 1;
            let best_dd = dd_pts.iter().max().copied().unwrap_or(0);
            if best_dd >= 80 {
                nn_pass_best_dd_ge80 += 1;
            }
        }

        // Find best DD suit
        let mut best_dd_suit = 0u8;
        let mut best_dd_pts = 0u8;
        for s in 0..4u8 {
            if dd_pts[s as usize] > best_dd_pts {
                best_dd_pts = dd_pts[s as usize];
                best_dd_suit = s;
            }
        }

        // Best-suit features
        {
            let bits = suit_bits(hand, Suit::from_u8(best_dd_suit));
            let count = bits.count_ones() as usize;
            let has_j = (bits >> RJ) & 1 == 1;
            let has_9 = (bits >> R9) & 1 == 1;
            let b = &mut best[has_j as usize][has_9 as usize][count.min(8)];
            b.total += 1;
            if best_dd_pts >= 80 {
                b.dd_ge80 += 1;
            }
            if best_dd_pts >= 100 {
                b.dd_ge100 += 1;
            }
            if nn_bids {
                b.nn_bids += 1;
            }
            if nn_bids && best_dd_pts >= 80 {
                b.nn_bids_and_dd_ge80 += 1;
            }
            b.sum_dd_pts += best_dd_pts as u64;
        }

        // Per-suit stats
        for s in 0..4u8 {
            let bits = suit_bits(hand, Suit::from_u8(s));
            let count = bits.count_ones() as usize;
            let has_j = (bits >> RJ) & 1 == 1;
            let has_9 = (bits >> R9) & 1 == 1;

            let b = &mut stats[has_j as usize][has_9 as usize][count.min(8)];
            b.total += 1;
            if dd_pts[s as usize] >= 80 {
                b.dd_ge80 += 1;
            }
            if dd_pts[s as usize] >= 100 {
                b.dd_ge100 += 1;
            }
            if nn_suit.map(|(ns, _)| ns) == Some(s) {
                b.nn_bids += 1;
                if dd_pts[s as usize] >= 80 {
                    b.nn_bids_and_dd_ge80 += 1;
                }
            }
            b.sum_dd_pts += dd_pts[s as usize] as u64;
        }

        // "Aux as" check
        let any_j = (0..4).any(|s| hand & (1 << (s * 8 + RJ)) != 0);
        let any_9 = (0..4).any(|s| hand & (1 << (s * 8 + R9)) != 0);
        if !any_j && !any_9 {
            no_honors_total += 1;
            if dd_pts.iter().any(|&p| p >= 80) {
                no_honors_dd_ge80 += 1;
            }
            if nn_bids {
                no_honors_nn_bids += 1;
            }
        }

        // Progress
        if (deal_idx + 1) % 2000 == 0 {
            let elapsed = start.elapsed().as_secs_f64();
            let rate = (deal_idx + 1) as f64 / elapsed;
            println!(
                "  [{}/{}] {:.0} deals/sec, {:.0}s elapsed",
                deal_idx + 1,
                n_deals,
                rate,
                elapsed
            );
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!(
        "\n  Completed in {:.1}s ({:.0} deals/sec)\n",
        elapsed,
        n_deals as f64 / elapsed
    );

    // === Overall NN Calibration ===
    println!("  ┌───────────────────────────────────────────────┐");
    println!("  │  Overall NN Calibration                       │");
    println!("  └───────────────────────────────────────────────┘");
    println!(
        "  NN bids:  {} / {} ({:.1}%)",
        nn_bid_total,
        n_deals,
        nn_bid_total as f64 / n_deals as f64 * 100.0
    );
    println!(
        "    → DD confirms ≥80 in bid suit:         {:>5} / {:>5} ({:.1}%)",
        nn_bid_dd_ge80,
        nn_bid_total,
        nn_bid_dd_ge80 as f64 / nn_bid_total.max(1) as f64 * 100.0
    );
    println!(
        "    → DD confirms ≥ actual contract level:  {:>5} / {:>5} ({:.1}%)",
        nn_bid_dd_ge_contract,
        nn_bid_total,
        nn_bid_dd_ge_contract as f64 / nn_bid_total.max(1) as f64 * 100.0
    );
    println!(
        "  NN passes: {} / {} ({:.1}%)",
        nn_pass_total,
        n_deals,
        nn_pass_total as f64 / n_deals as f64 * 100.0
    );
    println!(
        "    → DD says best suit ≥80 (missed opportunity): {:>5} / {:>5} ({:.1}%)",
        nn_pass_best_dd_ge80,
        nn_pass_total,
        nn_pass_best_dd_ge80 as f64 / nn_pass_total.max(1) as f64 * 100.0
    );

    // === Per-suit table: DD success rate vs NN bid rate ===
    println!("\n  ┌──────────────────────────────────────────────────────────────────────────┐");
    println!("  │  DD success rate vs NN bid rate — by trump suit features (per-suit)     │");
    println!("  └──────────────────────────────────────────────────────────────────────────┘");
    println!(
        "  {:>4} {:>3} {:>3}  {:>6} {:>7} {:>8} {:>8} {:>8} {:>10}",
        "J", "9", "Cnt", "Occur", "AvgPts", "DD≥80", "DD≥100", "NN bids", "NN prec."
    );
    println!("  {}", "-".repeat(72));

    for has_j in [true, false] {
        for has_9 in [true, false] {
            for count in 0..=8usize {
                let b = &stats[has_j as usize][has_9 as usize][count];
                if b.total < 30 {
                    continue;
                }
                let avg = b.sum_dd_pts as f64 / b.total as f64;
                let dd80 = b.dd_ge80 as f64 / b.total as f64 * 100.0;
                let dd100 = b.dd_ge100 as f64 / b.total as f64 * 100.0;
                let nn_rate = b.nn_bids as f64 / b.total as f64 * 100.0;
                let nn_prec = if b.nn_bids > 0 {
                    b.nn_bids_and_dd_ge80 as f64 / b.nn_bids as f64 * 100.0
                } else {
                    0.0
                };
                let j = if has_j { "J" } else { "-" };
                let n = if has_9 { "9" } else { "-" };

                // Color: green if NN bid rate ≈ DD success, red if big gap
                let gap = (nn_rate - dd80).abs();
                let color = if gap < 15.0 {
                    "\x1b[32m"
                } else if gap < 30.0 {
                    "\x1b[33m"
                } else {
                    "\x1b[31m"
                };

                println!(
                    "  {:>4} {:>3} {:>3}  {:>6} {:>7.1} {:>7.1}% {:>7.1}% {}{:>7.1}%\x1b[0m {:>9.1}%",
                    j, n, count, b.total, avg, dd80, dd100, color, nn_rate, nn_prec
                );
            }
        }
    }

    // === Best-suit table ===
    println!("\n  ┌──────────────────────────────────────────────────────────────────────────┐");
    println!("  │  Best DD suit features vs NN bid rate                                   │");
    println!("  └──────────────────────────────────────────────────────────────────────────┘");
    println!(
        "  {:>4} {:>3} {:>3}  {:>6} {:>7} {:>8} {:>8} {:>8}",
        "J", "9", "Cnt", "Hands", "AvgPts", "DD≥80", "NN bids", "NN+DD≥80"
    );
    println!("  {}", "-".repeat(60));

    for has_j in [true, false] {
        for has_9 in [true, false] {
            for count in 0..=8usize {
                let b = &best[has_j as usize][has_9 as usize][count];
                if b.total < 20 {
                    continue;
                }
                let avg = b.sum_dd_pts as f64 / b.total as f64;
                let dd80 = b.dd_ge80 as f64 / b.total as f64 * 100.0;
                let nn_rate = b.nn_bids as f64 / b.total as f64 * 100.0;
                let both = b.nn_bids_and_dd_ge80 as f64 / b.total as f64 * 100.0;
                let j = if has_j { "J" } else { "-" };
                let n = if has_9 { "9" } else { "-" };

                println!(
                    "  {:>4} {:>3} {:>3}  {:>6} {:>7.1} {:>7.1}% {:>7.1}% {:>7.1}%",
                    j, n, count, b.total, avg, dd80, nn_rate, both
                );
            }
        }
    }

    // "Aux as"
    println!(
        "\n  \"Aux as\" (no J, no 9 anywhere): {} hands ({:.1}%)",
        no_honors_total,
        no_honors_total as f64 / n_deals as f64 * 100.0
    );
    if no_honors_total > 0 {
        println!(
            "    DD says any suit ≥80: {:>5} ({:.1}%)",
            no_honors_dd_ge80,
            no_honors_dd_ge80 as f64 / no_honors_total as f64 * 100.0
        );
        println!(
            "    NN bids:              {:>5} ({:.1}%)",
            no_honors_nn_bids,
            no_honors_nn_bids as f64 / no_honors_total as f64 * 100.0
        );
    }
}

// =====================================================================
fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/bid_nn_final.bin");

    let mut n_sims = 200usize;
    let mut n_mc = 10_000usize;

    let mut i = 2;
    while i < args.len() {
        match args[i].as_str() {
            "--sims" => {
                n_sims = args[i + 1].parse().unwrap();
                i += 2;
            }
            "--mc" => {
                n_mc = args[i + 1].parse().unwrap();
                i += 2;
            }
            _ => i += 1,
        }
    }

    let mut net = BidNet::load_with_hidden(model_path, 256)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", model_path, e));

    println!(
        "Bid DD Probe — NN: {} | {} sims/hand | {} MC deals\n",
        model_path, n_sims, n_mc
    );

    let mut rng = StdRng::seed_from_u64(123);

    let t0 = Instant::now();
    exp_jack(&mut net, n_sims, &mut rng);
    exp_aux_as(&mut net, n_sims, &mut rng);
    exp_trump_length(&mut net, n_sims, &mut rng);
    exp_side_strength(&mut net, n_sims, &mut rng);
    exp_belote(&mut net, n_sims, &mut rng);
    println!(
        "\n  Controlled experiments: {:.1}s",
        t0.elapsed().as_secs_f64()
    );

    exp_monte_carlo(&mut net, n_mc);

    println!("\nDone.");
}
