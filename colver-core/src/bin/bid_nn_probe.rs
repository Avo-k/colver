/// Probe the bidding neural network to extract behavioral rules.
///
/// Runs controlled experiments + large-scale Monte Carlo to answer:
///   1. How does position (1st/2nd/3rd/4th to bid) affect decisions?
///   2. How important is the trump Jack?
///   3. Can you "annoncer aux as" (bid on aces alone)?
///   4. How many trump cards do you need?
///   5. How important is side strength (aces, voids)?
///   6. Does the NN support partner's bid? Coinche opponents?
///   7. Monte Carlo: statistical extraction of rules from 50k random deals
///
/// Usage:
///   cargo run -p colver-core --bin bid_nn_probe --release -- [model_path] [n_mc_deals]

use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bidding;
use colver_core::card::*;
use colver_core::state::GameState;
use rand::rngs::StdRng;
use rand::SeedableRng;

const SYM: [&str; 4] = ["♠", "♥", "♦", "♣"];
const RNK: [&str; 8] = ["7", "8", "9", "J", "Q", "K", "10", "A"];

// Rank indices
const R7: u8 = 0;
const R8: u8 = 1;
const R9: u8 = 2;
const RJ: u8 = 3;
const RQ: u8 = 4;
const RK: u8 = 5;
const R10: u8 = 6;
const RA: u8 = 7;

// Suit indices
const S: u8 = 0;
const H: u8 = 1;
const D: u8 = 2;
const C: u8 = 3;

fn c(suit: u8, rank: u8) -> u8 {
    suit * 8 + rank
}

fn hand_of(cards: &[u8]) -> u32 {
    assert_eq!(cards.len(), 8, "Hand must have exactly 8 cards");
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

/// Distribute remaining 24 cards to other 3 players deterministically.
fn fill_hands(seat: u8, hand: u32) -> [u32; 4] {
    let remaining: Vec<u8> = (0..32).filter(|&i| hand & (1 << i) == 0).collect();
    let mut hands = [0u32; 4];
    hands[seat as usize] = hand;
    let mut idx = 0;
    for p in 0..4u8 {
        if p == seat {
            continue;
        }
        for _ in 0..8 {
            hands[p as usize] |= 1u32 << remaining[idx];
            idx += 1;
        }
    }
    hands
}

/// Query the NN for a specific hand, position, and prior actions.
/// Returns (best_action, best_q, all_q_values).
fn query(
    net: &mut BidNet,
    hand: u32,
    seat: u8,
    position: u8,
    prior: &[(u8, u8)],
) -> (u8, f32, Vec<(u8, f32)>) {
    let dealer = (seat + 4 - position) % 4;
    let hands = fill_hands(seat, hand);
    let mut state = GameState::new(dealer, hands);

    let mut history = Vec::new();
    for &(s, a) in prior {
        history.push((s, a));
        state.step(a);
    }
    assert_eq!(
        state.current_player(),
        seat,
        "After replaying history, expected player {} but got {}",
        seat,
        state.current_player()
    );

    let obs = bid_obs::make_bid_observation(&state, &history);
    let legal = state.legal_actions();
    let (best, qvals) = net.best_action(&obs, legal);
    let best_q = qvals
        .iter()
        .find(|(a, _)| *a == best)
        .map(|(_, q)| *q)
        .unwrap_or(0.0);
    (best, best_q, qvals)
}

fn q_for(qvals: &[(u8, f32)], action: u8) -> f32 {
    qvals
        .iter()
        .find(|(a, _)| *a == action)
        .map(|(_, q)| *q)
        .unwrap_or(f32::NEG_INFINITY)
}

fn header(title: &str) {
    println!("\n{}", "=".repeat(90));
    println!("  {}", title);
    println!("{}\n", "=".repeat(90));
}

/// Print a one-line analysis: label → best action, PASS Q, Q(80) per suit
fn show(net: &mut BidNet, hand: u32, seat: u8, pos: u8, prior: &[(u8, u8)], label: &str) {
    let (best, best_q, qvals) = query(net, hand, seat, pos, prior);
    let qp = q_for(&qvals, 0);

    print!(
        "  {:<40} → {:>9} ({:+.3})  PASS={:+.3}  80:",
        label,
        act_str(best),
        best_q,
        qp
    );
    for s in 0..4u8 {
        let a80 = bidding::encode_bid(8, s);
        let q = q_for(&qvals, a80);
        if q > qp {
            // Green for bids better than pass
            print!(" \x1b[32m{}={:+.3}\x1b[0m", SYM[s as usize], q);
        } else {
            print!(" {}={:+.3}", SYM[s as usize], q);
        }
    }
    println!();
}

// =====================================================================
//  EXPERIMENT 1: Position Effect
// =====================================================================
fn exp_position(net: &mut BidNet) {
    header("EXPERIMENT 1: Position Effect");
    println!("  Theory: Position 4 (last) has more info → bids more confidently.");
    println!("  All other players pass before us.\n");

    let seat = 0u8;

    let hands: Vec<(&str, u32)> = vec![
        (
            "Strong (J9A♠ + sides)",
            hand_of(&[
                c(S, RJ),
                c(S, R9),
                c(S, RA),
                c(H, RA),
                c(D, R10),
                c(D, RK),
                c(C, R8),
                c(C, R7),
            ]),
        ),
        (
            "Medium (J87♠ + KQ♥)",
            hand_of(&[
                c(S, RJ),
                c(S, R8),
                c(S, R7),
                c(H, RK),
                c(H, RQ),
                c(D, R8),
                c(D, R7),
                c(C, R7),
            ]),
        ),
        (
            "Marginal (9♠ A10♠ + sides)",
            hand_of(&[
                c(S, R9),
                c(S, RA),
                c(S, R10),
                c(H, RK),
                c(D, R10),
                c(D, R8),
                c(C, R8),
                c(C, R7),
            ]),
        ),
    ];

    for (desc, hand) in &hands {
        println!("  Hand: {}  ({})\n", pretty(*hand), desc);
        for pos in 1..=4u8 {
            let dealer = (seat + 4 - pos) % 4;
            let mut prior = Vec::new();
            for i in 1..pos {
                prior.push(((dealer + i) % 4, 0u8));
            }
            let label = format!("Pos {} ({} pass before me)", pos, pos - 1);
            show(net, *hand, seat, pos, &prior, &label);
        }
        println!();
    }
}

// =====================================================================
//  EXPERIMENT 2: The Trump Jack
// =====================================================================
fn exp_jack(net: &mut BidNet) {
    header("EXPERIMENT 2: The Trump Jack");
    println!("  Theory: J is THE key card. 20 trump points, unbeatable.");
    println!("  All hands: 4 spades + K♥ Q♥ + 8♦ 7♦. Position 1.\n");

    let seat = 0u8;
    let side = [c(H, RK), c(H, RQ), c(D, R8), c(D, R7)];

    let cases: [(&str, [u8; 4]); 6] = [
        ("J 9 A 10  (monster)", [c(S, RJ), c(S, R9), c(S, RA), c(S, R10)]),
        ("J 9 A 8   (strong)", [c(S, RJ), c(S, R9), c(S, RA), c(S, R8)]),
        ("J A 10 8  (J no 9)", [c(S, RJ), c(S, RA), c(S, R10), c(S, R8)]),
        ("9 A 10 8  (9 no J)", [c(S, R9), c(S, RA), c(S, R10), c(S, R8)]),
        ("K Q A 10  (no J/9)", [c(S, RK), c(S, RQ), c(S, RA), c(S, R10)]),
        ("7 8 Q K   (garbage)", [c(S, R7), c(S, R8), c(S, RQ), c(S, RK)]),
    ];

    for (label, trump) in &cases {
        let mut cards = trump.to_vec();
        cards.extend_from_slice(&side);
        let hand = hand_of(&cards);
        show(net, hand, seat, 1, &[], label);
    }
}

// =====================================================================
//  EXPERIMENT 3: "Annoncer aux as"
// =====================================================================
fn exp_aux_as(net: &mut BidNet) {
    header("EXPERIMENT 3: \"Annoncer aux as\" — Bidding on Aces Alone");
    println!("  Theory: Aces without trump J/9 are not enough to bid.");
    println!("  Position 1.\n");

    let seat = 0u8;

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
            "A10KQ♠ (no J/9) + A♥ A♦ 7♥ 7♦",
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
            "A10KQ87♠ (6 trump no J/9) + A♥ 7♦",
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
            "A10KQ987♠ (7 trump no J) + 7♥",
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
        show(net, hand_of(cards), seat, 1, &[], label);
    }
}

// =====================================================================
//  EXPERIMENT 4: Trump Length
// =====================================================================
fn exp_trump_length(net: &mut BidNet) {
    header("EXPERIMENT 4: Trump Length (with Jack)");
    println!("  Theory: More trump = stronger bid. But 2 trump with J is marginal.\n");

    let seat = 0u8;

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
        show(net, hand_of(cards), seat, 1, &[], label);
    }
}

// =====================================================================
//  EXPERIMENT 5: Side Strength
// =====================================================================
fn exp_side_strength(net: &mut BidNet) {
    header("EXPERIMENT 5: Side Strength");
    println!("  Theory: Side aces protect against chute. J9A♠ trump is fixed.\n");

    let seat = 0u8;
    let trump = [c(S, RJ), c(S, R9), c(S, RA)];

    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "+ A♥ A♦ A♣ 10♥ 7♦  (3 side aces)",
            vec![c(H, RA), c(D, RA), c(C, RA), c(H, R10), c(D, R7)],
        ),
        (
            "+ A♥ A♦ K♣ Q♣ 7♦  (2 side aces)",
            vec![c(H, RA), c(D, RA), c(C, RK), c(C, RQ), c(D, R7)],
        ),
        (
            "+ A♥ K♦ Q♦ 8♣ 7♣  (1 side ace)",
            vec![c(H, RA), c(D, RK), c(D, RQ), c(C, R8), c(C, R7)],
        ),
        (
            "+ K♥ Q♥ K♦ Q♦ 7♣  (0 aces, KQ×2)",
            vec![c(H, RK), c(H, RQ), c(D, RK), c(D, RQ), c(C, R7)],
        ),
        (
            "+ 7♥ 8♥ 7♦ 8♦ 7♣  (0 aces, garbage)",
            vec![c(H, R7), c(H, R8), c(D, R7), c(D, R8), c(C, R7)],
        ),
    ];

    for (label, side) in &cases {
        let mut cards: Vec<u8> = trump.to_vec();
        cards.extend_from_slice(side);
        show(net, hand_of(&cards), seat, 1, &[], label);
    }
}

// =====================================================================
//  EXPERIMENT 6: Responding to Partner / Opponent
// =====================================================================
fn exp_responses(net: &mut BidNet) {
    header("EXPERIMENT 6a: Partner opened 80♠ — do we raise?");
    println!("  Seat 2 (partner) bid 80♠, seat 3 (opp) passed. Our turn.\n");

    let seat = 0u8;
    let partner_bid = bidding::encode_bid(8, S); // 80♠
    let prior_partner = vec![(2u8, partner_bid), (3u8, 0u8)];

    let cases: Vec<(&str, Vec<u8>)> = vec![
        (
            "J♠ 8♠ + A♥K♥ A♦10♦ 8♣7♣",
            vec![
                c(S, RJ),
                c(S, R8),
                c(H, RA),
                c(H, RK),
                c(D, RA),
                c(D, R10),
                c(C, R8),
                c(C, R7),
            ],
        ),
        (
            "9♠ A♠ + K♥Q♥ A♦ 8♦ 8♣7♣",
            vec![
                c(S, R9),
                c(S, RA),
                c(H, RK),
                c(H, RQ),
                c(D, RA),
                c(D, R8),
                c(C, R8),
                c(C, R7),
            ],
        ),
        (
            "A♠10♠ + A♥K♥ A♦10♦ 8♣7♣",
            vec![
                c(S, RA),
                c(S, R10),
                c(H, RA),
                c(H, RK),
                c(D, RA),
                c(D, R10),
                c(C, R8),
                c(C, R7),
            ],
        ),
        (
            "K♠Q♠ + A♥K♥ A♦10♦ A♣7♣",
            vec![
                c(S, RK),
                c(S, RQ),
                c(H, RA),
                c(H, RK),
                c(D, RA),
                c(D, R10),
                c(C, RA),
                c(C, R7),
            ],
        ),
        (
            "No ♠ at all: A♥K♥Q♥ A♦10♦K♦ A♣7♣",
            vec![
                c(H, RA),
                c(H, RK),
                c(H, RQ),
                c(D, RA),
                c(D, R10),
                c(D, RK),
                c(C, RA),
                c(C, R7),
            ],
        ),
        (
            "Garbage: 7♠ 7♥8♥ 7♦8♦ 7♣8♣Q♣",
            vec![
                c(S, R7),
                c(H, R7),
                c(H, R8),
                c(D, R7),
                c(D, R8),
                c(C, R7),
                c(C, R8),
                c(C, RQ),
            ],
        ),
    ];

    for (label, cards) in &cases {
        show(net, hand_of(cards), seat, 3, &prior_partner, label);
    }

    // --- 6b: Opponent opened 80♠, do we overbid or coinche? ---
    header("EXPERIMENT 6b: Opponent opened 80♠ — overbid or coinche?");
    println!("  Seat 1 (opp) bid 80♠, seat 2 (partner) passed, seat 3 (opp) passed. Our turn.\n");

    let opp_bid = bidding::encode_bid(8, S); // 80♠
    let prior_opp = vec![(1u8, opp_bid), (2u8, 0u8), (3u8, 0u8)];

    let cases_opp: Vec<(&str, Vec<u8>)> = vec![
        (
            "J♥9♥A♥10♥ + A♦K♦ 8♣7♣ (strong ♥)",
            vec![
                c(H, RJ),
                c(H, R9),
                c(H, RA),
                c(H, R10),
                c(D, RA),
                c(D, RK),
                c(C, R8),
                c(C, R7),
            ],
        ),
        (
            "J♥9♥A♥ + K♦Q♦ 8♣7♣10♣ (decent ♥)",
            vec![
                c(H, RJ),
                c(H, R9),
                c(H, RA),
                c(D, RK),
                c(D, RQ),
                c(C, R8),
                c(C, R7),
                c(C, R10),
            ],
        ),
        (
            "J♠9♠A♠10♠ + A♥K♥ 8♦7♦ (J♠ vs opp ♠!)",
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
            "Void ♠ + garbage (coinche?)",
            vec![
                c(H, R7),
                c(H, R8),
                c(D, R7),
                c(D, R8),
                c(D, RQ),
                c(C, R7),
                c(C, R8),
                c(C, RQ),
            ],
        ),
        (
            "A♠K♠Q♠10♠ + A♥A♦ 8♣7♣ (big ♠, coinche?)",
            vec![
                c(S, RA),
                c(S, RK),
                c(S, RQ),
                c(S, R10),
                c(H, RA),
                c(D, RA),
                c(C, R8),
                c(C, R7),
            ],
        ),
    ];

    for (label, cards) in &cases_opp {
        show(net, hand_of(cards), seat, 4, &prior_opp, label);
    }
}

// =====================================================================
//  EXPERIMENT 7: Belote (K+Q of trump)
// =====================================================================
fn exp_belote(net: &mut BidNet) {
    header("EXPERIMENT 7: Belote Bonus (K+Q of trump = +20 pts)");
    println!("  Theory: Having KQ of trump gives 20 bonus → easier to make contract.\n");

    let seat = 0u8;

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
        show(net, hand_of(cards), seat, 1, &[], label);
    }
}

// =====================================================================
//  EXPERIMENT 8: Monte Carlo — Statistical Rules Extraction
// =====================================================================
fn exp_monte_carlo(net: &mut BidNet, n_deals: usize) {
    header(&format!(
        "EXPERIMENT 8: Monte Carlo ({} random deals, position 1)",
        n_deals
    ));

    let mut rng = StdRng::seed_from_u64(42);

    // --- Per-suit stats by (has_jack, has_nine, count) ---
    // stats[j][n][count] = (total_occurrences, bid_this_suit_count, sum_q80)
    let mut stats = vec![vec![vec![(0u32, 0u32, 0f64); 9]; 2]; 2];

    // --- Overall action distribution ---
    let mut action_counts = [0u32; 43];
    let mut total_bids = 0u32;
    let mut bid_level_counts = [0u32; 10]; // idx 0=80, ..., 8=160, 9=capot

    // --- "Aux as" detection ---
    let mut no_j_no_9_anywhere = 0u32;
    let mut no_j_no_9_bids = 0u32;

    // --- Side aces when bidding ---
    let mut side_aces_bid = [0u32; 4]; // [0..3] side aces → count of bids
    let mut side_aces_total = [0u32; 4]; // for bid rate

    // --- Best-suit analysis ---
    // For each hand, find the suit with highest Q(80). Record characteristics and bid rate.
    // best_suit_stats[j][n][count] = (total, bid_count)
    let mut best_stats = vec![vec![vec![(0u32, 0u32); 9]; 2]; 2];

    for _ in 0..n_deals {
        // Deal random hand to seat 0, dealer=3 → seat 0 is position 1
        let state = GameState::deal_random(3, &mut rng);
        let hand = state.hands[0];

        let obs = bid_obs::make_bid_observation(&state, &[]);
        let legal = state.legal_actions();
        let (best, qvals) = net.best_action(&obs, legal);

        action_counts[best as usize] += 1;
        let bids = best >= 1 && best <= 40;
        if bids {
            total_bids += 1;
        }

        // Decode bid info
        let (bid_suit, _bid_level_idx) = if bids {
            let (val, suit) = bidding::decode_bid(best);
            let idx = if val == 25 {
                9
            } else {
                (val - 8) as usize
            };
            bid_level_counts[idx] += 1;
            (Some(suit), Some(idx))
        } else {
            (None, None)
        };

        // Find best suit by Q(80)
        let mut best_suit_q = f32::NEG_INFINITY;
        let mut best_suit_idx = 0u8;
        for s in 0..4u8 {
            let q = q_for(&qvals, bidding::encode_bid(8, s));
            if q > best_suit_q {
                best_suit_q = q;
                best_suit_idx = s;
            }
        }

        // Best-suit stats
        {
            let bits = suit_bits(hand, Suit::from_u8(best_suit_idx));
            let count = bits.count_ones() as usize;
            let has_j = (bits >> RJ) & 1 == 1;
            let has_9 = (bits >> R9) & 1 == 1;
            best_stats[has_j as usize][has_9 as usize][count].0 += 1;
            if bids {
                best_stats[has_j as usize][has_9 as usize][count].1 += 1;
            }
        }

        // Per-suit stats
        for s in 0..4u8 {
            let bits = suit_bits(hand, Suit::from_u8(s));
            let count = bits.count_ones() as usize;
            let has_j = (bits >> RJ) & 1 == 1;
            let has_9 = (bits >> R9) & 1 == 1;

            let q80 = q_for(&qvals, bidding::encode_bid(8, s));

            stats[has_j as usize][has_9 as usize][count].0 += 1;
            stats[has_j as usize][has_9 as usize][count].2 += q80 as f64;

            if bid_suit == Some(s) {
                stats[has_j as usize][has_9 as usize][count].1 += 1;
            }
        }

        // "Aux as" check
        let any_j = (0..4).any(|s| hand & (1 << (s * 8 + RJ)) != 0);
        let any_9 = (0..4).any(|s| hand & (1 << (s * 8 + R9)) != 0);
        if !any_j && !any_9 {
            no_j_no_9_anywhere += 1;
            if bids {
                no_j_no_9_bids += 1;
            }
        }

        // Side aces when bidding
        if let Some(bs) = bid_suit {
            let sa = (0..4u8)
                .filter(|&s| s != bs && hand & (1 << (s * 8 + RA)) != 0)
                .count();
            side_aces_bid[sa.min(3)] += 1;
        }
        // Count side aces for best suit (for total)
        {
            let sa = (0..4u8)
                .filter(|&s| s != best_suit_idx && hand & (1 << (s * 8 + RA)) != 0)
                .count();
            side_aces_total[sa.min(3)] += 1;
        }
    }

    // ===== Print results =====

    // Overall
    let pass_count = action_counts[0];
    let coinche_count = action_counts[41];
    println!(
        "  Overall: {} bids ({:.1}%), {} passes ({:.1}%), {} coinches\n",
        total_bids,
        total_bids as f64 / n_deals as f64 * 100.0,
        pass_count,
        pass_count as f64 / n_deals as f64 * 100.0,
        coinche_count
    );

    // Level distribution
    println!("  Bid level distribution (when bidding):");
    let names = [
        "80", "90", "100", "110", "120", "130", "140", "150", "160", "Capot",
    ];
    for (i, name) in names.iter().enumerate() {
        if bid_level_counts[i] > 0 {
            println!(
                "    {:>6}: {:>6} ({:>5.1}%)",
                name,
                bid_level_counts[i],
                bid_level_counts[i] as f64 / total_bids.max(1) as f64 * 100.0
            );
        }
    }

    // Per-suit Q(80) by characteristics
    println!("\n  ┌─────────────────────────────────────────────────────────────┐");
    println!("  │  Average Q(80) and bid rate by trump suit characteristics  │");
    println!("  │  (per-suit: each of 4 suits counted independently)         │");
    println!("  └─────────────────────────────────────────────────────────────┘");
    println!(
        "  {:>5} {:>4} {:>5}  {:>7} {:>9} {:>9}",
        "Jack", "Nine", "Count", "Occur", "Avg Q(80)", "Bid rate"
    );
    println!("  {}", "-".repeat(50));

    for has_j in [true, false] {
        for has_9 in [true, false] {
            for count in 0..=8usize {
                let (total, bids, sum_q) = stats[has_j as usize][has_9 as usize][count];
                if total < 20 {
                    continue;
                }
                let avg_q = sum_q / total as f64;
                let bid_rate = bids as f64 / total as f64 * 100.0;
                let j = if has_j { "J" } else { "-" };
                let n = if has_9 { "9" } else { "-" };

                let color = if avg_q > 0.01 {
                    "\x1b[32m"
                } else if avg_q < -0.01 {
                    "\x1b[31m"
                } else {
                    "\x1b[33m"
                };
                println!(
                    "  {:>5} {:>4} {:>5}  {:>7} {}{:>+9.4}\x1b[0m {:>8.1}%",
                    j, n, count, total, color, avg_q, bid_rate
                );
            }
        }
    }

    // Best-suit analysis
    println!("\n  ┌─────────────────────────────────────────────────────────────────┐");
    println!("  │  Bid rate when the BEST suit (highest Q(80)) has these traits  │");
    println!("  └─────────────────────────────────────────────────────────────────┘");
    println!(
        "  {:>5} {:>4} {:>5}  {:>7} {:>9}",
        "Jack", "Nine", "Count", "Hands", "Bid rate"
    );
    println!("  {}", "-".repeat(38));

    for has_j in [true, false] {
        for has_9 in [true, false] {
            for count in 0..=8usize {
                let (total, bids) = best_stats[has_j as usize][has_9 as usize][count];
                if total < 20 {
                    continue;
                }
                let bid_rate = bids as f64 / total as f64 * 100.0;
                let j = if has_j { "J" } else { "-" };
                let n = if has_9 { "9" } else { "-" };
                let color = if bid_rate > 50.0 {
                    "\x1b[32m"
                } else if bid_rate > 20.0 {
                    "\x1b[33m"
                } else {
                    "\x1b[31m"
                };
                println!(
                    "  {:>5} {:>4} {:>5}  {:>7} {}{:>8.1}%\x1b[0m",
                    j, n, count, total, color, bid_rate
                );
            }
        }
    }

    // Side aces
    println!("\n  Side aces (non-trump) when bidding:");
    for sa in 0..=3 {
        let total = side_aces_total[sa];
        let bids = side_aces_bid[sa];
        if total > 0 {
            println!(
                "    {} side ace(s): {:>5} / {:>5} bids ({:.1}%)",
                sa,
                bids,
                total,
                bids as f64 / total as f64 * 100.0
            );
        }
    }

    // "Aux as"
    println!(
        "\n  Hands with NO Jack and NO 9 in ANY suit: {} / {} ({:.1}%)",
        no_j_no_9_anywhere,
        n_deals,
        no_j_no_9_anywhere as f64 / n_deals as f64 * 100.0
    );
    if no_j_no_9_anywhere > 0 {
        println!(
            "    Of those, NN bids: {} ({:.1}%)",
            no_j_no_9_bids,
            no_j_no_9_bids as f64 / no_j_no_9_anywhere as f64 * 100.0
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
    let n_mc: usize = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(50_000);

    let mut net = BidNet::load_with_hidden(model_path, 256)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", model_path, e));

    println!(
        "Bid NN Probe — {} (obs={}, hidden={}, dueling={})\n",
        model_path,
        net.obs_dim(),
        net.hidden(),
        net.is_dueling()
    );

    exp_position(&mut net);
    exp_jack(&mut net);
    exp_aux_as(&mut net);
    exp_trump_length(&mut net);
    exp_side_strength(&mut net);
    exp_responses(&mut net);
    exp_belote(&mut net);
    exp_monte_carlo(&mut net, n_mc);

    println!("\nDone.");
}
