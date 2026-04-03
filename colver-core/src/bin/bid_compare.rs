//! Compare bidding strategies on real hands.
//!
//! Shows hands with complete per-suit scores for maxi_bid, improved_v2_bid, and bid_a_dd.
//!
//! Usage: cargo run --bin bid_compare --release -- [seed] [count]

use std::env;

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::bid_eval::{evaluate_for_trump, improved_v2_bid, quality_ok};
use colver_core::bidding;
use colver_core::card::*;
use colver_core::dd_bid::{DdBidConfig, DdBidder};
use colver_core::maxi::maxi_bid;
use colver_core::state::*;

const SUIT_CHARS: [char; 4] = ['♠', '♥', '♦', '♣'];
const RANK_CHARS: [char; 8] = ['7', '8', '9', 'J', 'Q', 'K', 'T', 'A'];

fn hand_string(hand: CardSet) -> String {
    let mut parts = Vec::new();
    for suit_idx in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        if bits == 0 {
            parts.push(format!("{}: -", SUIT_CHARS[suit_idx as usize]));
            continue;
        }
        let mut cards = String::new();
        for rank in [7, 6, 5, 4, 3, 2, 1, 0u8] {
            if bits & (1 << rank) != 0 {
                cards.push(RANK_CHARS[rank as usize]);
            }
        }
        parts.push(format!("{}: {}", SUIT_CHARS[suit_idx as usize], cards));
    }
    parts.join("  ")
}

fn action_name(action: u8) -> String {
    if action == 0 {
        return "PASS".to_string();
    }
    if action == 41 {
        return "COINCHE".to_string();
    }
    if action == 42 {
        return "SURCOINCHE".to_string();
    }
    let (value_enc, suit_idx) = bidding::decode_bid(action);
    let value = if value_enc == 25 {
        250
    } else {
        value_enc as u16 * 10
    };
    format!("{}{}", value, SUIT_CHARS[suit_idx as usize])
}

fn main() {
    let seed: u64 = env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);
    let count: usize = env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(5);

    let mut rng = StdRng::seed_from_u64(seed);
    let config = DdBidConfig {
        opening_dets: 12,
        ..Default::default()
    };
    let mut dd_bidder = DdBidder::new(config);

    println!("Bidding Comparison: Maxi vs ImprovedV2 vs BidADd");
    println!("Seed: {}, Count: {}", seed, count);
    println!();

    let mut shown = 0;
    let mut deal_idx = 0;
    while shown < count {
        deal_idx += 1;
        let state = GameState::deal_random(0, &mut rng);

        let player = state.current_player;
        let hand = state.hands[player as usize];

        // Get bids from all 3 strategies
        let maxi = maxi_bid(&state);
        let v2 = improved_v2_bid(&state);
        let dd_result = dd_bidder.bid_with_stats(&state, &mut rng);
        let dd = dd_result.action;

        // Skip if all three pass (boring)
        if maxi == 0 && v2 == 0 && dd == 0 {
            continue;
        }

        shown += 1;

        println!(
            "{}═══ Deal {} (player {}, dealer {}) ═══",
            if shown > 1 { "\n" } else { "" },
            deal_idx,
            player,
            state.dealer
        );
        println!("  Hand: {}", hand_string(hand));
        println!();

        // Per-suit detail table
        println!(
            "  {:>6}  {:>8}  {:>5}  {:>8}",
            "Suit", "Heur.scr", "QG", "DD pts"
        );
        println!("  {}", "-".repeat(35));

        let team = GameState::player_team(player);
        for suit_idx in 0..4u8 {
            let suit = Suit::from_u8(suit_idx);
            let score = evaluate_for_trump(hand, suit);
            let qg = quality_ok(hand, suit);
            let qg_str = if qg { "yes" } else { "no" };

            let dd_pts = dd_result.suit_expected_pts[suit_idx as usize];
            let dd_str = if dd_pts.is_nan() {
                "too weak".to_string()
            } else {
                let tp = if team == 0 { dd_pts } else { 162.0 - dd_pts };
                format!("{:.0}", tp)
            };

            println!(
                "  {:>6}  {:>8}  {:>5}  {:>8}",
                SUIT_CHARS[suit_idx as usize],
                score,
                qg_str,
                dd_str
            );
        }

        println!();
        println!("  Maxi:       {}", action_name(maxi));
        println!("  ImprovedV2: {}", action_name(v2));
        println!("  BidADd:     {}", action_name(dd));

        if maxi != v2 || v2 != dd || maxi != dd {
            let mut diffs = Vec::new();
            if maxi != v2 {
                diffs.push("Maxi≠V2");
            }
            if v2 != dd {
                diffs.push("V2≠DD");
            }
            if maxi != dd {
                diffs.push("Maxi≠DD");
            }
            println!("  ⚡ {}", diffs.join(", "));
        }
    }
}
