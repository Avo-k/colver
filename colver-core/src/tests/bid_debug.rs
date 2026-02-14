use colver_core::bid_eval::*;
use colver_core::bidding;
use colver_core::card::*;
use colver_core::state::*;
use rand::SeedableRng;

const SUIT_NAMES: [&str; 4] = ["S", "H", "D", "C"];
const SUIT_SYMBOLS: [&str; 4] = ["♠", "♥", "♦", "♣"];

fn hand_by_suit(hand: CardSet) -> String {
    let rank_names = ["7", "8", "9", "J", "Q", "K", "10", "A"];
    let mut parts = Vec::new();
    for suit_idx in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        if bits == 0 {
            parts.push(format!("{}: -", SUIT_SYMBOLS[suit_idx as usize]));
            continue;
        }
        let mut cards = Vec::new();
        let mut b = bits;
        while b != 0 {
            let rank = b.trailing_zeros() as usize;
            cards.push(rank_names[rank]);
            b &= b - 1;
        }
        // Show high to low
        cards.reverse();
        parts.push(format!("{}: {}", SUIT_SYMBOLS[suit_idx as usize], cards.join(" ")));
    }
    parts.join("  |  ")
}

fn eval_summary(hand: CardSet) -> String {
    let mut parts = Vec::new();
    for suit_idx in 0..4u8 {
        let suit = Suit::from_u8(suit_idx);
        let score = evaluate_for_trump(hand, suit);
        let bits = suit_bits(hand, suit);
        let count = bits.count_ones();
        let has_j = bits & (1 << 3) != 0;
        let has_9 = bits & (1 << 2) != 0;
        let has_a = bits & (1 << 7) != 0;
        let has_10 = bits & (1 << 6) != 0;
        let quality = has_j || has_9 || has_a || has_10 || count >= 3;
        parts.push(format!(
            "{}={:2}{}",
            SUIT_SYMBOLS[suit_idx as usize],
            score,
            if quality { "✓" } else { "✗" }
        ));
    }
    parts.join("  ")
}

fn action_str(action: u8) -> String {
    if action == 0 {
        "PASS".to_string()
    } else if action <= 40 {
        let (val, suit) = bidding::decode_bid(action);
        format!("{}{}", val * 10, SUIT_NAMES[suit as usize])
    } else if action == 41 {
        "COINCHE".to_string()
    } else if action == 42 {
        "SURCOINCHE".to_string()
    } else {
        format!("?{}", action)
    }
}

fn player_name(p: u8) -> &'static str {
    match p {
        0 => "North",
        1 => "East ",
        2 => "South",
        3 => "West ",
        _ => "?????",
    }
}

fn simulate_bidding(
    state: &GameState,
    bid_fn: BidFunction,
    label: &str,
) -> (GameState, Vec<(u8, u8)>) {
    let mut s = *state;
    let mut actions = Vec::new();
    while s.phase == Phase::Bidding && !s.is_terminal() {
        let action = bid_fn.bid(&s);
        actions.push((s.current_player, action));
        s.step(action);
    }
    // Print the bidding sequence
    let bid_strs: Vec<String> = actions
        .iter()
        .map(|(p, a)| format!("{}:{}", player_name(*p).trim(), action_str(*a)))
        .collect();
    let outcome = if s.phase == Phase::Playing {
        format!(
            "Contract: {}{} by team {}",
            s.contract.value * 10,
            SUIT_NAMES[s.contract.trump as usize],
            if s.contract.team == 0 { "NS" } else { "EW" }
        )
    } else {
        "VOID DEAL (4 passes)".to_string()
    };
    println!("  {:<12} {}  →  {}", label, bid_strs.join(", "), outcome);
    (s, actions)
}

fn main() {
    let num_deals: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(30);

    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);

    println!("=== BIDDING DIAGNOSTIC: {} deals (seed {}) ===\n", num_deals, seed);

    // Track statistics
    let mut heuristic_bids = 0u32;
    let mut heuristic_passes = 0u32;
    let mut improved_bids = 0u32;
    let mut improved_passes = 0u32;
    let mut disagree_count = 0u32;
    let mut heuristic_bids_improved_passes = 0u32;
    let mut improved_bids_heuristic_passes = 0u32;

    for deal_idx in 0..num_deals {
        let state = GameState::deal_random(deal_idx as u8 % 4, &mut rng);

        // Run both bidders
        let mut s_h = state;
        let mut h_actions = Vec::new();
        while s_h.phase == Phase::Bidding && !s_h.is_terminal() {
            let a = BidFunction::Heuristic.bid(&s_h);
            h_actions.push((s_h.current_player, a));
            s_h.step(a);
        }
        let h_void = s_h.phase == Phase::Done;

        let mut s_i = state;
        let mut i_actions = Vec::new();
        while s_i.phase == Phase::Bidding && !s_i.is_terminal() {
            let a = BidFunction::Improved.bid(&s_i);
            i_actions.push((s_i.current_player, a));
            s_i.step(a);
        }
        let i_void = s_i.phase == Phase::Done;

        if !h_void {
            heuristic_bids += 1;
        } else {
            heuristic_passes += 1;
        }
        if !i_void {
            improved_bids += 1;
        } else {
            improved_passes += 1;
        }

        // Check if they disagree
        let disagree = h_void != i_void
            || (!h_void
                && !i_void
                && (s_h.contract.team != s_i.contract.team
                    || s_h.contract.trump != s_i.contract.trump
                    || s_h.contract.value != s_i.contract.value));

        if disagree {
            disagree_count += 1;
        }
        if !h_void && i_void {
            heuristic_bids_improved_passes += 1;
        }
        if h_void && !i_void {
            improved_bids_heuristic_passes += 1;
        }

        // Only print interesting deals: where the bidders disagree
        // or every 5th deal for variety
        let print_this = disagree || deal_idx % 5 == 0;
        if !print_this {
            continue;
        }

        println!(
            "--- Deal {} (dealer={}) {} ---",
            deal_idx + 1,
            player_name(deal_idx as u8 % 4).trim(),
            if disagree { "*** DISAGREE ***" } else { "" }
        );
        for p in 0..4u8 {
            println!(
                "  {} [{}]: {}",
                player_name(p),
                if GameState::player_team(p) == 0 {
                    "NS"
                } else {
                    "EW"
                },
                hand_by_suit(state.hands[p as usize])
            );
            println!(
                "           Eval: {}",
                eval_summary(state.hands[p as usize])
            );
        }

        // Show bidding for each
        let h_strs: Vec<String> = h_actions
            .iter()
            .map(|(p, a)| format!("{}:{}", player_name(*p).trim(), action_str(*a)))
            .collect();
        let h_outcome = if !h_void {
            format!(
                "{}{} by {}",
                s_h.contract.value * 10,
                SUIT_NAMES[s_h.contract.trump as usize],
                if s_h.contract.team == 0 { "NS" } else { "EW" }
            )
        } else {
            "VOID".to_string()
        };
        println!("  Heuristic: {}  →  {}", h_strs.join(", "), h_outcome);

        let i_strs: Vec<String> = i_actions
            .iter()
            .map(|(p, a)| format!("{}:{}", player_name(*p).trim(), action_str(*a)))
            .collect();
        let i_outcome = if !i_void {
            format!(
                "{}{} by {}",
                s_i.contract.value * 10,
                SUIT_NAMES[s_i.contract.trump as usize],
                if s_i.contract.team == 0 { "NS" } else { "EW" }
            )
        } else {
            "VOID".to_string()
        };
        println!("  Improved:  {}  →  {}", i_strs.join(", "), i_outcome);

        println!();
    }

    println!("=== SUMMARY ({} deals) ===", num_deals);
    println!(
        "  Heuristic: {} contracts ({:.0}%), {} void ({:.0}%)",
        heuristic_bids,
        100.0 * heuristic_bids as f64 / num_deals as f64,
        heuristic_passes,
        100.0 * heuristic_passes as f64 / num_deals as f64,
    );
    println!(
        "  Improved:  {} contracts ({:.0}%), {} void ({:.0}%)",
        improved_bids,
        100.0 * improved_bids as f64 / num_deals as f64,
        improved_passes,
        100.0 * improved_passes as f64 / num_deals as f64,
    );
    println!("  Disagree: {} ({:.0}%)", disagree_count, 100.0 * disagree_count as f64 / num_deals as f64);
    println!(
        "  Heuristic bids, Improved passes: {} ({:.0}%)",
        heuristic_bids_improved_passes,
        100.0 * heuristic_bids_improved_passes as f64 / num_deals as f64
    );
    println!(
        "  Improved bids, Heuristic passes: {} ({:.0}%)",
        improved_bids_heuristic_passes,
        100.0 * improved_bids_heuristic_passes as f64 / num_deals as f64
    );
}
