use colver_core::bid_eval::*;
use colver_core::bidding;
use colver_core::card::*;
use colver_core::mcts::*;
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

    // Track statistics per bidder
    let bidders: Vec<(&str, BidFunction)> = vec![
        ("Heuristic", BidFunction::Heuristic),
        ("Improved", BidFunction::Improved),
        ("ImprovedV2", BidFunction::ImprovedV2),
    ];
    let n_bidders = bidders.len();
    let mut bid_counts = vec![0u32; n_bidders];   // non-void deals
    let mut void_counts = vec![0u32; n_bidders];
    let mut contract_values = vec![Vec::new(); n_bidders]; // bid values when not void

    for deal_idx in 0..num_deals {
        let state = GameState::deal_random(deal_idx as u8 % 4, &mut rng);

        // Run all bidders
        let mut results: Vec<(GameState, Vec<(u8, u8)>, bool)> = Vec::new();
        for (_, bid_fn) in &bidders {
            let mut s = state;
            let mut actions = Vec::new();
            while s.phase == Phase::Bidding && !s.is_terminal() {
                let a = bid_fn.bid(&s);
                actions.push((s.current_player, a));
                s.step(a);
            }
            let is_void = s.phase == Phase::Done;
            results.push((s, actions, is_void));
        }

        // Update stats
        for i in 0..n_bidders {
            if results[i].2 {
                void_counts[i] += 1;
            } else {
                bid_counts[i] += 1;
                contract_values[i].push(results[i].0.contract.point_value());
            }
        }

        // Check if any bidders disagree
        let any_disagree = (0..n_bidders).any(|i| {
            (0..n_bidders).any(|j| {
                if i == j { return false; }
                let (si, _, vi) = &results[i];
                let (sj, _, vj) = &results[j];
                vi != vj || (!vi && !vj && (
                    si.contract.team != sj.contract.team
                    || si.contract.trump != sj.contract.trump
                    || si.contract.value != sj.contract.value
                ))
            })
        });

        // Print every 3rd deal or all disagreements
        let print_this = any_disagree || deal_idx % 3 == 0;
        if !print_this {
            continue;
        }

        println!(
            "--- Deal {} (dealer={}) {} ---",
            deal_idx + 1,
            player_name(deal_idx as u8 % 4).trim(),
            if any_disagree { "*** DISAGREE ***" } else { "" }
        );
        for p in 0..4u8 {
            println!(
                "  {} [{}]: {}",
                player_name(p),
                if GameState::player_team(p) == 0 { "NS" } else { "EW" },
                hand_by_suit(state.hands[p as usize])
            );
            println!(
                "           Eval: {}",
                eval_summary(state.hands[p as usize])
            );
        }

        for (i, (name, _)) in bidders.iter().enumerate() {
            let (ref s, ref actions, is_void) = results[i];
            let strs: Vec<String> = actions
                .iter()
                .map(|(p, a)| format!("{}:{}", player_name(*p).trim(), action_str(*a)))
                .collect();
            let outcome = if !is_void {
                format!(
                    "{}{} by {} {}",
                    s.contract.point_value(),
                    SUIT_NAMES[s.contract.trump as usize],
                    if s.contract.team == 0 { "NS" } else { "EW" },
                    if s.contract.coinche > 0 { "(X)" } else { "" },
                )
            } else {
                "VOID".to_string()
            };
            println!("  {:<10}  {}  →  {}", name, strs.join(", "), outcome);
        }

        println!();
    }

    println!("=== SUMMARY ({} deals) ===", num_deals);
    for (i, (name, _)) in bidders.iter().enumerate() {
        let avg_val = if !contract_values[i].is_empty() {
            contract_values[i].iter().sum::<u16>() as f64 / contract_values[i].len() as f64
        } else { 0.0 };
        println!(
            "  {:<10}  {} contracts ({:.0}%), {} void ({:.0}%), avg bid {:.0}",
            name,
            bid_counts[i],
            100.0 * bid_counts[i] as f64 / num_deals as f64,
            void_counts[i],
            100.0 * void_counts[i] as f64 / num_deals as f64,
            avg_val,
        );
    }

    // === CROSS-STRATEGY: NS vs EW with different bidders, full deal with Oracle MCTS ===
    println!("\n\n======================================================================");
    println!("=== CROSS-STRATEGY: Smart vs Improved (Oracle MCTS play) ===\n");

    let oracle_cfg = MctsConfig {
        iterations: 2000,
        rollout_policy: RolloutPolicy::HeuristicPlay,
        ..MctsConfig::default()
    };

    let mut rng2 = rand::rngs::StdRng::seed_from_u64(seed);
    for deal_idx in 0..num_deals.min(20) {
        let state = GameState::deal_random(deal_idx as u8 % 4, &mut rng2);

        // Direction A: NS=Smart, EW=Improved
        let (sa, bids_a) = cross_bid(&state, BidFunction::Smart, BidFunction::Improved);
        // Direction B: NS=Improved, EW=Smart
        let (sb, bids_b) = cross_bid(&state, BidFunction::Improved, BidFunction::Smart);

        // Only print if the two directions disagree on who takes the contract or the value differs by 20+
        let both_playing = sa.phase == Phase::Playing && sb.phase == Phase::Playing;
        let interesting = !both_playing
            || sa.contract.team != sb.contract.team
            || sa.contract.point_value().abs_diff(sb.contract.point_value()) >= 20;
        if !interesting { continue; }

        println!("--- Cross Deal {} (dealer={}) ---", deal_idx + 1, player_name(deal_idx as u8 % 4).trim());
        for p in 0..4u8 {
            println!("  {} [{}]: {}", player_name(p),
                if GameState::player_team(p) == 0 { "NS" } else { "EW" },
                hand_by_suit(state.hands[p as usize]));
        }

        // Print bidding for both directions
        print_cross_bids("NS=Smart EW=Impr", &sa, &bids_a);
        print_cross_bids("NS=Impr EW=Smart", &sb, &bids_b);

        // Play out both with Oracle MCTS
        if sa.phase == Phase::Playing {
            println!("  --- Play: NS=Smart EW=Impr → {}{}  ---",
                sa.contract.point_value(), SUIT_NAMES[sa.contract.trump as usize]);
            play_oracle_deal(sa, &oracle_cfg, &mut rng2);
        }
        if sb.phase == Phase::Playing {
            println!("  --- Play: NS=Impr EW=Smart → {}{}  ---",
                sb.contract.point_value(), SUIT_NAMES[sb.contract.trump as usize]);
            play_oracle_deal(sb, &oracle_cfg, &mut rng2);
        }
        println!();
    }
}

fn cross_bid(state: &GameState, ns_bid: BidFunction, ew_bid: BidFunction) -> (GameState, Vec<(u8, u8)>) {
    let mut s = *state;
    let mut actions = Vec::new();
    while s.phase == Phase::Bidding && !s.is_terminal() {
        let p = s.current_player();
        let bid_fn = if p == 0 || p == 2 { ns_bid } else { ew_bid };
        let a = bid_fn.bid(&s);
        actions.push((p, a));
        s.step(a);
    }
    (s, actions)
}

fn print_cross_bids(label: &str, state: &GameState, bids: &[(u8, u8)]) {
    let strs: Vec<String> = bids.iter()
        .map(|(p, a)| format!("{}:{}", player_name(*p).trim(), action_str(*a)))
        .collect();
    let outcome = if state.phase == Phase::Playing {
        format!("{}{} by {} {}",
            state.contract.point_value(), SUIT_NAMES[state.contract.trump as usize],
            if state.contract.team == 0 { "NS" } else { "EW" },
            if state.contract.coinche > 0 { "(X)" } else { "" })
    } else { "VOID".to_string() };
    println!("  {:<18} {}  →  {}", label, strs.join(", "), outcome);
}

fn play_oracle_deal(mut state: GameState, cfg: &MctsConfig, rng: &mut impl rand::Rng) {
    let mut search = MctsSearch::new();
    let taker = state.contract.team;
    let ct = ContractType::Color(Suit::from_u8(state.contract.trump));
    let mut trick_num = 0u8;

    while state.phase == Phase::Playing {
        let lead = state.trick_lead;
        let mut trick_cards: Vec<(u8, Card)> = Vec::new();

        // Play 4 cards
        for _ in 0..4 {
            if state.phase != Phase::Playing { break; }
            let action = search.search(&state, cfg, rng);
            trick_cards.push((state.current_player(), action));
            state.step(action);
        }

        if trick_cards.len() == 4 {
            trick_num += 1;
            let pts: u8 = trick_cards.iter().map(|(_, c)| card_points(*c, ct)).sum();
            let card_strs: Vec<String> = trick_cards.iter()
                .map(|(p, c)| format!("{}:{}", player_name(*p).trim(), card_name(*c)))
                .collect();
            println!("    T{}: {}  ({}pts, lead={})",
                trick_num, card_strs.join(", "), pts, player_name(lead).trim());
        }
    }

    let pts_taker = state.points[taker as usize];
    let pts_def = state.points[1 - taker as usize];
    let made = pts_taker >= state.contract.point_value() as u8;
    println!("    Result: taker({})={} pts, defense={} pts → {}",
        if taker == 0 { "NS" } else { "EW" },
        pts_taker, pts_def,
        if made { "MADE" } else { "FAILED" });
}
