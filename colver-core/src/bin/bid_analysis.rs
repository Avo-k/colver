/// Diagnostic: play games with full logging to understand bidding patterns.
use colver_core::bid_eval::{heuristic_bid, smart_bid, BidFunction};
use colver_core::bidding::{decode_bid, BID_COINCHE, BID_PASS, BID_SURCOINCHE};
use colver_core::card::cardset_str;
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::scoring::compute_deal_score;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};
use rand::Rng;

const SUIT_NAMES: [&str; 4] = ["Spades", "Hearts", "Diamonds", "Clubs"];
const PLAYER_NAMES: [&str; 4] = ["North", "East", "South", "West"];

fn team_name(team: u8) -> &'static str {
    if team == 0 { "NS" } else { "EW" }
}

fn action_str(action: u8) -> String {
    if action == BID_PASS {
        "PASS".into()
    } else if action == BID_COINCHE {
        "COINCHE".into()
    } else if action == BID_SURCOINCHE {
        "SURCOINCHE".into()
    } else if action <= 40 {
        let (val, suit) = decode_bid(action);
        if val == 25 {
            format!("Capot {}", SUIT_NAMES[suit as usize])
        } else {
            format!("{} {}", val as u16 * 10, SUIT_NAMES[suit as usize])
        }
    } else {
        format!("?{}", action)
    }
}

struct GameLog {
    hands: [String; 4],
    bid_actions: Vec<(u8, u8)>, // (player, action)
    contract_team: u8,
    contract_value: u16,
    contract_trump: String,
    contract_coinche: u8,
    void_deal: bool,
    taker_points: u8,
    defense_points: u8,
    taker_tricks: u8,
    reussi: bool,
    ns_score: i16,
    ew_score: i16,
}

fn play_game_logged(
    ns_use_beliefs: bool,
    ns_smart_bid: bool,
    ew_smart_bid: bool,
    time_ms: u32,
    dealer: u8,
    rng: &mut impl Rng,
) -> GameLog {
    let mut state = GameState::deal_random(dealer, rng);

    let hands = [
        cardset_str(state.hands[0]),
        cardset_str(state.hands[1]),
        cardset_str(state.hands[2]),
        cardset_str(state.hands[3]),
    ];

    let mut bid_actions = Vec::new();

    // --- Bidding phase (fast, no MCTS needed) ---
    while state.phase == Phase::Bidding && !state.is_terminal() {
        let player = state.current_player();
        let team = GameState::player_team(player);

        let action = if team == 0 {
            // NS
            if ns_smart_bid { smart_bid(&state) } else { heuristic_bid(&state) }
        } else {
            // EW
            if ew_smart_bid { smart_bid(&state) } else { heuristic_bid(&state) }
        };

        bid_actions.push((player, action));
        state.step(action);
    }

    if state.is_terminal() && state.contract.value == 0 {
        return GameLog {
            hands,
            bid_actions,
            contract_team: 0,
            contract_value: 0,
            contract_trump: "none".into(),
            contract_coinche: 0,
            void_deal: true,
            taker_points: 0,
            defense_points: 0,
            taker_tricks: 0,
            reussi: false,
            ns_score: 0,
            ew_score: 0,
        };
    }

    let contract_team = state.contract.team;
    let contract_value = state.contract.point_value();
    let contract_trump = SUIT_NAMES[state.contract.trump as usize].to_string();
    let contract_coinche = state.contract.coinche;

    // --- Play phase with MCTS ---
    let ew_bf = if ew_smart_bid { BidFunction::Smart } else { BidFunction::Heuristic };
    let ns_bf = if ns_smart_bid { BidFunction::Smart } else { BidFunction::Heuristic };

    let ew_config = NaiveIsMctsConfig {
        iterations_per_det: 50,
        time_limit_ms: Some(time_ms),
        bid_function: ew_bf,
        ..Default::default()
    };

    if ns_use_beliefs {
        let ns_config = SmartIsMctsConfig {
            iterations_per_det: 50,
            time_limit_ms: Some(time_ms),
            bid_function: ns_bf,
            ..Default::default()
        };
        let mut sp0 = SmartIsMctsSearch::new();
        let mut sp2 = SmartIsMctsSearch::new();
        let mut ew_search = NaiveIsMctsSearch::new();
        // Re-init from current state (after bidding)
        sp0.init_deal(&state, 0, true);
        sp2.init_deal(&state, 2, true);

        while !state.is_terminal() {
            let player = state.current_player();
            let sb = state;
            let action = match player {
                0 => sp0.search(&state, &ns_config, rng),
                2 => sp2.search(&state, &ns_config, rng),
                _ => ew_search.search(&state, &ew_config, rng),
            };
            sp0.record_action(&sb, player, action);
            sp2.record_action(&sb, player, action);
            state.step(action);
        }
    } else {
        let ns_config = NaiveIsMctsConfig {
            iterations_per_det: 50,
            time_limit_ms: Some(time_ms),
            bid_function: ns_bf,
            ..Default::default()
        };
        let mut ns_search = NaiveIsMctsSearch::new();
        let mut ew_search = NaiveIsMctsSearch::new();

        while !state.is_terminal() {
            let player = state.current_player();
            let action = match player {
                0 | 2 => ns_search.search(&state, &ns_config, rng),
                _ => ew_search.search(&state, &ew_config, rng),
            };
            state.step(action);
        }
    }

    let taker = contract_team as usize;
    let defense = 1 - taker;
    let taker_points = state.points[taker];
    let defense_points = state.points[defense];
    let taker_tricks = state.tricks_won[taker];

    let reussi = if state.contract.is_capot() {
        taker_tricks == 8
    } else {
        let belote_bonus = if state.belote[taker] == 2 { 20 } else { 0 };
        (taker_points as u16 + belote_bonus) >= contract_value
    };

    let score = compute_deal_score(&state);

    GameLog {
        hands,
        bid_actions,
        contract_team,
        contract_value,
        contract_trump,
        contract_coinche,
        void_deal: false,
        taker_points,
        defense_points,
        taker_tricks,
        reussi,
        ns_score: score.scores[0],
        ew_score: score.scores[1],
    }
}

fn print_game(i: usize, log: &GameLog) {
    println!("--- Game {} ---", i + 1);
    for p in 0..4 {
        println!("  {} ({}): {}", PLAYER_NAMES[p], team_name(GameState::player_team(p as u8)), log.hands[p]);
    }

    print!("  Bidding: ");
    for (player, action) in &log.bid_actions {
        print!("{}={} ", PLAYER_NAMES[*player as usize], action_str(*action));
    }
    println!();

    if log.void_deal {
        println!("  => VOID DEAL (4 passes)");
        println!();
        return;
    }

    let coinche_str = match log.contract_coinche {
        1 => " (coinche)",
        2 => " (surcoinche)",
        _ => "",
    };

    println!(
        "  Contract: {} {} by {}{} ",
        log.contract_value,
        log.contract_trump,
        team_name(log.contract_team),
        coinche_str,
    );

    let result = if log.reussi { "REUSSI" } else { "CHUTE" };
    println!(
        "  Result: {} — taker made {} pts ({} tricks), defense {} pts",
        result, log.taker_points, log.taker_tricks, log.defense_points
    );
    println!("  Score: NS={} EW={}", log.ns_score, log.ew_score);
    println!();
}

fn run_analysis(
    label: &str,
    ns_beliefs: bool,
    ns_smart_bid: bool,
    ew_smart_bid: bool,
    n_games: usize,
    time_ms: u32,
    rng: &mut impl Rng,
) {
    println!("========================================");
    println!("  {}", label);
    println!("  NS bid: {}  |  EW bid: {}  |  NS beliefs: {}",
        if ns_smart_bid { "smart" } else { "heuristic" },
        if ew_smart_bid { "smart" } else { "heuristic" },
        ns_beliefs,
    );
    println!("========================================");

    let mut logs = Vec::new();
    for i in 0..n_games {
        let log = play_game_logged(ns_beliefs, ns_smart_bid, ew_smart_bid, time_ms, (i % 4) as u8, rng);
        logs.push(log);
    }

    // Print first 5 detailed games
    let show = n_games.min(5);
    println!();
    println!("--- Sample games (first {}) ---", show);
    for i in 0..show {
        print_game(i, &logs[i]);
    }

    // Aggregate stats
    let mut void_deals = 0u32;
    let mut ns_bid_count = 0u32;
    let mut ew_bid_count = 0u32;
    let mut ns_contracts = 0u32;
    let mut ew_contracts = 0u32;
    let mut ns_reussi = 0u32;
    let mut ew_reussi = 0u32;
    let mut ns_chute = 0u32;
    let mut ew_chute = 0u32;
    let mut ns_total_bid_value = 0u32;
    let mut ew_total_bid_value = 0u32;
    let mut coinche_count = 0u32;
    let mut total_ns_score = 0i32;
    let mut total_ew_score = 0i32;
    let mut ns_chute_overbid = 0u32; // chute where points were close
    let mut ew_chute_overbid = 0u32;

    for log in &logs {
        if log.void_deal {
            void_deals += 1;
            continue;
        }

        total_ns_score += log.ns_score as i32;
        total_ew_score += log.ew_score as i32;

        if log.contract_coinche > 0 { coinche_count += 1; }

        // Count who bid (non-pass bids)
        for (player, action) in &log.bid_actions {
            if *action != BID_PASS && *action != BID_COINCHE && *action != BID_SURCOINCHE {
                if GameState::player_team(*player) == 0 { ns_bid_count += 1; } else { ew_bid_count += 1; }
            }
        }

        if log.contract_team == 0 {
            ns_contracts += 1;
            ns_total_bid_value += log.contract_value as u32;
            if log.reussi { ns_reussi += 1; } else {
                ns_chute += 1;
                // Was it close? (within 20 pts)
                if (log.taker_points as u16) + 20 >= log.contract_value { ns_chute_overbid += 1; }
            }
        } else {
            ew_contracts += 1;
            ew_total_bid_value += log.contract_value as u32;
            if log.reussi { ew_reussi += 1; } else {
                ew_chute += 1;
                if (log.taker_points as u16) + 20 >= log.contract_value { ew_chute_overbid += 1; }
            }
        }
    }

    let played = n_games as u32 - void_deals;

    println!("--- Aggregate ({} games, {} void) ---", n_games, void_deals);
    println!();
    println!("  Bidding activity: NS made {} bids, EW made {} bids", ns_bid_count, ew_bid_count);
    println!("  Coinche/surcoinche: {} games", coinche_count);
    println!();
    println!("  NS took contract: {} ({:.0}%)", ns_contracts,
        if played > 0 { ns_contracts as f64 / played as f64 * 100.0 } else { 0.0 });
    if ns_contracts > 0 {
        println!("    Avg bid: {:.0}", ns_total_bid_value as f64 / ns_contracts as f64);
        println!("    Reussi: {} ({:.0}%)  Chute: {} ({:.0}%)",
            ns_reussi, ns_reussi as f64 / ns_contracts as f64 * 100.0,
            ns_chute, ns_chute as f64 / ns_contracts as f64 * 100.0);
        if ns_chute > 0 {
            println!("    Close chutes (within 20): {} ({:.0}% of chutes)",
                ns_chute_overbid, ns_chute_overbid as f64 / ns_chute as f64 * 100.0);
        }
    }

    println!();
    println!("  EW took contract: {} ({:.0}%)", ew_contracts,
        if played > 0 { ew_contracts as f64 / played as f64 * 100.0 } else { 0.0 });
    if ew_contracts > 0 {
        println!("    Avg bid: {:.0}", ew_total_bid_value as f64 / ew_contracts as f64);
        println!("    Reussi: {} ({:.0}%)  Chute: {} ({:.0}%)",
            ew_reussi, ew_reussi as f64 / ew_contracts as f64 * 100.0,
            ew_chute, ew_chute as f64 / ew_contracts as f64 * 100.0);
        if ew_chute > 0 {
            println!("    Close chutes (within 20): {} ({:.0}% of chutes)",
                ew_chute_overbid, ew_chute_overbid as f64 / ew_chute as f64 * 100.0);
        }
    }

    println!();
    println!("  Avg score: NS {:.0}  EW {:.0}  (delta {:+.0})",
        if played > 0 { total_ns_score as f64 / played as f64 } else { 0.0 },
        if played > 0 { total_ew_score as f64 / played as f64 } else { 0.0 },
        if played > 0 { (total_ns_score - total_ew_score) as f64 / played as f64 } else { 0.0 },
    );
    println!();
}

fn main() {
    let n_games: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);

    let time_ms: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(20); // low default for fast analysis

    let mut rng = rand::thread_rng();

    println!("Bid Analysis: {} games, {}ms/move\n", n_games, time_ms);

    // Exp4 equivalent: Naive+smart vs Naive+heuristic (pure bidding effect)
    run_analysis(
        "Naive+smart_bid (NS) vs Naive+heuristic (EW)",
        false, true, false, n_games, time_ms, &mut rng,
    );

    // Exp2 equivalent: Smart+heuristic vs Naive+heuristic (belief effect)
    run_analysis(
        "Smart+heuristic (NS) vs Naive+heuristic (EW)",
        true, false, false, n_games, time_ms, &mut rng,
    );

    // Symmetric baseline: both heuristic, both naive (sanity check)
    run_analysis(
        "Naive+heuristic (NS) vs Naive+heuristic (EW) [baseline]",
        false, false, false, n_games, time_ms, &mut rng,
    );
}
