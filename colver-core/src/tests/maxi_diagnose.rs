/// Diagnostic: play individual deals Maxi (NS) vs DMC-40M (EW) with full play-by-play.
///
/// Usage:
///   cargo run --bin maxi_diagnose --release -- [num_deals] [seed]

use colver_core::bid_eval::{self, BidFunction};
use colver_core::bidding;
use colver_core::card::*;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM};
use colver_core::maxi;
use colver_core::state::*;
use rand::rngs::StdRng;
use rand::SeedableRng;

const SUIT_SYM: [&str; 4] = ["♠", "♥", "♦", "♣"];
const RANK_NAMES: [&str; 8] = ["7", "8", "9", "J", "Q", "K", "10", "A"];

fn player_name(p: u8) -> &'static str {
    match p { 0 => "N", 1 => "E", 2 => "S", 3 => "W", _ => "?" }
}

fn team_name(t: u8) -> &'static str {
    match t { 0 => "NS", 1 => "EW", _ => "??" }
}

fn hand_str(hand: CardSet) -> String {
    let mut parts = Vec::new();
    for suit_idx in 0..4u8 {
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        if bits == 0 {
            continue;
        }
        let mut cards = Vec::new();
        for rank in (0..8).rev() {
            if bits & (1 << rank) != 0 {
                cards.push(RANK_NAMES[rank]);
            }
        }
        parts.push(format!("{}{}", SUIT_SYM[suit_idx as usize], cards.join("")));
    }
    parts.join(" ")
}

fn card_str(card: Card) -> String {
    let suit = card_suit(card);
    let rank = card_rank(card);
    format!("{}{}", RANK_NAMES[rank as usize], SUIT_SYM[suit as u8 as usize])
}

fn action_str(action: u8) -> String {
    if action == 0 {
        "Pass".to_string()
    } else if action <= 40 {
        let (val, suit) = bidding::decode_bid(action);
        format!("{}{}", val * 10, SUIT_SYM[suit as usize])
    } else if action == 41 {
        "COINCHE".to_string()
    } else if action == 42 {
        "SURCOINCHE".to_string()
    } else {
        format!("?{}", action)
    }
}

/// Explain Maxi's bid reasoning for a given hand.
fn maxi_bid_reasoning(hand: CardSet, position: u8) -> String {
    let mut best_suit_name = "";
    let mut best_class = 0u8;
    let mut best_score = 0u16;

    for suit_idx in 0..4u8 {
        let suit = Suit::from_u8(suit_idx);
        let eval = maxi_eval_suit(hand, suit);
        if eval.trump_count < 2 { continue; }
        let class = maxi_classify(hand, suit, &eval, position);
        if class > best_class || (class == best_class && eval.score > best_score) {
            best_class = class;
            best_suit_name = SUIT_SYM[suit_idx as usize];
            best_score = eval.score;
        }
    }

    if best_class == 0 {
        return "no openable suit".to_string();
    }

    let case = match best_class {
        8 => "A(80)", 9 => "B(90)", 10 => "C(100)",
        11 => "D(110)", 12 => "D(120)", 13 => "D(130)",
        _ => "??",
    };

    // Show all suit evals
    let mut suit_info = Vec::new();
    for suit_idx in 0..4u8 {
        let suit = Suit::from_u8(suit_idx);
        let eval = maxi_eval_suit(hand, suit);
        let cnt = eval.trump_count;
        if cnt == 0 { continue; }
        let mut honors = String::new();
        if eval.has_jack { honors.push('J'); }
        if eval.has_nine { honors.push('9'); }
        if eval.has_ace { honors.push('A'); }
        if eval.has_ten { honors.push('T'); }
        if eval.has_king { honors.push('K'); }
        if eval.has_queen { honors.push('Q'); }
        let class = if eval.trump_count >= 2 {
            maxi_classify(hand, suit, &eval, position)
        } else { 0 };
        let class_str = match class {
            0 => "-", 8 => "A", 9 => "B", 10 => "C",
            11 => "D1", 12 => "D2", 13 => "D3", _ => "?",
        };
        suit_info.push(format!("{}{}({},sc={},cl={})", SUIT_SYM[suit_idx as usize], honors, cnt, eval.score, class_str));
    }

    format!("best={}{} [{}]", best_suit_name, case, suit_info.join(", "))
}

// Re-expose private helpers via wrapper (they're private in maxi.rs, so we duplicate minimal logic)
fn maxi_eval_suit(hand: CardSet, suit: Suit) -> SuitInfo {
    let bits = suit_bits(hand, suit);
    SuitInfo {
        has_jack: bits & (1 << 3) != 0,
        has_nine: bits & (1 << 2) != 0,
        has_ace: bits & (1 << 7) != 0,
        has_ten: bits & (1 << 6) != 0,
        has_king: bits & (1 << 5) != 0,
        has_queen: bits & (1 << 4) != 0,
        trump_count: bits.count_ones(),
        score: bid_eval::evaluate_for_trump(hand, suit),
        has_belote: (bits & (1 << 5) != 0) && (bits & (1 << 4) != 0),
    }
}

struct SuitInfo {
    has_jack: bool, has_nine: bool, has_ace: bool, has_ten: bool,
    has_king: bool, has_queen: bool, trump_count: u32, score: u16, has_belote: bool,
}

fn count_side_aces(hand: CardSet, trump: Suit) -> u32 {
    let mut count = 0u32;
    for suit_idx in 0..4u8 {
        if suit_idx == trump as u8 { continue; }
        if suit_bits(hand, Suit::from_u8(suit_idx)) & (1 << 7) != 0 { count += 1; }
    }
    count
}

fn count_losers(hand: CardSet, trump: Suit, eval: &SuitInfo) -> u32 {
    let mut losers = 0u32;
    if !eval.has_jack { losers += 1; }
    if !eval.has_nine { losers += 1; }
    for suit_idx in 0..4u8 {
        if suit_idx == trump as u8 { continue; }
        let bits = suit_bits(hand, Suit::from_u8(suit_idx));
        let count = bits.count_ones();
        if count == 0 { continue; }
        let has_a = bits & (1 << 7) != 0;
        if count == 1 && has_a { continue; }
        if count == 1 { losers += 1; }
        else {
            if !has_a { losers += 1; }
            if count >= 3 && (bits & (1 << 6) == 0) && !has_a { losers += 1; }
        }
    }
    losers
}

fn maxi_classify(hand: CardSet, suit: Suit, eval: &SuitInfo, position: u8) -> u8 {
    let side_aces = count_side_aces(hand, suit);
    if eval.has_jack && eval.has_nine {
        let losers = count_losers(hand, suit, eval);
        if losers <= 1 { return 13; }
        if losers <= 2 { return 12; }
        if losers <= 3 && (side_aces >= 1 || eval.trump_count >= 4) { return 11; }
    }
    if eval.trump_count >= 4 {
        if (eval.has_jack || eval.has_nine) && (eval.has_ace || eval.has_ten) { return 10; }
        if (eval.has_jack || eval.has_nine) && eval.has_ace && eval.has_ten && side_aces >= 1 { return 10; }
        if eval.trump_count >= 5 && side_aces >= 1 { return 10; }
    }
    if eval.has_jack && eval.has_nine {
        if eval.trump_count >= 3 || side_aces >= 1 { return 9; }
        if side_aces >= 1 { return 9; }
    }
    if eval.trump_count >= 3 {
        if eval.has_jack && (eval.has_ace || eval.has_ten) {
            if side_aces >= 1 || position >= 2 { return 8; }
            return 0;
        }
        if eval.has_nine && eval.has_ace {
            if side_aces >= 1 || position >= 2 { return 8; }
            return 0;
        }
        if eval.has_ace && eval.has_ten && (eval.has_king || eval.has_queen) && eval.trump_count >= 4 {
            if side_aces >= 2 || position >= 3 { return 8; }
            return 0;
        }
    }
    if eval.has_jack && eval.has_belote && eval.trump_count >= 3 { return 8; }
    if eval.trump_count >= 5 && eval.has_ace && eval.has_ten && eval.has_king && eval.has_queen { return 8; }
    if eval.has_jack && eval.has_nine && eval.trump_count == 2 && side_aces >= 1 && position >= 3 { return 8; }
    0
}

fn main() {
    let num_deals: usize = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(10);
    let seed: u64 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(42);

    // Load DMC-40M
    let dmc_path = "models/dmc_40000000.bin";
    let mut dmc_net = DmcNet::load(dmc_path).expect("Failed to load DMC-40M");
    println!("Loaded DMC-40M (obs={}, h={}, duel={})", dmc_net.obs_dim(), dmc_net.hidden(), dmc_net.is_dueling());

    let mut rng = StdRng::seed_from_u64(seed);
    let mut obs_buf = vec![0.0f32; OBS_DIM];

    println!("\n{}", "=".repeat(80));
    println!("  MAXI vs DMC-40M DIAGNOSTIC — {} deals (seed {})", num_deals, seed);
    println!("  NS = Maxi (bid + play),  EW = ImprovedV2 + DMC-40M");
    println!("{}\n", "=".repeat(80));

    let mut ns_wins = 0u32;
    let mut ew_wins = 0u32;
    let mut ns_total_score = 0i32;
    let mut ew_total_score = 0i32;

    for deal_idx in 0..num_deals {
        let dealer = deal_idx as u8 % 4;
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);

        println!("{}", "=".repeat(80));
        println!("  DEAL {} — dealer={}", deal_idx + 1, player_name(dealer));
        println!("{}", "=".repeat(80));

        // Print hands
        for p in 0..4u8 {
            let team = team_name(GameState::player_team(p));
            let agent = if p == 0 || p == 2 { "MAXI" } else { "DMC " };
            println!("  {} [{}] {}: {}", player_name(p), team, agent, hand_str(state.hands[p as usize]));
        }

        // Print Maxi suit evals for NS players
        for &p in &[0u8, 2] {
            let hand = state.hands[p as usize];
            let reasoning = maxi_bid_reasoning(hand, 0);
            println!("  {} maxi eval: {}", player_name(p), reasoning);
        }

        // Print ImprovedV2 evals for EW players
        for &p in &[1u8, 3] {
            let hand = state.hands[p as usize];
            let mut best_suit = 0u8;
            let mut best_score = 0u16;
            for suit_idx in 0..4u8 {
                let sc = bid_eval::evaluate_for_trump(hand, Suit::from_u8(suit_idx));
                if sc > best_score { best_score = sc; best_suit = suit_idx; }
            }
            println!("  {} iv2 eval:  best={}(sc={})", player_name(p), SUIT_SYM[best_suit as usize], best_score);
        }

        println!();

        // === BIDDING ===
        println!("  Bidding:");
        let mut bid_actions = Vec::new();
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let player = state.current_player();
            let is_ns = player == 0 || player == 2;

            let action = if is_ns {
                maxi::maxi_bid(&state)
            } else {
                BidFunction::ImprovedV2.bid(&state)
            };

            let agent_label = if is_ns { "MAXI" } else { "IV2 " };
            println!("    {} {} → {}", player_name(player), agent_label, action_str(action));
            bid_actions.push((player, action));
            tracking.track_action(&state, action);
            state.step(action);
        }

        if state.is_terminal() && state.phase == Phase::Done {
            let score = state.deal_score();
            if score.scores[0] == 0 && score.scores[1] == 0 {
                println!("  → VOID DEAL (4 passes)\n");
                continue;
            }
        }

        if state.phase != Phase::Playing {
            println!("  → No contract\n");
            continue;
        }

        let contract_value = state.contract.point_value();
        let contract_suit = state.contract.trump;
        let taker_team = state.contract.team;
        let coinche = state.contract.coinche;
        println!("  → Contract: {}{} by {} {}",
            contract_value, SUIT_SYM[contract_suit as usize],
            team_name(taker_team),
            if coinche > 0 { "(coinche)" } else { "" });
        println!();

        // === PLAY ===
        let ct = ContractType::Color(Suit::from_u8(contract_suit));
        let trump_suit = Suit::from_u8(contract_suit);
        let mut trick_num = 0u8;

        println!("  Play:");
        while state.phase == Phase::Playing {
            let lead = state.trick_lead;
            let lead_team = team_name(GameState::player_team(lead));
            trick_num += 1;

            let mut trick_cards: Vec<(u8, Card)> = Vec::new();
            let mut trick_details: Vec<String> = Vec::new();

            for i in 0..4 {
                if state.phase != Phase::Playing { break; }
                let player = state.current_player();
                let is_ns = player == 0 || player == 2;
                let state_before = state;

                let action = if is_ns {
                    maxi::maxi_play_action(&state)
                } else {
                    // DMC
                    dmc_obs::write_observation(&mut obs_buf, 0, &state, &tracking);
                    let legal_mask = state.legal_actions() as u32;
                    let (a, top_actions) = dmc_net.best_action(&obs_buf, legal_mask);

                    // Show top Q-values for DMC decisions
                    if i == 0 || true {
                        // Show top 3 Q-values
                        let top3: Vec<String> = top_actions.iter().take(3)
                            .map(|(card, q)| format!("{}={:.2}", card_str(*card), q))
                            .collect();
                        trick_details.push(format!("    {} DMC Q: [{}]", player_name(player), top3.join(", ")));
                    }
                    a
                };

                // Show what Maxi is thinking on leads
                if is_ns && i == 0 {
                    let my_team = GameState::player_team(player);
                    let on_attack = my_team == taker_team;
                    let my_trumps = cards_in_suit(state.hands[player as usize], trump_suit).count_ones();
                    let opp_trumps = count_opp_trumps(&state, player, trump_suit);
                    trick_details.push(format!("    {} MAXI lead: {} (trumps:mine={},opp={})",
                        player_name(player),
                        if on_attack { "ATTACK" } else { "DEFENSE" },
                        my_trumps, opp_trumps));
                }

                trick_cards.push((player, action));
                tracking.track_action(&state_before, action);
                state.step(action);
            }

            // Determine winner
            let pts: u8 = trick_cards.iter().map(|(_, c)| card_points(*c, ct)).sum();
            let is_last = trick_num == 8;
            let total_pts = if is_last { pts + if pts > 0 || state.points[0] + state.points[1] > 162 - 10 { 10 } else { 100 } } else { pts };

            let card_strs: Vec<String> = trick_cards.iter()
                .map(|(p, c)| {
                    let is_trump = card_suit(*c) == trump_suit;
                    format!("{}:{}{}", player_name(*p), card_str(*c), if is_trump { "*" } else { "" })
                })
                .collect();

            // Print details first
            for detail in &trick_details {
                println!("{}", detail);
            }

            let dix = if is_last { " (+dix de der)" } else { "" };
            println!("  T{}: {}  (lead {}={}) → {}pts{}",
                trick_num, card_strs.join("  "), player_name(lead), lead_team,
                pts, dix);
        }

        // === RESULT ===
        let score = state.deal_score();
        let pts_ns = state.points[0];
        let pts_ew = state.points[1];
        let made = state.points[taker_team as usize] >= contract_value as u8;

        let score_ns = score.scores[0] as i32;
        let score_ew = score.scores[1] as i32;
        ns_total_score += score_ns;
        ew_total_score += score_ew;
        if score_ns > score_ew { ns_wins += 1; }
        else if score_ew > score_ns { ew_wins += 1; }

        println!();
        println!("  Card points: NS={}, EW={}", pts_ns, pts_ew);
        println!("  Contract {}{} by {} → {}",
            contract_value, SUIT_SYM[contract_suit as usize],
            team_name(taker_team),
            if made { "MADE ✓" } else { "FAILED ✗" });
        println!("  Deal score: NS={}, EW={}", score_ns, score_ew);
        println!("  Running total: NS={}, EW={}", ns_total_score, ew_total_score);
        println!();
    }

    println!("{}", "=".repeat(80));
    println!("  SUMMARY: {} deals", num_deals);
    println!("  NS (Maxi) wins: {}/{} deals, total score: {}",
        ns_wins, ns_wins + ew_wins, ns_total_score);
    println!("  EW (DMC-40M) wins: {}/{} deals, total score: {}",
        ew_wins, ns_wins + ew_wins, ew_total_score);
    println!("{}", "=".repeat(80));
}

fn count_opp_trumps(state: &GameState, player: u8, trump_suit: Suit) -> u32 {
    let opp1 = (player + 1) % 4;
    let opp2 = (player + 3) % 4;
    cards_in_suit(state.hands[opp1 as usize], trump_suit).count_ones()
        + cards_in_suit(state.hands[opp2 as usize], trump_suit).count_ones()
}
