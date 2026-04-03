/// Diagnostic binary: isolate why Smart IS-MCTS loses to Naive IS-MCTS.
///
/// Tests 5 configurations with identical budgets:
/// 1. Naive (heuristic_bid, greedy det)
/// 2. Smart (smart_bid, soft beliefs, weighted det) — current default
/// 3. Smart (heuristic_bid, soft beliefs, weighted det) — isolate bidding
/// 4. Smart (smart_bid, hard-only beliefs, weighted det) — isolate soft inference
/// 5. Smart (smart_bid, NO beliefs, greedy det) — isolate weighted det entirely
///
/// Logs per-game: contract, det success rates, belief accuracy, scores.
/// Prints full game traces for interesting games.
use colver_core::bid_eval::{heuristic_bid, improved_bid, smart_bid, BidFunction};
use colver_core::card::*;
use colver_core::card_beliefs::CardBeliefs;
use colver_core::determinize::{determinize_greedy, determinize_weighted};
use colver_core::mcts::{MctsConfig, MctsSearch, RolloutPolicy, SearchResult};
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout::select_nth_bit;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};
use rand::Rng;
use std::time::Instant;

const PLAYER_NAMES: [&str; 4] = ["North", "East", "South", "West"];
const SUIT_SYMBOLS: [&str; 4] = ["S", "H", "D", "C"];

fn bid_name(action: u8) -> String {
    match action {
        0 => "PASS".to_string(),
        41 => "COINCHE".to_string(),
        42 => "SURCOINCHE".to_string(),
        1..=36 => {
            let idx = action - 1;
            let value_idx = idx / 4;
            let suit_idx = idx % 4;
            let value = (value_idx as u16 + 8) * 10;
            format!("{}{}", value, SUIT_SYMBOLS[suit_idx as usize])
        }
        37..=40 => {
            let suit_idx = action - 37;
            format!("Capot{}", SUIT_SYMBOLS[suit_idx as usize])
        }
        _ => format!("?{}", action),
    }
}

#[derive(Clone)]
struct GameTrace {
    hands: [CardSet; 4],
    dealer: u8,
    bids: Vec<(u8, u8)>, // (player, action)
    plays: Vec<(u8, u8)>, // (player, card)
    contract_team: u8,
    contract_suit: u8,
    contract_value: u16,
    ns_score: i16,
    ew_score: i16,
}

impl GameTrace {
    fn print(&self) {
        println!("  Dealer: {}", PLAYER_NAMES[self.dealer as usize]);
        println!("  Hands:");
        for p in 0..4 {
            println!("    {}: {}", PLAYER_NAMES[p], cardset_str(self.hands[p]));
        }
        println!("  Bidding:");
        for (player, action) in &self.bids {
            println!(
                "    {}: {}",
                PLAYER_NAMES[*player as usize],
                bid_name(*action)
            );
        }
        if self.contract_value > 0 {
            println!(
                "  Contract: {}{} by {} (team {})",
                self.contract_value,
                SUIT_SYMBOLS[self.contract_suit as usize],
                PLAYER_NAMES[self.contract_team as usize * 2], // approximate
                if self.contract_team == 0 { "NS" } else { "EW" }
            );
        } else {
            println!("  Contract: VOID (4 passes)");
        }
        if !self.plays.is_empty() {
            println!("  Card play:");
            for (i, chunk) in self.plays.chunks(4).enumerate() {
                print!("    Trick {}: ", i + 1);
                for (player, card) in chunk {
                    print!("{}={} ", PLAYER_NAMES[*player as usize], card_name(*card));
                }
                println!();
            }
        }
        println!(
            "  Score: NS={}, EW={}",
            self.ns_score, self.ew_score
        );
    }
}

/// Measure determinization success rates for Smart IS-MCTS on a deal.
struct DetStats {
    weighted_success: u32,
    greedy_fallback: u32,
    total_failure: u32,
    total_attempts: u32,
}

impl DetStats {
    fn new() -> Self {
        DetStats {
            weighted_success: 0,
            greedy_fallback: 0,
            total_failure: 0,
            total_attempts: 0,
        }
    }

    fn print(&self) {
        if self.total_attempts == 0 {
            println!("  No determinization attempts");
            return;
        }
        println!(
            "  Dets: {} attempts, {} weighted ok ({:.0}%), {} greedy fallback ({:.0}%), {} failed ({:.0}%)",
            self.total_attempts,
            self.weighted_success,
            self.weighted_success as f64 / self.total_attempts as f64 * 100.0,
            self.greedy_fallback,
            self.greedy_fallback as f64 / self.total_attempts as f64 * 100.0,
            self.total_failure,
            self.total_failure as f64 / self.total_attempts as f64 * 100.0,
        );
    }
}

/// Measure belief accuracy: how well do normalized beliefs predict actual opponent cards?
fn belief_accuracy(
    beliefs: &CardBeliefs,
    actual_hands: &[CardSet; 4],
    observer: u8,
    played_cards: CardSet,
) -> (f32, f32, u32) {
    // For each unknown card, check if the player with highest belief actually has it
    let norm = beliefs.normalized_weights();
    let mut correct = 0u32;
    let mut total = 0u32;
    let mut kl_sum = 0.0f32;

    for card in 0..32u8 {
        let bit = card_to_bit(card);
        if played_cards & bit != 0 {
            continue;
        }
        if actual_hands[observer as usize] & bit != 0 {
            continue;
        }

        // Find who actually has this card
        let mut actual_holder = 255u8;
        for p in 0..4u8 {
            if actual_hands[p as usize] & bit != 0 {
                actual_holder = p;
                break;
            }
        }
        if actual_holder == 255 {
            continue;
        }

        // Find who beliefs think most likely has it
        let mut best_p = 0u8;
        let mut best_w = 0.0f32;
        for p in 0..4u8 {
            if p == observer {
                continue;
            }
            if norm[p as usize][card as usize] > best_w {
                best_w = norm[p as usize][card as usize];
                best_p = p;
            }
        }

        total += 1;
        if best_p == actual_holder {
            correct += 1;
        }

        // KL divergence contribution: -log(predicted_prob_of_actual_holder)
        let predicted_prob = norm[actual_holder as usize][card as usize].max(1e-10);
        kl_sum += -predicted_prob.ln();
    }

    let accuracy = if total > 0 {
        correct as f32 / total as f32
    } else {
        0.0
    };
    let avg_kl = if total > 0 {
        kl_sum / total as f32
    } else {
        0.0
    };

    (accuracy, avg_kl, total)
}

/// Test determinization success for a given state with beliefs
fn test_det_rates(
    state: &GameState,
    observer: u8,
    beliefs: &Option<CardBeliefs>,
    n_trials: u32,
    rng: &mut impl Rng,
) -> DetStats {
    let mut stats = DetStats::new();
    let weights = beliefs.as_ref().map(|b| b.normalized_weights());

    for _ in 0..n_trials {
        stats.total_attempts += 1;

        if let Some(ref w) = weights {
            if let Some(_) = determinize_weighted(state, observer, w, rng) {
                stats.weighted_success += 1;
            } else if let Some(_) = determinize_greedy(state, observer, rng) {
                stats.greedy_fallback += 1;
            } else {
                stats.total_failure += 1;
            }
        } else {
            if let Some(_) = determinize_greedy(state, observer, rng) {
                stats.weighted_success += 1; // counts as "primary success" for greedy-only
            } else {
                stats.total_failure += 1;
            }
        }
    }

    stats
}

#[derive(Clone, Copy)]
enum SmartVariant {
    /// Full Smart: smart_bid + soft beliefs + weighted det
    FullSmart,
    /// Smart with heuristic_bid (isolate bidding effect)
    HeuristicBidSmart,
    /// Smart with improved_bid + soft beliefs + weighted det
    ImprovedBidSmart,
    /// Smart with hard-only beliefs (no soft inference)
    HardOnlySmart,
    /// Smart with no beliefs (greedy det, same as Naive but with smart_bid)
    NoBeliefsSmartBid,
}

impl SmartVariant {
    fn label(&self) -> &'static str {
        match self {
            SmartVariant::FullSmart => "Smart(smart_bid+soft+weighted)",
            SmartVariant::HeuristicBidSmart => "Smart(heur_bid+soft+weighted)",
            SmartVariant::ImprovedBidSmart => "Smart(improved_bid+soft+weighted)",
            SmartVariant::HardOnlySmart => "Smart(smart_bid+hard+weighted)",
            SmartVariant::NoBeliefsSmartBid => "Smart(smart_bid+no_beliefs+greedy)",
        }
    }

    fn bid_function(&self) -> BidFunction {
        match self {
            SmartVariant::HeuristicBidSmart => BidFunction::Heuristic,
            SmartVariant::ImprovedBidSmart => BidFunction::Improved,
            _ => BidFunction::Smart,
        }
    }

    fn use_soft_inference(&self) -> bool {
        match self {
            SmartVariant::FullSmart | SmartVariant::HeuristicBidSmart | SmartVariant::ImprovedBidSmart => true,
            _ => false,
        }
    }

    fn use_beliefs(&self) -> bool {
        match self {
            SmartVariant::NoBeliefsSmartBid => false,
            _ => true,
        }
    }
}

struct MatchResult {
    ns_wins: u32,
    ew_wins: u32,
    draws: u32,
    ns_total_score: i64,
    ew_total_score: i64,
    elapsed: std::time::Duration,
    // Diagnostic counters
    ns_contracts: u32,
    ew_contracts: u32,
    ns_contracts_made: u32,
    ew_contracts_made: u32,
    void_deals: u32,
    // Det stats aggregated
    det_weighted_ok: u32,
    det_greedy_fallback: u32,
    det_failed: u32,
    det_total: u32,
    // Belief accuracy
    belief_accuracy_sum: f64,
    belief_accuracy_count: u32,
    // Collected traces for interesting games
    traces: Vec<GameTrace>,
}

impl MatchResult {
    fn new() -> Self {
        MatchResult {
            ns_wins: 0,
            ew_wins: 0,
            draws: 0,
            ns_total_score: 0,
            ew_total_score: 0,
            elapsed: std::time::Duration::ZERO,
            ns_contracts: 0,
            ew_contracts: 0,
            ns_contracts_made: 0,
            ew_contracts_made: 0,
            void_deals: 0,
            det_weighted_ok: 0,
            det_greedy_fallback: 0,
            det_failed: 0,
            det_total: 0,
            belief_accuracy_sum: 0.0,
            belief_accuracy_count: 0,
            traces: Vec::new(),
        }
    }
}

fn run_match(
    n_games: u32,
    ns_variant: SmartVariant,
    dets: u32,
    iters: u32,
    verbose: bool,
    max_traces: usize,
    rng: &mut impl Rng,
) -> MatchResult {
    let mut result = MatchResult::new();

    // NS config (Smart variant)
    let smart_config = SmartIsMctsConfig {
        determinizations: dets,
        iterations_per_det: iters,
        use_soft_inference: ns_variant.use_soft_inference(),
        bid_function: ns_variant.bid_function(),
        ..Default::default()
    };

    // EW config (Naive, always heuristic_bid)
    let naive_config = NaiveIsMctsConfig {
        determinizations: dets,
        iterations_per_det: iters,
        bid_function: BidFunction::Heuristic, // always heuristic_bid for baseline
        ..Default::default()
    };

    let start = Instant::now();

    for game in 0..n_games {
        let dealer = (game % 4) as u8;
        let mut state = GameState::deal_random(dealer, rng);
        let actual_hands = state.hands; // save for belief accuracy

        let mut search_p0 = SmartIsMctsSearch::new();
        let mut search_p2 = SmartIsMctsSearch::new();
        let mut naive_search = NaiveIsMctsSearch::new();

        // Initialize beliefs if using them
        if ns_variant.use_beliefs() {
            search_p0.init_deal(&state, 0, ns_variant.use_soft_inference());
            search_p2.init_deal(&state, 2, ns_variant.use_soft_inference());
        }

        let mut trace = GameTrace {
            hands: actual_hands,
            dealer,
            bids: Vec::new(),
            plays: Vec::new(),
            contract_team: 0,
            contract_suit: 0,
            contract_value: 0,
            ns_score: 0,
            ew_score: 0,
        };

        // Track det stats for this game
        let mut game_det_weighted = 0u32;
        let mut game_det_greedy = 0u32;
        let mut game_det_failed = 0u32;
        let mut game_det_total = 0u32;

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            let action = match player {
                0 | 2 => {
                    // NS: Smart variant
                    if ns_variant.use_beliefs() {
                        if player == 0 {
                            search_p0.search(&state, &smart_config, rng)
                        } else {
                            search_p2.search(&state, &smart_config, rng)
                        }
                    } else {
                        // No beliefs: use bid function for bids, then greedy det MCTS
                        if state.phase == Phase::Bidding {
                            ns_variant.bid_function().bid(&state)
                        } else {
                            // Use naive search (greedy det) but with smart_bid
                            let no_belief_config = NaiveIsMctsConfig {
                                determinizations: dets,
                                iterations_per_det: iters,
                                bid_function: ns_variant.bid_function(),
                                ..Default::default()
                            };
                            naive_search.search(&state, &no_belief_config, rng)
                        }
                    }
                }
                _ => {
                    // EW: Naive IS-MCTS (heuristic_bid + greedy det)
                    naive_search.search(&state, &naive_config, rng)
                }
            };

            // Record action in beliefs
            if ns_variant.use_beliefs() {
                search_p0.record_action(&state_before, player, action);
                search_p2.record_action(&state_before, player, action);
            }

            // Record for trace
            if state.phase == Phase::Bidding {
                trace.bids.push((player, action));
            } else {
                trace.plays.push((player, action));
            }

            state.step(action);
        }

        // Record contract info
        if state.contract.value > 0 {
            trace.contract_team = state.contract.team;
            trace.contract_suit = state.contract.trump;
            trace.contract_value = state.contract.point_value();
        }

        let score = state.deal_score();
        trace.ns_score = score.scores[0];
        trace.ew_score = score.scores[1];

        result.ns_total_score += score.scores[0] as i64;
        result.ew_total_score += score.scores[1] as i64;

        if score.scores[0] > score.scores[1] {
            result.ns_wins += 1;
        } else if score.scores[1] > score.scores[0] {
            result.ew_wins += 1;
        } else {
            result.draws += 1;
        }

        // Track contracts
        if state.contract.value == 0 {
            result.void_deals += 1;
        } else {
            if state.contract.team == 0 {
                result.ns_contracts += 1;
                if score.scores[0] > 0 {
                    result.ns_contracts_made += 1;
                }
            } else {
                result.ew_contracts += 1;
                if score.scores[1] > 0 {
                    result.ew_contracts_made += 1;
                }
            }
        }

        // Collect interesting traces
        if result.traces.len() < max_traces {
            // Collect games where NS lost badly (EW win by large margin)
            if score.scores[1] > score.scores[0] + 100 {
                result.traces.push(trace.clone());
            }
        }

        if verbose && (game + 1) % 50 == 0 {
            println!(
                "  [{:3}/{}] NS wins {:.1}%, avg NS={:.0} EW={:.0}",
                game + 1,
                n_games,
                result.ns_wins as f64 / (game + 1) as f64 * 100.0,
                result.ns_total_score as f64 / (game + 1) as f64,
                result.ew_total_score as f64 / (game + 1) as f64,
            );
        }
    }

    result.elapsed = start.elapsed();
    result
}

fn print_result(label: &str, n_games: u32, r: &MatchResult) {
    println!();
    println!("=== {} ({} games) ===", label, n_games);
    println!(
        "  NS wins: {} ({:.1}%), EW wins: {} ({:.1}%), Draws: {}",
        r.ns_wins,
        r.ns_wins as f64 / n_games as f64 * 100.0,
        r.ew_wins,
        r.ew_wins as f64 / n_games as f64 * 100.0,
        r.draws,
    );
    println!(
        "  Avg score: NS {:.0}, EW {:.0}",
        r.ns_total_score as f64 / n_games as f64,
        r.ew_total_score as f64 / n_games as f64,
    );
    let total_contracts = r.ns_contracts + r.ew_contracts;
    if total_contracts > 0 {
        println!(
            "  Contracts: NS took {}, made {:.0}% | EW took {}, made {:.0}% | Void: {}",
            r.ns_contracts,
            if r.ns_contracts > 0 {
                r.ns_contracts_made as f64 / r.ns_contracts as f64 * 100.0
            } else {
                0.0
            },
            r.ew_contracts,
            if r.ew_contracts > 0 {
                r.ew_contracts_made as f64 / r.ew_contracts as f64 * 100.0
            } else {
                0.0
            },
            r.void_deals,
        );
    }
    println!(
        "  Time: {:.2?} ({:.0}ms/game)",
        r.elapsed,
        r.elapsed.as_millis() as f64 / n_games as f64,
    );
}

/// Run a focused experiment: measure determinization success rates and belief accuracy
fn run_det_experiment(n_games: u32, dets: u32, rng: &mut impl Rng) {
    println!();
    println!("===== DETERMINIZATION DIAGNOSTICS ({} games) =====", n_games);

    let mut total_weighted_ok = 0u32;
    let mut total_greedy_fb = 0u32;
    let mut total_failed = 0u32;
    let mut total_attempts = 0u32;
    let mut total_belief_acc = 0.0f64;
    let mut total_belief_kl = 0.0f64;
    let mut belief_measurements = 0u32;

    // Track per-trick stats
    let mut per_trick_weighted_ok = [0u32; 8];
    let mut per_trick_total = [0u32; 8];
    let mut per_trick_belief_acc = [0.0f64; 8];
    let mut per_trick_belief_n = [0u32; 8];

    for game in 0..n_games {
        let dealer = (game % 4) as u8;
        let mut state = GameState::deal_random(dealer, rng);
        let actual_hands = state.hands;

        let mut beliefs_p0 = CardBeliefs::new(&state, 0);
        beliefs_p0.use_soft_inference = true;

        // Bid with heuristic_bid for simplicity (same for all)
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;
            let action = heuristic_bid(&state);
            beliefs_p0.record_action(&state_before, player, action);
            state.step(action);
        }

        if state.is_terminal() {
            continue;
        }

        let mut trick_num = 0u32;

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            // When it's P0's turn, measure det rates and belief accuracy
            if player == 0 {
                let weights = beliefs_p0.normalized_weights();

                // Test determinization rates
                for _ in 0..dets {
                    total_attempts += 1;
                    if trick_num < 8 {
                        per_trick_total[trick_num as usize] += 1;
                    }

                    if let Some(_) = determinize_weighted(&state, 0, &weights, rng) {
                        total_weighted_ok += 1;
                        if trick_num < 8 {
                            per_trick_weighted_ok[trick_num as usize] += 1;
                        }
                    } else if let Some(_) = determinize_greedy(&state, 0, rng) {
                        total_greedy_fb += 1;
                    } else {
                        total_failed += 1;
                    }
                }

                // Measure belief accuracy
                let (acc, kl, count) =
                    belief_accuracy(&beliefs_p0, &actual_hands, 0, state.played_cards);
                if count > 0 {
                    total_belief_acc += acc as f64;
                    total_belief_kl += kl as f64;
                    belief_measurements += 1;
                    if trick_num < 8 {
                        per_trick_belief_acc[trick_num as usize] += acc as f64;
                        per_trick_belief_n[trick_num as usize] += 1;
                    }
                }
            }

            // Play random action (fast)
            let legal = state.legal_actions();
            let count = legal.count_ones();
            let idx = rng.gen_range(0..count);
            let action = select_nth_bit(legal, idx);

            beliefs_p0.record_action(&state_before, player, action);
            state.step(action);

            if state_before.trick_count == 3 {
                trick_num += 1;
            }
        }
    }

    println!(
        "  Weighted success: {}/{} ({:.1}%)",
        total_weighted_ok,
        total_attempts,
        total_weighted_ok as f64 / total_attempts as f64 * 100.0,
    );
    println!(
        "  Greedy fallback:  {}/{} ({:.1}%)",
        total_greedy_fb,
        total_attempts,
        total_greedy_fb as f64 / total_attempts as f64 * 100.0,
    );
    println!(
        "  Total failures:   {}/{} ({:.1}%)",
        total_failed,
        total_attempts,
        total_failed as f64 / total_attempts as f64 * 100.0,
    );

    if belief_measurements > 0 {
        println!(
            "  Belief accuracy (avg): {:.1}% (baseline 33% = random among 3 players)",
            total_belief_acc / belief_measurements as f64 * 100.0,
        );
        println!(
            "  Belief KL div (avg):   {:.3}",
            total_belief_kl / belief_measurements as f64,
        );
    }

    println!();
    println!("  Per-trick determinization success (weighted):");
    for t in 0..8 {
        if per_trick_total[t] > 0 {
            let acc_str = if per_trick_belief_n[t] > 0 {
                format!(
                    ", belief_acc={:.1}%",
                    per_trick_belief_acc[t] / per_trick_belief_n[t] as f64 * 100.0
                )
            } else {
                String::new()
            };
            println!(
                "    Trick {}: {:.1}% weighted ok ({}/{}){}",
                t + 1,
                per_trick_weighted_ok[t] as f64 / per_trick_total[t] as f64 * 100.0,
                per_trick_weighted_ok[t],
                per_trick_total[t],
                acc_str,
            );
        }
    }
}

/// Print a few complete game traces to manually analyze
fn run_traced_games(n_games: u32, dets: u32, iters: u32, rng: &mut impl Rng) {
    println!();
    println!("===== FULL GAME TRACES (Smart NS vs Naive EW) =====");

    let smart_config = SmartIsMctsConfig {
        determinizations: dets,
        iterations_per_det: iters,
        bid_function: BidFunction::Smart,
        use_soft_inference: true,
        ..Default::default()
    };

    let naive_config = NaiveIsMctsConfig {
        determinizations: dets,
        iterations_per_det: iters,
        bid_function: BidFunction::Heuristic,
        ..Default::default()
    };

    let mut printed = 0u32;

    for game in 0..n_games {
        let dealer = (game % 4) as u8;
        let mut state = GameState::deal_random(dealer, rng);
        let actual_hands = state.hands;

        let mut search_p0 = SmartIsMctsSearch::new();
        let mut search_p2 = SmartIsMctsSearch::new();
        let mut naive_search = NaiveIsMctsSearch::new();

        search_p0.init_deal(&state, 0, true);
        search_p2.init_deal(&state, 2, true);

        let mut bids: Vec<(u8, u8)> = Vec::new();
        let mut plays: Vec<(u8, u8, String)> = Vec::new(); // (player, card, extra_info)

        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;

            let action = match player {
                0 | 2 => {
                    if player == 0 {
                        search_p0.search(&state, &smart_config, rng)
                    } else {
                        search_p2.search(&state, &smart_config, rng)
                    }
                }
                _ => naive_search.search(&state, &naive_config, rng),
            };

            search_p0.record_action(&state_before, player, action);
            search_p2.record_action(&state_before, player, action);

            if state.phase == Phase::Bidding {
                bids.push((player, action));
            } else {
                let legal = state.legal_actions();
                let legal_count = legal.count_ones();
                let team = if player % 2 == 0 { "NS" } else { "EW" };
                let engine = if player % 2 == 0 { "Smart" } else { "Naive" };
                let info = format!(
                    "[{} {}] legal={} hand={}",
                    engine,
                    team,
                    legal_count,
                    cardset_str(state.hands[player as usize]),
                );
                plays.push((player, action, info));
            }

            state.step(action);
        }

        let score = state.deal_score();

        // Print interesting games: NS lost, or close games
        let ns_lost = score.scores[1] > score.scores[0];
        if ns_lost && printed < 5 {
            printed += 1;
            println!();
            println!("--- Game {} (NS LOST) ---", game + 1);
            println!("  Dealer: {}", PLAYER_NAMES[dealer as usize]);
            println!("  Hands:");
            for p in 0..4 {
                println!(
                    "    {}: {}",
                    PLAYER_NAMES[p],
                    cardset_str(actual_hands[p])
                );
            }
            println!("  Bidding:");
            for (player, action) in &bids {
                println!(
                    "    {}: {}",
                    PLAYER_NAMES[*player as usize],
                    bid_name(*action)
                );
            }
            if state.contract.value > 0 {
                println!(
                    "  Contract: {}{} by team {}",
                    state.contract.point_value(),
                    SUIT_SYMBOLS[state.contract.trump as usize],
                    if state.contract.team == 0 { "NS" } else { "EW" }
                );
            }
            println!("  Card play:");
            for (i, chunk) in plays.chunks(4).enumerate() {
                println!("    Trick {}:", i + 1);
                for (player, card, info) in chunk {
                    println!(
                        "      {} plays {} {}",
                        PLAYER_NAMES[*player as usize],
                        card_name(*card),
                        info,
                    );
                }
            }
            println!(
                "  Score: NS={}, EW={}  → EW wins by {}",
                score.scores[0],
                score.scores[1],
                score.scores[1] - score.scores[0]
            );
        }

        if printed >= 5 {
            break;
        }
    }
}

fn main() {
    let mut rng = rand::thread_rng();
    let n_games: u32 = std::env::args()
        .nth(1)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200);
    let budget: u32 = std::env::args()
        .nth(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(1000);

    let dets = 20u32;
    let iters = budget / dets;

    println!("==========================================================");
    println!("  SMART IS-MCTS DIAGNOSTIC (NS vs Naive EW)");
    println!("  {} games, budget={}D x {}I = {}", n_games, dets, iters, dets * iters);
    println!("==========================================================");

    // Part 0: Determinization diagnostics (fast, random play)
    run_det_experiment(500, dets, &mut rng);

    // Part 1: Head-to-head comparisons isolating variables (1k budget)
    println!();
    println!("========== PART 1: ISOLATING VARIABLES (budget={}) ==========", dets * iters);

    let variants = [
        SmartVariant::FullSmart,
        SmartVariant::HeuristicBidSmart,
        SmartVariant::ImprovedBidSmart,
        SmartVariant::HardOnlySmart,
        SmartVariant::NoBeliefsSmartBid,
    ];

    for variant in &variants {
        println!();
        println!(
            "===== {} (NS) vs Naive(heur_bid) (EW) =====",
            variant.label()
        );

        let result = run_match(n_games, *variant, dets, iters, true, 3, &mut rng);
        print_result(
            &format!("{} vs Naive", variant.label()),
            n_games,
            &result,
        );

        // Print collected loss traces
        if !result.traces.is_empty() {
            println!("  Sample NS losses:");
            for (i, trace) in result.traces.iter().enumerate().take(2) {
                println!("  --- Loss #{} ---", i + 1);
                trace.print();
            }
        }
    }

    // Part 1b: Naive vs Naive baseline (both heuristic_bid)
    println!();
    println!("===== BASELINE: Naive(heur_bid) vs Naive(heur_bid) =====");
    {
        let naive_config = NaiveIsMctsConfig {
            determinizations: dets,
            iterations_per_det: iters,
            bid_function: BidFunction::Heuristic,
            ..Default::default()
        };
        let mut naive_result = MatchResult::new();
        let start = Instant::now();
        for game in 0..n_games {
            let dealer = (game % 4) as u8;
            let mut state = GameState::deal_random(dealer, &mut rng);
            let mut ns_search = NaiveIsMctsSearch::new();
            let mut ew_search = NaiveIsMctsSearch::new();
            while !state.is_terminal() {
                let player = state.current_player();
                let action = if player % 2 == 0 {
                    ns_search.search(&state, &naive_config, &mut rng)
                } else {
                    ew_search.search(&state, &naive_config, &mut rng)
                };
                state.step(action);
            }
            let score = state.deal_score();
            naive_result.ns_total_score += score.scores[0] as i64;
            naive_result.ew_total_score += score.scores[1] as i64;
            if score.scores[0] > score.scores[1] { naive_result.ns_wins += 1; }
            else if score.scores[1] > score.scores[0] { naive_result.ew_wins += 1; }
            else { naive_result.draws += 1; }
            if state.contract.value > 0 {
                if state.contract.team == 0 {
                    naive_result.ns_contracts += 1;
                    if score.scores[0] > 0 { naive_result.ns_contracts_made += 1; }
                } else {
                    naive_result.ew_contracts += 1;
                    if score.scores[1] > 0 { naive_result.ew_contracts_made += 1; }
                }
            } else { naive_result.void_deals += 1; }
        }
        naive_result.elapsed = start.elapsed();
        print_result("Naive vs Naive (same config)", n_games, &naive_result);
    }

    // Part 2: Higher budget test (beliefs should matter more)
    if budget >= 1000 {
        let hi_dets = 50u32;
        let hi_iters = 200u32;
        let hi_budget = hi_dets * hi_iters;
        println!();
        println!("========== PART 2: HIGHER BUDGET ({}) ==========", hi_budget);
        println!("  Testing if beliefs help with more computation...");

        let hi_variants = [
            SmartVariant::HeuristicBidSmart,  // beliefs + heur_bid
            SmartVariant::NoBeliefsSmartBid,   // no beliefs + smart_bid (as control)
        ];
        for variant in &hi_variants {
            let ew_bid = "heur_bid";
            println!();
            println!(
                "===== {} (NS) vs Naive({}) (EW), {}D x {}I =====",
                variant.label(), ew_bid, hi_dets, hi_iters
            );
            let result = run_match(n_games, *variant, hi_dets, hi_iters, true, 0, &mut rng);
            print_result(
                &format!("{} vs Naive ({})", variant.label(), hi_budget),
                n_games,
                &result,
            );
        }
    }

    // Part 3: Full game traces
    run_traced_games(100, dets, iters, &mut rng);
}
