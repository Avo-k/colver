/// Diagnostic binary: evaluate BisDd belief quality and profile inference time.
///
/// Plays N games (BisDd bid+play vs improved_v2 bid + rule play) and collects:
/// - Belief coverage: does reality pass check_constraints?
/// - Belief restriction: acceptance rate of unconstrained determinizations
/// - Timing breakdown: bid/play decision times and determinization counts
/// - EV calibration: predicted EV vs actual outcome for bids
///
/// Usage:
///   cargo run --bin bis_dd_diag --release -- [num_games]

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::bid_eval::improved_v2_bid;
use colver_core::bidding::{self, BID_PASS};
use colver_core::bis_dd::{BisDdAgent, BisDdConfig};
use colver_core::determinize::determinize_greedy;
use colver_core::rule_player::rule_play_action;
use colver_core::state::{GameState, Phase};

const SUIT_SYM: [&str; 4] = ["\u{2660}", "\u{2665}", "\u{2666}", "\u{2663}"];

struct DiagStats {
    // Timing
    bid_times_ms: Vec<f64>,
    play_times_ms: Vec<f64>,

    // Belief coverage
    bid_reality_accepted: u64,
    bid_reality_total: u64,
    play_reality_accepted: u64,
    play_reality_total: u64,

    // Acceptance rate (how restrictive are beliefs)
    bid_acceptance_rates: Vec<f64>,

    // Decision counts
    total_bids: u64,
    total_passes: u64,
    total_plays: u64,

    // Bid records for outcome tracking
    bid_ev_records: Vec<BidEvRecord>,

    // Void deals (no contract)
    void_deals: u64,
}

struct BidEvRecord {
    bid_value: u16,
    suit: u8,
    team: u8,
    // Filled in after game ends:
    contract_made: Option<bool>,
}

impl DiagStats {
    fn new() -> Self {
        DiagStats {
            bid_times_ms: Vec::new(),
            play_times_ms: Vec::new(),
            bid_reality_accepted: 0,
            bid_reality_total: 0,
            play_reality_accepted: 0,
            play_reality_total: 0,
            bid_acceptance_rates: Vec::new(),
            total_bids: 0,
            total_passes: 0,
            total_plays: 0,
            bid_ev_records: Vec::new(),
            void_deals: 0,
        }
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return 0.0;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let num_games: usize = args.get(1).and_then(|s| s.parse().ok()).unwrap_or(50);
    let seed: u64 = args.get(2).and_then(|s| s.parse().ok()).unwrap_or(42);

    println!("Bis-DD Diagnostics ({} games, seed={})", num_games, seed);
    println!("=================================");
    println!("NS = BisDd (bid+play), EW = improved_v2 bid + rule play");
    println!();

    let mut rng = StdRng::seed_from_u64(seed);
    let mut stats = DiagStats::new();
    let mut det_rng = StdRng::seed_from_u64(seed.wrapping_add(1000));

    // Track match scores
    let mut cumulative_scores = [0i32; 2];
    let mut games_with_contract = 0u64;

    let total_start = Instant::now();

    for game_idx in 0..num_games {
        let dealer = (game_idx % 4) as u8;
        let state = GameState::deal_random(dealer, &mut rng);
        let real_hands = state.hands;

        // Create BisDd agents for NS (players 0 and 2)
        // Use higher budgets for diagnostic quality
        let config = BisDdConfig {
            min_dets: 20,
            bid_time_ms: 2000,
            play_time_ms: 500,
            ..BisDdConfig::default()
        };
        let mut agents = [
            BisDdAgent::new(config.clone(), seed + game_idx as u64 * 4),
            BisDdAgent::new(config.clone(), seed + game_idx as u64 * 4 + 1),
            BisDdAgent::new(config.clone(), seed + game_idx as u64 * 4 + 2),
            BisDdAgent::new(config.clone(), seed + game_idx as u64 * 4 + 3),
        ];

        // Initialize beliefs for NS team (players 0 and 2)
        for p in [0u8, 2] {
            agents[p as usize].init_deal(p, state.hands[p as usize]);
        }

        let mut cur = state;

        // Track bids made by NS in this game for EV calibration
        let mut ns_bid_indices: Vec<usize> = Vec::new();

        // --- Game loop ---
        while !cur.is_terminal() {
            let player = cur.current_player;
            let team = GameState::player_team(player);
            let is_ns = team == 0;
            let state_before = cur;

            let action;

            match cur.phase {
                Phase::Bidding => {
                    if is_ns {
                        // BisDd bid
                        let t0 = Instant::now();
                        action = agents[player as usize].decide(&cur);
                        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        stats.bid_times_ms.push(elapsed_ms);

                        // Check if reality passes constraints
                        if let Some(belief) = agents[player as usize].belief() {
                            stats.bid_reality_total += 1;
                            if belief.check_constraints(&real_hands) {
                                stats.bid_reality_accepted += 1;
                            }

                            // Measure acceptance rate: how many unconstrained
                            // determinizations pass the constraints?
                            if !belief.constraints().is_empty() {
                                let mut accepted = 0u32;
                                let trials = 100u32;
                                for _ in 0..trials {
                                    if let Some(det) =
                                        determinize_greedy(&cur, player, &mut det_rng)
                                    {
                                        if belief.check_constraints(&det.hands) {
                                            accepted += 1;
                                        }
                                    }
                                }
                                stats
                                    .bid_acceptance_rates
                                    .push(accepted as f64 / trials as f64);
                            }
                        }

                        if action == BID_PASS {
                            stats.total_passes += 1;
                        } else {
                            stats.total_bids += 1;
                            // Record for EV calibration
                            let (val, suit) = if action <= 40 {
                                bidding::decode_bid(action)
                            } else {
                                (0, 0)
                            };
                            let record_idx = stats.bid_ev_records.len();
                            stats.bid_ev_records.push(BidEvRecord {
                                bid_value: val as u16 * 10,
                                suit,
                                team: 0,
                                contract_made: None,
                            });
                            ns_bid_indices.push(record_idx);
                        }
                    } else {
                        // Opponent: improved_v2 bid
                        action = improved_v2_bid(&cur);
                    }
                }
                Phase::Playing => {
                    if is_ns {
                        // BisDd play
                        let t0 = Instant::now();
                        action = agents[player as usize].decide(&cur);
                        let elapsed_ms = t0.elapsed().as_secs_f64() * 1000.0;
                        stats.play_times_ms.push(elapsed_ms);
                        stats.total_plays += 1;

                        // Check if reality passes constraints during play
                        if let Some(belief) = agents[player as usize].belief() {
                            stats.play_reality_total += 1;
                            if belief.check_constraints(&real_hands) {
                                stats.play_reality_accepted += 1;
                            }
                        }
                    } else {
                        // Opponent: rule-based play
                        action = rule_play_action(&cur);
                    }
                }
                Phase::Done => unreachable!(),
            }

            // All agents observe the action (NS agents only)
            for p in [0u8, 2] {
                agents[p as usize].observe(player, action, &state_before);
            }

            cur.step(action);
        }

        // Game finished -- collect outcome
        if cur.contract.value == 0 {
            stats.void_deals += 1;
        } else {
            games_with_contract += 1;
            let deal_score = cur.deal_score();
            cumulative_scores[0] += deal_score.scores[0] as i32;
            cumulative_scores[1] += deal_score.scores[1] as i32;

            // Fill in EV calibration for NS bids in this game
            let contract_team = cur.contract.team;
            let made = deal_score.scores[contract_team as usize] > 0;
            for &idx in &ns_bid_indices {
                stats.bid_ev_records[idx].contract_made = Some(made);
            }
        }

        // Progress
        if (game_idx + 1) % 10 == 0 || game_idx + 1 == num_games {
            let elapsed = total_start.elapsed().as_secs_f64();
            eprint!(
                "\r  Game {}/{} ({:.1}s, {:.1} games/s)",
                game_idx + 1,
                num_games,
                elapsed,
                (game_idx + 1) as f64 / elapsed
            );
        }
    }
    eprintln!();

    let total_elapsed = total_start.elapsed().as_secs_f64();

    // === Report ===

    println!();
    println!("BELIEF QUALITY:");
    if stats.bid_reality_total > 0 {
        let rate = stats.bid_reality_accepted as f64 / stats.bid_reality_total as f64 * 100.0;
        println!(
            "  Bid coverage (reality accepted):  {:.1}% ({}/{})",
            rate, stats.bid_reality_accepted, stats.bid_reality_total
        );
    } else {
        println!("  Bid coverage: no bid decisions recorded");
    }
    if stats.play_reality_total > 0 {
        let rate = stats.play_reality_accepted as f64 / stats.play_reality_total as f64 * 100.0;
        println!(
            "  Play coverage (reality accepted): {:.1}% ({}/{})",
            rate, stats.play_reality_accepted, stats.play_reality_total
        );
    } else {
        println!("  Play coverage: no play decisions recorded");
    }
    if !stats.bid_acceptance_rates.is_empty() {
        let avg: f64 =
            stats.bid_acceptance_rates.iter().sum::<f64>() / stats.bid_acceptance_rates.len() as f64;
        println!(
            "  Bid acceptance rate (avg):         {:.1}% (constraints reject {:.1}%)",
            avg * 100.0,
            (1.0 - avg) * 100.0
        );
        let mut sorted = stats.bid_acceptance_rates.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        println!(
            "    P5: {:.1}%  P50: {:.1}%  P95: {:.1}%",
            percentile(&sorted, 0.05) * 100.0,
            percentile(&sorted, 0.50) * 100.0,
            percentile(&sorted, 0.95) * 100.0
        );
    }

    println!();
    println!("TIMING:");

    // Bid timing
    if !stats.bid_times_ms.is_empty() {
        let mut sorted = stats.bid_times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg: f64 = sorted.iter().sum::<f64>() / sorted.len() as f64;
        println!("  Bid decisions:  {} total", sorted.len());
        println!(
            "    Avg: {:.0}ms  P50: {:.0}ms  P95: {:.0}ms  Max: {:.0}ms",
            avg,
            percentile(&sorted, 0.50),
            percentile(&sorted, 0.95),
            sorted.last().unwrap_or(&0.0)
        );
    }

    // Play timing
    if !stats.play_times_ms.is_empty() {
        let mut sorted = stats.play_times_ms.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
        let avg: f64 = sorted.iter().sum::<f64>() / sorted.len() as f64;
        println!("  Play decisions: {} total", sorted.len());
        println!(
            "    Avg: {:.0}ms  P50: {:.0}ms  P95: {:.0}ms  Max: {:.0}ms",
            avg,
            percentile(&sorted, 0.50),
            percentile(&sorted, 0.95),
            sorted.last().unwrap_or(&0.0)
        );
    }

    println!(
        "  Total wall time: {:.1}s ({:.2} games/s)",
        total_elapsed,
        num_games as f64 / total_elapsed
    );

    println!();
    println!("DECISIONS:");
    let total_bid_actions = stats.total_bids + stats.total_passes;
    if total_bid_actions > 0 {
        println!(
            "  Bids: {} ({:.0}%)  Passes: {} ({:.0}%)",
            stats.total_bids,
            stats.total_bids as f64 / total_bid_actions as f64 * 100.0,
            stats.total_passes,
            stats.total_passes as f64 / total_bid_actions as f64 * 100.0
        );
    }
    if !stats.bid_ev_records.is_empty() {
        let avg_bid_value: f64 = stats.bid_ev_records.iter().map(|r| r.bid_value as f64).sum::<f64>()
            / stats.bid_ev_records.len() as f64;
        println!("  Avg bid value: {:.0}", avg_bid_value);

        // Suit distribution of bids
        let mut suit_counts = [0u32; 4];
        for r in &stats.bid_ev_records {
            if (r.suit as usize) < 4 {
                suit_counts[r.suit as usize] += 1;
            }
        }
        println!(
            "  Suit distribution: {} {} {} {}",
            format!("{}:{}", SUIT_SYM[0], suit_counts[0]),
            format!("{}:{}", SUIT_SYM[1], suit_counts[1]),
            format!("{}:{}", SUIT_SYM[2], suit_counts[2]),
            format!("{}:{}", SUIT_SYM[3], suit_counts[3]),
        );
    }
    println!("  Play decisions: {}", stats.total_plays);
    println!("  Void deals: {}", stats.void_deals);

    // EV calibration
    println!();
    println!("OUTCOME:");
    if games_with_contract > 0 {
        let ns_avg = cumulative_scores[0] as f64 / games_with_contract as f64;
        let ew_avg = cumulative_scores[1] as f64 / games_with_contract as f64;
        println!(
            "  Games with contract: {}  Void: {}",
            games_with_contract, stats.void_deals
        );
        println!(
            "  NS avg score: {:.0}  EW avg score: {:.0}  (NS advantage: {:.0})",
            ns_avg,
            ew_avg,
            ns_avg - ew_avg
        );

        // Contract success rate for NS bids
        let ns_bids_with_outcome: Vec<&BidEvRecord> = stats
            .bid_ev_records
            .iter()
            .filter(|r| r.contract_made.is_some())
            .collect();
        if !ns_bids_with_outcome.is_empty() {
            // Count games where NS was the taker and contract was made/failed
            let ns_taker_made = ns_bids_with_outcome
                .iter()
                .filter(|r| r.contract_made == Some(true) && r.team == 0)
                .count();
            let ns_taker_total = ns_bids_with_outcome
                .iter()
                .filter(|r| r.team == 0)
                .count();
            if ns_taker_total > 0 {
                println!(
                    "  NS contract success rate: {:.1}% ({}/{})",
                    ns_taker_made as f64 / ns_taker_total as f64 * 100.0,
                    ns_taker_made,
                    ns_taker_total
                );
            }
        }

        // Bid value breakdown
        let mut by_value: std::collections::BTreeMap<u16, (u32, u32)> =
            std::collections::BTreeMap::new();
        for r in &stats.bid_ev_records {
            let entry = by_value.entry(r.bid_value).or_insert((0, 0));
            entry.0 += 1;
            if r.contract_made == Some(true) {
                entry.1 += 1;
            }
        }
        if !by_value.is_empty() {
            println!("  Bid value breakdown (count / made):");
            for (value, (count, made)) in &by_value {
                let pct = if *count > 0 {
                    *made as f64 / *count as f64 * 100.0
                } else {
                    0.0
                };
                println!("    {:>4}: {:>3} bids, {:>3} made ({:.0}%)", value, count, made, pct);
            }
        }
    } else {
        println!("  No games with contract (all void deals)");
    }
}
