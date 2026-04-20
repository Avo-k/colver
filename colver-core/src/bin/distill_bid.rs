/// Distill the bid NN V2 into interpretable features for ML training.
///
/// Generates a CSV with human-interpretable features + NN decisions
/// for hundreds of thousands of random hands across all positions and
/// common bidding scenarios.
///
/// Scenarios:
///   - pos1_open:       Position 1, opening bid (no prior actions)
///   - pos2_after_pass: Position 2, after 1 pass from opponent
///   - pos3_after_2p:   Position 3, after 2 passes (partner passed, opp passed)
///   - pos4_after_3p:   Position 4, after 3 passes (last chance)
///   - pos2_after_opp80: Position 2, opponent opened 80 in each suit
///   - pos3_partner80:  Position 3, partner opened 80 (opponent passed)
///   - pos4_partner80:  Position 4, partner (pos2) bid 80, opponent passed
///   - pos3_opp80:      Position 3, opponent (pos2) bid 80 over partner's pass
///   - pos4_opp80:      Position 4, opp (pos3) bid 80 over 2 passes
///
/// Usage:
///   cargo run -p colver-core --bin distill_bid --release -- [model_path] [n_deals] [output_path] [my_score] [opp_score]
///
/// Defaults: models/bid_v2/bid_nn_final.bin, 200000 deals, data/distill/bid_distill.csv, 0, 0
///
/// Observation dim is auto-detected from the weight file:
///   108 → legacy v1 bid obs (bid_v1/v2/v3/v4 with --reward real using 110 are not supported here)
///   110 → score-aware v1 (my_score/opp_score raw)
///   113 → score-aware v2 (v5 default: adds win_prob, leader_dist, diff)

use colver_core::bid_eval::evaluate_for_trump;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bidding;
use colver_core::card::*;
use colver_core::state::GameState;
use rand::rngs::StdRng;
use rand::SeedableRng;
use std::io::Write;

/// A bidding scenario to evaluate.
struct Scenario {
    /// Human-readable name for the scenario.
    name: &'static str,
    /// Dealer seat (determines who bids in what order).
    dealer: u8,
    /// The seat we're evaluating (will be current_player after replaying history).
    seat: u8,
    /// Bidding position (1-4) of our seat.
    position: u8,
    /// Prior actions: (seat, action) pairs to replay before our turn.
    /// For scenarios with a prior bid in a specific suit, we use suit_idx=0
    /// as placeholder and remap per-deal.
    prior_template: Vec<(u8, u8)>,
    /// If true, the prior bid suit is varied (4 sub-scenarios per suit).
    /// The template should contain encode_bid(8, 0) as placeholder for the bid.
    varies_by_suit: bool,
    /// Is there a partner bid in the history? (for feature: partner_bid_suit)
    partner_bid: bool,
    /// Is there an opponent bid in the history? (for feature: opp_bid_suit)
    opp_bid: bool,
}

fn scenarios() -> Vec<Scenario> {
    // Seat layout: dealer=0 → pos1=seat1, pos2=seat2, pos3=seat3, pos4=seat0
    // Teams: 0,2 (NS) and 1,3 (EW). Partner = seat^2.
    //
    // With dealer=0:
    //   pos1=seat1(EW), pos2=seat2(NS), pos3=seat3(EW), pos4=seat0(NS)
    //
    // We always evaluate seat=0 for simplicity. We pick dealer to control position.
    //   seat=0, dealer=3 → pos1 (first bidder)
    //   seat=0, dealer=2 → pos2 (seat3 acts first, then seat0)
    //   seat=0, dealer=1 → pos3 (seat2 acts first, seat3 second, then seat0)
    //   seat=0, dealer=0 → pos4 (seat1, seat2, seat3, then seat0)
    //
    // Partner of seat0 = seat2. Opponents = seat1, seat3.

    let pass = 0u8;
    let bid80s = bidding::encode_bid(8, 0); // 80♠ as placeholder

    vec![
        // ===== All-pass scenarios (by position) =====
        Scenario {
            name: "pos1_open",
            dealer: 3,
            seat: 0,
            position: 1,
            prior_template: vec![],
            varies_by_suit: false,
            partner_bid: false,
            opp_bid: false,
        },
        Scenario {
            name: "pos2_after_pass",
            dealer: 2,
            seat: 0,
            position: 2,
            // seat3 (opp) passes
            prior_template: vec![(3, pass)],
            varies_by_suit: false,
            partner_bid: false,
            opp_bid: false,
        },
        Scenario {
            name: "pos3_after_2p",
            dealer: 1,
            seat: 0,
            position: 3,
            // seat2 (partner) passes, seat3 (opp) passes
            prior_template: vec![(2, pass), (3, pass)],
            varies_by_suit: false,
            partner_bid: false,
            opp_bid: false,
        },
        Scenario {
            name: "pos4_after_3p",
            dealer: 0,
            seat: 0,
            position: 4,
            // seat1 (opp) passes, seat2 (partner) passes, seat3 (opp) passes
            prior_template: vec![(1, pass), (2, pass), (3, pass)],
            varies_by_suit: false,
            partner_bid: false,
            opp_bid: false,
        },
        // ===== Partner bid 80 scenarios =====
        Scenario {
            name: "pos3_partner80",
            dealer: 1,
            seat: 0,
            position: 3,
            // seat2 (partner) bids 80, seat3 (opp) passes
            prior_template: vec![(2, bid80s), (3, pass)],
            varies_by_suit: true,
            partner_bid: true,
            opp_bid: false,
        },
        Scenario {
            name: "pos4_partner80",
            dealer: 0,
            seat: 0,
            position: 4,
            // seat1 (opp) passes, seat2 (partner) bids 80, seat3 (opp) passes
            prior_template: vec![(1, pass), (2, bid80s), (3, pass)],
            varies_by_suit: true,
            partner_bid: true,
            opp_bid: false,
        },
        // ===== Opponent bid 80 scenarios =====
        Scenario {
            name: "pos2_opp80",
            dealer: 2,
            seat: 0,
            position: 2,
            // seat3 (opp) bids 80
            prior_template: vec![(3, bid80s)],
            varies_by_suit: true,
            partner_bid: false,
            opp_bid: true,
        },
        Scenario {
            name: "pos3_opp80",
            dealer: 1,
            seat: 0,
            position: 3,
            // seat2 (partner) passes, seat3 (opp) bids 80
            prior_template: vec![(2, pass), (3, bid80s)],
            varies_by_suit: true,
            partner_bid: false,
            opp_bid: true,
        },
        Scenario {
            name: "pos4_opp80",
            dealer: 0,
            seat: 0,
            position: 4,
            // seat1 (opp) passes, seat2 (partner) passes, seat3 (opp) bids 80
            prior_template: vec![(1, pass), (2, pass), (3, bid80s)],
            varies_by_suit: true,
            partner_bid: false,
            opp_bid: true,
        },
    ]
}

/// Compute hand features for a given suit and write a CSV row.
fn write_row(
    file: &mut impl Write,
    deal_id: usize,
    scenario: &str,
    position: u8,
    hand: CardSet,
    suit_idx: u8,
    qvals: &[(u8, f32)],
    best_action: u8,
    nn_suit: i8,
    nn_value: u16,
    q_pass: f32,
    q_coinche: f32,
    // Context features
    partner_bid_suit: i8,
    opp_bid_suit: i8,
    opp_bid_value: u16,
) {
    let suit = Suit::from_u8(suit_idx);
    let bits = suit_bits(hand, suit);
    let count = bits.count_ones();

    // Trump card features
    let has_jack = (bits >> 3) & 1 == 1;
    let has_nine = (bits >> 2) & 1 == 1;
    let has_ace = (bits >> 7) & 1 == 1;
    let has_ten = (bits >> 6) & 1 == 1;
    let has_king = (bits >> 5) & 1 == 1;
    let has_queen = (bits >> 4) & 1 == 1;
    let has_belote = has_king && has_queen;

    // Trump points
    let mut trump_pts = 0u16;
    let mut b = bits;
    while b != 0 {
        let rank = b.trailing_zeros() as usize;
        trump_pts += TRUMP_POINTS[rank] as u16;
        b &= b - 1;
    }

    let trump_score = evaluate_for_trump(hand, suit);

    // Side suit features
    let mut side_aces = 0u32;
    let mut side_tens = 0u32;
    let mut side_voids = 0u32;
    let mut side_singletons = 0u32;
    let mut side_doubletons = 0u32;
    let mut best_side_length = 0u32;
    for s in 0..4u8 {
        if s == suit_idx {
            continue;
        }
        let sb = suit_bits(hand, Suit::from_u8(s));
        let sc = sb.count_ones();
        if sb & (1 << 7) != 0 {
            side_aces += 1;
        }
        if sb & (1 << 6) != 0 {
            side_tens += 1;
        }
        if sc == 0 {
            side_voids += 1;
        } else if sc == 1 {
            side_singletons += 1;
        } else if sc == 2 {
            side_doubletons += 1;
        }
        if sc > best_side_length {
            best_side_length = sc;
        }
    }

    let total_aces = (0..4u8)
        .filter(|&s| suit_bits(hand, Suit::from_u8(s)) & (1 << 7) != 0)
        .count() as u32;

    // Contextual features: support for partner's suit
    let partner_support = if partner_bid_suit >= 0 && partner_bid_suit < 4 {
        let ps = partner_bid_suit as u8;
        let partner_bits = suit_bits(hand, Suit::from_u8(ps));
        partner_bits.count_ones() as i32
    } else {
        -1 // N/A
    };

    // How many cards do I have in opponent's bid suit?
    let opp_suit_cards = if opp_bid_suit >= 0 && opp_bid_suit < 4 {
        let os = opp_bid_suit as u8;
        suit_bits(hand, Suit::from_u8(os)).count_ones() as i32
    } else {
        -1
    };

    // Is this suit the same as partner's bid suit?
    let is_partner_suit = if partner_bid_suit >= 0 {
        (suit_idx == partner_bid_suit as u8) as u8
    } else {
        0
    };

    // Is this suit the same as opponent's bid suit?
    let is_opp_suit = if opp_bid_suit >= 0 {
        (suit_idx == opp_bid_suit as u8) as u8
    } else {
        0
    };

    // Q-values for this suit at various levels
    let q_80 = q_for(qvals, bidding::encode_bid(8, suit_idx));
    let q_90 = q_for(qvals, bidding::encode_bid(9, suit_idx));
    let q_100 = q_for(qvals, bidding::encode_bid(10, suit_idx));
    let q_110 = q_for(qvals, bidding::encode_bid(11, suit_idx));
    let q_120 = q_for(qvals, bidding::encode_bid(12, suit_idx));
    let q_capot = q_for(qvals, bidding::encode_bid(25, suit_idx));

    // Does the NN bid this suit?
    let nn_bids_this = if nn_suit == suit_idx as i8 { 1 } else { 0 };

    writeln!(
        file,
        "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
        deal_id,
        scenario,
        position,
        suit_idx,
        count,
        has_jack as u8,
        has_nine as u8,
        has_ace as u8,
        has_ten as u8,
        has_king as u8,
        has_queen as u8,
        trump_pts,
        trump_score,
        has_belote as u8,
        side_aces,
        side_tens,
        side_voids,
        side_singletons,
        side_doubletons,
        total_aces,
        best_side_length,
        partner_support,
        opp_suit_cards,
        is_partner_suit,
        is_opp_suit,
        format!("{:.6}", q_80),
        format!("{:.6}", q_90),
        format!("{:.6}", q_100),
        format!("{:.6}", q_110),
        format!("{:.6}", q_120),
        format!("{:.6}", q_capot),
        format!("{:.6}", q_pass),
        format!("{:.6}", q_coinche),
        best_action,
        nn_bids_this
    )
    .unwrap();
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let model_path = args
        .get(1)
        .map(|s| s.as_str())
        .unwrap_or("models/bid_v2/bid_nn_final.bin");
    let n_deals: usize = args
        .get(2)
        .and_then(|s| s.parse().ok())
        .unwrap_or(200_000);
    let output_path = args
        .get(3)
        .map(|s| s.as_str())
        .unwrap_or("data/distill/bid_distill.csv");
    let my_score: i32 = args.get(4).and_then(|s| s.parse().ok()).unwrap_or(0);
    let opp_score: i32 = args.get(5).and_then(|s| s.parse().ok()).unwrap_or(0);

    eprintln!("Loading model: {}", model_path);
    let mut net = BidNet::load_with_hidden(model_path, 512)
        .unwrap_or_else(|e| panic!("Failed to load {}: {}", model_path, e));
    let obs_dim = net.obs_dim();
    eprintln!(
        "Model loaded: obs={}, hidden={}, layers={}, dueling={}",
        obs_dim,
        net.hidden(),
        net.layers(),
        net.is_dueling()
    );
    match obs_dim {
        bid_obs::BID_OBS_DIM => eprintln!("  → legacy 108-dim obs (no score features)"),
        bid_obs::BID_OBS_DIM_SCORE_AWARE => eprintln!(
            "  → score-aware v1 (110-dim) — my_score={}, opp_score={}",
            my_score, opp_score
        ),
        bid_obs::BID_OBS_DIM_SCORE_AWARE_V2 => eprintln!(
            "  → score-aware v2 (113-dim) — my_score={}, opp_score={}",
            my_score, opp_score
        ),
        other => panic!(
            "Unsupported obs_dim={}, expected 108/110/113",
            other
        ),
    }

    if let Some(parent) = std::path::Path::new(output_path).parent() {
        std::fs::create_dir_all(parent).ok();
    }

    let mut file = std::io::BufWriter::new(
        std::fs::File::create(output_path)
            .unwrap_or_else(|e| panic!("Cannot create {}: {}", output_path, e)),
    );

    // CSV header
    writeln!(
        file,
        "deal_id,scenario,position,suit,\
trump_count,has_jack,has_nine,has_ace,has_ten,has_king,has_queen,\
trump_points,trump_score,has_belote,\
side_aces,side_tens,side_voids,side_singletons,side_doubletons,\
total_aces,best_side_length,\
partner_support,opp_suit_cards,is_partner_suit,is_opp_suit,\
q_80,q_90,q_100,q_110,q_120,q_capot,q_pass,q_coinche,\
nn_action,nn_bids_this_suit"
    )
    .unwrap();

    let all_scenarios = scenarios();
    let mut rng = StdRng::seed_from_u64(42);
    let mut row_count = 0usize;

    for scenario in &all_scenarios {
        eprintln!("\n=== Scenario: {} (position {}) ===", scenario.name, scenario.position);

        let suit_variants: Vec<u8> = if scenario.varies_by_suit {
            vec![0, 1, 2, 3]
        } else {
            vec![255] // sentinel: no suit variation
        };

        for &prior_suit in &suit_variants {
            let variant_label = if prior_suit < 4 {
                format!("{}_{}", scenario.name, ["s", "h", "d", "c"][prior_suit as usize])
            } else {
                scenario.name.to_string()
            };

            let deals_per_variant = if scenario.varies_by_suit {
                n_deals / 4
            } else {
                n_deals
            };

            let mut progress_next = deals_per_variant / 10;

            for deal_idx in 0..deals_per_variant {
                if deal_idx >= progress_next {
                    eprintln!(
                        "  {}: {}/{} ({:.0}%)",
                        variant_label,
                        deal_idx,
                        deals_per_variant,
                        deal_idx as f64 / deals_per_variant as f64 * 100.0
                    );
                    progress_next += deals_per_variant / 10;
                }

                let deal_id = row_count / 4; // unique per (scenario, deal)

                // Deal random hands with the right dealer
                let state_init = GameState::deal_random(scenario.dealer, &mut rng);
                let hand = state_init.hands[scenario.seat as usize];

                // Build history, substituting bid suit if needed
                let mut history: Vec<(u8, u8)> = Vec::new();
                let mut state = state_init;

                for &(seat, action_template) in &scenario.prior_template {
                    let action = if scenario.varies_by_suit && action_template > 0 && action_template <= 40 {
                        // Remap the placeholder bid to the correct suit
                        let (val, _) = bidding::decode_bid(action_template);
                        bidding::encode_bid(val, prior_suit)
                    } else {
                        action_template
                    };
                    history.push((seat, action));
                    state.step(action);
                }

                assert_eq!(
                    state.current_player(),
                    scenario.seat,
                    "Scenario {} seat mismatch: expected {}, got {}",
                    scenario.name,
                    scenario.seat,
                    state.current_player()
                );

                // Get NN decision — build obs matching the model's input dim
                let obs: Vec<f32> = match obs_dim {
                    bid_obs::BID_OBS_DIM => bid_obs::make_bid_observation(&state, &history),
                    bid_obs::BID_OBS_DIM_SCORE_AWARE => {
                        let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE];
                        bid_obs::write_bid_observation_score_aware(
                            &mut buf, 0, &state, &history, my_score, opp_score,
                        );
                        buf
                    }
                    bid_obs::BID_OBS_DIM_SCORE_AWARE_V2 => {
                        let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE_V2];
                        bid_obs::write_bid_observation_score_aware_v2(
                            &mut buf, 0, &state, &history, my_score, opp_score,
                        );
                        buf
                    }
                    other => unreachable!("obs_dim {} validated at startup", other),
                };
                let legal = state.legal_actions();
                let (best_action, qvals) = net.best_action(&obs, legal);

                // Decode NN decision
                let (nn_suit, nn_value) = if best_action >= 1 && best_action <= 40 {
                    let (val, suit) = bidding::decode_bid(best_action);
                    (suit as i8, val as u16 * 10)
                } else if best_action == 0 {
                    (-1i8, 0u16)
                } else if best_action == 41 {
                    (-3i8, 0u16) // COINCHE
                } else {
                    (-4i8, 0u16) // SURCOINCHE
                };

                let q_pass = q_for(&qvals, 0);
                let q_coinche = q_for(&qvals, 41);

                let partner_bid_suit = if scenario.partner_bid && prior_suit < 4 {
                    prior_suit as i8
                } else {
                    -1i8
                };
                let opp_bid_suit = if scenario.opp_bid && prior_suit < 4 {
                    prior_suit as i8
                } else {
                    -1i8
                };
                let opp_bid_value = if scenario.opp_bid { 80u16 } else { 0u16 };

                for suit_idx in 0..4u8 {
                    write_row(
                        &mut file,
                        deal_id,
                        &variant_label,
                        scenario.position,
                        hand,
                        suit_idx,
                        &qvals,
                        best_action,
                        nn_suit,
                        nn_value,
                        q_pass,
                        q_coinche,
                        partner_bid_suit,
                        opp_bid_suit,
                        opp_bid_value,
                    );
                    row_count += 1;
                }
            }
        }
    }

    eprintln!("\nDone! Wrote {} rows to {}", row_count, output_path);
}

fn q_for(qvals: &[(u8, f32)], action: u8) -> f32 {
    qvals
        .iter()
        .find(|(a, _)| *a == action)
        .map(|(_, q)| *q)
        .unwrap_or(f32::NEG_INFINITY)
}
