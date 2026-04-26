/// Offline regret audit for the v5 bid NN on held-out pool deals.
///
/// For each held-out deal:
/// 1. Play a full deterministic auction with the bid NN at all 4 seats (greedy, ε=0).
/// 2. Record the chosen contract (trump, value, declarer, coinche).
/// 3. Use the per-suit ISDD ground-truth points (real_pts[4]) to compute:
///    - actual deal score under the chosen contract
///    - best feasible contract for NS and for EW (assuming no overbid/contre)
///    - regret = best_for_declarer_team - actual_for_declarer_team
///
/// Emits a CSV that Python aggregates by archetype (position, suit, value bucket,
/// hand strength) to surface where the model leaves the most points on the table.
///
/// Usage:
///   cargo run -p colver-core --bin audit_bid_v5 --release -- \
///     --model models/bid_v5_isdd/bid_nn_final.bin \
///     --pool data/deals/base_5M.bin \
///     --scores data/deals/scores_isdd_5M.sc \
///     --offset 4500000 --count 100000 \
///     --output data/audit/audit_v5_100k.csv

use std::io::Write;
use std::time::Instant;

use colver_core::bid_eval::evaluate_for_trump;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bid_train_env::DealPool;
use colver_core::card::{suit_bits, Suit};
use colver_core::scoring::compute_deal_score;
use colver_core::state::{Contract, GameState, Phase};

struct Args {
    model: String,
    hidden: usize,
    pool: String,
    scores: String,
    score_layer: String,
    offset: usize,
    count: usize,
    output: String,
    match_scores: String,
}

impl Args {
    fn parse() -> Self {
        let mut a = Args {
            model: "models/bid_v5_isdd/bid_nn_final.bin".to_string(),
            hidden: 512,
            pool: "data/deals/base_5M.bin".to_string(),
            scores: "data/deals/scores_isdd_5M.sc".to_string(),
            score_layer: "isdd".to_string(),
            offset: 4_500_000,
            count: 100_000,
            output: "data/audit/audit_v5.csv".to_string(),
            match_scores: "0,0".to_string(),
        };
        let argv: Vec<String> = std::env::args().collect();
        let mut i = 1;
        while i < argv.len() {
            let flag = &argv[i];
            let need = |i: usize| {
                if i + 1 >= argv.len() {
                    panic!("flag {} needs a value", argv[i]);
                }
                argv[i + 1].clone()
            };
            match flag.as_str() {
                "--model" => { a.model = need(i); i += 2; }
                "--hidden" => { a.hidden = need(i).parse().unwrap(); i += 2; }
                "--pool" => { a.pool = need(i); i += 2; }
                "--scores" => { a.scores = need(i); i += 2; }
                "--score-layer" => { a.score_layer = need(i); i += 2; }
                "--offset" => { a.offset = need(i).parse().unwrap(); i += 2; }
                "--count" => { a.count = need(i).parse().unwrap(); i += 2; }
                "--output" => { a.output = need(i); i += 2; }
                "--match-scores" => { a.match_scores = need(i); i += 2; }
                "--help" | "-h" => {
                    print_help();
                    std::process::exit(0);
                }
                _ => panic!("unknown flag: {}", flag),
            }
        }
        a
    }
}

fn print_help() {
    eprintln!("audit_bid_v5 — offline regret audit for bid NN");
    eprintln!("  --model PATH            bid NN .bin (default models/bid_v5_isdd/bid_nn_final.bin)");
    eprintln!("  --hidden N              hidden size (default 512)");
    eprintln!("  --pool PATH             base pool (default data/deals/base_5M.bin)");
    eprintln!("  --scores PATH           score layer (default data/deals/scores_isdd_5M.sc)");
    eprintln!("  --score-layer NAME      layer name (default 'isdd')");
    eprintln!("  --offset N              held-out start (default 4500000)");
    eprintln!("  --count N               number of deals (default 100000)");
    eprintln!("  --output PATH           CSV output (default data/audit/audit_v5.csv)");
    eprintln!("  --match-scores 'ns,ew;ns,ew...'  score probes (default '0,0')");
}

/// Detect belote (Q+K of trump in same hand) from raw deal hands.
/// Returns [belote_ns, belote_ew] where each is 0 or 2 (full belote bonus).
fn detect_belote(hands: &[u32; 4], trump: u8) -> [u8; 2] {
    let qk = (1u32 << (trump as u32 * 8 + 4)) | (1u32 << (trump as u32 * 8 + 5));
    let mut b = [0u8; 2];
    for p in 0..4u8 {
        if (hands[p as usize] & qk) == qk {
            b[(p & 1) as usize] = 2;
        }
    }
    b
}

/// Build a synthetic terminal state for a hypothetical contract and compute the
/// deal score. Mirrors the synthesis in bid_train_env::compute_scores.
fn score_contract(
    trump: u8,
    value: u8,
    team: u8,
    coinche: u8,
    ns_pts: u8,
    belote: [u8; 2],
) -> [i16; 2] {
    let (ns_final, ew_final) = if ns_pts == 252 {
        (252u8, 0u8)
    } else if ns_pts == 0 {
        (0u8, 252u8)
    } else {
        (ns_pts, 162u8.saturating_sub(ns_pts))
    };

    let taker = team as usize;
    let defense = 1 - taker;
    let taker_pts = if taker == 0 { ns_final } else { ew_final };
    let defense_pts = if defense == 0 { ns_final } else { ew_final };

    let (taker_tricks, defense_tricks) = if defense_pts == 0 {
        (8u8, 0u8)
    } else if taker_pts == 0 {
        (0u8, 8u8)
    } else {
        let total = taker_pts as u16 + defense_pts as u16;
        let frac = taker_pts as f32 / total as f32;
        let t = (frac * 8.0).round().clamp(1.0, 7.0) as u8;
        (t, 8 - t)
    };

    let mut terminal = GameState::new(0, [0; 4]);
    terminal.phase = Phase::Done;
    terminal.contract = Contract { trump, value, team, coinche };
    terminal.points[taker] = taker_pts;
    terminal.points[defense] = defense_pts;
    terminal.tricks_won[taker] = taker_tricks;
    terminal.tricks_won[defense] = defense_tricks;
    terminal.belote = belote;

    let s = compute_deal_score(&terminal);
    s.scores
}

fn max_feasible_value(real_pts_ns: u8, team: u8) -> Option<u8> {
    let our_pts = if team == 0 {
        real_pts_ns as u16
    } else if real_pts_ns == 252 {
        0u16
    } else if real_pts_ns == 0 {
        252u16
    } else {
        162u16 - real_pts_ns as u16
    };

    let our_has_capot = (team == 0 && real_pts_ns == 252) || (team == 1 && real_pts_ns == 0);
    if our_has_capot {
        return Some(25);
    }

    for &v in [16u8, 15, 14, 13, 12, 11, 10, 9, 8].iter() {
        if our_pts >= v as u16 * 10 {
            return Some(v);
        }
    }
    None
}

fn best_contract_for_team(
    real_pts: [u8; 4],
    team: u8,
    hands: &[u32; 4],
) -> Option<(u8, u8, [i16; 2])> {
    let mut best: Option<(u8, u8, [i16; 2])> = None;
    for suit in 0..4u8 {
        // Belote must be recomputed per trump (Q+K of that suit).
        let belote = detect_belote(hands, suit);
        // Feasibility check must now include belote: taker_total (trick + belote) ≥ value.
        let real_ns = real_pts[suit as usize];
        let taker_trick = if team == 0 {
            real_ns as i16
        } else if real_ns == 252 { 0 } else if real_ns == 0 { 252 } else { 162 - real_ns as i16 };
        let taker_total = taker_trick + belote[team as usize] as i16;
        let feasible_value = if taker_total >= 250 && ((team == 0 && real_ns == 252) || (team == 1 && real_ns == 0)) {
            Some(25u8)
        } else {
            let mut found = None;
            for &v in [16u8, 15, 14, 13, 12, 11, 10, 9, 8].iter() {
                if taker_total >= v as i16 * 10 {
                    found = Some(v);
                    break;
                }
            }
            found
        };
        if let Some(v) = feasible_value {
            let scores = score_contract(suit, v, team, 0, real_ns, belote);
            let our = scores[team as usize];
            let cur_best = best.map(|(_, _, s)| s[team as usize]).unwrap_or(i16::MIN);
            if our > cur_best {
                best = Some((suit, v, scores));
            }
        }
    }
    best
}

fn parse_match_scores(s: &str) -> Vec<(i32, i32)> {
    s.split(';')
        .filter_map(|part| {
            let mut it = part.split(',');
            let ns = it.next()?.trim().parse::<i32>().ok()?;
            let ew = it.next()?.trim().parse::<i32>().ok()?;
            Some((ns, ew))
        })
        .collect()
}

fn bid_points(v: u8) -> u16 {
    if v == 25 { 250 } else { v as u16 * 10 }
}

fn main() {
    let args = Args::parse();

    // ---- Load pool + scores ----
    eprintln!("Loading pool: {}", args.pool);
    let mut pool = DealPool::load(&args.pool).expect("load pool");
    eprintln!("  {} deals", pool.len());
    eprintln!("Loading score layer: {}", args.scores);
    pool.load_scores(&args.scores).expect("load scores");
    pool.select_score_layer(Some(&args.score_layer));

    // ---- Load bid NN ----
    eprintln!("Loading bid NN: {} (hidden={})", args.model, args.hidden);
    let mut net = BidNet::load_with_hidden(&args.model, args.hidden).expect("load net");
    let obs_dim = net.obs_dim();
    eprintln!("  obs_dim={} (108/110/113/117 for v0/v1/v2/v3 score-aware)", obs_dim);
    assert!(
        obs_dim == 108 || obs_dim == 110 || obs_dim == 113 || obs_dim == 117,
        "unexpected obs dim {obs_dim}"
    );

    // ---- Held-out slice ----
    let end = (args.offset + args.count).min(pool.len());
    let actual = end.saturating_sub(args.offset);
    eprintln!(
        "Auditing deals [{}, {}) ({} deals)",
        args.offset, end, actual
    );

    // ---- Match-score probes ----
    let probes = parse_match_scores(&args.match_scores);
    assert!(!probes.is_empty(), "no match-score probes parsed");
    eprintln!("Match-score probes: {:?}", probes);

    // ---- Output ----
    if let Some(parent) = std::path::Path::new(&args.output).parent() {
        let _ = std::fs::create_dir_all(parent);
    }
    let mut out = std::io::BufWriter::new(
        std::fs::File::create(&args.output).expect("create output"),
    );
    writeln!(
        out,
        "deal_idx,probe_ns,probe_ew,dealer,trump,value,declarer,decl_team,decl_pos,coinche,passed_out,made,\
         actual_decl_pts,actual_opp_pts,\
         ns_best_suit,ns_best_val,ns_best_pts,\
         ew_best_suit,ew_best_val,ew_best_pts,\
         best_team_pts_decl,regret,\
         decl_hand_strength,decl_trump_count,decl_jack,decl_nine,\
         rp_s,rp_h,rp_d,rp_c"
    )
    .unwrap();

    let start = Instant::now();
    let mut obs_buf = vec![0.0f32; obs_dim];

    let mut n_done = 0usize;
    let mut n_passout = 0usize;
    let mut n_made = 0usize;
    let mut n_chute = 0usize;
    let mut sum_regret: f64 = 0.0;

    for deal_idx in args.offset..end {
        let deal = pool.get(deal_idx);
        let real_pts = deal.real_pts.expect("score layer must cover held-out slice");

        for &(probe_ns, probe_ew) in &probes {
            let mut state = GameState::new(deal.dealer, deal.hands);
            let mut bid_history: Vec<(u8, u8)> = Vec::with_capacity(16);

            while state.phase == Phase::Bidding {
                let me = state.current_player;
                let (my_s, opp_s) = if GameState::player_team(me) == 0 {
                    (probe_ns, probe_ew)
                } else {
                    (probe_ew, probe_ns)
                };
                match obs_dim {
                    108 => bid_obs::write_bid_observation(&mut obs_buf, 0, &state, &bid_history),
                    110 => bid_obs::write_bid_observation_score_aware(
                        &mut obs_buf, 0, &state, &bid_history, my_s, opp_s,
                    ),
                    113 => bid_obs::write_bid_observation_score_aware_v2(
                        &mut obs_buf, 0, &state, &bid_history, my_s, opp_s,
                    ),
                    117 => bid_obs::write_bid_observation_score_aware_v3(
                        &mut obs_buf, 0, &state, &bid_history, my_s, opp_s,
                    ),
                    _ => unreachable!(),
                }
                let legal = state.legal_actions();
                let (action, _) = net.best_action(&obs_buf, legal);
                bid_history.push((me, action));
                state.step(action);
            }

            let passed_out = state.contract.value == 0;
            let trump = state.contract.trump;
            let value = state.contract.value;
            let decl_team = state.contract.team;
            let coinche = state.contract.coinche;
            let declarer = state.last_bidder;
            // Position 1..4 counting from the first bidder (dealer+1).
            let decl_pos = if passed_out {
                0u8
            } else {
                ((declarer + 4 - deal.dealer - 1) % 4) + 1
            };

            let belote_actual = if passed_out {
                [0u8; 2]
            } else {
                detect_belote(&deal.hands, trump)
            };
            let (actual_ns, actual_ew) = if passed_out {
                (0i16, 0i16)
            } else {
                let s = score_contract(trump, value, decl_team, coinche, real_pts[trump as usize], belote_actual);
                (s[0], s[1])
            };
            let actual_decl = if decl_team == 0 { actual_ns } else { actual_ew };
            let actual_opp = if decl_team == 0 { actual_ew } else { actual_ns };

            let made = if passed_out {
                false
            } else if value == 25 {
                // Capot: declarer must have all trick points (no belote help).
                if decl_team == 0 { real_pts[trump as usize] == 252 } else { real_pts[trump as usize] == 0 }
            } else {
                let pv = bid_points(value) as i16;
                let decl_trick_pts = if decl_team == 0 {
                    real_pts[trump as usize] as i16
                } else if real_pts[trump as usize] == 252 {
                    0i16
                } else if real_pts[trump as usize] == 0 {
                    252i16
                } else {
                    162 - real_pts[trump as usize] as i16
                };
                // Belote for declarer team counts toward the threshold.
                (decl_trick_pts + belote_actual[decl_team as usize] as i16) >= pv
            };

            let ns_best = best_contract_for_team(real_pts, 0, &deal.hands);
            let ew_best = best_contract_for_team(real_pts, 1, &deal.hands);
            let (ns_bs, ns_bv, ns_bp) = match ns_best {
                Some((s, v, sc)) => (s as i32, v as i32, sc[0]),
                None => (-1, -1, 0),
            };
            let (ew_bs, ew_bv, ew_bp) = match ew_best {
                Some((s, v, sc)) => (s as i32, v as i32, sc[1]),
                None => (-1, -1, 0),
            };

            let best_team_pts_decl: i16 = if passed_out {
                ns_bp.max(ew_bp)
            } else if decl_team == 0 {
                ns_bp
            } else {
                ew_bp
            };
            let regret = (best_team_pts_decl - actual_decl).max(0);

            let (decl_hand_strength, decl_trump_count, decl_jack, decl_nine) = if passed_out {
                (0u16, 0u32, 0u8, 0u8)
            } else {
                let hand = deal.hands[declarer as usize];
                let t = Suit::from_u8(trump);
                let tb = suit_bits(hand, t) as u32;
                let hs = evaluate_for_trump(hand, t);
                let jack = if tb & (1 << 3) != 0 { 1 } else { 0 };
                let nine = if tb & (1 << 2) != 0 { 1 } else { 0 };
                (hs, tb.count_ones(), jack, nine)
            };

            if passed_out {
                n_passout += 1;
            } else if made {
                n_made += 1;
            } else {
                n_chute += 1;
            }
            sum_regret += regret as f64;

            writeln!(
                out,
                "{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}",
                deal_idx, probe_ns, probe_ew,
                deal.dealer, trump, value, declarer, decl_team, decl_pos, coinche,
                passed_out as u8, made as u8,
                actual_decl, actual_opp,
                ns_bs, ns_bv, ns_bp,
                ew_bs, ew_bv, ew_bp,
                best_team_pts_decl, regret,
                decl_hand_strength, decl_trump_count, decl_jack, decl_nine,
                real_pts[0], real_pts[1], real_pts[2], real_pts[3],
            )
            .unwrap();
        }

        n_done += 1;
        if n_done % 5000 == 0 {
            let el = start.elapsed().as_secs_f64();
            let rate = n_done as f64 / el;
            let eta = (actual - n_done) as f64 / rate;
            let total_rows = (n_done * probes.len()) as f64;
            eprintln!(
                "  {}/{} deals ({:.0}/s) — passout {}, made {}, chute {}, avg_regret {:.1} — ETA {:.0}s",
                n_done, actual, rate,
                n_passout, n_made, n_chute,
                sum_regret / total_rows, eta,
            );
        }
    }

    let el = start.elapsed().as_secs_f64();
    let total_rows = (n_done * probes.len()) as f64;
    eprintln!("\n=== Audit done in {:.1}s ===", el);
    eprintln!("Rows written: {} ({} deals × {} probes)", n_done * probes.len(), n_done, probes.len());
    eprintln!("  pass-out: {} ({:.1}%)", n_passout, n_passout as f64 * 100.0 / total_rows);
    eprintln!("  made:     {} ({:.1}%)", n_made, n_made as f64 * 100.0 / total_rows);
    eprintln!("  chute:    {} ({:.1}%)", n_chute, n_chute as f64 * 100.0 / total_rows);
    eprintln!("  avg regret (decl team): {:.1} pts", sum_regret / total_rows);
    eprintln!("Saved CSV to {}", args.output);
}
