/// Belief quality evaluation binary.
///
/// Plays many deals with NN bots, tracking beliefs vs ground truth at every
/// bid and play decision point. Compares up to four belief systems:
/// - **CB** (CardBeliefs): heuristic play inference
/// - **BS** (BeliefState): soft bid weights + play inference
/// - **BS+NN** (BeliefState with NN bid weights): NN bid beliefs + heuristic play inference
/// - **NN** (BeliefNet): neural network card location prediction (if model provided)
///
/// Usage:
///   cargo run --bin eval_beliefs --features "parallel,nn" --release -- [--deals 500] [--seed 42]
///   cargo run --bin eval_beliefs --features "parallel,nn" --release -- --deals 500 --nn models/belief_v3.bin
///   cargo run --bin eval_beliefs --features "parallel,nn" --release -- --deals 500 --bid-belief models/bid_belief.bin

use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::belief_net::{self, BeliefNet};
use colver_core::belief_obs;
use colver_core::belief_state::BeliefState;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::card::*;
use colver_core::card_beliefs::CardBeliefs;
use colver_core::determinize::determinize_weighted;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking};
use colver_core::state::{GameState, Phase};

// ── Bucket ──────────────────────────────────────────────────────────────

#[derive(Default, Clone, Copy)]
struct Bucket {
    cb_log_sum: f64,
    bs_log_sum: f64,
    bs_nn_log_sum: f64,
    nn_log_sum: f64,
    cb_excl: u64,
    nn_excl: u64,
    hidden: u64,
    decisions: u64,
    cb_entropy_sum: f64,
    bs_entropy_sum: f64,
    bs_nn_entropy_sum: f64,
    nn_entropy_sum: f64,
    det_correct: u64,
    det_total: u64,
    gt_reachable: u64,
    gt_checked: u64,
    elig_no_constraint: f64,
    elig_hard_only: f64,
    elig_soft: f64,
    elig_cards: u64,
}

impl Bucket {
    fn log(sum: f64, n: u64) -> f64 { if n > 0 { sum / n as f64 } else { 0.0 } }
    fn pct(num: u64, den: u64) -> f64 { if den > 0 { 100.0 * num as f64 / den as f64 } else { 0.0 } }
    fn avg(sum: f64, n: u64) -> f64 { if n > 0 { sum / n as f64 } else { 0.0 } }
}

// ── All metrics ─────────────────────────────────────────────────────────

struct Metrics {
    bid_step: [Bucket; 13],
    trick: [Bucket; 8],
    trick_pos: [Bucket; 4],
    play_total: Bucket,
    bid_total: Bucket,
    has_nn: bool,
    has_bid_belief: bool,
}

impl Metrics {
    fn new(has_nn: bool, has_bid_belief: bool) -> Self {
        Metrics {
            bid_step: [Bucket::default(); 13],
            trick: [Bucket::default(); 8],
            trick_pos: [Bucket::default(); 4],
            play_total: Bucket::default(),
            bid_total: Bucket::default(),
            has_nn,
            has_bid_belief,
        }
    }

    fn print(&self) {
        let uniform = (1.0f64 / 3.0).ln();
        let bsnn_col = if self.has_bid_belief { " BS+NN log  |" } else { "" };
        let nn_col = if self.has_nn { "  NN log(p) |" } else { "" };
        let bsnn_hdr = if self.has_bid_belief { "  BS+NN" } else { "" };
        let nn_hdr = if self.has_nn { "  NN" } else { "" };

        println!("\n══════════════════════════════════════════════════");
        println!("  BELIEF QUALITY EVALUATION");
        println!("  log(p): 0=perfect, {:.3}=uniform/random", uniform);
        println!("══════════════════════════════════════════════════");

        // ── Headlines ──
        let p = &self.play_total;
        let b = &self.bid_total;

        println!("\n  --- Play phase ({} decisions, {} hidden cards) ---", p.decisions, p.hidden);
        print!("  log(p):   CB {:.4}    BS {:.4}", Bucket::log(p.cb_log_sum, p.hidden), Bucket::log(p.bs_log_sum, p.hidden));
        if self.has_bid_belief { print!("    BS+NN {:.4}", Bucket::log(p.bs_nn_log_sum, p.hidden)); }
        if self.has_nn { print!("    NN {:.4}", Bucket::log(p.nn_log_sum, p.hidden)); }
        println!();
        print!("  false excl: CB {:.3}%", Bucket::pct(p.cb_excl, p.hidden));
        if self.has_nn { print!("    NN {:.3}%", Bucket::pct(p.nn_excl, p.hidden)); }
        println!();
        println!("  placement accuracy (CB det): {:.1}%    (uniform=33.3%)", Bucket::pct(p.det_correct, p.det_total));
        println!("  ground truth reachable: {:.1}%", Bucket::pct(p.gt_reachable, p.gt_checked));
        println!("  avg eligible/card: none={:.2}  hard={:.2}  soft={:.2}",
            Bucket::avg(p.elig_no_constraint, p.elig_cards),
            Bucket::avg(p.elig_hard_only, p.elig_cards),
            Bucket::avg(p.elig_soft, p.elig_cards));

        println!("\n  --- Bid phase ({} decisions, {} hidden cards) ---", b.decisions, b.hidden);
        print!("  log(p):   CB {:.4}    BS {:.4}", Bucket::log(b.cb_log_sum, b.hidden), Bucket::log(b.bs_log_sum, b.hidden));
        if self.has_nn { print!("    NN {:.4}", Bucket::log(b.nn_log_sum, b.hidden)); }
        println!();
        println!("  placement accuracy (CB det): {:.1}%", Bucket::pct(b.det_correct, b.det_total));

        // ── Per bid step ──
        println!("\n  ── Per Bid Step ──");
        println!("  Step | CB log(p) | BS log(p) |{} place%  | elig(soft) | Hidden", nn_col);
        println!("  -----|-----------|-----------|{}---------|------------|-------",
            if self.has_nn { "------------|" } else { "" });
        for s in 0..13 {
            let b = &self.bid_step[s];
            if b.hidden == 0 { continue; }
            print!("  {:>4} | {:>9.4} | {:>9.4} |",
                s, Bucket::log(b.cb_log_sum, b.hidden), Bucket::log(b.bs_log_sum, b.hidden));
            if self.has_nn { print!(" {:>9.4} |", Bucket::log(b.nn_log_sum, b.hidden)); }
            println!(" {:>6.1}% | {:>10.2} | {}",
                Bucket::pct(b.det_correct, b.det_total),
                Bucket::avg(b.elig_soft, b.elig_cards), b.hidden);
        }

        // ── Per trick ──
        println!("\n  ── Per Trick ──");
        println!("  Trick | CB log(p) | BS log(p) |{}{} place%  | GT ok%  | elig: none/hard/soft | excl%  | Hidden", bsnn_col, nn_col);
        println!("  ------|-----------|-----------|{}{}---------|---------|----------------------|--------|-------",
            if self.has_bid_belief { "------------|" } else { "" },
            if self.has_nn { "------------|" } else { "" });
        for t in 0..8 {
            let b = &self.trick[t];
            if b.hidden == 0 { continue; }
            print!("  {:>5} | {:>9.4} | {:>9.4} |",
                t, Bucket::log(b.cb_log_sum, b.hidden), Bucket::log(b.bs_log_sum, b.hidden));
            if self.has_bid_belief { print!(" {:>9.4} |", Bucket::log(b.bs_nn_log_sum, b.hidden)); }
            if self.has_nn { print!(" {:>9.4} |", Bucket::log(b.nn_log_sum, b.hidden)); }
            println!(" {:>6.1}% | {:>6.1}% | {:>4.2}/{:>4.2}/{:>4.2}        | {:>5.2}% | {}",
                Bucket::pct(b.det_correct, b.det_total),
                Bucket::pct(b.gt_reachable, b.gt_checked),
                Bucket::avg(b.elig_no_constraint, b.elig_cards),
                Bucket::avg(b.elig_hard_only, b.elig_cards),
                Bucket::avg(b.elig_soft, b.elig_cards),
                Bucket::pct(b.cb_excl, b.hidden), b.hidden);
        }

        // ── Per position ──
        println!("\n  ── Per Position in Trick ──");
        println!("  Pos  | CB log(p) |{}{} place%  | excl%  | Hidden", bsnn_hdr.trim(), nn_hdr.trim());
        println!("  -----|-----------|{}{}---------|--------|-------",
            if self.has_bid_belief { "------------|" } else { "" },
            if self.has_nn { "------------|" } else { "" });
        let labels = ["lead", "2nd ", "3rd ", "4th "];
        for p in 0..4 {
            let b = &self.trick_pos[p];
            if b.hidden == 0 { continue; }
            print!("  {}  | {:>9.4} |", labels[p], Bucket::log(b.cb_log_sum, b.hidden));
            if self.has_bid_belief { print!(" {:>9.4} |", Bucket::log(b.bs_nn_log_sum, b.hidden)); }
            if self.has_nn { print!(" {:>9.4} |", Bucket::log(b.nn_log_sum, b.hidden)); }
            println!(" {:>6.1}% | {:>5.2}% | {}",
                Bucket::pct(b.det_correct, b.det_total),
                Bucket::pct(b.cb_excl, b.hidden), b.hidden);
        }
    }
}

// ── Evaluate at one decision point ──────────────────────────────────────

fn evaluate_at_point(
    state: &GameState,
    cb: &CardBeliefs,
    bs: &BeliefState,
    bs_nn: Option<&BeliefState>,
    nn_weights: Option<&[[f32; 32]; 4]>,
    buckets: &mut [&mut Bucket],
    rng: &mut StdRng,
    sample_dets: bool,
) {
    let observer = state.current_player();
    let observer_hand = state.hands[observer as usize];

    let cb_w = cb.normalized_weights();
    let mut bs_w = bs.soft_weights;
    for card in 0..32 {
        let mut sum = 0.0f32;
        for p in 0..4 { sum += bs_w[p][card]; }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in 0..4 { bs_w[p][card] *= inv; }
        }
    }

    // Normalize BS+NN weights the same way
    let bs_nn_w = bs_nn.map(|bsn| {
        let mut w = bsn.soft_weights;
        for card in 0..32 {
            let mut sum = 0.0f32;
            for p in 0..4 { sum += w[p][card]; }
            if sum > 0.0 {
                let inv = 1.0 / sum;
                for p in 0..4 { w[p][card] *= inv; }
            }
        }
        w
    });

    for bucket in buckets.iter_mut() { bucket.decisions += 1; }

    // Ground truth reachable (CB hard constraints)
    let mut gt_ok = true;
    for card in 0..32u8 {
        let bit = card_to_bit(card);
        if observer_hand & bit != 0 || state.played_cards & bit != 0 { continue; }
        if (0..4).any(|i| state.current_trick[i] == card) { continue; }
        for p in 0..4u8 {
            if state.hands[p as usize] & bit != 0 {
                if cb_w[p as usize][card as usize] == 0.0 { gt_ok = false; }
                break;
            }
        }
    }
    for bucket in buckets.iter_mut() {
        bucket.gt_checked += 1;
        if gt_ok { bucket.gt_reachable += 1; }
    }

    // Per hidden card
    for card in 0..32u8 {
        let bit = card_to_bit(card);
        if observer_hand & bit != 0 || state.played_cards & bit != 0 { continue; }
        if (0..4).any(|i| state.current_trick[i] == card) { continue; }

        let mut true_p = 255u8;
        for p in 0..4u8 {
            if state.hands[p as usize] & bit != 0 { true_p = p; break; }
        }
        if true_p == 255 { continue; }

        let cb_p = (cb_w[true_p as usize][card as usize] as f64).max(1e-10);
        let bs_p = (bs_w[true_p as usize][card as usize] as f64).max(1e-10);
        let cb_lp = cb_p.ln();
        let bs_lp = bs_p.ln();
        let cb_is_excl = cb_w[true_p as usize][card as usize] == 0.0;

        let bs_nn_lp = if let Some(ref w) = bs_nn_w {
            let p_val = (w[true_p as usize][card as usize] as f64).max(1e-10);
            p_val.ln()
        } else {
            0.0
        };

        let nn_lp;
        let nn_is_excl;
        if let Some(nnw) = nn_weights {
            let nn_p = (nnw[true_p as usize][card as usize] as f64).max(1e-10);
            nn_lp = nn_p.ln();
            nn_is_excl = nnw[true_p as usize][card as usize] == 0.0;
        } else {
            nn_lp = 0.0;
            nn_is_excl = false;
        }

        let suit_idx = card_suit_u8(card);
        let mut cb_ent = 0.0f64;
        let mut bs_ent = 0.0f64;
        let mut bs_nn_ent = 0.0f64;
        let mut nn_ent = 0.0f64;
        let mut hard_eligible = 0u32;
        for p in 0..4u8 {
            if p == observer { continue; }
            if state.voids[p as usize] & (1 << suit_idx) == 0 { hard_eligible += 1; }
            let c = cb_w[p as usize][card as usize] as f64;
            if c > 1e-10 { cb_ent -= c * c.ln(); }
            let b = bs_w[p as usize][card as usize] as f64;
            if b > 1e-10 { bs_ent -= b * b.ln(); }
            if let Some(ref w) = bs_nn_w {
                let bn = w[p as usize][card as usize] as f64;
                if bn > 1e-10 { bs_nn_ent -= bn * bn.ln(); }
            }
            if let Some(nnw) = nn_weights {
                let n = nnw[p as usize][card as usize] as f64;
                if n > 1e-10 { nn_ent -= n * n.ln(); }
            }
        }

        for bucket in buckets.iter_mut() {
            bucket.hidden += 1;
            bucket.cb_log_sum += cb_lp;
            bucket.bs_log_sum += bs_lp;
            if bs_nn_w.is_some() { bucket.bs_nn_log_sum += bs_nn_lp; }
            if nn_weights.is_some() { bucket.nn_log_sum += nn_lp; }
            bucket.cb_entropy_sum += cb_ent;
            bucket.bs_entropy_sum += bs_ent;
            if bs_nn_w.is_some() { bucket.bs_nn_entropy_sum += bs_nn_ent; }
            bucket.nn_entropy_sum += nn_ent;
            if cb_is_excl { bucket.cb_excl += 1; }
            if nn_is_excl { bucket.nn_excl += 1; }
            bucket.elig_no_constraint += 3.0;
            bucket.elig_hard_only += hard_eligible as f64;
            bucket.elig_soft += cb_ent.exp();
            bucket.elig_cards += 1;
        }
    }

    // Placement accuracy (sampled)
    if sample_dets {
        for _ in 0..10u32 {
            let det = match determinize_weighted(state, observer, &cb_w, rng) {
                Some(d) => d,
                None => continue,
            };
            for card in 0..32u8 {
                let bit = card_to_bit(card);
                if observer_hand & bit != 0 || state.played_cards & bit != 0 { continue; }
                if (0..4).any(|i| state.current_trick[i] == card) { continue; }
                let mut true_p = 255u8;
                for p in 0..4u8 {
                    if state.hands[p as usize] & bit != 0 { true_p = p; break; }
                }
                if true_p == 255 { continue; }
                let ok = (0..4u8).any(|p| p == true_p && det.hands[p as usize] & bit != 0);
                for bucket in buckets.iter_mut() {
                    bucket.det_total += 1;
                    if ok { bucket.det_correct += 1; }
                }
            }
        }
    }
}

// ── NN belief helper ────────────────────────────────────────────────────

fn compute_nn_weights(
    belief_net: &mut Option<BeliefNet>,
    buf: &mut [f32],
    state: &GameState,
    tracking: &EnvTracking,
    observer: u8,
) -> Option<[[f32; 32]; 4]> {
    let net = belief_net.as_mut()?;
    let obs_d = net.obs_dim();
    if obs_d == belief_obs::BELIEF_OBS_DIM_V2 {
        let hc = [0.0f32; 96]; // no hard constraint mask for raw eval
        belief_obs::write_belief_observation_v2(buf, 0, state, tracking, observer, &hc);
    } else {
        // V1 (330)
        belief_obs::write_belief_observation(buf, 0, state, tracking, observer);
    }
    let logits = net.evaluate(buf);
    Some(belief_net::belief_to_weights(&logits, net.num_classes(), state, observer))
}

// ── Main ────────────────────────────────────────────────────────────────

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut num_deals = 500u32;
    let mut seed = 42u64;
    let mut nn_path: Option<String> = None;
    let mut bid_belief_path: Option<String> = None;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--deals" | "-n" => { i += 1; num_deals = args[i].parse().unwrap_or(500); }
            "--seed" | "-s" => { i += 1; seed = args[i].parse().unwrap_or(42); }
            "--nn" => { i += 1; nn_path = Some(args[i].clone()); }
            "--bid-belief" => { i += 1; bid_belief_path = Some(args[i].clone()); }
            "--help" | "-h" => {
                println!("Usage: eval_beliefs [--deals N] [--seed S] [--nn belief_model.bin] [--bid-belief bid_belief_model.bin]");
                return;
            }
            _ => { eprintln!("Unknown arg: {}", args[i]); }
        }
        i += 1;
    }

    println!("Loading models...");
    let mut bid_net = BidNet::load_with_hidden("models/bid_v2/bid_nn_final.bin", 512).expect("bid model");
    let mut play_net = DmcNet::load("models/play_v2/play_final.bin").expect("play model");
    play_net.set_residual(true);
    let obs_dim = play_net.obs_dim();
    let canonical = obs_dim == dmc_obs::OBS_DIM_TR;

    let mut belief_net: Option<BeliefNet> = None;
    if let Some(ref path) = nn_path {
        println!("Loading belief NN: {}", path);
        match BeliefNet::load(path) {
            Ok(net) => {
                println!("  obs_dim={}, classes={}", net.obs_dim(), net.num_classes());
                belief_net = Some(net);
            }
            Err(e) => eprintln!("  Failed: {}", e),
        }
    }

    let mut bid_belief_net: Option<BeliefNet> = bid_belief_path.as_ref().map(|path| {
        let net = BeliefNet::load_with_hidden(path, 256)
            .unwrap_or_else(|e| {
                eprintln!("Failed to load bid belief model: {}", e);
                std::process::exit(1);
            });
        println!("Bid belief model loaded (obs_dim={})", net.obs_dim());
        net
    });

    let has_nn = belief_net.is_some();
    let has_bid_belief = bid_belief_net.is_some();
    let mut rng = StdRng::seed_from_u64(seed);
    let mut m = Metrics::new(has_nn, has_bid_belief);
    let mut deals_played = 0u32;
    let mut void_deals = 0u32;
    let mut decision_counter = 0u64;
    let start = Instant::now();

    let mut bid_obs_buf = vec![0.0f32; bid_obs::BID_OBS_DIM];
    let mut obs_buf = vec![0.0f32; obs_dim];
    let belief_obs_dim = belief_net.as_ref().map(|n| n.obs_dim()).unwrap_or(0);
    let mut belief_obs_buf = vec![0.0f32; belief_obs_dim.max(1)];

    println!("Evaluating {} deals (seed={})...\n", num_deals, seed);

    for deal_idx in 0..num_deals {
        if deal_idx > 0 && deal_idx % 200 == 0 {
            let e = start.elapsed().as_secs_f64();
            println!("  [{}/{}] {:.1}s, {:.0} deals/s", deal_idx, num_deals, e, deal_idx as f64 / e);
        }

        let dealer = (deal_idx % 4) as u8;
        let mut state = GameState::deal_random(dealer, &mut rng);
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);

        let mut cb = [
            CardBeliefs::new(&state, 0), CardBeliefs::new(&state, 1),
            CardBeliefs::new(&state, 2), CardBeliefs::new(&state, 3),
        ];
        let mut bs = [
            BeliefState::new(0, state.hands[0]), BeliefState::new(1, state.hands[1]),
            BeliefState::new(2, state.hands[2]), BeliefState::new(3, state.hands[3]),
        ];

        // ── Bidding phase ──
        let mut bid_count = 0usize;
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;
            let step = bid_count.min(12);
            decision_counter += 1;
            let sample = decision_counter % 10 == 0;

            // NN belief weights (if available)
            let nn_w = compute_nn_weights(&mut belief_net, &mut belief_obs_buf, &state, &tracking, player);

            evaluate_at_point(
                &state, &cb[player as usize], &bs[player as usize],
                None, // BS+NN not yet initialized during bidding
                nn_w.as_ref(), &mut [&mut m.bid_step[step], &mut m.bid_total],
                &mut rng, sample,
            );

            bid_obs::write_bid_observation(&mut bid_obs_buf, 0, &state, &tracking.bid_history);
            let legal = state.legal_actions();
            let action = bid_net.best_action_fast(&bid_obs_buf, legal);

            for p in 0..4u8 {
                cb[p as usize].record_action(&state_before, player, action);
                bs[p as usize].record_bid(player, action, &state_before);
            }
            tracking.track_action(&state_before, action);
            state.step(action);
            bid_count += 1;
        }

        if state.is_terminal() { void_deals += 1; continue; }
        deals_played += 1;

        // ── Initialize BS+NN after bidding ──
        let mut bs_nn: Option<[BeliefState; 4]> = if bid_belief_net.is_some() {
            let mut arr = [
                BeliefState::new(0, state.hands[0]), BeliefState::new(1, state.hands[1]),
                BeliefState::new(2, state.hands[2]), BeliefState::new(3, state.hands[3]),
            ];
            if let Some(ref mut net) = bid_belief_net {
                for p in 0..4u8 {
                    arr[p as usize].apply_nn_bid_beliefs(net, &state, &tracking.bid_history);
                }
            }
            Some(arr)
        } else {
            None
        };

        // ── Playing phase ──
        while !state.is_terminal() {
            let player = state.current_player();
            let state_before = state;
            let trick_num = (state.tricks_won[0] + state.tricks_won[1]) as usize;
            let trick_idx = trick_num.min(7);
            let pos = state.trick_count as usize;
            decision_counter += 1;
            let sample = decision_counter % 10 == 0;

            // NN belief weights
            let nn_w = compute_nn_weights(&mut belief_net, &mut belief_obs_buf, &state, &tracking, player);

            evaluate_at_point(
                &state, &cb[player as usize], &bs[player as usize],
                bs_nn.as_ref().map(|arr| &arr[player as usize]),
                nn_w.as_ref(),
                &mut [&mut m.trick[trick_idx], &mut m.trick_pos[pos], &mut m.play_total],
                &mut rng, sample,
            );

            let action = if canonical {
                dmc_obs::write_observation_tr(&mut obs_buf, 0, &state, &tracking);
                let order = dmc_obs::current_player_order(&state, &tracking);
                let cm = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
                let (best, _) = play_net.best_action(&obs_buf, cm as u32);
                dmc_obs::card_to_physical(best, &order)
            } else {
                dmc_obs::write_observation(&mut obs_buf, 0, &state, &tracking);
                let (a, _) = play_net.best_action(&obs_buf, state.legal_actions() as u32);
                a
            };

            for p in 0..4u8 {
                cb[p as usize].record_action(&state_before, player, action);
                bs[p as usize].record_play(player, action as Card, &state_before);
            }
            if let Some(ref mut arr) = bs_nn {
                for p in 0..4u8 {
                    arr[p as usize].record_play(player, action as Card, &state_before);
                }
            }
            tracking.track_action(&state_before, action);
            state.step(action);
        }
    }

    let elapsed = start.elapsed().as_secs_f64();
    println!("\nDone: {} deals, {} void, {:.1}s ({:.0} deals/s)",
        deals_played, void_deals, elapsed, num_deals as f64 / elapsed);

    m.print();
}
