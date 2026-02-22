//! Shared evaluation functions for DMC Q-network agents.
//!
//! Used by both the `train_dmc` binary (inline eval during training) and
//! the `retro_eval` binary (retrospective re-evaluation of checkpoints).
//!
//! Key feature: **duplicate (seeded) matching** — each match count is split
//! into pairs where both matches in a pair use the same RNG seed (same deals)
//! but with teams swapped. This cancels deal luck, dramatically reducing
//! eval variance (the "duplicate" technique from bridge tournaments).

use std::time::Instant;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bid_eval;
use crate::bid_net::BidNet;
use crate::bid_obs;
use crate::dmc_net::DmcNet;
use crate::dmc_obs;
use crate::is_dd::{IsDdConfig, IsDdSearch};
use crate::rollout;
use crate::state::{GameState, Phase};

/// Configuration for a full eval run.
pub struct EvalConfig {
    /// Total matches vs random (must be even — played as N/2 duplicate pairs).
    pub random_matches: usize,
    /// Total matches vs frozen checkpoint (must be even).
    pub checkpoint_matches: usize,
    /// Total matches vs IS-DD (must be even).
    pub isdd_matches: usize,
    /// Time limit per IS-DD move in milliseconds.
    pub isdd_time_ms: u32,
}

/// Results from a full eval run.
pub struct EvalResult {
    /// Win rate vs random (0.0–1.0).
    pub rand_wr: f64,
    /// Win rate vs frozen checkpoint (0.0–1.0).
    pub ckpt_wr: f64,
    /// Win rate vs IS-DD (0.0–1.0).
    pub isdd_wr: f64,
    /// Wall-clock seconds for the full eval.
    pub elapsed: f64,
}

/// Get bid action using NN or heuristic for eval deals.
pub fn eval_bid_action(
    state: &GameState,
    bid_history: &[(u8, u8)],
    bid_net: &mut Option<BidNet>,
) -> u8 {
    if let Some(ref mut net) = bid_net {
        let obs = bid_obs::make_bid_observation(state, bid_history);
        let legal = state.legal_actions();
        let (action, _) = net.best_action(&obs, legal);
        action
    } else {
        bid_eval::improved_v2_bid(state)
    }
}

/// Run a full evaluation suite with duplicate matching.
///
/// For each eval type (random, checkpoint, IS-DD), plays N/2 duplicate pairs.
/// Each pair uses the same RNG seed for both sides, cancelling deal luck.
pub fn run_eval(
    q_net: &mut DmcNet,
    baseline_net: &mut Option<DmcNet>,
    bid_net: &mut Option<BidNet>,
    config: &EvalConfig,
) -> EvalResult {
    let start = Instant::now();

    // 1. Match play vs random (duplicate pairs)
    let rand_wr = if config.random_matches > 0 {
        let num_pairs = config.random_matches / 2;
        let mut wins = 0u32;
        for i in 0..num_pairs {
            let seed = 200_000 + i as u64;
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut rng_b = StdRng::seed_from_u64(seed);
            if play_match_eval(q_net, 0, "random", &mut None, bid_net, &mut rng_a) {
                wins += 1;
            }
            if play_match_eval(q_net, 1, "random", &mut None, bid_net, &mut rng_b) {
                wins += 1;
            }
        }
        wins as f64 / (num_pairs as f64 * 2.0)
    } else {
        0.0
    };

    // 2. Match play vs frozen checkpoint (duplicate pairs)
    let ckpt_wr = if config.checkpoint_matches > 0 && baseline_net.is_some() {
        let num_pairs = config.checkpoint_matches / 2;
        let mut wins = 0u32;
        for i in 0..num_pairs {
            let seed = 300_000 + i as u64;
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut rng_b = StdRng::seed_from_u64(seed);
            if play_match_eval(q_net, 0, "checkpoint", baseline_net, bid_net, &mut rng_a) {
                wins += 1;
            }
            if play_match_eval(q_net, 1, "checkpoint", baseline_net, bid_net, &mut rng_b) {
                wins += 1;
            }
        }
        wins as f64 / (num_pairs as f64 * 2.0)
    } else {
        0.0
    };

    // 3. Match play vs IS-DD (duplicate pairs)
    let isdd_wr = if config.isdd_matches > 0 {
        let num_pairs = config.isdd_matches / 2;
        let mut wins = 0u32;
        for i in 0..num_pairs {
            let seed = 400_000 + i as u64;
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut rng_b = StdRng::seed_from_u64(seed);
            if play_match_eval_isdd(q_net, 0, config.isdd_time_ms, bid_net, &mut rng_a) {
                wins += 1;
            }
            if play_match_eval_isdd(q_net, 1, config.isdd_time_ms, bid_net, &mut rng_b) {
                wins += 1;
            }
        }
        wins as f64 / (num_pairs as f64 * 2.0)
    } else {
        0.0
    };

    let elapsed = start.elapsed().as_secs_f64();
    EvalResult { rand_wr, ckpt_wr, isdd_wr, elapsed }
}

/// Play a match to 2000 for evaluation (vs random or checkpoint).
/// Both Q-net team and opponent use NN bid if available, else improved_v2.
pub fn play_match_eval(
    q_net: &mut DmcNet,
    q_team: u8,
    baseline: &str,
    baseline_net: &mut Option<DmcNet>,
    bid_net: &mut Option<BidNet>,
    rng: &mut StdRng,
) -> bool {
    let mut q_total = 0.0f32;
    let mut opp_total = 0.0f32;
    for _ in 0..50 {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, rng);
        let mut tracking = dmc_obs::EnvTracking::new();
        tracking.dealer = dealer;

        while !state.is_terminal() {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if state.phase == Phase::Bidding {
                eval_bid_action(&state, &tracking.bid_history, bid_net)
            } else if team == q_team {
                let obs = dmc_obs::make_observation(&state, &tracking);
                let legal_mask = state.legal_actions() as u32;
                let (best, _) = q_net.best_action(&obs, legal_mask);
                best
            } else {
                match baseline {
                    "random" => {
                        let mask = state.legal_actions();
                        let count = mask.count_ones();
                        let idx = rng.gen_range(0..count);
                        rollout::select_nth_bit(mask, idx)
                    }
                    "checkpoint" => {
                        let net = baseline_net.as_mut().unwrap();
                        let obs = dmc_obs::make_observation(&state, &tracking);
                        let legal_mask = state.legal_actions() as u32;
                        let (best, _) = net.best_action(&obs, legal_mask);
                        best
                    }
                    _ => unreachable!(),
                }
            };

            tracking.track_action(&state, action);
            state.step(action);
        }

        let rewards = state.rewards();
        q_total += rewards[q_team as usize];
        opp_total += rewards[1 - q_team as usize];
        if q_total >= 2000.0 || opp_total >= 2000.0 {
            break;
        }
    }
    q_total >= 2000.0
}

/// Play a match to 2000 vs IS-DD opponent.
pub fn play_match_eval_isdd(
    q_net: &mut DmcNet,
    q_team: u8,
    time_ms: u32,
    bid_net: &mut Option<BidNet>,
    rng: &mut StdRng,
) -> bool {
    let isdd_config = IsDdConfig {
        time_limit_ms: Some(time_ms),
        ..Default::default()
    };

    let mut q_total = 0.0f32;
    let mut opp_total = 0.0f32;
    for _ in 0..50 {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, rng);
        let mut tracking = dmc_obs::EnvTracking::new();
        tracking.dealer = dealer;

        // Initialize IS-DD searches for all 4 players
        let mut isdd_searches = [
            IsDdSearch::new(), IsDdSearch::new(),
            IsDdSearch::new(), IsDdSearch::new(),
        ];
        for (p, search) in isdd_searches.iter_mut().enumerate() {
            search.init_deal(&state, p as u8, true);
        }

        while !state.is_terminal() {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if state.phase == Phase::Bidding {
                eval_bid_action(&state, &tracking.bid_history, bid_net)
            } else if team == q_team {
                let obs = dmc_obs::make_observation(&state, &tracking);
                let legal_mask = state.legal_actions() as u32;
                let (best, _) = q_net.best_action(&obs, legal_mask);
                best
            } else {
                isdd_searches[player as usize].search(&state, &isdd_config, rng)
            };

            // Record action for IS-DD beliefs
            for search in isdd_searches.iter_mut() {
                search.record_action(&state, player, action);
            }

            tracking.track_action(&state, action);
            state.step(action);
        }

        let rewards = state.rewards();
        q_total += rewards[q_team as usize];
        opp_total += rewards[1 - q_team as usize];
        if q_total >= 2000.0 || opp_total >= 2000.0 {
            break;
        }
    }
    q_total >= 2000.0
}
