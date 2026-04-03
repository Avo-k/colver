//! Information Set Double-Dummy (IS-DD) agent.
//!
//! Combines the exact alpha-beta DD solver with determinization (like IS-MCTS,
//! but replacing approximate MCTS rollouts with exact DD solves). Each
//! determinized world gives a provably optimal answer, so fewer samples are
//! needed compared to IS-MCTS.
//!
//! Two modes:
//! - **Naive** (no beliefs): uniform determinization, like `naive_ismcts` but with DD.
//! - **Smart** (with beliefs): belief-weighted determinization, like `smart_ismcts` but with DD.
//!
//! Score-based aggregation: each DD solve returns exact NS points per card,
//! so we sum scores across determinizations rather than voting.

use std::time::{Duration, Instant};

use rand::Rng;

use crate::belief_net::BeliefNet;
use crate::belief_obs::{self, BELIEF_OBS_DIM, BELIEF_OBS_DIM_V2};
use crate::bid_eval::BidFunction;
use crate::card::card_count;
use crate::card_beliefs::CardBeliefs;
use crate::determinize::{determinize_greedy, determinize_weighted};
use crate::dmc_obs::EnvTracking;
use crate::solver::{new_tt_buffer, solve_with_scores};
use crate::state::{GameState, Phase};

/// Configuration for IS-DD search.
pub struct IsDdConfig {
    /// Number of determinized worlds to sample (default 20).
    pub determinizations: u32,
    /// Whether to use soft (probabilistic) inference in beliefs (default true).
    pub use_soft_inference: bool,
    /// Optional time limit in milliseconds (overrides `determinizations` count).
    pub time_limit_ms: Option<u32>,
    /// Which bid function to use during bidding phase.
    pub bid_function: BidFunction,
    /// If true and a BeliefNet is loaded, use NN beliefs instead of heuristic CardBeliefs.
    pub use_nn_beliefs: bool,
    /// If true (default), apply heuristic hard constraints (voids, trump ceiling) on top of NN beliefs.
    pub use_hard_constraints: bool,
}

impl Default for IsDdConfig {
    fn default() -> Self {
        IsDdConfig {
            determinizations: 20,
            use_soft_inference: true,
            time_limit_ms: None,
            bid_function: BidFunction::ImprovedV2,
            use_nn_beliefs: false,
            use_hard_constraints: true,
        }
    }
}

/// Per-card aggregated DD result.
pub struct IsDdResult {
    /// Best card for the current player's team.
    pub best_action: u8,
    /// (card, avg_score) for each legal move. Score is NS points (0-252).
    pub card_scores: Vec<(u8, f32)>,
    /// Number of successful determinizations.
    pub determinizations: u32,
}

/// IS-DD search using belief-weighted determinization + exact DD solving.
///
/// Maintains a `CardBeliefs` model (optional) and a pre-allocated TT buffer.
/// Optionally uses a `BeliefNet` for NN-based card location prediction.
/// API mirrors `SmartIsMctsSearch`.
pub struct IsDdSearch {
    beliefs: Option<CardBeliefs>,
    belief_net: Option<BeliefNet>,
    belief_tracking: Option<EnvTracking>,
    tt_buf: Vec<u64>,
}

impl IsDdSearch {
    pub fn new() -> Self {
        IsDdSearch {
            beliefs: None,
            belief_net: None,
            belief_tracking: None,
            tt_buf: new_tt_buffer(),
        }
    }

    /// Load a BeliefNet for NN-based beliefs.
    pub fn load_belief_net(&mut self, path: &str) -> std::io::Result<()> {
        self.belief_net = Some(BeliefNet::load(path)?);
        Ok(())
    }

    /// Check if a BeliefNet is loaded.
    pub fn has_belief_net(&self) -> bool {
        self.belief_net.is_some()
    }

    /// Initialize beliefs for a new deal from the given observer's perspective.
    pub fn init_deal(&mut self, state: &GameState, observer: u8, use_soft_inference: bool) {
        let mut beliefs = CardBeliefs::new(state, observer);
        beliefs.use_soft_inference = use_soft_inference;
        self.beliefs = Some(beliefs);

        // Also init NN belief tracking if BeliefNet is loaded
        if self.belief_net.is_some() {
            let mut tracking = EnvTracking::new();
            tracking.reset(state.dealer);
            self.belief_tracking = Some(tracking);
        }
    }

    /// Record an action by any player, updating beliefs.
    ///
    /// `state_before` is the state BEFORE the action was applied.
    pub fn record_action(&mut self, state_before: &GameState, player: u8, action: u8) {
        if let Some(beliefs) = &mut self.beliefs {
            beliefs.record_action(state_before, player, action);
        }
        if let Some(tracking) = &mut self.belief_tracking {
            tracking.track_action(state_before, action);
        }
    }

    /// Reset beliefs (e.g., between deals).
    pub fn reset(&mut self) {
        self.beliefs = None;
        self.belief_tracking = None;
    }

    /// Compute belief weights for determinization.
    /// When NN beliefs are enabled, applies hard constraints from heuristic CardBeliefs
    /// (voids, trump ceiling) on top of NN soft predictions.
    fn compute_weights(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        observer: u8,
    ) -> Option<[[f32; 32]; 4]> {
        if config.use_nn_beliefs && self.belief_net.is_some() {
            let net = self.belief_net.as_mut().unwrap();
            let tracking = self.belief_tracking.as_ref().unwrap();

            let logits = if net.obs_dim() == BELIEF_OBS_DIM_V2 {
                // V2: build hard constraints from CardBeliefs, then V2 obs
                let hard_constraints = if let Some(ref beliefs) = self.beliefs {
                    let raw = beliefs.raw_weights();
                    let observer_hand = state.hands[observer as usize];
                    let mut played = state.played_cards;
                    for i in 0..4 {
                        let c = state.current_trick[i];
                        if c != crate::card::EMPTY {
                            played |= 1u32 << c;
                        }
                    }
                    let known = observer_hand | played;
                    let hidden_players = [
                        ((observer + 1) % 4),
                        ((observer + 2) % 4),
                        ((observer + 3) % 4),
                    ];
                    let mut hc = [0.0f32; 96];
                    for (hp_idx, &hp) in hidden_players.iter().enumerate() {
                        let base = hp_idx * 32;
                        for card_idx in 0..32u32 {
                            if known & (1 << card_idx) != 0 {
                                hc[base + card_idx as usize] = 1.0;
                                continue;
                            }
                            if raw[hp as usize][card_idx as usize] == 0.0 {
                                hc[base + card_idx as usize] = 1.0;
                            }
                        }
                    }
                    hc
                } else {
                    [0.0f32; 96]
                };

                let mut obs_buf = [0.0f32; BELIEF_OBS_DIM_V2];
                belief_obs::write_belief_observation_v2(
                    &mut obs_buf, 0, state, tracking, observer, &hard_constraints,
                );
                net.evaluate(&obs_buf)
            } else {
                let mut obs_buf = [0.0f32; BELIEF_OBS_DIM];
                belief_obs::write_belief_observation(&mut obs_buf, 0, state, tracking, observer);
                net.evaluate(&obs_buf)
            };
            let mut nn_weights = crate::belief_net::belief_to_weights(&logits, net.num_classes(), state, observer);

            // Hybrid: apply hard constraints from CardBeliefs (voids, trump ceiling)
            if config.use_hard_constraints {
            if let Some(ref beliefs) = self.beliefs {
                let raw = beliefs.raw_weights();
                let observer_hand = state.hands[observer as usize];
                let mut played = state.played_cards;
                for i in 0..4 {
                    let c = state.current_trick[i];
                    if c != crate::card::EMPTY {
                        played |= 1u32 << c;
                    }
                }
                let known = observer_hand | played;

                for card in 0..32u32 {
                    if known & (1 << card) != 0 {
                        continue;
                    }
                    for p in 0..4usize {
                        if raw[p][card as usize] == 0.0 {
                            nn_weights[p][card as usize] = 0.0;
                        }
                    }
                    let sum: f32 = (0..4).map(|p| nn_weights[p][card as usize]).sum();
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for p in 0..4 {
                            nn_weights[p][card as usize] *= inv;
                        }
                    }
                }
            }
            }

            Some(nn_weights)
        } else {
            self.beliefs.as_ref().map(|b| b.normalized_weights())
        }
    }

    /// Get current belief weights for a given observer.
    /// Returns `(nn_weights, heuristic_weights)` where each is `weights[player][card]`.
    /// NN weights use hybrid mode (NN + hard constraints from heuristic).
    /// Heuristic weights are purely from `CardBeliefs::normalized_weights()`.
    pub fn get_belief_weights(
        &mut self,
        state: &GameState,
        observer: u8,
    ) -> (Option<[[f32; 32]; 4]>, Option<[[f32; 32]; 4]>) {
        let nn_config = IsDdConfig {
            use_nn_beliefs: true,
            use_hard_constraints: true,
            ..Default::default()
        };
        let nn_weights = if self.belief_net.is_some() {
            self.compute_weights(state, &nn_config, observer)
        } else {
            None
        };
        let heuristic_weights = self.beliefs.as_ref().map(|b| b.normalized_weights());
        (nn_weights, heuristic_weights)
    }

    pub fn search(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
    ) -> u8 {
        if state.phase == Phase::Bidding {
            return config.bid_function.bid(state);
        }
        self.search_with_stats(state, config, rng).best_action
    }

    pub fn search_with_stats(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
    ) -> IsDdResult {
        debug_assert!(!state.is_terminal(), "Cannot search from terminal state");

        let observer = state.current_player();
        let team = GameState::player_team(observer);
        let maximizing = team == 0; // NS maximizes, EW minimizes

        // Score accumulators: sum of NS points per card, count per card
        let mut score_sum = [0i64; 32];
        let mut score_count = [0u32; 32];

        // Scale time budget by cards remaining
        let cards_left = card_count(state.hands[observer as usize]);
        let deadline = config.time_limit_ms.map(|ms| {
            let scaled_ms = (ms as u64 * cards_left as u64) / 8;
            Instant::now() + Duration::from_millis(scaled_ms.max(1))
        });

        let weights = self.compute_weights(state, config, observer);

        let mut successful_dets = 0u32;
        let mut det_count = 0u32;

        loop {
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    break;
                }
            } else if det_count >= config.determinizations {
                break;
            }

            let det_state = if let Some(ref w) = weights {
                determinize_weighted(state, observer, w, rng)
                    .or_else(|| determinize_greedy(state, observer, rng))
            } else {
                determinize_greedy(state, observer, rng)
            };

            let det_state = match det_state {
                Some(s) => s,
                None => {
                    det_count += 1;
                    continue;
                }
            };

            let scores = solve_with_scores(&det_state, Some(&mut self.tt_buf));

            for i in 0..scores.count {
                let (card, ns_pts) = scores.scores[i];
                score_sum[card as usize] += ns_pts as i64;
                score_count[card as usize] += 1;
            }

            successful_dets += 1;
            det_count += 1;
        }

        // Build result: pick best card based on aggregated scores
        let legal = state.legal_actions();
        let mut best_action = legal.trailing_zeros() as u8;
        let mut best_avg: f32 = if maximizing { f32::NEG_INFINITY } else { f32::INFINITY };
        let mut card_scores = Vec::new();

        let mut mask = legal;
        while mask != 0 {
            let card = mask.trailing_zeros() as u8;
            let count = score_count[card as usize];
            let avg = if count > 0 {
                score_sum[card as usize] as f32 / count as f32
            } else {
                81.0 // neutral fallback (≈162/2)
            };

            card_scores.push((card, avg));

            let dominated = if maximizing {
                avg > best_avg
            } else {
                avg < best_avg
            };
            if dominated {
                best_avg = avg;
                best_action = card;
            }
            mask &= mask - 1;
        }

        // Fallback: if no determinization succeeded, pick first legal action
        if successful_dets == 0 {
            best_action = legal.trailing_zeros() as u8;
        }

        IsDdResult {
            best_action,
            card_scores,
            determinizations: successful_dets,
        }
    }

    /// Parallel search using rayon. Pre-generates seeds, runs determinizations in parallel.
    /// Each thread gets its own TT buffer (2MB).
    #[cfg(feature = "parallel")]
    pub fn search_parallel(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
    ) -> u8 {
        use rand::SeedableRng;
        use rayon::prelude::*;

        if state.phase == Phase::Bidding {
            return config.bid_function.bid(state);
        }

        let observer = state.current_player();
        let team = GameState::player_team(observer);
        let maximizing = team == 0;

        let num_dets = config.determinizations as usize;
        let seeds: Vec<u64> = (0..num_dets).map(|_| rng.gen()).collect();
        let weights = self.compute_weights(state, config, observer);
        let game_state = *state; // Copy for thread safety

        // Each thread returns (score_sum[32], score_count[32])
        let results: Vec<([i64; 32], [u32; 32])> = seeds
            .par_iter()
            .map(|&seed| {
                let mut local_rng = rand::rngs::StdRng::seed_from_u64(seed);
                let mut tt = new_tt_buffer();

                let det_state = if let Some(ref w) = weights {
                    determinize_weighted(&game_state, observer, w, &mut local_rng)
                        .or_else(|| determinize_greedy(&game_state, observer, &mut local_rng))
                } else {
                    determinize_greedy(&game_state, observer, &mut local_rng)
                };

                let det_state = match det_state {
                    Some(s) => s,
                    None => return ([0i64; 32], [0u32; 32]),
                };

                let scores = solve_with_scores(&det_state, Some(&mut tt));

                let mut sum = [0i64; 32];
                let mut count = [0u32; 32];
                for i in 0..scores.count {
                    let (card, ns_pts) = scores.scores[i];
                    sum[card as usize] += ns_pts as i64;
                    count[card as usize] += 1;
                }

                (sum, count)
            })
            .collect();

        // Aggregate
        let mut total_sum = [0i64; 32];
        let mut total_count = [0u32; 32];
        for (sum, count) in &results {
            for i in 0..32 {
                total_sum[i] += sum[i];
                total_count[i] += count[i];
            }
        }

        let legal = state.legal_actions();
        let mut best_action = legal.trailing_zeros() as u8;
        let mut best_avg: f32 = if maximizing { f32::NEG_INFINITY } else { f32::INFINITY };

        let mut mask = legal;
        while mask != 0 {
            let card = mask.trailing_zeros() as u8;
            let count = total_count[card as usize];
            let avg = if count > 0 {
                total_sum[card as usize] as f32 / count as f32
            } else {
                81.0
            };

            let dominated = if maximizing {
                avg > best_avg
            } else {
                avg < best_avg
            };
            if dominated {
                best_avg = avg;
                best_action = card;
            }
            mask &= mask - 1;
        }

        best_action
    }
}

/// Convenience wrapper that creates a temporary IsDdSearch without beliefs.
pub fn is_dd_search(state: &GameState, config: &IsDdConfig, rng: &mut impl Rng) -> u8 {
    let mut search = IsDdSearch::new();
    search.search(state, config, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollout::select_nth_bit;
    use crate::state::Phase;

    fn random_playing_state(rng: &mut impl Rng) -> Option<GameState> {
        let mut state = GameState::deal_random(0, rng);
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let legal = state.legal_actions();
            let count = legal.count_ones();
            let idx = rng.gen_range(0..count);
            let action = select_nth_bit(legal, idx);
            state.step(action);
        }
        if state.is_terminal() {
            None
        } else {
            Some(state)
        }
    }

    #[test]
    #[ignore]
    fn test_is_dd_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 5,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let action = is_dd_search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "IS-DD returned illegal action {}",
                    action
                );
                found += 1;
                if found >= 30 {
                    break;
                }
            }
        }
        assert!(found >= 10, "Not enough non-void deals to test");
    }

    #[test]
    #[ignore]
    fn test_is_dd_with_beliefs() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 3,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..50 {
            let state = GameState::deal_random(0, &mut rng);
            let mut search = IsDdSearch::new();
            search.init_deal(&state, 0, true);

            let mut current = state;
            while !current.is_terminal() {
                let player = current.current_player();
                let state_before = current;

                let action = if player == 0 {
                    search.search(&current, &config, &mut rng)
                } else {
                    let legal = current.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    select_nth_bit(legal, idx)
                };

                let legal = current.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Illegal action {} by player {}",
                    action,
                    player
                );

                search.record_action(&state_before, player, action);
                current.step(action);
                found += 1;

                if found >= 100 {
                    break;
                }
            }
            if found >= 100 {
                break;
            }
        }
        assert!(found >= 20, "Not enough actions played");
    }

    #[test]
    #[ignore]
    fn test_is_dd_works_during_bidding() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 3,
            ..Default::default()
        };
        let state = GameState::deal_random(0, &mut rng);
        assert_eq!(state.phase, Phase::Bidding);

        let mut search = IsDdSearch::new();
        search.init_deal(&state, state.current_player(), true);

        let action = search.search(&state, &config, &mut rng);
        let legal = state.legal_actions();
        assert!(
            legal & (1u64 << action) != 0,
            "IS-DD returned illegal bid action {}",
            action
        );
    }

    #[test]
    #[ignore]
    fn test_is_dd_reusable() {
        let mut rng = rand::thread_rng();
        let mut search = IsDdSearch::new();
        let config = IsDdConfig {
            determinizations: 3,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..20 {
            if let Some(state) = random_playing_state(&mut rng) {
                search.reset();
                let action = search.search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(legal & (1u64 << action) != 0);
                found += 1;
            }
            if found >= 10 {
                break;
            }
        }
        assert!(found >= 5);
    }

    #[test]
    #[ignore]
    fn test_is_dd_search_with_stats() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 5,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let mut search = IsDdSearch::new();
                let result = search.search_with_stats(&state, &config, &mut rng);

                assert!(result.determinizations > 0);
                assert!(!result.card_scores.is_empty());

                // Best action must be legal
                let legal = state.legal_actions();
                assert!(legal & (1u64 << result.best_action) != 0);

                // All scores should be in valid range
                for &(card, avg) in &result.card_scores {
                    assert!(legal & (1u64 << card) != 0);
                    assert!(avg >= 0.0 && avg <= 252.0, "avg={}", avg);
                }

                found += 1;
                if found >= 20 {
                    break;
                }
            }
        }
        assert!(found >= 10);
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[ignore]
    fn test_parallel_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 5,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let mut search = IsDdSearch::new();
                let action = search.search_parallel(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Parallel IS-DD returned illegal action {}",
                    action
                );
                found += 1;
                if found >= 20 {
                    break;
                }
            }
        }
        assert!(found >= 10, "Not enough non-void deals to test");
    }
}
