use std::time::{Duration, Instant};

use rand::Rng;

use crate::belief_net::BeliefNet;
use crate::belief_obs::{self, BELIEF_OBS_DIM};
use crate::bid_eval::BidFunction;
use crate::card::card_count;
use crate::card_beliefs::CardBeliefs;
use crate::determinize::{determinize_greedy, determinize_weighted};
use crate::dmc_obs::EnvTracking;
use crate::mcts::{MctsConfig, MctsSearch, RolloutPolicy, SearchResult};
use crate::state::{GameState, Phase};

/// Configuration for smart IS-MCTS.
pub struct SmartIsMctsConfig {
    /// Number of determinized worlds to sample.
    pub determinizations: u32,
    /// MCTS iterations per determinized world.
    pub iterations_per_det: u32,
    /// UCB1 exploration constant.
    pub exploration: f32,
    /// Whether to use soft (probabilistic) inference in addition to hard constraints.
    pub use_soft_inference: bool,
    /// Optional time limit in milliseconds (overrides `determinizations` count).
    pub time_limit_ms: Option<u32>,
    /// Which bid function to use during bidding phase.
    pub bid_function: BidFunction,
    /// If true and a BeliefNet is loaded, use NN beliefs instead of heuristic CardBeliefs.
    pub use_nn_beliefs: bool,
}

impl Default for SmartIsMctsConfig {
    fn default() -> Self {
        SmartIsMctsConfig {
            determinizations: 20,
            iterations_per_det: 50,
            exploration: std::f32::consts::SQRT_2,
            use_soft_inference: true,
            time_limit_ms: None,
            bid_function: BidFunction::ImprovedV2,
            use_nn_beliefs: false,
        }
    }
}

/// Smart IS-MCTS search using belief-weighted determinization.
///
/// Maintains a `CardBeliefs` model that is updated after every action in the deal.
/// When searching, it samples determinized worlds biased by these beliefs, then
/// runs standard MCTS on each and aggregates root visit counts.
///
/// Both teammates should each have their own `SmartIsMctsSearch` instance with
/// their own observer perspective, but both must call `record_action()` for ALL
/// actions by ALL players.
pub struct SmartIsMctsSearch {
    inner: MctsSearch,
    beliefs: Option<CardBeliefs>,
    belief_net: Option<BeliefNet>,
    belief_tracking: Option<EnvTracking>,
}

impl SmartIsMctsSearch {
    pub fn new() -> Self {
        SmartIsMctsSearch {
            inner: MctsSearch::new(),
            beliefs: None,
            belief_net: None,
            belief_tracking: None,
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

    pub fn search(
        &mut self,
        state: &GameState,
        config: &SmartIsMctsConfig,
        rng: &mut impl Rng,
    ) -> u8 {
        // Skip MCTS search during bidding — use configured bid function
        if state.phase == Phase::Bidding {
            return config.bid_function.bid(state);
        }
        self.search_with_stats(state, config, rng).best_action
    }

    pub fn search_with_stats(
        &mut self,
        state: &GameState,
        config: &SmartIsMctsConfig,
        rng: &mut impl Rng,
    ) -> SearchResult {
        debug_assert!(!state.is_terminal(), "Cannot search from terminal state");

        let observer = state.current_player();
        let mut action_votes = [0u32; 64];

        // Scale iterations by cards remaining: 8 cards → full, 1 card → 1/8
        let cards_left = card_count(state.hands[observer as usize]);
        let scaled_iters = (config.iterations_per_det * cards_left) / 8;

        let mcts_config = MctsConfig {
            iterations: scaled_iters.max(1),
            exploration: config.exploration,
            rollout_policy: RolloutPolicy::HeuristicPlay,
            ..Default::default()
        };

        let deadline = config.time_limit_ms.map(|ms| {
            let scaled_ms = (ms as u64 * cards_left as u64) / 8;
            Instant::now() + Duration::from_millis(scaled_ms.max(1))
        });

        let mut successful_dets = 0u32;
        let mut det_count = 0u32;

        // Get normalized weights from beliefs (NN or heuristic)
        let weights = if config.use_nn_beliefs && self.belief_net.is_some() {
            let net = self.belief_net.as_mut().unwrap();
            let tracking = self.belief_tracking.as_ref().unwrap();
            let mut obs_buf = [0.0f32; BELIEF_OBS_DIM];
            belief_obs::write_belief_observation(&mut obs_buf, 0, state, tracking, observer);
            let logits = net.evaluate(&obs_buf);
            Some(crate::belief_net::belief_to_weights(&logits, state, observer))
        } else {
            self.beliefs.as_ref().map(|b| b.normalized_weights())
        };

        loop {
            if let Some(d) = deadline {
                if Instant::now() >= d { break; }
            } else if det_count >= config.determinizations {
                break;
            }

            let det_state = if let Some(ref w) = weights {
                // Try weighted determinization first, fall back to greedy
                determinize_weighted(state, observer, w, rng)
                    .or_else(|| determinize_greedy(state, observer, rng))
            } else {
                determinize_greedy(state, observer, rng)
            };

            let det_state = match det_state {
                Some(s) => s,
                None => { det_count += 1; continue; }
            };

            let result = self.inner.search_with_stats(&det_state, &mcts_config, rng);

            for &(action, visits) in &result.visit_counts {
                action_votes[action as usize] += visits;
            }

            successful_dets += 1;
            det_count += 1;
        }

        // Build result from aggregated votes
        let legal = state.legal_actions();
        let mut best_action = 0u8;
        let mut best_votes = 0u32;
        let mut visit_counts = Vec::new();
        let mut total_votes = 0u32;

        let mut mask = legal;
        while mask != 0 {
            let bit = mask.trailing_zeros() as u8;
            let votes = action_votes[bit as usize];
            visit_counts.push((bit, votes));
            total_votes += votes;
            if votes > best_votes {
                best_votes = votes;
                best_action = bit;
            }
            mask &= mask - 1;
        }

        // Fallback: if no determinization succeeded, pick first legal action
        if successful_dets == 0 {
            best_action = legal.trailing_zeros() as u8;
        }

        SearchResult {
            best_action,
            visit_counts,
            root_visits: total_votes,
        }
    }

    /// Search using a neural network value function instead of rollouts.
    ///
    /// Same determinization logic as `search_with_stats`, but uses `MctsSearch::search_with_nn`
    /// for each determinized world.
    #[cfg(feature = "nn")]
    pub fn search_with_nn(
        &mut self,
        state: &GameState,
        config: &SmartIsMctsConfig,
        value_net: &mut crate::value_net::ValueNet,
        rng: &mut impl Rng,
    ) -> u8 {
        if state.phase == Phase::Bidding {
            return config.bid_function.bid(state);
        }

        let observer = state.current_player();
        let mut action_votes = [0u32; 64];

        let cards_left = card_count(state.hands[observer as usize]);
        let scaled_iters = (config.iterations_per_det * cards_left) / 8;

        let mcts_config = MctsConfig {
            iterations: scaled_iters.max(1),
            exploration: config.exploration,
            rollout_policy: RolloutPolicy::HeuristicPlay,
            ..Default::default()
        };

        let deadline = config.time_limit_ms.map(|ms| {
            let scaled_ms = (ms as u64 * cards_left as u64) / 8;
            Instant::now() + Duration::from_millis(scaled_ms.max(1))
        });

        let mut det_count = 0u32;

        let weights = if config.use_nn_beliefs && self.belief_net.is_some() {
            let net = self.belief_net.as_mut().unwrap();
            let tracking = self.belief_tracking.as_ref().unwrap();
            let mut obs_buf = [0.0f32; BELIEF_OBS_DIM];
            belief_obs::write_belief_observation(&mut obs_buf, 0, state, tracking, observer);
            let logits = net.evaluate(&obs_buf);
            Some(crate::belief_net::belief_to_weights(&logits, state, observer))
        } else {
            self.beliefs.as_ref().map(|b| b.normalized_weights())
        };

        loop {
            if let Some(d) = deadline {
                if Instant::now() >= d { break; }
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
                None => { det_count += 1; continue; }
            };

            let result = self.inner.search_with_nn(&det_state, &mcts_config, value_net, rng);

            for &(action, visits) in &result.visit_counts {
                action_votes[action as usize] += visits;
            }

            det_count += 1;
        }

        // Build result from aggregated votes
        let legal = state.legal_actions();
        let mut best_action = legal.trailing_zeros() as u8;
        let mut best_votes = 0u32;

        let mut mask = legal;
        while mask != 0 {
            let bit = mask.trailing_zeros() as u8;
            let votes = action_votes[bit as usize];
            if votes > best_votes {
                best_votes = votes;
                best_action = bit;
            }
            mask &= mask - 1;
        }

        best_action
    }

    /// Parallel search using rayon. Pre-generates seeds, runs determinizations in parallel.
    #[cfg(feature = "parallel")]
    pub fn search_parallel(
        &mut self,
        state: &GameState,
        config: &SmartIsMctsConfig,
        rng: &mut impl Rng,
    ) -> u8 {
        use rand::SeedableRng;
        use rayon::prelude::*;

        // Skip MCTS search during bidding
        if state.phase == Phase::Bidding {
            return config.bid_function.bid(state);
        }

        let observer = state.current_player();
        let cards_left = card_count(state.hands[observer as usize]);
        let scaled_iters = (config.iterations_per_det * cards_left) / 8;

        // Pre-generate seeds
        let num_dets = config.determinizations as usize;
        let seeds: Vec<u64> = (0..num_dets).map(|_| rng.gen()).collect();
        let weights = if config.use_nn_beliefs && self.belief_net.is_some() {
            let net = self.belief_net.as_mut().unwrap();
            let tracking = self.belief_tracking.as_ref().unwrap();
            let mut obs_buf = [0.0f32; BELIEF_OBS_DIM];
            belief_obs::write_belief_observation(&mut obs_buf, 0, state, tracking, observer);
            let logits = net.evaluate(&obs_buf);
            Some(crate::belief_net::belief_to_weights(&logits, state, observer))
        } else {
            self.beliefs.as_ref().map(|b| b.normalized_weights())
        };
        let game_state = *state; // Copy for thread safety

        // Run determinizations in parallel
        let vote_arrays: Vec<[u32; 64]> = seeds
            .par_iter()
            .map(|&seed| {
                let mut local_rng = rand::rngs::StdRng::seed_from_u64(seed);
                let mut local_search = MctsSearch::new();
                let mcts_config = MctsConfig {
                    iterations: scaled_iters.max(1),
                    exploration: config.exploration,
                    rollout_policy: RolloutPolicy::HeuristicPlay,
                    ..Default::default()
                };

                let det_state = if let Some(ref w) = weights {
                    crate::determinize::determinize_weighted(&game_state, observer, w, &mut local_rng)
                        .or_else(|| crate::determinize::determinize_greedy(&game_state, observer, &mut local_rng))
                } else {
                    crate::determinize::determinize_greedy(&game_state, observer, &mut local_rng)
                };

                let det_state = match det_state {
                    Some(s) => s,
                    None => return [0u32; 64],
                };

                let result = local_search.search_with_stats(&det_state, &mcts_config, &mut local_rng);
                let mut votes = [0u32; 64];
                for &(action, visits) in &result.visit_counts {
                    votes[action as usize] += visits;
                }
                votes
            })
            .collect();

        // Aggregate
        let mut action_votes = [0u32; 64];
        for votes in &vote_arrays {
            for i in 0..64 {
                action_votes[i] += votes[i];
            }
        }

        let legal = state.legal_actions();
        let mut best_action = legal.trailing_zeros() as u8;
        let mut best_votes = 0u32;
        let mut mask = legal;
        while mask != 0 {
            let bit = mask.trailing_zeros() as u8;
            if action_votes[bit as usize] > best_votes {
                best_votes = action_votes[bit as usize];
                best_action = bit;
            }
            mask &= mask - 1;
        }

        best_action
    }
}

/// Convenience wrapper that creates a temporary SmartIsMctsSearch without beliefs.
pub fn smart_ismcts_search(
    state: &GameState,
    config: &SmartIsMctsConfig,
    rng: &mut impl Rng,
) -> u8 {
    let mut search = SmartIsMctsSearch::new();
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
    fn test_smart_ismcts_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = SmartIsMctsConfig {
            determinizations: 5,
            iterations_per_det: 20,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let action = smart_ismcts_search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Smart IS-MCTS returned illegal action {}",
                    action
                );
                found += 1;
                if found >= 50 {
                    break;
                }
            }
        }
        assert!(found >= 10, "Not enough non-void deals to test");
    }

    #[test]
    fn test_smart_ismcts_with_beliefs() {
        let mut rng = rand::thread_rng();
        let config = SmartIsMctsConfig {
            determinizations: 5,
            iterations_per_det: 20,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..50 {
            let state = GameState::deal_random(0, &mut rng);
            let mut search = SmartIsMctsSearch::new();
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

                if found >= 200 {
                    break;
                }
            }
            if found >= 200 {
                break;
            }
        }
        assert!(found >= 20, "Not enough actions played");
    }

    #[test]
    fn test_smart_ismcts_works_during_bidding() {
        let mut rng = rand::thread_rng();
        let config = SmartIsMctsConfig {
            determinizations: 5,
            iterations_per_det: 20,
            ..Default::default()
        };
        let state = GameState::deal_random(0, &mut rng);
        assert_eq!(state.phase, Phase::Bidding);

        let mut search = SmartIsMctsSearch::new();
        search.init_deal(&state, state.current_player(), true);

        let action = search.search(&state, &config, &mut rng);
        let legal = state.legal_actions();
        assert!(
            legal & (1u64 << action) != 0,
            "Smart IS-MCTS returned illegal bid action {}",
            action
        );
    }

    #[test]
    fn test_smart_ismcts_reusable() {
        let mut rng = rand::thread_rng();
        let mut search = SmartIsMctsSearch::new();
        let config = SmartIsMctsConfig {
            determinizations: 3,
            iterations_per_det: 10,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..20 {
            if let Some(state) = random_playing_state(&mut rng) {
                // Reset beliefs for each new deal
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

    #[cfg(feature = "parallel")]
    #[test]
    fn test_parallel_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = SmartIsMctsConfig {
            determinizations: 5,
            iterations_per_det: 20,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let mut search = SmartIsMctsSearch::new();
                let action = search.search_parallel(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Parallel IS-MCTS returned illegal action {}",
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
}
