use std::time::{Duration, Instant};

use rand::Rng;

use crate::card::card_count;
use crate::determinize::determinize_greedy;
use crate::mcts::{BidPolicy, MctsConfig, MctsSearch, SearchResult};
use crate::state::{GameState, Phase};

/// Configuration for naive IS-MCTS (ensemble determinization).
pub struct NaiveIsMctsConfig {
    /// Number of determinized worlds to sample.
    pub determinizations: u32,
    /// MCTS iterations per determinized world.
    pub iterations_per_det: u32,
    /// UCB1 exploration constant.
    pub exploration: f32,
    /// Optional time limit in milliseconds (overrides `determinizations` count).
    pub time_limit_ms: Option<u32>,
    /// Which bid function to use during bidding phase.
    pub use_smart_bid: bool,
}

impl Default for NaiveIsMctsConfig {
    fn default() -> Self {
        NaiveIsMctsConfig {
            determinizations: 20,
            iterations_per_det: 50,
            exploration: std::f32::consts::SQRT_2,
            time_limit_ms: None,
            use_smart_bid: false,
        }
    }
}

/// Naive IS-MCTS search using ensemble determinization.
///
/// Samples multiple determinized worlds, runs standard MCTS on each,
/// and aggregates root visit counts to pick the best action.
pub struct NaiveIsMctsSearch {
    inner: MctsSearch,
}

impl NaiveIsMctsSearch {
    pub fn new() -> Self {
        NaiveIsMctsSearch {
            inner: MctsSearch::new(),
        }
    }

    pub fn search(&mut self, state: &GameState, config: &NaiveIsMctsConfig, rng: &mut impl Rng) -> u8 {
        // Skip MCTS search during bidding — use configured bid function
        if state.phase == Phase::Bidding {
            return if config.use_smart_bid {
                crate::bid_eval::smart_bid(state)
            } else {
                crate::bid_eval::heuristic_bid(state)
            };
        }
        self.search_with_stats(state, config, rng).best_action
    }

    pub fn search_with_stats(
        &mut self,
        state: &GameState,
        config: &NaiveIsMctsConfig,
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
            bid_policy: BidPolicy::Heuristic,
        };

        let deadline = config.time_limit_ms.map(|ms| {
            let scaled_ms = (ms as u64 * cards_left as u64) / 8;
            Instant::now() + Duration::from_millis(scaled_ms.max(1))
        });

        let mut successful_dets = 0u32;
        let mut det_count = 0u32;

        loop {
            if let Some(d) = deadline {
                if Instant::now() >= d { break; }
            } else if det_count >= config.determinizations {
                break;
            }

            let det_state = match determinize_greedy(state, observer, rng) {
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
}

/// Convenience wrapper that creates a temporary NaiveIsMctsSearch.
pub fn naive_ismcts_search(
    state: &GameState,
    config: &NaiveIsMctsConfig,
    rng: &mut impl Rng,
) -> u8 {
    let mut search = NaiveIsMctsSearch::new();
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
    fn test_naive_ismcts_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = NaiveIsMctsConfig {
            determinizations: 5,
            iterations_per_det: 20,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let action = naive_ismcts_search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "IS-MCTS returned illegal action {}",
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
    fn test_naive_ismcts_forced_move() {
        let mut rng = rand::thread_rng();
        let config = NaiveIsMctsConfig {
            determinizations: 3,
            iterations_per_det: 10,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..200 {
            let mut state = GameState::deal_random(0, &mut rng);
            while !state.is_terminal() {
                let legal = state.legal_actions();
                if legal.count_ones() == 1 {
                    let action = naive_ismcts_search(&state, &config, &mut rng);
                    let only_action = legal.trailing_zeros() as u8;
                    assert_eq!(action, only_action);
                    found += 1;
                    break;
                }
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = select_nth_bit(legal, idx);
                state.step(action);
            }
            if found >= 20 {
                break;
            }
        }
        assert!(found >= 5, "Not enough forced-move states found");
    }

    #[test]
    fn test_naive_ismcts_reusable() {
        let mut rng = rand::thread_rng();
        let mut search = NaiveIsMctsSearch::new();
        let config = NaiveIsMctsConfig {
            determinizations: 3,
            iterations_per_det: 10,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..20 {
            if let Some(state) = random_playing_state(&mut rng) {
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
    fn test_naive_ismcts_works_during_bidding() {
        let mut rng = rand::thread_rng();
        let config = NaiveIsMctsConfig {
            determinizations: 5,
            iterations_per_det: 20,
            ..Default::default()
        };
        let state = GameState::deal_random(0, &mut rng);
        assert_eq!(state.phase, Phase::Bidding);

        let action = naive_ismcts_search(&state, &config, &mut rng);
        let legal = state.legal_actions();
        assert!(
            legal & (1u64 << action) != 0,
            "IS-MCTS returned illegal bid action {}",
            action
        );
    }
}
