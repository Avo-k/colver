use std::time::{Duration, Instant};

use rand::Rng;

use crate::bid_eval::BidFunction;
use crate::card::card_count;
use crate::card_beliefs::CardBeliefs;
use crate::determinize::{determinize_greedy, determinize_weighted};
use crate::mcts::{RolloutPolicy, SearchResult};
use crate::rollout::{rollout_heuristic_bid, rollout_heuristic_play, rollout_random};
use crate::state::{GameState, Phase};

const REWARD_SCALE: f32 = 1.0 / 2000.0;
const UNEXPANDED: u32 = u32::MAX;

/// Configuration for single-tree IS-MCTS.
pub struct SingleTreeIsmctsConfig {
    /// Total MCTS iterations across all determinizations (default 1000).
    pub iterations: u32,
    /// UCB1 exploration constant.
    pub exploration: f32,
    /// Whether to use soft (probabilistic) inference in beliefs.
    pub use_soft_inference: bool,
    /// Which bid function to use during bidding phase.
    pub bid_function: BidFunction,
    /// Optional time limit in milliseconds (overrides `iterations`).
    pub time_limit_ms: Option<u32>,
    /// Rollout policy for simulations.
    pub rollout_policy: RolloutPolicy,
    /// Whether to reuse the tree between consecutive searches on the same deal.
    pub reuse_tree: bool,
    /// Decay factor applied to reused subtree statistics (0.0-1.0).
    pub decay_factor: f32,
}

impl Default for SingleTreeIsmctsConfig {
    fn default() -> Self {
        SingleTreeIsmctsConfig {
            iterations: 1000,
            exploration: std::f32::consts::SQRT_2,
            use_soft_inference: true,
            bid_function: BidFunction::ImprovedV2,
            time_limit_ms: None,
            rollout_policy: RolloutPolicy::HeuristicPlay,
            reuse_tree: true,
            decay_factor: 0.7,
        }
    }
}

struct IsmctsNode {
    children_start: u32,
    children_count: u16,
    player: u8,
    is_terminal: bool,
}

struct IsmctsEdge {
    action: u8,
    child: u32,              // UNEXPANDED = not yet expanded
    visit_count: u32,        // N(a): times this action was taken
    availability_count: u32, // N_avail(a): times this action was legal during selection
    total_reward: [f32; 2],  // cumulative reward per team when taking this action
}

/// Single-tree IS-MCTS with optional subtree persistence.
///
/// Maintains one shared tree across all determinizations. Each iteration:
/// (1) samples a determinized world, (2) traverses the shared tree selecting
/// only actions legal in that world, (3) expands/rollouts/backpropagates.
///
/// Optionally persists the tree between consecutive `search()` calls on the
/// same deal, re-rooting at the subtree reached by observed actions.
pub struct SingleTreeIsmctsSearch {
    nodes: Vec<IsmctsNode>,
    edges: Vec<IsmctsEdge>,
    path: Vec<usize>,           // edge indices for backprop
    beliefs: Option<CardBeliefs>,
    root: Option<u32>,
    pending_actions: Vec<u8>,
}

impl SingleTreeIsmctsSearch {
    pub fn new() -> Self {
        SingleTreeIsmctsSearch {
            nodes: Vec::with_capacity(4096),
            edges: Vec::with_capacity(32768),
            path: Vec::with_capacity(64),
            beliefs: None,
            root: None,
            pending_actions: Vec::new(),
        }
    }

    /// Initialize beliefs for a new deal. Clears the tree.
    pub fn init_deal(&mut self, state: &GameState, observer: u8, use_soft_inference: bool) {
        let mut beliefs = CardBeliefs::new(state, observer);
        beliefs.use_soft_inference = use_soft_inference;
        self.beliefs = Some(beliefs);
        self.clear();
    }

    /// Record that an action was played (by any player).
    /// Updates beliefs and tracks the action for tree re-rooting.
    pub fn advance(&mut self, state_before: &GameState, player: u8, action: u8) {
        if let Some(beliefs) = &mut self.beliefs {
            beliefs.record_action(state_before, player, action);
        }
        self.pending_actions.push(action);
    }

    /// Reset beliefs and tree (e.g., between deals).
    pub fn reset(&mut self) {
        self.beliefs = None;
        self.clear();
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.path.clear();
        self.root = None;
        self.pending_actions.clear();
    }

    fn new_node(&mut self, player: u8, is_terminal: bool) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(IsmctsNode {
            children_start: 0,
            children_count: 0,
            player,
            is_terminal,
        });
        idx
    }

    /// Ensure all legal actions in the current determinization have edges.
    /// If new actions are discovered, relocate edges to end of Vec and append new ones.
    /// Also bumps availability_count for all legal edges.
    fn ensure_edges(&mut self, node_idx: u32, legal_actions: u64) {
        let node = &self.nodes[node_idx as usize];
        let start = node.children_start as usize;
        let count = node.children_count as usize;

        // Build bitmask of existing edge actions
        let mut existing_mask = 0u64;
        for i in start..start + count {
            existing_mask |= 1u64 << self.edges[i].action;
        }

        let missing = legal_actions & !existing_mask;

        if missing != 0 {
            // Need to add new edges — relocate all edges to end of Vec
            let new_start = self.edges.len();

            // Copy existing edges
            for i in start..start + count {
                let edge = IsmctsEdge {
                    action: self.edges[i].action,
                    child: self.edges[i].child,
                    visit_count: self.edges[i].visit_count,
                    availability_count: self.edges[i].availability_count,
                    total_reward: self.edges[i].total_reward,
                };
                self.edges.push(edge);
            }

            // Append new edges for missing actions
            let mut m = missing;
            let mut new_count = count;
            while m != 0 {
                let bit = m.trailing_zeros() as u8;
                self.edges.push(IsmctsEdge {
                    action: bit,
                    child: UNEXPANDED,
                    visit_count: 0,
                    availability_count: 0,
                    total_reward: [0.0; 2],
                });
                new_count += 1;
                m &= m - 1;
            }

            // Update node
            let node = &mut self.nodes[node_idx as usize];
            node.children_start = new_start as u32;
            node.children_count = new_count as u16;
        }

        // Bump availability_count for all edges whose actions are legal
        let node = &self.nodes[node_idx as usize];
        let start = node.children_start as usize;
        let count = node.children_count as usize;
        for i in start..start + count {
            if legal_actions & (1u64 << self.edges[i].action) != 0 {
                self.edges[i].availability_count += 1;
            }
        }
    }

    /// Select an edge using modified UCB1 for IS-MCTS.
    /// Only considers edges whose actions are legal in the current determinization.
    /// Returns the edge index.
    fn ucb1_select(&self, node_idx: u32, legal_actions: u64, exploration: f32) -> usize {
        let node = &self.nodes[node_idx as usize];
        let team = (node.player & 1) as usize;
        let start = node.children_start as usize;
        let count = node.children_count as usize;

        let mut best_score = f32::NEG_INFINITY;
        let mut best_edge = start;

        for i in start..start + count {
            let edge = &self.edges[i];
            // Only consider legal actions
            if legal_actions & (1u64 << edge.action) == 0 {
                continue;
            }

            if edge.visit_count == 0 {
                return i; // Unvisited legal edge → infinite priority
            }

            // UCB1 with availability count: Q(a)/N(a) + C * sqrt(ln(N_avail(a)) / N(a))
            let exploitation = edge.total_reward[team] / edge.visit_count as f32;
            let ln_avail = (edge.availability_count as f32).ln();
            let explore = exploration * (ln_avail / edge.visit_count as f32).sqrt();
            let score = exploitation + explore;

            if score > best_score {
                best_score = score;
                best_edge = i;
            }
        }

        best_edge
    }

    /// Find the child node reached by a given action from a node.
    fn find_child_by_action(&self, node_idx: u32, action: u8) -> Option<u32> {
        let node = &self.nodes[node_idx as usize];
        let start = node.children_start as usize;
        let count = node.children_count as usize;
        for i in start..start + count {
            if self.edges[i].action == action && self.edges[i].child != UNEXPANDED {
                return Some(self.edges[i].child);
            }
        }
        None
    }

    /// Decay visit counts and rewards in a subtree (BFS).
    fn decay_subtree(&mut self, root: u32, factor: f32) {
        let mut queue = Vec::with_capacity(256);
        queue.push(root);
        let mut head = 0;
        while head < queue.len() {
            let node_idx = queue[head];
            head += 1;
            let node = &self.nodes[node_idx as usize];
            let start = node.children_start as usize;
            let count = node.children_count as usize;
            for i in start..start + count {
                let edge = &mut self.edges[i];
                edge.visit_count = (edge.visit_count as f32 * factor) as u32;
                edge.availability_count = (edge.availability_count as f32 * factor) as u32;
                edge.total_reward[0] *= factor;
                edge.total_reward[1] *= factor;
                if edge.child != UNEXPANDED {
                    queue.push(edge.child);
                }
            }
        }
    }

    /// Try to re-root the tree following pending actions.
    /// Returns the new root, or None if the path couldn't be followed.
    fn try_reroot(&mut self, config: &SingleTreeIsmctsConfig) -> Option<u32> {
        if !config.reuse_tree || self.root.is_none() || self.pending_actions.is_empty() {
            return None;
        }

        let mut current = self.root.unwrap();
        for &action in &self.pending_actions {
            match self.find_child_by_action(current, action) {
                Some(child) => current = child,
                None => return None,
            }
        }
        self.pending_actions.clear();

        // Decay old statistics
        self.decay_subtree(current, config.decay_factor);
        Some(current)
    }

    pub fn search(
        &mut self,
        state: &GameState,
        config: &SingleTreeIsmctsConfig,
        rng: &mut impl Rng,
    ) -> u8 {
        // During bidding, delegate to bid function
        if state.phase == Phase::Bidding {
            return config.bid_function.bid(state);
        }
        self.search_with_stats(state, config, rng).best_action
    }

    pub fn search_with_stats(
        &mut self,
        state: &GameState,
        config: &SingleTreeIsmctsConfig,
        rng: &mut impl Rng,
    ) -> SearchResult {
        debug_assert!(!state.is_terminal(), "Cannot search from terminal state");

        let observer = state.current_player();

        // Scale iterations by cards remaining
        let cards_left = card_count(state.hands[observer as usize]);
        let scaled_iters = (config.iterations * cards_left) / 8;
        let iterations = scaled_iters.max(1);

        // Try to reuse tree from previous search
        let root = match self.try_reroot(config) {
            Some(reused_root) => reused_root,
            None => {
                // Start fresh
                self.nodes.clear();
                self.edges.clear();
                self.pending_actions.clear();
                let r = self.new_node(state.current_player(), false);
                r
            }
        };
        self.root = Some(root);

        // Get normalized weights from beliefs
        let weights = self.beliefs.as_ref().map(|b| b.normalized_weights());

        let deadline = config.time_limit_ms.map(|ms| {
            let scaled_ms = (ms as u64 * cards_left as u64) / 8;
            Instant::now() + Duration::from_millis(scaled_ms.max(1))
        });

        let mut iter_count = 0u32;
        loop {
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    break;
                }
            } else if iter_count >= iterations {
                break;
            }
            iter_count += 1;

            // (1) Sample a determinized world
            let det_state = if let Some(ref w) = weights {
                determinize_weighted(state, observer, w, rng)
                    .or_else(|| determinize_greedy(state, observer, rng))
            } else {
                determinize_greedy(state, observer, rng)
            };

            let det_state = match det_state {
                Some(s) => s,
                None => continue,
            };

            let mut sim_state = det_state;
            self.path.clear();
            let mut current = root;

            // (2) Selection + Expansion
            loop {
                let node = &self.nodes[current as usize];
                if node.is_terminal {
                    break;
                }

                let legal = sim_state.legal_actions();
                if legal == 0 {
                    break;
                }

                self.ensure_edges(current, legal);

                let edge_idx = self.ucb1_select(current, legal, config.exploration);
                let action = self.edges[edge_idx].action;
                self.path.push(edge_idx);

                if self.edges[edge_idx].child == UNEXPANDED {
                    // Expansion
                    sim_state.step(action);
                    let is_term = sim_state.is_terminal();
                    let child = self.new_node(sim_state.current_player(), is_term);
                    self.edges[edge_idx].child = child;
                    break;
                } else {
                    sim_state.step(action);
                    current = self.edges[edge_idx].child;
                }
            }

            // (3) Simulation (rollout)
            let reward = if sim_state.is_terminal() {
                sim_state.rewards()
            } else {
                match config.rollout_policy {
                    RolloutPolicy::Random => rollout_random(&mut sim_state, rng),
                    RolloutPolicy::HeuristicBid => rollout_heuristic_bid(&mut sim_state, rng),
                    RolloutPolicy::HeuristicPlay => rollout_heuristic_play(&mut sim_state, rng),
                    RolloutPolicy::MaxiPlay => crate::rollout::rollout_maxi_play(&mut sim_state, rng),
                }
            };

            // (4) Backpropagation (on edges)
            let scaled = [reward[0] * REWARD_SCALE, reward[1] * REWARD_SCALE];
            for &edge_idx in &self.path {
                let edge = &mut self.edges[edge_idx];
                edge.visit_count += 1;
                edge.total_reward[0] += scaled[0];
                edge.total_reward[1] += scaled[1];
            }
        }

        // Best action: most-visited legal edge at root
        let legal = state.legal_actions();
        let node = &self.nodes[root as usize];
        let start = node.children_start as usize;
        let count = node.children_count as usize;

        let mut best_action = 0u8;
        let mut best_visits = 0u32;
        let mut visit_counts = Vec::new();
        let mut total_visits = 0u32;

        for i in start..start + count {
            let edge = &self.edges[i];
            if legal & (1u64 << edge.action) == 0 {
                continue;
            }
            visit_counts.push((edge.action, edge.visit_count));
            total_visits += edge.visit_count;
            if edge.visit_count > best_visits {
                best_visits = edge.visit_count;
                best_action = edge.action;
            }
        }

        // Fallback: if no edge was visited, pick first legal action
        if total_visits == 0 {
            best_action = legal.trailing_zeros() as u8;
        }

        SearchResult {
            best_action,
            visit_counts,
            root_visits: total_visits,
        }
    }
}

/// Convenience wrapper that creates a temporary search without beliefs.
pub fn single_tree_ismcts_search(
    state: &GameState,
    config: &SingleTreeIsmctsConfig,
    rng: &mut impl Rng,
) -> u8 {
    let mut search = SingleTreeIsmctsSearch::new();
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
    fn test_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = SingleTreeIsmctsConfig {
            iterations: 100,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let action = single_tree_ismcts_search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Single-tree IS-MCTS returned illegal action {}",
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
    fn test_with_beliefs() {
        let mut rng = rand::thread_rng();
        let config = SingleTreeIsmctsConfig {
            iterations: 100,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..50 {
            let state = GameState::deal_random(0, &mut rng);
            let mut search = SingleTreeIsmctsSearch::new();
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

                search.advance(&state_before, player, action);
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
    fn test_subtree_reuse() {
        let mut rng = rand::thread_rng();
        let config = SingleTreeIsmctsConfig {
            iterations: 200,
            reuse_tree: true,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..50 {
            let state = GameState::deal_random(0, &mut rng);
            let mut search = SingleTreeIsmctsSearch::new();
            search.init_deal(&state, 0, true);

            let mut current = state;
            let mut search_count = 0;

            while !current.is_terminal() {
                let player = current.current_player();
                let state_before = current;

                let action = if player == 0 {
                    let a = search.search(&current, &config, &mut rng);
                    if current.phase == Phase::Playing {
                        search_count += 1;

                        // After first play-phase search, tree should exist
                        if search_count > 1 {
                            assert!(search.root.is_some(), "Root should persist between searches");
                        }
                    }
                    a
                } else {
                    let legal = current.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    select_nth_bit(legal, idx)
                };

                search.advance(&state_before, player, action);
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
        assert!(found >= 10, "Not enough actions played");
    }

    #[test]
    fn test_subtree_fallback() {
        let mut rng = rand::thread_rng();
        let config_low = SingleTreeIsmctsConfig {
            iterations: 10, // Very few iterations — likely won't explore all children
            reuse_tree: true,
            ..Default::default()
        };
        let config_normal = SingleTreeIsmctsConfig {
            iterations: 100,
            reuse_tree: true,
            ..Default::default()
        };

        // Run a game with very low iterations to increase chance of missing paths
        for _ in 0..20 {
            let state = GameState::deal_random(0, &mut rng);
            let mut search = SingleTreeIsmctsSearch::new();
            search.init_deal(&state, 0, true);

            let mut current = state;
            while !current.is_terminal() {
                let player = current.current_player();
                let state_before = current;

                let action = if player == 0 {
                    // Alternate between low and normal to test both paths
                    search.search(&current, &config_low, &mut rng)
                } else {
                    let legal = current.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    select_nth_bit(legal, idx)
                };

                let legal = current.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Illegal action after fallback"
                );

                search.advance(&state_before, player, action);
                current.step(action);
            }

            // Now start a new search on same instance — should work fine
            let state2 = GameState::deal_random(0, &mut rng);
            search.init_deal(&state2, 0, true);
            let current2 = state2;
            if !current2.is_terminal() {
                let action = search.search(&current2, &config_normal, &mut rng);
                let legal = current2.legal_actions();
                assert!(legal & (1u64 << action) != 0);
            }
        }
    }

    #[test]
    fn test_reusable_across_deals() {
        let mut rng = rand::thread_rng();
        let mut search = SingleTreeIsmctsSearch::new();
        let config = SingleTreeIsmctsConfig {
            iterations: 50,
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
    fn test_works_during_bidding() {
        let mut rng = rand::thread_rng();
        let config = SingleTreeIsmctsConfig {
            iterations: 100,
            ..Default::default()
        };
        let state = GameState::deal_random(0, &mut rng);
        assert_eq!(state.phase, Phase::Bidding);

        let mut search = SingleTreeIsmctsSearch::new();
        search.init_deal(&state, state.current_player(), true);

        let action = search.search(&state, &config, &mut rng);
        let legal = state.legal_actions();
        assert!(
            legal & (1u64 << action) != 0,
            "Single-tree IS-MCTS returned illegal bid action {}",
            action
        );
    }

    #[test]
    fn test_no_reuse_mode() {
        let mut rng = rand::thread_rng();
        let config = SingleTreeIsmctsConfig {
            iterations: 100,
            reuse_tree: false,
            ..Default::default()
        };

        for _ in 0..10 {
            let state = GameState::deal_random(0, &mut rng);
            let mut search = SingleTreeIsmctsSearch::new();
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
                assert!(legal & (1u64 << action) != 0);

                search.advance(&state_before, player, action);
                current.step(action);
            }
        }
    }
}
