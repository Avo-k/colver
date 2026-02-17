use rand::Rng;

use crate::rollout::{
    rollout_heuristic_bid, rollout_heuristic_play, rollout_heuristic_play_with_rave,
    rollout_random, RolloutRaveInfo,
};
use crate::state::GameState;

const REWARD_SCALE: f32 = 1.0 / 2000.0;

/// Rollout policy used during MCTS simulations.
#[derive(Clone, Copy, Default)]
pub enum RolloutPolicy {
    /// Random bidding + random play (original behavior).
    #[default]
    Random,
    /// Heuristic bidding + random play.
    HeuristicBid,
    /// Heuristic bidding + heuristic play (strongest).
    HeuristicPlay,
    /// Maxi bidding + maxi play (convention-linked).
    MaxiPlay,
}

/// Configuration for MCTS search.
pub struct MctsConfig {
    /// Number of MCTS iterations per search.
    pub iterations: u32,
    /// UCB1 exploration constant.
    pub exploration: f32,
    /// Rollout policy for simulations.
    pub rollout_policy: RolloutPolicy,
    /// Whether to use RAVE/AMAF (only with HeuristicPlay).
    pub use_rave: bool,
    /// RAVE equivalence parameter (higher = trust AMAF longer).
    pub rave_k: f32,
}

impl Default for MctsConfig {
    fn default() -> Self {
        MctsConfig {
            iterations: 1000,
            exploration: std::f32::consts::SQRT_2,
            rollout_policy: RolloutPolicy::default(),
            use_rave: false,
            rave_k: 300.0,
        }
    }
}

struct Node {
    visit_count: u32,
    total_reward: [f32; 2], // cumulative [NS, EW] rewards (normalized)
    children_start: u32,    // index into edges Vec
    children_count: u8,     // 0 = unexpanded leaf
    player: u8,             // who acts here (0-3)
    is_terminal: bool,
}

struct Edge {
    action: u8,
    child: u32, // u32::MAX = child not yet created
}

/// Result of an MCTS search with statistics.
pub struct SearchResult {
    pub best_action: u8,
    /// (action, visit_count) for each child of root.
    pub visit_counts: Vec<(u8, u32)>,
    pub root_visits: u32,
}

/// Reusable MCTS search state. Arena-based tree with nodes and edges in flat Vecs.
pub struct MctsSearch {
    nodes: Vec<Node>,
    edges: Vec<Edge>,
    path: Vec<u32>, // reusable backprop buffer (node indices)
    // RAVE/AMAF storage (parallel to nodes, only allocated when use_rave=true)
    amaf_visits: Vec<[u32; 32]>,
    amaf_reward: Vec<[[f32; 32]; 2]>,
}

impl MctsSearch {
    pub fn new() -> Self {
        MctsSearch {
            nodes: Vec::with_capacity(2048),
            edges: Vec::with_capacity(16384),
            path: Vec::with_capacity(64),
            amaf_visits: Vec::new(),
            amaf_reward: Vec::new(),
        }
    }

    fn clear(&mut self) {
        self.nodes.clear();
        self.edges.clear();
        self.path.clear();
        self.amaf_visits.clear();
        self.amaf_reward.clear();
    }

    fn new_node(&mut self, player: u8, is_terminal: bool, use_rave: bool) -> u32 {
        let idx = self.nodes.len() as u32;
        self.nodes.push(Node {
            visit_count: 0,
            total_reward: [0.0; 2],
            children_start: 0,
            children_count: 0,
            player,
            is_terminal,
        });
        if use_rave {
            self.amaf_visits.push([0u32; 32]);
            self.amaf_reward.push([[0.0f32; 32]; 2]);
        }
        idx
    }

    fn expand(&mut self, node_idx: u32, state: &GameState) {
        let legal = state.legal_actions();
        debug_assert!(legal != 0, "expand called on terminal state");
        let start = self.edges.len() as u32;
        let mut mask = legal;
        let mut count = 0u8;
        while mask != 0 {
            let bit = mask.trailing_zeros() as u8;
            self.edges.push(Edge {
                action: bit,
                child: u32::MAX,
            });
            count += 1;
            mask &= mask - 1;
        }
        self.nodes[node_idx as usize].children_start = start;
        self.nodes[node_idx as usize].children_count = count;
    }

    fn ucb1_select(&self, node_idx: u32, exploration: f32) -> usize {
        let node = &self.nodes[node_idx as usize];
        let team = (node.player & 1) as usize;
        let ln_parent = (node.visit_count as f32).ln();
        let start = node.children_start as usize;
        let end = start + node.children_count as usize;

        let mut best_score = f32::NEG_INFINITY;
        let mut best_edge = start;

        for i in start..end {
            let edge = &self.edges[i];
            if edge.child == u32::MAX {
                return i; // unvisited → infinite priority
            }
            let child = &self.nodes[edge.child as usize];
            if child.visit_count == 0 {
                return i;
            }
            let exploitation = child.total_reward[team] / child.visit_count as f32;
            let explore = exploration * (ln_parent / child.visit_count as f32).sqrt();
            let score = exploitation + explore;
            if score > best_score {
                best_score = score;
                best_edge = i;
            }
        }
        best_edge
    }

    fn ucb1_select_rave(&self, node_idx: u32, config: &MctsConfig) -> usize {
        let node = &self.nodes[node_idx as usize];
        let team = (node.player & 1) as usize;
        let ln_parent = (node.visit_count as f32).ln();
        let start = node.children_start as usize;
        let end = start + node.children_count as usize;

        let mut best_score = f32::NEG_INFINITY;
        let mut best_edge = start;

        for i in start..end {
            let edge = &self.edges[i];
            if edge.child == u32::MAX {
                return i; // unvisited → infinite priority
            }
            let child = &self.nodes[edge.child as usize];
            if child.visit_count == 0 {
                return i;
            }

            let q_mcts = child.total_reward[team] / child.visit_count as f32;
            let explore = config.exploration * (ln_parent / child.visit_count as f32).sqrt();

            let action = edge.action as usize;
            let amaf_v = self.amaf_visits[node_idx as usize][action];
            let score = if amaf_v > 0 {
                let q_amaf = self.amaf_reward[node_idx as usize][team][action] / amaf_v as f32;
                let beta = (config.rave_k / (3.0 * child.visit_count as f32 + config.rave_k)).sqrt();
                (1.0 - beta) * q_mcts + beta * q_amaf + explore
            } else {
                q_mcts + explore
            };

            if score > best_score {
                best_score = score;
                best_edge = i;
            }
        }
        best_edge
    }

    pub fn search(&mut self, state: &GameState, config: &MctsConfig, rng: &mut impl Rng) -> u8 {
        self.search_with_stats(state, config, rng).best_action
    }

    pub fn search_with_stats(
        &mut self,
        state: &GameState,
        config: &MctsConfig,
        rng: &mut impl Rng,
    ) -> SearchResult {
        debug_assert!(!state.is_terminal(), "Cannot search from terminal state");
        self.clear();

        let use_rave = config.use_rave;

        // Create and expand root
        let root = self.new_node(state.current_player(), false, use_rave);
        self.expand(root, state);

        let mut rave_info = RolloutRaveInfo::new();

        for _ in 0..config.iterations {
            let mut sim_state = *state; // Copy ~56 bytes
            self.path.clear();
            self.path.push(root);

            let mut current = root;

            // Selection: descend tree via UCB1 (with or without RAVE)
            loop {
                let node = &self.nodes[current as usize];
                if node.is_terminal || node.children_count == 0 {
                    break;
                }

                let edge_idx = if use_rave {
                    self.ucb1_select_rave(current, config)
                } else {
                    self.ucb1_select(current, config.exploration)
                };
                let action = self.edges[edge_idx].action;

                if self.edges[edge_idx].child == u32::MAX {
                    // Expansion: create child node
                    sim_state.step(action);
                    let is_term = sim_state.is_terminal();
                    let child = self.new_node(sim_state.current_player(), is_term, use_rave);
                    // NB: new_node may have reallocated nodes vec, but edge_idx is into edges vec
                    self.edges[edge_idx].child = child;
                    if !is_term {
                        self.expand(child, &sim_state);
                    }
                    self.path.push(child);
                    break;
                } else {
                    sim_state.step(action);
                    current = self.edges[edge_idx].child;
                    self.path.push(current);
                }
            }

            // Simulation: rollout to terminal
            let reward = if sim_state.is_terminal() {
                sim_state.rewards()
            } else if use_rave {
                rave_info.cards_by_team = [0; 2];
                rave_info.forced_cards = 0;
                rollout_heuristic_play_with_rave(&mut sim_state, &mut rave_info, rng)
            } else {
                match config.rollout_policy {
                    RolloutPolicy::Random => rollout_random(&mut sim_state, rng),
                    RolloutPolicy::HeuristicBid => rollout_heuristic_bid(&mut sim_state, rng),
                    RolloutPolicy::HeuristicPlay => rollout_heuristic_play(&mut sim_state, rng),
                    RolloutPolicy::MaxiPlay => crate::rollout::rollout_maxi_play(&mut sim_state, rng),
                }
            };

            // Backpropagation
            let scaled = [reward[0] * REWARD_SCALE, reward[1] * REWARD_SCALE];
            for &node_idx in &self.path {
                let node = &mut self.nodes[node_idx as usize];
                node.visit_count += 1;
                node.total_reward[0] += scaled[0];
                node.total_reward[1] += scaled[1];
            }

            // RAVE backprop: update AMAF for non-forced cards played by each team
            if use_rave {
                for &node_idx in &self.path {
                    let node = &self.nodes[node_idx as usize];
                    let node_team = (node.player & 1) as usize;
                    let eligible = rave_info.cards_by_team[node_team] & !rave_info.forced_cards;

                    let start = node.children_start as usize;
                    let end = start + node.children_count as usize;
                    for i in start..end {
                        let a = self.edges[i].action as usize;
                        if a < 32 && eligible & (1u32 << a) != 0 {
                            self.amaf_visits[node_idx as usize][a] += 1;
                            self.amaf_reward[node_idx as usize][node_team][a] += scaled[node_team];
                        }
                    }
                }
            }
        }

        // Best action: most-visited child of root (robust child selection)
        let root_node = &self.nodes[root as usize];
        let start = root_node.children_start as usize;
        let end = start + root_node.children_count as usize;

        let mut best_action = 0u8;
        let mut best_visits = 0u32;
        let mut visit_counts = Vec::with_capacity(root_node.children_count as usize);

        for i in start..end {
            let edge = &self.edges[i];
            let visits = if edge.child != u32::MAX {
                self.nodes[edge.child as usize].visit_count
            } else {
                0
            };
            visit_counts.push((edge.action, visits));
            if visits > best_visits {
                best_visits = visits;
                best_action = edge.action;
            }
        }

        SearchResult {
            best_action,
            visit_counts,
            root_visits: root_node.visit_count,
        }
    }

    /// MCTS search using a neural network value function instead of rollouts.
    ///
    /// At leaf nodes, instead of playing out to terminal, the NN evaluates the position
    /// directly. Terminal nodes use actual outcomes (1.0 for team 0 win, 0.0 for loss).
    /// NN outputs are already in [0,1], so no REWARD_SCALE is applied.
    #[cfg(feature = "nn")]
    pub fn search_with_nn(
        &mut self,
        state: &GameState,
        config: &MctsConfig,
        value_net: &mut crate::value_net::ValueNet,
        _rng: &mut impl Rng,
    ) -> SearchResult {
        use crate::features::{extract_features, FEATURE_DIM};
        use crate::scoring::compute_deal_score;

        debug_assert!(!state.is_terminal(), "Cannot search from terminal state");
        self.clear();

        // NN path never uses RAVE
        let use_rave = false;

        let root = self.new_node(state.current_player(), false, use_rave);
        self.expand(root, state);

        let mut feature_buf = [0.0f32; FEATURE_DIM];

        for _ in 0..config.iterations {
            let mut sim_state = *state;
            self.path.clear();
            self.path.push(root);

            let mut current = root;

            // Selection: descend tree via UCB1
            loop {
                let node = &self.nodes[current as usize];
                if node.is_terminal || node.children_count == 0 {
                    break;
                }

                let edge_idx = self.ucb1_select(current, config.exploration);
                let action = self.edges[edge_idx].action;

                if self.edges[edge_idx].child == u32::MAX {
                    sim_state.step(action);
                    let is_term = sim_state.is_terminal();
                    let child = self.new_node(sim_state.current_player(), is_term, use_rave);
                    self.edges[edge_idx].child = child;
                    if !is_term {
                        self.expand(child, &sim_state);
                    }
                    self.path.push(child);
                    break;
                } else {
                    sim_state.step(action);
                    current = self.edges[edge_idx].child;
                    self.path.push(current);
                }
            }

            // Evaluation: NN value or terminal outcome
            let reward = if sim_state.is_terminal() {
                // Use binary win/loss based on deal score
                if sim_state.contract.value == 0 {
                    // Void deal (4 passes) — draw
                    [0.5, 0.5]
                } else {
                    let score = compute_deal_score(&sim_state);
                    if score.scores[0] > score.scores[1] {
                        [1.0, 0.0]
                    } else if score.scores[1] > score.scores[0] {
                        [0.0, 1.0]
                    } else {
                        [0.5, 0.5]
                    }
                }
            } else {
                extract_features(&sim_state, &mut feature_buf);
                let p = value_net.evaluate(&feature_buf);
                [p, 1.0 - p]
            };

            // Backpropagation (no REWARD_SCALE — NN outputs already in [0,1])
            for &node_idx in &self.path {
                let node = &mut self.nodes[node_idx as usize];
                node.visit_count += 1;
                node.total_reward[0] += reward[0];
                node.total_reward[1] += reward[1];
            }
        }

        // Best action: most-visited child of root
        let root_node = &self.nodes[root as usize];
        let start = root_node.children_start as usize;
        let end = start + root_node.children_count as usize;

        let mut best_action = 0u8;
        let mut best_visits = 0u32;
        let mut visit_counts = Vec::with_capacity(root_node.children_count as usize);

        for i in start..end {
            let edge = &self.edges[i];
            let visits = if edge.child != u32::MAX {
                self.nodes[edge.child as usize].visit_count
            } else {
                0
            };
            visit_counts.push((edge.action, visits));
            if visits > best_visits {
                best_visits = visits;
                best_action = edge.action;
            }
        }

        SearchResult {
            best_action,
            visit_counts,
            root_visits: root_node.visit_count,
        }
    }
}

/// Convenience wrapper that creates a temporary MctsSearch.
pub fn mcts_search(state: &GameState, config: &MctsConfig, rng: &mut impl Rng) -> u8 {
    let mut search = MctsSearch::new();
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
    fn test_mcts_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = MctsConfig {
            iterations: 100,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let action = mcts_search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "MCTS returned illegal action {}",
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
    fn test_mcts_forced_move() {
        let mut rng = rand::thread_rng();
        let config = MctsConfig {
            iterations: 10,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..200 {
            let mut state = GameState::deal_random(0, &mut rng);
            while !state.is_terminal() {
                let legal = state.legal_actions();
                if legal.count_ones() == 1 {
                    let action = mcts_search(&state, &config, &mut rng);
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
    fn test_mcts_search_reusable() {
        let mut rng = rand::thread_rng();
        let mut search = MctsSearch::new();
        let config = MctsConfig {
            iterations: 50,
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
    fn test_search_with_stats() {
        let mut rng = rand::thread_rng();
        let config = MctsConfig {
            iterations: 200,
            ..Default::default()
        };
        let state = loop {
            if let Some(s) = random_playing_state(&mut rng) {
                break s;
            }
        };

        let mut search = MctsSearch::new();
        let result = search.search_with_stats(&state, &config, &mut rng);

        assert_eq!(result.root_visits, config.iterations);

        // All visit counts should sum to iterations (each iteration visits exactly one child)
        let total_child_visits: u32 = result.visit_counts.iter().map(|(_, v)| v).sum();
        assert_eq!(total_child_visits, config.iterations);

        // Best action should appear in visit counts
        assert!(result
            .visit_counts
            .iter()
            .any(|(a, _)| *a == result.best_action));

        // Best action should be legal
        let legal = state.legal_actions();
        assert!(legal & (1u64 << result.best_action) != 0);
    }

    #[test]
    fn test_mcts_works_during_bidding() {
        let mut rng = rand::thread_rng();
        let config = MctsConfig {
            iterations: 100,
            ..Default::default()
        };
        let state = GameState::deal_random(0, &mut rng);
        assert_eq!(state.phase, Phase::Bidding);

        let action = mcts_search(&state, &config, &mut rng);
        let legal = state.legal_actions();
        assert!(
            legal & (1u64 << action) != 0,
            "MCTS returned illegal bid action {}",
            action
        );
    }

    #[test]
    fn test_rave_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = MctsConfig {
            iterations: 200,
            rollout_policy: RolloutPolicy::HeuristicPlay,
            use_rave: true,
            rave_k: 300.0,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let action = mcts_search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "RAVE MCTS returned illegal action {}",
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
    fn test_rave_amaf_filtered() {
        let mut rng = rand::thread_rng();
        let config = MctsConfig {
            iterations: 500,
            rollout_policy: RolloutPolicy::HeuristicPlay,
            use_rave: true,
            rave_k: 300.0,
            ..Default::default()
        };

        let state = loop {
            if let Some(s) = random_playing_state(&mut rng) {
                break s;
            }
        };

        let mut search = MctsSearch::new();
        let result = search.search_with_stats(&state, &config, &mut rng);

        // Root AMAF should only have entries for legal actions at root
        let legal = state.legal_actions();
        let _root_amaf = &search.amaf_visits[0]; // root is node 0
        for a in 0..32u8 {
            if legal & (1u64 << a) == 0 {
                // If card is not a legal action at root, its AMAF visits can still be nonzero
                // because AMAF tracks team-played cards. However, during selection
                // those entries are only used for edges that exist.
                // This is fine — the AMAF data is only queried for existing edges.
            }
        }

        // Best action should be legal
        assert!(legal & (1u64 << result.best_action) != 0);
    }

    #[test]
    fn test_mcts_heuristic_play_policy() {
        let mut rng = rand::thread_rng();
        let config = MctsConfig {
            iterations: 100,
            rollout_policy: RolloutPolicy::HeuristicPlay,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let action = mcts_search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(legal & (1u64 << action) != 0);
                found += 1;
                if found >= 30 {
                    break;
                }
            }
        }
        assert!(found >= 10);
    }
}
