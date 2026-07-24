//! IS-MCTS card players.
//!
//! Kept as baselines. IS-DD replaced them for real play — an exact DD solve on
//! a sampled world beats an approximate rollout tree on the same world — but
//! they are the reference every IS-DD number was first measured against, so
//! removing them would make old results unreproducible.

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::mcts::{MctsConfig, MctsSearch};
use crate::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use crate::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use crate::state::GameState;

use super::{AgentError, CardPlayer, Decision, MatchContext};

/// IS-MCTS with uniform determinization.
pub struct NaiveIsMctsPlayer {
    search: NaiveIsMctsSearch,
    config: NaiveIsMctsConfig,
    rng: StdRng,
}

impl NaiveIsMctsPlayer {
    pub fn new(config: NaiveIsMctsConfig, seed: u64) -> Self {
        NaiveIsMctsPlayer {
            search: NaiveIsMctsSearch::new(),
            config,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl CardPlayer for NaiveIsMctsPlayer {
    fn decide(&mut self, state: &GameState, _ctx: &MatchContext) -> Result<Decision, AgentError> {
        Ok(Decision::bare(self.search.search(state, &self.config, &mut self.rng), "naive_ismcts"))
    }
}

/// IS-MCTS with belief-weighted determinization.
pub struct SmartIsMctsPlayer {
    search: SmartIsMctsSearch,
    config: SmartIsMctsConfig,
    seat: u8,
    rng: StdRng,
}

impl SmartIsMctsPlayer {
    pub fn new(config: SmartIsMctsConfig, seat: u8, seed: u64) -> Self {
        SmartIsMctsPlayer {
            search: SmartIsMctsSearch::new(),
            config,
            seat,
            rng: StdRng::seed_from_u64(seed),
        }
    }
}

impl CardPlayer for SmartIsMctsPlayer {
    fn init_deal(&mut self, state: &GameState) {
        self.search.init_deal(state, self.seat, true);
    }

    fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        self.search.record_action(state_before, player, action);
    }

    fn decide(&mut self, state: &GameState, _ctx: &MatchContext) -> Result<Decision, AgentError> {
        Ok(Decision::bare(self.search.search(state, &self.config, &mut self.rng), "smart_ismcts"))
    }
}

/// Perfect-information MCTS: it sees all four hands. A ceiling, not a player.
pub struct OracleMctsPlayer {
    search: MctsSearch,
    config: MctsConfig,
    rng: StdRng,
}

impl OracleMctsPlayer {
    pub fn new(config: MctsConfig, seed: u64) -> Self {
        OracleMctsPlayer { search: MctsSearch::new(), config, rng: StdRng::seed_from_u64(seed) }
    }
}

impl CardPlayer for OracleMctsPlayer {
    fn decide(&mut self, state: &GameState, _ctx: &MatchContext) -> Result<Decision, AgentError> {
        Ok(Decision::bare(self.search.search(state, &self.config, &mut self.rng), "oracle_mcts"))
    }
}
