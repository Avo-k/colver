//! IS-DD card player — the production agent.
//!
//! Wraps [`IsDdSearch`] with everything it needs to run on its own:
//!
//! - a [`WorldSource`] (playgen over the GPU sidecar by default),
//! - an optional belief net and credibility judges,
//! - its own RNG and per-deal state.
//!
//! The point of the wrapper is that **world generation lives inside the
//! agent**. Before it, the web server sampled playgen worlds and pushed them
//! into the search while the arena did not, so the two ran different agents
//! under the same name. Now a caller cannot get that wrong: it builds the
//! player and asks for a card.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

use crate::is_dd::{IsDdConfig, IsDdSearch};
use crate::state::GameState;
use crate::worlds::WorldSource;

use super::models::DmcWeights;
use super::{AgentError, CardPlayer, Decision, MatchContext, Stats};

pub struct IsDdPlayer {
    search: IsDdSearch,
    config: IsDdConfig,
    source: Option<Box<dyn WorldSource>>,
    seat: u8,
    rng: StdRng,
    /// Play only from this trick onward, deferring earlier tricks to `early`.
    /// IS-DD's edge is concentrated in the endgame, where the remaining tree is
    /// small enough for the solves to be near-exact; a fast net is a better use
    /// of the budget before that.
    switch_at: Option<u8>,
    early: Option<Box<dyn CardPlayer>>,
}

impl IsDdPlayer {
    pub fn new(config: IsDdConfig, seat: u8, seed: u64) -> Self {
        IsDdPlayer {
            search: IsDdSearch::new(),
            config,
            source: None,
            seat,
            rng: StdRng::seed_from_u64(seed),
            switch_at: None,
            early: None,
        }
    }

    /// Attach the source of determinized worlds. Without one, the search falls
    /// back to belief-weighted / constraint-uniform sampling — legal, cheap,
    /// and measurably weaker.
    pub fn with_world_source(mut self, source: Box<dyn WorldSource>) -> Self {
        self.source = Some(source);
        self
    }

    /// Hand the first `switch_at` tricks to `early` (typically a DMC net) and
    /// take over for the endgame.
    pub fn with_early_player(mut self, early: Box<dyn CardPlayer>, switch_at: u8) -> Self {
        self.early = Some(early);
        self.switch_at = Some(switch_at);
        self
    }

    pub fn load_belief_net(&mut self, path: &str) -> Result<(), AgentError> {
        self.search
            .load_belief_net(path)
            .map_err(|e| AgentError::Model(format!("{path}: {e}")))?;
        self.config.use_nn_beliefs = true;
        Ok(())
    }

    /// Bid net used to judge how credible a world's auction is (`cred_alpha`).
    pub fn load_cred_bid_net(&mut self, path: &str) -> Result<(), AgentError> {
        self.search
            .load_cred_bid_net(path)
            .map_err(|e| AgentError::Model(format!("{path}: {e}")))
    }

    /// Canonical DMC net used to judge how credible a world's play is.
    pub fn load_cred_play_net(&mut self, path: &str) -> Result<(), AgentError> {
        self.search
            .load_cred_play_net(path)
            .map_err(|e| AgentError::Model(format!("{path}: {e}")))
    }

    pub fn config(&self) -> &IsDdConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut IsDdConfig {
        &mut self.config
    }

    pub fn search_mut(&mut self) -> &mut IsDdSearch {
        &mut self.search
    }

    /// Name of the attached world source, or `"none"`.
    pub fn world_source_name(&self) -> &'static str {
        self.source.as_ref().map(|s| s.name()).unwrap_or("none")
    }

    fn use_early(&self, state: &GameState) -> bool {
        match self.switch_at {
            Some(t) => (state.tricks_won[0] + state.tricks_won[1]) < t,
            None => false,
        }
    }
}

impl CardPlayer for IsDdPlayer {
    fn init_deal(&mut self, state: &GameState) {
        self.search.init_deal_with_config(state, self.seat, &self.config);
        if let Some(src) = self.source.as_mut() {
            src.init_deal(state, self.seat);
        }
        if let Some(early) = self.early.as_mut() {
            early.init_deal(state);
        }
    }

    fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        self.search.record_action(state_before, player, action);
        if let Some(src) = self.source.as_mut() {
            src.observe(state_before, player, action);
        }
        if let Some(early) = self.early.as_mut() {
            early.observe(state_before, player, action);
        }
    }

    /// Only meaningful in time mode; in count mode the budget is deliberately
    /// unbounded and a clock would silently truncate the world count.
    fn set_time_budget(&mut self, ms: u32) {
        if self.config.time_limit_ms.is_some() {
            self.config.time_limit_ms = Some(ms);
        }
        if let Some(early) = self.early.as_mut() {
            early.set_time_budget(ms);
        }
    }

    fn decide(&mut self, state: &GameState, ctx: &MatchContext) -> Result<Decision, AgentError> {
        if self.use_early(state) {
            return self.early.as_mut().expect("switch_at implies an early player").decide(state, ctx);
        }

        let start = Instant::now();
        let result = match self.source.as_deref_mut() {
            Some(src) => {
                self.search.search_with_source(state, &self.config, &mut self.rng, src)?
            }
            None => self.search.search_with_stats(state, &self.config, &mut self.rng),
        };

        Ok(Decision {
            action: result.best_action,
            stats: Stats {
                source: "isdd",
                candidates: result.card_scores,
                determinizations: result.determinizations,
                worlds: result.worlds,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
            },
        })
    }
}

/// Build the DMC player that fronts an [`IsDdPlayer`] in `dmc_then_dd` bots.
pub fn early_dmc(weights: Arc<DmcWeights>, residual: bool) -> Box<dyn CardPlayer> {
    Box::new(super::dmc::DmcPlayer::new(weights, residual))
}
