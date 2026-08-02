//! DMC card player: a DouZero-style Q-network over the 32 cards.
//!
//! # The canonical-observation trap
//!
//! Two observation layouts exist and they are **not** interchangeable:
//!
//! - **415-dim (legacy)** — suits in physical order, trump as a one-hot.
//! - **411-dim (canonical)** — suits *relabelled* so trump is always slot 0 and
//!   the others are sorted by length and rank pattern. No trump one-hot is
//!   needed because the layout itself encodes it.
//!
//! With a canonical model, the legal-action mask must be translated into
//! canonical space before the forward pass and the chosen action translated
//! back to a physical card afterwards. Skip either translation and the model
//! still returns a *legal* card — just an unrelated one. Nothing crashes; the
//! bot simply plays like a random legal player, which is easy to mistake for a
//! bad model.
//!
//! This type does the detection (from the weight file, via
//! [`crate::agent::models`]) and both translations, once, so no caller has to
//! remember.

use std::sync::Arc;

use crate::dmc_net::DmcNet;
use crate::dmc_obs::{self, OBS_DIM_TR};
use crate::state::GameState;

use super::models::DmcWeights;
use super::{AgentError, CardPlayer, Decision, MatchContext, Stats};

pub struct DmcPlayer {
    net: DmcNet,
    obs: Vec<f32>,
    /// True when the net expects the 411-dim canonical layout.
    canonical: bool,
}

impl DmcPlayer {
    pub fn new(weights: Arc<DmcWeights>, residual: bool) -> Self {
        let net = weights.instantiate(residual);
        let obs = vec![0.0f32; net.obs_dim()];
        let canonical = net.obs_dim() == OBS_DIM_TR;
        DmcPlayer { net, obs, canonical }
    }

    /// Q-value of every legal card, in **physical** card space.
    pub fn q_values(&mut self, state: &GameState, ctx: &MatchContext) -> Vec<(u8, f32)> {
        self.evaluate(state, ctx).1
    }

    fn evaluate(&mut self, state: &GameState, ctx: &MatchContext) -> (u8, Vec<(u8, f32)>) {
        let legal = state.legal_actions() as u32;
        if self.canonical {
            dmc_obs::write_observation_tr(&mut self.obs, 0, state, &ctx.tracking);
            let order = dmc_obs::current_player_order(state, &ctx.tracking);
            let mask = dmc_obs::cardset_to_canonical(legal, &order);
            let (best, q) = self.net.best_action(&self.obs, mask);
            let q = q
                .into_iter()
                .map(|(c, v)| (dmc_obs::card_to_physical(c, &order), v))
                .collect();
            (dmc_obs::card_to_physical(best, &order), q)
        } else {
            dmc_obs::write_observation(&mut self.obs, 0, state, &ctx.tracking);
            self.net.best_action(&self.obs, legal)
        }
    }
}

impl CardPlayer for DmcPlayer {
    fn decide(&mut self, state: &GameState, ctx: &MatchContext) -> Result<Decision, AgentError> {
        let (action, candidates) = self.evaluate(state, ctx);
        Ok(Decision {
            action,
            stats: Stats { source: "dmc", candidates, ..Stats::default() },
        })
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Simple card players
// ══════════════════════════════════════════════════════════════════════

/// Full-information heuristic — it reads all four hands. Useful as a rollout
/// policy and as a sparring partner, **not** as a fair opponent.
pub struct HeuristicPlayer;

impl CardPlayer for HeuristicPlayer {
    fn decide(&mut self, state: &GameState, _ctx: &MatchContext) -> Result<Decision, AgentError> {
        Ok(Decision::bare(crate::rollout::heuristic_play_action(state), "heuristic"))
    }
}

/// Hidden-information rule player: decides from its own hand, the trick, the
/// played cards and known voids only. The fair rule-based baseline.
pub struct RulePlayer;

impl CardPlayer for RulePlayer {
    fn decide(&mut self, state: &GameState, _ctx: &MatchContext) -> Result<Decision, AgentError> {
        Ok(Decision::bare(crate::rule_player::rule_play_action(state), "rule"))
    }
}

/// Exact double-dummy solver with all four hands visible. The ceiling, not a
/// player: it cheats by construction and exists to bound how much strength is
/// left on the table.
#[derive(Default)]
pub struct OraclePlayer {
    /// Kept across cards: `solve_with_scores(state, None)` allocated and zeroed a fresh 2 MB
    /// table on every single decision.
    tt: Option<crate::solver::TtBuf>,
}

impl CardPlayer for OraclePlayer {
    fn decide(&mut self, state: &GameState, _ctx: &MatchContext) -> Result<Decision, AgentError> {
        let tt = self.tt.get_or_insert_with(crate::solver::new_tt_buffer);
        // One search, not two: `solve_with_scores` already reports the best card. The old code
        // followed it with `solve_best_card`, which re-ran the identical tree from scratch on
        // its own fresh table — an exact doubling of the cost of every oracle move.
        let scores = crate::solver::solve_with_scores(state, Some(tt));
        let candidates: Vec<(u8, f32)> =
            (0..scores.count).map(|i| (scores.scores[i].0, scores.scores[i].1 as f32)).collect();
        Ok(Decision {
            action: scores.best_card,
            stats: Stats { source: "oracle", candidates, ..Stats::default() },
        })
    }
}
