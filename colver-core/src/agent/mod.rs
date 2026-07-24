//! Agents: **one object per seat that knows how to play a whole deal**.
//!
//! # Why this module exists
//!
//! Before it, every caller (the arena, the web server, each tournament binary)
//! re-implemented the same dispatch: "if this seat uses IS-DD call `search`,
//! else if it uses DMC write the observation, convert the mask to canonical
//! space, call `best_action`, convert back…". Ten copies of that logic drifted
//! apart — most visibly, the web fed IS-DD worlds sampled from the playgen GPU
//! sidecar while the arena did not, so the arena was silently benchmarking a
//! different agent than the one in production.
//!
//! The fix is ordinary object orientation. A [`Player`] is a trait — the
//! equivalent of an abstract base class — and each strategy is one
//! implementation that **owns everything it needs**: its models, its RNG, its
//! per-deal state, and (for IS-DD) its source of determinized worlds. Callers
//! build `Box<dyn Player>` from an [`AgentSpec`] and then only ever say
//! "here is the position, give me an action".
//!
//! ```ignore
//! let mut player = AgentSpec::from_toml_file("arena/bots/champion.toml")?.build(seat)?;
//! player.init_deal(&state);
//! let action = player.action(&state, &ctx)?;
//! ```
//!
//! # Structure
//!
//! A player is usually a [`ComposedPlayer`]: a [`BidPolicy`] for the auction
//! plus a [`CardPlayer`] for the play phase, exactly mirroring the `[bid]` and
//! `[play]` sections of a bot TOML. Strategies that decide both phases
//! themselves implement [`Player`] directly.
//!
//! # Failure policy
//!
//! Decisions return `Result`. A configured-but-unreachable dependency (most
//! importantly the playgen world sidecar) is an [`AgentError`], **not** a
//! silent downgrade to weaker worlds: an agent that quietly changes strength
//! turns every measurement into a lie. Callers that genuinely prefer a
//! degraded answer to no answer opt in explicitly via
//! [`crate::worlds::FallbackPolicy`].

use std::fmt;

use crate::dmc_obs::EnvTracking;
use crate::is_dd::WorldCounts;
use crate::state::GameState;

pub mod bid;
pub mod dmc;
pub mod isdd;
pub mod ismcts;
pub mod models;
pub mod spec;

pub use bid::BidNetPolicy;
pub use dmc::DmcPlayer;
pub use isdd::IsDdPlayer;
pub use spec::{AgentSpec, BidSpec, PlaySpec};
pub use crate::worlds::{
    FallbackPolicy, LocalPlaygenSource, SidecarWorldSource, UniformWorldSource, WorldSource,
};

// ══════════════════════════════════════════════════════════════════════
//  Errors
// ══════════════════════════════════════════════════════════════════════

/// Anything that can stop an agent from producing a decision.
#[derive(Debug, Clone)]
pub enum AgentError {
    /// A model file is missing, unreadable, or has an unexpected shape.
    Model(String),
    /// A configured world source failed and the policy is not to degrade.
    WorldSource(String),
    /// The spec asked for something this build cannot do.
    Unsupported(String),
    /// The spec is internally inconsistent (e.g. `strategy = "nn"` with no model).
    Config(String),
}

impl fmt::Display for AgentError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            AgentError::Model(m) => write!(f, "model error: {m}"),
            AgentError::WorldSource(m) => write!(f, "world source error: {m}"),
            AgentError::Unsupported(m) => write!(f, "unsupported: {m}"),
            AgentError::Config(m) => write!(f, "config error: {m}"),
        }
    }
}

impl std::error::Error for AgentError {}

// ══════════════════════════════════════════════════════════════════════
//  Context and decisions
// ══════════════════════════════════════════════════════════════════════

/// Everything a player needs about the game beyond the current `GameState`:
/// public action tracking (bid history, play order, voids) and the running
/// match score, which score-aware bidders condition on.
///
/// One context is shared by all four seats and maintained by the driver
/// ([`crate::game_loop`], or the PyO3 `Env` for the web).
#[derive(Clone)]
pub struct MatchContext {
    pub tracking: EnvTracking,
    /// Cumulative match points per team, `[NS, EW]`. Zero outside match play.
    pub scores: [i32; 2],
}

impl MatchContext {
    pub fn new(dealer: u8) -> Self {
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);
        MatchContext { tracking, scores: [0, 0] }
    }

    /// Start a new deal, keeping the cumulative match score.
    pub fn reset_deal(&mut self, dealer: u8) {
        self.tracking.reset(dealer);
    }

    /// Record an action that is about to be applied to `state_before`.
    pub fn track(&mut self, state_before: &GameState, action: u8) {
        self.tracking.track_action(state_before, action);
    }

    /// Cumulative match score as `(mine, theirs)` for the given seat.
    pub fn scores_for(&self, seat: u8) -> (i32, i32) {
        let team = GameState::player_team(seat) as usize;
        (self.scores[team], self.scores[1 - team])
    }
}

/// A chosen action plus whatever the strategy can say about how it got there.
#[derive(Clone)]
pub struct Decision {
    pub action: u8,
    pub stats: Stats,
}

impl Decision {
    /// A decision with no introspection to offer (forced move, plain heuristic).
    pub fn bare(action: u8, source: &'static str) -> Self {
        Decision { action, stats: Stats { source, ..Stats::default() } }
    }
}

/// Introspection about one decision. Fields that do not apply to a strategy
/// keep their zero value — the web reads this to draw its analysis panels and
/// the arena ignores it.
#[derive(Clone, Debug, Default)]
pub struct Stats {
    /// Short tag for the strategy that decided, e.g. `"isdd"`, `"dmc"`, `"bid_nn"`.
    pub source: &'static str,
    /// `(action, score)` for every candidate considered. Scores are on the
    /// strategy's own scale: NS deal points for IS-DD, Q-values for the nets.
    /// Empty when the strategy does not rank candidates.
    pub candidates: Vec<(u8, f32)>,
    /// IS-DD: worlds actually solved.
    pub determinizations: u32,
    /// IS-DD: where those worlds came from.
    pub worlds: WorldCounts,
    /// Wall-clock time spent deciding.
    pub elapsed_ms: f64,
}

// ══════════════════════════════════════════════════════════════════════
//  Traits
// ══════════════════════════════════════════════════════════════════════

/// A seated player for a full deal — bidding and play.
///
/// Implementations own their models and their per-deal state, so a caller
/// never has to know which strategy is behind the trait object. The lifecycle
/// is: [`init_deal`](Player::init_deal) once per deal, then for every action of
/// **every** seat [`observe`](Player::observe), with
/// [`decide`](Player::decide) called on the seat to move.
pub trait Player: Send {
    /// Human-readable identity, used in logs and result tables.
    fn label(&self) -> &str;

    /// The seat this player occupies (0=N, 1=E, 2=S, 3=W).
    fn seat(&self) -> u8;

    /// Start a new deal. `state` is the freshly dealt, pre-auction position;
    /// only this player's own hand may be read from it.
    fn init_deal(&mut self, state: &GameState);

    /// Observe an action by any seat, including this one. `state_before` is
    /// the position **before** the action is applied.
    fn observe(&mut self, state_before: &GameState, player: u8, action: u8);

    /// Choose an action for this player's seat.
    fn decide(&mut self, state: &GameState, ctx: &MatchContext) -> Result<Decision, AgentError>;

    /// Convenience wrapper that discards the introspection.
    fn action(&mut self, state: &GameState, ctx: &MatchContext) -> Result<u8, AgentError> {
        Ok(self.decide(state, ctx)?.action)
    }

    /// Retune the per-move time budget mid-game, for strategies that have one.
    /// The web's speed slider is the reason this exists: rebuilding the agent
    /// instead would throw away its belief state mid-deal.
    fn set_time_budget(&mut self, _ms: u32) {}
}

/// The auction half of a [`ComposedPlayer`].
pub trait BidPolicy: Send {
    fn init_deal(&mut self, _state: &GameState) {}
    fn observe(&mut self, _state_before: &GameState, _player: u8, _action: u8) {}
    fn decide(&mut self, state: &GameState, ctx: &MatchContext) -> Result<Decision, AgentError>;
}

/// The play half of a [`ComposedPlayer`].
pub trait CardPlayer: Send {
    fn init_deal(&mut self, _state: &GameState) {}
    fn observe(&mut self, _state_before: &GameState, _player: u8, _action: u8) {}
    fn decide(&mut self, state: &GameState, ctx: &MatchContext) -> Result<Decision, AgentError>;
    /// See [`Player::set_time_budget`].
    fn set_time_budget(&mut self, _ms: u32) {}
}

/// A player assembled from an independent bidder and card player — the shape
/// every bot TOML describes with its `[bid]` and `[play]` sections.
pub struct ComposedPlayer {
    label: String,
    seat: u8,
    bid: Box<dyn BidPolicy>,
    play: Box<dyn CardPlayer>,
}

impl ComposedPlayer {
    pub fn new(
        label: impl Into<String>,
        seat: u8,
        bid: Box<dyn BidPolicy>,
        play: Box<dyn CardPlayer>,
    ) -> Self {
        ComposedPlayer { label: label.into(), seat, bid, play }
    }
}

impl Player for ComposedPlayer {
    fn label(&self) -> &str {
        &self.label
    }

    fn seat(&self) -> u8 {
        self.seat
    }

    fn init_deal(&mut self, state: &GameState) {
        self.bid.init_deal(state);
        self.play.init_deal(state);
    }

    fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        self.bid.observe(state_before, player, action);
        self.play.observe(state_before, player, action);
    }

    fn decide(&mut self, state: &GameState, ctx: &MatchContext) -> Result<Decision, AgentError> {
        match state.phase {
            crate::state::Phase::Bidding => self.bid.decide(state, ctx),
            _ => self.play.decide(state, ctx),
        }
    }

    fn set_time_budget(&mut self, ms: u32) {
        self.play.set_time_budget(ms);
    }
}
