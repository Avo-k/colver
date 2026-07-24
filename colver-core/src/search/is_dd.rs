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
use crate::belief_obs::{self, BELIEF_OBS_DIM, BELIEF_OBS_DIM_V2, BELIEF_OBS_DIM_V3};
use crate::bid_eval::BidFunction;
use crate::card::card_count;
use crate::card_beliefs::CardBeliefs;
use crate::determinize::{determinize_greedy, determinize_weighted};
use crate::dmc_net::DmcNet;
use crate::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use crate::worlds::{World, WorldSource};
use crate::solver::{new_tt_buffer, solve_with_scores};
use crate::state::{GameState, Phase};

/// Configuration for IS-DD search.
///
/// **Hard constraints** (voids, trump ceiling, played cards) are facts, not beliefs:
/// they are always applied unconditionally and not exposed as a flag.
///
/// **Soft beliefs** (heuristic soft inference, NN beliefs) are **off by default** —
/// they introduce probabilistic adjustments that may help or hurt depending on
/// the opponents and the play model.
pub struct IsDdConfig {
    /// Number of determinized worlds to sample (default 20).
    pub determinizations: u32,
    /// Whether to use soft (probabilistic) heuristic inference from play (dominance,
    /// "ne pisse pas", etc.) in addition to hard constraints. **Off by default.**
    pub use_soft_inference: bool,
    /// Optional time limit in milliseconds (overrides `determinizations` count).
    pub time_limit_ms: Option<u32>,
    /// Which bid function to use during bidding phase.
    pub bid_function: BidFunction,
    /// If true and a BeliefNet is loaded, use NN soft beliefs (still combined with
    /// hard constraints, which are always applied). **Off by default.**
    pub use_nn_beliefs: bool,
    /// Play dominance inference factor for CardBeliefs.
    /// When a player follows suit without playing the highest, reduce weight for
    /// higher unknown cards by this factor. 1.0 = off, 0.3 = aggressive. Default 1.0.
    pub dominance_factor: f32,
    /// If true (default), skip search when only 1 legal action or position is fully resolved.
    pub early_termination: bool,
    /// How many worlds to request per [`WorldSource`] refill when running
    /// under a time budget (in count mode the whole remaining budget is asked
    /// for at once). One refill is one GPU round trip for the sidecar, so this
    /// trades latency granularity against overhead. Default 128.
    pub world_batch: usize,
    /// Fallback pool: when no [`WorldSource`] is attached (or it runs dry),
    /// the fraction of worlds drawn with belief weights rather than
    /// constraint-uniform. Only has an effect when a belief source is active.
    /// Default 1.0.
    pub belief_frac: f32,
    /// Credibility importance weighting of worlds in the DD aggregation. Each
    /// world's weight is the product of per-action rank factors — "would the
    /// reference policy replay the observed hidden action holding this world's
    /// hand?" — flattened by this exponent. Judges both phases when the
    /// corresponding net is loaded: the **auction** via the bid net
    /// (`load_cred_bid_net`) and the **play** via the DMC net
    /// (`load_cred_play_net`). 0.0 = off (default); 0.5 = recommended soft
    /// weighting. See [`IsDdSearch::credibility_weight`] for the mechanism.
    pub cred_alpha: f32,
    /// Solve the determinized worlds in parallel (rayon global pool) instead of
    /// sequentially. World *generation* is always sequential (the world source
    /// and RNG are stateful); only the embarrassingly-parallel DD
    /// solves are spread across threads, each with its own transposition table.
    /// Results are identical to the sequential path (DD is deterministic and the
    /// aggregation reduces in a fixed order). Requires the `parallel` cargo
    /// feature — ignored (falls back to sequential) when it is not compiled in.
    /// **Off by default**; the web/PyO3 layer turns it on for per-move latency.
    pub parallel: bool,
}

impl Default for IsDdConfig {
    fn default() -> Self {
        IsDdConfig {
            determinizations: 20,
            // All soft beliefs OFF by default. Hard constraints are facts, always applied.
            use_soft_inference: false,
            time_limit_ms: None,
            bid_function: BidFunction::ImprovedV2,
            use_nn_beliefs: false,
            dominance_factor: 1.0,
            early_termination: true,
            world_batch: 128,
            belief_frac: 1.0,
            cred_alpha: 0.0,
            parallel: false,
        }
    }
}

/// Derive V3 temporal features (trick lead suits, trick winners, suit-fail
/// counts relative to observer) from public play tracking. Mirrors the
/// training-time extraction in `game_replay.rs`.
fn derive_v3_temporal(
    state: &GameState,
    tracking: &EnvTracking,
    observer: u8,
) -> (Vec<u8>, Vec<u8>, [[u8; 4]; 3]) {
    use crate::card::{card_suit_u8, EMPTY};
    use crate::trick::trick_winner;

    let completed = tracking.play_order.len() / 4;
    let mut leads = Vec::with_capacity(8);
    let mut winners = Vec::with_capacity(8);
    let mut fails_abs = [[0u8; 4]; 4];

    for t in 0..completed {
        let base = t * 4;
        let c0 = tracking.play_order[base];
        let mut lead_seat = 0u8;
        for p in 0..4u8 {
            if tracking.played_by[p as usize] & (1u32 << c0) != 0 {
                lead_seat = p;
                break;
            }
        }
        let lead_suit = card_suit_u8(c0);
        leads.push(lead_suit);

        let mut trick_cards = [EMPTY; 4];
        for j in 0..4usize {
            let cj = tracking.play_order[base + j];
            trick_cards[(lead_seat as usize + j) % 4] = cj;
        }
        winners.push(trick_winner(&trick_cards, lead_seat, &state.contract));

        for j in 1..4usize {
            let cj = tracking.play_order[base + j];
            if card_suit_u8(cj) != lead_suit {
                let pj = (lead_seat as usize + j) % 4;
                fails_abs[pj][lead_suit as usize] =
                    fails_abs[pj][lead_suit as usize].saturating_add(1);
            }
        }
    }

    let rel_seats = [
        ((observer as usize + 1) % 4),
        ((observer as usize + 2) % 4),
        ((observer as usize + 3) % 4),
    ];
    let mut fail_rel = [[0u8; 4]; 3];
    for (i, &seat) in rel_seats.iter().enumerate() {
        fail_rel[i] = fails_abs[seat];
    }
    (leads, winners, fail_rel)
}

/// Credibility rank factor: how much to trust a world given that the reference
/// policy ranks `better` legal moves strictly above the one actually observed.
/// Argmax (0) is fully credible; a top-3 move is mildly discounted; anything
/// worse is heavily discounted. Shared by the auction and play judges.
#[inline]
fn rank_factor(better: u32) -> f32 {
    match better {
        0 => 1.0,
        1 | 2 => 0.7,
        _ => 0.35,
    }
}

/// Where a determinized world came from. The ensemble policy in
/// [`IsDdSearch::generate_world`] tries these in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldOrigin {
    /// Supplied by the caller via [`IsDdSearch::set_injected_worlds`] (e.g. the
    /// playgen GPU sidecar).
    Injected,
    /// Sampled in-process from the playgen transformer.
    Playgen,
    /// Belief-weighted determinization (NN or heuristic weights).
    Belief,
    /// Constraint-uniform determinization — the coverage floor.
    Uniform,
}

/// How many solved worlds came from each source. Reported so a degraded run
/// (e.g. a playgen sidecar that stopped answering) is visible in the stats
/// instead of silently changing the agent's strength.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldCounts {
    pub injected: u32,
    pub playgen: u32,
    pub belief: u32,
    pub uniform: u32,
}

impl WorldCounts {
    #[inline]
    fn record(&mut self, origin: WorldOrigin) {
        match origin {
            WorldOrigin::Injected => self.injected += 1,
            WorldOrigin::Playgen => self.playgen += 1,
            WorldOrigin::Belief => self.belief += 1,
            WorldOrigin::Uniform => self.uniform += 1,
        }
    }

    pub fn total(&self) -> u32 {
        self.injected + self.playgen + self.belief + self.uniform
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
    /// Provenance of those determinizations.
    pub worlds: WorldCounts,
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
    /// Bid net used as an auction-credibility judge (see `cred_alpha`).
    cred_bid_net: Option<crate::bid_net::BidNet>,
    /// DMC net used as a play-credibility judge (see `cred_alpha`). Canonical
    /// (411-dim, `OBS_DIM_TR`) obs only — mirrors `bench_world_cred`.
    cred_play_net: Option<DmcNet>,
    /// Observed auction this deal: (bidder, action) in order.
    auction: Vec<(u8, u8)>,
    /// Observed plays this deal: (player, card) in order. Together with
    /// `auction` this is the full replayable history used by the credibility
    /// judge to reconstruct each hidden decision point.
    plays: Vec<(u8, u8)>,
    /// State at deal start (pre-auction), for credibility replays.
    init_state: Option<GameState>,
    /// Cards played so far per seat (current trick included).
    played_by: [u32; 4],
    /// Worlds pulled from the [`WorldSource`] for the position currently
    /// being searched, not yet solved. Refilled on demand and dropped when the
    /// search ends, so a later search at another position cannot consume
    /// worlds sampled for the previous one.
    world_queue: Vec<World>,
    tt_buf: Vec<u64>,
}

impl IsDdSearch {
    pub fn new() -> Self {
        IsDdSearch {
            beliefs: None,
            belief_net: None,
            belief_tracking: None,
            cred_bid_net: None,
            cred_play_net: None,
            auction: Vec::new(),
            plays: Vec::new(),
            init_state: None,
            played_by: [0; 4],
            world_queue: Vec::new(),
            tt_buf: new_tt_buffer(),
        }
    }

    /// Load the bid net used as the auction-credibility judge (`cred_alpha`).
    pub fn load_cred_bid_net(&mut self, path: &str) -> std::io::Result<()> {
        self.cred_bid_net = Some(crate::bid_net::BidNet::load(path)?);
        Ok(())
    }

    /// Load the DMC net used as the play-credibility judge (`cred_alpha`).
    /// Must be a canonical (411-dim, `OBS_DIM_TR`) model.
    pub fn load_cred_play_net(&mut self, path: &str) -> std::io::Result<()> {
        let net = DmcNet::load(path)?;
        if net.obs_dim() != OBS_DIM_TR {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "play-credibility judge must be a canonical (411-dim) DMC model",
            ));
        }
        self.cred_play_net = Some(net);
        Ok(())
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

        // Credibility judge: remember the pre-auction state, reset the logs.
        self.auction.clear();
        self.plays.clear();
        self.init_state = Some(*state);
        self.played_by = [0; 4];
        self.world_queue.clear();

    }

    /// Initialize beliefs for a new deal, applying the config's belief knobs.
    pub fn init_deal_with_config(
        &mut self,
        state: &GameState,
        observer: u8,
        config: &IsDdConfig,
    ) {
        self.init_deal(state, observer, config.use_soft_inference);
        // Set dominance factor on beliefs.
        if let Some(ref mut beliefs) = self.beliefs {
            beliefs.dominance_factor = config.dominance_factor;
        }
    }

    /// Record an action by any player, updating beliefs and the world source's
    /// view of the history.
    ///
    /// `state_before` is the state BEFORE the action was applied.
    pub fn record_action(&mut self, state_before: &GameState, player: u8, action: u8) {
        if let Some(beliefs) = &mut self.beliefs {
            beliefs.record_action(state_before, player, action);
        }
        if let Some(tracking) = &mut self.belief_tracking {
            tracking.track_action(state_before, action);
        }
        if state_before.phase == Phase::Bidding {
            self.auction.push((player, action));
        }
        if state_before.phase == Phase::Playing {
            self.played_by[player as usize] |= 1u32 << action;
            self.plays.push((player, action));
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
        let base_weights = if config.use_nn_beliefs && self.belief_net.is_some() {
            let net = self.belief_net.as_mut().unwrap();
            let tracking = self.belief_tracking.as_ref().unwrap();

            // Hard constraints from CardBeliefs (shared by V2/V3 obs)
            let make_hc = |beliefs: &Option<CardBeliefs>| -> [f32; 96] {
                if let Some(beliefs) = beliefs {
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
                }
            };

            let logits = if net.obs_dim() == BELIEF_OBS_DIM_V2 {
                let hard_constraints = make_hc(&self.beliefs);
                let mut obs_buf = [0.0f32; BELIEF_OBS_DIM_V2];
                belief_obs::write_belief_observation_v2(
                    &mut obs_buf, 0, state, tracking, observer, &hard_constraints,
                );
                net.evaluate(&obs_buf)
            } else if net.obs_dim() == BELIEF_OBS_DIM_V3 {
                let hard_constraints = make_hc(&self.beliefs);
                let (trick_leads, trick_winners, suit_fail_rel) =
                    derive_v3_temporal(state, tracking, observer);
                let mut obs_buf = [0.0f32; BELIEF_OBS_DIM_V3];
                belief_obs::write_belief_observation_v3(
                    &mut obs_buf, 0, state, tracking, observer,
                    &hard_constraints, &trick_leads, &trick_winners, &suit_fail_rel,
                );
                net.evaluate(&obs_buf)
            } else {
                let mut obs_buf = [0.0f32; BELIEF_OBS_DIM];
                belief_obs::write_belief_observation(&mut obs_buf, 0, state, tracking, observer);
                net.evaluate(&obs_buf)
            };
            let mut nn_weights = crate::belief_net::belief_to_weights(&logits, net.num_classes(), state, observer);

            // Hard constraints (voids, trump ceiling, played cards) are facts, not beliefs.
            // Always apply them on top of NN soft predictions.
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

            Some(nn_weights)
        } else {
            self.beliefs.as_ref().map(|b| b.normalized_weights())
        };

        base_weights
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

    /// Credibility weight of a world: replay the observed history holding this
    /// world's reconstructed full hands and, for each action taken by a hidden
    /// player (`p != observer`), ask the reference policy whether it would
    /// replay that action. Both phases are judged when the corresponding net is
    /// loaded — **bids** by the bid net (`load_cred_bid_net`), **plays** by the
    /// canonical DMC net (`load_cred_play_net`). Each judged action contributes
    /// a rank factor by how many legal moves the net ranks strictly above the
    /// observed one:
    ///
    /// | net rates above observed | factor |
    /// |--------------------------|--------|
    /// | 0 (it *is* the argmax)   | 1.00   |
    /// | 1–2 (top-3)              | 0.70   |
    /// | ≥3                       | 0.35   |
    ///
    /// Factors multiply across judged actions; the product is flattened by
    /// `alpha` (`w.powf(alpha)`). Returns 1.0 when `alpha <= 0`, no judge is
    /// loaded, or the world cannot be reconstructed into four 8-card hands.
    ///
    /// Cost: one bid-net eval per hidden bid + one DMC eval per hidden play,
    /// per world. Negligible for the auction (~4–8 bids); the play path scales
    /// with tricks played, so keep world counts modest when it is enabled.
    fn credibility_weight(&mut self, world_hands: &[u32; 4], observer: u8, alpha: f32) -> f32 {
        if alpha <= 0.0 {
            return 1.0;
        }
        let Some(base) = self.init_state else { return 1.0 };
        if self.cred_bid_net.is_none() && self.cred_play_net.is_none() {
            return 1.0;
        }

        // Reconstruct full initial hands: the determinized world only assigns
        // cards still in hand, so add back what each seat has already played.
        let mut init_hands = [0u32; 4];
        for p in 0..4usize {
            init_hands[p] = world_hands[p] | self.played_by[p];
            if card_count(init_hands[p]) != 8 {
                return 1.0; // inconsistent reconstruction — skip weighting
            }
        }

        let mut s = base;
        s.hands = init_hands;
        // Replayed public tracking, needed for the canonical DMC play obs.
        let mut tracking = EnvTracking::new();
        tracking.reset(base.dealer);
        let mut w = 1.0f32;

        // --- Auction: judge each hidden bid with the bid net. ---
        let mut bid_hist: Vec<(u8, u8)> = Vec::with_capacity(self.auction.len());
        let mut bid_obs_buf = self
            .cred_bid_net
            .as_ref()
            .map(|n| vec![0.0f32; n.obs_dim()])
            .unwrap_or_default();
        for i in 0..self.auction.len() {
            let (p, a) = self.auction[i];
            if p != observer && s.phase == Phase::Bidding {
                if let Some(net) = self.cred_bid_net.as_mut() {
                    use crate::bid_obs::{
                        self, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2,
                        BID_OBS_DIM_SCORE_AWARE_V3,
                    };
                    // Standalone-deal convention: cumulative scores 0-0.
                    match net.obs_dim() {
                        BID_OBS_DIM_SCORE_AWARE_V3 => bid_obs::write_bid_observation_score_aware_v3(
                            &mut bid_obs_buf, 0, &s, &bid_hist, 0, 0,
                        ),
                        BID_OBS_DIM_SCORE_AWARE_V2 => bid_obs::write_bid_observation_score_aware_v2(
                            &mut bid_obs_buf, 0, &s, &bid_hist, 0, 0,
                        ),
                        BID_OBS_DIM_SCORE_AWARE => bid_obs::write_bid_observation_score_aware(
                            &mut bid_obs_buf, 0, &s, &bid_hist, 0, 0,
                        ),
                        _ => bid_obs::write_bid_observation(&mut bid_obs_buf, 0, &s, &bid_hist),
                    }
                    let q = net.evaluate(&bid_obs_buf);
                    let legal = s.legal_actions();
                    let mut better = 0u32;
                    let qa = q[a as usize];
                    for c in 0..43u8 {
                        if c != a && legal & (1u64 << c) != 0 && q[c as usize] > qa {
                            better += 1;
                        }
                    }
                    w *= rank_factor(better);
                }
            }
            bid_hist.push((p, a));
            tracking.track_action(&s, a);
            s.step(a);
        }

        // --- Play: judge each hidden play with the canonical DMC net. ---
        if self.cred_play_net.is_some() {
            for i in 0..self.plays.len() {
                let (p, a) = self.plays[i];
                if p != observer && s.phase == Phase::Playing {
                    let net = self.cred_play_net.as_mut().unwrap();
                    let obs = dmc_obs::make_observation_tr(&s, &tracking);
                    let order = dmc_obs::current_player_order(&s, &tracking);
                    let mask = dmc_obs::cardset_to_canonical(s.legal_actions() as u32, &order);
                    let q = net.evaluate(&obs);
                    let ca = dmc_obs::card_to_canonical(a, &order);
                    let qa = q[ca as usize];
                    let mut better = 0u32;
                    for c in 0..32u8 {
                        if c != ca && mask & (1u32 << c) != 0 && q[c as usize] > qa {
                            better += 1;
                        }
                    }
                    w *= rank_factor(better);
                }
                tracking.track_action(&s, a);
                s.step(a);
            }
        }

        w.powf(alpha)
    }

    /// Sample determinized play-phase worlds without solving them (used by the
    /// world-credibility benchmark). `use_beliefs` follows the same path as
    /// `search` (NN soft beliefs + hard constraints when a net is loaded);
    /// otherwise constraint-uniform sampling. Returns remaining-card hands.
    pub fn sample_worlds(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        observer: u8,
        n_worlds: usize,
        use_beliefs: bool,
        rng: &mut impl Rng,
    ) -> Vec<[u32; 4]> {
        let weights = if use_beliefs {
            self.compute_weights(state, config, observer)
        } else {
            None
        };
        (0..n_worlds)
            .filter_map(|_| match &weights {
                Some(w) => determinize_weighted(state, observer, w, rng)
                    .or_else(|| determinize_greedy(state, observer, rng)),
                None => determinize_greedy(state, observer, rng),
            })
            .map(|s| s.hands)
            .collect()
    }

    /// Heuristic `CardBeliefs` weights, before any NN blending.
    pub fn base_belief_weights(&self) -> Option<[[f32; 32]; 4]> {
        self.beliefs.as_ref().map(|b| b.normalized_weights())
    }

    /// Check if beliefs uniquely determine every unknown card's owner.
    /// If so, return the fully resolved GameState (perfect information) for direct DD solve.
    fn try_resolve_position(&self, state: &GameState, observer: u8) -> Option<GameState> {
        let beliefs = self.beliefs.as_ref()?;
        let raw = beliefs.raw_weights();

        let mut played = state.played_cards;
        for i in 0..4 {
            let c = state.current_trick[i];
            if c != crate::card::EMPTY {
                played |= 1u32 << c;
            }
        }
        let known = state.hands[observer as usize] | played;
        let unknown = crate::card::ALL_CARDS ^ known;

        if unknown == 0 {
            return Some(*state); // All cards already known.
        }

        let mut hands = [0u32; 4];
        hands[observer as usize] = state.hands[observer as usize];

        for card in crate::card::CardIter(unknown) {
            let mut owner: Option<u8> = None;
            for p in 0..4u8 {
                if p == observer {
                    continue;
                }
                if raw[p as usize][card as usize] > 0.0 {
                    if owner.is_some() {
                        return None; // Multiple candidates — not resolved.
                    }
                    owner = Some(p);
                }
            }
            let p = owner?;
            hands[p as usize] |= 1u32 << card;
        }

        // Verify card counts match original state.
        for p in 0..4u8 {
            if card_count(hands[p as usize]) != card_count(state.hands[p as usize]) {
                return None;
            }
        }

        let mut resolved = *state;
        resolved.hands = hands;
        Some(resolved)
    }

    /// Best card, sampling worlds from beliefs / constraint-uniform only.
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

    /// Full result, sampling worlds from beliefs / constraint-uniform only.
    ///
    /// Infallible: without a [`WorldSource`] there is nothing that can fail.
    /// Use [`search_with_source`](Self::search_with_source) to draw worlds from
    /// a playgen sampler — that is the strong configuration, and the one
    /// production uses.
    pub fn search_with_stats(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
    ) -> IsDdResult {
        self.run_search(state, config, rng, None)
            .expect("search without a world source cannot fail")
    }

    /// Full result, drawing determinized worlds from `source`.
    ///
    /// Worlds are pulled in batches and refilled on demand until the
    /// determinization count or the time budget is exhausted. If the source
    /// errors, the error propagates: a search that silently continued on
    /// constraint-uniform worlds would be a measurably weaker agent wearing
    /// the same name. A source that legitimately runs *dry* (returns an empty
    /// batch without erroring, as happens in over-constrained endgames) is not
    /// an error — the search falls back to its own sampling and reports the
    /// mix in [`IsDdResult::worlds`].
    pub fn search_with_source(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
        source: &mut dyn WorldSource,
    ) -> Result<IsDdResult, crate::agent::AgentError> {
        self.run_search(state, config, rng, Some(source))
    }

    fn run_search(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
        mut source: Option<&mut dyn WorldSource>,
    ) -> Result<IsDdResult, crate::agent::AgentError> {
        debug_assert!(!state.is_terminal(), "Cannot search from terminal state");

        let observer = state.current_player();
        let team = GameState::player_team(observer);
        let maximizing = team == 0; // NS maximizes, EW minimizes

        if config.early_termination {
            // Forced move: only 1 legal action — skip search entirely.
            let legal = state.legal_actions();
            if legal.count_ones() == 1 {
                let card = legal.trailing_zeros() as u8;
                return Ok(IsDdResult {
                    best_action: card,
                    card_scores: vec![(card, 81.0)],
                    determinizations: 0,
                    worlds: WorldCounts::default(),
                });
            }

            // Resolved position: beliefs uniquely determine all card locations.
            // Single DD solve gives the exact answer — no determinization needed.
            if let Some(resolved) = self.try_resolve_position(state, observer) {
                let scores = solve_with_scores(&resolved, Some(&mut self.tt_buf));
                let mut card_scores = Vec::new();
                let mut best_action = legal.trailing_zeros() as u8;
                let mut best_avg: f32 =
                    if maximizing { f32::NEG_INFINITY } else { f32::INFINITY };

                for i in 0..scores.count {
                    let (card, ns_pts) = scores.scores[i];
                    let avg = ns_pts as f32;
                    card_scores.push((card, avg));
                    let better = if maximizing { avg > best_avg } else { avg < best_avg };
                    if better {
                        best_avg = avg;
                        best_action = card;
                    }
                }

                // The position is fully resolved by facts, not by sampling —
                // count it as a world of its own kind rather than mislabeling it.
                return Ok(IsDdResult {
                    best_action,
                    card_scores,
                    determinizations: 1,
                    worlds: WorldCounts::default(),
                });
            }
        }

        // Score accumulators: weighted sum of NS points per card, weight per card
        // (weights are 1.0 unless credibility weighting is enabled).
        let mut score_sum = [0f64; 32];
        let mut weight_sum = [0f64; 32];

        // Scale time budget by cards remaining
        let cards_left = card_count(state.hands[observer as usize]);
        let deadline = config.time_limit_ms.map(|ms| {
            let scaled_ms = (ms as u64 * cards_left as u64) / 8;
            Instant::now() + Duration::from_millis(scaled_ms.max(1))
        });

        let weights = self.compute_weights(state, config, observer);

        let mut successful_dets = 0u32;
        let mut det_count = 0u32;
        let mut world_counts = WorldCounts::default();

        // Once the source stops producing worlds we stop asking, so an
        // over-constrained endgame costs one empty round trip, not one per world.
        let mut source_dry = false;

        // The search runs in chunks: **generate** a batch of worlds sequentially
        // (the world queue and the RNG are stateful), then **solve** the whole
        // batch — in parallel when `config.parallel` is set, otherwise one by one
        // reusing this search's TT. The chunk is one world in sequential mode
        // (tightest deadline adherence, identical to the legacy per-world loop)
        // and one worker-slot's worth in parallel mode.
        let chunk_size = solve_chunk_size(config.parallel);

        loop {
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    break;
                }
            } else if det_count >= config.determinizations {
                break;
            }

            // How many worlds to attempt this round.
            let remaining = if deadline.is_some() {
                chunk_size
            } else {
                chunk_size.min((config.determinizations - det_count) as usize)
            };

            // --- Refill from the world source when the queue cannot cover the
            // round. In count mode ask for the whole remaining budget (one round
            // trip per move); under a deadline ask for `world_batch` at a time. ---
            if let Some(src) = source.as_deref_mut() {
                if !source_dry && self.world_queue.len() < remaining {
                    let want = if deadline.is_some() {
                        config.world_batch.max(remaining)
                    } else {
                        ((config.determinizations - det_count) as usize).max(remaining)
                    };
                    let batch = src.worlds(state, observer, want, rng)?;
                    if batch.is_empty() {
                        source_dry = true;
                    } else {
                        self.world_queue.extend(batch);
                    }
                }
            }

            // --- Generate a chunk of worlds (sequential). ---
            let mut chunk: Vec<GameState> = Vec::with_capacity(remaining);
            let mut chunk_origins: Vec<WorldOrigin> = Vec::with_capacity(remaining);
            let mut attempted = 0u32;
            for _ in 0..remaining {
                attempted += 1;
                if let Some((s, origin)) = self.generate_world(state, observer, &weights, config, rng)
                {
                    chunk.push(s);
                    chunk_origins.push(origin);
                }
            }
            det_count += attempted;
            if chunk.is_empty() {
                // Every attempt this round failed to determinize; in count mode
                // `det_count` still advances so we terminate, in time mode we
                // retry until the deadline. Avoid touching the accumulators.
                continue;
            }

            // --- Credibility weights (sequential: the judge nets are stateful). ---
            let cred_weights: Vec<f64> = if config.cred_alpha > 0.0 {
                chunk
                    .iter()
                    .map(|s| self.credibility_weight(&s.hands, observer, config.cred_alpha) as f64)
                    .collect()
            } else {
                vec![1.0; chunk.len()]
            };

            // --- Solve the chunk (parallel or sequential). ---
            let chunk_scores = solve_worlds(&chunk, config.parallel, &mut self.tt_buf);

            // --- Aggregate in a fixed order (parallel result is identical). ---
            for (scores, &cw) in chunk_scores.iter().zip(cred_weights.iter()) {
                for i in 0..scores.count {
                    let (card, ns_pts) = scores.scores[i];
                    score_sum[card as usize] += ns_pts as f64 * cw;
                    weight_sum[card as usize] += cw;
                }
            }
            successful_dets += chunk.len() as u32;
            for origin in &chunk_origins {
                world_counts.record(*origin);
            }
        }

        // Sourced worlds are position-specific: drop any leftover so the next
        // search cannot consume worlds sampled for the previous position.
        self.world_queue.clear();

        // Build result: pick best card based on aggregated scores
        let legal = state.legal_actions();
        let mut best_action = legal.trailing_zeros() as u8;
        let mut best_avg: f32 = if maximizing { f32::NEG_INFINITY } else { f32::INFINITY };
        let mut card_scores = Vec::new();

        let mut mask = legal;
        while mask != 0 {
            let card = mask.trailing_zeros() as u8;
            let wsum = weight_sum[card as usize];
            let avg = if wsum > 1e-9 {
                (score_sum[card as usize] / wsum) as f32
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

        Ok(IsDdResult {
            best_action,
            card_scores,
            determinizations: successful_dets,
            worlds: world_counts,
        })
    }

    /// Take one determinized world for the current position.
    ///
    /// Worlds already pulled from the [`WorldSource`] are consumed first; when
    /// the queue is empty the search falls back to its own sampling — a
    /// **belief-weighted** world with probability `belief_frac` when a belief
    /// source is active, otherwise a **constraint-uniform** one. Hard
    /// constraints (voids, trump ceiling, played cards) are honored by every
    /// path. Returns `None` when a determinizer fails (an over-constrained
    /// position); the caller counts the attempt against the budget and moves
    /// on. The [`WorldOrigin`] says which branch produced the world so the
    /// caller can report the mix.
    fn generate_world(
        &mut self,
        state: &GameState,
        observer: u8,
        weights: &Option<[[f32; 32]; 4]>,
        config: &IsDdConfig,
        rng: &mut impl Rng,
    ) -> Option<(GameState, WorldOrigin)> {
        // Worlds from the source are pre-validated by `retain_valid`.
        if let Some(hands) = self.world_queue.pop() {
            let mut s = *state;
            s.hands = hands;
            return Some((s, WorldOrigin::Injected));
        }

        if weights.is_some() && rng.gen::<f32>() < config.belief_frac {
            let w = weights.as_ref().unwrap();
            return match determinize_weighted(state, observer, w, rng) {
                Some(s) => Some((s, WorldOrigin::Belief)),
                None => determinize_greedy(state, observer, rng)
                    .map(|s| (s, WorldOrigin::Uniform)),
            };
        }

        // Ensemble coverage floor: constraint-uniform world.
        determinize_greedy(state, observer, rng).map(|s| (s, WorldOrigin::Uniform))
    }
}

/// Worlds solved per generate/solve round. Sequential mode uses one world per
/// round (tightest deadline adherence, identical to the legacy per-world loop);
/// parallel mode fills one round of the rayon worker pool.
#[inline]
fn solve_chunk_size(parallel: bool) -> usize {
    #[cfg(feature = "parallel")]
    {
        if parallel {
            return rayon::current_num_threads().max(1);
        }
    }
    let _ = parallel;
    1
}

/// Solve a batch of fully-determinized worlds, returning per-world DD scores in
/// input order. In parallel mode each rayon worker keeps its own reusable TT
/// (`map_init`); sequential mode reuses the caller's `tt_buf`. DD is exact and
/// deterministic, so the two paths return identical scores.
#[cfg(feature = "parallel")]
fn solve_worlds(
    worlds: &[GameState],
    parallel: bool,
    tt_buf: &mut Vec<u64>,
) -> Vec<crate::solver::SolveScores> {
    use rayon::prelude::*;
    if parallel {
        worlds
            .par_iter()
            .map_init(new_tt_buffer, |tt, s| solve_with_scores(s, Some(tt)))
            .collect()
    } else {
        worlds
            .iter()
            .map(|s| solve_with_scores(s, Some(tt_buf)))
            .collect()
    }
}

#[cfg(not(feature = "parallel"))]
fn solve_worlds(
    worlds: &[GameState],
    _parallel: bool,
    tt_buf: &mut Vec<u64>,
) -> Vec<crate::solver::SolveScores> {
    worlds
        .iter()
        .map(|s| solve_with_scores(s, Some(tt_buf)))
        .collect()
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
            parallel: true,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let mut search = IsDdSearch::new();
                let action = search.search(&state, &config, &mut rng);
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

    /// Parallel and sequential solving must agree exactly: world generation is
    /// RNG-driven, so we seed identically and only flip `parallel`. DD is exact
    /// and the aggregation reduces in a fixed order, so `card_scores` must match.
    #[cfg(feature = "parallel")]
    #[test]
    #[ignore]
    fn test_parallel_matches_sequential() {
        use rand::SeedableRng;
        let mut src = rand::rngs::StdRng::seed_from_u64(12345);
        let mut checked = 0;
        for _ in 0..200 {
            let Some(state) = random_playing_state(&mut src) else { continue };
            let seq = {
                let mut rng = rand::rngs::StdRng::seed_from_u64(777);
                let cfg = IsDdConfig { determinizations: 12, parallel: false, ..Default::default() };
                IsDdSearch::new().search_with_stats(&state, &cfg, &mut rng)
            };
            let par = {
                let mut rng = rand::rngs::StdRng::seed_from_u64(777);
                let cfg = IsDdConfig { determinizations: 12, parallel: true, ..Default::default() };
                IsDdSearch::new().search_with_stats(&state, &cfg, &mut rng)
            };
            assert_eq!(seq.best_action, par.best_action, "best_action diverged");
            assert_eq!(seq.card_scores, par.card_scores, "card_scores diverged");
            checked += 1;
            if checked >= 15 {
                break;
            }
        }
        assert!(checked >= 5, "Not enough non-void deals to test");
    }
}
