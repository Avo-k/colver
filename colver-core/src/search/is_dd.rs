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
use crate::dmc_obs::EnvTracking;
use crate::elephant::{blend_with_evidence, ElephantMemory};
use crate::playgen::infer::{AuctionLogp, PlaygenModel, PlaygenSampler, WorldLogp};
use crate::solver::{new_tt_buffer, solve_with_scores};
use crate::state::{GameState, Phase};

/// Configuration for IS-DD search.
///
/// **Hard constraints** (voids, trump ceiling, played cards) are facts, not beliefs:
/// they are always applied unconditionally and not exposed as a flag.
///
/// **Soft beliefs** (heuristic soft inference, NN beliefs, elephant memory) are all
/// **off by default** — they introduce probabilistic adjustments that may help or hurt
/// depending on opponents and play model.
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
    /// If true, enable elephant memory (particle filter from past determinizations).
    /// **Off by default.**
    pub use_elephant_memory: bool,
    /// Smoothing factor for elephant memory evidence blending (default 0.05).
    /// Lower = stronger influence from particles; higher = more conservative.
    pub elephant_smoothing: f32,
    /// Penalty factor per dominant card not played (default 0.5).
    /// Only used when elephant memory is enabled.
    pub elephant_dominance_penalty: f32,
    /// Whether to use soft dominance penalty in elephant memory (default true).
    pub elephant_use_dominance: bool,
    /// Decay factor for elephant memory particles (default 0.8).
    /// 1.0 = no decay, 0.5 = aggressive decay of old particles.
    pub elephant_decay: f32,
    /// Play dominance inference factor for CardBeliefs.
    /// When a player follows suit without playing the highest, reduce weight for
    /// higher unknown cards by this factor. 1.0 = off, 0.3 = aggressive. Default 1.0.
    pub dominance_factor: f32,
    /// If true (default), skip search when only 1 legal action or position is fully resolved.
    pub early_termination: bool,
    /// Fraction of determinized worlds sampled from the playgen transformer
    /// (requires a loaded playgen model). 0.0 = all worlds from
    /// constraint-uniform sampling (default), 1.0 = all from playgen.
    pub playgen_frac: f32,
    /// Softmax temperature for playgen world sampling (default 1.0 = posterior).
    pub playgen_temp: f32,
    /// Ensemble pool: among NON-playgen worlds, fraction sampled with belief
    /// weights (when available); the rest use constraint-uniform sampling for
    /// coverage. Default 1.0 = previous behavior (all weighted when a belief
    /// source is active).
    pub belief_frac: f32,
    /// Auction-credibility importance weighting of worlds in the DD
    /// aggregation (requires `load_cred_bid_net`). Each world's weight is the
    /// product of per-bid rank factors (would the bid net replay the observed
    /// bid with this world's hand?), flattened by this exponent.
    /// 0.0 = off (default); 0.5 = recommended soft weighting.
    pub cred_alpha: f32,
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
            use_elephant_memory: false,
            elephant_smoothing: 0.05,
            elephant_dominance_penalty: 0.5,
            elephant_use_dominance: true,
            elephant_decay: 0.8,
            dominance_factor: 1.0,
            early_termination: true,
            playgen_frac: 0.0,
            playgen_temp: 1.0,
            belief_frac: 1.0,
            cred_alpha: 0.0,
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
    elephant: Option<ElephantMemory>,
    playgen: Option<PlaygenSampler>,
    /// Bid net used as an auction-credibility judge (see `cred_alpha`).
    cred_bid_net: Option<crate::bid_net::BidNet>,
    /// Observed auction this deal: (bidder, action) in order.
    auction: Vec<(u8, u8)>,
    /// State at deal start (pre-auction), for credibility replays.
    init_state: Option<GameState>,
    /// Cards played so far per seat (current trick included).
    played_by: [u32; 4],
    /// Externally provided worlds (e.g. GPU sidecar), consumed first by the
    /// next `search*` call. Remaining-hands format, current position.
    injected_worlds: Vec<[u32; 4]>,
    tt_buf: Vec<u64>,
}

impl IsDdSearch {
    pub fn new() -> Self {
        IsDdSearch {
            beliefs: None,
            belief_net: None,
            belief_tracking: None,
            elephant: None,
            playgen: None,
            cred_bid_net: None,
            auction: Vec::new(),
            init_state: None,
            played_by: [0; 4],
            injected_worlds: Vec::new(),
            tt_buf: new_tt_buffer(),
        }
    }

    /// Load the bid net used as auction-credibility judge (`cred_alpha`).
    pub fn load_cred_bid_net(&mut self, path: &str) -> std::io::Result<()> {
        self.cred_bid_net = Some(crate::bid_net::BidNet::load(path)?);
        Ok(())
    }

    /// Attach a playgen world-sampler model (shared, read-only).
    pub fn set_playgen_model(&mut self, model: std::sync::Arc<PlaygenModel>) {
        self.playgen = Some(PlaygenSampler::new(model));
    }

    pub fn has_playgen(&self) -> bool {
        self.playgen.is_some()
    }

    /// Provide externally sampled worlds (e.g. from the GPU sidecar) for the
    /// next `search*` call. Consumed before any internal sampler; invalid
    /// worlds (wrong counts or wrong observer hand) are skipped.
    pub fn set_injected_worlds(&mut self, worlds: Vec<[u32; 4]>) {
        self.injected_worlds = worlds;
    }

    /// Direct access to the playgen sampler (e.g. to read prefix tokens for
    /// an external GPU forward backend).
    pub fn playgen_sampler(&self) -> Option<&PlaygenSampler> {
        self.playgen.as_ref()
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

        // Reset playgen sampler for the new deal
        if let Some(sampler) = &mut self.playgen {
            sampler.init_deal(state, observer);
        }

        // Credibility judge: remember the pre-auction state, reset the log.
        self.auction.clear();
        self.init_state = Some(*state);
        self.played_by = [0; 4];

        // Reset elephant memory for new deal (re-initialized per config in search).
        if let Some(ref mut elephant) = self.elephant {
            elephant.clear();
        }
    }

    /// Initialize beliefs for a new deal, with elephant memory config.
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
        if config.use_elephant_memory {
            let mut elephant = ElephantMemory::new(observer);
            elephant.dominance_penalty = config.elephant_dominance_penalty;
            elephant.use_dominance = config.elephant_use_dominance;
            elephant.decay = config.elephant_decay;
            self.elephant = Some(elephant);
        } else {
            self.elephant = None;
        }
    }

    /// Record an action by any player, updating beliefs and elephant memory.
    ///
    /// `state_before` is the state BEFORE the action was applied.
    pub fn record_action(&mut self, state_before: &GameState, player: u8, action: u8) {
        if let Some(beliefs) = &mut self.beliefs {
            beliefs.record_action(state_before, player, action);
        }
        if let Some(sampler) = &mut self.playgen {
            sampler.record_action(state_before, player, action);
        }
        if let Some(tracking) = &mut self.belief_tracking {
            tracking.track_action(state_before, action);
        }
        if state_before.phase == Phase::Bidding {
            self.auction.push((player, action));
        }
        if state_before.phase == Phase::Playing {
            self.played_by[player as usize] |= 1u32 << action;
        }
        // Update elephant memory: filter particles based on observed play.
        if state_before.phase == Phase::Playing {
            if let Some(ref mut elephant) = self.elephant {
                elephant.observe_play(player, action, state_before);
            }
        }
    }

    /// Reset beliefs (e.g., between deals).
    pub fn reset(&mut self) {
        self.beliefs = None;
        self.belief_tracking = None;
        if let Some(ref mut elephant) = self.elephant {
            elephant.clear();
        }
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

        // Blend with elephant memory evidence if available.
        if config.use_elephant_memory {
            if let Some(ref elephant) = self.elephant {
                if let Some(evidence) = elephant.compute_evidence(state) {
                    if let Some(base) = base_weights {
                        return Some(blend_with_evidence(
                            &base,
                            &evidence,
                            state,
                            observer,
                            config.elephant_smoothing,
                        ));
                    }
                }
            }
        }

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

    /// Monte-Carlo card-location marginals from the playgen world sampler.
    ///
    /// Samples up to `n_worlds` determinized worlds from the current position
    /// (lockstep batches) and counts where each unseen card lands. Returns
    /// `weights[player][card]` probabilities, or `None` if no playgen model is
    /// attached or no world could be generated (e.g. during bidding, before
    /// the contract fixes the canonical trump permutation).
    pub fn playgen_marginals(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Option<[[f32; 32]; 4]> {
        let sampler = self.playgen.as_mut()?;
        const BATCH: usize = 16;
        let mut counts = [[0u32; 32]; 4];
        let mut total = 0u32;
        while (total as usize) < n_worlds {
            let want = BATCH.min(n_worlds - total as usize);
            let worlds = sampler.generate_worlds_batch(state, want, temperature, rng);
            if worlds.is_empty() {
                break;
            }
            for hands in worlds {
                for p in 0..4 {
                    let mut h = hands[p];
                    while h != 0 {
                        counts[p][h.trailing_zeros() as usize] += 1;
                        h &= h - 1;
                    }
                }
                total += 1;
            }
        }
        if total == 0 {
            return None;
        }
        let mut weights = [[0f32; 32]; 4];
        for p in 0..4 {
            for c in 0..32 {
                weights[p][c] = counts[p][c] as f32 / total as f32;
            }
        }
        Some(weights)
    }

    /// Playgen bid-policy logits (43) at the current auction point
    /// (v2 playgen models only; None otherwise).
    pub fn playgen_bid_policy(&mut self, state: &GameState) -> Option<[f32; 43]> {
        self.playgen.as_mut()?.bid_policy(state)
    }

    /// Sample full deals from a mid-auction position via the playgen model
    /// (v2 only): auction completed with the bid head, deal played out to
    /// reveal hands. Returns up to `n_worlds` hand assignments.
    pub fn playgen_auction_deals(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<[u32; 4]> {
        match self.playgen.as_mut() {
            Some(sampler) => sampler.generate_deals_from_auction(state, n_worlds, temperature, rng),
            None => Vec::new(),
        }
    }

    /// Scored variant of [`Self::playgen_auction_deals`]: each deal carries
    /// the cumulative log-probability of its sampled continuation.
    pub fn playgen_auction_deals_scored(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<([u32; 4], AuctionLogp)> {
        match self.playgen.as_mut() {
            Some(sampler) => {
                sampler.generate_deals_from_auction_scored(state, n_worlds, temperature, rng)
            }
            None => Vec::new(),
        }
    }

    /// Auction-credibility weight of a world: for each observed bid by a
    /// player other than `observer`, ask the credibility bid net whether it
    /// would replay that bid holding the world's hand for that player. Rank
    /// factors (argmax 1.0, top-3 0.7, else 0.35) multiply; the product is
    /// flattened by `alpha`. Returns 1.0 when disabled or no judge is loaded.
    fn credibility_weight(&mut self, world_hands: &[u32; 4], observer: u8, alpha: f32) -> f32 {
        if alpha <= 0.0 || self.auction.is_empty() {
            return 1.0;
        }
        let Some(base) = self.init_state else { return 1.0 };
        let Some(net) = self.cred_bid_net.as_mut() else { return 1.0 };

        let obs_dim = net.obs_dim();
        let mut obs = vec![0.0f32; obs_dim];
        let mut s = base;
        // Initial hands = world's remaining cards ∪ cards already played.
        // (The determinized world only assigns cards still in hand.)
        let mut init_hands = [0u32; 4];
        for p in 0..4usize {
            init_hands[p] = world_hands[p] | self.played_by[p];
            if crate::card::card_count(init_hands[p]) != 8 {
                return 1.0; // inconsistent reconstruction — skip weighting
            }
        }
        s.hands = init_hands;
        let mut hist: Vec<(u8, u8)> = Vec::with_capacity(self.auction.len());
        let mut w = 1.0f32;

        for &(p, a) in &self.auction {
            if p != observer && s.phase == Phase::Bidding {
                use crate::bid_obs::{
                    self, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2,
                    BID_OBS_DIM_SCORE_AWARE_V3,
                };
                // Standalone-deal convention: cumulative scores 0-0.
                match obs_dim {
                    BID_OBS_DIM_SCORE_AWARE_V3 => {
                        bid_obs::write_bid_observation_score_aware_v3(&mut obs, 0, &s, &hist, 0, 0)
                    }
                    BID_OBS_DIM_SCORE_AWARE_V2 => {
                        bid_obs::write_bid_observation_score_aware_v2(&mut obs, 0, &s, &hist, 0, 0)
                    }
                    BID_OBS_DIM_SCORE_AWARE => {
                        bid_obs::write_bid_observation_score_aware(&mut obs, 0, &s, &hist, 0, 0)
                    }
                    _ => bid_obs::write_bid_observation(&mut obs, 0, &s, &hist),
                }
                let q = net.evaluate(&obs);
                let legal = s.legal_actions();
                let qa = q[a as usize];
                let mut better = 0u32;
                for c in 0..43u8 {
                    if c != a && legal & (1u64 << c) != 0 && q[c as usize] > qa {
                        better += 1;
                    }
                }
                w *= match better {
                    0 => 1.0,
                    1 | 2 => 0.7,
                    _ => 0.35,
                };
            }
            hist.push((p, a));
            s.step(a);
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

    /// Sample play-phase worlds from the playgen model (batch lockstep).
    pub fn playgen_worlds(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<[u32; 4]> {
        match self.playgen.as_mut() {
            Some(sampler) => sampler.generate_worlds_batch(state, n_worlds, temperature, rng),
            None => Vec::new(),
        }
    }

    /// Scored variant of [`Self::playgen_worlds`]: each world carries the
    /// cumulative log-probability of its sampled continuation (hidden actors).
    pub fn playgen_worlds_scored(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<([u32; 4], WorldLogp)> {
        match self.playgen.as_mut() {
            Some(sampler) => {
                sampler.generate_worlds_batch_scored(state, n_worlds, temperature, rng)
            }
            None => Vec::new(),
        }
    }

    /// Get elephant memory stats: (surviving_particles, total_particles).
    pub fn elephant_stats(&self) -> (usize, usize) {
        match &self.elephant {
            Some(e) => (e.surviving_count(), e.total_count()),
            None => (0, 0),
        }
    }

    /// Get elephant evidence weights (particle-derived card distributions).
    /// Returns None if elephant memory is not active or has no surviving particles.
    pub fn elephant_evidence(&self, state: &GameState) -> Option<[[f32; 32]; 4]> {
        self.elephant.as_ref()?.compute_evidence(state)
    }

    /// Get base belief weights (heuristic CardBeliefs, without elephant blending).
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

        if config.early_termination {
            // Forced move: only 1 legal action — skip search entirely.
            let legal = state.legal_actions();
            if legal.count_ones() == 1 {
                let card = legal.trailing_zeros() as u8;
                return IsDdResult {
                    best_action: card,
                    card_scores: vec![(card, 81.0)],
                    determinizations: 0,
                };
            }

            // Resolved position: beliefs uniquely determine all card locations.
            // Single DD solve gives the exact answer — no determinization needed.
            if let Some(resolved) = self.try_resolve_position(state, observer) {
                // Feed the resolved hands as a single particle for elephant memory.
                if config.use_elephant_memory {
                    if let Some(ref mut elephant) = self.elephant {
                        elephant.add_particles(&[resolved.hands]);
                    }
                }

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

                return IsDdResult {
                    best_action,
                    card_scores,
                    determinizations: 1,
                };
            }
        }

        // Score accumulators: weighted sum of NS points per card, weight per card
        // (weights are 1.0 unless auction-credibility weighting is enabled).
        let mut score_sum = [0f64; 32];
        let mut weight_sum = [0f64; 32];

        // Collect determinized hands for elephant memory.
        let store_particles = config.use_elephant_memory && self.elephant.is_some();
        let mut det_hands: Vec<[u32; 4]> = Vec::new();

        // Scale time budget by cards remaining
        let cards_left = card_count(state.hands[observer as usize]);
        let deadline = config.time_limit_ms.map(|ms| {
            let scaled_ms = (ms as u64 * cards_left as u64) / 8;
            Instant::now() + Duration::from_millis(scaled_ms.max(1))
        });

        let weights = self.compute_weights(state, config, observer);

        let mut successful_dets = 0u32;
        let mut det_count = 0u32;

        // Playgen worlds are generated in lockstep batches (weights streamed
        // once per token-step for the whole batch) and consumed from a queue.
        const PLAYGEN_BATCH: usize = 16;
        let mut playgen_queue: Vec<[u32; 4]> = Vec::new();
        let mut playgen_dry = false;

        loop {
            if let Some(d) = deadline {
                if Instant::now() >= d {
                    break;
                }
            } else if det_count >= config.determinizations {
                break;
            }

            // Externally injected worlds are consumed first.
            let injected = loop {
                match self.injected_worlds.pop() {
                    Some(hands) => {
                        let ok = (0..4).all(|p| {
                            card_count(hands[p]) == card_count(state.hands[p])
                        }) && hands[observer as usize] == state.hands[observer as usize];
                        if ok {
                            break Some(hands);
                        }
                    }
                    None => break None,
                }
            };

            let use_playgen = config.playgen_frac > 0.0
                && self.playgen.is_some()
                && !playgen_dry
                && rng.gen::<f32>() < config.playgen_frac;

            let det_state = if let Some(hands) = injected {
                let mut s = *state;
                s.hands = hands;
                Some(s)
            } else if use_playgen {
                if playgen_queue.is_empty() {
                    let sampler = self.playgen.as_mut().unwrap();
                    playgen_queue =
                        sampler.generate_worlds_batch(state, PLAYGEN_BATCH, config.playgen_temp, rng);
                    if playgen_queue.is_empty() {
                        // Repeated dead-ends or too-long sequence: stop trying.
                        playgen_dry = true;
                    }
                }
                playgen_queue
                    .pop()
                    .map(|hands| {
                        let mut s = *state;
                        s.hands = hands;
                        s
                    })
                    .or_else(|| determinize_greedy(state, observer, rng))
            } else if weights.is_some() && rng.gen::<f32>() < config.belief_frac {
                let w = weights.as_ref().unwrap();
                determinize_weighted(state, observer, w, rng)
                    .or_else(|| determinize_greedy(state, observer, rng))
            } else {
                // Ensemble coverage floor: constraint-uniform world.
                determinize_greedy(state, observer, rng)
            };

            let det_state = match det_state {
                Some(s) => s,
                None => {
                    det_count += 1;
                    continue;
                }
            };

            // Store hand assignment for elephant memory.
            if store_particles {
                det_hands.push(det_state.hands);
            }

            let cred_w = if config.cred_alpha > 0.0 {
                self.credibility_weight(&det_state.hands, observer, config.cred_alpha) as f64
            } else {
                1.0
            };

            let scores = solve_with_scores(&det_state, Some(&mut self.tt_buf));

            for i in 0..scores.count {
                let (card, ns_pts) = scores.scores[i];
                score_sum[card as usize] += ns_pts as f64 * cred_w;
                weight_sum[card as usize] += cred_w;
            }

            successful_dets += 1;
            det_count += 1;
        }

        // Injected worlds are one-shot: drop any leftover so a later search
        // at another position cannot consume stale worlds.
        self.injected_worlds.clear();

        // Feed determinized hands into elephant memory.
        if store_particles && !det_hands.is_empty() {
            if let Some(ref mut elephant) = self.elephant {
                elephant.add_particles(&det_hands);
            }
        }

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
