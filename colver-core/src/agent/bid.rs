//! Auction policies.
//!
//! Two implementations cover everything the bot configs describe: a
//! [`RuleBidPolicy`] wrapping the hand-written strategies in
//! [`crate::bid_eval`], and a [`BidNetPolicy`] wrapping a trained Q-network.
//!
//! The observation layout is chosen from the *model file*, not from a flag:
//! bid nets come in a plain 108-dim flavour and score-aware 110/113/117-dim
//! flavours, and feeding one the other's layout produces confident nonsense
//! rather than an error. Detecting it in one place is the point of this type.
//!
//! The one thing the file *cannot* say is whether the net was trained on the
//! canonical suit ordering — a canonical net is byte-for-byte the same size as a
//! physical one. That comes from `BidSpec::canonical`, and this module is where the
//! consequence lives: the mask goes into canonical space before the argmax, and the
//! winner comes back out of it. Everything a caller sees stays in physical space.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::bid_eval::BidFunction;
use crate::bid_net::BidNet;
use crate::bid_obs::{
    self, BID_OBS_DIM, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2,
    BID_OBS_DIM_SCORE_AWARE_V3,
};
use crate::state::GameState;
use crate::suit_perm;

use super::models::BidWeights;
use super::{AgentError, BidPolicy, Decision, MatchContext, Stats};

/// One of the hand-written bidding strategies.
pub struct RuleBidPolicy {
    function: BidFunction,
}

impl RuleBidPolicy {
    pub fn new(function: BidFunction) -> Self {
        RuleBidPolicy { function }
    }
}

impl BidPolicy for RuleBidPolicy {
    fn decide(&mut self, state: &GameState, _ctx: &MatchContext) -> Result<Decision, AgentError> {
        Ok(Decision::bare(self.function.bid(state), "bid_rule"))
    }
}

/// A trained bid Q-network, with the optional post-hoc adjustments that the
/// bot configs can turn on.
pub struct BidNetPolicy {
    net: BidNet,
    obs: Vec<f32>,
    /// Discount applied to a bid's Q-value, growing with the contract level.
    /// Counteracts the networks' systematic optimism on high contracts.
    /// 0 = raw Q-values.
    penalty: f32,
    /// Softmax sampling temperature over raw Q-values. 0 = greedy argmax over
    /// *adjusted* Q-values. Sampling and the adjustments are alternatives, not
    /// a stack: temperature exists to diversify training/eval opponents, the
    /// adjustments to play better.
    temperature: f32,
    /// Apply endgame match-score adjustments. Only meaningful for a
    /// non-score-aware net; score-aware nets receive the scores in their
    /// observation and are left alone.
    score_aware: bool,
    /// The net answers in **canonical** suit space (see `BidSpec::canonical`).
    canonical: bool,
    rng: StdRng,
}

impl BidNetPolicy {
    pub fn new(
        weights: Arc<BidWeights>,
        penalty: f32,
        temperature: f32,
        score_aware: bool,
        canonical: bool,
        seed: u64,
    ) -> Self {
        let net = weights.instantiate();
        let obs = vec![0.0f32; net.obs_dim()];
        BidNetPolicy {
            net,
            obs,
            penalty,
            temperature,
            score_aware,
            canonical,
            rng: StdRng::seed_from_u64(seed),
        }
    }

    pub fn obs_dim(&self) -> usize {
        self.net.obs_dim()
    }

    /// Fill `self.obs` with the layout this particular network was trained on, and
    /// return the suit ordering its answers are expressed in (identity when the net
    /// is a physical-order one).
    fn write_obs(&mut self, state: &GameState, ctx: &MatchContext, seat: u8) -> [u8; 4] {
        let (mine, theirs) = ctx.scores_for(seat);
        let history = &ctx.tracking.bid_history;
        let dim = self.net.obs_dim();
        if self.canonical {
            return bid_obs::write_bid_observation_canonical(
                &mut self.obs, 0, state, history, mine, theirs, dim,
            );
        }
        match dim {
            BID_OBS_DIM_SCORE_AWARE_V3 => bid_obs::write_bid_observation_score_aware_v3(
                &mut self.obs, 0, state, history, mine, theirs,
            ),
            BID_OBS_DIM_SCORE_AWARE_V2 => bid_obs::write_bid_observation_score_aware_v2(
                &mut self.obs, 0, state, history, mine, theirs,
            ),
            BID_OBS_DIM_SCORE_AWARE => bid_obs::write_bid_observation_score_aware(
                &mut self.obs, 0, state, history, mine, theirs,
            ),
            _ => bid_obs::write_bid_observation(&mut self.obs, 0, state, history),
        }
        [0, 1, 2, 3]
    }
}

impl BidPolicy for BidNetPolicy {
    fn decide(&mut self, state: &GameState, ctx: &MatchContext) -> Result<Decision, AgentError> {
        let seat = state.current_player();
        let order = self.write_obs(state, ctx, seat);
        let legal = state.legal_actions();
        let q = self.net.evaluate(&self.obs);

        // `q` is indexed in the *net's* action space. For a canonical net that is not
        // the physical one, so the legal mask has to be pushed in before selecting and
        // the winner pulled back out. Omitting either still produces a legal bid — in
        // the wrong suit — which is exactly why it is done here and only here.
        let legal_net = if self.canonical {
            suit_perm::permute_bid_mask_u64(legal, &suit_perm::perm_from_order(&order))
        } else {
            legal
        };

        // Score adjustments only apply to nets that cannot see the score.
        let score_aware = self.score_aware && self.net.obs_dim() <= BID_OBS_DIM;
        let (mine, theirs) = ctx.scores_for(seat);

        let mut candidates: Vec<(u8, f32)> = (0..43u8)
            .filter(|a| legal_net & (1u64 << a) != 0)
            .map(|a| {
                // Back to physical immediately: penalties, stats and the returned
                // action are all in the caller's space.
                let phys = if self.canonical {
                    suit_perm::permute_bid_action(a, &order)
                } else {
                    a
                };
                let adjusted = if self.temperature > 0.0 {
                    q[a as usize]
                } else {
                    q[a as usize] - bid_penalty(phys, self.penalty)
                        + endgame_delta(phys, mine, theirs, score_aware)
                };
                (phys, adjusted)
            })
            .collect();

        let action = if self.temperature > 0.0 {
            sample_softmax(&candidates, self.temperature, &mut self.rng)
        } else {
            candidates
                .iter()
                .max_by(|a, b| a.1.total_cmp(&b.1))
                .map(|(a, _)| *a)
                .unwrap_or(0)
        };

        candidates.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(Decision {
            action,
            stats: Stats { source: "bid_nn", candidates, ..Stats::default() },
        })
    }
}

/// Level-dependent discount on a bid's Q-value.
///
/// Calibrated on 4M DD-vs-DouDou50 samples, where the network's optimism grows
/// with the contract: Δ≈0 at 80, ≈−9 at 120, ≈−12 at 130, and ≈−90 for capot —
/// hence the 2.5× multiplier there.
fn bid_penalty(action: u8, penalty: f32) -> f32 {
    if penalty <= 0.0 || action == 0 {
        return 0.0;
    }
    if action >= 41 {
        return penalty * 0.5; // coinche / surcoinche
    }
    if action >= 37 {
        return penalty * 2.5; // capot
    }
    // Regular bids: linear in the contract level, 0 at 80 and `penalty` at 160.
    let value_idx = (action - 1) / 4;
    penalty * (value_idx as f32 / 8.0)
}

/// Contract value of a regular bid action, or `None` for pass/capot/coinche.
fn bid_value_of(action: u8) -> Option<i32> {
    if action == 0 || action >= 37 {
        return None;
    }
    Some(80 + ((action as i32 - 1) / 4) * 10)
}

/// Endgame adjustment for networks that cannot see the match score.
///
/// Three regimes, all pass-through outside the endgame. Deltas are in Q-value
/// units, where ~1.0 ≈ 500 game points, so these are deliberately small nudges.
///
/// - **Leading** (mine ≥ 1700 and ahead by ≥ 300): discourage big gambles.
/// - **Trailing** (theirs ≥ 1700 and behind by ≥ 400): force risk; passing
///   cannot win from there.
/// - **Tight** (either ≥ 1600 and within 200): mild brake on overreach.
fn endgame_delta(action: u8, mine: i32, theirs: i32, enabled: bool) -> f32 {
    if !enabled {
        return 0.0;
    }
    let margin = mine - theirs;
    let leading = mine >= 1700 && margin >= 300;
    let trailing = theirs >= 1700 && margin <= -400;
    let tight = mine.max(theirs) >= 1600 && margin.abs() <= 200;
    let is_capot = (37..41).contains(&action);

    if leading {
        if let Some(v) = bid_value_of(action) {
            if v >= 130 {
                return -0.08 * ((v - 120) as f32 / 40.0);
            }
        }
        return if is_capot { -0.15 } else { 0.0 };
    }
    if trailing {
        if action == 0 {
            return -0.10; // PASS
        }
        if let Some(v) = bid_value_of(action) {
            if v <= 100 {
                return -0.04;
            }
            if v >= 120 {
                return 0.08 * ((v - 110) as f32 / 50.0);
            }
        }
        return 0.0;
    }
    if tight {
        if let Some(v) = bid_value_of(action) {
            if v >= 130 {
                return -0.04 * ((v - 120) as f32 / 40.0);
            }
        }
        return if is_capot { -0.08 } else { 0.0 };
    }
    0.0
}

fn sample_softmax(candidates: &[(u8, f32)], temperature: f32, rng: &mut StdRng) -> u8 {
    if candidates.is_empty() {
        return 0;
    }
    let t = temperature.max(1e-3);
    let max_q = candidates.iter().map(|(_, q)| *q).fold(f32::NEG_INFINITY, f32::max);
    let weights: Vec<f32> = candidates.iter().map(|(_, q)| ((q - max_q) / t).exp()).collect();
    let total: f32 = weights.iter().sum();
    let mut pick = rng.gen::<f32>() * total;
    for ((a, _), w) in candidates.iter().zip(&weights) {
        pick -= w;
        if pick <= 0.0 {
            return *a;
        }
    }
    candidates.last().unwrap().0
}

/// Playgen's own auction head used directly as a bidder.
///
/// The v2 world sampler carries a 43-way bid head, trained by next-token prediction
/// on games whose auctions came from bid v6. It is therefore a behaviour clone of v6
/// rather than an independent strategy, and it is *not* score-aware: the corpus was
/// generated on standalone deals at 0-0, so nothing in the prefix tells it the match
/// score. What it measures is how much auction structure the transformer captured
/// while learning to sample worlds.
///
/// It needs the whole visible prefix, so it tracks the deal from the start through
/// `init_deal` / `observe` exactly as the world source does.
pub struct PlaygenBidPolicy {
    model: Arc<crate::playgen::infer::PlaygenModel>,
    sampler: Option<crate::playgen::infer::PlaygenSampler>,
    seat: u8,
    temperature: f32,
    rng: StdRng,
    /// Fallback when the sampler cannot answer (over-long auction, non-v2 model).
    fallback: BidFunction,
}

impl PlaygenBidPolicy {
    pub fn new(
        model: Arc<crate::playgen::infer::PlaygenModel>,
        seat: u8,
        temperature: f32,
        seed: u64,
    ) -> Self {
        PlaygenBidPolicy {
            model,
            sampler: None,
            seat,
            temperature,
            rng: StdRng::seed_from_u64(seed),
            fallback: BidFunction::ImprovedV2,
        }
    }
}

impl BidPolicy for PlaygenBidPolicy {
    fn init_deal(&mut self, state: &GameState) {
        let mut s = crate::playgen::infer::PlaygenSampler::new(self.model.clone());
        s.init_deal(state, self.seat);
        self.sampler = Some(s);
    }

    fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        if let Some(s) = self.sampler.as_mut() {
            s.record_action(state_before, player, action);
        }
    }

    fn decide(&mut self, state: &GameState, _ctx: &MatchContext) -> Result<Decision, AgentError> {
        let legal = state.legal_actions();
        let logits = self.sampler.as_mut().and_then(|s| s.bid_policy(state));
        let Some(logits) = logits else {
            return Ok(Decision::bare(self.fallback.bid(state), "playgen_bid_fallback"));
        };

        let action = if self.temperature > 0.0 {
            Some(crate::playgen::infer::sample_bid_masked(
                &logits,
                legal,
                self.temperature,
                &mut self.rng,
            ))
        } else {
            let mut best = 0u8;
            let mut bv = f32::NEG_INFINITY;
            for a in 0..crate::bid_obs::BID_MASK_DIM {
                if legal & (1u64 << a) != 0 && logits[a] > bv {
                    bv = logits[a];
                    best = a as u8;
                }
            }
            Some(best)
        };

        match action {
            Some(a) if legal & (1u64 << a) != 0 => Ok(Decision::bare(a, "playgen_bid")),
            _ => Ok(Decision::bare(self.fallback.bid(state), "playgen_bid_fallback")),
        }
    }
}
