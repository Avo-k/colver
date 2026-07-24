//! Read-only introspection of the playgen world model.
//!
//! Distinct from [`crate::worlds`] on purpose. A [`WorldSource`](crate::worlds::WorldSource)
//! answers "give me worlds to solve" and is part of an *agent*; a
//! [`PlaygenAnalyst`] answers "what does the model believe" and is part of the
//! *analysis* surface — the web's Croyances and Annonces pages, and the
//! world-quality benchmarks. Keeping them apart means the analysis pages
//! cannot accidentally change how the agent plays, and the agent carries no
//! code it never runs.
//!
//! Lifecycle mirrors a player: [`init_deal`](PlaygenAnalyst::init_deal), then
//! [`observe`](PlaygenAnalyst::observe) for every action, then query.

use std::sync::Arc;

use rand::Rng;

use crate::state::GameState;

use super::infer::{AuctionLogp, PlaygenModel, PlaygenSampler, WorldLogp};
use super::tokens::NUM_BID_ACTIONS;

/// Worlds are sampled in lockstep batches: the transformer streams its weights
/// once per token step for the whole batch, so batching is what makes CPU
/// sampling tolerable at all.
const BATCH: usize = 16;

pub struct PlaygenAnalyst {
    sampler: PlaygenSampler,
}

impl PlaygenAnalyst {
    pub fn new(model: Arc<PlaygenModel>) -> Self {
        PlaygenAnalyst { sampler: PlaygenSampler::new(model) }
    }

    pub fn init_deal(&mut self, state: &GameState, observer: u8) {
        self.sampler.init_deal(state, observer);
    }

    pub fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        self.sampler.record_action(state_before, player, action);
    }

    /// The underlying sampler, for callers that drive the model directly
    /// (e.g. handing the prefix tokens to a GPU backend).
    pub fn sampler(&self) -> &PlaygenSampler {
        &self.sampler
    }

    pub fn sampler_mut(&mut self) -> &mut PlaygenSampler {
        &mut self.sampler
    }

    /// Monte-Carlo card-location marginals: sample up to `n_worlds` worlds from
    /// the current position and count where each unseen card lands.
    ///
    /// Returns `weights[player][card]`, or `None` if no world could be
    /// generated — notably during the auction, before the contract fixes the
    /// canonical trump permutation the model was trained on.
    pub fn marginals(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Option<[[f32; 32]; 4]> {
        let mut counts = [[0u32; 32]; 4];
        let mut total = 0u32;
        while (total as usize) < n_worlds {
            let want = BATCH.min(n_worlds - total as usize);
            let worlds = self.sampler.generate_worlds_batch(state, want, temperature, rng);
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

    /// Bid-policy logits (43-way) at the current auction point. `None` for v1
    /// models, which have no bid head.
    pub fn bid_policy(&mut self, state: &GameState) -> Option<[f32; NUM_BID_ACTIONS]> {
        self.sampler.bid_policy(state)
    }

    /// Sample full deals from a mid-auction position: the auction is completed
    /// with the bid head, then the deal is played out to reveal the hands.
    /// v2 models only.
    pub fn auction_deals(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<[u32; 4]> {
        self.sampler.generate_deals_from_auction(state, n_worlds, temperature, rng)
    }

    /// [`auction_deals`](Self::auction_deals) with each deal's cumulative
    /// log-probability under the model.
    pub fn auction_deals_scored(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<([u32; 4], AuctionLogp)> {
        self.sampler.generate_deals_from_auction_scored(state, n_worlds, temperature, rng)
    }

    /// Sample play-phase worlds (remaining hands) from the current position.
    pub fn play_worlds(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<[u32; 4]> {
        self.sampler.generate_worlds_batch(state, n_worlds, temperature, rng)
    }

    /// [`play_worlds`](Self::play_worlds) with each world's cumulative
    /// log-probability under the model.
    pub fn play_worlds_scored(
        &mut self,
        state: &GameState,
        n_worlds: usize,
        temperature: f32,
        rng: &mut impl Rng,
    ) -> Vec<([u32; 4], WorldLogp)> {
        self.sampler.generate_worlds_batch_scored(state, n_worlds, temperature, rng)
    }
}
