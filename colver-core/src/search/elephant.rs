//! Elephant memory: particle filter for IS-DD determinizations.
//!
//! During IS-DD search, determinized hand assignments are stored as "particles".
//! When opponents play cards, particles inconsistent with observed play are
//! eliminated or down-weighted. Surviving particles provide empirical card
//! distribution estimates that improve future determinization sampling.
//!
//! This is essentially sequential Monte Carlo (particle filtering) applied to
//! card game beliefs. Named "dédéléfant" because it remembers everything.

use crate::card::{
    card_rank, card_suit_u8, cards_in_suit_idx, CardIter, CardSet, TRUMP_STRENGTH,
};
use crate::state::GameState;

/// Elephant memory: accumulates evidence from past IS-DD determinizations.
pub struct ElephantMemory {
    /// Stored hand assignments from past determinizations.
    /// Each entry is `[hands[0], hands[1], hands[2], hands[3]]`.
    particles: Vec<[CardSet; 4]>,
    /// Weight of each particle (1.0 initially, reduced or zeroed on inconsistency).
    weights: Vec<f32>,
    /// The observer whose perspective we track from.
    observer: u8,
    /// Penalty factor applied per dominant card the player held but didn't play.
    /// 0.0 = hard elimination, 1.0 = no penalty. Default 0.5.
    pub dominance_penalty: f32,
    /// Whether to apply the soft dominance penalty (vs hard filter only).
    pub use_dominance: bool,
    /// Decay factor applied to existing particle weights when new particles are added.
    /// 1.0 = no decay (all particles equally weighted over time).
    /// 0.5 = older particles lose half their weight each search round.
    /// Default 0.8.
    pub decay: f32,
}

impl ElephantMemory {
    pub fn new(observer: u8) -> Self {
        ElephantMemory {
            particles: Vec::new(),
            weights: Vec::new(),
            observer,
            dominance_penalty: 0.5,
            use_dominance: true,
            decay: 0.8,
        }
    }

    /// Add particles (hand assignments) from a completed search.
    /// Applies decay to all existing particle weights before adding new ones,
    /// so recent particles are more influential than old ones.
    pub fn add_particles(&mut self, hands_list: &[[CardSet; 4]]) {
        // Decay existing particles.
        if self.decay < 1.0 {
            for w in &mut self.weights {
                *w *= self.decay;
            }
        }
        for hands in hands_list {
            self.particles.push(*hands);
            self.weights.push(1.0);
        }
    }

    /// Number of particles with non-zero weight.
    pub fn surviving_count(&self) -> usize {
        self.weights.iter().filter(|&&w| w > 0.0).count()
    }

    /// Total number of particles (including dead ones).
    pub fn total_count(&self) -> usize {
        self.particles.len()
    }

    /// Observe a play action and filter/reweight particles.
    ///
    /// `state_before` is the game state BEFORE the action was applied.
    /// Only processes play-phase actions (not bidding).
    pub fn observe_play(
        &mut self,
        player: u8,
        card: u8,
        state_before: &GameState,
    ) {
        if player == self.observer {
            return; // Observer's hand is known — no filtering needed.
        }
        if self.particles.is_empty() {
            return;
        }

        let card_bit = 1u32 << card;
        let card_suit = card_suit_u8(card);
        let is_trump = card_suit == state_before.contract.trump;

        for i in 0..self.particles.len() {
            if self.weights[i] == 0.0 {
                continue;
            }

            let player_hand = self.particles[i][player as usize];

            // Hard check: the player must have the played card in this particle.
            if player_hand & card_bit == 0 {
                self.weights[i] = 0.0;
                continue;
            }

            // Soft dominance penalty: if the player had a strictly stronger card
            // of the same suit that they didn't play, reduce particle weight.
            if self.use_dominance {
                let same_suit = cards_in_suit_idx(player_hand, card_suit);
                let card_strength = if is_trump {
                    TRUMP_STRENGTH[card_rank(card) as usize]
                } else {
                    card_rank(card)
                };

                for other in CardIter(same_suit) {
                    if other == card {
                        continue;
                    }
                    let other_strength = if is_trump {
                        TRUMP_STRENGTH[card_rank(other) as usize]
                    } else {
                        card_rank(other)
                    };
                    if other_strength > card_strength {
                        self.weights[i] *= self.dominance_penalty;
                    }
                }
            }
        }

        // Prune dead particles periodically to save memory.
        if self.particles.len() > 200 {
            self.prune();
        }
    }

    /// Compute evidence weights from surviving particles.
    ///
    /// Returns `weights[player][card]` representing the empirical probability
    /// that each player holds each unknown card, based on surviving particles.
    /// Returns `None` if no particles survive (fall back to other beliefs).
    pub fn compute_evidence(&self, state: &GameState) -> Option<[[f32; 32]; 4]> {
        let total_weight: f32 = self.weights.iter().sum();
        if total_weight < 1e-6 {
            return None;
        }

        // Known cards: observer's hand + played cards + current trick cards.
        let mut played = state.played_cards;
        for i in 0..4 {
            let c = state.current_trick[i];
            if c != crate::card::EMPTY {
                played |= 1u32 << c;
            }
        }
        let known = state.hands[self.observer as usize] | played;

        let inv_total = 1.0 / total_weight;
        let mut evidence = [[0.0f32; 32]; 4];

        for (i, particle) in self.particles.iter().enumerate() {
            let w = self.weights[i];
            if w == 0.0 {
                continue;
            }
            let weighted = w * inv_total;
            for p in 0..4u8 {
                if p == self.observer {
                    continue;
                }
                // Only count cards that are still unknown (not played, not in observer's hand).
                let relevant = particle[p as usize] & !known;
                for card in CardIter(relevant) {
                    evidence[p as usize][card as usize] += weighted;
                }
            }
        }

        Some(evidence)
    }

    /// Remove particles with zero weight to free memory.
    fn prune(&mut self) {
        let mut i = 0;
        while i < self.particles.len() {
            if self.weights[i] == 0.0 {
                self.particles.swap_remove(i);
                self.weights.swap_remove(i);
            } else {
                i += 1;
            }
        }
    }

    /// Clear all particles (e.g., between deals).
    pub fn clear(&mut self) {
        self.particles.clear();
        self.weights.clear();
    }
}

/// Blend belief weights with elephant memory evidence.
///
/// Uses multiplicative combination: `combined[p][c] = belief[p][c] * (ε + evidence[p][c])`.
/// This way, cards that no surviving particle assigns to a player get suppressed (×ε),
/// while cards consistent with particles are mostly unchanged.
///
/// `smoothing` prevents complete elimination (default ~0.05).
pub fn blend_with_evidence(
    belief_weights: &[[f32; 32]; 4],
    evidence: &[[f32; 32]; 4],
    state: &GameState,
    observer: u8,
    smoothing: f32,
) -> [[f32; 32]; 4] {
    let mut played = state.played_cards;
    for i in 0..4 {
        let c = state.current_trick[i];
        if c != crate::card::EMPTY {
            played |= 1u32 << c;
        }
    }
    let known = state.hands[observer as usize] | played;

    let mut result = [[0.0f32; 32]; 4];

    for card in 0..32u32 {
        if known & (1 << card) != 0 {
            continue;
        }

        // Multiply belief by (smoothing + evidence), then renormalize.
        let mut sum = 0.0f32;
        for p in 0..4usize {
            let combined =
                belief_weights[p][card as usize] * (smoothing + evidence[p][card as usize]);
            result[p][card as usize] = combined;
            sum += combined;
        }
        if sum > 0.0 {
            let inv = 1.0 / sum;
            for p in 0..4usize {
                result[p][card as usize] *= inv;
            }
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::GameState;

    #[test]
    fn test_elephant_basic_filtering() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);

        let mut mem = ElephantMemory::new(0);

        // Add a particle that is the actual deal.
        mem.add_particles(&[state.hands]);
        assert_eq!(mem.surviving_count(), 1);

        // Observe player 1 playing a card they actually have.
        let p1_hand = state.hands[1];
        let card = p1_hand.trailing_zeros() as u8;
        mem.observe_play(1, card, &state);
        // Should survive (player 1 has this card).
        assert!(mem.weights[0] > 0.0);

        // Now observe player 1 "playing" a card they DON'T have.
        let p2_hand = state.hands[2];
        let card2 = p2_hand.trailing_zeros() as u8;
        // Make sure this card isn't in p1's hand.
        if p1_hand & (1 << card2) == 0 {
            mem.observe_play(1, card2, &state);
            assert_eq!(mem.weights[0], 0.0);
        }
    }

    #[test]
    fn test_elephant_evidence_computation() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);

        let mut mem = ElephantMemory::new(0);

        // Add the true deal as a particle.
        mem.add_particles(&[state.hands]);

        let evidence = mem.compute_evidence(&state).unwrap();

        // For each unknown card, exactly one player should have it with weight 1.0.
        let known = state.hands[0] | state.played_cards;
        for card in 0..32u8 {
            if known & (1 << card) != 0 {
                continue;
            }
            let sum: f32 = (0..4).map(|p| evidence[p][card as usize]).sum();
            assert!(
                (sum - 1.0).abs() < 0.01,
                "Evidence should sum to ~1.0 for card {}, got {}",
                card,
                sum
            );
        }
    }

    #[test]
    fn test_elephant_no_filter_on_observer() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);

        let mut mem = ElephantMemory::new(0);
        mem.add_particles(&[state.hands]);

        // Observing the observer's own play should not filter.
        let card = state.hands[0].trailing_zeros() as u8;
        mem.observe_play(0, card, &state);
        assert_eq!(mem.surviving_count(), 1);
    }

    #[test]
    fn test_blend_with_evidence() {
        let mut rng = rand::thread_rng();
        let state = GameState::deal_random(0, &mut rng);

        // Uniform beliefs.
        let mut beliefs = [[0.0f32; 32]; 4];
        let known = state.hands[0] | state.played_cards;
        for card in 0..32u8 {
            if known & (1 << card) != 0 {
                continue;
            }
            for p in 1..4 {
                beliefs[p][card as usize] = 1.0 / 3.0;
            }
        }

        // Evidence: all cards assigned to player 1.
        let mut evidence = [[0.0f32; 32]; 4];
        for card in 0..32u8 {
            if known & (1 << card) != 0 {
                continue;
            }
            evidence[1][card as usize] = 1.0;
        }

        let blended = blend_with_evidence(&beliefs, &evidence, &state, 0, 0.05);

        // Player 1 should get much higher weight than 2 or 3.
        for card in 0..32u8 {
            if known & (1 << card) != 0 {
                continue;
            }
            assert!(
                blended[1][card as usize] > blended[2][card as usize],
                "Player 1 should have higher blended weight"
            );
        }
    }
}
