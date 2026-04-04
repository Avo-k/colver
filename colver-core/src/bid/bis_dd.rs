//! Unified DD-based agent for Belote Contree (Bid + Play).
//!
//! Uses `BeliefState` for belief-weighted determinization and the DD solver
//! for both bidding and play decisions. The bidding side evaluates candidate
//! bids by expected value (EV) across sampled worlds, while the play side
//! uses per-card score aggregation (like IS-DD).

use std::time::{Duration, Instant};

use rand::SeedableRng;

use crate::belief_state::BeliefState;
use crate::bid_eval::evaluate_for_trump;
use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::solver::{new_tt_buffer, solve_for_trump_reuse_tt, solve_with_scores};
use crate::state::{GameState, Phase};

/// Configuration for the BisDd agent.
#[derive(Clone, Debug)]
pub struct BisDdConfig {
    /// Minimum number of determinizations before considering a decision (default 20).
    pub min_dets: u32,
    /// Time budget per bid decision in milliseconds (default 2000).
    pub bid_time_ms: u32,
    /// Time budget per play decision in milliseconds (default 500).
    pub play_time_ms: u32,
    /// Minimum `evaluate_for_trump` score to consider a suit as candidate (default 6).
    pub prefilter_threshold: u16,
    /// Whether to evaluate capot bids (default true).
    pub evaluate_capot: bool,
    /// Maximum bid value (encoded /10, default 12 = 120). DD overestimates at high levels.
    pub max_bid_value: u8,
}

impl Default for BisDdConfig {
    fn default() -> Self {
        BisDdConfig {
            min_dets: 10,
            bid_time_ms: 500,
            play_time_ms: 200,
            prefilter_threshold: 6,
            evaluate_capot: true,
            max_bid_value: 12, // cap at 120
        }
    }
}

/// Unified DD-based agent that decides both bids and plays.
pub struct BisDdAgent {
    belief: Option<BeliefState>,
    #[cfg(not(feature = "parallel"))]
    tt_buf: Vec<u64>,
    config: BisDdConfig,
    rng: rand::rngs::StdRng,
}

impl BisDdAgent {
    /// Create a new agent with the given configuration and RNG seed.
    pub fn new(config: BisDdConfig, seed: u64) -> Self {
        BisDdAgent {
            belief: None,
            #[cfg(not(feature = "parallel"))]
            tt_buf: new_tt_buffer(),
            config,
            rng: rand::rngs::StdRng::seed_from_u64(seed),
        }
    }

    /// Access the current belief state (if initialized).
    pub fn belief(&self) -> Option<&BeliefState> {
        self.belief.as_ref()
    }

    /// Initialize beliefs for a new deal.
    pub fn init_deal(&mut self, observer: u8, hand: CardSet) {
        self.belief = Some(BeliefState::new(observer, hand));
    }

    /// Record an observed action (bid or play) by any player.
    ///
    /// `state` must be the state BEFORE the action was applied.
    pub fn observe(&mut self, player: u8, action: u8, state: &GameState) {
        if let Some(ref mut belief) = self.belief {
            match state.phase {
                Phase::Bidding => belief.record_bid(player, action, state),
                Phase::Playing => belief.record_play(player, action as Card, state),
                Phase::Done => {}
            }
        }
    }

    /// Decide the best action (bid or play) for the current player.
    pub fn decide(&mut self, state: &GameState) -> u8 {
        match state.phase {
            Phase::Bidding => self.decide_bid(state),
            Phase::Playing => self.decide_play(state),
            Phase::Done => 0,
        }
    }

    // ---- Bidding ----

    fn decide_bid(&mut self, state: &GameState) -> u8 {
        let player = state.current_player;
        let team = GameState::player_team(player);
        let hand = state.hands[player as usize];
        let legal = state.legal_actions();

        // Step 1: Prefilter candidate suits
        let mut candidates: Vec<u8> = Vec::new();
        for suit_idx in 0..4u8 {
            let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
            if score >= self.config.prefilter_threshold {
                candidates.push(suit_idx);
            }
        }

        // Also include opponent's suit if they have bid
        if state.last_bid_value > 0 {
            let bidder_team = GameState::player_team(state.last_bidder);
            if bidder_team != team && !candidates.contains(&state.last_bid_suit) {
                candidates.push(state.last_bid_suit);
            }
        }

        if candidates.is_empty() {
            return BID_PASS;
        }

        // Step 2: Generate determinizations (sequential — needs mutable rng)
        let deadline = Instant::now() + Duration::from_millis(self.config.bid_time_ms as u64);
        let max_attempts = (self.config.min_dets * 10) as usize;
        let mut dets: Vec<[CardSet; 4]> = Vec::new();

        for _ in 0..max_attempts {
            if dets.len() >= self.config.min_dets as usize && Instant::now() >= deadline {
                break;
            }

            let det = match self.belief.as_ref() {
                Some(belief) => belief.determinize(state, &mut self.rng),
                None => None,
            };
            if let Some(d) = det {
                dets.push(d.hands);
            }
        }

        if dets.is_empty() {
            return BID_PASS;
        }

        // Step 3: Solve all determinizations (parallel with rayon, sequential fallback)
        let results = self.solve_bid_dets(&dets, state.dealer, &candidates);

        // Step 4: Evaluate all candidate actions by EV
        // DD-based EV overestimates because it assumes perfect play.
        // Apply a margin that scales with bid level to compensate.
        let pass_ev = Self::ev_pass(&results, state, team);
        let base_margin = 50.0f32;

        let mut best_action = BID_PASS;
        let mut best_ev = pass_ev;

        for &suit_idx in &candidates {
            for bid_value in 8..=self.config.max_bid_value {
                let action = bidding::encode_bid(bid_value, suit_idx);
                if legal & (1u64 << action) == 0 {
                    continue;
                }
                let ev = Self::ev_bid(&results, suit_idx as usize, bid_value, team);
                // Scale margin: 50 for bid 80, +10 per level → 130 for bid 160
                let margin = base_margin + (bid_value - 8) as f32 * 10.0;
                if ev - pass_ev > margin && ev > best_ev {
                    best_ev = ev;
                    best_action = action;
                }
            }

            // Evaluate capot (very high margin — DD capots rarely hold)
            if self.config.evaluate_capot {
                let capot_action = 37 + suit_idx;
                if legal & (1u64 << capot_action) != 0 {
                    let ev = Self::ev_capot(&results, suit_idx as usize, team);
                    if ev - pass_ev > 200.0 && ev > best_ev {
                        best_ev = ev;
                        best_action = capot_action;
                    }
                }
            }
        }

        // Evaluate coinche
        if legal & (1u64 << BID_COINCHE) != 0 {
            let ev = Self::ev_coinche(&results, state, team);
            if ev - pass_ev > base_margin && ev > best_ev {
                best_ev = ev;
                best_action = BID_COINCHE;
            }
        }

        let _ = best_ev; // suppress unused warning
        best_action
    }

    /// Solve determinizations for bid evaluation.
    /// With `parallel` feature: uses rayon, each thread gets its own TT buffer.
    /// Without: sequential with shared TT buffer.
    #[cfg(feature = "parallel")]
    fn solve_bid_dets(
        &mut self,
        dets: &[[CardSet; 4]],
        dealer: u8,
        candidates: &[u8],
    ) -> Vec<[u8; 4]> {
        use rayon::prelude::*;

        let candidates = candidates.to_vec();
        dets.par_iter()
            .map(|hands| {
                let mut tt = new_tt_buffer();
                let mut ns_pts = [0u8; 4];
                for &suit_idx in &candidates {
                    let result = solve_for_trump_reuse_tt(*hands, dealer, suit_idx, &mut tt);
                    ns_pts[suit_idx as usize] = result[0];
                }
                ns_pts
            })
            .collect()
    }

    #[cfg(not(feature = "parallel"))]
    fn solve_bid_dets(
        &mut self,
        dets: &[[CardSet; 4]],
        dealer: u8,
        candidates: &[u8],
    ) -> Vec<[u8; 4]> {
        dets.iter()
            .map(|hands| {
                let mut ns_pts = [0u8; 4];
                for &suit_idx in candidates {
                    let result =
                        solve_for_trump_reuse_tt(*hands, dealer, suit_idx, &mut self.tt_buf);
                    ns_pts[suit_idx as usize] = result[0];
                }
                ns_pts
            })
            .collect()
    }

    // ---- EV helpers ----

    /// Expected value of bidding `bid_value` (encoded, e.g. 8=80) in `suit` for `team`.
    fn ev_bid(results: &[[u8; 4]], suit: usize, bid_value: u8, team: u8) -> f32 {
        let contract = bid_value as f32 * 10.0;
        let n = results.len() as f32;
        let mut score = 0.0f32;

        for r in results {
            let ns_pts = r[suit] as f32;
            let team_pts = if team == 0 { ns_pts } else { 162.0 - ns_pts };
            if team_pts >= contract {
                score += team_pts + contract; // made
            } else {
                score -= contract; // failed
            }
        }

        score / n
    }

    /// Expected value of passing for `team` given current auction state.
    fn ev_pass(results: &[[u8; 4]], state: &GameState, team: u8) -> f32 {
        if state.last_bid_value == 0 {
            return 0.0;
        }

        let bidder_team = GameState::player_team(state.last_bidder);
        let contract = state.last_bid_value as f32 * 10.0;
        let coinche_mult = match state.coinche_state {
            0 => 1.0f32,
            1 => 2.0,
            _ => 4.0,
        };
        let suit = state.last_bid_suit as usize;
        let n = results.len() as f32;
        let mut score = 0.0f32;

        for r in results {
            let ns_pts = r[suit] as f32;
            let bidder_pts = if bidder_team == 0 {
                ns_pts
            } else {
                162.0 - ns_pts
            };

            let our_gain = if bidder_pts >= contract {
                // Bidder makes it
                let bidder_gain = (bidder_pts + contract) * coinche_mult;
                if bidder_team == team {
                    bidder_gain
                } else {
                    -bidder_gain
                }
            } else {
                // Bidder fails
                let defender_gain = contract * coinche_mult;
                if bidder_team == team {
                    -defender_gain
                } else {
                    defender_gain
                }
            };
            score += our_gain;
        }

        score / n
    }

    /// Expected value of coinching for `team`.
    fn ev_coinche(results: &[[u8; 4]], state: &GameState, team: u8) -> f32 {
        if state.last_bid_value == 0 {
            return f32::NEG_INFINITY;
        }

        let bidder_team = GameState::player_team(state.last_bidder);
        let contract = state.last_bid_value as f32 * 10.0;
        let coinche_mult = 2.0f32; // coinche doubles
        let suit = state.last_bid_suit as usize;
        let n = results.len() as f32;
        let mut score = 0.0f32;

        for r in results {
            let ns_pts = r[suit] as f32;
            let bidder_pts = if bidder_team == 0 {
                ns_pts
            } else {
                162.0 - ns_pts
            };

            let our_gain = if bidder_pts >= contract {
                // Bidder makes it (bad for us, the coincher)
                let bidder_gain = (bidder_pts + contract) * coinche_mult;
                if bidder_team == team {
                    bidder_gain
                } else {
                    -bidder_gain
                }
            } else {
                // Bidder fails (good for us)
                let defender_gain = contract * coinche_mult;
                if bidder_team == team {
                    -defender_gain
                } else {
                    defender_gain
                }
            };
            score += our_gain;
        }

        score / n
    }

    /// Expected value of a capot bid in `suit` for `team`.
    fn ev_capot(results: &[[u8; 4]], suit: usize, team: u8) -> f32 {
        let n = results.len() as f32;
        let mut score = 0.0f32;

        for r in results {
            let ns_pts = r[suit] as f32;
            let team_pts = if team == 0 { ns_pts } else { 252.0 - ns_pts };
            if team_pts >= 252.0 {
                score += 252.0 + 250.0; // 502
            } else {
                score -= 250.0;
            }
        }

        score / n
    }

    // ---- Play ----

    fn decide_play(&mut self, state: &GameState) -> u8 {
        let player = state.current_player;
        let team = GameState::player_team(player);
        let maximizing = team == 0; // NS maximizes ns_pts

        // Scale time by cards remaining
        let cards_left = card_count(state.hands[player as usize]);
        let scaled_ms = (self.config.play_time_ms as u64 * cards_left as u64) / 8;
        let deadline = Instant::now() + Duration::from_millis(scaled_ms.max(1));

        // Step 1: Generate determinizations (sequential)
        let max_attempts = (self.config.min_dets * 10) as usize;
        let mut det_states: Vec<GameState> = Vec::new();

        for _ in 0..max_attempts {
            if det_states.len() >= self.config.min_dets as usize && Instant::now() >= deadline {
                break;
            }

            let det = match self.belief.as_ref() {
                Some(belief) => belief.determinize(state, &mut self.rng),
                None => None,
            };
            if let Some(d) = det {
                det_states.push(d);
            }
        }

        if det_states.is_empty() {
            let legal = state.legal_actions();
            return legal.trailing_zeros() as u8;
        }

        // Step 2: Solve all determinizations (parallel or sequential)
        let (score_sum, score_count) = self.solve_play_dets(&det_states);

        // Step 3: Pick best card
        let legal = state.legal_actions();
        let mut best_action = legal.trailing_zeros() as u8;
        let mut best_avg: f32 = if maximizing {
            f32::NEG_INFINITY
        } else {
            f32::INFINITY
        };

        let mut mask = legal;
        while mask != 0 {
            let card = mask.trailing_zeros() as u8;
            let count = score_count[card as usize];
            let avg = if count > 0 {
                score_sum[card as usize] as f32 / count as f32
            } else {
                81.0 // neutral fallback
            };

            let better = if maximizing {
                avg > best_avg
            } else {
                avg < best_avg
            };
            if better {
                best_avg = avg;
                best_action = card;
            }
            mask &= mask - 1;
        }

        best_action
    }

    /// Solve determinizations for play evaluation.
    #[cfg(feature = "parallel")]
    fn solve_play_dets(&mut self, dets: &[GameState]) -> ([i64; 32], [u32; 32]) {
        use rayon::prelude::*;

        let per_det: Vec<([i64; 32], [u32; 32])> = dets
            .par_iter()
            .map(|det| {
                let mut tt = new_tt_buffer();
                let scores = solve_with_scores(det, Some(&mut tt));
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

        let mut total_sum = [0i64; 32];
        let mut total_count = [0u32; 32];
        for (sum, count) in per_det {
            for i in 0..32 {
                total_sum[i] += sum[i];
                total_count[i] += count[i];
            }
        }
        (total_sum, total_count)
    }

    #[cfg(not(feature = "parallel"))]
    fn solve_play_dets(&mut self, dets: &[GameState]) -> ([i64; 32], [u32; 32]) {
        let mut total_sum = [0i64; 32];
        let mut total_count = [0u32; 32];
        for det in dets {
            let scores = solve_with_scores(det, Some(&mut self.tt_buf));
            for i in 0..scores.count {
                let (card, ns_pts) = scores.scores[i];
                total_sum[card as usize] += ns_pts as i64;
                total_count[card as usize] += 1;
            }
        }
        (total_sum, total_count)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ev_bid_all_make() {
        let results = vec![[100u8, 0, 0, 0]; 10];
        let ev = BisDdAgent::ev_bid(&results, 0, 8, 0); // suit 0, bid 80, team NS
        assert!(
            (ev - 180.0).abs() < 0.01,
            "Expected 180.0, got {}",
            ev
        );
    }

    #[test]
    fn test_ev_bid_all_fail() {
        let results = vec![[60u8, 0, 0, 0]; 10];
        let ev = BisDdAgent::ev_bid(&results, 0, 8, 0);
        assert!(
            (ev - (-80.0)).abs() < 0.01,
            "Expected -80.0, got {}",
            ev
        );
    }

    #[test]
    fn test_ev_bid_mixed() {
        let mut results = vec![[100u8, 0, 0, 0]; 5];
        results.extend(vec![[60u8, 0, 0, 0]; 5]);
        let ev = BisDdAgent::ev_bid(&results, 0, 8, 0);
        // 5 * 180 + 5 * (-80) = 900 - 400 = 500 => 500/10 = 50
        assert!(
            (ev - 50.0).abs() < 0.01,
            "Expected 50.0, got {}",
            ev
        );
    }

    #[test]
    fn test_ev_capot_success() {
        let results = vec![[252u8, 0, 0, 0]; 10];
        let ev = BisDdAgent::ev_capot(&results, 0, 0);
        assert!(
            (ev - 502.0).abs() < 0.01,
            "Expected 502.0, got {}",
            ev
        );
    }

    #[test]
    fn test_ev_capot_failure() {
        let results = vec![[200u8, 0, 0, 0]; 10];
        let ev = BisDdAgent::ev_capot(&results, 0, 0);
        assert!(
            (ev - (-250.0)).abs() < 0.01,
            "Expected -250.0, got {}",
            ev
        );
    }

    #[test]
    fn test_ev_bid_ew_team() {
        // EW team: team_pts = 162 - ns_pts
        // ns_pts=60 => ew_pts=102. bid 80 => 102>=80 => made: 102+80=182
        let results = vec![[60u8, 0, 0, 0]; 10];
        let ev = BisDdAgent::ev_bid(&results, 0, 8, 1);
        assert!(
            (ev - 182.0).abs() < 0.01,
            "Expected 182.0, got {}",
            ev
        );
    }

    #[test]
    fn test_ev_pass_no_bid() {
        let state = GameState::new(0, [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000]);
        let results = vec![[100u8, 0, 0, 0]; 10];
        let ev = BisDdAgent::ev_pass(&results, &state, 0);
        assert!(
            ev.abs() < 0.01,
            "Expected 0.0 when no bid, got {}",
            ev
        );
    }

    #[test]
    fn test_ev_coinche_no_bid() {
        let state = GameState::new(0, [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000]);
        let results = vec![[100u8, 0, 0, 0]; 10];
        let ev = BisDdAgent::ev_coinche(&results, &state, 0);
        assert!(
            ev == f32::NEG_INFINITY,
            "Expected NEG_INFINITY when no bid, got {}",
            ev
        );
    }

    #[test]
    fn test_default_config() {
        let config = BisDdConfig::default();
        assert_eq!(config.min_dets, 10);
        assert_eq!(config.bid_time_ms, 500);
        assert_eq!(config.play_time_ms, 200);
        assert_eq!(config.prefilter_threshold, 6);
        assert!(config.evaluate_capot);
        assert_eq!(config.max_bid_value, 12);
    }

    #[test]
    fn test_new_agent() {
        let agent = BisDdAgent::new(BisDdConfig::default(), 42);
        assert!(agent.belief.is_none());
    }

    #[test]
    fn test_init_deal() {
        let mut agent = BisDdAgent::new(BisDdConfig::default(), 42);
        agent.init_deal(0, 0xFF);
        assert!(agent.belief.is_some());
    }
}
