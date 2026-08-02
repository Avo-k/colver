//! DD-based bidding system for Belote Contrée.
//!
//! Uses the double-dummy solver with determinization to estimate expected team points
//! for each candidate trump suit. Replaces heuristic score→value mapping with principled
//! point estimation from sampled worlds.
//!
//! Two main components:
//! - `DdBidder`: real-time DD bidder (~300ms/opening) for agent play
//! - `dd_calibrate` binary: offline calibration tool (separate file)

use rand::Rng;

use crate::bid_eval::{evaluate_for_trump, quality_ok};
use crate::bidding::{self, BID_COINCHE, BID_PASS};
use crate::card::*;
use crate::determinize::determinize_greedy;
use crate::solver::{new_tt_buffer, solve_for_trump_reuse_tt};
use crate::state::*;

/// Configuration for DD-based bidding.
#[derive(Clone, Debug)]
pub struct DdBidConfig {
    /// Number of determinizations for opening bid (default 8).
    pub opening_dets: u32,
    /// Number of determinizations for partner response (default 4).
    pub response_dets: u32,
    /// Number of determinizations for overcall (default 4).
    pub overcall_dets: u32,
    /// Confidence margin: bid X if expected points >= X + margin (default 15).
    pub margin: i16,
    /// Minimum heuristic score to consider a suit as candidate (default 6).
    pub prefilter_threshold: u16,
    /// Maximum bid value for opening (encoded, 12=120, default 12).
    pub opening_cap: u8,
    /// Maximum bid value for overcall (encoded, 12=120, default 12).
    pub overcall_cap: u8,
    /// Maximum bid value for response (encoded, 13=130, default 13).
    pub response_cap: u8,
    /// Whether to apply quality gate on candidate suits.
    pub quality_gate: bool,
    /// Use ImprovedV2 coinche logic (fast, well-validated).
    pub use_heuristic_coinche: bool,
}

impl Default for DdBidConfig {
    fn default() -> Self {
        DdBidConfig {
            opening_dets: 8,
            response_dets: 4,
            overcall_dets: 4,
            margin: 15,
            prefilter_threshold: 6,
            opening_cap: 12,
            overcall_cap: 12,
            response_cap: 13,
            quality_gate: true,
            use_heuristic_coinche: true,
        }
    }
}

/// Diagnostic result from DD bidding.
pub struct DdBidResult {
    /// The chosen bid action (0=PASS, or encoded bid, or BID_COINCHE).
    pub action: u8,
    /// Per-suit expected team points from DD evaluation (4 suits).
    /// NaN if suit was not evaluated.
    pub suit_expected_pts: [f32; 4],
    /// Number of determinizations actually performed.
    pub total_solves: u32,
}

/// DD-based bidder. Holds a pre-allocated TT buffer (2MB) for reuse.
pub struct DdBidder {
    pub config: DdBidConfig,
    tt_buf: crate::solver::TtBuf,
}

impl DdBidder {
    pub fn new(config: DdBidConfig) -> Self {
        DdBidder {
            config,
            tt_buf: new_tt_buffer(),
        }
    }

    /// Main entry point: returns a legal bid action.
    pub fn bid(&mut self, state: &GameState, rng: &mut impl Rng) -> u8 {
        self.bid_with_stats(state, rng).action
    }

    /// Bid with diagnostic stats.
    pub fn bid_with_stats(&mut self, state: &GameState, rng: &mut impl Rng) -> DdBidResult {
        debug_assert_eq!(state.phase, Phase::Bidding);

        let player = state.current_player;
        let hand = state.hands[player as usize];
        let legal = state.legal_actions();

        // After coinche: always PASS
        if state.coinche_state > 0 {
            return DdBidResult {
                action: BID_PASS,
                suit_expected_pts: [f32::NAN; 4],
                total_solves: 0,
            };
        }

        // Coinche: reuse heuristic logic (fast, well-validated)
        if self.config.use_heuristic_coinche {
            if let Some(action) = heuristic_coinche(state, player, hand, legal) {
                return DdBidResult {
                    action,
                    suit_expected_pts: [f32::NAN; 4],
                    total_solves: 0,
                };
            }
        }

        let partner = GameState::partner(player);

        // Opening: no bid yet
        if state.last_bid_value == 0 {
            return self.dd_opening(state, player, hand, legal, rng);
        }

        // Partner response
        if state.last_bidder == partner {
            return self.dd_respond(state, player, hand, legal, rng);
        }

        // Overcall
        self.dd_overcall(state, player, hand, legal, rng)
    }

    /// DD-evaluate candidate suits and pick the best opening bid.
    fn dd_opening(
        &mut self,
        state: &GameState,
        player: u8,
        hand: CardSet,
        legal: u64,
        rng: &mut impl Rng,
    ) -> DdBidResult {
        let team = GameState::player_team(player);
        let candidates = self.prefilter_suits(hand);

        if candidates.is_empty() {
            return DdBidResult {
                action: BID_PASS,
                suit_expected_pts: [f32::NAN; 4],
                total_solves: 0,
            };
        }

        let (suit_pts, total_solves) =
            self.dd_evaluate_suits(state, player, &candidates, self.config.opening_dets, rng);

        // Pick best suit
        let (best_suit, best_pts) = self.pick_best_suit(&suit_pts, team);

        // Map expected points → bid value
        let action = self.points_to_bid(
            best_pts,
            best_suit,
            0, // no existing bid
            self.config.opening_cap,
            legal,
        );

        DdBidResult {
            action,
            suit_expected_pts: suit_pts,
            total_solves,
        }
    }

    /// DD-evaluate partner's suit for response.
    fn dd_respond(
        &mut self,
        state: &GameState,
        player: u8,
        _hand: CardSet,
        legal: u64,
        rng: &mut impl Rng,
    ) -> DdBidResult {
        let team = GameState::player_team(player);
        let partner_suit = state.last_bid_suit;
        let partner_value = state.last_bid_value;

        // Don't push past response cap
        if partner_value >= self.config.response_cap {
            return DdBidResult {
                action: BID_PASS,
                suit_expected_pts: [f32::NAN; 4],
                total_solves: 0,
            };
        }

        // Evaluate partner's suit
        let candidates = vec![partner_suit];
        let (suit_pts, total_solves) =
            self.dd_evaluate_suits(state, player, &candidates, self.config.response_dets, rng);

        let team_pts = if team == 0 {
            suit_pts[partner_suit as usize]
        } else {
            // EW team: their points = 162 - ns_points (or 252 for capot)
            // Use 162 - ns as approximation (capot is rare)
            162.0 - suit_pts[partner_suit as usize]
        };

        let action = self.points_to_bid(
            team_pts,
            partner_suit,
            partner_value,
            self.config.response_cap,
            legal,
        );

        DdBidResult {
            action,
            suit_expected_pts: suit_pts,
            total_solves,
        }
    }

    /// DD-evaluate own best suits for overcall.
    fn dd_overcall(
        &mut self,
        state: &GameState,
        player: u8,
        hand: CardSet,
        legal: u64,
        rng: &mut impl Rng,
    ) -> DdBidResult {
        let team = GameState::player_team(player);
        let opponent_value = state.last_bid_value;

        // Don't compete above overcall cap
        if opponent_value >= self.config.overcall_cap {
            return DdBidResult {
                action: BID_PASS,
                suit_expected_pts: [f32::NAN; 4],
                total_solves: 0,
            };
        }

        // Candidate suits: exclude opponent's suit, apply prefilter
        let mut candidates = Vec::new();
        for suit_idx in 0..4u8 {
            if suit_idx == state.last_bid_suit {
                continue;
            }
            let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
            if score >= self.config.prefilter_threshold {
                if !self.config.quality_gate || quality_ok(hand, Suit::from_u8(suit_idx)) {
                    candidates.push(suit_idx);
                }
            }
        }

        if candidates.is_empty() {
            return DdBidResult {
                action: BID_PASS,
                suit_expected_pts: [f32::NAN; 4],
                total_solves: 0,
            };
        }

        let (suit_pts, total_solves) =
            self.dd_evaluate_suits(state, player, &candidates, self.config.overcall_dets, rng);

        let (best_suit, best_pts) = self.pick_best_suit(&suit_pts, team);

        let action = self.points_to_bid(
            best_pts,
            best_suit,
            opponent_value,
            self.config.overcall_cap,
            legal,
        );

        DdBidResult {
            action,
            suit_expected_pts: suit_pts,
            total_solves,
        }
    }

    /// Pre-filter candidate suits using heuristic evaluation.
    fn prefilter_suits(&self, hand: CardSet) -> Vec<u8> {
        let mut candidates = Vec::new();
        for suit_idx in 0..4u8 {
            let score = evaluate_for_trump(hand, Suit::from_u8(suit_idx));
            if score >= self.config.prefilter_threshold {
                if !self.config.quality_gate || quality_ok(hand, Suit::from_u8(suit_idx)) {
                    candidates.push(suit_idx);
                }
            }
        }
        candidates
    }

    /// DD-evaluate a set of candidate suits.
    /// Returns (per-suit average NS points [4], total solves performed).
    fn dd_evaluate_suits(
        &mut self,
        state: &GameState,
        observer: u8,
        candidates: &[u8],
        num_dets: u32,
        rng: &mut impl Rng,
    ) -> ([f32; 4], u32) {
        let mut suit_sum = [0i64; 4];
        let mut suit_count = [0u32; 4];
        let mut total_solves = 0u32;

        // The dealer for DD setup: use the actual game dealer so trick lead is correct
        let dealer = state.dealer;

        for _ in 0..num_dets {
            // Determinize: sample opponent hands from observer's perspective
            let det = match determinize_greedy(state, observer, rng) {
                Some(s) => s,
                None => continue,
            };

            for &suit_idx in candidates {
                let result =
                    solve_for_trump_reuse_tt(det.hands, dealer, suit_idx, &mut self.tt_buf);
                suit_sum[suit_idx as usize] += result[0] as i64; // NS points
                suit_count[suit_idx as usize] += 1;
                total_solves += 1;
            }
        }

        let mut suit_avg = [f32::NAN; 4];
        for &suit_idx in candidates {
            let i = suit_idx as usize;
            if suit_count[i] > 0 {
                suit_avg[i] = suit_sum[i] as f32 / suit_count[i] as f32;
            }
        }

        (suit_avg, total_solves)
    }

    /// Pick the best suit for the given team from per-suit NS averages.
    fn pick_best_suit(&self, suit_pts: &[f32; 4], team: u8) -> (u8, f32) {
        let mut best_suit = 0u8;
        let mut best_pts = f32::NEG_INFINITY;

        for suit_idx in 0..4u8 {
            let ns_avg = suit_pts[suit_idx as usize];
            if ns_avg.is_nan() {
                continue;
            }
            // Convert to team points
            let team_pts = if team == 0 { ns_avg } else { 162.0 - ns_avg };
            if team_pts > best_pts {
                best_pts = team_pts;
                best_suit = suit_idx;
            }
        }

        (best_suit, best_pts)
    }

    /// Map expected team points to a bid action.
    /// Bids X if expected >= X + margin. Must overbid `min_value`.
    fn points_to_bid(
        &self,
        expected_pts: f32,
        suit: u8,
        min_value: u8,
        cap: u8,
        legal: u64,
    ) -> u8 {
        if expected_pts.is_nan() || expected_pts < 0.0 {
            return BID_PASS;
        }

        // Try bid levels from highest to lowest
        let bid_levels: [(u8, f32); 6] = [
            (13, 130.0), // 130
            (12, 120.0), // 120
            (11, 110.0), // 110
            (10, 100.0), // 100
            (9, 90.0),   // 90
            (8, 80.0),   // 80
        ];

        let mut best_action = BID_PASS;
        for &(value_enc, threshold) in &bid_levels {
            if value_enc > cap {
                continue;
            }
            if value_enc <= min_value {
                continue;
            }
            if expected_pts >= threshold + self.config.margin as f32 {
                let action = bidding::encode_bid(value_enc, suit);
                if legal & (1u64 << action) != 0 {
                    best_action = action;
                    break; // highest viable bid
                }
            }
        }

        best_action
    }
}

/// Heuristic coinche logic (reused from improved_v2).
/// Returns Some(action) if coinche should be played, None otherwise.
fn heuristic_coinche(state: &GameState, player: u8, hand: CardSet, legal: u64) -> Option<u8> {
    if state.last_bid_value == 0 || state.coinche_state != 0 {
        return None;
    }

    let bidder_team = GameState::player_team(state.last_bidder);
    let my_team = GameState::player_team(player);

    if bidder_team == my_team {
        return None;
    }

    let their_suit = Suit::from_u8(state.last_bid_suit);
    let their_suit_bits = suit_bits(hand, their_suit);
    let trump_count = their_suit_bits.count_ones();
    let has_jack = their_suit_bits & (1 << 3) != 0;
    let has_nine = their_suit_bits & (1 << 2) != 0;

    // J+9 in opponent's suit → COINCHE
    if has_jack && has_nine && legal & (1u64 << BID_COINCHE) != 0 {
        return Some(BID_COINCHE);
    }

    // 4+ trumps + side ace → COINCHE
    if trump_count >= 4 {
        let side_aces = count_side_aces_local(hand, their_suit);
        if side_aces >= 1 && legal & (1u64 << BID_COINCHE) != 0 {
            return Some(BID_COINCHE);
        }
    }

    // Théorème 3: 0 trumps in their suit + 3 aces → COINCHE
    if trump_count == 0 {
        let total_aces = count_total_aces_local(hand);
        if total_aces >= 3 && legal & (1u64 << BID_COINCHE) != 0 {
            return Some(BID_COINCHE);
        }
    }

    None
}

fn count_side_aces_local(hand: CardSet, trump: Suit) -> u32 {
    let mut count = 0u32;
    for suit_idx in 0..4u8 {
        if suit_idx == trump as u8 {
            continue;
        }
        if suit_bits(hand, Suit::from_u8(suit_idx)) & (1 << 7) != 0 {
            count += 1;
        }
    }
    count
}

fn count_total_aces_local(hand: CardSet) -> u32 {
    let mut count = 0u32;
    for suit_idx in 0..4u8 {
        if suit_bits(hand, Suit::from_u8(suit_idx)) & (1 << 7) != 0 {
            count += 1;
        }
    }
    count
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_dd_bid_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = DdBidConfig {
            opening_dets: 4,
            response_dets: 2,
            overcall_dets: 2,
            ..Default::default()
        };
        let mut bidder = DdBidder::new(config);

        for _ in 0..50 {
            let state = GameState::deal_random(0, &mut rng);
            let action = bidder.bid(&state, &mut rng);
            let legal = state.legal_actions();
            assert!(
                legal & (1u64 << action) != 0,
                "DD bid returned illegal action {}",
                action
            );
        }
    }

    #[test]
    fn test_dd_bid_with_stats() {
        let mut rng = rand::thread_rng();
        let config = DdBidConfig {
            opening_dets: 4,
            ..Default::default()
        };
        let mut bidder = DdBidder::new(config);

        let state = GameState::deal_random(0, &mut rng);
        let result = bidder.bid_with_stats(&state, &mut rng);
        let legal = state.legal_actions();
        assert!(legal & (1u64 << result.action) != 0);
    }

    #[test]
    fn test_dd_bid_all_positions() {
        let mut rng = rand::thread_rng();
        let config = DdBidConfig {
            opening_dets: 3,
            response_dets: 2,
            overcall_dets: 2,
            ..Default::default()
        };
        let mut bidder = DdBidder::new(config);

        // Play through several bidding rounds to test all positions
        let mut found_response = false;
        let mut found_overcall = false;

        for _ in 0..200 {
            let mut state = GameState::deal_random(0, &mut rng);

            // First player opens
            let a1 = bidder.bid(&state, &mut rng);
            let legal = state.legal_actions();
            assert!(legal & (1u64 << a1) != 0);
            state.step(a1);

            if state.phase != Phase::Bidding {
                continue;
            }

            // If someone bid, test response/overcall
            if state.last_bid_value > 0 {
                let a2 = bidder.bid(&state, &mut rng);
                let legal2 = state.legal_actions();
                assert!(
                    legal2 & (1u64 << a2) != 0,
                    "Illegal action {} at position 2",
                    a2
                );

                let bidder_team = GameState::player_team(state.last_bidder);
                let my_team = GameState::player_team(state.current_player);

                if bidder_team == my_team {
                    // This is partner responding
                    found_response = true;
                } else {
                    found_overcall = true;
                }

                state.step(a2);
                if state.phase != Phase::Bidding {
                    continue;
                }

                // Third player
                let a3 = bidder.bid(&state, &mut rng);
                let legal3 = state.legal_actions();
                assert!(
                    legal3 & (1u64 << a3) != 0,
                    "Illegal action {} at position 3",
                    a3
                );
            }

            if found_response && found_overcall {
                break;
            }
        }
    }

    #[test]
    fn test_points_to_bid_mapping() {
        let config = DdBidConfig::default(); // margin=15
        let bidder = DdBidder::new(config);
        let legal = u64::MAX; // all actions legal

        // 95 expected = 80+15 → should bid 80
        let action = bidder.points_to_bid(95.0, 0, 0, 12, legal);
        assert_ne!(action, BID_PASS);
        let (val, suit) = bidding::decode_bid(action);
        assert_eq!(val, 8); // 80
        assert_eq!(suit, 0);

        // 105 = 90+15 → should bid 90
        let action = bidder.points_to_bid(105.0, 1, 0, 12, legal);
        let (val, _) = bidding::decode_bid(action);
        assert_eq!(val, 9); // 90

        // 135 = 120+15 → should bid 120
        let action = bidder.points_to_bid(135.0, 2, 0, 12, legal);
        let (val, _) = bidding::decode_bid(action);
        assert_eq!(val, 12); // 120

        // 70 expected → too low for 80 → PASS
        let action = bidder.points_to_bid(70.0, 0, 0, 12, legal);
        assert_eq!(action, BID_PASS);

        // Respect cap: 145 expected but cap=11 (110) → bid 110
        let action = bidder.points_to_bid(145.0, 0, 0, 11, legal);
        let (val, _) = bidding::decode_bid(action);
        assert_eq!(val, 11); // 110

        // Must overbid min_value: expected=100 margin=15 → 80+15=95, but min_value=8 → need >80 → try 90
        // 90+15=105 > 100 → no. So PASS since can't meet threshold for any above min_value
        let action = bidder.points_to_bid(100.0, 0, 8, 12, legal);
        // 100 >= 90+15? 100>=105? No → try 80: 100>=95? Yes but 8 <= min_value=8 → skip
        assert_eq!(action, BID_PASS);

        // Can overbid: expected=110, min_value=8, margin=15 → 90+15=105<=110 → bid 90
        let action = bidder.points_to_bid(110.0, 0, 8, 12, legal);
        let (val, _) = bidding::decode_bid(action);
        assert_eq!(val, 9); // 90
    }

    #[test]
    fn test_dd_bid_full_game() {
        use crate::rollout::select_nth_bit;

        let mut rng = rand::thread_rng();
        let config = DdBidConfig {
            opening_dets: 3,
            response_dets: 2,
            overcall_dets: 2,
            ..Default::default()
        };
        let mut bidder = DdBidder::new(config);

        // Play 10 full games with DD bidding + random card play
        let mut completed = 0;
        for _ in 0..50 {
            let mut state = GameState::deal_random(0, &mut rng);

            while !state.is_terminal() {
                let legal = state.legal_actions();
                let action = if state.phase == Phase::Bidding {
                    bidder.bid(&state, &mut rng)
                } else {
                    // Random card play
                    let count = legal.count_ones();
                    select_nth_bit(legal, rng.gen_range(0..count))
                };
                assert!(
                    legal & (1u64 << action) != 0,
                    "Illegal action {} in phase {:?}",
                    action,
                    state.phase
                );
                state.step(action);
            }
            completed += 1;
            if completed >= 10 {
                break;
            }
        }
        assert!(completed >= 10, "Only completed {} games", completed);
    }
}
