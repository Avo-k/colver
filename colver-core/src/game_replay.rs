/// Game replay storage and extraction.
///
/// A game is fully deterministic from `(dealer, hands, action_sequence)` — just ~62 bytes.
/// Storing raw replays lets us re-extract any task-specific data cheaply without re-playing
/// expensive DMC/NN inference.
///
/// Binary format `COLVGM01` — variable-length records:
/// ```text
/// Header (16 bytes):
///   magic: "COLVGM01" (8 bytes)
///   num_games: u64 LE (8 bytes)
///
/// Per game:
///   dealer: u8 (1 byte)
///   hands: [u32; 4] LE (16 bytes)
///   num_actions: u8 (1 byte)
///   actions: [u8; num_actions] (variable, typically 36-48)
/// ```

use std::io::{self, Write};

use crate::belief_obs::{self, BELIEF_OBS_DIM};
use crate::card::{self, card_rank, card_suit, card_suit_u8, EMPTY, HIGHER_TRUMP_MASK, TRUMP_STRENGTH};
use crate::dmc_obs::EnvTracking;
use crate::state::{GameState, Phase};

const MAGIC: &[u8; 8] = b"COLVGM01";

/// A complete game replay: initial state + all actions taken.
pub struct GameReplay {
    pub dealer: u8,
    pub hands: [u32; 4],
    pub actions: Vec<u8>,
}

/// A single belief training sample extracted from a replay.
pub struct BeliefSample {
    pub obs: Vec<f32>,
    pub target: [u8; 32],
    pub mask: u32,
    /// V2: hard constraint mask (3 hidden players × 32 cards), 1.0 = impossible.
    pub hard_constraints: Option<[f32; 96]>,
}

impl GameReplay {
    /// Write a collection of game replays to a COLVGM01 binary file.
    pub fn write_all(path: &str, replays: &[GameReplay]) -> io::Result<()> {
        // Estimate total size: header + per-game (18 + avg ~40 actions)
        let est_size = 16 + replays.len() * 60;
        let mut buf = Vec::with_capacity(est_size);

        // Header
        buf.write_all(MAGIC)?;
        buf.write_all(&(replays.len() as u64).to_le_bytes())?;

        // Per game
        for replay in replays {
            buf.push(replay.dealer);
            for &hand in &replay.hands {
                buf.write_all(&hand.to_le_bytes())?;
            }
            buf.push(replay.actions.len() as u8);
            buf.write_all(&replay.actions)?;
        }

        if let Some(parent) = std::path::Path::new(path).parent() {
            std::fs::create_dir_all(parent).ok();
        }
        std::fs::write(path, &buf)
    }

    /// Load all game replays from a COLVGM01 binary file.
    pub fn load_all(path: &str) -> io::Result<Vec<GameReplay>> {
        let data = std::fs::read(path)?;
        if data.len() < 16 {
            return Err(io::Error::new(io::ErrorKind::InvalidData, "file too small"));
        }

        if &data[..8] != MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                format!("invalid magic: expected COLVGM01, got {:?}", &data[..8]),
            ));
        }

        let num_games = u64::from_le_bytes(data[8..16].try_into().unwrap()) as usize;
        let mut replays = Vec::with_capacity(num_games);
        let mut pos = 16;

        for _ in 0..num_games {
            if pos >= data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated game data"));
            }

            let dealer = data[pos];
            pos += 1;

            if pos + 16 > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated hands"));
            }
            let mut hands = [0u32; 4];
            for h in &mut hands {
                *h = u32::from_le_bytes(data[pos..pos + 4].try_into().unwrap());
                pos += 4;
            }

            if pos >= data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated num_actions"));
            }
            let num_actions = data[pos] as usize;
            pos += 1;

            if pos + num_actions > data.len() {
                return Err(io::Error::new(io::ErrorKind::UnexpectedEof, "truncated actions"));
            }
            let actions = data[pos..pos + num_actions].to_vec();
            pos += num_actions;

            replays.push(GameReplay { dealer, hands, actions });
        }

        Ok(replays)
    }

    /// Replay this game, calling `callback` at each step with
    /// `(&GameState, &EnvTracking, action)` before the action is applied.
    pub fn replay_with<F>(&self, mut callback: F)
    where
        F: FnMut(&GameState, &EnvTracking, u8),
    {
        let mut state = GameState::new(self.dealer, self.hands);
        let mut tracking = EnvTracking::new();
        tracking.dealer = self.dealer;

        for &action in &self.actions {
            callback(&state, &tracking, action);
            tracking.track_action(&state, action);
            state.step(action);
        }
    }

    /// Extract V2 belief samples with hard constraints.
    fn extract_belief_samples_v2_into(&self, samples: &mut Vec<BeliefSample>) {
        let mut state = GameState::new(self.dealer, self.hands);
        let mut tracking = EnvTracking::new();
        tracking.dealer = self.dealer;

        let true_hands = self.hands;
        let mut obs_buf = vec![0.0f32; BELIEF_OBS_DIM];
        let mut tracker = TrumpCeilingTracker::new();

        for &action in &self.actions {
            if state.phase == Phase::Playing {
                let observer = state.current_player();

                belief_obs::write_belief_observation(&mut obs_buf, 0, &state, &tracking, observer);

                let hard_constraints = tracker.compute_hard_constraints(&state, observer);

                // Target: player-relative card locations
                let mut target = [0u8; 32];
                for p in 0..4u8 {
                    for c in 0..32u8 {
                        if true_hands[p as usize] & (1u32 << c) != 0 {
                            let rel_p = (p + 4 - observer) % 4;
                            target[c as usize] = rel_p;
                        }
                    }
                }

                // Unknown mask
                let observer_hand = state.hands[observer as usize];
                let mut played = state.played_cards;
                for j in 0..4 {
                    let c = state.current_trick[j];
                    if c != card::EMPTY {
                        played |= 1u32 << c;
                    }
                }
                let unknown_mask = !observer_hand & !played;

                if unknown_mask != 0 {
                    samples.push(BeliefSample {
                        obs: obs_buf.clone(),
                        target,
                        mask: unknown_mask,
                        hard_constraints: Some(hard_constraints),
                    });
                }

                // Update tracker after recording sample
                tracker.record_play(&state, observer, action);
            }

            tracking.track_action(&state, action);
            state.step(action);
        }
    }

    /// Extract all belief training samples from this replay.
    ///
    /// At each playing-phase step, records:
    /// - Belief observation from the current player's perspective
    /// - Ground truth target: which player holds each card (player-relative)
    /// - Unknown mask: cards the observer doesn't know about
    fn extract_belief_samples_into(&self, samples: &mut Vec<BeliefSample>) {
        let mut state = GameState::new(self.dealer, self.hands);
        let mut tracking = EnvTracking::new();
        tracking.dealer = self.dealer;

        let true_hands = self.hands;
        let mut obs_buf = vec![0.0f32; BELIEF_OBS_DIM];

        for &action in &self.actions {
            if state.phase == Phase::Playing {
                let observer = state.current_player();

                belief_obs::write_belief_observation(&mut obs_buf, 0, &state, &tracking, observer);

                // Target: player-relative card locations
                let mut target = [0u8; 32];
                for p in 0..4u8 {
                    for c in 0..32u8 {
                        if true_hands[p as usize] & (1u32 << c) != 0 {
                            let rel_p = (p + 4 - observer) % 4;
                            target[c as usize] = rel_p;
                        }
                    }
                }

                // Unknown mask
                let observer_hand = state.hands[observer as usize];
                let mut played = state.played_cards;
                for j in 0..4 {
                    let c = state.current_trick[j];
                    if c != card::EMPTY {
                        played |= 1u32 << c;
                    }
                }
                let unknown_mask = !observer_hand & !played;

                if unknown_mask != 0 {
                    samples.push(BeliefSample {
                        obs: obs_buf.clone(),
                        target,
                        mask: unknown_mask,
                        hard_constraints: None,
                    });
                }
            }

            tracking.track_action(&state, action);
            state.step(action);
        }
    }
}

/// Tracks trump ceiling deductions during game replay.
///
/// Accumulates per-player bitmasks of trump ranks proven impossible
/// (from inability to overtrump), plus deduced trump voids (from not
/// trumping when required).
pub struct TrumpCeilingTracker {
    /// Per-player (absolute index) bitmask of impossible trump ranks.
    ceiling_mask: [u8; 4],
    /// Per-player deduced void mask (suit bitmask, same format as state.voids).
    /// Captures "void in trump" deduction that state.voids doesn't track.
    deduced_voids: [u8; 4],
}

impl TrumpCeilingTracker {
    pub fn new() -> Self {
        TrumpCeilingTracker {
            ceiling_mask: [0; 4],
            deduced_voids: [0; 4],
        }
    }

    /// Record a play action and update trump ceiling / deduced voids.
    /// `state` is the state BEFORE the action is applied.
    pub fn record_play(&mut self, state: &GameState, player: u8, card_played: u8) {
        if state.phase != Phase::Playing {
            return;
        }

        let card_s = card_suit_u8(card_played);
        let trump_suit = state.contract.trump;

        if state.trick_count > 0 {
            // Not the leader
            let lead_card = state.current_trick[state.trick_lead as usize];
            let lead_suit_idx = card_suit(lead_card) as u8;

            if card_s != lead_suit_idx {
                // Didn't follow lead suit

                if card_s != trump_suit {
                    // Didn't play trump either. If partner isn't master,
                    // rules require trumping if possible -> void in trump.
                    if !partner_is_master_before_play(state, player) {
                        self.deduced_voids[player as usize] |= 1 << trump_suit;
                    }
                }

                // Trump ceiling: played trump but couldn't overtrump
                if card_s == trump_suit {
                    if let Some(best_rank) = best_trump_rank_on_trick(state, trump_suit) {
                        let played_strength = TRUMP_STRENGTH[card_rank(card_played) as usize];
                        let best_strength = TRUMP_STRENGTH[best_rank as usize];
                        if played_strength < best_strength {
                            self.ceiling_mask[player as usize] |=
                                HIGHER_TRUMP_MASK[best_rank as usize];
                        }
                    }
                }
            } else if lead_suit_idx == trump_suit {
                // Following trump suit - check overtrump constraint
                if let Some(best_rank) = best_trump_rank_on_trick(state, trump_suit) {
                    let played_strength = TRUMP_STRENGTH[card_rank(card_played) as usize];
                    let best_strength = TRUMP_STRENGTH[best_rank as usize];
                    if played_strength < best_strength {
                        self.ceiling_mask[player as usize] |=
                            HIGHER_TRUMP_MASK[best_rank as usize];
                    }
                }
            }
        }
    }

    /// Compute hard constraints: 96 floats (3 hidden players x 32 cards).
    /// Layout: [left_opp(32), partner(32), right_opp(32)] in player-relative order.
    /// 1.0 = impossible (player cannot hold this card).
    pub fn compute_hard_constraints(&self, state: &GameState, observer: u8) -> [f32; 96] {
        let mut hc = [0.0f32; 96];
        let seats = [
            ((observer as usize + 1) % 4),
            ((observer as usize + 2) % 4),
            ((observer as usize + 3) % 4),
        ];

        let observer_hand = state.hands[observer as usize];
        let mut played = state.played_cards;
        for j in 0..4 {
            let c = state.current_trick[j];
            if c != EMPTY {
                played |= 1u32 << c;
            }
        }
        let known = observer_hand | played;
        let trump_suit = state.contract.trump;

        for (i, &seat) in seats.iter().enumerate() {
            let combined_voids = state.voids[seat] | self.deduced_voids[seat];

            for card_idx in 0..32u32 {
                let offset = i * 32 + card_idx as usize;
                let suit = (card_idx / 8) as u8;
                let rank = (card_idx % 8) as u8;

                // Known cards (observer hand, played, current trick)
                if known & (1 << card_idx) != 0 {
                    hc[offset] = 1.0;
                    continue;
                }

                // Void constraint (game-tracked + deduced)
                if combined_voids & (1 << suit) != 0 {
                    hc[offset] = 1.0;
                    continue;
                }

                // Trump ceiling
                if suit == trump_suit && self.ceiling_mask[seat] & (1 << rank) != 0 {
                    hc[offset] = 1.0;
                }
            }
        }

        hc
    }
}

/// Find the best trump rank currently on the trick (helper for TrumpCeilingTracker).
fn best_trump_rank_on_trick(state: &GameState, trump_suit: u8) -> Option<u8> {
    let trump = card::Suit::from_u8(trump_suit);
    let mut best: Option<u8> = None;
    let mut best_strength = 0u8;

    for i in 0..state.trick_count {
        let seat = (state.trick_lead + i) % 4;
        let c = state.current_trick[seat as usize];
        if c == EMPTY {
            continue;
        }
        if card_suit(c) == trump {
            let rank = card_rank(c);
            let strength = TRUMP_STRENGTH[rank as usize];
            if best.is_none() || strength > best_strength {
                best_strength = strength;
                best = Some(rank);
            }
        }
    }

    best
}

/// Check if player's partner is currently winning the trick (before player plays).
fn partner_is_master_before_play(state: &GameState, player: u8) -> bool {
    if state.trick_count < 2 {
        return false;
    }

    let partner = GameState::partner(player);
    let lead = state.trick_lead;
    let lead_card = state.current_trick[lead as usize];
    let lead_suit = card_suit(lead_card);
    let trump_suit = state.contract.trump_suit();

    let mut best_seat = lead;
    let mut has_trump = false;
    let mut best_trump_strength = 0u8;
    let mut best_lead_rank = card_rank(lead_card);
    let mut best_lead_seat = lead;

    if lead_suit == trump_suit {
        has_trump = true;
        best_trump_strength = TRUMP_STRENGTH[card_rank(lead_card) as usize];
        best_seat = lead;
    }

    for i in 1..state.trick_count {
        let seat = (lead + i) % 4;
        let c = state.current_trick[seat as usize];
        if c == EMPTY {
            continue;
        }
        let suit = card_suit(c);

        if suit == trump_suit {
            let s = TRUMP_STRENGTH[card_rank(c) as usize];
            if !has_trump || s > best_trump_strength {
                best_trump_strength = s;
                best_seat = seat;
                has_trump = true;
            }
        } else if suit == lead_suit && !has_trump {
            let r = card_rank(c);
            if r > best_lead_rank {
                best_lead_rank = r;
                best_lead_seat = seat;
            }
        }
    }

    if !has_trump {
        best_seat = best_lead_seat;
    }

    best_seat == partner
}

/// Extract all belief samples from a set of game replays (single-threaded).
pub fn extract_belief_samples(replays: &[GameReplay]) -> Vec<BeliefSample> {
    let est = replays.len() * 31;
    let mut samples = Vec::with_capacity(est);
    for replay in replays {
        replay.extract_belief_samples_into(&mut samples);
    }
    samples
}

/// Extract all belief samples using rayon for parallelism.
#[cfg(feature = "parallel")]
pub fn extract_belief_samples_parallel(replays: &[GameReplay]) -> Vec<BeliefSample> {
    use rayon::prelude::*;

    replays
        .par_iter()
        .fold(
            || Vec::with_capacity(1024),
            |mut acc, replay| {
                replay.extract_belief_samples_into(&mut acc);
                acc
            },
        )
        .reduce(
            || Vec::new(),
            |mut a, b| {
                a.extend(b);
                a
            },
        )
}

/// Extract V2 belief samples (with hard constraints) from game replays (single-threaded).
pub fn extract_belief_samples_v2(replays: &[GameReplay]) -> Vec<BeliefSample> {
    let est = replays.len() * 31;
    let mut samples = Vec::with_capacity(est);
    for replay in replays {
        replay.extract_belief_samples_v2_into(&mut samples);
    }
    samples
}

/// Extract V2 belief samples using rayon for parallelism.
#[cfg(feature = "parallel")]
pub fn extract_belief_samples_v2_parallel(replays: &[GameReplay]) -> Vec<BeliefSample> {
    use rayon::prelude::*;

    replays
        .par_iter()
        .fold(
            || Vec::with_capacity(1024),
            |mut acc, replay| {
                replay.extract_belief_samples_v2_into(&mut acc);
                acc
            },
        )
        .reduce(
            || Vec::new(),
            |mut a, b| {
                a.extend(b);
                a
            },
        )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_write_load() {
        let replays = vec![
            GameReplay {
                dealer: 0,
                hands: [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000],
                actions: vec![0, 0, 0, 5, 0, 0, 0], // 3 passes + bid + 3 passes
            },
            GameReplay {
                dealer: 2,
                hands: [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000],
                actions: vec![0, 0, 0, 0], // 4 passes (void deal)
            },
        ];

        let path = "/tmp/test_game_replay_roundtrip.bin";
        GameReplay::write_all(path, &replays).unwrap();
        let loaded = GameReplay::load_all(path).unwrap();

        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].dealer, 0);
        assert_eq!(loaded[0].hands, [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000]);
        assert_eq!(loaded[0].actions, vec![0, 0, 0, 5, 0, 0, 0]);
        assert_eq!(loaded[1].dealer, 2);
        assert_eq!(loaded[1].actions, vec![0, 0, 0, 0]);

        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_empty_file() {
        let path = "/tmp/test_game_replay_empty.bin";
        GameReplay::write_all(path, &[]).unwrap();
        let loaded = GameReplay::load_all(path).unwrap();
        assert_eq!(loaded.len(), 0);
        std::fs::remove_file(path).ok();
    }

    #[test]
    fn test_replay_with_callback() {
        let replay = GameReplay {
            dealer: 0,
            hands: [0xFF, 0xFF00, 0xFF_0000, 0xFF00_0000],
            actions: vec![0, 0, 0, 0], // 4 passes
        };

        let mut step_count = 0;
        replay.replay_with(|_state, _tracking, _action| {
            step_count += 1;
        });
        assert_eq!(step_count, 4);
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_replay_produces_valid_states() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(123);

        for _ in 0..20 {
            let dealer = rng.gen_range(0..4u8);
            let mut state = GameState::deal_random(dealer, &mut rng);
            let hands = state.hands;
            let mut tracking = EnvTracking::new();
            tracking.dealer = dealer;
            let mut actions = Vec::new();

            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = crate::rollout::select_nth_bit(legal, idx);
                actions.push(action);
                tracking.track_action(&state, action);
                state.step(action);
            }

            let replay = GameReplay { dealer, hands, actions: actions.clone() };

            // Verify replay produces same terminal state
            let mut replayed_state = GameState::new(dealer, hands);
            let mut replayed_tracking = EnvTracking::new();
            replayed_tracking.dealer = dealer;

            for &a in &actions {
                replayed_tracking.track_action(&replayed_state, a);
                replayed_state.step(a);
            }
            assert!(replayed_state.is_terminal());
            assert_eq!(replayed_state.points, state.points);

            // Verify replay_with visits correct number of steps
            let mut count = 0;
            replay.replay_with(|_, _, _| count += 1);
            assert_eq!(count, actions.len());
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_extract_belief_samples() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(456);
        let mut replays = Vec::new();

        for _ in 0..10 {
            let dealer = rng.gen_range(0..4u8);
            let mut state = GameState::deal_random(dealer, &mut rng);
            let hands = state.hands;
            let mut actions = Vec::new();

            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = crate::rollout::select_nth_bit(legal, idx);
                actions.push(action);
                state.step(action);
            }

            replays.push(GameReplay { dealer, hands, actions });
        }

        let samples = extract_belief_samples(&replays);
        assert!(!samples.is_empty());

        for sample in &samples {
            assert_eq!(sample.obs.len(), BELIEF_OBS_DIM);
            assert_ne!(sample.mask, 0);
            for &t in &sample.target {
                assert!(t < 4);
            }
        }
    }

    #[test]
    #[cfg(feature = "rand")]
    fn test_extract_belief_samples_v2() {
        use rand::rngs::StdRng;
        use rand::{Rng, SeedableRng};

        let mut rng = StdRng::seed_from_u64(789);
        let mut replays = Vec::new();

        for _ in 0..10 {
            let dealer = rng.gen_range(0..4u8);
            let mut state = GameState::deal_random(dealer, &mut rng);
            let hands = state.hands;
            let mut actions = Vec::new();

            while !state.is_terminal() {
                let legal = state.legal_actions();
                let count = legal.count_ones();
                let idx = rng.gen_range(0..count);
                let action = crate::rollout::select_nth_bit(legal, idx);
                actions.push(action);
                state.step(action);
            }

            replays.push(GameReplay { dealer, hands, actions });
        }

        let samples = extract_belief_samples_v2(&replays);
        assert!(!samples.is_empty());

        for sample in &samples {
            assert_eq!(sample.obs.len(), BELIEF_OBS_DIM);
            assert_ne!(sample.mask, 0);
            for &t in &sample.target {
                assert!(t < 4);
            }
            // V2: must have hard constraints
            let hc = sample.hard_constraints.as_ref().unwrap();
            // All values should be 0.0 or 1.0
            for &v in hc.iter() {
                assert!(v == 0.0 || v == 1.0, "hard constraint should be 0 or 1, got {}", v);
            }
            // Known cards (observer hand + played) should be marked impossible
            // At minimum, observer's own cards should be 1.0 for all hidden players
            let mut has_some_impossible = false;
            for &v in hc.iter() {
                if v == 1.0 {
                    has_some_impossible = true;
                    break;
                }
            }
            assert!(has_some_impossible, "V2 hard constraints should have some impossible cards");
        }
    }
}
