use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use std::collections::HashMap;

use colver_core::bid_eval;
use colver_core::bidding;
use colver_core::card;
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{Contract, GameState, Phase};

const OBS_V2_DIM: usize = 372;

/// Find the highest trump strength currently on the trick (before current player's move).
fn best_trump_strength_on_trick(state: &GameState) -> Option<u8> {
    let trump = state.contract.trump;
    let mut best: Option<u8> = None;
    for i in 0..state.trick_count {
        let seat = (state.trick_lead + i) % 4;
        let c = state.current_trick[seat as usize];
        if c != card::EMPTY && (c >> 3) == trump {
            let strength = card::TRUMP_STRENGTH[(c & 7) as usize];
            best = Some(match best {
                Some(b) => b.max(strength),
                None => strength,
            });
        }
    }
    best
}

/// Determine the currently winning seat in a partial trick.
fn compute_partial_trick_winner(state: &GameState) -> Option<u8> {
    if state.trick_count == 0 {
        return None;
    }
    let trump = state.contract.trump;
    let lead_card = state.current_trick[state.trick_lead as usize];
    let lead_suit = lead_card >> 3;

    let mut best_seat = state.trick_lead;
    let mut best_is_trump = lead_suit == trump;
    let mut best_val: u8 = if best_is_trump {
        card::TRUMP_STRENGTH[(lead_card & 7) as usize]
    } else {
        lead_card & 7
    };

    for i in 1..state.trick_count {
        let seat = (state.trick_lead + i) % 4;
        let c = state.current_trick[seat as usize];
        if c == card::EMPTY {
            continue;
        }
        let suit = c >> 3;
        let is_trump = suit == trump;

        if is_trump && !best_is_trump {
            best_seat = seat;
            best_is_trump = true;
            best_val = card::TRUMP_STRENGTH[(c & 7) as usize];
        } else if is_trump && best_is_trump {
            let s = card::TRUMP_STRENGTH[(c & 7) as usize];
            if s > best_val {
                best_seat = seat;
                best_val = s;
            }
        } else if !is_trump && !best_is_trump && suit == lead_suit {
            let r = c & 7;
            if r > best_val {
                best_seat = seat;
                best_val = r;
            }
        }
    }
    Some(best_seat)
}

/// Track a play action: update played_by mask and trump ceiling.
fn track_play(
    state: &GameState,
    action: u8,
    played_by: &mut [u32; 4],
    trump_ceiling: &mut [u8; 4],
) {
    if state.phase != Phase::Playing {
        return;
    }
    let player = state.current_player() as usize;
    played_by[player] |= 1u32 << action;

    let card_suit = action >> 3;
    let trump = state.contract.trump;
    if card_suit == trump {
        if let Some(best_str) = best_trump_strength_on_trick(state) {
            let played_str = card::TRUMP_STRENGTH[(action & 7) as usize];
            if played_str < best_str && best_str > 0 {
                trump_ceiling[player] = trump_ceiling[player].min(best_str - 1);
            }
        }
    }
}

/// Build observation v2 (372 floats) from game state + tracking arrays.
fn make_observation_v2(
    state: &GameState,
    played_by: &[u32; 4],
    trump_ceiling: &[u8; 4],
) -> Vec<f32> {
    let mut obs = Vec::with_capacity(OBS_V2_DIM);
    let me = state.current_player() as usize;
    let my_team = me & 1;
    let opp_team = 1 - my_team;
    let trump = state.contract.trump;

    // Player-relative seats: [me, left_opp, partner, right_opp]
    let seats = [me, (me + 1) % 4, (me + 2) % 4, (me + 3) % 4];

    // === Block 1: My hand (32) ===
    let my_hand = state.hands[me];
    for i in 0..32u32 {
        obs.push(if my_hand & (1 << i) != 0 { 1.0 } else { 0.0 });
    }

    // === Block 2: Current trick, player-relative (128) ===
    for &seat in &seats {
        let c = state.current_trick[seat];
        for i in 0..32u8 {
            obs.push(if c != card::EMPTY && c == i { 1.0 } else { 0.0 });
        }
    }

    // Current trick union
    let mut trick_union: u32 = 0;
    for i in 0..4 {
        let c = state.current_trick[i];
        if c != card::EMPTY {
            trick_union |= 1u32 << c;
        }
    }

    // === Block 3: Per-player played cards in past tricks (96) ===
    // For left, partner, right (not me)
    for &seat in &seats[1..] {
        let past = played_by[seat] & !trick_union;
        for i in 0..32u32 {
            obs.push(if past & (1 << i) != 0 { 1.0 } else { 0.0 });
        }
    }

    // === Block 4: All played cards (32) ===
    for i in 0..32u32 {
        obs.push(if state.played_cards & (1 << i) != 0 { 1.0 } else { 0.0 });
    }

    // === Block 5: Trump-aware card point values (32) ===
    for i in 0..32u8 {
        let suit = i >> 3;
        let rank = (i & 7) as usize;
        let pts = if suit == trump {
            card::TRUMP_POINTS[rank]
        } else {
            card::PLAIN_POINTS[rank]
        };
        obs.push(pts as f32 / 20.0);
    }

    // === Block 6: Contract (7) ===
    for t in 0..4u8 {
        obs.push(if trump == t { 1.0 } else { 0.0 });
    }
    obs.push(state.contract.point_value() as f32 / 250.0);
    obs.push(if state.contract.team as usize == my_team {
        1.0
    } else {
        0.0
    });
    obs.push(state.contract.coinche as f32 / 2.0);

    // === Block 7: Void tracking (12) ===
    for &seat in &seats[1..] {
        for s in 0..4u8 {
            obs.push(if state.voids[seat] & (1 << s) != 0 {
                1.0
            } else {
                0.0
            });
        }
    }

    // === Block 8: Scoring context (12) ===
    obs.push(state.points[my_team] as f32 / 252.0);
    obs.push(state.points[opp_team] as f32 / 252.0);
    obs.push(state.tricks_won[my_team] as f32 / 8.0);
    obs.push(state.tricks_won[opp_team] as f32 / 8.0);

    // Points in current trick
    let mut trick_pts: u16 = 0;
    for i in 0..4 {
        let c = state.current_trick[i];
        if c != card::EMPTY {
            let suit = c >> 3;
            let rank = (c & 7) as usize;
            trick_pts += if suit == trump {
                card::TRUMP_POINTS[rank]
            } else {
                card::PLAIN_POINTS[rank]
            } as u16;
        }
    }
    obs.push(trick_pts as f32 / 62.0);

    // Remaining points in play
    let scored = state.points[0] as u16 + state.points[1] as u16;
    obs.push(152u16.saturating_sub(scored) as f32 / 152.0);

    // Points needed for contract (0 if not taker)
    let cv = state.contract.point_value();
    if state.contract.team as usize == my_team && cv > state.points[my_team] as u16 {
        obs.push((cv - state.points[my_team] as u16) as f32 / 252.0);
    } else {
        obs.push(0.0);
    }

    // Belote
    obs.push(state.belote[my_team] as f32 / 2.0);
    obs.push(state.belote[opp_team] as f32 / 2.0);

    // Trick number
    let trick_num = state.tricks_won[0] + state.tricks_won[1];
    obs.push(trick_num as f32 / 7.0);

    // Position in trick
    obs.push(state.trick_count as f32 / 3.0);

    // Cards in my hand
    obs.push(my_hand.count_ones() as f32 / 8.0);

    // === Block 9: Tactical features (21) ===
    let trump_mask: u32 = 0xFF << (trump as u32 * 8);
    // Cards still in hands (not in completed tricks or current trick)
    let remaining = !(state.played_cards | trick_union);

    // My trump count
    obs.push((my_hand & trump_mask).count_ones() as f32 / 8.0);

    // Remaining trumps in play (in hands, not yet played)
    let remaining_trumps = remaining & trump_mask;
    obs.push(remaining_trumps.count_ones() as f32 / 8.0);

    // I hold master trump
    let has_master_trump = if remaining_trumps != 0 {
        let trump_base = trump as u32 * 8;
        let mut master_bit: u32 = 0;
        let mut master_str: u8 = 0;
        for rank in 0..8u8 {
            let bit = 1u32 << (trump_base + rank as u32);
            if remaining_trumps & bit != 0 {
                let s = card::TRUMP_STRENGTH[rank as usize];
                if s >= master_str {
                    master_str = s;
                    master_bit = bit;
                }
            }
        }
        my_hand & master_bit != 0
    } else {
        false
    };
    obs.push(if has_master_trump { 1.0 } else { 0.0 });

    // I hold master in each suit (4)
    for suit in 0..4u8 {
        let smask: u32 = 0xFF << (suit as u32 * 8);
        let rem_suit = remaining & smask;
        let has_master = if rem_suit != 0 {
            if suit == trump {
                let base = suit as u32 * 8;
                let mut mb: u32 = 0;
                let mut ms: u8 = 0;
                for rank in 0..8u8 {
                    let bit = 1u32 << (base + rank as u32);
                    if rem_suit & bit != 0 {
                        let s = card::TRUMP_STRENGTH[rank as usize];
                        if s >= ms {
                            ms = s;
                            mb = bit;
                        }
                    }
                }
                my_hand & mb != 0
            } else {
                // Plain: highest rank = highest bit set in rem_suit
                let highest = 1u32 << (31 - rem_suit.leading_zeros());
                my_hand & highest != 0
            }
        } else {
            false
        };
        obs.push(if has_master { 1.0 } else { 0.0 });
    }

    // Remaining cards per suit (4)
    for suit in 0..4u8 {
        let smask: u32 = 0xFF << (suit as u32 * 8);
        obs.push((remaining & smask).count_ones() as f32 / 8.0);
    }

    // Partner winning current trick
    let partner_winning = if state.trick_count >= 1 {
        compute_partial_trick_winner(state)
            .map(|w| w as usize == seats[2])
            .unwrap_or(false)
    } else {
        false
    };
    obs.push(if partner_winning { 1.0 } else { 0.0 });

    // Led suit is trump (0 if leader)
    let is_leader = state.trick_count == 0;
    if is_leader {
        obs.push(0.0);
    } else {
        let lead_card = state.current_trick[state.trick_lead as usize];
        obs.push(if (lead_card >> 3) == trump {
            1.0
        } else {
            0.0
        });
    }

    // Led suit one-hot (4, zeros if leader)
    if is_leader {
        for _ in 0..4 {
            obs.push(0.0);
        }
    } else {
        let lead_card = state.current_trick[state.trick_lead as usize];
        let lead_suit = lead_card >> 3;
        for s in 0..4u8 {
            obs.push(if lead_suit == s { 1.0 } else { 0.0 });
        }
    }

    // My team led this trick
    obs.push(if (state.trick_lead as usize & 1) == my_team {
        1.0
    } else {
        0.0
    });

    // Trump ceiling for [left, partner, right]
    for &seat in &seats[1..] {
        obs.push(trump_ceiling[seat] as f32 / 7.0);
    }

    debug_assert_eq!(
        obs.len(),
        OBS_V2_DIM,
        "obs v2 len = {}, expected {}",
        obs.len(),
        OBS_V2_DIM
    );
    obs
}

fn legal_actions_list(state: &GameState) -> Vec<u8> {
    let mask = state.legal_actions();
    let mut actions = Vec::new();
    let mut remaining = mask;
    while remaining != 0 {
        let bit = remaining.trailing_zeros() as u8;
        actions.push(bit);
        remaining &= remaining - 1;
    }
    actions
}

fn legal_mask_vec(state: &GameState) -> Vec<f32> {
    let mask = state.legal_actions();
    let size = if state.phase == Phase::Bidding { 43 } else { 32 };
    let mut arr = vec![0.0f32; 43];
    for i in 0..size {
        if mask & (1u64 << i) != 0 {
            arr[i] = 1.0;
        }
    }
    arr
}

/// Single environment wrapping a Belote Contrée deal.
#[pyclass]
struct Env {
    state: GameState,
    rng: StdRng,
    // IS-MCTS search objects (lazily initialized)
    naive_search: Option<NaiveIsMctsSearch>,
    smart_searches: Option<[SmartIsMctsSearch; 4]>,
    smart_initialized: bool,
    // Per-player card tracking for obs v2
    played_by: [u32; 4],
    trump_ceiling: [u8; 4],
}

#[pymethods]
impl Env {
    #[new]
    fn new() -> Self {
        let mut rng = StdRng::from_entropy();
        let state = GameState::deal_random(0, &mut rng);
        Env {
            state,
            rng,
            naive_search: None,
            smart_searches: None,
            smart_initialized: false,
            played_by: [0; 4],
            trump_ceiling: [7; 4],
        }
    }

    /// Reset the environment with a new random deal.
    fn reset(&mut self) -> PyResult<(Vec<f32>, Vec<u8>)> {
        let dealer = self.rng.gen_range(0..4u8);
        self.state = GameState::deal_random(dealer, &mut self.rng);
        self.smart_initialized = false;
        self.played_by = [0; 4];
        self.trump_ceiling = [7; 4];
        Ok((
            make_observation_v2(&self.state, &self.played_by, &self.trump_ceiling),
            legal_actions_list(&self.state),
        ))
    }

    /// Take an action. Returns (observation, reward, done, legal_actions).
    fn step(&mut self, action: u8) -> PyResult<(Vec<f32>, f32, bool, Vec<u8>)> {
        let player = self.state.current_player();
        let team = GameState::player_team(player) as usize;

        track_play(
            &self.state,
            action,
            &mut self.played_by,
            &mut self.trump_ceiling,
        );
        self.state.step(action);

        let done = self.state.is_terminal();
        let reward = if done {
            self.state.rewards()[team]
        } else {
            0.0
        };

        Ok((
            make_observation_v2(&self.state, &self.played_by, &self.trump_ceiling),
            reward,
            done,
            legal_actions_list(&self.state),
        ))
    }

    /// Get current player index (0-3).
    fn current_player(&self) -> u8 {
        self.state.current_player()
    }

    /// Get current phase: 0=Bidding, 1=Playing, 2=Done.
    fn phase(&self) -> u8 {
        self.state.phase as u8
    }

    /// Is the game over?
    fn is_terminal(&self) -> bool {
        self.state.is_terminal()
    }

    /// Get legal actions as a list of action indices.
    fn legal_actions(&self) -> Vec<u8> {
        legal_actions_list(&self.state)
    }

    /// Get legal actions as a binary mask.
    fn legal_action_mask<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<f32>> {
        let arr = legal_mask_vec(&self.state);
        PyArray1::from_slice_bound(py, &arr)
    }

    /// Get rewards for both teams [NS, EW].
    fn rewards(&self) -> [f32; 2] {
        if self.state.is_terminal() {
            self.state.rewards()
        } else {
            [0.0, 0.0]
        }
    }

    /// Get improved_bid action for current state (only valid during bidding phase).
    fn bid_improved(&self) -> u8 {
        bid_eval::improved_bid(&self.state)
    }

    /// Get binary deal outcome [NS_outcome, EW_outcome].
    /// 1.0/0.0 for win/loss, 0.5/0.5 for void/tie.
    fn deal_outcome(&self) -> [f32; 2] {
        if !self.state.is_terminal() {
            return [0.0, 0.0];
        }
        let r = self.state.rewards();
        if r[0] == 0.0 && r[1] == 0.0 {
            [0.5, 0.5] // void deal
        } else if r[0] > r[1] {
            [1.0, 0.0]
        } else if r[1] > r[0] {
            [0.0, 1.0]
        } else {
            [0.5, 0.5] // tie
        }
    }

    /// Get Naive IS-MCTS action for current state. time_ms is the search budget in ms.
    fn action_naive_ismcts(&mut self, time_ms: u32) -> u8 {
        let config = NaiveIsMctsConfig {
            time_limit_ms: Some(time_ms),
            ..Default::default()
        };
        let search = self.naive_search.get_or_insert_with(NaiveIsMctsSearch::new);
        search.search(&self.state, &config, &mut self.rng)
    }

    /// Initialize Smart IS-MCTS beliefs for a new deal.
    /// Must be called after reset() and before action_smart_ismcts().
    fn smart_ismcts_init(&mut self) {
        let searches = self.smart_searches.get_or_insert_with(|| {
            [
                SmartIsMctsSearch::new(),
                SmartIsMctsSearch::new(),
                SmartIsMctsSearch::new(),
                SmartIsMctsSearch::new(),
            ]
        });
        for (player, search) in searches.iter_mut().enumerate() {
            search.init_deal(&self.state, player as u8, true);
        }
        self.smart_initialized = true;
    }

    /// Record an action for Smart IS-MCTS beliefs, then step the game.
    /// Returns (observation, reward, done, legal_actions) like step().
    /// Must be used instead of step() when using Smart IS-MCTS.
    fn smart_ismcts_step(&mut self, action: u8) -> PyResult<(Vec<f32>, f32, bool, Vec<u8>)> {
        let player = self.state.current_player();
        let team = GameState::player_team(player) as usize;

        track_play(
            &self.state,
            action,
            &mut self.played_by,
            &mut self.trump_ceiling,
        );

        // Record action in all 4 belief models before stepping
        if let Some(ref mut searches) = self.smart_searches {
            for search in searches.iter_mut() {
                search.record_action(&self.state, player, action);
            }
        }

        self.state.step(action);

        let done = self.state.is_terminal();
        let reward = if done {
            self.state.rewards()[team]
        } else {
            0.0
        };

        Ok((
            make_observation_v2(&self.state, &self.played_by, &self.trump_ceiling),
            reward,
            done,
            legal_actions_list(&self.state),
        ))
    }

    /// Get Smart IS-MCTS action for current state. time_ms is the search budget in ms.
    /// Must call smart_ismcts_init() first and use smart_ismcts_step() for all moves.
    fn action_smart_ismcts(&mut self, time_ms: u32) -> PyResult<u8> {
        if !self.smart_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call smart_ismcts_init() first",
            ));
        }
        let player = self.state.current_player() as usize;
        let config = SmartIsMctsConfig {
            time_limit_ms: Some(time_ms),
            ..Default::default()
        };
        let searches = self.smart_searches.as_mut().unwrap();
        let action = searches[player].search(&self.state, &config, &mut self.rng);
        Ok(action)
    }

    /// Run n random rollouts from current state and return average rewards.
    fn rollout(&mut self, n: u32) -> [f32; 2] {
        colver_core::rollout::rollout_batch(&self.state, n, &mut self.rng)
    }

    /// Get a copy of the internal state as bytes (for MCTS).
    fn get_state_bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let bytes: &[u8] = unsafe {
            core::slice::from_raw_parts(
                &self.state as *const GameState as *const u8,
                core::mem::size_of::<GameState>(),
            )
        };
        PyArray1::from_slice_bound(py, bytes)
    }

    /// Restore state from bytes (for MCTS).
    fn set_state_bytes(&mut self, bytes: Vec<u8>) -> PyResult<()> {
        if bytes.len() != core::mem::size_of::<GameState>() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Invalid state bytes length",
            ));
        }
        unsafe {
            core::ptr::copy_nonoverlapping(
                bytes.as_ptr(),
                &mut self.state as *mut GameState as *mut u8,
                core::mem::size_of::<GameState>(),
            );
        }
        Ok(())
    }

    /// Get all 4 hands as lists of card indices.
    fn get_hands(&self) -> Vec<Vec<u8>> {
        self.state
            .hands
            .iter()
            .map(|&h| card::CardIter(h).collect())
            .collect()
    }

    /// Get contract info as a dict. Empty if no contract yet.
    fn get_contract(&self) -> HashMap<String, i32> {
        let c = &self.state.contract;
        if self.state.phase == Phase::Bidding && self.state.last_bid_value == 0 {
            return HashMap::new();
        }
        let mut m = HashMap::new();
        m.insert("trump".into(), c.trump as i32);
        m.insert("value".into(), c.point_value() as i32);
        m.insert("team".into(), c.team as i32);
        m.insert("coinche".into(), c.coinche as i32);
        m
    }

    /// Get current trick as 4 elements (-1 if seat hasn't played).
    fn get_current_trick(&self) -> Vec<i32> {
        self.state
            .current_trick
            .iter()
            .map(|&c| if c == card::EMPTY { -1 } else { c as i32 })
            .collect()
    }

    /// Get list of all played card indices.
    fn get_played_cards(&self) -> Vec<u8> {
        card::CardIter(self.state.played_cards).collect()
    }

    /// Get points per team [NS, EW].
    fn get_points(&self) -> [u8; 2] {
        self.state.points
    }

    /// Get tricks won per team [NS, EW].
    fn get_tricks_won(&self) -> [u8; 2] {
        self.state.tricks_won
    }

    /// Get dealer seat (0-3).
    fn get_dealer(&self) -> u8 {
        self.state.dealer
    }

    /// Get trick lead seat (0-3).
    fn get_trick_lead(&self) -> u8 {
        self.state.trick_lead
    }

    /// Get card name from index (e.g. "7S", "JH", "AD").
    #[staticmethod]
    fn card_name(card_idx: u8) -> String {
        card::card_name(card_idx)
    }

    /// Get action name. Phase 0=bidding, 1=playing.
    #[staticmethod]
    fn action_name(action: u8, phase: u8) -> String {
        if phase == 0 {
            // Bidding
            match action {
                0 => "Pass".into(),
                41 => "Coinche".into(),
                42 => "Surcoinche".into(),
                1..=40 => {
                    let (val_enc, suit_idx) = bidding::decode_bid(action);
                    let suit_names = ["S", "H", "D", "C"];
                    let value = if val_enc == 25 {
                        "Capot".to_string()
                    } else {
                        format!("{}", val_enc as u16 * 10)
                    };
                    format!("{}{}", value, suit_names[suit_idx as usize])
                }
                _ => format!("?{}", action),
            }
        } else {
            // Playing - action is card index
            card::card_name(action)
        }
    }

    /// Create env with specific hands. hands: list of 4 lists of card indices.
    #[staticmethod]
    fn deal_with_hands(dealer: u8, hands: Vec<Vec<u8>>) -> PyResult<Self> {
        if hands.len() != 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "Need exactly 4 hands",
            ));
        }
        let mut hand_sets = [0u32; 4];
        for (i, hand) in hands.iter().enumerate() {
            for &c in hand {
                if c >= 32 {
                    return Err(pyo3::exceptions::PyValueError::new_err(format!(
                        "Invalid card index: {}",
                        c
                    )));
                }
                hand_sets[i] |= card::card_to_bit(c);
            }
        }
        let state = GameState::new(dealer, hand_sets);
        Ok(Env {
            state,
            rng: StdRng::from_entropy(),
            naive_search: None,
            smart_searches: None,
            smart_initialized: false,
            played_by: [0; 4],
            trump_ceiling: [7; 4],
        })
    }

    /// Set contract manually (for analysis mode). trump: 0-3, value: e.g. 80/90/.../160/250, team: 0-1, coinche: 0-2.
    fn set_contract(&mut self, trump: u8, value: u16, team: u8, coinche: u8) {
        self.state.contract = Contract {
            trump,
            value: (value / 10) as u8,
            team,
            coinche,
        };
    }

    /// Skip bidding phase and go directly to playing.
    fn set_phase_playing(&mut self) {
        self.state.phase = Phase::Playing;
        // Trick lead = player after dealer
        self.state.trick_lead = (self.state.dealer + 1) % 4;
        self.state.current_player = self.state.trick_lead;
    }

    /// Get bidding history as list of (player, action) tuples.
    fn get_bid_history(&self) -> Vec<(u8, u8)> {
        // We don't store history in GameState, so return minimal info
        Vec::new()
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.state)
    }
}

/// Vectorized environment for batch RL training.
#[pyclass]
struct VecEnv {
    states: Vec<GameState>,
    rng: StdRng,
    // Per-env card tracking for obs v2
    played_by: Vec<[u32; 4]>,
    trump_ceiling: Vec<[u8; 4]>,
}

#[pymethods]
impl VecEnv {
    #[new]
    fn new(n: usize) -> Self {
        let mut rng = StdRng::from_entropy();
        let states: Vec<GameState> = (0..n)
            .map(|_| {
                let dealer = rng.gen_range(0..4u8);
                GameState::deal_random(dealer, &mut rng)
            })
            .collect();
        VecEnv {
            states,
            rng,
            played_by: vec![[0u32; 4]; n],
            trump_ceiling: vec![[7u8; 4]; n],
        }
    }

    /// Number of environments.
    fn num_envs(&self) -> usize {
        self.states.len()
    }

    /// Current player index (0-3) for each environment.
    fn current_players<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let v: Vec<u8> = self.states.iter().map(|s| s.current_player()).collect();
        PyArray1::from_slice_bound(py, &v)
    }

    /// Current phase (0=Bidding, 1=Playing, 2=Done) for each environment.
    fn phases<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let v: Vec<u8> = self.states.iter().map(|s| s.phase as u8).collect();
        PyArray1::from_slice_bound(py, &v)
    }

    /// Get improved_bid action for each environment (only valid for bidding-phase envs).
    fn bid_improved<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let v: Vec<u8> = self
            .states
            .iter()
            .map(|s| {
                if s.phase == Phase::Bidding {
                    bid_eval::improved_bid(s)
                } else {
                    0 // placeholder for non-bidding envs
                }
            })
            .collect();
        PyArray1::from_slice_bound(py, &v)
    }

    /// Reset all environments. Returns (observations, legal_masks).
    fn reset<'py>(
        &mut self,
        py: Python<'py>,
    ) -> PyResult<(Bound<'py, PyArray2<f32>>, Bound<'py, PyArray2<f32>>)> {
        let n = self.states.len();
        for i in 0..n {
            let dealer = self.rng.gen_range(0..4u8);
            self.states[i] = GameState::deal_random(dealer, &mut self.rng);
            self.played_by[i] = [0; 4];
            self.trump_ceiling[i] = [7; 4];
        }

        let mut obs_data = Vec::with_capacity(n * OBS_V2_DIM);
        let mut mask_data = Vec::with_capacity(n * 43);

        for i in 0..n {
            obs_data.extend(make_observation_v2(
                &self.states[i],
                &self.played_by[i],
                &self.trump_ceiling[i],
            ));
            mask_data.extend(legal_mask_vec(&self.states[i]));
        }

        let obs = numpy::PyArray::from_vec_bound(py, obs_data)
            .reshape([n, OBS_V2_DIM])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let masks = numpy::PyArray::from_vec_bound(py, mask_data)
            .reshape([n, 43])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        Ok((obs, masks))
    }

    /// Step all environments with given actions.
    /// Auto-resets terminated environments.
    /// Returns (observations, rewards, dones, legal_masks, deal_outcomes).
    /// deal_outcomes is (n, 2) with [NS_outcome, EW_outcome] — binary 1.0/0.0/0.5
    /// for terminated envs (captured before auto-reset), [0, 0] for non-terminal.
    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<u8>,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<bool>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
    )> {
        let n = self.states.len();
        assert_eq!(actions.len(), n);

        let mut rewards_vec = Vec::with_capacity(n);
        let mut dones_vec = Vec::with_capacity(n);
        let mut outcomes_vec = Vec::with_capacity(n * 2);

        for (i, &action) in actions.iter().enumerate() {
            let player = self.states[i].current_player();
            let team = GameState::player_team(player) as usize;

            track_play(
                &self.states[i],
                action,
                &mut self.played_by[i],
                &mut self.trump_ceiling[i],
            );
            self.states[i].step(action);

            let done = self.states[i].is_terminal();
            let reward = if done {
                self.states[i].rewards()[team]
            } else {
                0.0
            };

            rewards_vec.push(reward);
            dones_vec.push(done);

            // Capture deal outcomes before auto-reset
            if done {
                let r = self.states[i].rewards();
                if r[0] == 0.0 && r[1] == 0.0 {
                    // void deal (4 passes)
                    outcomes_vec.push(0.5);
                    outcomes_vec.push(0.5);
                } else if r[0] > r[1] {
                    outcomes_vec.push(1.0);
                    outcomes_vec.push(0.0);
                } else if r[1] > r[0] {
                    outcomes_vec.push(0.0);
                    outcomes_vec.push(1.0);
                } else {
                    outcomes_vec.push(0.5);
                    outcomes_vec.push(0.5);
                }

                let dealer = self.rng.gen_range(0..4u8);
                self.states[i] = GameState::deal_random(dealer, &mut self.rng);
                self.played_by[i] = [0; 4];
                self.trump_ceiling[i] = [7; 4];
            } else {
                outcomes_vec.push(0.0);
                outcomes_vec.push(0.0);
            }
        }

        let mut obs_data = Vec::with_capacity(n * OBS_V2_DIM);
        let mut mask_data = Vec::with_capacity(n * 43);

        for i in 0..n {
            obs_data.extend(make_observation_v2(
                &self.states[i],
                &self.played_by[i],
                &self.trump_ceiling[i],
            ));
            mask_data.extend(legal_mask_vec(&self.states[i]));
        }

        let obs = numpy::PyArray::from_vec_bound(py, obs_data)
            .reshape([n, OBS_V2_DIM])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let rewards = PyArray1::from_slice_bound(py, &rewards_vec);
        let dones = PyArray1::from_slice_bound(py, &dones_vec);
        let masks = numpy::PyArray::from_vec_bound(py, mask_data)
            .reshape([n, 43])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let outcomes = numpy::PyArray::from_vec_bound(py, outcomes_vec)
            .reshape([n, 2])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;

        Ok((obs, rewards, dones, masks, outcomes))
    }
}

#[pymodule]
fn colver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<VecEnv>()?;
    Ok(())
}
