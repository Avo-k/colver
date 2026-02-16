use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use std::collections::HashMap;
use std::time::Instant;

use colver_core::bid_eval;
use colver_core::bidding;
use colver_core::card;
use colver_core::card::Suit;
use colver_core::dmc_net::DmcNet;
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{Contract, GameState, Phase};

const OBS_V2_DIM: usize = 415;
const BID_HISTORY_FLOATS: usize = 72; // 12 slots × 6 floats per slot

/// Track a play action: update played_by mask.
fn track_play(
    state: &GameState,
    action: u8,
    played_by: &mut [u32; 4],
) {
    if state.phase != Phase::Playing {
        return;
    }
    let player = state.current_player() as usize;
    played_by[player] |= 1u32 << action;
}

/// Encode bid history into 72 floats (12 slots × 6 floats).
/// Slots are in player-relative order: [me, left, partner, right] × 3 rounds.
fn encode_bid_history(
    bid_history: &[(u8, u8)], // (seat, action) pairs
    me: usize,
    dealer: u8,
) -> [f32; BID_HISTORY_FLOATS] {
    let mut out = [0.0f32; BID_HISTORY_FLOATS];

    // First bidder is after dealer
    let first_bidder = ((dealer + 1) % 4) as usize;
    // Relative offset: how many slots of padding before first bid
    let offset = (first_bidder + 4 - me) % 4;

    // Use last 12 actions if history is longer (extremely rare)
    let history = if bid_history.len() > 12 {
        &bid_history[bid_history.len() - 12..]
    } else {
        bid_history
    };

    for (i, &(_seat, action)) in history.iter().enumerate() {
        let slot = offset + i;
        if slot >= 12 {
            break;
        }
        let base = slot * 6;

        match action {
            0 => {
                // Pass
                out[base] = 0.2;
            }
            41 => {
                // Coinche
                out[base] = 0.8;
            }
            42 => {
                // Surcoinche
                out[base] = 1.0;
            }
            1..=40 => {
                let (val_enc, suit_idx) = bidding::decode_bid(action);
                if val_enc == 25 {
                    // Capot
                    out[base] = 0.6;
                    out[base + 1] = 1.0;
                } else {
                    // Regular bid
                    out[base] = 0.4;
                    out[base + 1] = (val_enc as f32 * 10.0) / 250.0;
                }
                out[base + 2 + suit_idx as usize] = 1.0;
            }
            _ => {}
        }
    }

    out
}

/// Build observation v4 (415 floats) from game state + tracking arrays.
fn make_observation_v2(
    state: &GameState,
    played_by: &[u32; 4],
    play_order: &[u8],
    bid_history: &[(u8, u8)],
    dealer: u8,
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

    // === Block 8: Scoring context (4) ===
    obs.push(state.points[my_team] as f32 / 252.0);
    obs.push(state.points[opp_team] as f32 / 252.0);
    obs.push(state.tricks_won[my_team] as f32 / 8.0);
    obs.push(state.tricks_won[opp_team] as f32 / 8.0);

    // === Block 10: Bid history (72) ===
    let bid_enc = encode_bid_history(bid_history, me, dealer);
    obs.extend_from_slice(&bid_enc);

    // === Block 11: Card trick index (32) ===
    // For each card 0-31: trick_number/8.0 (1-8), 0.0 if not played
    let mut card_trick = [0.0f32; 32];
    let mut card_seq = [0.0f32; 32];
    for (i, &card) in play_order.iter().enumerate() {
        card_trick[card as usize] = (i / 4 + 1) as f32 / 8.0;
        card_seq[card as usize] = (i % 4 + 1) as f32 / 4.0;
    }
    obs.extend_from_slice(&card_trick);

    // === Block 12: Card sequence index (32) ===
    // For each card 0-31: position_in_trick/4.0 (1-4), 0.0 if not played
    obs.extend_from_slice(&card_seq);

    debug_assert_eq!(
        obs.len(),
        OBS_V2_DIM,
        "obs v4 len = {}, expected {}",
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
    // Chronological play order (card indices) for timing features
    play_order: Vec<u8>,
    // Bid history: (seat, bid_action) pairs, cleared on reset
    bid_history: Vec<(u8, u8)>,
    // DMC Q-network (loaded lazily)
    dmc_net: Option<DmcNet>,
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
            play_order: Vec::with_capacity(32),
            bid_history: Vec::new(),
            dmc_net: None,
        }
    }

    /// Reset the environment with a new random deal.
    fn reset(&mut self) -> PyResult<(Vec<f32>, Vec<u8>)> {
        let dealer = self.rng.gen_range(0..4u8);
        self.state = GameState::deal_random(dealer, &mut self.rng);
        self.smart_initialized = false;
        self.played_by = [0; 4];
        self.play_order.clear();
        self.bid_history.clear();
        Ok((
            make_observation_v2(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer),
            legal_actions_list(&self.state),
        ))
    }

    /// Take an action. Returns (observation, reward, done, legal_actions).
    fn step(&mut self, action: u8) -> PyResult<(Vec<f32>, f32, bool, Vec<u8>)> {
        let player = self.state.current_player();
        let team = GameState::player_team(player) as usize;

        if self.state.phase == Phase::Bidding {
            self.bid_history.push((player, action));
        }
        if self.state.phase == Phase::Playing {
            self.play_order.push(action);
        }
        track_play(
            &self.state,
            action,
            &mut self.played_by,
        );
        self.state.step(action);

        let done = self.state.is_terminal();
        let reward = if done {
            self.state.rewards()[team]
        } else {
            0.0
        };

        Ok((
            make_observation_v2(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer),
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
        bid_eval::improved_v2_bid(&self.state)
    }

    /// Get improved_v2_bid action for current state (only valid during bidding phase).
    fn bid_improved_v2(&self) -> u8 {
        bid_eval::improved_v2_bid(&self.state)
    }

    /// Get roro_bid action for current state (only valid during bidding phase).
    fn bid_roro(&self) -> u8 {
        bid_eval::roro_bid(&self.state)
    }

    /// Get petit_bide_bid action for current state (only valid during bidding phase).
    fn bid_petit_bide(&self) -> u8 {
        bid_eval::petit_bide_bid(&self.state)
    }

    /// Get moelleux_bid action for current state (only valid during bidding phase).
    fn bid_moelleux(&self) -> u8 {
        bid_eval::moelleux_bid(&self.state)
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

        if self.state.phase == Phase::Bidding {
            self.bid_history.push((player, action));
        }
        if self.state.phase == Phase::Playing {
            self.play_order.push(action);
        }
        track_play(
            &self.state,
            action,
            &mut self.played_by,
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
            make_observation_v2(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer),
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

    /// Get belote state per team [NS, EW]. 0=none, 1=belote, 2=rebelote.
    fn get_belote(&self) -> [u8; 2] {
        self.state.belote
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
            play_order: Vec::new(),
            bid_history: Vec::new(),
            dmc_net: None,
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
        self.bid_history.clone()
    }

    /// Get Naive IS-MCTS action with search statistics.
    /// Returns dict: {best_action, visit_counts: [(action, visits)...], root_visits, elapsed_ms}
    /// During bidding, returns bid_improved_v2() with minimal stats.
    fn action_naive_ismcts_with_stats<'py>(&mut self, py: Python<'py>, time_ms: u32) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        if self.state.phase == Phase::Bidding {
            let action = bid_eval::improved_v2_bid(&self.state);
            dict.set_item("best_action", action)?;
            dict.set_item("visit_counts", Vec::<(u8, u32)>::new())?;
            dict.set_item("root_visits", 0u32)?;
            dict.set_item("elapsed_ms", 0.0f64)?;
            return Ok(dict);
        }

        let config = NaiveIsMctsConfig {
            time_limit_ms: Some(time_ms),
            ..Default::default()
        };
        let search = self.naive_search.get_or_insert_with(NaiveIsMctsSearch::new);
        let start = Instant::now();
        let result = search.search_with_stats(&self.state, &config, &mut self.rng);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        dict.set_item("best_action", result.best_action)?;
        dict.set_item("visit_counts", result.visit_counts)?;
        dict.set_item("root_visits", result.root_visits)?;
        dict.set_item("elapsed_ms", elapsed)?;
        Ok(dict)
    }

    /// Get Smart IS-MCTS action with search statistics.
    /// Returns dict: {best_action, visit_counts: [(action, visits)...], root_visits, elapsed_ms}
    /// During bidding, returns bid_improved_v2() with minimal stats.
    fn action_smart_ismcts_with_stats<'py>(&mut self, py: Python<'py>, time_ms: u32) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        if self.state.phase == Phase::Bidding {
            let action = bid_eval::improved_v2_bid(&self.state);
            dict.set_item("best_action", action)?;
            dict.set_item("visit_counts", Vec::<(u8, u32)>::new())?;
            dict.set_item("root_visits", 0u32)?;
            dict.set_item("elapsed_ms", 0.0f64)?;
            return Ok(dict);
        }

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
        let start = Instant::now();
        let result = searches[player].search_with_stats(&self.state, &config, &mut self.rng);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        dict.set_item("best_action", result.best_action)?;
        dict.set_item("visit_counts", result.visit_counts)?;
        dict.set_item("root_visits", result.root_visits)?;
        dict.set_item("elapsed_ms", elapsed)?;
        Ok(dict)
    }

    /// Evaluate a player's hand for all 4 trump suits.
    /// Returns dict: {scores: [s0,s1,s2,s3], best_suit, best_score}
    fn evaluate_hand<'py>(&self, py: Python<'py>, player: u8) -> PyResult<Bound<'py, PyDict>> {
        if player >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("player must be 0-3"));
        }
        let hand = self.state.hands[player as usize];
        let mut scores = [0u16; 4];
        let mut best_suit = 0u8;
        let mut best_score = 0u16;
        for suit in 0..4u8 {
            let s = bid_eval::evaluate_for_trump(hand, Suit::from_u8(suit));
            scores[suit as usize] = s;
            if s > best_score {
                best_score = s;
                best_suit = suit;
            }
        }
        let dict = PyDict::new_bound(py);
        dict.set_item("scores", scores.to_vec())?;
        dict.set_item("best_suit", best_suit)?;
        dict.set_item("best_score", best_score)?;
        Ok(dict)
    }

    /// Get heuristic play action (perfect-info deterministic heuristic).
    /// Only valid during play phase.
    fn action_heuristic_play(&self) -> PyResult<u8> {
        if self.state.phase != Phase::Playing {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Heuristic play only valid during play phase",
            ));
        }
        Ok(rollout::heuristic_play_action(&self.state))
    }

    /// Get a random legal action.
    fn action_random(&mut self) -> u8 {
        let mask = self.state.legal_actions();
        let count = mask.count_ones();
        let n = self.rng.gen_range(0..count);
        rollout::select_nth_bit(mask, n)
    }

    /// Get observation v4 (415 floats) for current state.
    fn get_observation_v2(&self) -> Vec<f32> {
        make_observation_v2(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer)
    }

    /// Load DMC Q-network weights from a raw binary file.
    /// Call once, then use action_dmc_with_stats() for inference.
    #[pyo3(signature = (path, hidden=None))]
    fn load_dmc_model(&mut self, path: &str, hidden: Option<usize>) -> PyResult<()> {
        let h = hidden.unwrap_or(1024);
        let net = DmcNet::load_with_hidden(path, h).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load DMC model: {}", e))
        })?;
        self.dmc_net = Some(net);
        Ok(())
    }

    /// Check if DMC model is loaded.
    fn has_dmc_model(&self) -> bool {
        self.dmc_net.is_some()
    }

    /// Get DMC Q-network action with Q-value statistics.
    /// Returns dict: {best_action, q_values: [(action, q)...], elapsed_ms}
    /// Only valid during play phase. Requires load_dmc_model() first.
    fn action_dmc_with_stats<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.state.phase != Phase::Playing {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "DMC only valid during play phase",
            ));
        }
        let net = self.dmc_net.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Call load_dmc_model() first")
        })?;

        let obs_full = make_observation_v2(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer);
        // Truncate obs to match model's expected obs_dim (backward compat with 372-dim models)
        let obs = &obs_full[..net.obs_dim()];
        let legal_mask = self.state.legal_actions() as u32;

        let start = Instant::now();
        let (best_action, q_values) = net.best_action(obs, legal_mask);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let dict = PyDict::new_bound(py);
        dict.set_item("best_action", best_action)?;
        dict.set_item("q_values", q_values)?;
        dict.set_item("elapsed_ms", elapsed)?;
        Ok(dict)
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
    // Per-env chronological play order for timing features
    play_orders: Vec<Vec<u8>>,
    // Per-env bid history
    bid_histories: Vec<Vec<(u8, u8)>>,
    // Per-env bidding strategy: 0=improved (default), 1-6=BidParams presets
    // Presets: 1=ultra_conservative, 2=conservative, 3=moderate,
    //          4=balanced, 5=aggressive, 6=very_aggressive, 7=heuristic
    bid_strategy: Vec<u8>,
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
            play_orders: (0..n).map(|_| Vec::with_capacity(32)).collect(),
            bid_histories: vec![Vec::new(); n],
            bid_strategy: vec![0u8; n],
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

    /// Set bidding strategy per environment.
    /// 0=improved (default), 1=ultra_conservative, 2=conservative, 3=moderate,
    /// 4=balanced, 5=aggressive, 6=very_aggressive, 7=heuristic.
    fn set_bid_strategies(&mut self, strategies: Vec<u8>) {
        assert_eq!(strategies.len(), self.states.len());
        self.bid_strategy = strategies;
    }

    /// Get bid action for each environment using its assigned strategy.
    fn bid_improved<'py>(&self, py: Python<'py>) -> Bound<'py, PyArray1<u8>> {
        let presets = bid_eval::BidParams::all_presets();
        let v: Vec<u8> = self
            .states
            .iter()
            .enumerate()
            .map(|(i, s)| {
                if s.phase != Phase::Bidding {
                    return 0;
                }
                match self.bid_strategy[i] {
                    0 => bid_eval::improved_v2_bid(s),
                    idx @ 1..=6 => {
                        bid_eval::parametric_bid(s, &presets[(idx - 1) as usize])
                    }
                    7 => bid_eval::heuristic_bid(s),
                    8 => bid_eval::roro_bid(s),
                    _ => bid_eval::improved_v2_bid(s),
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
            self.play_orders[i].clear();
            self.bid_histories[i].clear();
        }

        let mut obs_data = Vec::with_capacity(n * OBS_V2_DIM);
        let mut mask_data = Vec::with_capacity(n * 43);

        for i in 0..n {
            obs_data.extend(make_observation_v2(
                &self.states[i],
                &self.played_by[i],
                &self.play_orders[i],
                &self.bid_histories[i],
                self.states[i].dealer,
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

            if self.states[i].phase == Phase::Bidding {
                self.bid_histories[i].push((player, action));
            }
            if self.states[i].phase == Phase::Playing {
                self.play_orders[i].push(action);
            }
            track_play(
                &self.states[i],
                action,
                &mut self.played_by[i],
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
                self.play_orders[i].clear();
                self.bid_histories[i].clear();
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
                &self.play_orders[i],
                &self.bid_histories[i],
                self.states[i].dealer,
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

// ============================================================
// Rust-based Prioritized Experience Replay for fast training
// ============================================================

/// Binary sum tree for O(log n) proportional sampling.
struct SumTree {
    capacity: usize,
    tree: Vec<f64>,
    data_pointer: usize,
    n_entries: usize,
}

impl SumTree {
    fn new(capacity: usize) -> Self {
        SumTree {
            capacity,
            tree: vec![0.0f64; 2 * capacity],
            data_pointer: 0,
            n_entries: 0,
        }
    }

    #[inline]
    fn update(&mut self, idx: usize, priority: f64) {
        let tree_idx = idx + self.capacity;
        let change = priority - self.tree[tree_idx];
        self.tree[tree_idx] = priority;
        let mut i = tree_idx >> 1;
        while i >= 1 {
            self.tree[i] += change;
            i >>= 1;
        }
    }

    #[inline]
    fn add(&mut self, priority: f64) -> usize {
        let idx = self.data_pointer;
        self.update(idx, priority);
        self.data_pointer = (self.data_pointer + 1) % self.capacity;
        if self.n_entries < self.capacity {
            self.n_entries += 1;
        }
        idx
    }

    #[inline]
    fn get(&self, mut s: f64) -> usize {
        let mut idx = 1;
        let cap2 = 2 * self.capacity;
        loop {
            let left = 2 * idx;
            if left >= cap2 {
                break;
            }
            if s <= self.tree[left] {
                idx = left;
            } else {
                s -= self.tree[left];
                idx = left + 1;
            }
        }
        idx - self.capacity
    }

    #[inline]
    fn total(&self) -> f64 {
        self.tree[1]
    }

    #[inline]
    fn priority(&self, idx: usize) -> f64 {
        self.tree[idx + self.capacity]
    }
}

const PER_OBS_DIM: usize = OBS_V2_DIM; // 444
const PER_NUM_CARDS: usize = 32;

/// Rust-based Prioritized Experience Replay buffer.
#[pyclass]
struct PrioritizedReplayBuffer {
    capacity: usize,
    alpha: f64,
    tree: SumTree,
    obs: Vec<f32>,       // capacity * OBS_DIM, row-major
    masks: Vec<f32>,     // capacity * NUM_CARDS
    actions: Vec<i64>,
    returns: Vec<f32>,
    max_priority: f64,
    cached_priority: f64,
    size: usize,
}

#[pymethods]
impl PrioritizedReplayBuffer {
    #[new]
    #[pyo3(signature = (capacity=2_000_000, alpha=0.6))]
    fn new(capacity: usize, alpha: f64) -> Self {
        let cached_priority = 1.0f64.powf(alpha);
        PrioritizedReplayBuffer {
            capacity,
            alpha,
            tree: SumTree::new(capacity),
            obs: vec![0.0f32; capacity * PER_OBS_DIM],
            masks: vec![0.0f32; capacity * PER_NUM_CARDS],
            actions: vec![0i64; capacity],
            returns: vec![0.0f32; capacity],
            max_priority: 1.0,
            cached_priority,
            size: 0,
        }
    }

    /// Current buffer size.
    #[getter]
    fn size(&self) -> usize {
        self.size
    }

    /// Push a batch of transitions with max priority.
    /// obs: (n, 372), masks: (n, 32), actions: (n,), returns: (n,)
    fn push_batch(
        &mut self,
        obs: numpy::PyReadonlyArray2<f32>,
        masks: numpy::PyReadonlyArray2<f32>,
        actions: numpy::PyReadonlyArray1<i64>,
        returns: numpy::PyReadonlyArray1<f32>,
    ) {
        let obs = obs.as_slice().unwrap();
        let masks = masks.as_slice().unwrap();
        let actions = actions.as_slice().unwrap();
        let returns = returns.as_slice().unwrap();
        let n = actions.len();
        let p = self.cached_priority;

        for i in 0..n {
            let idx = self.tree.add(p);
            let obs_start = idx * PER_OBS_DIM;
            let mask_start = idx * PER_NUM_CARDS;
            self.obs[obs_start..obs_start + PER_OBS_DIM]
                .copy_from_slice(&obs[i * PER_OBS_DIM..(i + 1) * PER_OBS_DIM]);
            self.masks[mask_start..mask_start + PER_NUM_CARDS]
                .copy_from_slice(&masks[i * PER_NUM_CARDS..(i + 1) * PER_NUM_CARDS]);
            self.actions[idx] = actions[i];
            self.returns[idx] = returns[i];
        }
        self.size = self.tree.n_entries;
    }

    /// Sample with priorities.
    /// Returns (obs, masks, actions, returns, weights, indices) as numpy arrays.
    fn sample<'py>(
        &self,
        py: Python<'py>,
        batch_size: usize,
        beta: f64,
    ) -> PyResult<(
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray2<f32>>,
        Bound<'py, PyArray1<i64>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<f32>>,
        Bound<'py, PyArray1<i64>>,
    )> {
        let total = self.tree.total();
        let segment = total / batch_size as f64;

        let mut indices = Vec::with_capacity(batch_size);
        let mut priorities = Vec::with_capacity(batch_size);
        let mut rng = rand::thread_rng();

        for i in 0..batch_size {
            let lo = segment * i as f64;
            let hi = segment * (i + 1) as f64;
            let s: f64 = lo + rng.gen::<f64>() * (hi - lo);
            let mut idx = self.tree.get(s);
            if idx >= self.size {
                idx = self.size - 1;
            }
            indices.push(idx);
            let p = self.tree.priority(idx);
            priorities.push(if p > 1e-8 { p } else { 1e-8 });
        }

        // IS weights
        let mut weights = Vec::with_capacity(batch_size);
        let mut max_weight: f32 = 0.0;
        let size_f = self.size as f64;
        for &p in &priorities {
            let prob = p / total;
            let w = ((size_f * prob).powf(-beta)) as f32;
            if w > max_weight {
                max_weight = w;
            }
            weights.push(w);
        }
        if max_weight > 0.0 {
            for w in weights.iter_mut() {
                *w /= max_weight;
            }
        }

        // Gather data
        let mut obs_data = vec![0.0f32; batch_size * PER_OBS_DIM];
        let mut mask_data = vec![0.0f32; batch_size * PER_NUM_CARDS];
        let mut act_data = Vec::with_capacity(batch_size);
        let mut ret_data = Vec::with_capacity(batch_size);
        let mut idx_data = Vec::with_capacity(batch_size);

        for (j, &idx) in indices.iter().enumerate() {
            let obs_src = idx * PER_OBS_DIM;
            let obs_dst = j * PER_OBS_DIM;
            obs_data[obs_dst..obs_dst + PER_OBS_DIM]
                .copy_from_slice(&self.obs[obs_src..obs_src + PER_OBS_DIM]);
            let mask_src = idx * PER_NUM_CARDS;
            let mask_dst = j * PER_NUM_CARDS;
            mask_data[mask_dst..mask_dst + PER_NUM_CARDS]
                .copy_from_slice(&self.masks[mask_src..mask_src + PER_NUM_CARDS]);
            act_data.push(self.actions[idx]);
            ret_data.push(self.returns[idx]);
            idx_data.push(idx as i64);
        }

        let obs = numpy::PyArray::from_vec_bound(py, obs_data)
            .reshape([batch_size, PER_OBS_DIM])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let masks = numpy::PyArray::from_vec_bound(py, mask_data)
            .reshape([batch_size, PER_NUM_CARDS])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(format!("{}", e)))?;
        let actions = PyArray1::from_slice_bound(py, &act_data);
        let returns = PyArray1::from_slice_bound(py, &ret_data);
        let weights_arr = PyArray1::from_slice_bound(py, &weights);
        let indices_arr = PyArray1::from_slice_bound(py, &idx_data);

        Ok((obs, masks, actions, returns, weights_arr, indices_arr))
    }

    /// Update priorities based on TD errors.
    fn update_priorities(
        &mut self,
        indices: numpy::PyReadonlyArray1<i64>,
        td_errors: numpy::PyReadonlyArray1<f32>,
    ) {
        let indices = indices.as_slice().unwrap();
        let td_errors = td_errors.as_slice().unwrap();
        let alpha = self.alpha;
        let mut max_p = self.max_priority;

        for i in 0..indices.len() {
            let p = (td_errors[i].abs() + 1e-6) as f64;
            if p > max_p {
                max_p = p;
            }
            self.tree.update(indices[i] as usize, p.powf(alpha));
        }

        if max_p > self.max_priority {
            self.max_priority = max_p;
            self.cached_priority = max_p.powf(alpha);
        }
    }
}

#[pymodule]
fn colver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<VecEnv>()?;
    m.add_class::<PrioritizedReplayBuffer>()?;
    Ok(())
}
