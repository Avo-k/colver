use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use std::collections::HashMap;
use std::time::Instant;

use colver_core::bid_eval;
use colver_core::bidding;
use colver_core::card;
use colver_core::card::Suit;
use colver_core::bid_net::BidNet;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::EnvTracking;
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{Contract, GameState, Phase};

const OBS_DIM: usize = 415;
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
fn make_observation(
    state: &GameState,
    played_by: &[u32; 4],
    play_order: &[u8],
    bid_history: &[(u8, u8)],
    dealer: u8,
) -> Vec<f32> {
    let mut obs = Vec::with_capacity(OBS_DIM);
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
        OBS_DIM,
        "obs v4 len = {}, expected {}",
        obs.len(),
        OBS_DIM
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
    // IS-DD search objects (lazily initialized)
    dede_searches: Option<[IsDdSearch; 4]>,
    dede_initialized: bool,
    // Per-player card tracking for obs v2
    played_by: [u32; 4],
    // Chronological play order (card indices) for timing features
    play_order: Vec<u8>,
    // Bid history: (seat, bid_action) pairs, cleared on reset
    bid_history: Vec<(u8, u8)>,
    // DMC Q-network (loaded lazily)
    dmc_net: Option<DmcNet>,
    // Bid Q-network (loaded lazily)
    bid_net: Option<BidNet>,
    // Belief net model path (shared across dede searches)
    belief_net_path: Option<String>,
    // Playgen world-sampler model (shared read-only across dede searches)
    playgen_model: Option<std::sync::Arc<colver_core::playgen::infer::PlaygenModel>>,
    // Credibility world-weighting for IS-DD (see IsDdConfig::cred_alpha).
    // 0.0 = off (default). Judge nets are applied to every dede search.
    dede_cred_alpha: f32,
    dede_cred_bid_net_path: Option<String>,
    dede_cred_play_net_path: Option<String>,
    // Fixed determinization count for IS-DD. 0 = time mode (default, budget =
    // action_dede's time_ms arg); >0 = count mode (solve exactly N worlds,
    // ignoring the time budget). Count mode gives a machine-independent number
    // of worlds at the cost of variable latency.
    dede_determinizations: u32,
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
            dede_searches: None,
            dede_initialized: false,
            played_by: [0; 4],
            play_order: Vec::with_capacity(32),
            bid_history: Vec::new(),
            dmc_net: None,
            bid_net: None,
            belief_net_path: None,
            playgen_model: None,
            dede_cred_alpha: 0.0,
            dede_cred_bid_net_path: None,
            dede_cred_play_net_path: None,
            dede_determinizations: 0,
        }
    }

    /// Reset the environment with a new random deal.
    fn reset(&mut self) -> PyResult<(Vec<f32>, Vec<u8>)> {
        let dealer = self.rng.gen_range(0..4u8);
        self.state = GameState::deal_random(dealer, &mut self.rng);
        self.smart_initialized = false;
        self.dede_initialized = false;
        self.played_by = [0; 4];
        self.play_order.clear();
        self.bid_history.clear();
        Ok((
            make_observation(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer),
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
            make_observation(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer),
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

    /// Legal play actions with equivalent cards collapsed to one representative.
    /// Two cards are equivalent when adjacent in their suit ordering with no
    /// outstanding card between them and identical point value — playing either
    /// is indistinguishable for the rest of the deal. Only valid in play phase.
    fn legal_actions_reduced(&self) -> Vec<u8> {
        if self.state.phase != Phase::Playing {
            return legal_actions_list(&self.state);
        }
        let reduced = colver_core::solver::reduce_equivalent(
            self.state.legal_actions() as u32,
            &self.state,
        );
        let mut actions = Vec::new();
        let mut remaining = reduced;
        while remaining != 0 {
            actions.push(remaining.trailing_zeros() as u8);
            remaining &= remaining - 1;
        }
        actions
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

    /// Get improved_v2_bid action (default bidding strategy).
    fn bid_improved(&self) -> u8 {
        bid_eval::improved_v2_bid(&self.state)
    }

    /// Alias for bid_improved (same as improved_v2).
    fn bid_improved_v2(&self) -> u8 {
        bid_eval::improved_v2_bid(&self.state)
    }

    /// Get the OLD improved_bid action (legacy, pre-v2).
    fn bid_improved_v1(&self) -> u8 {
        bid_eval::improved_bid(&self.state)
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

    /// Get maxi_bid action for current state (only valid during bidding phase).
    fn bid_maxi(&self) -> u8 {
        colver_core::maxi::maxi_bid(&self.state)
    }

    /// Get "Bid à DD" action: NN bidder if model loaded, else improved_v2 fallback.
    /// Only valid during bidding phase.
    ///
    /// Auto-dispatches on the NN's obs_dim:
    ///   108 → standard bid obs (v1/v2/v3/v4 models)
    ///   110 → score-aware v1 (my_score/opp_score raw), uses 0/0 for match-neutral
    ///   113 → score-aware v2 (my/opp/win_prob/leader_dist/diff), uses 0/0
    ///   117 → score-aware v3 (v6 default, +4 belote bits), uses 0/0
    fn bid_a_dd(&mut self) -> u8 {
        if self.state.phase != Phase::Bidding {
            return 0;
        }
        if let Some(ref mut net) = self.bid_net {
            let obs = build_bid_obs(net, &self.state, &self.bid_history);
            let legal = self.state.legal_actions();
            let (action, _) = net.best_action(&obs, legal);
            action
        } else {
            bid_eval::improved_v2_bid(&self.state)
        }
    }

    /// Get maxi play action (perfect-info convention-linked heuristic).
    /// Only valid during play phase.
    fn action_maxi_play(&self) -> PyResult<u8> {
        if self.state.phase != Phase::Playing {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Maxi play only valid during play phase",
            ));
        }
        Ok(colver_core::maxi::maxi_play_action(&self.state))
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
            make_observation(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer),
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

    /// Initial (pre-play) hands: current hand plus every card the seat has
    /// already played. After `from_cfn` this reconstructs the full deal.
    fn get_initial_hands(&self) -> Vec<Vec<u8>> {
        (0..4)
            .map(|s| card::CardIter(self.state.hands[s] | self.played_by[s]).collect())
            .collect()
    }

    /// Play actions (card indices) in the order they were played.
    fn get_play_order(&self) -> Vec<u8> {
        self.play_order.clone()
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
            dede_searches: None,
            dede_initialized: false,
            played_by: [0; 4],
            play_order: Vec::new(),
            bid_history: Vec::new(),
            dmc_net: None,
            bid_net: None,
            belief_net_path: None,
            playgen_model: None,
            dede_cred_alpha: 0.0,
            dede_cred_bid_net_path: None,
            dede_cred_play_net_path: None,
            dede_determinizations: 0,
        })
    }

    /// Re-deal this env with specific hands, keeping loaded models (DMC/bid/belief).
    /// Much cheaper than deal_with_hands + load_*_model when simulating many worlds.
    fn redeal_with_hands(&mut self, dealer: u8, hands: Vec<Vec<u8>>) -> PyResult<()> {
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
        self.state = GameState::new(dealer, hand_sets);
        self.smart_initialized = false;
        self.dede_initialized = false;
        self.played_by = [0; 4];
        self.play_order.clear();
        self.bid_history.clear();
        Ok(())
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

    /// Oracle MCTS: perfect-information MCTS with time budget.
    /// Sees all hands — strongest possible play but "cheats".
    fn action_oracle_mcts(&mut self, time_ms: u32) -> PyResult<u8> {
        use colver_core::mcts::{MctsConfig, MctsSearch, RolloutPolicy};
        if self.state.phase != Phase::Playing {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Oracle MCTS only valid during play phase",
            ));
        }
        // ~769K rollouts/sec with HeuristicPlay, so time_ms * 769 ≈ iterations
        let iterations = (time_ms as u32) * 700;
        let config = MctsConfig {
            iterations,
            rollout_policy: RolloutPolicy::HeuristicPlay,
            ..Default::default()
        };
        let mut search = MctsSearch::new();
        Ok(search.search(&self.state, &config, &mut self.rng))
    }

    /// Load a BeliefNet model for NN-based beliefs in IS-DD.
    /// Call once, then dede_init() will use it for all searches.
    fn load_belief_net(&mut self, path: &str) -> PyResult<()> {
        // Validate by loading once, then store path for dede_init to load per-search
        colver_core::belief_net::BeliefNet::load(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load belief net: {}", e))
        })?;
        self.belief_net_path = Some(path.to_string());
        // If searches already exist, load into them
        if let Some(ref mut searches) = self.dede_searches {
            for search in searches.iter_mut() {
                search.load_belief_net(path).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!("Failed to load belief net: {}", e))
                })?;
            }
        }
        Ok(())
    }

    /// Check if a BeliefNet model is loaded.
    fn has_belief_net(&self) -> bool {
        self.belief_net_path.is_some()
    }

    /// Load a playgen world-sampler model (COLVPG01) for IS-DD searches.
    /// Call before dede_init() so the sampler sees the full action prefix.
    fn load_playgen_model(&mut self, path: &str) -> PyResult<()> {
        let model = colver_core::playgen::infer::PlaygenModel::load(path).map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load playgen model: {}", e))
        })?;
        let model = std::sync::Arc::new(model);
        if let Some(ref mut searches) = self.dede_searches {
            for search in searches.iter_mut() {
                search.set_playgen_model(model.clone());
            }
        }
        self.playgen_model = Some(model);
        Ok(())
    }

    /// Check if a playgen model is loaded.
    fn has_playgen_model(&self) -> bool {
        self.playgen_model.is_some()
    }

    /// Set the IS-DD credibility exponent (`IsDdConfig::cred_alpha`). 0.0 = off
    /// (default). Requires at least one judge net loaded to have any effect;
    /// see `dede_load_cred_bid_net` / `dede_load_cred_play_net`.
    fn dede_set_cred_alpha(&mut self, alpha: f32) {
        self.dede_cred_alpha = alpha;
    }

    /// Set a fixed IS-DD determinization count. 0 = time mode (default: the
    /// `action_dede` time_ms arg bounds the search). >0 = count mode: solve
    /// exactly N determinized worlds regardless of time. Count mode gives a
    /// machine-independent number of worlds (reproducible strength) at the cost
    /// of variable latency — prefer time mode when latency must stay bounded.
    fn dede_set_determinizations(&mut self, n: u32) {
        self.dede_determinizations = n;
    }

    /// Load the bid net used as the IS-DD auction-credibility judge. Applied to
    /// every dede search (and remembered so a later `dede_init` re-applies it).
    fn dede_load_cred_bid_net(&mut self, path: &str) -> PyResult<()> {
        if let Some(ref mut searches) = self.dede_searches {
            for search in searches.iter_mut() {
                search.load_cred_bid_net(path).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to load cred bid net: {}",
                        e
                    ))
                })?;
            }
        }
        self.dede_cred_bid_net_path = Some(path.to_string());
        Ok(())
    }

    /// Load the canonical DMC net used as the IS-DD play-credibility judge.
    /// Applied to every dede search (and remembered for later `dede_init`).
    fn dede_load_cred_play_net(&mut self, path: &str) -> PyResult<()> {
        if let Some(ref mut searches) = self.dede_searches {
            for search in searches.iter_mut() {
                search.load_cred_play_net(path).map_err(|e| {
                    pyo3::exceptions::PyIOError::new_err(format!(
                        "Failed to load cred play net: {}",
                        e
                    ))
                })?;
            }
        }
        self.dede_cred_play_net_path = Some(path.to_string());
        Ok(())
    }

    /// Monte-Carlo card-location marginals from the playgen world sampler,
    /// from `observer`'s perspective. Samples up to `n_worlds` determinized
    /// worlds and counts where each card lands. Returns weights[player][card]
    /// (4×32), or None during bidding / if sampling fails.
    #[pyo3(signature = (observer, n_worlds=50, temperature=1.0))]
    fn get_playgen_beliefs(
        &mut self,
        py: Python<'_>,
        observer: u8,
        n_worlds: usize,
        temperature: f32,
    ) -> PyResult<Option<Vec<Vec<f32>>>> {
        if !self.dede_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call dede_init() first",
            ));
        }
        if observer >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "observer must be 0-3",
            ));
        }
        // MC sampling takes ~1s: release the GIL so the web server's event
        // loop (and parallel precompute workers) keep running.
        let state = self.state;
        let searches = self.dede_searches.as_mut().unwrap();
        let search = &mut searches[observer as usize];
        let rng = &mut self.rng;
        let marginals = py.allow_threads(move || {
            search.playgen_marginals(&state, n_worlds, temperature, rng)
        });
        Ok(marginals.map(|w| w.iter().map(|row| row.to_vec()).collect()))
    }

    /// Playgen bid-policy probabilities at the current auction point
    /// (v2 playgen models only). Returns 43 masked-softmax probabilities
    /// (0.0 for illegal bids), or None if unavailable.
    fn get_playgen_bid_policy(
        &mut self,
        observer: u8,
        temperature: f32,
    ) -> PyResult<Option<Vec<f32>>> {
        if !self.dede_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call dede_init() first",
            ));
        }
        if observer >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "observer must be 0-3",
            ));
        }
        let state = self.state;
        let searches = self.dede_searches.as_mut().unwrap();
        let search = &mut searches[observer as usize];
        let Some(logits) = search.playgen_bid_policy(&state) else {
            return Ok(None);
        };
        let mask = state.legal_actions();
        let t = temperature.max(1e-3);
        let mut max_l = f32::NEG_INFINITY;
        for c in 0..43 {
            if mask & (1u64 << c) != 0 && logits[c] > max_l {
                max_l = logits[c];
            }
        }
        let mut probs = vec![0.0f32; 43];
        let mut total = 0.0f32;
        for c in 0..43 {
            if mask & (1u64 << c) != 0 {
                let p = ((logits[c] - max_l) / t).exp();
                probs[c] = p;
                total += p;
            }
        }
        if total > 0.0 {
            for p in probs.iter_mut() {
                *p /= total;
            }
        }
        Ok(Some(probs))
    }

    /// Sample full deals from the current mid-auction position via the
    /// playgen model (v2 only): the auction is completed with the bid head,
    /// then the deal is played out to reveal the hidden hands.
    /// Returns up to `n_worlds` deals as 4 lists of 8 card ids, or None.
    fn playgen_sample_auction_deals(
        &mut self,
        py: Python<'_>,
        observer: u8,
        n_worlds: usize,
        temperature: f32,
    ) -> PyResult<Option<Vec<Vec<Vec<u8>>>>> {
        if !self.dede_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call dede_init() first",
            ));
        }
        if observer >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "observer must be 0-3",
            ));
        }
        let state = self.state;
        let searches = self.dede_searches.as_mut().unwrap();
        let search = &mut searches[observer as usize];
        let rng = &mut self.rng;
        // Slow (~0.4s/deal): release the GIL for the web event loop.
        let worlds = py.allow_threads(move || {
            search.playgen_auction_deals(&state, n_worlds, temperature, rng)
        });
        if worlds.is_empty() {
            return Ok(None);
        }
        let out = worlds
            .iter()
            .map(|hands| {
                hands
                    .iter()
                    .map(|&h| {
                        let mut cards: Vec<u8> =
                            (0..32u8).filter(|&c| h & (1 << c) != 0).collect();
                        cards.sort_unstable();
                        cards
                    })
                    .collect()
            })
            .collect();
        Ok(Some(out))
    }

    /// Sample full deals from the current mid-auction position using the bid
    /// belief net (COLVBB, obs 108): NN marginals conditioned on the auction
    /// drive a weighted determinization. Hands during bidding are complete, so
    /// each draw is a full deal. Returns up to `n_worlds` deals as 4 lists of
    /// 8 card ids, or None if sampling failed.
    fn bid_belief_sample_deals(
        &mut self,
        py: Python<'_>,
        observer: u8,
        n_worlds: usize,
        model_path: &str,
    ) -> PyResult<Option<Vec<Vec<Vec<u8>>>>> {
        if observer >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "observer must be 0-3",
            ));
        }
        if self.state.phase != Phase::Bidding {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "state must be in bidding phase",
            ));
        }
        // Bid belief nets (COLVBB) use hidden=256; fall back from the default.
        let mut net = colver_core::belief_net::BeliefNet::load(model_path)
            .or_else(|_| colver_core::belief_net::BeliefNet::load_with_hidden(model_path, 256))
            .map_err(|e| {
                pyo3::exceptions::PyIOError::new_err(format!("Failed to load belief net: {}", e))
            })?;

        let state = self.state;
        let bid_history = self.bid_history.clone();
        let rng = &mut self.rng;
        let worlds: Vec<[u32; 4]> = py.allow_threads(move || {
            use colver_core::belief_state::BeliefState;
            let mut bs = BeliefState::new(observer, state.hands[observer as usize]);
            // Replay the auction so heuristic constraints accumulate.
            let mut s = GameState::new(state.dealer, state.hands);
            for &(p, a) in &bid_history {
                bs.record_bid(p, a, &s);
                s.step(a);
            }
            bs.apply_nn_bid_beliefs(&mut net, &state, &bid_history);
            (0..n_worlds)
                .filter_map(|_| bs.determinize(&state, rng).map(|d| d.hands))
                .collect()
        });
        if worlds.is_empty() {
            return Ok(None);
        }
        let out = worlds
            .iter()
            .map(|hands| {
                hands
                    .iter()
                    .map(|&h| {
                        let mut cards: Vec<u8> =
                            (0..32u8).filter(|&c| h & (1 << c) != 0).collect();
                        cards.sort_unstable();
                        cards
                    })
                    .collect()
            })
            .collect();
        Ok(Some(out))
    }

    /// Initialize IS-DD (Dédé) beliefs for a new deal.
    /// Must be called after reset() and before action_dede().
    fn dede_init(&mut self) {
        let belief_path = self.belief_net_path.clone();
        let playgen_model = self.playgen_model.clone();
        let cred_bid_path = self.dede_cred_bid_net_path.clone();
        let cred_play_path = self.dede_cred_play_net_path.clone();
        let searches = self.dede_searches.get_or_insert_with(|| {
            let mut s = [
                IsDdSearch::new(),
                IsDdSearch::new(),
                IsDdSearch::new(),
                IsDdSearch::new(),
            ];
            if let Some(ref path) = belief_path {
                for search in s.iter_mut() {
                    let _ = search.load_belief_net(path);
                }
            }
            if let Some(ref model) = playgen_model {
                for search in s.iter_mut() {
                    search.set_playgen_model(model.clone());
                }
            }
            if let Some(ref path) = cred_bid_path {
                for search in s.iter_mut() {
                    let _ = search.load_cred_bid_net(path);
                }
            }
            if let Some(ref path) = cred_play_path {
                for search in s.iter_mut() {
                    let _ = search.load_cred_play_net(path);
                }
            }
            s
        });
        for (player, search) in searches.iter_mut().enumerate() {
            search.init_deal(&self.state, player as u8, true);
        }
        self.dede_initialized = true;
    }

    /// Record an action for IS-DD beliefs, then step the game.
    /// Returns (observation, reward, done, legal_actions) like step().
    fn dede_step(&mut self, action: u8) -> PyResult<(Vec<f32>, f32, bool, Vec<u8>)> {
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
        if let Some(ref mut searches) = self.dede_searches {
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
            make_observation(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer),
            reward,
            done,
            legal_actions_list(&self.state),
        ))
    }

    /// Provide externally sampled worlds (remaining hands per seat, current
    /// position — e.g. from the GPU playgen sidecar) to the current player's
    /// IS-DD search. Consumed first by the next action_dede*() call; invalid
    /// worlds are skipped. Each world is a list of 4 u32 CardSet bitmasks.
    fn dede_inject_worlds(&mut self, worlds: Vec<Vec<u32>>) -> PyResult<()> {
        if !self.dede_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call dede_init() first",
            ));
        }
        let converted: Vec<[u32; 4]> = worlds
            .into_iter()
            .filter_map(|w| w.try_into().ok())
            .collect();
        let player = self.state.current_player() as usize;
        let searches = self.dede_searches.as_mut().unwrap();
        searches[player].set_injected_worlds(converted);
        Ok(())
    }

    /// Get IS-DD (Dédé) action for current state. time_ms is the search budget in ms.
    /// Must call dede_init() first and use dede_step() for all moves.
    fn action_dede(&mut self, time_ms: u32) -> PyResult<u8> {
        if !self.dede_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call dede_init() first",
            ));
        }
        let player = self.state.current_player() as usize;
        // Count mode (dede_determinizations > 0) solves a fixed number of worlds
        // and ignores the time budget; otherwise time mode uses `time_ms`.
        let count_mode = self.dede_determinizations > 0;
        let config = IsDdConfig {
            determinizations: if count_mode { self.dede_determinizations } else { 20 },
            time_limit_ms: if count_mode { None } else { Some(time_ms) },
            use_nn_beliefs: self.belief_net_path.is_some(),
            // Solve the determinized worlds across the rayon global pool (shared
            // and bounded, so concurrent games/rooms don't oversubscribe).
            parallel: true,
            cred_alpha: self.dede_cred_alpha,
            ..Default::default()
        };
        let searches = self.dede_searches.as_mut().unwrap();
        let action = searches[player].search(&self.state, &config, &mut self.rng);
        Ok(action)
    }

    /// Get IS-DD (Dédé) action with search statistics.
    /// Returns dict: {best_action, card_scores: [[card, avg_score]...], determinizations, elapsed_ms}
    /// During bidding, returns bid_improved_v2() with minimal stats.
    fn action_dede_with_stats<'py>(&mut self, py: Python<'py>, time_ms: u32) -> PyResult<Bound<'py, PyDict>> {
        let dict = PyDict::new_bound(py);

        if self.state.phase == Phase::Bidding {
            let action = bid_eval::improved_v2_bid(&self.state);
            dict.set_item("best_action", action)?;
            dict.set_item("card_scores", Vec::<(u8, f32)>::new())?;
            dict.set_item("determinizations", 0u32)?;
            dict.set_item("elapsed_ms", 0.0f64)?;
            return Ok(dict);
        }

        if !self.dede_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call dede_init() first",
            ));
        }
        let player = self.state.current_player() as usize;
        // Count mode (dede_determinizations > 0) solves a fixed number of worlds
        // and ignores the time budget; otherwise time mode uses `time_ms`.
        let count_mode = self.dede_determinizations > 0;
        let config = IsDdConfig {
            determinizations: if count_mode { self.dede_determinizations } else { 20 },
            time_limit_ms: if count_mode { None } else { Some(time_ms) },
            use_nn_beliefs: self.belief_net_path.is_some(),
            // Solve the determinized worlds across the rayon global pool (shared
            // and bounded, so concurrent games/rooms don't oversubscribe).
            parallel: true,
            cred_alpha: self.dede_cred_alpha,
            ..Default::default()
        };
        let searches = self.dede_searches.as_mut().unwrap();
        let start = Instant::now();
        let result = searches[player].search_with_stats(&self.state, &config, &mut self.rng);
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        dict.set_item("best_action", result.best_action)?;
        dict.set_item("card_scores", result.card_scores)?;
        dict.set_item("determinizations", result.determinizations)?;
        dict.set_item("elapsed_ms", elapsed)?;
        Ok(dict)
    }

    /// Get belief weights from IS-DD for a given observer.
    /// Returns dict: {nn: [[f32; 32]; 4] | None, heuristic: [[f32; 32]; 4] | None}
    /// Each is weights[player][card] probability distribution.
    fn get_belief_weights<'py>(&mut self, py: Python<'py>, observer: u8) -> PyResult<Bound<'py, PyDict>> {
        if !self.dede_initialized {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Call dede_init() first",
            ));
        }
        if observer >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "observer must be 0-3",
            ));
        }
        let searches = self.dede_searches.as_mut().unwrap();
        let (nn, heuristic) = searches[observer as usize].get_belief_weights(&self.state, observer);

        let dict = PyDict::new_bound(py);
        match nn {
            Some(w) => {
                let nn_list: Vec<Vec<f32>> = w.iter().map(|row| row.to_vec()).collect();
                dict.set_item("nn", nn_list)?;
            }
            None => {
                dict.set_item("nn", py.None())?;
            }
        }
        match heuristic {
            Some(w) => {
                let h_list: Vec<Vec<f32>> = w.iter().map(|row| row.to_vec()).collect();
                dict.set_item("heuristic", h_list)?;
            }
            None => {
                dict.set_item("heuristic", py.None())?;
            }
        }
        Ok(dict)
    }

    /// Oracle DD: exact double-dummy solver. Returns optimal card for current player.
    /// Only valid during play phase. No time budget needed (~7ms median).
    fn action_oracle_dd(&self) -> PyResult<u8> {
        if self.state.phase != Phase::Playing {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Oracle DD only valid during play phase",
            ));
        }
        Ok(colver_core::solver::solve_best_card(&self.state))
    }

    /// DD scores for every legal root move of the current player.
    /// Returns dict: {scores: [[card, ns_points], ...], best_card: u8}.
    /// Only valid during play phase on a non-terminal state.
    fn solve_scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.state.phase != Phase::Playing || self.state.is_terminal() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "solve_scores only valid during play phase",
            ));
        }
        let result = colver_core::solver::solve_with_scores(&self.state, None);
        let scores: Vec<Vec<i32>> = result.scores[..result.count]
            .iter()
            .map(|&(card, ns)| vec![card as i32, ns as i32])
            .collect();
        let dict = PyDict::new_bound(py);
        dict.set_item("scores", scores)?;
        dict.set_item("best_card", result.best_card)?;
        Ok(dict)
    }

    /// Solve all 4 trump suits with the DD solver.
    /// Returns dict: {suits: [[ns_pts, ew_pts], ...4], elapsed_ms: f64}
    fn solve_all_suits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let start = Instant::now();
        let hands = self.state.hands;
        let dealer = self.state.dealer;
        // Release the GIL during the solve (~300ms) so callers can run several
        // solves in parallel from a Python thread pool.
        let suits = py.allow_threads(move || {
            let mut tt_buf = colver_core::solver::new_tt_buffer();
            let mut suits = Vec::with_capacity(4);
            for suit in 0..4u8 {
                let [ns, ew] = colver_core::solver::solve_for_trump_reuse_tt(hands, dealer, suit, &mut tt_buf);
                suits.push(vec![ns, ew]);
            }
            suits
        });
        let elapsed = start.elapsed().as_secs_f64() * 1000.0;
        let dict = PyDict::new_bound(py);
        dict.set_item("suits", suits)?;
        dict.set_item("elapsed_ms", elapsed)?;
        Ok(dict)
    }

    /// Get observation (415 floats) for current state.
    fn get_observation(&self) -> Vec<f32> {
        make_observation(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer)
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
        // Canonical (411-dim) models use residual skip connections
        if let Some(ref mut n) = self.dmc_net {
            if n.obs_dim() == colver_core::dmc_obs::OBS_DIM_TR {
                n.set_residual(true);
            }
        }
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

        let start = Instant::now();

        let (best_action, q_values) = if net.obs_dim() == colver_core::dmc_obs::OBS_DIM_TR {
            // Canonical obs (411-dim): build EnvTracking, use canonical mask/action
            let tracking = EnvTracking {
                played_by: self.played_by,
                play_order: self.play_order.clone(),
                bid_history: self.bid_history.clone(),
                dealer: self.state.dealer,
            };
            let obs = colver_core::dmc_obs::make_observation_tr(&self.state, &tracking);
            let order = colver_core::dmc_obs::current_player_order(&self.state, &tracking);
            let canonical_mask = colver_core::dmc_obs::cardset_to_canonical(self.state.legal_actions() as u32, &order);
            let (canonical_best, q_vals) = net.best_action(&obs, canonical_mask);
            let physical_action = colver_core::dmc_obs::card_to_physical(canonical_best, &order);
            (physical_action, q_vals)
        } else {
            // Legacy obs (415-dim): use existing make_observation
            let obs_full = make_observation(&self.state, &self.played_by, &self.play_order, &self.bid_history, self.state.dealer);
            let obs = &obs_full[..net.obs_dim()];
            let legal_mask = self.state.legal_actions() as u32;
            net.best_action(obs, legal_mask)
        };

        let elapsed = start.elapsed().as_secs_f64() * 1000.0;

        let dict = PyDict::new_bound(py);
        dict.set_item("best_action", best_action)?;
        dict.set_item("q_values", q_values)?;
        dict.set_item("elapsed_ms", elapsed)?;
        Ok(dict)
    }

    /// Load NN bid model weights from a raw binary file.
    /// Call once, then use action_bid_nn() for inference.
    #[pyo3(signature = (path, hidden=None))]
    fn load_bid_model(&mut self, path: &str, hidden: Option<usize>) -> PyResult<()> {
        let net = if let Some(h) = hidden {
            BidNet::load_with_hidden(path, h)
        } else {
            BidNet::load(path)  // auto-detects hidden size (tries 256, 512, 1024)
        }.map_err(|e| {
            pyo3::exceptions::PyIOError::new_err(format!("Failed to load bid model: {}", e))
        })?;
        self.bid_net = Some(net);
        Ok(())
    }

    /// Check if bid model is loaded.
    fn has_bid_model(&self) -> bool {
        self.bid_net.is_some()
    }

    /// Get NN bid action with Q-value statistics.
    /// Returns dict: {best_action, q_values: [(action, q)...]}
    /// Only valid during bidding phase. Requires load_bid_model() first.
    fn action_bid_nn<'py>(&mut self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.state.phase != Phase::Bidding {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Bid NN only valid during bidding phase",
            ));
        }
        let net = self.bid_net.as_mut().ok_or_else(|| {
            pyo3::exceptions::PyRuntimeError::new_err("Call load_bid_model() first")
        })?;

        let obs = build_bid_obs(net, &self.state, &self.bid_history);
        let legal = self.state.legal_actions();
        let (best_action, q_values) = net.best_action(&obs, legal);

        let dict = PyDict::new_bound(py);
        dict.set_item("best_action", best_action)?;
        dict.set_item("q_values", q_values)?;
        Ok(dict)
    }

    /// Get the bidding observation vector (108/110/113-dim depending on obs).
    /// Without NN loaded, returns the base 108-dim obs for feature inspection.
    fn get_bid_observation(&self) -> Vec<f32> {
        colver_core::bid_obs::make_bid_observation(&self.state, &self.bid_history)
    }

    /// Get the 43-float legal bid action mask.
    fn get_bid_mask(&self) -> Vec<f32> {
        let mut mask = vec![0.0f32; 43];
        colver_core::bid_obs::write_bid_mask(&mut mask, 0, &self.state);
        mask
    }

    /// Convert game state to CFN (Contrée FEN Notation) string.
    fn to_cfn(&self) -> String {
        self.state.to_cfn()
    }

    /// Create an Env from a CFN string.
    #[staticmethod]
    fn from_cfn(cfn: &str) -> PyResult<Self> {
        let state = GameState::from_cfn(cfn).map_err(|e| {
            pyo3::exceptions::PyValueError::new_err(format!("{}", e))
        })?;

        // Reconstruct play_order and played_by from trick_history
        let mut played_by = [0u32; 4];
        let mut play_order = Vec::with_capacity(32);
        let bid_history = Vec::new();

        let completed = (state.tricks_won[0] + state.tricks_won[1]) as usize;
        let first_lead = (state.dealer + 1) % 4;
        let mut current_lead = first_lead;

        for t in 0..completed {
            let trick = state.trick_history[t];
            for i in 0..4u8 {
                let seat = (current_lead + i) % 4;
                let card = trick[seat as usize];
                played_by[seat as usize] |= 1u32 << card;
                play_order.push(card);
            }
            let winner = colver_core::trick::trick_winner(&trick, current_lead, &state.contract);
            current_lead = winner;
        }

        // Current partial trick
        if state.phase == Phase::Playing && state.trick_count > 0 {
            for i in 0..state.trick_count {
                let seat = (state.trick_lead + i) % 4;
                let card = state.current_trick[seat as usize];
                played_by[seat as usize] |= 1u32 << card;
                play_order.push(card);
            }
        }

        Ok(Env {
            state,
            rng: StdRng::from_entropy(),
            naive_search: None,
            smart_searches: None,
            smart_initialized: false,
            dede_searches: None,
            dede_initialized: false,
            played_by,
            play_order,
            bid_history,
            dmc_net: None,
            bid_net: None,
            belief_net_path: None,
            playgen_model: None,
            dede_cred_alpha: 0.0,
            dede_cred_bid_net_path: None,
            dede_cred_play_net_path: None,
            dede_determinizations: 0,
        })
    }

    fn __repr__(&self) -> String {
        format!("{:?}", self.state)
    }
}

/// Build a bid observation matching the NN's obs_dim.
///
/// - 108: base obs (v1/v2/v3/v4)
/// - 110: score-aware v1 (my_score, opp_score raw) — uses 0/0
/// - 113: score-aware v2 (v5) — uses 0/0
/// - 117: score-aware v3 (v6 default, +4 belote bits) — uses 0/0
///
/// At inference time on a single deal (no multi-deal match context on the
/// web), match scores default to 0/0 which matches the NN's neutral-match
/// behaviour (what the distillation used).
fn build_bid_obs(
    net: &BidNet,
    state: &colver_core::state::GameState,
    history: &[(u8, u8)],
) -> Vec<f32> {
    use colver_core::bid_obs;
    let obs_dim = net.obs_dim();
    match obs_dim {
        bid_obs::BID_OBS_DIM => bid_obs::make_bid_observation(state, history),
        bid_obs::BID_OBS_DIM_SCORE_AWARE => {
            let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE];
            bid_obs::write_bid_observation_score_aware(&mut buf, 0, state, history, 0, 0);
            buf
        }
        bid_obs::BID_OBS_DIM_SCORE_AWARE_V2 => {
            let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE_V2];
            bid_obs::write_bid_observation_score_aware_v2(&mut buf, 0, state, history, 0, 0);
            buf
        }
        bid_obs::BID_OBS_DIM_SCORE_AWARE_V3 => {
            let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE_V3];
            bid_obs::write_bid_observation_score_aware_v3(&mut buf, 0, state, history, 0, 0);
            buf
        }
        other => panic!(
            "Unsupported bid NN obs_dim={} (expected 108/110/113/117)",
            other
        ),
    }
}

#[pymodule]
fn _colver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    Ok(())
}
