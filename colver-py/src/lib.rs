use pyo3::prelude::*;
use numpy::{PyArray1, PyArray2, PyArrayMethods};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_eval;
use colver_core::card;
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};

fn make_observation(state: &GameState) -> Vec<f32> {
    let mut obs = Vec::with_capacity(222);
    let player = state.current_player() as usize;

    // Player's hand: 32 floats
    for i in 0..32 {
        obs.push(if state.hands[player] & (1u32 << i) != 0 { 1.0 } else { 0.0 });
    }

    // Cards played in current trick: 4 × 32 floats
    for seat in 0..4u8 {
        let c = state.current_trick[seat as usize];
        for i in 0..32u8 {
            obs.push(if c != card::EMPTY && c == i { 1.0 } else { 0.0 });
        }
    }

    // Played cards: 32 floats
    for i in 0..32 {
        obs.push(if state.played_cards & (1u32 << i) != 0 { 1.0 } else { 0.0 });
    }

    // Contract info: trump(4), value(1 normalized), coinche(3), taker_team(2)
    for t in 0..4u8 {
        obs.push(if state.contract.trump == t { 1.0 } else { 0.0 });
    }
    obs.push(state.contract.value as f32 / 25.0);
    for c in 0..3u8 {
        obs.push(if state.contract.coinche == c { 1.0 } else { 0.0 });
    }
    for t in 0..2u8 {
        obs.push(if state.contract.team == t { 1.0 } else { 0.0 });
    }

    // Points per team: 2 normalized
    obs.push(state.points[0] as f32 / 162.0);
    obs.push(state.points[1] as f32 / 162.0);

    // Tricks won per team
    obs.push(state.tricks_won[0] as f32 / 8.0);
    obs.push(state.tricks_won[1] as f32 / 8.0);

    // Phase: 3 floats one-hot
    for p in 0..3u8 {
        obs.push(if state.phase as u8 == p { 1.0 } else { 0.0 });
    }

    // Current player position relative to dealer
    let rel_pos = (state.current_player + 4 - state.dealer) % 4;
    for p in 0..4u8 {
        obs.push(if rel_pos == p { 1.0 } else { 0.0 });
    }

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
        }
    }

    /// Reset the environment with a new random deal.
    fn reset(&mut self) -> PyResult<(Vec<f32>, Vec<u8>)> {
        let dealer = self.rng.gen_range(0..4u8);
        self.state = GameState::deal_random(dealer, &mut self.rng);
        self.smart_initialized = false;
        Ok((make_observation(&self.state), legal_actions_list(&self.state)))
    }

    /// Take an action. Returns (observation, reward, done, legal_actions).
    fn step(&mut self, action: u8) -> PyResult<(Vec<f32>, f32, bool, Vec<u8>)> {
        let player = self.state.current_player();
        let team = GameState::player_team(player) as usize;

        self.state.step(action);

        let done = self.state.is_terminal();
        let reward = if done {
            self.state.rewards()[team]
        } else {
            0.0
        };

        Ok((make_observation(&self.state), reward, done, legal_actions_list(&self.state)))
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

        Ok((make_observation(&self.state), reward, done, legal_actions_list(&self.state)))
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

    fn __repr__(&self) -> String {
        format!("{:?}", self.state)
    }
}

/// Vectorized environment for batch RL training.
#[pyclass]
struct VecEnv {
    states: Vec<GameState>,
    rng: StdRng,
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
        VecEnv { states, rng }
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
        }

        let obs_size = make_observation(&self.states[0]).len();
        let mut obs_data = Vec::with_capacity(n * obs_size);
        let mut mask_data = Vec::with_capacity(n * 43);

        for state in &self.states {
            obs_data.extend(make_observation(state));
            mask_data.extend(legal_mask_vec(state));
        }

        let obs = numpy::PyArray::from_vec_bound(py, obs_data)
            .reshape([n, obs_size])
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
            } else {
                outcomes_vec.push(0.0);
                outcomes_vec.push(0.0);
            }
        }

        let obs_size = make_observation(&self.states[0]).len();
        let mut obs_data = Vec::with_capacity(n * obs_size);
        let mut mask_data = Vec::with_capacity(n * 43);

        for state in &self.states {
            obs_data.extend(make_observation(state));
            mask_data.extend(legal_mask_vec(state));
        }

        let obs = numpy::PyArray::from_vec_bound(py, obs_data)
            .reshape([n, obs_size])
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
