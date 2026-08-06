use pyo3::prelude::*;
use pyo3::types::PyDict;
use numpy::PyArray1;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use colver_core::bid_eval;
use colver_core::bidding;
use colver_core::card;
use colver_core::card::Suit;
use colver_core::hand_class;
use colver_core::bid_net::BidNet;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::EnvTracking;
use colver_core::agent::{AgentSpec, MatchContext, Player};
use colver_core::is_dd::IsDdSearch;
use colver_core::playgen::analysis::PlaygenAnalyst;
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{Contract, GameState, Phase};

/// Les chemins entrent en `PathBuf` — PyO3 accepte alors `str` comme tout
/// `os.PathLike`, donc un `pathlib.Path` (ce que rendent `colver.*_path()` et
/// `download_*`) passe sans `str()`. Le cœur, lui, prend des `&str`, d'où cette
/// conversion à la frontière. L'erreur est un garde-fou : un chemin venu de
/// Python est déjà de l'UTF-8 valide, on ne veut simplement pas d'un `unwrap`.
fn path_str(p: &Path) -> PyResult<&str> {
    p.to_str().ok_or_else(|| {
        pyo3::exceptions::PyValueError::new_err(format!("chemin non-UTF-8 : {}", p.display()))
    })
}

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
    // Cumulative match score [NS, EW] that score-aware bid nets condition on.
    // Survives `reset` / `redeal_with_hands`: it belongs to the match, not the deal.
    match_scores: [i32; 2],
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
            bid_net: None,
            match_scores: [0, 0],
        }
    }

    /// Cumulative match score `[NS, EW]` fed to score-aware bid nets (obs 110/113/117).
    ///
    /// Without this, `action_bid_nn` always observed 0-0 — so every measurement made
    /// through `Env` described v6 at the start of a match only, and the score half of
    /// its observation was untestable from Python. `Agent.set_scores` is the equivalent
    /// on the production path; this one keeps the Q-values, which `Agent.decide` does
    /// not return for bidding.
    fn set_match_scores(&mut self, ns: i32, ew: i32) {
        self.match_scores = [ns, ew];
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
            let obs = build_bid_obs(net, &self.state, &self.bid_history, self.match_scores);
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

    /// Belote **finale** par camp (0 ou 20), cartes pas encore jouées comprises.
    ///
    /// [`get_belote`] compte ce qui a déjà été posé et sous-estime donc en cours
    /// de donne. Celle-ci regarde `hands | played_by` : la belote est acquise dès
    /// qu'un joueur détient Dame **et** Roi d'atout, puisqu'il finira par jouer
    /// les deux. C'est la version qui décide de la réussite du contrat — elle
    /// **déplace le seuil** au lieu d'ajouter 20 points au bout.
    fn belote_final(&self) -> [i16; 2] {
        if self.state.phase == Phase::Bidding {
            return [0, 0];
        }
        colver_core::scoring::final_belote(
            &self.state.hands,
            &self.played_by,
            self.state.contract.trump,
        )
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

    /// Deal 8 cards to each seat and return a fresh env, ready to bid.
    ///
    /// `seed` makes the deal reproducible — and seeds the env's own RNG, so every
    /// later random draw on it (`reset`, the rollouts, the IS-MCTS searches) follows
    /// from it too. `dealer` defaults to a random seat; seat `(dealer + 1) % 4` speaks
    /// first.
    ///
    /// `Env()` + `reset()` already deals at random, but it hands back an observation
    /// vector and keeps the cards to itself, so every caller who wanted *a deal* was
    /// shuffling `range(32)` by hand and going through `deal_with_hands`.
    #[staticmethod]
    #[pyo3(signature = (dealer=None, seed=None))]
    fn deal(dealer: Option<u8>, seed: Option<u64>) -> PyResult<Self> {
        let mut rng = match seed {
            Some(s) => StdRng::seed_from_u64(s),
            None => StdRng::from_entropy(),
        };
        let dealer = match dealer {
            Some(d) if d >= 4 => {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "Invalid dealer: {} (expected 0-3)",
                    d
                )))
            }
            Some(d) => d,
            None => rng.gen_range(0..4u8),
        };
        let state = GameState::deal_random(dealer, &mut rng);
        Ok(Env {
            state,
            rng,
            naive_search: None,
            smart_searches: None,
            smart_initialized: false,
            played_by: [0; 4],
            play_order: Vec::with_capacity(32),
            bid_history: Vec::new(),
            dmc_net: None,
            bid_net: None,
            match_scores: [0, 0],
        })
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
            bid_net: None,
            match_scores: [0, 0],
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


    fn bid_belief_sample_deals(
        &mut self,
        py: Python<'_>,
        observer: u8,
        n_worlds: usize,
        model_path: PathBuf,
    ) -> PyResult<Option<Vec<Vec<Vec<u8>>>>> {
        let model_path = path_str(&model_path)?;
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

    fn action_oracle_dd(&self, py: Python<'_>) -> PyResult<u8> {
        if self.state.phase != Phase::Playing {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "Oracle DD only valid during play phase",
            ));
        }
        // Release the GIL. Three callers offload this to a thread (`card_analysis.opinions`,
        // `agent_review`, `game_manager`) on the documented assumption that the Rust solver
        // lets go — it did not, so the event loop was blocked for one whole DD search per
        // card. Invisible while review positions are mid/endgame (190 µs / 1.5 µs); a 35 ms
        // stall at the opening lead.
        let state = self.state;
        Ok(py.allow_threads(move || colver_core::solver::solve_best_card(&state)))
    }

    /// DD scores for every legal root move of the current player.
    ///
    /// Returns dict: `{scores, deal_scores, best_card}`.
    ///
    /// - `scores` — `[[card, ns_card_points], …]`, l'échelle historique : les
    ///   points **cartes** N-S en fin de donne, 0-252.
    /// - `deal_scores` — `[[card, ns_minus_ew], …]`, le même solve passé au
    ///   barème : l'écart de score **marqué**, contrat compris. C'est en
    ///   escalier, pas linéaire — plat sous le seuil du contrat, marche de `4V`
    ///   au seuil — donc **les deux échelles ne se soustraient pas** et un écart
    ///   en points cartes ne dit pas ce qu'un coup a coûté. Même conversion que
    ///   `is_dd::world_value` sous `PlayObjective::DealScore`.
    /// - `contract_made` — `[[card, bool], …]`, le contrat est-il tenu dans
    ///   cette branche. Ne se déduit **pas** du signe de `deal_scores` (un écart
    ///   négatif peut être une chute du preneur N-S ou un contrat tenu par E-O),
    ///   et c'est le seul prédicat qui sépare « ce coup a coûté des points » de
    ///   « ce coup a renversé la donne ».
    /// - `best_card` — inchangé. La conversion étant monotone non décroissante,
    ///   la carte qui maximise les points cartes maximise aussi le score de
    ///   donne ; en revanche **la classe des cartes optimales s'élargit** dans la
    ///   seconde échelle, et un appelant qui veut l'afficher doit la relire
    ///   depuis `deal_scores` plutôt que de désigner ce représentant.
    ///
    /// Only valid during play phase on a non-terminal state.
    fn solve_scores<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        if self.state.phase != Phase::Playing || self.state.is_terminal() {
            return Err(pyo3::exceptions::PyRuntimeError::new_err(
                "solve_scores only valid during play phase",
            ));
        }
        // Release the GIL for the search, exactly like `solve_all_suits` below.
        // `card_analysis.py` fans this call out across a thread pool on the stated assumption
        // that "the Rust solver releases the GIL" — for this entry point it did not, so that
        // fan-out was serialised and one page load paid 200-500 solves back to back.
        let state = self.state;
        let result = py.allow_threads(move || colver_core::solver::solve_with_scores(&state, None));
        let scores: Vec<Vec<i32>> = result.scores[..result.count]
            .iter()
            .map(|&(card, ns)| vec![card as i32, ns as i32])
            .collect();
        // Le barème appliqué au même solve : aucune recherche supplémentaire,
        // juste de l'arithmétique par carte.
        let played_by = self.played_by;
        let deal_scores: Vec<Vec<i32>> = result.scores[..result.count]
            .iter()
            .map(|&(card, ns)| {
                let delta = colver_core::scoring::deal_score_delta(&state, &played_by, ns);
                vec![card as i32, delta as i32]
            })
            .collect();
        let contract_made: Vec<(i32, bool)> = result.scores[..result.count]
            .iter()
            .map(|&(card, ns)| {
                (card as i32, colver_core::scoring::contract_made(&state, &played_by, ns))
            })
            .collect();
        let dict = PyDict::new_bound(py);
        dict.set_item("scores", scores)?;
        dict.set_item("deal_scores", deal_scores)?;
        dict.set_item("contract_made", contract_made)?;
        dict.set_item("best_card", result.best_card)?;
        Ok(dict)
    }

    /// La **marche** du barème : le plus grand saut d'un écart de score de donne
    /// entre deux totaux de points cartes voisins.
    ///
    /// `4V` sur un contrat normal, `2(162 + V·mult)` sous coinche — donc 320 à
    /// 640 d'un côté, 804 à 1044 de l'autre. C'est l'unité dans laquelle un
    /// écart rendu par `deal_scores` se lit : **un seuil absolu ne peut pas
    /// servir les deux régimes**, il s'exprime en fraction de cette marche.
    ///
    /// Vaut 0 hors contrat, et sur un capot déjà chuté où le barème est
    /// effectivement constant. Le saut à la frontière du capot dépendant des
    /// plis déjà ramassés, un appelant qui veut une constante de donne
    /// l'évalue à la première décision — cf. `scoring::deal_score_step`.
    fn deal_score_step(&self) -> i32 {
        colver_core::scoring::deal_score_step(&self.state, &self.played_by) as i32
    }

    /// Solve all 4 trump suits with the DD solver.
    /// Returns dict: {suits: [[ns_pts, ew_pts], ...4], elapsed_ms: f64}
    fn solve_all_suits<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let start = Instant::now();
        let hands = self.state.hands;
        let dealer = self.state.dealer;
        // Release the GIL during the solve (~70 ms for four suits on a median deal, up to
        // ~300 ms on tail deals) so callers can run several
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
    fn load_dmc_model(&mut self, path: PathBuf, hidden: Option<usize>) -> PyResult<()> {
        let path = path_str(&path)?;
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
    fn load_bid_model(&mut self, path: PathBuf, hidden: Option<usize>) -> PyResult<()> {
        let path = path_str(&path)?;
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

        let obs = build_bid_obs(net, &self.state, &self.bid_history, self.match_scores);
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
            played_by,
            play_order,
            bid_history,
            dmc_net: None,
            bid_net: None,
            match_scores: [0, 0],
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
    match_scores: [i32; 2],
) -> Vec<f32> {
    use colver_core::bid_obs;
    let obs_dim = net.obs_dim();
    // The score tail is written from the *speaking seat's* point of view, so it has to
    // be re-oriented per decision — the same match score is "my 1500" for one seat and
    // "their 1500" for the next.
    let my_team = (state.current_player() & 1) as usize;
    let (my_score, opp_score) = (match_scores[my_team], match_scores[1 - my_team]);
    match obs_dim {
        bid_obs::BID_OBS_DIM => bid_obs::make_bid_observation(state, history),
        bid_obs::BID_OBS_DIM_SCORE_AWARE => {
            let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE];
            bid_obs::write_bid_observation_score_aware(&mut buf, 0, state, history, my_score, opp_score);
            buf
        }
        bid_obs::BID_OBS_DIM_SCORE_AWARE_V2 => {
            let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE_V2];
            bid_obs::write_bid_observation_score_aware_v2(&mut buf, 0, state, history, my_score, opp_score);
            buf
        }
        bid_obs::BID_OBS_DIM_SCORE_AWARE_V3 => {
            let mut buf = vec![0.0f32; bid_obs::BID_OBS_DIM_SCORE_AWARE_V3];
            bid_obs::write_bid_observation_score_aware_v3(&mut buf, 0, state, history, my_score, opp_score);
            buf
        }
        other => panic!(
            "Unsupported bid NN obs_dim={} (expected 108/110/113/117)",
            other
        ),
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Agent — the one way to make a bot decide
// ══════════════════════════════════════════════════════════════════════

/// A seated bot, built from a bot spec (the same TOML the arena reads).
///
/// The agent owns everything it needs to play: its models, its beliefs, its
/// RNG, and — crucially — **its own source of determinized worlds**. Callers
/// no longer sample playgen worlds and push them in; that job moved inside,
/// which is what stopped the web and the arena from silently running different
/// agents under the same name.
///
/// Lifecycle, per deal:
///
/// ```python
/// agent = Agent(spec_toml, seat=1)
/// agent.init_deal(env)
/// while not env.is_terminal():
///     if env.current_player() == agent.seat:
///         d = agent.decide(env)          # {"action": …, "candidates": …, …}
///         action = d["action"]
///     else:
///         action = human_move()
///     for a in agents:                   # every agent sees every action…
///         a.observe(env, action)         # …with env still *before* the move
///     env.step(action)
/// ```
#[pyclass]
struct Agent {
    player: Box<dyn Player>,
    ctx: MatchContext,
    seat: u8,
    label: String,
}

fn agent_err(e: colver_core::agent::AgentError) -> PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(e.to_string())
}

#[pymethods]
impl Agent {
    /// Build from a bot spec written as TOML text.
    #[new]
    #[pyo3(signature = (spec, seat, seed=0))]
    fn new(spec: &str, seat: u8, seed: u64) -> PyResult<Self> {
        if seat >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("seat must be 0-3"));
        }
        let mut parsed = AgentSpec::from_toml_str(spec).map_err(agent_err)?;
        parsed.seed = seed;
        let player = parsed.build(seat).map_err(agent_err)?;
        let label = player.label().to_string();
        Ok(Agent { player, ctx: MatchContext::new(0), seat, label })
    }

    /// Build from a bot spec file, e.g. `arena/bots/v6_isdd_75M_belief.toml`.
    #[staticmethod]
    #[pyo3(signature = (path, seat, seed=0))]
    fn from_file(path: PathBuf, seat: u8, seed: u64) -> PyResult<Self> {
        let path = path_str(&path)?;
        let spec = std::fs::read_to_string(path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?;
        Agent::new(&spec, seat, seed)
    }

    #[getter]
    fn seat(&self) -> u8 {
        self.seat
    }

    #[getter]
    fn label(&self) -> &str {
        &self.label
    }

    /// Retune the per-move time budget without rebuilding the agent (which
    /// would discard its belief state mid-deal). No-op in count mode.
    fn set_time_ms(&mut self, ms: u32) {
        self.player.set_time_budget(ms);
    }

    /// Cumulative match score, which score-aware bidders condition on.
    /// Leave at zero for single-deal play.
    fn set_scores(&mut self, ns: i32, ew: i32) {
        self.ctx.scores = [ns, ew];
    }

    /// Start a new deal. `env` must be the freshly dealt, pre-auction position.
    fn init_deal(&mut self, env: PyRef<Env>) {
        self.ctx.reset_deal(env.state.dealer);
        self.player.init_deal(&env.state);
    }

    /// Observe an action. `env` must still hold the position **before** the
    /// action is applied — call this on every agent, for every seat's move,
    /// then `env.step(action)`.
    fn observe(&mut self, env: PyRef<Env>, action: u8) {
        let before = env.state;
        let player = before.current_player();
        self.player.observe(&before, player, action);
        self.ctx.track(&before, action);
    }

    /// Decide at the current position.
    ///
    /// Returns `{action, source, candidates: [[action, score], …],
    /// determinizations, worlds: {injected, playgen, belief, uniform},
    /// elapsed_ms}`. `worlds` is what makes a degraded run visible: if the
    /// playgen sidecar were substituted by uniform sampling, the counts would
    /// say so instead of the agent quietly getting weaker.
    fn decide<'py>(&mut self, py: Python<'py>, env: PyRef<Env>) -> PyResult<Bound<'py, PyDict>> {
        let state = env.state;
        drop(env);
        // Searching can take seconds; let the web's event loop run meanwhile.
        let decision = py
            .allow_threads(|| self.player.decide(&state, &self.ctx))
            .map_err(agent_err)?;

        let dict = PyDict::new_bound(py);
        dict.set_item("action", decision.action)?;
        dict.set_item("source", decision.stats.source)?;
        dict.set_item("candidates", decision.stats.candidates)?;
        dict.set_item("determinizations", decision.stats.determinizations)?;
        dict.set_item("elapsed_ms", decision.stats.elapsed_ms)?;
        let worlds = PyDict::new_bound(py);
        worlds.set_item("injected", decision.stats.worlds.injected)?;
        worlds.set_item("playgen", decision.stats.worlds.playgen)?;
        worlds.set_item("belief", decision.stats.worlds.belief)?;
        worlds.set_item("uniform", decision.stats.worlds.uniform)?;
        dict.set_item("worlds", worlds)?;
        Ok(dict)
    }

    /// Convenience: just the action.
    fn action(&mut self, py: Python<'_>, env: PyRef<Env>) -> PyResult<u8> {
        let state = env.state;
        drop(env);
        py.allow_threads(|| self.player.decide(&state, &self.ctx))
            .map(|d| d.action)
            .map_err(agent_err)
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Analyst — read-only introspection of the playgen world model
// ══════════════════════════════════════════════════════════════════════

/// What the playgen model *believes*, as opposed to what an agent *does*.
///
/// Kept separate from [`Agent`] on purpose: the analysis pages cannot change
/// how a bot plays, and a bot carries no code it never runs. Same lifecycle:
/// `init_deal`, then `observe` for every action, then query.
#[pyclass]
struct Analyst {
    inner: PlaygenAnalyst,
    rng: StdRng,
}

#[pymethods]
impl Analyst {
    #[new]
    #[pyo3(signature = (model_path, seed=0))]
    fn new(model_path: PathBuf, seed: u64) -> PyResult<Self> {
        let model_path = path_str(&model_path)?;
        let model = colver_core::agent::models::playgen_model(model_path).map_err(agent_err)?;
        Ok(Analyst { inner: PlaygenAnalyst::new(model), rng: StdRng::seed_from_u64(seed) })
    }

    /// Rebuild the sampler state at a position by replaying the deal, so the
    /// analysis pages can jump around without keeping a live object per view.
    #[staticmethod]
    #[pyo3(signature = (model_path, dealer, hands, actions, observer, seed=0))]
    fn replay(
        model_path: PathBuf,
        dealer: u8,
        hands: Vec<Vec<u8>>,
        actions: Vec<u8>,
        observer: u8,
        seed: u64,
    ) -> PyResult<Self> {
        if observer >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("observer must be 0-3"));
        }
        let mut analyst = Analyst::new(model_path, seed)?;
        let mut env = Env::deal_with_hands(dealer, hands)?;
        analyst.inner.init_deal(&env.state, observer);
        for action in actions {
            let before = env.state;
            analyst.inner.observe(&before, before.current_player(), action);
            env.state.step(action);
        }
        Ok(analyst)
    }

    fn init_deal(&mut self, env: PyRef<Env>, observer: u8) -> PyResult<()> {
        if observer >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("observer must be 0-3"));
        }
        self.inner.init_deal(&env.state, observer);
        Ok(())
    }

    fn observe(&mut self, env: PyRef<Env>, action: u8) {
        let before = env.state;
        let player = before.current_player();
        self.inner.observe(&before, player, action);
    }

    /// Card-location marginals `weights[player][card]`, or `None` if the model
    /// cannot sample here (notably mid-auction for v1 models).
    #[pyo3(signature = (env, n_worlds=50, temperature=1.0))]
    fn marginals(
        &mut self,
        py: Python<'_>,
        env: PyRef<Env>,
        n_worlds: usize,
        temperature: f32,
    ) -> Option<Vec<Vec<f32>>> {
        let state = env.state;
        drop(env);
        py.allow_threads(|| self.inner.marginals(&state, n_worlds, temperature, &mut self.rng))
            .map(|w| w.iter().map(|row| row.to_vec()).collect())
    }

    /// Masked-softmax bid probabilities (43) at the current auction point, or
    /// `None` for models without a bid head.
    #[pyo3(signature = (env, temperature=1.0))]
    fn bid_policy(&mut self, env: PyRef<Env>, temperature: f32) -> Option<Vec<f32>> {
        let state = env.state;
        drop(env);
        let logits = self.inner.bid_policy(&state)?;
        Some(masked_softmax(&logits, state.legal_actions(), temperature))
    }

    /// Full deals sampled from a mid-auction position: the auction is finished
    /// with the bid head, then the deal is played out to reveal the hands.
    #[pyo3(signature = (env, n_worlds, temperature=1.0))]
    fn auction_deals(
        &mut self,
        py: Python<'_>,
        env: PyRef<Env>,
        n_worlds: usize,
        temperature: f32,
    ) -> Option<Vec<Vec<Vec<u8>>>> {
        let state = env.state;
        drop(env);
        let worlds = py.allow_threads(|| {
            self.inner.auction_deals(&state, n_worlds, temperature, &mut self.rng)
        });
        if worlds.is_empty() {
            return None;
        }
        Some(worlds.iter().map(|hands| hands.iter().map(|&h| mask_to_cards(h)).collect()).collect())
    }

    /// Worlds sampled from a mid-**play** position: `worlds[i][seat]` is that
    /// seat's *remaining* cards, the observer's own hand being its real one.
    ///
    /// Unlike [`auction_deals`](Self::auction_deals) these are not full deals —
    /// the already-played cards are not repeated. A caller that needs a
    /// solvable position must fold each seat's played cards back in.
    #[pyo3(signature = (env, n_worlds, temperature=1.0))]
    fn play_worlds(
        &mut self,
        py: Python<'_>,
        env: PyRef<Env>,
        n_worlds: usize,
        temperature: f32,
    ) -> Option<Vec<Vec<Vec<u8>>>> {
        let state = env.state;
        drop(env);
        let worlds = py.allow_threads(|| {
            self.inner.play_worlds(&state, n_worlds, temperature, &mut self.rng)
        });
        if worlds.is_empty() {
            return None;
        }
        Some(worlds.iter().map(|hands| hands.iter().map(|&h| mask_to_cards(h)).collect()).collect())
    }
}

fn mask_to_cards(mask: u32) -> Vec<u8> {
    (0..32u8).filter(|c| mask & (1 << c) != 0).collect()
}

/// Softmax over the legal actions only; illegal entries stay at 0.
fn masked_softmax(logits: &[f32], legal: u64, temperature: f32) -> Vec<f32> {
    let t = temperature.max(1e-3);
    let n = logits.len();
    let max_l = (0..n)
        .filter(|&c| legal & (1u64 << c) != 0)
        .map(|c| logits[c])
        .fold(f32::NEG_INFINITY, f32::max);
    let mut probs = vec![0.0f32; n];
    let mut total = 0.0f32;
    for c in 0..n {
        if legal & (1u64 << c) != 0 {
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
    probs
}

// ══════════════════════════════════════════════════════════════════════
//  Beliefs — what the card-belief models think, for the analysis pages
// ══════════════════════════════════════════════════════════════════════

/// Card-location beliefs from one observer's seat: the neural belief net's
/// soft prediction (combined with the hard constraints that are facts) and the
/// heuristic `CardBeliefs` model on its own.
///
/// This is IS-DD's belief machinery exposed for display. It is deliberately
/// separate from [`Agent`]: looking at what a model believes must never be
/// able to change how a bot plays.
#[pyclass]
struct Beliefs {
    search: IsDdSearch,
    observer: u8,
}

#[pymethods]
impl Beliefs {
    /// Rebuild the belief state at a position by replaying the deal.
    ///
    /// Replaying (rather than holding a live object) keeps the analysis pages
    /// stateless: they can jump to any action index and ask again. It is cheap
    /// — belief updates involve no search.
    #[staticmethod]
    #[pyo3(signature = (dealer, hands, actions, observer, belief_model=None))]
    fn replay(
        dealer: u8,
        hands: Vec<Vec<u8>>,
        actions: Vec<u8>,
        observer: u8,
        belief_model: Option<PathBuf>,
    ) -> PyResult<Self> {
        if observer >= 4 {
            return Err(pyo3::exceptions::PyValueError::new_err("observer must be 0-3"));
        }
        let mut env = Env::deal_with_hands(dealer, hands)?;
        let mut search = IsDdSearch::new();
        if let Some(ref buf) = belief_model {
            let path = path_str(buf)?;
            search
                .load_belief_net(path)
                .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("{path}: {e}")))?;
        }
        search.init_deal(&env.state, observer, false);
        for action in actions {
            let before = env.state;
            search.record_action(&before, before.current_player(), action);
            env.state.step(action);
        }
        Ok(Beliefs { search, observer })
    }

    /// `{nn: [[f32; 32]; 4] | None, heuristic: [[f32; 32]; 4] | None}` —
    /// `weights[player][card]`. `nn` is `None` when no belief net was given.
    fn weights<'py>(&mut self, py: Python<'py>, env: PyRef<Env>) -> PyResult<Bound<'py, PyDict>> {
        let state = env.state;
        drop(env);
        let (nn, heuristic) = self.search.get_belief_weights(&state, self.observer);
        let dict = PyDict::new_bound(py);
        let to_py = |w: Option<[[f32; 32]; 4]>| -> Option<Vec<Vec<f32>>> {
            w.map(|w| w.iter().map(|row| row.to_vec()).collect())
        };
        dict.set_item("nn", to_py(nn))?;
        dict.set_item("heuristic", to_py(heuristic))?;
        Ok(dict)
    }
}

// ---------------------------------------------------------------------------
// Classification des mains (colver_core::hand_class)
// ---------------------------------------------------------------------------

/// Convertit une liste d'indices de carte 0-31 en CardSet, en validant le compte.
fn cards_to_set(cards: &[u8], expect: Option<usize>) -> PyResult<card::CardSet> {
    let mut set: card::CardSet = 0;
    for &c in cards {
        if c >= 32 {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "indice de carte hors bornes : {c}"
            )));
        }
        set |= 1 << c;
    }
    if set.count_ones() as usize != cards.len() {
        return Err(pyo3::exceptions::PyValueError::new_err("cartes en double"));
    }
    if let Some(n) = expect {
        if cards.len() != n {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "main de {n} cartes attendue, {} reçues",
                cards.len()
            )));
        }
    }
    Ok(set)
}

fn set_to_cards(set: card::CardSet) -> Vec<u8> {
    (0..32u8).filter(|c| set >> c & 1 == 1).collect()
}

fn check_trump(trump: u8) -> PyResult<()> {
    if trump >= 4 {
        return Err(pyo3::exceptions::PyValueError::new_err(
            "couleur d'atout hors bornes (0-3)",
        ));
    }
    Ok(())
}

/// Index canonique d'une main de 8 cartes, dans [0, 472579).
///
/// Constant sur les 24 permutations de couleurs : deux mains identiques à un
/// échange de couleurs près ont le même index.
#[pyfunction]
fn hand_class_id(cards: Vec<u8>) -> PyResult<u32> {
    Ok(hand_class::hand_class_id(cards_to_set(&cards, Some(8))?))
}

/// Index canonique à atout désigné, dans [0, 1820803).
#[pyfunction]
fn hand_class_id_trump(cards: Vec<u8>, trump: u8) -> PyResult<u32> {
    check_trump(trump)?;
    Ok(hand_class::hand_class_id_trump(
        cards_to_set(&cards, Some(8))?,
        trump,
    ))
}

/// Une main représentative de la classe `class_id`, en indices de carte.
#[pyfunction]
fn hand_from_class_id(class_id: u32) -> PyResult<Vec<u8>> {
    if class_id >= hand_class::NUM_HAND_CLASSES {
        return Err(pyo3::exceptions::PyValueError::new_err("class_id hors bornes"));
    }
    Ok(set_to_cards(hand_class::hand_from_class_id(class_id)))
}

/// Idem à atout désigné ; l'atout est rendu en couleur 0 (pique).
#[pyfunction]
fn hand_from_class_id_trump(class_id: u32) -> PyResult<Vec<u8>> {
    if class_id >= hand_class::NUM_HAND_CLASSES_TRUMP {
        return Err(pyo3::exceptions::PyValueError::new_err("class_id hors bornes"));
    }
    Ok(set_to_cards(hand_class::hand_from_class_id_trump(class_id)))
}

/// Code de main lisible et insensible aux couleurs, p.ex. `"T5.J9AT.A1/A1/x1"`.
///
/// `level` ∈ {"length", "trump", "shape", "tops", "full"} — du plus grossier
/// (9 codes) au plus fin (6654). La chaîne est la clé de regroupement.
#[pyfunction]
#[pyo3(signature = (cards, trump, level="full"))]
fn hand_code(cards: Vec<u8>, trump: u8, level: &str) -> PyResult<String> {
    use hand_class::CodeLevel::*;
    check_trump(trump)?;
    let lvl = match level {
        "length" => Length,
        "trump" => Trump,
        "shape" => Shape,
        "tops" => Tops,
        "full" => Full,
        other => {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "niveau inconnu : {other:?} (length|trump|shape|tops|full)"
            )))
        }
    };
    let set = cards_to_set(&cards, Some(8))?;
    Ok(hand_class::HandCode::from_hand(set, trump)
        .coarsen(lvl)
        .to_string())
}

/// Matadors signés (« mit N / ohne N » du Skat) : longueur de la série
/// ininterrompue des plus gros atouts, positive si on la détient.
#[pyfunction]
fn matadors(cards: Vec<u8>, trump: u8) -> PyResult<i8> {
    check_trump(trump)?;
    Ok(hand_class::matadors(cards_to_set(&cards, Some(8))?, trump))
}

#[pymodule]
fn _colver(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Env>()?;
    m.add_class::<Agent>()?;
    m.add_class::<Analyst>()?;
    m.add_class::<Beliefs>()?;
    m.add_function(wrap_pyfunction!(hand_class_id, m)?)?;
    m.add_function(wrap_pyfunction!(hand_class_id_trump, m)?)?;
    m.add_function(wrap_pyfunction!(hand_from_class_id, m)?)?;
    m.add_function(wrap_pyfunction!(hand_from_class_id_trump, m)?)?;
    m.add_function(wrap_pyfunction!(hand_code, m)?)?;
    m.add_function(wrap_pyfunction!(matadors, m)?)?;
    m.add("NUM_HAND_CLASSES", hand_class::NUM_HAND_CLASSES)?;
    m.add("NUM_HAND_CLASSES_TRUMP", hand_class::NUM_HAND_CLASSES_TRUMP)?;
    // Empreinte des sources playgen/engine de *ce* binaire — donc de celles du
    // conteneur web, puisque c'est le même build. `playgen_gpu.probe` la compare
    // à celle que le sidecar publie sur son /health : c'est ce qui permet à
    // /health de dire qu'un sidecar déployé à la main a pris du retard.
    m.add("PLAYGEN_SURFACE", colver_core::playgen::SURFACE)?;
    Ok(())
}
