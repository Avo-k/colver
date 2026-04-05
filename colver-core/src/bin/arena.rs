/// Arena: systematic bot comparison framework.
///
/// Bots are defined as TOML files in arena/bots/. Results stored in arena/results/matches.csv.
///
/// Usage:
///   cargo run --bin arena --release -- list
///   cargo run --bin arena --release -- h2h bot_a bot_b --matches 200
///   cargo run --bin arena --release -- round-robin --matches 100 [--bots a,b,c]
///   cargo run --bin arena --release -- results [--bot name]

use colver_core::bid_eval::BidFunction;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{self, BID_OBS_DIM};
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM, OBS_DIM_TR};
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::mcts::{MctsConfig, MctsSearch, RolloutPolicy};
use colver_core::naive_ismcts::{NaiveIsMctsConfig, NaiveIsMctsSearch};
use colver_core::rollout::heuristic_play_action;
use colver_core::rule_player::rule_play_action;
use colver_core::smart_ismcts::{SmartIsMctsConfig, SmartIsMctsSearch};
use colver_core::state::{GameState, Phase};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::Instant;

const MATCH_TARGET: i32 = 2000;
const RESULTS_PATH: &str = "arena/results/matches.csv";
const BOTS_DIR: &str = "arena/bots";

// ══════════════════════════════════════════════════════════════════════
//  DD calibration penalty for bid NN Q-values
// ══════════════════════════════════════════════════════════════════════

/// Select best bid action with a penalty on high bids to correct DD overestimation.
///
/// The NN was trained with DD-optimal rewards which overestimate success at high
/// contract levels. This wrapper subtracts a scaled penalty from Q-values.
///
/// Penalty shape is linear in bid value (calibrated from DD vs DouDou50 data):
///   - PASS (0): no penalty
///   - Bids 80 (actions 1-4): no penalty (DD well-calibrated here)
///   - Bids 90-160: penalty scales linearly with (value - 80)
///   - Capot (37-40): max penalty (DD says 252, real ≈ 162)
///   - Coinche/Surcoinche (41-42): moderate penalty
///
/// `penalty` is the scaling factor — try 0.05 to 0.3.
fn bid_action_with_penalty(
    bn: &mut BidNet,
    obs: &[f32],
    legal_mask: u64,
    penalty: f32,
) -> u8 {
    let q = bn.evaluate(obs);

    // Penalty per action based on bid level
    // Calibrated from DD vs DouDou50 data (4M samples):
    //   80: Δ≈0, 90: Δ≈-2, 100: Δ≈-4, 110: Δ≈-7, 120: Δ≈-9, 130: Δ≈-12, capot: Δ≈-90
    let bid_penalty = |action: u8| -> f32 {
        if action == 0 { return 0.0; }             // PASS
        if action >= 41 { return penalty * 0.5; }   // COINCHE/SURCOINCHE
        if action >= 37 { return penalty * 2.5; }   // CAPOT (massive DD gap)

        // Regular bids 1-36: value_idx = (action-1)/4, value = 80 + value_idx*10
        let value_idx = (action - 1) / 4;
        // Linear scaling: 0 at 80, penalty at 160
        let level = value_idx as f32 / 8.0; // 0.0 at 80, 1.0 at 160
        penalty * level
    };

    let mut best_action = 0u8;
    let mut best_q = f32::NEG_INFINITY;
    let mut mask = legal_mask;
    while mask != 0 {
        let bit = mask.trailing_zeros() as u8;
        if (bit as usize) < 43 {
            let q_adj = q[bit as usize] - bid_penalty(bit);
            if q_adj > best_q {
                best_q = q_adj;
                best_action = bit;
            }
        }
        mask &= mask - 1;
    }
    best_action
}

// ══════════════════════════════════════════════════════════════════════
//  Bot config (parsed from TOML)
// ══════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
struct BotConfig {
    name: String,
    bid_strategy: String,     // "heuristic", "improved", "improved_v2", "smart", "roro", "maxi", "petit_bide", "moelleux", "nn"
    bid_model: Option<String>,
    play_method: String,      // "naive_ismcts", "smart_ismcts", "is_dd", "smart_is_dd", "dmc", "oracle", "heuristic"
    play_model: Option<String>,
    play_residual: bool,
    time_ms: u32,
    determinizations: u32,
    oracle_iters: u32,
    switch_at: u8,        // for dmc_then_dd: switch to DD after this many tricks (default 5)
    bid_hidden: usize,        // hidden size for bid NN (default 256)
    bid_penalty: f32,             // Q-value penalty for high bids (0.0 = off, calibrated ~0.1-0.3)
    belief_model: Option<String>,
    belief_hard_constraints: bool,
    bid_belief_model: Option<String>,
}

impl Default for BotConfig {
    fn default() -> Self {
        Self {
            name: String::new(),
            bid_strategy: "improved_v2".into(),
            bid_model: None,
            play_method: "naive_ismcts".into(),
            play_model: None,
            play_residual: false,
            time_ms: 20,
            determinizations: 20,
            oracle_iters: 2000,
            switch_at: 5,
            bid_hidden: 256,
            bid_penalty: 0.0,
            belief_model: None,
            belief_hard_constraints: true,
            bid_belief_model: None,
        }
    }
}

impl BotConfig {
    /// Short model name: include parent dir if not "models/"
    fn short_model_name(path: &str) -> String {
        let p = std::path::Path::new(path);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or(path);
        let parent = p.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str());
        match parent {
            Some("models") | None => stem.to_string(),
            Some(dir) => format!("{}/{}", dir, stem),
        }
    }

    /// Short label for the bid strategy, e.g. "nn:bid_v2/bid_nn_final" or "improved_v2"
    fn bid_label(&self) -> String {
        if let Some(ref model) = self.bid_model {
            format!("{}:{}", self.bid_strategy, Self::short_model_name(model))
        } else {
            self.bid_strategy.clone()
        }
    }

    /// Short label for the play method, e.g. "dmc:play_20M" or "smart_is_dd:50ms"
    fn play_label(&self) -> String {
        if let Some(ref model) = self.play_model {
            format!("{}:{}", self.play_method, Self::short_model_name(model))
        } else {
            let mut label = self.play_method.clone();
            if self.play_method.contains("ismcts") || self.play_method.contains("is_dd") {
                label = format!("{}:{}ms", label, self.time_ms);
            }
            label
        }
    }
}

fn parse_bot_config(path: &str) -> Result<BotConfig, String> {
    let content = std::fs::read_to_string(path)
        .map_err(|e| format!("cannot read {}: {}", path, e))?;

    let name = std::path::Path::new(path)
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("unknown")
        .to_string();

    let mut cfg = BotConfig { name, ..Default::default() };
    let mut section = "";

    for line in content.lines() {
        let line = line.trim();
        if line.is_empty() || line.starts_with('#') {
            continue;
        }
        if line.starts_with('[') && line.ends_with(']') {
            section = match &line[1..line.len() - 1] {
                "bid" => "bid",
                "play" => "play",
                "belief" => "belief",
                other => return Err(format!("unknown section [{}] in {}", other, path)),
            };
            continue;
        }
        if let Some(eq_pos) = line.find('=') {
            let key = line[..eq_pos].trim();
            let val = line[eq_pos + 1..].trim().trim_matches('"');

            match (section, key) {
                ("bid", "strategy") => cfg.bid_strategy = val.to_string(),
                ("bid", "model") => cfg.bid_model = Some(val.to_string()),
                ("bid", "hidden") => cfg.bid_hidden = val.parse().unwrap_or(256),
                ("bid", "penalty") => cfg.bid_penalty = val.parse().unwrap_or(0.0),
                ("play", "method") => cfg.play_method = val.to_string(),
                ("play", "model") => cfg.play_model = Some(val.to_string()),
                ("play", "residual") => cfg.play_residual = val == "true",
                ("play", "time_ms") => cfg.time_ms = val.parse().unwrap_or(20),
                ("play", "determinizations") => cfg.determinizations = val.parse().unwrap_or(20),
                ("play", "oracle_iters") => cfg.oracle_iters = val.parse().unwrap_or(2000),
                ("play", "switch_at") => cfg.switch_at = val.parse().unwrap_or(5),
                ("belief", "model") => cfg.belief_model = Some(val.to_string()),
                ("belief", "use_hard_constraints") => cfg.belief_hard_constraints = val == "true",
                ("belief", "bid_model") => cfg.bid_belief_model = Some(val.to_string()),
                _ => {} // ignore unknown keys
            }
        }
    }

    Ok(cfg)
}

fn load_all_bots() -> Vec<BotConfig> {
    let mut bots = Vec::new();
    let dir = match std::fs::read_dir(BOTS_DIR) {
        Ok(d) => d,
        Err(e) => {
            eprintln!("Cannot read {}: {}", BOTS_DIR, e);
            return bots;
        }
    };
    let mut paths: Vec<_> = dir
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().and_then(|s| s.to_str()) == Some("toml"))
        .map(|e| e.path())
        .collect();
    paths.sort();
    for path in paths {
        match parse_bot_config(path.to_str().unwrap_or("")) {
            Ok(cfg) => bots.push(cfg),
            Err(e) => eprintln!("  Warning: {}", e),
        }
    }
    bots
}

// ══════════════════════════════════════════════════════════════════════
//  Shared model weights (thread-safe)
// ══════════════════════════════════════════════════════════════════════

struct DmcWeights {
    floats: Vec<f32>,
    hidden: usize,
    obs_dim: usize,
    dueling: bool,
    residual: bool,
}

impl DmcWeights {
    fn load(path: &str, residual: bool) -> std::io::Result<Self> {
        let net = DmcNet::load(path)?;
        let obs_dim = net.obs_dim();
        let hidden = net.hidden();
        let dueling = net.is_dueling();
        drop(net);
        let data = std::fs::read(path)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(DmcWeights { floats, hidden, obs_dim, dueling, residual })
    }

    fn make_net(&self) -> DmcNet {
        let mut net = DmcNet::from_floats(&self.floats, self.hidden, self.obs_dim, self.dueling).unwrap();
        if self.residual {
            net.set_residual(true);
        }
        net
    }
}

struct BidNetWeights {
    floats: Vec<f32>,
    hidden: usize,
    obs_dim: usize,
    dueling: bool,
    layers: usize,
}

impl BidNetWeights {
    fn load(path: &str, hidden: usize) -> std::io::Result<Self> {
        let net = BidNet::load_with_hidden(path, hidden)?;
        let obs_dim = net.obs_dim();
        let hidden = net.hidden();
        let dueling = net.is_dueling();
        let layers = net.layers();
        drop(net);
        let data = std::fs::read(path)?;
        let floats: Vec<f32> = data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect();
        Ok(BidNetWeights { floats, hidden, obs_dim, dueling, layers })
    }

    fn make_net(&self) -> BidNet {
        BidNet::from_floats_with_layers(&self.floats, self.hidden, self.obs_dim, self.dueling, self.layers).unwrap()
    }
}

// ══════════════════════════════════════════════════════════════════════
//  Agent (runtime representation of a bot)
// ══════════════════════════════════════════════════════════════════════

#[derive(Clone)]
enum CardPlayMethod {
    NaiveIsMcts,
    SmartIsMcts,
    Oracle,
    Dmc(usize),      // index into shared dmc_weights
    IsDd,
    SmartIsDd,
    Heuristic,
    RulePlayer,       // Fair rule-based player (no peeking at opponent hands)
    DmcThenDd {       // DMC for early tricks, IS-DD for endgame
        dmc_idx: usize,
        switch_at: u8, // switch to DD when tricks_completed >= this (e.g. 5 = last 3 tricks)
    },
    BisDd,            // Unified DD-based agent (bid + play via BisDdAgent)
}

#[derive(Clone)]
struct Agent {
    name: String,
    bid_label: String,
    play_label: String,
    bid_function: BidFunction,
    card_play: CardPlayMethod,
    bid_weights_idx: Option<usize>,  // index into shared bid_weights, None = use bid_function
    bid_penalty: f32,                // Q-value penalty scaling for high bids
    belief_path: Option<String>,
    bid_belief_path: Option<String>,
    time_ms: u32,
    determinizations: u32,
    oracle_iters: u32,
}

/// All shared model weights for the tournament.
struct SharedModels {
    dmc_weights: Vec<DmcWeights>,
    bid_weights: Vec<BidNetWeights>,
}

fn build_agent(cfg: &BotConfig, models: &mut SharedModels) -> Result<Agent, String> {
    let bid_function = match cfg.bid_strategy.as_str() {
        "heuristic" => BidFunction::Heuristic,
        "improved" => BidFunction::Improved,
        "improved_v2" => BidFunction::ImprovedV2,
        "improved_v3" => BidFunction::ImprovedV3,
        "smart" => BidFunction::Smart,
        "roro" => BidFunction::Roro,
        "petit_bide" => BidFunction::PetitBide,
        "moelleux" => BidFunction::Moelleux,
        "maxi" => BidFunction::Maxi,
        "bis_dd" => BidFunction::BisDd, // placeholder, actual decisions via stateful BisDdAgent
        "nn" => BidFunction::ImprovedV2, // placeholder, actual NN used via bid_weights_idx
        other => return Err(format!("unknown bid strategy '{}' for bot {}", other, cfg.name)),
    };

    let bid_weights_idx = if cfg.bid_strategy == "nn" {
        let path = cfg.bid_model.as_deref()
            .ok_or_else(|| format!("bot {} has bid strategy 'nn' but no bid model", cfg.name))?;
        // Dedup: reuse if same path already loaded
        let existing = models.bid_weights.iter().position(|_| {
            // Simple: always add (models are small). Could dedup by path if needed.
            false
        });
        let idx = if let Some(idx) = existing {
            idx
        } else {
            let w = BidNetWeights::load(path, cfg.bid_hidden)
                .map_err(|e| format!("cannot load bid model {} for bot {}: {}", path, cfg.name, e))?;
            models.bid_weights.push(w);
            models.bid_weights.len() - 1
        };
        Some(idx)
    } else {
        None
    };

    let card_play = match cfg.play_method.as_str() {
        "naive_ismcts" => CardPlayMethod::NaiveIsMcts,
        "smart_ismcts" => CardPlayMethod::SmartIsMcts,
        "is_dd" => CardPlayMethod::IsDd,
        "smart_is_dd" => CardPlayMethod::SmartIsDd,
        "oracle" => CardPlayMethod::Oracle,
        "heuristic" => CardPlayMethod::Heuristic,
        "rule" => CardPlayMethod::RulePlayer,
        "dmc" => {
            let path = cfg.play_model.as_deref()
                .ok_or_else(|| format!("bot {} has play method 'dmc' but no play model", cfg.name))?;
            let w = DmcWeights::load(path, cfg.play_residual)
                .map_err(|e| format!("cannot load play model {} for bot {}: {}", path, cfg.name, e))?;
            let idx = models.dmc_weights.len();
            models.dmc_weights.push(w);
            CardPlayMethod::Dmc(idx)
        }
        "dmc_then_dd" => {
            let path = cfg.play_model.as_deref()
                .ok_or_else(|| format!("bot {} has play method 'dmc_then_dd' but no play model", cfg.name))?;
            let w = DmcWeights::load(path, cfg.play_residual)
                .map_err(|e| format!("cannot load play model {} for bot {}: {}", path, cfg.name, e))?;
            let idx = models.dmc_weights.len();
            models.dmc_weights.push(w);
            CardPlayMethod::DmcThenDd { dmc_idx: idx, switch_at: cfg.switch_at }
        }
        "bis_dd" => CardPlayMethod::BisDd,
        other => return Err(format!("unknown play method '{}' for bot {}", other, cfg.name)),
    };

    Ok(Agent {
        name: cfg.name.clone(),
        bid_label: cfg.bid_label(),
        play_label: cfg.play_label(),
        bid_function,
        card_play,
        bid_weights_idx,
        bid_penalty: cfg.bid_penalty,
        belief_path: cfg.belief_model.clone(),
        bid_belief_path: cfg.bid_belief_model.clone(),
        time_ms: cfg.time_ms,
        determinizations: cfg.determinizations,
        oracle_iters: cfg.oracle_iters,
    })
}

// ══════════════════════════════════════════════════════════════════════
//  Match play (adapted from agent_tournament.rs)
// ══════════════════════════════════════════════════════════════════════

struct MatchResult {
    winner: u8,
    ns_final: i32,
    ew_final: i32,
}

fn play_match(
    ns_agent: &Agent,
    ew_agent: &Agent,
    models: &SharedModels,
    rng: &mut StdRng,
) -> MatchResult {
    let make_naive_config = |a: &Agent| NaiveIsMctsConfig {
        iterations_per_det: 50,
        time_limit_ms: Some(a.time_ms),
        ..Default::default()
    };
    let make_smart_config = |a: &Agent| SmartIsMctsConfig {
        iterations_per_det: 50,
        time_limit_ms: Some(a.time_ms),
        ..Default::default()
    };
    let make_dd_config = |a: &Agent| IsDdConfig {
        determinizations: a.determinizations,
        time_limit_ms: Some(a.time_ms),
        ..Default::default()
    };
    let make_oracle_config = |a: &Agent| MctsConfig {
        iterations: a.oracle_iters,
        rollout_policy: RolloutPolicy::HeuristicPlay,
        ..Default::default()
    };

    // Pre-create thread-local models
    let mut dmc_nets: Vec<Option<DmcNet>> = (0..models.dmc_weights.len()).map(|_| None).collect();
    for agent in [ns_agent, ew_agent] {
        let idx = match agent.card_play {
            CardPlayMethod::Dmc(idx) => Some(idx),
            CardPlayMethod::DmcThenDd { dmc_idx, .. } => Some(dmc_idx),
            _ => None,
        };
        if let Some(idx) = idx {
            if dmc_nets[idx].is_none() {
                dmc_nets[idx] = Some(models.dmc_weights[idx].make_net());
            }
        }
    }

    // Pre-create thread-local BidNet if needed
    let mut bid_nets: Vec<Option<BidNet>> = (0..models.bid_weights.len()).map(|_| None).collect();
    for agent in [ns_agent, ew_agent] {
        if let Some(idx) = agent.bid_weights_idx {
            if bid_nets[idx].is_none() {
                bid_nets[idx] = Some(models.bid_weights[idx].make_net());
            }
        }
    }
    let max_bid_obs = models.bid_weights.iter().map(|w| w.obs_dim).max().unwrap_or(BID_OBS_DIM);
    let mut bid_obs_buf = vec![0.0f32; max_bid_obs];

    // Determine obs buffer size (use largest needed)
    let max_obs_dim = models.dmc_weights.iter().map(|w| w.obs_dim).max().unwrap_or(OBS_DIM);
    let mut obs_buf = vec![0.0f32; max_obs_dim];

    let mut ns_cumulative: i32 = 0;
    let mut ew_cumulative: i32 = 0;
    let mut dealer: u8 = rng.gen_range(0..4);

    // Pre-create Smart IS-DD searches with belief net (also for DmcThenDd)
    let ns_is_smart_dd = matches!(ns_agent.card_play, CardPlayMethod::SmartIsDd | CardPlayMethod::DmcThenDd { .. });
    let ew_is_smart_dd = matches!(ew_agent.card_play, CardPlayMethod::SmartIsDd | CardPlayMethod::DmcThenDd { .. });
    let mut ns_smart_dd = [IsDdSearch::new(), IsDdSearch::new()];
    let mut ew_smart_dd = [IsDdSearch::new(), IsDdSearch::new()];
    if ns_is_smart_dd {
        if let Some(path) = &ns_agent.belief_path {
            let _ = ns_smart_dd[0].load_belief_net(path);
            let _ = ns_smart_dd[1].load_belief_net(path);
        }
    }
    if ew_is_smart_dd {
        if let Some(path) = &ew_agent.belief_path {
            let _ = ew_smart_dd[0].load_belief_net(path);
            let _ = ew_smart_dd[1].load_belief_net(path);
        }
    }

    // BisDd agents (2 per team: one per player)
    let ns_bis_dd_bid = matches!(ns_agent.bid_function, BidFunction::BisDd);
    let ns_bis_dd_play = matches!(ns_agent.card_play, CardPlayMethod::BisDd);
    let ns_is_bis_dd = ns_bis_dd_bid || ns_bis_dd_play;
    let ew_bis_dd_bid = matches!(ew_agent.bid_function, BidFunction::BisDd);
    let ew_bis_dd_play = matches!(ew_agent.card_play, CardPlayMethod::BisDd);
    let ew_is_bis_dd = ew_bis_dd_bid || ew_bis_dd_play;
    let bis_dd_config_ns = colver_core::bis_dd::BisDdConfig {
        min_dets: ns_agent.determinizations,
        ..Default::default()
    };
    let bis_dd_config_ew = colver_core::bis_dd::BisDdConfig {
        min_dets: ew_agent.determinizations,
        ..Default::default()
    };
    let mut ns_bis_dd = [
        colver_core::bis_dd::BisDdAgent::new(bis_dd_config_ns.clone(), rng.gen()),
        colver_core::bis_dd::BisDdAgent::new(bis_dd_config_ns, rng.gen()),
    ];
    let mut ew_bis_dd = [
        colver_core::bis_dd::BisDdAgent::new(bis_dd_config_ew.clone(), rng.gen()),
        colver_core::bis_dd::BisDdAgent::new(bis_dd_config_ew, rng.gen()),
    ];

    // Load bid belief nets for BisDd agents
    let mut ns_bid_belief_net = if ns_is_bis_dd {
        ns_agent.bid_belief_path.as_ref().and_then(|path| {
            colver_core::belief_net::BeliefNet::load_with_hidden(path, 256).ok()
        })
    } else { None };
    let mut ew_bid_belief_net = if ew_is_bis_dd {
        ew_agent.bid_belief_path.as_ref().and_then(|path| {
            colver_core::belief_net::BeliefNet::load_with_hidden(path, 256).ok()
        })
    } else { None };

    while ns_cumulative < MATCH_TARGET && ew_cumulative < MATCH_TARGET {
        let mut state = GameState::deal_random(dealer, rng);
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);

        // Per-deal search objects
        let mut ns_naive = NaiveIsMctsSearch::new();
        let mut ew_naive = NaiveIsMctsSearch::new();
        let mut ns_smart = [SmartIsMctsSearch::new(), SmartIsMctsSearch::new()];
        let mut ew_smart = [SmartIsMctsSearch::new(), SmartIsMctsSearch::new()];
        let mut ns_dd = IsDdSearch::new();
        let mut ew_dd = IsDdSearch::new();
        let mut oracle = MctsSearch::new();

        let ns_is_smart = matches!(ns_agent.card_play, CardPlayMethod::SmartIsMcts);
        let ew_is_smart = matches!(ew_agent.card_play, CardPlayMethod::SmartIsMcts);

        if ns_is_smart {
            ns_smart[0].init_deal(&state, 0, true);
            ns_smart[1].init_deal(&state, 2, true);
        }
        if ew_is_smart {
            ew_smart[0].init_deal(&state, 1, true);
            ew_smart[1].init_deal(&state, 3, true);
        }
        if ns_is_smart_dd {
            ns_smart_dd[0].init_deal(&state, 0, true);
            ns_smart_dd[1].init_deal(&state, 2, true);
        }
        if ew_is_smart_dd {
            ew_smart_dd[0].init_deal(&state, 1, true);
            ew_smart_dd[1].init_deal(&state, 3, true);
        }
        if ns_is_bis_dd {
            ns_bis_dd[0].init_deal(0, state.hands[0]);
            ns_bis_dd[1].init_deal(2, state.hands[2]);
        }
        if ew_is_bis_dd {
            ew_bis_dd[0].init_deal(1, state.hands[1]);
            ew_bis_dd[1].init_deal(3, state.hands[3]);
        }

        while !state.is_terminal() {
            let player = state.current_player();
            let is_ns = player == 0 || player == 2;
            let agent = if is_ns { ns_agent } else { ew_agent };
            let state_before = state;

            let action = if state.phase == Phase::Bidding {
                if ns_bis_dd_bid && is_ns {
                    let idx = if player == 0 { 0 } else { 1 };
                    ns_bis_dd[idx].decide(&state)
                } else if ew_bis_dd_bid && !is_ns {
                    let idx = if player == 1 { 0 } else { 1 };
                    ew_bis_dd[idx].decide(&state)
                } else if let Some(idx) = agent.bid_weights_idx {
                    if let Some(ref mut bn) = bid_nets[idx] {
                        bid_obs::write_bid_observation(
                            &mut bid_obs_buf, 0, &state, &tracking.bid_history,
                        );
                        let legal_mask = state.legal_actions();
                        if agent.bid_penalty > 0.0 {
                            bid_action_with_penalty(bn, &bid_obs_buf, legal_mask, agent.bid_penalty)
                        } else {
                            bn.best_action_fast(&bid_obs_buf, legal_mask)
                        }
                    } else {
                        agent.bid_function.bid(&state)
                    }
                } else {
                    agent.bid_function.bid(&state)
                }
            } else {
                match &agent.card_play {
                    CardPlayMethod::NaiveIsMcts => {
                        let config = make_naive_config(agent);
                        if is_ns {
                            ns_naive.search(&state, &config, rng)
                        } else {
                            ew_naive.search(&state, &config, rng)
                        }
                    }
                    CardPlayMethod::SmartIsMcts => {
                        let config = make_smart_config(agent);
                        if is_ns {
                            let idx = if player == 0 { 0 } else { 1 };
                            ns_smart[idx].search(&state, &config, rng)
                        } else {
                            let idx = if player == 1 { 0 } else { 1 };
                            ew_smart[idx].search(&state, &config, rng)
                        }
                    }
                    CardPlayMethod::Oracle => {
                        let config = make_oracle_config(agent);
                        oracle.search(&state, &config, rng)
                    }
                    CardPlayMethod::Dmc(model_idx) => {
                        let net = dmc_nets[*model_idx].as_mut().unwrap();
                        let dmc_w = &models.dmc_weights[*model_idx];
                        if dmc_w.obs_dim == OBS_DIM_TR {
                            // Canonical obs: need canonical mask + physical conversion
                            dmc_obs::write_observation_tr(&mut obs_buf, 0, &state, &tracking);
                            let order = dmc_obs::current_player_order(&state, &tracking);
                            let canonical_mask = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
                            let (canonical_best, _) = net.best_action(&obs_buf, canonical_mask as u32);
                            dmc_obs::card_to_physical(canonical_best, &order)
                        } else {
                            dmc_obs::write_observation(&mut obs_buf, 0, &state, &tracking);
                            let legal_mask = state.legal_actions() as u32;
                            let (action, _) = net.best_action(&obs_buf, legal_mask);
                            action
                        }
                    }
                    CardPlayMethod::IsDd => {
                        let config = make_dd_config(agent);
                        if is_ns {
                            ns_dd.search(&state, &config, rng)
                        } else {
                            ew_dd.search(&state, &config, rng)
                        }
                    }
                    CardPlayMethod::SmartIsDd => {
                        let config = make_dd_config(agent);
                        if is_ns {
                            let idx = if player == 0 { 0 } else { 1 };
                            ns_smart_dd[idx].search(&state, &config, rng)
                        } else {
                            let idx = if player == 1 { 0 } else { 1 };
                            ew_smart_dd[idx].search(&state, &config, rng)
                        }
                    }
                    CardPlayMethod::Heuristic => {
                        heuristic_play_action(&state)
                    }
                    CardPlayMethod::RulePlayer => {
                        rule_play_action(&state)
                    }
                    CardPlayMethod::DmcThenDd { dmc_idx, switch_at } => {
                        let tricks_done = state.tricks_won[0] + state.tricks_won[1];
                        if tricks_done >= *switch_at {
                            // Endgame: use IS-DD (exact solver)
                            let config = make_dd_config(agent);
                            if is_ns {
                                let idx = if player == 0 { 0 } else { 1 };
                                ns_smart_dd[idx].search(&state, &config, rng)
                            } else {
                                let idx = if player == 1 { 0 } else { 1 };
                                ew_smart_dd[idx].search(&state, &config, rng)
                            }
                        } else {
                            // Early game: use DMC (fast NN)
                            let net = dmc_nets[*dmc_idx].as_mut().unwrap();
                            let dmc_w = &models.dmc_weights[*dmc_idx];
                            if dmc_w.obs_dim == OBS_DIM_TR {
                                dmc_obs::write_observation_tr(&mut obs_buf, 0, &state, &tracking);
                                let order = dmc_obs::current_player_order(&state, &tracking);
                                let canonical_mask = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
                                let (canonical_best, _) = net.best_action(&obs_buf, canonical_mask as u32);
                                dmc_obs::card_to_physical(canonical_best, &order)
                            } else {
                                dmc_obs::write_observation(&mut obs_buf, 0, &state, &tracking);
                                let legal_mask = state.legal_actions() as u32;
                                let (action, _) = net.best_action(&obs_buf, legal_mask);
                                action
                            }
                        }
                    }
                    CardPlayMethod::BisDd => {
                        if is_ns {
                            let idx = if player == 0 { 0 } else { 1 };
                            ns_bis_dd[idx].decide(&state)
                        } else {
                            let idx = if player == 1 { 0 } else { 1 };
                            ew_bis_dd[idx].decide(&state)
                        }
                    }
                }
            };

            // Record action on smart searches
            if ns_is_smart {
                ns_smart[0].record_action(&state_before, player, action);
                ns_smart[1].record_action(&state_before, player, action);
            }
            if ew_is_smart {
                ew_smart[0].record_action(&state_before, player, action);
                ew_smart[1].record_action(&state_before, player, action);
            }
            if ns_is_smart_dd {
                ns_smart_dd[0].record_action(&state_before, player, action);
                ns_smart_dd[1].record_action(&state_before, player, action);
            }
            if ew_is_smart_dd {
                ew_smart_dd[0].record_action(&state_before, player, action);
                ew_smart_dd[1].record_action(&state_before, player, action);
            }
            if ns_is_bis_dd {
                ns_bis_dd[0].observe(player, action, &state_before);
                ns_bis_dd[1].observe(player, action, &state_before);
            }
            if ew_is_bis_dd {
                ew_bis_dd[0].observe(player, action, &state_before);
                ew_bis_dd[1].observe(player, action, &state_before);
            }
            tracking.track_action(&state_before, action);
            state.step(action);

            // Apply bid belief NN when bidding just ended
            if state_before.phase == Phase::Bidding && state.phase == Phase::Playing {
                if let Some(ref mut net) = ns_bid_belief_net {
                    if ns_is_bis_dd {
                        ns_bis_dd[0].apply_bid_belief(net, &state, &tracking.bid_history);
                        ns_bis_dd[1].apply_bid_belief(net, &state, &tracking.bid_history);
                    }
                }
                if let Some(ref mut net) = ew_bid_belief_net {
                    if ew_is_bis_dd {
                        ew_bis_dd[0].apply_bid_belief(net, &state, &tracking.bid_history);
                        ew_bis_dd[1].apply_bid_belief(net, &state, &tracking.bid_history);
                    }
                }
            }
        }

        let score = state.deal_score();
        if !(score.scores[0] == 0 && score.scores[1] == 0) {
            ns_cumulative += score.scores[0] as i32;
            ew_cumulative += score.scores[1] as i32;
        }
        dealer = (dealer + 3) % 4;
    }

    let winner = if ns_cumulative >= MATCH_TARGET && ew_cumulative >= MATCH_TARGET {
        if ns_cumulative >= ew_cumulative { 0 } else { 1 }
    } else if ns_cumulative >= MATCH_TARGET {
        0
    } else {
        1
    };

    MatchResult { winner, ns_final: ns_cumulative, ew_final: ew_cumulative }
}

// ══════════════════════════════════════════════════════════════════════
//  Matchup runner (parallel)
// ══════════════════════════════════════════════════════════════════════

#[derive(Default, Clone)]
struct MatchupResult {
    n_matches: u32,
    ns_wins: u32,
    ew_wins: u32,
    total_margin: i64,
}

impl MatchupResult {
    fn merge(&mut self, other: &MatchupResult) {
        self.n_matches += other.n_matches;
        self.ns_wins += other.ns_wins;
        self.ew_wins += other.ew_wins;
        self.total_margin += other.total_margin;
    }
}

fn run_matchup(
    ns_agent: &Agent,
    ew_agent: &Agent,
    n_matches: u32,
    models: &SharedModels,
    n_threads: usize,
    base_seed: u64,
    progress: &AtomicU32,
) -> MatchupResult {
    let per_thread = (n_matches as usize + n_threads - 1) / n_threads;

    let results: Vec<MatchupResult> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..n_threads {
            let start = t * per_thread;
            let end = ((t + 1) * per_thread).min(n_matches as usize);
            if start >= end { continue; }
            let count = end - start;

            handles.push(s.spawn(move || {
                let mut rng = StdRng::seed_from_u64(base_seed.wrapping_add(t as u64 * 7919));
                let mut result = MatchupResult::default();
                for _ in 0..count {
                    let mr = play_match(ns_agent, ew_agent, models, &mut rng);
                    result.n_matches += 1;
                    if mr.winner == 0 { result.ns_wins += 1; } else { result.ew_wins += 1; }
                    result.total_margin += (mr.ns_final - mr.ew_final) as i64;
                    progress.fetch_add(1, Ordering::Relaxed);
                }
                result
            }));
        }
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut combined = MatchupResult::default();
    for r in &results { combined.merge(r); }
    combined
}

/// Run H2H with duplicate matching (both directions, same seeds).
fn run_h2h(
    agent_a: &Agent,
    agent_b: &Agent,
    n_matches: u32,
    models: &SharedModels,
    n_threads: usize,
    base_seed: u64,
    progress: &AtomicU32,
) -> (MatchupResult, MatchupResult) {
    // Direction 1: A as NS, B as EW
    let r1 = run_matchup(agent_a, agent_b, n_matches, models, n_threads, base_seed, progress);
    // Direction 2: B as NS, A as EW (same seed for duplicate matching)
    let r2 = run_matchup(agent_b, agent_a, n_matches, models, n_threads, base_seed.wrapping_add(1_000_000), progress);
    (r1, r2)
}

// ══════════════════════════════════════════════════════════════════════
//  CSV results persistence
// ══════════════════════════════════════════════════════════════════════

fn now_iso() -> String {
    // Simple timestamp without chrono crate
    let dur = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default();
    let secs = dur.as_secs();
    // Approximate: good enough for logging
    let days = secs / 86400;
    let time_of_day = secs % 86400;
    let hours = time_of_day / 3600;
    let minutes = (time_of_day % 3600) / 60;
    let seconds = time_of_day % 60;
    // Days since 1970-01-01
    let (year, month, day) = days_to_ymd(days);
    format!("{:04}-{:02}-{:02}T{:02}:{:02}:{:02}", year, month, day, hours, minutes, seconds)
}

fn days_to_ymd(days: u64) -> (u64, u64, u64) {
    // Simple Gregorian calendar calculation
    let mut y = 1970;
    let mut remaining = days;
    loop {
        let days_in_year = if is_leap(y) { 366 } else { 365 };
        if remaining < days_in_year { break; }
        remaining -= days_in_year;
        y += 1;
    }
    let month_days: [u64; 12] = if is_leap(y) {
        [31, 29, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    } else {
        [31, 28, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    };
    let mut m = 0;
    while m < 12 && remaining >= month_days[m] {
        remaining -= month_days[m];
        m += 1;
    }
    (y, m as u64 + 1, remaining + 1)
}

fn is_leap(y: u64) -> bool {
    (y % 4 == 0 && y % 100 != 0) || y % 400 == 0
}

fn append_csv_result(
    bot_a: &str, bot_b: &str,
    bid_a: &str, play_a: &str,
    bid_b: &str, play_b: &str,
    r1: &MatchupResult, r2: &MatchupResult,
    seed: u64, wall_secs: f64,
) {
    let results_dir = std::path::Path::new(RESULTS_PATH).parent().unwrap();
    let _ = std::fs::create_dir_all(results_dir);

    let write_header = !std::path::Path::new(RESULTS_PATH).exists();
    let mut file = match std::fs::OpenOptions::new().create(true).append(true).open(RESULTS_PATH) {
        Ok(f) => f,
        Err(e) => {
            eprintln!("Cannot write results to {}: {}", RESULTS_PATH, e);
            return;
        }
    };

    use std::io::Write;
    if write_header {
        let _ = writeln!(file, "timestamp,bot_a,bid_a,play_a,bot_b,bid_b,play_b,matches,a_wins,b_wins,win_pct,avg_margin,seed,wall_secs,matches_per_min");
    }

    let ts = now_iso();
    // Aggregate both directions: A wins = r1.ns_wins (A as NS) + r2.ew_wins (A as EW)
    let total_matches = r1.n_matches + r2.n_matches;
    let a_wins = r1.ns_wins + r2.ew_wins;
    let b_wins = r1.ew_wins + r2.ns_wins;
    let a_margin = r1.total_margin - r2.total_margin;
    let win_pct = 100.0 * a_wins as f64 / total_matches as f64;
    let avg_margin = a_margin as f64 / total_matches as f64;
    let matches_per_min = if wall_secs > 0.0 { total_matches as f64 / wall_secs * 60.0 } else { 0.0 };

    let _ = writeln!(file, "{},{},{},{},{},{},{},{},{},{},{:.1},{:+.0},{},{:.1},{:.1}",
        ts, bot_a, bid_a, play_a, bot_b, bid_b, play_b,
        total_matches, a_wins, b_wins, win_pct, avg_margin, seed, wall_secs, matches_per_min);
}

// ══════════════════════════════════════════════════════════════════════
//  Results display
// ══════════════════════════════════════════════════════════════════════

struct AggResult {
    bot_a: String,
    bot_b: String,
    matches: u32,
    a_wins: u32,
    avg_margin: f64,
    wall_secs: f64,
    matches_per_min: f64,
    timestamp: String,
}

fn load_results() -> Vec<AggResult> {
    let content = match std::fs::read_to_string(RESULTS_PATH) {
        Ok(c) => c,
        Err(_) => return Vec::new(),
    };

    let mut results = Vec::new();
    for (i, line) in content.lines().enumerate() {
        if i == 0 { continue; } // skip header
        let cols: Vec<&str> = line.split(',').collect();
        // New format (15 cols): timestamp,bot_a,bid_a,play_a,bot_b,bid_b,play_b,matches,...
        // Old format (10-11 cols): timestamp,bot_a,bot_b,matches,...
        if cols.len() >= 14 {
            // New format
            let wall_secs: f64 = cols[13].parse().unwrap_or(0.0);
            let matches: u32 = cols[7].parse().unwrap_or(0);
            let matches_per_min = if cols.len() > 14 {
                cols[14].parse().unwrap_or(0.0)
            } else if wall_secs > 0.0 {
                matches as f64 / wall_secs * 60.0
            } else { 0.0 };
            results.push(AggResult {
                timestamp: cols[0].to_string(),
                bot_a: cols[1].to_string(),
                bot_b: cols[4].to_string(),
                matches,
                a_wins: cols[8].parse().unwrap_or(0),
                avg_margin: cols[11].parse().unwrap_or(0.0),
                wall_secs,
                matches_per_min,
            });
        } else if cols.len() >= 10 {
            // Old format
            let wall_secs: f64 = cols[9].parse().unwrap_or(0.0);
            let matches: u32 = cols[3].parse().unwrap_or(0);
            let matches_per_min = if cols.len() > 10 {
                cols[10].parse().unwrap_or(0.0)
            } else if wall_secs > 0.0 {
                matches as f64 / wall_secs * 60.0
            } else { 0.0 };
            results.push(AggResult {
                timestamp: cols[0].to_string(),
                bot_a: cols[1].to_string(),
                bot_b: cols[2].to_string(),
                matches,
                a_wins: cols[4].parse().unwrap_or(0),
                avg_margin: cols[7].parse().unwrap_or(0.0),
                wall_secs,
                matches_per_min,
            });
        }
    }
    results
}

fn cmd_results(filter_bot: Option<&str>) {
    let results = load_results();
    if results.is_empty() {
        println!("No results yet. Run some matches first!");
        return;
    }

    // Load bot configs for bid/play labels
    use std::collections::HashMap;
    let bot_configs: HashMap<String, BotConfig> = load_all_bots()
        .into_iter().map(|b| (b.name.clone(), b)).collect();
    let bot_label = |name: &str| -> (String, String) {
        if let Some(cfg) = bot_configs.get(name) {
            (cfg.bid_label(), cfg.play_label())
        } else {
            ("?".into(), "?".into())
        }
    };

    // Build leaderboard: aggregate all H2H into per-bot stats
    let mut wins: HashMap<String, u32> = HashMap::new();
    let mut played: HashMap<String, u32> = HashMap::new();
    let mut margin: HashMap<String, f64> = HashMap::new();

    for r in &results {
        if let Some(f) = filter_bot {
            if r.bot_a != f && r.bot_b != f { continue; }
        }
        *wins.entry(r.bot_a.clone()).or_default() += r.a_wins;
        *wins.entry(r.bot_b.clone()).or_default() += r.matches - r.a_wins;
        *played.entry(r.bot_a.clone()).or_default() += r.matches;
        *played.entry(r.bot_b.clone()).or_default() += r.matches;
        *margin.entry(r.bot_a.clone()).or_default() += r.avg_margin * r.matches as f64;
        *margin.entry(r.bot_b.clone()).or_default() -= r.avg_margin * r.matches as f64;
    }

    let mut ranking: Vec<_> = played.keys().map(|bot| {
        let w = *wins.get(bot).unwrap_or(&0);
        let p = *played.get(bot).unwrap_or(&1);
        let m = *margin.get(bot).unwrap_or(&0.0);
        (bot.clone(), w, p, 100.0 * w as f64 / p as f64, m / p as f64)
    }).collect();
    ranking.sort_by(|a, b| b.3.partial_cmp(&a.3).unwrap());

    println!("=============================================================");
    println!("  ARENA LEADERBOARD");
    if let Some(f) = filter_bot {
        println!("  (filtered: matches involving '{}')", f);
    }
    println!("=============================================================");
    println!("  {:>3} {:<20} {:<22} {:<22} {:>6} {:>8}", "#", "Bot", "Bid", "Play", "Win%", "Margin");
    println!("  {}", "-".repeat(90));

    for (rank, (bot, _w, _p, pct, m)) in ranking.iter().enumerate() {
        let (bid, play) = bot_label(bot);
        println!("  {:>3} {:<20} {:<22} {:<22} {:>5.1}% {:>+7.0}", rank + 1, bot, bid, play, pct, m);
    }

    // Bot speed estimates: for each bot, average matches_per_min across all matchups
    // The slowest bot in a pair determines the speed, so we track per-pair speeds
    // and attribute to both bots.
    let mut bot_speed_sum: HashMap<String, f64> = HashMap::new();
    let mut bot_speed_count: HashMap<String, u32> = HashMap::new();
    for r in &results {
        if r.matches_per_min <= 0.0 { continue; }
        if let Some(f) = filter_bot {
            if r.bot_a != f && r.bot_b != f { continue; }
        }
        // A pair's speed reflects both bots — attribute to each
        *bot_speed_sum.entry(r.bot_a.clone()).or_default() += r.matches_per_min;
        *bot_speed_count.entry(r.bot_a.clone()).or_default() += 1;
        *bot_speed_sum.entry(r.bot_b.clone()).or_default() += r.matches_per_min;
        *bot_speed_count.entry(r.bot_b.clone()).or_default() += 1;
    }

    if !bot_speed_sum.is_empty() {
        println!();
        println!("  BOT SPEEDS (avg matches/min across observed matchups)");
        println!("  {}", "-".repeat(60));
        // Sort by speed descending
        let mut speeds: Vec<_> = bot_speed_sum.keys().map(|bot| {
            let avg = bot_speed_sum[bot] / *bot_speed_count.get(bot).unwrap_or(&1) as f64;
            (bot.clone(), avg)
        }).collect();
        speeds.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());
        for (bot, avg_speed) in &speeds {
            let time_200 = if *avg_speed > 0.0 { 400.0 / avg_speed } else { f64::INFINITY };
            println!("  {:<20} {:>6.0} matches/min   (~{:.0}min for 200-match H2H)",
                bot, avg_speed, time_200);
        }
    }

    // Show recent H2H results
    println!();
    println!("  RECENT MATCHES");
    println!("  {}", "-".repeat(60));
    let start = if results.len() > 20 { results.len() - 20 } else { 0 };
    for r in &results[start..] {
        if let Some(f) = filter_bot {
            if r.bot_a != f && r.bot_b != f { continue; }
        }
        let pct = 100.0 * r.a_wins as f64 / r.matches as f64;
        let speed_str = if r.matches_per_min > 0.0 {
            format!(" {:.0}m/min", r.matches_per_min)
        } else {
            String::new()
        };
        println!("  {} vs {}: {:.1}% ({}/{}) margin {:+.0}{}  [{}]",
            r.bot_a, r.bot_b, pct, r.a_wins, r.matches, r.avg_margin, speed_str, r.timestamp);
    }
}

// ══════════════════════════════════════════════════════════════════════
//  CLI
// ══════════════════════════════════════════════════════════════════════

fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(4)
}

fn print_usage() {
    eprintln!("Usage:");
    eprintln!("  arena list                                 List available bots");
    eprintln!("  arena h2h <bot_a> <bot_b> [--matches N]   Head-to-head comparison");
    eprintln!("  arena round-robin [--matches N] [--bots a,b,c]  Round-robin tournament");
    eprintln!("  arena results [--bot name]                 Show results leaderboard");
    eprintln!("  arena trace <bot_a> <bot_b> [--deals N]   Play same deals with both bots, show diffs");
    eprintln!();
    eprintln!("Options:");
    eprintln!("  --matches N     Matches per direction (default 100)");
    eprintln!("  --threads N     Thread count (default auto)");
    eprintln!("  --seed N        Base RNG seed (default 42)");
    eprintln!("  --bots a,b,c    Only include these bots in round-robin");
    eprintln!("  --no-save       Don't persist results to CSV (h2h only)");
    eprintln!("  --deals N       Number of deals to trace (default 50)");
}

fn main() {
    let args: Vec<String> = std::env::args().collect();

    if args.len() < 2 {
        print_usage();
        std::process::exit(1);
    }

    let subcmd = args[1].as_str();

    match subcmd {
        "list" => cmd_list(),
        "h2h" => cmd_h2h(&args[2..]),
        "round-robin" => cmd_round_robin(&args[2..]),
        "results" => {
            let filter = parse_flag(&args[2..], "--bot");
            cmd_results(filter.as_deref());
        }
        "trace" => cmd_trace(&args[2..]),
        "--help" | "-h" | "help" => print_usage(),
        other => {
            eprintln!("Unknown command: {}", other);
            print_usage();
            std::process::exit(1);
        }
    }
}

fn parse_flag(args: &[String], flag: &str) -> Option<String> {
    for i in 0..args.len() {
        if args[i] == flag && i + 1 < args.len() {
            return Some(args[i + 1].clone());
        }
    }
    None
}

fn parse_flag_u32(args: &[String], flag: &str, default: u32) -> u32 {
    parse_flag(args, flag).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn parse_flag_u64(args: &[String], flag: &str, default: u64) -> u64 {
    parse_flag(args, flag).and_then(|s| s.parse().ok()).unwrap_or(default)
}

fn cmd_list() {
    let bots = load_all_bots();
    if bots.is_empty() {
        println!("No bots found in {}/", BOTS_DIR);
        return;
    }
    println!("Available bots ({}):", bots.len());
    println!("  {:<20} {:<22} {:<22}", "Name", "Bid", "Play");
    println!("  {}", "-".repeat(66));
    for b in &bots {
        println!("  {:<20} {:<22} {:<22}", b.name, b.bid_label(), b.play_label());
    }
}

fn has_flag(args: &[String], flag: &str) -> bool {
    args.iter().any(|a| a == flag)
}

fn cmd_h2h(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: arena h2h <bot_a> <bot_b> [--matches N] [--threads N] [--seed N] [--no-save]");
        std::process::exit(1);
    }

    let bot_a_name = &args[0];
    let bot_b_name = &args[1];
    let rest = &args[2..];
    let n_matches = parse_flag_u32(rest, "--matches", 100);
    let n_threads = parse_flag_u32(rest, "--threads", default_threads() as u32) as usize;
    let seed = parse_flag_u64(rest, "--seed", 42);
    let no_save = has_flag(rest, "--no-save");

    let all_bots = load_all_bots();
    let cfg_a = all_bots.iter().find(|b| b.name == *bot_a_name)
        .unwrap_or_else(|| { eprintln!("Bot '{}' not found in {}/", bot_a_name, BOTS_DIR); std::process::exit(1); });
    let cfg_b = all_bots.iter().find(|b| b.name == *bot_b_name)
        .unwrap_or_else(|| { eprintln!("Bot '{}' not found in {}/", bot_b_name, BOTS_DIR); std::process::exit(1); });

    let mut models = SharedModels { dmc_weights: Vec::new(), bid_weights: Vec::new() };
    let agent_a = build_agent(cfg_a, &mut models).unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
    let agent_b = build_agent(cfg_b, &mut models).unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });

    println!("=============================================================");
    println!("  ARENA H2H: {} vs {}", agent_a.name, agent_b.name);
    println!("  {} matches/direction ({}x2 total), {} threads, seed {}",
        n_matches, n_matches, n_threads, seed);
    println!("=============================================================");
    println!();

    let progress = Arc::new(AtomicU32::new(0));
    let total = n_matches * 2;

    // Progress monitor
    let progress_clone = progress.clone();
    let monitor = std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(5));
            let done = progress_clone.load(Ordering::Relaxed);
            if done >= total { break; }
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed * 60.0;
            let eta = if rate > 0.0 { (total - done) as f64 / rate * 60.0 } else { 0.0 };
            eprint!("\r  Progress: {}/{} matches ({:.0}%), {:.0}/min, ETA {:.0}s   ",
                done, total, 100.0 * done as f64 / total as f64, rate, eta);
        }
    });

    let start = Instant::now();
    let (r1, r2) = run_h2h(&agent_a, &agent_b, n_matches, &models, n_threads, seed, &progress);
    let elapsed = start.elapsed();

    progress.store(total, Ordering::Relaxed);
    let _ = monitor.join();
    eprintln!();

    // Aggregate
    let a_wins = r1.ns_wins + r2.ew_wins;
    let b_wins = r1.ew_wins + r2.ns_wins;
    let total_matches = r1.n_matches + r2.n_matches;
    let a_margin = r1.total_margin - r2.total_margin;
    let a_pct = 100.0 * a_wins as f64 / total_matches as f64;
    let avg_margin = a_margin as f64 / total_matches as f64;

    println!("  RESULT: {} {:.1}% vs {} {:.1}%",
        agent_a.name, a_pct, agent_b.name, 100.0 - a_pct);
    println!("    A: {} | {}     B: {} | {}", agent_a.bid_label, agent_a.play_label, agent_b.bid_label, agent_b.play_label);
    println!("  Wins: {} {} — {} {}", agent_a.name, a_wins, agent_b.name, b_wins);
    println!("  Avg margin: {:+.0} (from {}'s perspective)", avg_margin, agent_a.name);
    println!("  Dir 1 ({}=NS): {}-{}", agent_a.name, r1.ns_wins, r1.ew_wins);
    println!("  Dir 2 ({}=NS): {}-{}", agent_b.name, r2.ns_wins, r2.ew_wins);
    println!("  Wall: {:.1}s ({:.1} matches/min)", elapsed.as_secs_f64(),
        total_matches as f64 / elapsed.as_secs_f64() * 60.0);

    // Persist
    if !no_save {
        append_csv_result(&agent_a.name, &agent_b.name,
            &agent_a.bid_label, &agent_a.play_label,
            &agent_b.bid_label, &agent_b.play_label,
            &r1, &r2, seed, elapsed.as_secs_f64());
        println!();
        println!("  Results saved to {}", RESULTS_PATH);
    } else {
        println!();
        println!("  (--no-save: results NOT written to CSV)");
    }
}

fn cmd_round_robin(args: &[String]) {
    let n_matches = parse_flag_u32(args, "--matches", 100);
    let n_threads = parse_flag_u32(args, "--threads", default_threads() as u32) as usize;
    let seed = parse_flag_u64(args, "--seed", 42);
    let bot_filter = parse_flag(args, "--bots");

    let all_bots = load_all_bots();
    let bots: Vec<&BotConfig> = if let Some(ref filter) = bot_filter {
        let names: Vec<&str> = filter.split(',').collect();
        all_bots.iter().filter(|b| names.contains(&b.name.as_str())).collect()
    } else {
        all_bots.iter().collect()
    };

    if bots.len() < 2 {
        eprintln!("Need at least 2 bots for round-robin. Found {} in {}/", bots.len(), BOTS_DIR);
        std::process::exit(1);
    }

    let mut models = SharedModels { dmc_weights: Vec::new(), bid_weights: Vec::new() };
    let agents: Vec<Agent> = bots.iter().map(|cfg| {
        build_agent(cfg, &mut models).unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); })
    }).collect();

    let n = agents.len();
    let total_matchups = n * (n - 1) / 2;
    let total_matches = total_matchups as u32 * n_matches * 2; // both directions

    println!("=============================================================");
    println!("  ARENA ROUND-ROBIN — First to {}", MATCH_TARGET);
    println!("  {} bots, {} matches/direction, {} threads", n, n_matches, n_threads);
    println!("  {} matchups, {} total matches", total_matchups, total_matches);
    println!("=============================================================");
    println!();

    for (i, a) in agents.iter().enumerate() {
        println!("  [{:>2}] {:<20} {} | {}", i, a.name, a.bid_label, a.play_label);
    }
    println!();

    let progress = Arc::new(AtomicU32::new(0));
    let start = Instant::now();

    // Progress monitor
    let progress_clone = progress.clone();
    let monitor = std::thread::spawn(move || {
        let start = Instant::now();
        loop {
            std::thread::sleep(std::time::Duration::from_secs(10));
            let done = progress_clone.load(Ordering::Relaxed);
            if done >= total_matches { break; }
            let elapsed = start.elapsed().as_secs_f64();
            let rate = done as f64 / elapsed * 60.0;
            let eta = if rate > 0.0 { (total_matches - done) as f64 / rate * 60.0 } else { 0.0 };
            eprint!("\r  Progress: {}/{} ({:.0}%), {:.0}/min, ETA {:.0}s   ",
                done, total_matches, 100.0 * done as f64 / total_matches as f64, rate, eta);
        }
    });

    // Win/margin matrices
    let mut win_matrix = vec![vec![0u32; n]; n];
    let mut margin_matrix = vec![vec![0i64; n]; n];
    let mut matches_matrix = vec![vec![0u32; n]; n];

    for i in 0..n {
        for j in (i + 1)..n {
            let pair_seed = seed.wrapping_add((i * 1000 + j * 100) as u64);
            let pair_start = Instant::now();
            let (r1, r2) = run_h2h(&agents[i], &agents[j], n_matches, &models, n_threads, pair_seed, &progress);
            let pair_secs = pair_start.elapsed().as_secs_f64();

            // Persist each pair with its own wall time
            append_csv_result(&agents[i].name, &agents[j].name,
                &agents[i].bid_label, &agents[i].play_label,
                &agents[j].bid_label, &agents[j].play_label,
                &r1, &r2, pair_seed, pair_secs);

            // Aggregate into matrices
            win_matrix[i][j] += r1.ns_wins + r2.ew_wins;
            win_matrix[j][i] += r1.ew_wins + r2.ns_wins;
            margin_matrix[i][j] += r1.total_margin - r2.total_margin;
            margin_matrix[j][i] += r2.total_margin - r1.total_margin;
            let total = r1.n_matches + r2.n_matches;
            matches_matrix[i][j] += total;
            matches_matrix[j][i] += total;
        }
    }

    progress.store(total_matches, Ordering::Relaxed);
    let _ = monitor.join();
    let elapsed = start.elapsed();
    eprintln!();

    // Print win matrix
    println!("=============================================================");
    println!("  WIN MATRIX (row win% vs column)");
    println!("=============================================================");
    print!("  {:>16}", "");
    for a in &agents { print!("  {:>8}", &a.name[..a.name.len().min(8)]); }
    println!("    TOTAL");
    println!("  {}", "-".repeat(16 + 10 * (n + 1) + 4));

    let mut total_wins = vec![0u32; n];
    let mut total_played = vec![0u32; n];

    for i in 0..n {
        print!("  {:>16}", agents[i].name);
        let mut row_wins = 0u32;
        let mut row_played = 0u32;
        for j in 0..n {
            if i == j {
                print!("       - ");
            } else {
                let wins = win_matrix[i][j];
                let played = matches_matrix[i][j];
                let pct = 100.0 * wins as f64 / played as f64;
                print!("   {:5.1}% ", pct);
                row_wins += wins;
                row_played += played;
            }
        }
        let total_pct = 100.0 * row_wins as f64 / row_played as f64;
        print!("   {:5.1}%", total_pct);
        println!();
        total_wins[i] = row_wins;
        total_played[i] = row_played;
    }

    // Rankings
    println!();
    println!("=============================================================");
    println!("  RANKINGS");
    println!("=============================================================");

    let mut ranking: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, 100.0 * total_wins[i] as f64 / total_played[i] as f64))
        .collect();
    ranking.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

    for (rank, (idx, pct)) in ranking.iter().enumerate() {
        let avg_m: f64 = {
            let total_m: i64 = (0..n).filter(|&j| j != *idx).map(|j| margin_matrix[*idx][j]).sum();
            let total_p: u32 = (0..n).filter(|&j| j != *idx).map(|j| matches_matrix[*idx][j]).sum();
            total_m as f64 / total_p as f64
        };
        println!("  {:>2}. {:<20} {:<22} {:<22} win {:5.1}%  margin {:+5.0}",
            rank + 1, agents[*idx].name, agents[*idx].bid_label, agents[*idx].play_label, pct, avg_m);
    }

    println!();
    println!("  Wall: {:.1}s ({} matches, {:.1}/min)",
        elapsed.as_secs_f64(), total_matches, total_matches as f64 / elapsed.as_secs_f64() * 60.0);
    println!("  Results saved to {}", RESULTS_PATH);
}

// ══════════════════════════════════════════════════════════════════════
//  Trace: play same deals with two bots, compare decisions
// ══════════════════════════════════════════════════════════════════════

use colver_core::card::{card_name, cardset_str, card_suit, CardSet};

const SUIT_SYMS: [&str; 4] = ["S", "H", "D", "C"];

fn bid_action_str(action: u8) -> String {
    match action {
        0 => "PASS".to_string(),
        41 => "COINCHE".to_string(),
        42 => "SURCOINCHE".to_string(),
        1..=36 => {
            let idx = action - 1;
            let value_idx = idx / 4;
            let suit_idx = idx % 4;
            let value = (value_idx as u16 + 8) * 10;
            format!("{}{}", value, SUIT_SYMS[suit_idx as usize])
        }
        37..=40 => {
            let suit_idx = action - 37;
            format!("Capot{}", SUIT_SYMS[suit_idx as usize])
        }
        _ => format!("?{}", action),
    }
}

const SEAT_NAMES: [&str; 4] = ["N", "E", "S", "W"];

#[allow(dead_code)]
struct DealTrace {
    hands: [CardSet; 4],
    dealer: u8,
    bids: Vec<(u8, u8)>,      // (player, action)
    plays: Vec<(u8, u8)>,     // (player, card)
    trick_leads: Vec<u8>,     // trick lead player for each trick
    contract_str: String,
    ns_score: i16,
    ew_score: i16,
    void_deal: bool,
}

/// Play a single deal with a given agent pair, recording trace.
fn play_deal_traced(
    state_orig: &GameState,
    ns_agent: &Agent,
    ew_agent: &Agent,
    models: &SharedModels,
    dmc_nets: &mut Vec<Option<DmcNet>>,
    bid_nets: &mut Vec<Option<BidNet>>,
    obs_buf: &mut Vec<f32>,
    bid_obs_buf: &mut Vec<f32>,
) -> DealTrace {
    let mut state = *state_orig;
    let mut tracking = EnvTracking::new();
    tracking.reset(state.dealer);

    let mut trace = DealTrace {
        hands: state.hands,
        dealer: state.dealer,
        bids: Vec::new(),
        plays: Vec::new(),
        trick_leads: Vec::new(),
        contract_str: String::new(),
        ns_score: 0,
        ew_score: 0,
        void_deal: false,
    };

    while !state.is_terminal() {
        let player = state.current_player();
        let is_ns = player == 0 || player == 2;
        let agent = if is_ns { ns_agent } else { ew_agent };

        let action = if state.phase == Phase::Bidding {
            if let Some(idx) = agent.bid_weights_idx {
                if let Some(ref mut bn) = bid_nets[idx] {
                    bid_obs::write_bid_observation(bid_obs_buf, 0, &state, &tracking.bid_history);
                    let legal_mask = state.legal_actions();
                    if agent.bid_penalty > 0.0 {
                        bid_action_with_penalty(bn, bid_obs_buf, legal_mask, agent.bid_penalty)
                    } else {
                        bn.best_action_fast(bid_obs_buf, legal_mask)
                    }
                } else {
                    agent.bid_function.bid(&state)
                }
            } else {
                agent.bid_function.bid(&state)
            }
        } else {
            match &agent.card_play {
                CardPlayMethod::Heuristic => heuristic_play_action(&state),
                CardPlayMethod::RulePlayer => rule_play_action(&state),
                CardPlayMethod::Dmc(model_idx) => {
                    let net = dmc_nets[*model_idx].as_mut().unwrap();
                    let dmc_w = &models.dmc_weights[*model_idx];
                    if dmc_w.obs_dim == OBS_DIM_TR {
                        dmc_obs::write_observation_tr(obs_buf, 0, &state, &tracking);
                        let order = dmc_obs::current_player_order(&state, &tracking);
                        let canonical_mask = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
                        let (canonical_best, _) = net.best_action(obs_buf, canonical_mask as u32);
                        dmc_obs::card_to_physical(canonical_best, &order)
                    } else {
                        dmc_obs::write_observation(obs_buf, 0, &state, &tracking);
                        let legal_mask = state.legal_actions() as u32;
                        let (action, _) = net.best_action(obs_buf, legal_mask);
                        action
                    }
                }
                _ => {
                    // Fallback for search-based methods: use heuristic for trace speed
                    heuristic_play_action(&state)
                }
            }
        };

        if state.phase == Phase::Bidding {
            trace.bids.push((player, action));
        } else {
            if state.trick_count == 0 {
                trace.trick_leads.push(player);
            }
            trace.plays.push((player, action));
        }

        tracking.track_action(&state, action);
        state.step(action);
    }

    let score = state.deal_score();
    trace.ns_score = score.scores[0];
    trace.ew_score = score.scores[1];
    trace.void_deal = score.scores[0] == 0 && score.scores[1] == 0;

    if state.contract.value > 0 {
        let val = (state.contract.value as u16 + 8) * 10;
        let coinche_str = match state.contract.coinche {
            1 => "x",
            2 => "xx",
            _ => "",
        };
        trace.contract_str = format!("{}{}{} by {}",
            val, SUIT_SYMS[state.contract.trump as usize], coinche_str,
            if state.contract.team == 0 { "NS" } else { "EW" });
    } else {
        trace.contract_str = "VOID".to_string();
    }

    trace
}

fn cmd_trace(args: &[String]) {
    if args.len() < 2 {
        eprintln!("Usage: arena trace <bot_a> <bot_b> [--deals N] [--seed N]");
        std::process::exit(1);
    }

    let bot_a_name = &args[0];
    let bot_b_name = &args[1];
    let rest = &args[2..];
    let n_deals = parse_flag_u32(rest, "--deals", 50);
    let seed = parse_flag_u64(rest, "--seed", 42);

    let all_bots = load_all_bots();
    let cfg_a = all_bots.iter().find(|b| b.name == *bot_a_name)
        .unwrap_or_else(|| { eprintln!("Bot '{}' not found", bot_a_name); std::process::exit(1); });
    let cfg_b = all_bots.iter().find(|b| b.name == *bot_b_name)
        .unwrap_or_else(|| { eprintln!("Bot '{}' not found", bot_b_name); std::process::exit(1); });

    let mut models = SharedModels { dmc_weights: Vec::new(), bid_weights: Vec::new() };
    let agent_a = build_agent(cfg_a, &mut models).unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });
    let agent_b = build_agent(cfg_b, &mut models).unwrap_or_else(|e| { eprintln!("Error: {}", e); std::process::exit(1); });

    println!("═══════════════════════════════════════════════════════════════");
    println!("  TRACE: {} vs {}", agent_a.name, agent_b.name);
    println!("  {} deals, seed {}", n_deals, seed);
    println!("═══════════════════════════════════════════════════════════════");
    println!();

    let mut rng = StdRng::seed_from_u64(seed);

    // Create two full sets of model instances (one per direction)
    let make_dmc_nets = |models: &SharedModels| -> Vec<Option<DmcNet>> {
        models.dmc_weights.iter().map(|w| Some(w.make_net())).collect()
    };
    let make_bid_nets = |models: &SharedModels| -> Vec<Option<BidNet>> {
        models.bid_weights.iter().map(|w| Some(w.make_net())).collect()
    };
    let mut dmc_nets_a = make_dmc_nets(&models);
    let mut dmc_nets_b = make_dmc_nets(&models);
    let mut bid_nets_a = make_bid_nets(&models);
    let mut bid_nets_b = make_bid_nets(&models);

    let max_obs_dim = models.dmc_weights.iter().map(|w| w.obs_dim).max().unwrap_or(OBS_DIM);
    let mut obs_buf = vec![0.0f32; max_obs_dim];
    let max_bid_obs = models.bid_weights.iter().map(|w| w.obs_dim).max().unwrap_or(BID_OBS_DIM);
    let mut bid_obs_buf = vec![0.0f32; max_bid_obs];

    // Stats
    let mut a_better = 0u32;
    let mut b_better = 0u32;
    let mut same = 0u32;
    let mut void_deals = 0u32;
    let mut a_total_ns: i32 = 0;
    let mut b_total_ns: i32 = 0;

    // Categorize differences
    let mut bid_diffs = 0u32;
    let mut play_diffs = 0u32;

    for deal_idx in 0..n_deals {
        let dealer = (deal_idx % 4) as u8;
        let state = GameState::deal_random(dealer, &mut rng);

        // Both bots play as NS, opponent is the other bot playing as EW
        // Config A: A=NS, B=EW
        let trace_a = play_deal_traced(&state, &agent_a, &agent_b, &models,
            &mut dmc_nets_a, &mut bid_nets_a, &mut obs_buf, &mut bid_obs_buf);
        // Config B: B=NS, A=EW
        let trace_b = play_deal_traced(&state, &agent_b, &agent_a, &models,
            &mut dmc_nets_b, &mut bid_nets_b, &mut obs_buf, &mut bid_obs_buf);

        if trace_a.void_deal && trace_b.void_deal {
            void_deals += 1;
            continue;
        }

        let a_net = trace_a.ns_score - trace_a.ew_score;
        let b_net = trace_b.ns_score - trace_b.ew_score;
        a_total_ns += a_net as i32;
        b_total_ns += b_net as i32;

        // Check if bids differ
        let bids_same = trace_a.bids.len() == trace_b.bids.len()
            && trace_a.bids.iter().zip(&trace_b.bids).all(|(a, b)| a.1 == b.1);
        if !bids_same { bid_diffs += 1; }

        // Check if plays differ
        let plays_same = trace_a.plays.len() == trace_b.plays.len()
            && trace_a.plays.iter().zip(&trace_b.plays).all(|(a, b)| a.1 == b.1);
        if !plays_same { play_diffs += 1; }

        let score_diff = a_net - b_net;

        if score_diff > 0 {
            a_better += 1;
        } else if score_diff < 0 {
            b_better += 1;
        } else {
            same += 1;
            continue;
        }

        // Print interesting deals (score diff >= 50 points)
        if score_diff.abs() >= 50 {
            let winner_name = if score_diff > 0 { &agent_a.name } else { &agent_b.name };
            println!("─────────────────────────────────────────────────────────");
            println!("Deal #{} (dealer={}) — {} better by {} pts",
                deal_idx, SEAT_NAMES[dealer as usize], winner_name, score_diff.abs());
            println!();

            // Hands
            for p in 0..4 {
                println!("  {} {}: {}", SEAT_NAMES[p],
                    if p % 2 == 0 { "(NS)" } else { "(EW)" },
                    cardset_str(state.hands[p]));
            }
            println!();

            // Bidding comparison
            println!("  Bidding ({}):", agent_a.name);
            print!("    ");
            for (player, action) in &trace_a.bids {
                print!("{}:{} ", SEAT_NAMES[*player as usize], bid_action_str(*action));
            }
            println!(" → {}", trace_a.contract_str);

            println!("  Bidding ({}):", agent_b.name);
            print!("    ");
            for (player, action) in &trace_b.bids {
                print!("{}:{} ", SEAT_NAMES[*player as usize], bid_action_str(*action));
            }
            println!(" → {}", trace_b.contract_str);
            println!();

            // Play comparison (trick by trick)
            let trump_a = if !trace_a.void_deal {
                format!(" ({})", trace_a.contract_str)
            } else { String::new() };
            println!("  {} as NS{}: score NS={} EW={}",
                agent_a.name, trump_a, trace_a.ns_score, trace_a.ew_score);
            for (i, chunk) in trace_a.plays.chunks(4).enumerate() {
                let lead = if i < trace_a.trick_leads.len() {
                    SEAT_NAMES[trace_a.trick_leads[i] as usize]
                } else { "?" };
                print!("    T{} (lead {}): ", i + 1, lead);
                for (player, card) in chunk {
                    print!("{}={} ", SEAT_NAMES[*player as usize], card_name(*card));
                }
                println!();
            }

            println!("  {} as NS: score NS={} EW={}",
                agent_b.name, trace_b.ns_score, trace_b.ew_score);
            for (i, chunk) in trace_b.plays.chunks(4).enumerate() {
                let lead = if i < trace_b.trick_leads.len() {
                    SEAT_NAMES[trace_b.trick_leads[i] as usize]
                } else { "?" };
                print!("    T{} (lead {}): ", i + 1, lead);
                for (player, card) in chunk {
                    print!("{}={} ", SEAT_NAMES[*player as usize], card_name(*card));
                }
                println!();
            }
            println!();
        }
    }

    // Summary
    let played = n_deals - void_deals;
    println!("═══════════════════════════════════════════════════════════════");
    println!("  SUMMARY ({} deals played, {} void)", played, void_deals);
    println!("  {} better in {} deals ({:.1}%)",
        agent_a.name, a_better, 100.0 * a_better as f64 / played as f64);
    println!("  {} better in {} deals ({:.1}%)",
        agent_b.name, b_better, 100.0 * b_better as f64 / played as f64);
    println!("  Same score: {} deals ({:.1}%)",
        same, 100.0 * same as f64 / played as f64);
    println!("  Avg NS score: {} {:.1}, {} {:.1}",
        agent_a.name, a_total_ns as f64 / played as f64,
        agent_b.name, b_total_ns as f64 / played as f64);
    println!("  Bid differences: {} ({:.0}%)", bid_diffs, 100.0 * bid_diffs as f64 / played as f64);
    println!("  Play differences: {} ({:.0}%)", play_diffs, 100.0 * play_diffs as f64 / played as f64);
    println!("═══════════════════════════════════════════════════════════════");
}
