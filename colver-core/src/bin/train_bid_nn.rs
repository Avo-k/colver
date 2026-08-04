/// Pure Rust NN bidding training binary using Candle + Dueling DQN.
///
/// Trains a bidding Q-network using DD oracle for reward computation.
/// Each env DD-solves all 4 suits at deal start (~52ms), then runs
/// bidding in microseconds. Reward = (my_team_score - opp_score) / 500.
///
/// Opponent diversity: configurable mix of self-play, improved_v2,
/// aggressive, conservative, and random bidding opponents. The non-self-play
/// ratio anneals linearly from --diversity-start to --diversity-end.
///
/// Usage:
///   cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- --num-envs 64 --steps 5000000

use std::time::Instant;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use candle_core::Device;

use colver_core::bid_candle::BiddingTrainer;
use colver_core::bid::bumblebid_candle::BumblebidTrainer;
use colver_core::bid_eval;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{
    self, BID_OBS_DIM, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2,
    BID_OBS_DIM_SCORE_AWARE_V3, BID_OBS_DIM_V7,
};
use colver_core::suit_perm;
use colver_core::bid_train_env::{BidReplayBuffer, DealPool, RewardMode, VecBidEnv};
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use colver_core::rollout;
use colver_core::state::{GameState, Phase};

const NUM_ACTIONS: usize = 43;

/// Wrapper enum so the training loop works with both MLP and transformer.
enum Trainer {
    Mlp(BiddingTrainer),
    Transformer(BumblebidTrainer),
}

impl Trainer {
    fn act(
        &self,
        obs: &candle_core::Tensor,
        mask: &candle_core::Tensor,
        epsilon: f32,
        rng: &mut impl Rng,
    ) -> candle_core::Result<Vec<u8>> {
        match self {
            Trainer::Mlp(t) => t.net.act(obs, mask, epsilon, rng),
            Trainer::Transformer(t) => t.net.act(obs, mask, epsilon, rng),
        }
    }

    fn train_step(
        &mut self,
        obs: &[f32],
        masks: &[f32],
        actions: &[u8],
        returns: &[f32],
        weights: &[f32],
    ) -> candle_core::Result<(f32, Vec<f32>)> {
        match self {
            Trainer::Mlp(t) => t.train_step(obs, masks, actions, returns, weights),
            Trainer::Transformer(t) => t.train_step(obs, masks, actions, returns, weights),
        }
    }

    fn save_checkpoint(&self, path: &str) -> candle_core::Result<()> {
        match self {
            Trainer::Mlp(t) => t.save_checkpoint(path),
            Trainer::Transformer(t) => t.save_checkpoint(path),
        }
    }

    fn load_checkpoint(&mut self, path: &str) -> candle_core::Result<()> {
        match self {
            Trainer::Mlp(t) => t.load_checkpoint(path),
            Trainer::Transformer(t) => t.load_checkpoint(path),
        }
    }

    fn device(&self) -> &Device {
        match self {
            Trainer::Mlp(t) => t.device(),
            Trainer::Transformer(t) => t.device(),
        }
    }

    fn set_lr(&mut self, lr: f64) {
        match self {
            Trainer::Mlp(t) => t.set_lr(lr),
            Trainer::Transformer(t) => t.set_lr(lr),
        }
    }

    fn snapshot_weights(&self) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error>> {
        match self {
            Trainer::Mlp(t) => t.snapshot_weights(),
            Trainer::Transformer(t) => t.snapshot_weights().map_err(|e| e.into()),
        }
    }

    fn set_ema_tau(&mut self, tau: f32) {
        if let Trainer::Mlp(t) = self {
            t.set_ema_tau(tau);
        }
    }

    fn update_ema(&mut self) {
        if let Trainer::Mlp(t) = self {
            t.update_ema();
        }
    }

    /// Returns EMA-shadow snapshot if EMA is enabled (MLP only), else current weights.
    fn eval_snapshot(&self) -> std::result::Result<Vec<f32>, Box<dyn std::error::Error>> {
        if let Trainer::Mlp(t) = self {
            if let Some(ema) = t.ema_snapshot() {
                return Ok(ema.to_vec());
            }
        }
        self.snapshot_weights()
    }
}

/// Write a flat f32 vector to disk as raw little-endian bytes (matches `BiddingTrainer::export_binary` format).
fn save_bin_from_floats(path: &str, floats: &[f32]) -> std::io::Result<()> {
    let mut bytes: Vec<u8> = Vec::with_capacity(floats.len() * 4);
    for f in floats {
        bytes.extend_from_slice(&f.to_le_bytes());
    }
    std::fs::write(path, bytes)
}

/// The single place the `--sa-features-*` flags become a width. Three call sites read it
/// — the model, the replay buffer and the env — and a disagreement between them is
/// silent: every layout is a valid buffer, just not the one the net was built for.
fn score_aware_dim(args: &Args) -> usize {
    if args.sa_features_v7 {
        BID_OBS_DIM_V7
    } else if args.sa_features_v3 {
        BID_OBS_DIM_SCORE_AWARE_V3
    } else if args.sa_features_v2 {
        BID_OBS_DIM_SCORE_AWARE_V2
    } else {
        BID_OBS_DIM_SCORE_AWARE
    }
}

/// Opponent bidding strategy for non-self-play environments.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum OpponentMode {
    /// Both teams use the NN (pure self-play).
    SelfPlay,
    /// Opponent uses improved_v2 (the default balanced bidder).
    ImprovedV2,
    /// Opponent uses aggressive bidding (low thresholds, high caps, no quality gate).
    Aggressive,
    /// Opponent uses conservative bidding (high thresholds, low caps).
    Conservative,
    /// Opponent bids randomly among legal actions.
    Random,
}

/// Within non-self-play, relative weights for each opponent type.
/// These stay constant; only the total non-self-play ratio anneals.
const OPP_WEIGHT_IMPROVED: f32 = 8.0;
const OPP_WEIGHT_AGGRESSIVE: f32 = 8.0;
const OPP_WEIGHT_CONSERVATIVE: f32 = 8.0;
const OPP_WEIGHT_RANDOM: f32 = 16.0;
const OPP_WEIGHT_TOTAL: f32 =
    OPP_WEIGHT_IMPROVED + OPP_WEIGHT_AGGRESSIVE + OPP_WEIGHT_CONSERVATIVE + OPP_WEIGHT_RANDOM;

/// Pick an opponent mode based on current training progress.
fn pick_opponent_mode(
    step: usize,
    total_steps: usize,
    diversity_start: f32,
    diversity_end: f32,
    rng: &mut impl Rng,
) -> OpponentMode {
    let progress = (step as f32 / total_steps as f32).min(1.0);
    let non_self_play = diversity_start + (diversity_end - diversity_start) * progress;

    let r: f32 = rng.gen();
    if r >= non_self_play {
        return OpponentMode::SelfPlay;
    }

    // Within non-self-play, distribute by relative weights
    let inner_r: f32 = rng.gen::<f32>() * OPP_WEIGHT_TOTAL;
    if inner_r < OPP_WEIGHT_IMPROVED {
        OpponentMode::ImprovedV2
    } else if inner_r < OPP_WEIGHT_IMPROVED + OPP_WEIGHT_AGGRESSIVE {
        OpponentMode::Aggressive
    } else if inner_r < OPP_WEIGHT_IMPROVED + OPP_WEIGHT_AGGRESSIVE + OPP_WEIGHT_CONSERVATIVE {
        OpponentMode::Conservative
    } else {
        OpponentMode::Random
    }
}

/// Get action from a fixed bidding strategy.
fn opponent_action(state: &GameState, mode: OpponentMode, rng: &mut impl Rng) -> u8 {
    match mode {
        OpponentMode::SelfPlay => unreachable!("SelfPlay has no fixed action"),
        OpponentMode::ImprovedV2 => bid_eval::improved_v2_bid(state),
        OpponentMode::Aggressive => {
            bid_eval::parametric_bid(state, &bid_eval::BidParams::very_aggressive())
        }
        OpponentMode::Conservative => {
            bid_eval::parametric_bid(state, &bid_eval::BidParams::ultra_conservative())
        }
        OpponentMode::Random => {
            let mask = state.legal_actions();
            let count = mask.count_ones();
            let idx = rng.gen_range(0..count);
            rollout::select_nth_bit(mask, idx)
        }
    }
}

#[derive(Parser)]
#[command(
    name = "train_bid_nn",
    about = "NN bidding training with DD oracle + Dueling DQN"
)]
struct Args {
    #[arg(long, default_value_t = 64)]
    num_envs: usize,
    #[arg(long, default_value_t = 5_000_000)]
    steps: usize,
    #[arg(long, default_value_t = 512)]
    batch_size: usize,
    #[arg(long, default_value_t = 3e-4)]
    lr: f64,
    #[arg(long, default_value_t = 256)]
    hidden: usize,
    #[arg(long, default_value_t = 2)]
    layers: usize,
    #[arg(long, default_value_t = 0.3)]
    eps_start: f32,
    #[arg(long, default_value_t = 0.02)]
    eps_end: f32,
    #[arg(long, default_value_t = 3_000_000)]
    eps_decay_steps: usize,
    #[arg(long, default_value_t = 500_000)]
    buffer_size: usize,
    #[arg(long, default_value_t = 5_000)]
    min_buffer: usize,
    #[arg(long, default_value_t = 4)]
    train_freq: usize,
    #[arg(long, default_value_t = 50_000)]
    eval_freq: usize,
    #[arg(long, default_value_t = 200)]
    eval_matches: usize,
    #[arg(long, default_value_t = 200_000)]
    save_freq: usize,
    #[arg(long, default_value = "models")]
    save_dir: String,
    #[arg(long)]
    resume: Option<String>,
    #[arg(long, default_value_t = 42)]
    seed: u64,
    #[arg(long, default_value_t = 0.6)]
    per_alpha: f64,
    #[arg(long, default_value_t = 0.4)]
    per_beta_start: f64,
    #[arg(long, default_value_t = 1.0)]
    per_beta_end: f64,
    /// Initial non-self-play ratio (0.0-1.0). Default 0.40.
    #[arg(long, default_value_t = 0.40)]
    diversity_start: f32,
    /// Final non-self-play ratio (0.0-1.0). Default 0.15.
    #[arg(long, default_value_t = 0.15)]
    diversity_end: f32,
    /// Number of deals to pre-solve with DD.
    #[arg(long, default_value_t = 1_000_000)]
    pool_size: usize,
    /// Path to save/load deal pool (auto-generates if missing).
    #[arg(long, default_value = "data/deals/archive/dd_2.5M.bin")]
    pool_file: String,
    /// Reward mode: "dd", "real", "blend:0.7" (alpha=DD weight), or "curriculum:0.95:0.3" (DD weight start:end).
    #[arg(long, default_value = "dd")]
    reward: String,
    /// Score file(s) to load (COLVSC01 format). Can be repeated. Last one loaded becomes the active score layer.
    #[arg(long)]
    scores: Vec<String>,
    /// Use Bumblebid transformer instead of MLP.
    #[arg(long)]
    transformer: bool,
    /// Transformer model dimension (d_model). Only used with --transformer.
    #[arg(long, default_value_t = 128)]
    d_model: usize,
    /// Transformer number of heads. Only used with --transformer.
    #[arg(long, default_value_t = 4)]
    n_heads: usize,
    /// Enable score-aware training: 110-dim obs with match scores, Δ-winprob reward.
    #[arg(long)]
    score_aware: bool,
    /// Reward scale for score-aware mode (default 3.0).
    #[arg(long, default_value_t = 3.0)]
    sa_scale: f32,
    /// Score distribution CSV for score-aware mode (ns,ew,winner).
    /// If provided, 80% of match scores are sampled from this file, 20% uniform.
    #[arg(long)]
    score_dist: Option<String>,
    /// Fraction of score samples drawn uniformly (rest from --score-dist). Default 0.2.
    #[arg(long, default_value_t = 0.2)]
    sa_uniform_ratio: f32,
    /// Use v2 score features (5 derived features → 113-dim obs). Requires --score-aware.
    #[arg(long)]
    sa_features_v2: bool,
    /// Use v3 score features (v2 + 4 self-belote bits → 117-dim obs). Requires --score-aware.
    /// Implies --sa-features-v2 semantics for the first 5 extras; overrides if both set.
    #[arg(long)]
    sa_features_v3: bool,
    /// Use v7 features (v3 + 4 per-suit trump scores + 2 auction-conditioned
    /// reductions → 123-dim obs). Requires --score-aware.
    ///
    /// The per-suit scores restate the hand; what is new is `opp_best_other_ts` —
    /// the best of them *excluding* the suit an opponent is contracting in. See
    /// `bid_obs::write_bid_observation_v7`.
    #[arg(long)]
    sa_features_v7: bool,
    /// Train on the **canonical** suit ordering (v7). Suits are sorted by
    /// (length, rank pattern) descending, ties broken by the auction, so two hands
    /// identical up to renaming become one training sample instead of 24.
    ///
    /// Turns suit augmentation off — the canonical form is already invariant, so
    /// permuting a sample would only decorrelate its obs from its stored action.
    ///
    /// A net trained this way is **not** interchangeable with a physical-order one of
    /// the same width: arena bots must declare `canonical = true` under `[bid]`.
    #[arg(long)]
    canonical: bool,
    /// Enable match simulation: cumulative scores + dealer rotation across deals
    /// until one team reaches 2000. Replaces the random score injection at reset.
    /// Requires --score-aware. Builds a dealer index on the pool at startup.
    #[arg(long)]
    match_sim: bool,
    /// Clip Δ-winprob reward (post scale) to [-clip, +clip]. 0 disables.
    #[arg(long, default_value_t = 0.0)]
    reward_clip: f32,
    /// Polyak EMA τ (per-step) for exported weights / eval. 0 disables.
    #[arg(long, default_value_t = 0.0)]
    ema_tau: f32,
    /// Final LR for cosine decay over training. <= 0 keeps lr constant.
    #[arg(long, default_value_t = 0.0)]
    lr_end: f64,
    /// Play model for eval matches (default: models/play_v2/play_final.bin).
    #[arg(long, default_value = "models/play_v2/play_final.bin")]
    eval_play_model: String,
    /// Baseline bid model for eval matches (default: models/bid_v3_max_20M/bid_nn_final.bin).
    #[arg(long, default_value = "models/bid_v3_max_20M/bid_nn_final.bin")]
    eval_baseline_bid: String,
    /// Baseline bid hidden size (default 512).
    #[arg(long, default_value_t = 512)]
    eval_baseline_hidden: usize,
    /// Diagnostic : journaliser dans ce fichier JSON l'histogramme des contrats
    /// atteints — rang DD de l'atout contracté, camp preneur, valeur.
    ///
    /// Sert à dimensionner une couche de scores : la reward ne lit qu'une case
    /// `[atout]` par épisode, donc ce que la boucle *consulte* décide de ce qu'il
    /// faut étiqueter. N'affecte en rien l'entraînement.
    #[arg(long)]
    log_contract_ranks: Option<String>,
}

/// Écrire l'histogramme des contrats atteints. Réécrit à chaque intervalle de log,
/// pour qu'une interruption laisse un fichier exploitable.
fn write_contract_stats(path: &str, st: &colver_core::bid_train_env::ContractStats, step: usize) {
    let total: u64 = st.by_rank.iter().sum();
    let json = format!(
        "{{\n  \"steps\": {},\n  \"contracts\": {},\n  \"voids\": {},\n  \
         \"by_rank\": [{}],\n  \"by_rank_taker\": [{}],\n  \"by_rank_matched\": [{}],\n  \
         \"taker_pts_bucket\": [{}],\n  \"by_value_80_to_160_then_capot\": [{}],\n  \
         \"capot_bids\": {},\n  \"capot_bids_sound\": {}\n}}\n",
        step,
        total,
        st.voids,
        st.by_rank.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        st.by_rank_taker.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        st.by_rank_matched.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        st.taker_pts_bucket.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        st.by_value.iter().map(|v| v.to_string()).collect::<Vec<_>>().join(", "),
        st.capot_bids,
        st.capot_bids_sound,
    );
    if let Err(e) = std::fs::write(path, json) {
        eprintln!("write_contract_stats({}): {}", path, e);
    }
}

/// Evaluate training model vs baseline in full 2000-point matches.
///
/// Training model: score-aware (110-dim) or standard (108-dim) bid NN + DouDou50 play.
/// Baseline: nn_v3_max_20M bid NN + DouDou50 play (same play model for both).
/// Each match alternates sides (training=NS for even, training=EW for odd).
///
/// Returns (training_wins, total_matches, avg_margin).
fn evaluate_full_matches(
    trainer: &Trainer,
    hidden: usize,
    layers: usize,
    obs_dim: usize,
    num_matches: usize,
    play_model_path: &str,
    baseline_bid_path: &str,
    baseline_bid_hidden: usize,
    canonical: bool,
) -> (usize, usize, f64) {
    let weights = match trainer.eval_snapshot() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("Failed to snapshot weights: {}", e);
            return (0, 0, 0.0);
        }
    };
    let mut train_bid = match BidNet::from_floats_with_layers(&weights, hidden, obs_dim, true, layers) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to load eval net: {}", e);
            return (0, 0, 0.0);
        }
    };
    let mut base_bid = match BidNet::load_with_hidden(baseline_bid_path, baseline_bid_hidden) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to load baseline bid: {}", e);
            return (0, 0, 0.0);
        }
    };
    let mut dmc = match DmcNet::load(play_model_path) {
        Ok(n) => n,
        Err(e) => {
            eprintln!("Failed to load play model: {}", e);
            return (0, 0, 0.0);
        }
    };
    dmc.set_residual(true);

    let score_aware = obs_dim > BID_OBS_DIM;
    let base_obs_dim = base_bid.obs_dim();
    let base_score_aware = base_obs_dim > BID_OBS_DIM;
    let mut rng = StdRng::seed_from_u64(12345);
    let mut train_wins = 0usize;
    let mut total_margin = 0i64;

    let mut bid_obs_buf = vec![0.0f32; obs_dim];
    let mut base_obs_buf = vec![0.0f32; base_obs_dim];
    let mut play_obs_buf = vec![0.0f32; OBS_DIM_TR];

    for match_idx in 0..num_matches {
        let train_team: u8 = (match_idx % 2) as u8;
        let mut ns_cum: i32 = 0;
        let mut ew_cum: i32 = 0;
        let mut dealer: u8 = rng.gen_range(0..4);

        while ns_cum < 2000 && ew_cum < 2000 {
            let mut state = GameState::deal_random(dealer, &mut rng);
            let mut tracking = EnvTracking::new();
            tracking.reset(dealer);

            // Bidding
            while state.phase == Phase::Bidding {
                let player = state.current_player();
                let team = GameState::player_team(player);

                let (my, opp) = if team == 0 { (ns_cum, ew_cum) } else { (ew_cum, ns_cum) };
                let action = if team == train_team {
                    let (my, opp) = if score_aware { (my, opp) } else { (0, 0) };
                    if canonical {
                        // Same round trip as inference: obs and mask into canonical
                        // space, chosen action back out. Skipping it here would make
                        // the eval report a plausible, meaningless win rate.
                        let order = bid_obs::write_bid_observation_canonical(
                            &mut bid_obs_buf, 0, &state, &tracking.bid_history, my, opp, obs_dim,
                        );
                        let perm = suit_perm::perm_from_order(&order);
                        let legal = suit_perm::permute_bid_mask_u64(state.legal_actions(), &perm);
                        let a = train_bid.best_action_fast(&bid_obs_buf, legal);
                        suit_perm::permute_bid_action(a, &order)
                    } else {
                        bid_obs::write_bid_observation_dim(
                            &mut bid_obs_buf, 0, &state, &tracking.bid_history, my, opp, obs_dim,
                        );
                        train_bid.best_action_fast(&bid_obs_buf, state.legal_actions())
                    }
                } else {
                    let (my, opp) = if base_score_aware { (my, opp) } else { (0, 0) };
                    bid_obs::write_bid_observation_dim(
                        &mut base_obs_buf, 0, &state, &tracking.bid_history, my, opp, base_obs_dim,
                    );
                    base_bid.best_action_fast(&base_obs_buf, state.legal_actions())
                };

                tracking.track_action(&state, action);
                state.step(action);
            }

            // Play (DMC canonical for both teams)
            while !state.is_terminal() {
                dmc_obs::write_observation_tr(&mut play_obs_buf, 0, &state, &tracking);
                let order = dmc_obs::current_player_order(&state, &tracking);
                let canonical_mask = dmc_obs::cardset_to_canonical(state.legal_actions() as u32, &order);
                let (canonical_best, _) = dmc.best_action(&play_obs_buf, canonical_mask as u32);
                let action = dmc_obs::card_to_physical(canonical_best, &order);
                tracking.track_action(&state, action);
                state.step(action);
            }

            let score = state.deal_score();
            if score.scores[0] != 0 || score.scores[1] != 0 {
                ns_cum += score.scores[0] as i32;
                ew_cum += score.scores[1] as i32;
            }
            dealer = (dealer + 1) % 4;
        }

        let winner = if ns_cum >= 2000 && ew_cum >= 2000 {
            if ns_cum >= ew_cum { 0u8 } else { 1 }
        } else if ns_cum >= 2000 { 0 } else { 1 };

        if winner == train_team {
            train_wins += 1;
        }
        let train_final = if train_team == 0 { ns_cum } else { ew_cum };
        let base_final = if train_team == 0 { ew_cum } else { ns_cum };
        total_margin += (train_final - base_final) as i64;
    }

    let avg_margin = total_margin as f64 / num_matches as f64;
    (train_wins, num_matches, avg_margin)
}

fn main() {
    let args = Args::parse();

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).expect("CUDA device creation failed")
    } else {
        eprintln!("WARNING: CUDA not available, using CPU");
        Device::Cpu
    };
    let device_name = match &device {
        Device::Cpu => "CPU",
        Device::Cuda(_) => "CUDA",
        _ => "Other",
    };

    println!("=== NN Bidding Training (DD Oracle) ===");
    println!("Device: {}", device_name);
    println!(
        "Envs: {}, Steps: {}, LR: {}, Hidden: {}, Layers: {}",
        args.num_envs, args.steps, args.lr, args.hidden, args.layers
    );
    println!(
        "Training: batch={}, freq={}, buffer={}",
        args.batch_size, args.train_freq, args.buffer_size
    );
    println!(
        "PER: alpha={}, beta={}->{}",
        args.per_alpha, args.per_beta_start, args.per_beta_end
    );
    println!(
        "Epsilon: {}->{}  over {} steps",
        args.eps_start, args.eps_end, args.eps_decay_steps
    );
    println!(
        "Diversity: {:.0}% -> {:.0}% non-self-play (improved_v2 + aggressive + conservative + random)",
        args.diversity_start * 100.0,
        args.diversity_end * 100.0
    );

    // Initialize trainer
    let mut trainer = if args.transformer {
        println!(
            "Model: Bumblebid transformer d={} L={} H={}",
            args.d_model, args.layers, args.n_heads
        );
        Trainer::Transformer(
            BumblebidTrainer::new(args.d_model, args.layers, args.n_heads, args.lr, 0.0, device)
                .expect("Failed to create Bumblebid trainer"),
        )
    } else {
        let obs_dim = if args.score_aware {
            score_aware_dim(&args)
        } else {
            BID_OBS_DIM
        };
        println!("Model: Dueling MLP H={} L={} obs_dim={}", args.hidden, args.layers, obs_dim);
        Trainer::Mlp(
            BiddingTrainer::with_layers_and_obs(args.layers, args.hidden, obs_dim, args.lr, 0.0, device)
                .expect("Failed to create trainer"),
        )
    };

    if args.sa_features_v2 && !args.score_aware {
        panic!("--sa-features-v2 requires --score-aware");
    }
    if args.sa_features_v3 && !args.score_aware {
        panic!("--sa-features-v3 requires --score-aware");
    }
    if args.sa_features_v7 && !args.score_aware {
        panic!("--sa-features-v7 requires --score-aware");
    }
    if args.canonical && !args.score_aware {
        // The canonical path only exists on the score-aware transition type; a plain
        // 108-dim run would silently store physical samples under a canonical banner.
        panic!("--canonical requires --score-aware");
    }
    if args.ema_tau > 0.0 {
        trainer.set_ema_tau(args.ema_tau);
        println!("EMA tracking enabled (τ={})", args.ema_tau);
    }
    if args.reward_clip > 0.0 {
        println!("Reward clip: ±{}", args.reward_clip);
    }
    if args.lr_end > 0.0 {
        println!("LR cosine decay: {} → {}", args.lr, args.lr_end);
    }

    if let Some(ref path) = args.resume {
        trainer
            .load_checkpoint(path)
            .expect("Failed to load checkpoint");
        println!("Resumed from {}", path);
    }

    // Initialize replay buffer (obs dim matches model: 117 v3, 113 v2, 110 v1, 108 standard)
    let replay_obs_dim = if args.score_aware {
        score_aware_dim(&args)
    } else {
        BID_OBS_DIM
    };
    let mut replay_buffer = BidReplayBuffer::with_obs_dim(args.buffer_size, args.per_alpha, replay_obs_dim);

    // Parse reward mode (and optional curriculum schedule)
    let mut curriculum: Option<(f32, f32)> = None; // (dd_start, dd_end)
    let reward_mode = if args.reward == "dd" {
        RewardMode::DdOnly
    } else if args.reward == "real" {
        RewardMode::RealOnly
    } else if args.reward.starts_with("blend:") {
        let alpha: f32 = args.reward[6..].parse().expect("Bad blend alpha, use e.g. blend:0.7");
        RewardMode::Blend(alpha)
    } else if args.reward.starts_with("curriculum:") {
        let parts: Vec<&str> = args.reward[11..].split(':').collect();
        assert!(parts.len() == 2, "Use curriculum:0.95:0.3 (dd_start:dd_end)");
        let dd_start: f32 = parts[0].parse().expect("Bad curriculum dd_start");
        let dd_end: f32 = parts[1].parse().expect("Bad curriculum dd_end");
        curriculum = Some((dd_start, dd_end));
        println!("Curriculum: DD weight {:.0}% -> {:.0}%", dd_start * 100.0, dd_end * 100.0);
        RewardMode::Blend(dd_start)
    } else {
        panic!("Unknown reward mode '{}'. Use: dd, real, blend:0.7, curriculum:0.95:0.3", args.reward);
    };
    println!("Reward mode: {:?}", reward_mode);

    // Phase 1: Load deal pool
    println!(
        "\n--- Phase 1: Deal pool (file: {}) ---",
        args.pool_file
    );
    // Ensure data directory exists
    if let Some(parent) = std::path::Path::new(&args.pool_file).parent() {
        std::fs::create_dir_all(parent).ok();
    }
    let pool_start = Instant::now();
    let mut pool = if args.pool_file.contains("enriched") || matches!(reward_mode, RewardMode::RealOnly | RewardMode::Blend(_)) {
        // Try enriched format first, fall back to standard
        match DealPool::load_enriched(&args.pool_file) {
            Ok(p) => p,
            Err(_) => DealPool::load_or_generate(&args.pool_file, args.pool_size, args.seed + 100),
        }
    } else {
        DealPool::load_or_generate(&args.pool_file, args.pool_size, args.seed + 100)
    };

    // Load score files (COLVSC01) if provided
    let mut last_score_name = None;
    for score_path in &args.scores {
        pool.load_scores(score_path).unwrap_or_else(|e| {
            panic!("Failed to load score file {}: {}", score_path, e);
        });
        // Track last loaded name for activation
        last_score_name = pool.score_layer_names().last().map(|s| s.to_string());
    }
    // Activate the last loaded score layer for real_pts
    if let Some(name) = &last_score_name {
        pool.select_score_layer(Some(name));
    }

    println!(
        "Deal pool ready: {} deals in {:.1}s (score layers: {:?})",
        pool.len(),
        pool_start.elapsed().as_secs_f64(),
        pool.score_layer_names(),
    );

    if args.match_sim {
        if !args.score_aware {
            panic!("--match-sim requires --score-aware");
        }
        pool.build_dealer_index();
    }

    // Phase 2: Initialize envs from pool (instant)
    println!("\n--- Phase 2: Training ---");
    if args.score_aware {
        // `score_aware_dim`, pas une constante : la ligne affichait 110 quel que soit le
        // jeu de features, donc les journaux de v5 comme de v6 annoncent une largeur
        // qu'ils n'ont pas utilisée. Le seul endroit où la largeur est visible à l'œil
        // doit être celui que le modèle reçoit.
        println!(
            "Score-aware mode: obs_dim={}, Δ-winprob reward (scale={})",
            score_aware_dim(&args),
            args.sa_scale
        );
    }
    let mut vec_env = VecBidEnv::new_with_pool_and_mode(args.num_envs, args.seed, &pool, reward_mode);

    let mut rng = StdRng::seed_from_u64(args.seed + 1);

    // In score-aware mode, randomize match scores for each env
    let score_pool: Vec<(i32, i32)> = if args.score_aware {
        let pool_data = if let Some(ref path) = args.score_dist {
            let pts = colver_core::bid_train_env::load_score_points(path)
                .expect("Failed to load score distribution CSV");
            println!("Score distribution: {} points from {}, uniform ratio={:.0}%",
                pts.len(), path, args.sa_uniform_ratio * 100.0);
            pts
        } else {
            println!("Score distribution: uniform [0, 2000)");
            Vec::new()
        };
        let sa_dim = score_aware_dim(&args);
        vec_env.enable_score_aware_with_dim(sa_dim, &pool_data, args.sa_uniform_ratio);
        if args.reward_clip > 0.0 {
            vec_env.set_reward_clip(Some(args.reward_clip));
        }
        if args.match_sim {
            vec_env.set_match_sim(true);
            println!("Match simulation enabled: cumulative scores + dealer rotation, reset @ 2000.");
        }
        if args.canonical {
            vec_env.set_canonical(true);
            println!(
                "Canonical suit ordering: obs, mask and stored actions in canonical space; \
                 suit augmentation disabled."
            );
        }
        if let Some(ref p) = args.log_contract_ranks {
            vec_env.contract_stats = Some(Default::default());
            println!("Contract-rank histogram → {}", p);
        }
        pool_data
    } else {
        Vec::new()
    };
    let obs_dim = vec_env.obs_dim;

    // Per-env opponent tracking
    let n = args.num_envs;
    let mut opp_modes: Vec<OpponentMode> = (0..n)
        .map(|_| {
            pick_opponent_mode(
                0,
                args.steps,
                args.diversity_start,
                args.diversity_end,
                &mut rng,
            )
        })
        .collect();
    let mut nn_teams: Vec<u8> = (0..n).map(|_| rng.gen_range(0..2u8)).collect();

    // Stats
    let mut total_transitions = 0usize;
    let mut total_episodes = 0usize;
    let mut total_loss = 0.0f64;
    let mut loss_count = 0usize;
    let mut total_void_deals = 0usize;
    let mut self_play_episodes = 0usize;
    let mut diverse_episodes = 0usize;
    let step_start = Instant::now();
    let mut last_log_time = Instant::now();
    let mut last_log_step = 0usize;

    println!(
        "\n{:>10} | {:>5} | {:>5} | {:>7} | {:>8} | {:>8} | {:>6} | {:>5} | {:>7}",
        "Step", "Eps", "Beta", "Buffer", "Loss", "Episodes", "Voids", "Div%", "Steps/s"
    );
    println!("{}", "-".repeat(88));

    for step in 0..args.steps {
        // Epsilon schedule
        let progress = ((step as f64) / args.eps_decay_steps as f64).min(1.0);
        let eps = args.eps_start + (args.eps_end - args.eps_start) * progress as f32;

        // Beta schedule
        let beta_progress = ((step as f64) / args.steps as f64).min(1.0);
        let beta =
            args.per_beta_start + (args.per_beta_end - args.per_beta_start) * beta_progress;

        // Curriculum: update DD blend ratio over training
        if let Some((dd_start, dd_end)) = curriculum {
            if step % 1000 == 0 {
                let t = (step as f32 / args.steps as f32).min(1.0);
                let alpha = dd_start + (dd_end - dd_start) * t;
                vec_env.set_reward_mode(RewardMode::Blend(alpha));
            }
        }

        // LR cosine decay (per-step, lazy: only update when value changes appreciably)
        if args.lr_end > 0.0 && step % 1000 == 0 {
            let t = (step as f64 / args.steps as f64).min(1.0);
            let cos_factor = 0.5 * (1.0 + (std::f64::consts::PI * t).cos());
            let lr_now = args.lr_end + (args.lr - args.lr_end) * cos_factor;
            trainer.set_lr(lr_now);
        }

        // --- Collect actions for all envs ---
        // GPU batch forward pass for all envs (even opponent-controlled ones, for simplicity)
        let obs_flat = &vec_env.obs_buf;
        let mask_flat = &vec_env.mask_buf;

        let nn_actions = match trainer.act(
            &candle_core::Tensor::from_slice(obs_flat, (n, obs_dim), trainer.device())
                .unwrap(),
            &candle_core::Tensor::from_slice(mask_flat, (n, NUM_ACTIONS), trainer.device())
                .unwrap(),
            eps,
            &mut rng,
        ) {
            Ok(a) => a,
            Err(e) => {
                eprintln!("GPU forward failed: {}", e);
                (0..n).map(|i| vec_env.random_action(i)).collect()
            }
        };

        // --- Determine final actions: NN or opponent ---
        // The net answered in its own action space. Under --canonical that is not the
        // physical one, so every NN action goes back through the env's ordering before
        // anything else looks at it; opponent strategies below already speak physical.
        let mut actions = nn_actions;
        for i in 0..n {
            actions[i] = vec_env.to_physical(i, actions[i]);
        }
        for i in 0..n {
            if opp_modes[i] != OpponentMode::SelfPlay {
                let player = vec_env.envs[i].state.current_player();
                let team = GameState::player_team(player);
                if team != nn_teams[i] {
                    // Opponent's turn — use fixed strategy
                    actions[i] = opponent_action(&vec_env.envs[i].state, opp_modes[i], &mut rng);
                }
            }
        }

        // --- Step all envs, flush completed episodes ---
        for i in 0..n {
            let is_self_play = opp_modes[i] == OpponentMode::SelfPlay;

            if args.score_aware {
                if let Some(transitions) = vec_env.step_env_pooled_score_aware(i, actions[i], &pool, args.sa_scale, &score_pool, args.sa_uniform_ratio) {
                    let is_void = transitions.iter().all(|(_, _, _, r, _)| *r == 0.0);
                    if is_void { total_void_deals += 1; }

                    for (obs, mask, action, reward, team) in &transitions {
                        if is_self_play || *team == nn_teams[i] {
                            replay_buffer.push(obs.as_slice(), mask.as_slice(), *action, *reward);
                            total_transitions += 1;
                        }
                    }

                    if is_self_play { self_play_episodes += 1; } else { diverse_episodes += 1; }
                    total_episodes += 1;
                    opp_modes[i] = pick_opponent_mode(step, args.steps, args.diversity_start, args.diversity_end, &mut rng);
                    nn_teams[i] = rng.gen_range(0..2u8);
                }
            } else {
                if let Some(transitions) = vec_env.step_env_pooled(i, actions[i], &pool) {
                    let is_void = transitions.iter().all(|(_, _, _, r, _)| *r == 0.0);
                    if is_void { total_void_deals += 1; }

                    for (obs, mask, action, reward, team) in &transitions {
                        if is_self_play || *team == nn_teams[i] {
                            replay_buffer.push(obs, mask, *action, *reward);
                            total_transitions += 1;
                        }
                    }

                    if is_self_play { self_play_episodes += 1; } else { diverse_episodes += 1; }
                    total_episodes += 1;
                    opp_modes[i] = pick_opponent_mode(step, args.steps, args.diversity_start, args.diversity_end, &mut rng);
                    nn_teams[i] = rng.gen_range(0..2u8);
                }
            }
        }

        // --- Train ---
        if replay_buffer.size() >= args.min_buffer && step % args.train_freq == 0 {
            let mut sample = replay_buffer.sample(args.batch_size, beta, &mut rng);

            // 24× suit augmentation: random permutation per sample.
            // Canonical samples are already invariant — the 24 relabelings collapsed
            // to one at collection time — so permuting here would add no information
            // and would move the obs away from the canonical form the net expects.
            if !args.canonical {
                suit_perm::augment_bid_batch_with_obs_dim(
                    &mut sample.obs_data,
                    &mut sample.mask_data,
                    &mut sample.actions,
                    obs_dim,
                    &mut rng,
                );
            }

            match trainer.train_step(
                &sample.obs_data,
                &sample.mask_data,
                &sample.actions,
                &sample.returns,
                &sample.weights,
            ) {
                Ok((loss, td_errors)) => {
                    replay_buffer.update_priorities(&sample.indices, &td_errors);
                    total_loss += loss as f64;
                    loss_count += 1;
                    trainer.update_ema();
                }
                Err(e) => {
                    eprintln!("Training step failed: {}", e);
                }
            }
        }

        // --- Logging ---
        if (step + 1) % 1_000 == 0 {
            if let (Some(p), Some(st)) = (&args.log_contract_ranks, &vec_env.contract_stats) {
                write_contract_stats(p, st, step + 1);
            }
            let elapsed = last_log_time.elapsed().as_secs_f64();
            let steps_done = step + 1 - last_log_step;
            let sps = steps_done as f64 / elapsed.max(1e-6);
            let avg_loss = if loss_count > 0 {
                total_loss / loss_count as f64
            } else {
                0.0
            };
            let div_pct = if total_episodes > 0 {
                diverse_episodes as f64 / total_episodes as f64 * 100.0
            } else {
                0.0
            };
            println!(
                "{:>10} | {:>5.3} | {:>5.3} | {:>7} | {:>8.5} | {:>8} | {:>6} | {:>4.0}% | {:>7.0}",
                step + 1,
                eps,
                beta,
                replay_buffer.size(),
                avg_loss,
                total_episodes,
                total_void_deals,
                div_pct,
                sps
            );
            total_loss = 0.0;
            loss_count = 0;
            last_log_time = Instant::now();
            last_log_step = step + 1;
        }

        // --- Evaluate ---
        if (step + 1) % args.eval_freq == 0 {
            let eval_start = Instant::now();
            let (wins, total, margin) = evaluate_full_matches(
                &trainer,
                args.hidden,
                args.layers,
                obs_dim,
                args.eval_matches,
                &args.eval_play_model,
                &args.eval_baseline_bid,
                args.eval_baseline_hidden,
                args.canonical,
            );
            let wr = if total > 0 {
                wins as f64 / total as f64
            } else {
                0.0
            };
            println!(
                "  [EVAL] vs baseline (200 full matches): {:.1}% ({}/{}) margin={:+.0}  ({:.0}s)",
                wr * 100.0,
                wins,
                total,
                margin,
                eval_start.elapsed().as_secs_f64()
            );
        }

        // --- Save checkpoint ---
        if (step + 1) % args.save_freq == 0 {
            std::fs::create_dir_all(&args.save_dir).ok();
            let st_path = format!("{}/bid_nn_{}.safetensors", args.save_dir, step + 1);
            let bin_path = format!("{}/bid_nn_{}.bin", args.save_dir, step + 1);
            if let Err(e) = trainer.save_checkpoint(&st_path) {
                eprintln!("Failed to save safetensors: {}", e);
            }
            if matches!(trainer, Trainer::Mlp(_)) {
                // Use EMA snapshot for the .bin export when EMA is enabled (matches eval).
                match trainer.eval_snapshot() {
                    Ok(snap) => {
                        if let Err(e) = save_bin_from_floats(&bin_path, &snap) {
                            eprintln!("Failed to write {}: {}", bin_path, e);
                        }
                    }
                    Err(e) => eprintln!("Failed to take eval snapshot: {}", e),
                }
            }
            // Also save as latest
            let latest_st = format!("{}/bid_nn_latest.safetensors", args.save_dir);
            trainer.save_checkpoint(&latest_st).ok();
            if matches!(trainer, Trainer::Mlp(_)) {
                let latest_bin = format!("{}/bid_nn_latest.bin", args.save_dir);
                if let Ok(snap) = trainer.eval_snapshot() {
                    save_bin_from_floats(&latest_bin, &snap).ok();
                }
            }
            println!("  [SAVE] {}", st_path);
        }
    }

    // Final eval and save
    println!("\n--- Final Evaluation ---");
    let eval_start = Instant::now();
    let (wins, total, margin) = evaluate_full_matches(
        &trainer,
        args.hidden,
        args.layers,
        obs_dim,
        args.eval_matches,
        &args.eval_play_model,
        &args.eval_baseline_bid,
        args.eval_baseline_hidden,
        args.canonical,
    );
    let wr = if total > 0 {
        wins as f64 / total as f64
    } else {
        0.0
    };
    if total > 0 {
        println!(
            "vs baseline (full matches): {:.1}% ({}/{}) margin={:+.0}",
            wr * 100.0, wins, total, margin
        );
    } else {
        println!("(eval returned no results)");
    }
    println!(
        "Self-play: {} episodes, Diverse: {} episodes ({:.0}%)",
        self_play_episodes,
        diverse_episodes,
        if total_episodes > 0 {
            diverse_episodes as f64 / total_episodes as f64 * 100.0
        } else {
            0.0
        }
    );
    println!("Eval time: {:.0}s", eval_start.elapsed().as_secs_f64());
    println!(
        "Total training time: {:.0}s",
        step_start.elapsed().as_secs_f64()
    );
    println!(
        "Total transitions: {}, episodes: {}, void deals: {}",
        total_transitions, total_episodes, total_void_deals
    );

    std::fs::create_dir_all(&args.save_dir).ok();
    let final_st = format!("{}/bid_nn_final.safetensors", args.save_dir);
    let final_bin = format!("{}/bid_nn_final.bin", args.save_dir);
    trainer.save_checkpoint(&final_st).ok();
    if matches!(trainer, Trainer::Mlp(_)) {
        if let Ok(snap) = trainer.eval_snapshot() {
            save_bin_from_floats(&final_bin, &snap).ok();
        }
    }
    println!("Saved final model to {}", final_st);
}
