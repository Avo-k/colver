/// Joint Bid+Play training: trains both a bid NN and a play NN from scratch
/// using full-game self-play. No DD oracle — both NNs learn from actual outcomes.
///
/// Key design:
/// - Two GPU networks: BiddingQNet (114→256²→43) + DuelingQNet (415→1024³→32)
/// - Two PER buffers: bid (FlexReplayBuffer) + play (PrioritizedReplayBuffer)
/// - Phased warm-up: heuristic-dominant → gradual handover → joint self-play
/// - Suit augmentation (24×) applied at sample time for both networks
/// - Normalized score reward: (my_score - opp_score) / 500

use std::io::Write;
use std::time::Instant;

use clap::Parser;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use candle_core::Device;

use colver_core::belief_net::BeliefNet;
use colver_core::belief_obs::{self as belief_obs_mod, BELIEF_OBS_DIM};
use colver_core::bid_candle::BiddingTrainer;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::BID_OBS_DIM;
use colver_core::dmc_candle::{DuelingTrainer, PoolNet};
use colver_core::dmc_eval::{self, EvalResult};
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self as obs, OBS_DIM_TR};
use colver_core::dmc_replay::FlexReplayBuffer;
use colver_core::joint_env::{VecJointEnv, BELIEF_PRED_DIM};
use colver_core::state::{GameState, Phase};
use colver_core::suit_perm;

const PLAY_ACTIONS: usize = 32;
const BID_ACTIONS: usize = 43;
const BID_MASK_DIM: usize = 43;

#[derive(Clone, PartialEq)]
enum TrainMode {
    Joint,
    PlayOnly,
    BidOnly,
}

impl std::str::FromStr for TrainMode {
    type Err = String;
    fn from_str(s: &str) -> Result<Self, String> {
        match s {
            "joint" => Ok(TrainMode::Joint),
            "play-only" | "play" => Ok(TrainMode::PlayOnly),
            "bid-only" | "bid" => Ok(TrainMode::BidOnly),
            _ => Err(format!("unknown mode '{}', expected: joint, play-only, bid-only", s)),
        }
    }
}

impl std::fmt::Display for TrainMode {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        match self {
            TrainMode::Joint => write!(f, "joint"),
            TrainMode::PlayOnly => write!(f, "play-only"),
            TrainMode::BidOnly => write!(f, "bid-only"),
        }
    }
}

#[derive(Parser)]
#[command(name = "train_joint", about = "Joint bid+play training (triforge)")]
struct Args {
    /// Training mode: joint (both), play-only (freeze bid), bid-only (freeze play)
    #[arg(long, default_value = "joint")]
    mode: TrainMode,

    #[arg(long, default_value_t = 256)]
    num_envs: usize,
    #[arg(long, default_value_t = 35_000_000)]
    steps: usize,

    // --- Network sizes ---
    #[arg(long, default_value_t = 1024)]
    play_hidden: usize,
    #[arg(long, default_value_t = 512)]
    bid_hidden: usize,
    #[arg(long, default_value_t = 3)]
    bid_layers: usize,

    // --- Learning rates ---
    #[arg(long, default_value_t = 3e-4)]
    play_lr: f64,
    #[arg(long, default_value_t = 3e-4)]
    bid_lr: f64,

    // --- Batch sizes ---
    #[arg(long, default_value_t = 1024)]
    play_batch: usize,
    #[arg(long, default_value_t = 512)]
    bid_batch: usize,

    // --- Training frequencies ---
    /// Train play NN every N steps.
    #[arg(long, default_value_t = 4)]
    play_train_freq: usize,
    /// Train bid NN every N steps.
    #[arg(long, default_value_t = 16)]
    bid_train_freq: usize,

    // --- Replay buffers ---
    #[arg(long, default_value_t = 2_000_000)]
    play_buffer_size: usize,
    #[arg(long, default_value_t = 2_000_000)]
    bid_buffer_size: usize,
    #[arg(long, default_value_t = 10_000)]
    play_min_buffer: usize,
    #[arg(long, default_value_t = 5_000)]
    bid_min_buffer: usize,

    // --- Exploration ---
    #[arg(long, default_value_t = 0.25)]
    play_eps_start: f32,
    #[arg(long, default_value_t = 0.01)]
    play_eps_end: f32,
    #[arg(long, default_value_t = 8_000_000)]
    play_eps_decay: usize,
    #[arg(long, default_value_t = 0.40)]
    bid_eps_start: f32,
    #[arg(long, default_value_t = 0.03)]
    bid_eps_end: f32,
    #[arg(long, default_value_t = 15_000_000)]
    bid_eps_decay: usize,

    // --- PER ---
    #[arg(long, default_value_t = 0.6)]
    per_alpha: f64,
    #[arg(long, default_value_t = 0.4)]
    per_beta_start: f64,
    #[arg(long, default_value_t = 1.0)]
    per_beta_end: f64,

    // --- Phased warm-up ---
    /// Steps for Phase 0 (heuristic-dominant bidding).
    #[arg(long, default_value_t = 2_000_000)]
    warmup_steps: usize,
    /// Steps to complete Phase 1 handover (heuristic → NN bidding).
    #[arg(long, default_value_t = 8_000_000)]
    handover_steps: usize,
    /// Fraction of heuristic opponents that remain even after handover.
    #[arg(long, default_value_t = 0.30)]
    heuristic_floor: f32,

    // --- Play opponent diversity ---
    #[arg(long, default_value_t = 0.0)]
    play_pool_frac: f32,
    #[arg(long, default_value_t = 0.1)]
    play_random_frac: f32,
    #[arg(long, default_value_t = 500_000)]
    play_pool_save_freq: usize,
    #[arg(long, default_value_t = 10)]
    play_pool_size: usize,

    // --- Bid opponent pool ---
    #[arg(long, default_value_t = 1_000_000)]
    bid_pool_save_freq: usize,
    #[arg(long, default_value_t = 5)]
    bid_pool_size: usize,

    // --- Evaluation ---
    #[arg(long, default_value_t = 1_000_000)]
    eval_freq: usize,
    #[arg(long, default_value_t = 500)]
    eval_random_matches: usize,
    #[arg(long, default_value_t = 0)]
    eval_isdd_matches: usize,
    #[arg(long, default_value_t = 20)]
    eval_isdd_time_ms: u32,
    /// Frozen play .bin checkpoint for comparison.
    #[arg(long)]
    eval_play_checkpoint: Option<String>,
    #[arg(long, default_value_t = 500)]
    eval_checkpoint_matches: usize,
    /// Fixed bid model for eval (e.g. bid_nn_final.bin). Both sides use this bidder.
    #[arg(long)]
    eval_bid_model: Option<String>,

    // --- Checkpointing ---
    #[arg(long, default_value_t = 500_000)]
    save_freq: usize,
    #[arg(long, default_value = "models/joint")]
    save_dir: String,
    #[arg(long)]
    resume_play: Option<String>,
    #[arg(long)]
    resume_bid: Option<String>,

    /// Frozen belief net for card location prediction (appended to play obs).
    #[arg(long)]
    belief_model: Option<String>,

    #[arg(long, default_value_t = 42)]
    seed: u64,
}

/// Opponent type for the play phase.
const OPP_SELF: u8 = 0;
const OPP_POOL: u8 = 1;
const OPP_RANDOM: u8 = 2;

/// Buffered bid transition awaiting end-of-game reward.
struct BidTrans {
    obs: Vec<f32>,
    mask: Vec<f32>,
    action: u8,
    team: u8,
}

/// Buffered play transition awaiting end-of-game reward.
struct PlayTrans {
    obs: Vec<f32>,
    mask: Vec<f32>,
    action: u8,
    team: u8,
}

/// Compute the NN bid fraction at a given step.
/// Phase 0 (step < warmup): heuristic_floor only.
/// Phase 1 (warmup..handover): linear from heuristic_floor to (1 - heuristic_floor).
/// Phase 2 (step >= handover): 1 - heuristic_floor.
fn nn_bid_fraction(step: usize, warmup: usize, handover: usize, heuristic_floor: f32) -> f32 {
    let max_nn = 1.0 - heuristic_floor;
    if step < warmup {
        // Phase 0: small NN fraction for initial data collection
        0.15
    } else if step < handover {
        let progress = (step - warmup) as f32 / (handover - warmup) as f32;
        0.15 + (max_nn - 0.15) * progress
    } else {
        max_nn
    }
}

/// Evaluate: (our_bid + our_play) vs (opp_bid + opp_play).
/// Our play NN uses trump-relative (canonical) observations, optionally augmented with belief predictions.
/// Checkpoint opponent auto-detects obs layout: canonical (411, e.g. DouDou50 ResNet) or legacy (415, e.g. DouDou35).
fn evaluate_play(
    play_trainer: &DuelingTrainer,
    play_hidden: usize,
    play_obs_dim: usize,
    our_bid_net: &mut Option<BidNet>,
    opp_bid_net: &mut Option<BidNet>,
    random_matches: usize,
    checkpoint_matches: usize,
    baseline_net: &mut Option<DmcNet>,
    eval_belief_net: &mut Option<BeliefNet>,
) -> EvalResult {
    let start = std::time::Instant::now();

    let weights = match play_trainer.snapshot_weights() {
        Ok(w) => w,
        Err(e) => {
            eprintln!("  Failed to snapshot play weights: {}", e);
            return EvalResult { rand_wr: 0.0, ckpt_wr: 0.0, isdd_wr: 0.0, elapsed: 0.0 };
        }
    };
    let mut q_net = match DmcNet::from_floats(&weights, play_hidden, play_obs_dim, true) {
        Ok(mut n) => { n.set_residual(true); n }
        Err(e) => {
            eprintln!("  Failed to load eval play net: {}", e);
            return EvalResult { rand_wr: 0.0, ckpt_wr: 0.0, isdd_wr: 0.0, elapsed: 0.0 };
        }
    };

    // vs random: our_bid + our_play(TR) vs our_bid + random_play
    let rand_wr = if random_matches > 0 {
        let num_pairs = random_matches / 2;
        let mut wins = 0u32;
        for i in 0..num_pairs {
            let seed = 200_000 + i as u64;
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut rng_b = StdRng::seed_from_u64(seed);
            if play_match_eval_tr(&mut q_net, 0, "random", &mut None, our_bid_net, &mut rng_a, eval_belief_net) {
                wins += 1;
            }
            if play_match_eval_tr(&mut q_net, 1, "random", &mut None, our_bid_net, &mut rng_b, eval_belief_net) {
                wins += 1;
            }
        }
        wins as f64 / (num_pairs as f64 * 2.0)
    } else {
        0.0
    };

    // vs checkpoint: (our_bid + our_play_TR) vs (opp_bid + checkpoint_play_legacy)
    let ckpt_wr = if checkpoint_matches > 0 && baseline_net.is_some() {
        let num_pairs = checkpoint_matches / 2;
        let mut wins = 0u32;
        for i in 0..num_pairs {
            let seed = 300_000 + i as u64;
            let mut rng_a = StdRng::seed_from_u64(seed);
            let mut rng_b = StdRng::seed_from_u64(seed);
            if play_match_eval_tr_dual(&mut q_net, 0, baseline_net, our_bid_net, opp_bid_net, &mut rng_a, eval_belief_net) {
                wins += 1;
            }
            if play_match_eval_tr_dual(&mut q_net, 1, baseline_net, our_bid_net, opp_bid_net, &mut rng_b, eval_belief_net) {
                wins += 1;
            }
        }
        wins as f64 / (num_pairs as f64 * 2.0)
    } else {
        0.0
    };

    let elapsed = start.elapsed().as_secs_f64();
    EvalResult { rand_wr, ckpt_wr, isdd_wr: 0.0, elapsed }
}

/// Build belief-augmented canonical observation for eval.
/// Returns the base tr_obs extended with 96 belief prediction floats in canonical card order.
fn make_belief_augmented_obs(
    state: &GameState,
    tracking: &colver_core::dmc_obs::EnvTracking,
    belief_net: &mut BeliefNet,
    order: &[u8; 4],
) -> Vec<f32> {
    use colver_core::card;

    let observer = state.current_player();
    let mut belief_obs = vec![0.0f32; BELIEF_OBS_DIM];
    belief_obs_mod::write_belief_observation(&mut belief_obs, 0, state, tracking, observer);
    let logits = belief_net.evaluate(&belief_obs);
    let num_classes = belief_net.num_classes();

    let observer_hand = state.hands[observer as usize];
    let mut played = state.played_cards;
    for ci in 0..4 {
        let c = state.current_trick[ci];
        if c != card::EMPTY {
            played |= 1u32 << c;
        }
    }
    let known = observer_hand | played;

    let mut preds = vec![0.0f32; BELIEF_PRED_DIM];
    for card_phys in 0..32u8 {
        let card_canon = obs::card_to_canonical(card_phys, order);
        let canon_base = card_canon as usize * 3;

        if known & (1u32 << card_phys) != 0 {
            continue; // already zeroed
        }

        let (p_left, p_partner, p_right) = if num_classes == 3 {
            let base = card_phys as usize * 3;
            let max_l = logits[base].max(logits[base + 1]).max(logits[base + 2]);
            let e0 = (logits[base] - max_l).exp();
            let e1 = (logits[base + 1] - max_l).exp();
            let e2 = (logits[base + 2] - max_l).exp();
            let s = e0 + e1 + e2;
            (e0 / s, e1 / s, e2 / s)
        } else {
            let base = card_phys as usize * 4;
            let max_l = logits[base].max(logits[base + 1]).max(logits[base + 2]).max(logits[base + 3]);
            let e1 = (logits[base + 1] - max_l).exp();
            let e2 = (logits[base + 2] - max_l).exp();
            let e3 = (logits[base + 3] - max_l).exp();
            let s = e1 + e2 + e3;
            (e1 / s, e2 / s, e3 / s)
        };

        preds[canon_base] = p_left;
        preds[canon_base + 1] = p_partner;
        preds[canon_base + 2] = p_right;
    }

    let mut full_obs = obs::make_observation_tr(state, tracking);
    full_obs.extend_from_slice(&preds);
    full_obs
}

/// Play a match to 2000: our_play uses trump-relative obs, opponent uses random.
fn play_match_eval_tr(
    q_net: &mut DmcNet,
    q_team: u8,
    baseline: &str,
    _baseline_net: &mut Option<DmcNet>,
    bid_net: &mut Option<BidNet>,
    rng: &mut StdRng,
    belief_net: &mut Option<BeliefNet>,
) -> bool {
    use colver_core::dmc_obs::EnvTracking;
    use colver_core::rollout;

    let mut q_total = 0.0f32;
    let mut opp_total = 0.0f32;
    for _ in 0..50 {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, rng);
        let mut tracking = EnvTracking::new();
        tracking.dealer = dealer;

        while !state.is_terminal() {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if state.phase == Phase::Bidding {
                dmc_eval::eval_bid_action(&state, &tracking.bid_history, bid_net)
            } else if team == q_team {
                let order = obs::current_player_order(&state, &tracking);
                let canonical_mask = obs::cardset_to_canonical(state.legal_actions() as u32, &order);
                let full_obs = if let Some(ref mut bnet) = belief_net {
                    make_belief_augmented_obs(&state, &tracking, bnet, &order)
                } else {
                    obs::make_observation_tr(&state, &tracking)
                };
                let (canonical_best, _) = q_net.best_action(&full_obs, canonical_mask as u32);
                obs::card_to_physical(canonical_best, &order)
            } else {
                match baseline {
                    "random" => {
                        let mask = state.legal_actions();
                        let count = mask.count_ones();
                        let idx = rng.gen_range(0..count);
                        rollout::select_nth_bit(mask, idx)
                    }
                    _ => unreachable!(),
                }
            };

            tracking.track_action(&state, action);
            state.step(action);
        }

        let rewards = state.rewards();
        q_total += rewards[q_team as usize];
        opp_total += rewards[1 - q_team as usize];
        if q_total >= 2000.0 || opp_total >= 2000.0 {
            break;
        }
    }
    q_total >= 2000.0
}

/// Play a match to 2000: our team uses canonical play + our bid, opponent uses canonical or legacy play + opp bid.
fn play_match_eval_tr_dual(
    q_net: &mut DmcNet,
    q_team: u8,
    baseline_net: &mut Option<DmcNet>,
    q_bid_net: &mut Option<BidNet>,
    opp_bid_net: &mut Option<BidNet>,
    rng: &mut StdRng,
    belief_net: &mut Option<BeliefNet>,
) -> bool {
    use colver_core::dmc_obs::{EnvTracking, make_observation};

    let mut q_total = 0.0f32;
    let mut opp_total = 0.0f32;
    for _ in 0..50 {
        let dealer = rng.gen_range(0..4u8);
        let mut state = GameState::deal_random(dealer, rng);
        let mut tracking = EnvTracking::new();
        tracking.dealer = dealer;

        while !state.is_terminal() {
            let player = state.current_player();
            let team = GameState::player_team(player);

            let action = if state.phase == Phase::Bidding {
                if team == q_team {
                    dmc_eval::eval_bid_action(&state, &tracking.bid_history, q_bid_net)
                } else {
                    dmc_eval::eval_bid_action(&state, &tracking.bid_history, opp_bid_net)
                }
            } else if team == q_team {
                let order = obs::current_player_order(&state, &tracking);
                let canonical_mask = obs::cardset_to_canonical(state.legal_actions() as u32, &order);
                let full_obs = if let Some(ref mut bnet) = belief_net {
                    make_belief_augmented_obs(&state, &tracking, bnet, &order)
                } else {
                    obs::make_observation_tr(&state, &tracking)
                };
                let (canonical_best, _) = q_net.best_action(&full_obs, canonical_mask as u32);
                obs::card_to_physical(canonical_best, &order)
            } else {
                // Opponent: auto-detect canonical (411) vs legacy (415) from baseline obs_dim
                let net = baseline_net.as_mut().unwrap();
                if net.obs_dim() == OBS_DIM_TR {
                    let order = obs::current_player_order(&state, &tracking);
                    let canonical_mask = obs::cardset_to_canonical(state.legal_actions() as u32, &order);
                    let canonical_obs = obs::make_observation_tr(&state, &tracking);
                    let (canonical_best, _) = net.best_action(&canonical_obs, canonical_mask as u32);
                    obs::card_to_physical(canonical_best, &order)
                } else {
                    let legacy_obs = make_observation(&state, &tracking);
                    let legal_mask = state.legal_actions() as u32;
                    let (best, _) = net.best_action(&legacy_obs, legal_mask);
                    best
                }
            };

            tracking.track_action(&state, action);
            state.step(action);
        }

        let rewards = state.rewards();
        q_total += rewards[q_team as usize];
        opp_total += rewards[1 - q_team as usize];
        if q_total >= 2000.0 || opp_total >= 2000.0 {
            break;
        }
    }
    q_total >= 2000.0
}

fn main() {
    let args = Args::parse();

    let device = if candle_core::utils::cuda_is_available() {
        Device::new_cuda(0).expect("CUDA device creation failed")
    } else {
        eprintln!("WARNING: CUDA not available, using CPU");
        Device::Cpu
    };

    let play_obs_dim = if args.belief_model.is_some() { OBS_DIM_TR + BELIEF_PRED_DIM } else { OBS_DIM_TR };

    println!("=== Triforge Training ({}) ===", args.mode);
    println!("Device: {:?}", device);
    println!("Envs: {}, Steps: {}M", args.num_envs, args.steps / 1_000_000);
    println!("Play NN: {}→{}³→{} (ResNet, canonical{}) | Bid NN: {}→{}×{}→{}",
        play_obs_dim, args.play_hidden, PLAY_ACTIONS,
        if args.belief_model.is_some() { ", +belief 96" } else { "" },
        BID_OBS_DIM, args.bid_hidden, args.bid_layers, BID_ACTIONS);
    match args.mode {
        TrainMode::Joint => {
            println!("Mode: joint — training both networks");
            println!("Warm-up: {}M steps → Handover: {}M → Floor: {:.0}% heuristic",
                args.warmup_steps / 1_000_000, args.handover_steps / 1_000_000,
                args.heuristic_floor * 100.0);
        }
        TrainMode::PlayOnly => {
            println!("Mode: play-only — bid NN frozen, training play NN only");
            if args.resume_bid.is_none() {
                eprintln!("WARNING: --resume-bid required for play-only mode!");
            }
        }
        TrainMode::BidOnly => {
            println!("Mode: bid-only — play NN frozen, training bid NN only");
            if args.resume_play.is_none() {
                eprintln!("WARNING: --resume-play required for bid-only mode!");
            }
        }
    }
    println!();

    // --- Initialize trainers ---
    let mut play_trainer = DuelingTrainer::with_residual(play_obs_dim, args.play_hidden, args.play_lr, 0.0, device.clone())
        .expect("Failed to create play trainer");
    let mut bid_trainer = BiddingTrainer::with_layers(args.bid_layers, args.bid_hidden, args.bid_lr, 0.0, device.clone())
        .expect("Failed to create bid trainer");

    if let Some(ref path) = args.resume_play {
        play_trainer.load_checkpoint(path).expect("Failed to load play checkpoint");
        println!("Resumed play NN from {}", path);
    }
    if let Some(ref path) = args.resume_bid {
        bid_trainer.load_checkpoint(path).expect("Failed to load bid checkpoint");
        println!("Resumed bid NN from {}", path);
    }

    // Frozen play checkpoint for eval. Canonical baselines (e.g. DouDou50) need residual ResNet forward.
    let mut eval_play_baseline: Option<DmcNet> = args.eval_play_checkpoint.as_ref().map(|path| {
        let mut net = DmcNet::load(path).unwrap_or_else(|e| panic!("Failed to load {}: {}", path, e));
        let canonical = net.obs_dim() == OBS_DIM_TR;
        if canonical {
            net.set_residual(true);
        }
        println!("Eval play baseline: {} (obs_dim={}, {})", path, net.obs_dim(),
            if canonical { "canonical ResNet" } else { "legacy" });
        net
    });

    // Opponent bid NN for eval: fixed model (Bid à Dédé) used by DouDou35 side
    let mut eval_opp_bid_net: Option<BidNet> = args.eval_bid_model.as_ref().map(|path| {
        let net = BidNet::load(path).unwrap_or_else(|e| panic!("Failed to load eval bid model {}: {}", path, e));
        println!("Eval opponent bidder: {} (Bid à Dédé)", path);
        net
    });

    // Our bid NN for eval: snapshot from bid_trainer (updated each eval)
    let mut eval_our_bid_net: Option<BidNet> = None;

    // Belief net for eval (separate instance from vec_env's, since eval runs independently)
    let mut eval_belief_net: Option<BeliefNet> = args.belief_model.as_ref().map(|path| {
        BeliefNet::load(path).unwrap_or_else(|e| panic!("Failed to load eval belief net {}: {}", path, e))
    });

    // --- Replay buffers ---
    let mut play_buffer = FlexReplayBuffer::new(
        args.play_buffer_size, args.per_alpha, play_obs_dim, PLAY_ACTIONS,
    );
    let mut bid_buffer = FlexReplayBuffer::new(
        args.bid_buffer_size, args.per_alpha, BID_OBS_DIM, BID_MASK_DIM,
    );

    // --- Environments ---
    let mut vec_env = VecJointEnv::new(args.num_envs, args.seed);
    if let Some(ref path) = args.belief_model {
        vec_env.load_belief_net(path).expect("Failed to load belief net");
        println!("Belief net loaded: {} (+{} dims → play obs {})", path, BELIEF_PRED_DIM, play_obs_dim);
    }
    let mut rng = StdRng::seed_from_u64(args.seed + 1);

    // --- Episode transition buffers ---
    let mut bid_episode_bufs: Vec<Vec<BidTrans>> = (0..args.num_envs)
        .map(|_| Vec::with_capacity(12))
        .collect();
    let mut play_episode_bufs: Vec<Vec<PlayTrans>> = (0..args.num_envs)
        .map(|_| Vec::with_capacity(32))
        .collect();

    // --- Play opponent tracking ---
    let mut play_opp_type = vec![OPP_SELF; args.num_envs];
    let mut play_opp_team = vec![0u8; args.num_envs];

    // --- Bid opponent tracking ---
    // Per-env: which team uses the NN bidder (the other uses heuristic/pool).
    let mut bid_nn_team = vec![0u8; args.num_envs]; // 0=NS uses NN, 1=EW uses NN
    // Heuristic strategy for the opponent team (0-7).
    let mut bid_opp_strategy = vec![0u8; args.num_envs];

    // Initialize opponent assignments
    let effective_random_frac = if args.mode == TrainMode::BidOnly { 0.0 } else { args.play_random_frac };
    let effective_pool_frac = if args.mode == TrainMode::BidOnly { 0.0 } else { args.play_pool_frac };
    for i in 0..args.num_envs {
        bid_nn_team[i] = rng.gen_range(0..2);
        bid_opp_strategy[i] = rng.gen_range(0..8);

        let roll: f32 = rng.gen();
        if roll < effective_random_frac {
            play_opp_type[i] = OPP_RANDOM;
        } else if roll < effective_random_frac + effective_pool_frac {
            play_opp_type[i] = OPP_POOL;
        }
        play_opp_team[i] = rng.gen_range(0..2);
    }

    // --- Play opponent pool ---
    let mut play_pool_weights: Vec<Vec<f32>> = Vec::new();
    let mut play_pool_net: Option<PoolNet> = None;

    // --- Bid opponent pool ---
    let mut bid_pool_weights: Vec<Vec<f32>> = Vec::new();

    // --- Stats ---
    let mut total_episodes = 0usize;
    let mut play_loss_total = 0.0f64;
    let mut play_loss_count = 0usize;
    let mut bid_loss_total = 0.0f64;
    let mut bid_loss_count = 0usize;
    let mut play_transitions_total = 0usize;
    let mut bid_transitions_total = 0usize;
    let last_log = Instant::now();
    let mut last_log_step = 0usize;
    let mut last_log_time = last_log;

    // --- CSV log file ---
    std::fs::create_dir_all(&args.save_dir).ok();
    let csv_path = format!("{}/training_log.csv", args.save_dir);
    let csv_exists = std::path::Path::new(&csv_path).exists();
    let mut csv_file = std::fs::OpenOptions::new()
        .create(true).append(true).open(&csv_path)
        .expect("Failed to open CSV log");
    if !csv_exists {
        writeln!(csv_file, "step,play_eps,bid_eps,beta,play_buf,bid_buf,play_loss,bid_loss,episodes,nn_pct,steps_per_sec,play_trans,bid_trans,rand_wr,ckpt_wr,isdd_wr").ok();
    }
    println!("CSV log: {}", csv_path);

    println!("{:>10} | {:>5} {:>5} | {:>5} | {:>7} {:>7} | {:>8} {:>8} | {:>7} | {:>5}",
        "Step", "pEps", "bEps", "Beta", "pBuf", "bBuf", "pLoss", "bLoss", "Ep", "NN%");
    println!("{}", "-".repeat(100));

    // Track latest eval results for CSV
    let mut last_rand_wr = 0.0f64;
    let mut last_ckpt_wr = 0.0f64;
    let mut last_isdd_wr = 0.0f64;

    // --- Main training loop ---
    for step in 0..args.steps {
        // Schedules
        let play_progress = (step as f64 / args.play_eps_decay as f64).min(1.0);
        let play_eps = args.play_eps_start + (args.play_eps_end - args.play_eps_start) * play_progress as f32;

        let bid_progress = (step as f64 / args.bid_eps_decay as f64).min(1.0);
        let bid_eps = args.bid_eps_start + (args.bid_eps_end - args.bid_eps_start) * bid_progress as f32;

        let beta_progress = (step as f64 / args.steps as f64).min(1.0);
        let beta = args.per_beta_start + (args.per_beta_end - args.per_beta_start) * beta_progress;

        let nn_frac = match args.mode {
            TrainMode::PlayOnly => 1.0, // always use frozen bid NN
            _ => nn_bid_fraction(step, args.warmup_steps, args.handover_steps, args.heuristic_floor),
        };

        let n = args.num_envs;
        let phases = vec_env.phases();
        let players = vec_env.current_players();

        let mut actions = vec![0u8; n];

        // ========== BIDDING PHASE ==========
        {
            // Collect bidding envs
            let mut nn_bid_indices = Vec::new();
            let mut opp_bid_indices = Vec::new();

            for i in 0..n {
                if phases[i] != Phase::Bidding {
                    continue;
                }
                let player = players[i];
                let team = GameState::player_team(player);

                // Decide if this player uses NN or heuristic
                let use_nn = if team == bid_nn_team[i] {
                    // NN team: use NN with probability nn_frac
                    rng.gen::<f32>() < nn_frac
                } else {
                    false
                };

                if use_nn {
                    nn_bid_indices.push(i);
                } else {
                    opp_bid_indices.push(i);
                }
            }

            // NN bid: GPU batch forward
            if !nn_bid_indices.is_empty() {
                let batch = nn_bid_indices.len();
                let mut obs_flat = vec![0.0f32; batch * BID_OBS_DIM];
                let mut mask_flat = vec![0.0f32; batch * BID_MASK_DIM];

                for (j, &i) in nn_bid_indices.iter().enumerate() {
                    obs_flat[j * BID_OBS_DIM..(j + 1) * BID_OBS_DIM]
                        .copy_from_slice(vec_env.bid_obs_slice(i));
                    mask_flat[j * BID_MASK_DIM..(j + 1) * BID_MASK_DIM]
                        .copy_from_slice(vec_env.bid_mask_slice(i));
                }

                let bid_actions = match bid_trainer.net.act(
                    &candle_core::Tensor::from_slice(&obs_flat, (batch, BID_OBS_DIM), bid_trainer.device()).unwrap(),
                    &candle_core::Tensor::from_slice(&mask_flat, (batch, BID_MASK_DIM), bid_trainer.device()).unwrap(),
                    bid_eps,
                    &mut rng,
                ) {
                    Ok(a) => a,
                    Err(e) => {
                        eprintln!("Bid GPU forward failed: {}", e);
                        nn_bid_indices.iter().map(|&i| vec_env.random_action(i)).collect()
                    }
                };

                for (j, &i) in nn_bid_indices.iter().enumerate() {
                    actions[i] = bid_actions[j];
                }
            }

            // Opponent bids: heuristic or bid pool
            for &i in &opp_bid_indices {
                let strategy = bid_opp_strategy[i];
                // Occasionally use bid pool if available
                if !bid_pool_weights.is_empty() && rng.gen::<f32>() < 0.3 {
                    actions[i] = vec_env.nn_bid(i);
                } else {
                    actions[i] = vec_env.heuristic_bid(i, strategy);
                }
            }

            // Record bid transitions for NN team players (skip in play-only mode)
            if args.mode != TrainMode::PlayOnly {
                for &i in &nn_bid_indices {
                    let team = GameState::player_team(players[i]);
                    let mask = vec_env.bid_mask_slice(i);
                    let n_legal: usize = mask.iter().filter(|&&v| v > 0.5).count();
                    if n_legal > 1 {
                        bid_episode_bufs[i].push(BidTrans {
                            obs: vec_env.bid_obs_slice(i).to_vec(),
                            mask: mask.to_vec(),
                            action: actions[i],
                            team,
                        });
                    }
                }
            }
        }

        // ========== PLAYING PHASE ==========
        {
            let mut play_indices = Vec::new();
            for i in 0..n {
                if phases[i] == Phase::Playing {
                    play_indices.push(i);
                }
            }

            if !play_indices.is_empty() {
                // Compute belief predictions only for envs that need play actions
                if vec_env.has_belief_net() {
                    vec_env.refresh_belief_preds_for(&play_indices);
                }

                let mut self_indices = Vec::new();
                let mut pool_indices = Vec::new();
                let mut random_indices = Vec::new();

                for &i in &play_indices {
                    let team = GameState::player_team(players[i]);
                    if play_opp_type[i] != OPP_SELF && team == play_opp_team[i] {
                        match play_opp_type[i] {
                            OPP_POOL => pool_indices.push(i),
                            OPP_RANDOM => random_indices.push(i),
                            _ => self_indices.push(i),
                        }
                    } else {
                        self_indices.push(i);
                    }
                }

                // Self-play: GPU Q-network + ε-greedy
                if !self_indices.is_empty() {
                    let batch = self_indices.len();
                    let mut obs_flat = vec![0.0f32; batch * play_obs_dim];
                    let mut mask_flat = vec![0.0f32; batch * PLAY_ACTIONS];
                    for (j, &i) in self_indices.iter().enumerate() {
                        obs_flat[j * play_obs_dim..j * play_obs_dim + OBS_DIM_TR]
                            .copy_from_slice(vec_env.play_obs_slice(i));
                        if vec_env.has_belief_net() {
                            obs_flat[j * play_obs_dim + OBS_DIM_TR..(j + 1) * play_obs_dim]
                                .copy_from_slice(vec_env.belief_pred_slice(i));
                        }
                        mask_flat[j * PLAY_ACTIONS..(j + 1) * PLAY_ACTIONS]
                            .copy_from_slice(vec_env.play_mask_slice(i));
                    }

                    let q_actions = match play_trainer.net.act(
                        &candle_core::Tensor::from_slice(&obs_flat, (batch, play_obs_dim), play_trainer.device()).unwrap(),
                        &candle_core::Tensor::from_slice(&mask_flat, (batch, PLAY_ACTIONS), play_trainer.device()).unwrap(),
                        play_eps,
                        &mut rng,
                    ) {
                        Ok(a) => a,
                        Err(e) => {
                            eprintln!("Play GPU forward failed: {}", e);
                            self_indices.iter().map(|&i| vec_env.random_action(i)).collect()
                        }
                    };

                    for (j, &i) in self_indices.iter().enumerate() {
                        actions[i] = q_actions[j];
                    }
                }

                // Pool: GPU batched greedy
                if !pool_indices.is_empty() {
                    if let Some(ref pnet) = play_pool_net {
                        let pbatch = pool_indices.len();
                        let mut pool_obs = vec![0.0f32; pbatch * play_obs_dim];
                        let mut pool_mask = vec![0.0f32; pbatch * PLAY_ACTIONS];
                        for (j, &i) in pool_indices.iter().enumerate() {
                            pool_obs[j * play_obs_dim..j * play_obs_dim + OBS_DIM_TR]
                                .copy_from_slice(vec_env.play_obs_slice(i));
                            if vec_env.has_belief_net() {
                                pool_obs[j * play_obs_dim + OBS_DIM_TR..(j + 1) * play_obs_dim]
                                    .copy_from_slice(vec_env.belief_pred_slice(i));
                            }
                            pool_mask[j * PLAY_ACTIONS..(j + 1) * PLAY_ACTIONS]
                                .copy_from_slice(vec_env.play_mask_slice(i));
                        }
                        let obs_t = candle_core::Tensor::from_slice(&pool_obs, (pbatch, play_obs_dim), play_trainer.device()).unwrap();
                        let mask_t = candle_core::Tensor::from_slice(&pool_mask, (pbatch, PLAY_ACTIONS), play_trainer.device()).unwrap();
                        match pnet.act_greedy(&obs_t, &mask_t) {
                            Ok(pa) => {
                                for (j, &i) in pool_indices.iter().enumerate() {
                                    actions[i] = pa[j];
                                }
                            }
                            Err(_) => {
                                for &i in &pool_indices {
                                    actions[i] = vec_env.random_action(i);
                                }
                            }
                        }
                    } else {
                        for &i in &pool_indices {
                            actions[i] = vec_env.random_action(i);
                        }
                    }
                }

                // Random (convert physical → canonical for consistent replay storage)
                for &i in &random_indices {
                    let phys = vec_env.random_action(i);
                    let order = obs::current_player_order(&vec_env.envs[i].state, &vec_env.envs[i].tracking);
                    actions[i] = obs::card_to_canonical(phys, &order);
                }

                // Record play transitions (skip in bid-only mode)
                if args.mode != TrainMode::BidOnly {
                    for &i in &play_indices {
                        let mask = vec_env.play_mask_slice(i);
                        let n_legal: usize = mask.iter().filter(|&&v| v > 0.5).count();
                        if n_legal > 1 {
                            let team = GameState::player_team(players[i]);
                            let mut obs = vec_env.play_obs_slice(i).to_vec();
                            if vec_env.has_belief_net() {
                                obs.extend_from_slice(vec_env.belief_pred_slice(i));
                            }
                            play_episode_bufs[i].push(PlayTrans {
                                obs,
                                mask: mask.to_vec(),
                                action: actions[i],
                                team,
                            });
                        }
                    }
                }
            }
        }

        // ========== STEP ALL ENVIRONMENTS ==========
        // Convert playing-phase canonical actions to physical for env.step
        let mut physical_actions = actions.clone();
        for i in 0..n {
            if phases[i] == Phase::Playing {
                let order = obs::current_player_order(&vec_env.envs[i].state, &vec_env.envs[i].tracking);
                physical_actions[i] = obs::card_to_physical(actions[i], &order);
            }
        }
        let (dones, rewards) = vec_env.step_all(&physical_actions);

        // ========== FLUSH COMPLETED EPISODES ==========
        for i in 0..n {
            if !dones[i] {
                continue;
            }

            let (ns_reward, ew_reward) = rewards[i];

            // Flush play transitions
            let play_buf = std::mem::replace(&mut play_episode_bufs[i], Vec::with_capacity(32));
            for trans in &play_buf {
                let ret = if trans.team == 0 { ns_reward } else { ew_reward };
                play_buffer.push(&trans.obs, &trans.mask, trans.action, ret);
            }
            play_transitions_total += play_buf.len();

            // Flush bid transitions
            let bid_buf = std::mem::replace(&mut bid_episode_bufs[i], Vec::with_capacity(12));
            for trans in &bid_buf {
                let ret = if trans.team == 0 { ns_reward } else { ew_reward };
                bid_buffer.push(&trans.obs, &trans.mask, trans.action, ret);
            }
            bid_transitions_total += bid_buf.len();

            total_episodes += 1;

            // Re-randomize opponent assignments for new deal
            bid_nn_team[i] = rng.gen_range(0..2);
            bid_opp_strategy[i] = rng.gen_range(0..8);

            let roll: f32 = rng.gen();
            if roll < effective_random_frac {
                play_opp_type[i] = OPP_RANDOM;
                play_opp_team[i] = rng.gen_range(0..2);
            } else if roll < effective_random_frac + effective_pool_frac && play_pool_net.is_some() {
                play_opp_type[i] = OPP_POOL;
                play_opp_team[i] = rng.gen_range(0..2);
            } else {
                play_opp_type[i] = OPP_SELF;
            }
        }

        // ========== TRAIN PLAY NN ========== (skip in bid-only mode)
        if args.mode != TrainMode::BidOnly && play_buffer.size() >= args.play_min_buffer && step % args.play_train_freq == 0 {
            let sample = play_buffer.sample(args.play_batch, beta, &mut rng);

            match play_trainer.train_step(
                &sample.obs_data,
                &sample.mask_data,
                &sample.actions,
                &sample.returns,
                &sample.weights,
            ) {
                Ok((loss, td_errors)) => {
                    play_buffer.update_priorities(&sample.indices, &td_errors);
                    play_loss_total += loss as f64;
                    play_loss_count += 1;
                }
                Err(e) => eprintln!("Play train failed: {}", e),
            }
        }

        // ========== TRAIN BID NN ========== (skip in play-only mode)
        if args.mode != TrainMode::PlayOnly && bid_buffer.size() >= args.bid_min_buffer && step % args.bid_train_freq == 0 {
            let mut sample = bid_buffer.sample(args.bid_batch, beta, &mut rng);

            // Suit augmentation
            suit_perm::augment_bid_batch(
                &mut sample.obs_data,
                &mut sample.mask_data,
                &mut sample.actions,
                &mut rng,
            );

            match bid_trainer.train_step(
                &sample.obs_data,
                &sample.mask_data,
                &sample.actions,
                &sample.returns,
                &sample.weights,
            ) {
                Ok((loss, td_errors)) => {
                    bid_buffer.update_priorities(&sample.indices, &td_errors);
                    bid_loss_total += loss as f64;
                    bid_loss_count += 1;
                }
                Err(e) => eprintln!("Bid train failed: {}", e),
            }
        }

        // ========== PLAY OPPONENT POOL ==========
        // Only start pool after warmup — early models are garbage and destabilize training
        if (step + 1) % args.play_pool_save_freq == 0 && args.play_pool_frac > 0.0 && step >= args.warmup_steps {
            if let Ok(weights) = play_trainer.snapshot_weights() {
                play_pool_weights.push(weights);
                if play_pool_weights.len() > args.play_pool_size {
                    play_pool_weights.remove(0);
                }
                let idx = rng.gen_range(0..play_pool_weights.len());
                match &play_pool_net {
                    Some(pnet) => { pnet.load_weights(&play_pool_weights[idx]).ok(); }
                    None => {
                        if let Ok(pnet) = PoolNet::with_residual(OBS_DIM_TR, args.play_hidden, play_trainer.device()) {
                            pnet.load_weights(&play_pool_weights[idx]).ok();
                            play_pool_net = Some(pnet);
                        }
                    }
                }
                println!("  [PLAY POOL] size: {}", play_pool_weights.len());
            }
        }

        // ========== BID OPPONENT POOL ==========
        if (step + 1) % args.bid_pool_save_freq == 0 && step >= args.warmup_steps {
            if let Ok(weights) = bid_trainer.snapshot_weights() {
                bid_pool_weights.push(weights);
                if bid_pool_weights.len() > args.bid_pool_size {
                    bid_pool_weights.remove(0);
                }
                // Load latest bid pool model into vec_env for nn_bid() calls
                let idx = rng.gen_range(0..bid_pool_weights.len());
                vec_env.set_bid_net_from_floats(&bid_pool_weights[idx], args.bid_hidden).ok();
                println!("  [BID POOL] size: {}", bid_pool_weights.len());
            }
        }

        // ========== LOGGING ==========
        if (step + 1) % 10_000 == 0 {
            let elapsed = last_log_time.elapsed().as_secs_f64();
            let steps_done = step + 1 - last_log_step;
            let sps = steps_done as f64 / elapsed.max(1e-6);
            let play_loss_avg = if play_loss_count > 0 { play_loss_total / play_loss_count as f64 } else { 0.0 };
            let bid_loss_avg = if bid_loss_count > 0 { bid_loss_total / bid_loss_count as f64 } else { 0.0 };
            println!(
                "{:>10} | {:>5.3} {:>5.3} | {:>5.3} | {:>7} {:>7} | {:>8.4} {:>8.4} | {:>7} | {:>4.0}%  {:.0}s/s",
                step + 1,
                play_eps, bid_eps,
                beta,
                play_buffer.size(), bid_buffer.size(),
                play_loss_avg, bid_loss_avg,
                total_episodes,
                nn_frac * 100.0,
                sps,
            );
            // CSV row
            writeln!(csv_file,
                "{},{:.4},{:.4},{:.4},{},{},{:.6},{:.6},{},{:.1},{:.0},{},{},{:.4},{:.4},{:.4}",
                step + 1,
                play_eps, bid_eps, beta,
                play_buffer.size(), bid_buffer.size(),
                play_loss_avg, bid_loss_avg,
                total_episodes,
                nn_frac * 100.0, sps,
                play_transitions_total, bid_transitions_total,
                last_rand_wr, last_ckpt_wr, last_isdd_wr,
            ).ok();
            csv_file.flush().ok();

            play_loss_total = 0.0;
            play_loss_count = 0;
            bid_loss_total = 0.0;
            bid_loss_count = 0;
            last_log_time = Instant::now();
            last_log_step = step + 1;
        }

        // ========== EVALUATE ==========
        if (step + 1) % args.eval_freq == 0 {
            // Snapshot our joint bid NN for eval
            if let Ok(bw) = bid_trainer.snapshot_weights() {
                if let Ok(net) = BidNet::from_floats_with_layers(&bw, args.bid_hidden, BID_OBS_DIM, true, args.bid_layers) {
                    eval_our_bid_net = Some(net);
                }
            }

            let er = evaluate_play(
                &play_trainer, args.play_hidden, play_obs_dim,
                &mut eval_our_bid_net,
                &mut eval_opp_bid_net,
                args.eval_random_matches,
                args.eval_checkpoint_matches,
                &mut eval_play_baseline,
                &mut eval_belief_net,
            );

            last_rand_wr = er.rand_wr;
            last_ckpt_wr = er.ckpt_wr;
            last_isdd_wr = er.isdd_wr;

            let mut parts = Vec::new();
            if args.eval_random_matches > 0 {
                parts.push(format!("rand {:.0}%", er.rand_wr * 100.0));
            }
            if eval_play_baseline.is_some() && args.eval_checkpoint_matches > 0 {
                parts.push(format!("ckpt {:.0}%", er.ckpt_wr * 100.0));
            }
            if args.eval_isdd_matches > 0 {
                parts.push(format!("isdd {:.0}%", er.isdd_wr * 100.0));
            }
            parts.push(format!("play_trans {}", play_transitions_total));
            parts.push(format!("bid_trans {}", bid_transitions_total));
            println!("  [EVAL] {} ({:.0}s)", parts.join(" | "), er.elapsed);
        }

        // ========== CHECKPOINT ==========
        if (step + 1) % args.save_freq == 0 {
            std::fs::create_dir_all(&args.save_dir).ok();
            let step_label = step + 1;

            // Play NN
            let play_st = format!("{}/play_{}.safetensors", args.save_dir, step_label);
            let play_bin = format!("{}/play_{}.bin", args.save_dir, step_label);
            play_trainer.save_checkpoint(&play_st).ok();
            play_trainer.export_binary(&play_bin).ok();

            // Bid NN
            let bid_st = format!("{}/bid_{}.safetensors", args.save_dir, step_label);
            let bid_bin = format!("{}/bid_{}.bin", args.save_dir, step_label);
            bid_trainer.save_checkpoint(&bid_st).ok();
            bid_trainer.export_binary(&bid_bin).ok();

            // Latest aliases
            play_trainer.save_checkpoint(&format!("{}/play_latest.safetensors", args.save_dir)).ok();
            play_trainer.export_binary(&format!("{}/play_latest.bin", args.save_dir)).ok();
            bid_trainer.save_checkpoint(&format!("{}/bid_latest.safetensors", args.save_dir)).ok();
            bid_trainer.export_binary(&format!("{}/bid_latest.bin", args.save_dir)).ok();

            println!("  [SAVE] play_{} + bid_{}", step_label, step_label);
        }
    }

    // Final save
    std::fs::create_dir_all(&args.save_dir).ok();
    play_trainer.save_checkpoint(&format!("{}/play_final.safetensors", args.save_dir)).ok();
    play_trainer.export_binary(&format!("{}/play_final.bin", args.save_dir)).ok();
    bid_trainer.save_checkpoint(&format!("{}/bid_final.safetensors", args.save_dir)).ok();
    bid_trainer.export_binary(&format!("{}/bid_final.bin", args.save_dir)).ok();
    println!("\n=== Training complete ===");
    println!("Episodes: {}, Play trans: {}, Bid trans: {}",
        total_episodes, play_transitions_total, bid_transitions_total);
    println!("Saved to {}/{{play,bid}}_final.{{safetensors,bin}}", args.save_dir);
}
