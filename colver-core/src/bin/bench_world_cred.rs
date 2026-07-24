//! World-credibility benchmark: how consistent are sampled hidden-hand worlds
//! with the actions actually observed?
//!
//! Self-supervised (idea: user, 2026-07-23): the observed auction/plays serve
//! as the oracle. For each seeded position, sample K worlds per sampler, then
//! ask the reference policy whether it would replay each observed hidden
//! action holding that world's hand for the actor.
//!
//! - Bid phase: judge = bid NN on every non-pass bid by a hidden player.
//! - Play phase: judge = DMC net (DouDou50) on every play by a hidden player.
//!
//! Samplers: playgen (COLVPG02), belief NN (bid_belief for auctions, play
//! belief net for plays), constraint-uniform.
//!
//! **Two-phase design (2026-07-23).** All positions are generated *first*,
//! from a dedicated RNG, and only then answered. Positions depend on the bid
//! net and the DMC net alone — never on a sampler. Each (position, sampler)
//! pair then draws from its own `sub_rng` stream, so a sampler that consumes
//! a different amount of randomness (a new playgen checkpoint, say) cannot
//! shift any other sampler's worlds, nor the positions themselves.
//!
//! The earlier single-loop version interleaved position drawing and world
//! sampling on one shared RNG: changing the playgen checkpoint desynced the
//! stream and silently re-drew every position after the first, moving the
//! untouched belief/uniform baselines by up to 5 pp. Any pre-2026-07-23
//! cross-checkpoint comparison from this bench is void. Baselines that do not
//! depend on the varied component must now be bit-identical across runs —
//! that is the built-in control.
//!
//! Usage:
//!   cargo run -p colver-core --bin bench_world_cred --release -- \
//!     --bid-positions 30 --play-positions 30 --worlds 12 --seed 42
//!
//! `--gpu` (requires `--features dmc_train`) runs playgen generation on CUDA,
//! for both the auction and the play path.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::belief_net::BeliefNet;
use colver_core::playgen::analysis::PlaygenAnalyst;
use colver_core::belief_state::BeliefState;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{
    self, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2, BID_OBS_DIM_SCORE_AWARE_V3,
};
use colver_core::card::card_count;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use colver_core::is_dd::{IsDdConfig, IsDdSearch};
use colver_core::playgen::infer::PlaygenModel;
use colver_core::state::{GameState, Phase};

#[derive(Default, Clone)]
struct Tally {
    argmax: usize,
    top3: usize,
    n: usize,
    worlds: usize,
    missing: usize,
}

impl Tally {
    fn print(&self, label: &str) {
        let n = self.n.max(1);
        println!(
            "  {:8}: argmax {:>3.0}%  top3 {:>3.0}%  ({} jugements, {} mondes, {} manquants)",
            label,
            self.argmax as f64 / n as f64 * 100.0,
            self.top3 as f64 / n as f64 * 100.0,
            self.n,
            self.worlds,
            self.missing,
        );
    }
}

/// Independent RNG stream per (phase, position, sampler).
///
/// The whole point is that no stream can be perturbed by another: adding a
/// sampler, reordering them, or swapping a model must leave every other
/// stream untouched.
fn sub_rng(seed: u64, phase: u64, pos: usize, sampler: usize) -> StdRng {
    let mut h = seed
        ^ phase.wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (pos as u64).wrapping_mul(0xBF58_476D_1CE4_E5B9)
        ^ (sampler as u64).wrapping_mul(0x94D0_49BB_1331_11EB);
    // splitmix64 finalizer — decorrelate neighbouring (pos, sampler) pairs.
    h ^= h >> 30;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 27;
    h = h.wrapping_mul(0x94D0_49BB_1331_11EB);
    h ^= h >> 31;
    StdRng::seed_from_u64(h)
}

fn write_bid_obs(net_obs_dim: usize, obs: &mut [f32], s: &GameState, hist: &[(u8, u8)]) {
    match net_obs_dim {
        BID_OBS_DIM_SCORE_AWARE_V3 => {
            bid_obs::write_bid_observation_score_aware_v3(obs, 0, s, hist, 0, 0)
        }
        BID_OBS_DIM_SCORE_AWARE_V2 => {
            bid_obs::write_bid_observation_score_aware_v2(obs, 0, s, hist, 0, 0)
        }
        BID_OBS_DIM_SCORE_AWARE => {
            bid_obs::write_bid_observation_score_aware(obs, 0, s, hist, 0, 0)
        }
        _ => bid_obs::write_bid_observation(obs, 0, s, hist),
    }
}

/// Rank of `action` among legal actions by Q (0 = argmax).
fn rank_of(q: &[f32], legal: u64, action: u8) -> usize {
    let qa = q[action as usize];
    (0..q.len() as u8)
        .filter(|&c| c != action && legal & (1u64 << c) != 0 && q[c as usize] > qa)
        .count()
}

fn uniform_bid_worlds(
    hands: &[u32; 4],
    observer: u8,
    k: usize,
    rng: &mut StdRng,
) -> Vec<[u32; 4]> {
    let obs_hand = hands[observer as usize];
    let mut rest: Vec<u8> = (0..32u8).filter(|&c| obs_hand & (1 << c) == 0).collect();
    let others: Vec<u8> = (0..4u8).filter(|&p| p != observer).collect();
    (0..k)
        .map(|_| {
            for i in (1..rest.len()).rev() {
                let j = rng.gen_range(0..=i);
                rest.swap(i, j);
            }
            let mut w = [0u32; 4];
            w[observer as usize] = obs_hand;
            for (j, &p) in others.iter().enumerate() {
                for &c in &rest[j * 8..(j + 1) * 8] {
                    w[p as usize] |= 1 << c;
                }
            }
            w
        })
        .collect()
}

fn dmc_play(net: &mut DmcNet, state: &GameState, tracking: &EnvTracking) -> (u8, Vec<(u8, f32)>) {
    let legal = state.legal_actions() as u32;
    if net.obs_dim() != OBS_DIM_TR {
        panic!("legacy DMC obs not supported in this bench");
    }
    let obs = dmc_obs::make_observation_tr(state, tracking);
    let order = dmc_obs::current_player_order(state, tracking);
    let mask = dmc_obs::cardset_to_canonical(legal, &order);
    let (best, q) = net.best_action(&obs, mask);
    let phys_q = q
        .into_iter()
        .map(|(c, v)| (dmc_obs::card_to_physical(c, &order), v))
        .collect();
    (dmc_obs::card_to_physical(best, &order), phys_q)
}

/// A mid-auction position: the deal, the observed prefix, and which of those
/// bid entries are hidden-player bids to be judged.
struct BidPosition {
    dealer: u8,
    state0: GameState,
    observer: u8,
    actions: Vec<(u8, u8)>,
    targets: Vec<usize>,
    state: GameState,
}

/// A mid-play position: the deal and the full observed history up to the stop
/// point. `played_by` lets a sampled world be completed back into full hands.
struct PlayPosition {
    dealer: u8,
    state0: GameState,
    observer: u8,
    history: Vec<(u8, u8)>,
    played_by: [u32; 4],
    state: GameState,
}

/// Phase 0a — draw auction positions. Depends on the bid net only.
fn generate_bid_positions(
    n: usize,
    rng: &mut StdRng,
    bid_net: &mut BidNet,
    bid_dim: usize,
    buf: &mut [f32],
) -> (Vec<BidPosition>, usize) {
    let mut out = Vec::with_capacity(n);
    let mut drawn = 0usize;
    while out.len() < n {
        drawn += 1;
        let dealer = rng.gen_range(0..4u8);
        let state0 = GameState::deal_random(dealer, rng);
        let n_steps = rng.gen_range(2..5usize);
        let observer = rng.gen_range(0..4u8);

        let mut state = state0;
        let mut actions: Vec<(u8, u8)> = Vec::new();
        let mut ok = true;
        for _ in 0..n_steps {
            let p = state.current_player();
            write_bid_obs(bid_dim, buf, &state, &actions);
            let a = bid_net.best_action_fast(buf, state.legal_actions());
            actions.push((p, a));
            state.step(a);
            if state.phase != Phase::Bidding {
                ok = false;
                break;
            }
        }
        let targets: Vec<usize> = actions
            .iter()
            .enumerate()
            .filter(|(_, &(p, a))| a > 0 && p != observer)
            .map(|(i, _)| i)
            .collect();
        if !ok || targets.is_empty() {
            continue;
        }
        out.push(BidPosition { dealer, state0, observer, actions, targets, state });
    }
    (out, drawn)
}

/// Phase 0b — draw play positions. Depends on the bid net and DMC net only.
fn generate_play_positions(
    n: usize,
    rng: &mut StdRng,
    bid_net: &mut BidNet,
    bid_dim: usize,
    buf: &mut [f32],
    dmc: &mut DmcNet,
) -> (Vec<PlayPosition>, usize) {
    let mut out = Vec::with_capacity(n);
    let mut drawn = 0usize;
    while out.len() < n {
        drawn += 1;
        let dealer = rng.gen_range(0..4u8);
        let state0 = GameState::deal_random(dealer, rng);
        let observer = rng.gen_range(0..4u8);
        let stop_plays = rng.gen_range(6..25usize);

        let mut state = state0;
        let mut tracking = EnvTracking::new();
        tracking.reset(dealer);
        let mut history: Vec<(u8, u8)> = Vec::new();
        let mut bid_actions: Vec<(u8, u8)> = Vec::new();
        let mut plays_done = 0usize;
        let mut ok = true;

        while plays_done < stop_plays {
            if state.is_terminal() {
                ok = false;
                break;
            }
            let p = state.current_player();
            let a = match state.phase {
                Phase::Bidding => {
                    write_bid_obs(bid_dim, buf, &state, &bid_actions);
                    bid_net.best_action_fast(buf, state.legal_actions())
                }
                Phase::Playing => dmc_play(dmc, &state, &tracking).0,
                Phase::Done => {
                    ok = false;
                    break;
                }
            };
            tracking.track_action(&state, a);
            if state.phase == Phase::Bidding {
                bid_actions.push((p, a));
            } else {
                plays_done += 1;
            }
            history.push((p, a));
            state.step(a);
            if state.phase == Phase::Done && plays_done < stop_plays {
                ok = false; // void deal
                break;
            }
        }
        if !ok || state.phase != Phase::Playing {
            continue;
        }
        out.push(PlayPosition {
            dealer,
            state0,
            observer,
            history,
            played_by: tracking.played_by,
            state,
        });
    }
    (out, drawn)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut bid_positions = 30usize;
    let mut play_positions = 30usize;
    let mut worlds = 12usize;
    let mut seed = 42u64;
    let mut temperature = 1.0f32;
    let mut bid_model = String::from("models/bid_v6_isdd_resume/bid_nn_final.bin");
    let mut bid_hidden = 512usize;
    let mut dmc_model = String::from("models/play_v2/play_final.bin");
    let mut playgen_path = String::from("models/playgen_v2/playgen_v2_half.bin");
    let mut bid_belief_path = String::from("models/bid_belief_v4.bin");
    let mut play_belief_path = String::from("models/belief_v4_fix_v2.bin");
    let mut use_gpu = cfg!(feature = "dmc_train");

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--bid-positions" => { bid_positions = args[i + 1].parse().unwrap(); i += 2; }
            "--play-positions" => { play_positions = args[i + 1].parse().unwrap(); i += 2; }
            "--worlds" => { worlds = args[i + 1].parse().unwrap(); i += 2; }
            "--seed" => { seed = args[i + 1].parse().unwrap(); i += 2; }
            "--temperature" => { temperature = args[i + 1].parse().unwrap(); i += 2; }
            "--bid-model" => { bid_model = args[i + 1].clone(); i += 2; }
            "--bid-hidden" => { bid_hidden = args[i + 1].parse().unwrap(); i += 2; }
            "--dmc-model" => { dmc_model = args[i + 1].clone(); i += 2; }
            "--playgen" => { playgen_path = args[i + 1].clone(); i += 2; }
            "--bid-belief" => { bid_belief_path = args[i + 1].clone(); i += 2; }
            "--play-belief" => { play_belief_path = args[i + 1].clone(); i += 2; }
            "--gpu" => { use_gpu = true; i += 1; }
            "--cpu" => { use_gpu = false; i += 1; }
            other => { eprintln!("unknown arg {}", other); std::process::exit(1); }
        }
    }

    let mut bid_net = BidNet::load_with_hidden(&bid_model, bid_hidden).expect("bid model");
    let bid_dim = bid_net.obs_dim();
    let mut bid_obs_buf = vec![0.0f32; bid_dim];
    let playgen = std::sync::Arc::new(PlaygenModel::load(&playgen_path).expect("playgen model"));
    let mut dmc = DmcNet::load(&dmc_model).expect("dmc model");
    if dmc.obs_dim() == OBS_DIM_TR {
        dmc.set_residual(true);
    }

    #[cfg(feature = "dmc_train")]
    let gpu: Option<colver_core::playgen::gpu::GpuPlaygen> = if use_gpu {
        let dev = candle_core::Device::new_cuda(0).expect("CUDA device");
        Some(colver_core::playgen::gpu::GpuPlaygen::new(&playgen, dev).expect("GPU model"))
    } else {
        None
    };
    #[cfg(not(feature = "dmc_train"))]
    if use_gpu {
        eprintln!("--gpu nécessite --features dmc_train");
        std::process::exit(1);
    }

    println!(
        "bench_world_cred — seed {}, {} mondes/position, temp {}, playgen {}",
        seed,
        worlds,
        temperature,
        if use_gpu { "GPU" } else { "CPU" },
    );

    // =====================================================================
    // Phase 0: draw every position up front, from a dedicated RNG.
    // =====================================================================
    // One stream per phase: `--bid-positions 0` must not shift the play
    // positions, so the two generators never share an RNG either.
    let t_gen = std::time::Instant::now();
    let mut bid_pos_rng = sub_rng(seed, 0, 0, 0);
    let mut play_pos_rng = sub_rng(seed, 0, 0, 1);
    let (bid_pos, bid_drawn) = generate_bid_positions(
        bid_positions, &mut bid_pos_rng, &mut bid_net, bid_dim, &mut bid_obs_buf,
    );
    let (play_pos, play_drawn) = generate_play_positions(
        play_positions, &mut play_pos_rng, &mut bid_net, bid_dim, &mut bid_obs_buf, &mut dmc,
    );
    println!(
        "  positions : {} enchères ({} donnes tirées), {} jeu ({} tirées) en {:.1}s",
        bid_pos.len(), bid_drawn, play_pos.len(), play_drawn, t_gen.elapsed().as_secs_f64(),
    );

    // =====================================================================
    // Phase 1: auctions
    // =====================================================================
    if !bid_pos.is_empty() {
        let t0 = std::time::Instant::now();
        let mut tallies = [Tally::default(), Tally::default(), Tally::default()];
        let labels = ["playgen", "belief", "uniform"];
        let mut bid_belief = BeliefNet::load(&bid_belief_path)
            .or_else(|_| BeliefNet::load_with_hidden(&bid_belief_path, 256))
            .expect("bid belief net");

        for (pi, pos) in bid_pos.iter().enumerate() {
            let mut search = IsDdSearch::new();
            let mut analyst = PlaygenAnalyst::new(playgen.clone());
            search.init_deal(&pos.state0, pos.observer, false);
            analyst.init_deal(&pos.state0, pos.observer);
            {
                let mut s = pos.state0;
                for &(p, a) in &pos.actions {
                    search.record_action(&s, p, a);
                    analyst.observe(&s, p, a);
                    s.step(a);
                }
            }

            // Sampler 1: playgen (auction completed by the bid head + playout).
            let mut r0 = sub_rng(seed, 1, pi, 0);
            #[cfg(feature = "dmc_train")]
            let pg_worlds: Vec<[u32; 4]> = match &gpu {
                Some(g) => {
                    let sampler = analyst.sampler();
                    g.generate_deals_from_auction_scored(
                        &sampler.prefix_tokens(),
                        &pos.state,
                        sampler.observer(),
                        sampler.observer_hand(),
                        sampler.bid_entries_count(),
                        worlds,
                        temperature,
                        &mut r0,
                    )
                    .expect("GPU generation")
                    .into_iter()
                    .map(|(h, _)| h)
                    .collect()
                }
                None => analyst.auction_deals(&pos.state, worlds, temperature, &mut r0),
            };
            #[cfg(not(feature = "dmc_train"))]
            let pg_worlds = analyst.auction_deals(&pos.state, worlds, temperature, &mut r0);

            // Sampler 2: bid belief net marginals → weighted determinize.
            let mut r1 = sub_rng(seed, 1, pi, 1);
            let bel_worlds: Vec<[u32; 4]> = {
                let mut bs = BeliefState::new(pos.observer, pos.state0.hands[pos.observer as usize]);
                let mut s = pos.state0;
                for &(p, a) in &pos.actions {
                    bs.record_bid(p, a, &s);
                    s.step(a);
                }
                bs.apply_nn_bid_beliefs(&mut bid_belief, &pos.state, &pos.actions);
                (0..worlds)
                    .filter_map(|_| bs.determinize(&pos.state, &mut r1).map(|d| d.hands))
                    .collect()
            };

            // Sampler 3: uniform.
            let mut r2 = sub_rng(seed, 1, pi, 2);
            let un_worlds = uniform_bid_worlds(&pos.state0.hands, pos.observer, worlds, &mut r2);

            for (si, ws) in [&pg_worlds, &bel_worlds, &un_worlds].iter().enumerate() {
                tallies[si].worlds += ws.len();
                tallies[si].missing += worlds - ws.len();
                for w in ws.iter() {
                    let mut s = GameState::new(pos.dealer, *w);
                    let mut hist: Vec<(u8, u8)> = Vec::new();
                    for (ai, &(p, a)) in pos.actions.iter().enumerate() {
                        if pos.targets.contains(&ai) {
                            write_bid_obs(bid_dim, &mut bid_obs_buf, &s, &hist);
                            let q = bid_net.evaluate(&bid_obs_buf);
                            let r = rank_of(&q, s.legal_actions(), a);
                            tallies[si].argmax += (r == 0) as usize;
                            tallies[si].top3 += (r < 3) as usize;
                            tallies[si].n += 1;
                        }
                        hist.push((p, a));
                        s.step(a);
                    }
                }
            }
        }
        println!(
            "\n== Enchères : {} positions, juge bid NN ({:.1}s) ==",
            bid_pos.len(),
            t0.elapsed().as_secs_f64()
        );
        for (si, label) in labels.iter().enumerate() {
            tallies[si].print(label);
        }
    }

    // =====================================================================
    // Phase 2: play
    // =====================================================================
    if !play_pos.is_empty() {
        let t0 = std::time::Instant::now();
        let mut tallies = [Tally::default(), Tally::default(), Tally::default()];
        let labels = ["playgen", "belief", "uniform"];

        for (pi, pos) in play_pos.iter().enumerate() {
            // Search fed with the full history (playgen prefix + beliefs).
            let mut search = IsDdSearch::new();
            let mut analyst = PlaygenAnalyst::new(playgen.clone());
            let _ = search.load_belief_net(&play_belief_path);
            search.init_deal(&pos.state0, pos.observer, false);
            analyst.init_deal(&pos.state0, pos.observer);
            {
                let mut s = pos.state0;
                for &(p, a) in &pos.history {
                    search.record_action(&s, p, a);
                    analyst.observe(&s, p, a);
                    s.step(a);
                }
            }

            let cfg_belief = IsDdConfig { use_nn_beliefs: true, ..Default::default() };
            let cfg_uniform = IsDdConfig::default();
            let mut r0 = sub_rng(seed, 2, pi, 0);
            let mut r1 = sub_rng(seed, 2, pi, 1);
            let mut r2 = sub_rng(seed, 2, pi, 2);
            #[cfg(feature = "dmc_train")]
            let pg_worlds: Vec<[u32; 4]> = match &gpu {
                Some(g) => g
                    .generate_worlds_scored(
                        analyst.sampler(),
                        &pos.state,
                        worlds,
                        temperature,
                        &mut r0,
                    )
                    .expect("GPU generation")
                    .into_iter()
                    .map(|(h, _)| h)
                    .collect(),
                None => analyst.play_worlds(&pos.state, worlds, temperature, &mut r0),
            };
            #[cfg(not(feature = "dmc_train"))]
            let pg_worlds = analyst.play_worlds(&pos.state, worlds, temperature, &mut r0);
            let bel_worlds =
                search.sample_worlds(&pos.state, &cfg_belief, pos.observer, worlds, true, &mut r1);
            let un_worlds =
                search.sample_worlds(&pos.state, &cfg_uniform, pos.observer, worlds, false, &mut r2);

            for (si, ws) in [&pg_worlds, &bel_worlds, &un_worlds].iter().enumerate() {
                tallies[si].worlds += ws.len();
                tallies[si].missing += worlds - ws.len();
                'world: for w in ws.iter() {
                    // Reconstruct full initial hands: remaining ∪ already played.
                    let mut init_hands = [0u32; 4];
                    for p in 0..4usize {
                        init_hands[p] = w[p] | pos.played_by[p];
                        if card_count(init_hands[p]) != 8 {
                            continue 'world; // inconsistent world
                        }
                    }
                    let mut s = GameState::new(pos.dealer, init_hands);
                    let mut jt = EnvTracking::new();
                    jt.reset(pos.dealer);
                    for &(p, a) in &pos.history {
                        if s.phase == Phase::Playing && p != pos.observer {
                            let (_, q) = dmc_play(&mut dmc, &s, &jt);
                            let qa = q.iter().find(|(c, _)| *c == a).map(|(_, v)| *v);
                            if let Some(qa) = qa {
                                let better =
                                    q.iter().filter(|(c, v)| *c != a && *v > qa).count();
                                tallies[si].argmax += (better == 0) as usize;
                                tallies[si].top3 += (better < 3) as usize;
                                tallies[si].n += 1;
                            }
                        }
                        jt.track_action(&s, a);
                        s.step(a);
                    }
                }
            }
        }
        println!(
            "\n== Jeu : {} positions, juge DMC ({:.1}s) ==",
            play_pos.len(),
            t0.elapsed().as_secs_f64()
        );
        for (si, label) in labels.iter().enumerate() {
            tallies[si].print(label);
        }
    }
}
