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
//! Positions are fully driven by --seed: same seed = same deals, auctions and
//! stop points (stable benchmark); change the seed for a fresh draw.
//!
//! Usage:
//!   cargo run -p colver-core --bin bench_world_cred --release -- \
//!     --bid-positions 30 --play-positions 30 --worlds 12 --seed 42
//!
//! Référence (playgen v2 @60K, seed 42, 30+30 positions, 12 mondes) :
//!   Enchères : playgen 60%/92%  belief 34%/64%  uniform 12%/32%  (argmax/top3)
//!   Jeu      : playgen 85%/98%  belief 78%/96%  uniform 70%/94%

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::belief_net::BeliefNet;
use colver_core::belief_state::BeliefState;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs::{
    self, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2, BID_OBS_DIM_SCORE_AWARE_V3,
};
use colver_core::card::card_count;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{
    self, EnvTracking, OBS_DIM_TR,
};
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
            other => { eprintln!("unknown arg {}", other); std::process::exit(1); }
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut bid_net = BidNet::load_with_hidden(&bid_model, bid_hidden).expect("bid model");
    let bid_dim = bid_net.obs_dim();
    let mut bid_obs_buf = vec![0.0f32; bid_dim];
    let playgen = std::sync::Arc::new(PlaygenModel::load(&playgen_path).expect("playgen model"));
    let mut bid_belief = BeliefNet::load(&bid_belief_path)
        .or_else(|_| BeliefNet::load_with_hidden(&bid_belief_path, 256))
        .expect("bid belief net");

    println!(
        "bench_world_cred — seed {}, {} mondes/position, temp {}",
        seed, worlds, temperature
    );

    // =====================================================================
    // Phase 1: auctions
    // =====================================================================
    if bid_positions > 0 {
        let t0 = std::time::Instant::now();
        let mut tallies = [Tally::default(), Tally::default(), Tally::default()];
        let labels = ["playgen", "belief", "uniform"];
        let mut done = 0usize;

        while done < bid_positions {
            let dealer = rng.gen_range(0..4u8);
            let state0 = GameState::deal_random(dealer, &mut rng);
            let n_steps = rng.gen_range(2..5usize);
            let observer = rng.gen_range(0..4u8);

            // Self-play auction prefix with the bid net.
            let mut state = state0;
            let mut actions: Vec<(u8, u8)> = Vec::new();
            let mut ok = true;
            for _ in 0..n_steps {
                let p = state.current_player();
                write_bid_obs(bid_dim, &mut bid_obs_buf, &state, &actions);
                let a = bid_net.best_action_fast(&bid_obs_buf, state.legal_actions());
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

            // Sampler 1: playgen (auction completed by the bid head + playout).
            let mut search = IsDdSearch::new();
            search.set_playgen_model(playgen.clone());
            search.init_deal(&state0, observer, false);
            {
                let mut s = state0;
                for &(p, a) in &actions {
                    search.record_action(&s, p, a);
                    s.step(a);
                }
            }
            let pg_worlds = search.playgen_auction_deals(&state, worlds, temperature, &mut rng);

            // Sampler 2: bid belief net marginals → weighted determinize.
            let bel_worlds: Vec<[u32; 4]> = {
                let mut bs = BeliefState::new(observer, state0.hands[observer as usize]);
                let mut s = state0;
                for &(p, a) in &actions {
                    bs.record_bid(p, a, &s);
                    s.step(a);
                }
                bs.apply_nn_bid_beliefs(&mut bid_belief, &state, &actions);
                (0..worlds)
                    .filter_map(|_| bs.determinize(&state, &mut rng).map(|d| d.hands))
                    .collect()
            };

            // Sampler 3: uniform.
            let un_worlds = uniform_bid_worlds(&state0.hands, observer, worlds, &mut rng);

            for (si, ws) in [&pg_worlds, &bel_worlds, &un_worlds].iter().enumerate() {
                tallies[si].worlds += ws.len();
                tallies[si].missing += worlds - ws.len();
                for w in ws.iter() {
                    let mut s = GameState::new(dealer, *w);
                    let mut hist: Vec<(u8, u8)> = Vec::new();
                    for (ai, &(p, a)) in actions.iter().enumerate() {
                        if targets.contains(&ai) {
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
            done += 1;
        }
        println!(
            "\n== Enchères : {} positions, juge bid NN ({:.1}s) ==",
            done,
            t0.elapsed().as_secs_f64()
        );
        for (si, label) in labels.iter().enumerate() {
            tallies[si].print(label);
        }
    }

    // =====================================================================
    // Phase 2: play
    // =====================================================================
    if play_positions > 0 {
        let t0 = std::time::Instant::now();
        let mut dmc = DmcNet::load(&dmc_model).expect("dmc model");
        if dmc.obs_dim() == OBS_DIM_TR {
            dmc.set_residual(true);
        }
        let mut tallies = [Tally::default(), Tally::default(), Tally::default()];
        let labels = ["playgen", "belief", "uniform"];
        let mut done = 0usize;

        let mut dmc_play = |net: &mut DmcNet, state: &GameState, tracking: &EnvTracking| -> (u8, Vec<(u8, f32)>) {
            let legal = state.legal_actions() as u32;
            if net.obs_dim() == OBS_DIM_TR {
                let obs = dmc_obs::make_observation_tr(state, tracking);
                let order = dmc_obs::current_player_order(state, tracking);
                let mask = dmc_obs::cardset_to_canonical(legal, &order);
                let (best, q) = net.best_action(&obs, mask);
                let phys_q = q
                    .into_iter()
                    .map(|(c, v)| (dmc_obs::card_to_physical(c, &order), v))
                    .collect();
                (dmc_obs::card_to_physical(best, &order), phys_q)
            } else {
                panic!("legacy DMC obs not supported in this bench");
            }
        };

        while done < play_positions {
            let dealer = rng.gen_range(0..4u8);
            let state0 = GameState::deal_random(dealer, &mut rng);
            let observer = rng.gen_range(0..4u8);
            let stop_plays = rng.gen_range(6..25usize);

            // Search fed with the full history (playgen prefix + beliefs).
            let mut search = IsDdSearch::new();
            search.set_playgen_model(playgen.clone());
            let _ = search.load_belief_net(&play_belief_path);
            search.init_deal(&state0, observer, false);

            let mut state = state0;
            let mut tracking = EnvTracking::new();
            tracking.reset(dealer);
            // (player, action) full history; play targets judged later.
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
                        write_bid_obs(bid_dim, &mut bid_obs_buf, &state, &bid_actions);
                        bid_net.best_action_fast(&bid_obs_buf, state.legal_actions())
                    }
                    Phase::Playing => dmc_play(&mut dmc, &state, &tracking).0,
                    Phase::Done => { ok = false; break; }
                };
                search.record_action(&state, p, a);
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

            // Cards already played per seat (current trick included).
            let played_by = tracking.played_by;

            let cfg_belief = IsDdConfig { use_nn_beliefs: true, ..Default::default() };
            let cfg_uniform = IsDdConfig::default();
            let pg_worlds = search.playgen_worlds(&state, worlds, temperature, &mut rng);
            let bel_worlds =
                search.sample_worlds(&state, &cfg_belief, observer, worlds, true, &mut rng);
            let un_worlds =
                search.sample_worlds(&state, &cfg_uniform, observer, worlds, false, &mut rng);

            for (si, ws) in [&pg_worlds, &bel_worlds, &un_worlds].iter().enumerate() {
                tallies[si].worlds += ws.len();
                tallies[si].missing += worlds - ws.len();
                'world: for w in ws.iter() {
                    // Reconstruct full initial hands: remaining ∪ already played.
                    let mut init_hands = [0u32; 4];
                    for p in 0..4usize {
                        init_hands[p] = w[p] | played_by[p];
                        if card_count(init_hands[p]) != 8 {
                            continue 'world; // inconsistent world
                        }
                    }
                    let mut s = GameState::new(dealer, init_hands);
                    let mut jt = EnvTracking::new();
                    jt.reset(dealer);
                    for &(p, a) in &history {
                        if s.phase == Phase::Playing && p != observer {
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
            done += 1;
        }
        println!(
            "\n== Jeu : {} positions, juge DMC ({:.1}s) ==",
            done,
            t0.elapsed().as_secs_f64()
        );
        for (si, label) in labels.iter().enumerate() {
            tallies[si].print(label);
        }
    }
}
