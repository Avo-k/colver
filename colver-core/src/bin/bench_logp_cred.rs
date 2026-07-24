//! Est-ce que le log p cumulé pendant la génération playgen prédit la
//! crédibilité-feuille des mondes échantillonnés ?
//!
//! Hypothèse à tester (pré-requis du pruning de branches pendant la
//! génération) : les mondes dont la continuation a un log p élevé sous le
//! modèle sont aussi ceux que le juge (bid NN / DMC) trouve crédibles.
//!
//! Protocole : mêmes positions seedées que `bench_world_cred`, mais un seul
//! sampler (playgen) et K mondes par position, chacun annoté de son log p
//! cumulé (variantes `_scored`). Par position : Spearman(log p / token,
//! crédibilité-juge). Pool global : terciles intra-position par log p →
//! argmax% / top3% / MRR du juge par tercile.
//!
//! Usage:
//!   cargo run -p colver-core --bin bench_logp_cred --release -- \
//!     --bid-positions 30 --play-positions 30 --worlds 32 --seed 42

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::bid_net::BidNet;
use colver_core::playgen::analysis::PlaygenAnalyst;
use colver_core::bid_obs::{
    self, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2, BID_OBS_DIM_SCORE_AWARE_V3,
};
use colver_core::card::card_count;
use colver_core::dmc_net::DmcNet;
use colver_core::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use colver_core::is_dd::IsDdSearch;
use colver_core::playgen::infer::PlaygenModel;
use colver_core::state::{GameState, Phase};

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

/// Average ranks (ties → mean rank).
fn ranks(v: &[f64]) -> Vec<f64> {
    let n = v.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&a, &b| v[a].partial_cmp(&v[b]).unwrap());
    let mut r = vec![0.0; n];
    let mut i = 0;
    while i < n {
        let mut j = i;
        while j + 1 < n && v[idx[j + 1]] == v[idx[i]] {
            j += 1;
        }
        let avg = (i + j) as f64 / 2.0 + 1.0;
        for &k in &idx[i..=j] {
            r[k] = avg;
        }
        i = j + 1;
    }
    r
}

fn pearson(x: &[f64], y: &[f64]) -> Option<f64> {
    let n = x.len() as f64;
    let mx = x.iter().sum::<f64>() / n;
    let my = y.iter().sum::<f64>() / n;
    let (mut sxy, mut sxx, mut syy) = (0.0, 0.0, 0.0);
    for i in 0..x.len() {
        let (dx, dy) = (x[i] - mx, y[i] - my);
        sxy += dx * dy;
        sxx += dx * dx;
        syy += dy * dy;
    }
    if sxx <= 0.0 || syy <= 0.0 {
        return None; // constant vector → correlation undefined
    }
    Some(sxy / (sxx * syy).sqrt())
}

fn spearman(x: &[f64], y: &[f64]) -> Option<f64> {
    if x.len() < 6 {
        return None;
    }
    pearson(&ranks(x), &ranks(y))
}

/// One sampled world's measurements at a position.
struct WorldPoint {
    /// Predictor values (one per predictor, log p / token).
    preds: Vec<f64>,
    /// Judge credibility: mean reciprocal rank over judged actions.
    mrr: f64,
    argmax: usize,
    top3: usize,
    n_judged: usize,
}

#[derive(Default, Clone)]
struct TercileTally {
    argmax: usize,
    top3: usize,
    n: usize,
    mrr_sum: f64,
    worlds: usize,
}

/// Per-predictor analysis over positions: within-position Spearman +
/// pooled tercile tallies (low/mid/high by predictor value).
struct PredictorStats {
    name: &'static str,
    spearmans: Vec<f64>,
    terciles: [TercileTally; 3],
}

impl PredictorStats {
    fn new(name: &'static str) -> Self {
        PredictorStats { name, spearmans: Vec::new(), terciles: Default::default() }
    }

    fn add_position(&mut self, pi: usize, points: &[WorldPoint]) {
        let x: Vec<f64> = points.iter().map(|p| p.preds[pi]).collect();
        let y: Vec<f64> = points.iter().map(|p| p.mrr).collect();
        if let Some(rho) = spearman(&x, &y) {
            self.spearmans.push(rho);
        }
        // Terciles by predictor value within this position.
        let mut idx: Vec<usize> = (0..points.len()).collect();
        idx.sort_by(|&a, &b| x[a].partial_cmp(&x[b]).unwrap());
        let n = idx.len();
        for (r, &wi) in idx.iter().enumerate() {
            let t = (r * 3 / n).min(2);
            let p = &points[wi];
            self.terciles[t].argmax += p.argmax;
            self.terciles[t].top3 += p.top3;
            self.terciles[t].n += p.n_judged;
            self.terciles[t].mrr_sum += p.mrr;
            self.terciles[t].worlds += 1;
        }
    }

    fn print(&self) {
        let ns = self.spearmans.len().max(1) as f64;
        let mean = self.spearmans.iter().sum::<f64>() / ns;
        let var = self.spearmans.iter().map(|r| (r - mean) * (r - mean)).sum::<f64>() / ns;
        println!(
            "  {:16} Spearman intra-position: {:+.3} ± {:.3} ({} positions)",
            self.name,
            mean,
            var.sqrt(),
            self.spearmans.len()
        );
        for (t, label) in ["bas", "moyen", "haut"].iter().enumerate() {
            let tt = &self.terciles[t];
            let n = tt.n.max(1) as f64;
            println!(
                "    tercile {:5}: argmax {:>4.1}%  top3 {:>4.1}%  MRR {:.3}  ({} mondes)",
                label,
                tt.argmax as f64 / n * 100.0,
                tt.top3 as f64 / n * 100.0,
                tt.mrr_sum / tt.worlds.max(1) as f64,
                tt.worlds,
            );
        }
    }
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut bid_positions = 30usize;
    let mut play_positions = 30usize;
    let mut worlds = 32usize;
    let mut seed = 42u64;
    let mut temperature = 1.0f32;
    let mut bid_model = String::from("models/bid_v6_isdd_resume/bid_nn_final.bin");
    let mut bid_hidden = 512usize;
    let mut dmc_model = String::from("models/play_v2/play_final.bin");
    let mut playgen_path = String::from("models/playgen_v2/playgen_v2_half.bin");

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
            other => { eprintln!("unknown arg {}", other); std::process::exit(1); }
        }
    }

    let mut rng = StdRng::seed_from_u64(seed);
    let mut bid_net = BidNet::load_with_hidden(&bid_model, bid_hidden).expect("bid model");
    let bid_dim = bid_net.obs_dim();
    let mut bid_obs_buf = vec![0.0f32; bid_dim];
    let playgen = std::sync::Arc::new(PlaygenModel::load(&playgen_path).expect("playgen model"));

    println!(
        "bench_logp_cred — seed {}, {} mondes/position, temp {}",
        seed, worlds, temperature
    );

    // =====================================================================
    // Phase 1: auctions — predictors: log p enchères / token, log p total
    // =====================================================================
    if bid_positions > 0 {
        let t0 = std::time::Instant::now();
        let mut stats = [
            PredictorStats::new("logp enchères"),
            PredictorStats::new("logp total"),
        ];
        let mut done = 0usize;

        while done < bid_positions {
            let dealer = rng.gen_range(0..4u8);
            let state0 = GameState::deal_random(dealer, &mut rng);
            let n_steps = rng.gen_range(2..5usize);
            let observer = rng.gen_range(0..4u8);

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

            let mut search = IsDdSearch::new();
            let mut analyst = PlaygenAnalyst::new(playgen.clone());
            search.init_deal(&state0, observer, false);
            analyst.init_deal(&state0, observer);
            {
                let mut s = state0;
                for &(p, a) in &actions {
                    search.record_action(&s, p, a);
                    analyst.observe(&s, p, a);
                    s.step(a);
                }
            }
            let scored = analyst.auction_deals_scored(&state, worlds, temperature, &mut rng);
            if scored.len() < 9 {
                continue; // too few worlds for terciles/Spearman
            }

            let points: Vec<WorldPoint> = scored
                .iter()
                .map(|(w, lp)| {
                    let bid_lp = lp.bid_sum as f64 / lp.bid_n.max(1) as f64;
                    let tot_lp = (lp.bid_sum + lp.play_sum) as f64
                        / (lp.bid_n + lp.play_n).max(1) as f64;
                    let mut s = GameState::new(dealer, *w);
                    let mut hist: Vec<(u8, u8)> = Vec::new();
                    let (mut mrr, mut argmax, mut top3, mut n_judged) = (0.0, 0, 0, 0);
                    for (ai, &(p, a)) in actions.iter().enumerate() {
                        if targets.contains(&ai) {
                            write_bid_obs(bid_dim, &mut bid_obs_buf, &s, &hist);
                            let q = bid_net.evaluate(&bid_obs_buf);
                            let r = rank_of(&q, s.legal_actions(), a);
                            mrr += 1.0 / (1.0 + r as f64);
                            argmax += (r == 0) as usize;
                            top3 += (r < 3) as usize;
                            n_judged += 1;
                        }
                        hist.push((p, a));
                        s.step(a);
                    }
                    WorldPoint {
                        preds: vec![bid_lp, tot_lp],
                        mrr: mrr / n_judged.max(1) as f64,
                        argmax,
                        top3,
                        n_judged,
                    }
                })
                .collect();

            for (pi, st) in stats.iter_mut().enumerate() {
                st.add_position(pi, &points);
            }
            done += 1;
        }
        println!(
            "\n== Enchères : {} positions, juge bid NN ({:.1}s) ==",
            done,
            t0.elapsed().as_secs_f64()
        );
        for st in &stats {
            st.print();
        }
    }

    // =====================================================================
    // Phase 2: play — predictors: log p complet / token, log p 1re moitié
    // =====================================================================
    if play_positions > 0 {
        let t0 = std::time::Instant::now();
        let mut dmc = DmcNet::load(&dmc_model).expect("dmc model");
        if dmc.obs_dim() == OBS_DIM_TR {
            dmc.set_residual(true);
        }
        let mut stats = [
            PredictorStats::new("logp complet"),
            PredictorStats::new("logp 1re moitié"),
        ];
        let mut done = 0usize;

        let dmc_play = |net: &mut DmcNet, state: &GameState, tracking: &EnvTracking| -> (u8, Vec<(u8, f32)>) {
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

            let mut search = IsDdSearch::new();
            let mut analyst = PlaygenAnalyst::new(playgen.clone());
            search.init_deal(&state0, observer, false);
            analyst.init_deal(&state0, observer);

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
                        write_bid_obs(bid_dim, &mut bid_obs_buf, &state, &bid_actions);
                        bid_net.best_action_fast(&bid_obs_buf, state.legal_actions())
                    }
                    Phase::Playing => dmc_play(&mut dmc, &state, &tracking).0,
                    Phase::Done => { ok = false; break; }
                };
                search.record_action(&state, p, a);
                analyst.observe(&state, p, a);
                tracking.track_action(&state, a);
                if state.phase == Phase::Bidding {
                    bid_actions.push((p, a));
                } else {
                    plays_done += 1;
                }
                history.push((p, a));
                state.step(a);
                if state.phase == Phase::Done && plays_done < stop_plays {
                    ok = false;
                    break;
                }
            }
            if !ok || state.phase != Phase::Playing {
                continue;
            }

            let played_by = tracking.played_by;
            let scored = analyst.play_worlds_scored(&state, worlds, temperature, &mut rng);
            if scored.len() < 9 {
                continue;
            }

            let mut points: Vec<WorldPoint> = Vec::with_capacity(scored.len());
            'world: for (w, lp) in scored.iter() {
                let mut init_hands = [0u32; 4];
                for p in 0..4usize {
                    init_hands[p] = w[p] | played_by[p];
                    if card_count(init_hands[p]) != 8 {
                        continue 'world;
                    }
                }
                let full_lp = lp.sum as f64 / lp.n.max(1) as f64;
                let half_lp = if lp.half_n > 0 {
                    lp.half_sum as f64 / lp.half_n as f64
                } else {
                    full_lp
                };
                let mut s = GameState::new(dealer, init_hands);
                let mut jt = EnvTracking::new();
                jt.reset(dealer);
                let (mut mrr, mut argmax, mut top3, mut n_judged) = (0.0, 0, 0, 0);
                for &(p, a) in &history {
                    if s.phase == Phase::Playing && p != observer {
                        let (_, q) = dmc_play(&mut dmc, &s, &jt);
                        if let Some(qa) = q.iter().find(|(c, _)| *c == a).map(|(_, v)| *v) {
                            let better = q.iter().filter(|(c, v)| *c != a && *v > qa).count();
                            mrr += 1.0 / (1.0 + better as f64);
                            argmax += (better == 0) as usize;
                            top3 += (better < 3) as usize;
                            n_judged += 1;
                        }
                    }
                    jt.track_action(&s, a);
                    s.step(a);
                }
                points.push(WorldPoint {
                    preds: vec![full_lp, half_lp],
                    mrr: mrr / n_judged.max(1) as f64,
                    argmax,
                    top3,
                    n_judged,
                });
            }
            if points.len() < 9 {
                continue;
            }

            for (pi, st) in stats.iter_mut().enumerate() {
                st.add_position(pi, &points);
            }
            done += 1;
        }
        println!(
            "\n== Jeu : {} positions, juge DMC ({:.1}s) ==",
            done,
            t0.elapsed().as_secs_f64()
        );
        for st in &stats {
            st.print();
        }
    }
}
