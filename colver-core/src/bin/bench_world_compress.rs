//! Compression de posterior : N mondes playgen → K atomes pondérés.
//!
//! Question (utilisateur, 2026-07-23) : si on peut générer N≫K mondes (GPU),
//! peut-on fournir à IS-DD un menu de K mondes plus « cohérent » que K tirages
//! directs, en exploitant les tendances statistiques des N tirages ?
//!
//! Trois menus de K mondes comparés à budget solver égal :
//!   - direct   : les K premiers tirages (comportement IS-DD actuel)
//!   - compress : k-médoïdes sur les N tirages, un représentant par cluster,
//!                pondéré par la masse du cluster (compression de posterior)
//!   - topcons  : les K mondes les plus consensuels (idée brute « les moins
//!                bizarres » — attendu : biais de mode, queues amputées)
//!
//! Métriques par menu, contre la vérité terrain des positions seedées :
//!   - logp de placement de la vérité sous les marginales du menu (nats/carte,
//!    ↑ mieux ; référence : marginales des N tirages complets)
//!   - distance min du menu à la vérité (cartes cachées mal placées, ↓ mieux)
//!   - crédibilité-juge (bid NN / DMC rejouent les actions observées), pondérée
//!   - diagnostic multi-modalité : nb effectif de clusters (1/Σm²)
//!
//! Usage:
//!   cargo run -p colver-core --bin bench_world_compress --release -- \
//!     --bid-positions 20 --play-positions 20 --worlds 200 --menu 20 --seed 42

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

fn rank_of(q: &[f32], legal: u64, action: u8) -> usize {
    let qa = q[action as usize];
    (0..q.len() as u8)
        .filter(|&c| c != action && legal & (1u64 << c) != 0 && q[c as usize] > qa)
        .count()
}

/// Cartes cachées placées différemment entre deux mondes (0 = identiques).
fn world_dist(a: &[u32; 4], b: &[u32; 4], observer: u8) -> u32 {
    let mut d = 0u32;
    for p in 0..4u8 {
        if p == observer {
            continue;
        }
        d += (a[p as usize] ^ b[p as usize]).count_ones();
    }
    d / 2
}

/// k-médoïdes (init k-means++, itérations d'affectation/médoïde).
/// Retourne (indice du médoïde, masse du cluster).
fn kmedoids(
    worlds: &[[u32; 4]],
    observer: u8,
    k: usize,
    rng: &mut StdRng,
) -> Vec<(usize, f64)> {
    let n = worlds.len();
    assert!(k <= n);
    let mut dm = vec![0u32; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let d = world_dist(&worlds[i], &worlds[j], observer);
            dm[i * n + j] = d;
            dm[j * n + i] = d;
        }
    }

    // Init k-means++
    let mut medoids: Vec<usize> = vec![rng.gen_range(0..n)];
    while medoids.len() < k {
        let d2: Vec<f64> = (0..n)
            .map(|i| {
                let dmin = medoids.iter().map(|&m| dm[i * n + m]).min().unwrap();
                (dmin as f64) * (dmin as f64)
            })
            .collect();
        let total: f64 = d2.iter().sum();
        if total <= 0.0 {
            // Tous les points couverts exactement — compléter arbitrairement.
            for i in 0..n {
                if !medoids.contains(&i) {
                    medoids.push(i);
                    break;
                }
            }
            continue;
        }
        let mut r = rng.gen::<f64>() * total;
        let mut pick = n - 1;
        for i in 0..n {
            r -= d2[i];
            if r <= 0.0 {
                pick = i;
                break;
            }
        }
        medoids.push(pick);
    }

    let mut assign = vec![0usize; n];
    for _iter in 0..10 {
        // Affectation
        for i in 0..n {
            assign[i] = (0..k)
                .min_by_key(|&c| dm[i * n + medoids[c]])
                .unwrap();
        }
        // Recalcul des médoïdes
        let mut changed = false;
        for c in 0..k {
            let members: Vec<usize> = (0..n).filter(|&i| assign[i] == c).collect();
            if members.is_empty() {
                // Cluster vide : re-seed au point le plus loin de tout médoïde.
                let far = (0..n)
                    .max_by_key(|&i| medoids.iter().map(|&m| dm[i * n + m]).min().unwrap())
                    .unwrap();
                if medoids[c] != far {
                    medoids[c] = far;
                    changed = true;
                }
                continue;
            }
            let best = *members
                .iter()
                .min_by_key(|&&i| members.iter().map(|&j| dm[i * n + j] as u64).sum::<u64>())
                .unwrap();
            if medoids[c] != best {
                medoids[c] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
    }
    for i in 0..n {
        assign[i] = (0..k).min_by_key(|&c| dm[i * n + medoids[c]]).unwrap();
    }
    (0..k)
        .map(|c| {
            let mass = assign.iter().filter(|&&a| a == c).count() as f64 / n as f64;
            (medoids[c], mass)
        })
        .filter(|&(_, m)| m > 0.0)
        .collect()
}

/// Marginales p(owner | carte) d'un menu pondéré, seats cachés uniquement.
fn menu_marginals(
    worlds: &[[u32; 4]],
    menu: &[(usize, f64)],
    unseen: u32,
    observer: u8,
) -> [[f64; 4]; 32] {
    let mut marg = [[0.0f64; 4]; 32];
    let total: f64 = menu.iter().map(|&(_, w)| w).sum();
    for &(wi, w) in menu {
        for p in 0..4u8 {
            if p == observer {
                continue;
            }
            let mut h = worlds[wi][p as usize] & unseen;
            while h != 0 {
                let c = h.trailing_zeros() as usize;
                marg[c][p as usize] += w / total;
                h &= h - 1;
            }
        }
    }
    marg
}

/// logp moyen (nats/carte) du placement vrai sous des marginales.
fn truth_logp(marg: &[[f64; 4]; 32], truth: &[u32; 4], unseen: u32, observer: u8) -> f64 {
    let mut sum = 0.0f64;
    let mut n = 0usize;
    for p in 0..4u8 {
        if p == observer {
            continue;
        }
        let mut h = truth[p as usize] & unseen;
        while h != 0 {
            let c = h.trailing_zeros() as usize;
            sum += marg[c][p as usize].max(1e-3).ln();
            n += 1;
            h &= h - 1;
        }
    }
    sum / n.max(1) as f64
}

/// Score consensus d'un monde : logp moyen de ses placements sous les
/// marginales des N tirages.
fn consensus_scores(worlds: &[[u32; 4]], unseen: u32, observer: u8) -> Vec<f64> {
    let all_menu: Vec<(usize, f64)> = (0..worlds.len()).map(|i| (i, 1.0)).collect();
    let marg = menu_marginals(worlds, &all_menu, unseen, observer);
    worlds
        .iter()
        .map(|w| truth_logp(&marg, w, unseen, observer))
        .collect()
}

#[derive(Default)]
struct MenuAgg {
    logp: f64,
    min_dist: f64,
    judge_rate: f64,
    exact_hits: usize,
    n: usize,
}

impl MenuAgg {
    fn add(&mut self, logp: f64, min_dist: f64, judge_rate: f64, exact: bool) {
        self.logp += logp;
        self.min_dist += min_dist;
        self.judge_rate += judge_rate;
        self.exact_hits += exact as usize;
        self.n += 1;
    }
    fn print(&self, label: &str) {
        let n = self.n.max(1) as f64;
        println!(
            "  {:9}: logp vérité {:+.3}  dist min {:.1}  juge argmax {:.1}%  vérité exacte {}/{}",
            label,
            self.logp / n,
            self.min_dist / n,
            self.judge_rate / n * 100.0,
            self.exact_hits,
            self.n
        );
    }
}

fn min_dist_and_exact(
    worlds: &[[u32; 4]],
    menu: &[(usize, f64)],
    truth: &[u32; 4],
    observer: u8,
) -> (f64, bool) {
    let md = menu
        .iter()
        .map(|&(wi, _)| world_dist(&worlds[wi], truth, observer))
        .min()
        .unwrap();
    (md as f64, md == 0)
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut bid_positions = 20usize;
    let mut play_positions = 20usize;
    let mut n_worlds = 200usize;
    let mut menu_k = 20usize;
    let mut seed = 42u64;
    let mut temperature = 1.0f32;
    let mut bid_model = String::from("models/bid_v6_isdd_resume/bid_nn_final.bin");
    let mut bid_hidden = 512usize;
    let mut dmc_model = String::from("models/play_v2/play_final.bin");
    let mut playgen_path = String::from("models/playgen_v2/playgen_v2_half.bin");
    let mut use_gpu = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--gpu" => { use_gpu = true; i += 1; }
            "--bid-positions" => { bid_positions = args[i + 1].parse().unwrap(); i += 2; }
            "--play-positions" => { play_positions = args[i + 1].parse().unwrap(); i += 2; }
            "--worlds" => { n_worlds = args[i + 1].parse().unwrap(); i += 2; }
            "--menu" => { menu_k = args[i + 1].parse().unwrap(); i += 2; }
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
        "bench_world_compress — seed {}, N={} mondes → menus K={}, temp {}, gen {}",
        seed,
        n_worlds,
        menu_k,
        temperature,
        if use_gpu { "GPU" } else { "CPU" }
    );

    // =====================================================================
    // Phase 1 : enchères
    // =====================================================================
    if bid_positions > 0 {
        let t0 = std::time::Instant::now();
        let mut aggs = [MenuAgg::default(), MenuAgg::default(), MenuAgg::default()];
        let mut full_logp = 0.0f64;
        let mut eff_clusters = 0.0f64;
        let mut biggest_mass = 0.0f64;
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
            #[cfg(feature = "dmc_train")]
            let scored = match &gpu {
                Some(g) => {
                    let sampler = analyst.sampler();
                    g.generate_deals_from_auction_scored(
                        &sampler.prefix_tokens(),
                        &state,
                        sampler.observer(),
                        sampler.observer_hand(),
                        sampler.bid_entries_count(),
                        n_worlds,
                        temperature,
                        &mut rng,
                    )
                    .expect("GPU generation")
                }
                None => analyst.auction_deals_scored(&state, n_worlds, temperature, &mut rng),
            };
            #[cfg(not(feature = "dmc_train"))]
            let scored =
                analyst.auction_deals_scored(&state, n_worlds, temperature, &mut rng);
            if scored.len() < menu_k * 3 {
                continue; // trop peu de mondes pour comparer proprement
            }
            let worlds: Vec<[u32; 4]> = scored.iter().map(|(w, _)| *w).collect();
            let unseen = colver_core::card::ALL_CARDS & !state0.hands[observer as usize];
            let truth = state0.hands;

            // Menus
            let direct: Vec<(usize, f64)> = (0..menu_k).map(|i| (i, 1.0)).collect();
            let compress = kmedoids(&worlds, observer, menu_k, &mut rng);
            let cons = consensus_scores(&worlds, unseen, observer);
            let mut by_cons: Vec<usize> = (0..worlds.len()).collect();
            by_cons.sort_by(|&a, &b| cons[b].partial_cmp(&cons[a]).unwrap());
            let topcons: Vec<(usize, f64)> =
                by_cons[..menu_k].iter().map(|&i| (i, 1.0)).collect();

            // Diagnostic multi-modalité
            let sum_m2: f64 = compress.iter().map(|&(_, m)| m * m).sum();
            eff_clusters += 1.0 / sum_m2;
            biggest_mass += compress.iter().map(|&(_, m)| m).fold(0.0, f64::max);

            // Référence : marginales des N tirages
            let all_menu: Vec<(usize, f64)> = (0..worlds.len()).map(|i| (i, 1.0)).collect();
            let marg_full = menu_marginals(&worlds, &all_menu, unseen, observer);
            full_logp += truth_logp(&marg_full, &truth, unseen, observer);

            // Juge : rejouer les enchères observées avec les mains du monde
            let mut judge = |menu: &[(usize, f64)]| -> f64 {
                let total: f64 = menu.iter().map(|&(_, w)| w).sum();
                let mut rate = 0.0f64;
                for &(wi, w) in menu {
                    let mut s = GameState::new(dealer, worlds[wi]);
                    let mut hist: Vec<(u8, u8)> = Vec::new();
                    let (mut hits, mut n) = (0usize, 0usize);
                    for (ai, &(p, a)) in actions.iter().enumerate() {
                        if targets.contains(&ai) {
                            write_bid_obs(bid_dim, &mut bid_obs_buf, &s, &hist);
                            let q = bid_net.evaluate(&bid_obs_buf);
                            hits += (rank_of(&q, s.legal_actions(), a) == 0) as usize;
                            n += 1;
                        }
                        hist.push((p, a));
                        s.step(a);
                    }
                    rate += w / total * hits as f64 / n.max(1) as f64;
                }
                rate
            };

            for (mi, menu) in [&direct, &compress, &topcons].iter().enumerate() {
                let marg = menu_marginals(&worlds, menu, unseen, observer);
                let lp = truth_logp(&marg, &truth, unseen, observer);
                let (md, exact) = min_dist_and_exact(&worlds, menu, &truth, observer);
                let jr = judge(menu);
                aggs[mi].add(lp, md, jr, exact);
            }
            done += 1;
        }
        println!(
            "\n== Enchères : {} positions ({:.0}s) — 24 cartes cachées ==",
            done,
            t0.elapsed().as_secs_f64()
        );
        println!(
            "  référence marginales N={} : logp vérité {:+.3}",
            n_worlds,
            full_logp / done.max(1) as f64
        );
        println!(
            "  multi-modalité : {:.1} clusters effectifs, plus gros cluster {:.0}%",
            eff_clusters / done.max(1) as f64,
            biggest_mass / done.max(1) as f64 * 100.0
        );
        for (mi, label) in ["direct", "compress", "topcons"].iter().enumerate() {
            aggs[mi].print(label);
        }
    }

    // =====================================================================
    // Phase 2 : jeu
    // =====================================================================
    if play_positions > 0 {
        let t0 = std::time::Instant::now();
        let mut dmc = DmcNet::load(&dmc_model).expect("dmc model");
        if dmc.obs_dim() == OBS_DIM_TR {
            dmc.set_residual(true);
        }
        let mut aggs = [MenuAgg::default(), MenuAgg::default(), MenuAgg::default()];
        let mut full_logp = 0.0f64;
        let mut eff_clusters = 0.0f64;
        let mut biggest_mass = 0.0f64;
        let mut done = 0usize;

        let dmc_play = |net: &mut DmcNet, state: &GameState, tracking: &EnvTracking| -> (u8, Vec<(u8, f32)>) {
            let legal = state.legal_actions() as u32;
            let obs = dmc_obs::make_observation_tr(state, tracking);
            let order = dmc_obs::current_player_order(state, tracking);
            let mask = dmc_obs::cardset_to_canonical(legal, &order);
            let (best, q) = net.best_action(&obs, mask);
            let phys_q = q
                .into_iter()
                .map(|(c, v)| (dmc_obs::card_to_physical(c, &order), v))
                .collect();
            (dmc_obs::card_to_physical(best, &order), phys_q)
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
            let scored = analyst.play_worlds_scored(&state, n_worlds, temperature, &mut rng);
            if scored.len() < menu_k * 3 {
                continue;
            }
            let worlds: Vec<[u32; 4]> = scored.iter().map(|(w, _)| *w).collect();
            // Cartes cachées restantes (vue observateur)
            let mut trick_mask = 0u32;
            for i in 0..4 {
                let c = state.current_trick[i];
                if c != colver_core::card::EMPTY {
                    trick_mask |= 1u32 << c;
                }
            }
            let known = state.hands[observer as usize] | state.played_cards | trick_mask;
            let unseen = colver_core::card::ALL_CARDS & !known;
            let truth = state.hands;

            let direct: Vec<(usize, f64)> = (0..menu_k).map(|i| (i, 1.0)).collect();
            let compress = kmedoids(&worlds, observer, menu_k, &mut rng);
            let cons = consensus_scores(&worlds, unseen, observer);
            let mut by_cons: Vec<usize> = (0..worlds.len()).collect();
            by_cons.sort_by(|&a, &b| cons[b].partial_cmp(&cons[a]).unwrap());
            let topcons: Vec<(usize, f64)> =
                by_cons[..menu_k].iter().map(|&i| (i, 1.0)).collect();

            let sum_m2: f64 = compress.iter().map(|&(_, m)| m * m).sum();
            eff_clusters += 1.0 / sum_m2;
            biggest_mass += compress.iter().map(|&(_, m)| m).fold(0.0, f64::max);

            let all_menu: Vec<(usize, f64)> = (0..worlds.len()).map(|i| (i, 1.0)).collect();
            let marg_full = menu_marginals(&worlds, &all_menu, unseen, observer);
            full_logp += truth_logp(&marg_full, &truth, unseen, observer);

            // Juge : rejouer l'historique avec les mains initiales du monde
            let judge = |menu: &[(usize, f64)], dmc: &mut DmcNet| -> f64 {
                let total: f64 = menu.iter().map(|&(_, w)| w).sum();
                let mut rate = 0.0f64;
                for &(wi, w) in menu {
                    let mut init_hands = [0u32; 4];
                    let mut valid = true;
                    for p in 0..4usize {
                        init_hands[p] = worlds[wi][p] | played_by[p];
                        valid &= card_count(init_hands[p]) == 8;
                    }
                    if !valid {
                        continue;
                    }
                    let mut s = GameState::new(dealer, init_hands);
                    let mut jt = EnvTracking::new();
                    jt.reset(dealer);
                    let (mut hits, mut n) = (0usize, 0usize);
                    for &(p, a) in &history {
                        if s.phase == Phase::Playing && p != observer {
                            let (_, q) = dmc_play(dmc, &s, &jt);
                            if let Some(qa) = q.iter().find(|(c, _)| *c == a).map(|(_, v)| *v) {
                                let better =
                                    q.iter().filter(|(c, v)| *c != a && *v > qa).count();
                                hits += (better == 0) as usize;
                                n += 1;
                            }
                        }
                        jt.track_action(&s, a);
                        s.step(a);
                    }
                    rate += w / total * hits as f64 / n.max(1) as f64;
                }
                rate
            };

            for (mi, menu) in [&direct, &compress, &topcons].iter().enumerate() {
                let marg = menu_marginals(&worlds, menu, unseen, observer);
                let lp = truth_logp(&marg, &truth, unseen, observer);
                let (md, exact) = min_dist_and_exact(&worlds, menu, &truth, observer);
                let jr = judge(menu, &mut dmc);
                aggs[mi].add(lp, md, jr, exact);
            }
            done += 1;
        }
        println!(
            "\n== Jeu : {} positions ({:.0}s) ==",
            done,
            t0.elapsed().as_secs_f64()
        );
        println!(
            "  référence marginales N={} : logp vérité {:+.3}",
            n_worlds,
            full_logp / done.max(1) as f64
        );
        println!(
            "  multi-modalité : {:.1} clusters effectifs, plus gros cluster {:.0}%",
            eff_clusters / done.max(1) as f64,
            biggest_mass / done.max(1) as f64 * 100.0
        );
        for (mi, label) in ["direct", "compress", "topcons"].iter().enumerate() {
            aggs[mi].print(label);
        }
    }
}
