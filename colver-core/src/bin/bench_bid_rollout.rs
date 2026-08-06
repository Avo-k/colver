//! Ce que coûte une décision du bidder par simulation, et ce qu'elle sépare.
//!
//! Deux questions, dans cet ordre, parce que la seconde décide si la première
//! valait la peine d'être posée :
//!
//! 1. **Le coût.** Une décision = `candidates × sims` donnes simulées, chacune
//!    une enchère au réseau plus 32 coups de DouDou50. On mesure la donne
//!    simulée, la décision, et on extrapole le match.
//! 2. **Le pouvoir de séparation.** `quick_bid_spread` a mesuré σ ≈ 310-370
//!    points par monde sur cette même simulation, et quelques points d'écart
//!    *vrai* entre deux annonces voisines. Un classement construit là-dessus
//!    peut n'être qu'un tirage au sort déguisé, et l'écart affiché entre la
//!    première et la deuxième candidate ne le dirait pas : **c'est un maximum
//!    sur des estimations bruitées, donc il est positif même sous bruit pur.**
//!
//!    L'instrument est donc un **contrôle moitié/moitié** et non une formule :
//!    les mondes de rang pair et ceux de rang impair donnent deux classements
//!    indépendants de la *même* décision. S'ils désignent la même annonce plus
//!    souvent que le hasard (1/k), le classement porte quelque chose ; sinon le
//!    bidder ne fait que randomiser son a priori, en mille fois plus cher. Le
//!    même contrôle rend une erreur type **mesurée**, sans supposer de σ.
//!
//! On mesure sur la **première parole**, la plus chère (l'enchère simulée est
//! la plus longue) et la plus ouverte (le plus de candidates plausibles).
//!
//! ```bash
//! cargo run -p colver-core --release --bin bench_bid_rollout -- --deals 30 --sims 20
//! ```

use colver_core::agent::{AgentSpec, MatchContext};
use colver_core::state::GameState;
use rand::rngs::StdRng;
use rand::SeedableRng;

const SUITS: [char; 4] = ['S', 'H', 'D', 'C'];

fn bid_label(action: u8) -> String {
    match action {
        0 => "passe".into(),
        1..=36 => format!("{}{}", 80 + (action as u16 - 1) / 4 * 10, SUITS[((action - 1) % 4) as usize]),
        37..=40 => format!("capot{}", SUITS[(action - 37) as usize]),
        41 => "coinche".into(),
        42 => "surcoinche".into(),
        _ => format!("?{action}"),
    }
}

/// Des positions de **jeu** pour le contrôle d'équivalence : on distribue, on
/// laisse une heuristique mener l'enchère, et on garde ce qui a pris un contrat.
/// Le suivi (`MatchContext`) est construit en même temps — c'est lui qui porte
/// l'ordre canonique, donc une position sans lui ne contrôlerait rien.
#[cfg(feature = "dmc_train")]
fn playing_positions(n: usize, seed: u64) -> Vec<colver_core::gpu_rollout::Lane> {
    use colver_core::state::Phase;
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::with_capacity(n);
    while out.len() < n {
        let mut state = GameState::deal_random(0, &mut rng);
        let mut ctx = MatchContext::new(state.dealer);
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let before = state;
            let a = colver_core::bid_eval::improved_v2_bid(&before);
            ctx.track(&before, a);
            state.step(a);
        }
        if state.phase != Phase::Playing {
            continue; // donne passée : rien à jouer
        }
        // Quelques cartes posées, pour contrôler aussi en milieu de donne.
        let depth = out.len() % 8;
        for _ in 0..depth {
            if state.phase != Phase::Playing {
                break;
            }
            let before = state;
            let a = colver_core::rollout::heuristic_play_action(&before);
            ctx.track(&before, a);
            state.step(a);
        }
        if state.phase == Phase::Playing {
            out.push(colver_core::gpu_rollout::Lane { state, ctx });
        }
    }
    out
}

fn arg<T: std::str::FromStr>(name: &str, default: T) -> T {
    let args: Vec<String> = std::env::args().collect();
    args.iter()
        .position(|a| a == name)
        .and_then(|i| args.get(i + 1))
        .and_then(|v| v.parse().ok())
        .unwrap_or(default)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let deals: u32 = arg("--deals", 30);
    let sims: u32 = arg("--sims", 20);
    let candidates: usize = arg("--candidates", 4);
    let seed: u64 = arg("--seed", 42);
    let bid_model: String = arg("--bid-model", "models/bid_v6_isdd_resume/bid_nn_final.bin".to_string());
    let play_model: String = arg("--play-model", "models/play_v2/play_final.bin".to_string());

    let mode: String = arg("--mode", "probe".to_string());
    let worlds: String = arg("--worlds", "uniform".to_string());
    let parallel: bool = std::env::args().any(|a| a == "--parallel");
    let gpu: bool = std::env::args().any(|a| a == "--gpu");

    // ⚠️ **Avant tout chiffre : le contrôle d'équivalence.** Deux pièges
    // silencieux vivent sur le chemin GPU — l'orientation des matrices et
    // l'espace canonique des couleurs — et aucun des deux ne lève d'erreur : le
    // réseau rend une carte légale et sans rapport, ce qui se lit comme un
    // joueur un peu faible, pas comme un bug. Un déroulement groupé qu'on n'a
    // pas confronté au réseau CPU ne mesure rien.
    #[cfg(feature = "dmc_train")]
    if gpu {
        let w = colver_core::agent::models::dmc_weights(&play_model)?;
        let engine = colver_core::gpu_rollout::GpuRollout::new(&w, true)?;
        let positions = playing_positions(64, seed ^ 0xC0FFEE);
        let worst = engine.check_against_cpu(&w, true, &positions)?;
        println!(
            "contrôle GPU/CPU : écart max {worst:.2e} sur {} positions — appareil {}",
            positions.len(),
            if engine.device_is_cuda() { "CUDA" } else { "CPU" }
        );
        // Un ordre de réduction différent coûte ~1e-3 ; une matrice transposée
        // coûte des unités de Q. Le seuil sépare les deux sans ambiguïté.
        if worst > 0.05 {
            return Err(format!(
                "le déroulement GPU ne reproduit pas le réseau CPU (écart {worst:.3}) — \
                 mesurer serait mesurer autre chose"
            )
            .into());
        }
    }

    let toml = format!(
        "[bid]\nstrategy = \"rollout\"\nmodel = \"{bid_model}\"\nhidden = 512\n\
         score_aware = true\nsims = {sims}\ncandidates = {candidates}\n\
         candidate_mode = \"{mode}\"\nparallel = {parallel}\ngpu = {gpu}\n\
         [play]\nmethod = \"dmc\"\nmodel = \"{play_model}\"\nresidual = true\n\
         [worlds]\nsource = \"{worlds}\"\n"
    );
    // Deux instances du *même* bidder, graines différentes : elles voient la
    // même position et la même présélection, et ne diffèrent que par le tirage
    // des mondes. C'est le contrôle — ce sur quoi elles s'accordent est ce que
    // la simulation sait, le reste est le tirage.
    let mut spec_a = AgentSpec::from_toml_str(&toml)?;
    spec_a.seed = seed;
    let mut roll_a = spec_a.build(0)?;
    let mut spec_b = AgentSpec::from_toml_str(&toml)?;
    spec_b.seed = seed ^ 0xA5A5_5A5A;
    let mut roll_b = spec_b.build(0)?;

    let nn_toml = toml.replacen("strategy = \"rollout\"", "strategy = \"nn\"", 1);
    let mut nn = AgentSpec::from_toml_str(&nn_toml)?.build(0)?;

    println!("rollout bidder : {sims} mondes × {candidates} candidates ({mode}, worlds={worlds}\
              {}) = {} donnes simulées/décision",
             if parallel { ", parallèle" } else { "" },
             sims as usize * candidates);
    println!("  bid  {bid_model}\n  play {play_model}\n");

    let mut rng = StdRng::seed_from_u64(seed);
    let mut agree_nn = 0u32;
    let mut agree_self = 0u32;
    let mut times = Vec::new();
    let mut gaps = Vec::new(); // écart 1ʳᵉ − 2ᵉ candidate, run A
    let mut deltas = Vec::new(); // |moyenne_A − moyenne_B| par candidate
    let mut worlds_used = 0u64;

    println!("{:>4}  {:>7}  {:>8}  {:>8}  {:>8}  {:>5}   {}",
             "#", "ms", "réseau", "sim. A", "sim. B", "A=B", "candidates de A (espérance d'écart)");
    for d in 0..deals {
        // Une donne fraîche, position d'ouverture : personne n'a encore parlé.
        let state = GameState::deal_random(0, &mut rng);
        let ctx = MatchContext::new(state.dealer);
        for p in [&mut roll_a, &mut roll_b, &mut nn] {
            p.init_deal(&state);
        }

        let t = std::time::Instant::now();
        let a = roll_a.decide(&state, &ctx)?;
        let ms = t.elapsed().as_secs_f64() * 1000.0;
        let b = roll_b.decide(&state, &ctx)?;
        let base = nn.decide(&state, &ctx)?;

        if a.action == base.action {
            agree_nn += 1;
        }
        if a.action == b.action {
            agree_self += 1;
        }
        times.push(ms);
        worlds_used += a.stats.determinizations as u64;
        if a.stats.candidates.len() >= 2 {
            gaps.push((a.stats.candidates[0].1 - a.stats.candidates[1].1) as f64);
        }
        for (act, va) in &a.stats.candidates {
            if let Some((_, vb)) = b.stats.candidates.iter().find(|(x, _)| x == act) {
                deltas.push((va - vb).abs() as f64);
            }
        }

        let shown: Vec<String> = a
            .stats
            .candidates
            .iter()
            .take(4)
            .map(|(act, v)| format!("{} {:+.0}", bid_label(*act), v))
            .collect();
        println!(
            "{:>4}  {:>7.0}  {:>8}  {:>8}  {:>8}  {:>5}   {}",
            d,
            ms,
            bid_label(base.action),
            bid_label(a.action),
            bid_label(b.action),
            if a.action == b.action { "=" } else { "≠" },
            shown.join("  ")
        );
    }

    let mean = |v: &[f64]| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let ms_mean = mean(&times);
    let sims_done = worlds_used as f64 / deals as f64;
    let per_deal = ms_mean / (sims_done * candidates as f64);

    println!("\n── coût ──");
    println!("  décision      : {ms_mean:.0} ms  (min {:.0}, max {:.0})",
             times.iter().cloned().fold(f64::INFINITY, f64::min),
             times.iter().cloned().fold(0.0, f64::max));
    println!("  donne simulée : {per_deal:.1} ms  ({sims_done:.0} mondes × {candidates} candidates)");
    // Un match à 2000 points fait ~13 donnes ; le bot tient deux sièges et
    // chacun parle une à deux fois, donc ~4 décisions de bidder par donne.
    println!("  match (est.)  : {:.0} s  pour ~13 donnes × ~4 décisions du bot",
             ms_mean * 13.0 * 4.0 / 1000.0);

    println!("\n── séparation (deux tirages indépendants de la même décision) ──");
    // Deux moyennes indépendantes de même erreur type s : E|A−B| = 1,128·s.
    let se = mean(&deltas) / 1.128;
    let chance = 100.0 / candidates.max(1) as f64;
    let self_pct = 100.0 * agree_self as f64 / deals as f64;
    println!("  A et B choisissent pareil : {agree_self}/{deals} ({self_pct:.0} %)  — hasard ≈ {chance:.0} %");
    println!("  erreur type mesurée       : ±{se:.0} points sur la moyenne d'une candidate");
    println!("  écart 1ʳᵉ − 2ᵉ (run A)    : {:.0} points  ⚠️ maximum d'estimations bruitées, positif même sous bruit pur",
             mean(&gaps));
    println!("  accord avec le réseau seul : {}/{} ({:.0} %)",
             agree_nn, deals, 100.0 * agree_nn as f64 / deals as f64);
    println!("  → {}", if self_pct > chance + 25.0 {
        "reproductible : la simulation classe, elle ne tire pas au sort"
    } else {
        "peu reproductible : à ce budget le bidder randomise surtout son a priori"
    });
    Ok(())
}
