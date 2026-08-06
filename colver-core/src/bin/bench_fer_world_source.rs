//! Pour une case « fer », vaut-il mieux étiqueter avec des **mondes uniformes** qu'avec
//! des mondes playgen conditionnés sur une enchère fabriquée ?
//!
//! ## D'où vient la question
//!
//! `bench_prefix_worlds` (2026-08-06, 199 donnes) a mesuré la qualité des *mondes* sous
//! chaque rang de préfixe, contre la vérité terrain. Résultat gênant pour le fer :
//!
//! | | exactitude de placement | NLL |
//! |---|--:|--:|
//! | or (enchère réelle) | 43,23 % | 1,0421 |
//! | **fer** (construite) | 37,80 % | **1,1510** |
//! | uniforme sous contrainte *(analytique)* | 33,33 % | **1,0986** |
//!
//! Le fer est **plus exact** que l'uniforme (37,8 contre 33,3 %) et pourtant **plus mal
//! calibré** (1,151 contre 1,099) : playgen sous une fausse enchère devient confiant et
//! faux. Ce n'est donc pas une domination, c'est un arbitrage — et rien dans une note de
//! croyances ne dit lequel des deux produit la meilleure **étiquette**, qui est une
//! valeur en points cartes sortie d'IS-DD.
//!
//! ## Pourquoi ce n'est pas un sixième bras de la mesure B
//!
//! B juge au **score de la donne**, dont le σ apparié est de 24,4 points cartes. Voir un
//! écart de l'ordre de 4 points y demanderait ~670 donnes × 2 bras playgen, soit ~5 h de
//! CPU. Or ici la vérité terrain est connue — le corpus porte les quatre mains — donc le
//! solveur donne la valeur DD **exacte** de chaque carte jouable à chaque position. On
//! juge donc **décision par décision** contre un oracle parfait, ce qui est à la fois
//! beaucoup plus puissant (≈ 30 décisions par donne au lieu d'un chiffre) et beaucoup
//! moins cher. Même patron que l'A/B apparié de la belote.
//!
//! ## Le protocole, et le seul point qui demande de l'attention
//!
//! Les trois configurations sont interrogées **aux mêmes positions** : la donne est
//! déroulée par une seule d'entre elles, et les deux autres sont *consultées* sans jamais
//! jouer. Les laisser jouer chacune sa donne ferait diverger les positions au premier
//! désaccord, et la comparaison cesserait d'être appariée — c'est exactement pourquoi un
//! h2h d'arène ne peut pas répondre à ce genre de question.
//!
//! ⚠️ **La trajectoire appartient donc au bras qui conduit** (`fer + playgen`, graine A).
//! Les positions visitées sont celles que *lui* atteint. C'est le même compromis que
//! `bench_belote_ab` ; le noter plutôt que le corriger, parce que la seule correction
//! serait de dérouler chaque bras séparément, ce qui casse l'appariement.
//!
//! Le **témoin** est le bras conducteur retiré une seconde fois avec une autre graine.
//! Son coût moyen doit être indiscernable de celui du bras A ; sans lui, un écart
//! playgen/uniforme est indiscernable de deux tirages de mondes du même bras.
//!
//! ## Le coût d'une décision
//!
//! `solve_with_scores` rend `(carte, points N-S)` pour chaque coup légal à la position
//! **réelle**. Le coût d'une carte est l'écart à la meilleure, **du côté du camp qui
//! joue** — les points cartes sont à somme constante, donc pour E-O la meilleure carte
//! est celle qui *minimise* les points N-S. Le compter en N-S annulerait l'effet une
//! décision sur deux (cf. `feedback_orient_by_the_taker`).
//!
//! Les positions à un seul coup légal sont écartées : il n'y a pas de décision, et les
//! compter diluerait l'écart avec des zéros communs aux trois bras.
//!
//! ```bash
//! cargo build -p colver-core --release --features parallel --bin bench_fer_world_source
//! ./target/release/bench_fer_world_source --deals 150 --threads 8 --json out.json
//! ```
//!
//! **Aucun sidecar** : les mondes playgen sont produits en CPU (`worlds.source =
//! "playgen"`), pour que la mesure cohabite avec une génération qui monopolise les GPU.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use colver_core::agent::spec::WorldSourceKind;
use colver_core::agent::{AgentSpec, MatchContext, Player};
use colver_core::game_replay::GameReplay;
use colver_core::solver;
use colver_core::state::{GameState, Phase};

#[path = "shared/auction.rs"]
mod auction;
use auction::built_auction;

const N_ARMS: usize = 3;
const ARM_NAMES: [&str; N_ARMS] = ["fer + playgen (A)", "fer + playgen (B) témoin", "fer + UNIFORME"];

/// Une donne : coût DD cumulé et nombre de décisions, par bras.
#[derive(Clone)]
struct Row {
    /// Coût moyen en points DD par décision, pour chaque bras.
    cost: [f64; N_ARMS],
    /// Fraction des décisions où le bras choisit une carte DD-optimale.
    optimal: [f64; N_ARMS],
    decisions: usize,
}

struct Args {
    games: String,
    bot: String,
    bid_model: String,
    playgen: String,
    deals: usize,
    dets: Option<u32>,
    threads: usize,
    seed: u64,
    json: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        games: "data/training/isdd_games_v1.bin".into(),
        bot: "arena/bots/gen_isdd_cardpts.toml".into(),
        bid_model: "models/bid_v6_isdd_resume/bid_nn_final.bin".into(),
        playgen: "models/playgen/playgen_v2_final.bin".into(),
        deals: 150,
        dets: None,
        threads: 8,
        seed: 20260806,
        json: None,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let nx = |i: usize| v.get(i + 1).cloned().unwrap_or_default();
    while i < v.len() {
        match v[i].as_str() {
            "--games" => { a.games = nx(i); i += 2 }
            "--bot" => { a.bot = nx(i); i += 2 }
            "--bid-model" => { a.bid_model = nx(i); i += 2 }
            "--playgen" => { a.playgen = nx(i); i += 2 }
            "--deals" => { a.deals = nx(i).parse().unwrap(); i += 2 }
            "--dets" => { a.dets = Some(nx(i).parse().unwrap()); i += 2 }
            "--threads" => { a.threads = nx(i).parse().unwrap(); i += 2 }
            "--seed" => { a.seed = nx(i).parse().unwrap(); i += 2 }
            "--json" => { a.json = Some(nx(i)); i += 2 }
            "--help" | "-h" => {
                eprintln!("bench_fer_world_source : mondes uniformes contre playgen, sur un préfixe « fer »");
                eprintln!("  --deals N     donnes                (défaut 150)");
                eprintln!("  --dets N      mondes par décision   (défaut : celui du bot, 40)");
                eprintln!("  --threads N   CPU pur, aucun GPU    (défaut 8)");
                eprintln!("  --json <path> brut par donne");
                eprintln!();
                eprintln!("  ⚠️ ne jamais rediriger vers `head` : le SIGPIPE tue le processus");
                eprintln!("     avant l'écriture du JSON.");
                std::process::exit(0)
            }
            other => { eprintln!("argument inconnu : {other}"); std::process::exit(1) }
        }
    }
    a
}

fn paired(xs: &[f64]) -> (f64, f64, f64, usize) {
    let n = xs.len();
    if n == 0 {
        return (f64::NAN, f64::NAN, f64::NAN, 0);
    }
    let m = xs.iter().sum::<f64>() / n as f64;
    let var = xs.iter().map(|x| (x - m) * (x - m)).sum::<f64>() / (n as f64 - 1.0).max(1.0);
    let sd = var.sqrt();
    (m, sd, sd / (n as f64).sqrt(), n)
}

fn show(name: &str, xs: &[f64], unit: &str) {
    let (m, sd, se, n) = paired(xs);
    let z = if se > 0.0 { m / se } else { f64::NAN };
    println!("  {name:<34} n={n:<5} {m:+8.4} ±{se:.4} {unit}  (z={z:+6.1})   σ={sd:.4}");
}

fn main() {
    let args = parse_args();

    let base = AgentSpec::from_toml_file(&args.bot).unwrap_or_else(|e| {
        eprintln!("bot {} : {e}", args.bot);
        std::process::exit(1);
    });

    // Trois specs qui ne diffèrent que par la source de mondes et la graine. Tout le
    // reste — 40 mondes, `objective = card_points`, `parallel = false` — vient du bot de
    // génération, pour que la mesure porte sur l'étiqueteur réel et pas sur un cousin.
    let mut spec_a = base.clone();
    spec_a.worlds.kind = WorldSourceKind::LocalPlaygen;
    spec_a.worlds.model = Some(args.playgen.clone());
    spec_a.seed = args.seed;
    let mut spec_b = spec_a.clone();
    spec_b.seed = args.seed ^ 0xA5A5_5A5A_A5A5_5A5A;
    let mut spec_u = base.clone();
    spec_u.worlds.kind = WorldSourceKind::Uniform;
    spec_u.worlds.model = None;
    spec_u.seed = args.seed;
    if let Some(d) = args.dets {
        for s in [&mut spec_a, &mut spec_b, &mut spec_u] {
            s.play.determinizations = d;
        }
    }
    let dets = spec_a.play.determinizations;

    let replays = GameReplay::load_all(&args.games).expect("lecture du corpus");
    let n = replays.len().min(args.deals);
    eprintln!(
        "bench_fer_world_source : {n} donnes × {N_ARMS} bras, {dets} mondes/décision, \
         {} threads, playgen en CPU (AUCUN GPU)",
        args.threads
    );

    let rows: Arc<Mutex<Vec<Row>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let next = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let skipped = Arc::new(AtomicUsize::new(0));
    let start = Instant::now();
    let replays = Arc::new(replays);

    let mut handles = Vec::with_capacity(args.threads);
    for _tid in 0..args.threads {
        let (rows, next, done, skipped) = (rows.clone(), next.clone(), done.clone(), skipped.clone());
        let replays = replays.clone();
        let (sa, sb, su) = (spec_a.clone(), spec_b.clone(), spec_u.clone());
        let seed = args.seed;
        handles.push(std::thread::spawn(move || {
            let build = |sp: &AgentSpec| -> Result<[Box<dyn Player>; 4], _> {
                (0..4).map(|s| sp.build(s)).collect::<Result<Vec<_>, _>>()
                    .map(|v| <[Box<dyn Player>; 4]>::try_from(v).ok().expect("4 joueurs"))
            };
            let mut arms: [[Box<dyn Player>; 4]; N_ARMS] = match (build(&sa), build(&sb), build(&su)) {
                (Ok(a), Ok(b), Ok(u)) => [a, b, u],
                (e1, e2, e3) => {
                    for e in [e1.err(), e2.err(), e3.err()].into_iter().flatten() {
                        eprintln!("construction : {e}");
                    }
                    std::process::exit(1)
                }
            };
            let mut tt = solver::new_tt_buffer();
            let mut judge_tt = solver::new_tt_buffer();
            let mut ctx = MatchContext::new(0);

            loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= n || idx >= replays.len() {
                    break;
                }
                let r = &replays[idx];

                // t₁ = l'atout que l'enchère réelle a nommé. Le fer est construit
                // dessus, ce qui rend ce bras directement comparable au `fer1` de B.
                let mut g = GameState::new(r.dealer, r.hands);
                for &a in &r.actions {
                    if g.phase != Phase::Bidding {
                        break;
                    }
                    g.step(a);
                }
                if g.phase != Phase::Playing {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let t1 = g.contract.trump;
                let dd1 = solver::solve_for_trump_reuse_tt(r.hands, r.dealer, t1, &mut tt)[0];
                let key = (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
                let fer = built_auction(&r.hands, r.dealer, t1, dd1, key);

                // --- montrer l'enchère aux trois bras ---
                let mut state = GameState::new(r.dealer, r.hands);
                ctx.reset_deal(r.dealer);
                for arm in arms.iter_mut() {
                    for p in arm.iter_mut() {
                        p.init_deal(&state);
                    }
                }
                let mut bad = false;
                for &a in &fer {
                    if state.phase != Phase::Bidding {
                        break;
                    }
                    let before = state;
                    let seat = before.current_player();
                    for arm in arms.iter_mut() {
                        for p in arm.iter_mut() {
                            p.observe(&before, seat, a);
                        }
                    }
                    ctx.track(&before, a);
                    state.step(a);
                }
                if state.phase != Phase::Playing {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                // --- dérouler, en consultant les trois bras à chaque décision ---
                let mut cost = [0.0f64; N_ARMS];
                let mut optimal = [0usize; N_ARMS];
                let mut decisions = 0usize;
                while !state.is_terminal() {
                    let seat = state.current_player();
                    let before = state;

                    let mut chosen = [0u8; N_ARMS];
                    for (k, arm) in arms.iter_mut().enumerate() {
                        match arm[seat as usize].action(&before, &ctx) {
                            Ok(a) => chosen[k] = a,
                            Err(e) => {
                                eprintln!("donne {idx} bras {k} : {e}");
                                bad = true;
                                break;
                            }
                        }
                    }
                    if bad {
                        break;
                    }

                    // Le juge : valeurs DD exactes à la position RÉELLE. Les positions à
                    // un seul coup ne sont pas des décisions.
                    let sc = solver::solve_with_scores(&before, Some(&mut judge_tt));
                    if sc.count > 1 {
                        let team = (seat & 1) as usize;
                        let vals = &sc.scores[..sc.count];
                        // Points cartes à somme constante : N-S veut le maximum, E-O le
                        // minimum de la même quantité.
                        let best = if team == 0 {
                            vals.iter().map(|&(_, v)| v).max().unwrap()
                        } else {
                            vals.iter().map(|&(_, v)| v).min().unwrap()
                        };
                        decisions += 1;
                        for k in 0..N_ARMS {
                            let v = vals.iter().find(|&&(c, _)| c == chosen[k]).map(|&(_, v)| v);
                            let Some(v) = v else { continue };
                            let c = if team == 0 { (best - v) as f64 } else { (v - best) as f64 };
                            cost[k] += c;
                            if c == 0.0 {
                                optimal[k] += 1;
                            }
                        }
                    }

                    // Le bras A conduit ; les deux autres ont été consultés sans jouer.
                    let action = chosen[0];
                    for arm in arms.iter_mut() {
                        for p in arm.iter_mut() {
                            p.observe(&before, seat, action);
                        }
                    }
                    ctx.track(&before, action);
                    state.step(action);
                }
                if bad || decisions == 0 {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                let d = decisions as f64;
                rows.lock().unwrap().push(Row {
                    cost: [cost[0] / d, cost[1] / d, cost[2] / d],
                    optimal: [
                        optimal[0] as f64 / d,
                        optimal[1] as f64 / d,
                        optimal[2] as f64 / d,
                    ],
                    decisions,
                });
                let k = done.fetch_add(1, Ordering::Relaxed) + 1;
                if k % 10 == 0 {
                    let el = start.elapsed().as_secs_f64();
                    eprintln!(
                        "  {k}/{n} donnes  {:.3} donnes/s  ETA {:.0} s",
                        k as f64 / el,
                        (n - k) as f64 / (k as f64 / el)
                    );
                }
            }
        }));
    }
    for h in handles {
        h.join().expect("thread paniqué");
    }

    let rows = rows.lock().unwrap().clone();
    let el = start.elapsed().as_secs_f64();
    let total_dec: usize = rows.iter().map(|r| r.decisions).sum();
    eprintln!(
        "\n{} donnes ({} décisions) en {:.0} s, {} écartées",
        rows.len(), total_dec, el, skipped.load(Ordering::Relaxed)
    );
    if rows.is_empty() {
        return;
    }

    println!("\n=== Coût DD moyen par décision ({} donnes, {total_dec} décisions, {dets} mondes) ===",
             rows.len());
    println!("  (points cartes perdus contre le meilleur coup, vu du camp qui joue ;\n   plus bas = mieux)\n");
    for k in 0..N_ARMS {
        let c: Vec<f64> = rows.iter().map(|r| r.cost[k]).collect();
        let o: Vec<f64> = rows.iter().map(|r| r.optimal[k] * 100.0).collect();
        let (mc, _, sec, _) = paired(&c);
        let (mo, _, seo, _) = paired(&o);
        println!("  {:<26} coût {mc:6.3} ±{sec:.3} pt   coup optimal {mo:5.2} % ±{seo:.2}",
                 ARM_NAMES[k]);
    }

    println!("\n=== Écarts appariés contre « fer + playgen (A) » ===");
    println!("  (NÉGATIF = le bras coûte PLUS cher, donc étiquette moins bien)\n");
    for k in 1..N_ARMS {
        let d: Vec<f64> = rows.iter().map(|r| r.cost[0] - r.cost[k]).collect();
        show(&format!("{} : −Δcoût", ARM_NAMES[k]), &d, "pt");
    }
    println!("\n  Le témoin donne le plancher : son écart au bras A est du bruit de");
    println!("  tirage de mondes pur. Si l'uniforme tient dedans, les deux sources");
    println!("  étiquettent aussi bien — ce qui serait déjà une réponse.");

    if let Some(path) = args.json {
        let mut s = String::from("{\"arms\":[");
        for (k, name) in ARM_NAMES.iter().enumerate() {
            if k > 0 { s.push(','); }
            s.push_str(&format!("\"{name}\""));
        }
        s.push_str(&format!("],\"dets\":{dets},\"rows\":["));
        for (i, r) in rows.iter().enumerate() {
            if i > 0 { s.push(','); }
            s.push_str(&format!("{{\"decisions\":{},\"cost\":[", r.decisions));
            for (k, v) in r.cost.iter().enumerate() {
                if k > 0 { s.push(','); }
                s.push_str(&format!("{v:.6}"));
            }
            s.push_str("],\"optimal\":[");
            for (k, v) in r.optimal.iter().enumerate() {
                if k > 0 { s.push(','); }
                s.push_str(&format!("{v:.6}"));
            }
            s.push_str("]}");
        }
        s.push_str("]}");
        std::fs::write(&path, s).expect("écriture du JSON");
        eprintln!("brut → {path}");
    }
}
