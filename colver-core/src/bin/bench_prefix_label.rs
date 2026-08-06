//! **Mesure B** du plan de regénération de la couche de scores
//! ([docs/data_gen/isdd_score_layer_v2.md](../../../docs/data_gen/isdd_score_layer_v2.md) §10) :
//! le **préfixe d'enchère** déplace-t-il l'étiquette, et de combien ?
//!
//! ## Ce que la mesure A a laissé
//!
//! A a établi qu'une enchère fabriquée est hors distribution, et classé les préfixes
//! possibles en quatre rangs par ce qu'ils **mentent** :
//!
//! | rang | source | mensonge |
//! |---|---|---|
//! | **or** | l'enchère réelle de v6 | rien |
//! | **argent** | épluchage, **relance** retirée | une enchère de plus ; l'auteur reste visible |
//! | **bronze** | épluchage, **ouverture** retirée | un siège devient muet |
//! | **fer** | construction §4 | tout le préfixe |
//!
//! **C'est une hypothèse ordonnée, pas un résultat.** Rien ne dit que l'ordre du
//! mensonge soit l'ordre du coût en points cartes. B le mesure.
//!
//! ## Le bras témoin, et pourquoi il n'est pas facultatif
//!
//! Deux étiquetages IS-DD de la **même** case ne rendent pas le même nombre : les
//! mondes sont échantillonnés. Sans mesurer ce bruit-là, un écart or/fer est
//! indiscernable de deux tirages du même bras. Le témoin est donc **le même préfixe,
//! une autre graine** — un cinquième du budget qui décide si les quatre autres
//! cinquièmes veulent dire quelque chose.
//!
//! ## Les cases ne sont pas interchangeables
//!
//! « or » nomme l'atout t₁, l'épluchage tombe sur t₂ ≠ t₁ le plus souvent. On ne
//! compare donc que ce qui partage une case :
//!
//! ```text
//! t₁ :  or(graine 1)   or(graine 2)   fer     ← le témoin et l'écart or/fer
//! t₂ :  épluchage      fer                    ← l'écart épluchage/fer
//! ```
//!
//! ## Usage
//!
//! ```bash
//! cargo build -p colver-core --release --features parallel --bin bench_prefix_label
//! ./target/release/bench_prefix_label --deals 500 --threads 256 --json pilote.json
//! ```
//!
//! Le sidecar playgen doit être debout (`playgen-up`) — et redescendre après.

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use colver_core::agent::{AgentSpec, MatchContext, Player};
use colver_core::bid_net::BidNet;
use colver_core::bidding::{self, BID_PASS};
use colver_core::game_replay::GameReplay;
use colver_core::solver;
use colver_core::state::{GameState, Phase};

#[path = "shared/auction.rs"]
mod auction;
use auction::{built_auction, dd_side, is_bid, is_raise, run_v6};

/// Joue la donne avec une enchère **imposée**, et rend les points cartes N-S.
///
/// L'enchère n'est pas demandée aux joueurs, elle leur est **montrée** : chaque action
/// passe par `observe` et `ctx.track` exactement comme si elle avait été choisie. C'est
/// ce qui fait que playgen voit le bon préfixe — il tokenise les jetons d'enchère du
/// contexte, pas ceux d'un état interne au bidder.
fn label_with_auction(
    hands: [u32; 4],
    dealer: u8,
    auction: &[u8],
    players: &mut [Box<dyn Player>; 4],
    ctx: &mut MatchContext,
) -> Result<u16, colver_core::agent::AgentError> {
    let mut state = GameState::new(dealer, hands);
    ctx.reset_deal(dealer);
    for p in players.iter_mut() {
        p.init_deal(&state);
    }

    for &a in auction {
        if state.phase != Phase::Bidding {
            break;
        }
        let before = state;
        let seat = before.current_player();
        for p in players.iter_mut() {
            p.observe(&before, seat, a);
        }
        ctx.track(&before, a);
        state.step(a);
    }
    if state.phase != Phase::Playing {
        // 4 passes : la donne n'a pas de contrat, donc pas d'étiquette.
        return Ok(u16::MAX);
    }

    while !state.is_terminal() {
        let seat = state.current_player();
        let before = state;
        let action = players[seat as usize].action(&before, ctx)?;
        for p in players.iter_mut() {
            p.observe(&before, seat, action);
        }
        ctx.track(&before, action);
        state.step(action);
    }
    Ok(state.points[0] as u16)
}

/// Une donne étiquetée sous les cinq bras.
#[derive(Clone, Copy)]
struct Row {
    t1: u8,
    t2: u8,
    /// L'épluchage a-t-il retiré une **relance** (argent) ou une **ouverture** (bronze) ?
    peel_is_raise: bool,
    or1: u16,
    or2: u16,
    fer1: u16,
    peel: u16,
    fer2: u16,
    /// Valeur DD N-S à t₁ et t₂ — repère, pas une étiquette.
    dd1: u8,
    dd2: u8,
}

struct Args {
    games: String,
    bot: String,
    bid_model: String,
    deals: usize,
    dets: Option<u32>,
    threads: usize,
    seed: u64,
    json: Option<String>,
    url: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        games: "data/training/isdd_games_v1.bin".into(),
        bot: "arena/bots/gen_isdd_cardpts.toml".into(),
        bid_model: "models/bid_v6_isdd_resume/bid_nn_final.bin".into(),
        deals: 500,
        dets: None,
        threads: 0,
        seed: 12345,
        json: None,
        url: None,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let next = |i: usize| -> String { v.get(i + 1).cloned().unwrap_or_default() };
    while i < v.len() {
        match v[i].as_str() {
            "--games" => { a.games = next(i); i += 2 }
            "--bot" => { a.bot = next(i); i += 2 }
            "--bid-model" => { a.bid_model = next(i); i += 2 }
            "--deals" => { a.deals = next(i).parse().unwrap(); i += 2 }
            "--dets" => { a.dets = Some(next(i).parse().unwrap()); i += 2 }
            "--threads" => { a.threads = next(i).parse().unwrap(); i += 2 }
            "--seed" => { a.seed = next(i).parse().unwrap(); i += 2 }
            "--json" => { a.json = Some(next(i)); i += 2 }
            "--url" => { a.url = Some(next(i)); i += 2 }
            "--help" | "-h" => {
                eprintln!("bench_prefix_label : le préfixe d'enchère déplace-t-il l'étiquette ?");
                eprintln!("  --games <path>   corpus dont on reprend donnes ET enchères réelles");
                eprintln!("  --bot <path>     spec du joueur étiqueteur (objective = card_points)");
                eprintln!("  --deals N        donnes (5 étiquetages chacune)");
                eprintln!("  --dets N         mondes par décision (défaut : celui du TOML)");
                eprintln!("  --threads N      sur-souscrire : le mur est le sidecar, pas le CPU");
                eprintln!("  --url <u[,u..]>  sidecar(s) ; défaut $COLVER_PLAYGEN_GPU_URL");
                eprintln!("  --json <path>");
                std::process::exit(0)
            }
            other => { eprintln!("argument inconnu : {other}"); std::process::exit(1) }
        }
    }
    if a.threads == 0 {
        a.threads = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(8) * 8;
    }
    a
}

/// Moyenne, écart-type et erreur-type d'une série de différences appariées.
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

fn show(name: &str, xs: &[f64], floor_sd: f64) {
    let (m, sd, se, n) = paired(xs);
    let z = if se > 0.0 { m / se } else { f64::NAN };
    let ratio = if floor_sd > 0.0 { sd / floor_sd } else { f64::NAN };
    println!(
        "  {name:<26} n={n:<6} moyenne {m:+7.2}  ±{se:.2}  (z={z:+5.1})   écart-type {sd:6.2}  = {ratio:.2}× le plancher"
    );
}

fn main() {
    let args = parse_args();

    let mut spec = AgentSpec::from_toml_file(&args.bot).unwrap_or_else(|e| {
        eprintln!("bot {} : {e}", args.bot);
        std::process::exit(1);
    });
    if let Some(d) = args.dets {
        spec.play.determinizations = d;
    }
    if let Some(u) = args.url.clone().or_else(|| std::env::var("COLVER_PLAYGEN_GPU_URL").ok()) {
        spec.worlds.url = Some(u);
    }
    if spec.worlds.url.is_none() {
        eprintln!("❌ pas de sidecar : --url ou $COLVER_PLAYGEN_GPU_URL");
        std::process::exit(1);
    }

    let replays = GameReplay::load_all(&args.games).expect("lecture du corpus");
    let n = replays.len().min(args.deals);
    eprintln!(
        "bench_prefix_label : {n} donnes × 5 bras = {} étiquetages, {} threads, {} mondes/décision",
        n * 5,
        args.threads,
        spec.play.determinizations
    );

    let rows: Arc<Mutex<Vec<Row>>> = Arc::new(Mutex::new(Vec::with_capacity(n)));
    let next = Arc::new(AtomicUsize::new(0));
    let done = Arc::new(AtomicUsize::new(0));
    let errors = Arc::new(AtomicUsize::new(0));
    let extra = Arc::new(AtomicUsize::new(0));
    let failed = Arc::new(AtomicBool::new(false));
    // Même politique que `gen_games_isdd` : un hoquet du sidecar ne doit pas tuer le
    // run, une panne qui dure doit l'arrêter. Le budget se compte en durée de panne.
    let budget = (args.threads * 4).max(n / 4).max(50);
    let start = Instant::now();
    let replays = Arc::new(replays);

    let mut handles = Vec::with_capacity(args.threads);
    for tid in 0..args.threads {
        let (rows, next, done, errors, failed) =
            (rows.clone(), next.clone(), done.clone(), errors.clone(), failed.clone());
        let extra = extra.clone();
        let replays = replays.clone();
        let spec = spec.clone();
        let bid_model = args.bid_model.clone();
        let seed = args.seed;
        handles.push(std::thread::spawn(move || {
            // Deux jeux de joueurs qui ne diffèrent QUE par la graine : c'est eux qui
            // font le bras témoin. Construits une fois par thread — les poids sont dans
            // un cache global, mais l'état par donne et les RNG ne le sont pas.
            let mut spec_a = spec.clone();
            spec_a.seed = seed ^ 0xA1;
            let mut spec_b = spec.clone();
            spec_b.seed = seed ^ 0xB2;
            let build = |sp: &AgentSpec| -> Option<[Box<dyn Player>; 4]> {
                (0..4).map(|s| sp.build(s)).collect::<Result<Vec<_>, _>>()
                    .ok()
                    .and_then(|v| v.try_into().ok())
            };
            let (mut pa, mut pb) = match (build(&spec_a), build(&spec_b)) {
                (Some(a), Some(b)) => (a, b),
                _ => {
                    eprintln!("thread {tid} : construction du bot impossible");
                    failed.store(true, Ordering::Relaxed);
                    return;
                }
            };
            let mut net = match BidNet::load(&bid_model) {
                Ok(nt) => nt,
                Err(e) => {
                    eprintln!("thread {tid} : modèle d'enchère : {e}");
                    failed.store(true, Ordering::Relaxed);
                    return;
                }
            };
            let mut obs: Vec<f32> = Vec::new();
            let mut ctx = MatchContext::new(0);
            let mut tt = solver::new_tt_buffer();

            loop {
                if failed.load(Ordering::Relaxed) {
                    break;
                }
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= n + extra.load(Ordering::Relaxed) || idx >= replays.len() {
                    break;
                }
                let r = &replays[idx];

                // --- le préfixe « or » : l'enchère réelle du corpus ---
                let mut g = GameState::new(r.dealer, r.hands);
                let mut or_auction: Vec<u8> = Vec::with_capacity(12);
                for &a in &r.actions {
                    if g.phase != Phase::Bidding {
                        break;
                    }
                    g.step(a);
                    or_auction.push(a);
                }
                if g.phase != Phase::Playing {
                    continue; // donne passée : rien à étiqueter
                }
                let t1 = g.contract.trump;

                // --- l'épluchage : la dernière annonce devient une passe, v6 continue ---
                let Some(bi) = or_auction.iter().rposition(|&a| is_bid(a)) else { continue };
                let peel_is_raise = is_raise(&or_auction, bi);
                let mut prefix: Vec<u8> = or_auction[..bi].to_vec();
                prefix.push(BID_PASS);
                let peel_auction = run_v6(&r.hands, r.dealer, u64::MAX, &prefix, &mut net, &mut obs);
                let mut g2 = GameState::new(r.dealer, r.hands);
                for &a in &peel_auction {
                    if g2.phase != Phase::Bidding {
                        break;
                    }
                    g2.step(a);
                }
                if g2.phase != Phase::Playing {
                    continue; // l'épluchage a vidé l'enchère
                }
                let t2 = g2.contract.trump;

                // --- les deux fers, construits sur t₁ et t₂ ---
                let dd1 = solver::solve_for_trump_reuse_tt(r.hands, r.dealer, t1, &mut tt)[0];
                let dd2 = solver::solve_for_trump_reuse_tt(r.hands, r.dealer, t2, &mut tt)[0];
                let key = (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
                let fer1 = built_auction(&r.hands, r.dealer, t1, dd1, key);
                let fer2 = built_auction(&r.hands, r.dealer, t2, dd2, key ^ 0x5DEE);

                // --- les cinq étiquetages ---
                let arms: [(&Vec<u8>, bool); 5] = [
                    (&or_auction, true),   // or, graine A
                    (&or_auction, false),  // or, graine B  ← le témoin
                    (&fer1, true),
                    (&peel_auction, true),
                    (&fer2, true),
                ];
                let mut out = [0u16; 5];
                let mut why: Option<String> = None;
                for (k, (auc, use_a)) in arms.iter().enumerate() {
                    let players = if *use_a { &mut pa } else { &mut pb };
                    match label_with_auction(r.hands, r.dealer, auc, players, &mut ctx) {
                        Ok(v) if v != u16::MAX => out[k] = v,
                        Ok(_) => { why = Some("enchère imposée sans contrat".into()); break }
                        Err(e) => { why = Some(format!("bras {k} : {e}")); break }
                    }
                }
                if let Some(w) = why {
                    let e = errors.fetch_add(1, Ordering::Relaxed) + 1;
                    // Le jeton est **rendu** : `--deals N` doit rester un compte de donnes
                    // abouties, sinon un hoquet du sidecar rétrécit l'échantillon en
                    // silence — et les donnes perdues ne sont pas un tirage au hasard,
                    // ce sont celles jouées pendant la saturation.
                    extra.fetch_add(1, Ordering::Relaxed);
                    if e <= 5 || e % 200 == 0 {
                        eprintln!("thread {tid} donne {idx} abandonnée ({e}e) : {w}");
                    }
                    if e > budget {
                        eprintln!("❌ {e} erreurs > budget {budget} — la panne dure, arrêt");
                        failed.store(true, Ordering::Relaxed);
                    }
                    continue;
                }

                rows.lock().unwrap().push(Row {
                    t1, t2, peel_is_raise,
                    or1: out[0], or2: out[1], fer1: out[2], peel: out[3], fer2: out[4],
                    dd1, dd2,
                });
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 50 == 0 {
                    let el = start.elapsed().as_secs_f64();
                    let rate = d as f64 / el;
                    eprintln!(
                        "  {d}/{n} donnes  {:.2} donnes/s  {:.1} étiquetages/s  ETA {:.0} s",
                        rate, rate * 5.0, (n - d) as f64 / rate
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
    eprintln!(
        "\n{} donnes étiquetées en {:.0} s ({:.2} donnes/s, {:.1} étiquetages/s), {} erreurs",
        rows.len(), el, rows.len() as f64 / el, rows.len() as f64 * 5.0 / el,
        errors.load(Ordering::Relaxed)
    );
    if rows.is_empty() {
        eprintln!("aucune donne — rien à dire");
        std::process::exit(1);
    }

    // --- le témoin d'abord : sans son écart-type, rien d'autre n'est lisible ---
    let ctl: Vec<f64> = rows.iter().map(|r| r.or1 as f64 - r.or2 as f64).collect();
    let (cm, csd, cse, cn) = paired(&ctl);
    println!("\n=== TÉMOIN — même préfixe « or », deux graines ===");
    println!("  n={cn}  moyenne {cm:+.2} ±{cse:.2}  **écart-type apparié {csd:.2} points cartes**");
    println!("  C'est le plancher : tout écart ci-dessous se lit contre lui, pas dans l'absolu.");

    println!("\n=== CE QUE LE PRÉFIXE DÉPLACE (points cartes N-S, différence appariée) ===");
    let or_fer: Vec<f64> = rows.iter().map(|r| r.or1 as f64 - r.fer1 as f64).collect();
    let peel_fer: Vec<f64> = rows.iter().map(|r| r.peel as f64 - r.fer2 as f64).collect();
    show("or − fer   (case t₁)", &or_fer, csd);
    show("épluchage − fer (t₂)", &peel_fer, csd);

    let ag: Vec<f64> = rows.iter().filter(|r| r.peel_is_raise)
        .map(|r| r.peel as f64 - r.fer2 as f64).collect();
    let br: Vec<f64> = rows.iter().filter(|r| !r.peel_is_raise)
        .map(|r| r.peel as f64 - r.fer2 as f64).collect();
    println!("\n  dont, selon ce que l'épluchage a retiré :");
    show("argent (relance) − fer", &ag, csd);
    show("bronze (ouverture) − fer", &br, csd);

    println!("\n  repères : t₁ = t₂ dans {:.1} % des donnes ; valeur DD moyenne {:.0} / {:.0}",
             100.0 * rows.iter().filter(|r| r.t1 == r.t2).count() as f64 / rows.len() as f64,
             rows.iter().map(|r| r.dd1 as f64).sum::<f64>() / rows.len() as f64,
             rows.iter().map(|r| r.dd2 as f64).sum::<f64>() / rows.len() as f64);

    // --- dimensionnement du run principal, dérivé du témoin ---
    println!("\n=== DIMENSIONNEMENT ===");
    for target in [1.0f64, 2.0, 5.0] {
        // n pour que l'erreur-type soit à `target / 2` — soit un effet de `target`
        // points détectable à 2 sigma.
        let need = (2.0 * csd / target).powi(2);
        println!("  détecter {target:.0} pt à 2σ : {need:.0} donnes ({:.0} étiquetages, {:.0} min à {:.2} donnes/s)",
                 need * 5.0, need / (rows.len() as f64 / el) / 60.0, rows.len() as f64 / el);
    }

    if let Some(p) = args.json {
        let body = format!(
            "{{\"deals\":{},\"secs\":{:.1},\"control_sd\":{:.4},\"control_mean\":{:.4},\
             \"or_minus_fer\":{{\"mean\":{:.4},\"sd\":{:.4},\"se\":{:.4},\"n\":{}}},\
             \"peel_minus_fer\":{{\"mean\":{:.4},\"sd\":{:.4},\"se\":{:.4},\"n\":{}}},\
             \"silver_minus_fer\":{{\"mean\":{:.4},\"sd\":{:.4},\"se\":{:.4},\"n\":{}}},\
             \"bronze_minus_fer\":{{\"mean\":{:.4},\"sd\":{:.4},\"se\":{:.4},\"n\":{}}},\
             \"rows\":[{}]}}",
            rows.len(), el, csd, cm,
            paired(&or_fer).0, paired(&or_fer).1, paired(&or_fer).2, or_fer.len(),
            paired(&peel_fer).0, paired(&peel_fer).1, paired(&peel_fer).2, peel_fer.len(),
            paired(&ag).0, paired(&ag).1, paired(&ag).2, ag.len(),
            paired(&br).0, paired(&br).1, paired(&br).2, br.len(),
            rows.iter()
                .map(|r| format!(
                    "[{},{},{},{},{},{},{},{},{},{}]",
                    r.t1, r.t2, r.peel_is_raise as u8,
                    r.or1, r.or2, r.fer1, r.peel, r.fer2, r.dd1, r.dd2))
                .collect::<Vec<_>>().join(","),
        );
        std::fs::write(&p, body).expect("écriture json");
        eprintln!("[json] {p}");
    }
}
