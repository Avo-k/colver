//! **Le mécanisme derrière la mesure B** : une fausse enchère dégrade-t-elle les *mondes*
//! que playgen produit, et de combien ?
//!
//! ## Ce que B a laissé ouvert
//!
//! B a mesuré l'effet **de bout en bout** — un préfixe réaliste fait jouer le preneur
//! +4,36 points cartes mieux qu'un préfixe construit
//! ([docs/data_gen/isdd_score_layer_v2.md](../../../docs/data_gen/isdd_score_layer_v2.md) §4).
//! L'explication avancée était : *fausse enchère → playgen place mal la force → IS-DD
//! cherche avec de mauvaises croyances → il joue moins bien.* Trois maillons, un seul
//! chiffre. Ce binaire mesure **le premier maillon isolément**.
//!
//! ## Pourquoi c'est beaucoup moins cher que B
//!
//! B devait jouer la donne, donc résoudre des milliers de positions en double dummy.
//! Ici la vérité terrain est **déjà connue** — le corpus porte les quatre mains — donc
//! il n'y a rien à résoudre : on échantillonne des mondes et on note directement où
//! playgen place chaque carte cachée. **Aucun solve DD dans la boucle de mesure**, et
//! l'échantillonnage tourne en CPU pur (`playgen::infer`), donc **aucun GPU** : la
//! mesure cohabite avec une génération de couche qui monopolise les deux cartes.
//!
//! ## Les bras, repris de B à l'identique
//!
//! ```text
//! t₁ :  or(graine A)   or(graine B)   fer₁      ← le témoin et l'écart or/fer
//! t₂ :  épluchage      fer₂                     ← l'écart épluchage/fer
//! ```
//!
//! `or` est l'enchère réelle du corpus, l'épluchage remplace sa dernière annonce par une
//! passe et laisse v6 continuer, le fer est la construction du §4. La construction vit
//! dans `shared/auction.rs`, partagée avec A et B — une copie dériverait, et la mesure
//! ne porterait plus sur les mêmes enchères.
//!
//! **Le bras témoin n'est pas facultatif.** Deux tirages de mondes du *même* préfixe ne
//! rendent pas la même marginale ; sans ce plancher, un écart or/fer est indiscernable
//! de deux tirages du même bras. C'est la leçon de B, et elle vaut ici aussi.
//!
//! ## La note
//!
//! À l'entame, l'observateur voit ses 8 cartes ; les 24 autres sont réparties entre les
//! trois sièges restants. On échantillonne `--worlds` mondes et on compte où chaque
//! carte cachée atterrit, ce qui donne `p(carte → siège)`. Deux notes contre la vérité :
//!
//! - **exactitude** — la carte est-elle *le plus souvent* chez son vrai porteur ;
//! - **log-vraisemblance négative** — `−ln p(vrai porteur)`, lissée à la Laplace sur les
//!   trois sièges candidats. C'est une règle de score propre : contrairement à
//!   l'exactitude, elle punit une distribution confiante et fausse.
//!
//! **Le repère absolu est analytique, pas échantillonné** : à l'entame les trois sièges
//! cachés tiennent 8 cartes chacun, donc un tirage uniforme sous contrainte donne
//! exactement `p = 1/3` par carte — exactitude 33,3 %, NLL `ln 3 = 1,0986`. Inutile de
//! le simuler, et le simuler ajouterait du bruit à une constante.
//!
//! Les quatre sièges servent d'observateur à tour de rôle. C'est ce qui **supprime le
//! confondant** : l'épluchage peut changer le preneur, donc l'entameur, donc quel siège
//! observerait en production. Les mains, elles, ne dépendent que de la donne.
//!
//! ```bash
//! cargo build -p colver-core --release --features parallel --bin bench_prefix_worlds
//! ./target/release/bench_prefix_worlds --deals 150 --worlds 32 --threads 8 --json out.json
//! ```
//!
//! Aucun sidecar requis — c'est le propos.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use colver_core::bid_net::BidNet;
use colver_core::bidding::BID_PASS;
use colver_core::game_replay::GameReplay;
use colver_core::playgen::analysis::PlaygenAnalyst;
use colver_core::playgen::infer::PlaygenModel;
use colver_core::solver;
use colver_core::state::{GameState, Phase};

use rand::{Rng, SeedableRng};
use rand::rngs::StdRng;

#[path = "shared/auction.rs"]
mod auction;
use auction::{built_auction, is_bid, is_raise, run_v6};

const N_ARMS: usize = 5;
const ARM_NAMES: [&str; N_ARMS] =
    ["or (graine A)", "or (graine B) témoin", "fer sur t1", "épluchage sur t2", "fer sur t2"];

/// Une donne mesurée : la note de chaque bras, plus de quoi stratifier.
#[derive(Clone)]
struct Row {
    peel_is_raise: bool,
    /// Exactitude par bras — fraction de cartes cachées placées chez leur vrai porteur.
    acc: [f64; N_ARMS],
    /// NLL par bras.
    nll: [f64; N_ARMS],
}

struct Args {
    games: String,
    bid_model: String,
    playgen: String,
    deals: usize,
    worlds: usize,
    threads: usize,
    temperature: f32,
    seed: u64,
    json: Option<String>,
}

fn parse_args() -> Args {
    let mut a = Args {
        games: "data/training/isdd_games_v1.bin".into(),
        bid_model: "models/bid_v6_isdd_resume/bid_nn_final.bin".into(),
        playgen: "models/playgen/playgen_v2_final.bin".into(),
        deals: 150,
        worlds: 32,
        threads: 8,
        temperature: 1.0,
        seed: 20260806,
        json: None,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let nx = |i: usize| v.get(i + 1).cloned().unwrap_or_default();
    while i < v.len() {
        match v[i].as_str() {
            "--games" => { a.games = nx(i); i += 2 }
            "--bid-model" => { a.bid_model = nx(i); i += 2 }
            "--playgen" => { a.playgen = nx(i); i += 2 }
            "--deals" => { a.deals = nx(i).parse().unwrap(); i += 2 }
            "--worlds" => { a.worlds = nx(i).parse().unwrap(); i += 2 }
            "--threads" => { a.threads = nx(i).parse().unwrap(); i += 2 }
            "--temperature" => { a.temperature = nx(i).parse().unwrap(); i += 2 }
            "--seed" => { a.seed = nx(i).parse().unwrap(); i += 2 }
            "--json" => { a.json = Some(nx(i)); i += 2 }
            "--help" | "-h" => {
                eprintln!("bench_prefix_worlds : le préfixe d'enchère dégrade-t-il les mondes playgen ?");
                eprintln!("  --games <COLVGM0x>  corpus (défaut isdd_games_v1.bin)");
                eprintln!("  --deals N           donnes           (défaut 150)");
                eprintln!("  --worlds N          mondes par (bras, observateur)  (défaut 32)");
                eprintln!("  --threads N         CPU pur, aucun GPU  (défaut 8)");
                eprintln!("  --json <path>       brut par donne");
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

/// Le siège qui détient chaque carte, ou 255 pour aucun.
fn holder_of(hands: &[u32; 4]) -> [u8; 32] {
    let mut h = [255u8; 32];
    for (p, &set) in hands.iter().enumerate() {
        let mut s = set;
        while s != 0 {
            h[s.trailing_zeros() as usize] = p as u8;
            s &= s - 1;
        }
    }
    h
}

/// Note d'un préfixe : moyenne sur les quatre observateurs de l'exactitude et de la NLL
/// des marginales, contre les vraies mains.
///
/// Rend `None` si l'enchère ne mène pas au jeu (donne passée) ou si le sampler refuse —
/// un bras muet doit faire tomber la donne entière, pas rendre une note incomparable.
fn score_prefix(
    hands: &[u32; 4],
    dealer: u8,
    auction: &[u8],
    model: &Arc<PlaygenModel>,
    n_worlds: usize,
    temperature: f32,
    rng: &mut impl Rng,
) -> Option<(f64, f64)> {
    let truth = holder_of(hands);
    let mut acc_sum = 0.0;
    let mut nll_sum = 0.0;
    let mut cards = 0usize;

    for observer in 0..4u8 {
        let mut st = GameState::new(dealer, *hands);
        let mut an = PlaygenAnalyst::new(model.clone());
        an.init_deal(&st, observer);
        // L'enchère est **montrée** au sampler action par action : playgen tokenise les
        // jetons d'enchère du contexte, donc c'est là, et nulle part ailleurs, que le
        // préfixe entre dans la mesure.
        for &a in auction {
            if st.phase != Phase::Bidding {
                break;
            }
            let before = st;
            let p = st.current_player();
            an.observe(&before, p, a);
            st.step(a);
        }
        if st.phase != Phase::Playing {
            return None;
        }
        let w = an.marginals(&st, n_worlds, temperature, rng)?;

        let mine = hands[observer as usize];
        for c in 0..32usize {
            if mine & (1u32 << c) != 0 {
                continue; // l'observateur la voit
            }
            let t = truth[c] as usize;
            // argmax sur les sièges — l'observateur ne peut pas la tenir, mais on le
            // laisse concourir : s'il gagne, c'est que le sampler s'est trompé, et le
            // masquer maquillerait l'erreur.
            let mut best = 0usize;
            for p in 1..4 {
                if w[p][c] > w[best][c] {
                    best = p;
                }
            }
            acc_sum += if best == t { 1.0 } else { 0.0 };
            // Lissage de Laplace sur les trois sièges candidats : à 32 mondes, une carte
            // jamais placée chez son vrai porteur donnerait p = 0 et une NLL infinie,
            // donc une seule carte dominerait toute la moyenne.
            let count = w[t][c] as f64 * n_worlds as f64;
            let p_hat = (count + 1.0) / (n_worlds as f64 + 3.0);
            nll_sum += -p_hat.ln();
            cards += 1;
        }
    }
    if cards == 0 {
        return None;
    }
    Some((acc_sum / cards as f64, nll_sum / cards as f64))
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
    println!("  {name:<30} n={n:<5} {m:+8.4} ±{se:.4} {unit}  (z={z:+6.1})   σ={sd:.4}");
}

/// Écrit le brut, par fichier temporaire puis `rename`, **en cours de run**.
///
/// Ne garder les lignes qu'en mémoire jusqu'au `join` rend un run non interruptible :
/// l'arrêter à 110 donnes sur 150 perd tout, sans rien sur disque. `gen_score_layer`
/// a un checkpoint pour exactement cette raison. `rename` plutôt qu'une écriture en
/// place, sinon un lecteur qui tombe pendant l'écriture voit un JSON tronqué.
fn write_json(path: &str, rows: &[Row], worlds: usize) {
    let mut s = String::from("{\"arms\":[");
    for (k, name) in ARM_NAMES.iter().enumerate() {
        if k > 0 { s.push(','); }
        s.push_str(&format!("\"{name}\""));
    }
    s.push_str(&format!("],\"worlds\":{worlds},\"rows\":["));
    for (i, r) in rows.iter().enumerate() {
        if i > 0 { s.push(','); }
        s.push_str(&format!("{{\"peel_is_raise\":{},\"acc\":[", r.peel_is_raise));
        for (k, v) in r.acc.iter().enumerate() {
            if k > 0 { s.push(','); }
            s.push_str(&format!("{v:.6}"));
        }
        s.push_str("],\"nll\":[");
        for (k, v) in r.nll.iter().enumerate() {
            if k > 0 { s.push(','); }
            s.push_str(&format!("{v:.6}"));
        }
        s.push_str("]}");
    }
    s.push_str("]}");
    let tmp = format!("{path}.tmp");
    if let Err(e) = std::fs::write(&tmp, s).and_then(|()| std::fs::rename(&tmp, path)) {
        eprintln!("  ⚠ écriture de {path} : {e}");
    }
}

fn main() {
    let args = parse_args();

    let model = Arc::new(
        PlaygenModel::load(&args.playgen).unwrap_or_else(|e| {
            eprintln!("playgen {} : {e}", args.playgen);
            std::process::exit(1);
        }),
    );
    let replays = GameReplay::load_all(&args.games).expect("lecture du corpus");
    let n = replays.len().min(args.deals);
    eprintln!(
        "bench_prefix_worlds : {n} donnes × {N_ARMS} bras × 4 observateurs × {} mondes \
         = {} mondes CPU, {} threads, AUCUN GPU",
        args.worlds,
        n * N_ARMS * 4 * args.worlds,
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
        let model = model.clone();
        let (bid_model, seed, worlds, temperature) =
            (args.bid_model.clone(), args.seed, args.worlds, args.temperature);
        let json = args.json.clone();
        handles.push(std::thread::spawn(move || {
            let mut net = BidNet::load(&bid_model).expect("bid v6");
            let mut obs = vec![0.0f32; net.obs_dim()];
            let mut tt = solver::new_tt_buffer();

            loop {
                let idx = next.fetch_add(1, Ordering::Relaxed);
                if idx >= n || idx >= replays.len() {
                    break;
                }
                let r = &replays[idx];

                // --- « or » : l'enchère réelle du corpus ---
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
                    skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let t1 = g.contract.trump;

                // --- l'épluchage : la dernière annonce devient une passe, v6 continue ---
                let Some(bi) = or_auction.iter().rposition(|&a| is_bid(a)) else {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                };
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
                    skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }
                let t2 = g2.contract.trump;

                // --- les deux fers ---
                let dd1 = solver::solve_for_trump_reuse_tt(r.hands, r.dealer, t1, &mut tt)[0];
                let dd2 = solver::solve_for_trump_reuse_tt(r.hands, r.dealer, t2, &mut tt)[0];
                let key = (idx as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) ^ seed;
                let fer1 = built_auction(&r.hands, r.dealer, t1, dd1, key);
                let fer2 = built_auction(&r.hands, r.dealer, t2, dd2, key ^ 0x5DEE);

                // Deux graines par donne : le témoin est « or » tiré une seconde fois.
                // Les trois autres bras partagent la graine A, donc leur écart à `or1`
                // est apparié sur le même tirage de mondes.
                let mut rng_a = StdRng::seed_from_u64(key);
                let mut rng_b = StdRng::seed_from_u64(key ^ 0xA5A5_5A5A);

                let arms: [(&Vec<u8>, bool); N_ARMS] = [
                    (&or_auction, true),
                    (&or_auction, false), // témoin
                    (&fer1, true),
                    (&peel_auction, true),
                    (&fer2, true),
                ];
                let mut acc = [0.0f64; N_ARMS];
                let mut nll = [0.0f64; N_ARMS];
                let mut ok = true;
                for (k, (auc, use_a)) in arms.iter().enumerate() {
                    let rng: &mut StdRng = if *use_a { &mut rng_a } else { &mut rng_b };
                    match score_prefix(&r.hands, r.dealer, auc, &model, worlds, temperature, rng) {
                        Some((a, l)) => { acc[k] = a; nll[k] = l }
                        None => { ok = false; break }
                    }
                }
                if !ok {
                    skipped.fetch_add(1, Ordering::Relaxed);
                    continue;
                }

                rows.lock().unwrap().push(Row { peel_is_raise, acc, nll });
                let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                if d % 10 == 0 {
                    if let Some(p) = json.as_deref() {
                        let snap = rows.lock().unwrap().clone();
                        write_json(p, &snap, worlds);
                    }
                    let el = start.elapsed().as_secs_f64();
                    eprintln!(
                        "  {d}/{n} donnes  {:.2} donnes/s  ETA {:.0} s",
                        d as f64 / el,
                        (n - d) as f64 / (d as f64 / el)
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
        "\n{} donnes en {:.0} s ({:.2} donnes/s), {} écartées",
        rows.len(), el, rows.len() as f64 / el, skipped.load(Ordering::Relaxed)
    );
    if rows.is_empty() {
        eprintln!("aucune donne — rien à dépouiller");
        return;
    }

    let uniform_nll = 3f64.ln();
    println!("\n=== Niveau absolu par bras ({} donnes, {} mondes/observateur) ===",
             rows.len(), args.worlds);
    println!("  repère uniforme sous contrainte (analytique) : exactitude 33,33 %, NLL {uniform_nll:.4}\n");
    for k in 0..N_ARMS {
        let a: Vec<f64> = rows.iter().map(|r| r.acc[k] * 100.0).collect();
        let l: Vec<f64> = rows.iter().map(|r| r.nll[k]).collect();
        let (ma, _, sea, _) = paired(&a);
        let (ml, _, sel, _) = paired(&l);
        println!("  {:<22} exactitude {ma:6.2} % ±{sea:.2}   NLL {ml:.4} ±{sel:.4}",
                 ARM_NAMES[k]);
    }

    // Écarts appariés contre le bras « or (graine A) ». Le témoin donne le plancher :
    // il est le même préfixe, donc son écart à `or1` est du bruit d'échantillonnage pur.
    println!("\n=== Écarts appariés contre « or (graine A) » ===");
    println!("  (exactitude en points de %, NLL en nats ; NÉGATIF = moins bon que l'or)\n");
    let mut floor_sd = f64::NAN;
    for k in 1..N_ARMS {
        let da: Vec<f64> = rows.iter().map(|r| (r.acc[k] - r.acc[0]) * 100.0).collect();
        let dl: Vec<f64> = rows.iter().map(|r| -(r.nll[k] - r.nll[0])).collect();
        if k == 1 {
            floor_sd = paired(&da).1;
        }
        show(&format!("{} : exactitude", ARM_NAMES[k]), &da, "pt");
        show(&format!("{} : −ΔNLL", ARM_NAMES[k]), &dl, "nat");
    }
    println!("\n  plancher de bruit (témoin, σ de l'exactitude) : {floor_sd:.4} pt");
    println!("  Un bras dont l'écart tient dans ce plancher n'est pas distinguable de");
    println!("  deux tirages du même préfixe. ⚠️ Mais « petit devant le bruit » ne veut");
    println!("  PAS dire négligeable : le bruit du témoin est non biaisé et se moyenne,");
    println!("  un décalage de rang est systématique et reste dans chaque étiquette.");

    // La distinction argent/bronze est le pari de la couche : elle sépare l'épluchage
    // d'une relance (l'auteur reste visible) de celui d'une ouverture (siège muet).
    println!("\n=== L'épluchage, séparé en argent et bronze ===");
    for (label, want) in [("argent (relance retirée)", true), ("bronze (ouverture retirée)", false)] {
        let da: Vec<f64> = rows.iter().filter(|r| r.peel_is_raise == want)
            .map(|r| (r.acc[3] - r.acc[0]) * 100.0).collect();
        let dl: Vec<f64> = rows.iter().filter(|r| r.peel_is_raise == want)
            .map(|r| -(r.nll[3] - r.nll[0])).collect();
        show(&format!("{label} : exactitude"), &da, "pt");
        show(&format!("{label} : −ΔNLL"), &dl, "nat");
    }

    if let Some(path) = args.json.as_deref() {
        write_json(path, &rows, args.worlds);
        eprintln!("brut → {path}");
    }
}
