//! **Mesure C** du plan de regénération de la couche de scores
//! ([docs/data_gen/isdd_score_layer_v2.md](../../../docs/data_gen/isdd_score_layer_v2.md) §6) :
//! `P(capot | mes 8 cartes)`.
//!
//! ## Pourquoi ce n'est pas 16,08 %
//!
//! `base_5M.bin` dit qu'un capot N-S est **atteignable en DD** dans 16,08 % des donnes.
//! Ce chiffre est un piège pour dimensionner une strate d'entraînement : le solveur voit
//! les quatre mains. Un bidder ne voit que la sienne, et la question qu'il se pose est
//! *« la mienne suffit-elle à espérer les huit levées ? »* — une **conditionnelle**, pas
//! une fréquence globale.
//!
//! L'écart entre les deux est tout l'enjeu. Si `P(capot | main)` est plate et basse
//! partout, aucune main ne justifie l'annonce et sur-échantillonner « les mains à capot »
//! revient à enseigner un a priori faux. Si elle a une queue — quelques milliers de
//! classes de mains où elle dépasse 50 % — alors ces classes *sont* `tail_100k`, et le
//! modèle peut apprendre à les reconnaître.
//!
//! ## Méthode
//!
//! Une main tirée, placée au siège 0, puis `K` **complétions** : les 24 cartes restantes
//! redistribuées au hasard aux trois autres sièges, donneur retiré à chaque fois. Chaque
//! complétion est résolue aux 4 atouts. `P(capot | main) ≈` fraction des complétions où
//! au moins un atout donne 252 à N-S.
//!
//! **Le donneur est retiré à chaque complétion, pas fixé** : il décide qui entame, donc
//! il change la valeur DD. Le fixer mesurerait `P(capot | main, position)`, qui est une
//! autre question — et la moyenne sur les positions est ce qu'un bidder voit avant de
//! connaître la sienne.
//!
//! Aucun GPU : c'est le solveur DD qu'on interroge, pas playgen.
//!
//! ```bash
//! cargo build -p colver-core --release --bin bench_capot_prior
//! ./target/release/bench_capot_prior --hands 2000 --worlds 100 --json c.json
//! ```

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::rngs::StdRng;
use rand::seq::SliceRandom;
use rand::{Rng, SeedableRng};

use colver_core::bid_eval::evaluate_for_trump;
use colver_core::card::Suit;
use colver_core::solver;

/// Ce qu'on retient d'une main : sa probabilité de capot, et deux repères bon marché
/// pour savoir si un modèle pourrait la reconnaître **sans** simuler.
#[derive(Clone, Copy)]
struct HandRow {
    hand: u32,
    /// Fraction des complétions où N-S réalise un capot à au moins un atout.
    p_any: f32,
    /// Fraction où le **meilleur atout de la main** (au sens `evaluate_for_trump`)
    /// donne le capot. C'est la quantité qu'un bidder utiliserait vraiment : il annonce
    /// une couleur, pas « l'une des quatre ».
    p_best: f32,
    /// Points N-S moyens au meilleur atout — pour situer la main sans le capot.
    mean_best: f32,
    /// `max evaluate_for_trump` sur les 4 couleurs (0-40).
    eval_max: u16,
}

fn main() {
    let mut hands_n = 2000usize;
    let mut worlds = 100usize;
    let mut seed = 42u64;
    let mut threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(8);
    let mut json: Option<String> = None;

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--hands" => { i += 1; hands_n = args[i].parse().unwrap() }
            "--worlds" => { i += 1; worlds = args[i].parse().unwrap() }
            "--seed" => { i += 1; seed = args[i].parse().unwrap() }
            "--threads" => { i += 1; threads = args[i].parse().unwrap() }
            "--json" => { i += 1; json = Some(args[i].clone()) }
            "--help" | "-h" => {
                eprintln!("bench_capot_prior : P(capot | mes 8 cartes), sans GPU");
                eprintln!("  --hands N    mains tirées (défaut 2000)");
                eprintln!("  --worlds K   complétions par main (défaut 100)");
                eprintln!("  --threads N  défaut : nproc");
                eprintln!("  --json <p>");
                std::process::exit(0)
            }
            other => { eprintln!("argument inconnu : {other}"); std::process::exit(1) }
        }
        i += 1;
    }

    eprintln!(
        "bench_capot_prior : {hands_n} mains × {worlds} complétions × 4 atouts = {} solves, {threads} threads",
        hands_n * worlds * 4
    );

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let start = std::time::Instant::now();

    let parts: Vec<Vec<HandRow>> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|t| {
                let next = &next;
                let done = &done;
                s.spawn(move || {
                    let mut rng = StdRng::seed_from_u64(seed ^ (t as u64).wrapping_mul(0x9E37_79B9));
                    let mut tt = solver::new_tt_buffer();
                    let mut out: Vec<HandRow> = Vec::new();
                    let mut deck: Vec<u8> = Vec::with_capacity(32);

                    loop {
                        let idx = next.fetch_add(1, Ordering::Relaxed);
                        if idx >= hands_n {
                            break;
                        }
                        // Une main uniforme : 8 cartes tirées des 32. C'est le bon
                        // a priori — la question est « parmi les mains que je reçois »,
                        // pas « parmi les mains intéressantes ».
                        deck.clear();
                        deck.extend(0..32u8);
                        deck.shuffle(&mut rng);
                        let hand: u32 = deck[..8].iter().fold(0u32, |m, &c| m | (1 << c));
                        let rest: Vec<u8> = deck[8..].to_vec();

                        let eval: [u16; 4] =
                            std::array::from_fn(|k| evaluate_for_trump(hand, Suit::from_u8(k as u8)));
                        let best_t = (0..4).max_by_key(|&k| eval[k]).unwrap() as u8;
                        let eval_max = eval[best_t as usize];

                        let (mut any, mut at_best, mut sum_best) = (0usize, 0usize, 0f64);
                        let mut pool = rest.clone();
                        for _ in 0..worlds {
                            pool.shuffle(&mut rng);
                            let mut hs = [hand, 0, 0, 0];
                            for (k, chunk) in pool.chunks(8).enumerate() {
                                hs[k + 1] = chunk.iter().fold(0u32, |m, &c| m | (1 << c));
                            }
                            // Le donneur est retiré à chaque complétion : il décide qui
                            // entame, donc il change la valeur DD.
                            let dealer: u8 = rng.gen_range(0..4);
                            let mut capot = false;
                            for tr in 0..4u8 {
                                let ns = solver::solve_for_trump_reuse_tt(hs, dealer, tr, &mut tt)[0];
                                if tr == best_t {
                                    sum_best += ns as f64;
                                    if ns == 252 {
                                        at_best += 1;
                                    }
                                }
                                if ns == 252 {
                                    capot = true;
                                }
                            }
                            if capot {
                                any += 1;
                            }
                        }

                        out.push(HandRow {
                            hand,
                            p_any: (any as f64 / worlds as f64) as f32,
                            p_best: (at_best as f64 / worlds as f64) as f32,
                            mean_best: (sum_best / worlds as f64) as f32,
                            eval_max,
                        });

                        let d = done.fetch_add(1, Ordering::Relaxed) + 1;
                        if d % 200 == 0 {
                            let el = start.elapsed().as_secs_f64();
                            eprintln!("  {d}/{hands_n} mains  {:.1} mains/s  ETA {:.0} s",
                                      d as f64 / el, (hands_n - d) as f64 / (d as f64 / el));
                        }
                    }
                    out
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });

    let mut rows: Vec<HandRow> = parts.into_iter().flatten().collect();
    let n = rows.len() as f64;
    let el = start.elapsed().as_secs_f64();
    eprintln!("\n{} mains en {:.0} s", rows.len(), el);

    let mean_any = rows.iter().map(|r| r.p_any as f64).sum::<f64>() / n;
    let mean_best = rows.iter().map(|r| r.p_best as f64).sum::<f64>() / n;
    println!("\n=== P(capot | main) — {} mains × {worlds} complétions ===", rows.len());
    println!("  moyenne, un atout quelconque : {:.4} %", 100.0 * mean_any);
    println!("  moyenne, au meilleur atout   : {:.4} %", 100.0 * mean_best);
    println!("  (repère : 16,08 % des donnes de base_5M ont un capot N-S atteignable en DD —");
    println!("   c'est la marginale, vue des QUATRE mains, pas cette conditionnelle-ci)");

    println!("\n  combien de mains dépassent un seuil (au meilleur atout) :");
    for thr in [0.01f32, 0.05, 0.10, 0.25, 0.50, 0.75] {
        let k = rows.iter().filter(|r| r.p_best >= thr).count();
        println!("    P ≥ {:>4.0} % : {k:>6} mains ({:.3} %)", 100.0 * thr, 100.0 * k as f64 / n);
    }

    // Une strate ne vaut que si elle est *reconnaissable*. `evaluate_for_trump` est le
    // repère le moins cher qui existe ; s'il sépare déjà, la strate se construit sans
    // simuler.
    rows.sort_by(|a, b| b.p_best.total_cmp(&a.p_best));
    let top = (rows.len() / 100).max(1);
    let top_eval = rows[..top].iter().map(|r| r.eval_max as f64).sum::<f64>() / top as f64;
    let all_eval = rows.iter().map(|r| r.eval_max as f64).sum::<f64>() / n;
    println!("\n  centile supérieur en P(capot) : eval_max moyen {top_eval:.1} contre {all_eval:.1} sur l'ensemble");
    println!("  P(capot au meilleur atout) par tranche d'eval_max :");
    for lo in (0..40).step_by(5) {
        let sel: Vec<&HandRow> = rows.iter().filter(|r| r.eval_max >= lo && r.eval_max < lo + 5).collect();
        if sel.len() < 5 {
            continue;
        }
        let p = sel.iter().map(|r| r.p_best as f64).sum::<f64>() / sel.len() as f64;
        let mb = sel.iter().map(|r| r.mean_best as f64).sum::<f64>() / sel.len() as f64;
        println!("    [{lo:>2} ; {:>2}) : n={:>6}  P={:>7.3} %  points moyens {mb:.0}",
                 lo + 5, sel.len(), 100.0 * p);
    }

    if let Some(p) = json {
        let body = format!(
            "{{\"hands\":{},\"worlds\":{worlds},\"secs\":{:.1},\"mean_p_any\":{:.6},\
             \"mean_p_best\":{:.6},\"rows\":[{}]}}",
            rows.len(), el, mean_any, mean_best,
            rows.iter()
                .map(|r| format!("[{},{:.4},{:.4},{:.1},{}]",
                                 r.hand, r.p_any, r.p_best, r.mean_best, r.eval_max))
                .collect::<Vec<_>>().join(","),
        );
        std::fs::write(&p, body).expect("écriture json");
        eprintln!("[json] {p}");
    }
}
