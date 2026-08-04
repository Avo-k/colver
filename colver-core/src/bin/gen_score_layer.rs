//! Génère une **couche de scores** `COLVSC01` avec des mondes playgen.
//!
//! Plan complet, mesures et pièges :
//! [docs/data_gen/isdd_score_layer_v2.md](../../../docs/data_gen/isdd_score_layer_v2.md).
//!
//! Une couche est un `[u8; 4]` par donne — les points cartes N-S sous chaque atout, sous
//! jeu fort. C'est l'entrée de la reward du bidder (`train_bid_nn --reward real
//! --scores`). L'existante date d'avril 2026 et a été produite en **mondes uniformes**
//! par un IS-DD à 2 mondes au pli 1.
//!
//! ## Ce que ce binaire fait de plus que `enrich_pool_isdd`
//!
//! 1. **Des mondes playgen** (sidecar), pas des mondes contraints-uniformes.
//! 2. **Une vraie enchère par case**, dans la mesure du possible — voir ci-dessous.
//! 3. **L'objectif `card_points`**, seul compatible avec un `[u8;4]` : un tableau indexé
//!    par le seul atout ne peut porter qu'une quantité indépendante du contrat.
//!
//! ## Les quatre rangs de préfixe (mesure A)
//!
//! playgen ne peut pas échantillonner un monde sans jetons d'enchère dans son préfixe, et
//! l'enchère fabriquée par la construction naïve est **hors distribution** : jamais
//! contestée (0 % contre 81,3 % en réel), une seule annonce (100 % contre 11,9 %). Pire,
//! aucune enchère à *atout imposé* ne peut être réaliste — la contestation **est** le
//! mécanisme qui sélectionne l'atout, donc la forcer sur une couleur la supprime.
//!
//! D'où la hiérarchie, par ordre décroissant de fidélité :
//!
//! | rang | source | ce qu'on ment |
//! |---|---|---|
//! | **or** | l'enchère libre de v6 | rien |
//! | **argent** | épluchage, **relance** retirée | une enchère de plus |
//! | **bronze** | épluchage, **ouverture** retirée | un siège devient muet |
//! | **fer** | construction | tout le préfixe |
//!
//! L'épluchage retire la dernière annonce d'une vraie enchère et laisse v6 continuer :
//! ce qui reste est un **vrai** préfixe. Il couvre ~2,4 des 4 cases ; les autres
//! couleurs n'ont jamais été annoncées par personne, et pour elles aucune enchère
//! réaliste n'existe **sur cette donne** — c'est la question posée, pas un défaut.
//!
//! ## Reprise
//!
//! Un run de plusieurs jours doit survivre à tout. `--out` est réécrit tous les
//! `--checkpoint` donnes avec le **préfixe dense** des donnes abouties, et un relancement
//! repart de là. Le fichier est minuscule (4 o/donne), donc réécrire est plus simple et
//! plus sûr qu'ajouter.
//!
//! ```bash
//! ./target/release/gen_score_layer --pool data/deals/base_5M.bin \
//!   --count 500000 --threads 96 --out data/deals/scores_isdd_v2.sc \
//!   --url "http://localhost:8003,http://localhost:8003,http://localhost:8003,http://192.168.1.23:8003"
//! ```

use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Instant;

use colver_core::agent::{AgentSpec, MatchContext, Player};
use colver_core::bid_net::BidNet;
use colver_core::bid_train_env::DealPool;
use colver_core::bidding::{self, BID_PASS};
use colver_core::solver;
use colver_core::state::{GameState, Phase};

#[path = "shared/auction.rs"]
mod auction;
use auction::{constructed_seat, dd_side, is_bid, is_raise, run_v6, speak_pos};

/// Répartition empirique de la valeur du contrat sur 43 031 enchères réelles
/// (`bench_taker_position`, 2026-08-04), capot écarté et le reste renormalisé.
///
/// On tire dedans plutôt que de dériver la valeur de la force de main : mesuré, la
/// relation est plate (112-124 pour tous les scores `evaluate_for_trump` de 1 à 31),
/// parce que dans une enchère contestée à 81 % c'est la pression de l'enchère qui
/// décide. Une échelle main → palier serait une règle inventée présentée comme mesurée.
const VALUE_CDF: [(f64, u8); 9] = [
    (0.0223, 8), (0.1109, 9), (0.2415, 10), (0.4701, 11), (0.6977, 12),
    (0.8933, 13), (0.9740, 14), (0.9952, 15), (1.0000, 16),
];

fn value_for(key: u64) -> u8 {
    let x = (key.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11) as f64 / (1u64 << 53) as f64;
    VALUE_CDF.iter().find(|&&(c, _)| x < c).map(|&(_, v)| v).unwrap_or(16)
}

/// L'enchère « fer » : passes jusqu'au siège qui tient la couleur, une annonce, trois
/// passes. Hors distribution et assumée comme telle — c'est le seul recours pour une
/// couleur que personne n'a annoncée.
fn built_auction(hands: &[u32; 4], dealer: u8, trump: u8, ns_pts: u8, key: u64) -> Vec<u8> {
    let seat = constructed_seat(hands, dealer, trump, dd_side(ns_pts));
    let mut a: Vec<u8> = vec![BID_PASS; speak_pos(seat, dealer)];
    a.push(bidding::encode_bid(value_for(key), trump));
    a.extend_from_slice(&[BID_PASS; 3]);
    a
}

/// Joue la donne sous une enchère **imposée** et rend les points cartes N-S.
///
/// L'enchère n'est pas demandée aux joueurs, elle leur est **montrée** : chaque action
/// passe par `observe` et `ctx.track` comme si elle avait été choisie. C'est ce qui fait
/// que playgen voie le bon préfixe — il tokenise les jetons du contexte.
fn label(
    hands: [u32; 4],
    dealer: u8,
    auction: &[u8],
    players: &mut [Box<dyn Player>; 4],
    ctx: &mut MatchContext,
) -> Result<Option<u8>, colver_core::agent::AgentError> {
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
        return Ok(None);
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
    Ok(Some(state.points[0]))
}

/// Construit les 4 préfixes d'une donne, un par atout, du plus fidèle au moins.
///
/// Rend aussi le rang de chacun (0 = or, 1 = argent/bronze, 3 = fer) pour la
/// journalisation : c'est la seule trace de la qualité du préfixe qui a produit chaque
/// étiquette, et elle n'est pas dans `COLVSC01`.
fn prefixes(
    hands: &[u32; 4],
    dealer: u8,
    dd: &[u8; 4],
    key: u64,
    net: &mut BidNet,
    obs: &mut Vec<f32>,
) -> [(Vec<u8>, u8); 4] {
    let mut out: [Option<(Vec<u8>, u8)>; 4] = [None, None, None, None];

    // Or, puis épluchages successifs : chacun tombe sur l'atout qu'il tombe.
    //
    // Le rang d'un préfixe épluché n'est **pas** son niveau dans la chaîne, c'est la
    // nature de l'annonce qu'on a retirée pour l'obtenir. Retirer une relance laisse son
    // auteur visible (argent) ; retirer son ouverture le rend muet (bronze), et c'est le
    // mensonge le plus cher qu'on puisse mettre dans un préfixe.
    let mut actions = run_v6(hands, dealer, u64::MAX, &[], net, obs);
    let mut rank = 0u8; // le premier est l'enchère libre : or
    for _ in 0..4u8 {
        let mut g = GameState::new(dealer, *hands);
        for &a in &actions {
            if g.phase != Phase::Bidding {
                break;
            }
            g.step(a);
        }
        if g.phase != Phase::Playing {
            break; // l'enchère s'est vidée
        }
        let t = g.contract.trump as usize;
        if out[t].is_none() {
            out[t] = Some((actions.clone(), rank));
        }
        let Some(bi) = actions.iter().rposition(|&a| is_bid(a)) else { break };
        rank = if is_raise(&actions, bi) { 1 } else { 2 };
        let mut prefix: Vec<u8> = actions[..bi].to_vec();
        prefix.push(BID_PASS);
        actions = run_v6(hands, dealer, u64::MAX, &prefix, net, obs);
    }

    // Fer pour ce qui reste.
    std::array::from_fn(|t| match out[t].take() {
        Some(v) => v,
        None => (built_auction(hands, dealer, t as u8, dd[t], key ^ (t as u64 * 0x5DEE)), 3),
    })
}

struct Args {
    pool: String,
    bot: String,
    bid_model: String,
    offset: usize,
    count: usize,
    threads: usize,
    checkpoint: usize,
    out: String,
    url: Option<String>,
    dets: Option<u32>,
}

/// Écrit le rang de préfixe de chaque case, un octet par case, 4 par donne.
///
/// **Ce fichier rend le choix réversible, et c'est sa seule raison d'être.** La mesure B
/// a montré que le préfixe déplace l'étiquette de **+4,36 points pour le preneur** (or
/// contre fer) et **+2,73** (épluchage contre fer) — monotone, z > 4. Une couche bâtie
/// sur la hiérarchie porte donc un écart *entre ses propres cases* : celle que v6 a
/// annoncée est systématiquement mieux étiquetée que les autres, ce qui incline vers la
/// politique qu'on cherche justement à dépasser.
///
/// On ne tranche pas ça en pleine nuit sur un run de plusieurs jours. On **enregistre**
/// de quoi le trancher après : avec le rang de chaque case, la correction se calcule,
/// s'annule ou se mesure. Sans lui, la couche est un mélange irrécupérable.
fn save_ranks(path: &str, ranks: &[[u8; 4]]) -> std::io::Result<()> {
    use std::io::Write;
    let mut f = std::io::BufWriter::new(std::fs::File::create(path)?);
    f.write_all(b"COLVRK01")?;
    f.write_all(&(ranks.len() as u32).to_le_bytes())?;
    for r in ranks {
        f.write_all(r)?;
    }
    f.flush()
}

fn parse() -> Args {
    let mut a = Args {
        pool: "data/deals/base_5M.bin".into(),
        bot: "arena/bots/gen_isdd_cardpts.toml".into(),
        bid_model: "models/bid_v6_isdd_resume/bid_nn_final.bin".into(),
        offset: 0,
        count: 500_000,
        threads: 96,
        checkpoint: 500,
        out: "data/deals/scores_isdd_v2.sc".into(),
        url: None,
        dets: None,
    };
    let v: Vec<String> = std::env::args().collect();
    let mut i = 1;
    let nx = |i: usize| v.get(i + 1).cloned().unwrap_or_default();
    while i < v.len() {
        match v[i].as_str() {
            "--pool" => { a.pool = nx(i); i += 2 }
            "--bot" => { a.bot = nx(i); i += 2 }
            "--bid-model" => { a.bid_model = nx(i); i += 2 }
            "--offset" => { a.offset = nx(i).parse().unwrap(); i += 2 }
            "--count" => { a.count = nx(i).parse().unwrap(); i += 2 }
            "--threads" => { a.threads = nx(i).parse().unwrap(); i += 2 }
            "--checkpoint" => { a.checkpoint = nx(i).parse().unwrap(); i += 2 }
            "--out" => { a.out = nx(i); i += 2 }
            "--url" => { a.url = Some(nx(i)); i += 2 }
            "--dets" => { a.dets = Some(nx(i).parse().unwrap()); i += 2 }
            "--help" | "-h" => {
                eprintln!("gen_score_layer : couche COLVSC01 en mondes playgen");
                eprintln!("  --pool/--offset/--count   donnes source");
                eprintln!("  --out <path>              couche (réécrite à chaque checkpoint)");
                eprintln!("  --threads N               96 mesuré sans erreur ; 256 noie le sidecar");
                eprintln!("  --checkpoint N            donnes entre deux écritures");
                eprintln!("  --url <u[,u..]>           sidecar(s), répéter une URL la pondère");
                std::process::exit(0)
            }
            o => { eprintln!("argument inconnu : {o}"); std::process::exit(1) }
        }
    }
    a
}

fn main() {
    let args = parse();
    let mut spec = AgentSpec::from_toml_file(&args.bot).unwrap_or_else(|e| {
        eprintln!("bot {} : {e}", args.bot);
        std::process::exit(1)
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

    eprintln!("chargement de {} …", args.pool);
    let pool = DealPool::load(&args.pool).unwrap_or_else(|e| {
        eprintln!("pool : {e}");
        std::process::exit(1)
    });
    let count = args.count.min(pool.len().saturating_sub(args.offset));

    // Reprise : le préfixe déjà écrit est relu et sauté.
    let mut scores: Vec<[u8; 4]> = vec![[0; 4]; count];
    let mut resume = 0usize;
    if let Ok(data) = std::fs::read(&args.out) {
        // COLVSC01 : magic[8] + name_len u16 + name + count u32 + offset u32 + data.
        // L'en-tête est de **taille variable** — le nom de la couche est dedans — donc
        // le lire à un décalage fixe donne des chiffres plausibles et faux.
        if data.len() >= 10 && &data[..8] == b"COLVSC01" {
            let nl = u16::from_le_bytes(data[8..10].try_into().unwrap()) as usize;
            let hdr = 10 + nl + 8;
            if data.len() >= hdr {
                let n = u32::from_le_bytes(data[10 + nl..14 + nl].try_into().unwrap()) as usize;
                let off = u32::from_le_bytes(data[14 + nl..18 + nl].try_into().unwrap()) as usize;
                if off != args.offset {
                    eprintln!("❌ {} porte offset {off}, on demande {} — refus d'écraser",
                              args.out, args.offset);
                    std::process::exit(1);
                }
                let n = n.min(count).min((data.len() - hdr) / 4);
                for k in 0..n {
                    scores[k].copy_from_slice(&data[hdr + 4 * k..hdr + 4 * k + 4]);
                }
                resume = n;
                eprintln!("reprise : {n} donnes déjà étiquetées dans {}", args.out);
            }
        }
    }

    eprintln!(
        "gen_score_layer : donnes {}..{} ({} à faire), {} threads, {} mondes/décision",
        args.offset + resume, args.offset + count, count - resume,
        args.threads, spec.play.determinizations
    );

    // Rangs de préfixe, parallèles aux scores. Repartis à zéro sur une reprise : les
    // rangs des donnes déjà faites sont dans le fichier précédent, et les réinventer
    // serait pire que de les laisser vides.
    let mut ranks_v: Vec<[u8; 4]> = vec![[9; 4]; count];
    if resume > 0 {
        if let Ok(d) = std::fs::read(format!("{}.ranks", args.out)) {
            if d.len() >= 12 && &d[..8] == b"COLVRK01" {
                let n = (u32::from_le_bytes(d[8..12].try_into().unwrap()) as usize)
                    .min(resume)
                    .min((d.len() - 12) / 4);
                for k in 0..n {
                    ranks_v[k].copy_from_slice(&d[12 + 4 * k..16 + 4 * k]);
                }
                eprintln!("reprise : {n} lignes de rangs relues");
            }
        }
    }
    let ranks_v = Arc::new(Mutex::new(ranks_v));
    let scores = Arc::new(Mutex::new(scores));
    // `done[k]` : la donne `offset+k` est étiquetée. Sert à trouver le préfixe dense.
    let done = Arc::new(Mutex::new(vec![false; count]));
    for k in 0..resume {
        done.lock().unwrap()[k] = true;
    }
    let next = Arc::new(AtomicUsize::new(resume));
    let n_done = Arc::new(AtomicUsize::new(resume));
    let errors = Arc::new(AtomicUsize::new(0));
    let ranks = Arc::new([AtomicUsize::new(0), AtomicUsize::new(0),
                          AtomicUsize::new(0), AtomicUsize::new(0)]);
    let stop = Arc::new(AtomicBool::new(false));
    let start = Instant::now();
    let pool = Arc::new(pool);

    // Écrivain : réécrit le préfixe dense à intervalle régulier. Un run de plusieurs
    // jours doit survivre à une coupure, et 4 o/donne rend la réécriture triviale.
    {
        let (scores, done, n_done, stop, ranks) =
            (scores.clone(), done.clone(), n_done.clone(), stop.clone(), ranks.clone());
        let ranks_v = ranks_v.clone();
        let (out, offset, checkpoint) = (args.out.clone(), args.offset, args.checkpoint);
        std::thread::spawn(move || {
            let mut last = 0usize;
            loop {
                std::thread::sleep(std::time::Duration::from_secs(20));
                let d = n_done.load(Ordering::Relaxed);
                let ending = stop.load(Ordering::Relaxed);
                if d < last + checkpoint && !ending {
                    continue;
                }
                last = d;
                let dense = {
                    let dn = done.lock().unwrap();
                    dn.iter().position(|&x| !x).unwrap_or(dn.len())
                };
                if dense == 0 {
                    continue;
                }
                let sc = scores.lock().unwrap()[..dense].to_vec();
                let rk = ranks_v.lock().unwrap()[..dense].to_vec();
                if let Err(e) = save_ranks(&format!("{out}.ranks"), &rk) {
                    eprintln!("  ⚠ écriture des rangs : {e}");
                }
                match DealPool::save_scores("isdd_v2", offset, &sc, &out) {
                    Ok(()) => {
                        let el = start.elapsed().as_secs_f64();
                        let r: Vec<String> = (0..4)
                            .map(|i| ranks[i].load(Ordering::Relaxed).to_string())
                            .collect();
                        eprintln!(
                            "  ✔ {dense} donnes écrites ({d} abouties) | {:.2} donnes/s | \
                             {:.1} h écoulées | préfixes or/arg/bro/fer {}",
                            d as f64 / el, el / 3600.0, r.join("/")
                        );
                    }
                    Err(e) => eprintln!("  ⚠ écriture {out} : {e}"),
                }
                if ending {
                    break;
                }
            }
        });
    }

    let mut handles = Vec::with_capacity(args.threads);
    for tid in 0..args.threads {
        let (scores, done, next, n_done, errors, ranks, stop) =
            (scores.clone(), done.clone(), next.clone(), n_done.clone(),
             errors.clone(), ranks.clone(), stop.clone());
        let (pool, spec, bid_model) = (pool.clone(), spec.clone(), args.bid_model.clone());
        let ranks_v = ranks_v.clone();
        let offset = args.offset;
        handles.push(std::thread::spawn(move || {
            let mut players: [Box<dyn Player>; 4] = match (0..4)
                .map(|s| spec.build(s)).collect::<Result<Vec<_>, _>>()
            {
                Ok(v) => v.try_into().map_err(|_| ()).expect("4 sièges"),
                Err(e) => { eprintln!("thread {tid} : {e}"); stop.store(true, Ordering::Relaxed); return }
            };
            let mut net = match BidNet::load(&bid_model) {
                Ok(n) => n,
                Err(e) => { eprintln!("thread {tid} : {e}"); stop.store(true, Ordering::Relaxed); return }
            };
            let mut obs: Vec<f32> = Vec::new();
            let mut ctx = MatchContext::new(0);
            let mut tt = solver::new_tt_buffer();

            loop {
                if stop.load(Ordering::Relaxed) {
                    break;
                }
                let k = next.fetch_add(1, Ordering::Relaxed);
                if k >= scores.lock().unwrap().len() {
                    break;
                }
                let d = pool.get(offset + k);
                let (hands, dealer) = (d.hands, d.dealer);

                // Les `dd_pts` du pool sont **antérieures au retrait de `quick_tricks`**
                // (2026-07-23) : on les recalcule. ~96 ms contre ~1,5 s d'étiquetage,
                // soit 6 % — le prix de ne pas bâtir sur une valeur périmée.
                let dd: [u8; 4] = std::array::from_fn(|t| {
                    solver::solve_for_trump_reuse_tt(hands, dealer, t as u8, &mut tt)[0]
                });

                let key = (offset + k) as u64;
                let pfx = prefixes(&hands, dealer, &dd, key, &mut net, &mut obs);

                let mut row = [0u8; 4];
                let mut rrow = [0u8; 4];
                let mut ok = true;
                for (t, (auc, rank)) in pfx.iter().enumerate() {
                    match label(hands, dealer, auc, &mut players, &mut ctx) {
                        Ok(Some(v)) => {
                            row[t] = v;
                            rrow[t] = *rank;
                            ranks[*rank as usize].fetch_add(1, Ordering::Relaxed);
                        }
                        _ => { ok = false; break }
                    }
                }
                if !ok {
                    let e = errors.fetch_add(1, Ordering::Relaxed) + 1;
                    if e <= 5 || e % 500 == 0 {
                        eprintln!("thread {tid} donne {} abandonnée ({e}e)", offset + k);
                    }
                    // Pas de jeton rendu : la donne est simplement laissée à 0 et le
                    // préfixe dense s'arrête avant elle. Une reprise la reprendra.
                    continue;
                }
                scores.lock().unwrap()[k] = row;
                ranks_v.lock().unwrap()[k] = rrow;
                done.lock().unwrap()[k] = true;
                n_done.fetch_add(1, Ordering::Relaxed);
            }
        }));
    }
    for h in handles {
        let _ = h.join();
    }
    stop.store(true, Ordering::Relaxed);
    std::thread::sleep(std::time::Duration::from_secs(22));

    let el = start.elapsed().as_secs_f64();
    let d = n_done.load(Ordering::Relaxed);
    eprintln!(
        "\nterminé : {d} donnes en {:.1} h ({:.2} donnes/s, {:.1} étiquetages/s), {} erreurs",
        el / 3600.0, d as f64 / el, 4.0 * d as f64 / el, errors.load(Ordering::Relaxed)
    );
}
