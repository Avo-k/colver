//! Combien de mondes le modèle hésite-t-il encore, pli par pli ?
//!
//! Perplexité teacher-forcing d'un modèle playgen sur un corpus **retenu**,
//! restreinte aux coups des sièges **cachés** — les seuls qui parlent du monde ;
//! les coups de l'observateur ne disent rien de ce qu'il ignore.
//!
//! Trois choses que la CE par pli de `train_playgen` ne donne pas :
//!
//! 1. **Un point d'entrée autonome** : n'importe quel checkpoint, n'importe quel
//!    corpus. La courbe existante ne vit qu'à l'intérieur de l'entraînement, sur
//!    une tranche découpée du corpus d'entraînement lui-même.
//! 2. **La vue cumulative** : par la règle de chaîne, log p(reste de la donne)
//!    est la **somme** des log p par coup. À l'entame du pli t, `exp(Σ nll
//!    restantes)` est donc le nombre effectif de **continuations** encore
//!    indistinguables. Ce n'est pas une mesure de plus, c'est la même en unités
//!    interprétables — et elle doit tomber vers 1 au pli 8.
//!
//!    **Ce ne sont pas des mondes, et l'écart n'est pas petit.** Une même
//!    distribution des mains se réalise par de nombreux ordres de jeu, donc ce
//!    compte majore le nombre de mondes — au pli 1 il n'existe que
//!    `24!/(8!)³ = 9,47e9` distributions, et la colonne en affiche cent fois
//!    plus. La table imprime donc ce plafond combinatoire à côté, pour que le
//!    sur-comptage se voie. **C'est le rapport modèle/uniforme qui est lisible**,
//!    pas la valeur absolue : le facteur d'ordres se simplifie entre les deux.
//!    Le compte de mondes proprement dit demande de l'échantillonnage (taux de
//!    monde exact), c'est-à-dire une autre mesure.
//! 3. **Le plancher de contrainte** : la même quantité sous une loi uniforme sur
//!    le masque. `bench_world_cred` a montré qu'un tirage uniforme contraint
//!    atteint déjà 70 % d'argmax en jeu, donc l'arithmétique d'ensembles porte
//!    l'essentiel du signal. Sans ce plancher, on lit comme un mérite du modèle
//!    ce qui n'est qu'une déduction de règles.
//!
//! **Ce que la mesure n'est pas** : un chiffre absolu. Le corpus est joué par
//! bid v6 + DouDou50, donc elle répond à « sait-il prédire *ces bots-là* ».
//! Comparer deux checkpoints sur le même corpus est légitime ; comparer deux
//! corpus ne l'est pas.
//!
//! Aucun échantillonnage : le teacher forcing est déterministe, donc ce bench
//! est immunisé par construction contre le piège maison (« ne jamais tirer les
//! questions d'un flux que la chose mesurée consomme aussi »).
//!
//! Usage :
//!   cargo run -p colver-core --bin bench_playgen_ppl --release --features parallel -- \
//!     --model models/playgen/playgen_v2_final.bin \
//!     --games data/training/heldout_20k_s90210.bin --n 2000

use colver_core::game_replay::GameReplay;
use colver_core::play::belote_facts;
use colver_core::playgen::infer::{KvCache, PlaygenModel, Tok};
use colver_core::playgen::tokens::{identity_perm, tokenize_replay_v2};
use colver_core::state::{GameState, Phase};

const TRICKS: usize = 8;
/// Tours d'enchère suivis. Un tour = les quatre sièges parlent une fois
/// (`MAX_BID_ENTRIES_V2` = 24 places, soit 6 tours), mais la longueur réelle est
/// **variable** : la plupart des enchères meurent au 1er ou 2e tour, donc les
/// derniers tours reposent sur peu d'échantillons. La table imprime `n` pour que
/// ça se voie, et saute les tours vides.
const BID_ROUNDS: usize = 6;

/// Accumulateur d'un lot de donnes.
#[derive(Clone, Default)]
struct Acc {
    /// Somme des nll modèle par pli (acteurs cachés).
    nll: [f64; TRICKS],
    /// Idem sous la loi uniforme sur le masque — le plancher de contrainte.
    unif: [f64; TRICKS],
    /// Nombre de prédictions par pli.
    n: [u64; TRICKS],
    /// Somme, par pli d'entrée t, de `Σ_{u ≥ t} nll` d'un échantillon — la
    /// moyenne *des sommes*, donc `exp` de sa moyenne = moyenne géométrique du
    /// nombre de mondes. Moyenner les perplexités serait faux.
    cum: [f64; TRICKS],
    cum_unif: [f64; TRICKS],
    /// Échantillons ayant au moins une prédiction cachée au pli t ou après.
    cum_n: [u64; TRICKS],
    /// Idem pour la tête d'enchère, par tour, sièges cachés seulement.
    bnll: [f64; BID_ROUNDS],
    bunif: [f64; BID_ROUNDS],
    bn: [u64; BID_ROUNDS],
    /// Prédictions de jeu séparées selon qu'une déduction belote est disponible
    /// à la position. Sert à démêler la capacité de la belote : v2 a été
    /// entraîné avec un masque qui l'ignorait, donc toute comparaison globale
    /// v3-vs-v2 confond les deux. Ici la contrainte est dans le masque des deux
    /// côtés — ce qui diffère, c'est d'avoir appris ou non ses conséquences.
    bel_nll: f64,
    bel_unif: f64,
    bel_n: u64,
    nobel_nll: f64,
    nobel_unif: f64,
    nobel_n: u64,
    samples: u64,
    skipped: u64,
}

impl Acc {
    fn merge(&mut self, o: &Acc) {
        for t in 0..TRICKS {
            self.nll[t] += o.nll[t];
            self.unif[t] += o.unif[t];
            self.n[t] += o.n[t];
            self.cum[t] += o.cum[t];
            self.cum_unif[t] += o.cum_unif[t];
            self.cum_n[t] += o.cum_n[t];
        }
        for r in 0..BID_ROUNDS {
            self.bnll[r] += o.bnll[r];
            self.bunif[r] += o.bunif[r];
            self.bn[r] += o.bn[r];
        }
        self.bel_nll += o.bel_nll;
        self.bel_unif += o.bel_unif;
        self.bel_n += o.bel_n;
        self.nobel_nll += o.nobel_nll;
        self.nobel_unif += o.nobel_unif;
        self.nobel_n += o.nobel_n;
        self.samples += o.samples;
        self.skipped += o.skipped;
    }
}

/// Une donne vue par un observateur : nll par pli, acteurs cachés seulement.
fn score_sample(model: &PlaygenModel, replay: &GameReplay, observer: u8, acc: &mut Acc) {
    let Some(s) = tokenize_replay_v2(replay, observer, &identity_perm()) else {
        acc.skipped += 1;
        return;
    };
    // Une déduction belote est-elle disponible à chaque coup joué ? Re-parcours
    // de la donne plutôt qu'un champ de plus dans `PlaygenSampleV2` : le
    // tokenizer est partagé avec l'entraînement, et le coût d'un replay est
    // négligeable devant un forward de transformeur.
    let bel: Vec<bool> = {
        let mut st = GameState::new(replay.dealer, replay.hands);
        let mut v = Vec::with_capacity(32);
        for &a in &replay.actions {
            if st.phase == Phase::Playing {
                v.push(!belote_facts(&st).is_empty());
            }
            st.step(a);
        }
        v
    };

    let mut nll = [0.0f64; TRICKS];
    let mut unif = [0.0f64; TRICKS];
    let mut n = [0u64; TRICKS];

    let mut cache = KvCache::new(model);
    let mut pred_i = 0usize;
    let mut bid_i = 0usize;
    for j in 0..s.primary.len() {
        let tok = Tok {
            primary: s.primary[j],
            suit: s.suit[j],
            actor: s.actor[j],
            segment: s.segment[j],
        };
        let hidden = model.forward_token(&mut cache, tok, j);
        if bid_i < s.bid_pred_pos.len() && s.bid_pred_pos[bid_i] as usize == j {
            // `actor` est relatif à l'observateur : 0 = lui-même. Le masque
            // d'enchère est **public** (la légalité ne lit aucune main), donc
            // ce qui distingue un siège caché ici n'est pas le masque mais la
            // main que le modèle doit deviner derrière l'annonce.
            if s.actor[j] != 0 {
                let logits = model.bid_logits(&hidden);
                let mask = s.bid_masks[bid_i];
                let target = s.bid_targets[bid_i];
                let mut max_l = f32::NEG_INFINITY;
                for a in 0..logits.len() {
                    if mask & (1u64 << a) != 0 && logits[a] > max_l {
                        max_l = logits[a];
                    }
                }
                let mut denom = 0.0f64;
                for a in 0..logits.len() {
                    if mask & (1u64 << a) != 0 {
                        denom += ((logits[a] - max_l) as f64).exp();
                    }
                }
                let logp = (logits[target as usize] - max_l) as f64 - denom.ln();
                let r = (bid_i / 4).min(BID_ROUNDS - 1);
                acc.bnll[r] += -logp;
                acc.bunif[r] += (mask.count_ones() as f64).ln();
                acc.bn[r] += 1;
            }
            bid_i += 1;
        }
        if pred_i < s.pred_pos.len() && s.pred_pos[pred_i] as usize == j {
            // Seuls les sièges cachés portent de l'information sur le monde.
            if s.hidden_actor[pred_i] {
                let logits = model.logits(&hidden);
                let mask = s.masks[pred_i];
                let target = s.targets[pred_i];
                let mut max_l = f32::NEG_INFINITY;
                for c in 0..32u8 {
                    if mask & (1 << c) != 0 && logits[c as usize] > max_l {
                        max_l = logits[c as usize];
                    }
                }
                let mut denom = 0.0f64;
                for c in 0..32u8 {
                    if mask & (1 << c) != 0 {
                        denom += ((logits[c as usize] - max_l) as f64).exp();
                    }
                }
                let logp = (logits[target as usize] - max_l) as f64 - denom.ln();
                let t = (s.trick_idx[pred_i] as usize).min(TRICKS - 1);
                nll[t] += -logp;
                unif[t] += (mask.count_ones() as f64).ln();
                n[t] += 1;
                if bel.get(pred_i).copied().unwrap_or(false) {
                    acc.bel_nll += -logp;
                    acc.bel_unif += (mask.count_ones() as f64).ln();
                    acc.bel_n += 1;
                } else {
                    acc.nobel_nll += -logp;
                    acc.nobel_unif += (mask.count_ones() as f64).ln();
                    acc.nobel_n += 1;
                }
            }
            pred_i += 1;
        }
    }

    // Cumul depuis le pli t jusqu'à la fin, par échantillon.
    let (mut tail, mut tail_u, mut tail_n) = (0.0f64, 0.0f64, 0u64);
    for t in (0..TRICKS).rev() {
        tail += nll[t];
        tail_u += unif[t];
        tail_n += n[t];
        if tail_n > 0 {
            acc.cum[t] += tail;
            acc.cum_unif[t] += tail_u;
            acc.cum_n[t] += 1;
        }
    }
    for t in 0..TRICKS {
        acc.nll[t] += nll[t];
        acc.unif[t] += unif[t];
        acc.n[t] += n[t];
    }
    acc.samples += 1;
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let mut model_path = String::from("models/playgen/playgen_v2_final.bin");
    let mut games_path = String::from("data/training/heldout_20k_s90210.bin");
    let mut n_games: usize = 2000;
    let mut json = false;

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => { model_path = args[i + 1].clone(); i += 2; }
            "--games" => { games_path = args[i + 1].clone(); i += 2; }
            "--n" => { n_games = args[i + 1].parse().unwrap(); i += 2; }
            "--json" => { json = true; i += 1; }
            other => { eprintln!("Unknown argument: {}", other); std::process::exit(1); }
        }
    }

    let model = PlaygenModel::load(&model_path).unwrap_or_else(|e| {
        eprintln!("load model {}: {}", model_path, e);
        std::process::exit(1);
    });
    if !model.v2 {
        eprintln!("bench_playgen_ppl attend un modèle COLVPG02 (tokenisation v2)");
        std::process::exit(1);
    }
    let replays = GameReplay::load_all(&games_path).unwrap_or_else(|e| {
        eprintln!("load games {}: {}", games_path, e);
        std::process::exit(1);
    });
    let games: Vec<&GameReplay> = replays.iter().take(n_games).collect();

    println!("=== Perplexité playgen (acteurs cachés) ===");
    println!("Modèle : {}", model_path);
    println!("Corpus : {} ({} donnes × 4 observateurs)", games_path, games.len());

    let t0 = std::time::Instant::now();

    #[cfg(feature = "parallel")]
    let acc = {
        use rayon::prelude::*;
        games
            .par_iter()
            .fold(Acc::default, |mut a, replay| {
                for observer in 0..4u8 {
                    score_sample(&model, replay, observer, &mut a);
                }
                a
            })
            .reduce(Acc::default, |mut a, b| {
                a.merge(&b);
                a
            })
    };
    #[cfg(not(feature = "parallel"))]
    let acc = {
        let mut a = Acc::default();
        for replay in &games {
            for observer in 0..4u8 {
                score_sample(&model, replay, observer, &mut a);
            }
        }
        a
    };

    let elapsed = t0.elapsed().as_secs_f64();
    let total_preds: u64 = acc.n.iter().sum();
    println!(
        "{} échantillons ({} ignorés), {} prédictions cachées, {:.1}s\n",
        acc.samples, acc.skipped, total_preds, elapsed
    );

    let bid_preds: u64 = acc.bn.iter().sum();
    if bid_preds > 0 {
        println!("Par tour d'enchère — sièges cachés seulement, masque légal public :");
        println!("  tour |    n   | nll modèle | branch. | nll uniforme | branch. | gain");
        for r in 0..BID_ROUNDS {
            if acc.bn[r] == 0 {
                continue;
            }
            let m = acc.bnll[r] / acc.bn[r] as f64;
            let u = acc.bunif[r] / acc.bn[r] as f64;
            println!(
                "  {:>4} | {:>6} |     {:.4} | {:>7.2} |       {:.4} | {:>7.2} | {:.2}×",
                r + 1,
                acc.bn[r],
                m,
                m.exp(),
                u,
                u.exp(),
                (u - m).exp(),
            );
        }
        println!();
    }

    if acc.bel_n > 0 {
        let bm = acc.bel_nll / acc.bel_n as f64;
        let bu = acc.bel_unif / acc.bel_n as f64;
        let nm = acc.nobel_nll / acc.nobel_n as f64;
        let nu = acc.nobel_unif / acc.nobel_n as f64;
        println!("Selon qu'une déduction belote est disponible à la position :");
        println!("  position          |     n  | nll modèle | branch. | uniforme | gain");
        println!(
            "  belote disponible | {:>6} |     {:.4} | {:>7.2} | {:>8.2} | {:.2}×",
            acc.bel_n, bm, bm.exp(), bu.exp(), (bu - bm).exp()
        );
        println!(
            "  aucune            | {:>6} |     {:.4} | {:>7.2} | {:>8.2} | {:.2}×",
            acc.nobel_n, nm, nm.exp(), nu.exp(), (nu - nm).exp()
        );
        println!(
            "  ({:.1} % des prédictions cachées portent une déduction)\n",
            acc.bel_n as f64 / (acc.bel_n + acc.nobel_n) as f64 * 100.0
        );
    }

    println!("Par pli — nll moyenne et facteur de branchement effectif exp(nll) :");
    println!("  pli |    n  | nll modèle | branch. | nll uniforme | branch. | gain");
    for t in 0..TRICKS {
        if acc.n[t] == 0 {
            continue;
        }
        let m = acc.nll[t] / acc.n[t] as f64;
        let u = acc.unif[t] / acc.n[t] as f64;
        println!(
            "  {:>3} | {:>6} |     {:.4} | {:>7.2} |       {:.4} | {:>7.2} | {:.2}×",
            t + 1,
            acc.n[t],
            m,
            m.exp(),
            u,
            u.exp(),
            (u - m).exp(),
        );
    }

    println!("\nCumul depuis le pli t — continuations encore indistinguables.");
    println!("  ⚠ pas des mondes : plusieurs ordres de jeu réalisent la même distribution,");
    println!("    donc ces comptes majorent le nombre de mondes (colonne « distributions »).");
    println!("    Seul le rapport modèle/uniforme est lisible — le facteur d'ordres s'y simplifie.");
    println!("  depuis pli | continuations (modèle) | (uniforme) |     gain | distributions possibles");
    for t in 0..TRICKS {
        if acc.cum_n[t] == 0 {
            continue;
        }
        let m = acc.cum[t] / acc.cum_n[t] as f64;
        let u = acc.cum_unif[t] / acc.cum_n[t] as f64;
        // Plafond combinatoire : (3k)!/(k!)^3 mains cachées de k cartes chacune.
        let k = 8 - t;
        let ln_worlds: f64 = (1..=3 * k).map(|x| (x as f64).ln()).sum::<f64>()
            - 3.0 * (1..=k).map(|x| (x as f64).ln()).sum::<f64>();
        println!(
            "  {:>10} | {:>22.3e} | {:>10.3e} | {:>8.3e} | {:>10.3e}",
            t + 1,
            m.exp(),
            u.exp(),
            (u - m).exp(),
            ln_worlds.exp(),
        );
    }

    if json {
        let f = |v: &[f64; TRICKS], d: &[u64; TRICKS]| -> Vec<f64> {
            (0..TRICKS)
                .map(|t| if d[t] > 0 { v[t] / d[t] as f64 } else { f64::NAN })
                .collect()
        };
        println!(
            "\nJSON {}",
            format!(
                "{{\"model\":\"{}\",\"games\":\"{}\",\"samples\":{},\"preds\":{},\
                 \"nll_by_trick\":{:?},\"unif_by_trick\":{:?},\
                 \"cum_nll_from_trick\":{:?},\"cum_unif_from_trick\":{:?}}}",
                model_path,
                games_path,
                acc.samples,
                total_preds,
                f(&acc.nll, &acc.n),
                f(&acc.unif, &acc.n),
                f(&acc.cum, &acc.cum_n),
                f(&acc.cum_unif, &acc.cum_n),
            )
        );
    }
}
