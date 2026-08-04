//! Ce que coûte l'aveuglement d'IS-DD au **score de partie**, en probabilité de
//! gagner le match.
//!
//! IS-DD maximise `E[écart de score de donne]` (`PlayObjective::DealScore`).
//! C'est une fonction **linéaire** de l'écart, donc neutre au risque. L'objectif
//! vrai est `E[P(gagner la partie)]`, qui sature près de 2000 : loin devant on
//! doit être averse au risque, loin derrière preneur de risque. La question
//! chiffrée ici : à quelle fréquence les deux objectifs désignent une carte
//! différente, et combien vaut l'écart.
//!
//! **La mesure est appariée** : les deux objectifs lisent les *mêmes* mondes
//! résolus une seule fois. Une divergence est donc entièrement imputable à
//! l'agrégation, sans plancher de bruit — même principe que `bench_belote_ab`,
//! qu'un h2h d'arène ne peut pas remplacer.
//!
//! ```bash
//! cargo build --release --features parallel --bin bench_match_equity
//! ./target/release/bench_match_equity --corpus data/training/isdd_games_v1.bin \
//!     --positions 400 --worlds 60
//! ```
//!
//! ⚠️ Trois approximations, toutes dans le même sens (elles **surestiment**) :
//! (1) les mondes sont tirés **uniformément sous contraintes**, pas par playgen —
//! une postérieure uniforme est plus large, donc plus de masse près du seuil ;
//! (2) la table d'équité est bâtie sur la distribution de scores de donne d'un
//! corpus joué **à 0-0**, alors qu'une vraie fin de partie se joue autrement ;
//! (3) la grille d'équité est à 10 points (la FFB arrondit là, le moteur non).

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

#[cfg(feature = "parallel")]
use rayon::prelude::*;

use colver_core::card::CardSet;
use colver_core::determinize::determinize;
use colver_core::game_replay::GameReplay;
use colver_core::scoring::{deal_score_from_card_points, CAPOT_PTS, TOTAL_PTS};
use colver_core::solver::{new_tt_buffer, solve_with_scores};
use colver_core::state::{GameState, Phase};

const TARGET: i32 = 2000;
/// Pas de la grille d'équité, en points de partie.
const G: i32 = 10;
/// Nombre de cases par axe : 0, 10, … 1990. Au-delà la partie est finie.
const N: usize = (TARGET / G) as usize;

// ---------------------------------------------------------------------------
// Barème d'un monde résolu
// ---------------------------------------------------------------------------

/// Total de points cartes de la donne, déduit du total N-S. Copie locale du
/// privé d'`is_dd.rs` — même règle, même angle mort (`ns == 0` sans capot E-O).
#[inline]
fn total_card_points(state: &GameState, ns_card_pts: i16) -> i16 {
    if ns_card_pts == CAPOT_PTS {
        return CAPOT_PTS;
    }
    if ns_card_pts == 0 && state.tricks_won[0] == 0 {
        return CAPOT_PTS;
    }
    TOTAL_PTS
}

/// Belote finale d'un monde, par camp. Comme dans `is_dd.rs` : acquise dès qu'un
/// joueur détient Dame **et** Roi d'atout, cartes déjà jouées comprises.
#[inline]
fn world_belote(hands: &[CardSet; 4], played_by: &[CardSet; 4], trump: u8) -> [i16; 2] {
    let mask = (1u32 << (trump * 8 + 4)) | (1u32 << (trump * 8 + 5));
    let mut bonus = [0i16; 2];
    for p in 0..4usize {
        if (hands[p] | played_by[p]) & mask == mask {
            bonus[p % 2] = 20;
        }
    }
    bonus
}

// ---------------------------------------------------------------------------
// Table d'équité de match
// ---------------------------------------------------------------------------

/// `P(N-S gagne la partie)` pour tout score cumulé non terminal, sur la grille.
struct Equity {
    table: Vec<f32>,
}

impl Equity {
    /// Récurrence exacte sur la distribution empirique des scores de donne.
    ///
    /// Tout tirage fait strictement croître `a + b` (les donnes passées, qui ne
    /// marquent rien, sont exclues du tirage — elles sont simplement redonnées),
    /// donc l'espace d'états est un DAG et un seul balayage par `a + b`
    /// décroissant suffit : aucune itération de valeur.
    fn build(outcomes: &[(i32, i32)]) -> Self {
        let mut table = vec![0f32; N * N];
        let mut states: Vec<(usize, usize)> =
            (0..N).flat_map(|a| (0..N).map(move |b| (a, b))).collect();
        states.sort_by_key(|&(a, b)| std::cmp::Reverse(a + b));

        let inv = 1.0 / outcomes.len() as f64;
        for (a, b) in states {
            let (sa, sb) = (a as i32 * G, b as i32 * G);
            let mut acc = 0f64;
            for &(ds, de) in outcomes {
                let (na, nb) = (sa + ds, sb + de);
                acc += if na >= TARGET || nb >= TARGET {
                    winner_is_ns(na, nb) as u8 as f64
                } else {
                    table[(na / G) as usize * N + (nb / G) as usize] as f64
                };
            }
            table[a * N + b] = (acc * inv) as f32;
        }
        Equity { table }
    }

    /// Équité après une donne. Le score n'est arrondi qu'ici, au plus proche.
    #[inline]
    fn at(&self, ns: i32, ew: i32) -> f32 {
        if ns >= TARGET || ew >= TARGET {
            return winner_is_ns(ns, ew) as u8 as f32;
        }
        let a = (((ns.max(0) + G / 2) / G) as usize).min(N - 1);
        let b = (((ew.max(0) + G / 2) / G) as usize).min(N - 1);
        self.table[a * N + b]
    }
}

/// Règle de fin de `game_loop::play_match` : les deux camps peuvent franchir la
/// ligne sur la même donne, le plus haut total l'emporte.
#[inline]
fn winner_is_ns(ns: i32, ew: i32) -> bool {
    if ns >= TARGET && ew >= TARGET {
        ns >= ew
    } else {
        ns >= TARGET
    }
}

// ---------------------------------------------------------------------------
// Corpus → distribution de scores de donne, et positions de jeu
// ---------------------------------------------------------------------------

/// Rejoue une donne du corpus et rend son score, arrondi sur la grille.
/// `None` pour une donne passée : elle ne marque rien et se redonne.
fn deal_outcome(r: &GameReplay) -> Option<(i32, i32)> {
    let mut s = GameState::new(r.dealer, r.hands);
    for &a in &r.actions {
        if s.is_terminal() || s.legal_actions() & (1u64 << a) == 0 {
            return None;
        }
        s.step(a);
    }
    if !s.is_terminal() || s.contract.value == 0 {
        return None;
    }
    let sc = colver_core::scoring::compute_deal_score(&s);
    let round = |x: i16| ((x as i32 + G / 2) / G) * G;
    let (ns, ew) = (round(sc.scores[0]), round(sc.scores[1]));
    if ns == 0 && ew == 0 {
        return None;
    }
    Some((ns, ew))
}

/// Une décision de jeu réelle : la position, plus les cartes déjà posées par
/// chaque siège (le monde en a besoin pour la belote).
struct Position {
    state: GameState,
    played_by: [CardSet; 4],
}

/// Toutes les positions de jeu d'une donne où il y a **une décision** (au moins
/// deux coups légaux) et où il reste assez de cartes pour que le choix pèse.
fn positions_of(r: &GameReplay, min_cards: u32, max_cards: u32) -> Vec<Position> {
    let mut out = Vec::new();
    let mut s = GameState::new(r.dealer, r.hands);
    let mut played_by = [0u32; 4];
    for &a in &r.actions {
        if s.is_terminal() || s.legal_actions() & (1u64 << a) == 0 {
            return Vec::new();
        }
        if s.phase == Phase::Playing {
            let p = s.current_player();
            let left = s.hands[p as usize].count_ones();
            if s.legal_actions().count_ones() >= 2 && left >= min_cards && left <= max_cards {
                out.push(Position { state: s, played_by });
            }
            played_by[p as usize] |= 1u32 << a;
        }
        s.step(a);
    }
    out
}

// ---------------------------------------------------------------------------
// La mesure
// ---------------------------------------------------------------------------

#[derive(Default, Clone)]
struct Tally {
    positions: u32,
    diverged: u32,
    /// Équité perdue en choisissant la carte de `DealScore`, en points de %.
    loss_pp: Vec<f64>,
    /// Écart de score de donne abandonné par la carte de l'équité.
    margin_given_up: Vec<f64>,
}

impl Tally {
    fn merge(&mut self, o: Tally) {
        self.positions += o.positions;
        self.diverged += o.diverged;
        self.loss_pp.extend(o.loss_pp);
        self.margin_given_up.extend(o.margin_given_up);
    }
}

/// Une position, un score de partie : agrège les mêmes mondes deux fois.
fn measure(
    pos: &Position,
    eq: &Equity,
    cum: (i32, i32),
    worlds: u32,
    rng: &mut StdRng,
    tt: &mut colver_core::solver::TtBuf,
    out: &mut Tally,
) {
    let state = pos.state;
    let observer = state.current_player();
    let team = GameState::player_team(observer) as usize;

    let mut sum_margin = [0f64; 32];
    let mut sum_eq = [0f64; 32];
    let mut n = [0u32; 32];
    let mut solved = 0u32;

    for _ in 0..worlds {
        let Some(world) = determinize(&state, observer, rng) else { continue };
        let belote = world_belote(&world.hands, &pos.played_by, world.contract.trump);
        let taker = world.contract.team as usize;
        let sc = solve_with_scores(&world, Some(tt));
        for i in 0..sc.count {
            let (card, ns_pts) = sc.scores[i];
            let total = total_card_points(&world, ns_pts);
            let cp = [ns_pts, total - ns_pts];
            let s = deal_score_from_card_points(&world.contract, cp, belote, cp[taker] == CAPOT_PTS);
            let c = card as usize;
            sum_margin[c] += (s.scores[0] - s.scores[1]) as f64;
            sum_eq[c] += eq.at(cum.0 + s.scores[0] as i32, cum.1 + s.scores[1] as i32) as f64;
            n[c] += 1;
        }
        solved += 1;
    }
    if solved == 0 {
        return;
    }

    // Les deux objectifs, du point de vue du camp qui joue.
    let margin_of = |c: usize| {
        let v = sum_margin[c] / n[c] as f64;
        if team == 0 { v } else { -v }
    };
    let equity_of = |c: usize| {
        let v = sum_eq[c] / n[c] as f64;
        if team == 0 { v } else { 1.0 - v }
    };

    let cards: Vec<usize> = (0..32).filter(|&c| n[c] > 0).collect();
    if cards.len() < 2 {
        return;
    }
    let pick = |f: &dyn Fn(usize) -> f64| -> usize {
        *cards
            .iter()
            .max_by(|&&a, &&b| f(a).partial_cmp(&f(b)).unwrap())
            .unwrap()
    };
    let best_m = pick(&margin_of);
    let best_e = pick(&equity_of);

    out.positions += 1;
    if best_m != best_e {
        out.diverged += 1;
        out.loss_pp.push((equity_of(best_e) - equity_of(best_m)) * 100.0);
        out.margin_given_up.push(margin_of(best_m) - margin_of(best_e));
    }
}

// ---------------------------------------------------------------------------

fn mean(v: &[f64]) -> f64 {
    if v.is_empty() { 0.0 } else { v.iter().sum::<f64>() / v.len() as f64 }
}

fn quantile(v: &mut Vec<f64>, q: f64) -> f64 {
    if v.is_empty() {
        return 0.0;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[(((v.len() - 1) as f64) * q).round() as usize]
}

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str, def: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| def.to_string())
    };
    let corpus = get("--corpus", "data/training/isdd_games_v1.bin");
    let n_positions: usize = get("--positions", "400").parse().unwrap();
    let worlds: u32 = get("--worlds", "60").parse().unwrap();
    let seed: u64 = get("--seed", "42").parse().unwrap();
    let min_cards: u32 = get("--min-cards", "3").parse().unwrap();
    let max_cards: u32 = get("--max-cards", "8").parse().unwrap();
    let states_arg = get(
        "--states",
        "0:0,1000:1000,1500:750,1800:600,1900:200,1800:1750,600:1800,200:1900",
    );

    let probes: Vec<(i32, i32)> = states_arg
        .split(',')
        .map(|s| {
            let (a, b) = s.split_once(':').expect("état = ns:ew");
            (a.parse().unwrap(), b.parse().unwrap())
        })
        .collect();

    eprintln!("Lecture de {corpus} …");
    let replays = GameReplay::load_all(&corpus).expect("corpus illisible");
    eprintln!("  {} donnes", replays.len());

    // --- 1. Distribution empirique des scores de donne. ---
    let mut raw: Vec<(i32, i32)> = replays.iter().filter_map(deal_outcome).collect();
    eprintln!("  {} issues de donne", raw.len());
    // Un sous-échantillon suffit à la récurrence et la rend ~instantanée. Il est
    // tiré **avant** la symétrisation : tirer après casserait l'invariant
    // `E(a,b) + E(b,a) = 1`, et une table qui donne 0,477 à 0-0 est fausse d'une
    // façon qui se voit — c'est le contrôle qui a attrapé le défaut.
    let mut rng = StdRng::seed_from_u64(seed);
    if raw.len() > 2048 {
        for i in 0..2048 {
            let j = rng.gen_range(i..raw.len());
            raw.swap(i, j);
        }
        raw.truncate(2048);
    }
    let mut outcomes: Vec<(i32, i32)> = Vec::with_capacity(raw.len() * 2);
    for &(a, b) in &raw {
        outcomes.push((a, b));
        outcomes.push((b, a)); // le barème n'a pas de camp privilégié
    }

    // --- 2. Table d'équité. ---
    let t0 = std::time::Instant::now();
    let eq = Equity::build(&outcomes);
    eprintln!("  table d'équité en {:?}", t0.elapsed());
    println!("=== Table d'équité — P(N-S gagne) au score courant");
    for &(a, b) in &probes {
        if a < TARGET && b < TARGET {
            println!("  {a:>4}-{b:<4}  {:.3}", eq.at(a, b));
        }
    }

    // --- 3. Combien de donnes se jouent où ? ---
    let mut visits = [0u64; 5]; // par distance de la ligne, camp en tête
    let mut total_deals = 0u64;
    let mut matches = 0u64;
    // Les scores réellement visités, avant chaque donne. C'est la pondération
    // qui transforme un coût par position en coût par partie.
    let mut visited: Vec<(i32, i32)> = Vec::new();
    let mut sim = StdRng::seed_from_u64(seed ^ 0x5eed);
    for _ in 0..20_000 {
        let (mut a, mut b) = (0i32, 0i32);
        matches += 1;
        while a < TARGET && b < TARGET {
            let lead = a.max(b);
            let bucket = ((lead / 500).min(3)) as usize;
            visits[bucket] += 1;
            total_deals += 1;
            visited.push((a, b));
            let (ds, de) = outcomes[sim.gen_range(0..outcomes.len())];
            a += ds;
            b += de;
        }
    }
    let deals_per_match = total_deals as f64 / matches as f64;
    println!("\n=== Où se jouent les donnes (20 000 parties simulées)");
    for (i, lbl) in ["0-499", "500-999", "1000-1499", "1500-1999"].iter().enumerate() {
        println!(
            "  meneur à {lbl:<10} {:>6.1} % des donnes",
            visits[i] as f64 * 100.0 / total_deals as f64
        );
    }
    println!("  {deals_per_match:.1} donnes par partie");
    // La zone qui compte : les deux camps à portée de la ligne ET serrés. C'est
    // là que l'équité a un vrai coude ; ailleurs elle est quasi linéaire (loin
    // de la ligne) ou quasi plate (partie décidée).
    let both_close = visited.iter().filter(|&&(a, b)| a >= 1500 && b >= 1500).count();
    let tight = visited
        .iter()
        .filter(|&&(a, b)| a >= 1500 && b >= 1500 && (a - b).abs() <= 250)
        .count();
    println!(
        "  les deux camps ≥ 1500 : {:.1} % des donnes ; et serrés (écart ≤ 250) : {:.1} %",
        both_close as f64 * 100.0 / total_deals as f64,
        tight as f64 * 100.0 / total_deals as f64
    );

    // --- 4. Positions réelles. ---
    let mut positions: Vec<Position> = Vec::new();
    let (mut decisions, mut scanned) = (0u64, 0u64);
    for r in &replays {
        if positions.len() >= n_positions {
            break;
        }
        let mut ps = positions_of(r, min_cards, max_cards);
        if ps.is_empty() {
            continue;
        }
        decisions += ps.len() as u64;
        scanned += 1;
        // Une seule position par donne : sinon l'échantillon est corrélé.
        let k = rng.gen_range(0..ps.len());
        positions.push(ps.swap_remove(k));
    }
    let dec_per_deal = decisions as f64 / scanned.max(1) as f64;
    eprintln!("  {} positions ({min_cards}-{max_cards} cartes restantes)", positions.len());
    println!("  {dec_per_deal:.1} décisions par donne (4 sièges), soit {:.1} pour un camp", dec_per_deal / 2.0);

    // --- 5. L'A/B apparié, par score de partie. ---
    println!(
        "\n=== A/B apparié : DealScore vs équité de match ({} positions × {worlds} mondes)",
        positions.len()
    );
    println!(
        "{:<12} {:>8} {:>12} {:>12} {:>12} {:>10}",
        "score", "diverge", "perte moy.", "perte|div", "p95|div", "écart cédé"
    );

    // `state_for` donne le score de partie de la position `i` : constant pour un
    // état sondé, tiré de la distribution réellement visitée pour la ligne
    // « réel ». C'est cette dernière qui répond à « combien ça coûte », les
    // autres disent seulement *où* ça coûte.
    let mut probe = |label: String, state_for: &(dyn Fn(usize) -> (i32, i32) + Sync)| -> f64 {
        let t = std::time::Instant::now();
        let run = |(i, p): (usize, &Position)| -> Tally {
            let cum = state_for(i);
            let mut rng = StdRng::seed_from_u64(seed ^ (i as u64) << 16 ^ cum.0 as u64);
            let mut tt = new_tt_buffer();
            let mut tally = Tally::default();
            measure(p, &eq, cum, worlds, &mut rng, &mut tt, &mut tally);
            tally
        };

        #[cfg(feature = "parallel")]
        let mut tally = positions
            .par_iter()
            .enumerate()
            .map(run)
            .reduce(Tally::default, |mut a, b| {
                a.merge(b);
                a
            });
        #[cfg(not(feature = "parallel"))]
        let mut tally = {
            let mut acc = Tally::default();
            for x in positions.iter().enumerate() {
                acc.merge(run(x));
            }
            acc
        };

        let div_pct = tally.diverged as f64 * 100.0 / tally.positions.max(1) as f64;
        let loss_all = tally.loss_pp.iter().sum::<f64>() / tally.positions.max(1) as f64;
        let loss_div = mean(&tally.loss_pp);
        let ceded = mean(&tally.margin_given_up);
        let p95 = quantile(&mut tally.loss_pp, 0.95);
        println!(
            "{label:<12} {div_pct:>7.1}% {loss_all:>11.3} {loss_div:>11.3} {p95:>11.3} {ceded:>9.1}"
        );
        eprintln!("  ({:?})", t.elapsed());
        loss_all
    };

    for &cum in &probes {
        probe(format!("{}-{}", cum.0, cum.1), &|_| cum);
    }

    // La ligne qui compte : chaque position reçoit un score de partie tiré de la
    // distribution des scores réellement visités.
    let picks: Vec<(i32, i32)> = (0..positions.len())
        .map(|i| visited[(i * 2_654_435_761) % visited.len()])
        .collect();
    let real_loss = probe("réel (pondéré)".to_string(), &|i| picks[i]);

    println!("\nPertes en points de % de probabilité de gagner la partie.");
    println!("« écart cédé » = points de score de donne abandonnés par la carte de l'équité.");
    println!(
        "\nCoût par partie, borne haute naïve : {:.3} pp/décision × {:.1} décisions × {:.1} donnes\n  = {:.2} pp de probabilité de gagner la partie.",
        real_loss,
        dec_per_deal / 2.0,
        deals_per_match,
        real_loss * (dec_per_deal / 2.0) * deals_per_match
    );
    println!("  (borne haute : les pertes sont sommées comme si elles étaient indépendantes,");
    println!("   et chaque perte est déjà biaisée à la hausse — l'argmax de l'équité est choisi");
    println!("   sur l'échantillon même qui l'évalue.)");
}
