//! Deux sidecars échantillonnent-ils la **même distribution** de mondes ?
//!
//! Sert à valider un changement du chemin GPU (ici le préfixe groupé) qui est
//! *mathématiquement* identique mais **pas bit-à-bit** : les matmuls changent
//! de forme, donc l'ordre de réduction flottant aussi, et un logit qui bouge de
//! 1e-6 peut faire basculer un tirage sur une quasi-égalité. Comparer les
//! mondes un à un ne dirait donc rien.
//!
//! ## Le témoin est le cœur de la mesure
//!
//! Deux tirages de 512 mondes diffèrent **toujours** : à p = 0,5 l'écart-type
//! d'une marginale vaut 0,022, donc deux échantillons indépendants s'écartent
//! de ~0,031 en moyenne. Un « écart max de 0,09 entre A et B » ne veut donc
//! rien dire tout seul.
//!
//! On mesure donc **trois** écarts sur les mêmes positions :
//!
//! - `A↔B` : nouveau contre ancien — ce qu'on veut juger ;
//! - `B↔B'` : ancien contre lui-même, deux tirages — le **bruit
//!   d'échantillonnage**, la seule référence honnête ;
//! - `A↔A'` : nouveau contre lui-même, pour vérifier qu'il n'a pas *réduit* sa
//!   propre variabilité (un préfixe cassé qui rendrait toujours le même monde
//!   passerait un test A↔B mal construit).
//!
//! Verdict : `A↔B` doit être du même ordre que `B↔B'`. S'il est nettement plus
//! grand, les distributions diffèrent.
//!
//! ```bash
//! cargo run --release --bin bench_prefill_eq -- \
//!   --a http://moxxi:8013 --b http://moxxi:8014 --positions 40 --worlds 256
//! ```

use std::time::Duration;

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::agent::{AgentSpec, MatchContext, Player};
use colver_core::state::{GameState, Phase};
use colver_core::worlds::{SidecarWorldSource, WorldSource};

/// Une position de jeu, rejouable telle que le sidecar l'attend.
struct Position {
    dealer: u8,
    hands: [u32; 4],
    history: Vec<(u8, u8)>,
    observer: u8,
    cards_left: usize,
}

/// Marginales p(carte → siège), l'agrégat que deux échantillons d'une même
/// distribution doivent partager.
fn marginals(worlds: &[[u32; 4]]) -> Vec<f32> {
    let mut counts = vec![0f32; 128];
    for w in worlds {
        for (p, &h) in w.iter().enumerate() {
            let mut b = h;
            while b != 0 {
                counts[p * 32 + b.trailing_zeros() as usize] += 1.0;
                b &= b - 1;
            }
        }
    }
    let n = worlds.len().max(1) as f32;
    counts.iter().map(|c| c / n).collect()
}

/// (écart moyen, écart max) entre deux jeux de marginales.
fn compare(a: &[f32], b: &[f32]) -> (f64, f64) {
    let mut sum = 0.0;
    let mut max: f64 = 0.0;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (x - y).abs() as f64;
        sum += d;
        max = max.max(d);
    }
    (sum / a.len() as f64, max)
}

fn sample(url: &str, pos: &Position, n: usize) -> Option<Vec<[u32; 4]>> {
    let mut src = SidecarWorldSource::new(url, 0.8, Duration::from_secs(30));
    let state0 = GameState::new(pos.dealer, pos.hands);
    src.init_deal(&state0, pos.observer);
    let mut state = state0;
    for &(p, a) in &pos.history {
        src.observe(&state, p, a);
        state.step(a);
    }
    match src.worlds_unfiltered(pos.observer, n) {
        Ok(w) if !w.is_empty() => Some(w),
        Ok(_) => None,
        Err(e) => {
            eprintln!("  {url}: {e}");
            None
        }
    }
}

/// Joue des donnes avec un bot bon marché et retient une position par palier de
/// cartes restantes : le coût du préfixe croît avec l'avancement de la donne,
/// donc un échantillon pris à un seul stade ne testerait qu'un régime.
fn collect_positions(want: usize, seed: u64) -> Vec<Position> {
    let spec = AgentSpec::from_toml_str(
        "[bid]\nstrategy = \"improved_v2\"\n[play]\nmethod = \"heuristic\"\n",
    )
    .expect("spec");
    let mut players: [Box<dyn Player>; 4] = (0..4)
        .map(|s| spec.build(s).expect("build"))
        .collect::<Vec<_>>()
        .try_into()
        .map_err(|_| ())
        .expect("4");
    let mut rng = StdRng::seed_from_u64(seed);
    let mut out = Vec::new();
    let mut dealer = 0u8;

    while out.len() < want {
        let mut state = GameState::deal_random(dealer, &mut rng);
        let hands = state.hands;
        let mut ctx = MatchContext::new(dealer);
        ctx.reset_deal(dealer);
        for p in players.iter_mut() {
            p.init_deal(&state);
        }
        let mut history: Vec<(u8, u8)> = Vec::new();
        let mut taken_this_deal = 0;
        while !state.is_terminal() {
            if state.phase == Phase::Playing && out.len() < want && taken_this_deal < 4 {
                let obs = state.current_player();
                let left = state.hands[obs as usize].count_ones() as usize;
                // Un palier sur deux, pour couvrir l'entame comme la finale sans
                // saturer l'échantillon avec une seule donne.
                if left >= 2 && (left + out.len()) % 2 == 0 {
                    out.push(Position {
                        dealer,
                        hands,
                        history: history.clone(),
                        observer: obs,
                        cards_left: left,
                    });
                    taken_this_deal += 1;
                }
            }
            let seat = state.current_player();
            let before = state;
            let action = match players[seat as usize].action(&before, &ctx) {
                Ok(a) => a,
                Err(_) => break,
            };
            for p in players.iter_mut() {
                p.observe(&before, seat, action);
            }
            ctx.track(&before, action);
            history.push((seat, action));
            state.step(action);
        }
        dealer = (dealer + 1) % 4;
    }
    out
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut a_url = String::from("http://127.0.0.1:8013");
    let mut b_url = String::from("http://127.0.0.1:8014");
    let mut positions = 40usize;
    let mut worlds = 256usize;
    let mut seed = 7u64;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--a" => { a_url = argv[i + 1].clone(); i += 2 }
            "--b" => { b_url = argv[i + 1].clone(); i += 2 }
            "--positions" => { positions = argv[i + 1].parse().unwrap(); i += 2 }
            "--worlds" => { worlds = argv[i + 1].parse().unwrap(); i += 2 }
            "--seed" => { seed = argv[i + 1].parse().unwrap(); i += 2 }
            o => { eprintln!("argument inconnu : {o}"); std::process::exit(2) }
        }
    }

    let pos = collect_positions(positions, seed);
    eprintln!("{} positions, {worlds} mondes chacune\n  A = {a_url}\n  B = {b_url}\n", pos.len());

    let (mut ab, mut bb, mut aa) = (Vec::new(), Vec::new(), Vec::new());
    let (mut ab_max, mut bb_max, mut aa_max) = (0f64, 0f64, 0f64);
    let mut skipped = 0;

    for (i, p) in pos.iter().enumerate() {
        let (wa, wa2, wb, wb2) = (
            sample(&a_url, p, worlds),
            sample(&a_url, p, worlds),
            sample(&b_url, p, worlds),
            sample(&b_url, p, worlds),
        );
        let (Some(wa), Some(wa2), Some(wb), Some(wb2)) = (wa, wa2, wb, wb2) else {
            skipped += 1;
            continue;
        };
        let (ma, ma2, mb, mb2) =
            (marginals(&wa), marginals(&wa2), marginals(&wb), marginals(&wb2));
        let (d1, m1) = compare(&ma, &mb);
        let (d2, m2) = compare(&mb, &mb2);
        let (d3, m3) = compare(&ma, &ma2);
        ab.push(d1);
        bb.push(d2);
        aa.push(d3);
        ab_max = ab_max.max(m1);
        bb_max = bb_max.max(m2);
        aa_max = aa_max.max(m3);
        if i < 8 {
            eprintln!(
                "  pos {i:>2} ({} cartes) : A↔B {:.4}  B↔B' {:.4}  A↔A' {:.4}   (mondes A {} / B {})",
                p.cards_left, d1, d2, d3, wa.len(), wb.len()
            );
        }
    }

    let mean = |v: &Vec<f64>| v.iter().sum::<f64>() / v.len().max(1) as f64;
    let (m_ab, m_bb, m_aa) = (mean(&ab), mean(&bb), mean(&aa));
    println!("\n{} positions comparées ({skipped} sautées)", ab.len());
    println!("  A↔B   (nouveau vs ancien) : moyenne {m_ab:.5}   max {ab_max:.4}");
    println!("  B↔B'  (ancien vs lui-même, TÉMOIN) : moyenne {m_bb:.5}   max {bb_max:.4}");
    println!("  A↔A'  (nouveau vs lui-même)        : moyenne {m_aa:.5}   max {aa_max:.4}");
    // Un témoin nul n'est pas un excellent résultat, c'est une mesure qui n'a
    // pas eu lieu : deux tirages indépendants de centaines de mondes ne
    // coïncident jamais exactement. Sans ce garde-fou, un sidecar qui ne rend
    // rien du tout fait afficher « même distribution » — le pire mode de
    // défaillance possible pour un test de non-régression.
    if ab.is_empty() || m_bb <= 1e-9 {
        println!(
            "\n  ❌ témoin nul ({} positions retenues) — les sidecars n'ont rien échantillonné, \
             la comparaison n'a pas eu lieu",
            ab.len()
        );
        std::process::exit(2);
    }
    let ratio = m_ab / m_bb;
    println!("\n  rapport A↔B / témoin = {ratio:.3}");
    // 1,15 laisse passer le bruit d'estimation sur ~40 positions sans laisser
    // passer un vrai décalage de distribution : un préfixe faux déplace les
    // marginales bien au-delà de 15 % du bruit d'échantillonnage.
    if ratio <= 1.15 {
        println!("  ✅ même distribution — l'écart A↔B est du niveau du bruit d'échantillonnage");
    } else {
        println!("  ❌ A↔B dépasse le témoin de {:.0} % — les distributions diffèrent", (ratio - 1.0) * 100.0);
        std::process::exit(1);
    }
}
