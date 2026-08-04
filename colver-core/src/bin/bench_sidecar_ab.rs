//! A/B **alterné** de latence entre deux sidecars playgen.
//!
//! Le pendant chronométré de [`bench_prefill_eq`], qui compare les
//! *distributions* : celui-ci compare le temps vu du client, qui est ce que
//! Dédé dépense réellement dans son budget de 1 200 ms par coup.
//!
//! ## Pourquoi alterné, et pourquoi la médiane
//!
//! Deux exécutions séquentielles ne se comparent pas sur cette machine : la
//! charge dérive de 20 %, ce qui est plus grand que la plupart des gains qu'on
//! cherche. La même règle que `dd_ab_revs.sh` côté solveur DD. Ici chaque
//! position est donc envoyée **à A puis à B immédiatement**, et on répète : une
//! dérive de charge frappe les deux bras au même instant. On rend la médiane
//! des ratios appariés plutôt que le ratio des moyennes — une seule requête
//! ralentie par un voisin (le sidecar de prod qui sert un joueur, un autre
//! run) déplace une moyenne et pas une médiane.
//!
//! ## Un seul thread client, délibérément
//!
//! C'est le régime de la prod : un joueur seul sur le web envoie **une requête
//! à la fois**, donc rien n'amortit le coût fixe d'une requête. C'est aussi le
//! régime où le préfixe groupé a rendu 1,9×. Pour le débit en génération de
//! masse, c'est `gen_games_isdd` qui mesure, pas ce binaire.
//!
//! ```bash
//! cargo run --release --bin bench_sidecar_ab -- \
//!   --a http://moxxi:8013 --b http://moxxi:8014 \
//!   --positions 12 --worlds 40 --rounds 5
//! ```

use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::SeedableRng;

use colver_core::agent::{AgentSpec, MatchContext, Player};
use colver_core::state::{GameState, Phase};
use colver_core::worlds::{SidecarWorldSource, WorldSource};

struct Position {
    dealer: u8,
    hands: [u32; 4],
    history: Vec<(u8, u8)>,
    observer: u8,
    cards_left: usize,
}

/// Une requête, chronométrée de bout en bout (réseau et HTTP compris).
fn timed(url: &str, pos: &Position, n: usize) -> Option<(f64, usize)> {
    let mut src = SidecarWorldSource::new(url, 0.8, Duration::from_secs(30));
    let state0 = GameState::new(pos.dealer, pos.hands);
    src.init_deal(&state0, pos.observer);
    let mut state = state0;
    for &(p, a) in &pos.history {
        src.observe(&state, p, a);
        state.step(a);
    }
    // Le chrono ne démarre qu'ici : le rejeu ci-dessus est du travail client,
    // identique pour les deux bras, et il n'a rien à voir avec le sidecar.
    let t = Instant::now();
    let w = src.worlds_unfiltered(pos.observer, n).ok()?;
    let ms = t.elapsed().as_secs_f64() * 1e3;
    if w.is_empty() {
        return None;
    }
    Some((ms, w.len()))
}

/// Positions réparties sur tous les stades de la donne : le nombre de pas de
/// décodage — donc tout l'effet cherché — décroît avec les cartes restantes.
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
    // Une position par palier, en faisant tourner le palier visé d'une donne à
    // l'autre. Un filtre de parité sur l'index de sortie — ce que fait
    // `bench_prefill_eq` — a l'air de couvrir les stades et n'en couvre en
    // réalité que le début : il prend jusqu'à quatre positions par donne, donc
    // toujours les quatre premières levées, et la finale n'apparaît jamais.
    // C'est précisément le régime où le nombre de pas de décodage est le plus
    // petit, donc celui où un gain sur le compte de pas doit le moins payer.
    let mut target = 8usize;

    while out.len() < want {
        let mut state = GameState::deal_random(dealer, &mut rng);
        let hands = state.hands;
        let mut ctx = MatchContext::new(dealer);
        ctx.reset_deal(dealer);
        for p in players.iter_mut() {
            p.init_deal(&state);
        }
        let mut history: Vec<(u8, u8)> = Vec::new();
        let mut taken = 0;
        while !state.is_terminal() {
            if state.phase == Phase::Playing && out.len() < want && taken < 1 {
                let obs = state.current_player();
                let left = state.hands[obs as usize].count_ones() as usize;
                if left == target {
                    out.push(Position {
                        dealer,
                        hands,
                        history: history.clone(),
                        observer: obs,
                        cards_left: left,
                    });
                    taken += 1;
                }
            }
            let seat = state.current_player();
            let before = state;
            let Ok(action) = players[seat as usize].action(&before, &ctx) else {
                break;
            };
            for p in players.iter_mut() {
                p.observe(&before, seat, action);
            }
            ctx.track(&before, action);
            history.push((seat, action));
            state.step(action);
        }
        dealer = (dealer + 1) % 4;
        target = if target <= 2 { 8 } else { target - 1 };
    }
    out
}

fn median(v: &mut [f64]) -> f64 {
    if v.is_empty() {
        return f64::NAN;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let n = v.len();
    if n % 2 == 1 {
        v[n / 2]
    } else {
        (v[n / 2 - 1] + v[n / 2]) / 2.0
    }
}

fn main() {
    let argv: Vec<String> = std::env::args().collect();
    let mut a_url = String::from("http://127.0.0.1:8013");
    let mut b_url = String::from("http://127.0.0.1:8014");
    let mut positions = 12usize;
    let mut worlds = 40usize;
    let mut rounds = 5usize;
    let mut seed = 7u64;
    let mut i = 1;
    while i < argv.len() {
        match argv[i].as_str() {
            "--a" => { a_url = argv[i + 1].clone(); i += 2 }
            "--b" => { b_url = argv[i + 1].clone(); i += 2 }
            "--positions" => { positions = argv[i + 1].parse().unwrap(); i += 2 }
            "--worlds" => { worlds = argv[i + 1].parse().unwrap(); i += 2 }
            "--rounds" => { rounds = argv[i + 1].parse().unwrap(); i += 2 }
            "--seed" => { seed = argv[i + 1].parse().unwrap(); i += 2 }
            o => { eprintln!("argument inconnu : {o}"); std::process::exit(2) }
        }
    }

    let pos = collect_positions(positions, seed);
    eprintln!(
        "{} positions × {rounds} tours, {worlds} mondes par requête\n  A = {a_url}\n  B = {b_url}\n",
        pos.len()
    );

    // Tour de chauffe, jeté : la première requête d'un sidecar paie ses
    // allocations de cache et fausserait le premier ratio apparié.
    for p in pos.iter().take(2) {
        let _ = timed(&a_url, p, worlds);
        let _ = timed(&b_url, p, worlds);
    }

    let mut ratios: Vec<f64> = Vec::new();
    let mut a_all: Vec<f64> = Vec::new();
    let mut b_all: Vec<f64> = Vec::new();
    // Par palier de cartes restantes : le gain doit croître avec le nombre de
    // pas de décodage, donc décroître avec l'avancement de la donne. Un gain
    // plat sur tous les stades serait le signe qu'on mesure autre chose.
    let mut by_left: std::collections::BTreeMap<usize, Vec<f64>> = Default::default();

    for r in 0..rounds {
        for p in &pos {
            // Alterné à la requête, et le sens s'inverse d'un tour à l'autre
            // pour que « qui passe en premier » ne soit pas un biais constant.
            let (first, second) = if r % 2 == 0 { (&a_url, &b_url) } else { (&b_url, &a_url) };
            let t1 = timed(first, p, worlds);
            let t2 = timed(second, p, worlds);
            let (Some((m1, _)), Some((m2, _))) = (t1, t2) else { continue };
            let (ma, mb) = if r % 2 == 0 { (m1, m2) } else { (m2, m1) };
            a_all.push(ma);
            b_all.push(mb);
            ratios.push(ma / mb);
            by_left.entry(p.cards_left).or_default().push(ma / mb);
        }
    }

    if ratios.is_empty() {
        eprintln!("aucune paire exploitable");
        std::process::exit(1);
    }

    println!("\n  cartes restantes │ ratio A/B (médiane) │ n");
    println!("  ─────────────────┼─────────────────────┼────");
    for (left, v) in by_left.iter_mut() {
        println!("  {left:>16} │ {:>19.3} │ {:>3}", median(v), v.len());
    }

    let (ma, mb, mr) = (median(&mut a_all), median(&mut b_all), median(&mut ratios));
    println!("\n  A : médiane {ma:.1} ms   ({} requêtes)", a_all.len());
    println!("  B : médiane {mb:.1} ms   ({} requêtes)", b_all.len());
    println!("\n  ratio apparié médian A/B = {mr:.3}   →  A est {:.2}× {}", 1.0 / mr,
        if mr < 1.0 { "plus rapide" } else { "plus LENT" });
}
