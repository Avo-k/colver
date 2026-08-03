//! A/B **apparié** de la déduction de belote, donne par donne.
//!
//! L'arène compare deux bots dans un même processus ; ici les deux bras sont le
//! *même* bot dans deux processus, l'un avec `COLVER_NO_BELOTE_FACTS=1`. Un h2h
//! d'arène ne conviendrait pas : son RNG par thread sert à la fois à distribuer
//! les donnes et à alimenter la recherche, donc le premier coup qui diverge
//! décale toutes les donnes suivantes et la comparaison redevient non appariée —
//! soit ~±1,5 pp de bruit à 1000 matchs, pour un effet attendu plus petit.
//!
//! Ici la donne `k` vient d'un RNG **dédié** (`seed ^ k`), identique dans les
//! deux bras quoi que fassent les agents. La différence par donne est donc
//! exactement nulle partout où le jeu n'a pas divergé, et le test ne porte que
//! sur les donnes où il a divergé.
//!
//! ```bash
//! cargo build --release --features "parallel belief_ablation" --bin bench_belote_ab
//! ./target/release/bench_belote_ab --ns v6_isdd_75M_isdd --ew v6_isdd_75M --deals 2000 > on.txt
//! COLVER_NO_BELOTE_FACTS=1 ./target/release/bench_belote_ab ... > off.txt
//! python3 scripts/analysis/belote_ab_diff.py on.txt off.txt
//! ```
//!
//! ⚠️ Le sidecar playgen doit tourner (les bots IS-DD y prennent leurs mondes),
//! et **le nombre de threads doit être le même dans les deux bras** : il décide
//! quel flux d'aléa reçoit chaque donne.

use std::sync::atomic::{AtomicUsize, Ordering};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::agent::spec::AgentSpec;
use colver_core::agent::MatchContext;
use colver_core::game_loop;
use colver_core::state::GameState;

const BOTS_DIR: &str = "arena/bots";

fn main() {
    let args: Vec<String> = std::env::args().collect();
    let get = |name: &str, def: &str| -> String {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1))
            .cloned()
            .unwrap_or_else(|| def.to_string())
    };
    let ns_name = get("--ns", "v6_isdd_75M_isdd");
    let ew_name = get("--ew", "v6_isdd_75M");
    let deals: usize = get("--deals", "2000").parse().unwrap();
    let seed: u64 = get("--seed", "42").parse().unwrap();
    let threads: usize = get("--threads", "8").parse().unwrap();

    let find = |name: &str| -> AgentSpec {
        let path = format!("{BOTS_DIR}/{name}.toml");
        AgentSpec::from_toml_file(&path).unwrap_or_else(|e| {
            eprintln!("{path}: {e}");
            std::process::exit(1);
        })
    };
    let mut ns_spec = find(&ns_name);
    let ew_spec = find(&ew_name);

    // **Mode compte, jamais échéance.** Les bots IS-DD de référence tournent à
    // `time_ms = 50` ; sous budget de temps, le nombre de mondes traversés dépend
    // de la charge de la machine, donc les deux bras ne feraient pas la même
    // recherche et l'écart mesuré porterait autant sur la charge que sur la
    // déduction. Le mode compte fige ce nombre.
    let dets: u32 = get("--dets", "60").parse().unwrap();
    if ns_spec.play.time_ms > 0 || ns_spec.play.determinizations != dets {
        eprintln!(
            "note : {ns_name} passé en mode compte ({dets} mondes) au lieu de time_ms={}",
            ns_spec.play.time_ms
        );
        ns_spec.play.time_ms = 0;
        ns_spec.play.determinizations = dets;
    }
    // Le sidecar met en file d'attente les requêtes de tous les threads : sous
    // charge, une demande de 60 mondes dépasse les 6 s du défaut et le run meurt
    // au bout d'une demi-heure. Ici la latence n'entre dans aucun résultat, seul
    // le nombre de mondes compte — donc on attend.
    ns_spec.worlds.timeout = std::time::Duration::from_secs(
        get("--source-timeout-s", "60").parse().unwrap(),
    );

    eprintln!(
        "{ns_name} (N-S) contre {ew_name} (E-O), {deals} donnes, {threads} threads, {}",
        colver_core::play::belote_ablation_label()
    );

    let done = AtomicUsize::new(0);
    let mut lines: Vec<(usize, i32, i32)> = std::thread::scope(|s| {
        let mut handles = Vec::new();
        for t in 0..threads {
            let (ns_spec, ew_spec) = (ns_spec.clone(), ew_spec.clone());
            let done = &done;
            handles.push(s.spawn(move || {
                let build = |spec: &AgentSpec, seat: u8| {
                    let mut spec = spec.clone();
                    spec.seed = seed.wrapping_add(t as u64 * 7919);
                    spec.build(seat).unwrap_or_else(|e| {
                        eprintln!("{}: {e}", spec.name);
                        std::process::exit(1);
                    })
                };
                let mut players = [
                    build(&ns_spec, 0),
                    build(&ew_spec, 1),
                    build(&ns_spec, 2),
                    build(&ew_spec, 3),
                ];
                let mut out = Vec::new();
                for k in (t..deals).step_by(threads) {
                    // RNG **dédié** à la distribution : identique dans les deux bras.
                    let mut deal_rng = StdRng::seed_from_u64(seed ^ (k as u64).wrapping_mul(2654435761));
                    let dealer = deal_rng.gen_range(0..4);
                    let mut state = GameState::deal_random(dealer, &mut deal_rng);
                    let mut ctx = MatchContext::new(dealer);
                    match game_loop::play_deal(&mut state, &mut players, &mut ctx) {
                        Ok(score) => out.push((k, score[0], score[1])),
                        // Une donne perdue n'est pas une mesure fausse : le
                        // dépouillement apparie sur l'intersection des deux bras,
                        // donc elle disparaît des deux côtés. Tuer le run entier
                        // pour un aller-retour raté, si.
                        Err(e) => eprintln!("donne {k} abandonnée : {e}"),
                    }
                    let n = done.fetch_add(1, Ordering::Relaxed) + 1;
                    if n % 100 == 0 {
                        eprintln!("  … {n}/{deals}");
                    }
                }
                out
            }));
        }
        handles.into_iter().flat_map(|h| h.join().unwrap()).collect()
    });

    lines.sort_unstable_by_key(|&(k, _, _)| k);
    let (mut ns_tot, mut ew_tot) = (0i64, 0i64);
    for &(k, ns, ew) in &lines {
        ns_tot += ns as i64;
        ew_tot += ew as i64;
        println!("{k}:{ns}:{ew}");
    }
    eprintln!(
        "N-S {:.1} pts/donne, E-O {:.1} pts/donne sur {} donnes",
        ns_tot as f64 / lines.len() as f64,
        ew_tot as f64 / lines.len() as f64,
        lines.len()
    );
}
