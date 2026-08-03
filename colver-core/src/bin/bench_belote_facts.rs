//! Que vaut la belote annoncée comme contrainte de déterminisation ?
//!
//! Deux questions, deux sources :
//!
//! 1. **Fréquence** — sur des donnes réellement jouées (corpus COLVGM01), à
//!    combien de positions la déduction porte-t-elle sur un siège *caché* ?
//!    Deux formes, comptées séparément (cf. `play::belote_facts`) :
//!    l'**annonce** place une carte, le **silence** en exclut une.
//! 2. **Coût de l'avoir ignorée** — quelle fraction des mondes tirés était
//!    impossible ? Analytique pour le tirage uniforme (un tirage aveugle place
//!    la carte au hasard parmi les sièges cachés), **empirique** pour playgen,
//!    qui est la source par défaut d'IS-DD et ne peut pas déduire l'annonce :
//!    elle n'est pas dans son flux de tokens, il ne voit qu'un Roi d'atout
//!    tomber.
//!
//! Le second volet demande le sidecar (`--sidecar $COLVER_PLAYGEN_GPU_URL`) et
//! interroge `worlds_unfiltered`, sans quoi on mesurerait ce qui reste après le
//! filtre au lieu de ce qu'il jette.
//!
//! ```bash
//! cargo run -p colver-core --bin bench_belote_facts --release -- \
//!   --corpus data/training/games_500k.bin --deals 20000
//! cargo run -p colver-core --bin bench_belote_facts --release -- \
//!   --corpus data/training/games_500k.bin --deals 20000 \
//!   --sidecar "$COLVER_PLAYGEN_GPU_URL" --positions 200 --worlds 32
//! ```
//!
//! Sortie JSON sur stdout (`--json`), pour `scripts/analysis/belote_facts.py`.
//!
//! Troisième mode : `--isdd N [--ref-worlds M]` rend une décision IS-DD par
//! position contrainte non forcée, et la table EV d'un juge à `M` mondes. Deux
//! exécutions, la seconde sous `COLVER_NO_BELOTE_FACTS=1` (feature
//! `belief_ablation`), donnent l'A/B ; `scripts/analysis/belote_regret.py` le
//! dépouille.
//!
//! ```bash
//! cargo build --release --features "parallel belief_ablation" --bin bench_belote_facts
//! ./target/release/bench_belote_facts --deals 50000 --isdd 2000 --ref-worlds 400 > on.txt
//! COLVER_NO_BELOTE_FACTS=1 ./target/release/bench_belote_facts --deals 50000 --isdd 2000 > off.txt
//! ```

use std::time::{Duration, Instant};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use colver_core::card::{card_count, CardIter, CardSet};
use colver_core::game_replay::GameReplay;
use colver_core::play::{belote_facts, BeloteFacts};
use colver_core::state::{GameState, Phase};
use colver_core::worlds::{SidecarWorldSource, WorldSource};

/// Ce que les faits disent des seuls sièges cachés d'un observateur.
struct HiddenView {
    held: usize,
    banned: usize,
    /// Probabilité qu'un tirage aveugle produise un monde impossible.
    p_impossible: f64,
}

fn hidden_view(facts: &BeloteFacts, state: &GameState, observer: u8) -> HiddenView {
    let hidden_cards: u32 = (0..4)
        .filter(|&p| p != observer as usize)
        .map(|p| card_count(state.hands[p]))
        .sum();
    let mut view = HiddenView { held: 0, banned: 0, p_impossible: 0.0 };
    if hidden_cards == 0 {
        return view;
    }
    let mut p_ok = 1.0f64;
    for p in 0..4usize {
        if p == observer as usize {
            continue;
        }
        let n_p = card_count(state.hands[p]) as f64;
        for _ in CardIter(facts.held[p]) {
            view.held += 1;
            // Un tirage aveugle place cette carte chez `p` avec la fréquence de
            // ses cartes parmi les cartes cachées.
            p_ok *= n_p / hidden_cards as f64;
        }
        // Les exclusions qui découlent d'un `held` sont déjà comptées par la
        // ligne du dessus : ne garder que celles qu'aucun `held` n'implique.
        let implied: CardSet = (0..4)
            .filter(|&q| q != p)
            .fold(0, |acc, q| acc | facts.held[q]);
        for _ in CardIter(facts.banned[p] & !implied) {
            view.banned += 1;
            p_ok *= 1.0 - n_p / hidden_cards as f64;
        }
    }
    view.p_impossible = 1.0 - p_ok;
    view
}

/// Position susceptible de porter une déduction sur un siège caché, **sans passer
/// par `belote_facts`** : le Roi ou la Dame d'atout est tombé, l'autre n'est ni
/// tombé ni dans la main de l'observateur.
///
/// L'A/B ne peut pas sélectionner ses positions avec la chose qu'il ablate — sous
/// `COLVER_NO_BELOTE_FACTS=1` la liste serait vide et les deux exécutions ne
/// compareraient rien.
fn constrainable(state: &GameState) -> bool {
    use colver_core::card::{card_to_bit, make_card};
    let trump = state.contract.trump_suit();
    let q = card_to_bit(make_card(trump, 4));
    let k = card_to_bit(make_card(trump, 5));
    let played = state.played_cards;
    let (q_out, k_out) = (played & q != 0, played & k != 0);
    if q_out == k_out {
        return false; // aucun des deux, ou les deux : rien à déduire
    }
    let other = if q_out { k } else { q };
    if state.hands[state.current_player as usize] & other != 0 {
        return false;
    }
    // Une position forcée ne peut pas changer d'avis : la garder diluerait les
    // deux mesures (taux de bascule *et* regret) sans rien y ajouter.
    state.legal_actions().count_ones() >= 2
}

#[derive(Default)]
struct Tally {
    deals: usize,
    positions: usize,
    /// Positions où un siège caché porte au moins une déduction.
    constrained: usize,
    /// … dont une carte *placée* (annonce entendue).
    with_held: usize,
    /// … dont seulement des cartes *exclues* (silence sur un Roi ou une Dame d'atout).
    banned_only: usize,
    deals_with_announcement: usize,
    /// Somme des probabilités qu'un monde uniforme aveugle soit impossible.
    p_impossible_sum: f64,
    p_impossible_sum_constrained: f64,
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
    let corpus = get("--corpus", "data/training/games_500k.bin");
    let deals: usize = get("--deals", "20000").parse().unwrap();
    let seed: u64 = get("--seed", "42").parse().unwrap();
    let sidecar = get("--sidecar", "");
    let positions: usize = get("--positions", "200").parse().unwrap();
    let worlds_per_pos: usize = get("--worlds", "32").parse().unwrap();
    let isdd_positions: usize = get("--isdd", "0").parse().unwrap();
    let isdd_worlds: u32 = get("--isdd-worlds", "60").parse().unwrap();
    let ref_worlds: u32 = get("--ref-worlds", "0").parse().unwrap();
    let json = args.iter().any(|a| a == "--json");

    let t0 = Instant::now();
    let replays = match GameReplay::load_all(&corpus) {
        Ok(r) => r,
        Err(e) => {
            eprintln!("corpus {corpus}: {e}");
            std::process::exit(1);
        }
    };
    let replays: Vec<_> = replays.into_iter().take(deals).collect();
    eprintln!("{} donnes chargées en {:?}", replays.len(), t0.elapsed());

    // ---- 1. Fréquence sur donnes jouées -------------------------------------
    let mut tally = Tally::default();
    // (donne, index d'action) des positions contraintes, pour le second volet.
    let mut hot: Vec<(usize, usize)> = Vec::new();
    // Idem pour l'A/B de décision, mais sélectionnées sans `belote_facts`.
    let mut hot_raw: Vec<(usize, usize)> = Vec::new();

    for (di, replay) in replays.iter().enumerate() {
        tally.deals += 1;
        let mut announced = false;
        let mut ai = 0usize;
        replay.replay_with(|state, _tracking, _action| {
            let idx = ai;
            ai += 1;
            if state.phase != Phase::Playing {
                return;
            }
            tally.positions += 1;
            if state.belote[0] >= 1 || state.belote[1] >= 1 {
                announced = true;
            }
            if constrainable(state) {
                hot_raw.push((di, idx));
            }
            let observer = state.current_player;
            let facts = belote_facts(state);
            if facts.is_empty() {
                return;
            }
            let view = hidden_view(&facts, state, observer);
            if view.held == 0 && view.banned == 0 {
                return; // la déduction ne portait que sur l'observateur lui-même
            }
            tally.constrained += 1;
            if view.held > 0 {
                tally.with_held += 1;
            } else {
                tally.banned_only += 1;
            }
            tally.p_impossible_sum += view.p_impossible;
            tally.p_impossible_sum_constrained += view.p_impossible;
            hot.push((di, idx));
        });
        if announced {
            tally.deals_with_announcement += 1;
        }
    }

    let pct = |a: usize, b: usize| if b == 0 { 0.0 } else { 100.0 * a as f64 / b as f64 };
    eprintln!(
        "positions de jeu : {} sur {} donnes",
        tally.positions, tally.deals
    );
    eprintln!(
        "  contraintes sur un siège caché : {} ({:.2} %) — placement {} ({:.2} %), exclusion seule {} ({:.2} %)",
        tally.constrained,
        pct(tally.constrained, tally.positions),
        tally.with_held,
        pct(tally.with_held, tally.positions),
        tally.banned_only,
        pct(tally.banned_only, tally.positions),
    );
    eprintln!(
        "  donnes avec au moins une annonce : {} ({:.2} %)",
        tally.deals_with_announcement,
        pct(tally.deals_with_announcement, tally.deals)
    );
    eprintln!(
        "  mondes impossibles d'un tirage uniforme aveugle : {:.2} % des mondes aux positions contraintes, {:.3} % sur l'ensemble des positions",
        100.0 * tally.p_impossible_sum_constrained / tally.constrained.max(1) as f64,
        100.0 * tally.p_impossible_sum / tally.positions.max(1) as f64,
    );

    // ---- 2. Mondes playgen réellement produits ------------------------------
    let mut sidecar_stats = None;
    if !sidecar.is_empty() && !hot.is_empty() {
        let mut rng = StdRng::seed_from_u64(seed);
        let mut chosen: Vec<(usize, usize)> = Vec::new();
        for _ in 0..positions.min(hot.len()) {
            chosen.push(hot[rng.gen_range(0..hot.len())]);
        }

        let mut asked = 0usize;
        let mut returned = 0usize;
        let mut belote_bad = 0usize;
        let mut void_bad = 0usize;
        let mut count_bad = 0usize;
        let t = Instant::now();

        for (n, &(di, target)) in chosen.iter().enumerate() {
            let replay = &replays[di];
            let mut src =
                SidecarWorldSource::new(sidecar.clone(), 1.0, Duration::from_secs(30));
            let mut state = GameState::new(replay.dealer, replay.hands);
            src.init_deal(&state, 0);
            let mut observer = 0u8;
            let mut ready = false;
            for (i, &action) in replay.actions.iter().enumerate() {
                if i == target {
                    observer = state.current_player;
                    ready = true;
                    break;
                }
                let before = state;
                let player = state.current_player;
                state.step(action);
                src.observe(&before, player, action);
            }
            if !ready {
                continue;
            }
            let facts = belote_facts(&state);
            let got = match src.worlds_unfiltered(observer, worlds_per_pos) {
                Ok(w) => w,
                Err(e) => {
                    eprintln!("sidecar: {e}");
                    break;
                }
            };
            asked += worlds_per_pos;
            returned += got.len();
            for hands in &got {
                if !facts.allows(hands) {
                    belote_bad += 1;
                }
                let counts_ok =
                    (0..4).all(|p| card_count(hands[p]) == card_count(state.hands[p]));
                if !counts_ok {
                    count_bad += 1;
                }
                let voids_ok = (0..4).all(|p| {
                    (0..4).all(|s| {
                        state.voids[p] & (1 << s) == 0
                            || hands[p] & (0xFFu32 << (s * 8)) == 0
                    })
                });
                if !voids_ok {
                    void_bad += 1;
                }
            }
            if n % 25 == 0 {
                eprintln!("  … {n}/{} positions", chosen.len());
            }
        }

        eprintln!(
            "playgen : {returned} mondes rendus sur {asked} demandés en {:?}",
            t.elapsed()
        );
        eprintln!(
            "  incompatibles avec la belote : {belote_bad} ({:.2} %)",
            pct(belote_bad, returned)
        );
        eprintln!(
            "  mauvais compte de cartes : {count_bad} ({:.2} %) — coupe révélée ignorée : {void_bad} ({:.2} %)",
            pct(count_bad, returned),
            pct(void_bad, returned)
        );
        sidecar_stats = Some((asked, returned, belote_bad, count_bad, void_bad));
    }

    // ---- 3. La décision change-t-elle ? -------------------------------------
    // Une carte par position, à comparer entre deux exécutions (feature
    // `belief_ablation` + COLVER_NO_BELOTE_FACTS=1). Mondes uniformes et **mode
    // compte** : sous échéance, deux exécutions ne traverseraient pas le même
    // nombre de mondes et la comparaison ne dirait plus rien.
    if isdd_positions > 0 && !hot_raw.is_empty() {
        use colver_core::is_dd::{IsDdConfig, IsDdSearch};
        let mut rng = StdRng::seed_from_u64(seed);
        let mut chosen: Vec<(usize, usize)> = Vec::new();
        for _ in 0..isdd_positions.min(hot_raw.len()) {
            chosen.push(hot_raw[rng.gen_range(0..hot_raw.len())]);
        }
        let config = IsDdConfig {
            determinizations: isdd_worlds,
            time_limit_ms: None,
            ..Default::default()
        };
        // Référence optionnelle : la même position résolue avec **beaucoup** plus
        // de mondes, pour chiffrer le regret de la carte choisie au lieu de se
        // contenter de « elle a changé ». À ne calculer que dans le bras qui a la
        // déduction — c'est la distribution correcte qui sert d'arbitre, et une
        // graine distincte évite que le bras à 60 mondes hérite des mondes de son
        // propre juge.
        let ref_config = IsDdConfig {
            determinizations: ref_worlds,
            time_limit_ms: None,
            parallel: true,
            ..Default::default()
        };
        let t = Instant::now();
        let mut decisions: Vec<String> = Vec::with_capacity(chosen.len());
        for &(di, target) in &chosen {
            let replay = &replays[di];
            let mut state = GameState::new(replay.dealer, replay.hands);
            let mut search = IsDdSearch::new();
            // L'observateur est le siège au trait à la position visée ; les
            // croyances doivent être bâties de son point de vue depuis le début.
            let mut probe = state;
            for &action in replay.actions.iter().take(target) {
                probe.step(action);
            }
            let observer = probe.current_player;
            search.init_deal_with_config(&state, observer, &config);
            for &action in replay.actions.iter().take(target) {
                let before = state;
                let player = state.current_player;
                state.step(action);
                search.record_action(&before, player, action);
            }
            let mut prng = StdRng::seed_from_u64(seed ^ ((di as u64) << 8) ^ target as u64);
            let res = search.search_with_stats(&state, &config, &mut prng);
            let mut line = format!("{di}:{target}:{}", res.best_action);
            if ref_worlds > 0 {
                let mut rprng =
                    StdRng::seed_from_u64(!(seed ^ ((di as u64) << 8) ^ target as u64));
                let r = search.search_with_stats(&state, &ref_config, &mut rprng);
                line.push(':');
                for (i, (card, ev)) in r.card_scores.iter().enumerate() {
                    if i > 0 {
                        line.push(';');
                    }
                    line.push_str(&format!("{card}={ev:.3}"));
                }
            }
            decisions.push(line);
        }
        eprintln!(
            "IS-DD : {} décisions ({} mondes, {}) en {:?}",
            decisions.len(),
            isdd_worlds,
            colver_core::play::belote_ablation_label(),
            t.elapsed()
        );
        for d in &decisions {
            println!("{d}");
        }
    }

    if json {
        let mut out = format!(
            r#"{{"deals":{},"positions":{},"constrained":{},"with_held":{},"banned_only":{},"deals_with_announcement":{},"p_impossible_uniform_constrained":{:.6},"p_impossible_uniform_all":{:.6}"#,
            tally.deals,
            tally.positions,
            tally.constrained,
            tally.with_held,
            tally.banned_only,
            tally.deals_with_announcement,
            tally.p_impossible_sum_constrained / tally.constrained.max(1) as f64,
            tally.p_impossible_sum / tally.positions.max(1) as f64,
        );
        if let Some((asked, returned, belote_bad, count_bad, void_bad)) = sidecar_stats {
            out.push_str(&format!(
                r#","playgen":{{"asked":{asked},"returned":{returned},"belote_violations":{belote_bad},"count_violations":{count_bad},"void_violations":{void_bad}}}"#
            ));
        }
        out.push('}');
        println!("{out}");
    }
}
