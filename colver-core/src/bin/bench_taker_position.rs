//! **Mesure A** du plan de regénération de la couche de scores
//! ([docs/data_gen/isdd_score_layer_v2.md](../../../docs/data_gen/isdd_score_layer_v2.md) §10) :
//! l'enchère **synthétique** que le générateur fabriquera est-elle dans la distribution
//! des enchères **réelles** ?
//!
//! ## La question
//!
//! Étiqueter une case `(donne, atout)` avec IS-DD sur mondes playgen exige une enchère,
//! parce que les jetons d'enchère sont dans le préfixe que playgen consomme
//! (`playgen/tokens.rs:6`), avec l'acteur codé **relativement à l'observateur**. Le plan
//! propose de la construire :
//!
//! ```text
//! camp   = N-S si dd_pts[t] > 81, sinon E-O
//! siège  = argmax evaluate_for_trump dans ce camp
//! valeur = palier dérivé de ce score
//! enchère = passes depuis (dealer+1)%4 jusqu'au siège, puis <valeur>t, puis P P P
//! ```
//!
//! Si cette construction place le preneur à une position qu'une vraie enchère ne
//! produit jamais, playgen échantillonne des mondes pour une table qui n'existe pas —
//! **sans erreur, sans signal**. C'est exactement le mode de panne du sidecar périmé.
//!
//! ## La méthode
//!
//! Le corpus `isdd_games_v1.bin` (COLVGM02) porte des enchères **réelles** — bid v6, IS-DD
//! au jeu. Pour chaque donne on rejoue l'enchère, on relève le contrat, puis on résout
//! les 4 atouts en DD et on applique la construction **sur la même donne**. La
//! comparaison est donc appariée, pas deux marginales de populations différentes.
//!
//! Trois familles de chiffres :
//!   1. **position du preneur** dans l'ordre de parole — réel contre construit ;
//!   2. **forme du préfixe** — longueur de l'enchère, nombre d'annonces, contestation,
//!      coinche. La construction en produit toujours la même : k passes, 1 annonce,
//!      3 passes ;
//!   3. **l'échelle valeur ↔ `evaluate_for_trump`** — le plan dit « palier plausible
//!      dérivé de ce score » sans donner la règle. Elle se lit ici.
//!
//! ## Les variantes candidates (`--bid-model`)
//!
//! Le même binaire déroule deux enchères pilotées par bid v6 sur les mêmes donnes :
//!
//!   * **v6 masqué sur la couleur cible** — le masque légal est réduit à `PASS`,
//!     `COINCHE`, `SURCOINCHE` et aux paliers de l'atout `t`. Une case `(donne, atout)`
//!     par appel, donc les 4 cases d'une donne sont couvertes.
//!   * **v6 libre** — aucun masque. C'est **le témoin**, et c'en est un très fort : le
//!     réseau est déterministe et les donnes sont celles du corpus, donc le rejeu doit
//!     rendre l'enchère **à l'identique**. Mesuré à 99,97 %. Sans ce contrôle, un défaut
//!     du pilote — historique mal suivi, score passé du mauvais côté, pénalité oubliée —
//!     se lirait comme une propriété de la variante masquée.
//!
//! ```bash
//! cargo build -p colver-core --release --bin bench_taker_position
//! ./target/release/bench_taker_position --games data/training/isdd_games_v1.bin \
//!     --bid-model models/bid_v6_isdd_resume/bid_nn_final.bin --json out.json
//! ```
//!
//! ⚠️ **Ne pas rediriger dans `head`** : le SIGPIPE tue le processus avant l'écriture du
//! JSON, et le run a l'air d'avoir réussi.

use std::sync::atomic::{AtomicUsize, Ordering};

use colver_core::bid_eval::evaluate_for_trump;
use colver_core::bid_net::BidNet;
use colver_core::bid_obs;
use colver_core::bidding::{self, BID_COINCHE, BID_PASS, BID_SURCOINCHE};
use colver_core::card::Suit;
use colver_core::dmc_obs::EnvTracking;
use colver_core::game_replay::GameReplay;
use colver_core::solver;
use colver_core::state::{GameState, Phase};

/// Position d'un siège dans l'ordre de parole : 0 = premier à parler = `(dealer+1)%4`,
/// 3 = le donneur, qui parle en dernier.
#[inline]
fn speak_pos(seat: u8, dealer: u8) -> usize {
    ((seat + 4 - dealer + 3) % 4) as usize
}

/// Le siège que la construction ferait annoncer : dans le camp que `dd_pts` désigne,
/// celui des deux partenaires dont la main est la plus forte à cet atout.
///
/// Égalité départagée par la **position de parole**, pas par l'indice de siège : à
/// force égale c'est celui qui parle le premier qui prend l'initiative sur la couleur.
fn constructed_seat(hands: &[u32; 4], dealer: u8, trump: u8, side: u8) -> u8 {
    let suit = Suit::from_u8(trump);
    let (a, b) = if side == 0 { (0u8, 2u8) } else { (1u8, 3u8) };
    let (ea, eb) = (
        evaluate_for_trump(hands[a as usize], suit),
        evaluate_for_trump(hands[b as usize], suit),
    );
    match ea.cmp(&eb) {
        std::cmp::Ordering::Greater => a,
        std::cmp::Ordering::Less => b,
        std::cmp::Ordering::Equal => {
            if speak_pos(a, dealer) < speak_pos(b, dealer) { a } else { b }
        }
    }
}

/// Camp qui peut tenir un contrat à cet atout. Les points cartes sont à somme
/// constante, donc un seul des deux le peut — c'est ce qui donne le bénéfice d'un
/// `[u8;8]` au prix d'un `[u8;4]`.
#[inline]
fn dd_side(ns_pts: u8) -> u8 {
    if ns_pts > 81 { 0 } else { 1 }
}

#[derive(Default, Clone)]
struct Stats {
    deals: u64,
    voids: u64,
    /// Donnes où `dd_pts[t*] == 81` — partage exact, le camp preneur n'est pas désigné.
    dd_ties: u64,

    // --- enchères réelles ---
    real_pos: [u64; 4],
    real_first_bid_pos: [u64; 4],
    /// Longueur du préfixe d'enchère en jetons (borné à 24 par le tokeniseur).
    real_len: [u64; 25],
    /// Nombre d'annonces (hors passe / coinche / surcoinche).
    real_nbids: [u64; 12],
    real_contested: u64,
    real_coinched: u64,
    /// Valeur du contrat : indices 0..8 = 80..160, 9 = capot.
    real_value: [u64; 10],
    /// Le camp preneur est-il celui que `dd_pts` désigne à l'atout contracté ?
    real_side_is_dd: u64,
    /// Le siège preneur est-il l'argmax `evaluate_for_trump` de son propre camp ?
    real_seat_is_argmax: u64,

    // --- construction ---
    /// Position du preneur construit, à l'atout **réellement** contracté (apparié).
    cons_pos_at_real: [u64; 4],
    cons_seat_agree: u64,
    cons_side_agree: u64,
    /// Position du preneur construit sur les 4 atouts — ce que le générateur produira
    /// vraiment, puisqu'il étiquette les 4 cases de chaque donne.
    cons_pos_all: [u64; 4],

    /// Échelle : `evaluate_for_trump` du preneur à l'atout contracté → valeur annoncée.
    /// `[score 0..=40][valeur 0..9]`.
    ladder: Vec<[u64; 10]>,
}

impl Stats {
    fn new() -> Self {
        Stats { ladder: vec![[0u64; 10]; 41], ..Default::default() }
    }

    fn merge(&mut self, o: &Stats) {
        self.deals += o.deals;
        self.voids += o.voids;
        self.dd_ties += o.dd_ties;
        self.real_contested += o.real_contested;
        self.real_coinched += o.real_coinched;
        self.real_side_is_dd += o.real_side_is_dd;
        self.real_seat_is_argmax += o.real_seat_is_argmax;
        self.cons_seat_agree += o.cons_seat_agree;
        self.cons_side_agree += o.cons_side_agree;
        for i in 0..4 {
            self.real_pos[i] += o.real_pos[i];
            self.real_first_bid_pos[i] += o.real_first_bid_pos[i];
            self.cons_pos_at_real[i] += o.cons_pos_at_real[i];
            self.cons_pos_all[i] += o.cons_pos_all[i];
        }
        for i in 0..25 { self.real_len[i] += o.real_len[i]; }
        for i in 0..12 { self.real_nbids[i] += o.real_nbids[i]; }
        for i in 0..10 { self.real_value[i] += o.real_value[i]; }
        for s in 0..self.ladder.len() {
            for v in 0..10 { self.ladder[s][v] += o.ladder[s][v]; }
        }
    }
}

/// Valeur du contrat → index d'histogramme (0..8 = 80..160, 9 = capot).
#[inline]
fn value_idx(v: u8) -> usize {
    if v == 25 { 9 } else { (v - 8) as usize }
}

/// Ce qu'une enchère candidate produit, mesuré aux **mêmes** cinq statistiques de forme
/// que les enchères réelles. Une case `(donne, atout)` par observation.
#[derive(Default, Clone)]
struct VariantStats {
    cells: u64,
    /// Cases où tout le monde passe : la variante ne rend **aucune** enchère, donc
    /// aucune étiquette. C'est le coût propre du masquage, et il n'a pas d'équivalent
    /// dans la construction du plan, qui annonce toujours.
    voids: u64,
    pos: [u64; 4],
    first_bid_pos: [u64; 4],
    len: [u64; 25],
    nbids: [u64; 12],
    contested: u64,
    coinched: u64,
    value: [u64; 10],
    /// Le camp preneur est-il celui que `dd_pts` désigne à cet atout ?
    side_is_dd: u64,
    /// **Témoin, variante non masquée seulement** : l'enchère rejouée est-elle
    /// *identique* à celle du corpus ? Le modèle est déterministe et les donnes sont
    /// les mêmes, donc la réponse doit être « toujours ». Tout écart est un défaut du
    /// pilote (suivi d'historique, score, pénalité), pas une propriété de l'enchère —
    /// et sans ce contrôle les chiffres de la variante masquée ne valent rien.
    exact_match: u64,
    /// Rang de l'atout choisi parmi les 4, **vu du camp preneur** (0 = le meilleur).
    /// C'est la lecture de `bid_contract_ranks` ; le rang « de la donne » est un piège.
    rank: [u64; 4],
}

impl VariantStats {
    fn merge(&mut self, o: &VariantStats) {
        self.cells += o.cells;
        self.voids += o.voids;
        self.contested += o.contested;
        self.coinched += o.coinched;
        self.side_is_dd += o.side_is_dd;
        self.exact_match += o.exact_match;
        for i in 0..4 {
            self.pos[i] += o.pos[i];
            self.first_bid_pos[i] += o.first_bid_pos[i];
            self.rank[i] += o.rank[i];
        }
        for i in 0..25 { self.len[i] += o.len[i]; }
        for i in 0..12 { self.nbids[i] += o.nbids[i]; }
        for i in 0..10 { self.value[i] += o.value[i]; }
    }
}

/// Masque légal restreint à la couleur `t` : `PASS`, `COINCHE`, `SURCOINCHE` et les
/// seuls paliers de cet atout.
///
/// C'est le cœur de la variante. Laisser passer les autres couleurs rendrait
/// simplement v6 à lui-même et l'atout ne serait plus celui qu'on veut étiqueter ;
/// retirer `PASS` rendrait l'enchère infinie et surtout **fabriquerait** la
/// contestation qu'on cherche justement à mesurer.
fn suit_mask(t: u8) -> u64 {
    let mut m = (1u64 << BID_PASS) | (1u64 << BID_COINCHE) | (1u64 << BID_SURCOINCHE);
    for v in 0..9u8 {
        m |= 1u64 << bidding::encode_bid(8 + v, t);
    }
    m |= 1u64 << bidding::encode_bid(25, t); // capot
    m
}

/// Points cartes du camp `team` sous l'atout dont `dd_pts` vaut `ns`.
#[inline]
fn side_pts(ns: u8, team: u8) -> i16 {
    let ns = ns as i16;
    let ew = if ns == 252 { 0 } else if ns == 0 { 252 } else { 162 - ns };
    if team == 0 { ns } else { ew }
}

/// Déroule une enchère complète pilotée par v6 sous le masque `mask`, et la mesure.
///
/// `mask = u64::MAX` rend l'enchère **non masquée**, donc la vraie politique : c'est le
/// témoin. `reference` est la suite d'actions du corpus quand on veut vérifier qu'on la
/// reproduit.
#[allow(clippy::too_many_arguments)]
fn drive_auction(
    hands: &[u32; 4],
    dealer: u8,
    mask: u64,
    dd: &[u8; 4],
    reference: Option<&[u8]>,
    net: &mut BidNet,
    obs: &mut Vec<f32>,
    st: &mut VariantStats,
) {
    st.cells += 1;
    let mut g = GameState::new(dealer, *hands);
    let mut tr = EnvTracking::new();
    tr.dealer = dealer;
    let dim = net.obs_dim();
    let mut mine: Vec<u8> = Vec::with_capacity(12);

    let mut len = 0usize;
    let mut nbids = 0usize;
    let mut first_bid_pos: Option<usize> = None;
    let mut bid_by_team = [false; 2];
    let mut coinched = false;

    while g.phase == Phase::Bidding {
        let seat = g.current_player();
        let legal = g.legal_actions() & mask;
        // `PASS` est toujours dans les deux, donc l'intersection n'est jamais vide.
        debug_assert!(legal & (1 << BID_PASS) != 0);

        // Donnes isolées : 0-0. Même contexte de score que le corpus de référence et
        // que la génération de la couche.
        obs.clear();
        obs.resize(dim, 0.0);
        bid_obs::write_bid_observation_dim(obs, 0, &g, &tr.bid_history, 0, 0, dim);
        let action = net.best_action_fast(obs, legal);

        match action {
            BID_PASS => {}
            BID_COINCHE | BID_SURCOINCHE => coinched = true,
            _ => {
                nbids += 1;
                bid_by_team[GameState::player_team(seat) as usize] = true;
                first_bid_pos.get_or_insert_with(|| speak_pos(seat, dealer));
            }
        }
        tr.track_action(&g, action);
        g.step(action);
        mine.push(action);
        len += 1;
        if len > 40 {
            break; // garde-fou : une enchère ne peut pas durer, mais ne pas boucler
        }
    }

    if let Some(r) = reference {
        // Le corpus contient l'enchère PUIS les 32 cartes ; on ne compare que le préfixe.
        if r.len() >= mine.len() && r[..mine.len()] == mine[..] {
            st.exact_match += 1;
        }
    }

    if g.phase != Phase::Playing {
        st.voids += 1;
        return;
    }

    let taker = g.last_bidder;
    st.pos[speak_pos(taker, dealer)] += 1;
    if let Some(p) = first_bid_pos {
        st.first_bid_pos[p] += 1;
    }
    st.len[len.min(24)] += 1;
    st.nbids[nbids.min(11)] += 1;
    if bid_by_team[0] && bid_by_team[1] {
        st.contested += 1;
    }
    if coinched {
        st.coinched += 1;
    }
    st.value[value_idx(g.contract.value)] += 1;
    let t = g.contract.trump;
    if dd_side(dd[t as usize]) == g.contract.team {
        st.side_is_dd += 1;
    }

    // Rang de l'atout choisi parmi les 4, du point de vue du camp qui l'a pris.
    let own = side_pts(dd[t as usize], g.contract.team);
    let better = (0..4usize)
        .filter(|&i| i != t as usize && side_pts(dd[i], g.contract.team) > own)
        .count();
    st.rank[better.min(3)] += 1;
}

fn process(
    r: &GameReplay,
    tt: &mut solver::TtBuf,
    st: &mut Stats,
    v6: Option<(&mut BidNet, &mut Vec<f32>, &mut VariantStats, &mut VariantStats)>,
) {
    st.deals += 1;

    // --- rejouer l'enchère ---
    let mut g = GameState::new(r.dealer, r.hands);
    let mut len = 0usize;
    let mut nbids = 0usize;
    let mut first_bid_pos: Option<usize> = None;
    let mut bid_by_team = [false; 2];
    let mut coinched = false;

    for &a in &r.actions {
        if g.phase != Phase::Bidding {
            break;
        }
        let seat = g.current_player;
        match a {
            BID_PASS => {}
            BID_COINCHE | BID_SURCOINCHE => coinched = true,
            _ => {
                nbids += 1;
                bid_by_team[GameState::player_team(seat) as usize] = true;
                first_bid_pos.get_or_insert_with(|| speak_pos(seat, r.dealer));
            }
        }
        g.step(a);
        len += 1;
    }

    if g.phase == Phase::Done {
        st.voids += 1; // 4 passes — aucun contrat, rien à comparer
        return;
    }

    let taker = g.last_bidder;
    let trump = g.contract.trump;
    let side = g.contract.team;

    st.real_pos[speak_pos(taker, r.dealer)] += 1;
    if let Some(p) = first_bid_pos {
        st.real_first_bid_pos[p] += 1;
    }
    st.real_len[len.min(24)] += 1;
    st.real_nbids[nbids.min(11)] += 1;
    if bid_by_team[0] && bid_by_team[1] {
        st.real_contested += 1;
    }
    if coinched {
        st.real_coinched += 1;
    }
    st.real_value[value_idx(g.contract.value)] += 1;

    // --- les 4 atouts, résolus en DD ---
    let mut dd = [0u8; 4];
    for (t, slot) in dd.iter_mut().enumerate() {
        *slot = solver::solve_for_trump_reuse_tt(r.hands, r.dealer, t as u8, tt)[0];
    }
    let ns = dd[trump as usize];
    if ns == 81 {
        st.dd_ties += 1;
    }
    let cside = dd_side(ns);
    if cside == side {
        st.real_side_is_dd += 1;
    }

    let argmax_own = constructed_seat(&r.hands, r.dealer, trump, side);
    if argmax_own == taker {
        st.real_seat_is_argmax += 1;
    }

    let cseat = constructed_seat(&r.hands, r.dealer, trump, cside);
    st.cons_pos_at_real[speak_pos(cseat, r.dealer)] += 1;
    if cseat == taker {
        st.cons_seat_agree += 1;
    }
    if cside == side {
        st.cons_side_agree += 1;
    }

    // --- ce que la construction produit sur les 4 cases de cette donne ---
    let mut v6 = v6;
    for t in 0..4u8 {
        let s = dd_side(dd[t as usize]);
        let seat = constructed_seat(&r.hands, r.dealer, t, s);
        st.cons_pos_all[speak_pos(seat, r.dealer)] += 1;

        if let Some((ref mut net, ref mut obs, ref mut vst, _)) = v6 {
            drive_auction(&r.hands, r.dealer, suit_mask(t), &dd, None, net, obs, vst);
        }
    }

    // --- le témoin : la même mécanique sans masque doit reproduire le corpus ---
    if let Some((ref mut net, ref mut obs, _, ref mut ust)) = v6 {
        drive_auction(&r.hands, r.dealer, u64::MAX, &dd, Some(&r.actions), net, obs, ust);
    }

    // --- échelle valeur ↔ force ---
    let e = evaluate_for_trump(r.hands[taker as usize], Suit::from_u8(trump)) as usize;
    st.ladder[e.min(40)][value_idx(g.contract.value)] += 1;
}

fn pct(x: u64, n: u64) -> f64 {
    if n == 0 { 0.0 } else { 100.0 * x as f64 / n as f64 }
}

fn arr_pct(a: &[u64], n: u64) -> Vec<f64> {
    a.iter().map(|&x| (pct(x, n) * 100.0).round() / 100.0).collect()
}

fn json_nums(v: &[f64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
    format!("[{}]", parts.join(","))
}

fn json_u64(v: &[u64]) -> String {
    let parts: Vec<String> = v.iter().map(|x| format!("{x}")).collect();
    format!("[{}]", parts.join(","))
}

fn main() {
    let mut games = String::from("data/training/isdd_games_v1.bin");
    let mut json_out: Option<String> = None;
    let mut bid_model: Option<String> = None;
    let mut limit = usize::MAX;
    let mut threads = std::thread::available_parallelism().map(|p| p.get()).unwrap_or(4);

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--games" => { i += 1; games = args[i].clone(); }
            "--json" => { i += 1; json_out = Some(args[i].clone()); }
            "--bid-model" => { i += 1; bid_model = Some(args[i].clone()); }
            "--limit" => { i += 1; limit = args[i].parse().expect("limit"); }
            "--threads" => { i += 1; threads = args[i].parse().expect("threads"); }
            "--help" | "-h" => {
                eprintln!("bench_taker_position: enchère synthétique vs enchères réelles");
                eprintln!("  --games <path>      corpus COLVGM01/02 (défaut: data/training/isdd_games_v1.bin)");
                eprintln!("  --json <path>       écrit les histogrammes");
                eprintln!("  --bid-model <path>  mesure en plus la variante « v6 masqué sur la couleur »");
                eprintln!("  --limit N           n'examine que les N premières donnes");
                eprintln!("  --threads N");
                eprintln!("\n  ⚠️ ne pas rediriger la sortie dans `head` : le SIGPIPE tue le");
                eprintln!("     processus avant l'écriture du JSON.");
                std::process::exit(0);
            }
            other => { eprintln!("argument inconnu: {other}"); std::process::exit(1); }
        }
        i += 1;
    }

    let replays = GameReplay::load_all(&games).expect("lecture du corpus");
    let n = replays.len().min(limit);
    eprintln!("bench_taker_position: {n} donnes de {games}, {threads} threads");

    // Un `BidNet` par thread : `evaluate` prend `&mut self` (buffers internes), donc il
    // n'est pas partageable. Charger le fichier N fois coûte 2,4 Mo × N, négligeable.
    if let Some(ref p) = bid_model {
        let net = BidNet::load(p).expect("chargement du modèle d'enchère");
        eprintln!("  variante masquée : {p} (obs {})", net.obs_dim());
    }

    let next = AtomicUsize::new(0);
    let done = AtomicUsize::new(0);
    let start = std::time::Instant::now();
    let mut total = Stats::new();
    let mut total_masked = VariantStats::default();
    let mut total_free = VariantStats::default();

    let parts: Vec<(Stats, VariantStats, VariantStats)> = std::thread::scope(|s| {
        let handles: Vec<_> = (0..threads)
            .map(|_| {
                let next = &next;
                let done = &done;
                let replays = &replays;
                let bid_model = &bid_model;
                s.spawn(move || {
                    let mut tt = solver::new_tt_buffer();
                    let mut st = Stats::new();
                    let mut vst = VariantStats::default();
                    let mut ust = VariantStats::default();
                    let mut net = bid_model
                        .as_ref()
                        .map(|p| BidNet::load(p).expect("chargement du modèle d'enchère"));
                    let mut obs: Vec<f32> = Vec::new();
                    loop {
                        // Un lot par prise : une donne coûte 4 solves, la contention
                        // sur le compteur serait visible à l'unité.
                        let lo = next.fetch_add(64, Ordering::Relaxed);
                        if lo >= n {
                            break;
                        }
                        let hi = (lo + 64).min(n);
                        for r in &replays[lo..hi] {
                            let m = net.as_mut().map(|nt| (nt, &mut obs, &mut vst, &mut ust));
                            process(r, &mut tt, &mut st, m);
                        }
                        let d = done.fetch_add(hi - lo, Ordering::Relaxed) + (hi - lo);
                        if d % 5_000 < 64 {
                            let el = start.elapsed().as_secs_f64();
                            eprintln!("  {d}/{n}  {:.0} donnes/s", d as f64 / el);
                        }
                    }
                    (st, vst, ust)
                })
            })
            .collect();
        handles.into_iter().map(|h| h.join().unwrap()).collect()
    });
    for (p, v, u) in &parts {
        total.merge(p);
        total_masked.merge(v);
        total_free.merge(u);
    }

    let st = total;
    let c = st.deals - st.voids; // contrats
    let all = c * 4; // cases (donne, atout)

    println!("\n=== {} donnes, {} contrats, {} donnes passées ===", st.deals, c, st.voids);
    println!("  dd_pts == 81 (partage exact, camp non désigné) : {} ({:.3} %)",
             st.dd_ties, pct(st.dd_ties, c));

    println!("\n--- position du preneur dans l'ordre de parole (0 = premier, 3 = donneur) ---");
    let rp = arr_pct(&st.real_pos, c);
    let cp = arr_pct(&st.cons_pos_at_real, c);
    let ca = arr_pct(&st.cons_pos_all, all);
    println!("  {:<28} {:>7} {:>7} {:>7} {:>7}", "", "pos 0", "pos 1", "pos 2", "pos 3");
    println!("  {:<28} {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}%",
             "enchères réelles (v6)", rp[0], rp[1], rp[2], rp[3]);
    println!("  {:<28} {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}%",
             "construction, atout réel", cp[0], cp[1], cp[2], cp[3]);
    println!("  {:<28} {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}%",
             "construction, 4 atouts", ca[0], ca[1], ca[2], ca[3]);
    let tvd: f64 = (0..4).map(|i| (rp[i] - cp[i]).abs()).sum::<f64>() / 2.0;
    println!("  distance en variation totale (réel vs construit, atout réel) : {tvd:.2} pp");

    println!("\n--- accord apparié, sur l'atout réellement contracté ---");
    println!("  camp  : {:.2} %", pct(st.cons_side_agree, c));
    println!("  siège : {:.2} %", pct(st.cons_seat_agree, c));
    println!("  (le preneur réel est l'argmax evaluate_for_trump de SON camp : {:.2} %)",
             pct(st.real_seat_is_argmax, c));

    println!("\n--- forme du préfixe (la construction produit toujours k·P + 1 annonce + 3·P) ---");
    let mean_len: f64 = st.real_len.iter().enumerate()
        .map(|(i, &x)| i as f64 * x as f64).sum::<f64>() / c.max(1) as f64;
    println!("  longueur moyenne : {mean_len:.2} jetons");
    print!("  longueur :");
    for (i, &x) in st.real_len.iter().enumerate() {
        if x > 0 { print!("  {i}:{:.1}%", pct(x, c)); }
    }
    println!();
    print!("  annonces :");
    for (i, &x) in st.real_nbids.iter().enumerate() {
        if x > 0 { print!("  {i}:{:.1}%", pct(x, c)); }
    }
    println!();
    println!("  enchère contestée (les deux camps annoncent) : {:.2} %", pct(st.real_contested, c));
    println!("  coinchée : {:.2} %", pct(st.real_coinched, c));
    print!("  première annonce, position :");
    for (i, p) in arr_pct(&st.real_first_bid_pos, c).iter().enumerate() {
        print!("  {i}:{p:.1}%");
    }
    println!();

    println!("\n--- valeur du contrat ---");
    let labels = ["80", "90", "100", "110", "120", "130", "140", "150", "160", "capot"];
    for (i, &x) in st.real_value.iter().enumerate() {
        if x > 0 { println!("  {:>5} : {:>6.2} %", labels[i], pct(x, c)); }
    }

    println!("\n--- échelle : evaluate_for_trump du preneur → valeur annoncée ---");
    println!("  {:>5} {:>8} {:>9}  {}", "score", "n", "moyenne", "répartition");
    for (e, row) in st.ladder.iter().enumerate() {
        let tot: u64 = row.iter().sum();
        if tot < 30 { continue; }
        let non_capot: u64 = row[..9].iter().sum();
        let mean = if non_capot == 0 { f64::NAN } else {
            row[..9].iter().enumerate()
                .map(|(v, &x)| (80.0 + 10.0 * v as f64) * x as f64).sum::<f64>() / non_capot as f64
        };
        let mut bars = String::new();
        for (v, &x) in row.iter().enumerate() {
            if x * 20 >= tot { bars.push_str(&format!(" {}:{:.0}%", labels[v], pct(x, tot))); }
        }
        println!("  {e:>5} {tot:>8} {mean:>9.1} {bars}");
    }

    // --- la variante candidate, en face des mêmes cibles ---
    let mut vjson = String::from("null");
    let mut ujson = String::from("null");
    if bid_model.is_some() {
        // Le témoin d'abord : sans lui, rien de ce qui suit n'est interprétable.
        let u = &total_free;
        let ua = u.cells - u.voids;
        let umean: f64 = u.len.iter().enumerate()
            .map(|(i, &x)| i as f64 * x as f64).sum::<f64>() / ua.max(1) as f64;
        println!("\n=== TÉMOIN : la même mécanique SANS masque, rejouée sur les mêmes donnes ===");
        println!("  enchère identique à celle du corpus : {:.2} % ({} / {})",
                 pct(u.exact_match, u.cells), u.exact_match, u.cells);
        println!("  {:<44} {:>10} {:>10}", "", "corpus", "rejeu");
        for (lbl, r, m) in [
            ("première annonce par le premier parleur",
             pct(st.real_first_bid_pos[0], c), pct(u.first_bid_pos[0], ua)),
            ("une seule annonce", pct(st.real_nbids[1], c), pct(u.nbids[1], ua)),
            ("contestée", pct(st.real_contested, c), pct(u.contested, ua)),
            ("coinchée", pct(st.real_coinched, c), pct(u.coinched, ua)),
            ("longueur du préfixe (jetons)", mean_len, umean),
        ] {
            println!("  {lbl:<44} {r:>9.2}  {m:>9.2}");
        }
        println!("  rang de l'atout choisi, vu du camp preneur : {}",
                 arr_pct(&u.rank, ua).iter().enumerate()
                     .map(|(i, p)| format!("rang {i}:{p:.1}%"))
                     .collect::<Vec<_>>().join("  "));
        ujson = format!(
            "{{\"cells\":{},\"exact_match_pct\":{:.3},\"first_bid_pos0_pct\":{:.3},\
             \"single_bid_pct\":{:.3},\"contested_pct\":{:.3},\"coinched_pct\":{:.3},\
             \"mean_prefix_len\":{:.3},\"rank_pct\":{}}}",
            u.cells, pct(u.exact_match, u.cells), pct(u.first_bid_pos[0], ua),
            pct(u.nbids[1], ua), pct(u.contested, ua), pct(u.coinched, ua), umean,
            json_nums(&arr_pct(&u.rank, ua)),
        );

        let v = &total_masked;
        let a = v.cells - v.voids; // cases où la variante rend bien une enchère
        println!("\n=== variante « v6 masqué sur la couleur » — {} cases, {} sans enchère ===",
                 v.cells, v.voids);
        println!("  ⚠️ {:.2} % des cases n'ont AUCUNE enchère : masqué sur cette couleur, v6",
                 pct(v.voids, v.cells));
        println!("     passe aux quatre sièges. Ces cases-là restent à étiqueter autrement.");

        let vmean: f64 = v.len.iter().enumerate()
            .map(|(i, &x)| i as f64 * x as f64).sum::<f64>() / a.max(1) as f64;
        println!("\n  {:<44} {:>10} {:>10}", "", "réel", "masqué");
        let rows: [(&str, f64, f64); 5] = [
            ("première annonce par le premier parleur",
             pct(st.real_first_bid_pos[0], c), pct(v.first_bid_pos[0], a)),
            ("une seule annonce", pct(st.real_nbids[1], c), pct(v.nbids[1], a)),
            ("contestée", pct(st.real_contested, c), pct(v.contested, a)),
            ("coinchée", pct(st.real_coinched, c), pct(v.coinched, a)),
            ("longueur du préfixe (jetons)", mean_len, vmean),
        ];
        for (lbl, r, m) in rows {
            println!("  {lbl:<44} {r:>9.2}  {m:>9.2}");
        }
        println!("\n  position du preneur :");
        let vp = arr_pct(&v.pos, a);
        println!("    {:<24} {:>7} {:>7} {:>7} {:>7}", "", "pos 0", "pos 1", "pos 2", "pos 3");
        println!("    {:<24} {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}%",
                 "réel", rp[0], rp[1], rp[2], rp[3]);
        println!("    {:<24} {:>6.2}% {:>6.2}% {:>6.2}% {:>6.2}%",
                 "masqué", vp[0], vp[1], vp[2], vp[3]);
        let vtvd: f64 = (0..4).map(|i| (rp[i] - vp[i]).abs()).sum::<f64>() / 2.0;
        println!("    distance en variation totale : {vtvd:.2} pp  (construction : {tvd:.2} pp)");
        println!("  camp preneur = celui de dd_pts : {:.2} %", pct(v.side_is_dd, a));
        print!("  valeur :");
        for (i, &x) in v.value.iter().enumerate() {
            if x > 0 { print!("  {}:{:.1}%", labels[i], pct(x, a)); }
        }
        println!();

        vjson = format!(
            "{{\"cells\":{},\"voids\":{},\"void_pct\":{:.3},\"pos_pct\":{},\"tvd_pp\":{:.3},\
             \"first_bid_pos0_pct\":{:.3},\"single_bid_pct\":{:.3},\"contested_pct\":{:.3},\
             \"coinched_pct\":{:.3},\"mean_prefix_len\":{:.3},\"side_is_dd_pct\":{:.3},\
             \"nbids\":{},\"len\":{},\"value\":{}}}",
            v.cells, v.voids, pct(v.voids, v.cells), json_nums(&vp), vtvd,
            pct(v.first_bid_pos[0], a), pct(v.nbids[1], a), pct(v.contested, a),
            pct(v.coinched, a), vmean, pct(v.side_is_dd, a),
            json_u64(&v.nbids), json_u64(&v.len), json_u64(&v.value),
        );
    }

    if let Some(path) = json_out {
        let ladder: Vec<String> = st.ladder.iter().enumerate()
            .filter(|(_, r)| r.iter().sum::<u64>() > 0)
            .map(|(e, r)| format!("{{\"score\":{e},\"hist\":{}}}", json_u64(r)))
            .collect();
        let body = format!(
            "{{\n \"deals\":{},\n \"contracts\":{},\n \"voids\":{},\n \"dd_ties\":{},\n \
             \"real_pos_pct\":{},\n \"cons_pos_at_real_pct\":{},\n \"cons_pos_all_pct\":{},\n \
             \"tvd_pp\":{:.3},\n \"side_agree_pct\":{:.3},\n \"seat_agree_pct\":{:.3},\n \
             \"real_seat_is_argmax_pct\":{:.3},\n \"mean_prefix_len\":{:.3},\n \
             \"real_len\":{},\n \"real_nbids\":{},\n \"contested_pct\":{:.3},\n \
             \"coinched_pct\":{:.3},\n \"real_first_bid_pos_pct\":{},\n \
             \"real_value\":{},\n \"free_v6\":{},\n \"masked_v6\":{},\n \"ladder\":[{}]\n}}\n",
            st.deals, c, st.voids, st.dd_ties,
            json_nums(&rp), json_nums(&cp), json_nums(&ca),
            tvd, pct(st.cons_side_agree, c), pct(st.cons_seat_agree, c),
            pct(st.real_seat_is_argmax, c), mean_len,
            json_u64(&st.real_len), json_u64(&st.real_nbids),
            pct(st.real_contested, c), pct(st.real_coinched, c),
            json_nums(&arr_pct(&st.real_first_bid_pos, c)),
            json_u64(&st.real_value), ujson, vjson, ladder.join(","),
        );
        std::fs::write(&path, body).expect("écriture json");
        eprintln!("[json] {path}");
    }

    eprintln!("terminé en {:.1}s", start.elapsed().as_secs_f64());
}
