//! Information Set Double-Dummy (IS-DD) agent.
//!
//! Combines the exact alpha-beta DD solver with determinization (like IS-MCTS,
//! but replacing approximate MCTS rollouts with exact DD solves). Each
//! determinized world gives a provably optimal answer, so fewer samples are
//! needed compared to IS-MCTS.
//!
//! Two modes:
//! - **Naive** (no beliefs): uniform determinization, like `naive_ismcts` but with DD.
//! - **Smart** (with beliefs): belief-weighted determinization, like `smart_ismcts` but with DD.
//!
//! Score-based aggregation: each DD solve returns exact NS points per card,
//! so we sum scores across determinizations rather than voting.

use std::time::{Duration, Instant};

use rand::Rng;

use crate::belief_net::BeliefNet;
use crate::belief_obs::{self, BELIEF_OBS_DIM, BELIEF_OBS_DIM_V2, BELIEF_OBS_DIM_V3};
use crate::bid_eval::BidFunction;
use crate::card::card_count;
use crate::card_beliefs::CardBeliefs;
use crate::determinize::{determinize_greedy, determinize_weighted};
use crate::dmc_net::DmcNet;
use crate::dmc_obs::{self, EnvTracking, OBS_DIM_TR};
use crate::worlds::{World, WorldSource};
use crate::scoring::{deal_score_from_card_points, CAPOT_PTS, TOTAL_PTS};
use crate::solver::{new_tt_buffer, solve_with_scores};
use crate::state::{GameState, Phase};

/// Ce qu'une recherche IS-DD maximise en agrégeant ses mondes.
///
/// Le solveur DD rend des **points cartes**, mais ce ne sont pas eux qui
/// décident une donne. Écart N-S − E-O en fonction des points cartes `x` du
/// preneur, contrat de valeur `V` (`engine/scoring.rs`) :
///
/// ```text
///   x < V   :  -(162 + V)      CONSTANT, pente 0
///   x >= V  :  2x + V - 162    pente 2
///   saut en x = V : 4V
/// ```
///
/// Maximiser `E[x]` est donc une approximation linéaire d'une marche. Elle est
/// **correcte au-dessus du seuil** (la pente y vaut bien 2), et fausse en
/// dessous — une fois la chute acquise un point carte de plus vaut exactement
/// zéro, et le solveur continue pourtant à se battre pour lui : c'est de là que
/// viennent les coups qui paraissent arbitraires en fin de donne perdue.
///
/// # Pourquoi c'est l'agrégation qui décide, et pas le solveur
///
/// L'écart de score est une fonction **monotone non décroissante** des points
/// cartes du preneur. Dans un monde déterminisé, où tout est décidé, les deux
/// objectifs classent donc les coups à l'identique et le camp qui minimise l'un
/// minimise l'autre : le solveur DD n'a rien à savoir du contrat, et
/// `solve_with_scores` reste inchangé.
///
/// L'écart n'apparaît qu'à la moyenne, parce que `E[f(x)] ≠ f(E[x])` dès qu'il
/// y a une marche entre les deux. Trois mondes à 90/70/70 sous un contrat à 80
/// donnent une espérance de 76,7 — « chute » — alors que la bonne lecture est
/// un tiers de contrat réussi contre deux tiers de chute. C'est exactement le
/// coup à tenter quand la seule ligne gagnante est improbable, et la raison
/// pour laquelle on ne rattrape pas ça après coup sur une moyenne de points
/// cartes. D'où [`IsDdSearch::world_value`], appliqué **monde par monde** avant
/// la somme.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlayObjective {
    /// Espérance de points cartes N-S. Le comportement historique, conservé
    /// pour l'A/B (`arena/bots/web_dede_cardpts.toml`) et pour les mesures qui
    /// veulent une échelle 0-252.
    CardPoints,
    /// Espérance d'écart de **score de donne** N-S − E-O, contrat compris :
    /// réussite ou chute, valeur du contrat, contré/surcontré, capot, dix de
    /// der et belote. **Le défaut.**
    ///
    /// La belote n'est pas un simple bonus de 20 points : `scoring.rs` la
    /// compte dans `taker_total` pour décider de la réussite, donc elle
    /// **déplace le seuil du contrat**. C'est la raison pour laquelle
    /// [`IsDdSearch::world_belote_for`] la recalcule dans chaque monde au lieu
    /// de lire `state.belote`, qui ne compte que ce qui a déjà été joué.
    DealScore,
}

// `total_card_points` et `world_belote` vivaient ici, en privé. Elles sont
// remontées dans `scoring.rs` le 2026-08-05 sous les noms
// [`scoring::total_card_points`] et [`scoring::final_belote`] : les pages
// d'analyse ont le même besoin de convertir une valeur DD en score de donne, et
// le barème n'a le droit d'exister qu'à un seul endroit.
use crate::scoring::{final_belote, total_card_points};

/// Configuration for IS-DD search.
///
/// **Hard constraints** (voids, trump ceiling, played cards) are facts, not beliefs:
/// they are always applied unconditionally and not exposed as a flag.
///
/// **Soft beliefs** (heuristic soft inference, NN beliefs) are **off by default** —
/// they introduce probabilistic adjustments that may help or hurt depending on
/// the opponents and the play model.
pub struct IsDdConfig {
    /// Number of determinized worlds to sample (default 20).
    pub determinizations: u32,
    /// Mondes par décision **selon les cartes restantes** (index 1..=8), en
    /// mode compte. `None` = compte fixe, le comportement historique.
    ///
    /// Un compte plat dépense pareil partout, alors que le besoin ne l'est pas :
    /// `isdd_dets_by_stage` place tout le regret au-dessus de 0,10 point DD à
    /// 8-6 cartes restantes, et **zéro en dessous de 3 cartes, à n'importe quel
    /// budget**. Un monde tiré en finale n'achète donc rien.
    ///
    /// ⚠️ **Un total constant ne fait pas un coût constant** — c'est l'erreur
    /// qu'il faut ne pas refaire ici. Un monde à 8 cartes restantes demande au
    /// sampler 24 cartes cachées, soit 48 pas de décodage ; un monde à 2 cartes
    /// en demande 6, soit 12 pas. Le premier coûte donc ~4× le second sur le
    /// GPU, et bien davantage encore au solveur DD. Mesuré : le calendrier
    /// `60,60,60,30,30,20,20`, qui dépense exactement les mêmes 280 mondes
    /// qu'un plat à 40, tourne à **1,90 et 1,35 donnes/s contre 2,15 et 2,34** —
    /// il est plus *lent*, pas gratuit.
    ///
    /// Le sens qui paie est donc le sens **décroissant**, et il paie deux fois :
    /// `40,40,40,30,20,15,15` garde 40 mondes là où vit le regret, coupe là où
    /// il est nul, et rend **2,50 contre 2,02 donnes/s — 1,24×**.
    ///
    /// Change l'agent, donc se déclare : `[play] dets_schedule` ou
    /// `--dets-schedule`.
    pub det_schedule: Option<[u32; 9]>,
    /// Whether to use soft (probabilistic) heuristic inference from play (dominance,
    /// "ne pisse pas", etc.) in addition to hard constraints. **Off by default.**
    pub use_soft_inference: bool,
    /// Optional time limit in milliseconds (overrides `determinizations` count).
    pub time_limit_ms: Option<u32>,
    /// Which bid function to use during bidding phase.
    pub bid_function: BidFunction,
    /// If true and a BeliefNet is loaded, use NN soft beliefs (still combined with
    /// hard constraints, which are always applied). **Off by default.**
    pub use_nn_beliefs: bool,
    /// Play dominance inference factor for CardBeliefs.
    /// When a player follows suit without playing the highest, reduce weight for
    /// higher unknown cards by this factor. 1.0 = off, 0.3 = aggressive. Default 1.0.
    pub dominance_factor: f32,
    /// If true (default), skip search when only 1 legal action or position is fully resolved.
    pub early_termination: bool,
    /// How many worlds to request per [`WorldSource`] refill when running
    /// under a time budget (in count mode the whole remaining budget is asked
    /// for at once). One refill is one GPU round trip for the sidecar, so this
    /// trades latency granularity against overhead. Default 128.
    pub world_batch: usize,
    /// Fallback pool: when no [`WorldSource`] is attached (or it runs dry),
    /// the fraction of worlds drawn with belief weights rather than
    /// constraint-uniform. Only has an effect when a belief source is active.
    /// Default 1.0.
    pub belief_frac: f32,
    /// Ce que la recherche maximise. Défaut [`PlayObjective::DealScore`] depuis
    /// le 2026-08-03 : c'est le score de donne qui décide une partie, pas les
    /// points cartes, et l'écart entre les deux est une marche (voir
    /// [`PlayObjective`]).
    ///
    /// ⚠️ L'échelle de `card_scores` en dépend : écarts de score de donne
    /// (±500, contrat compris) sous `DealScore`, points cartes (0-252) sous
    /// `CardPoints`. Tout consommateur qui affiche ces valeurs doit lire
    /// l'objectif — côté web c'est le champ `score_scale` du blob de stats.
    pub objective: PlayObjective,
    /// Plafond de mondes **résolus** par décision, sous échéance uniquement.
    ///
    /// Sous budget de temps la boucle ne sortait que sur l'échéance, donc elle
    /// consommait tout le temps disponible même quand la réponse avait cessé de
    /// bouger. Mesuré (`isdd_dets_by_stage`, 250 positions) : le regret contre
    /// une référence à 2000 mondes est sous 0,10 point DD dès **60** mondes, et
    /// sous 0,03 dès **15** en dessous de cinq cartes restantes — alors que
    /// Dédé en traversait de 256 à 697 selon le stade.
    ///
    /// `None` par défaut : aucun appelant existant ne change de comportement, et
    /// aucune donnée déjà produite n'est périmée. C'est aux specs qui le veulent
    /// de le poser (`[play] max_worlds`). Sans effet en mode compte, où
    /// `determinizations` borne déjà la boucle — c'est ce qui garde
    /// reproductibles le sweep ci-dessus et `enrich_pool_isdd`.
    pub max_worlds: Option<u32>,
    /// Plancher de mondes **résolus** par décision, sous échéance uniquement.
    ///
    /// Le pendant de `max_worlds`, et il répond à une question de politique et
    /// non de performance : **sous pression de calcul, un bot doit-il rendre sa
    /// réponse à l'heure en cherchant moins, ou chercher autant en rendant sa
    /// réponse plus tard ?** Sans plancher c'est la première option, et la
    /// dégradation est invisible — le joueur voit un coup arriver au même rythme,
    /// simplement moins bon. Avec un plancher c'est la seconde : la charge se
    /// paie en latence, qui se voit, au lieu de se payer en force, qui ne se voit
    /// pas. Le GPU de la prod étant partagé (sidecar, llama-server, et toute
    /// génération de données qui passe), le cas n'est pas théorique.
    ///
    /// `None` par défaut : aucun appelant existant ne change de comportement.
    /// `min_worlds == max_worlds` dégénère en mode compte, l'échéance ne servant
    /// alors plus à rien.
    ///
    /// **« Lent » n'est pas « bloqué ».** Le plancher suspend l'échéance, mais
    /// pas le garde-fou de progression : si aucun monde n'est produit pendant
    /// `STUCK_ROUNDS` tours consécutifs *après* l'échéance, la recherche rend ce
    /// qu'elle a. Sans ça, une position dont la déterminisation échoue toujours
    /// — le `continue` de la boucle est explicitement conçu pour réessayer
    /// jusqu'à l'échéance — tournerait sans fin.
    pub min_worlds: Option<u32>,
    /// Credibility importance weighting of worlds in the DD aggregation. Each
    /// world's weight is the product of per-action rank factors — "would the
    /// reference policy replay the observed hidden action holding this world's
    /// hand?" — flattened by this exponent. Judges both phases when the
    /// corresponding net is loaded: the **auction** via the bid net
    /// (`load_cred_bid_net`) and the **play** via the DMC net
    /// (`load_cred_play_net`). 0.0 = off (default); 0.5 = recommended soft
    /// weighting. See [`IsDdSearch::credibility_weight`] for the mechanism.
    pub cred_alpha: f32,
    /// Solve the determinized worlds in parallel (rayon global pool) instead of
    /// sequentially. World *generation* is always sequential (the world source
    /// and RNG are stateful); only the embarrassingly-parallel DD
    /// solves are spread across threads, each with its own transposition table.
    /// Results are identical to the sequential path (DD is deterministic and the
    /// aggregation reduces in a fixed order). Requires the `parallel` cargo
    /// feature — ignored (falls back to sequential) when it is not compiled in.
    /// **Off by default**; the web/PyO3 layer turns it on for per-move latency.
    pub parallel: bool,
}

impl Default for IsDdConfig {
    fn default() -> Self {
        IsDdConfig {
            determinizations: 20,
            det_schedule: None,
            // All soft beliefs OFF by default. Hard constraints are facts, always applied.
            use_soft_inference: false,
            time_limit_ms: None,
            bid_function: BidFunction::ImprovedV2,
            use_nn_beliefs: false,
            dominance_factor: 1.0,
            early_termination: true,
            world_batch: 128,
            belief_frac: 1.0,
            objective: PlayObjective::DealScore,
            max_worlds: None,
            min_worlds: None,
            cred_alpha: 0.0,
            parallel: false,
        }
    }
}

/// Derive V3 temporal features (trick lead suits, trick winners, suit-fail
/// counts relative to observer) from public play tracking. Mirrors the
/// training-time extraction in `game_replay.rs`.
fn derive_v3_temporal(
    state: &GameState,
    tracking: &EnvTracking,
    observer: u8,
) -> (Vec<u8>, Vec<u8>, [[u8; 4]; 3]) {
    use crate::card::{card_suit_u8, EMPTY};
    use crate::trick::trick_winner;

    let completed = tracking.play_order.len() / 4;
    let mut leads = Vec::with_capacity(8);
    let mut winners = Vec::with_capacity(8);
    let mut fails_abs = [[0u8; 4]; 4];

    for t in 0..completed {
        let base = t * 4;
        let c0 = tracking.play_order[base];
        let mut lead_seat = 0u8;
        for p in 0..4u8 {
            if tracking.played_by[p as usize] & (1u32 << c0) != 0 {
                lead_seat = p;
                break;
            }
        }
        let lead_suit = card_suit_u8(c0);
        leads.push(lead_suit);

        let mut trick_cards = [EMPTY; 4];
        for j in 0..4usize {
            let cj = tracking.play_order[base + j];
            trick_cards[(lead_seat as usize + j) % 4] = cj;
        }
        winners.push(trick_winner(&trick_cards, lead_seat, &state.contract));

        for j in 1..4usize {
            let cj = tracking.play_order[base + j];
            if card_suit_u8(cj) != lead_suit {
                let pj = (lead_seat as usize + j) % 4;
                fails_abs[pj][lead_suit as usize] =
                    fails_abs[pj][lead_suit as usize].saturating_add(1);
            }
        }
    }

    let rel_seats = [
        ((observer as usize + 1) % 4),
        ((observer as usize + 2) % 4),
        ((observer as usize + 3) % 4),
    ];
    let mut fail_rel = [[0u8; 4]; 3];
    for (i, &seat) in rel_seats.iter().enumerate() {
        fail_rel[i] = fails_abs[seat];
    }
    (leads, winners, fail_rel)
}

/// Credibility rank factor: how much to trust a world given that the reference
/// policy ranks `better` legal moves strictly above the one actually observed.
/// Argmax (0) is fully credible; a top-3 move is mildly discounted; anything
/// worse is heavily discounted. Shared by the auction and play judges.
#[inline]
fn rank_factor(better: u32) -> f32 {
    match better {
        0 => 1.0,
        1 | 2 => 0.7,
        _ => 0.35,
    }
}

/// Where a determinized world came from. The ensemble policy in
/// [`IsDdSearch::generate_world`] tries these in order.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldOrigin {
    /// Supplied by the caller via [`IsDdSearch::set_injected_worlds`] (e.g. the
    /// playgen GPU sidecar).
    Injected,
    /// Sampled in-process from the playgen transformer.
    Playgen,
    /// Belief-weighted determinization (NN or heuristic weights).
    Belief,
    /// Constraint-uniform determinization — the coverage floor.
    Uniform,
}

/// How many solved worlds came from each source. Reported so a degraded run
/// (e.g. a playgen sidecar that stopped answering) is visible in the stats
/// instead of silently changing the agent's strength.
#[derive(Default, Clone, Copy, Debug, PartialEq, Eq)]
pub struct WorldCounts {
    pub injected: u32,
    pub playgen: u32,
    pub belief: u32,
    pub uniform: u32,
}

impl WorldCounts {
    #[inline]
    fn record(&mut self, origin: WorldOrigin) {
        match origin {
            WorldOrigin::Injected => self.injected += 1,
            WorldOrigin::Playgen => self.playgen += 1,
            WorldOrigin::Belief => self.belief += 1,
            WorldOrigin::Uniform => self.uniform += 1,
        }
    }

    pub fn total(&self) -> u32 {
        self.injected + self.playgen + self.belief + self.uniform
    }
}

/// Per-card aggregated DD result.
pub struct IsDdResult {
    /// Best card for the current player's team.
    pub best_action: u8,
    /// (card, avg_score) pour chaque coup légal. **L'échelle dépend de
    /// [`IsDdConfig::objective`]** : écart de score de donne N-S − E-O (±500,
    /// contrat compris) sous `DealScore` — le défaut — ou points cartes N-S
    /// (0-252) sous `CardPoints`.
    pub card_scores: Vec<(u8, f32)>,
    /// Number of successful determinizations.
    pub determinizations: u32,
    /// Provenance of those determinizations.
    pub worlds: WorldCounts,
    /// Ce que la recherche a demandé à la source, ce qu'elle a reçu, et ce
    /// qu'elle a jeté. Trois chiffres distincts, et l'écart entre eux est le
    /// seul moyen de savoir si le sampler est sous-employé : sous budget de
    /// temps on demande `world_batch` mondes d'un coup, on n'en résout que ce
    /// que la deadline permet, et **le reste part à la poubelle**
    /// (`world_queue.clear()`, obligatoire — un monde échantillonné pour cette
    /// position-ci ne vaut rien à la suivante).
    pub source: SourceUsage,
}

/// Ce qu'une recherche a tiré de sa source de mondes.
#[derive(Debug, Default, Clone, Copy)]
pub struct SourceUsage {
    /// Allers-retours vers la source.
    pub rounds: u32,
    /// Mondes demandés, cumulés sur les allers-retours.
    pub requested: u32,
    /// Mondes effectivement rendus (après `retain_valid`).
    pub delivered: u32,
    /// Mondes reçus mais jamais résolus, jetés en fin de recherche.
    pub discarded: u32,
    /// Temps passé **dans** `WorldSource::worlds`, en microsecondes. C'est la
    /// part du budget qui n'est pas de la recherche : sous échéance, chaque
    /// microseconde ici est une microseconde de moins pour résoudre. La
    /// comparer au temps total de la décision est ce qui distingue « le
    /// solveur sature » de « on attend le GPU ».
    pub source_us: u64,
    /// Cartes restantes en main de l'observateur — 8 à l'entame, 1 au dernier
    /// pli. C'est l'axe qui compte : le coût d'un solve et la richesse de
    /// l'espace des mondes varient de plusieurs ordres de grandeur le long de
    /// la donne, donc un agrégat sur toute la donne ne veut rien dire.
    pub cards_left: u8,
}

/// IS-DD search using belief-weighted determinization + exact DD solving.
///
/// Maintains a `CardBeliefs` model (optional) and a pre-allocated TT buffer.
/// Optionally uses a `BeliefNet` for NN-based card location prediction.
/// API mirrors `SmartIsMctsSearch`.
pub struct IsDdSearch {
    beliefs: Option<CardBeliefs>,
    belief_net: Option<BeliefNet>,
    belief_tracking: Option<EnvTracking>,
    /// Bid net used as an auction-credibility judge (see `cred_alpha`).
    cred_bid_net: Option<crate::bid_net::BidNet>,
    /// DMC net used as a play-credibility judge (see `cred_alpha`). Canonical
    /// (411-dim, `OBS_DIM_TR`) obs only — mirrors `bench_world_cred`.
    cred_play_net: Option<DmcNet>,
    /// Observed auction this deal: (bidder, action) in order.
    auction: Vec<(u8, u8)>,
    /// Observed plays this deal: (player, card) in order. Together with
    /// `auction` this is the full replayable history used by the credibility
    /// judge to reconstruct each hidden decision point.
    plays: Vec<(u8, u8)>,
    /// State at deal start (pre-auction), for credibility replays.
    init_state: Option<GameState>,
    /// Cards played so far per seat (current trick included).
    played_by: [u32; 4],
    /// Worlds pulled from the [`WorldSource`] for the position currently
    /// being searched, not yet solved. Refilled on demand and dropped when the
    /// search ends, so a later search at another position cannot consume
    /// worlds sampled for the previous one.
    world_queue: Vec<World>,
    /// Latence observée d'un aller-retour vers la source, en microsecondes,
    /// moyenne mobile sur toutes les recherches de cette instance.
    ///
    /// Sert à ne pas démarrer une requête qu'on n'aura pas le temps de
    /// consommer : sous échéance, une requête lancée trop tard rend des mondes
    /// que la boucle jettera sans les résoudre — mesuré à 45 % des mondes reçus
    /// en fin de donne, pour 164 ms d'attente pure.
    ///
    /// Une seule moyenne pour toute la donne, alors que la latence décroît avec
    /// les cartes restantes (224 ms à l'entame, 164 ms à deux cartes) : elle
    /// sur-estime donc légèrement en fin de donne, ce qui fait renoncer un peu
    /// trop tôt plutôt qu'un peu trop tard. C'est le bon sens de l'erreur, et
    /// l'`ALPHA` la fait redescendre au fil de la donne.
    source_latency_us: f64,
    /// Part des mondes demandés que la source rend effectivement, en moyenne
    /// mobile sur la donne. Sert à **sur-commander** : `retain_valid` écarte
    /// les mondes que la belote rend impossibles — le sampler ne voit pas
    /// l'annonce — donc demander `n` en rend typiquement 0,85 n, et il faut un
    /// second aller-retour pour finir le compte. Or un aller-retour coûte une
    /// séquence de jetons entière sur le GPU, alors que les lanes
    /// supplémentaires d'une seule requête sont quasi gratuites : le coût d'un
    /// lot est dominé par le nombre de pas, pas par la largeur.
    ///
    /// Mesuré sur le corpus de donnes complètes : 1,15 aller-retour par
    /// décision, et un taux de rendu de 71 à 98 % selon le stade. La
    /// sur-commande ne change **pas** le nombre de mondes résolus — le mode
    /// compte s'arrête à `determinizations` — seulement lesquels, et elle
    /// réduit le repli local en fin de recherche.
    source_fill: f64,
    tt_buf: crate::solver::TtBuf,
}

impl IsDdConfig {
    /// Mondes visés à cette position, en mode compte.
    ///
    /// Le `max(1)` n'est pas de la superstition : un objectif à zéro rendrait
    /// `det_count >= det_target` vrai au premier tour, donc une recherche qui
    /// sort **sans avoir résolu un seul monde** — toutes les cartes à la valeur
    /// neutre, et le premier coup légal joué. Une régression silencieuse, pas un
    /// plantage. `parse_det_schedule` refuse déjà les zéros, mais le champ est
    /// public et se construit aussi à la main.
    #[inline]
    pub fn dets_for(&self, cards_left: u32) -> u32 {
        match &self.det_schedule {
            Some(s) => s[(cards_left as usize).min(8)].max(1),
            None => self.determinizations,
        }
    }
}

/// Poids de la dernière mesure dans la moyenne mobile de latence.
const LATENCY_ALPHA: f64 = 0.25;

/// Plafond de sur-commande. Sans borne, une position dont la source ne peut
/// presque rien produire (finale sur-contrainte) ferait demander des milliers
/// de mondes pour n'en obtenir aucun — le sampler est à court, pas timide.
const MAX_OVERASK: f64 = 1.8;

impl IsDdSearch {
    pub fn new() -> Self {
        IsDdSearch {
            beliefs: None,
            belief_net: None,
            belief_tracking: None,
            cred_bid_net: None,
            cred_play_net: None,
            auction: Vec::new(),
            plays: Vec::new(),
            init_state: None,
            played_by: [0; 4],
            world_queue: Vec::new(),
            source_latency_us: 0.0,
            source_fill: 1.0,
            tt_buf: new_tt_buffer(),
        }
    }

    /// Load the bid net used as the auction-credibility judge (`cred_alpha`).
    pub fn load_cred_bid_net(&mut self, path: &str) -> std::io::Result<()> {
        self.cred_bid_net = Some(crate::bid_net::BidNet::load(path)?);
        Ok(())
    }

    /// Load the DMC net used as the play-credibility judge (`cred_alpha`).
    /// Must be a canonical (411-dim, `OBS_DIM_TR`) model.
    pub fn load_cred_play_net(&mut self, path: &str) -> std::io::Result<()> {
        let net = DmcNet::load(path)?;
        if net.obs_dim() != OBS_DIM_TR {
            return Err(std::io::Error::new(
                std::io::ErrorKind::InvalidData,
                "play-credibility judge must be a canonical (411-dim) DMC model",
            ));
        }
        self.cred_play_net = Some(net);
        Ok(())
    }

    /// Load a BeliefNet for NN-based beliefs.
    pub fn load_belief_net(&mut self, path: &str) -> std::io::Result<()> {
        self.belief_net = Some(BeliefNet::load(path)?);
        Ok(())
    }

    /// Check if a BeliefNet is loaded.
    pub fn has_belief_net(&self) -> bool {
        self.belief_net.is_some()
    }

    /// Initialize beliefs for a new deal from the given observer's perspective.
    pub fn init_deal(&mut self, state: &GameState, observer: u8, use_soft_inference: bool) {
        let mut beliefs = CardBeliefs::new(state, observer);
        beliefs.use_soft_inference = use_soft_inference;
        self.beliefs = Some(beliefs);

        // Also init NN belief tracking if BeliefNet is loaded
        if self.belief_net.is_some() {
            let mut tracking = EnvTracking::new();
            tracking.reset(state.dealer);
            self.belief_tracking = Some(tracking);
        }

        // Credibility judge: remember the pre-auction state, reset the logs.
        self.auction.clear();
        self.plays.clear();
        self.init_state = Some(*state);
        self.played_by = [0; 4];
        self.world_queue.clear();

    }

    /// Initialize beliefs for a new deal, applying the config's belief knobs.
    pub fn init_deal_with_config(
        &mut self,
        state: &GameState,
        observer: u8,
        config: &IsDdConfig,
    ) {
        self.init_deal(state, observer, config.use_soft_inference);
        // Set dominance factor on beliefs.
        if let Some(ref mut beliefs) = self.beliefs {
            beliefs.dominance_factor = config.dominance_factor;
        }
    }

    /// Record an action by any player, updating beliefs and the world source's
    /// view of the history.
    ///
    /// `state_before` is the state BEFORE the action was applied.
    pub fn record_action(&mut self, state_before: &GameState, player: u8, action: u8) {
        if let Some(beliefs) = &mut self.beliefs {
            beliefs.record_action(state_before, player, action);
        }
        if let Some(tracking) = &mut self.belief_tracking {
            tracking.track_action(state_before, action);
        }
        if state_before.phase == Phase::Bidding {
            self.auction.push((player, action));
        }
        if state_before.phase == Phase::Playing {
            self.played_by[player as usize] |= 1u32 << action;
            self.plays.push((player, action));
        }
    }

    /// Reset beliefs (e.g., between deals).
    pub fn reset(&mut self) {
        self.beliefs = None;
        self.belief_tracking = None;
    }

    /// Valeur d'un coup dans un monde donné, selon l'objectif configuré.
    ///
    /// `ns_card_pts` est ce que rend le solveur : le total N-S **final** de la
    /// donne dans ce monde (`state.points[0]` au terminal), dix de der compris.
    #[inline]
    fn world_value(
        &self,
        world: &GameState,
        ns_card_pts: i16,
        belote: [i16; 2],
        objective: PlayObjective,
    ) -> f64 {
        match objective {
            PlayObjective::CardPoints => ns_card_pts as f64,
            PlayObjective::DealScore => {
                let total = total_card_points(world, ns_card_pts);
                let card = [ns_card_pts, total - ns_card_pts];
                let taker = world.contract.team as usize;
                let s = deal_score_from_card_points(
                    &world.contract,
                    card,
                    belote,
                    card[taker] == CAPOT_PTS,
                );
                (s.scores[0] - s.scores[1]) as f64
            }
        }
    }

    /// Valeur neutre d'un coup dont aucun monde n'a été résolu, dans l'échelle
    /// de l'objectif : la moitié du total en points cartes, l'écart nul entre
    /// les deux camps en score de donne.
    ///
    /// N'intervient jamais dans une décision — soit le coup est forcé, soit
    /// aucun monde n'a abouti pour cette carte — mais elle sort dans
    /// `card_scores`, que les pages d'analyse affichent.
    #[inline]
    fn neutral_value(objective: PlayObjective) -> f32 {
        match objective {
            PlayObjective::CardPoints => (TOTAL_PTS / 2) as f32,
            PlayObjective::DealScore => 0.0,
        }
    }

    /// Belote finale du monde, ou zéro quand l'objectif ne la regarde pas.
    #[inline]
    fn world_belote_for(&self, world: &GameState, objective: PlayObjective) -> [i16; 2] {
        match objective {
            PlayObjective::CardPoints => [0, 0],
            PlayObjective::DealScore => {
                final_belote(&world.hands, &self.played_by, world.contract.trump)
            }
        }
    }

    /// Compute belief weights for determinization.
    /// When NN beliefs are enabled, applies hard constraints from heuristic CardBeliefs
    /// (voids, trump ceiling) on top of NN soft predictions.
    fn compute_weights(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        observer: u8,
    ) -> Option<[[f32; 32]; 4]> {
        // `record_action` ne voit que l'état d'*avant* une action : la belote
        // annoncée par la carte qui vient d'être posée n'y est pas encore. Ici
        // l'état est celui de la position courante, donc la déduction est à jour.
        if let Some(beliefs) = self.beliefs.as_mut() {
            beliefs.apply_belote_facts(state);
        }
        let base_weights = if config.use_nn_beliefs && self.belief_net.is_some() {
            let net = self.belief_net.as_mut().unwrap();
            let tracking = self.belief_tracking.as_ref().unwrap();

            // Hard constraints from CardBeliefs (shared by V2/V3 obs)
            let make_hc = |beliefs: &Option<CardBeliefs>| -> [f32; 96] {
                if let Some(beliefs) = beliefs {
                    let raw = beliefs.raw_weights();
                    let observer_hand = state.hands[observer as usize];
                    let mut played = state.played_cards;
                    for i in 0..4 {
                        let c = state.current_trick[i];
                        if c != crate::card::EMPTY {
                            played |= 1u32 << c;
                        }
                    }
                    let known = observer_hand | played;
                    let hidden_players = [
                        ((observer + 1) % 4),
                        ((observer + 2) % 4),
                        ((observer + 3) % 4),
                    ];
                    let mut hc = [0.0f32; 96];
                    for (hp_idx, &hp) in hidden_players.iter().enumerate() {
                        let base = hp_idx * 32;
                        for card_idx in 0..32u32 {
                            if known & (1 << card_idx) != 0 {
                                hc[base + card_idx as usize] = 1.0;
                                continue;
                            }
                            if raw[hp as usize][card_idx as usize] == 0.0 {
                                hc[base + card_idx as usize] = 1.0;
                            }
                        }
                    }
                    hc
                } else {
                    [0.0f32; 96]
                }
            };

            let logits = if net.obs_dim() == BELIEF_OBS_DIM_V2 {
                let hard_constraints = make_hc(&self.beliefs);
                let mut obs_buf = [0.0f32; BELIEF_OBS_DIM_V2];
                belief_obs::write_belief_observation_v2(
                    &mut obs_buf, 0, state, tracking, observer, &hard_constraints,
                );
                net.evaluate(&obs_buf)
            } else if net.obs_dim() == BELIEF_OBS_DIM_V3 {
                let hard_constraints = make_hc(&self.beliefs);
                let (trick_leads, trick_winners, suit_fail_rel) =
                    derive_v3_temporal(state, tracking, observer);
                let mut obs_buf = [0.0f32; BELIEF_OBS_DIM_V3];
                belief_obs::write_belief_observation_v3(
                    &mut obs_buf, 0, state, tracking, observer,
                    &hard_constraints, &trick_leads, &trick_winners, &suit_fail_rel,
                );
                net.evaluate(&obs_buf)
            } else {
                let mut obs_buf = [0.0f32; BELIEF_OBS_DIM];
                belief_obs::write_belief_observation(&mut obs_buf, 0, state, tracking, observer);
                net.evaluate(&obs_buf)
            };
            let mut nn_weights = crate::belief_net::belief_to_weights(&logits, net.num_classes(), state, observer);

            // Hard constraints (voids, trump ceiling, played cards) are facts, not beliefs.
            // Always apply them on top of NN soft predictions.
            if let Some(ref beliefs) = self.beliefs {
                let raw = beliefs.raw_weights();
                let observer_hand = state.hands[observer as usize];
                let mut played = state.played_cards;
                for i in 0..4 {
                    let c = state.current_trick[i];
                    if c != crate::card::EMPTY {
                        played |= 1u32 << c;
                    }
                }
                let known = observer_hand | played;

                for card in 0..32u32 {
                    if known & (1 << card) != 0 {
                        continue;
                    }
                    for p in 0..4usize {
                        if raw[p][card as usize] == 0.0 {
                            nn_weights[p][card as usize] = 0.0;
                        }
                    }
                    let sum: f32 = (0..4).map(|p| nn_weights[p][card as usize]).sum();
                    if sum > 0.0 {
                        let inv = 1.0 / sum;
                        for p in 0..4 {
                            nn_weights[p][card as usize] *= inv;
                        }
                    }
                }
            }

            Some(nn_weights)
        } else {
            self.beliefs.as_ref().map(|b| b.normalized_weights())
        };

        base_weights
    }

    /// Get current belief weights for a given observer.
    /// Returns `(nn_weights, heuristic_weights)` where each is `weights[player][card]`.
    /// NN weights use hybrid mode (NN + hard constraints from heuristic).
    /// Heuristic weights are purely from `CardBeliefs::normalized_weights()`.
    pub fn get_belief_weights(
        &mut self,
        state: &GameState,
        observer: u8,
    ) -> (Option<[[f32; 32]; 4]>, Option<[[f32; 32]; 4]>) {
        let nn_config = IsDdConfig {
            use_nn_beliefs: true,
            ..Default::default()
        };
        let nn_weights = if self.belief_net.is_some() {
            self.compute_weights(state, &nn_config, observer)
        } else {
            None
        };
        let heuristic_weights = self.beliefs.as_ref().map(|b| b.normalized_weights());
        (nn_weights, heuristic_weights)
    }

    /// Credibility weight of a world: replay the observed history holding this
    /// world's reconstructed full hands and, for each action taken by a hidden
    /// player (`p != observer`), ask the reference policy whether it would
    /// replay that action. Both phases are judged when the corresponding net is
    /// loaded — **bids** by the bid net (`load_cred_bid_net`), **plays** by the
    /// canonical DMC net (`load_cred_play_net`). Each judged action contributes
    /// a rank factor by how many legal moves the net ranks strictly above the
    /// observed one:
    ///
    /// | net rates above observed | factor |
    /// |--------------------------|--------|
    /// | 0 (it *is* the argmax)   | 1.00   |
    /// | 1–2 (top-3)              | 0.70   |
    /// | ≥3                       | 0.35   |
    ///
    /// Factors multiply across judged actions; the product is flattened by
    /// `alpha` (`w.powf(alpha)`). Returns 1.0 when `alpha <= 0`, no judge is
    /// loaded, or the world cannot be reconstructed into four 8-card hands.
    ///
    /// Cost: one bid-net eval per hidden bid + one DMC eval per hidden play,
    /// per world. Negligible for the auction (~4–8 bids); the play path scales
    /// with tricks played, so keep world counts modest when it is enabled.
    fn credibility_weight(&mut self, world_hands: &[u32; 4], observer: u8, alpha: f32) -> f32 {
        if alpha <= 0.0 {
            return 1.0;
        }
        let Some(base) = self.init_state else { return 1.0 };
        if self.cred_bid_net.is_none() && self.cred_play_net.is_none() {
            return 1.0;
        }

        // Reconstruct full initial hands: the determinized world only assigns
        // cards still in hand, so add back what each seat has already played.
        let mut init_hands = [0u32; 4];
        for p in 0..4usize {
            init_hands[p] = world_hands[p] | self.played_by[p];
            if card_count(init_hands[p]) != 8 {
                return 1.0; // inconsistent reconstruction — skip weighting
            }
        }

        let mut s = base;
        s.hands = init_hands;
        // Replayed public tracking, needed for the canonical DMC play obs.
        let mut tracking = EnvTracking::new();
        tracking.reset(base.dealer);
        let mut w = 1.0f32;

        // --- Auction: judge each hidden bid with the bid net. ---
        let mut bid_hist: Vec<(u8, u8)> = Vec::with_capacity(self.auction.len());
        let mut bid_obs_buf = self
            .cred_bid_net
            .as_ref()
            .map(|n| vec![0.0f32; n.obs_dim()])
            .unwrap_or_default();
        for i in 0..self.auction.len() {
            let (p, a) = self.auction[i];
            if p != observer && s.phase == Phase::Bidding {
                if let Some(net) = self.cred_bid_net.as_mut() {
                    use crate::bid_obs::{
                        self, BID_OBS_DIM_SCORE_AWARE, BID_OBS_DIM_SCORE_AWARE_V2,
                        BID_OBS_DIM_SCORE_AWARE_V3,
                    };
                    // Standalone-deal convention: cumulative scores 0-0.
                    match net.obs_dim() {
                        BID_OBS_DIM_SCORE_AWARE_V3 => bid_obs::write_bid_observation_score_aware_v3(
                            &mut bid_obs_buf, 0, &s, &bid_hist, 0, 0,
                        ),
                        BID_OBS_DIM_SCORE_AWARE_V2 => bid_obs::write_bid_observation_score_aware_v2(
                            &mut bid_obs_buf, 0, &s, &bid_hist, 0, 0,
                        ),
                        BID_OBS_DIM_SCORE_AWARE => bid_obs::write_bid_observation_score_aware(
                            &mut bid_obs_buf, 0, &s, &bid_hist, 0, 0,
                        ),
                        _ => bid_obs::write_bid_observation(&mut bid_obs_buf, 0, &s, &bid_hist),
                    }
                    let q = net.evaluate(&bid_obs_buf);
                    let legal = s.legal_actions();
                    let mut better = 0u32;
                    let qa = q[a as usize];
                    for c in 0..43u8 {
                        if c != a && legal & (1u64 << c) != 0 && q[c as usize] > qa {
                            better += 1;
                        }
                    }
                    w *= rank_factor(better);
                }
            }
            bid_hist.push((p, a));
            tracking.track_action(&s, a);
            s.step(a);
        }

        // --- Play: judge each hidden play with the canonical DMC net. ---
        if self.cred_play_net.is_some() {
            for i in 0..self.plays.len() {
                let (p, a) = self.plays[i];
                if p != observer && s.phase == Phase::Playing {
                    let net = self.cred_play_net.as_mut().unwrap();
                    let obs = dmc_obs::make_observation_tr(&s, &tracking);
                    let order = dmc_obs::current_player_order(&s, &tracking);
                    let mask = dmc_obs::cardset_to_canonical(s.legal_actions() as u32, &order);
                    let q = net.evaluate(&obs);
                    let ca = dmc_obs::card_to_canonical(a, &order);
                    let qa = q[ca as usize];
                    let mut better = 0u32;
                    for c in 0..32u8 {
                        if c != ca && mask & (1u32 << c) != 0 && q[c as usize] > qa {
                            better += 1;
                        }
                    }
                    w *= rank_factor(better);
                }
                tracking.track_action(&s, a);
                s.step(a);
            }
        }

        w.powf(alpha)
    }

    /// Sample determinized play-phase worlds without solving them (used by the
    /// world-credibility benchmark). `use_beliefs` follows the same path as
    /// `search` (NN soft beliefs + hard constraints when a net is loaded);
    /// otherwise constraint-uniform sampling. Returns remaining-card hands.
    pub fn sample_worlds(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        observer: u8,
        n_worlds: usize,
        use_beliefs: bool,
        rng: &mut impl Rng,
    ) -> Vec<[u32; 4]> {
        let weights = if use_beliefs {
            self.compute_weights(state, config, observer)
        } else {
            None
        };
        (0..n_worlds)
            .filter_map(|_| match &weights {
                Some(w) => determinize_weighted(state, observer, w, rng)
                    .or_else(|| determinize_greedy(state, observer, rng)),
                None => determinize_greedy(state, observer, rng),
            })
            .map(|s| s.hands)
            .collect()
    }

    /// Heuristic `CardBeliefs` weights, before any NN blending.
    pub fn base_belief_weights(&self) -> Option<[[f32; 32]; 4]> {
        self.beliefs.as_ref().map(|b| b.normalized_weights())
    }

    /// Check if beliefs uniquely determine every unknown card's owner.
    /// If so, return the fully resolved GameState (perfect information) for direct DD solve.
    fn try_resolve_position(&self, state: &GameState, observer: u8) -> Option<GameState> {
        let beliefs = self.beliefs.as_ref()?;
        let raw = beliefs.raw_weights();

        let mut played = state.played_cards;
        for i in 0..4 {
            let c = state.current_trick[i];
            if c != crate::card::EMPTY {
                played |= 1u32 << c;
            }
        }
        let known = state.hands[observer as usize] | played;
        let unknown = crate::card::ALL_CARDS ^ known;

        if unknown == 0 {
            return Some(*state); // All cards already known.
        }

        let mut hands = [0u32; 4];
        hands[observer as usize] = state.hands[observer as usize];

        for card in crate::card::CardIter(unknown) {
            let mut owner: Option<u8> = None;
            for p in 0..4u8 {
                if p == observer {
                    continue;
                }
                if raw[p as usize][card as usize] > 0.0 {
                    if owner.is_some() {
                        return None; // Multiple candidates — not resolved.
                    }
                    owner = Some(p);
                }
            }
            let p = owner?;
            hands[p as usize] |= 1u32 << card;
        }

        // Verify card counts match original state.
        for p in 0..4u8 {
            if card_count(hands[p as usize]) != card_count(state.hands[p as usize]) {
                return None;
            }
        }

        let mut resolved = *state;
        resolved.hands = hands;
        Some(resolved)
    }

    /// Best card, sampling worlds from beliefs / constraint-uniform only.
    pub fn search(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
    ) -> u8 {
        if state.phase == Phase::Bidding {
            return config.bid_function.bid(state);
        }
        self.search_with_stats(state, config, rng).best_action
    }

    /// Full result, sampling worlds from beliefs / constraint-uniform only.
    ///
    /// Infallible: without a [`WorldSource`] there is nothing that can fail.
    /// Use [`search_with_source`](Self::search_with_source) to draw worlds from
    /// a playgen sampler — that is the strong configuration, and the one
    /// production uses.
    pub fn search_with_stats(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
    ) -> IsDdResult {
        self.run_search(state, config, rng, None)
            .expect("search without a world source cannot fail")
    }

    /// Full result, drawing determinized worlds from `source`.
    ///
    /// Worlds are pulled in batches and refilled on demand until the
    /// determinization count or the time budget is exhausted. If the source
    /// errors, the error propagates: a search that silently continued on
    /// constraint-uniform worlds would be a measurably weaker agent wearing
    /// the same name. A source that legitimately runs *dry* (returns an empty
    /// batch without erroring, as happens in over-constrained endgames) is not
    /// an error — the search falls back to its own sampling and reports the
    /// mix in [`IsDdResult::worlds`].
    pub fn search_with_source(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
        source: &mut dyn WorldSource,
    ) -> Result<IsDdResult, crate::agent::AgentError> {
        self.run_search(state, config, rng, Some(source))
    }

    fn run_search(
        &mut self,
        state: &GameState,
        config: &IsDdConfig,
        rng: &mut impl Rng,
        mut source: Option<&mut dyn WorldSource>,
    ) -> Result<IsDdResult, crate::agent::AgentError> {
        debug_assert!(!state.is_terminal(), "Cannot search from terminal state");

        let observer = state.current_player();
        let team = GameState::player_team(observer);
        let maximizing = team == 0; // NS maximizes, EW minimizes

        if config.early_termination {
            // Forced move: only 1 legal action — skip search entirely.
            let legal = state.legal_actions();
            if legal.count_ones() == 1 {
                let card = legal.trailing_zeros() as u8;
                return Ok(IsDdResult {
                    best_action: card,
                    card_scores: vec![(card, Self::neutral_value(config.objective))],
                    determinizations: 0,
                    worlds: WorldCounts::default(),
                    source: SourceUsage {
                        cards_left: card_count(state.hands[observer as usize]) as u8,
                        ..Default::default()
                    },
                });
            }

            // Resolved position: beliefs uniquely determine all card locations.
            // Single DD solve gives the exact answer — no determinization needed.
            if let Some(resolved) = self.try_resolve_position(state, observer) {
                let scores = solve_with_scores(&resolved, Some(&mut self.tt_buf));
                let belote = self.world_belote_for(&resolved, config.objective);
                let mut card_scores = Vec::new();
                let mut best_action = legal.trailing_zeros() as u8;
                let mut best_avg: f32 =
                    if maximizing { f32::NEG_INFINITY } else { f32::INFINITY };

                for i in 0..scores.count {
                    let (card, ns_pts) = scores.scores[i];
                    let avg =
                        self.world_value(&resolved, ns_pts, belote, config.objective) as f32;
                    card_scores.push((card, avg));
                    let better = if maximizing { avg > best_avg } else { avg < best_avg };
                    if better {
                        best_avg = avg;
                        best_action = card;
                    }
                }

                // The position is fully resolved by facts, not by sampling —
                // count it as a world of its own kind rather than mislabeling it.
                return Ok(IsDdResult {
                    best_action,
                    card_scores,
                    determinizations: 1,
                    worlds: WorldCounts::default(),
                    source: SourceUsage {
                        cards_left: card_count(state.hands[observer as usize]) as u8,
                        ..Default::default()
                    },
                });
            }
        }

        // Score accumulators: weighted sum of NS points per card, weight per card
        // (weights are 1.0 unless credibility weighting is enabled).
        let mut score_sum = [0f64; 32];
        let mut weight_sum = [0f64; 32];

        // Scale time budget by cards remaining
        let cards_left = card_count(state.hands[observer as usize]);
        // Mondes visés ici : un compte plat, ou l'échelon du calendrier
        // correspondant au stade de la donne (cf. `IsDdConfig::det_schedule`).
        let det_target = config.dets_for(cards_left);
        let deadline = config.time_limit_ms.map(|ms| {
            let scaled_ms = (ms as u64 * cards_left as u64) / 8;
            Instant::now() + Duration::from_millis(scaled_ms.max(1))
        });

        // Belief weights are computed **on first need**, not up front. Since a
        // world from the source always wins over a belief-weighted draw
        // (`generate_world`), a search whose queue never runs dry never looks at
        // them — and that is the normal case in production, where playgen fills
        // the queue. Computing them eagerly ran a belief-net forward pass per
        // decision whose result was then thrown away.
        //
        // `compute_weights` reads no RNG and depends only on (state, observer,
        // config), all fixed for the duration of a search, so deferring it
        // changes neither the weights nor the random stream.
        let mut weights: Option<[[f32; 32]; 4]> = None;
        let mut weights_ready = false;

        let mut successful_dets = 0u32;
        let mut det_count = 0u32;
        let mut world_counts = WorldCounts::default();
        let mut usage = SourceUsage { cards_left: cards_left as u8, ..Default::default() };

        // Once the source stops producing worlds we stop asking, so an
        // over-constrained endgame costs one empty round trip, not one per world.
        let mut source_dry = false;
        // La source a encore des mondes, mais l'échéance ne laisse plus le temps
        // d'en attendre un lot. Distinct de `source_dry` : là c'est nous qui
        // renonçons, pas le sampler qui s'épuise.
        let mut out_of_time_to_refill = false;

        // The search runs in chunks: **generate** a batch of worlds sequentially
        // (the world queue and the RNG are stateful), then **solve** the whole
        // batch — in parallel when `config.parallel` is set, otherwise one by one
        // reusing this search's TT. The chunk is one world in sequential mode
        // (tightest deadline adherence, identical to the legacy per-world loop)
        // and one worker-slot's worth in parallel mode.
        let chunk_size = solve_chunk_size(config.parallel);

        // Nombre de tours consécutifs sans aucun monde résolu. Ne sert qu'au
        // plancher : c'est ce qui sépare « le GPU est lent » de « cette position
        // ne se déterminise pas », deux situations que l'échéance confondait
        // parce qu'elle coupait les deux.
        let mut barren_rounds = 0u32;
        const STUCK_ROUNDS: u32 = 64;

        loop {
            // Sous le plancher, l'échéance ne coupe pas : la pression de calcul
            // doit se payer en latence, pas en force de jeu.
            let below_floor = config.min_worlds.is_some_and(|m| successful_dets < m);
            if let Some(d) = deadline {
                let past_deadline = Instant::now() >= d;
                if past_deadline && (!below_floor || barren_rounds >= STUCK_ROUNDS) {
                    break;
                }
                // Assez de mondes : la réponse a cessé de bouger, le temps qui
                // reste n'achèterait que de la charge GPU. Sous échéance
                // seulement — en mode compte c'est `determinizations` qui borne.
                if config.max_worlds.is_some_and(|m| successful_dets >= m) {
                    break;
                }
            } else if det_count >= det_target {
                break;
            }

            // How many worlds to attempt this round.
            let remaining = if deadline.is_some() {
                chunk_size
            } else {
                chunk_size.min((det_target - det_count) as usize)
            };

            // --- Refill from the world source when the queue cannot cover the
            // round. In count mode ask for the whole remaining budget (one round
            // trip per move); under a deadline ask for `world_batch` at a time. ---
            if let Some(src) = source.as_deref_mut() {
                if !source_dry && !out_of_time_to_refill && self.world_queue.len() < remaining {
                    // Un aller-retour coûte une latence quasi fixe (~164-224 ms
                    // sur le sidecar local, le double sous concurrence), et
                    // l'échéance n'est testée qu'en tête de boucle : rien
                    // n'empêchait de lancer une requête de 164 ms avec 10 ms au
                    // compteur, puis de jeter tout ce qu'elle rendait. Le
                    // **premier** aller-retour part toujours — sans lui la
                    // recherche n'aurait aucun monde appris, et c'est justement
                    // en fin de donne que playgen sert le plus, en écartant les
                    // mondes quasi impossibles qu'un tirage uniforme traite à
                    // égalité des mondes plausibles.
                    //
                    // Sous le plancher ce calcul ne s'applique pas : renoncer à
                    // l'aller-retour faute de temps, c'est précisément rendre la
                    // réponse à l'heure en cherchant moins.
                    let no_time_left = match deadline {
                        _ if below_floor => false,
                        Some(d) if usage.rounds > 0 && self.source_latency_us > 0.0 => {
                            let left = d.saturating_duration_since(Instant::now());
                            (left.as_micros() as f64) < self.source_latency_us
                        }
                        _ => false,
                    };
                    if no_time_left {
                        out_of_time_to_refill = true;
                    } else {
                        let mut want = if deadline.is_some() {
                            config.world_batch.max(remaining)
                        } else {
                            ((det_target - det_count) as usize).max(remaining)
                        };
                        // Ne pas commander plus que le plafond ne laisse résoudre.
                        // Sans ça le plafond n'économise que du CPU : on
                        // recevait le lot entier puis on s'arrêtait au milieu —
                        // mesuré à 34,5 % de mondes jetés, le sidecar les ayant
                        // fabriqués pour rien.
                        if let Some(m) = config.max_worlds {
                            let already = successful_dets as usize + self.world_queue.len();
                            want = want.min((m as usize).saturating_sub(already));
                        }
                        if want == 0 {
                            out_of_time_to_refill = true;
                        } else {
                        // Sur-commande : cf. `source_fill`. Ce qui revient en
                        // trop reste dans la file et sert au tour suivant, ou
                        // est jeté en fin de recherche comme n'importe quel
                        // monde de la position précédente.
                        let ask = ((want as f64 / self.source_fill.max(1.0 / MAX_OVERASK)).ceil()
                            as usize)
                            .max(want);
                        let t0 = Instant::now();
                        let batch = src.worlds(state, observer, ask, rng)?;
                        let took_us = t0.elapsed().as_micros() as f64;
                        let fill = batch.len() as f64 / ask as f64;
                        self.source_fill =
                            (1.0 - LATENCY_ALPHA) * self.source_fill + LATENCY_ALPHA * fill;
                        self.source_latency_us = if self.source_latency_us == 0.0 {
                            took_us
                        } else {
                            (1.0 - LATENCY_ALPHA) * self.source_latency_us
                                + LATENCY_ALPHA * took_us
                        };
                        usage.source_us += took_us as u64;
                        usage.rounds += 1;
                        usage.requested += ask as u32;
                        usage.delivered += batch.len() as u32;
                        if batch.is_empty() {
                            source_dry = true;
                        } else {
                            self.world_queue.extend(batch);
                        }
                        }
                    }
                }
            }

            // Plus le temps de redemander : on finit la réserve et on s'arrête.
            //
            // Le plafonnement de `remaining` n'est pas cosmétique. En mode
            // parallèle un chunk vaut un tour de pool (32 mondes ici), donc sans
            // lui un chunk qui déborde d'une file presque vide se complète en
            // mondes **locaux** — mesuré à 62,6 % de décisions partielles et
            // 806 mondes belief. Or playgen n'apporte pas que des mondes, il
            // apporte une pondération : il écarte les mondes quasi impossibles
            // qu'un tirage uniforme sous contraintes traite à égalité des
            // mondes plausibles. Diluer l'agrégat avec ceux-là, c'est défaire
            // en fin de donne ce qu'on est allé chercher.
            //
            // `source_dry` est un cas différent et garde son repli local : là
            // le sampler ne *peut* plus produire, et un plancher de couverture
            // vaut mieux que rien.
            let remaining = if out_of_time_to_refill {
                remaining.min(self.world_queue.len())
            } else {
                remaining
            };
            if remaining == 0 {
                break;
            }

            // --- Generate a chunk of worlds (sequential). ---
            let mut chunk: Vec<GameState> = Vec::with_capacity(remaining);
            let mut chunk_origins: Vec<WorldOrigin> = Vec::with_capacity(remaining);
            let mut attempted = 0u32;
            for _ in 0..remaining {
                attempted += 1;
                // The queue is empty, so this world will come from the belief
                // net or from uniform sampling — now the weights are needed.
                if self.world_queue.is_empty() && !weights_ready {
                    weights = self.compute_weights(state, config, observer);
                    weights_ready = true;
                }
                if let Some((s, origin)) = self.generate_world(state, observer, &weights, config, rng)
                {
                    chunk.push(s);
                    chunk_origins.push(origin);
                }
            }
            det_count += attempted;
            if chunk.is_empty() {
                // Every attempt this round failed to determinize; in count mode
                // `det_count` still advances so we terminate, in time mode we
                // retry until the deadline. Avoid touching the accumulators.
                barren_rounds += 1;
                continue;
            }
            barren_rounds = 0;

            // --- Credibility weights (sequential: the judge nets are stateful). ---
            let cred_weights: Vec<f64> = if config.cred_alpha > 0.0 {
                chunk
                    .iter()
                    .map(|s| self.credibility_weight(&s.hands, observer, config.cred_alpha) as f64)
                    .collect()
            } else {
                vec![1.0; chunk.len()]
            };

            // --- Solve the chunk (parallel or sequential). ---
            let chunk_scores = solve_worlds(&chunk, config.parallel, &mut self.tt_buf);

            // --- Aggregate in a fixed order (parallel result is identical). ---
            for ((world, scores), &cw) in
                chunk.iter().zip(chunk_scores.iter()).zip(cred_weights.iter())
            {
                let belote = self.world_belote_for(world, config.objective);
                for i in 0..scores.count {
                    let (card, ns_pts) = scores.scores[i];
                    score_sum[card as usize] +=
                        self.world_value(world, ns_pts, belote, config.objective) * cw;
                    weight_sum[card as usize] += cw;
                }
            }
            successful_dets += chunk.len() as u32;
            for origin in &chunk_origins {
                world_counts.record(*origin);
            }
        }

        // Sourced worlds are position-specific: drop any leftover so the next
        // search cannot consume worlds sampled for the previous position.
        usage.discarded = self.world_queue.len() as u32;
        self.world_queue.clear();

        // Build result: pick best card based on aggregated scores
        let legal = state.legal_actions();
        let mut best_action = legal.trailing_zeros() as u8;
        let mut best_avg: f32 = if maximizing { f32::NEG_INFINITY } else { f32::INFINITY };
        let mut card_scores = Vec::new();

        let mut mask = legal;
        while mask != 0 {
            let card = mask.trailing_zeros() as u8;
            let wsum = weight_sum[card as usize];
            let avg = if wsum > 1e-9 {
                (score_sum[card as usize] / wsum) as f32
            } else {
                Self::neutral_value(config.objective)
            };

            card_scores.push((card, avg));

            let dominated = if maximizing {
                avg > best_avg
            } else {
                avg < best_avg
            };
            if dominated {
                best_avg = avg;
                best_action = card;
            }
            mask &= mask - 1;
        }

        // Fallback: if no determinization succeeded, pick first legal action
        if successful_dets == 0 {
            best_action = legal.trailing_zeros() as u8;
        }

        Ok(IsDdResult {
            best_action,
            card_scores,
            determinizations: successful_dets,
            worlds: world_counts,
            source: usage,
        })
    }

    /// Take one determinized world for the current position.
    ///
    /// Worlds already pulled from the [`WorldSource`] are consumed first; when
    /// the queue is empty the search falls back to its own sampling — a
    /// **belief-weighted** world with probability `belief_frac` when a belief
    /// source is active, otherwise a **constraint-uniform** one. Hard
    /// constraints (voids, trump ceiling, played cards) are honored by every
    /// path. Returns `None` when a determinizer fails (an over-constrained
    /// position); the caller counts the attempt against the budget and moves
    /// on. The [`WorldOrigin`] says which branch produced the world so the
    /// caller can report the mix.
    fn generate_world(
        &mut self,
        state: &GameState,
        observer: u8,
        weights: &Option<[[f32; 32]; 4]>,
        config: &IsDdConfig,
        rng: &mut impl Rng,
    ) -> Option<(GameState, WorldOrigin)> {
        // Worlds from the source are pre-validated by `retain_valid`.
        if let Some(hands) = self.world_queue.pop() {
            let mut s = *state;
            s.hands = hands;
            return Some((s, WorldOrigin::Injected));
        }

        if weights.is_some() && rng.gen::<f32>() < config.belief_frac {
            let w = weights.as_ref().unwrap();
            return match determinize_weighted(state, observer, w, rng) {
                Some(s) => Some((s, WorldOrigin::Belief)),
                None => determinize_greedy(state, observer, rng)
                    .map(|s| (s, WorldOrigin::Uniform)),
            };
        }

        // Ensemble coverage floor: constraint-uniform world.
        determinize_greedy(state, observer, rng).map(|s| (s, WorldOrigin::Uniform))
    }
}

/// Worlds solved per generate/solve round. Sequential mode uses one world per
/// round (tightest deadline adherence, identical to the legacy per-world loop);
/// parallel mode fills one round of the rayon worker pool.
#[inline]
fn solve_chunk_size(parallel: bool) -> usize {
    #[cfg(feature = "parallel")]
    {
        if parallel {
            return rayon::current_num_threads().max(1);
        }
    }
    let _ = parallel;
    1
}

/// Solve a batch of fully-determinized worlds, returning per-world DD scores in
/// input order. In parallel mode each rayon worker keeps its own reusable TT
/// (`map_init`); sequential mode reuses the caller's `tt_buf`. DD is exact and
/// deterministic, so the two paths return identical scores.
#[cfg(feature = "parallel")]
fn solve_worlds(
    worlds: &[GameState],
    parallel: bool,
    tt_buf: &mut crate::solver::TtBuf,
) -> Vec<crate::solver::SolveScores> {
    use rayon::prelude::*;
    if parallel {
        worlds
            .par_iter()
            .map_init(new_tt_buffer, |tt, s| solve_with_scores(s, Some(tt)))
            .collect()
    } else {
        worlds
            .iter()
            .map(|s| solve_with_scores(s, Some(tt_buf)))
            .collect()
    }
}

#[cfg(not(feature = "parallel"))]
fn solve_worlds(
    worlds: &[GameState],
    _parallel: bool,
    tt_buf: &mut crate::solver::TtBuf,
) -> Vec<crate::solver::SolveScores> {
    worlds
        .iter()
        .map(|s| solve_with_scores(s, Some(tt_buf)))
        .collect()
}

/// Convenience wrapper that creates a temporary IsDdSearch without beliefs.
pub fn is_dd_search(state: &GameState, config: &IsDdConfig, rng: &mut impl Rng) -> u8 {
    let mut search = IsDdSearch::new();
    search.search(state, config, rng)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rollout::select_nth_bit;
    use crate::state::Phase;

    fn random_playing_state(rng: &mut impl Rng) -> Option<GameState> {
        let mut state = GameState::deal_random(0, rng);
        while state.phase == Phase::Bidding && !state.is_terminal() {
            let legal = state.legal_actions();
            let count = legal.count_ones();
            let idx = rng.gen_range(0..count);
            let action = select_nth_bit(legal, idx);
            state.step(action);
        }
        if state.is_terminal() {
            None
        } else {
            Some(state)
        }
    }

    #[test]
    #[ignore]
    fn test_is_dd_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 5,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let action = is_dd_search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "IS-DD returned illegal action {}",
                    action
                );
                found += 1;
                if found >= 30 {
                    break;
                }
            }
        }
        assert!(found >= 10, "Not enough non-void deals to test");
    }

    #[test]
    #[ignore]
    fn test_is_dd_with_beliefs() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 3,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..50 {
            let state = GameState::deal_random(0, &mut rng);
            let mut search = IsDdSearch::new();
            search.init_deal(&state, 0, true);

            let mut current = state;
            while !current.is_terminal() {
                let player = current.current_player();
                let state_before = current;

                let action = if player == 0 {
                    search.search(&current, &config, &mut rng)
                } else {
                    let legal = current.legal_actions();
                    let count = legal.count_ones();
                    let idx = rng.gen_range(0..count);
                    select_nth_bit(legal, idx)
                };

                let legal = current.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Illegal action {} by player {}",
                    action,
                    player
                );

                search.record_action(&state_before, player, action);
                current.step(action);
                found += 1;

                if found >= 100 {
                    break;
                }
            }
            if found >= 100 {
                break;
            }
        }
        assert!(found >= 20, "Not enough actions played");
    }

    #[test]
    #[ignore]
    fn test_is_dd_works_during_bidding() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 3,
            ..Default::default()
        };
        let state = GameState::deal_random(0, &mut rng);
        assert_eq!(state.phase, Phase::Bidding);

        let mut search = IsDdSearch::new();
        search.init_deal(&state, state.current_player(), true);

        let action = search.search(&state, &config, &mut rng);
        let legal = state.legal_actions();
        assert!(
            legal & (1u64 << action) != 0,
            "IS-DD returned illegal bid action {}",
            action
        );
    }

    #[test]
    #[ignore]
    fn test_is_dd_reusable() {
        let mut rng = rand::thread_rng();
        let mut search = IsDdSearch::new();
        let config = IsDdConfig {
            determinizations: 3,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..20 {
            if let Some(state) = random_playing_state(&mut rng) {
                search.reset();
                let action = search.search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(legal & (1u64 << action) != 0);
                found += 1;
            }
            if found >= 10 {
                break;
            }
        }
        assert!(found >= 5);
    }

    #[test]
    #[ignore]
    fn test_is_dd_search_with_stats() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 5,
            ..Default::default()
        };

        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let mut search = IsDdSearch::new();
                let result = search.search_with_stats(&state, &config, &mut rng);

                assert!(result.determinizations > 0);
                assert!(!result.card_scores.is_empty());

                // Best action must be legal
                let legal = state.legal_actions();
                assert!(legal & (1u64 << result.best_action) != 0);

                // All scores should be in valid range. The range is the
                // objective's, not a constant: `DealScore` (the default) yields
                // a signed deal-score margin, `CardPoints` an NS card total.
                let (lo, hi) = match config.objective {
                    PlayObjective::CardPoints => (0.0, 252.0),
                    // Borne large : surcontré capot réussi vaut 252 + 250×3 +
                    // belote = 1022 pour le preneur, zéro en face.
                    PlayObjective::DealScore => (-1100.0, 1100.0),
                };
                for &(card, avg) in &result.card_scores {
                    assert!(legal & (1u64 << card) != 0);
                    assert!(avg >= lo && avg <= hi, "avg={} hors [{lo}, {hi}]", avg);
                }

                found += 1;
                if found >= 20 {
                    break;
                }
            }
        }
        assert!(found >= 10);
    }

    #[cfg(feature = "parallel")]
    #[test]
    #[ignore]
    fn test_parallel_returns_legal_action() {
        let mut rng = rand::thread_rng();
        let config = IsDdConfig {
            determinizations: 5,
            parallel: true,
            ..Default::default()
        };
        let mut found = 0;
        for _ in 0..100 {
            if let Some(state) = random_playing_state(&mut rng) {
                let mut search = IsDdSearch::new();
                let action = search.search(&state, &config, &mut rng);
                let legal = state.legal_actions();
                assert!(
                    legal & (1u64 << action) != 0,
                    "Parallel IS-DD returned illegal action {}",
                    action
                );
                found += 1;
                if found >= 20 {
                    break;
                }
            }
        }
        assert!(found >= 10, "Not enough non-void deals to test");
    }

    /// Parallel and sequential solving must agree exactly: world generation is
    /// RNG-driven, so we seed identically and only flip `parallel`. DD is exact
    /// and the aggregation reduces in a fixed order, so `card_scores` must match.
    #[cfg(feature = "parallel")]
    #[test]
    #[ignore]
    fn test_parallel_matches_sequential() {
        use rand::SeedableRng;
        let mut src = rand::rngs::StdRng::seed_from_u64(12345);
        let mut checked = 0;
        for _ in 0..200 {
            let Some(state) = random_playing_state(&mut src) else { continue };
            let seq = {
                let mut rng = rand::rngs::StdRng::seed_from_u64(777);
                let cfg = IsDdConfig { determinizations: 12, parallel: false, ..Default::default() };
                IsDdSearch::new().search_with_stats(&state, &cfg, &mut rng)
            };
            let par = {
                let mut rng = rand::rngs::StdRng::seed_from_u64(777);
                let cfg = IsDdConfig { determinizations: 12, parallel: true, ..Default::default() };
                IsDdSearch::new().search_with_stats(&state, &cfg, &mut rng)
            };
            assert_eq!(seq.best_action, par.best_action, "best_action diverged");
            assert_eq!(seq.card_scores, par.card_scores, "card_scores diverged");
            checked += 1;
            if checked >= 15 {
                break;
            }
        }
        assert!(checked >= 5, "Not enough non-void deals to test");
    }

    /// Le barème, vu depuis les points cartes : plate sous le seuil, pente 2
    /// au-dessus, et une marche de `4V` entre les deux.
    ///
    /// C'est l'invariant qui justifie tout `PlayObjective::DealScore`. S'il
    /// tombe, l'objectif « score de donne » ne décrit plus le barème et il vaut
    /// mieux revenir aux points cartes qu'optimiser une fiction.
    #[test]
    fn deal_score_is_flat_below_the_contract_and_jumps_at_it() {
        use crate::state::Contract;
        // Contrat à 100 pour N-S, sans coinche, sans belote.
        let contract = Contract { trump: 0, value: 10, team: 0, coinche: 0 };
        let delta = |x: i16| -> i16 {
            let s = deal_score_from_card_points(&contract, [x, TOTAL_PTS - x], [0, 0], false);
            s.scores[0] - s.scores[1]
        };

        // Sous le seuil : strictement constant. Un point carte de plus ne vaut
        // rien — c'est le coup « au hasard » quand la chute est acquise.
        let below: Vec<i16> = (0..100).step_by(7).map(|x| delta(x as i16)).collect();
        assert!(below.windows(2).all(|w| w[0] == w[1]),
                "la zone de chute doit être plate, vu : {below:?}");
        assert_eq!(below[0], -(TOTAL_PTS + 100), "chute = -(162 + V)");

        // Au-dessus : pente 2 par point carte.
        for x in [100i16, 110, 130, 162] {
            assert_eq!(delta(x), 2 * x + 100 - TOTAL_PTS, "pente 2 attendue en x={x}");
        }

        // La marche vaut 4V.
        assert_eq!(delta(100) - delta(99), 4 * 100);
    }

    /// La belote **déplace le seuil**, elle n'ajoute pas 20 points au bout.
    ///
    /// `scoring.rs` la compte dans `taker_total` pour décider de la réussite,
    /// donc un preneur à 100 avec belote réussit à 80 points cartes là où il
    /// lui en faudrait 100 sans. C'est ce qui interdit de la rattraper après
    /// coup sur une moyenne : la corriger en aval déplacerait la valeur sans
    /// déplacer la marche.
    #[test]
    fn belote_moves_the_contract_threshold_not_just_the_total() {
        use crate::state::Contract;
        let contract = Contract { trump: 0, value: 10, team: 0, coinche: 0 }; // 100 pour N-S
        let made = |x: i16, belote: [i16; 2]| -> bool {
            let s = deal_score_from_card_points(&contract, [x, TOTAL_PTS - x], belote, false);
            // Chute ⇒ le preneur marque zéro (la défense prend tout).
            s.scores[0] > 0
        };

        assert!(!made(80, [0, 0]), "80 points cartes sans belote : chute à 100");
        assert!(made(80, [20, 0]), "80 + belote doit passer le contrat à 100");
        // Et la belote de l'adversaire ne sauve pas le preneur.
        assert!(!made(80, [0, 20]), "la belote adverse ne compte pas pour le preneur");
    }

    /// Un appelant qui ne dit rien joue pour le score de donne.
    #[test]
    fn default_objective_is_the_deal_score() {
        assert_eq!(IsDdConfig::default().objective, PlayObjective::DealScore);
    }

    /// L'objectif « score de donne » doit effectivement changer le classement
    /// des cartes quelque part — sinon le drapeau ne sert à rien.
    ///
    /// On ne teste pas *quelle* carte est choisie (elle dépend de la donne),
    /// mais que les deux objectifs ne rendent pas systématiquement les mêmes
    /// valeurs : `card_scores` doit changer d'échelle, et l'ordre doit différer
    /// au moins une fois sur un échantillon de positions.
    #[test]
    fn deal_score_objective_reranks_some_positions() {
        use rand::SeedableRng;
        let mut src = rand::rngs::StdRng::seed_from_u64(20260803);
        let mut differ = 0;
        let mut checked = 0;
        for _ in 0..600 {
            let Some(mut state) = random_playing_state(&mut src) else { continue };
            // Milieu de donne : un solve y coûte des ordres de grandeur de moins
            // qu'à l'entame, *et* c'est là que le seuil du contrat devient
            // décidable — donc là où les deux objectifs ont une chance de
            // diverger. À l'entame ils sont presque toujours d'accord.
            while !state.is_terminal()
                && card_count(state.hands[state.current_player() as usize]) > 5
            {
                let legal = state.legal_actions();
                let idx = src.gen_range(0..legal.count_ones());
                state.step(select_nth_bit(legal, idx));
            }
            if state.is_terminal() || state.legal_actions().count_ones() < 2 {
                continue;
            }
            let run = |obj: PlayObjective| {
                let cfg = IsDdConfig {
                    determinizations: 24,
                    parallel: false,
                    early_termination: false,
                    objective: obj,
                    ..Default::default()
                };
                let mut rng = rand::rngs::StdRng::seed_from_u64(1234);
                let mut s = IsDdSearch::new();
                s.init_deal_with_config(&state, state.current_player(), &cfg);
                s.search_with_stats(&state, &cfg, &mut rng)
            };
            let pts = run(PlayObjective::CardPoints);
            let sco = run(PlayObjective::DealScore);
            // Mêmes mondes (même graine), donc toute différence vient de l'objectif.
            if pts.best_action != sco.best_action {
                differ += 1;
            }
            checked += 1;
            if checked >= 60 {
                break;
            }
        }
        assert!(checked >= 30, "pas assez de positions jouables");
        assert!(differ > 0,
                "les deux objectifs choisissent toujours la même carte sur {checked} positions : \
                 le drapeau ne change rien");
    }

    /// Une source lente ne doit pas se faire redemander un lot qu'on n'aura
    /// pas le temps de consommer.
    ///
    /// C'est la fin de donne mesurée : un aller-retour coûte ~164 ms de latence
    /// quasi fixe, le budget à deux cartes vaut 250 ms, et l'échéance n'était
    /// testée qu'en tête de boucle — d'où deux requêtes par recherche, 512
    /// mondes demandés, 283 résolus et 230 jetés. Le premier aller-retour doit
    /// toujours partir (sinon plus aucun monde appris), le second jamais.
    #[test]
    fn a_slow_source_is_asked_exactly_once_under_a_tight_deadline() {
        use rand::SeedableRng;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        /// Source qui répond lentement et rend toujours de quoi remplir.
        struct SlowSource {
            calls: Arc<AtomicU32>,
            delay: Duration,
        }
        impl crate::worlds::WorldSource for SlowSource {
            fn name(&self) -> &'static str {
                "slow-test"
            }
            fn init_deal(&mut self, _state: &GameState, _observer: u8) {}
            fn observe(&mut self, _s: &GameState, _p: u8, _a: u8) {}
            fn worlds(
                &mut self,
                state: &GameState,
                observer: u8,
                n: usize,
                rng: &mut dyn rand::RngCore,
            ) -> Result<Vec<World>, crate::agent::AgentError> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(self.delay);
                struct Ad<'a>(&'a mut dyn rand::RngCore);
                impl rand::RngCore for Ad<'_> {
                    fn next_u32(&mut self) -> u32 { self.0.next_u32() }
                    fn next_u64(&mut self) -> u64 { self.0.next_u64() }
                    fn fill_bytes(&mut self, d: &mut [u8]) { self.0.fill_bytes(d) }
                    fn try_fill_bytes(&mut self, d: &mut [u8]) -> Result<(), rand::Error> {
                        self.0.try_fill_bytes(d)
                    }
                }
                let mut r = Ad(rng);
                Ok((0..n)
                    .filter_map(|_| determinize_greedy(state, observer, &mut r).map(|s| s.hands))
                    .collect())
            }
        }

        // Une position de **fin** de donne : les solves y coûtent des
        // microsecondes, donc la file se vide bien avant l'échéance et c'est
        // le second aller-retour — et lui seul — qui décide du sort du budget.
        // Sur une donne complète le solveur mangerait l'échéance tout seul et
        // le test ne prouverait rien.
        let mut src_rng = rand::rngs::StdRng::seed_from_u64(31337);
        let state = loop {
            let Some(mut s) = random_playing_state(&mut src_rng) else { continue };
            while !s.is_terminal()
                && card_count(s.hands[s.current_player() as usize]) > 3
            {
                let legal = s.legal_actions();
                let idx = src_rng.gen_range(0..legal.count_ones());
                s.step(select_nth_bit(legal, idx));
            }
            if !s.is_terminal() && s.legal_actions().count_ones() >= 2 {
                break s;
            }
        };
        let cards_left = card_count(state.hands[state.current_player() as usize]);

        let calls = Arc::new(AtomicU32::new(0));
        // Latence 60 ms. Le budget est mis à l'échelle par les cartes restantes
        // (`ms * cards_left / 8`), donc on vise ~100 ms réels : le premier
        // aller-retour tient, le second déborderait de 20 ms.
        let mut source = SlowSource {
            calls: Arc::clone(&calls),
            delay: Duration::from_millis(60),
        };
        let cfg = IsDdConfig {
            time_limit_ms: Some(100 * 8 / cards_left.max(1)),
            world_batch: 4,
            early_termination: false,
            parallel: false,
            ..Default::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut search = IsDdSearch::new();
        search.init_deal_with_config(&state, state.current_player(), &cfg);
        let r = search
            .search_with_source(&state, &cfg, &mut rng, &mut source)
            .expect("la source ne renvoie pas d'erreur");

        assert_eq!(
            calls.load(Ordering::Relaxed),
            1,
            "un seul aller-retour attendu, {} observés (position à {} cartes)",
            calls.load(Ordering::Relaxed),
            cards_left
        );
        assert!(r.worlds.injected > 0, "le premier lot doit avoir servi");
        assert_eq!(r.source.discarded, 0, "rien ne doit être jeté");
    }

    /// Sous pression de calcul, un plancher fait payer la **latence** et non la
    /// **force de jeu**.
    ///
    /// C'est un choix de politique, et c'est pour ça qu'il se teste : sans
    /// plancher, une source lente fait rendre à Dédé une réponse à l'heure
    /// fondée sur une poignée de mondes, et **rien ne le signale** — le joueur
    /// voit un coup arriver au rythme habituel, simplement moins bon. Avec
    /// plancher, la même position rend le nombre de mondes demandé et met plus
    /// longtemps. Le test tourne les deux configurations sur la même position et
    /// la même graine, donc seul le plancher les sépare.
    #[test]
    fn a_floor_buys_worlds_with_latency_instead_of_giving_up_strength() {
        use rand::SeedableRng;
        use std::sync::atomic::{AtomicU32, Ordering};
        use std::sync::Arc;

        struct SlowSource {
            calls: Arc<AtomicU32>,
            delay: Duration,
        }
        impl crate::worlds::WorldSource for SlowSource {
            fn name(&self) -> &'static str {
                "slow-floor-test"
            }
            fn init_deal(&mut self, _state: &GameState, _observer: u8) {}
            fn observe(&mut self, _s: &GameState, _p: u8, _a: u8) {}
            fn worlds(
                &mut self,
                state: &GameState,
                observer: u8,
                n: usize,
                rng: &mut dyn rand::RngCore,
            ) -> Result<Vec<World>, crate::agent::AgentError> {
                self.calls.fetch_add(1, Ordering::Relaxed);
                std::thread::sleep(self.delay);
                struct Ad<'a>(&'a mut dyn rand::RngCore);
                impl rand::RngCore for Ad<'_> {
                    fn next_u32(&mut self) -> u32 { self.0.next_u32() }
                    fn next_u64(&mut self) -> u64 { self.0.next_u64() }
                    fn fill_bytes(&mut self, d: &mut [u8]) { self.0.fill_bytes(d) }
                    fn try_fill_bytes(&mut self, d: &mut [u8]) -> Result<(), rand::Error> {
                        self.0.try_fill_bytes(d)
                    }
                }
                let mut r = Ad(rng);
                Ok((0..n)
                    .filter_map(|_| determinize_greedy(state, observer, &mut r).map(|s| s.hands))
                    .collect())
            }
        }

        // Fin de donne : les solves y coûtent des microsecondes, donc c'est bien
        // la source — et elle seule — qui consomme l'échéance.
        let mut src_rng = rand::rngs::StdRng::seed_from_u64(90210);
        let state = loop {
            let Some(mut s) = random_playing_state(&mut src_rng) else { continue };
            while !s.is_terminal() && card_count(s.hands[s.current_player() as usize]) > 3 {
                let legal = s.legal_actions();
                let idx = src_rng.gen_range(0..legal.count_ones());
                s.step(select_nth_bit(legal, idx));
            }
            if !s.is_terminal() && s.legal_actions().count_ones() >= 2 {
                break s;
            }
        };
        let cards_left = card_count(state.hands[state.current_player() as usize]);
        const FLOOR: u32 = 24;

        let run = |min_worlds: Option<u32>| {
            let calls = Arc::new(AtomicU32::new(0));
            let mut source = SlowSource {
                calls: Arc::clone(&calls),
                delay: Duration::from_millis(25),
            };
            let cfg = IsDdConfig {
                // Echeance volontairement plus courte qu'un seul aller-retour.
                time_limit_ms: Some(10 * 8 / cards_left.max(1)),
                world_batch: 8,
                early_termination: false,
                parallel: false,
                min_worlds,
                ..Default::default()
            };
            let mut rng = rand::rngs::StdRng::seed_from_u64(7);
            let mut search = IsDdSearch::new();
            search.init_deal_with_config(&state, state.current_player(), &cfg);
            let t0 = Instant::now();
            let r = search
                .search_with_source(&state, &cfg, &mut rng, &mut source)
                .expect("la source ne renvoie pas d'erreur");
            (r.determinizations, t0.elapsed())
        };

        let (dets_free, t_free) = run(None);
        let (dets_floor, t_floor) = run(Some(FLOOR));

        assert!(
            dets_floor >= FLOOR,
            "le plancher n'a pas tenu : {dets_floor} mondes resolus pour un plancher de {FLOOR}"
        );
        assert!(
            dets_free < FLOOR,
            "l'echeance aurait du couper sous le plancher sans lui ({dets_free} mondes) — \
             sinon ce test ne prouve rien"
        );
        assert!(
            t_floor > t_free,
            "le plancher doit se payer en temps : {t_floor:?} contre {t_free:?}"
        );
    }

    /// Une source morte ne doit pas faire tourner le plancher dans le vide.
    ///
    /// `source_dry` (lot vide = le sampler ne *peut* plus produire) garde son
    /// repli local, donc le plancher reste atteignable en mondes échantillonnés
    /// sur place. C'est ce qui rend le plancher sûr : il ne peut pas boucler
    /// indéfiniment tant que la déterminisation locale aboutit — et quand elle
    /// n'aboutit pas, `STUCK_ROUNDS` coupe.
    ///
    /// ⚠️ La contrepartie, et elle compte pour la prod : dans **ce** cas précis
    /// le plancher est rempli avec des mondes uniformes, pas des mondes playgen.
    /// Il garantit le *nombre*, pas la *provenance*. Un sidecar simplement lent
    /// n'est pas concerné (`source_dry` ne se déclenche que sur un lot vide, et
    /// sous le plancher on ne renonce jamais à l'aller-retour).
    #[test]
    fn a_dead_source_still_lets_the_floor_terminate() {
        use rand::SeedableRng;

        struct DeadSource;
        impl crate::worlds::WorldSource for DeadSource {
            fn name(&self) -> &'static str {
                "dead-test"
            }
            fn init_deal(&mut self, _state: &GameState, _observer: u8) {}
            fn observe(&mut self, _s: &GameState, _p: u8, _a: u8) {}
            fn worlds(
                &mut self,
                _state: &GameState,
                _observer: u8,
                _n: usize,
                _rng: &mut dyn rand::RngCore,
            ) -> Result<Vec<World>, crate::agent::AgentError> {
                Ok(Vec::new())
            }
        }

        let mut src_rng = rand::rngs::StdRng::seed_from_u64(1234);
        let state = loop {
            if let Some(s) = random_playing_state(&mut src_rng) {
                if !s.is_terminal() && s.legal_actions().count_ones() >= 2 {
                    break s;
                }
            }
        };
        const FLOOR: u32 = 24;
        let cfg = IsDdConfig {
            time_limit_ms: Some(1), // echeance immediate : seul le plancher decide
            min_worlds: Some(FLOOR),
            early_termination: false,
            parallel: false,
            ..Default::default()
        };
        let mut rng = rand::rngs::StdRng::seed_from_u64(7);
        let mut search = IsDdSearch::new();
        search.init_deal_with_config(&state, state.current_player(), &cfg);
        let t0 = Instant::now();
        let mut source = DeadSource;
        let r = search
            .search_with_source(&state, &cfg, &mut rng, &mut source)
            .expect("pas d'erreur");
        assert!(
            t0.elapsed() < Duration::from_secs(20),
            "la recherche ne s'est pas arretee ({:?})",
            t0.elapsed()
        );
        assert!(
            r.determinizations >= FLOOR,
            "plancher non tenu par le repli local : {} mondes",
            r.determinizations
        );
        assert!(r.best_action < 32, "une carte doit quand meme sortir");
    }

    /// Les poids de croyance sont calculés **paresseusement** — seulement quand
    /// la file de mondes est vide. Sans source, elle l'est toujours, donc les
    /// mondes doivent continuer à sortir pondérés (`Belief`), pas uniformes.
    ///
    /// C'est l'invariant que le passage au calcul paresseux pouvait casser en
    /// silence : un `weights` resté `None` ne fait pas d'erreur, il fait
    /// seulement échantillonner à plat, et rien ne le dirait.
    #[test]
    fn lazy_weights_still_reach_the_sampler_without_a_source() {
        use rand::SeedableRng;
        let mut src = rand::rngs::StdRng::seed_from_u64(4242);
        let mut checked = 0;
        for _ in 0..200 {
            let Some(state) = random_playing_state(&mut src) else { continue };
            if state.legal_actions().count_ones() < 2 {
                continue; // coup forcé : sortie anticipée, aucun monde demandé
            }
            let mut rng = rand::rngs::StdRng::seed_from_u64(99);
            let cfg = IsDdConfig {
                determinizations: 8,
                parallel: false,
                early_termination: false,
                ..Default::default()
            };
            let mut search = IsDdSearch::new();
            search.init_deal_with_config(&state, state.current_player(), &cfg);
            let r = search.search_with_stats(&state, &cfg, &mut rng);
            assert_eq!(r.worlds.injected, 0, "aucune source n'est branchée");
            assert!(r.worlds.total() > 0, "aucun monde généré");
            assert!(
                r.worlds.belief > 0,
                "les poids de croyance n'ont pas atteint l'échantillonneur : \
                 {:?}",
                r.worlds
            );
            checked += 1;
            if checked >= 10 {
                break;
            }
        }
        assert!(checked >= 5, "Not enough non-void deals to test");
    }
}
