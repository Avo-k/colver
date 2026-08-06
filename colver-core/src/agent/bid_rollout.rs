//! Annoncer en jouant la donne : la simulation de la page Annonces, comme
//! politique d'enchère.
//!
//! # Ce que c'est
//!
//! À son tour de parole, ce bidder prend les annonces candidates, en simule
//! chacune jusqu'au bout — mondes échantillonnés, reste de l'enchère par un
//! réseau de référence, jeu par DouDou50 — et annonce celle dont l'**espérance
//! d'écart de score** est la meilleure. C'est le calcul que la page Annonces
//! affiche, tourné en décision au lieu d'un tableau.
//!
//! Il n'apprend rien : il n'a pas de poids à lui. Ce qu'il vaut est ce que
//! valent son réseau de référence et son joueur de cartes, plus le budget de
//! simulation qu'on lui donne.
//!
//! # Ce que coûte une simulation, et ce que ça implique
//!
//! Un monde simulé, c'est **32 passes avant de DouDou50**, soit ~178 µs pièce
//! ([docs/BENCH.md](../../../docs/BENCH.md)) — ~6 ms de plancher, ~11 ms
//! mesurés. Le tirage du monde, lui, est négligeable devant ça, y compris quand
//! il vient du GPU. **Donc playgen améliore la qualité des mondes, il ne réduit
//! pas le coût** : c'est DouDou50 qui paie, une carte à la fois, sur CPU.
//!
//! D'où le seul vrai levier de latence : `parallel`, qui éclate les
//! déroulements sur rayon. 3 000 déroulements font 33 s de CPU et ~1 s au mur
//! sur 32 cœurs. C'est la différence entre « utilisable sur le site » et « non ».
//! En arène ça ne change rien : les matchs y sont déjà parallèles, les cœurs
//! sont pris, et le total de CPU est ce qu'il est.
//!
//! # Trois choix de conception, et pourquoi
//!
//! **Les mondes sont partagés entre les candidates** (« common random
//! numbers ») : on tire `sims` mondes une fois, puis on rejoue *chaque*
//! candidate sur *chacun* d'eux. Comparer deux annonces sur des tirages
//! indépendants ajoute deux fois la variance de tirage à une différence qui,
//! elle, est petite. La boucle est donc « pour chaque monde, pour chaque
//! candidate », et non l'inverse : un budget épuisé en cours de route laisse
//! toutes les candidates au **même** nombre de mondes, donc comparables.
//!
//! **Les mondes viennent de playgen** ([`BidWorlds`]), pas d'un mélange
//! uniforme. Ce n'est pas un raffinement : un tirage uniforme donne une main au
//! hasard au siège qui vient d'annoncer 100♥, donc **le monde contredit
//! l'enchère sous laquelle il est tiré**, et le réseau fera passer ce siège
//! dans la suite de la simulation. Le contrat simulé est alors systématiquement
//! moins disputé qu'il ne le sera — biais de sur-annonce, croissant avec la
//! longueur de l'enchère déjà entendue. playgen v2 complète l'enchère avec sa
//! propre tête d'annonces, donc les mains qu'il invente **expliquent les
//! annonces déjà entendues**.
//!
//! **La liste des candidates est courte, et construite, pas triée.** Une
//! première parole a jusqu'à 37 annonces légales ; à ~11 ms le monde, les
//! balayer toutes est hors de prix. Mais un simple top-K au Q du réseau est
//! pire qu'inutile — voir [`CandidateMode`].
//!
//! # Ce que la mesure existante dit de ses chances
//!
//! `scripts/analysis/quick_bid_spread.py` (2026-08-06) a mesuré la dispersion
//! de cette même simulation : **σ ≈ 310 à 370 points par monde**, et l'écart
//! *vrai* entre deux annonces voisines (X contre X+10, même couleur) vaut
//! **quelques points**, compatible avec zéro à 600 simulations. Le contrôle de
//! `bench_bid_rollout` le confirme sur le bidder lui-même : à 20 mondes, deux
//! tirages de la même décision s'accordent **2 fois sur 20** pour un hasard à
//! 25 %. À ce budget le bot ne corrige pas son a priori, il le randomise.
//!
//! Ce n'est pas rédhibitoire, parce que ce n'est pas là que se jouent les
//! points : passer ou parler, et dans quelle couleur, sont des décisions à
//! grande amplitude, et ce sont elles que [`CandidateMode::Probe`] met en
//! concurrence.

use std::sync::Arc;
use std::time::Instant;

use rand::rngs::StdRng;
use rand::SeedableRng;

use crate::determinize::determinize;
use crate::playgen::analysis::PlaygenAnalyst;
use crate::playgen::infer::PlaygenModel;
use crate::state::{GameState, Phase};
use crate::worlds::{FallbackPolicy, SidecarWorldSource, World};

use super::bid::BidNetPolicy;
use super::dmc::DmcPlayer;
use super::models::{BidWeights, DmcWeights};
use super::{AgentError, BidPolicy, CardPlayer, Decision, MatchContext, Stats};

// ══════════════════════════════════════════════════════════════════════
//  D'où viennent les mondes
// ══════════════════════════════════════════════════════════════════════

/// Échantillonneur de donnes complètes pour une position d'**enchère**.
///
/// Volontairement séparé de [`crate::worlds::WorldSource`], qui sert la phase de
/// jeu et rend les cartes qu'il **reste** à chaque siège. Ici rien n'a été joué :
/// un monde est quatre mains complètes, et la route du sidecar n'est pas la même
/// (`/auction_deals` et non `/play_worlds`).
pub enum BidWorlds {
    /// Les 24 cartes invisibles au hasard. Aveugle à l'enchère entendue —
    /// c'est ce que fait `annonces_doudou` côté web, et c'est le biais décrit
    /// en tête de module.
    Uniform,
    /// playgen sur le sidecar GPU. Le défaut quand une URL est disponible :
    /// un lot de 512 mondes y coûte à peu près ce que coûte un seul.
    Sidecar(Box<SidecarWorldSource>),
    /// playgen en CPU, dans le processus. Correct mais lent — utile sans GPU.
    Local(Box<PlaygenAnalyst>),
}

impl BidWorlds {
    pub fn name(&self) -> &'static str {
        match self {
            BidWorlds::Uniform => "uniform",
            BidWorlds::Sidecar(_) => "playgen_sidecar",
            BidWorlds::Local(_) => "playgen_cpu",
        }
    }

    fn init_deal(&mut self, state: &GameState, observer: u8) {
        match self {
            BidWorlds::Uniform => {}
            BidWorlds::Sidecar(s) => {
                use crate::worlds::WorldSource;
                s.init_deal(state, observer)
            }
            BidWorlds::Local(a) => a.init_deal(state, observer),
        }
    }

    fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        match self {
            BidWorlds::Uniform => {}
            BidWorlds::Sidecar(s) => {
                use crate::worlds::WorldSource;
                s.observe(state_before, player, action)
            }
            BidWorlds::Local(a) => a.observe(state_before, player, action),
        }
    }

    /// Jusqu'à `n` donnes complètes. Rendre moins que `n` est permis.
    fn sample(
        &mut self,
        state: &GameState,
        observer: u8,
        n: usize,
        rng: &mut StdRng,
    ) -> Result<Vec<World>, AgentError> {
        match self {
            BidWorlds::Uniform => Ok(uniform_deals(state, observer, n, rng)),
            BidWorlds::Sidecar(s) => s.auction_deals(observer, n),
            BidWorlds::Local(a) => Ok(a.auction_deals(state, n, 1.0, rng)),
        }
    }
}

fn uniform_deals(state: &GameState, observer: u8, n: usize, rng: &mut StdRng) -> Vec<World> {
    (0..n).filter_map(|_| determinize(state, observer, rng).map(|s| s.hands)).collect()
}

/// Un monde d'enchère est valide s'il rend sa main à l'observateur et donne 8
/// cartes à chacun. Le sampler ne voit pas la main qu'il doit préserver, donc
/// ce contrôle est à la charge de l'appelant — et une position d'enchère n'a ni
/// coupe révélée ni belote annoncée à vérifier en plus.
fn valid_deal(hands: &World, state: &GameState, observer: u8) -> bool {
    hands[observer as usize] == state.hands[observer as usize]
        && hands.iter().all(|h| h.count_ones() == 8)
        && hands.iter().fold(0u32, |a, h| a | h).count_ones() == 32
}

// ══════════════════════════════════════════════════════════════════════
//  Quelles annonces on met en concurrence
// ══════════════════════════════════════════════════════════════════════

/// Ce n'est pas un détail de budget : c'est **le** paramètre qui décide si la
/// simulation peut répondre. Elle sépare mal les annonces voisines, donc lui
/// donner quatre paliers adjacents de la même couleur — ce que fait
/// naturellement un top-K au Q — c'est lui poser la seule question à laquelle
/// elle ne sait pas répondre, en laissant de côté « passer ou parler » et « dans
/// quelle couleur », qui sont les décisions à grande amplitude.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum CandidateMode {
    /// Les `candidates` meilleures annonces au Q du réseau.
    ///
    /// Gardé pour l'A/B et parce que c'est le comportement naïf qu'on veut
    /// pouvoir mesurer. Observé tel quel : `120♦ 110♦ 100♦ 90♦`, sans passe.
    Top,
    /// **Le sondage.** Le réseau dit où chercher, la simulation regarde autour :
    ///
    /// 1. l'annonce du réseau ;
    /// 2. **passe** — l'alternative dont l'écart au reste est le plus grand,
    ///    donc la seule que ce budget puisse trancher à coup sûr ;
    /// 3. la meilleure annonce de la **deuxième couleur** du réseau, même loin
    ///    dans son classement : changer de couleur est une décision à grande
    ///    amplitude, et un top-K ne la propose presque jamais ;
    /// 4. les voisines dans la couleur : −10 et +10 si les deux sont légales,
    ///    sinon +10 et +20 — on explore toujours deux paliers, l'enchère en
    ///    cours décide seulement lesquels.
    #[default]
    Probe,
}

#[derive(Clone, Copy, Debug)]
pub struct RolloutBidConfig {
    /// Mondes tirés par décision. Chaque candidate est jouée sur tous.
    pub sims: u32,
    /// Plafond du nombre de candidates. 0 = pas de plafond.
    pub candidates: usize,
    pub mode: CandidateMode,
    pub objective: RolloutObjective,
    /// Échéance par décision, en ms. 0 = pas d'horloge, on fait les `sims`.
    pub time_ms: u32,
    /// Éclater les déroulements sur rayon. Ne vaut que **1,4×** — DouDou50 est
    /// limité par la bande passante mémoire, pas par le calcul. Voir `gpu`.
    pub parallel: bool,
    /// Dérouler les mondes **en lot sur GPU** ([`crate::gpu_rollout`]).
    ///
    /// C'est le vrai levier : grouper les déroulements lit les poids une fois
    /// pour tout le lot au lieu d'une fois par carte, ce qui fait passer le
    /// calcul de « limité par la mémoire » à « limité par le calcul ».
    /// Demande la feature `dmc_train` (candle + CUDA) ; sans elle, retombe sur
    /// le chemin CPU.
    pub gpu: bool,
    /// Que faire si l'échantillonneur ne répond pas.
    pub fallback: FallbackPolicy,
}

impl Default for RolloutBidConfig {
    fn default() -> Self {
        RolloutBidConfig {
            sims: 20,
            candidates: 4,
            mode: CandidateMode::Probe,
            objective: RolloutObjective::Margin,
            time_ms: 0,
            parallel: false,
            gpu: false,
            fallback: FallbackPolicy::Strict,
        }
    }
}

/// Ce que la simulation maximise.
#[derive(Clone, Copy, PartialEq, Eq, Debug, Default)]
pub enum RolloutObjective {
    /// Espérance de l'écart de score de donne (mon camp − l'autre), en points
    /// marqués. Le défaut : c'est la quantité qu'un match à 2000 points
    /// accumule.
    #[default]
    Margin,
    /// Fraction des mondes où mon camp marque plus que l'autre. Une donne
    /// passée compte comme nulle, comme dans `game_loop::play_match`.
    ///
    /// Maximiser ça n'est pas la même chose que maximiser l'espérance : le
    /// barème est très asymétrique (une chute vaut 162 + contrat à l'adversaire),
    /// donc un contrat qui passe souvent mais coûte cher quand il tombe est
    /// meilleur ici et pire là. Offert parce que c'est le chiffre-phare de la
    /// page Annonces, pas parce que c'est le bon objectif.
    WinRate,
}

/// La couleur d'une annonce ordinaire (1..=36), `None` pour passe, capot et
/// coinche. Encodage : `value_idx × 4 + suit_idx + 1`.
fn bid_suit(action: u8) -> Option<u8> {
    (1..=36).contains(&action).then(|| (action - 1) % 4)
}

/// Les annonces mises en concurrence, dans l'ordre de départage : la première
/// gagne les égalités, donc c'est celle du réseau.
///
/// `prior` est la liste du réseau, triée par Q décroissant, en espace physique.
fn shortlist(cfg: &RolloutBidConfig, prior: &[(u8, f32)], legal: u64) -> Vec<u8> {
    let mut out: Vec<u8> = Vec::new();
    let push = |a: u8, out: &mut Vec<u8>| {
        if legal & (1u64 << a) != 0 && !out.contains(&a) {
            out.push(a);
        }
    };

    match cfg.mode {
        CandidateMode::Top => {
            for (a, _) in prior {
                push(*a, &mut out);
            }
        }
        CandidateMode::Probe => {
            let best = prior.first().map(|(a, _)| *a).unwrap_or(0);
            push(best, &mut out); // 1. l'annonce du réseau
            push(0, &mut out); // 2. passe

            // 3. la meilleure annonce de la deuxième couleur du réseau. On la
            //    cherche loin dans le classement s'il le faut : un top-K ne la
            //    propose presque jamais, et c'est une décision à grande
            //    amplitude, donc exactement ce que la simulation sait trancher.
            if let Some(s0) = bid_suit(best) {
                if let Some((a, _)) =
                    prior.iter().find(|(a, _)| bid_suit(*a).is_some_and(|s| s != s0))
                {
                    push(*a, &mut out);
                }
            }

            // 4. deux paliers dans la couleur : −10/+10 si les deux passent,
            //    sinon +10/+20. On explore toujours deux voisines ; l'enchère
            //    en cours décide seulement lesquelles.
            if (1..=36).contains(&best) {
                let step = |d: i16| {
                    let n = best as i16 + d;
                    ((1..=36).contains(&n) && legal & (1u64 << n) != 0).then_some(n as u8)
                };
                match (step(-4), step(4)) {
                    (Some(down), Some(up)) => {
                        push(down, &mut out);
                        push(up, &mut out);
                    }
                    (None, Some(up)) => {
                        push(up, &mut out);
                        if let Some(up2) = step(8) {
                            push(up2, &mut out);
                        }
                    }
                    (Some(down), None) => push(down, &mut out),
                    (None, None) => {}
                }
            }

            // Le réseau dit « passe », ou la liste est maigre : on complète par
            // ses meilleures, c'est-à-dire les portes d'entrée dans l'enchère.
            for (a, _) in prior {
                if cfg.candidates > 0 && out.len() >= cfg.candidates {
                    break;
                }
                push(*a, &mut out);
            }
        }
    }

    if cfg.candidates > 0 {
        out.truncate(cfg.candidates);
    }
    out
}

// ══════════════════════════════════════════════════════════════════════
//  La politique
// ══════════════════════════════════════════════════════════════════════

/// Les réseaux d'une simulation. Le parallélisme en veut un jeu par fil — un
/// réseau porte ses tampons de calcul, donc `evaluate` prend `&mut self`.
struct SimNets {
    bid: BidNetPolicy,
    play: DmcPlayer,
}

pub struct RolloutBidPolicy {
    /// Le réseau d'enchère qui présélectionne. Le même jeu de poids sert dans
    /// les simulations, où il parle pour les quatre sièges — `BidNetPolicy` lit
    /// le siège dans la position, donc une instance suffit par fil.
    prior: BidNetPolicy,
    nets: SimNets,
    /// De quoi instancier un jeu de réseaux de plus, par fil rayon.
    bid_weights: Arc<BidWeights>,
    dmc_weights: Arc<DmcWeights>,
    residual: bool,
    penalty: f32,
    score_aware: bool,
    canonical: bool,
    seed: u64,

    worlds: BidWorlds,
    cfg: RolloutBidConfig,
    rng: StdRng,
    seat: u8,
    /// Moteur de déroulement groupé. `None` quand `cfg.gpu` est faux, ou quand
    /// le chargement échoue — auquel cas on retombe sur le CPU **en le disant**
    /// à la construction, pas en silence à la première décision.
    #[cfg(feature = "dmc_train")]
    gpu: Option<crate::gpu_rollout::GpuRollout>,
    #[cfg(not(feature = "dmc_train"))]
    gpu: Option<()>,
}

impl RolloutBidPolicy {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        bid_weights: Arc<BidWeights>,
        dmc_weights: Arc<DmcWeights>,
        residual: bool,
        penalty: f32,
        score_aware: bool,
        canonical: bool,
        worlds: BidWorlds,
        cfg: RolloutBidConfig,
        seat: u8,
        seed: u64,
    ) -> Self {
        let mk_bid = || {
            BidNetPolicy::new(
                bid_weights.clone(),
                penalty,
                // Température 0 : un évaluateur qui échantillonne ajouterait du
                // bruit à une mesure déjà dominée par le tirage des mondes.
                0.0,
                score_aware,
                canonical,
                seed,
            )
        };
        #[cfg(feature = "dmc_train")]
        let gpu = cfg.gpu.then(|| {
            match crate::gpu_rollout::GpuRollout::new(&dmc_weights, residual) {
                Ok(g) => {
                    if !g.device_is_cuda() {
                        eprintln!(
                            "bid rollout : CUDA indisponible, le déroulement groupé tourne sur CPU"
                        );
                    }
                    Some(g)
                }
                Err(e) => {
                    eprintln!("bid rollout : moteur GPU indisponible ({e}), repli sur le CPU");
                    None
                }
            }
        }).flatten();
        #[cfg(not(feature = "dmc_train"))]
        let gpu = None;

        RolloutBidPolicy {
            prior: mk_bid(),
            nets: SimNets { bid: mk_bid(), play: DmcPlayer::new(dmc_weights.clone(), residual) },
            gpu,
            bid_weights,
            dmc_weights,
            residual,
            penalty,
            score_aware,
            canonical,
            seed,
            worlds,
            cfg,
            rng: StdRng::seed_from_u64(seed),
            seat,
        }
    }

    fn fresh_nets(&self) -> SimNets {
        SimNets {
            bid: BidNetPolicy::new(
                self.bid_weights.clone(),
                self.penalty,
                0.0,
                self.score_aware,
                self.canonical,
                self.seed,
            ),
            play: DmcPlayer::new(self.dmc_weights.clone(), self.residual),
        }
    }
}

/// Une simulation : on force `forced`, puis l'enchère et le jeu se déroulent
/// tout seuls. Rend l'écart de score de donne vu de `team`.
///
/// `ctx` est cloné et suivi dans la simulation — c'est ce qui donne au réseau
/// d'enchère son historique d'annonces et à DouDou50 son suivi de coupes. Le
/// score de match voyage avec, donc les simulations sont score-aware comme le
/// vrai jeu.
///
/// Fonction libre et non méthode : le parallélisme a besoin de l'appeler avec
/// un jeu de réseaux propre au fil, sans emprunter la politique.
fn rollout(
    nets: &mut SimNets,
    world: &World,
    state: &GameState,
    ctx: &MatchContext,
    forced: u8,
    team: usize,
) -> Result<i32, AgentError> {
    let mut sim = *state;
    sim.hands = *world;
    let mut track = ctx.clone();

    track.track(&sim, forced);
    sim.step(forced);

    while !sim.is_terminal() {
        let before = sim;
        let action = match before.phase {
            Phase::Bidding => nets.bid.decide(&before, &track)?.action,
            _ => nets.play.decide(&before, &track)?.action,
        };
        track.track(&before, action);
        sim.step(action);
    }

    let score = sim.deal_score().scores;
    Ok(score[team] as i32 - score[1 - team] as i32)
}

impl BidPolicy for RolloutBidPolicy {
    fn init_deal(&mut self, state: &GameState) {
        self.worlds.init_deal(state, self.seat);
    }

    fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        self.worlds.observe(state_before, player, action);
    }

    fn decide(&mut self, state: &GameState, ctx: &MatchContext) -> Result<Decision, AgentError> {
        let seat = state.current_player();
        let team = GameState::player_team(seat) as usize;
        let legal = state.legal_actions();

        // Le réseau donne l'a priori, trié par Q décroissant. Il sert aussi de
        // repli : sans candidate à départager, ou sans monde à tirer, sa
        // réponse est la nôtre.
        let prior = self.prior.decide(state, ctx)?;
        let fall_back = |p: Decision| {
            Ok(Decision { action: p.action, stats: Stats { source: "bid_rollout", ..p.stats } })
        };
        if legal.count_ones() <= 1 || self.cfg.sims == 0 {
            return fall_back(prior);
        }

        let shortlist = shortlist(&self.cfg, &prior.stats.candidates, legal);
        if shortlist.len() <= 1 {
            return fall_back(prior);
        }

        let start = Instant::now();

        // Les mondes sont tirés une fois et rejoués par toutes les candidates.
        // Un échantillonneur qui rend moins que demandé n'est pas une erreur ;
        // un échantillonneur muet en est une, sauf repli explicite.
        let sampled = match self.worlds.sample(state, seat, self.cfg.sims as usize, &mut self.rng) {
            Ok(w) => w,
            Err(e) => match self.cfg.fallback {
                FallbackPolicy::Strict => return Err(e),
                FallbackPolicy::Uniform => Vec::new(),
            },
        };
        let mut worlds: Vec<World> =
            sampled.into_iter().filter(|w| valid_deal(w, state, seat)).collect();
        // Compléter en uniforme plutôt que chercher moins de mondes : le
        // sampler peut légitimement rendre moins que demandé, et un budget qui
        // fond en silence ferait varier la précision sans le dire.
        if worlds.len() < self.cfg.sims as usize {
            let missing = self.cfg.sims as usize - worlds.len();
            worlds.extend(uniform_deals(state, seat, missing, &mut self.rng));
        }
        if worlds.is_empty() {
            return fall_back(prior);
        }

        let (totals, wins, done) = if self.cfg.gpu && self.gpu.is_some() {
            self.run_gpu(&worlds, state, ctx, &shortlist, team)?
        } else if self.cfg.parallel {
            self.run_parallel(&worlds, state, ctx, &shortlist, team)?
        } else {
            self.run_sequential(&worlds, state, ctx, &shortlist, team, start)?
        };

        let n = done.max(1) as f32;
        let mut scored: Vec<(u8, f32)> = shortlist
            .iter()
            .enumerate()
            .map(|(i, &a)| {
                let v = match self.cfg.objective {
                    RolloutObjective::Margin => totals[i] as f32 / n,
                    RolloutObjective::WinRate => wins[i] as f32 / n,
                };
                (a, v)
            })
            .collect();

        // Départage : à valeur égale, l'ordre du réseau. `shortlist` est déjà
        // dans cet ordre et on garde le premier maximum, donc il suffit de
        // choisir avant de trier pour l'affichage.
        let action = scored
            .iter()
            .copied()
            .reduce(|a, b| if b.1 > a.1 { b } else { a })
            .map(|(a, _)| a)
            .unwrap_or(prior.action);

        scored.sort_by(|a, b| b.1.total_cmp(&a.1));
        Ok(Decision {
            action,
            stats: Stats {
                source: "bid_rollout",
                candidates: scored,
                determinizations: done,
                elapsed_ms: start.elapsed().as_secs_f64() * 1000.0,
                ..Stats::default()
            },
        })
    }
}

type Tallies = (Vec<i64>, Vec<u32>, u32);

impl RolloutBidPolicy {
    /// Monde par monde, candidate par candidate — dans cet ordre, jamais
    /// l'inverse. Une échéance qui coupe entre deux candidates laisserait les
    /// premières avec un monde de plus que les dernières, et le classement
    /// lirait ce déséquilibre comme un écart de valeur.
    fn run_sequential(
        &mut self,
        worlds: &[World],
        state: &GameState,
        ctx: &MatchContext,
        shortlist: &[u8],
        team: usize,
        start: Instant,
    ) -> Result<Tallies, AgentError> {
        let mut totals = vec![0i64; shortlist.len()];
        let mut wins = vec![0u32; shortlist.len()];
        let mut done = 0u32;
        for world in worlds {
            for (i, &cand) in shortlist.iter().enumerate() {
                let margin = rollout(&mut self.nets, world, state, ctx, cand, team)?;
                totals[i] += margin as i64;
                if margin > 0 {
                    wins[i] += 1;
                }
            }
            done += 1;
            if self.cfg.time_ms > 0 && start.elapsed().as_millis() as u32 >= self.cfg.time_ms {
                break;
            }
        }
        Ok((totals, wins, done))
    }

    /// Le même calcul, éclaté sur rayon. **Pas d'échéance ici** : couper un lot
    /// parallèle laisserait un nombre de mondes différent par candidate, ce que
    /// la version séquentielle prend soin d'éviter. On fait les `sims`, ou rien.
    #[cfg(feature = "parallel")]
    fn run_parallel(
        &mut self,
        worlds: &[World],
        state: &GameState,
        ctx: &MatchContext,
        shortlist: &[u8],
        team: usize,
    ) -> Result<Tallies, AgentError> {
        use rayon::prelude::*;

        let k = shortlist.len();
        let per_world: Result<Vec<(Vec<i64>, Vec<u32>)>, AgentError> = worlds
            .par_iter()
            .map_init(
                || self.fresh_nets(),
                |nets, world| {
                    let mut t = vec![0i64; k];
                    let mut w = vec![0u32; k];
                    for (i, &cand) in shortlist.iter().enumerate() {
                        let margin = rollout(nets, world, state, ctx, cand, team)?;
                        t[i] = margin as i64;
                        w[i] = (margin > 0) as u32;
                    }
                    Ok((t, w))
                },
            )
            .collect();

        let per_world = per_world?;
        let mut totals = vec![0i64; k];
        let mut wins = vec![0u32; k];
        for (t, w) in &per_world {
            for i in 0..k {
                totals[i] += t[i];
                wins[i] += w[i];
            }
        }
        Ok((totals, wins, per_world.len() as u32))
    }

    #[cfg(not(feature = "parallel"))]
    fn run_parallel(
        &mut self,
        worlds: &[World],
        state: &GameState,
        ctx: &MatchContext,
        shortlist: &[u8],
        team: usize,
    ) -> Result<Tallies, AgentError> {
        self.run_sequential(worlds, state, ctx, shortlist, team, Instant::now())
    }

    /// Une lane par (monde, candidate) : **l'enchère sur CPU**, puis les 32
    /// cartes de toutes les lanes en lockstep sur GPU.
    ///
    /// Découpage assumé : l'enchère ne se met pas en lockstep (les lanes en
    /// sortent à des instants différents, certaines par une donne passée) et
    /// elle est bon marché par appel. Si elle devient le goulot une fois le jeu
    /// déporté, c'est la mesure qui le dira — pas une intuition.
    #[cfg(feature = "dmc_train")]
    fn run_gpu(
        &mut self,
        worlds: &[World],
        state: &GameState,
        ctx: &MatchContext,
        shortlist: &[u8],
        team: usize,
    ) -> Result<Tallies, AgentError> {
        use crate::gpu_rollout::Lane;

        let k = shortlist.len();
        let mut lanes: Vec<Lane> = Vec::with_capacity(worlds.len() * k);

        // Phase 1 — l'enchère, sur CPU, lane par lane.
        for world in worlds {
            for &cand in shortlist {
                let mut sim = *state;
                sim.hands = *world;
                let mut track = ctx.clone();
                track.track(&sim, cand);
                sim.step(cand);
                while sim.phase == Phase::Bidding && !sim.is_terminal() {
                    let before = sim;
                    let action = self.nets.bid.decide(&before, &track)?.action;
                    track.track(&before, action);
                    sim.step(action);
                }
                lanes.push(Lane { state: sim, ctx: track });
            }
        }

        // Phase 2 — les 32 cartes, en lot.
        let gpu = self.gpu.as_ref().expect("run_gpu appelé sans moteur GPU");
        gpu.play_out(&mut lanes)
            .map_err(|e| AgentError::Model(format!("déroulement GPU : {e}")))?;

        // L'ordre des lanes est (monde-major, candidate-mineur) : le même monde
        // sert donc bien toutes les candidates, comme sur le chemin CPU.
        let mut totals = vec![0i64; k];
        let mut wins = vec![0u32; k];
        for (idx, lane) in lanes.iter().enumerate() {
            let score = lane.state.deal_score().scores;
            let margin = score[team] as i32 - score[1 - team] as i32;
            let i = idx % k;
            totals[i] += margin as i64;
            if margin > 0 {
                wins[i] += 1;
            }
        }
        Ok((totals, wins, worlds.len() as u32))
    }

    #[cfg(not(feature = "dmc_train"))]
    fn run_gpu(
        &mut self,
        worlds: &[World],
        state: &GameState,
        ctx: &MatchContext,
        shortlist: &[u8],
        team: usize,
    ) -> Result<Tallies, AgentError> {
        self.run_sequential(worlds, state, ctx, shortlist, team, Instant::now())
    }
}

/// Construire l'échantillonneur décrit par un spec. Vit ici plutôt que dans
/// `spec.rs` pour que la règle « playgen par défaut, uniforme sur demande » et
/// son message d'erreur restent avec le code qui les subit.
pub fn build_bid_worlds(
    kind: &str,
    url: Option<&str>,
    model: Option<Arc<PlaygenModel>>,
    temperature: f32,
    timeout: std::time::Duration,
    env_url: Option<String>,
) -> Result<BidWorlds, AgentError> {
    match kind {
        "uniform" | "none" => Ok(BidWorlds::Uniform),
        "playgen" | "local" | "cpu" => {
            let m = model.ok_or_else(|| {
                AgentError::Config(
                    "worlds.source = \"playgen\" requires worlds.model (a playgen v2 .bin)".into(),
                )
            })?;
            Ok(BidWorlds::Local(Box::new(PlaygenAnalyst::new(m))))
        }
        "sidecar" | "gpu" => {
            let url = url
                .map(|u| u.to_string())
                .or(env_url)
                .filter(|u| !u.is_empty())
                .ok_or_else(|| {
                    AgentError::Config(
                        "bid worlds 'sidecar' needs a URL: set worlds.url or \
                         $COLVER_PLAYGEN_GPU_URL, or choose worlds.source = \"uniform\" \
                         to sample without a model"
                            .into(),
                    )
                })?;
            let s = SidecarWorldSource::new(url, temperature, timeout);
            s.health_check()?;
            Ok(BidWorlds::Sidecar(Box::new(s)))
        }
        other => Err(AgentError::Config(format!(
            "unknown world source '{other}' (sidecar|playgen|uniform)"
        ))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// `value_idx × 4 + suit_idx + 1`. Pique=0, Cœur=1, Carreau=2, Trèfle=3.
    const B90D: u8 = 7;
    const B100D: u8 = 11;
    const B110D: u8 = 15;
    const B120D: u8 = 19;
    const B100H: u8 = 10;
    const B160D: u8 = 35;
    const B150D: u8 = 31;

    fn all_legal() -> u64 {
        (1u64 << 43) - 1
    }

    fn probe(candidates: usize) -> RolloutBidConfig {
        RolloutBidConfig { mode: CandidateMode::Probe, candidates, ..Default::default() }
    }

    /// Le sondage complet : l'annonce du réseau, passe, la deuxième couleur,
    /// puis les deux voisines. L'ordre est celui du départage — le réseau gagne
    /// les égalités.
    #[test]
    fn probe_covers_pass_the_second_suit_and_the_neighbours() {
        let prior =
            [(B100D, 1.0), (B110D, 0.9), (B90D, 0.8), (0, 0.5), (B100H, 0.4)];
        assert_eq!(
            shortlist(&probe(0), &prior, all_legal()),
            vec![B100D, 0, B100H, B90D, B110D]
        );
    }

    /// La deuxième couleur se cherche **loin** dans le classement du réseau :
    /// c'est tout l'intérêt, un top-K ne la propose presque jamais.
    #[test]
    fn the_second_suit_is_found_however_deep_it_sits() {
        let prior = [(B100D, 1.0), (B110D, 0.9), (B120D, 0.8), (B90D, 0.7), (B100H, 0.01)];
        assert!(
            shortlist(&probe(0), &prior, all_legal()).contains(&B100H),
            "la meilleure annonce d'une autre couleur doit entrer"
        );
    }

    /// Quand −10 n'est pas légal (l'enchère est déjà passée par là), on explore
    /// +10 et +20 : deux paliers dans tous les cas.
    #[test]
    fn when_the_step_down_is_illegal_it_probes_two_steps_up() {
        // 100♦ est l'annonce du réseau et le plancher légal : pas de 90♦.
        let legal: u64 = (B100D..=36).fold(0u64, |m, a| m | (1u64 << a)) | 1;
        let prior = [(B100D, 1.0)];
        let list = shortlist(&probe(0), &prior, legal);
        assert!(list.contains(&B110D) && list.contains(&B120D), "attendu +10 et +20 : {list:?}");
        assert!(!list.contains(&B90D));
    }

    /// Le voisinage ne déborde pas sur le capot ni sur la coinche : ce sont
    /// d'autres décisions, pas des paliers de la même échelle.
    #[test]
    fn the_neighbourhood_stays_inside_the_ordinary_bids() {
        let prior = [(B160D, 1.0)];
        let list = shortlist(&probe(0), &prior, all_legal());
        assert!(list.contains(&B150D), "150♦ doit être exploré");
        assert!(!list.iter().any(|&a| a >= 37), "capot/coinche ne sont pas des voisines : {list:?}");
    }

    /// Quand le réseau dit « passe » il n'y a ni couleur ni voisinage : on
    /// prend ses meilleures annonces, c'est-à-dire les portes d'entrée.
    #[test]
    fn probe_falls_back_to_the_prior_when_the_net_passes() {
        let prior = [(0, 1.0), (B90D, 0.9), (B100H, 0.5)];
        assert_eq!(shortlist(&probe(3), &prior, all_legal()), vec![0, B90D, B100H]);
    }

    /// Une candidate illégale ne doit jamais entrer : un voisinage se calcule
    /// sur l'échelle des annonces, pas sur ce que l'enchère en cours autorise.
    #[test]
    fn illegal_candidates_are_dropped() {
        let legal = (1u64 << B100D) | (1u64 << B110D);
        let prior = [(B100D, 1.0), (B110D, 0.9)];
        assert_eq!(shortlist(&probe(0), &prior, legal), vec![B100D, B110D]);
    }

    /// Le mode `top` reste ce qu'il était : les meilleures au Q, plafonnées.
    #[test]
    fn top_is_the_prior_order() {
        let prior = [(B100D, 1.0), (B110D, 0.9), (B90D, 0.8), (0, 0.1)];
        let cfg = RolloutBidConfig { mode: CandidateMode::Top, candidates: 3, ..Default::default() };
        assert_eq!(shortlist(&cfg, &prior, all_legal()), vec![B100D, B110D, B90D]);
    }

    /// Un monde d'enchère doit rendre sa main à l'observateur, et 32 cartes en
    /// tout. Sans ce contrôle, un sampler qui dérape empoisonne la moyenne avec
    /// des positions qui n'existent pas.
    #[test]
    fn a_world_that_moves_the_observers_hand_is_rejected() {
        let mut rng = StdRng::seed_from_u64(1);
        let state = GameState::deal_random(0, &mut rng);
        assert!(valid_deal(&state.hands, &state, 2));
        let mut bad = state.hands;
        bad.swap(0, 2);
        assert!(!valid_deal(&bad, &state, 2), "la main de l'observateur a bougé");
    }
}
