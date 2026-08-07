//! Declarative agent description, and the one place that turns it into a
//! live [`Player`].
//!
//! An [`AgentSpec`] is what a bot TOML says. `build(seat)` is what makes it
//! run. Everything that used to be assembled by hand at each call site —
//! loading models, wiring a belief net, choosing an observation layout,
//! deciding where worlds come from — happens here, so the arena, the web and
//! any script get identical agents from identical specs.
//!
//! ```toml
//! [bid]
//! strategy = "nn"                 # heuristic|improved|improved_v2|improved_v3|smart|maxi|nn|playgen
//! model = "models/bid_v6_isdd_resume/bid_nn_final.bin"
//!
//! [play]
//! method = "isdd"                 # isdd|dmc|dmc_then_isdd|ismcts|smart_ismcts|oracle|heuristic|rule
//! max_worlds = 256                # IS-DD sous échéance : plafond de mondes résolus par coup
//! min_worlds = 60                 # ... et plancher : sous pression, plus lent plutôt que plus faible
//! objective = "deal_score"        # IS-DD : deal_score (défaut, contrat compris) | card_points
//! time_ms = 1000                  # time budget; 0 = use `determinizations` instead
//! determinizations = 240
//!
//! [worlds]
//! source = "sidecar"              # sidecar|playgen|uniform  (default: sidecar)
//! url = "http://gpu-host:8003"   # or the COLVER_PLAYGEN_GPU_URL env var
//! temperature = 0.8
//! fallback = "strict"             # strict|uniform
//!
//! [belief]
//! model = "models/belief_v4_fix_v2.bin"
//! ```

use std::time::Duration;

use crate::bid_eval::BidFunction;
use crate::is_dd::IsDdConfig;
use crate::mcts::{MctsConfig, RolloutPolicy};
use crate::naive_ismcts::NaiveIsMctsConfig;
use crate::smart_ismcts::SmartIsMctsConfig;
use crate::worlds::{
    FallbackPolicy, LocalPlaygenSource, PolicyWorldSource, SidecarWorldSource, UniformWorldSource,
    WorldSource,
};

use super::ismcts::{NaiveIsMctsPlayer, OracleMctsPlayer, SmartIsMctsPlayer};
use super::{
    bid::{BidNetPolicy, RuleBidPolicy},
    dmc::{DmcPlayer, HeuristicPlayer, OraclePlayer, RulePlayer},
    isdd::IsDdPlayer,
    models, AgentError, BidPolicy, CardPlayer, ComposedPlayer, Player,
};

/// Environment variable holding the playgen GPU sidecar URL. Used when a spec
/// asks for `source = "sidecar"` without spelling out a `url`, so the same bot
/// file works on a laptop, in CI and in production.
pub const SIDECAR_URL_ENV: &str = "COLVER_PLAYGEN_GPU_URL";

// ══════════════════════════════════════════════════════════════════════
//  Spec
// ══════════════════════════════════════════════════════════════════════

#[derive(Clone, Debug)]
pub struct BidSpec {
    /// Strategy name; `"nn"` requires `model`.
    pub strategy: String,
    pub model: Option<String>,
    /// Hint for weight files whose hidden size is ambiguous.
    pub hidden: usize,
    pub penalty: f32,
    pub temperature: f32,
    /// Apply endgame match-score adjustments to a non-score-aware net.
    pub score_aware: bool,
    /// The net was trained on the **canonical** suit ordering (v7 and later).
    ///
    /// This cannot be auto-detected: a canonical net has exactly the same weight-file
    /// size as a physical one of the same width, so the flag is the only thing that
    /// distinguishes them. Getting it wrong is silent — the net returns a legal bid,
    /// in the wrong suit. Same footgun as `residual` on the play side, and the same
    /// reason it is explicit.
    pub canonical: bool,

    // ── `strategy = "rollout"` seulement ─────────────────────────────
    //
    // Annoncer en simulant la donne. `model` reste le réseau de référence :
    // il présélectionne les candidates *et* parle pour les quatre sièges dans
    // la suite de chaque simulation. Voir [`crate::agent::bid_rollout`].
    /// Mondes tirés par décision, rejoués par chaque candidate.
    pub sims: u32,
    /// Plafond du nombre d'annonces simulées. 0 = pas de plafond.
    pub candidates: usize,
    /// Comment la liste est construite : `probe` (défaut — celle du réseau,
    /// passe, la deuxième couleur, deux voisines) ou `top` (les meilleures au Q).
    pub candidate_mode: crate::agent::bid_rollout::CandidateMode,
    /// Éclater les déroulements d'une décision sur rayon. Utile au web (latence
    /// d'un coup), inutile en arène où les matchs saturent déjà les cœurs.
    pub parallel: bool,
    /// Dérouler les mondes en lot sur GPU (feature `dmc_train`). Le vrai levier.
    pub gpu: bool,
    /// Le modèle de jeu des simulations. `None` reprend `[play] model`, ce qui
    /// est le cas normal : on simule avec le joueur qu'on est.
    pub play_model: Option<String>,
    /// Idem pour les connexions résiduelles ; `None` reprend `[play] residual`.
    pub play_residual: Option<bool>,
    /// Ce que la simulation maximise (`margin` par défaut, `winrate` offert).
    pub objective: crate::agent::bid_rollout::RolloutObjective,
    /// Échéance par décision d'enchère, en ms. 0 = pas d'horloge.
    pub time_ms: u32,
}

impl Default for BidSpec {
    fn default() -> Self {
        BidSpec {
            strategy: "improved_v2".into(),
            model: None,
            hidden: 256,
            penalty: 0.0,
            temperature: 0.0,
            score_aware: false,
            canonical: false,
            sims: 20,
            candidates: 5,
            candidate_mode: crate::agent::bid_rollout::CandidateMode::Probe,
            parallel: false,
            gpu: false,
            play_model: None,
            play_residual: None,
            objective: crate::agent::bid_rollout::RolloutObjective::Margin,
            time_ms: 0,
        }
    }
}

#[derive(Clone, Debug)]
pub struct PlaySpec {
    pub method: String,
    pub model: Option<String>,
    /// `oracle_dd` only: which card to play when several are DD-equal. **57,8 % of positions
    /// have more than one optimal card**, so this is not an edge case — it is the majority.
    /// `"order"` (the default and the historical behaviour) takes the first in the solver's own
    /// move-ordering preference; `"lowest"` / `"highest"` take the extreme card index, which is
    /// deterministic but meaningless at the table; `"cheapest"` / `"dearest"` take the fewest /
    /// most card points, which is the closest thing here to a Belote principle.
    ///
    /// Exists to answer a question nobody had measured: does the choice among DD-equal cards
    /// change actual playing strength? Against a perfect opponent it cannot — both realise the
    /// same DD value — so it can only be measured against an imperfect one.
    pub tiebreak: String,
    /// Skip connections (DouDou50 / triforge architecture). Same weights,
    /// different forward pass, so it cannot be detected from the file.
    pub residual: bool,
    /// Per-move budget in ms. 0 = ignore the clock and solve exactly
    /// `determinizations` worlds.
    pub time_ms: u32,
    pub determinizations: u32,
    /// Mondes par décision selon les cartes restantes (index 1..=8), en mode
    /// compte. TOML : `dets_schedule = "60,60,60,30,30,20,20"`, de 8 cartes
    /// restantes à 2. Voir [`crate::is_dd::IsDdConfig::det_schedule`].
    pub det_schedule: Option<[u32; 9]>,
    pub oracle_iters: u32,
    /// `dmc_then_isdd`: trick index at which IS-DD takes over.
    pub switch_at: u8,
    pub early_termination: Option<bool>,
    pub dominance_factor: f32,
    pub belief_frac: f32,
    /// Ce que la recherche IS-DD maximise : `deal_score` (défaut — contrat,
    /// chute, belote, capot) ou `card_points` (l'ancien défaut, points cartes
    /// nus). Voir `PlayObjective`.
    pub objective: crate::is_dd::PlayObjective,
    /// Plafond de mondes résolus par décision, sous échéance. `None` = pas de
    /// plafond (le défaut historique : consommer tout le budget).
    pub max_worlds: Option<u32>,
    /// Plancher de mondes résolus sous échéance (`[play] min_worlds`).
    ///
    /// Choix de politique : sous pression de calcul, la dégradation se paie en
    /// **latence** (visible) plutôt qu'en **force de jeu** (invisible). `None`
    /// laisse le comportement d'origine — l'échéance coupe, quel que soit le
    /// nombre de mondes atteint.
    pub min_worlds: Option<u32>,
    pub cred_alpha: f32,
    pub cred_bid_model: Option<String>,
    pub cred_play_model: Option<String>,
    pub parallel: bool,
}

impl Default for PlaySpec {
    fn default() -> Self {
        PlaySpec {
            method: "isdd".into(),
            model: None,
            tiebreak: "order".into(),
            residual: false,
            time_ms: 0,
            determinizations: 20,
            det_schedule: None,
            oracle_iters: 2000,
            switch_at: 5,
            early_termination: None,
            dominance_factor: 1.0,
            belief_frac: 1.0,
            objective: crate::is_dd::PlayObjective::DealScore,
            max_worlds: None,
            min_worlds: None,
            cred_alpha: 0.0,
            cred_bid_model: None,
            cred_play_model: None,
            // Fan DD solves across the rayon pool. Bit-identical to sequential
            // (DD is deterministic, aggregation order is fixed), so there is no
            // reason for the default to be "slow".
            parallel: true,
        }
    }
}

/// `"60,60,60,30,30,20,20"` → un tableau indexé par cartes restantes.
///
/// La liste se lit **de 8 cartes restantes vers 1**, sens de lecture d'une
/// donne. L'index 0 n'est jamais consulté (zéro carte = pas de décision), et
/// une liste plus courte que 8 reconduit sa dernière valeur — écrire les
/// échelons qui comptent suffit, les finales n'en ont pas besoin.
pub fn parse_det_schedule(s: &str) -> Result<[u32; 9], String> {
    let vals: Vec<u32> = s
        .split(',')
        .map(|t| t.trim().parse::<u32>().map_err(|_| format!("« {} » n'est pas un entier", t.trim())))
        .collect::<Result<_, _>>()?;
    if vals.is_empty() {
        return Err("liste vide".into());
    }
    if vals.len() > 8 {
        return Err(format!("{} valeurs pour 8 échelons au plus", vals.len()));
    }
    if vals.iter().any(|&v| v == 0) {
        return Err("un échelon à zéro ne chercherait aucun monde".into());
    }
    let mut out = [0u32; 9];
    for (i, cards) in (1..=8u32).rev().enumerate() {
        out[cards as usize] = *vals.get(i).unwrap_or_else(|| vals.last().unwrap());
    }
    Ok(out)
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum WorldSourceKind {
    /// Playgen transformer on a remote GPU. The default: it is the only source
    /// fast enough to supply hundreds of high-quality worlds per move.
    Sidecar,
    /// The same transformer in-process on CPU. Same distribution, ~50× slower.
    LocalPlaygen,
    /// Constraint-uniform. Needs no model; measurably weaker.
    Uniform,
}

#[derive(Clone, Debug)]
pub struct WorldSpec {
    pub kind: WorldSourceKind,
    /// Sidecar URL; falls back to [`SIDECAR_URL_ENV`].
    pub url: Option<String>,
    /// Playgen model path, for `LocalPlaygen`.
    pub model: Option<String>,
    pub temperature: f32,
    pub timeout: Duration,
    pub fallback: FallbackPolicy,
    /// Worlds requested per refill under a time budget.
    pub batch: usize,
}

impl Default for WorldSpec {
    fn default() -> Self {
        WorldSpec {
            kind: WorldSourceKind::Sidecar,
            url: None,
            model: None,
            temperature: 0.8,
            timeout: Duration::from_secs(6),
            fallback: FallbackPolicy::Strict,
            batch: 128,
        }
    }
}

#[derive(Clone, Debug, Default)]
pub struct BeliefSpec {
    pub model: Option<String>,
}

/// A complete agent description.
#[derive(Clone, Debug, Default)]
pub struct AgentSpec {
    pub name: String,
    pub bid: BidSpec,
    pub play: PlaySpec,
    pub worlds: WorldSpec,
    pub belief: BeliefSpec,
    /// Base RNG seed. Each seat derives its own stream from this, so a match
    /// replays exactly.
    pub seed: u64,
}

impl AgentSpec {
    pub fn from_toml_file(path: &str) -> Result<Self, AgentError> {
        let content = std::fs::read_to_string(path)
            .map_err(|e| AgentError::Config(format!("cannot read {path}: {e}")))?;
        let name = std::path::Path::new(path)
            .file_stem()
            .and_then(|s| s.to_str())
            .unwrap_or("unknown")
            .to_string();
        let mut spec = Self::from_toml_str(&content)?;
        spec.name = name;
        Ok(spec)
    }

    /// Parse the flat `key = value` subset of TOML the bot files use.
    ///
    /// Hand-rolled on purpose: the format is a handful of scalars under three
    /// or four headers, and a TOML crate would be the only dependency of the
    /// whole core beyond `rand`.
    pub fn from_toml_str(content: &str) -> Result<Self, AgentError> {
        let mut spec = AgentSpec::default();
        let mut section = String::new();
        // Legacy `[play] playgen_model = …` means "sample worlds from playgen
        // in-process"; honour it so old bot files keep working.
        let mut legacy_playgen: Option<String> = None;

        for raw in content.lines() {
            let line = raw.split('#').next().unwrap_or("").trim();
            if line.is_empty() {
                continue;
            }
            if let Some(header) = line.strip_prefix('[').and_then(|l| l.strip_suffix(']')) {
                section = header.to_string();
                continue;
            }
            let Some((key, val)) = line.split_once('=') else { continue };
            let key = key.trim();
            let val = val.trim().trim_matches('"');

            let num = |d: f32| val.parse::<f32>().unwrap_or(d);
            let int = |d: u32| val.parse::<u32>().unwrap_or(d);
            let flag = || val == "true";

            match (section.as_str(), key) {
                ("bid", "strategy") => spec.bid.strategy = val.into(),
                ("bid", "model") => spec.bid.model = Some(val.into()),
                ("bid", "hidden") => spec.bid.hidden = int(256) as usize,
                ("bid", "penalty") => spec.bid.penalty = num(0.0),
                ("bid", "temperature") => spec.bid.temperature = num(0.0),
                ("bid", "score_aware") => spec.bid.score_aware = flag(),
                ("bid", "canonical") => spec.bid.canonical = flag(),
                ("bid", "sims") => spec.bid.sims = int(20),
                ("bid", "candidates") => spec.bid.candidates = int(5) as usize,
                ("bid", "parallel") => spec.bid.parallel = flag(),
                ("bid", "gpu") => spec.bid.gpu = flag(),
                ("bid", "candidate_mode") => {
                    spec.bid.candidate_mode = match val {
                        "top" => crate::agent::bid_rollout::CandidateMode::Top,
                        "probe" | "around" | "sondage" => {
                            crate::agent::bid_rollout::CandidateMode::Probe
                        }
                        other => {
                            return Err(AgentError::Config(format!(
                                "unknown bid candidate_mode '{other}' (expected 'probe' or 'top')"
                            )))
                        }
                    }
                }
                ("bid", "play_model") => spec.bid.play_model = Some(val.into()),
                ("bid", "play_residual") => spec.bid.play_residual = Some(flag()),
                ("bid", "time_ms") => spec.bid.time_ms = int(0),
                ("bid", "objective") => {
                    // Même règle que `[play] objective` : pas de repli
                    // silencieux, une faute de frappe changerait ce que le bot
                    // maximise sans rien afficher.
                    spec.bid.objective = match val {
                        "margin" | "score" => crate::agent::bid_rollout::RolloutObjective::Margin,
                        "winrate" | "win" => crate::agent::bid_rollout::RolloutObjective::WinRate,
                        other => {
                            return Err(AgentError::Config(format!(
                                "unknown bid objective '{other}' (expected 'margin' or 'winrate')"
                            )))
                        }
                    }
                }

                ("play", "method") => spec.play.method = val.into(),
                ("play", "model") => spec.play.model = Some(val.into()),
                ("play", "tiebreak") => spec.play.tiebreak = val.into(),
                ("play", "residual") => spec.play.residual = flag(),
                ("play", "time_ms") => spec.play.time_ms = int(0),
                ("play", "determinizations") => spec.play.determinizations = int(20),
                ("play", "dets_schedule") => {
                    spec.play.det_schedule = Some(parse_det_schedule(val).map_err(|e| {
                        AgentError::Config(format!("play.dets_schedule: {e}"))
                    })?);
                }
                ("play", "oracle_iters") => spec.play.oracle_iters = int(2000),
                ("play", "switch_at") => spec.play.switch_at = int(5) as u8,
                ("play", "early_termination") => spec.play.early_termination = Some(flag()),
                ("play", "dominance_factor") => spec.play.dominance_factor = num(1.0),
                ("play", "belief_frac") => spec.play.belief_frac = num(1.0),
                ("play", "max_worlds") => spec.play.max_worlds = Some(int(0) as u32),
                ("play", "min_worlds") => spec.play.min_worlds = Some(int(0) as u32),
                ("play", "objective") => {
                    // Pas de repli silencieux : une faute de frappe rendrait
                    // l'objectif le plus faible sans rien dire, et un bot qui
                    // maximise les points cartes joue *différemment* — pas un
                    // peu moins bien, autre chose.
                    spec.play.objective = match val {
                        "deal_score" | "score" => crate::is_dd::PlayObjective::DealScore,
                        "card_points" | "cards" => crate::is_dd::PlayObjective::CardPoints,
                        other => {
                            return Err(AgentError::Config(format!(
                                "unknown play objective '{other}' \
                                 (expected 'deal_score' or 'card_points')"
                            )))
                        }
                    }
                }
                ("play", "cred_alpha") => spec.play.cred_alpha = num(0.0),
                ("play", "cred_bid_model") => spec.play.cred_bid_model = Some(val.into()),
                ("play", "cred_play_model") => spec.play.cred_play_model = Some(val.into()),
                ("play", "parallel") => spec.play.parallel = flag(),
                ("play", "playgen_model") => legacy_playgen = Some(val.into()),
                ("play", "playgen_temp") => spec.worlds.temperature = num(0.8),

                ("worlds", "source") => {
                    spec.worlds.kind = match val {
                        "sidecar" | "gpu" => WorldSourceKind::Sidecar,
                        "playgen" | "local" | "cpu" => WorldSourceKind::LocalPlaygen,
                        "uniform" | "none" => WorldSourceKind::Uniform,
                        other => {
                            return Err(AgentError::Config(format!(
                                "unknown world source '{other}' (sidecar|playgen|uniform)"
                            )))
                        }
                    }
                }
                ("worlds", "url") => spec.worlds.url = Some(val.into()),
                ("worlds", "model") => spec.worlds.model = Some(val.into()),
                ("worlds", "temperature") => spec.worlds.temperature = num(0.8),
                ("worlds", "timeout_ms") => {
                    spec.worlds.timeout = Duration::from_millis(int(6000) as u64)
                }
                ("worlds", "batch") => spec.worlds.batch = int(128) as usize,
                ("worlds", "fallback") => {
                    spec.worlds.fallback = match val {
                        "strict" => FallbackPolicy::Strict,
                        "uniform" => FallbackPolicy::Uniform,
                        other => {
                            return Err(AgentError::Config(format!(
                                "unknown fallback '{other}' (strict|uniform)"
                            )))
                        }
                    }
                }

                ("belief", "model") => spec.belief.model = Some(val.into()),
                // Hard constraints are facts and always applied; the old flag
                // is accepted and ignored so historical bot files still parse.
                ("belief", "use_hard_constraints") => {}
                _ => {}
            }
        }

        if let Some(model) = legacy_playgen {
            if spec.worlds.model.is_none() {
                spec.worlds.kind = WorldSourceKind::LocalPlaygen;
                spec.worlds.model = Some(model);
            }
        }
        Ok(spec)
    }

    /// Instantiate this spec for one seat.
    pub fn build(&self, seat: u8) -> Result<Box<dyn Player>, AgentError> {
        let seed = self.seed ^ ((seat as u64 + 1).wrapping_mul(0x9E37_79B9_7F4A_7C15));
        let bid = self.build_bid(seat, seed)?;
        let play = self.build_play(seat, seed)?;
        Ok(Box::new(ComposedPlayer::new(self.label(), seat, bid, play)))
    }

    fn build_bid(&self, seat: u8, seed: u64) -> Result<Box<dyn BidPolicy>, AgentError> {
        if self.bid.strategy == "playgen" {
            let path = self.bid.model.as_deref().ok_or_else(|| {
                AgentError::Config("bid strategy 'playgen' requires a model path".into())
            })?;
            let model = models::playgen_model(path)?;
            return Ok(Box::new(super::bid::PlaygenBidPolicy::new(
                model,
                seat,
                self.bid.temperature,
                seed,
            )));
        }
        if self.bid.strategy == "rollout" {
            // Le réseau de référence : présélection des candidates, et parole
            // des quatre sièges dans la suite de chaque simulation.
            let path = self.bid.model.as_deref().ok_or_else(|| {
                AgentError::Config("bid strategy 'rollout' requires a model path".into())
            })?;
            let bid_weights = models::bid_weights(path, self.bid.hidden)?;
            // Le joueur des simulations est celui du bot, sauf mention
            // contraire : simuler avec un autre joueur que soi mesurerait
            // l'annonce d'un bot qui n'existe pas.
            let play_path = self
                .bid
                .play_model
                .as_deref()
                .or(self.play.model.as_deref())
                .ok_or_else(|| {
                    AgentError::Config(
                        "bid strategy 'rollout' needs a play model: set bid.play_model, \
                         or a [play] model it can borrow"
                            .into(),
                    )
                })?;
            let dmc_weights = models::dmc_weights(play_path)?;
            // Les mondes d'enchère se règlent dans `[worlds]`, comme ceux du
            // jeu — même section, même défaut sidecar, même refus de dégrader
            // en silence. `worlds.model` sert le mode CPU.
            let pg_model = match self.worlds.kind {
                WorldSourceKind::LocalPlaygen => {
                    Some(models::playgen_model(self.worlds.model.as_deref().ok_or_else(|| {
                        AgentError::Config(
                            "worlds.source = \"playgen\" requires worlds.model".into(),
                        )
                    })?)?)
                }
                _ => None,
            };
            let kind = match self.worlds.kind {
                WorldSourceKind::Sidecar => "sidecar",
                WorldSourceKind::LocalPlaygen => "playgen",
                WorldSourceKind::Uniform => "uniform",
            };
            let worlds = super::bid_rollout::build_bid_worlds(
                kind,
                self.worlds.url.as_deref(),
                pg_model,
                self.worlds.temperature,
                self.worlds.timeout,
                std::env::var(SIDECAR_URL_ENV).ok(),
            )?;
            return Ok(Box::new(super::bid_rollout::RolloutBidPolicy::new(
                bid_weights,
                dmc_weights,
                self.bid.play_residual.unwrap_or(self.play.residual),
                self.bid.penalty,
                self.bid.score_aware,
                self.bid.canonical,
                worlds,
                super::bid_rollout::RolloutBidConfig {
                    sims: self.bid.sims,
                    candidates: self.bid.candidates,
                    mode: self.bid.candidate_mode,
                    objective: self.bid.objective,
                    time_ms: self.bid.time_ms,
                    parallel: self.bid.parallel,
                    gpu: self.bid.gpu,
                    fallback: self.worlds.fallback,
                },
                seat,
                seed,
            )));
        }
        if self.bid.strategy == "nn" {
            let path = self.bid.model.as_deref().ok_or_else(|| {
                AgentError::Config("bid strategy 'nn' requires a model path".into())
            })?;
            let weights = models::bid_weights(path, self.bid.hidden)?;
            return Ok(Box::new(BidNetPolicy::new(
                weights,
                self.bid.penalty,
                self.bid.temperature,
                self.bid.score_aware,
                self.bid.canonical,
                seed,
            )));
        }
        let function = match self.bid.strategy.as_str() {
            "heuristic" => BidFunction::Heuristic,
            "improved" => BidFunction::Improved,
            "improved_v2" => BidFunction::ImprovedV2,
            "improved_v3" => BidFunction::ImprovedV3,
            "smart" => BidFunction::Smart,
            "maxi" => BidFunction::Maxi,
            other => {
                return Err(AgentError::Config(format!("unknown bid strategy '{other}'")))
            }
        };
        Ok(Box::new(RuleBidPolicy::new(function)))
    }

    fn is_dd_config(&self) -> IsDdConfig {
        let mut cfg = IsDdConfig {
            determinizations: self.play.determinizations,
            det_schedule: self.play.det_schedule,
            // A zero budget means "count mode": solve exactly N worlds however
            // long it takes. That is what evaluations want; production sets a
            // clock instead.
            time_limit_ms: (self.play.time_ms > 0).then_some(self.play.time_ms),
            use_nn_beliefs: self.belief.model.is_some(),
            dominance_factor: self.play.dominance_factor,
            belief_frac: self.play.belief_frac,
            objective: self.play.objective,
            max_worlds: self.play.max_worlds,
            min_worlds: self.play.min_worlds,
            cred_alpha: self.play.cred_alpha,
            parallel: self.play.parallel,
            world_batch: self.worlds.batch,
            ..Default::default()
        };
        if let Some(et) = self.play.early_termination {
            cfg.early_termination = et;
        }
        cfg
    }

    /// Build the world source. Errors rather than defaulting to uniform: a
    /// missing sidecar URL is a configuration mistake, and silently sampling
    /// weaker worlds instead would hide it behind a few points per deal.
    pub fn build_world_source(&self) -> Result<Box<dyn WorldSource>, AgentError> {
        let inner: Box<dyn WorldSource> = match self.worlds.kind {
            WorldSourceKind::Uniform => Box::new(UniformWorldSource),
            WorldSourceKind::LocalPlaygen => {
                let path = self.worlds.model.as_deref().ok_or_else(|| {
                    AgentError::Config(
                        "worlds.source = \"playgen\" requires worlds.model (a playgen .bin)".into(),
                    )
                })?;
                let model = models::playgen_model(path)?;
                Box::new(LocalPlaygenSource::new(model, self.worlds.temperature))
            }
            WorldSourceKind::Sidecar => {
                let url = self
                    .worlds
                    .url
                    .clone()
                    .or_else(|| std::env::var(SIDECAR_URL_ENV).ok())
                    .filter(|u| !u.is_empty())
                    .ok_or_else(|| {
                        AgentError::Config(format!(
                            "world source 'sidecar' needs a URL: set worlds.url or ${SIDECAR_URL_ENV}, \
                             or choose worlds.source = \"uniform\" to sample without a model"
                        ))
                    })?;
                let source =
                    SidecarWorldSource::new(url, self.worlds.temperature, self.worlds.timeout);
                // Fail at construction rather than mid-deal.
                source.health_check()?;
                Box::new(source)
            }
        };
        Ok(Box::new(PolicyWorldSource::new(inner, self.worlds.fallback)))
    }

    fn build_isdd(&self, seat: u8, seed: u64) -> Result<IsDdPlayer, AgentError> {
        let mut player = IsDdPlayer::new(self.is_dd_config(), seat, seed)
            .with_world_source(self.build_world_source()?);
        if let Some(path) = &self.belief.model {
            player.load_belief_net(path)?;
        }
        if self.play.cred_alpha > 0.0 {
            // Default the auction judge to the bot's own bidder: "would I have
            // bid this?" is the question the weighting is asking.
            let bid_judge = self.play.cred_bid_model.as_ref().or(self.bid.model.as_ref());
            if let Some(path) = bid_judge {
                player.load_cred_bid_net(path)?;
            }
            if let Some(path) = &self.play.cred_play_model {
                player.load_cred_play_net(path)?;
            }
        }
        Ok(player)
    }

    fn build_dmc(&self) -> Result<DmcPlayer, AgentError> {
        let path = self.play.model.as_deref().ok_or_else(|| {
            AgentError::Config(format!("play method '{}' requires a model", self.play.method))
        })?;
        Ok(DmcPlayer::new(models::dmc_weights(path)?, self.play.residual))
    }

    fn build_play(&self, seat: u8, seed: u64) -> Result<Box<dyn CardPlayer>, AgentError> {
        match self.play.method.as_str() {
            // `is_dd` / `smart_is_dd` are the historical names; the difference
            // between them was only whether a belief net was configured, which
            // the `[belief]` section already says.
            "isdd" | "is_dd" | "smart_is_dd" => Ok(Box::new(self.build_isdd(seat, seed)?)),
            "dmc" => Ok(Box::new(self.build_dmc()?)),
            "dmc_then_isdd" | "dmc_then_dd" => {
                let early = Box::new(self.build_dmc()?);
                Ok(Box::new(
                    self.build_isdd(seat, seed)?.with_early_player(early, self.play.switch_at),
                ))
            }
            "heuristic" => Ok(Box::new(HeuristicPlayer)),
            "rule" => Ok(Box::new(RulePlayer)),
            "oracle_dd" => Ok(Box::new(OraclePlayer::with_tiebreak(&self.play.tiebreak)?)),
            "oracle" => Ok(Box::new(OracleMctsPlayer::new(
                MctsConfig {
                    iterations: self.play.oracle_iters,
                    rollout_policy: RolloutPolicy::HeuristicPlay,
                    ..Default::default()
                },
                seed,
            ))),
            "naive_ismcts" | "ismcts" => Ok(Box::new(NaiveIsMctsPlayer::new(
                NaiveIsMctsConfig {
                    iterations_per_det: 50,
                    time_limit_ms: Some(self.play.time_ms),
                    ..Default::default()
                },
                seed,
            ))),
            "smart_ismcts" => Ok(Box::new(SmartIsMctsPlayer::new(
                SmartIsMctsConfig {
                    iterations_per_det: 50,
                    time_limit_ms: Some(self.play.time_ms),
                    ..Default::default()
                },
                seat,
                seed,
            ))),
            other => Err(AgentError::Config(format!("unknown play method '{other}'"))),
        }
    }

    /// Whether the play method draws determinized worlds — i.e. whether the
    /// `[worlds]` section means anything for this bot.
    pub fn uses_worlds(&self) -> bool {
        matches!(
            self.play.method.as_str(),
            "isdd" | "is_dd" | "smart_is_dd" | "dmc_then_isdd" | "dmc_then_dd"
        )
    }

    /// `bid_label + play_label`, the identity used in result tables.
    pub fn label(&self) -> String {
        if self.name.is_empty() {
            format!("{}/{}", self.bid_label(), self.play_label())
        } else {
            self.name.clone()
        }
    }

    /// Short model name, keeping the parent directory when it is informative.
    fn short_model(path: &str) -> String {
        let p = std::path::Path::new(path);
        let stem = p.file_stem().and_then(|s| s.to_str()).unwrap_or(path);
        match p.parent().and_then(|d| d.file_name()).and_then(|s| s.to_str()) {
            Some("models") | None => stem.to_string(),
            Some(dir) => format!("{dir}/{stem}"),
        }
    }

    pub fn bid_label(&self) -> String {
        let mut label = match &self.bid.model {
            Some(m) => format!("{}:{}", self.bid.strategy, Self::short_model(m)),
            None => self.bid.strategy.clone(),
        };
        if self.bid.temperature > 0.0 {
            label.push_str(&format!("@T{}", self.bid.temperature));
        }
        // Le budget de simulation *est* l'identité de ce bidder — deux
        // `rollout:bid_v6` à 10 et à 200 mondes ne sont pas le même joueur, et
        // `matches.csv` doit pouvoir les distinguer.
        if self.bid.strategy == "rollout" {
            label.push_str(&format!("@{}x{}", self.bid.sims, self.bid.candidates));
            if self.bid.candidate_mode == crate::agent::bid_rollout::CandidateMode::Top {
                label.push_str("+top");
            }
            // D'où viennent les mondes fait partie de l'identité : `uniform`
            // contredit l'enchère entendue, playgen non — deux joueurs.
            label.push_str(match self.worlds.kind {
                WorldSourceKind::Sidecar => "+pg",
                WorldSourceKind::LocalPlaygen => "+pgcpu",
                WorldSourceKind::Uniform => "+unif",
            });
            if self.bid.objective == crate::agent::bid_rollout::RolloutObjective::WinRate {
                label.push_str("+win");
            }
        }
        label
    }

    pub fn play_label(&self) -> String {
        let mut label = match &self.play.model {
            Some(m) => format!("{}:{}", self.play.method, Self::short_model(m)),
            None if self.play.time_ms > 0 => format!("{}:{}ms", self.play.method, self.play.time_ms),
            None => format!("{}:{}d", self.play.method, self.play.determinizations),
        };
        // Only IS-DD consumes determinized worlds, so only IS-DD labels say
        // where they came from.
        if self.uses_worlds() {
            match self.worlds.kind {
                WorldSourceKind::Sidecar => label.push_str("+pg"),
                WorldSourceKind::LocalPlaygen => label.push_str("+pgcpu"),
                WorldSourceKind::Uniform => label.push_str("+unif"),
            }
        }
        if self.belief.model.is_some() {
            label.push_str("+bel");
        }
        // L'objectif change ce que le bot *joue*, donc il doit se voir dans
        // `matches.csv` : sans ça `web_dede` et `web_dede_cardpts` rendent le
        // même `play_a`/`play_b` et un A/B est illisible une fois écrit. Seul
        // l'ancien objectif est marqué — le défaut reste muet, comme partout
        // ailleurs dans ce label.
        if self.uses_worlds() && self.play.objective == crate::is_dd::PlayObjective::CardPoints {
            label.push_str("+cardpts");
        }
        if self.play.cred_alpha > 0.0 {
            label.push_str(&format!("+cred{:.1}", self.play.cred_alpha));
        }
        label
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn parses_a_bot_file() {
        let spec = AgentSpec::from_toml_str(
            r#"
            [bid]
            strategy = "nn"
            model = "models/bid_v6/bid_nn_final.bin"
            hidden = 512

            [play]
            method = "isdd"
            determinizations = 240   # count mode

            [worlds]
            source = "uniform"
            "#,
        )
        .unwrap();
        assert_eq!(spec.bid.strategy, "nn");
        assert_eq!(spec.bid.hidden, 512);
        assert_eq!(spec.play.determinizations, 240);
        assert_eq!(spec.play.time_ms, 0);
        assert_eq!(spec.worlds.kind, WorldSourceKind::Uniform);
    }

    /// Un bot qui ne dit rien doit jouer pour le **score de donne**.
    ///
    /// Le défaut a de la valeur ici : `card_points` ignore la réussite du
    /// contrat, la belote (qui déplace le seuil de 20 points) et la différence
    /// entre faire chuter l'adversaire et lui grappiller quelques points. Un
    /// refactor qui reperdrait ce défaut ferait jouer *autre chose* à tous les
    /// bots sans en changer un seul fichier.
    #[test]
    fn default_play_objective_is_the_deal_score() {
        let spec = AgentSpec::from_toml_str("[play]\nmethod = \"isdd\"\n").unwrap();
        assert_eq!(spec.play.objective, crate::is_dd::PlayObjective::DealScore);
        assert_eq!(
            spec.is_dd_config().objective,
            crate::is_dd::PlayObjective::DealScore,
            "le défaut du spec doit atteindre la config de recherche"
        );
    }

    #[test]
    fn card_points_objective_is_still_reachable() {
        let spec = AgentSpec::from_toml_str(
            "[play]\nmethod = \"isdd\"\nobjective = \"card_points\"\n",
        )
        .unwrap();
        assert_eq!(spec.play.objective, crate::is_dd::PlayObjective::CardPoints);
    }

    /// Une faute de frappe doit être bruyante. Un repli silencieux sur
    /// `card_points` ferait jouer un autre agent que celui décrit par le
    /// fichier, sans rien afficher.
    #[test]
    fn an_unknown_objective_is_a_config_error() {
        let err = AgentSpec::from_toml_str(
            "[play]\nmethod = \"isdd\"\nobjective = \"deal-score\"\n",
        );
        assert!(matches!(err, Err(AgentError::Config(_))), "got {err:?}");
    }

    #[test]
    fn legacy_playgen_model_selects_the_cpu_source() {
        let spec = AgentSpec::from_toml_str(
            "[play]\nmethod = \"smart_is_dd\"\nplaygen_model = \"models/playgen_v2_final.bin\"\n",
        )
        .unwrap();
        assert_eq!(spec.worlds.kind, WorldSourceKind::LocalPlaygen);
        assert_eq!(spec.worlds.model.as_deref(), Some("models/playgen_v2_final.bin"));
    }

    #[test]
    fn sidecar_without_a_url_is_a_config_error() {
        // Default source is the sidecar; with no URL anywhere this must fail
        // loudly rather than quietly sampling uniform worlds.
        let spec = AgentSpec::from_toml_str("[play]\nmethod = \"isdd\"\n").unwrap();
        if std::env::var(SIDECAR_URL_ENV).is_err() {
            assert!(matches!(spec.build_world_source(), Err(AgentError::Config(_))));
        }
    }

    #[test]
    fn rejects_unknown_names() {
        assert!(AgentSpec::from_toml_str("[worlds]\nsource = \"magic\"\n").is_err());
        let spec = AgentSpec::from_toml_str("[play]\nmethod = \"telepathy\"\n").unwrap();
        assert!(spec.build(0).is_err());
    }
}

#[cfg(test)]
mod det_schedule_tests {
    use super::parse_det_schedule;

    /// La liste se lit de 8 cartes restantes vers 1, et une liste courte
    /// reconduit sa dernière valeur — on n'écrit que les échelons qui comptent.
    #[test]
    fn schedule_reads_from_eight_cards_down() {
        let s = parse_det_schedule("60,60,60,30,30,20,20").unwrap();
        assert_eq!(s[8], 60, "entame");
        assert_eq!(s[6], 60);
        assert_eq!(s[5], 30);
        assert_eq!(s[2], 20);
        assert_eq!(s[1], 20, "la dernière valeur se reconduit jusqu'à 1 carte");
    }

    /// À total constant, un calendrier ne change que la répartition : c'est
    /// l'argument qui justifie de l'offrir, il doit rester vérifiable.
    #[test]
    fn the_reference_schedule_costs_the_same_as_a_flat_forty() {
        let s = parse_det_schedule("60,60,60,30,30,20,20").unwrap();
        let total: u32 = (2..=8).map(|c| s[c]).sum();
        assert_eq!(total, 7 * 40, "280 mondes, comme un plat à 40 sur 7 échelons");
    }

    #[test]
    fn a_broken_schedule_is_a_config_error() {
        assert!(parse_det_schedule("").is_err());
        assert!(parse_det_schedule("60,x").is_err());
        assert!(parse_det_schedule("60,0,20").is_err(), "un échelon nul ne cherche rien");
        assert!(parse_det_schedule("1,2,3,4,5,6,7,8,9").is_err(), "9 échelons pour 8 stades");
    }
}
