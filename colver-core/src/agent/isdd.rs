//! IS-DD card player — the production agent.
//!
//! Wraps [`IsDdSearch`] with everything it needs to run on its own:
//!
//! - a [`WorldSource`] (playgen over the GPU sidecar by default),
//! - an optional belief net and credibility judges,
//! - its own RNG and per-deal state.
//!
//! The point of the wrapper is that **world generation lives inside the
//! agent**. Before it, the web server sampled playgen worlds and pushed them
//! into the search while the arena did not, so the two ran different agents
//! under the same name. Now a caller cannot get that wrong: it builds the
//! player and asks for a card.

use std::sync::Arc;

use rand::rngs::StdRng;
use rand::SeedableRng;
use std::time::Instant;

use crate::is_dd::{IsDdConfig, IsDdSearch};
use crate::state::GameState;
use crate::worlds::WorldSource;

use super::models::DmcWeights;
use super::{AgentError, CardPlayer, Decision, MatchContext, Stats};

pub struct IsDdPlayer {
    search: IsDdSearch,
    config: IsDdConfig,
    source: Option<Box<dyn WorldSource>>,
    seat: u8,
    rng: StdRng,
    /// Play only from this trick onward, deferring earlier tricks to `early`.
    /// IS-DD's edge is concentrated in the endgame, where the remaining tree is
    /// small enough for the solves to be near-exact; a fast net is a better use
    /// of the budget before that.
    switch_at: Option<u8>,
    early: Option<Box<dyn CardPlayer>>,
}

impl IsDdPlayer {
    pub fn new(config: IsDdConfig, seat: u8, seed: u64) -> Self {
        IsDdPlayer {
            search: IsDdSearch::new(),
            config,
            source: None,
            seat,
            rng: StdRng::seed_from_u64(seed),
            switch_at: None,
            early: None,
        }
    }

    /// Attach the source of determinized worlds. Without one, the search falls
    /// back to belief-weighted / constraint-uniform sampling — legal, cheap,
    /// and measurably weaker.
    pub fn with_world_source(mut self, source: Box<dyn WorldSource>) -> Self {
        self.source = Some(source);
        self
    }

    /// Hand the first `switch_at` tricks to `early` (typically a DMC net) and
    /// take over for the endgame.
    pub fn with_early_player(mut self, early: Box<dyn CardPlayer>, switch_at: u8) -> Self {
        self.early = Some(early);
        self.switch_at = Some(switch_at);
        self
    }

    pub fn load_belief_net(&mut self, path: &str) -> Result<(), AgentError> {
        self.search
            .load_belief_net(path)
            .map_err(|e| AgentError::Model(format!("{path}: {e}")))?;
        self.config.use_nn_beliefs = true;
        Ok(())
    }

    /// Bid net used to judge how credible a world's auction is (`cred_alpha`).
    pub fn load_cred_bid_net(&mut self, path: &str) -> Result<(), AgentError> {
        self.search
            .load_cred_bid_net(path)
            .map_err(|e| AgentError::Model(format!("{path}: {e}")))
    }

    /// Canonical DMC net used to judge how credible a world's play is.
    pub fn load_cred_play_net(&mut self, path: &str) -> Result<(), AgentError> {
        self.search
            .load_cred_play_net(path)
            .map_err(|e| AgentError::Model(format!("{path}: {e}")))
    }

    pub fn config(&self) -> &IsDdConfig {
        &self.config
    }

    pub fn config_mut(&mut self) -> &mut IsDdConfig {
        &mut self.config
    }

    pub fn search_mut(&mut self) -> &mut IsDdSearch {
        &mut self.search
    }

    /// Name of the attached world source, or `"none"`.
    pub fn world_source_name(&self) -> &'static str {
        self.source.as_ref().map(|s| s.name()).unwrap_or("none")
    }

    fn use_early(&self, state: &GameState) -> bool {
        match self.switch_at {
            Some(t) => (state.tricks_won[0] + state.tricks_won[1]) < t,
            None => false,
        }
    }
}

impl CardPlayer for IsDdPlayer {
    fn init_deal(&mut self, state: &GameState) {
        self.search.init_deal_with_config(state, self.seat, &self.config);
        if let Some(src) = self.source.as_mut() {
            src.init_deal(state, self.seat);
        }
        if let Some(early) = self.early.as_mut() {
            early.init_deal(state);
        }
    }

    fn observe(&mut self, state_before: &GameState, player: u8, action: u8) {
        self.search.record_action(state_before, player, action);
        if let Some(src) = self.source.as_mut() {
            src.observe(state_before, player, action);
        }
        if let Some(early) = self.early.as_mut() {
            early.observe(state_before, player, action);
        }
    }

    /// Only meaningful in time mode; in count mode the budget is deliberately
    /// unbounded and a clock would silently truncate the world count.
    fn set_time_budget(&mut self, ms: u32) {
        if self.config.time_limit_ms.is_some() {
            self.config.time_limit_ms = Some(ms);
        }
        if let Some(early) = self.early.as_mut() {
            early.set_time_budget(ms);
        }
    }

    fn decide(&mut self, state: &GameState, ctx: &MatchContext) -> Result<Decision, AgentError> {
        if self.use_early(state) {
            return self.early.as_mut().expect("switch_at implies an early player").decide(state, ctx);
        }

        let start = Instant::now();
        let result = match self.source.as_deref_mut() {
            Some(src) => {
                self.search.search_with_source(state, &self.config, &mut self.rng, src)?
            }
            None => self.search.search_with_stats(state, &self.config, &mut self.rng),
        };

        let elapsed = start.elapsed();
        telemetry::record(
            &result.worlds, &result.source, result.determinizations,
            elapsed.as_micros() as u64,
        );

        Ok(Decision {
            action: result.best_action,
            stats: Stats {
                source: "isdd",
                candidates: result.card_scores,
                determinizations: result.determinizations,
                worlds: result.worlds,
                elapsed_ms: elapsed.as_secs_f64() * 1000.0,
            },
        })
    }
}

/// D'où viennent les mondes que les recherches IS-DD résolvent, cumulé sur le
/// processus.
///
/// Le comptage par décision existe depuis toujours ([`WorldCounts`]) mais
/// n'était agrégé nulle part, si bien que « la file playgen s'assèche-t-elle ? »
/// n'avait pas de réponse — alors que c'est cette question, et elle seule, qui
/// dit si le belief net sert encore à quelque chose. Des atomiques relâchées :
/// le coût est nul devant une recherche DD, et le banc d'essai est
/// multi-thread.
pub mod telemetry {
    use super::super::super::is_dd::{SourceUsage, WorldCounts};
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Une case par nombre de cartes restantes (index 0 inutilisé, 1..=8).
    /// L'axe est obligatoire : entre l'entame et la finale, le coût d'un solve
    /// varie de quatre ordres de grandeur et la taille de l'espace des mondes
    /// s'effondre. Un agrégat sur toute la donne mélangerait les deux régimes
    /// et ne dirait rien de ni l'un ni l'autre.
    const LANES: usize = 9;

    macro_rules! lanes {
        ($name:ident) => {
            static $name: [AtomicU64; LANES] = [
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
                AtomicU64::new(0), AtomicU64::new(0), AtomicU64::new(0),
            ];
        };
    }

    lanes!(L_DECISIONS);
    lanes!(L_SEARCHED);
    lanes!(L_ROUNDS);
    lanes!(L_REQUESTED);
    lanes!(L_DELIVERED);
    lanes!(L_SOLVED);
    lanes!(L_DISCARDED);
    lanes!(L_SOURCE_US);
    lanes!(L_TOTAL_US);

    static DECISIONS: AtomicU64 = AtomicU64::new(0);
    static NO_SAMPLING: AtomicU64 = AtomicU64::new(0);
    static ALL_PLAYGEN: AtomicU64 = AtomicU64::new(0);
    static PARTIAL: AtomicU64 = AtomicU64::new(0);
    static NO_PLAYGEN: AtomicU64 = AtomicU64::new(0);
    static W_INJECTED: AtomicU64 = AtomicU64::new(0);
    static W_PLAYGEN: AtomicU64 = AtomicU64::new(0);
    static W_BELIEF: AtomicU64 = AtomicU64::new(0);
    static W_UNIFORM: AtomicU64 = AtomicU64::new(0);

    /// Ce qu'une décision a consommé comme mondes.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct Snapshot {
        /// Décisions IS-DD, toutes catégories confondues.
        pub decisions: u64,
        /// Coup forcé ou position résolue : aucun monde n'a été demandé. Ce
        /// n'est pas une dégradation, c'est une recherche qui n'a pas eu lieu.
        pub no_sampling: u64,
        /// Décisions échantillonnées dont *tous* les mondes viennent de la
        /// source (playgen).
        pub all_playgen: u64,
        /// ... dont une partie seulement : la file s'est vidée en cours de
        /// recherche.
        pub partial: u64,
        /// ... dont aucun : la source n'a rien produit du tout.
        pub no_playgen: u64,
        pub worlds_injected: u64,
        pub worlds_playgen: u64,
        /// Mondes tirés par déterminisation **pondérée**. À lire avec la
        /// configuration en main : ce sont les poids du belief net quand
        /// `use_nn_beliefs` est vrai (le cas de la prod et de `web_dede`), et
        /// ceux de `CardBeliefs` heuristique sinon. Le compteur dit « la file
        /// a séché », pas « le réseau a servi ».
        pub worlds_belief: u64,
        pub worlds_uniform: u64,
    }

    impl Snapshot {
        /// Décisions ayant réellement échantillonné des mondes.
        pub fn sampled(&self) -> u64 {
            self.all_playgen + self.partial + self.no_playgen
        }

        /// Part des mondes résolus qui ne venaient pas de la source, en %.
        /// C'est le chiffre qui décide du sort du belief net.
        pub fn fallback_world_pct(&self) -> f64 {
            let total = self.worlds_injected
                + self.worlds_playgen
                + self.worlds_belief
                + self.worlds_uniform;
            if total == 0 {
                return 0.0;
            }
            100.0 * (self.worlds_belief + self.worlds_uniform) as f64 / total as f64
        }
    }

    /// Ce qu'une recherche a demandé, reçu, résolu et jeté, par nombre de
    /// cartes restantes.
    #[derive(Debug, Default, Clone, Copy)]
    pub struct Lane {
        pub cards_left: u8,
        /// Décisions atteintes à ce stade, coups forcés compris.
        pub decisions: u64,
        /// ... dont celles qui ont réellement cherché.
        pub searched: u64,
        pub rounds: u64,
        pub requested: u64,
        pub delivered: u64,
        pub solved: u64,
        pub discarded: u64,
        /// Temps passé à attendre la source, cumulé (µs).
        pub source_us: u64,
        /// Temps total des décisions de cette tranche (µs).
        pub total_us: u64,
    }

    impl Lane {
        /// Part des mondes reçus qui a été résolue. Le complément est jeté à la
        /// fin de la recherche : un monde est échantillonné pour *une*
        /// position, il ne survit pas au coup suivant.
        pub fn used_pct(&self) -> f64 {
            if self.delivered == 0 {
                return 0.0;
            }
            100.0 * (self.delivered.saturating_sub(self.discarded)) as f64
                / self.delivered as f64
        }

        /// Part du temps de décision passée à attendre la source plutôt qu'à
        /// résoudre. Au-delà de ~50 % la recherche n'est plus limitée par le
        /// solveur : demander moins de mondes, ou les demander plus tôt,
        /// devient le seul levier.
        pub fn wait_pct(&self) -> f64 {
            if self.total_us == 0 {
                return 0.0;
            }
            100.0 * self.source_us as f64 / self.total_us as f64
        }

        /// Durée moyenne d'un aller-retour, en ms.
        pub fn ms_per_round(&self) -> f64 {
            if self.rounds == 0 {
                return 0.0;
            }
            self.source_us as f64 / self.rounds as f64 / 1000.0
        }

        /// Part des mondes demandés que la source a effectivement rendus.
        /// En dessous de 100 %, le sampler n'arrive plus à produire — c'est ce
        /// qui fait sécher la file et réveille le repli.
        pub fn fill_pct(&self) -> f64 {
            if self.requested == 0 {
                return 0.0;
            }
            100.0 * self.delivered as f64 / self.requested as f64
        }
    }

    pub(super) fn record(
        counts: &WorldCounts,
        usage: &SourceUsage,
        solved: u32,
        total_us: u64,
    ) {
        let lane = (usage.cards_left as usize).min(LANES - 1);
        L_DECISIONS[lane].fetch_add(1, Ordering::Relaxed);
        if counts.total() > 0 {
            L_SEARCHED[lane].fetch_add(1, Ordering::Relaxed);
        }
        L_ROUNDS[lane].fetch_add(usage.rounds as u64, Ordering::Relaxed);
        L_REQUESTED[lane].fetch_add(usage.requested as u64, Ordering::Relaxed);
        L_DELIVERED[lane].fetch_add(usage.delivered as u64, Ordering::Relaxed);
        L_SOLVED[lane].fetch_add(solved as u64, Ordering::Relaxed);
        L_DISCARDED[lane].fetch_add(usage.discarded as u64, Ordering::Relaxed);
        L_SOURCE_US[lane].fetch_add(usage.source_us, Ordering::Relaxed);
        L_TOTAL_US[lane].fetch_add(total_us, Ordering::Relaxed);

        DECISIONS.fetch_add(1, Ordering::Relaxed);
        W_INJECTED.fetch_add(counts.injected as u64, Ordering::Relaxed);
        W_PLAYGEN.fetch_add(counts.playgen as u64, Ordering::Relaxed);
        W_BELIEF.fetch_add(counts.belief as u64, Ordering::Relaxed);
        W_UNIFORM.fetch_add(counts.uniform as u64, Ordering::Relaxed);

        let total = counts.total();
        if total == 0 {
            NO_SAMPLING.fetch_add(1, Ordering::Relaxed);
        } else if counts.injected == total {
            ALL_PLAYGEN.fetch_add(1, Ordering::Relaxed);
        } else if counts.injected > 0 {
            PARTIAL.fetch_add(1, Ordering::Relaxed);
        } else {
            NO_PLAYGEN.fetch_add(1, Ordering::Relaxed);
        }
    }

    /// Les tranches par cartes restantes, de l'entame (8) à la dernière (1).
    pub fn lanes() -> Vec<Lane> {
        (1..LANES)
            .rev()
            .map(|i| Lane {
                cards_left: i as u8,
                decisions: L_DECISIONS[i].load(Ordering::Relaxed),
                searched: L_SEARCHED[i].load(Ordering::Relaxed),
                rounds: L_ROUNDS[i].load(Ordering::Relaxed),
                requested: L_REQUESTED[i].load(Ordering::Relaxed),
                delivered: L_DELIVERED[i].load(Ordering::Relaxed),
                solved: L_SOLVED[i].load(Ordering::Relaxed),
                discarded: L_DISCARDED[i].load(Ordering::Relaxed),
                source_us: L_SOURCE_US[i].load(Ordering::Relaxed),
                total_us: L_TOTAL_US[i].load(Ordering::Relaxed),
            })
            .filter(|l| l.decisions > 0)
            .collect()
    }

    pub fn snapshot() -> Snapshot {
        Snapshot {
            decisions: DECISIONS.load(Ordering::Relaxed),
            no_sampling: NO_SAMPLING.load(Ordering::Relaxed),
            all_playgen: ALL_PLAYGEN.load(Ordering::Relaxed),
            partial: PARTIAL.load(Ordering::Relaxed),
            no_playgen: NO_PLAYGEN.load(Ordering::Relaxed),
            worlds_injected: W_INJECTED.load(Ordering::Relaxed),
            worlds_playgen: W_PLAYGEN.load(Ordering::Relaxed),
            worlds_belief: W_BELIEF.load(Ordering::Relaxed),
            worlds_uniform: W_UNIFORM.load(Ordering::Relaxed),
        }
    }
}

/// Build the DMC player that fronts an [`IsDdPlayer`] in `dmc_then_dd` bots.
pub fn early_dmc(weights: Arc<DmcWeights>, residual: bool) -> Box<dyn CardPlayer> {
    Box::new(super::dmc::DmcPlayer::new(weights, residual))
}
