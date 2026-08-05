"""Bots for the web, as `colver.Agent` objects.

The web used to assemble its bots by hand: load a belief net onto the `Env`,
call `dede_init()`, fetch playgen worlds from the GPU sidecar, inject them,
then call `action_dede()`. The arena did not do the injection step, so the two
ran measurably different agents under the same name.

Now a bot is described by a spec (the same TOML the arena reads) and built by
the Rust side, which owns its own world source. This module is only the
translation from the web's agent-type names to specs, plus the four-seat
bookkeeping.

Every agent must see every action — that is what keeps beliefs and world
samplers in sync with the game — so `AgentTable.observe()` must be called
before `env.step()` for **all** moves, human ones included.
"""

import collections
import logging
import os
import statistics
import time

import colver

logger = logging.getLogger(__name__)

# Per-move IS-DD budget, in ms, when the session does not set one.
DEFAULT_TIME_MS = 1000

# Fixed determinization count. 0 (default) = time mode: search until the
# per-move budget runs out. >0 = count mode: solve exactly N worlds however
# long that takes — reproducible, but the opening lead can take many seconds
# on a small machine, so production stays in time mode.
ISDD_DETS = int(os.environ.get("COLVER_ISDD_DETS", "0"))

# Worlds requested per sidecar round trip under a time budget.
ISDD_WORLD_BATCH = int(os.environ.get("COLVER_ISDD_PLAYGEN_WORLDS", "256"))

# Plafond et **plancher** de mondes résolus par coup, sous échéance.
# 0 = pas de borne.
#
# Le plancher est un choix de politique, pas un réglage de performance : sous
# pression de calcul — le GPU de la prod est partagé — la dégradation doit se
# payer en **latence**, qui se voit, et non en **force de jeu**, qui ne se voit
# pas. Sans lui, un GPU chargé faisait rendre à Dédé une réponse à l'heure
# fondée sur une poignée de mondes, au rythme habituel, simplement moins bonne.
#
# Le plancher est posé **au genou mesuré** : le regret contre une référence à
# 2000 mondes passe sous 0,10 point DD dès 60 mondes (`isdd_dets_by_stage`, 250
# positions). 64 garantit donc du jeu essentiellement convergé, quoi qu'il
# arrive au GPU.
#
# Le plafond reste large (256, ~4× le genou) : quand la carte est libre, ce
# temps ne coûte rien à personne. Mais c'est **lui** qui fabrique la contention,
# pas le plancher — si la latence par coup devient visible pour les joueurs,
# c'est 256 qu'il faut baisser, pas 64.
ISDD_MAX_WORLDS = int(os.environ.get("COLVER_ISDD_MAX_WORLDS", "256"))
ISDD_MIN_WORLDS = int(os.environ.get("COLVER_ISDD_MIN_WORLDS", "64"))

# Playgen GPU sidecar. Empty = no sidecar configured, in which case IS-DD bots
# sample constraint-uniform worlds and say so in their stats.
SIDECAR_URL = os.environ.get("COLVER_PLAYGEN_GPU_URL", "").rstrip("/")

# Le déploiement *déclare* qu'il attend un sidecar. Sans ça, « pas de sidecar »
# est indiscernable d'un choix : une machine de dev sans GPU est parfaitement
# normale, une prod sans playgen ne l'est pas. La prod a tourné plus d'un jour
# sur des mondes uniformes sans que rien ne l'annonce, parce que la seule
# différence entre les deux cas vivait dans la tête de l'exploitant.
REQUIRE_SIDECAR = os.environ.get("COLVER_REQUIRE_SIDECAR", "").strip().lower() \
    in ("1", "true", "yes", "on")

# Journalisation des décisions dégradées : au plus une ligne par fenêtre, avec
# le compte. Un coup de bot par siège et par pli, c'est ~24 par donne — sans
# plafond, une panne de sidecar noierait le journal au lieu de le renseigner.
_DEGRADED_LOG_WINDOW = 60.0
_degraded = {"since": 0.0, "count": 0}


def sidecar_expected() -> bool:
    return REQUIRE_SIDECAR


def log_startup_state():
    """Dire, au démarrage, avec quels mondes Dédé va jouer.

    `[worlds] fallback = "uniform"` est un choix délibéré — le web préfère finir
    la donne à être exactement aussi fort que le banc d'essai — mais un repli
    silencieux est un repli qu'on découvre des semaines plus tard, à la force de
    jeu. Une ligne au démarrage coûte zéro et ferme ce trou-là.
    """
    if SIDECAR_URL:
        logger.info("IS-DD : mondes playgen via le sidecar %s "
                    "(repli uniforme si indisponible)", SIDECAR_URL)
        return
    message = ("IS-DD : aucun sidecar playgen configuré "
               "(COLVER_PLAYGEN_GPU_URL vide) — Dédé échantillonne des mondes "
               "contraints-uniformes et joue donc plus faiblement qu'attendu")
    if REQUIRE_SIDECAR:
        logger.error("%s ; COLVER_REQUIRE_SIDECAR est pourtant activé", message)
    else:
        logger.warning(message)


# Origine des mondes, cumulée sur la vie du processus. C'est la seule façon de
# répondre à « la file playgen s'assèche-t-elle, et à quelle fréquence ? » : les
# compteurs par décision existaient déjà (`WorldCounts`) mais partaient au
# client et nulle part ailleurs. Publié par `/health`.
_WORLD_STATS = {
    "decisions": 0,      # décisions IS-DD, toutes catégories
    "no_sampling": 0,    # coup forcé ou position résolue : aucun monde demandé
    "sampled": 0,        # décisions ayant réellement échantillonné
    "all_playgen": 0,    # ... dont tous les mondes viennent du sidecar
    "partial": 0,        # ... dont une partie seulement
    "no_playgen": 0,     # ... dont aucun
    "worlds_injected": 0,
    "worlds_playgen": 0,
    "worlds_belief": 0,
    "worlds_uniform": 0,
}

# Mondes résolus des 200 dernières décisions échantillonnées, toutes tables
# confondues. 200 ≈ sept donnes : assez long pour ne pas sauter sur une
# position atypique, assez court pour qu'une dégradation apparaisse en quelques
# minutes plutôt que d'être diluée dans l'historique du processus.
_RECENT_WORLDS = collections.deque(maxlen=200)


def recent_worlds_per_decision():
    """Mondes résolus par décision sur la fenêtre récente, ou `None` si vide.

    C'est **la** jauge de santé d'IS-DD : elle doit se tenir entre le plancher
    et le plafond de la spec (`min_worlds` / `max_worlds`). En dessous du
    plancher, quelque chose empêche la recherche d'atteindre son compte ; au
    plafond en permanence, le budget de temps n'est jamais la contrainte.

    **Seul le jeu au budget de production l'alimente** (`window=True`). Les
    pages d'analyse et le panneau des problèmes cherchent à 500 et 100 ms : ils
    atteignent le plancher, donc la lecture « entre plancher et plafond »
    survivrait — mais ils tireraient la **moyenne** vers le bas dès qu'un joueur
    ouvre Rejouer, et cette baisse se lirait comme une pression GPU. Leurs
    mondes sont comptés ailleurs, dans [`world_stats`].
    """
    if not _RECENT_WORLDS:
        return None
    xs = sorted(_RECENT_WORLDS)
    return {
        "n": len(xs),
        "mean": round(statistics.fmean(xs), 1),
        "min": xs[0],
        "p50": xs[len(xs) // 2],
        "max": xs[-1],
    }


def world_stats():
    """Instantané des compteurs d'origine de mondes."""
    return dict(_WORLD_STATS)


def _note_worlds(stats, window=True):
    """Comptabiliser l'origine des mondes d'une décision IS-DD.

    `window` : cette décision entre-t-elle dans la fenêtre glissante ? **Non**
    pour les pages d'analyse — voir [`note_decision`].
    """
    worlds = stats.get("worlds") or {}
    total = sum(int(v) for v in worlds.values())
    _WORLD_STATS["decisions"] += 1
    # Fenêtre glissante des dernières décisions échantillonnées. Les compteurs
    # ci-dessous sont **cumulés depuis le démarrage** : parfaits pour un total,
    # inutilisables pour répondre à « est-ce que ça va *maintenant* ? » — une
    # heure de bon fonctionnement noie une dégradation en cours. La fenêtre
    # répond à cette question-là, et c'est elle que /health publie.
    if total and window:
        _RECENT_WORLDS.append(int(stats.get("determinizations") or total))
    for key in ("injected", "playgen", "belief", "uniform"):
        _WORLD_STATS["worlds_" + key] += int(worlds.get(key, 0))
    if total == 0:
        _WORLD_STATS["no_sampling"] += 1
        return
    _WORLD_STATS["sampled"] += 1
    injected = int(worlds.get("injected", 0))
    if injected == total:
        _WORLD_STATS["all_playgen"] += 1
    elif injected:
        _WORLD_STATS["partial"] += 1
    else:
        _WORLD_STATS["no_playgen"] += 1


def _note_degraded(seat, stats):
    """Compter, et dire de temps en temps, qu'une décision s'est repliée.

    `worlds_source` était déjà calculé — mais envoyé au *client*, jamais au
    journal. Personne ne regarde une interface à trois heures du matin ; c'est
    exactement la deuxième dégradation silencieuse que le backlog signale
    (docs/web_todo.md §2.2).

    **Une décision qui n'a demandé aucun monde n'est pas une décision
    dégradée.** Sur un coup forcé (une seule carte légale) comme sur une
    position résolue, `run_search` sort avant d'échantillonner et renvoie des
    compteurs à zéro ; `decision_stats` lit alors `sourced == 0` et étiquette
    `"cpu"`, ce qui n'a rien à voir avec l'état du sidecar. Les 14 alertes
    présentes en prod le 2026-08-03 étaient *toutes* de cette forme
    (`source cpu, 0 mondes`) : l'unique alarme censée détecter une panne
    silencieuse criait au loup plusieurs fois par jour, ce qui est la façon la
    plus sûre de la rendre illisible.
    """
    if not SIDECAR_URL or stats.get("worlds_source") == "playgen-gpu":
        return
    if not sum(int(v) for v in (stats.get("worlds") or {}).values()):
        return  # rien n'a été échantillonné : rien n'est dégradé
    now = time.monotonic()
    _degraded["count"] += 1
    if now - _degraded["since"] < _DEGRADED_LOG_WINDOW:
        return
    logger.warning(
        "IS-DD dégradé : %d décision(s) sans mondes playgen depuis %.0f s "
        "(dernière : siège %s, source %s, %s mondes) — sidecar %s injoignable ?",
        _degraded["count"], now - _degraded["since"] if _degraded["since"] else 0,
        seat, stats.get("worlds_source"), stats.get("determinizations"),
        SIDECAR_URL)
    _degraded["since"] = now
    _degraded["count"] = 0


def note_decision(kind, decision, *, seat=None, window=True):
    """Comptabiliser une décision IS-DD, **d'où qu'elle vienne**.

    Les compteurs vivaient dans `AgentTable.decide`, donc seul le jeu réel y
    entrait : la revue d'agents (`agent_review`) et l'analyse du jeu
    (`card_analysis`) construisent leurs `colver.Agent` à la main et appellent
    `decide` en direct. Or ce sont de **gros** consommateurs de mondes — une
    revue, c'est ~20 recherches par donne. Résultat : `/health` publiait un bloc
    `worlds` entièrement à zéro juste après une revue, et une jauge qui
    sous-rapporte est pire qu'aucune, parce qu'elle rassure. Constaté sur la
    prod le 2026-08-06, après le redéploiement du sidecar.

    `window` sépare les deux questions, et **c'est le point à ne pas rater** :

    - les compteurs d'**origine** (playgen / belief / uniforme) répondent à « la
      file playgen s'assèche-t-elle ? ». La réponse ne dépend pas du budget, donc
      toutes les décisions comptent ;
    - la fenêtre glissante (`recent_worlds_per_decision`) répond à « IS-DD
      atteint-il son compte de mondes ? », et se lit **contre le plancher et le
      plafond de la spec**. Or les pages d'analyse tournent à
      `COLVER_REVIEW_ISDD_MS` (500 ms), moitié moins que le jeu : les y verser
      ferait passer la jauge sous son plancher en permanence dès qu'un joueur
      ouvre Rejouer. On transformerait une alarme qui marche en bruit de fond —
      exactement ce que `_note_degraded` a déjà eu à corriger une fois.

    Rend le blob de stats, que l'appelant peut réutiliser.
    """
    if decision is None or decision.get("source") != "isdd":
        return None
    stats = decision_stats(kind, decision)
    _note_worlds(stats, window=window)
    _note_degraded(seat, stats)
    return stats


AGENT_NAMES = {
    "dede": "Dédé (IS-DD)",
    "doudou": "DouDou50",
    "oracle_dd": "Oracle (DD)",
}


def _worlds_section() -> str:
    """World source for IS-DD bots.

    The web prefers finishing the deal to being exactly as strong as the
    benchmark, so it opts into `fallback = "uniform"`: if the GPU sidecar goes
    down mid-game the player still gets a move. The substitution is not hidden
    — it shows up in the decision's `worlds` counts.
    """
    if not SIDECAR_URL:
        return '[worlds]\nsource = "uniform"\n'
    return (
        "[worlds]\n"
        'source = "sidecar"\n'
        f'url = "{SIDECAR_URL}"\n'
        f"batch = {ISDD_WORLD_BATCH}\n"
        'fallback = "uniform"\n'
    )


def spec_for(kind, *, bid_model=None, play_model=None, belief_model=None, time_ms=None) -> str:
    """Bot spec (TOML text) for one of the web's agent types."""
    time_ms = DEFAULT_TIME_MS if time_ms is None else int(time_ms)

    if bid_model:
        bid = f'[bid]\nstrategy = "nn"\nmodel = "{bid_model}"\nhidden = 512\n'
    else:
        bid = '[bid]\nstrategy = "improved_v2"\n'

    if kind == "doudou":
        if not play_model:
            raise ValueError("agent 'doudou' needs a play model")
        return bid + f'\n[play]\nmethod = "dmc"\nmodel = "{play_model}"\nresidual = true\n'

    if kind == "oracle_dd":
        return bid + '\n[play]\nmethod = "oracle_dd"\n'

    # "dede" and anything unrecognised: IS-DD, the production agent.
    play = (
        "\n[play]\n"
        'method = "isdd"\n'
        # In count mode the time budget must be zero, or it wins.
        f"time_ms = {0 if ISDD_DETS > 0 else time_ms}\n"
        f"determinizations = {ISDD_DETS if ISDD_DETS > 0 else 20}\n"
        # Sous échéance, plafonner les mondes résolus par coup. La réponse cesse
        # de bouger bien avant : regret sous 0,10 point DD dès 60 mondes, sous
        # 0,03 dès 15 en fin de donne (`isdd_dets_by_stage`, 250 positions),
        # alors que Dédé en traversait 256 à 697 selon le stade. Le temps
        # au-delà n'achète que de la charge GPU — et le joueur ne la voit pas,
        # `pacing.hold` absorbe le temps rendu. Sans effet en mode compte.
        f"max_worlds = {ISDD_MAX_WORLDS}\n"
        f"min_worlds = {ISDD_MIN_WORLDS}\n"
    )
    belief = f'\n[belief]\nmodel = "{belief_model}"\n' if belief_model else ""
    return bid + play + "\n" + _worlds_section() + belief


class AgentTable:
    """The bots seated at one table, keyed by seat.

    Seats without a bot (human players, or a spec that failed to build) simply
    have no entry; `observe` still runs for every seat that does.
    """

    def __init__(self, kinds, *, bid_model=None, play_model=None, belief_model=None,
                 time_ms=None, window=True):
        """`kinds`: {seat: agent_type} for the seats played by bots.

        `window` : cette table joue-t-elle au budget du jeu réel ? Mettre
        `False` pour une table sonde (cf. [`note_decision`]) — le panneau des
        problèmes interroge Dédé à 100 ms, un dixième du jeu, et ses décisions
        tireraient la jauge vers le bas sans qu'aucune dégradation ait eu lieu.
        """
        self.kinds = dict(kinds)
        self.agents = {}
        self.errors = {}
        self.window = bool(window)
        # Mondes résolus et temps par décision IS-DD de la donne en cours.
        self._deal_worlds = []
        self._deal_ms = []
        for seat, kind in self.kinds.items():
            spec = spec_for(
                kind,
                bid_model=bid_model,
                play_model=play_model,
                belief_model=belief_model,
                time_ms=time_ms,
            )
            try:
                self.agents[int(seat)] = colver.Agent(spec, int(seat))
            except Exception as e:  # noqa: BLE001 — a bad model must not kill the session
                self.errors[int(seat)] = str(e)
                logger.warning("seat %s (%s) unavailable: %s", seat, kind, e)

    def __bool__(self):
        return bool(self.agents)

    def init_deal(self, env):
        """Start a deal. `env` must be the fresh, pre-auction position."""
        # Filet : si un pilote oublie `end_deal()`, la donne précédente est tout
        # de même rapportée ici. Ne couvre pas la dernière donne d'une session,
        # d'où `end_deal()` sur les deux chemins terminaux.
        self.end_deal()
        for agent in self.agents.values():
            agent.init_deal(env)

    def end_deal(self):
        """Journaliser les mondes par décision de la donne écoulée, puis remettre à zéro.

        La jauge que ça produit répond à « DD travaille-t-il normalement ? »
        **à l'échelle d'une donne**, ce qu'aucun compteur cumulé ne peut dire :
        une heure de bon fonctionnement noie une donne dégradée. Une ligne par
        donne (~24 décisions de bot) est un débit de journal négligeable, et
        c'est la granularité à laquelle on lit ensuite une régression.

        Le `min` compte autant que la moyenne : c'est lui qui dit si le plancher
        a tenu. Une moyenne confortable peut cacher deux coups à 5 mondes.
        """
        if not self._deal_worlds:
            return
        xs = sorted(self._deal_worlds)
        self._deal_worlds = []
        logger.info(
            "IS-DD donne terminée : %d décisions échantillonnées, "
            "%.1f mondes/décision (min %d, médiane %d, max %d), %.0f ms/coup",
            len(xs), statistics.fmean(xs), xs[0], xs[len(xs) // 2], xs[-1],
            statistics.fmean(self._deal_ms) if self._deal_ms else 0.0,
        )
        self._deal_ms = []

    def set_time_ms(self, ms):
        """Retune the per-move budget without rebuilding (the Regarder page)."""
        for agent in self.agents.values():
            agent.set_time_ms(int(ms))

    def set_scores(self, ns, ew):
        for agent in self.agents.values():
            agent.set_scores(int(ns), int(ew))

    def observe(self, env, action):
        """Show an action to every bot. Call with `env` still *before* the move."""
        for agent in self.agents.values():
            agent.observe(env, int(action))

    def error(self, seat):
        """Why this seat has no bot, if it should have had one."""
        return self.errors.get(int(seat))

    def kind(self, seat):
        return self.kinds.get(int(seat), "dede")

    def label(self, seat):
        return AGENT_NAMES.get(self.kind(seat), self.kind(seat))

    def decide(self, env, seat):
        """Full decision dict for `seat`, or `None` if that seat has no bot.

        Point de passage unique de tout coup de bot **joué**, solo comme salon.
        Le comptage lui-même vit dans [`note_decision`] : les pages d'analyse
        décident hors de cette table et doivent compter aussi. Ce qui reste ici
        est ce qui n'a de sens qu'à une table — la jauge par donne, et la
        fenêtre glissante, qui se lit contre le budget du jeu réel.
        """
        agent = self.agents.get(int(seat))
        if agent is None:
            return None
        decision = agent.decide(env)
        noted = note_decision(self.kind(seat), decision, seat=seat, window=self.window)
        if noted is not None and self.window:
            dets = int(decision.get("determinizations") or 0)
            if dets:
                # Les décisions sans échantillonnage (coup forcé, position
                # résolue) sont exclues : les compter tirerait la moyenne vers
                # le bas sans qu'aucune dégradation n'ait eu lieu.
                self._deal_worlds.append(dets)
                self._deal_ms.append(float(decision.get("elapsed_ms") or 0.0))
                logger.debug("IS-DD siège %s : %d mondes en %.0f ms",
                             seat, dets, decision.get("elapsed_ms") or 0.0)
        return decision


def decision_stats(kind, decision, error=None):
    """Shape a Rust decision into the stats blob the frontend expects.

    When the seat's bot failed to build, `error` is carried through instead of
    being swallowed: a seat quietly playing heuristic moves under a bot's name
    is exactly the kind of silent degradation this refactor exists to stop.
    """
    stats = {"agent": kind, "agent_label": AGENT_NAMES.get(kind, kind)}
    if error:
        stats["error"] = error
    if decision is None:
        return stats

    source = decision.get("source")
    candidates = [[int(a), float(s)] for a, s in decision.get("candidates", [])]
    if source == "isdd":
        stats["card_scores"] = [[a, round(s, 1)] for a, s in candidates]
        # IS-DD et l'Oracle publient tous deux `card_scores`, sur deux échelles
        # différentes depuis que l'objectif par défaut est le score de donne :
        # écart marqué N-S − E-O ici, points cartes 0-252 là. Sans ce champ le
        # client afficherait les deux avec la même légende.
        stats["score_scale"] = "deal_score"
        stats["determinizations"] = int(decision.get("determinizations", 0))
        worlds = decision.get("worlds") or {}
        stats["worlds"] = {k: int(v) for k, v in worlds.items()}
        # Which sampler actually produced the solved worlds, so a sidecar
        # outage is visible in the UI rather than just felt in the play.
        sourced = int(worlds.get("injected", 0))
        total = sum(int(v) for v in worlds.values())
        stats["worlds_source"] = (
            "playgen-gpu" if sourced and sourced == total
            else "mixed" if sourced
            else "cpu"
        )
    elif source == "dmc":
        stats["q_values"] = [[a, round(s, 4)] for a, s in candidates]
    elif source == "oracle":
        stats["card_scores"] = [[a, round(s, 1)] for a, s in candidates]
        # Un solve DD nu : points cartes, sans contrat ni belote.
        stats["score_scale"] = "card_points"
    elif source == "bid_nn":
        stats["bid_nn"] = {
            "q_values": [[a, round(s, 3)] for a, s in candidates],
            "best_action": int(decision["action"]),
        }

    elapsed = decision.get("elapsed_ms")
    if elapsed:
        stats["elapsed_ms"] = round(float(elapsed), 1)
    return stats
