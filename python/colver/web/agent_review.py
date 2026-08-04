"""What the reference bots would have played, card by card, on a stored game.

For every card of the play phase, the position is replayed and three agents are
asked what they would do there — regardless of who actually played the card:

- **DouDou50** — the DMC Q-network, one forward pass, no search.
- **Oracle (DD)** — the double-dummy solver, all four hands visible. The
  ceiling, not a player.
- **Dédé (IS-DD)** — the production agent: samples worlds from playgen and
  solves each.

IS-DD is *seat-bound*: its beliefs, void tracking and world sampler all run
from one observer's point of view. So four of them are built, one per seat,
every one shown every action, and the one whose seat is to play is the one
asked. Anything else would be asking a bot what it would do with information
it could not have had.

The work is exposed as a **stream**, one card at a time and in play order, so
the page can annotate the opening lead while the endgame is still being
searched. [`stream`] is the real entry point; [`get_or_compute`] just drains it
for callers that only want the finished blob.

The result is cached in the `agent_review` table and computed once. Cost is
dominated by IS-DD — `COLVER_REVIEW_ISDD_MS` per card, minus the forced ones
and minus early termination, so ~9s per deal at the 500ms default. Every step
runs in a thread (the Rust search releases the GIL — true of `Agent.decide`, `solve_scores` and, since 2026-08-02, `action_oracle_dd`) so the event loop stays
free, and reviews are serialised across games so a burst of replay loads
cannot pile searches onto the playgen sidecar.
"""

import asyncio
import json
import logging
import os

import colver
import colver.web.agents as agents
import colver.web.database as db

logger = logging.getLogger(__name__)

# v2 : même purge que ANALYSIS_VERSION v5 — invalide les revues calculées en
# pleine donne avant le filtre `get_game` (2026-08-01).
REVIEW_VERSION = 2

# Per-card IS-DD budget for the review. Lower than the 1000ms the web plays at:
# a full deal is ~24 decisions and someone is watching them land.
ISDD_MS = int(os.environ.get("COLVER_REVIEW_ISDD_MS", "500"))

# One game reviewed at a time — the searches all hit the same GPU sidecar, and
# IS-DD is built with rayon here, so a second concurrent review would be taking
# cores away from people actually playing.
_gate = asyncio.Semaphore(1)
_locks = {}  # game_id -> asyncio.Lock, to avoid duplicate computations


def _build(kind, seat, **kw):
    """A bot for one seat, or None if its spec cannot be built."""
    try:
        return colver.Agent(agents.spec_for(kind, **kw), seat)
    except Exception as e:  # noqa: BLE001 — a missing model must not kill the review
        logger.warning("%s seat %s unavailable: %s", kind, seat, e)
        return None


def _ask(agent, env):
    """The agent's card here, or None if its search failed (sidecar down…)."""
    try:
        return int(agent.decide(env)["action"])
    except Exception as e:  # noqa: BLE001 — one bad move must not lose the review
        logger.warning("decide failed: %s", e)
        return None


def _count_cards(game):
    """How many cards the review will cover. Stepping is free; searching isn't."""
    env = colver.Env.deal_with_hands(game["dealer"], game["hands"])
    n = 0
    for entry in game["actions"]:
        if env.is_terminal():
            break
        action = int(entry["action"])
        if int(env.phase()) == 1:
            if action not in list(env.legal_actions()):
                break  # corrupt record — the runner will stop here too
            n += 1
        env.step(action)
    return n


class _Runner:
    """One game's review, advanced one card per `step()`.

    Split this way so the caller can hand each step to a thread and push the
    result out before starting the next one.
    """

    def __init__(self, game, *, play_model=None, belief_model=None):
        self._game = game
        self._play_model = play_model
        self._belief_model = belief_model
        self._env = None
        self._doudou = None
        self._dede = [None] * 4
        self._table = []
        self._idx = 0
        self._done = False
        self.moves = []

    def start(self):
        """Build the bots, seat them, and return the number of cards to come."""
        game = self._game
        self._env = colver.Env.deal_with_hands(game["dealer"], game["hands"])
        # DouDou50 reads the position from whoever is to play, so one instance
        # covers all four seats. IS-DD does not — hence one per seat.
        if self._play_model:
            self._doudou = _build("doudou", 0, play_model=self._play_model)
        self._dede = [
            _build("dede", s, belief_model=self._belief_model, time_ms=ISDD_MS)
            for s in range(4)
        ]
        self._table = [a for a in ([self._doudou] + self._dede) if a is not None]
        for agent in self._table:
            agent.init_deal(self._env)
        return _count_cards(game)

    def step(self):
        """Next card's entry, or None once the deal is exhausted."""
        actions = self._game["actions"]
        while not self._done and self._idx < len(actions):
            if self._env.is_terminal():
                break
            action = int(actions[self._idx]["action"])
            phase = int(self._env.phase())
            player = int(self._env.current_player())

            move = None
            if phase == 1:
                legals = list(self._env.legal_actions())
                if action not in legals:
                    break  # corrupt record — stop rather than emit nonsense
                move = {"idx": self._idx, "player": player, "action": action}
                if len(legals) == 1:
                    move["forced"] = True
                else:
                    if self._doudou is not None:
                        move["doudou"] = _ask(self._doudou, self._env)
                    try:
                        move["oracle"] = int(self._env.action_oracle_dd())
                    except Exception:  # noqa: BLE001
                        pass
                    if self._dede[player] is not None:
                        move["isdd"] = _ask(self._dede[player], self._env)

            for agent in self._table:
                agent.observe(self._env, action)
            self._env.step(action)
            self._idx += 1

            if move is not None:
                self.moves.append(move)
                return move

        self._done = True
        return None

    def finish(self):
        """The cacheable blob, once `step()` has returned None."""
        # `available` is reported from what actually landed in the data, not
        # from what we set out to build: a bot that built fine but errored on
        # every search is absent.
        decisions = [m for m in self.moves if not m.get("forced")]
        return {
            "version": REVIEW_VERSION,
            "isdd_ms": ISDD_MS,
            "available": {
                key: any(m.get(key) is not None for m in decisions)
                for key in ("doudou", "oracle", "isdd")
            },
            "moves": self.moves,
        }


def _expected(play_model):
    """Les bots dont l'absence, dans une revue en cache, justifie de la refaire.

    « Attendu » veut dire disponible *maintenant*. Exiger IS-DD sur une machine
    qui n'a pas de sidecar ferait recalculer la revue à chaque ouverture de
    Rejouer sans qu'elle puisse jamais aboutir — d'où `REQUIRE_SIDECAR`, qui est
    précisément la déclaration « ce déploiement attend un sidecar ».
    """
    keys = ["oracle"]  # DD pur : ni modèle ni réseau, toujours attendu
    if play_model:
        keys.append("doudou")
    if agents.sidecar_expected():
        keys.append("isdd")
    return keys


def _fresh(blob, play_model=None):
    """Une revue en cache est périmée à un bump de version — **et aussi quand
    elle a été calculée sans un bot qui répond aujourd'hui.**

    Sans ce second test, une revue produite pendant que le sidecar playgen était
    injoignable (les quatre Dédé échouent alors à la construction, cf. `_build`)
    est écrite avec `available.isdd = false` et servie **pour toujours** : aucun
    bump ne la rattrapera, puisqu'elle porte bien la version courante. La donne
    perd sa colonne IS-DD définitivement, en silence.

    `analysis._is_fresh` ferme exactement ce trou-là pour playgen ; c'est le
    même raisonnement, sur le bot au lieu du modèle. Le test est volontairement
    dissymétrique : on recalcule quand un bot manquant est *redevenu*
    disponible, jamais l'inverse — sinon un sidecar tombé invaliderait tout le
    cache au pire moment.
    """
    if blob is None or blob.get("version") != REVIEW_VERSION:
        return False
    available = blob.get("available") or {}
    return all(available.get(key) for key in _expected(play_model))


async def stream(game_id, *, play_model=None, belief_model=None):
    """Yield `(kind, payload)` as the review progresses, in play order.

    `("done", blob)` immediately when it was already cached, otherwise
    `("start", total)`, then one `("move", entry)` per card, then
    `("done", blob)`. `("error", msg)` if the game cannot be reviewed at all.

    Abandoning the generator (task cancellation, client gone) unwinds the lock
    and the semaphore, and nothing partial is written to the cache.
    """
    cached = await db.get_agent_review(game_id)
    if _fresh(cached, play_model):
        yield "done", cached
        return

    game = await db.get_game(game_id)
    if game is None:
        yield "error", "Partie introuvable"
        return
    if not game["actions"]:
        yield "error", "Aucune action à analyser"
        return

    lock = _locks.setdefault(game_id, asyncio.Lock())
    async with lock:
        # Another request may have computed it while we waited on the lock
        cached = await db.get_agent_review(game_id)
        if _fresh(cached, play_model):
            yield "done", cached
            return

        async with _gate:
            runner = _Runner(
                game, play_model=play_model, belief_model=belief_model)
            total = await asyncio.to_thread(runner.start)
            yield "start", total
            while True:
                move = await asyncio.to_thread(runner.step)
                if move is None:
                    break
                yield "move", move
            blob = runner.finish()

        await db.save_agent_review(game_id, json.dumps(blob))
    _locks.pop(game_id, None)
    yield "done", blob


async def get_or_compute(game_id, *, play_model=None, belief_model=None):
    """The finished review for a game, computing it on first request."""
    result = None
    gen = stream(game_id, play_model=play_model, belief_model=belief_model)
    try:
        async for kind, payload in gen:
            if kind == "error":
                return None, payload
            if kind == "done":
                result = payload
    finally:
        await gen.aclose()
    return result, None
