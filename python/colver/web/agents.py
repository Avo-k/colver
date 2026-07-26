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

import os

import colver

# Per-move IS-DD budget, in ms, when the session does not set one.
DEFAULT_TIME_MS = 1000

# Fixed determinization count. 0 (default) = time mode: search until the
# per-move budget runs out. >0 = count mode: solve exactly N worlds however
# long that takes — reproducible, but the opening lead can take many seconds
# on a small machine, so production stays in time mode.
ISDD_DETS = int(os.environ.get("COLVER_ISDD_DETS", "0"))

# Worlds requested per sidecar round trip under a time budget.
ISDD_WORLD_BATCH = int(os.environ.get("COLVER_ISDD_PLAYGEN_WORLDS", "256"))

# Playgen GPU sidecar. Empty = no sidecar configured, in which case IS-DD bots
# sample constraint-uniform worlds and say so in their stats.
SIDECAR_URL = os.environ.get("COLVER_PLAYGEN_GPU_URL", "").rstrip("/")

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
    )
    belief = f'\n[belief]\nmodel = "{belief_model}"\n' if belief_model else ""
    return bid + play + "\n" + _worlds_section() + belief


class AgentTable:
    """The bots seated at one table, keyed by seat.

    Seats without a bot (human players, or a spec that failed to build) simply
    have no entry; `observe` still runs for every seat that does.
    """

    def __init__(self, kinds, *, bid_model=None, play_model=None, belief_model=None, time_ms=None):
        """`kinds`: {seat: agent_type} for the seats played by bots."""
        self.kinds = dict(kinds)
        self.agents = {}
        self.errors = {}
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
                print(f"[agents] seat {seat} ({kind}) unavailable: {e}")

    def __bool__(self):
        return bool(self.agents)

    def init_deal(self, env):
        """Start a deal. `env` must be the fresh, pre-auction position."""
        for agent in self.agents.values():
            agent.init_deal(env)

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
        """Full decision dict for `seat`, or `None` if that seat has no bot."""
        agent = self.agents.get(int(seat))
        if agent is None:
            return None
        return agent.decide(env)


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
    elif source == "bid_nn":
        stats["bid_nn"] = {
            "q_values": [[a, round(s, 3)] for a, s in candidates],
            "best_action": int(decision["action"]),
        }

    elapsed = decision.get("elapsed_ms")
    if elapsed:
        stats["elapsed_ms"] = round(float(elapsed), 1)
    return stats
