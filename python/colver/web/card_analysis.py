"""One card decision, weighed against the worlds the player could not see.

The annonces page asks « que vaut cette main ? » from a hand and an auction.
This module asks the same question one level down: **at this exact position,
what was each legal card worth?** — and it insists on separating two questions
that look alike and are not:

1. **Le vrai monde.** One double-dummy solve on the real deal. Exact, and
   already what [`analysis.py`] shows in Rejouer: the best card *in perfect
   information*. It says nothing about whether the choice was reasonable.
2. **Les mondes de l'information set.** Worlds are sampled from what the
   acting seat could actually know, each is solved, and the candidates are
   compared across that distribution. This is the honest question — « était-ce
   un bon choix compte tenu de ce que ce siège savait ? » — and it is the only
   thing here that Rejouer does not already answer.

Both are reported, never merged. A card that is second-best in the real deal
but best in 70 % of the worlds was a *good* card played against bad luck, and
collapsing the two columns would hide exactly that.

Two structural notes:

- **Rows are every legal card**, and `legal_actions_reduced()` is used only to
  decide whether there is a decision to analyse at all. The reduction returns
  one representative per equivalence class without saying which class a card
  belongs to, so using it for the rows would drop the row a bot's answer needs
  to land on. Equivalent cards therefore appear as separate rows carrying
  near-identical numbers — visibly redundant rather than invisibly missing.
- **Un solve couvre toutes les candidates.** `solve_scores()` returns the NS
  points of *every* legal card at once, so the Oracle side costs `n_worlds`
  solves regardless of how many candidates there are. The played-out side does
  not share: each candidate must be forced separately, so it is
  `n_cards × n_worlds` and it is what actually bounds the page.

Points are always reported **from Nord-Sud's side** (as the Oracle panel of the
annonces page does); only the regret is oriented towards the acting team.
"""

import logging
import os
import random
import threading

import colver
import colver.web.agents as agents
import colver.web.playgen_gpu as playgen_gpu

logger = logging.getLogger(__name__)

# Worlds solved by the Oracle, keyed by cards left in a hand. A solve gets
# cheaper fast as the deal empties, so the count rises instead of the cost.
WORLDS_BY_CARDS_LEFT = {8: 200, 7: 260, 6: 340, 5: 420, 4: 500, 3: 500, 2: 500, 1: 500}

# Total forced playouts the « jeu réel » side may spend. Divided by the number
# of candidates, so a 6-way choice does not cost triple a 2-way one.
REAL_PLAYOUT_BUDGET = int(os.environ.get("COLVER_CARD_REAL_BUDGET", "600"))

# Per-move IS-DD budget for the single Dédé opinion on this position.
ISDD_MS = int(os.environ.get("COLVER_CARD_ISDD_MS", "800"))

# A regret this large or worse is what the risk column counts, in card points.
RISK_THRESHOLD = 10

_tls = threading.local()


# ── position ──

def replay_to(dealer, initial_hands, actions, upto):
    """`(env, played)` at action index `upto`.

    `played[seat]` is the cards that seat has already laid down — what a
    sampled world has to be completed with to become a solvable deal.
    """
    env = colver.Env.deal_with_hands(int(dealer), [list(h) for h in initial_hands])
    played = [[] for _ in range(4)]
    for a in actions[:upto]:
        a = int(a)
        if int(env.phase()) == 1:
            played[int(env.current_player())].append(a)
        env.step(a)
    return env, played


def describe(dealer, initial_hands, actions, upto):
    """Everything the page needs to draw the position, or `{"error": ...}`.

    Kept separate from the simulation so the page can paint the trick, the
    hand and the candidates immediately and let the numbers stream in.
    """
    try:
        env, played = replay_to(dealer, initial_hands, actions, upto)
    except Exception as e:  # noqa: BLE001 — a bad CFN is a user error, not a crash
        return {"error": f"Position illisible : {e}"}

    if env.is_terminal():
        return {"error": "La donne est terminée à cet endroit"}
    if int(env.phase()) != 1:
        return {"error": "Cette action est une annonce, pas une carte",
                "phase": 0}

    seat = int(env.current_player())
    legal = [int(a) for a in env.legal_actions()]
    reduced = [int(a) for a in env.legal_actions_reduced()]
    # Rows are the **full** legal set, not the reduced one. The reduction only
    # hands back one representative per equivalence class and never says which
    # class a given card fell into, so a bot answering with a collapsed card —
    # the 8 when the 9 was kept — would have had no row to land on and its badge
    # would silently vanish. One solve covers every legal card anyway, and the
    # playout budget already divides by the row count, so completeness is free
    # on the Oracle side and self-limiting on the played-out side. A card absent
    # from `reduced` is equivalent to another legal one; that is worth marking,
    # and it is all the reduction can honestly tell us.
    candidates = legal
    tricks_won = list(env.get_tricks_won())
    return {
        "seat": seat,
        "phase": 1,
        "contract": env.get_contract(),
        "current_trick": list(env.get_current_trick()),
        "trick_lead": int(env.get_trick_lead()),
        "hands": [list(h) for h in env.get_hands()],
        "played": played,
        "legal": legal,
        "candidates": candidates,
        "reduced": reduced,
        # No real decision when every legal card collapses to one class: the 7
        # and the 8 of a plain suit with nothing outstanding between them are
        # the same move. `legal` can hold several cards while `reduced` holds one.
        "forced": len(reduced) < 2,
        "points": list(env.get_points()),
        "tricks_won": tricks_won,
        "trick_no": tricks_won[0] + tricks_won[1] + 1,
        "bid_history": [[int(p), int(a)] for p, a in env.get_bid_history()],
        "played_action": int(actions[upto]) if upto < len(actions) else None,
    }


def plan(pos):
    """How many worlds each side gets, from the size of the position."""
    cards_left = len(pos["hands"][pos["seat"]])
    oracle = WORLDS_BY_CARDS_LEFT.get(cards_left, 300)
    n_cands = max(1, len(pos["candidates"]))
    real = min(oracle, max(20, REAL_PLAYOUT_BUDGET // n_cands))
    return {"oracle_worlds": oracle, "real_worlds": real}


# ── le vrai monde, et les avis ──

def true_world(dealer, initial_hands, actions, upto, candidates, seat):
    """DD cost of each candidate on the deal as it really was."""
    env, _ = replay_to(dealer, initial_hands, actions, upto)
    result = env.solve_scores()
    scores = {int(c): int(ns) for c, ns in result["scores"]}
    team = seat % 2
    best = max(scores.values()) if team == 0 else min(scores.values())
    return {
        "best_card": int(result["best_card"]),
        "ns": {str(c): scores.get(c) for c in candidates},
        "cost": {str(c): _regret(scores.get(c), best, team) for c in candidates},
    }


def opinions(dealer, initial_hands, actions, upto, seat,
             *, play_model=None, belief_model=None):
    """What DouDou50, l'Oracle and Dédé would play here.

    Only **one** Dédé is built — the one seated where the decision is. IS-DD's
    beliefs, void tracking and world sampler all run from one point of view, so
    asking an instance seated elsewhere would hand it information this seat
    never had.
    """
    env = colver.Env.deal_with_hands(int(dealer), [list(h) for h in initial_hands])
    table = {}
    if play_model:
        table["doudou"] = _agent("doudou", 0, play_model=play_model)
    table["isdd"] = _agent("dede", seat, belief_model=belief_model, time_ms=ISDD_MS)
    live = [a for a in table.values() if a is not None]
    for agent in live:
        agent.init_deal(env)

    for a in actions[:upto]:
        a = int(a)
        for agent in live:
            agent.observe(env, a)
        env.step(a)

    out = {}
    for key, agent in table.items():
        if agent is None:
            continue
        try:
            out[key] = int(agent.decide(env)["action"])
        except Exception as e:  # noqa: BLE001 — sidecar down must not lose the page
            logger.warning("%s failed: %s", key, e)
    try:
        out["oracle"] = int(env.action_oracle_dd())
    except Exception:  # noqa: BLE001
        pass
    return out


def _agent(kind, seat, **kw):
    try:
        return colver.Agent(agents.spec_for(kind, **kw), seat)
    except Exception as e:  # noqa: BLE001 — a missing model must not kill the page
        logger.warning("%s seat %s unavailable: %s", kind, seat, e)
        return None


# ── worlds ──

def sample_worlds(dealer, initial_hands, actions, upto, observer, n_worlds,
                  *, playgen_model=None, temperature=1.0):
    """`(worlds, source)` — each world is `[cards_left_per_seat]` for 4 seats.

    Sidecar first (batched on the GPU, ~50× the CPU path), then the local
    playgen sampler, then a count-respecting shuffle. The shuffle is a genuine
    downgrade — it ignores the coupes the play has already revealed, so some of
    its worlds contradict the observed cards — hence it is named in the result
    and the page says so rather than passing it off as playgen.
    """
    action_ids = [int(a) for a in actions[:upto]]

    if playgen_gpu.enabled():
        pairs = _player_action_pairs(dealer, initial_hands, action_ids)
        worlds = playgen_gpu.play_worlds(
            dealer, initial_hands, pairs, observer, n_worlds, temperature)
        if worlds:
            return worlds, "playgen"

    if playgen_model:
        try:
            analyst = colver.Analyst.replay(
                playgen_model, int(dealer), [list(h) for h in initial_hands],
                action_ids, int(observer))
            env, _ = replay_to(dealer, initial_hands, actions, upto)
            worlds = analyst.play_worlds(env, n_worlds, temperature)
            if worlds:
                return worlds, "playgen"
        except BaseException as e:  # noqa: BLE001 — a Rust panic is a BaseException
            logger.warning("local playgen failed: %s", e)

    return _uniform_worlds(dealer, initial_hands, actions, upto, observer, n_worlds), "uniform"


def _player_action_pairs(dealer, initial_hands, action_ids):
    """`[(player, action)]` — the shape the sidecar wants for a prefix."""
    env = colver.Env.deal_with_hands(int(dealer), [list(h) for h in initial_hands])
    pairs = []
    for a in action_ids:
        pairs.append((int(env.current_player()), int(a)))
        env.step(a)
    return pairs


def _uniform_worlds(dealer, initial_hands, actions, upto, observer, n_worlds):
    """Last-resort worlds: unseen cards shuffled, per-seat counts respected."""
    env, _ = replay_to(dealer, initial_hands, actions, upto)
    hands = [list(h) for h in env.get_hands()]
    own = set(hands[observer])
    hidden = [s for s in range(4) if s != observer]
    pool = [c for s in hidden for c in hands[s]]
    counts = {s: len(hands[s]) for s in hidden}

    worlds = []
    for _ in range(n_worlds):
        deck = list(pool)
        random.shuffle(deck)
        world = [None] * 4
        world[observer] = sorted(own)
        at = 0
        for s in hidden:
            world[s] = sorted(deck[at:at + counts[s]])
            at += counts[s]
        worlds.append(world)
    return worlds


# ── one world's work ──

def world_job(dealer, actions, upto, played, world, candidates, team,
              play_model=None, want_real=False):
    """Solve one world, and optionally play each candidate out in it.

    Runs in a worker thread: both the solver and the DMC forward pass release
    the GIL, so these genuinely overlap. Every parameter is positional —
    `run_in_executor` cannot pass keywords.
    """
    hands = [sorted(list(played[s]) + [int(c) for c in world[s]]) for s in range(4)]
    env = colver.Env.deal_with_hands(int(dealer), hands)
    for a in actions[:upto]:
        env.step(int(a))

    result = env.solve_scores()
    scores = {int(c): int(ns) for c, ns in result["scores"]}
    ns = {c: scores.get(c) for c in candidates}
    known = [v for v in ns.values() if v is not None]
    best = (max(known) if team == 0 else min(known)) if known else None

    out = {"ns": ns, "best": best, "hands": hands}
    if want_real and play_model:
        out["real"] = {c: _playout(dealer, hands, actions, upto, c, play_model)
                       for c in candidates}
    return out


def _playout(dealer, hands, actions, upto, card, play_model):
    """Force `card`, then let DouDou50 finish the deal for all four seats."""
    env = _play_env(play_model, dealer, hands)
    for a in actions[:upto]:
        env.step(int(a))
    if int(card) not in [int(a) for a in env.legal_actions()]:
        return None
    env.step(int(card))
    guard = 0
    while not env.is_terminal() and guard < 40:
        env.step(int(env.action_dmc_with_stats()["best_action"]))
        guard += 1
    if not env.is_terminal():
        return None
    rewards = env.rewards()
    contract = env.get_contract()
    taker = int(contract["team"]) if contract else 0
    return {
        "ns": float(rewards[0]),
        "ew": float(rewards[1]),
        "achieved": bool(rewards[taker] > 0),
        "taker": taker,
    }


def _play_env(play_model, dealer, hands):
    """Thread-local Env with the play model already resident.

    Loading DouDou50 is ~10 MB off disk; doing it per world instead of per
    worker thread would dominate the whole computation.
    """
    env = getattr(_tls, "env", None)
    if env is not None and getattr(_tls, "model", None) == play_model:
        env.redeal_with_hands(int(dealer), hands)
        return env
    env = colver.Env.deal_with_hands(int(dealer), hands)
    env.load_dmc_model(play_model)
    _tls.env = env
    _tls.model = play_model
    return env


# ── accumulation ──

def new_totals(candidates):
    return {str(c): {"ns": [], "best": 0, "risk": 0, "n": 0,
                     "real_ns": 0.0, "real_ew": 0.0, "real_ok": 0,
                     "real_win": 0, "real_n": 0}
            for c in candidates}


def accumulate(totals, job, team):
    """Fold one world's result into the running per-candidate totals."""
    best = job.get("best")
    for card, value in job["ns"].items():
        acc = totals[str(card)]
        if value is None:
            continue
        acc["ns"].append(value)
        acc["n"] += 1
        if best is not None:
            if value == best:
                acc["best"] += 1
            if _regret(value, best, team) >= RISK_THRESHOLD:
                acc["risk"] += 1
    for card, real in (job.get("real") or {}).items():
        if real is None:
            continue
        acc = totals[str(card)]
        acc["real_ns"] += real["ns"]
        acc["real_ew"] += real["ew"]
        acc["real_ok"] += 1 if real["achieved"] else 0
        # « Contrat réussi » is the *taker's* outcome, so on a defending seat a
        # high rate means the worst card. `real_win` is the same event read from
        # the acting team's side, so the column ranks the same way on every row
        # whichever side of the contract this seat is on.
        favourable = real["achieved"] if team == real["taker"] else not real["achieved"]
        acc["real_win"] += 1 if favourable else 0
        acc["real_n"] += 1


def summarize(totals, team):
    """Per-candidate rows, ready to send: no lists, only what the table draws.

    The two sides are **not on the same scale** and must not be subtracted from
    each other: `mean_ns` / `median_ns` are double-dummy *card* points (0-252
    with capot and dix de der), while the played-out side reports the *scored*
    deal (contract included, so a made 160 lands past 320). The real side is
    therefore reported as a signed Nord-Sud − Est-Ouest differential, which
    cannot be misread as a card-point total.
    """
    rows = {}
    for card, acc in totals.items():
        n = acc["n"]
        values = sorted(acc["ns"])
        rn = acc["real_n"]
        rows[card] = {
            "n": n,
            "mean_ns": round(sum(values) / n, 1) if n else None,
            "median_ns": _median(values),
            "best_pct": round(acc["best"] / n * 100) if n else None,
            "risk_pct": round(acc["risk"] / n * 100) if n else None,
            "real_n": rn,
            "real_diff": round((acc["real_ns"] - acc["real_ew"]) / rn, 1) if rn else None,
            "real_ok": acc["real_ok"],
            "real_win": acc["real_win"],
        }
    # Regret is oriented towards the acting team, so the sign reads the same
    # way in both directions: 0 is best, positive is worse.
    means = [r["mean_ns"] for r in rows.values() if r["mean_ns"] is not None]
    if means:
        top = max(means) if team == 0 else min(means)
        for row in rows.values():
            row["regret"] = (round(_regret(row["mean_ns"], top, team), 1)
                             if row["mean_ns"] is not None else None)
    return rows


def _regret(value, best, team):
    """How much worse than `best` this value is, for the acting team."""
    if value is None or best is None:
        return None
    return (best - value) if team == 0 else (value - best)


def _median(sorted_values):
    n = len(sorted_values)
    if not n:
        return None
    mid = n // 2
    if n % 2:
        return float(sorted_values[mid])
    return round((sorted_values[mid - 1] + sorted_values[mid]) / 2, 1)
