"""Post-game oracle analysis: DD cost of every card played + bid review.

For each play-phase action of a stored game, the position is re-solved with
the double-dummy solver (all hands known) and the played card is compared to
the DD-optimal one. The cost is expressed in real card points (0-162 scale,
252 with capot) from the acting team's perspective.

Bid moves get two extra signals: the bid NN's preferred action from the same
position (model annonce), and the DD-best contract each team could have
declared on the deal (oracle annonce, one solve per trump suit).

Results are cached in the `analysis` table; computation runs in a thread
(pure Rust solver, a few seconds per game).
"""

import asyncio
import json

import colver
import colver.web.database as db

ANALYSIS_VERSION = 3

# Cost thresholds (card points) -> category label
CATEGORIES = [
    (0, "parfait"),
    (4, "bon"),
    (14, "imprecision"),
    (29, "erreur"),
    (10_000, "faute"),
]

_locks = {}  # game_id -> asyncio.Lock, to avoid duplicate computations


def _categorize(cost):
    for threshold, label in CATEGORIES:
        if cost <= threshold:
            return label
    return "faute"


def _best_contract(suits, team):
    """DD-best declarable contract for a team from per-suit DD points.

    Returns {suit, pts, value} — value 0 when no contract is makeable (< 80),
    250 (capot) when the team takes every point (162).
    """
    pts = [int(s[team]) for s in suits]
    best_suit = max(range(4), key=lambda s: pts[s])
    p = pts[best_suit]
    if p >= 162:
        value = 250
    else:
        value = min(160, p // 10 * 10) if p >= 80 else 0
    return {"suit": best_suit, "pts": p, "value": value}


def _analyze_sync(game, bid_model_path=None):
    """Replay the stored actions, solving each play decision. CPU-bound."""
    env = colver.Env.deal_with_hands(game["dealer"], game["hands"])

    # Oracle annonces: DD solve of the full deal, one solve per trump suit
    oracle_bids = None
    try:
        dd = env.solve_all_suits()
        suits = [[int(ns), int(ew)] for ns, ew in dd["suits"]]
        oracle_bids = {
            "suits": suits,
            "best": [_best_contract(suits, 0), _best_contract(suits, 1)],
        }
    except Exception:
        pass

    if bid_model_path:
        try:
            env.load_bid_model(bid_model_path)
        except Exception:
            bid_model_path = None

    moves = []
    bids = []
    for idx, entry in enumerate(game["actions"]):
        if env.is_terminal():
            break
        phase = int(env.phase())
        player = int(env.current_player())
        action = int(entry["action"])
        if phase == 0 and bid_model_path:
            try:
                result = env.action_bid_nn()
                q = {int(a): float(v) for a, v in result["q_values"]}
                best = int(result["best_action"])
                bids.append({
                    "idx": idx, "player": player, "action": action,
                    "model_best": best,
                    "q_best": round(q.get(best, 0.0), 3),
                    "q_played": round(q[action], 3) if action in q else None,
                })
            except Exception:
                pass
        if phase == 1:
            legals = list(env.legal_actions())
            if action not in legals:
                break  # corrupt record — stop rather than emit nonsense
            if len(legals) == 1:
                moves.append({
                    "idx": idx, "player": player, "action": action,
                    "best": action, "cost": 0, "forced": True,
                })
            else:
                result = env.solve_scores()
                scores = {c: ns for c, ns in result["scores"]}
                team = player % 2
                played_ns = scores[action]
                best_ns = max(scores.values()) if team == 0 else min(scores.values())
                cost = (best_ns - played_ns) if team == 0 else (played_ns - best_ns)
                moves.append({
                    "idx": idx, "player": player, "action": action,
                    "best": int(result["best_card"]),
                    "cost": int(cost),
                    "category": _categorize(cost),
                })
        env.step(action)

    summary = _summarize(moves)
    return {
        "version": ANALYSIS_VERSION,
        "moves": moves,
        "bids": bids,
        "oracle_bids": oracle_bids,
        "summary": summary,
    }


def _summarize(moves):
    players = []
    for p in range(4):
        pm = [m for m in moves if m["player"] == p]
        decisions = [m for m in pm if not m.get("forced")]
        total_cost = sum(m["cost"] for m in decisions)
        counts = {label: 0 for _, label in CATEGORIES}
        for m in decisions:
            counts[m["category"]] += 1
        players.append({
            "player": p,
            "moves": len(pm),
            "forced": len(pm) - len(decisions),
            "decisions": len(decisions),
            "total_cost": total_cost,
            "avg_cost": round(total_cost / len(decisions), 1) if decisions else 0.0,
            "counts": counts,
        })
    return {"players": players}


async def get_or_compute(game_id, bid_model_path=None):
    """Return the cached analysis for a game, computing it on first request."""
    cached = await db.get_analysis(game_id)
    if cached is not None and cached.get("version") == ANALYSIS_VERSION:
        return cached, None

    game = await db.get_game(game_id)
    if game is None:
        return None, "Partie introuvable"
    if not game["actions"]:
        return None, "Aucune action à analyser"

    lock = _locks.setdefault(game_id, asyncio.Lock())
    async with lock:
        # Another request may have computed it while we waited on the lock
        cached = await db.get_analysis(game_id)
        if cached is not None and cached.get("version") == ANALYSIS_VERSION:
            return cached, None
        analysis = await asyncio.to_thread(_analyze_sync, game, bid_model_path)
        await db.save_analysis(game_id, json.dumps(analysis))
    _locks.pop(game_id, None)
    return analysis, None
