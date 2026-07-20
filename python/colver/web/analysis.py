"""Post-game oracle analysis: DD cost of every card played.

For each play-phase action of a stored game, the position is re-solved with
the double-dummy solver (all hands known) and the played card is compared to
the DD-optimal one. The cost is expressed in real card points (0-162 scale,
252 with capot) from the acting team's perspective.

Results are cached in the `analysis` table; computation runs in a thread
(pure Rust solver, a few seconds per game).
"""

import asyncio
import json

import colver
import colver.web.database as db

ANALYSIS_VERSION = 2

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


def _analyze_sync(game):
    """Replay the stored actions, solving each play decision. CPU-bound."""
    env = colver.Env.deal_with_hands(game["dealer"], game["hands"])
    moves = []
    for idx, entry in enumerate(game["actions"]):
        if env.is_terminal():
            break
        phase = int(env.phase())
        player = int(env.current_player())
        action = int(entry["action"])
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
    return {"version": ANALYSIS_VERSION, "moves": moves, "summary": summary}


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


async def get_or_compute(game_id):
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
        analysis = await asyncio.to_thread(_analyze_sync, game)
        await db.save_analysis(game_id, json.dumps(analysis))
    _locks.pop(game_id, None)
    return analysis, None
