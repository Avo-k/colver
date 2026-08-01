"""Elo ratings for users and bot types.

Every rated entity — a user account or a bot type ("dede", "doudou", …) —
carries one Elo. A deal is rated when all four seats are identifiable
(bots, or humans with an account): team rating = mean of the two partners,
classic Elo expectation, score 1/0/0.5 from the deal's final rewards
(contract-aware, obtained by replaying the stored actions). Void deals
(four passes) are not rated.

Bots use a lower K than humans: they are reference points and occupy up to
three seats per game, so their ratings move fast enough even with a small K.
The same bot type may sit on both teams; its deltas are summed.

`rate_game` is idempotent (elo_history has one row per game × entity), which
makes the startup backfill safe to run on every boot.
"""

import asyncio
import logging

import colver
import colver.web.database as db

logger = logging.getLogger(__name__)

START_ELO = 1000.0
K_USER = 32.0
K_BOT = 8.0

_lock = asyncio.Lock()  # serialize read-modify-write across concurrent games


def _seat_entities(game, player_rows):
    """Map each seat to a rated entity (kind, ref), or None if unratable."""
    agents = game["agents"]
    humans = {row["seat"]: row["user_id"] for row in player_rows}
    if game["mode"] == "play" and game["human_seat"] is not None:
        if game.get("user_id") is None:
            return None  # anonymous solo game
        humans[game["human_seat"]] = game["user_id"]

    seats = []
    for s in range(4):
        if s in humans:
            seats.append(("user", str(humans[s])))
        else:
            bot = agents.get(str(s))
            if not bot or bot == "human":
                return None
            seats.append(("bot", bot))
    return seats


def _replay_rewards(game):
    """Final rewards of a stored game, or None if unusable for rating."""
    try:
        env = colver.Env.deal_with_hands(game["dealer"], game["hands"])
        for entry in game["actions"]:
            if env.is_terminal():
                break
            env.step(int(entry["action"]))
        if not env.is_terminal():
            return None
        if not env.get_contract().get("value"):
            return None  # void deal (four passes)
        return list(env.rewards())
    except Exception:
        return None


async def rate_game(game_id):
    """Rate one completed game. Idempotent; never raises."""
    try:
        async with _lock:
            return await _rate_game_locked(game_id)
    except Exception:
        logger.exception("rating of %s failed", game_id)
        return False


async def _rate_game_locked(game_id):
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT 1 FROM elo_history WHERE game_id = ? LIMIT 1", (game_id,))
    if rows:
        return False  # already rated

    game = await db.get_game(game_id)
    if game is None or not game["is_complete"] or game["mode"] not in ("play", "multi"):
        return False
    player_rows = await conn.execute_fetchall(
        "SELECT seat, user_id FROM game_players WHERE game_id = ?", (game_id,))
    seats = _seat_entities(game, [dict(r) for r in player_rows])
    if seats is None:
        return False
    rewards = _replay_rewards(game)
    if rewards is None:
        return False

    score_ns = 1.0 if rewards[0] > rewards[1] else 0.0 if rewards[0] < rewards[1] else 0.5

    # Current ratings
    ratings = {}
    for ent in set(seats):
        r = await conn.execute_fetchall(
            "SELECT elo, games FROM elo_ratings WHERE kind = ? AND ref = ?", ent)
        ratings[ent] = (r[0][0], r[0][1]) if r else (START_ELO, 0)

    team_elo = [
        (ratings[seats[0]][0] + ratings[seats[2]][0]) / 2,
        (ratings[seats[1]][0] + ratings[seats[3]][0]) / 2,
    ]
    expected_ns = 1.0 / (1.0 + 10 ** ((team_elo[1] - team_elo[0]) / 400))

    # Per-entity aggregated deltas (a bot type can occupy several seats)
    deltas = {}
    counts = {}
    for seat, ent in enumerate(seats):
        team = seat % 2
        s = score_ns if team == 0 else 1.0 - score_ns
        e = expected_ns if team == 0 else 1.0 - expected_ns
        k = K_USER if ent[0] == "user" else K_BOT
        deltas[ent] = deltas.get(ent, 0.0) + k * (s - e)
        counts[ent] = counts.get(ent, 0) + 1

    now = db._now()
    for ent, delta in deltas.items():
        new_elo = ratings[ent][0] + delta
        await conn.execute(
            "INSERT INTO elo_ratings (kind, ref, elo, games, updated_at) "
            "VALUES (?, ?, ?, ?, ?) "
            "ON CONFLICT(kind, ref) DO UPDATE SET elo = ?, games = games + ?, updated_at = ?",
            (*ent, new_elo, counts[ent], now, new_elo, counts[ent], now),
        )
        await conn.execute(
            "INSERT INTO elo_history (game_id, kind, ref, delta, elo_after) "
            "VALUES (?, ?, ?, ?, ?)",
            (game_id, *ent, round(delta, 2), round(new_elo, 2)),
        )
    await conn.commit()
    return True


async def backfill():
    """Rate every completed game not yet in elo_history, oldest first."""
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT id FROM games WHERE is_complete = 1 AND mode IN ('play', 'multi') "
        "AND id NOT IN (SELECT DISTINCT game_id FROM elo_history) "
        "ORDER BY created_at",
    )
    rated = 0
    for (game_id,) in rows:
        if await rate_game(game_id):
            rated += 1
    if rated:
        logger.info("backfill: rated %d game(s)", rated)


async def get_rating(kind, ref):
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT elo, games FROM elo_ratings WHERE kind = ? AND ref = ?",
        (kind, str(ref)))
    if not rows:
        return {"elo": START_ELO, "games": 0}
    return {"elo": round(rows[0][0], 1), "games": rows[0][1]}


async def leaderboard():
    """All rated entities, best first, with display names for users."""
    conn = await db.get_db()
    rows = await conn.execute_fetchall(
        "SELECT r.kind, r.ref, r.elo, r.games, u.username "
        "FROM elo_ratings r "
        "LEFT JOIN users u ON r.kind = 'user' AND u.id = CAST(r.ref AS INTEGER) "
        "ORDER BY r.elo DESC",
    )
    return [
        {
            "kind": row[0],
            "ref": row[1],
            "elo": round(row[2], 1),
            "games": row[3],
            "name": row[4] if row[0] == "user" else row[1],
        }
        for row in rows
    ]
