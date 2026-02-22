"""SQLite database for persisting game history and bug reports."""

import json
import os
import random
import string
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

_DEFAULT_DB_DIR = Path.home() / ".local" / "share" / "colver"
DB_PATH = os.environ.get(
    "COLVER_DB_PATH",
    str(_DEFAULT_DB_DIR / "colver.db"),
)

_db = None

SCHEMA = """
CREATE TABLE IF NOT EXISTS games (
    id          TEXT PRIMARY KEY,
    mode        TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    dealer      INTEGER NOT NULL,
    hands       TEXT NOT NULL,
    agents      TEXT NOT NULL,
    human_seat  INTEGER,
    actions     TEXT NOT NULL DEFAULT '[]',
    is_complete INTEGER NOT NULL DEFAULT 0,
    points_ns   INTEGER,
    points_ew   INTEGER,
    contract    TEXT
);

CREATE TABLE IF NOT EXISTS bug_reports (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    game_id     TEXT NOT NULL REFERENCES games(id),
    action_idx  INTEGER,
    message     TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    user_agent  TEXT
);
"""


async def get_db():
    global _db
    if _db is not None:
        return _db
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _db = await aiosqlite.connect(DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.commit()
    # executescript commits implicitly, don't call commit() after
    await _db.executescript(SCHEMA)
    print(f"[database] Connected to {DB_PATH}")
    return _db


def _gen_id():
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(4))


async def create_game(mode, dealer, hands, agents, human_seat=None):
    db = await get_db()
    for _ in range(20):
        game_id = _gen_id()
        try:
            await db.execute(
                "INSERT INTO games (id, mode, created_at, dealer, hands, agents, human_seat) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    game_id,
                    mode,
                    datetime.now(timezone.utc).isoformat(),
                    dealer,
                    json.dumps(hands),
                    json.dumps(agents),
                    human_seat,
                ),
            )
            await db.commit()
            return game_id
        except aiosqlite.IntegrityError:
            continue
    raise RuntimeError("Failed to generate unique game ID")


async def append_action(game_id, action_entry):
    """Append a single action to the game's action list."""
    db = await get_db()
    row = await db.execute_fetchall(
        "SELECT actions FROM games WHERE id = ?", (game_id,)
    )
    if not row:
        return
    actions = json.loads(row[0][0])
    actions.append(action_entry)
    await db.execute(
        "UPDATE games SET actions = ? WHERE id = ?",
        (json.dumps(actions), game_id),
    )
    await db.commit()


async def complete_game(game_id, points_ns, points_ew, contract):
    db = await get_db()
    await db.execute(
        "UPDATE games SET is_complete = 1, points_ns = ?, points_ew = ?, contract = ? "
        "WHERE id = ?",
        (points_ns, points_ew, json.dumps(contract) if contract else None, game_id),
    )
    await db.commit()


async def get_game(game_id):
    db = await get_db()
    rows = await db.execute_fetchall("SELECT * FROM games WHERE id = ?", (game_id,))
    if not rows:
        return None
    row = rows[0]
    return _row_to_dict(row)


async def list_games(limit=50, offset=0):
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT id, mode, created_at, dealer, agents, human_seat, is_complete, "
        "points_ns, points_ew, contract FROM games "
        "WHERE mode = 'play' AND is_complete = 1 "
        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (limit, offset),
    )
    result = []
    for row in rows:
        d = {
            "id": row[0],
            "mode": row[1],
            "created_at": row[2],
            "dealer": row[3],
            "agents": json.loads(row[4]),
            "human_seat": row[5],
            "is_complete": bool(row[6]),
            "points_ns": row[7],
            "points_ew": row[8],
            "contract": json.loads(row[9]) if row[9] else None,
        }
        result.append(d)
    return result


async def create_bug_report(game_id, action_idx, message, user_agent=None):
    db = await get_db()
    await db.execute(
        "INSERT INTO bug_reports (game_id, action_idx, message, created_at, user_agent) "
        "VALUES (?, ?, ?, ?, ?)",
        (game_id, action_idx, message, datetime.now(timezone.utc).isoformat(), user_agent),
    )
    await db.commit()


def _row_to_dict(row):
    return {
        "id": row[0],
        "mode": row[1],
        "created_at": row[2],
        "dealer": row[3],
        "hands": json.loads(row[4]),
        "agents": json.loads(row[5]),
        "human_seat": row[6],
        "actions": json.loads(row[7]),
        "is_complete": bool(row[8]),
        "points_ns": row[9],
        "points_ew": row[10],
        "contract": json.loads(row[11]) if row[11] else None,
    }
