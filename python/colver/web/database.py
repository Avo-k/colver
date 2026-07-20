"""SQLite database for persisting users, game history and bug reports.

Schema changes go through MIGRATIONS: each entry bumps PRAGMA user_version.
Migration 1 is idempotent (IF NOT EXISTS) so pre-migration prod databases
(user_version=0 with existing tables) adopt the system cleanly.
"""

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

MIGRATIONS = [
    # v1 — base tables (matches the historical implicit schema)
    """
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
    """,
    # v2 — user accounts and sessions
    """
    CREATE TABLE users (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        username      TEXT NOT NULL UNIQUE COLLATE NOCASE,
        password_hash TEXT NOT NULL,
        created_at    TEXT NOT NULL
    );

    CREATE TABLE sessions (
        token_hash  TEXT PRIMARY KEY,
        user_id     INTEGER NOT NULL REFERENCES users(id),
        created_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL
    );

    ALTER TABLE games ADD COLUMN user_id INTEGER REFERENCES users(id);
    CREATE INDEX idx_games_user ON games(user_id, created_at);
    """,
]


async def _migrate(db):
    cur = await db.execute("PRAGMA user_version")
    version = (await cur.fetchone())[0]
    for i, script in enumerate(MIGRATIONS, start=1):
        if i <= version:
            continue
        await db.executescript(script)
        await db.execute(f"PRAGMA user_version = {i}")
        await db.commit()
        print(f"[database] Applied migration v{i}")


async def get_db():
    global _db
    if _db is not None:
        return _db
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    _db = await aiosqlite.connect(DB_PATH)
    _db.row_factory = aiosqlite.Row
    await _db.execute("PRAGMA journal_mode=WAL")
    await _db.commit()
    await _migrate(_db)
    print(f"[database] Connected to {DB_PATH}")
    return _db


def _now():
    return datetime.now(timezone.utc).isoformat()


def _gen_id():
    chars = string.ascii_lowercase + string.digits
    return "".join(random.choice(chars) for _ in range(4))


# ===== Users & sessions =====

async def create_user(username, password_hash):
    db = await get_db()
    try:
        cur = await db.execute(
            "INSERT INTO users (username, password_hash, created_at) VALUES (?, ?, ?)",
            (username, password_hash, _now()),
        )
        await db.commit()
        return cur.lastrowid
    except aiosqlite.IntegrityError:
        return None


async def get_user_by_username(username):
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM users WHERE username = ?", (username,)
    )
    return dict(rows[0]) if rows else None


async def get_user_by_id(user_id):
    db = await get_db()
    rows = await db.execute_fetchall("SELECT * FROM users WHERE id = ?", (user_id,))
    return dict(rows[0]) if rows else None


async def create_session(token_hash, user_id, expires_at):
    db = await get_db()
    await db.execute(
        "INSERT INTO sessions (token_hash, user_id, created_at, expires_at) "
        "VALUES (?, ?, ?, ?)",
        (token_hash, user_id, _now(), expires_at),
    )
    # Opportunistic cleanup of expired sessions
    await db.execute("DELETE FROM sessions WHERE expires_at < ?", (_now(),))
    await db.commit()


async def get_session_user(token_hash):
    """Return the user dict for a valid (non-expired) session token, else None."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT u.* FROM sessions s JOIN users u ON u.id = s.user_id "
        "WHERE s.token_hash = ? AND s.expires_at >= ?",
        (token_hash, _now()),
    )
    return dict(rows[0]) if rows else None


async def delete_session(token_hash):
    db = await get_db()
    await db.execute("DELETE FROM sessions WHERE token_hash = ?", (token_hash,))
    await db.commit()


# ===== Games =====

async def create_game(mode, dealer, hands, agents, human_seat=None, user_id=None):
    db = await get_db()
    for _ in range(20):
        game_id = _gen_id()
        try:
            await db.execute(
                "INSERT INTO games (id, mode, created_at, dealer, hands, agents, human_seat, user_id) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    game_id,
                    mode,
                    _now(),
                    dealer,
                    json.dumps(hands),
                    json.dumps(agents),
                    human_seat,
                    user_id,
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
    return _row_to_dict(rows[0])


async def list_games(limit=50, offset=0, user_id=None):
    db = await get_db()
    where = "WHERE mode = 'play' AND is_complete = 1"
    params = []
    if user_id is not None:
        where += " AND user_id = ?"
        params.append(user_id)
    rows = await db.execute_fetchall(
        "SELECT id, mode, created_at, dealer, agents, human_seat, is_complete, "
        f"points_ns, points_ew, contract FROM games {where} "
        "ORDER BY created_at DESC LIMIT ? OFFSET ?",
        (*params, limit, offset),
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


async def user_game_stats(user_id):
    """Aggregate win/loss stats for a user's completed play games.

    The user sits at human_seat: even seats are team NS, odd seats team EW.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        """
        SELECT
            COUNT(*) AS games,
            SUM(CASE WHEN (human_seat % 2 = 0 AND points_ns > points_ew)
                       OR (human_seat % 2 = 1 AND points_ew > points_ns)
                THEN 1 ELSE 0 END) AS wins
        FROM games
        WHERE mode = 'play' AND is_complete = 1 AND user_id = ?
              AND human_seat IS NOT NULL
        """,
        (user_id,),
    )
    row = rows[0]
    return {"games": row[0] or 0, "wins": row[1] or 0}


async def create_bug_report(game_id, action_idx, message, user_agent=None):
    db = await get_db()
    await db.execute(
        "INSERT INTO bug_reports (game_id, action_idx, message, created_at, user_agent) "
        "VALUES (?, ?, ?, ?, ?)",
        (game_id, action_idx, message, _now(), user_agent),
    )
    await db.commit()


def _row_to_dict(row):
    d = dict(row)
    d["hands"] = json.loads(d["hands"])
    d["agents"] = json.loads(d["agents"])
    d["actions"] = json.loads(d["actions"])
    d["is_complete"] = bool(d["is_complete"])
    d["contract"] = json.loads(d["contract"]) if d["contract"] else None
    return d
