"""SQLite database for persisting users, game history and bug reports.

Schema changes go through MIGRATIONS: each entry bumps PRAGMA user_version.
Migration 1 is idempotent (IF NOT EXISTS) so pre-migration prod databases
(user_version=0 with existing tables) adopt the system cleanly.
"""

import asyncio
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
# La connexion n'est publiée qu'une fois migrée, et une seule tâche migre :
# au démarrage, le backfill Elo et la première partie ouvrent la base en même
# temps, et le second voyait sinon une base sans tables (`_db` était affecté
# avant `_migrate`).
_db_lock = asyncio.Lock()

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
    # v3 — multiplayer: per-seat participants of a game
    """
    CREATE TABLE game_players (
        game_id  TEXT NOT NULL REFERENCES games(id),
        seat     INTEGER NOT NULL,
        user_id  INTEGER NOT NULL REFERENCES users(id),
        PRIMARY KEY (game_id, seat)
    );
    CREATE INDEX idx_game_players_user ON game_players(user_id);
    """,
    # v4 — cached post-game oracle analysis
    """
    CREATE TABLE analysis (
        game_id     TEXT PRIMARY KEY REFERENCES games(id),
        created_at  TEXT NOT NULL,
        data        TEXT NOT NULL
    );
    """,
    # v5 — Elo ratings for users AND bot types
    """
    CREATE TABLE elo_ratings (
        kind        TEXT NOT NULL,
        ref         TEXT NOT NULL,
        elo         REAL NOT NULL,
        games       INTEGER NOT NULL DEFAULT 0,
        updated_at  TEXT NOT NULL,
        PRIMARY KEY (kind, ref)
    );

    CREATE TABLE elo_history (
        game_id    TEXT NOT NULL REFERENCES games(id),
        kind       TEXT NOT NULL,
        ref        TEXT NOT NULL,
        delta      REAL NOT NULL,
        elo_after  REAL NOT NULL,
        PRIMARY KEY (game_id, kind, ref)
    );
    """,
    # v6 — cached per-card "what would the reference bots have played"
    # Kept out of `analysis`: that one is DD solves only, this one also runs a
    # full IS-DD search per card. Separate rows so a slow review never
    # invalidates (or blocks) the fast oracle annotations.
    """
    CREATE TABLE agent_review (
        game_id     TEXT PRIMARY KEY REFERENCES games(id),
        created_at  TEXT NOT NULL,
        data        TEXT NOT NULL
    );
    """,
    # v7 — parties en plusieurs donnes (1000 / 2000 points)
    # Une donne reste une ligne de `games` : c'est elle qui porte les actions,
    # l'analyse et le partage. `matches` ne fait que les regrouper, avec le
    # score cumulé et le vainqueur.
    """
    CREATE TABLE matches (
        id          TEXT PRIMARY KEY,
        mode        TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        target      INTEGER NOT NULL,
        user_id     INTEGER REFERENCES users(id),
        points_ns   INTEGER NOT NULL DEFAULT 0,
        points_ew   INTEGER NOT NULL DEFAULT 0,
        deals       INTEGER NOT NULL DEFAULT 0,
        is_complete INTEGER NOT NULL DEFAULT 0,
        winner      INTEGER
    );

    ALTER TABLE games ADD COLUMN match_id TEXT REFERENCES matches(id);
    ALTER TABLE games ADD COLUMN deal_no INTEGER;
    CREATE INDEX idx_games_match ON games(match_id, deal_no);
    CREATE INDEX idx_matches_user ON matches(user_id, created_at);
    """,
    # v8 — reprendre une partie interrompue
    # Le rythme est un réglage de *partie*, pas de donne : sans lui, reprendre
    # une partie « rapide » la rendrait à Dédé. `abandoned` sépare la partie
    # concédée de la partie jouée jusqu'au bout — les deux sont `is_complete`,
    # seule la seconde a un vainqueur.
    """
    ALTER TABLE matches ADD COLUMN pacing TEXT;
    ALTER TABLE matches ADD COLUMN human_seat INTEGER;
    ALTER TABLE matches ADD COLUMN abandoned INTEGER NOT NULL DEFAULT 0;
    CREATE INDEX idx_matches_open ON matches(user_id, is_complete);
    """,
    # v9 — purge des caches d'analyse calculés sur des donnes non terminées.
    # `/api/games/{id}/analysis` et `/agents` acceptaient les donnes en cours ;
    # or `get_or_compute` sert le cache *avant* de relire la donne : une
    # analyse partielle déjà écrite serait servie pour toujours (une donne
    # abandonnée reste `is_complete = 0`). `get_game` filtre désormais, mais
    # les lignes déjà en base doivent partir.
    """
    DELETE FROM analysis WHERE game_id IN
        (SELECT id FROM games WHERE is_complete = 0);
    DELETE FROM agent_review WHERE game_id IN
        (SELECT id FROM games WHERE is_complete = 0);
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
    async with _db_lock:
        if _db is not None:
            return _db
        os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
        conn = await aiosqlite.connect(DB_PATH)
        conn.row_factory = aiosqlite.Row
        await conn.execute("PRAGMA journal_mode=WAL")
        await conn.commit()
        await _migrate(conn)
        _db = conn
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

async def create_game(mode, dealer, hands, agents, human_seat=None, user_id=None,
                      match_id=None, deal_no=None):
    db = await get_db()
    for _ in range(20):
        game_id = _gen_id()
        try:
            await db.execute(
                "INSERT INTO games (id, mode, created_at, dealer, hands, agents, human_seat, user_id, "
                "match_id, deal_no) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    game_id,
                    mode,
                    _now(),
                    dealer,
                    json.dumps(hands),
                    json.dumps(agents),
                    human_seat,
                    user_id,
                    match_id,
                    deal_no,
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


# ===== Parties (plusieurs donnes) =====

async def create_match(mode, target, user_id=None, pacing=None, human_seat=None):
    """Ouvrir une partie. Les donnes s'y rattachent par `games.match_id`.

    `pacing` et `human_seat` sont les réglages à rejouer si la partie est
    reprise plus tard : ils valent pour toutes ses donnes.
    """
    db = await get_db()
    for _ in range(20):
        match_id = _gen_id()
        try:
            await db.execute(
                "INSERT INTO matches (id, mode, created_at, target, user_id, "
                "pacing, human_seat) VALUES (?, ?, ?, ?, ?, ?, ?)",
                (match_id, mode, _now(), int(target), user_id, pacing, human_seat),
            )
            await db.commit()
            return match_id
        except aiosqlite.IntegrityError:
            continue
    raise RuntimeError("Failed to generate unique match ID")


async def update_match(match_id, points_ns, points_ew, deals, is_complete, winner=None):
    db = await get_db()
    await db.execute(
        "UPDATE matches SET points_ns = ?, points_ew = ?, deals = ?, "
        "is_complete = ?, winner = ? WHERE id = ?",
        (int(points_ns), int(points_ew), int(deals),
         1 if is_complete else 0, winner, match_id),
    )
    await db.commit()


async def list_open_matches(user_id, limit=8):
    """Les parties solo d'un joueur qui n'ont pas été jouées jusqu'au bout.

    `pending` dit qu'une donne était en cours au moment de la coupure : elle
    n'est pas reprenable (les bots n'ont pas d'état persistant), donc la
    reprendre l'abandonne. C'est la seule information dont l'affichage a besoin
    pour poser honnêtement le choix.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT m.id, m.target, m.created_at, m.points_ns, m.points_ew, m.deals, "
        "m.pacing, m.human_seat, "
        "EXISTS (SELECT 1 FROM games g WHERE g.match_id = m.id "
        "        AND g.is_complete = 0) AS pending "
        "FROM matches m "
        "WHERE m.user_id = ? AND m.mode = 'play' AND m.is_complete = 0 "
        "      AND m.target > 0 "
        "ORDER BY m.created_at DESC LIMIT ?",
        (user_id, limit),
    )
    return [
        {
            "id": r[0], "target": r[1], "created_at": r[2],
            "points_ns": r[3], "points_ew": r[4], "deals": r[5],
            "pacing": r[6], "human_seat": r[7], "pending": bool(r[8]),
        }
        for r in rows
    ]


async def open_match_summary(match_id, user_id):
    """Une partie en cours appartenant à ce joueur, ou None. Sert au lien de
    reprise que Rejouer affiche sous une donne de partie."""
    matches = await list_open_matches(user_id, limit=50)
    for m in matches:
        if m["id"] == match_id:
            return m
    return None


async def load_open_match(match_id, user_id):
    """Tout ce qu'il faut pour reconstruire une partie interrompue.

    `deals` ne porte que les donnes **terminées** : leur identifiant et leur
    donneur. Pas leur score — `games.points_ns/ew` sont les points *cartes* de
    la donne (`env.get_points()`), pas les points *marqués* (`env.rewards()`,
    contrat compris) qui font le score de la partie. Ces derniers ne sont
    cumulés que dans `matches.points_ns/ew`, seule source du score repris.

    Une donne interrompue reste dans `games` (invisible partout ailleurs : tous
    les listings filtrent `is_complete = 1`), on n'en garde ici que le donneur,
    pour que la donne rejouée soit redonnée par le même joueur.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM matches WHERE id = ? AND user_id = ? AND is_complete = 0",
        (match_id, user_id),
    )
    if not rows:
        return None
    match = dict(rows[0])
    games = await db.execute_fetchall(
        "SELECT id, dealer, is_complete FROM games "
        "WHERE match_id = ? ORDER BY deal_no",
        (match_id,),
    )
    match["deals"] = [{"game_id": g[0], "dealer": g[1]} for g in games if g[2]]
    pending = [g for g in games if not g[2]]
    match["pending_dealer"] = pending[-1][1] if pending else None
    return match


async def abandon_match(match_id, user_id):
    """Concéder une partie : close, mais sans vainqueur.

    Une partie jouée jusqu'au bout a toujours un `winner` (`Match.finished`
    exige un écart), donc `is_complete = 1` + `winner IS NULL` ne peut désigner
    qu'un abandon ; `abandoned` le dit quand même explicitement.
    """
    db = await get_db()
    cur = await db.execute(
        "UPDATE matches SET is_complete = 1, abandoned = 1, winner = NULL "
        "WHERE id = ? AND user_id = ? AND is_complete = 0",
        (match_id, user_id),
    )
    await db.commit()
    return cur.rowcount > 0


async def get_match(match_id):
    """Une partie et la liste ordonnée de ses donnes."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM matches WHERE id = ?", (match_id,))
    if not rows:
        return None
    match = dict(rows[0])
    match["is_complete"] = bool(match["is_complete"])
    deals = await db.execute_fetchall(
        "SELECT id, deal_no, dealer, points_ns, points_ew, contract, is_complete "
        "FROM games WHERE match_id = ? ORDER BY deal_no",
        (match_id,),
    )
    match["games"] = [
        {
            "id": d[0],
            "deal_no": d[1],
            "dealer": d[2],
            "points_ns": d[3],
            "points_ew": d[4],
            "contract": json.loads(d[5]) if d[5] else None,
            "is_complete": bool(d[6]),
        }
        for d in deals
    ]
    return match


async def get_game(game_id, include_incomplete=False):
    """Une donne enregistrée — par défaut, seulement si elle est terminée.

    Une ligne `games` porte les quatre mains en clair (`hands`), et son
    identifiant est public : le salon le diffuse à tous dans chaque
    `room_game_state`, et quatre caractères s'énumèrent de toute façon.
    Servir une donne en cours reviendrait donc à montrer le jeu des
    adversaires — même politique que `list_games`, qui filtre déjà
    `is_complete = 1`. Une donne abandonnée reste `is_complete = 0` pour
    toujours : elle n'est jamais rendue. Les appelants qui ont réellement
    besoin d'une donne en cours (rapport de bug, donne personnalisée à
    regarder) le disent avec `include_incomplete=True` — à charge pour eux
    de n'en rien divulguer.
    """
    db = await get_db()
    where = "" if include_incomplete else " AND is_complete = 1"
    rows = await db.execute_fetchall(
        f"SELECT * FROM games WHERE id = ?{where}", (game_id,))
    if not rows:
        return None
    return _row_to_dict(rows[0])


async def add_game_player(game_id, seat, user_id):
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO game_players (game_id, seat, user_id) VALUES (?, ?, ?)",
        (game_id, seat, user_id),
    )
    await db.commit()


async def game_seat_names(game):
    """Qui occupait chaque siège d'une partie enregistrée.

    Rend 4 entrées `{"name", "bot"}` dans l'ordre moteur (N, E, S, O) ; `bot`
    vrai, `name` porte alors la *clé* d'agent (« dede », « doudou »…) et c'est
    à l'affichage de la traduire.

    `games.agents` ne suffit pas : en solo le siège humain y vaut « human » (le
    pseudo vit dans `games.user_id`), et en salon le pseudo y est bien écrit
    mais rien ne le distingue d'une clé de bot — un joueur nommé « dede »
    passerait pour un robot. Les sièges humains sont donc résolus par la base,
    comme le fait déjà l'Elo (`elo._seat_entities`).
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT gp.seat, u.username FROM game_players gp"
        " LEFT JOIN users u ON u.id = gp.user_id WHERE gp.game_id = ?",
        (game["id"],),
    )
    humans = {row[0]: row[1] for row in rows}
    if game["mode"] == "play" and game["human_seat"] is not None:
        name = None
        if game.get("user_id") is not None:
            urows = await db.execute_fetchall(
                "SELECT username FROM users WHERE id = ?", (game["user_id"],))
            name = urows[0][0] if urows else None
        humans[game["human_seat"]] = name

    agents = game["agents"] or {}
    seats = []
    for s in range(4):
        if s in humans:
            # Partie jouée sans compte : personne à nommer, mais bien un humain.
            seats.append({"name": humans[s] or "Invité", "bot": False})
        else:
            seats.append({"name": agents.get(str(s)) or "?", "bot": True})
    return seats


async def list_games(limit=50, offset=0, user_id=None):
    db = await get_db()
    where = "WHERE mode IN ('play', 'multi') AND is_complete = 1"
    seat_col = "human_seat AS user_seat, NULL AS elo_delta"
    params = []
    if user_id is not None:
        # Solo games carry user_id directly; multiplayer games via game_players.
        # user_seat = the requesting user's seat (their team's perspective).
        seat_col = ("COALESCE(human_seat, (SELECT seat FROM game_players gp"
                    " WHERE gp.game_id = games.id AND gp.user_id = ?)) AS user_seat, "
                    "(SELECT delta FROM elo_history eh WHERE eh.game_id = games.id"
                    " AND eh.kind = 'user' AND eh.ref = CAST(? AS TEXT)) AS elo_delta")
        params += [user_id, user_id]
        where += (" AND (user_id = ? OR id IN"
                  " (SELECT game_id FROM game_players WHERE user_id = ?))")
        params += [user_id, user_id]
    rows = await db.execute_fetchall(
        "SELECT id, mode, created_at, dealer, agents, human_seat, is_complete, "
        f"points_ns, points_ew, contract, {seat_col} FROM games {where} "
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
            "user_seat": row[10],
            "elo_delta": row[11],
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
        SELECT COUNT(*) AS games, SUM(win) AS wins FROM (
            SELECT CASE WHEN (human_seat % 2 = 0 AND points_ns > points_ew)
                          OR (human_seat % 2 = 1 AND points_ew > points_ns)
                   THEN 1 ELSE 0 END AS win
            FROM games
            WHERE mode = 'play' AND is_complete = 1 AND user_id = ?
                  AND human_seat IS NOT NULL
            UNION ALL
            SELECT CASE WHEN (gp.seat % 2 = 0 AND g.points_ns > g.points_ew)
                          OR (gp.seat % 2 = 1 AND g.points_ew > g.points_ns)
                   THEN 1 ELSE 0 END
            FROM games g JOIN game_players gp ON gp.game_id = g.id
            WHERE g.mode = 'multi' AND g.is_complete = 1 AND gp.user_id = ?
        )
        """,
        (user_id, user_id),
    )
    row = rows[0]
    return {"games": row[0] or 0, "wins": row[1] or 0}


async def get_analysis(game_id):
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT data FROM analysis WHERE game_id = ?", (game_id,))
    return json.loads(rows[0][0]) if rows else None


async def save_analysis(game_id, data_json):
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO analysis (game_id, created_at, data) VALUES (?, ?, ?)",
        (game_id, _now(), data_json),
    )
    await db.commit()


async def get_agent_review(game_id):
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT data FROM agent_review WHERE game_id = ?", (game_id,))
    return json.loads(rows[0][0]) if rows else None


async def save_agent_review(game_id, data_json):
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO agent_review (game_id, created_at, data) VALUES (?, ?, ?)",
        (game_id, _now(), data_json),
    )
    await db.commit()


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
