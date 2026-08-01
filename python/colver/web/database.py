"""SQLite database for persisting users, game history and bug reports.

Schema changes go through MIGRATIONS: each entry bumps PRAGMA user_version.
Migration 1 is idempotent (IF NOT EXISTS) so pre-migration prod databases
(user_version=0 with existing tables) adopt the system cleanly.
"""

import asyncio
import json
import logging
import os
import random
import re
import sqlite3
import string
from datetime import datetime, timezone
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

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
    # v10 — mise en quarantaine des donnes dont le journal ne se rejoue pas.
    # Des donnes enregistrées décrivent une partie impossible (une carte jouée
    # deux fois, une autre jamais) : `env.step()` ne valide pas la légalité, et
    # le gestionnaire `play` du solo lui passait l'action reçue telle quelle.
    # Le trou est bouché en écriture (`game_manager.check_legal`) ; ces deux
    # colonnes servent à écarter ce qui est déjà là. `checked_at` dit qu'une
    # ligne a été examinée (donc chacune ne l'est qu'une fois, cf.
    # `integrity.scan`), `invalid` ce que l'examen a conclu. On marque au lieu
    # d'effacer : une donne fausse est un incident, et sa trace vaut mieux que
    # sa disparition.
    """
    ALTER TABLE games ADD COLUMN invalid INTEGER NOT NULL DEFAULT 0;
    ALTER TABLE games ADD COLUMN checked_at TEXT;
    ALTER TABLE games ADD COLUMN invalid_reason TEXT;
    CREATE INDEX idx_games_unchecked ON games(checked_at) WHERE checked_at IS NULL;
    """,
    # v11 — cycle de vie du compte : adresse e-mail et réinitialisation.
    # `email` est facultative et unique — un compte sans adresse reste
    # parfaitement jouable, il n'a simplement aucun recours en cas d'oubli, et
    # c'est à l'interface de le dire. L'unicité est `COLLATE NOCASE` comme le
    # pseudo : deux comptes sur la même adresse rendraient le « qui suis-je »
    # d'un lien de réinitialisation ambigu.
    #
    # Les jetons sont stockés **hachés**, comme les sessions : une fuite de la
    # base ne doit pas donner de quoi prendre un compte. `used_at` les rend à
    # usage unique — sans lui, un lien qui traîne dans une boîte mail reste une
    # clé valable pendant toute sa durée de vie.
    """
    ALTER TABLE users ADD COLUMN email TEXT COLLATE NOCASE;
    CREATE UNIQUE INDEX idx_users_email ON users(email) WHERE email IS NOT NULL;

    CREATE TABLE password_resets (
        token_hash  TEXT PRIMARY KEY,
        user_id     INTEGER NOT NULL REFERENCES users(id),
        created_at  TEXT NOT NULL,
        expires_at  TEXT NOT NULL,
        used_at     TEXT
    );
    CREATE INDEX idx_password_resets_user ON password_resets(user_id);
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
        logger.info("Applied migration v%d", i)


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
        logger.info("Connected to %s", DB_PATH)
    return _db


def backup_db(dest_dir, keep=14):
    """Copie un instantané de la base dans `dest_dir`, avec rétention.

    Synchrone, à lancer via `asyncio.to_thread`. Le VACUUM INTO tourne sur une
    connexion sqlite3 dédiée, pas sur `_db` : la connexion partagée sérialise
    tout par un seul thread aiosqlite, et une copie complète y mettrait chaque
    écriture de partie en file d'attente le temps du backup. VACUUM INTO ne
    fait que lire la source, et en WAL un lecteur ne bloque jamais les
    écrivains — le serveur continue de jouer pendant la copie.

    L'appelant doit avoir attendu `get_db()` avant le premier backup : elle ne
    rend la main qu'une fois les migrations passées. Sinon l'instantané peut
    capturer un état mi-migration (script committé, `user_version` pas encore
    bumpé — `executescript` committe implicitement) que la restauration
    re-migrerait, et les migrations v2+ ne sont pas idempotentes.
    """
    dest_dir = Path(dest_dir)
    dest_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
    dest = dest_dir / f"colver-{stamp}.db"
    conn = sqlite3.connect(DB_PATH)
    try:
        # VACUUM INTO échoue si la cible existe ; le nom horodaté l'évite, et
        # un échec (loggé par l'appelant) vaut mieux qu'un écrasement. Un
        # VACUUM interrompu laisse un fichier partiel — c'est à l'application
        # de le supprimer (doc SQLite) ; sans ça, la rétention compterait des
        # cadavres comme des sauvegardes et évincerait les vraies.
        try:
            conn.execute("VACUUM INTO ?", (str(dest),))
        except BaseException:
            dest.unlink(missing_ok=True)
            raise
    finally:
        conn.close()
    if keep > 0:
        # Ne tourner que sur nos propres fichiers : une copie posée à la main
        # par un opérateur (`colver-avant-restauration.db`) ne doit ni compter
        # dans la rétention ni disparaître.
        ours = [p for p in dest_dir.glob("colver-*.db")
                if re.fullmatch(r"colver-\d{8}-\d{6}\.db", p.name)]
        for old in sorted(ours)[:-keep]:
            old.unlink()
    return dest


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


async def get_user_by_email(email):
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM users WHERE email = ?", (email,))
    return dict(rows[0]) if rows else None


async def set_user_email(user_id, email):
    """Poser (ou retirer, avec None) l'adresse d'un compte.

    Rend False si l'adresse est déjà prise : l'index unique est la seule
    autorité là-dessus — un « SELECT puis INSERT » laisserait passer deux
    inscriptions simultanées sur la même adresse.
    """
    db = await get_db()
    try:
        await db.execute("UPDATE users SET email = ? WHERE id = ?", (email, user_id))
        await db.commit()
        return True
    except aiosqlite.IntegrityError:
        return False


async def set_password(user_id, password_hash):
    db = await get_db()
    await db.execute("UPDATE users SET password_hash = ? WHERE id = ?",
                     (password_hash, user_id))
    await db.commit()


async def delete_user_sessions(user_id, keep_token_hash=None):
    """Révoquer les sessions d'un compte, sauf éventuellement celle en cours.

    Tout changement d'identifiant les fait tomber : après un mot de passe
    changé, une session ouverte ailleurs est exactement ce dont on voulait se
    débarrasser. On garde la sienne pour ne pas se déconnecter soi-même en se
    protégeant.
    """
    db = await get_db()
    if keep_token_hash:
        await db.execute(
            "DELETE FROM sessions WHERE user_id = ? AND token_hash != ?",
            (user_id, keep_token_hash))
    else:
        await db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    await db.commit()


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


# ===== Réinitialisation du mot de passe =====

async def create_password_reset(token_hash, user_id, expires_at):
    """Ouvrir un jeton de réinitialisation, en annulant les précédents.

    Un seul jeton vivant par compte : redemander un lien doit invalider le
    précédent, sinon chaque demande laisse derrière elle une clé de plus, et un
    ancien courriel resterait exploitable.
    """
    db = await get_db()
    await db.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
    await db.execute(
        "INSERT INTO password_resets (token_hash, user_id, created_at, expires_at) "
        "VALUES (?, ?, ?, ?)",
        (token_hash, user_id, _now(), expires_at),
    )
    # Ménage opportuniste, comme pour les sessions.
    await db.execute("DELETE FROM password_resets WHERE expires_at < ?", (_now(),))
    await db.commit()


async def get_password_reset(token_hash):
    """Le compte visé par un jeton encore valable, ou None.

    « Valable » = non expiré **et** jamais consommé : un lien de
    réinitialisation qui traîne dans une boîte mail ne doit servir qu'une fois.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT u.* FROM password_resets r JOIN users u ON u.id = r.user_id "
        "WHERE r.token_hash = ? AND r.expires_at >= ? AND r.used_at IS NULL",
        (token_hash, _now()),
    )
    return dict(rows[0]) if rows else None


async def consume_password_reset(token_hash):
    """Marquer un jeton comme utilisé. Rend False s'il ne l'était pas déjà pas.

    Le `used_at IS NULL` dans le UPDATE est ce qui rend l'usage unique
    atomique : deux requêtes concurrentes sur le même lien, une seule change
    une ligne.
    """
    db = await get_db()
    cur = await db.execute(
        "UPDATE password_resets SET used_at = ? "
        "WHERE token_hash = ? AND used_at IS NULL AND expires_at >= ?",
        (_now(), token_hash, _now()),
    )
    await db.commit()
    return cur.rowcount > 0


async def delete_account(user_id):
    """Effacer un compte en gardant ses donnes, détachées de lui.

    Une donne de salon appartient à quatre joueurs : l'effacer prendrait la
    partie des trois autres avec elle. On efface donc la *personne* — compte,
    sessions, jetons, classement — et on anonymise ce qu'elle laisse : la donne
    reste rejouable, le siège devient « Invité » (c'est déjà ce que
    `game_seat_names` affiche pour un `user_id` absent).

    L'Elo part avec le compte : une ligne de classement est une identité. Les
    lignes d'`elo_history` restent, rattachées à des donnes qui existent encore
    — les effacer ne rendrait pas leurs points aux adversaires, ça retirerait
    juste la trace de parties qu'ils ont bien jouées.
    """
    db = await get_db()
    await db.execute("DELETE FROM sessions WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM password_resets WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM game_players WHERE user_id = ?", (user_id,))
    await db.execute("UPDATE games SET user_id = NULL WHERE user_id = ?", (user_id,))
    await db.execute("UPDATE matches SET user_id = NULL WHERE user_id = ?", (user_id,))
    await db.execute("DELETE FROM elo_ratings WHERE kind = 'user' AND ref = ?",
                     (str(user_id),))
    await db.execute("DELETE FROM elo_history WHERE kind = 'user' AND ref = ?",
                     (str(user_id),))
    cur = await db.execute("DELETE FROM users WHERE id = ?", (user_id,))
    await db.commit()
    return cur.rowcount > 0


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


async def pending_deal(user_id, match_id=None):
    """La donne solo interrompue de ce joueur, avec de quoi la rejouer.

    Une donne coupée en plein milieu (rechargement, socket tuée, déploiement)
    reste `is_complete = 0` pour toujours. Elle porte pourtant tout ce qu'il
    faut pour la reprendre : `hands` et `dealer` écrits à la création,
    `actions` à chaque coup. C'est cette ligne-là que `game_manager.PlaySession`
    rejoue.

    Deux formes de donne interrompue, une seule requête : celle d'une partie
    (`match_id`, propriété vérifiée sur la partie) et celle qui n'appartient à
    aucune (le cas par défaut du site — `target = 0` ne crée pas de `matches`).
    Restreint à `mode = 'play'` : en salon c'est le pilote qui tient la donne,
    et il survit à la déconnexion d'un joueur.

    Renvoie la ligne entière, **mains comprises** — au seul appelant qui a
    prouvé qu'il en est le propriétaire, et à charge pour lui de n'en montrer
    que le siège du joueur (cf. `get_game`).
    """
    if not user_id:
        return None
    db = await get_db()
    if match_id:
        sql = ("SELECT g.* FROM games g JOIN matches m ON m.id = g.match_id "
               "WHERE g.match_id = ? AND m.user_id = ? AND m.is_complete = 0 "
               "  AND g.mode = 'play' AND g.is_complete = 0 "
               "ORDER BY g.deal_no DESC LIMIT 1")
        params = (match_id, user_id)
    else:
        sql = ("SELECT * FROM games "
               "WHERE user_id = ? AND match_id IS NULL "
               "  AND mode = 'play' AND is_complete = 0 "
               "ORDER BY created_at DESC LIMIT 1")
        params = (user_id,)
    rows = await db.execute_fetchall(sql, params)
    if not rows:
        return None
    return _row_to_dict(rows[0])


async def drop_deal(game_id):
    """Effacer une donne interrompue devenue irrejouable.

    Sert quand la reprise échoue (journal et donne qui ne se correspondent
    pas) : sans ça la ligne serait reproposée à chaque passage sur Jouer, et
    échouerait à chaque fois. Une donne jamais terminée n'est comptée nulle
    part — ni score de partie, ni Elo, ni listing — donc rien ne la pleure.
    """
    db = await get_db()
    await db.execute("DELETE FROM games WHERE id = ? AND is_complete = 0",
                     (game_id,))
    await db.commit()


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


async def mark_game_checked(game_id, reason=None):
    """Consigner le verdict d'`integrity.check_deal` sur une donne.

    Écrit toujours `checked_at`, y compris quand la donne est saine : c'est lui
    qui empêche de la réexaminer à chaque démarrage. Une donne écartée voit ses
    analyses en cache partir avec elle — elles ont été calculées depuis un état
    impossible, et `get_or_compute` sert le cache avant de relire la donne (même
    raison qu'à la migration v9).
    """
    db = await get_db()
    await db.execute(
        "UPDATE games SET checked_at = ?, invalid = ?, invalid_reason = ? "
        "WHERE id = ?",
        (_now(), 0 if reason is None else 1, reason, game_id),
    )
    if reason is not None:
        await db.execute("DELETE FROM analysis WHERE game_id = ?", (game_id,))
        await db.execute("DELETE FROM agent_review WHERE game_id = ?", (game_id,))
    await db.commit()


async def list_invalid_games(limit=100):
    """Les donnes écartées, pour l'exploitant — rien ne les montre ailleurs."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT id, mode, created_at, checked_at, invalid_reason FROM games "
        "WHERE invalid = 1 ORDER BY created_at DESC LIMIT ?", (limit,))
    return [
        {"id": r[0], "mode": r[1], "created_at": r[2],
         "checked_at": r[3], "reason": r[4]}
        for r in rows
    ]


async def get_game(game_id, include_incomplete=False, include_invalid=False):
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

    Une donne écartée (`invalid = 1`, cf. `integrity`) n'est pas rendue non
    plus : son journal ne décrit pas une partie jouable, donc Rejouer y montre
    des cartes en double et son analyse part d'un état impossible. Seul le scan
    d'intégrité lui-même passe outre (`include_invalid=True`), pour pouvoir la
    relire.
    """
    db = await get_db()
    where = "" if include_incomplete else " AND is_complete = 1"
    if not include_invalid:
        where += " AND invalid = 0"
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


async def random_user_game(user_id, exclude=()):
    """Une donne terminée du joueur, tirée au hasard — pour l'entraînement.

    `exclude` évite de resservir les dernières : le vivier d'un joueur se
    compte en dizaines de donnes, donc sans mémoire il rejouerait la même au
    bout de quelques tirages. La liste vient du client (elle est à lui), et
    elle est bornée là-bas ; on la borne aussi ici, un `IN (...)` de longueur
    libre n'a rien à faire dans une requête.
    """
    db = await get_db()
    exclude = [str(g) for g in list(exclude)[:20]]
    holes = ",".join("?" * len(exclude))
    where = ("WHERE mode IN ('play', 'multi') AND is_complete = 1 AND invalid = 0"
             " AND (user_id = ? OR id IN"
             " (SELECT game_id FROM game_players WHERE user_id = ?))")
    params = [user_id, user_id]
    if exclude:
        where += f" AND id NOT IN ({holes})"
        params += exclude
    rows = await db.execute_fetchall(
        f"SELECT * FROM games {where} ORDER BY RANDOM() LIMIT 1", tuple(params))
    if not rows:
        return None
    return _row_to_dict(rows[0])


async def list_games(limit=50, offset=0, user_id=None):
    db = await get_db()
    where = "WHERE mode IN ('play', 'multi') AND is_complete = 1 AND invalid = 0"
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
            WHERE mode = 'play' AND is_complete = 1 AND invalid = 0 AND user_id = ?
                  AND human_seat IS NOT NULL
            UNION ALL
            SELECT CASE WHEN (gp.seat % 2 = 0 AND g.points_ns > g.points_ew)
                          OR (gp.seat % 2 = 1 AND g.points_ew > g.points_ns)
                   THEN 1 ELSE 0 END
            FROM games g JOIN game_players gp ON gp.game_id = g.id
            WHERE g.mode = 'multi' AND g.is_complete = 1 AND g.invalid = 0
                  AND gp.user_id = ?
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
