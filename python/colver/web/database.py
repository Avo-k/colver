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

    # v12 — les bots deviennent l'étalon : Elo figé, plus de dérive.
    #
    # Jusqu'ici les bots étaient notés comme des joueurs (K = 8) et dérivaient
    # avec la population : Dédé est passé de 1000 à 1044 (pic 1119) simplement
    # parce que les humains perdent contre lui. Or il n'est pas un joueur du
    # pool — il tient trois sièges sur quatre et 881 donnes sur 1073, donc il
    # *est* l'échelle. Une échelle qui bouge dévalue en silence tous les
    # inscrits dès qu'un joueur plus faible arrive.
    #
    # Les classements sont des données **dérivées** : on vide les deux tables et
    # le backfill du démarrage les reconstruit sur la nouvelle échelle. Rien
    # n'est perdu, `games` porte tout ce qu'il faut pour recalculer.
    """
    DELETE FROM elo_history;
    DELETE FROM elo_ratings;
    """,

    # v13 — R3 : une donne se note à la marge, plus au signe.
    #
    # Le barème interdit les marges proches de zéro (un contrat réussi rapporte
    # au moins 3V−162, une chute exactement −(162+V)) : zéro donne sur 2 999 sous
    # 78 points d'écart. Le signe ne bouge donc presque jamais alors que la marge
    # suit — d'où un classement qui voyait une fraction de ce qui se passait, et
    # l'écart entre « Dédé gagne 55,4 % des donnes » et « Dédé gagne 72 % des
    # matchs ».
    #
    # Comme en v12 : les classements sont dérivés, on vide et le backfill du
    # démarrage reconstruit sur la nouvelle échelle.
    """
    DELETE FROM elo_history;
    DELETE FROM elo_ratings;
    """,
    # v14 — l'unité notée devient la **partie en 2000 points**, plus la donne.
    #
    # Trois raisons, dans l'ordre d'importance : c'est le format des tournois
    # réels, c'est celui de l'arène (donc l'ancre d'un bot devient une mesure
    # directe au lieu d'une conversion), et c'est le seul levier honnête qui
    # élargisse l'échelle — mesuré **×3,4** sur deux couples indépendants
    # (DouDou35→DouDou50 : 62,5 % des parties contre 46,0 % des donnes ;
    # Heuristique→DouDou50 : 69,3 % contre 42,7 %). L'étendue du jeu de la carte
    # passe ainsi de 171 à ~580 Elo. Voir `docs/classement_et_scoring.md` §8.
    #
    # `elo_history` était clé sur `game_id` : on la reconstruit sur `match_id`.
    # Les donnes isolées et les parties en 1000 restent jouables et analysables,
    # elles ne comptent simplement plus au classement.
    """
    DROP TABLE elo_history;
    CREATE TABLE elo_history (
        match_id   TEXT NOT NULL REFERENCES matches(id),
        kind       TEXT NOT NULL,
        ref        TEXT NOT NULL,
        delta      REAL NOT NULL,
        elo_after  REAL NOT NULL,
        PRIMARY KEY (match_id, kind, ref)
    );
    DELETE FROM elo_ratings;
    """,
    # v15 — la version d'une analyse devient une colonne.
    #
    # `analysis.data` est un blob JSON de ~5 ko qui porte son `version` à
    # l'intérieur. Savoir « combien de donnes sont analysées à la version
    # courante » demandait donc de désérialiser **toutes** les lignes : 25 ms
    # sur les 1 200 donnes d'aujourd'hui, ~1,1 s et ~250 Mo lus à 50 000. Une
    # requête de page qui grossit avec le corpus n'a rien à faire sur une base
    # à connexion unique, où elle met en file d'attente les coups de toutes les
    # parties en cours.
    #
    # La colonne est **dérivée** du blob, jamais saisie : `save_analysis`
    # l'extrait à l'écriture. Le remplissage rétroactif passe par JSON1 plutôt
    # que par du Python — une seule passe, pas de va-et-vient.
    """
    ALTER TABLE analysis ADD COLUMN version INTEGER;
    UPDATE analysis SET version = json_extract(data, '$.version');
    CREATE INDEX idx_analysis_version ON analysis(version);
    """,
    # v16 — la donne enregistre enfin ce qu'elle a **marqué**.
    #
    # `games.points_ns/ew` sont les points **cartes** (`env.get_points()`,
    # 0-252), pas les points marqués (`env.rewards()`, contrat compris). Les
    # deux ne se déduisent pas l'un de l'autre : le second dépend du contrat, de
    # la contre, de la belote et du dix de der. Jusqu'ici le score marqué d'une
    # donne n'était enregistré **nulle part** — seul son cumul vivait dans
    # `matches.points_ns/ew`.
    #
    # Ce trou n'est pas théorique, il produisait un chiffre faux à l'écran :
    # `user_game_stats` décidait « victoire » sur `points_ns > points_ew`, donc
    # sur les cartes. Une chute où le preneur garde la majorité des cartes —
    # 110♠ annoncé, 90 points faits, 0 marqué contre 272 — était comptée comme
    # une **victoire** du preneur, et s'affichait telle quelle sur /compte.
    #
    # Deux colonnes plutôt qu'un rejeu à la lecture : rejouer une donne coûte
    # ~0,3 ms, donc rien pour une donne et tout pour un classement (O(corpus),
    # ~54 000 donnes/an au rythme actuel, sur la connexion qui sérialise les
    # écritures de jeu). Elles rendent en prime le **contrat tenu** lisible en
    # SQL : sous ce barème un preneur qui chute marque exactement 0, donc
    # « réussi » ⟺ `score[camp du preneur] > 0`.
    #
    # Remplissage rétroactif par rejeu au démarrage (`_backfill_scores`), pas
    # ici : SQLite ne connaît pas le barème.
    """
    ALTER TABLE games ADD COLUMN score_ns INTEGER;
    ALTER TABLE games ADD COLUMN score_ew INTEGER;
    CREATE INDEX idx_games_unscored ON games(is_complete, score_ns)
        WHERE score_ns IS NULL;
    """,
    # v17 — la note devient un posterior, plus une récurrence.
    #
    # `elo.py` calculait la note par mise à jour incrémentale (K décroissant
    # depuis 1000). Trois défauts mesurés, détaillés dans son en-tête ; le plus
    # visible est que le tableau **ordonnait par inexpérience** (Spearman −0,89
    # entre parties jouées et note affichée) : tout le monde partait de 1000 et
    # descendait, donc le classement disait surtout qui avait le moins joué.
    #
    # La note est désormais le posterior exact recalculé depuis le bilan complet,
    # publié sous sa forme conservatrice `mu - 2*sigma`. Ce qui change en base :
    #
    # - `elo_history` porte le **bilan** et non plus le pas d'une récurrence :
    #   `score`, `partner_elo`, `opp_elo` suffisent à refaire le calcul de zéro.
    #   `delta` / `elo_after` gardent leurs noms et changent de sens (déplacement
    #   de la note publiée causé par cette partie), donc `list_matches` et
    #   `get_match` n'ont rien à changer.
    # - `elo_ratings` gagne `level` (niveau estimé) et `sigma` (son incertitude).
    #   `elo` reste la note, donc la clé de tri, donc `ORDER BY elo DESC` tient.
    #
    # On vide les deux tables : `backfill()` les reconstruit au démarrage à
    # partir de `matches`, qui est la source de vérité. Rien n'est perdu — les
    # anciennes valeurs étaient sur une échelle qui n'existe plus.
    """
    DROP TABLE elo_history;
    CREATE TABLE elo_history (
        match_id    TEXT NOT NULL REFERENCES matches(id),
        kind        TEXT NOT NULL,
        ref         TEXT NOT NULL,
        score       REAL NOT NULL,
        partner_elo REAL NOT NULL,
        opp_elo     REAL NOT NULL,
        delta       REAL NOT NULL,
        elo_after   REAL NOT NULL,
        PRIMARY KEY (match_id, kind, ref)
    );
    ALTER TABLE elo_ratings ADD COLUMN level REAL;
    ALTER TABLE elo_ratings ADD COLUMN sigma REAL;
    DELETE FROM elo_ratings;
    """,
    # v18 — les deux simulations d'analyse cessent d'être jetées.
    #
    # `annonces_sim` (200 solves DD sur donne complète + 1 000 déroulements
    # DouDou50) et `card_analysis` (200-500 solves + 600 déroulements + une
    # recherche IS-DD) sont les deux calculs les plus chers du site, et le seul
    # à ne rien laisser derrière lui. La barre « Mains analysées » de la page
    # annonces le montre bien : elle enregistre l'**entrée** (main, enchères,
    # score) et pas le résultat, donc rouvrir une main enregistrée relançait
    # tout — et rendait au passage des chiffres différents, les mondes venant
    # de playgen.
    #
    # Table à part de `analysis` / `agent_review`, pour une raison de forme :
    # celles-ci sont clés sur `game_id`, donc bornées par le corpus de donnes.
    # Ici la clé est un hachage d'entrées **non bornées** (un CFN se tape à la
    # main, une main aussi), d'où `used_at` et l'éviction LRU — sans quoi la
    # table grossirait sans plafond.
    #
    # `version` sort du blob plutôt que d'y vivre (leçon de v15) : c'est le
    # prédicat de fraîcheur, il est lu à chaque requête, et le désérialiser
    # coûterait un blob entier pour un entier.
    """
    CREATE TABLE analysis_cache (
        kind       TEXT NOT NULL,
        cache_key  TEXT NOT NULL,
        version    INTEGER NOT NULL,
        created_at TEXT NOT NULL,
        used_at    TEXT NOT NULL,
        hits       INTEGER NOT NULL DEFAULT 0,
        data       TEXT NOT NULL,
        PRIMARY KEY (kind, cache_key)
    );
    CREATE INDEX idx_analysis_cache_lru ON analysis_cache(used_at);
    """,
    # v19 — le progrès sur « Compter les points » suit le compte, plus le
    # navigateur.
    #
    # Série, record, taux de justesse et écart moyen vivaient en localStorage
    # (`colver:compter:stats`). C'est la seule chose de tout le site qu'un
    # changement d'appareil ou un vidage de cache **détruit sans recours** : une
    # analyse se recalcule, une donne est en base, un record d'exercice n'existe
    # nulle part ailleurs.
    #
    # Les compteurs s'incrémentent par **delta** et non par totaux poussés
    # depuis le client. Deux raisons : un total envoyé écrase le travail d'un
    # second onglet ou d'un second appareil, et il fait du client l'autorité sur
    # son propre record. `streak` fait exception — ce n'est pas une somme mais
    # un état courant, donc dernière valeur écrite ; `best` en dérive par un max
    # côté serveur, pour qu'un client ne puisse pas s'en déclarer un.
    #
    # Le local reste : il porte le jeu **anonyme**, qui n'a pas de ligne ici. On
    # ne fusionne pas les deux — additionner des essais anonymes à la connexion
    # les compterait deux fois sur l'appareil qui les a déjà envoyés.
    """
    CREATE TABLE exercise_stats (
        user_id       INTEGER NOT NULL REFERENCES users(id),
        exercise      TEXT NOT NULL,
        variant       TEXT NOT NULL,
        plays         INTEGER NOT NULL DEFAULT 0,
        exact         INTEGER NOT NULL DEFAULT 0,
        sum_abs_delta INTEGER NOT NULL DEFAULT 0,
        streak        INTEGER NOT NULL DEFAULT 0,
        best          INTEGER NOT NULL DEFAULT 0,
        updated_at    TEXT NOT NULL,
        PRIMARY KEY (user_id, exercise, variant)
    );
    """,
    # v20 — la marge d'une partie entre au classement.
    #
    # Gagner 2000-1900 et gagner 2000-200 comptaient pareil. Le score d'une
    # partie est désormais `sigma(marge / 1047)` au lieu de 1/0 — mesuré avant
    # d'être écrit (log-perte leave-one-out 0,6770 → 0,6567, soit le double
    # d'information utile au-dessus du hasard ; détail dans l'en-tête d'`elo`).
    #
    # Aucun changement de schéma : `elo_history.score` était déjà un REAL, et le
    # posterior accepte un score fractionnaire depuis v17. Ce qui change est la
    # **valeur** de toutes les lignes, donc on les jette et `backfill()` les
    # reconstruit au démarrage depuis `matches`, la source de vérité. On vide
    # aussi `elo_ratings` : ses notes dérivent des lignes qu'on efface.
    #
    # Ce que ça ne change pas, et qu'il ne faut pas promettre : l'incertitude
    # affichée. La courbure d'une vraisemblance de Bernoulli ne dépend pas du
    # score observé, donc sigma reste où il était.
    #
    # Effet visible à la bascule, simulé sur la base de prod du 2026-08-06 : les
    # neuf comptes notés montent de +4 à +92 points d'affichage (adoucir un score
    # le tire vers 1/2, et la population gagne 28 % de ses parties). **L'ordre du
    # classement est inchangé.**
    """
    DELETE FROM elo_history;
    DELETE FROM elo_ratings;
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
    # Le progrès sur les exercices n'appartient qu'à cette personne — rien ne
    # s'y rattache, contrairement à une donne de salon : il part entièrement.
    await db.execute("DELETE FROM exercise_stats WHERE user_id = ?", (user_id,))
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


async def complete_game(game_id, points_ns, points_ew, contract,
                        score_ns=None, score_ew=None):
    """Clore une donne. **Deux échelles, et il faut les deux.**

    `points_*` sont les points **cartes** (`env.get_points()`, 0-252) : ce que
    les plis ont rapporté. `score_*` sont les points **marqués**
    (`env.rewards()`) : ce que la donne inscrit au tableau, contrat, contre,
    belote et dix de der compris. Les seconds ne se déduisent pas des premiers.

    `score_*` reste facultatif pour que les appelants historiques et les tests
    ne se cassent pas, mais les deux chemins de production le passent — sans
    quoi la donne repartirait dans la file du rattrapage au démarrage suivant.
    """
    db = await get_db()
    await db.execute(
        "UPDATE games SET is_complete = 1, points_ns = ?, points_ew = ?, contract = ?, "
        "score_ns = ?, score_ew = ? WHERE id = ?",
        (points_ns, points_ew, json.dumps(contract) if contract else None,
         None if score_ns is None else int(score_ns),
         None if score_ew is None else int(score_ew),
         game_id),
    )
    await db.commit()


async def set_deal_scores(game_id, score_ns, score_ew):
    """Poser les points marqués d'une donne déjà close (rattrapage)."""
    db = await get_db()
    await db.execute(
        "UPDATE games SET score_ns = ?, score_ew = ? WHERE id = ?",
        (int(score_ns), int(score_ew), game_id),
    )
    await db.commit()


async def games_missing_scores(limit=5000):
    """Les donnes closes et saines dont les points marqués manquent encore."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT id FROM games WHERE is_complete = 1 AND invalid = 0 "
        "AND score_ns IS NULL ORDER BY created_at LIMIT ?",
        (limit,),
    )
    return [r[0] for r in rows]


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


async def list_matches(user_id, limit=20, offset=0):
    """Les parties **terminées** d'un joueur, la plus récente d'abord.

    Le complément exact de `list_open_matches` : celle-ci ne rend que
    `is_complete = 1`, **abandons compris**. Une partie concédée est un
    résultat — l'Elo la note comme une défaite (`elo._losing_team`), donc la
    cacher ici ferait deux comptes différents de la même chose.

    Le siège du joueur ne se lit pas au même endroit selon le mode : en solo
    c'est `matches.human_seat`, en salon c'est `game_players` — là-bas
    `matches.user_id` désigne **l'hôte**, pas les trois autres. D'où le
    COALESCE, et d'où l'appartenance en deux branches : sans la seconde, un
    invité de salon ne verrait aucune des parties qu'il a jouées.

    `elo_delta` est nul-able et doit le rester à l'affichage : seule une partie
    en 2000 points jouée jusqu'au bout est notée (migration v14), et une partie
    dont une donne est en quarantaine ne l'est pas non plus. « Non noté » et
    « noté 0 » sont deux choses différentes.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT m.id, m.mode, m.created_at, m.target, m.points_ns, m.points_ew, "
        "m.deals, m.winner, m.abandoned, m.pacing, "
        "COALESCE(m.human_seat, (SELECT gp.seat FROM game_players gp "
        "        JOIN games g ON g.id = gp.game_id "
        "       WHERE g.match_id = m.id AND gp.user_id = ? LIMIT 1)) AS user_seat, "
        "(SELECT e.delta FROM elo_history e WHERE e.match_id = m.id "
        "   AND e.kind = 'user' AND e.ref = ?) AS elo_delta "
        "FROM matches m "
        "WHERE m.is_complete = 1 AND m.target > 0 "
        "  AND (m.user_id = ? OR EXISTS (SELECT 1 FROM game_players gp "
        "       JOIN games g ON g.id = gp.game_id "
        "      WHERE g.match_id = m.id AND gp.user_id = ?)) "
        "ORDER BY m.created_at DESC LIMIT ? OFFSET ?",
        (user_id, str(user_id), user_id, user_id, limit, offset),
    )
    return [
        {
            "id": r[0], "mode": r[1], "created_at": r[2], "target": r[3],
            "points_ns": r[4], "points_ew": r[5], "deals": r[6],
            "winner": r[7], "abandoned": bool(r[8]), "pacing": r[9],
            "user_seat": r[10], "elo_delta": r[11],
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

    **`mode = 'play'` est une garde, pas un filtre de confort.** Un salon écrit
    `matches.user_id = host_id` : sans elle, l'hôte pouvait concéder sa partie
    de *salon* par le chemin solo, et le pilote du salon remettait ensuite
    `is_complete = 0` au `update_match` suivant — sans toucher `abandoned`,
    donc dans un état qu'aucun code ne prévoit. L'UI ne l'offrait pas
    (`list_open_matches` filtre déjà le mode), mais le message WebSocket, lui,
    est accessible : la garde tenait par coïncidence.
    """
    db = await get_db()
    cur = await db.execute(
        "UPDATE matches SET is_complete = 1, abandoned = 1, winner = NULL "
        "WHERE id = ? AND user_id = ? AND is_complete = 0 AND mode = 'play'",
        (match_id, user_id),
    )
    await db.commit()
    return cur.rowcount > 0


def _taker_seat(actions_json):
    """Le siège qui a pris : la **dernière** annonce chiffrée de l'enchère.

    `games.contract` ne porte que l'*équipe* (`$.team`), et en solo trois sièges
    sur quatre sont des bots : « mon camp a pris » ne dit pas « j'ai pris ».
    Même définition qu'en SQL dans `stats._TAKER` — là-bas parce qu'il faut
    agréger des centaines de donnes sans les relire, ici parce que le journal
    est déjà en main.

    Les actions 1-40 sont des annonces (0 = passe, 41/42 = contre et surcontre),
    mais **seulement en phase 0** : en phase de jeu les mêmes entiers sont des
    indices de carte. Un journal sans `phase` rend donc None plutôt qu'un siège
    tiré d'une carte — c'est déjà ce que fait `stats`.
    """
    try:
        actions = json.loads(actions_json) if actions_json else []
    except (TypeError, ValueError):
        return None
    taker = None
    for entry in actions:
        if entry.get("phase") == 0 and 1 <= int(entry.get("action", 0)) <= 40:
            taker = entry.get("player")
    return taker


async def _match_seat_names(match_id):
    """Qui tenait les quatre sièges d'une partie, lus sur sa première donne.

    Une donne suffit : en salon les sièges sont liés aux comptes pour toute la
    partie, en solo `human_seat` est un réglage de partie (`matches.human_seat`,
    migration v8). C'est la même lecture que fait l'Elo (`elo._match_seats`).
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT id, mode, human_seat, user_id, agents FROM games "
        "WHERE match_id = ? AND is_complete = 1 ORDER BY deal_no LIMIT 1",
        (match_id,),
    )
    if not rows:
        return None
    r = rows[0]
    return await game_seat_names({
        "id": r[0], "mode": r[1], "human_seat": r[2], "user_id": r[3],
        "agents": json.loads(r[4]) if r[4] else {},
    })


async def get_match(match_id):
    """Une partie et sa **feuille de marque** : une ligne par donne jouée, avec
    son score marqué et le cumul courant.

    Ce cumul-là n'était pas calculable avant la migration **v16** : le score
    marqué d'une donne n'était enregistré nulle part, seul son total vivait dans
    `matches.points_ns/ew`, et `games.points_ns/ew` sont les points *cartes* —
    une autre échelle, qui donne des chiffres plausibles et faux (152 au lieu
    de 380).

    **Les deux totaux sont rendus, et ils peuvent diverger** (`points_ns/ew` de
    la partie contre `sheet_ns/ew` recalculé). Trois causes, toutes réelles :
    une donne pas encore rattrapée par `integrity.backfill_scores`, une donne
    mise en quarantaine après coup, et surtout le **barème** — la ligne
    `matches` a été écrite au fil de la partie sous le barème du jour, tandis
    que les scores par donne d'une vieille partie ont été *rejoués* sous le
    barème courant, qui a changé deux fois (2026-04-16, 2026-07-31). C'est le
    total de `matches` qui fait foi : c'est lui qui a désigné le vainqueur et
    nourri l'Elo. L'écart se dit, il ne se masque pas — d'où les deux compteurs
    `unscored_deals` / `invalid_deals` qui l'expliquent.

    Ne rend que les donnes **terminées et saines** : une donne en plan n'est
    comptée nulle part ailleurs non plus, et une donne en quarantaine décrit une
    partie impossible.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT * FROM matches WHERE id = ?", (match_id,))
    if not rows:
        return None
    match = dict(rows[0])
    match["is_complete"] = bool(match["is_complete"])
    match["abandoned"] = bool(match["abandoned"])
    deals = await db.execute_fetchall(
        "SELECT id, deal_no, dealer, points_ns, points_ew, score_ns, score_ew, "
        "contract, actions, created_at, is_complete, invalid "
        "FROM games WHERE match_id = ? ORDER BY deal_no",
        (match_id,),
    )
    games, total, unscored, invalid = [], [0, 0], 0, 0
    for d in deals:
        if not d[10]:
            continue
        if d[11]:
            invalid += 1
            continue
        if d[5] is None:
            unscored += 1
        else:
            total = [total[0] + d[5], total[1] + d[6]]
        games.append({
            "id": d[0],
            "deal_no": d[1],
            "dealer": d[2],
            "points_ns": d[3],
            "points_ew": d[4],
            "score_ns": d[5],
            "score_ew": d[6],
            "contract": json.loads(d[7]) if d[7] else None,
            "taker": _taker_seat(d[8]),
            "created_at": d[9],
            # Le cumul *après* cette donne — c'est la colonne qu'on lit sur une
            # feuille de marque, et elle n'a de sens qu'en ordre de donne.
            "total_ns": total[0],
            "total_ew": total[1],
        })
    match["games"] = games
    match["sheet_ns"], match["sheet_ew"] = total
    match["unscored_deals"] = unscored
    match["invalid_deals"] = invalid
    match["seats"] = await _match_seat_names(match_id)
    # Le déplacement de note causé par la partie, humains seulement : la note
    # d'un bot est une ancre figée, donc ses lignes sont des zéros écrits pour
    # l'idempotence de `rate_match`, pas une information.
    #
    # ⚠️ Depuis la v17 `delta` n'est plus le pas d'une récurrence K mais l'écart
    # entre la note publiée avant et après cette partie — deux définitions qui
    # donnent des ordres de grandeur différents. Les anciennes lignes ont été
    # supprimées par la migration, il n'y a donc pas de mélange en base.
    elo_rows = await db.execute_fetchall(
        "SELECT e.ref, u.username, e.delta, e.elo_after FROM elo_history e "
        "LEFT JOIN users u ON u.id = CAST(e.ref AS INTEGER) "
        "WHERE e.match_id = ? AND e.kind = 'user'",
        (match_id,),
    )
    match["elo"] = [
        {"ref": e[0], "name": e[1], "delta": e[2], "elo_after": e[3]}
        for e in elo_rows
    ]
    return match


async def deal_match_context(game):
    """Où en était la partie **avant** que cette donne se joue.

    Le score cumulé n'est pas qu'un affichage : bid v6 lit une observation
    *score-aware*, donc la même main s'annonce autrement à 900-200 qu'à 0-0
    (cf. `match_state`). Une donne relue hors de son score est donc analysée
    sous une autre question que celle que le joueur s'est posée à la table.

    `before` est le cumul des donnes **antérieures** — c'est lui qui a compté
    au moment d'annoncer, pas le total final. Il se recalcule depuis
    `games.score_ns/ew` (migration v16) : `matches.points_ns/ew` ne porte que
    le total de la partie entière, sans découpage par donne.

    Rend None hors partie : `target = 0` ne crée pas de ligne `matches`, et
    c'est le cas par défaut du site.

    Le repère est physique (`[NS, EW]`), comme partout côté serveur. Rendre
    `owner_id` laisse l'appelant appliquer la règle de diffusion : une partie
    close se lit comme une donne se lit, une partie **en cours** dirait le
    score en direct d'une table où l'on joue encore (même règle que
    `get_match`, servi seulement une fois la partie finie).
    """
    match_id = game.get("match_id")
    if not match_id:
        return None
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT target, user_id, is_complete, abandoned, points_ns, points_ew "
        "FROM matches WHERE id = ?", (match_id,))
    if not rows:
        return None
    target, owner_id, is_complete, abandoned, pts_ns, pts_ew = rows[0]
    if not target:
        return None

    deal_no = game.get("deal_no")
    deals = await db.execute_fetchall(
        "SELECT deal_no, score_ns, score_ew FROM games "
        "WHERE match_id = ? AND is_complete = 1 AND invalid = 0 "
        "ORDER BY deal_no", (match_id,))

    before, unscored = [0, 0], 0
    for d_no, s_ns, s_ew in deals:
        if deal_no is None or d_no is None or d_no >= deal_no:
            continue
        if s_ns is None:
            # Donne pas encore rattrapée par `integrity.backfill_scores` : le
            # cumul est alors incomplet, et il vaut mieux le dire que servir un
            # chiffre trop bas sans le signaler.
            unscored += 1
            continue
        before = [before[0] + s_ns, before[1] + s_ew]

    score = ([game["score_ns"], game["score_ew"]]
             if game.get("score_ns") is not None else None)
    return {
        "id": match_id,
        "target": int(target),
        "deal_no": deal_no,
        "deals": len(deals),
        "before": before,
        "after": [before[0] + score[0], before[1] + score[1]] if score else None,
        "score": score,
        "unscored_before": unscored,
        "is_complete": bool(is_complete),
        "abandoned": bool(abandoned),
        "final": [pts_ns, pts_ew],
        "owner_id": owner_id,
    }


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
    """Les donnes terminées, éventuellement filtrées sur un joueur.

    **Une donne ne porte plus de variation d'Elo.** Cette liste en rendait une
    (`elo_delta`), lue sur `elo_history.game_id` — colonne que la migration v14
    a supprimée en reconstruisant la table sur `match_id`, l'unité notée étant
    devenue la partie en 2000 points. La requête levait donc `no such column`
    dès qu'un `user_id` était passé, et « Mes donnes » ne s'affichait plus pour
    personne de connecté. Le chemin anonyme, lui, n'assemblait pas ce
    sous-select : les tests, qui appellent `list_games()` sans joueur, sont
    restés verts pendant tout ce temps — d'où le test de non-régression qui
    passe explicitement un `user_id` (`tests/test_elo.py`).

    La réparation est de **retirer** la colonne, pas de la rebrancher sur
    `match_id` : la variation appartient à la partie. La reporter sur chacune de
    ses ~10 donnes afficherait dix fois le même chiffre en laissant croire que
    chaque donne l'a gagné.
    """
    db = await get_db()
    where = "WHERE mode IN ('play', 'multi') AND is_complete = 1 AND invalid = 0"
    seat_col = "human_seat AS user_seat"
    params = []
    if user_id is not None:
        # Solo games carry user_id directly; multiplayer games via game_players.
        # user_seat = the requesting user's seat (their team's perspective).
        seat_col = ("COALESCE(human_seat, (SELECT seat FROM game_players gp"
                    " WHERE gp.game_id = games.id AND gp.user_id = ?)) AS user_seat")
        params += [user_id]
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
        }
        result.append(d)
    return result


async def user_game_stats(user_id):
    """Donnes jouées et gagnées par un joueur, **au score marqué**.

    Le siège se lit sur `human_seat` en solo et sur `game_players` en salon ;
    les sièges pairs sont N-S, les impairs E-O.

    ⚠️ **Cette fonction comparait les points *cartes*** (`points_ns/ew`) jusqu'à
    la migration v16, ce qui inversait le résultat de toute chute où le preneur
    gardait la majorité des cartes : 110♠ annoncé, 90 points faits, 0 marqué
    contre 272 — et l'humain était crédité d'une victoire, affichée sur
    /compte. Le sens de « gagner une donne » est le score marqué, et il n'était
    enregistré nulle part avant v16.

    Une donne dont le score n'est pas encore rattrapé est **exclue du
    dénominateur** plutôt que comptée perdue : mieux vaut un compte qui
    grandit au fil du rattrapage qu'un taux faux servi tout de suite.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        """
        SELECT COUNT(*) AS games, SUM(win) AS wins FROM (
            SELECT CASE WHEN (human_seat % 2 = 0 AND score_ns > score_ew)
                          OR (human_seat % 2 = 1 AND score_ew > score_ns)
                   THEN 1 ELSE 0 END AS win
            FROM games
            WHERE mode = 'play' AND is_complete = 1 AND invalid = 0 AND user_id = ?
                  AND human_seat IS NOT NULL AND score_ns IS NOT NULL
            UNION ALL
            SELECT CASE WHEN (gp.seat % 2 = 0 AND g.score_ns > g.score_ew)
                          OR (gp.seat % 2 = 1 AND g.score_ew > g.score_ns)
                   THEN 1 ELSE 0 END
            FROM games g JOIN game_players gp ON gp.game_id = g.id
            WHERE g.mode = 'multi' AND g.is_complete = 1 AND g.invalid = 0
                  AND gp.user_id = ? AND g.score_ns IS NOT NULL
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
    """Écrire une analyse en cache, avec sa version **extraite du blob**.

    La colonne `version` (v15) est dérivée, jamais fournie par l'appelant : une
    seule source de vérité, celle qui voyage avec les données. `json_extract`
    la relit dans la même instruction que l'insertion, donc les deux ne peuvent
    pas diverger.
    """
    db = await get_db()
    await db.execute(
        "INSERT OR REPLACE INTO analysis (game_id, created_at, data, version) "
        "VALUES (?, ?, ?, json_extract(?, '$.version'))",
        (game_id, _now(), data_json, data_json),
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


# ===== Cache des simulations d'analyse (v18) =====
#
# Contrairement à `analysis` / `agent_review`, la clé n'est pas un `game_id`
# mais un hachage d'entrées non bornées : la table a donc besoin d'un plafond.
# Voir `sim_cache.py` pour la dérivation des clés et les règles de fraîcheur.

# Au-delà, `put_sim_cache` évince les entrées les moins récemment servies. À
# ~5 ko l'entrée, 20 000 tiennent dans ~100 Mo — l'ordre de grandeur de la base
# de donnes elle-même, pas de quoi surprendre une sauvegarde `VACUUM INTO`.
SIM_CACHE_MAX_ROWS = int(os.environ.get("COLVER_SIM_CACHE_MAX", "20000"))


async def get_sim_cache(kind, cache_key, version):
    """L'entrée en cache pour cette clé **à cette version**, ou None.

    Marque le service à chaque fois : `used_at` alimente l'éviction LRU, `hits`
    est publié sur /health. Oui, c'est une écriture par lecture sur une base à
    connexion unique — mais elle vaut une ligne, contre les 11 s de CPU que la
    lecture vient d'économiser, et un compteur amorti à la journée compterait
    des *jours* et non des services : une métrique qui ment coûte plus cher que
    l'UPDATE qu'elle évite.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT data FROM analysis_cache "
        "WHERE kind = ? AND cache_key = ? AND version = ?",
        (kind, cache_key, version),
    )
    if not rows:
        return None
    try:
        blob = json.loads(rows[0][0])
    except json.JSONDecodeError:
        logger.warning("cache d'analyse illisible : %s/%s", kind, cache_key)
        return None
    await db.execute(
        "UPDATE analysis_cache SET used_at = ?, hits = hits + 1 "
        "WHERE kind = ? AND cache_key = ?",
        (_now(), kind, cache_key),
    )
    await db.commit()
    return blob


async def put_sim_cache(kind, cache_key, version, data_json):
    """Écrire une entrée, puis rendre la table à son plafond.

    L'éviction est faite ici plutôt que par une tâche de fond : elle ne touche
    que les lignes en trop, donc elle ne coûte rien tant que le plafond n'est
    pas atteint, et elle ne peut pas être oubliée au démarrage.
    """
    db = await get_db()
    now = _now()
    await db.execute(
        "INSERT OR REPLACE INTO analysis_cache "
        "(kind, cache_key, version, created_at, used_at, hits, data) "
        "VALUES (?, ?, ?, ?, ?, 0, ?)",
        (kind, cache_key, version, now, now, data_json),
    )
    await db.execute(
        "DELETE FROM analysis_cache WHERE rowid IN ("
        "  SELECT rowid FROM analysis_cache ORDER BY used_at DESC LIMIT -1 OFFSET ?"
        ")",
        (SIM_CACHE_MAX_ROWS,),
    )
    await db.commit()


async def sim_cache_stats():
    """`{kind: {rows, hits}}` — de quoi publier le cache sur /health."""
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT kind, COUNT(*), COALESCE(SUM(hits), 0) FROM analysis_cache GROUP BY kind")
    return {r[0]: {"rows": r[1], "hits": r[2]} for r in rows}


# ===== Progrès sur les exercices (v19) =====

async def exercise_stats(user_id, exercise):
    """`{variant: {plays, exact, sumAbsDelta, streak, best}}` pour ce joueur.

    Les clés sortent en camelCase parce que c'est la forme que le client avait
    déjà en localStorage : le rendu doit pouvoir lire l'une ou l'autre sans
    savoir laquelle il tient.
    """
    db = await get_db()
    rows = await db.execute_fetchall(
        "SELECT variant, plays, exact, sum_abs_delta, streak, best "
        "FROM exercise_stats WHERE user_id = ? AND exercise = ?",
        (user_id, exercise),
    )
    return {r[0]: {"plays": r[1], "exact": r[2], "sumAbsDelta": r[3],
                   "streak": r[4], "best": r[5]} for r in rows}


async def record_exercise_attempt(user_id, exercise, variant, *, delta, exact, streak):
    """Enregistrer **un** essai. Incrémental, jamais un total reçu du client.

    `best` est recalculé ici par un max plutôt que fourni : un record est la
    seule valeur de cette table qu'un client aurait intérêt à s'attribuer, et
    c'est aussi celle qu'un second onglet écraserait en poussant la sienne.
    """
    db = await get_db()
    await db.execute(
        """
        INSERT INTO exercise_stats
            (user_id, exercise, variant, plays, exact, sum_abs_delta,
             streak, best, updated_at)
        VALUES (?, ?, ?, 1, ?, ?, ?, ?, ?)
        ON CONFLICT (user_id, exercise, variant) DO UPDATE SET
            plays         = plays + 1,
            exact         = exact + excluded.exact,
            sum_abs_delta = sum_abs_delta + excluded.sum_abs_delta,
            streak        = excluded.streak,
            best          = MAX(best, excluded.streak),
            updated_at    = excluded.updated_at
        """,
        (user_id, exercise, variant, 1 if exact else 0, int(delta),
         int(streak), int(streak), _now()),
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
