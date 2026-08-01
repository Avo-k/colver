"""FastAPI server for Colver web UI."""

import asyncio
import json
import logging
import os
import re
import time
from datetime import datetime, timezone
from html import escape as _html_escape
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse, PlainTextResponse, Response

from colver.web.game_manager import PlaySession, WatchSession, ReplaySession, BidProblemSession, PlayProblemSession, BeliefSession, only_pass_is_legal, in_last_trick, cards_in_trick, trick_snapshot
from colver.web import playgen_gpu as _playgen_gpu
from colver.web import game_notation
import colver.web.card_analysis as _card_analysis
import colver.web.database as db
import colver.web.elo as elo
import colver.web.rooms as rooms
import colver.web.pacing as pacing
import colver.web.match_state as match_state
import colver.web.ratelimit as ratelimit
from colver.web.auth import router as auth_router, user_from_cookies

# Base path for reverse proxy deployment (e.g. ROOT_PATH=/colver/)
ROOT_PATH = os.environ.get("ROOT_PATH", "/")
if not ROOT_PATH.endswith("/"):
    ROOT_PATH += "/"

# Journalisation applicative. `basicConfig` ne touche que le logger racine et
# ne fait rien s'il est déjà configuré ; uvicorn garde ses propres loggers
# (`uvicorn`, `uvicorn.access`, propagate=False), donc rien n'est écrit deux
# fois. L'access log uvicorn tient lieu de journal de requêtes — pas de
# middleware à nous. Niveaux valides : DEBUG / INFO / WARNING / ERROR.
_log_level = (os.environ.get("COLVER_LOG_LEVEL") or "INFO").upper()
if _log_level not in ("DEBUG", "INFO", "WARNING", "ERROR"):
    _log_level = "INFO"  # une coquille d'env ne doit pas empêcher le démarrage
logging.basicConfig(
    level=_log_level,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
if _log_level != (os.environ.get("COLVER_LOG_LEVEL") or "INFO").upper():
    logger.warning("COLVER_LOG_LEVEL=%r invalide — repli sur INFO",
                   os.environ.get("COLVER_LOG_LEVEL"))

# Sentry optionnel : SENTRY_DSN non vide + sentry-sdk importable. Il n'est
# volontairement pas dans les dépendances — l'installer suffit à l'activer.
# L'observabilité ne doit jamais coûter la disponibilité : un DSN malformé
# (BadDsn) dégrade en warning, comme le SDK absent.
if os.environ.get("SENTRY_DSN"):
    try:
        import sentry_sdk

        sentry_sdk.init(dsn=os.environ["SENTRY_DSN"])
        logger.info("Sentry actif")
    except ImportError:
        logger.warning("SENTRY_DSN défini mais sentry-sdk non installé — remontée d'erreurs désactivée")
    except Exception:
        logger.exception("SENTRY_DSN invalide — remontée d'erreurs désactivée")

# Instant de démarrage, pour l'uptime de /health.
_START_TIME = time.monotonic()

app = FastAPI(title="Colver")

# Locate static assets bundled in the package
_WEB_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(_WEB_DIR, "static")
CARDS_DIR = os.path.join(_WEB_DIR, "cards")

# DouDou50 model path (Rust inference, no PyTorch needed)
import colver as _colver_pkg

_model = _colver_pkg.model_path()
if _model is None:
    # Auto-download model if not found
    logger.info("No DouDou50 model found, downloading...")
    try:
        _model = _colver_pkg.download_model()
    except Exception as e:
        logger.warning("Download failed: %s", e)
        _model = None

DMC_MODEL_PATH = str(_model) if _model else None
doudou_available = _model is not None
if doudou_available:
    logger.info("DouDou50 model available at %s (Rust inference)", DMC_MODEL_PATH)
else:
    logger.warning("No DouDou50 model found and download failed")

# Bid NN model path
_bid_model = _colver_pkg.bid_model_path()
if _bid_model is None:
    try:
        _bid_model = _colver_pkg.download_bid_model()
    except Exception as e:
        logger.warning("Bid model download failed: %s", e)
        _bid_model = None

BID_MODEL_PATH = str(_bid_model) if _bid_model else None
if _bid_model:
    logger.info("Bid model available at %s", BID_MODEL_PATH)
else:
    logger.warning("No bid model found, using improved_v2 fallback")

# Belief net model path (NN-based card location prediction for IS-DD)
_belief_model = _colver_pkg.belief_model_path()
if _belief_model is None:
    try:
        _belief_model = _colver_pkg.download_belief_model()
    except Exception as e:
        logger.warning("Belief model download failed: %s", e)
        _belief_model = None

BELIEF_MODEL_PATH = str(_belief_model) if _belief_model else None
if _belief_model:
    logger.info("Belief net model available at %s", BELIEF_MODEL_PATH)
else:
    logger.warning("No belief net model found, using heuristic beliefs")

# Playgen world-sampler model (transformer, MC belief marginals)
_playgen_model = _colver_pkg.playgen_model_path()
if _playgen_model is None:
    try:
        _playgen_model = _colver_pkg.download_playgen_model()
    except Exception as e:
        logger.warning("Playgen model download failed: %s", e)
        _playgen_model = None

PLAYGEN_MODEL_PATH = str(_playgen_model) if _playgen_model else None
if _playgen_model:
    logger.info("Playgen model available at %s", PLAYGEN_MODEL_PATH)
else:
    logger.warning("No playgen model found, playgen beliefs disabled")

logger.info("ROOT_PATH=%s", ROOT_PATH)

# URL publique du site — URL canoniques, Open Graph, sitemap. Derrière un
# reverse proxy avec préfixe (ROOT_PATH), inclure le préfixe dans la valeur.
PUBLIC_URL = os.environ.get("COLVER_PUBLIC_URL", "https://colver.net").rstrip("/")

# ===== Métadonnées SEO par route =====
# Le catch-all sert le même index.html pour toutes les routes du SPA : sans
# injection côté serveur, tout le site partage un seul <title> et un lien
# partagé s'affiche vide. Table alignée sur static/js/router.js.
_SUIT_SYMBOLS = "♠♥♦♣"  # ordre moteur : 0=♠ 1=♥ 2=♦ 3=♣ (cf. shared/suits.js)
_TEAM_NAMES = ["Nord-Sud", "Est-Ouest"]

_ROUTE_META = {
    "/": {
        "title": "Colver — Belote contrée en ligne contre des IA",
        "description": "Jouez à la belote contrée en ligne contre des IA fortes, "
                       "en solo ou en salon entre amis. Entraînez vos annonces et "
                       "analysez vos donnes carte par carte.",
    },
    "/jouer/humain": {
        "title": "Jouer contre l'IA — Colver",
        "description": "Une table de belote contrée en solo : trois sièges tenus "
                       "par l'IA, parties en une donne, 1000 ou 2000 points.",
    },
    "/jouer/salon": {
        "title": "Salon multijoueur — Colver",
        "description": "Créez un salon de belote contrée et invitez vos amis avec "
                       "un code à quatre lettres — les sièges vides sont tenus par l'IA.",
    },
    "/jouer/ia": {
        "title": "Regarder l'IA jouer — Colver",
        "description": "Observez des parties de belote contrée entre IA, avec les "
                       "statistiques du moteur en temps réel.",
    },
    "/analyse/rejouer": {
        "title": "Rejouer une partie — Colver",
        "description": "Rejouez une donne de belote contrée coup par coup, avec "
                       "l'avis de l'Oracle et des IA sur chaque carte.",
    },
    "/analyse/annonces": {
        "title": "Analyser une annonce — Colver",
        "description": "Composez une main et évaluez chaque annonce : réseau "
                       "d'enchères, jeu parfait et simulations de donnes.",
    },
    "/analyse/jeu": {
        "title": "Analyser le jeu de la carte — Colver",
        "description": "Comparez les cartes jouables d'une position de belote "
                       "contrée : valeur exacte à l'Oracle et taux de réussite simulé.",
    },
    "/analyse/croyances": {
        "title": "Croyances de l'IA — Colver",
        "description": "Visualisez ce que l'IA déduit des mains cachées au fil "
                       "d'une donne de belote contrée.",
    },
    "/problemes/annonce": {
        "title": "Problèmes d'annonce — Colver",
        "description": "Entraînez vos enchères à la belote contrée sur des donnes "
                       "générées, avec la correction de l'IA.",
    },
    "/problemes/jeu": {
        "title": "Problèmes de jeu — Colver",
        "description": "Entraînez votre jeu de la carte : trouvez la meilleure "
                       "carte de la position, l'Oracle corrige.",
    },
    "/aide": {
        "title": "Aide-mémoire de la belote contrée — Colver",
        "description": "Les règles essentielles de la belote contrée : ordre et "
                       "valeur des cartes, enchères, coinche, belote.",
    },
    "/annoncer": {
        "title": "Guide des annonces — Colver",
        "description": "Que demander avec sa main : des repères simples pour "
                       "annoncer juste à la belote contrée.",
    },
    "/score": {
        "title": "Marquer les points — Colver",
        "description": "Compter et marquer les points à la belote contrée : "
                       "contrats, coinche, surcoinche, belote et capot.",
    },
    "/about": {
        "title": "À propos — Colver",
        "description": "Colver : un moteur de belote contrée open source et des "
                       "IA entraînées par apprentissage par renforcement.",
    },
    "/compte": {
        "title": "Mon compte — Colver",
        "description": "Vos parties terminées, vos parties en cours et votre "
                       "classement.",
    },
    "/classement": {
        "title": "Classement — Colver",
        "description": "Le classement Elo des joueurs et des IA de Colver, mis à "
                       "jour à chaque donne.",
    },
}

# Pages publiques de contenu : ni /compte (espace personnel), ni /analyse/jeu
# (vide tant qu'on n'y colle pas une position).
_SITEMAP_ROUTES = [
    "/", "/jouer/humain", "/jouer/salon", "/jouer/ia",
    "/analyse/rejouer", "/analyse/annonces", "/analyse/croyances",
    "/problemes/annonce", "/problemes/jeu",
    "/aide", "/annoncer", "/score", "/classement", "/about",
]


# L'injection SEO entière repose sur ce remplacement de chaîne : si le <title>
# d'index.html change d'un octet, tout devient un no-op silencieux. Le contrôle
# au démarrage transforme cette mort silencieuse en erreur visible.
_TITLE_ANCHOR = "<title>Colver - Belote Contree</title>"
try:
    with open(os.path.join(FRONTEND_DIR, "index.html")) as _f:
        if _TITLE_ANCHOR not in _f.read():
            logger.error(
                "index.html ne contient plus l'ancre %r — l'injection SEO "
                "(title/description/OG) est morte, remettre le <title> en phase",
                _TITLE_ANCHOR)
except OSError:
    pass


def _route_meta(path):
    """Métadonnées d'une route du SPA. Route inconnue → celles de l'accueil,
    avec la canonique forcée sur `/` (le routeur client y redirige aussi)."""
    meta = _ROUTE_META.get(path)
    if meta is None:
        meta = dict(_ROUTE_META["/"], canonical=PUBLIC_URL + "/")
    return meta


def _serve_index(path="/", meta=None):
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(html_path) as f:
        html = f.read()
    # Inject the correct base href for reverse proxy support
    html = html.replace('<base href="/">', f'<base href="{ROOT_PATH}">')
    # Le <title> du fichier sert d'ancre : il est remplacé par le bloc complet
    # (title, description, canonique, Open Graph). Tout est échappé — les
    # métadonnées d'une donne partagée viennent de la base.
    meta = meta or _route_meta(path)
    title = _html_escape(meta["title"])
    desc = _html_escape(meta["description"])
    canon = _html_escape(meta.get("canonical") or PUBLIC_URL + path)
    html = html.replace(
        _TITLE_ANCHOR,
        f"<title>{title}</title>\n"
        f'    <meta name="description" content="{desc}">\n'
        f'    <link rel="canonical" href="{canon}">\n'
        f'    <meta property="og:title" content="{title}">\n'
        f'    <meta property="og:description" content="{desc}">\n'
        f'    <meta property="og:type" content="website">\n'
        f'    <meta property="og:url" content="{canon}">\n'
        f'    <meta property="og:image" content="{PUBLIC_URL}/static/og-image.png">\n'
        f'    <meta property="og:image:width" content="1024">\n'
        f'    <meta property="og:image:height" content="1024">\n'
        f'    <meta property="og:site_name" content="Colver">',
    )
    return HTMLResponse(html)


async def _replay_meta(game_id):
    """Métadonnées OG d'une donne partagée, ou None → celles de la route.

    Le contrat stocké ne dit pas si la donne est réussie : `games.points_ns/ew`
    sont les points *cartes* et la belote n'est pas en base. On rejoue donc les
    actions (même idiome que `elo._replay_rewards`) — quelques microsecondes
    côté Rust. Donne inconnue, incomplète ou passée → None.
    """
    try:
        game = await db.get_game(game_id)
    except Exception:
        return None
    if not game or not game["is_complete"]:
        return None
    contract = game.get("contract")
    if not contract or not contract.get("value"):
        return None  # donne passée (4 passes)
    try:
        env = _colver_pkg.Env.deal_with_hands(game["dealer"], game["hands"])
        for entry in game["actions"]:
            if env.is_terminal():
                break
            env.step(int(entry["action"]))
        if not env.is_terminal():
            return None
        rewards = list(env.rewards())
        points = list(env.get_points())
        belote = list(env.get_belote())
    except Exception:
        return None

    value = contract["value"]
    taker = contract.get("team", 0)
    made = rewards[taker] > 0  # même idiome que `contract_made` (game_manager)
    attack = points[taker] + (20 if belote[taker] == 2 else 0)
    sym = _SUIT_SYMBOLS[contract.get("trump", 0)]
    # « 160♥ coinchée chutée » mais « Capot ♥ coinché chuté » : le capot est
    # masculin, l'annonce en points est féminine.
    fem = value != 250
    e = "e" if fem else ""
    label = f"{value}{sym}" if fem else f"Capot {sym}"
    coinche = {1: f" coinché{e}", 2: f" surcoinché{e}"}.get(contract.get("coinche", 0), "")
    # Un capot chute sur les plis, pas sur les points : « chuté de N » n'a de
    # sens que pour un contrat en points.
    if made:
        outcome = f"réussi{e}"
    elif fem:
        outcome = f"chuté{e} de {max(1, value - attack)}"
    else:
        outcome = "chuté"
    verb = "réussit" if made else "chute"
    description = (
        f"{_TEAM_NAMES[taker]} joue {label}{coinche} et {verb} "
        f"({points[taker]} points cartes contre {points[1 - taker]}). "
        f"Rejouez la donne carte par carte, avec l'avis de l'Oracle "
        f"et des IA de Colver."
    )
    return {
        "title": f"{label}{coinche} {outcome} — rejouez la donne sur Colver",
        "description": description,
        # `i` (position dans la donne) est volontairement absent : toutes les
        # positions d'une donne se replient sur la même page canonique.
        "canonical": f"{PUBLIC_URL}/analyse/rejouer?game={game['id']}",
    }


@app.get("/")
async def index():
    return _serve_index("/")


app.mount("/cards", StaticFiles(directory=CARDS_DIR), name="cards")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")

app.include_router(auth_router)

# Cache policy. Without explicit Cache-Control, browsers use heuristic
# caching (and Cloudflare caches js/css by extension), so after a deploy
# clients keep stale ES modules and imports break. HTML/JS/CSS are marked
# no-cache: always revalidated via ETag/Last-Modified — a 304 unless the
# file changed. Heavy, stable assets (cards, wasm, models, images) get a
# long shared cache with background revalidation.
_LONG_CACHE_EXT = (".svg", ".png", ".ico", ".wasm", ".bin", ".json",
                   ".woff", ".woff2", ".mp3", ".webp", ".jpg")


@app.middleware("http")
async def cache_control(request: Request, call_next):
    response = await call_next(request)
    if request.method != "GET" or "cache-control" in response.headers:
        return response
    path = request.url.path
    if "/cards/" in path or path.endswith(_LONG_CACHE_EXT):
        response.headers["Cache-Control"] = (
            "public, max-age=3600, stale-while-revalidate=86400")
    else:
        # HTML shell, JS modules, CSS, API responses
        response.headers["Cache-Control"] = "no-cache"
    return response


# ===== REST API =====

@app.get("/health")
async def health():
    """État du service : base, version, uptime, modèles.

    Déclarée ici, au niveau module — donc avant le catch-all SPA de fin de
    fichier : l'ordre d'enregistrement des routes fait foi.
    """
    db_ok = True
    try:
        conn = await db.get_db()
        cur = await conn.execute("SELECT 1")
        await cur.fetchone()
    except Exception:
        logger.exception("health : SELECT 1 en échec")
        db_ok = False
    payload = {
        "status": "ok" if db_ok else "degraded",
        "db": db_ok,
        "version": _colver_pkg.__version__,
        "uptime_s": round(time.monotonic() - _START_TIME, 1),
        "models": {
            "doudou": doudou_available,
            "bid": BID_MODEL_PATH is not None,
            "belief": BELIEF_MODEL_PATH is not None,
            "playgen": PLAYGEN_MODEL_PATH is not None,
            # URL du sidecar configurée — pas une sonde : /health doit rester
            # instantané.
            "sidecar_configured": _playgen_gpu.enabled(),
        },
    }
    return JSONResponse(payload, status_code=200 if db_ok else 503)


@app.get("/api/games")
async def api_list_games(limit: int = 50, offset: int = 0):
    games = await db.list_games(limit=min(limit, 200), offset=offset)
    return JSONResponse(games)


@app.get("/api/games/{game_id}")
async def api_get_game(game_id: str):
    game = await db.get_game(game_id)
    if not game:
        return JSONResponse({"error": "Game not found"}, status_code=404)
    return JSONResponse(game)


@app.on_event("startup")
async def _elo_backfill():
    _spawn(elo.backfill())


# ===== Sauvegarde périodique de la base =====

# `colver.db` n'existe qu'en un exemplaire (comptes, historique, Elo, analyses
# en cache) : un VACUUM INTO périodique en copie un instantané cohérent hors
# du volume. COLVER_BACKUP_DIR vide ou "0" → désactivé ; non défini → un
# sous-répertoire backups/ à côté du fichier .db (en Docker, le compose le
# pointe hors du volume de la base — c'est tout l'intérêt).
_backup_env = os.environ.get("COLVER_BACKUP_DIR")
if _backup_env in ("", "0"):
    BACKUP_DIR = None
else:
    BACKUP_DIR = _backup_env or os.path.join(
        os.path.dirname(db.DB_PATH), "backups")
BACKUP_INTERVAL_H = float(os.environ.get("COLVER_BACKUP_INTERVAL_H", "24"))
BACKUP_KEEP = int(os.environ.get("COLVER_BACKUP_KEEP", "14"))


def _backup_due_delay():
    """Secondes avant le prochain backup dû, d'après la copie la plus récente.

    Sans ce délai, chaque redémarrage écrirait un instantané : une rafale de
    déploiements (ou un crash-loop sous `restart: unless-stopped`) remplacerait
    les 14 sauvegardes retenues par autant de copies de la même heure — la
    fenêtre de rétention passerait de 14 jours à quelques minutes, précisément
    le jour où on casse quelque chose.
    """
    try:
        stamps = [
            datetime.strptime(p.name, "colver-%Y%m%d-%H%M%S.db")
            for p in Path(BACKUP_DIR).glob("colver-*.db")
            if re.fullmatch(r"colver-\d{8}-\d{6}\.db", p.name)
        ]
        if not stamps:
            return 0.0
        age = datetime.now(timezone.utc) - max(stamps).replace(tzinfo=timezone.utc)
        return max(0.0, BACKUP_INTERVAL_H * 3600 - age.total_seconds())
    except Exception:
        return 0.0


async def _backup_loop():
    delay = await asyncio.to_thread(_backup_due_delay)
    if delay > 0:
        logger.info("Dernier backup récent, prochain dans %.1f h", delay / 3600)
        await asyncio.sleep(delay)
    while True:
        try:
            # Les migrations d'abord : `get_db()` ne rend la main qu'une fois
            # la base migrée (verrou `_db_lock`), sinon le backup peut capturer
            # un état mi-migration impossible à restaurer proprement. Dans le
            # try : une base indisponible au boot ne doit pas tuer la tâche
            # pour toujours, juste faire rater ce tour-ci.
            await db.get_db()
            path = await asyncio.to_thread(
                db.backup_db, BACKUP_DIR, BACKUP_KEEP)
            logger.info("Database backed up to %s", path)
        except Exception:
            logger.exception("Database backup failed")
        await asyncio.sleep(BACKUP_INTERVAL_H * 3600)


# L'event loop ne tient que des références faibles sur les tâches : sans ce
# set, une tâche de fond peut être ramassée en plein vol.
_BG_TASKS = set()


def _spawn(coro):
    t = asyncio.create_task(coro)
    _BG_TASKS.add(t)
    t.add_done_callback(_BG_TASKS.discard)
    return t


@app.on_event("startup")
async def _db_backup():
    if BACKUP_DIR and BACKUP_INTERVAL_H > 0:
        _spawn(_backup_loop())


@app.get("/api/leaderboard")
async def api_leaderboard():
    return JSONResponse(await elo.leaderboard())


@app.get("/api/games/{game_id}/analysis")
async def api_game_analysis(game_id: str):
    import colver.web.analysis as analysis
    result, err = await analysis.get_or_compute(
        game_id,
        bid_model_path=BID_MODEL_PATH,
        playgen_model_path=PLAYGEN_MODEL_PATH,
    )
    if err:
        return JSONResponse({"error": err}, status_code=404)
    return JSONResponse(result)


@app.get("/api/games/{game_id}/agents")
async def api_game_agent_review(game_id: str):
    """What DouDou50 / Oracle / IS-DD would have played at every card."""
    import colver.web.agent_review as agent_review
    result, err = await agent_review.get_or_compute(
        game_id,
        play_model=DMC_MODEL_PATH if doudou_available else None,
        belief_model=BELIEF_MODEL_PATH,
    )
    if err:
        return JSONResponse({"error": err}, status_code=404)
    return JSONResponse(result)


@app.post("/api/games/{game_id}/report")
async def api_bug_report(game_id: str, request: Request):
    body = await request.json()
    message = body.get("message", "").strip()
    if not message:
        return JSONResponse({"error": "Message required"}, status_code=400)
    # Un bug se signale aussi en pleine donne (le bouton vit sur la table de
    # jeu) : on accepte les donnes en cours — rien de la donne n'est renvoyé,
    # l'appel ne vérifie que l'existence de l'identifiant.
    game = await db.get_game(game_id, include_incomplete=True)
    if not game:
        return JSONResponse({"error": "Game not found"}, status_code=404)
    action_idx = body.get("action_idx")
    user_agent = request.headers.get("user-agent")
    await db.create_bug_report(game_id, action_idx, message, user_agent)
    return JSONResponse({"ok": True})


# ===== Annonces simulation tasks (cancellable) =====

# Dedicated pool for oracle DD solves — solve_all_suits releases the GIL, so
# these actually run in parallel (one solve ≈ 300ms; 8 workers ≈ 8× wall-clock).
import concurrent.futures as _cf
import threading as _threading

_DD_EXECUTOR = _cf.ThreadPoolExecutor(
    max_workers=min(16, os.cpu_count() or 4), thread_name_prefix="dd-solve")

# Per-thread Env cache for Dédé sims: model loading (~10MB from disk) happens
# once per worker thread instead of once per simulated world.
_doudou_tls = _threading.local()


def _get_doudou_env(bid_model_path, dmc_model_path, dealer, hands):
    key = (bid_model_path, dmc_model_path)
    env = getattr(_doudou_tls, "env", None)
    if env is not None and getattr(_doudou_tls, "key", None) == key:
        env.redeal_with_hands(dealer, hands)
        return env
    env = _colver_pkg.Env.deal_with_hands(dealer, hands)
    env.load_bid_model(bid_model_path)
    env.load_dmc_model(dmc_model_path)
    _doudou_tls.env = env
    _doudou_tls.key = key
    return env


def _bid_action_suit(action):
    """Suit of a bid action (1-36 value bids, 37-40 capots), else None."""
    if 1 <= action <= 36:
        return (action - 1) % 4
    if 37 <= action <= 40:
        return action - 37
    return None


def _doudou_new_cells():
    # cells[suit][col] = [ns_count, ns_achieved, ew_count, ew_achieved]
    return [[[0, 0, 0, 0] for _ in range(10)] for _ in range(4)]


def _doudou_new_stats():
    return {
        "voids": 0,
        "ns_contracts": 0, "ns_achieved": 0,
        "ew_contracts": 0, "ew_achieved": 0,
        "taker_seats": [0, 0, 0, 0],        # N, E, S, W
        "trump_counts": [0, 0, 0, 0],       # S, H, D, C
        "coinche": 0, "coinche_achieved": 0, "surcoinche": 0,
        "south_bids": 0,                    # deals where South made a suit bid
        "partner_support": 0,               # Nord's next action: same-suit bid
        "partner_other": 0,                 # Nord's next action: other-suit bid
        "partner_pass": 0,                  # Nord's next action: pass/coinche
        "opp_overbid": 0,                   # deals where E/W bid over South's bid
        "ns_value_sum": 0,                  # for avg NS contract value
        "pts_ns_sum": 0.0, "pts_ew_sum": 0.0, "pts_n": 0,
        # Issue de la donne, tous contrats confondus (donne passée = nulle).
        # Dénominateur = wins_ns + wins_ew + draws = nombre de sims terminées.
        "deal_wins_ns": 0, "deal_wins_ew": 0, "deal_draws": 0,
    }


def _doudou_accumulate(cells, stats, dd):
    """Fold one Dédé sim result into the aggregates."""
    if dd["void"]:
        stats["voids"] += 1
        stats["deal_draws"] += 1  # personne ne marque : la donne est nulle
        return
    suit, value, team, achieved = dd["trump"], dd["value"], dd["team"], dd["achieved"]
    col = 9 if value == 250 else (value - 80) // 10
    if 0 <= col <= 9 and 0 <= suit <= 3:
        base = 0 if team == 0 else 2
        cells[suit][col][base] += 1
        if achieved:
            cells[suit][col][base + 1] += 1
    key = "ns" if team == 0 else "ew"
    stats[f"{key}_contracts"] += 1
    if achieved:
        stats[f"{key}_achieved"] += 1

    stats["trump_counts"][suit] += 1
    coinche = dd.get("coinche", 0)
    if coinche >= 1:
        stats["coinche"] += 1
        if achieved:
            stats["coinche_achieved"] += 1
    if coinche == 2:
        stats["surcoinche"] += 1
    if team == 0:
        stats["ns_value_sum"] += value

    scores = dd.get("scores")
    if scores:
        stats["pts_ns_sum"] += scores[0]
        stats["pts_ew_sum"] += scores[1]
        stats["pts_n"] += 1
        if scores[0] > scores[1]:
            stats["deal_wins_ns"] += 1
        elif scores[1] > scores[0]:
            stats["deal_wins_ew"] += 1
        else:
            stats["deal_draws"] += 1

    auction = dd.get("auction") or []
    taker_seat = None
    for s, a in auction:
        if _bid_action_suit(a) is not None:
            taker_seat = s
    if taker_seat is not None:
        stats["taker_seats"][taker_seat] += 1

    # South's first suit bid → partner reaction + adverse overbid
    s_idx = next((i for i, (s, a) in enumerate(auction)
                  if s == 2 and _bid_action_suit(a) is not None), None)
    if s_idx is not None:
        stats["south_bids"] += 1
        south_suit = _bid_action_suit(auction[s_idx][1])
        partner_done = False
        overbid = False
        for s, a in auction[s_idx + 1:]:
            a_suit = _bid_action_suit(a)
            if s == 0 and not partner_done:
                partner_done = True
                if a_suit is None:
                    stats["partner_pass"] += 1
                elif a_suit == south_suit:
                    stats["partner_support"] += 1
                else:
                    stats["partner_other"] += 1
            if s in (1, 3) and a_suit is not None:
                overbid = True
        if overbid:
            stats["opp_overbid"] += 1


async def _run_annonces_sim(ws: WebSocket, data: dict):
    """Oracle DD + Dédé simulation, runs as a background task."""
    import time as _time
    import random as _random

    hand = data.get("hand", [])
    # Les deux tableaux ne tirent plus le même nombre de donnes : un solve
    # double-dummy coûte ~50× une donne jouée. Le pool de mondes est commun —
    # l'Oracle résout les `oracle_sims` premiers, Dédé les joue tous — pour que
    # les deux tableaux parlent bien du même échantillon.
    num_sims = max(1, min(1000, int(data.get("num_sims", 50))))
    oracle_sims = max(1, min(1000, int(data.get("oracle_sims", num_sims))))
    doudou_sims = max(1, min(2000, int(data.get("doudou_sims", num_sims))))
    world_total = max(oracle_sims, doudou_sims)
    prior_actions_raw = data.get("prior_actions", None)
    prior_actions = [int(a) for a in prior_actions_raw] if prior_actions_raw else []
    # Identifiant d'onglet côté client : renvoyé tel quel dans chaque message
    # pour que le flux atterrisse dans l'onglet qui l'a demandé, même si une
    # analyse plus récente a déjà annulé celle-ci.
    req_id = data.get("req_id")

    if len(hand) != 8:
        await ws.send_json({"type": "annonces_sim_update", "req_id": req_id,
                            "error": "8 cartes requises"})
        return

    gen_task = None  # génération de mondes en cours, à annuler en sortie
    try:
        loop = asyncio.get_event_loop()
        remaining = list(set(range(32)) - set(hand))
        seat = 2
        dealer = (seat - 1 - len(prior_actions) + 32) % 4
        THRESHOLDS = [80, 90, 100, 110, 120, 130, 140, 150, 160, 162]

        others = [s for s in range(4) if s != seat]

        def _uniform_hands():
            r = list(remaining)
            _random.shuffle(r)
            h = [None] * 4
            h[seat] = sorted(hand)
            for j, p in enumerate(others):
                h[p] = sorted(r[j * 8:(j + 1) * 8])
            return h

        # Worlds: playgen v2 (conditioned on the current auction) when
        # available, uniform otherwise. Generation is slow (~0.4s/deal) so it
        # runs chunk-wise in an executor, overlapped with the DD solves below.
        playgen_env = None
        playgen_analyst = None
        worlds_source = "uniform"
        # (player, action) pairs of the auction prefix, for the GPU sidecar.
        gpu_prior_pairs = []
        gpu_deal_hands = None
        if PLAYGEN_MODEL_PATH:
            def _mk_playgen():
                nonlocal gpu_deal_hands, gpu_prior_pairs
                gpu_deal_hands = _uniform_hands()
                e = _colver_pkg.Env.deal_with_hands(dealer, gpu_deal_hands)
                gpu_prior_pairs = []
                for a in prior_actions:
                    gpu_prior_pairs.append((int(e.current_player()), int(a)))
                    e.step(a)
                analyst = _colver_pkg.Analyst.replay(
                    PLAYGEN_MODEL_PATH, dealer, gpu_deal_hands,
                    [int(a) for a in prior_actions], seat,
                )
                # Probe: v1 playgen weights cannot sample auctions (empty).
                return (e, analyst) if analyst.auction_deals(e, 1, 1.0) else (None, None)
            try:
                playgen_env, playgen_analyst = await loop.run_in_executor(None, _mk_playgen)
            except Exception:
                playgen_env, playgen_analyst = None, None
            if playgen_analyst is not None:
                worlds_source = "playgen"

        all_hands = []
        all_sources = []
        if playgen_env is None:
            for _ in range(world_total):
                all_hands.append(_uniform_hands())
                all_sources.append("uniform")

        # GPU sidecar: one batched call is far cheaper than chunked CPU calls
        # (shared prefill), so generate everything in a single chunk.
        _gpu = _playgen_gpu.enabled()
        PLAYGEN_CHUNK = world_total if _gpu else 8

        def _gen_chunk(n):
            if _gpu:
                deals = _playgen_gpu.auction_deals(
                    dealer, gpu_deal_hands, gpu_prior_pairs, seat, n, 1.0)
                if deals:
                    return deals
            deals = playgen_analyst.auction_deals(playgen_env, n, 1.0)
            return deals or []

        # Phase 1: Oracle (DD solves) — parallel sliding window on _DD_EXECUTOR
        # (solve_all_suits releases the GIL, so the solves genuinely overlap).
        success_counts = [[0] * len(THRESHOLDS) for _ in range(4)]
        oracle_ns_sums = [0, 0, 0, 0]
        oracle_ns_vals = [[], [], [], []]  # per-suit NS points, for medians
        oracle_best_counts = [0, 0, 0, 0]
        sampled_deals = []
        sampled_sources = []
        worlds_counts = {}
        oracle_start = _time.monotonic()

        def _oracle_synth():
            medians = []
            for vals in oracle_ns_vals:
                if not vals:
                    medians.append(0)
                    continue
                s = sorted(vals)
                mid = len(s) // 2
                medians.append(s[mid] if len(s) % 2 else (s[mid - 1] + s[mid]) / 2)
            return {"ns_sums": oracle_ns_sums, "ns_medians": medians,
                    "best_counts": oracle_best_counts}

        window = min(_DD_EXECUTOR._max_workers, oracle_sims)
        completed = 0
        next_i = 0
        pending = set()
        gen_requested = 0
        if playgen_env is not None:
            gen_requested = min(PLAYGEN_CHUNK, world_total)
            gen_task = loop.run_in_executor(None, _gen_chunk, gen_requested)

        def _uniform_topup():
            nonlocal worlds_source
            while len(all_hands) < world_total:
                all_hands.append(_uniform_hands())
                all_sources.append("uniform")
                worlds_source = "mixte"

        try:
            while completed < oracle_sims:
                if gen_task is not None and gen_task.done():
                    chunk = gen_task.result()
                    all_hands.extend(chunk)
                    all_sources.extend(["playgen"] * len(chunk))
                    if chunk and gen_requested < world_total:
                        n = min(PLAYGEN_CHUNK, world_total - gen_requested)
                        gen_requested += n
                        gen_task = loop.run_in_executor(None, _gen_chunk, n)
                    else:
                        gen_task = None
                        _uniform_topup()  # génération épuisée ou en échec

                while next_i < min(oracle_sims, len(all_hands)) and len(pending) < window:
                    fut = loop.run_in_executor(
                        _DD_EXECUTOR, _run_dd_sim_with_hands, all_hands[next_i], dealer)
                    fut.world_index = next_i
                    pending.add(fut)
                    next_i += 1

                wait_set = set(pending)
                if gen_task is not None:
                    wait_set.add(gen_task)
                if not wait_set:
                    break
                done, _ = await asyncio.wait(
                    wait_set, return_when=asyncio.FIRST_COMPLETED)
                for fut in done:
                    if fut is gen_task:
                        continue  # consumed at the top of the loop
                    pending.discard(fut)
                    result = fut.result()
                    for suit_idx, (ns, ew) in enumerate(result["suits"]):
                        oracle_ns_sums[suit_idx] += ns
                        oracle_ns_vals[suit_idx].append(ns)
                        for t_idx, thr in enumerate(THRESHOLDS):
                            if ns >= thr:
                                success_counts[suit_idx][t_idx] += 1
                    best = max(range(4), key=lambda s: result["suits"][s][0])
                    oracle_best_counts[best] += 1
                    sampled_deals.append(result["hands"])
                    src = all_sources[fut.world_index]
                    sampled_sources.append(src)
                    worlds_counts[src] = worlds_counts.get(src, 0) + 1
                    completed += 1

                await ws.send_json({
                    "type": "annonces_sim_update", "req_id": req_id,
                    "completed": completed, "total": oracle_sims,
                    "elapsed_ms": round((_time.monotonic() - oracle_start) * 1000, 1),
                    "success_counts": success_counts,
                    "oracle_synth": _oracle_synth(),
                    "worlds_source": worlds_source,
                    "worlds_counts": worlds_counts,
                })
        finally:
            for fut in pending:
                fut.cancel()

        await ws.send_json({
            "type": "annonces_sim_done", "req_id": req_id,
            "completed": completed, "total": oracle_sims,
            "elapsed_ms": round((_time.monotonic() - oracle_start) * 1000, 1),
            "success_counts": success_counts,
            "oracle_synth": _oracle_synth(),
            "sampled_deals": sampled_deals,
            "sampled_sources": sampled_sources,
            "worlds_source": worlds_source,
            "worlds_counts": worlds_counts,
        })

        # Phase 2: Dédé (NN bid + DMC play) — slow
        if BID_MODEL_PATH and DMC_MODEL_PATH:
            # L'Oracle s'arrête avant la fin de la génération quand il tire
            # moins de donnes ; on draine le reste du pool avant de jouer.
            while len(all_hands) < world_total:
                if gen_task is None:
                    if gen_requested >= world_total:
                        break
                    n = min(PLAYGEN_CHUNK, world_total - gen_requested)
                    gen_requested += n
                    gen_task = loop.run_in_executor(None, _gen_chunk, n)
                chunk = await gen_task
                gen_task = None
                all_hands.extend(chunk)
                all_sources.extend(["playgen"] * len(chunk))
                if not chunk:
                    break  # génération en échec : on complète en uniforme
            _uniform_topup()

            doudou_cells = _doudou_new_cells()
            doudou_stats = _doudou_new_stats()
            doudou_start = _time.monotonic()

            for i in range(doudou_sims):
                dd = await loop.run_in_executor(
                    None, _run_doudou_sim_with_hands, all_hands[i],
                    BID_MODEL_PATH, DMC_MODEL_PATH, dealer, prior_actions)

                _doudou_accumulate(doudou_cells, doudou_stats, dd)

                await ws.send_json({
                    "type": "annonces_doudou_update", "req_id": req_id,
                    "completed": i + 1, "total": doudou_sims,
                    "elapsed_ms": round((_time.monotonic() - doudou_start) * 1000, 1),
                    "doudou_cells": doudou_cells,
                    "doudou_stats": doudou_stats,
                })

            await ws.send_json({
                "type": "annonces_doudou_done", "req_id": req_id,
                "completed": doudou_sims, "total": doudou_sims,
                "elapsed_ms": round((_time.monotonic() - doudou_start) * 1000, 1),
                "doudou_cells": doudou_cells,
                "doudou_stats": doudou_stats,
            })

    except asyncio.CancelledError:
        return
    except Exception as e:
        # L'erreur part au client, mais elle doit aussi laisser une trace ici.
        logger.exception("annonces_sim : simulation en échec")
        try:
            await ws.send_json({"type": "annonces_sim_update", "req_id": req_id,
                                "error": str(e)})
        except Exception:
            pass
    finally:
        if gen_task is not None:
            gen_task.cancel()


async def _run_card_analysis(ws: WebSocket, data: dict):
    """Analyse d'une décision de carte, mondes échantillonnés en streaming.

    Un monde = un solve (qui couvre **toutes** les candidates d'un coup) plus,
    tant que le budget de déroulements n'est pas épuisé, un déroulement forcé
    par candidate. Les deux groupes de colonnes décrivent donc par construction
    le même échantillon : c'est le même monde qui alimente les deux, dans le
    même ordre, et le seul écart est la profondeur (`real_worlds` ≤
    `oracle_worlds`, un déroulement coûtant bien plus cher qu'un solve).
    """
    import time as _time

    req_id = data.get("req_id")

    def _err(msg, **extra):
        return {"type": "card_analysis_error", "req_id": req_id, "error": msg, **extra}

    cfn = (data.get("cfn") or "").strip()
    try:
        idx = int(data.get("idx", 0))
    except (TypeError, ValueError):
        idx = 0

    try:
        core, bid_actions = game_notation.parse_full_cfn(cfn)
        src = _colver_pkg.Env.from_cfn(core)
        dealer = int(src.get_dealer())
        initial_hands = [list(h) for h in src.get_initial_hands()]
        play_actions = [int(a) for a in src.get_play_order()]
        actions = [int(a) for a in bid_actions] + play_actions
    except Exception as e:
        await ws.send_json(_err(f"CFN illisible : {e}"))
        return

    # Un CFN cœur 3 sections ne porte pas l'enchère : en phase de jeu
    # `format_contract` n'émet que le contrat résolu. Rejouer les cartes sur un
    # état encore en phase d'annonces les ferait interpréter comme des annonces,
    # et l'analyse porterait sur une position qui n'a jamais existé. Un contrat
    # résolu demande au minimum une annonce et trois passes, donc une enchère
    # vide avec des cartes jouées est toujours une notation tronquée.
    if play_actions and not bid_actions:
        await ws.send_json(_err(
            "Ce CFN ne contient pas l'enchère (3 sections) : impossible de "
            "reconstruire la position. Copiez le CFN complet depuis Rejouer."))
        return

    if not 0 <= idx < len(actions):
        await ws.send_json(_err("Index hors de la partie"))
        return

    pos = _card_analysis.describe(dealer, initial_hands, actions, idx)
    if "error" in pos:
        await ws.send_json(_err(pos["error"], phase=pos.get("phase")))
        return

    budget = _card_analysis.plan(pos)
    seat = pos["seat"]
    team = seat % 2
    candidates = pos["candidates"]

    await ws.send_json({
        "type": "card_analysis_position", "req_id": req_id,
        "dealer": dealer, "idx": idx, "cfn": cfn,
        "position": pos, "plan": budget,
    })

    if pos["forced"]:
        return  # une seule carte jouable : rien à comparer

    try:
        loop = asyncio.get_event_loop()

        # Le vrai monde et les avis d'abord : c'est rapide et ça remplit déjà
        # deux colonnes pendant que les mondes s'échantillonnent.
        truth = await loop.run_in_executor(
            _DD_EXECUTOR, _card_analysis.true_world,
            dealer, initial_hands, actions, idx, candidates, seat)
        await ws.send_json({"type": "card_analysis_truth", "req_id": req_id,
                            "truth": truth})

        avis = await asyncio.to_thread(
            _card_analysis.opinions, dealer, initial_hands, actions, idx, seat,
            play_model=DMC_MODEL_PATH, belief_model=BELIEF_MODEL_PATH)
        await ws.send_json({"type": "card_analysis_opinions", "req_id": req_id,
                            "opinions": avis})

        n_worlds = budget["oracle_worlds"]
        worlds, worlds_source = await asyncio.to_thread(
            _card_analysis.sample_worlds, dealer, initial_hands, actions, idx,
            seat, n_worlds, playgen_model=PLAYGEN_MODEL_PATH)
        n_worlds = min(n_worlds, len(worlds))
        real_worlds = min(budget["real_worlds"], n_worlds)

        totals = _card_analysis.new_totals(candidates)
        start = _time.monotonic()
        window = min(_DD_EXECUTOR._max_workers, max(1, n_worlds))
        next_i = 0
        completed = 0
        pending = set()
        sample_hands = []

        try:
            while completed < n_worlds:
                while next_i < n_worlds and len(pending) < window:
                    fut = loop.run_in_executor(
                        _DD_EXECUTOR, _card_analysis.world_job,
                        dealer, actions, idx, pos["played"], worlds[next_i],
                        candidates, team,
                        DMC_MODEL_PATH, next_i < real_worlds)
                    pending.add(fut)
                    next_i += 1
                if not pending:
                    break
                done, _ = await asyncio.wait(
                    pending, return_when=asyncio.FIRST_COMPLETED)
                for fut in done:
                    pending.discard(fut)
                    job = fut.result()
                    _card_analysis.accumulate(totals, job, team)
                    if len(sample_hands) < 10:
                        sample_hands.append(job["hands"])
                    completed += 1

                await ws.send_json({
                    "type": "card_analysis_update", "req_id": req_id,
                    "completed": completed, "total": n_worlds,
                    "real_total": real_worlds,
                    "elapsed_ms": round((_time.monotonic() - start) * 1000, 1),
                    "rows": _card_analysis.summarize(totals, team),
                    "worlds_source": worlds_source,
                })
        finally:
            for fut in pending:
                fut.cancel()

        await ws.send_json({
            "type": "card_analysis_done", "req_id": req_id,
            "completed": completed, "total": n_worlds,
            "real_total": real_worlds,
            "elapsed_ms": round((_time.monotonic() - start) * 1000, 1),
            "rows": _card_analysis.summarize(totals, team),
            "worlds_source": worlds_source,
            "sample_hands": sample_hands,
        })

    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.exception("card_analysis : simulation en échec")
        try:
            await ws.send_json(_err(str(e)))
        except Exception:
            pass


async def _run_annonces_doudou(ws: WebSocket, data: dict):
    """Dédé-only simulation (used by local/WASM mode), runs as a background task."""
    import time as _time

    hand = data.get("hand", [])
    num_sims = max(1, min(1000, int(data.get("num_sims", 50))))
    prior_actions_raw = data.get("prior_actions", None)
    prior_actions = [int(a) for a in prior_actions_raw] if prior_actions_raw else []
    forced_action_raw = data.get("forced_action", None)
    forced_action = int(forced_action_raw) if forced_action_raw is not None else None
    # Cf. _run_annonces_sim : identifiant d'onglet renvoyé tel quel.
    req_id = data.get("req_id")

    if len(hand) != 8:
        await ws.send_json({"type": "annonces_doudou_update", "req_id": req_id,
                            "error": "8 cartes requises"})
        return
    if not BID_MODEL_PATH or not DMC_MODEL_PATH:
        await ws.send_json({"type": "annonces_doudou_update", "req_id": req_id,
                            "error": "Modèles Dédé non disponibles"})
        return

    try:
        loop = asyncio.get_event_loop()
        remaining = list(set(range(32)) - set(hand))
        seat = 2
        dealer = (seat - 1 - len(prior_actions) + 32) % 4
        start = _time.monotonic()

        # Bid legality depends only on the auction history, not on the deal —
        # validate the forced action once before launching the sims.
        if forced_action is not None:
            hands_check = [None] * 4
            hands_check[seat] = sorted(hand)
            others = [s for s in range(4) if s != seat]
            for j, p in enumerate(others):
                hands_check[p] = sorted(remaining[j * 8:(j + 1) * 8])
            env_check = _colver_pkg.Env.deal_with_hands(dealer, hands_check)
            for action in prior_actions:
                env_check.step(action)
            if env_check.phase() != 0 or forced_action not in env_check.legal_actions():
                await ws.send_json({"type": "annonces_doudou_update", "req_id": req_id,
                                    "error": "Annonce illégale dans cette situation"})
                return

        doudou_cells = _doudou_new_cells()
        doudou_stats = _doudou_new_stats()

        for i in range(num_sims):
            result = await loop.run_in_executor(
                None, _run_single_doudou_sim, hand, list(remaining),
                BID_MODEL_PATH, DMC_MODEL_PATH, dealer, prior_actions, forced_action)

            dd = result.get("doudou")
            if dd:
                _doudou_accumulate(doudou_cells, doudou_stats, dd)

            await ws.send_json({
                "type": "annonces_doudou_update", "req_id": req_id,
                "completed": i + 1, "total": num_sims,
                "elapsed_ms": round((_time.monotonic() - start) * 1000, 1),
                "doudou_cells": doudou_cells,
                "doudou_stats": doudou_stats,
            })

        await ws.send_json({
            "type": "annonces_doudou_done", "req_id": req_id,
            "completed": num_sims, "total": num_sims,
            "elapsed_ms": round((_time.monotonic() - start) * 1000, 1),
            "doudou_cells": doudou_cells,
            "doudou_stats": doudou_stats,
        })

    except asyncio.CancelledError:
        return
    except Exception as e:
        logger.exception("annonces_doudou : simulation en échec")
        try:
            await ws.send_json({"type": "annonces_doudou_update", "req_id": req_id,
                                "error": str(e)})
        except Exception:
            pass


# ===== WebSocket =====

# Plafond de connexions simultanées : chaque socket peut lancer des simulations
# coûteuses (une annonces_sim ≈ 200 solves DD sur _DD_EXECUTOR) et une seule
# tourne à la fois PAR socket — rien ne limitait le nombre de sockets qu'un
# client anonyme peut ouvrir. Par IP + global, configurables par env.
_WS_CAP = ratelimit.ConnectionCap(
    per_key=int(os.environ.get("COLVER_WS_PER_IP", "8")),
    total=int(os.environ.get("COLVER_WS_TOTAL", "200")),
)

# Le jour où il faudra distinguer une attaque d'un plafond mal calibré, c'est
# cette trace qu'on cherchera — une ligne par minute et par IP suffit.
_WS_REFUSAL_LOG = ratelimit.RateLimiter(limit=1, window=60.0)


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    ip = ws.client.host if ws.client else "?"
    if not _WS_CAP.acquire(ip):
        if _WS_REFUSAL_LOG.allow(ip):
            logger.warning("connexion WS refusée (plafond) pour %s", ip)
        try:
            # Accepter d'abord : un refus avant l'accept se traduit par un 403
            # opaque côté client, alors que 1013 (« try again later ») lui dit
            # proprement de réessayer plus tard.
            await ws.accept()
            await ws.close(code=1013)
        except Exception:
            # Client déjà reparti — un refus de routine, pas une erreur ASGI.
            pass
        return
    try:
        await _websocket_session(ws)
    finally:
        _WS_CAP.release(ip)


async def _websocket_session(ws: WebSocket):
    await ws.accept()
    ws_user = await user_from_cookies(ws.cookies)
    play_session = None
    watch_session = None
    replay_session = None
    bid_problem_session = None
    play_problem_session = None
    belief_session = None
    play_game_id = None
    watch_game_id = None
    play_mode = pacing.DEFAULT_MODE
    play_match = None   # match_state.Match — une donne (target 0) ou une partie
    play_cfg = None     # réglages de la partie, rejoués à chaque donne
    sim_task = None  # background task for annonces_sim / annonces_doudou
    belief_precompute_task = None  # background playgen precompute sweep
    agent_review_task = None  # background per-card bot review, streamed
    # Starlette WebSockets are NOT safe for concurrent sends: a background task
    # (e.g. the playgen precompute sweep) sending at the same time as the main
    # handler corrupts the ASGI state and raises RuntimeError, killing the
    # socket. Serialize every send through this lock.
    send_lock = asyncio.Lock()
    # Lecture du socket engagée par la course du dernier pli et pas encore
    # servie (`pending_recv`), et messages qu'elle a lus sans les vouloir
    # (`deferred`). La boucle principale reprend les deux avant de lire.
    pending_recv = None
    deferred = []

    async def wsend(payload):
        async with send_lock:
            await ws.send_json(payload)

    async def _wait_human_card(delay):
        """La carte que le joueur pose dans les `delay` s, ou None à l'échéance.

        Sert au dernier pli, le seul moment où le serveur joue une carte à la
        place d'un humain. La lecture du socket n'est jamais annulée — on garde
        la tâche en attente et la boucle principale en récupère le résultat —
        sinon un `play` arrivé pile à l'échéance serait perdu avec elle. Tout ce
        qui n'est pas la carte attendue (navigation, autre page) est laissé à la
        boucle principale, qui le traitera juste après.
        """
        nonlocal pending_recv
        deadline = time.monotonic() + delay
        while True:
            if pending_recv is None:
                pending_recv = asyncio.ensure_future(ws.receive_json())
            rest = deadline - time.monotonic()
            if rest <= 0:
                return None
            await asyncio.wait({pending_recv}, timeout=rest)
            if not pending_recv.done():
                return None
            data = pending_recv.result()
            pending_recv = None
            if (data.get("type") == "play" and play_session is not None
                    and not play_session.env.is_terminal()
                    and int(play_session.env.current_player())
                        == int(data.get("human_seat", 2))
                    and data.get("action") in list(play_session.env.legal_actions())):
                return int(data["action"])
            # Pas la carte : on le laisse à la boucle principale, qui le
            # traitera juste après, et on attend le reste du délai.
            deferred.append(data)

    async def _finish_deal():
        """Clore la donne en cours : base, Elo, puis score de la partie.

        Appelée avant l'envoi de l'état terminal, pour que le panneau de fin
        montre le score de partie à jour. Plusieurs chemins peuvent y mener pour
        une même donne (coup humain terminal, puis `_run_ai_turns`) — d'où le
        garde-fou dans `Match.record`, sans lequel la donne serait comptée deux
        fois.
        """
        if play_game_id is None or play_session is None:
            return
        await _complete_game(play_game_id, play_session)
        if play_match is None:
            return
        if play_match.record(play_game_id, play_session.env.rewards()):
            if play_match.id:
                await db.update_match(
                    play_match.id, play_match.totals[0], play_match.totals[1],
                    len(play_match.deals), play_match.finished, play_match.winner)

    async def _begin_deal():
        """Distribuer et lancer la donne suivante de la partie en cours."""
        nonlocal play_session, play_game_id
        cfg = play_cfg
        human_seat = cfg["human_seat"]
        bot = cfg["bot"]
        ai_types = {s: bot for s in range(4) if s != human_seat}
        # En partie le donneur tourne ; sur une donne isolée on laisse le
        # tirage au sort d'origine.
        dealer = play_match.dealer if play_match.is_match else None
        play_session = await asyncio.to_thread(
            PlaySession,
            ai_types=ai_types,
            human_seat=human_seat,
            dmc_model_path=DMC_MODEL_PATH if bot == "doudou" else None,
            bid_model_path=BID_MODEL_PATH,
            belief_model_path=BELIEF_MODEL_PATH,
            dede_time_ms=cfg["time_ms"],
            dealer=dealer,
            scores=play_match.totals,
        )

        agents_map = {str(s): t for s, t in ai_types.items()}
        agents_map[str(human_seat)] = "human"
        play_game_id = await db.create_game(
            mode="play",
            dealer=int(play_session.env.get_dealer()),
            hands=play_session.env.get_hands(),
            agents=agents_map,
            human_seat=human_seat,
            user_id=ws_user["id"] if ws_user else None,
            match_id=play_match.id,
            deal_no=play_match.deal_no if play_match.is_match else None,
        )

        await ws.send_json({
            "type": "game_state",
            "state": play_session.get_state(human_seat),
            "doudou_available": doudou_available,
            "game_id": play_game_id,
            "initial_hands": play_session.initial_hands,
            "mode": cfg["mode"],
            "bot": bot,
            "mode_degraded": cfg["degraded"],
            "match": play_match.payload(),
        })
        await _run_ai_turns(ws, play_session, human_seat, play_game_id,
                            mode=cfg["mode"], match=play_match,
                            on_deal_end=_finish_deal, wait_human=_wait_human_card)

    def _play_state_msg():
        """La position courante, telle qu'on la renverrait à un client revenu.

        Même forme que le message de `_begin_deal`, plus l'historique d'enchères :
        il vit côté client, il est parti avec la vue, et seul le serveur peut le
        rendre (`GameTable` reconstruit son panneau à partir de `bid_history`).
        """
        cfg = play_cfg or {}
        human_seat = cfg.get("human_seat", 2)
        msg = {
            "type": "game_state",
            "state": play_session.get_state(human_seat),
            "doudou_available": doudou_available,
            "game_id": play_game_id,
            "initial_hands": play_session.initial_hands,
            "mode": cfg.get("mode", play_mode),
            "bot": cfg.get("bot"),
            "mode_degraded": cfg.get("degraded", False),
            "match": play_match.payload() if play_match is not None else None,
            "bid_history": play_session.bid_history,
        }
        return _enrich_terminal_msg(msg, play_session, play_match)

    async def _send_open_matches():
        """La liste « à reprendre » : parties ouvertes du compte, sauf celle qui
        est déjà à l'écran — on ne propose pas de reprendre là où on est."""
        matches = await db.list_open_matches(ws_user["id"]) if ws_user else []
        live = play_match.id if play_match is not None else None
        await ws.send_json({
            "type": "play_open",
            "matches": [m for m in matches if m["id"] != live],
        })

    async def _resume_match(match_id):
        """Reprendre une partie interrompue, à la donne suivante.

        La donne en cours au moment de la coupure n'est pas rejouable : les bots
        n'ont pas d'état persistant, et rejouer la donne depuis ses actions ne
        leur rendrait pas le leur. Elle est donc abandonnée et la partie repart
        sur une donne neuve, au score acquis — c'est le prix affiché du bouton.
        """
        nonlocal play_match, play_cfg, play_mode
        if ws_user is None:
            await ws.send_json({"type": "error",
                                "msg": "Connectez-vous pour reprendre une partie"})
            return
        row = await db.load_open_match(match_id, ws_user["id"])
        if row is None:
            await ws.send_json({"type": "error",
                                "msg": "Partie introuvable ou déjà terminée"})
            return
        deals = row["deals"]
        if deals:
            # Le donneur passe à gauche après une donne jouée…
            dealer = (int(deals[-1]["dealer"]) + 1) % 4
        elif row["pending_dealer"] is not None:
            # …mais une donne abandonnée n'a pas eu lieu : le même redonne.
            dealer = int(row["pending_dealer"])
        else:
            dealer = None  # partie sans aucune donne : tirage au sort
        play_match = match_state.Match.restore(
            row["target"], [row["points_ns"], row["points_ew"]], deals, dealer,
            match_id=row["id"])
        play_mode = pacing.normalize(row["pacing"])
        bot, dede_time_ms, degraded = pacing.resolve(play_mode, doudou_available)
        play_cfg = {
            "human_seat": 2 if row["human_seat"] is None else int(row["human_seat"]),
            "mode": play_mode,
            "bot": bot,
            "time_ms": dede_time_ms,
            "degraded": degraded,
        }
        await _begin_deal()

    async def _cancel_sim_task():
        nonlocal sim_task
        if sim_task and not sim_task.done():
            sim_task.cancel()
            try:
                await sim_task
            except asyncio.CancelledError:
                pass
        sim_task = None

    async def _cancel_agent_review():
        nonlocal agent_review_task
        if agent_review_task and not agent_review_task.done():
            agent_review_task.cancel()
            try:
                await agent_review_task
            except asyncio.CancelledError:
                pass
        agent_review_task = None

    async def _agent_review_loop(game_id):
        """Stream the bots' choices card by card, in play order.

        Runs as its own task so the ~9s of IS-DD search never sits between the
        client and the next message it sends: loading another game cancels this
        rather than queueing behind it.
        """
        import colver.web.agent_review as agent_review
        gen = agent_review.stream(
            game_id,
            play_model=DMC_MODEL_PATH if doudou_available else None,
            belief_model=BELIEF_MODEL_PATH,
        )
        try:
            async for kind, payload in gen:
                msg = {"type": f"agent_review_{kind}", "game_id": game_id}
                if kind == "start":
                    msg["total"] = payload
                elif kind == "move":
                    msg["move"] = payload
                elif kind == "done":
                    msg.update(payload)
                else:  # error
                    msg["msg"] = payload
                await wsend(msg)
        except asyncio.CancelledError:
            raise
        except Exception as e:  # noqa: BLE001
            logger.exception("agent_review : revue en échec (game %s)", game_id)
            try:
                await wsend({"type": "agent_review_error",
                             "game_id": game_id, "msg": str(e)})
            except Exception:
                pass
        finally:
            await gen.aclose()

    async def _cancel_belief_precompute():
        nonlocal belief_precompute_task
        if belief_precompute_task and not belief_precompute_task.done():
            belief_precompute_task.cancel()
            try:
                await belief_precompute_task
            except asyncio.CancelledError:
                pass
        belief_precompute_task = None

    async def _belief_precompute_loop(session, observer):
        """Sweep all play positions, filling the session playgen cache, streaming progress."""
        loop = asyncio.get_event_loop()
        try:
            total = await loop.run_in_executor(None, session.precompute_start, observer)
            await wsend({"type": "belief_precompute", "observer": observer,
                         "done": 0, "total": total})
            while True:
                r = await loop.run_in_executor(None, session.precompute_step)
                if r is None:
                    break
                done, total = r
                await wsend({"type": "belief_precompute", "observer": observer,
                             "done": done, "total": total})
        except asyncio.CancelledError:
            raise
        except Exception as e:
            logger.exception("belief : pré-calcul playgen en échec")
            try:
                await wsend({"type": "error", "msg": f"Pré-calcul playgen : {e}"})
            except Exception:
                pass

    msg_type = None  # dernier type traité — contexte de la trace d'erreur
    try:
        while True:
            try:
                if deferred:
                    data = deferred.pop(0)
                elif pending_recv is not None:
                    # Lecture engagée par la course du dernier pli et jamais
                    # annulée : c'est elle qui porte le message suivant.
                    recv, pending_recv = pending_recv, None
                    data = await recv
                else:
                    data = await ws.receive_json()
            except json.JSONDecodeError:
                # Frame illisible : au client de se corriger — pas de traceback
                # ERROR (ni d'événement Sentry) pour du bruit d'entrée.
                logger.warning("frame WS ignorée (JSON invalide)")
                continue
            if not isinstance(data, dict) or not isinstance(data.get("type"), str):
                logger.warning("frame WS ignorée (payload non conforme) : %.80r", data)
                continue
            msg_type = data["type"]

            if msg_type and msg_type.startswith("room_"):
                await rooms.handle_message(ws_user, ws, data, {
                    "dmc": DMC_MODEL_PATH if doudou_available else None,
                    "bid": BID_MODEL_PATH,
                    "belief": BELIEF_MODEL_PATH,
                })
                continue

            if msg_type == "start_game":
                play_mode = pacing.normalize(data.get("mode"))
                # One mode picks both the tempo and the bot, and all three AI
                # seats run that same bot.
                bot, dede_time_ms, degraded = pacing.resolve(play_mode, doudou_available)
                play_cfg = {
                    "human_seat": data.get("human_seat", 2),
                    "mode": play_mode,
                    "bot": bot,
                    "time_ms": dede_time_ms,
                    "degraded": degraded,
                }
                target = match_state.normalize_target(data.get("target"))
                play_match = match_state.Match(target)
                if play_match.is_match:
                    play_match.id = await db.create_match(
                        mode="play", target=target,
                        user_id=ws_user["id"] if ws_user else None,
                        pacing=play_mode, human_seat=play_cfg["human_seat"])
                await _begin_deal()

            elif msg_type == "play_status":
                # Retour sur la page Jouer. Trois cas, dans cet ordre : la partie
                # est encore vivante sur cette socket (aller-retour vers
                # l'analyse) — on la remet à l'écran telle quelle, sans rien
                # abandonner ; sinon l'URL demande une reprise ; dans tous les
                # cas on renvoie ce qu'il reste à reprendre.
                if play_session is not None and play_match is not None \
                        and not play_match.finished:
                    await ws.send_json(_play_state_msg())
                elif data.get("resume"):
                    await _resume_match(str(data["resume"]).strip().lower())
                await _send_open_matches()

            elif msg_type == "resume_match":
                await _resume_match(str(data.get("match_id") or "").strip().lower())
                await _send_open_matches()

            elif msg_type == "abandon_match":
                # Concéder : la partie est close sans vainqueur. Si c'est celle
                # qui est en cours, la table est lâchée avec elle.
                dropped = str(data.get("match_id") or "").strip().lower()
                if ws_user is not None and dropped:
                    await db.abandon_match(dropped, ws_user["id"])
                    if play_match is not None and play_match.id == dropped:
                        play_match = None
                        play_session = None
                        play_game_id = None
                await _send_open_matches()

            elif msg_type == "next_deal":
                # Donne suivante d'une partie : mêmes réglages, donneur suivant,
                # score cumulé conservé (et transmis aux bots).
                if play_match is None or play_session is None:
                    await ws.send_json({"type": "error", "msg": "No game in progress"})
                elif not play_session.env.is_terminal():
                    await ws.send_json({"type": "error", "msg": "Donne en cours"})
                elif play_match.finished:
                    await ws.send_json({"type": "error", "msg": "Partie terminée"})
                else:
                    play_match.next_deal()
                    await _begin_deal()

            elif msg_type == "play":
                if play_session is None:
                    await ws.send_json({"type": "error", "msg": "No game in progress"})
                    continue
                action = data["action"]
                human_seat = data.get("human_seat", 2)

                # Ignore duplicate clicks when it's not the human's turn. Le cas
                # se produit aussi au dernier pli, quand le clic arrive juste
                # après que le serveur ait joué la carte à notre place.
                if (play_session.env.is_terminal()
                        or play_session.env.current_player() != human_seat):
                    continue

                state = play_session.play_action(action)
                if play_game_id:
                    await db.append_action(play_game_id, play_session.history[-1])
                # Clore la donne *avant* d'envoyer l'état terminal : le score de
                # la partie voyage avec lui (cf. `_enrich_terminal_msg`). Un
                # dernier passe humain peut terminer la donne sans pli, donc les
                # deux branches ci-dessous peuvent être terminales.
                if play_session.env.is_terminal():
                    await _finish_deal()

                msg = {"type": "game_state", "state": state}
                if play_session._belote_event:
                    msg["belote_event"] = play_session._belote_event
                    msg["belote_player"] = play_session._belote_player

                if play_session.trick_just_completed:
                    play_session.trick_just_completed = False
                    # Show completed trick (4 cards visible), pause, then clear
                    snapshot_msg = dict(msg)
                    snapshot_msg["state"] = trick_snapshot(state)
                    await ws.send_json(snapshot_msg)
                    # tricks_won is post-increment here, so the trick that just
                    # completed is one below the count.
                    await asyncio.sleep(pacing.trick_delay(
                        play_mode, sum(state["tricks_won"]) - 1,
                        deal_over=state["is_terminal"]))
                    final_msg = {"type": "game_state", "state": state}
                    _enrich_terminal_msg(final_msg, play_session, play_match)
                    await ws.send_json(final_msg)
                else:
                    _enrich_terminal_msg(msg, play_session, play_match)
                    await ws.send_json(msg)
                    # No pause here: _run_ai_turns holds this position itself
                    # before revealing the next bot's move.

                await _run_ai_turns(ws, play_session, human_seat, play_game_id,
                                    mode=play_mode, match=play_match,
                                    on_deal_end=_finish_deal,
                                    wait_human=_wait_human_card)

            elif msg_type == "watch_start":
                agents = data.get("agents", {0: "smart", 1: "smart", 2: "smart", 3: "smart"})
                # Convert string keys from JSON to int
                agents = {int(k): v for k, v in agents.items()}
                dede_time_ms = max(1000, min(15000, int(data.get("dede_time_ms", 5000))))
                watch_session = WatchSession(
                    agents=agents,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    bid_model_path=BID_MODEL_PATH,
                    belief_model_path=BELIEF_MODEL_PATH,
                    dede_time_ms=dede_time_ms,
                )
                replay_session = None

                # Save game to DB
                agents_map = {str(k): v for k, v in agents.items()}
                watch_game_id = await db.create_game(
                    mode="watch",
                    dealer=int(watch_session.env.get_dealer()),
                    hands=watch_session.env.get_hands(),
                    agents=agents_map,
                )

                watch_started_msg = {
                    "type": "watch_started",
                    "state": watch_session.get_state(),
                    "doudou_available": doudou_available,
                    "bid_history": [],
                    "completed_tricks": [],
                    "game_id": watch_game_id,
                }
                if watch_session.dd_scores is not None:
                    watch_started_msg["dd_scores"] = watch_session.dd_scores
                    watch_started_msg["dd_elapsed_ms"] = watch_session.dd_elapsed_ms
                await ws.send_json(watch_started_msg)

            elif msg_type == "watch_step":
                if watch_session is None:
                    await ws.send_json({"type": "error", "msg": "No watch session"})
                    continue
                if "dede_time_ms" in data:
                    watch_session.dede_time_ms = max(1000, min(15000, int(data["dede_time_ms"])))
                    watch_session.bots.set_time_ms(watch_session.dede_time_ms)
                if watch_session.env.is_terminal():
                    await ws.send_json({
                        "type": "watch_move",
                        "move": None,
                        "state": watch_session.get_state(),
                        "completed_tricks": watch_session.completed_tricks,
                        "bid_history": watch_session.bid_history,
                        "finished": True,
                    })
                    continue

                # Run in executor since IS-MCTS can block
                loop = asyncio.get_event_loop()
                move, state, tricks = await loop.run_in_executor(None, watch_session.step)
                finished = watch_session.env.is_terminal()
                watch_msg = {
                    "type": "watch_move",
                    "move": move,
                    "state": state,
                    "completed_tricks": tricks,
                    "bid_history": watch_session.bid_history,
                    "finished": finished,
                }
                if watch_session._belote_event:
                    watch_msg["belote_event"] = watch_session._belote_event
                    watch_msg["belote_player"] = watch_session._belote_player
                await ws.send_json(watch_msg)

                # Save action + check terminal
                if watch_game_id and move:
                    await db.append_action(watch_game_id, {
                        "player": move["player"],
                        "action": move["action"],
                        "phase": move["phase"],
                    })
                if watch_game_id and finished:
                    await _complete_game(watch_game_id, watch_session)

            elif msg_type == "replay_load":
                game_id = data.get("game_id", "").strip().lower()
                # A review of the game being left has no reader left; stop it
                # before it can interleave sends with the new game's messages.
                await _cancel_agent_review()
                game_data = await db.get_game(game_id)
                if not game_data:
                    await wsend({"type": "error", "msg": f"Partie '{game_id}' introuvable"})
                    continue
                replay_session = ReplaySession(game_data)
                watch_session = None
                initial_state = replay_session.get_state()
                # Precompute the whole game so the client can navigate (and
                # click-jump to any move) without further round-trips.
                moves = []
                while True:
                    move, state, tricks, finished = replay_session.step()
                    if move is None:
                        break
                    entry = {
                        "move": move,
                        "state": state,
                        "completed_tricks": [dict(t) for t in tricks],
                        "bid_history": list(replay_session.bid_history),
                        "finished": finished,
                        "action_idx": replay_session.action_idx,
                    }
                    if replay_session._belote_event:
                        entry["belote_event"] = replay_session._belote_event
                        entry["belote_player"] = replay_session._belote_player
                    moves.append(entry)
                    if finished:
                        break
                # Cette donne appartenait-elle à une partie encore en cours, et
                # au joueur qui regarde ? Alors la page d'analyse peut proposer
                # d'y retourner — c'est le chemin inverse du bouton « Analyser ».
                resume = None
                if ws_user is not None and game_data.get("match_id"):
                    resume = await db.open_match_summary(
                        game_data["match_id"], ws_user["id"])
                await wsend({
                    "type": "replay_loaded",
                    "state": initial_state,
                    "game_id": game_id,
                    "resume": resume,
                    "mode": game_data["mode"],
                    "agents": game_data["agents"],
                    "seat_names": await db.game_seat_names(game_data),
                    "total_actions": len(game_data["actions"]),
                    "moves": moves,
                    "bid_history": [],
                    "completed_tricks": [],
                    "game_cfn": replay_session.game_cfn,
                })

            elif msg_type == "replay_agents":
                await _cancel_agent_review()
                review_id = data.get("game_id", "").strip().lower()
                if review_id:
                    agent_review_task = asyncio.create_task(
                        _agent_review_loop(review_id))

            elif msg_type == "replay_step":
                if replay_session is None:
                    await wsend({"type": "error", "msg": "No replay session"})
                    continue
                move, state, tricks, finished = replay_session.step()
                replay_msg = {
                    "type": "replay_move",
                    "move": move,
                    "state": state,
                    "completed_tricks": tricks,
                    "bid_history": replay_session.bid_history,
                    "finished": finished,
                    "action_idx": replay_session.action_idx,
                }
                if replay_session._belote_event:
                    replay_msg["belote_event"] = replay_session._belote_event
                    replay_msg["belote_player"] = replay_session._belote_player
                await wsend(replay_msg)

            elif msg_type == "save_custom_deal":
                hands = data["hands"]
                dealer = data.get("dealer", 0)
                agents = data.get("agents", {})
                agents_map = {str(k): v for k, v in agents.items()}
                game_id = await db.create_game(
                    mode="custom",
                    dealer=dealer,
                    hands=hands,
                    agents=agents_map,
                )
                await ws.send_json({
                    "type": "deal_saved",
                    "game_id": game_id,
                })

            elif msg_type == "watch_cfn":
                cfn_str = data.get("cfn", "").strip()
                if not cfn_str:
                    await ws.send_json({"type": "error", "msg": "CFN vide"})
                    continue
                try:
                    import colver as _cfn_colver
                    cfn_env = _cfn_colver.Env.from_cfn(cfn_str)
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": f"CFN invalide : {e}"})
                    continue
                agents = data.get("agents", {0: "dede", 1: "dede", 2: "dede", 3: "dede"})
                agents = {int(k): v for k, v in agents.items()}
                watch_session = WatchSession(
                    agents=agents,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    bid_model_path=BID_MODEL_PATH,
                    belief_model_path=BELIEF_MODEL_PATH,
                    env=cfn_env,
                    dede_time_ms=max(1000, min(15000, int(data.get("dede_time_ms", 5000)))),
                )
                replay_session = None
                watch_game_id = None

                cfn_started_msg = {
                    "type": "watch_started",
                    "state": watch_session.get_state(),
                    "doudou_available": doudou_available,
                    "bid_history": [],
                    "completed_tricks": [],
                    "game_id": watch_game_id,
                }
                if watch_session.dd_scores is not None:
                    cfn_started_msg["dd_scores"] = watch_session.dd_scores
                    cfn_started_msg["dd_elapsed_ms"] = watch_session.dd_elapsed_ms
                await ws.send_json(cfn_started_msg)

            elif msg_type == "bid_eval":
                hand = data.get("hand", [])
                # Accept prior_actions (list of action indices) or fall back to prior_passes
                prior_actions_raw = data.get("prior_actions", None)
                if prior_actions_raw is not None:
                    prior_actions = [int(a) for a in prior_actions_raw]
                else:
                    prior_passes = min(3, max(0, int(data.get("prior_passes", 0))))
                    prior_actions = [0] * prior_passes
                if len(hand) != 8:
                    await ws.send_json({"type": "bid_eval_result", "error": "8 cartes requises"})
                    continue
                if not BID_MODEL_PATH:
                    await ws.send_json({"type": "bid_eval_result", "error": "Modèle d'enchères non disponible"})
                    continue
                try:
                    import random as _random
                    remaining = list(set(range(32)) - set(hand))
                    _random.shuffle(remaining)
                    hands = [None] * 4
                    seat = 2  # always evaluate as Sud (seat 2)
                    hands[seat] = sorted(hand)
                    others = [s for s in range(4) if s != seat]
                    for i, p in enumerate(others):
                        hands[p] = sorted(remaining[i * 8:(i + 1) * 8])
                    n_prior = len(prior_actions)
                    dealer = (seat - 1 - n_prior + 32) % 4
                    env = _colver_pkg.Env.deal_with_hands(dealer, hands)
                    env.load_bid_model(BID_MODEL_PATH)
                    analyst = None
                    if PLAYGEN_MODEL_PATH:
                        try:
                            analyst = _colver_pkg.Analyst.replay(
                                PLAYGEN_MODEL_PATH, dealer, hands,
                                [int(a) for a in prior_actions], seat,
                            )
                        except Exception:
                            analyst = None
                    for action in prior_actions:
                        env.step(action)
                    result = env.action_bid_nn()
                    playgen_policy = None
                    if analyst is not None:
                        # v2 playgen models only; returns None on v1 weights.
                        pol = analyst.bid_policy(env, 1.0)
                        if pol is not None:
                            playgen_policy = [
                                [a, round(float(p), 4)]
                                for a, p in enumerate(pol) if p > 0.0005
                            ]
                    await ws.send_json({
                        "type": "bid_eval_result",
                        "q_values": [[int(a), round(float(q), 3)] for a, q in result["q_values"]],
                        "best_action": int(result["best_action"]),
                        "playgen_policy": playgen_policy,
                    })
                except Exception as e:
                    await ws.send_json({"type": "bid_eval_result", "error": str(e)})

            elif msg_type == "dd_sim":
                hand = data.get("hand", [])
                num_sims = max(1, min(200, int(data.get("num_sims", 50))))
                if len(hand) != 8:
                    await ws.send_json({"type": "dd_sim_result", "error": "8 cartes requises"})
                    continue
                try:
                    loop = asyncio.get_event_loop()
                    result = await loop.run_in_executor(None, _run_dd_sim, hand, num_sims)
                    await ws.send_json(result)
                except Exception as e:
                    await ws.send_json({"type": "dd_sim_result", "error": str(e)})

            elif msg_type == "annonces_sim":
                await _cancel_sim_task()
                sim_task = asyncio.create_task(
                    _run_annonces_sim(ws, data))

            elif msg_type == "annonces_doudou":
                await _cancel_sim_task()
                sim_task = asyncio.create_task(
                    _run_annonces_doudou(ws, data))

            elif msg_type == "card_analysis":
                # Même règle que les annonces : une seule simulation à la fois,
                # et le `req_id` renvoyé évite qu'une analyse annulée peigne
                # ses derniers messages dans la position suivante.
                await _cancel_sim_task()
                sim_task = asyncio.create_task(
                    _run_card_analysis(ws, data))

            elif msg_type == "watch_custom":
                game_id = data.get("game_id", "").strip().lower()
                # Une donne personnalisée fraîchement enregistrée n'est pas
                # encore `is_complete` : c'est le seul cas où une donne en
                # cours peut être regardée ici — en servir une autre (donne
                # solo ou salon en train de se jouer) montrerait les mains.
                game_data = await db.get_game(game_id, include_incomplete=True)
                if not game_data or (not game_data["is_complete"]
                                     and game_data["mode"] != "custom"):
                    await ws.send_json({"type": "error", "msg": f"Partie '{game_id}' introuvable"})
                    continue
                # Client may override agents (e.g. from Watch tab's dropdowns)
                if "agents" in data:
                    agents = {int(k): v for k, v in data["agents"].items()}
                else:
                    agents = {int(k): v for k, v in game_data["agents"].items()}
                watch_session = WatchSession(
                    agents=agents,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    bid_model_path=BID_MODEL_PATH,
                    belief_model_path=BELIEF_MODEL_PATH,
                    dealer=game_data["dealer"],
                    hands=game_data["hands"],
                    dede_time_ms=max(1000, min(15000, int(data.get("dede_time_ms", 5000)))),
                )
                replay_session = None
                # N'écrire en base que pour la donne custom en cours de
                # composition : rejouer une donne déjà terminée dans Regarder
                # ne doit pas concaténer une seconde trajectoire dans son
                # enregistrement (watch_step append dans games.actions, et la
                # fin réécrirait ses points). Session éphémère, comme watch_cfn.
                watch_game_id = game_id if not game_data["is_complete"] else None

                custom_started_msg = {
                    "type": "watch_started",
                    "state": watch_session.get_state(),
                    "doudou_available": doudou_available,
                    "bid_history": [],
                    "completed_tricks": [],
                    "game_id": watch_game_id,
                }
                if watch_session.dd_scores is not None:
                    custom_started_msg["dd_scores"] = watch_session.dd_scores
                    custom_started_msg["dd_elapsed_ms"] = watch_session.dd_elapsed_ms
                await ws.send_json(custom_started_msg)

            elif msg_type == "bid_problem_generate":
                loop = asyncio.get_event_loop()
                bid_problem_session = BidProblemSession(
                    bid_model_path=BID_MODEL_PATH,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                )
                try:
                    result = await loop.run_in_executor(None, bid_problem_session.generate)
                    await ws.send_json({"type": "bid_problem_ready", **result,
                                        "has_bid_model": bool(BID_MODEL_PATH)})
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": f"Génération échouée : {e}"})

            elif msg_type == "bid_problem_submit":
                if bid_problem_session is None or bid_problem_session.env is None:
                    await ws.send_json({"type": "error", "msg": "Pas de problème en cours"})
                    continue
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, bid_problem_session.evaluate, int(data["action"]))
                    await ws.send_json({"type": "bid_problem_correction", **result})
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": f"Évaluation échouée : {e}"})

            elif msg_type == "play_problem_generate":
                loop = asyncio.get_event_loop()
                play_problem_session = PlayProblemSession(
                    bid_model_path=BID_MODEL_PATH,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                )
                try:
                    result = await loop.run_in_executor(None, play_problem_session.generate)
                    await ws.send_json({"type": "play_problem_ready", **result,
                                        "has_dmc_model": doudou_available})
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": f"Génération échouée : {e}"})

            elif msg_type == "play_problem_submit":
                if play_problem_session is None or play_problem_session.env is None:
                    await ws.send_json({"type": "error", "msg": "Pas de problème en cours"})
                    continue
                try:
                    result = await asyncio.get_event_loop().run_in_executor(
                        None, play_problem_session.evaluate, int(data["action"]))
                    await ws.send_json({"type": "play_problem_correction", **result})
                except Exception as e:
                    await ws.send_json({"type": "error", "msg": f"Évaluation échouée : {e}"})

            elif msg_type == "belief_generate":
                await _cancel_belief_precompute()
                loop = asyncio.get_event_loop()
                belief_session = BeliefSession(
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    bid_model_path=BID_MODEL_PATH,
                    belief_model_path=BELIEF_MODEL_PATH,
                    playgen_model_path=PLAYGEN_MODEL_PATH,
                )
                try:
                    result = await loop.run_in_executor(None, belief_session.generate)
                    await wsend({"type": "belief_generated", **result})
                    # Warm the playgen cache in the background for the default observer
                    if PLAYGEN_MODEL_PATH:
                        belief_precompute_task = asyncio.create_task(
                            _belief_precompute_loop(belief_session, 0))
                except Exception as e:
                    await wsend({"type": "error", "msg": f"Génération échouée : {e}"})

            elif msg_type == "belief_import":
                # Load a specific game from a pasted full-game CFN (auction + play).
                await _cancel_belief_precompute()
                loop = asyncio.get_event_loop()
                cfn = str(data.get("cfn", "")).strip()
                belief_session = BeliefSession(
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    bid_model_path=BID_MODEL_PATH,
                    belief_model_path=BELIEF_MODEL_PATH,
                    playgen_model_path=PLAYGEN_MODEL_PATH,
                )
                try:
                    result = await loop.run_in_executor(None, belief_session.import_cfn, cfn)
                    await wsend({"type": "belief_generated", **result})
                    if PLAYGEN_MODEL_PATH:
                        belief_precompute_task = asyncio.create_task(
                            _belief_precompute_loop(belief_session, 0))
                except Exception as e:
                    belief_session = None
                    await wsend({"type": "error", "msg": f"CFN invalide : {e}"})

            elif msg_type == "belief_restore":
                # WS reconnected: rebuild the (per-connection) session from the
                # deal the client still holds, then jump to its last position.
                await _cancel_belief_precompute()
                loop = asyncio.get_event_loop()
                belief_session = BeliefSession(
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    bid_model_path=BID_MODEL_PATH,
                    belief_model_path=BELIEF_MODEL_PATH,
                    playgen_model_path=PLAYGEN_MODEL_PATH,
                )
                try:
                    belief_session.restore(
                        data["dealer"], data["initial_hands"], data["actions"])
                    target = int(data.get("target", 0))
                    result = await loop.run_in_executor(
                        None, belief_session.step_to, target)
                    await wsend({"type": "belief_state", **result})
                    if PLAYGEN_MODEL_PATH:
                        observer = int(data.get("observer", 0))
                        belief_precompute_task = asyncio.create_task(
                            _belief_precompute_loop(belief_session, observer))
                except Exception as e:
                    belief_session = None
                    await wsend({"type": "error", "msg": f"Restauration échouée : {e}"})

            elif msg_type == "belief_precompute":
                if belief_session is None:
                    await wsend({"type": "error", "msg": "Pas de session croyances"})
                    continue
                if PLAYGEN_MODEL_PATH:
                    observer = int(data.get("observer", 0))
                    await _cancel_belief_precompute()
                    belief_precompute_task = asyncio.create_task(
                        _belief_precompute_loop(belief_session, observer))

            elif msg_type == "belief_step":
                if belief_session is None:
                    await wsend({"type": "error", "msg": "Pas de session croyances"})
                    continue
                result = belief_session.step_forward()
                await wsend({"type": "belief_state", **result})

            elif msg_type == "belief_step_to":
                if belief_session is None:
                    await wsend({"type": "error", "msg": "Pas de session croyances"})
                    continue
                target = int(data.get("target", 0))
                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(None, belief_session.step_to, target)
                    await wsend({"type": "belief_state", **result})
                except Exception as e:
                    await wsend({"type": "error", "msg": f"Erreur step_to : {e}"})

            elif msg_type == "belief_get_weights":
                if belief_session is None:
                    await wsend({"type": "error", "msg": "Pas de session croyances"})
                    continue
                observer = int(data.get("observer", 0))
                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(None, belief_session.get_beliefs, observer)
                    await wsend({"type": "belief_weights", **result})
                except Exception as e:
                    await wsend({"type": "error", "msg": f"Erreur croyances : {e}"})

            else:
                await ws.send_json({"type": "error", "msg": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        if sim_task and not sim_task.done():
            sim_task.cancel()
        if belief_precompute_task and not belief_precompute_task.done():
            belief_precompute_task.cancel()
        if agent_review_task and not agent_review_task.done():
            agent_review_task.cancel()
    except Exception:
        # Une exception imprévue tue le socket. uvicorn la loggerait sans
        # contexte (« Exception in ASGI application ») et les tâches de fond
        # resteraient orphelines : on la trace avec le dernier msg_type traité
        # et on annule comme à la déconnexion.
        logger.exception(
            "_websocket_session : exception non gérée (dernier msg_type=%r)", msg_type)
        if sim_task and not sim_task.done():
            sim_task.cancel()
        if belief_precompute_task and not belief_precompute_task.done():
            belief_precompute_task.cancel()
        if agent_review_task and not agent_review_task.done():
            agent_review_task.cancel()
    finally:
        # Lecture laissée en attente par la course du dernier pli : sans ça la
        # déconnexion lui remonte une exception que personne ne réclame.
        if pending_recv is not None:
            if not pending_recv.done():
                pending_recv.cancel()
            elif not pending_recv.cancelled():
                pending_recv.exception()  # marque l'erreur comme réclamée
        await rooms.handle_disconnect(ws)


def _enrich_terminal_msg(msg, play_session, match=None):
    """Add review data (initial hands, bids, tricks) to terminal game_state messages.

    Le score de la partie voyage avec l'état terminal : c'est lui qui décide si
    le panneau de fin propose « Donne suivante » ou clôt la partie.
    """
    if play_session.env.is_terminal():
        msg["initial_hands"] = play_session.initial_hands
        msg["bid_history"] = play_session.bid_history
        msg["completed_tricks"] = play_session.completed_tricks
        if match is not None:
            msg["match"] = match.payload()
    return msg


async def _complete_game(game_id, session):
    """Mark a game as complete in the database."""
    points = list(session.env.get_points())
    contract = session.env.get_contract()
    await db.complete_game(game_id, points[0], points[1], contract)
    await elo.rate_game(game_id)


async def _run_ai_turns(ws, session, human_seat, game_id=None, mode=pacing.DEFAULT_MODE,
                        match=None, on_deal_end=None, wait_human=None):
    """Auto-play AI turns until human's turn or game over.

    A forced pass on the human's seat counts as an AI turn: when passing is the
    only legal bid there is nothing to decide, so the server plays it instead of
    waiting for a click on the single available button.

    Le dernier pli est du même ordre — chaque siège n'a plus qu'une carte — mais
    il se joue, pas seulement il se décide : on le déroule donc sans rendre la
    main, en laissant au siège humain le délai du pli pour poser sa carte
    lui-même (`wait_human`, qui rend l'action cliquée ou `None` à l'échéance).
    Sans `wait_human` la boucle s'arrête au siège humain comme avant.

    Pacing note: the pause belongs to the position *preceding* a move, and the
    bot's own thinking is spent inside it rather than on top of it. So each
    iteration holds the current position for the mode's target, computes the
    move while it is still on screen, and only then reveals it. That is why
    there is no trailing sleep — and why the caller must not pause before
    handing over.
    """
    while not session.env.is_terminal():
        # Un siège humain qui n'a rien à décider est joué par le serveur : passe
        # forcé, ou carte unique du dernier pli.
        human_turn = (session.env.current_player() == human_seat
                      and not only_pass_is_legal(session.env))
        if human_turn and not (wait_human is not None and in_last_trick(session.env)):
            break
        trick_idx = sum(session.env.get_tricks_won())
        target = pacing.move_delay(
            mode, session.env.phase(), trick_idx, cards_in_trick(session.env))
        t0 = time.monotonic()
        clicked = False
        if human_turn:
            # Dernier pli : la carte est forcée, mais le joueur garde le droit
            # de la poser. On attend `target`, puis on la joue pour lui.
            action = await wait_human(target)
            clicked = action is not None
            if not clicked:
                action = int(session.env.legal_actions()[0])
            state = await asyncio.to_thread(session.play_action, action)
            name = _colver_pkg.Env.action_name(int(action),
                                               int(session.history[-1]["phase"]))
        else:
            # The Rust search releases the GIL, so this keeps the event loop free.
            action, name, state = await asyncio.to_thread(session.play_ai_turn)
            await pacing.hold(target, time.monotonic() - t0)
        player = session.history[-1]["player"]
        # La phase du coup, comme en salon (`room_move`) : sans elle le client
        # ne peut pas distinguer une annonce d'une carte, et rangeait chaque
        # carte jouée dans l'historique des enchères (au son d'annonce).
        ai_msg = {
            "type": "ai_move",
            "player": player,
            "action": action,
            "name": name,
            "phase": int(session.history[-1]["phase"]),
        }
        if player == human_seat and not clicked:
            # Our own seat, played for us — the client's local echo never fired.
            ai_msg["auto"] = True
        if session._belote_event:
            ai_msg["belote_event"] = session._belote_event
            ai_msg["belote_player"] = session._belote_player
        await ws.send_json(ai_msg)

        if game_id:
            await db.append_action(game_id, session.history[-1])
        # Clore la donne avant l'état terminal, qui emporte le score de partie.
        if session.env.is_terminal() and on_deal_end is not None:
            await on_deal_end()

        if session.trick_just_completed:
            session.trick_just_completed = False
            # Show completed trick (4 cards visible), pause, then clear
            await ws.send_json({"type": "game_state", "state": trick_snapshot(state)})
            await asyncio.sleep(pacing.trick_delay(
                mode, trick_idx, deal_over=state["is_terminal"]))
            # Send cleared state — no delay after (the next iteration holds it)
            final_msg = {"type": "game_state", "state": state}
            _enrich_terminal_msg(final_msg, session, match)
            await ws.send_json(final_msg)
        else:
            state_msg = {"type": "game_state", "state": state}
            _enrich_terminal_msg(state_msg, session, match)
            await ws.send_json(state_msg)

    # Check terminal after AI turns (no-op when the loop already closed the
    # deal: `on_deal_end` is idempotent).
    if session.env.is_terminal() and on_deal_end is not None:
        await on_deal_end()
    elif game_id and session.env.is_terminal():
        await _complete_game(game_id, session)


def _run_dd_sim(hand, num_sims):
    """Run DD simulation: deal random opponent hands and solve all 4 suits."""
    import random
    import time

    start = time.monotonic()
    remaining = list(set(range(32)) - set(hand))
    seat = 2  # Sud
    others = [s for s in range(4) if s != seat]
    totals = [[0.0, 0.0] for _ in range(4)]  # per-suit [ns_sum, ew_sum]

    for _ in range(num_sims):
        random.shuffle(remaining)
        hands = [None] * 4
        hands[seat] = sorted(hand)
        for i, p in enumerate(others):
            hands[p] = sorted(remaining[i * 8:(i + 1) * 8])
        env = _colver_pkg.Env.deal_with_hands(0, hands)
        result = env.solve_all_suits()
        for suit_idx, (ns, ew) in enumerate(result["suits"]):
            totals[suit_idx][0] += ns
            totals[suit_idx][1] += ew

    elapsed_ms = (time.monotonic() - start) * 1000.0
    suits = []
    for suit_idx in range(4):
        suits.append({
            "suit": suit_idx,
            "avg_ns": round(totals[suit_idx][0] / num_sims, 1),
            "avg_ew": round(totals[suit_idx][1] / num_sims, 1),
        })

    return {
        "type": "dd_sim_result",
        "suits": suits,
        "num_sims": num_sims,
        "elapsed_ms": round(elapsed_ms, 1),
    }


def _run_dd_sim_with_hands(hands, dealer=0):
    """Run DD solve on pre-generated hands."""
    seat = 2
    others = [s for s in range(4) if s != seat]
    env = _colver_pkg.Env.deal_with_hands(dealer, hands)
    dd_result = env.solve_all_suits()
    deal_hands = {str(p): list(hands[p]) for p in others}
    return {"suits": dd_result["suits"], "hands": deal_hands}


def _run_doudou_sim_with_hands(hands, bid_model_path, dmc_model_path,
                                dealer=0, prior_actions=None):
    """Run Dédé (NN bid + DMC play) on pre-generated hands."""
    env = _get_doudou_env(bid_model_path, dmc_model_path, dealer, hands)

    if prior_actions:
        for action in prior_actions:
            env.step(action)

    safety = 0
    while env.phase() == 0 and not env.is_terminal() and safety < 50:
        env.step(int(env.bid_a_dd()))
        safety += 1

    contract = env.get_contract()
    if not contract:
        return {"void": True}

    while not env.is_terminal():
        result = env.action_dmc_with_stats()
        env.step(int(result["best_action"]))
    rewards = env.rewards()
    taker = contract["team"]
    return {
        "void": False,
        "trump": contract["trump"],
        "value": contract["value"],
        "team": taker,
        "coinche": contract["coinche"],
        "achieved": rewards[taker] > 0,
        "auction": [[int(s), int(a)] for s, a in env.get_bid_history()],
        "scores": [float(rewards[0]), float(rewards[1])],
    }


def _run_single_dd_sim(hand, remaining, dealer=0):
    """Run ONE DD simulation: shuffle remaining cards, solve all 4 suits."""
    import random

    remaining = list(remaining)  # copy to avoid mutating caller's list
    random.shuffle(remaining)
    seat = 2  # Sud
    others = [s for s in range(4) if s != seat]
    hands = [None] * 4
    hands[seat] = sorted(hand)
    for i, p in enumerate(others):
        hands[p] = sorted(remaining[i * 8:(i + 1) * 8])

    env = _colver_pkg.Env.deal_with_hands(dealer, hands)
    dd_result = env.solve_all_suits()

    deal_hands = {}
    for p in others:
        deal_hands[str(p)] = list(hands[p])

    return {
        "suits": dd_result["suits"],
        "hands": deal_hands,
    }


def _run_single_combined_sim(hand, remaining, bid_model_path, dmc_model_path,
                             dealer=0, prior_actions=None):
    """Run ONE combined sim: same deal for DD solve + Dédé full game."""
    import random

    remaining = list(remaining)
    random.shuffle(remaining)
    seat = 2  # Sud
    others = [s for s in range(4) if s != seat]
    hands = [None] * 4
    hands[seat] = sorted(hand)
    for i, p in enumerate(others):
        hands[p] = sorted(remaining[i * 8:(i + 1) * 8])

    # 1. DD solve (Oracle)
    env_dd = _colver_pkg.Env.deal_with_hands(dealer, hands)
    dd_result = env_dd.solve_all_suits()
    deal_hands = {str(p): list(hands[p]) for p in others}

    # 2. Dédé full game (if models available)
    doudou = None
    if bid_model_path and dmc_model_path:
        env = _colver_pkg.Env.deal_with_hands(dealer, hands)
        env.load_bid_model(bid_model_path)
        env.load_dmc_model(dmc_model_path)

        # Replay user's bid history before letting Dédé continue
        if prior_actions:
            for action in prior_actions:
                env.step(action)

        # Bidding phase (Dédé continues from where history left off)
        safety = 0
        while env.phase() == 0 and not env.is_terminal() and safety < 50:
            env.step(int(env.bid_a_dd()))
            safety += 1

        contract = env.get_contract()
        if not contract:
            doudou = {"void": True}
        else:
            # Play phase with DMC
            while not env.is_terminal():
                result = env.action_dmc_with_stats()
                env.step(int(result["best_action"]))
            rewards = env.rewards()
            taker = contract["team"]
            doudou = {
                "void": False,
                "trump": contract["trump"],
                "value": contract["value"],
                "team": taker,
                "coinche": contract["coinche"],
                "achieved": rewards[taker] > 0,
                "auction": [[int(s), int(a)] for s, a in env.get_bid_history()],
                "scores": [float(rewards[0]), float(rewards[1])],
            }

    return {"suits": dd_result["suits"], "hands": deal_hands, "doudou": doudou}


def _run_single_doudou_sim(hand, remaining, bid_model_path, dmc_model_path,
                            dealer=0, prior_actions=None, forced_action=None):
    """Run ONE Dédé-only sim: shuffle opponent hands, NN bid + DMC play (no DD solve).

    If forced_action is given, South's next bid (right after prior_actions) is
    forced to it; the rest of the auction and the play stay NN-driven.
    """
    import random

    remaining = list(remaining)
    random.shuffle(remaining)
    seat = 2  # Sud
    others = [s for s in range(4) if s != seat]
    hands = [None] * 4
    hands[seat] = sorted(hand)
    for i, p in enumerate(others):
        hands[p] = sorted(remaining[i * 8:(i + 1) * 8])

    doudou = None
    env = _get_doudou_env(bid_model_path, dmc_model_path, dealer, hands)

    # Replay user's bid history
    if prior_actions:
        for action in prior_actions:
            env.step(action)

    # Forced bid: South's turn comes right after the replayed history
    if forced_action is not None and env.phase() == 0 and not env.is_terminal():
        env.step(forced_action)

    # Bidding phase
    safety = 0
    while env.phase() == 0 and not env.is_terminal() and safety < 50:
        env.step(int(env.bid_a_dd()))
        safety += 1

    contract = env.get_contract()
    if not contract:
        doudou = {"void": True}
    else:
        # Play phase with DMC
        while not env.is_terminal():
            result = env.action_dmc_with_stats()
            env.step(int(result["best_action"]))
        rewards = env.rewards()
        taker = contract["team"]
        doudou = {
            "void": False,
            "trump": contract["trump"],
            "value": contract["value"],
            "team": taker,
            "coinche": contract["coinche"],
            "achieved": rewards[taker] > 0,
            "auction": [[int(s), int(a)] for s, a in env.get_bid_history()],
            "scores": [float(rewards[0]), float(rewards[1])],
        }

    return {"doudou": doudou}


# robots.txt / sitemap.xml — déclarés AVANT le catch-all, qui les servirait
# sinon en HTML. Robots : tout est permis sauf l'API et le WebSocket.
@app.get("/robots.txt")
async def robots_txt():
    return PlainTextResponse(
        "User-agent: *\n"
        "Disallow: /api/\n"
        "Disallow: /ws\n"
        "\n"
        f"Sitemap: {PUBLIC_URL}/sitemap.xml\n"
    )


@app.get("/sitemap.xml")
async def sitemap_xml():
    urls = "\n".join(
        f"  <url><loc>{PUBLIC_URL}{p}</loc></url>" for p in _SITEMAP_ROUTES)
    return Response(
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{urls}\n"
        "</urlset>\n",
        media_type="application/xml",
    )


# Catch-all for client-side routes (pushState).
# Must be registered AFTER all API/WS/static mounts.
@app.get("/{full_path:path}")
async def spa_catchall(full_path: str, request: Request):
    path = "/" + full_path
    # Une donne partagée mérite de vraies métadonnées : titre et description
    # composés depuis la donne elle-même quand elle existe et est terminée.
    if path == "/analyse/rejouer" and request.query_params.get("game"):
        meta = await _replay_meta(request.query_params["game"])
        if meta is not None:
            return _serve_index(path, meta)
    return _serve_index(path)
