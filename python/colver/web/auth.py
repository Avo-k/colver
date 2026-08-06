"""Account management: register / login / logout with DB-backed session cookies.

Sessions are random tokens stored hashed (sha256) in SQLite, so a DB leak
doesn't expose usable cookies and individual sessions can be revoked.

Le cycle de vie complet vit ici : changer son mot de passe, en réinitialiser un
oublié par courriel, renseigner une adresse, supprimer son compte. Trois règles
traversent tout le fichier.

1. **Ne jamais dire si un compte existe.** `forgot` répond la même chose pour un
   pseudo connu et un inconnu ; `login` a déjà la même discipline (il brûle un
   bcrypt sur un utilisateur absent pour que le temps de réponse ne trahisse
   rien). Sinon le formulaire d'oubli devient un annuaire.
2. **Tout changement d'identifiant révoque les autres sessions.** Un mot de
   passe changé qui laisserait vivre une session ouverte ailleurs ne protège de
   rien — c'est précisément le cas qu'on veut couvrir.
3. **Toute opération sensible redemande le mot de passe**, même connecté : le
   cookie prouve qu'on a ouvert la session, pas qu'on est encore devant
   l'écran.
"""

import asyncio
import hashlib
import logging
import os
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

import colver.web.analysis as analysis
import colver.web.database as db
import colver.web.elo as elo
import colver.web.stats as stats
import colver.web.mail as mail
from colver.web.ratelimit import RateLimiter

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api")

# Base publique des liens envoyés par courriel. Même variable que le SEO
# (`server.PUBLIC_URL`), relue ici pour ne pas importer `server` — il importe
# déjà ce module, et le cycle casserait au démarrage.
PUBLIC_URL = os.environ.get("COLVER_PUBLIC_URL", "https://colver.net").rstrip("/")

COOKIE_NAME = "colver_session"
SESSION_DAYS = 30

# ===== Identité d'invité =====
#
# Un visiteur sans compte peut s'asseoir à une table partagée, mais seulement
# pour une **donne isolée** : une partie en 1000/2000 points dure une demi-heure
# et trois autres personnes en dépendent, donc elle exige un compte — c'est lui
# qui permet de reconnaître quelqu'un qui revient, et c'est lui qu'on classe.
#
# Ce jeton n'est **pas** une session : aucune ligne en base, aucun droit, rien à
# révoquer. Il ne répond qu'à « est-ce le même visiteur qu'il y a trente
# secondes ? » — et sans lui un F5 ferait perdre le siège, alors que toute la
# règle du temps de jeu repose sur le fait qu'un rechargement n'est pas un
# abandon.
GUEST_COOKIE = "colver_guest"
GUEST_DAYS = 30
_GUEST_RE = re.compile(r"^[A-Za-z0-9_-]{16,64}$")


def guest_from_cookies(cookies):
    """Le jeton d'invité que le navigateur renvoie, ou None."""
    token = cookies.get(GUEST_COOKIE)
    return token if token and _GUEST_RE.match(token) else None


def new_guest_token():
    return secrets.token_urlsafe(16)


def guest_cookie_header(token, secure):
    """En-tête `set-cookie` pour la **poignée de main WebSocket**.

    Posé là et nulle part ailleurs, pour trois raisons qui tiennent ensemble :
    c'est la seule réponse à la fois jamais mise en cache (un `Set-Cookie` sur
    le HTML servi derrière Cloudflare serait partagé entre visiteurs),
    antérieure au premier message de salon, et écrite par le serveur — un jeton
    fabriqué par le client laisserait prendre le siège d'un autre.
    """
    parts = [
        f"{GUEST_COOKIE}={token}",
        f"Max-Age={GUEST_DAYS * 86400}",
        "Path=/",
        "HttpOnly",
        "SameSite=Lax",
    ]
    if secure:
        parts.append("Secure")
    return (b"set-cookie", "; ".join(parts).encode())


USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,20}$")
MIN_PASSWORD_LEN = 8

# Validation d'adresse volontairement grossière : elle n'attrape que les fautes
# de frappe évidentes. La seule preuve qu'une adresse existe est qu'un message
# y arrive — c'est le lien de réinitialisation qui la vérifie, pas une regex.
EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s.]+\.[^@\s]+$")
EMAIL_MAX_LEN = 254

# Durée de vie d'un lien de réinitialisation. Assez court pour qu'un courriel
# oublié dans une boîte cesse vite d'être une clé, assez long pour survivre à
# « je regarderai ce soir ».
RESET_HOURS = 2

# bcrypt is deliberately slow (~100ms per call). Run it on its own tiny pool:
# asyncio.to_thread shares the default executor with the AI turns
# (server._run_ai_turns), so a login brute-force would also stall the tables.
_BCRYPT_EXECUTOR = ThreadPoolExecutor(max_workers=2, thread_name_prefix="bcrypt")

# One shared budget for login + register — both burn a bcrypt per call.
_AUTH_LIMITER = RateLimiter(limit=5, window=60.0)


async def _bcrypt(fn, *args):
    return await asyncio.get_running_loop().run_in_executor(_BCRYPT_EXECUTOR, fn, *args)


def _rate_limited(request):
    """429 response when this client has spent its auth budget, else None."""
    ip = request.client.host if request.client else "?"
    if _AUTH_LIMITER.allow(ip):
        return None
    return JSONResponse(
        {"error": "Trop de tentatives — réessayez dans une minute"},
        status_code=429,
        headers={"Retry-After": str(_AUTH_LIMITER.retry_after(ip))},
    )


def _refund(request):
    """A successful auth is not brute force: give the token back, so a group
    behind one NAT (friends joining a salon) doesn't burn the shared budget."""
    _AUTH_LIMITER.refund(request.client.host if request.client else "?")


def _hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _check_password_rules(password):
    """Message d'erreur si le mot de passe ne convient pas, sinon None."""
    if len(password or "") < MIN_PASSWORD_LEN:
        return f"Mot de passe trop court ({MIN_PASSWORD_LEN} caractères minimum)"
    return None


def _normalize_email(raw):
    """(adresse, erreur). Une chaîne vide vaut « retirer l'adresse »."""
    email = (raw or "").strip().lower()
    if not email:
        return None, None
    if len(email) > EMAIL_MAX_LEN or not EMAIL_RE.match(email):
        return None, "Adresse e-mail invalide"
    return email, None


async def _verify_password(user, password):
    return await _bcrypt(bcrypt.checkpw, (password or "").encode(),
                         user["password_hash"].encode())


def _public_base(request):
    """L'origine à mettre dans un lien de courriel.

    `COLVER_PUBLIC_URL` fait foi : derrière Cloudflare puis Caddy, l'hôte vu par
    l'application n'est pas celui que le joueur a tapé, et un lien fabriqué
    depuis `request.url` pointerait vers `localhost:8000`. On ne lit l'en-tête
    `Host` que si rien n'est configuré — jamais en production, donc jamais dans
    un cas où il serait forgeable à conséquence.
    """
    if PUBLIC_URL:
        return PUBLIC_URL
    return str(request.base_url).rstrip("/")


async def _set_password(user_id, password):
    pw_hash = await _bcrypt(bcrypt.hashpw, password.encode(), bcrypt.gensalt())
    await db.set_password(user_id, pw_hash.decode())


def _is_secure(request):
    # Behind Caddy/Cloudflare the app sees plain HTTP. With uvicorn's
    # proxy_headers, request.url.scheme is rewritten from X-Forwarded-Proto
    # when the connection comes from a trusted proxy; the raw header stays as
    # a fallback for launches without one. Reading it can only upgrade to
    # https — a real https scheme always wins — so spoofing it is harmless.
    if request.url.scheme == "https":
        return True
    return request.headers.get("x-forwarded-proto") == "https"


async def _start_session(response, request, user_id):
    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(days=SESSION_DAYS)
    await db.create_session(_hash_token(token), user_id, expires.isoformat())
    response.set_cookie(
        COOKIE_NAME,
        token,
        max_age=SESSION_DAYS * 86400,
        httponly=True,
        samesite="lax",
        secure=_is_secure(request),
        path="/",
    )


async def user_from_cookies(cookies):
    """Resolve a user dict from a cookie mapping (works for HTTP and WS)."""
    token = cookies.get(COOKIE_NAME)
    if not token:
        return None
    return await db.get_session_user(_hash_token(token))


async def current_user(request: Request):
    return await user_from_cookies(request.cookies)


def _public_user(user):
    return {
        "id": user["id"],
        "username": user["username"],
        "created_at": user["created_at"],
        # L'adresse revient au propriétaire du compte, pour qu'il sache s'il en
        # a une — c'est la différence entre « j'ai un recours » et « je n'en ai
        # pas », et il n'a aucun autre moyen de la connaître.
        "email": user.get("email"),
    }


@router.post("/auth/register")
async def register(request: Request):
    body = await request.json()
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    if not USERNAME_RE.match(username):
        return JSONResponse(
            {"error": "Pseudo invalide : 3-20 caractères (lettres, chiffres, - ou _)"},
            status_code=400,
        )
    bad = _check_password_rules(password)
    if bad:
        return JSONResponse({"error": bad}, status_code=400)
    # L'adresse est facultative à l'inscription : sans elle le compte est
    # parfaitement jouable, il n'a simplement aucun recours si le mot de passe
    # se perd — c'est à l'interface de le dire, pas à ce formulaire de forcer.
    email, bad_email = _normalize_email(body.get("email"))
    if bad_email:
        return JSONResponse({"error": bad_email}, status_code=400)
    # The limiter protects the bcrypt below; the free validation 400s above
    # shouldn't spend the budget.
    limited = _rate_limited(request)
    if limited is not None:
        return limited
    pw_hash = await _bcrypt(bcrypt.hashpw, password.encode(), bcrypt.gensalt())
    user_id = await db.create_user(username, pw_hash.decode())
    if user_id is None:
        return JSONResponse({"error": "Ce pseudo est déjà pris"}, status_code=409)
    if email and not await db.set_user_email(user_id, email):
        # Adresse déjà prise : le compte est créé, il lui manque son recours.
        # On ne défait pas l'inscription pour ça — on le dit, et l'intéressé
        # corrigera depuis son compte.
        logger.info("compte %s créé sans adresse (déjà utilisée)", user_id)
    _refund(request)
    user = await db.get_user_by_id(user_id)
    response = JSONResponse({"user": _public_user(user)})
    await _start_session(response, request, user_id)
    return response


@router.post("/auth/login")
async def login(request: Request):
    limited = _rate_limited(request)
    if limited is not None:
        return limited
    body = await request.json()
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    user = await db.get_user_by_username(username)
    if user is None:
        # Burn comparable time so missing users aren't detectable by timing.
        await _bcrypt(bcrypt.hashpw, password.encode(), bcrypt.gensalt())
        return JSONResponse({"error": "Pseudo ou mot de passe incorrect"}, status_code=401)
    ok = await _verify_password(user, password)
    if not ok:
        return JSONResponse({"error": "Pseudo ou mot de passe incorrect"}, status_code=401)
    _refund(request)
    response = JSONResponse({"user": _public_user(user)})
    await _start_session(response, request, user["id"])
    return response


@router.post("/auth/logout")
async def logout(request: Request):
    token = request.cookies.get(COOKIE_NAME)
    if token:
        await db.delete_session(_hash_token(token))
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@router.post("/auth/password")
async def change_password(request: Request):
    """Changer son mot de passe, en connaissant l'actuel.

    Le mot de passe actuel est redemandé bien qu'on soit connecté : le cookie
    prouve qu'une session a été ouverte, pas qu'on est encore devant l'écran.
    C'est ce qui rend un poste laissé déverrouillé inoffensif.
    """
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    body = await request.json()
    new_password = body.get("new_password") or ""
    bad = _check_password_rules(new_password)
    if bad:
        return JSONResponse({"error": bad}, status_code=400)
    limited = _rate_limited(request)
    if limited is not None:
        return limited
    if not await _verify_password(user, body.get("current_password")):
        return JSONResponse({"error": "Mot de passe actuel incorrect"},
                            status_code=403)
    _refund(request)
    await _set_password(user["id"], new_password)
    # Les autres sessions tombent : une session ouverte ailleurs est exactement
    # ce dont on veut se débarrasser en changeant de mot de passe. La sienne
    # survit, sinon on se déconnecterait en se protégeant.
    token = request.cookies.get(COOKIE_NAME)
    await db.delete_user_sessions(
        user["id"], keep_token_hash=_hash_token(token) if token else None)
    return JSONResponse({"ok": True})


@router.post("/auth/email")
async def change_email(request: Request):
    """Renseigner, changer ou retirer l'adresse de son compte.

    Une chaîne vide retire l'adresse — et donc le recours ; c'est un choix
    qu'on doit pouvoir faire, l'interface se charge de le dire.
    """
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    body = await request.json()
    email, bad = _normalize_email(body.get("email"))
    if bad:
        return JSONResponse({"error": bad}, status_code=400)
    limited = _rate_limited(request)
    if limited is not None:
        return limited
    if not await _verify_password(user, body.get("password")):
        return JSONResponse({"error": "Mot de passe incorrect"}, status_code=403)
    _refund(request)
    if not await db.set_user_email(user["id"], email):
        return JSONResponse({"error": "Cette adresse est déjà utilisée"},
                            status_code=409)
    return JSONResponse({"ok": True, "email": email})


@router.post("/auth/forgot")
async def forgot(request: Request):
    """Demander un lien de réinitialisation. **Répond toujours la même chose.**

    Pseudo inconnu, compte sans adresse, SMTP en panne : même corps, même code.
    Sinon ce formulaire public devient un annuaire — « ce pseudo existe », puis
    « voici son adresse ». Ce que l'appelant apprend est donc strictement : la
    demande a été prise en compte.

    Sans SMTP configuré, `mail.send` écrit le lien au journal (cf. `mail`) : la
    réinitialisation reste utilisable de bout en bout en développement.
    """
    limited = _rate_limited(request)
    if limited is not None:
        return limited
    body = await request.json()
    identifier = (body.get("identifier") or "").strip()
    answer = JSONResponse({"ok": True, "sent": True})

    user = None
    if identifier:
        user = await db.get_user_by_username(identifier)
        if user is None and "@" in identifier:
            user = await db.get_user_by_email(identifier.lower())
    if user is None or not user.get("email"):
        logger.info("réinitialisation demandée sans suite (identifiant inconnu "
                    "ou compte sans adresse)")
        return answer

    token = secrets.token_urlsafe(32)
    expires = datetime.now(timezone.utc) + timedelta(hours=RESET_HOURS)
    await db.create_password_reset(_hash_token(token), user["id"],
                                   expires.isoformat())
    link = f"{_public_base(request)}/mot-de-passe/nouveau?token={token}"
    subject, text = mail.reset_email(user["username"], link, RESET_HOURS)
    # Dans un thread : `smtplib` est bloquant, et l'event loop fait tourner les
    # tables pendant ce temps-là.
    await asyncio.to_thread(mail.send, user["email"], subject, text)
    return answer


@router.post("/auth/reset")
async def reset(request: Request):
    """Poser un nouveau mot de passe à partir d'un jeton reçu par courriel."""
    limited = _rate_limited(request)
    if limited is not None:
        return limited
    body = await request.json()
    token = (body.get("token") or "").strip()
    new_password = body.get("new_password") or ""
    bad = _check_password_rules(new_password)
    if bad:
        return JSONResponse({"error": bad}, status_code=400)

    token_hash = _hash_token(token) if token else ""
    user = await db.get_password_reset(token_hash)
    if user is None:
        return JSONResponse(
            {"error": "Lien invalide ou expiré — redemandez-en un"},
            status_code=400)
    # Consommer d'abord : le UPDATE conditionnel est ce qui rend l'usage unique
    # atomique. Deux requêtes sur le même lien, une seule passe.
    if not await db.consume_password_reset(token_hash):
        return JSONResponse(
            {"error": "Lien invalide ou expiré — redemandez-en un"},
            status_code=400)
    _refund(request)
    await _set_password(user["id"], new_password)
    # Toutes les sessions tombent, la nouvelle comprise : quelqu'un a peut-être
    # pris le compte, c'est le moment de le mettre dehors partout.
    await db.delete_user_sessions(user["id"])
    response = JSONResponse({"ok": True, "user": _public_user(user)})
    await _start_session(response, request, user["id"])
    return response


@router.post("/account/delete")
async def delete_account(request: Request):
    """Supprimer son compte. Les donnes restent, détachées de lui.

    Une donne de salon appartient à quatre joueurs : l'effacer prendrait la
    partie des trois autres avec elle (cf. `db.delete_account`). Le mot de passe
    est redemandé — c'est irréversible.
    """
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    body = await request.json()
    limited = _rate_limited(request)
    if limited is not None:
        return limited
    if not await _verify_password(user, body.get("password")):
        return JSONResponse({"error": "Mot de passe incorrect"}, status_code=403)
    # Garde-fou contre le clic malheureux : on redemande le pseudo en clair.
    if (body.get("confirm") or "").strip().lower() != user["username"].lower():
        return JSONResponse(
            {"error": "Saisissez votre pseudo pour confirmer"}, status_code=400)
    _refund(request)
    await db.delete_account(user["id"])
    logger.info("compte %s supprimé (donnes anonymisées)", user["id"])
    response = JSONResponse({"ok": True})
    response.delete_cookie(COOKIE_NAME, path="/")
    return response


@router.get("/me")
async def me(request: Request):
    user = await current_user(request)
    if user is None:
        return JSONResponse({"user": None})
    stats = await db.user_game_stats(user["id"])
    # `standing` = `get_rating` plus de quoi expliquer une absence du tableau
    # (`ranked`, `remaining`) : un joueur sous le seuil doit savoir pourquoi.
    stats["elo"] = await elo.standing("user", user["id"])
    return JSONResponse({"user": _public_user(user), "stats": stats})


@router.get("/me/games")
async def my_games(request: Request, limit: int = 50, offset: int = 0):
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    games = await db.list_games(limit=min(limit, 200), offset=offset, user_id=user["id"])
    return JSONResponse(games)


def _scope(request):
    """Le périmètre demandé, ramené à une valeur connue.

    Vérifié ici plutôt qu'en aval : `stats` construit du SQL avec, donc une
    valeur inattendue ne doit jamais y arriver. `stats._seats` retombe aussi sur
    « all » de son côté — deux gardes valent mieux qu'une pour un littéral.
    """
    want = (request.query_params.get("scope") or "all").strip()
    return want if want in stats.SCOPES else "all"


@router.get("/me/stats")
async def my_stats(request: Request):
    """Le portrait chiffré du joueur connecté — taux, moyennes, intervalles.

    Séparé de `/me`, que toutes les pages appellent au chargement : ces
    agrégats parcourent toutes les donnes du joueur et n'ont d'utilité que sur
    /stats. Les faire payer à chaque navigation serait une taxe permanente
    pour un affichage occasionnel.

    Ne calcule **rien** de coûteux : que du SQL sur des colonnes déjà écrites.
    Le jeu comparé à l'oracle est ailleurs, derrière un bouton.
    """
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    return JSONResponse(await stats.my_stats(user["id"], _scope(request)))


@router.get("/me/oracle")
async def my_oracle(request: Request):
    """Ce que le solveur dit du jeu du joueur, **d'après ce qui est en cache**.

    Ne lance aucun calcul : un GET qui déclencherait des recherches DD serait
    déclenché par un préchargeur de navigateur ou un robot. Le POST le fait,
    lui, et seulement sur demande explicite.
    """
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    blob = await stats.oracle_stats(user["id"], _scope(request))
    blob["job"] = _job_view(analysis.bulk_status(user["id"]))
    return JSONResponse(blob)


@router.post("/me/oracle")
async def start_my_oracle(request: Request):
    """Lancer l'analyse des donnes du joueur qui n'en ont pas encore.

    Idempotent tant qu'un balayage tourne : rappeler rend l'état courant au
    lieu d'en démarrer un second. Un seul balayage tourne sur tout le serveur
    (`analysis._bulk_gate`) — ce sont les recherches les plus chères du site,
    et elles ne doivent pas prendre les cœurs des gens qui jouent.
    """
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    pending = await stats.unanalysed_games(user["id"], _scope(request))
    # Les chemins de modèles vivent dans `server`, qui importe `auth` : on les
    # lit tardivement pour ne pas créer de cycle à l'import.
    from colver.web import server as _server
    job = await analysis.bulk(
        user["id"], pending,
        bid_model_path=_server.BID_MODEL_PATH,
        playgen_model_path=_server.PLAYGEN_MODEL_PATH)
    return JSONResponse(_job_view(job))


def _job_view(job):
    """L'état d'un balayage, sans la tâche asyncio (non sérialisable)."""
    if not job:
        return None
    return {k: v for k, v in job.items() if k != "task"}


# Les exercices dont le progrès est suivi côté serveur. Liste fermée : la
# `variant` est libre (elle vient du préréglage choisi), mais l'`exercise` est
# ce qui décide de quel écran on parle, et un client ne doit pas pouvoir en
# inventer — ce serait une table à clés arbitraires, remplissable à volonté.
_EXERCISES = {"compter"}


@router.get("/me/exercises")
async def my_exercises(request: Request, exercise: str = "compter"):
    """Le progrès du joueur connecté sur un exercice.

    Anonyme, la réponse est vide plutôt qu'une erreur : la page marche sans
    compte, elle retombe simplement sur son localStorage. Un 401 ferait écrire
    une gestion d'erreur à un appelant pour qui ce n'est pas une erreur.
    """
    if exercise not in _EXERCISES:
        return JSONResponse({"error": "Exercice inconnu"}, status_code=400)
    user = await current_user(request)
    if user is None:
        return JSONResponse({"stats": {}, "synced": False})
    return JSONResponse({"stats": await db.exercise_stats(user["id"], exercise),
                         "synced": True})


@router.post("/me/exercises")
async def record_exercise(request: Request):
    """Enregistrer **un** essai. Le corps porte un delta, jamais un total.

    Voir la migration v19 : un total poussé par le client écraserait le travail
    d'un autre onglet et ferait de lui l'autorité sur son propre record.
    """
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    body = await request.json()
    exercise = str(body.get("exercise") or "")
    if exercise not in _EXERCISES:
        return JSONResponse({"error": "Exercice inconnu"}, status_code=400)
    variant = str(body.get("variant") or "")[:64]
    if not variant:
        return JSONResponse({"error": "Variante manquante"}, status_code=400)
    try:
        # Borné : ces deux nombres viennent du client et alimentent des
        # compteurs cumulés. L'écart d'un essai ne peut pas dépasser le total de
        # points d'une donne, une série ne peut pas dépasser le nombre d'essais.
        delta = max(0, min(504, int(body.get("delta", 0))))
        streak = max(0, min(100_000, int(body.get("streak", 0))))
    except (TypeError, ValueError):
        return JSONResponse({"error": "Requête invalide"}, status_code=400)
    await db.record_exercise_attempt(
        user["id"], exercise, variant,
        delta=delta, exact=bool(body.get("exact")), streak=streak)
    return JSONResponse(
        {"stats": await db.exercise_stats(user["id"], exercise)})


@router.get("/me/matches")
async def my_matches(request: Request, status: str = "open",
                     limit: int = 20, offset: int = 0):
    """Les parties en 1000 / 2000 points d'un joueur.

    `status=open` (le défaut) : celles laissées en plan, à reprendre.
    `status=done` : celles qui sont allées au bout, abandons compris.

    Le défaut reste « en cours » parce que deux appelants existaient avant
    l'historique — la carte « Parties en cours » de /compte et le bandeau de
    reprise de Jouer — et qu'ils n'ont pas à apprendre qu'un paramètre est
    apparu.
    """
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    if status == "done":
        return JSONResponse(await db.list_matches(
            user["id"], limit=min(max(limit, 1), 100), offset=max(offset, 0)))
    return JSONResponse(await db.list_open_matches(user["id"]))
