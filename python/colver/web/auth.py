"""Account management: register / login / logout with DB-backed session cookies.

Sessions are random tokens stored hashed (sha256) in SQLite, so a DB leak
doesn't expose usable cookies and individual sessions can be revoked.
"""

import asyncio
import hashlib
import re
import secrets
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

import colver.web.database as db
import colver.web.elo as elo
from colver.web.ratelimit import RateLimiter

router = APIRouter(prefix="/api")

COOKIE_NAME = "colver_session"
SESSION_DAYS = 30
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,20}$")
MIN_PASSWORD_LEN = 8

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
    return {"id": user["id"], "username": user["username"], "created_at": user["created_at"]}


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
    if len(password) < MIN_PASSWORD_LEN:
        return JSONResponse(
            {"error": f"Mot de passe trop court ({MIN_PASSWORD_LEN} caractères minimum)"},
            status_code=400,
        )
    # The limiter protects the bcrypt below; the free validation 400s above
    # shouldn't spend the budget.
    limited = _rate_limited(request)
    if limited is not None:
        return limited
    pw_hash = await _bcrypt(bcrypt.hashpw, password.encode(), bcrypt.gensalt())
    user_id = await db.create_user(username, pw_hash.decode())
    if user_id is None:
        return JSONResponse({"error": "Ce pseudo est déjà pris"}, status_code=409)
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
    ok = await _bcrypt(bcrypt.checkpw, password.encode(), user["password_hash"].encode())
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


@router.get("/me")
async def me(request: Request):
    user = await current_user(request)
    if user is None:
        return JSONResponse({"user": None})
    stats = await db.user_game_stats(user["id"])
    stats["elo"] = await elo.get_rating("user", user["id"])
    return JSONResponse({"user": _public_user(user), "stats": stats})


@router.get("/me/games")
async def my_games(request: Request, limit: int = 50, offset: int = 0):
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    games = await db.list_games(limit=min(limit, 200), offset=offset, user_id=user["id"])
    return JSONResponse(games)


@router.get("/me/matches")
async def my_open_matches(request: Request):
    """Les parties en 1000 / 2000 points laissées en plan, à reprendre."""
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    return JSONResponse(await db.list_open_matches(user["id"]))
