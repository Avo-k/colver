"""Account management: register / login / logout with DB-backed session cookies.

Sessions are random tokens stored hashed (sha256) in SQLite, so a DB leak
doesn't expose usable cookies and individual sessions can be revoked.
"""

import asyncio
import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone

import bcrypt
from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse

import colver.web.database as db

router = APIRouter(prefix="/api")

COOKIE_NAME = "colver_session"
SESSION_DAYS = 30
USERNAME_RE = re.compile(r"^[A-Za-z0-9_-]{3,20}$")
MIN_PASSWORD_LEN = 8


def _hash_token(token):
    return hashlib.sha256(token.encode()).hexdigest()


def _is_secure(request):
    # Behind Caddy/Cloudflare the app sees plain HTTP; trust the proxy header.
    proto = request.headers.get("x-forwarded-proto", request.url.scheme)
    return proto == "https"


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
    pw_hash = await asyncio.to_thread(
        bcrypt.hashpw, password.encode(), bcrypt.gensalt()
    )
    user_id = await db.create_user(username, pw_hash.decode())
    if user_id is None:
        return JSONResponse({"error": "Ce pseudo est déjà pris"}, status_code=409)
    user = await db.get_user_by_id(user_id)
    response = JSONResponse({"user": _public_user(user)})
    await _start_session(response, request, user_id)
    return response


@router.post("/auth/login")
async def login(request: Request):
    body = await request.json()
    username = (body.get("username") or "").strip()
    password = body.get("password") or ""
    user = await db.get_user_by_username(username)
    if user is None:
        # Burn comparable time so missing users aren't detectable by timing.
        await asyncio.to_thread(
            bcrypt.hashpw, password.encode(), bcrypt.gensalt()
        )
        return JSONResponse({"error": "Pseudo ou mot de passe incorrect"}, status_code=401)
    ok = await asyncio.to_thread(
        bcrypt.checkpw, password.encode(), user["password_hash"].encode()
    )
    if not ok:
        return JSONResponse({"error": "Pseudo ou mot de passe incorrect"}, status_code=401)
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
    return JSONResponse({"user": _public_user(user), "stats": stats})


@router.get("/me/games")
async def my_games(request: Request, limit: int = 50, offset: int = 0):
    user = await current_user(request)
    if user is None:
        return JSONResponse({"error": "Non connecté"}, status_code=401)
    games = await db.list_games(limit=min(limit, 200), offset=offset, user_id=user["id"])
    return JSONResponse(games)
