"""FastAPI server for Colver web UI."""

import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from colver.web.game_manager import PlaySession, WatchSession, ReplaySession, BidProblemSession, PlayProblemSession, BeliefSession
from colver.web import playgen_gpu as _playgen_gpu
import colver.web.database as db
import colver.web.elo as elo
import colver.web.rooms as rooms
from colver.web.auth import router as auth_router, user_from_cookies

# Base path for reverse proxy deployment (e.g. ROOT_PATH=/colver/)
ROOT_PATH = os.environ.get("ROOT_PATH", "/")
if not ROOT_PATH.endswith("/"):
    ROOT_PATH += "/"

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
    print("[server] No DouDou50 model found, downloading...")
    try:
        _model = _colver_pkg.download_model()
    except Exception as e:
        print(f"[server] Download failed: {e}")
        _model = None

DMC_MODEL_PATH = str(_model) if _model else None
doudou_available = _model is not None
if doudou_available:
    print(f"[server] DouDou50 model available at {DMC_MODEL_PATH} (Rust inference)")
else:
    print("[server] No DouDou50 model found and download failed")

# Bid NN model path
_bid_model = _colver_pkg.bid_model_path()
if _bid_model is None:
    try:
        _bid_model = _colver_pkg.download_bid_model()
    except Exception as e:
        print(f"[server] Bid model download failed: {e}")
        _bid_model = None

BID_MODEL_PATH = str(_bid_model) if _bid_model else None
if _bid_model:
    print(f"[server] Bid model available at {BID_MODEL_PATH}")
else:
    print("[server] No bid model found, using improved_v2 fallback")

# Belief net model path (NN-based card location prediction for IS-DD)
_belief_model = _colver_pkg.belief_model_path()
if _belief_model is None:
    try:
        _belief_model = _colver_pkg.download_belief_model()
    except Exception as e:
        print(f"[server] Belief model download failed: {e}")
        _belief_model = None

BELIEF_MODEL_PATH = str(_belief_model) if _belief_model else None
if _belief_model:
    print(f"[server] Belief net model available at {BELIEF_MODEL_PATH}")
else:
    print("[server] No belief net model found, using heuristic beliefs")

# Playgen world-sampler model (transformer, MC belief marginals)
_playgen_model = _colver_pkg.playgen_model_path()
if _playgen_model is None:
    try:
        _playgen_model = _colver_pkg.download_playgen_model()
    except Exception as e:
        print(f"[server] Playgen model download failed: {e}")
        _playgen_model = None

PLAYGEN_MODEL_PATH = str(_playgen_model) if _playgen_model else None
if _playgen_model:
    print(f"[server] Playgen model available at {PLAYGEN_MODEL_PATH}")
else:
    print("[server] No playgen model found, playgen beliefs disabled")

print(f"[server] ROOT_PATH={ROOT_PATH}")


def _serve_index():
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(html_path) as f:
        html = f.read()
    # Inject the correct base href for reverse proxy support
    html = html.replace('<base href="/">', f'<base href="{ROOT_PATH}">')
    return HTMLResponse(html)


@app.get("/")
async def index():
    return _serve_index()


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
    asyncio.create_task(elo.backfill())


@app.get("/api/leaderboard")
async def api_leaderboard():
    return JSONResponse(await elo.leaderboard())


@app.get("/api/games/{game_id}/analysis")
async def api_game_analysis(game_id: str):
    import colver.web.analysis as analysis
    result, err = await analysis.get_or_compute(game_id, bid_model_path=BID_MODEL_PATH)
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
    game = await db.get_game(game_id)
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
    }


def _doudou_accumulate(cells, stats, dd):
    """Fold one Dédé sim result into the aggregates."""
    if dd["void"]:
        stats["voids"] += 1
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
    num_sims = max(1, min(1000, int(data.get("num_sims", 50))))
    prior_actions_raw = data.get("prior_actions", None)
    prior_actions = [int(a) for a in prior_actions_raw] if prior_actions_raw else []

    if len(hand) != 8:
        await ws.send_json({"type": "annonces_sim_update", "error": "8 cartes requises"})
        return

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
            for _ in range(num_sims):
                all_hands.append(_uniform_hands())
                all_sources.append("uniform")

        # GPU sidecar: one batched call is far cheaper than chunked CPU calls
        # (shared prefill), so generate everything in a single chunk.
        _gpu = _playgen_gpu.enabled()
        PLAYGEN_CHUNK = num_sims if _gpu else 8

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

        window = min(_DD_EXECUTOR._max_workers, num_sims)
        completed = 0
        next_i = 0
        pending = set()
        gen_task = None
        gen_requested = 0
        if playgen_env is not None:
            gen_requested = min(PLAYGEN_CHUNK, num_sims)
            gen_task = loop.run_in_executor(None, _gen_chunk, gen_requested)
        try:
            while completed < num_sims:
                if gen_task is not None and gen_task.done():
                    chunk = gen_task.result()
                    all_hands.extend(chunk)
                    all_sources.extend(["playgen"] * len(chunk))
                    if gen_requested < num_sims:
                        n = min(PLAYGEN_CHUNK, num_sims - gen_requested)
                        gen_requested += n
                        gen_task = loop.run_in_executor(None, _gen_chunk, n)
                    else:
                        gen_task = None
                        # Shortfall (failed generations) → uniform top-up.
                        while len(all_hands) < num_sims:
                            all_hands.append(_uniform_hands())
                            all_sources.append("uniform")
                            worlds_source = "mixte"

                while next_i < min(num_sims, len(all_hands)) and len(pending) < window:
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
                    "type": "annonces_sim_update",
                    "completed": completed, "total": num_sims,
                    "elapsed_ms": round((_time.monotonic() - oracle_start) * 1000, 1),
                    "success_counts": success_counts,
                    "oracle_synth": _oracle_synth(),
                    "worlds_source": worlds_source,
                    "worlds_counts": worlds_counts,
                })
        finally:
            for fut in pending:
                fut.cancel()
            if gen_task is not None:
                gen_task.cancel()

        await ws.send_json({
            "type": "annonces_sim_done",
            "completed": num_sims, "total": num_sims,
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
            doudou_cells = _doudou_new_cells()
            doudou_stats = _doudou_new_stats()
            doudou_start = _time.monotonic()

            for i in range(num_sims):
                dd = await loop.run_in_executor(
                    None, _run_doudou_sim_with_hands, all_hands[i],
                    BID_MODEL_PATH, DMC_MODEL_PATH, dealer, prior_actions)

                _doudou_accumulate(doudou_cells, doudou_stats, dd)

                await ws.send_json({
                    "type": "annonces_doudou_update",
                    "completed": i + 1, "total": num_sims,
                    "elapsed_ms": round((_time.monotonic() - doudou_start) * 1000, 1),
                    "doudou_cells": doudou_cells,
                    "doudou_stats": doudou_stats,
                })

            await ws.send_json({
                "type": "annonces_doudou_done",
                "completed": num_sims, "total": num_sims,
                "elapsed_ms": round((_time.monotonic() - doudou_start) * 1000, 1),
                "doudou_cells": doudou_cells,
                "doudou_stats": doudou_stats,
            })

    except asyncio.CancelledError:
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "annonces_sim_update", "error": str(e)})
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

    if len(hand) != 8:
        await ws.send_json({"type": "annonces_doudou_update", "error": "8 cartes requises"})
        return
    if not BID_MODEL_PATH or not DMC_MODEL_PATH:
        await ws.send_json({"type": "annonces_doudou_update", "error": "Modèles Dédé non disponibles"})
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
                await ws.send_json({"type": "annonces_doudou_update",
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
                "type": "annonces_doudou_update",
                "completed": i + 1, "total": num_sims,
                "elapsed_ms": round((_time.monotonic() - start) * 1000, 1),
                "doudou_cells": doudou_cells,
                "doudou_stats": doudou_stats,
            })

        await ws.send_json({
            "type": "annonces_doudou_done",
            "completed": num_sims, "total": num_sims,
            "elapsed_ms": round((_time.monotonic() - start) * 1000, 1),
            "doudou_cells": doudou_cells,
            "doudou_stats": doudou_stats,
        })

    except asyncio.CancelledError:
        return
    except Exception as e:
        try:
            await ws.send_json({"type": "annonces_doudou_update", "error": str(e)})
        except Exception:
            pass


# ===== WebSocket =====

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
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
    play_move_delay = 2.0
    sim_task = None  # background task for annonces_sim / annonces_doudou
    belief_precompute_task = None  # background playgen precompute sweep
    agent_review_task = None  # background per-card bot review, streamed
    # Starlette WebSockets are NOT safe for concurrent sends: a background task
    # (e.g. the playgen precompute sweep) sending at the same time as the main
    # handler corrupts the ASGI state and raises RuntimeError, killing the
    # socket. Serialize every send through this lock.
    send_lock = asyncio.Lock()

    async def wsend(payload):
        async with send_lock:
            await ws.send_json(payload)

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
            try:
                await wsend({"type": "error", "msg": f"Pré-calcul playgen : {e}"})
            except Exception:
                pass

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type and msg_type.startswith("room_"):
                await rooms.handle_message(ws_user, ws, data, {
                    "dmc": DMC_MODEL_PATH if doudou_available else None,
                    "bid": BID_MODEL_PATH,
                    "belief": BELIEF_MODEL_PATH,
                })
                continue

            if msg_type == "start_game":
                human_seat = data.get("human_seat", 2)
                opponent_ai = data.get("opponent_ai", "dede")
                partner_ai = data.get("partner_ai", "dede")
                play_move_delay = max(1.0, min(8.0, float(data.get("move_delay", 2))))
                # Build per-seat AI mapping (human excluded)
                ai_types = {}
                for seat in range(4):
                    if seat == human_seat:
                        continue
                    if seat == (human_seat ^ 2):  # partner
                        ai_types[seat] = partner_ai
                    else:  # opponents
                        ai_types[seat] = opponent_ai
                needs_dmc = any(t == "doudou" for t in ai_types.values())
                dmc_path = DMC_MODEL_PATH if (doudou_available and needs_dmc) else None
                bid_path = BID_MODEL_PATH
                dede_time_ms = int(play_move_delay * 1000)
                play_session = PlaySession(ai_types=ai_types, human_seat=human_seat, dmc_model_path=dmc_path, bid_model_path=bid_path, belief_model_path=BELIEF_MODEL_PATH, dede_time_ms=dede_time_ms)

                # Save game to DB
                agents_map = {str(s): t for s, t in ai_types.items()}
                agents_map[str(human_seat)] = "human"
                play_game_id = await db.create_game(
                    mode="play",
                    dealer=int(play_session.env.get_dealer()),
                    hands=play_session.env.get_hands(),
                    agents=agents_map,
                    human_seat=human_seat,
                    user_id=ws_user["id"] if ws_user else None,
                )

                init_msg = {
                    "type": "game_state",
                    "state": play_session.get_state(human_seat),
                    "doudou_available": doudou_available,
                    "game_id": play_game_id,
                    "initial_hands": play_session.initial_hands,
                }
                await ws.send_json(init_msg)
                await _run_ai_turns(ws, play_session, human_seat, play_game_id, move_delay=play_move_delay)

            elif msg_type == "play":
                if play_session is None:
                    await ws.send_json({"type": "error", "msg": "No game in progress"})
                    continue
                action = data["action"]
                human_seat = data.get("human_seat", 2)

                # Ignore duplicate clicks when it's not the human's turn
                if play_session.env.current_player() != human_seat:
                    continue

                # Update move delay dynamically from slider
                if "move_delay" in data:
                    play_move_delay = max(1.0, min(8.0, float(data["move_delay"])))
                    play_session.dede_time_ms = int(play_move_delay * 1000)
                    play_session.bots.set_time_ms(play_session.dede_time_ms)

                state = play_session.play_action(action)
                msg = {"type": "game_state", "state": state}
                if play_session._belote_event:
                    msg["belote_event"] = play_session._belote_event
                    msg["belote_player"] = play_session._belote_player

                if play_session.trick_just_completed:
                    play_session.trick_just_completed = False
                    # Show completed trick (4 cards visible), pause, then clear
                    snapshot_state = dict(state)
                    snapshot_state["current_trick"] = state["last_trick"]
                    # tricks_won already incremented; roll back so hand counts stay correct
                    tw = list(snapshot_state["tricks_won"])
                    tw[state["last_trick_winner"] % 2] = max(0, tw[state["last_trick_winner"] % 2] - 1)
                    snapshot_state["tricks_won"] = tw
                    snapshot_msg = dict(msg)
                    snapshot_msg["state"] = snapshot_state
                    await ws.send_json(snapshot_msg)
                    if play_game_id:
                        await db.append_action(play_game_id, play_session.history[-1])
                    if play_game_id and play_session.env.is_terminal():
                        await _complete_game(play_game_id, play_session)
                    await asyncio.sleep(play_move_delay)
                    final_msg = {"type": "game_state", "state": state}
                    _enrich_terminal_msg(final_msg, play_session)
                    await ws.send_json(final_msg)
                else:
                    _enrich_terminal_msg(msg, play_session)
                    await ws.send_json(msg)
                    if play_game_id:
                        await db.append_action(play_game_id, play_session.history[-1])
                    if play_game_id and play_session.env.is_terminal():
                        await _complete_game(play_game_id, play_session)
                    # Pause after human's card is visible (simulates next player thinking)
                    await asyncio.sleep(play_move_delay)

                await _run_ai_turns(ws, play_session, human_seat, play_game_id, move_delay=play_move_delay)

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
                await wsend({
                    "type": "replay_loaded",
                    "state": initial_state,
                    "game_id": game_id,
                    "mode": game_data["mode"],
                    "agents": game_data["agents"],
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

            elif msg_type == "watch_custom":
                game_id = data.get("game_id", "").strip().lower()
                game_data = await db.get_game(game_id)
                if not game_data:
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
                watch_game_id = game_id

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
                with_playgen = bool(data.get("playgen", False))
                loop = asyncio.get_event_loop()
                try:
                    result = await loop.run_in_executor(None, belief_session.get_beliefs, observer, with_playgen)
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
    finally:
        await rooms.handle_disconnect(ws)


def _enrich_terminal_msg(msg, play_session):
    """Add review data (initial hands, bids, tricks) to terminal game_state messages."""
    if play_session.env.is_terminal():
        msg["initial_hands"] = play_session.initial_hands
        msg["bid_history"] = play_session.bid_history
        msg["completed_tricks"] = play_session.completed_tricks
    return msg


async def _complete_game(game_id, session):
    """Mark a game as complete in the database."""
    points = list(session.env.get_points())
    contract = session.env.get_contract()
    await db.complete_game(game_id, points[0], points[1], contract)
    await elo.rate_game(game_id)


async def _run_ai_turns(ws, session, human_seat, game_id=None, move_delay=2.0):
    """Auto-play AI turns until human's turn or game over."""
    while not session.env.is_terminal() and session.env.current_player() != human_seat:
        action, name, state = session.play_ai_turn()
        player = session.history[-1]["player"]
        ai_msg = {
            "type": "ai_move",
            "player": player,
            "action": action,
            "name": name,
        }
        if session._belote_event:
            ai_msg["belote_event"] = session._belote_event
            ai_msg["belote_player"] = session._belote_player
        await ws.send_json(ai_msg)

        if game_id:
            await db.append_action(game_id, session.history[-1])

        if session.trick_just_completed:
            session.trick_just_completed = False
            # Show completed trick (4 cards visible), pause, then clear
            snapshot = dict(state)
            snapshot["current_trick"] = state["last_trick"]
            # tricks_won is already incremented; roll it back so hand counts stay correct
            tw = list(snapshot["tricks_won"])
            winner_team = state["last_trick_winner"] % 2
            tw[winner_team] = max(0, tw[winner_team] - 1)
            snapshot["tricks_won"] = tw
            await ws.send_json({"type": "game_state", "state": snapshot})
            await asyncio.sleep(move_delay)
            # Send cleared state — no delay after (next card arrives immediately)
            final_msg = {"type": "game_state", "state": state}
            _enrich_terminal_msg(final_msg, session)
            await ws.send_json(final_msg)
        else:
            state_msg = {"type": "game_state", "state": state}
            _enrich_terminal_msg(state_msg, session)
            await ws.send_json(state_msg)
            await asyncio.sleep(move_delay)

    # Check terminal after AI turns
    if game_id and session.env.is_terminal():
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


# Catch-all for client-side routes (pushState).
# Must be registered AFTER all API/WS/static mounts.
@app.get("/{full_path:path}")
async def spa_catchall(full_path: str):
    return _serve_index()
