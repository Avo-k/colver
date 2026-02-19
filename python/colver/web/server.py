"""FastAPI server for Colver web UI."""

import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from colver.web.game_manager import PlaySession, WatchSession, ReplaySession
import colver.web.database as db

# Base path for reverse proxy deployment (e.g. ROOT_PATH=/colver/)
ROOT_PATH = os.environ.get("ROOT_PATH", "/")
if not ROOT_PATH.endswith("/"):
    ROOT_PATH += "/"

app = FastAPI(title="Colver")

# Locate static assets bundled in the package
_WEB_DIR = os.path.dirname(__file__)
FRONTEND_DIR = os.path.join(_WEB_DIR, "static")
CARDS_DIR = os.path.join(_WEB_DIR, "cards")

# DouDou35 model path (Rust inference, no PyTorch needed)
import colver as _colver_pkg

_model = _colver_pkg.model_path()
if _model is None:
    # Auto-download model if not found
    print("[server] No DouDou35 model found, downloading...")
    try:
        _model = _colver_pkg.download_model()
    except Exception as e:
        print(f"[server] Download failed: {e}")
        _model = None

DMC_MODEL_PATH = str(_model) if _model else None
doudou_available = _model is not None
if doudou_available:
    print(f"[server] DouDou35 model available at {DMC_MODEL_PATH} (Rust inference)")
else:
    print("[server] No DouDou35 model found and download failed")

print(f"[server] ROOT_PATH={ROOT_PATH}")


@app.get("/")
async def index():
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(html_path) as f:
        html = f.read()
    # Inject the correct base href for reverse proxy support
    html = html.replace('<base href="/">', f'<base href="{ROOT_PATH}">')
    return HTMLResponse(html)


app.mount("/cards", StaticFiles(directory=CARDS_DIR), name="cards")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


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


# ===== WebSocket =====

@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    play_session = None
    watch_session = None
    replay_session = None
    play_game_id = None
    watch_game_id = None
    play_move_delay = 2.0

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "start_game":
                human_seat = data.get("human_seat", 2)
                opponent_ai = data.get("opponent_ai", "doudou")
                partner_ai = data.get("partner_ai", "doudou")
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
                play_session = PlaySession(ai_types=ai_types, human_seat=human_seat, dmc_model_path=dmc_path)

                # Save game to DB
                agents_map = {str(s): t for s, t in ai_types.items()}
                agents_map[str(human_seat)] = "human"
                play_game_id = await db.create_game(
                    mode="play",
                    dealer=int(play_session.env.get_dealer()),
                    hands=play_session.env.get_hands(),
                    agents=agents_map,
                    human_seat=human_seat,
                )

                await ws.send_json({
                    "type": "game_state",
                    "state": play_session.get_state(human_seat),
                    "doudou_available": doudou_available,
                    "game_id": play_game_id,
                })
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
                    snapshot_msg = dict(msg)
                    snapshot_msg["state"] = snapshot_state
                    await ws.send_json(snapshot_msg)
                    if play_game_id:
                        await db.append_action(play_game_id, play_session.history[-1])
                    if play_game_id and play_session.env.is_terminal():
                        await _complete_game(play_game_id, play_session)
                    await asyncio.sleep(play_move_delay)
                    await ws.send_json({"type": "game_state", "state": state})
                else:
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
                watch_session = WatchSession(
                    agents=agents,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
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

                await ws.send_json({
                    "type": "watch_started",
                    "state": watch_session.get_state(),
                    "doudou_available": doudou_available,
                    "bid_history": [],
                    "completed_tricks": [],
                    "game_id": watch_game_id,
                })

            elif msg_type == "watch_step":
                if watch_session is None:
                    await ws.send_json({"type": "error", "msg": "No watch session"})
                    continue
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
                game_data = await db.get_game(game_id)
                if not game_data:
                    await ws.send_json({"type": "error", "msg": f"Partie '{game_id}' introuvable"})
                    continue
                replay_session = ReplaySession(game_data)
                watch_session = None
                await ws.send_json({
                    "type": "replay_loaded",
                    "state": replay_session.get_state(),
                    "game_id": game_id,
                    "mode": game_data["mode"],
                    "agents": game_data["agents"],
                    "total_actions": len(game_data["actions"]),
                    "bid_history": [],
                    "completed_tricks": [],
                })

            elif msg_type == "replay_step":
                if replay_session is None:
                    await ws.send_json({"type": "error", "msg": "No replay session"})
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
                await ws.send_json(replay_msg)

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
                agents = data.get("agents", {0: "doudou", 1: "doudou", 2: "doudou", 3: "doudou"})
                agents = {int(k): v for k, v in agents.items()}
                watch_session = WatchSession(
                    agents=agents,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    env=cfn_env,
                )
                replay_session = None
                watch_game_id = None

                await ws.send_json({
                    "type": "watch_started",
                    "state": watch_session.get_state(),
                    "doudou_available": doudou_available,
                    "bid_history": [],
                    "completed_tricks": [],
                    "game_id": watch_game_id,
                })

            elif msg_type == "watch_custom":
                game_id = data.get("game_id", "").strip().lower()
                game_data = await db.get_game(game_id)
                if not game_data:
                    await ws.send_json({"type": "error", "msg": f"Partie '{game_id}' introuvable"})
                    continue
                agents = {int(k): v for k, v in game_data["agents"].items()}
                watch_session = WatchSession(
                    agents=agents,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    dealer=game_data["dealer"],
                    hands=game_data["hands"],
                )
                replay_session = None
                watch_game_id = game_id

                await ws.send_json({
                    "type": "watch_started",
                    "state": watch_session.get_state(),
                    "doudou_available": doudou_available,
                    "bid_history": [],
                    "completed_tricks": [],
                    "game_id": watch_game_id,
                })

            else:
                await ws.send_json({"type": "error", "msg": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        pass


async def _complete_game(game_id, session):
    """Mark a game as complete in the database."""
    points = list(session.env.get_points())
    contract = session.env.get_contract()
    await db.complete_game(game_id, points[0], points[1], contract)


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
            await ws.send_json({"type": "game_state", "state": snapshot})
            await asyncio.sleep(move_delay)
            # Send cleared state — no delay after (next card arrives immediately)
            await ws.send_json({"type": "game_state", "state": state})
        else:
            await ws.send_json({"type": "game_state", "state": state})
            await asyncio.sleep(move_delay)

    # Check terminal after AI turns
    if game_id and session.env.is_terminal():
        await _complete_game(game_id, session)
