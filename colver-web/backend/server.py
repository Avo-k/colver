"""FastAPI server for Colver web UI."""

import asyncio
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from game_manager import PlaySession, WatchSession, ReplaySession, AnalysisSession
import database as db

# Base path for reverse proxy deployment (e.g. ROOT_PATH=/colver/)
ROOT_PATH = os.environ.get("ROOT_PATH", "/")
if not ROOT_PATH.endswith("/"):
    ROOT_PATH += "/"

app = FastAPI(title="Colver")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# DouDou35 model path (Rust inference, no PyTorch needed)
DMC_MODEL_PATH = os.path.join(REPO_ROOT, "models", "dmc_35.bin")
doudou_available = os.path.exists(DMC_MODEL_PATH)
if doudou_available:
    print(f"[server] DouDou35 model available at {DMC_MODEL_PATH} (Rust inference)")
else:
    print(f"[server] No DouDou35 model at {DMC_MODEL_PATH}")

print(f"[server] ROOT_PATH={ROOT_PATH}")


@app.get("/")
async def index():
    html_path = os.path.join(FRONTEND_DIR, "index.html")
    with open(html_path) as f:
        html = f.read()
    # Inject the correct base href for reverse proxy support
    html = html.replace('<base href="/">', f'<base href="{ROOT_PATH}">')
    return HTMLResponse(html)


CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images", "cards")

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
    analysis_session = None
    play_game_id = None
    watch_game_id = None

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "start_game":
                human_seat = data.get("human_seat", 2)
                opponent_ai = data.get("opponent_ai", "doudou")
                partner_ai = data.get("partner_ai", "doudou")
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
                await _run_ai_turns(ws, play_session, human_seat, play_game_id)

            elif msg_type == "play":
                if play_session is None:
                    await ws.send_json({"type": "error", "msg": "No game in progress"})
                    continue
                action = data["action"]
                human_seat = data.get("human_seat", 2)
                state = play_session.play_action(action)
                msg = {"type": "game_state", "state": state}
                if play_session._belote_event:
                    msg["belote_event"] = play_session._belote_event
                    msg["belote_player"] = play_session._belote_player
                await ws.send_json(msg)

                # Save action to DB
                if play_game_id:
                    entry = play_session.history[-1]
                    await db.append_action(play_game_id, entry)

                # Check terminal after human move
                if play_game_id and play_session.env.is_terminal():
                    await _complete_game(play_game_id, play_session)

                await _run_ai_turns(ws, play_session, human_seat, play_game_id)

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

            elif msg_type == "setup_analysis":
                hands = data["hands"]
                contract = data["contract"]
                dealer = data.get("dealer", 0)
                analysis_session = AnalysisSession()
                state = analysis_session.setup(dealer, hands, contract)
                await ws.send_json({
                    "type": "analysis_ready",
                    "state": state,
                })

            elif msg_type == "analyze":
                if analysis_session is None:
                    await ws.send_json({"type": "error", "msg": "No analysis position"})
                    continue
                agent = data.get("agent", "naive")
                time_ms = data.get("time_ms", 200)
                result = analysis_session.analyze(agent=agent, time_ms=time_ms)
                await ws.send_json({
                    "type": "analysis_result",
                    **result,
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


async def _run_ai_turns(ws, session, human_seat, game_id=None):
    """Auto-play AI turns until human's turn or game over."""
    while not session.env.is_terminal() and session.env.current_player() != human_seat:
        await asyncio.sleep(0.3)
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
        await ws.send_json({"type": "game_state", "state": state})

        # Save action to DB
        if game_id:
            await db.append_action(game_id, session.history[-1])

    # Check terminal after AI turns
    if game_id and session.env.is_terminal():
        await _complete_game(game_id, session)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
