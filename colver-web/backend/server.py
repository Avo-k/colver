"""FastAPI server for Colver web UI."""

import asyncio
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, FileResponse

# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from game_manager import PlaySession, WatchSession, AnalysisSession

# Base path for reverse proxy deployment (e.g. ROOT_PATH=/colver/)
ROOT_PATH = os.environ.get("ROOT_PATH", "/")
if not ROOT_PATH.endswith("/"):
    ROOT_PATH += "/"

app = FastAPI(title="Colver")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

# DouDou model path (Rust inference, no PyTorch needed)
DMC_MODEL_PATH = os.path.join(REPO_ROOT, "models", "dmc_final.bin")
doudou_available = os.path.exists(DMC_MODEL_PATH)
if doudou_available:
    print(f"[server] DouDou model available at {DMC_MODEL_PATH} (Rust inference)")
else:
    print(f"[server] No DouDou model at {DMC_MODEL_PATH}")

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


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    play_session = None
    watch_session = None
    analysis_session = None

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "start_game":
                ai = data.get("ai", "smart")
                time_ms = data.get("time_ms", 50)
                human_seat = data.get("human_seat", 2)
                dmc_path = DMC_MODEL_PATH if (doudou_available and ai == "doudou") else None
                play_session = PlaySession(ai_type=ai, time_ms=time_ms, dmc_model_path=dmc_path)

                await ws.send_json({
                    "type": "game_state",
                    "state": play_session.get_state(human_seat),
                    "doudou_available": doudou_available,
                })
                await _run_ai_turns(ws, play_session, human_seat)

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
                await _run_ai_turns(ws, play_session, human_seat)

            elif msg_type == "watch_start":
                agents = data.get("agents", {0: "smart", 1: "smart", 2: "smart", 3: "smart"})
                # Convert string keys from JSON to int
                agents = {int(k): v for k, v in agents.items()}
                time_ms = data.get("time_ms", 50)
                watch_session = WatchSession(
                    agents=agents,
                    time_ms=time_ms,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                )
                await ws.send_json({
                    "type": "watch_started",
                    "state": watch_session.get_state(),
                    "doudou_available": doudou_available,
                    "bid_history": [],
                    "completed_tricks": [],
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
                watch_msg = {
                    "type": "watch_move",
                    "move": move,
                    "state": state,
                    "completed_tricks": tricks,
                    "bid_history": watch_session.bid_history,
                    "finished": watch_session.env.is_terminal(),
                }
                if watch_session._belote_event:
                    watch_msg["belote_event"] = watch_session._belote_event
                    watch_msg["belote_player"] = watch_session._belote_player
                await ws.send_json(watch_msg)

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


async def _run_ai_turns(ws, session, human_seat):
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


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
