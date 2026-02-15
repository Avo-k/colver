"""FastAPI server for Colver web UI."""

import asyncio
import json
import os
import sys

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse

# Add backend dir to path
sys.path.insert(0, os.path.dirname(__file__))
REPO_ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
from game_manager import PlaySession, ReplaySession, AnalysisSession

app = FastAPI(title="Colver")

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")


@app.get("/")
async def index():
    return FileResponse(os.path.join(FRONTEND_DIR, "index.html"))


CARDS_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "images", "cards")

app.mount("/cards", StaticFiles(directory=CARDS_DIR), name="cards")
app.mount("/static", StaticFiles(directory=FRONTEND_DIR), name="static")


@app.websocket("/ws")
async def websocket_endpoint(ws: WebSocket):
    await ws.accept()
    play_session = None
    replay_session = None
    analysis_session = None

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

            if msg_type == "start_game":
                ai = data.get("ai", "smart")
                time_ms = data.get("time_ms", 50)
                human_seat = data.get("human_seat", 2)
                play_session = PlaySession(ai_type=ai, time_ms=time_ms)

                # Send initial state
                await ws.send_json({
                    "type": "game_state",
                    "state": play_session.get_state(human_seat),
                })

                # Auto-play AI turns until it's human's turn or game over
                await _run_ai_turns(ws, play_session, human_seat)

            elif msg_type == "play":
                if play_session is None:
                    await ws.send_json({"type": "error", "msg": "No game in progress"})
                    continue
                action = data["action"]
                human_seat = data.get("human_seat", 2)
                state = play_session.play_action(action)
                await ws.send_json({"type": "game_state", "state": state})

                # Auto-play AI turns
                await _run_ai_turns(ws, play_session, human_seat)

            elif msg_type == "load_replay":
                log_data = data.get("log")
                if log_data is None:
                    await ws.send_json({"type": "error", "msg": "No log data"})
                    continue
                replay_session = ReplaySession(log_data)
                await ws.send_json({
                    "type": "replay_loaded",
                    "total_steps": replay_session.total_steps,
                    "state": replay_session.get_state(0),
                    "actions": log_data["actions"],
                })

            elif msg_type == "replay_seek":
                if replay_session is None:
                    await ws.send_json({"type": "error", "msg": "No replay loaded"})
                    continue
                step = data["step"]
                await ws.send_json({
                    "type": "replay_state",
                    "step": step,
                    "state": replay_session.get_state(step),
                })

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

            elif msg_type == "generate_replay":
                # Generate a game using AI vs AI for replay
                log = _generate_game_log(
                    ai=data.get("ai", "naive"),
                    time_ms=data.get("time_ms", 20),
                )
                await ws.send_json({"type": "generated_replay", "log": log})

            else:
                await ws.send_json({"type": "error", "msg": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        pass


async def _run_ai_turns(ws, session, human_seat):
    """Auto-play AI turns until human's turn or game over."""
    while not session.env.is_terminal() and session.env.current_player() != human_seat:
        await asyncio.sleep(0.3)  # Small delay for animation
        action, name, state = session.play_ai_turn()
        player = session.history[-1]["player"]
        await ws.send_json({
            "type": "ai_move",
            "player": player,
            "action": action,
            "name": name,
        })
        await ws.send_json({"type": "game_state", "state": state})


def _generate_game_log(ai="naive", time_ms=20):
    """Generate a full game log using AI vs AI."""
    import colver
    env = colver.Env()
    env.reset()
    hands = env.get_hands()
    dealer = int(env.get_dealer())
    actions = []

    if ai == "smart":
        env.smart_ismcts_init()

    while not env.is_terminal():
        player = int(env.current_player())
        phase = int(env.phase())
        if phase == 0:
            action = int(env.bid_improved())
        else:
            if ai == "smart":
                action = int(env.action_smart_ismcts(time_ms))
            else:
                action = int(env.action_naive_ismcts(time_ms))

        actions.append({
            "player": player,
            "action": action,
            "phase": phase,
            "name": colver.Env.action_name(action, phase),
        })

        if ai == "smart":
            env.smart_ismcts_step(action)
        else:
            env.step(action)

    return {
        "dealer": dealer,
        "hands": hands,
        "actions": actions,
        "result": {
            "points": list(env.get_points()),
            "tricks_won": list(env.get_tricks_won()),
            "contract": env.get_contract(),
        },
    }


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
