"""FastAPI server for Colver web UI."""

import asyncio
import os

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.staticfiles import StaticFiles
from fastapi.responses import HTMLResponse, JSONResponse

from colver.web.game_manager import PlaySession, WatchSession, ReplaySession, BidProblemSession, PlayProblemSession
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

# DouDou27 model path (Rust inference, no PyTorch needed)
import colver as _colver_pkg

_model = _colver_pkg.model_path()
if _model is None:
    # Auto-download model if not found
    print("[server] No DouDou27 model found, downloading...")
    try:
        _model = _colver_pkg.download_model()
    except Exception as e:
        print(f"[server] Download failed: {e}")
        _model = None

DMC_MODEL_PATH = str(_model) if _model else None
doudou_available = _model is not None
if doudou_available:
    print(f"[server] DouDou27 model available at {DMC_MODEL_PATH} (Rust inference)")
else:
    print("[server] No DouDou27 model found and download failed")

# Bid NN model path (DD-trained bidder)
_bid_model = _colver_pkg.bid_model_path()
if _bid_model is None:
    try:
        _bid_model = _colver_pkg.download_bid_model()
    except Exception as e:
        print(f"[server] Bid model download failed: {e}")
        _bid_model = None

BID_MODEL_PATH = str(_bid_model) if _bid_model else None
if _bid_model:
    print(f"[server] Bid à DD model available at {BID_MODEL_PATH}")
else:
    print("[server] No Bid à DD model found, using improved_v2 fallback")

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
    bid_problem_session = None
    play_problem_session = None
    play_game_id = None
    watch_game_id = None
    play_move_delay = 2.0

    try:
        while True:
            data = await ws.receive_json()
            msg_type = data.get("type")

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
                difficulty = data.get("difficulty", "difficile")
                play_session = PlaySession(ai_types=ai_types, human_seat=human_seat, dmc_model_path=dmc_path, bid_model_path=bid_path, difficulty=difficulty)

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
                difficulty = data.get("difficulty", "difficile")
                watch_session = WatchSession(
                    agents=agents,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    bid_model_path=BID_MODEL_PATH,
                    difficulty=difficulty,
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
                agents = data.get("agents", {0: "dede", 1: "dede", 2: "dede", 3: "dede"})
                agents = {int(k): v for k, v in agents.items()}
                watch_session = WatchSession(
                    agents=agents,
                    dmc_model_path=DMC_MODEL_PATH if doudou_available else None,
                    bid_model_path=BID_MODEL_PATH,
                    env=cfn_env,
                    difficulty=data.get("difficulty", "difficile"),
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
                    for action in prior_actions:
                        env.step(action)
                    result = env.action_bid_nn()
                    await ws.send_json({
                        "type": "bid_eval_result",
                        "q_values": [[int(a), round(float(q), 3)] for a, q in result["q_values"]],
                        "best_action": int(result["best_action"]),
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
                    dealer=game_data["dealer"],
                    hands=game_data["hands"],
                    difficulty=data.get("difficulty", "difficile"),
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

            else:
                await ws.send_json({"type": "error", "msg": f"Unknown type: {msg_type}"})

    except WebSocketDisconnect:
        pass


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
