"""Multiplayer rooms: lobby, per-room game driver, seat-rotated broadcasts.

A Room holds up to 4 seated humans (identified by account) plus bots on the
remaining seats. The driver task advances the game: bots play after a pacing
delay, humans are awaited on an action queue. All game state sent to a viewer
is filtered to their hand and ROTATED so the viewer is always display-seat 2
(South) — this lets the frontend reuse the solo table rendering unchanged.

The DB (games.actions, game_players) stores PHYSICAL seats; rotation happens
only at broadcast time.
"""

import asyncio
import random

import colver.web.database as db
from colver.web.game_manager import PlaySession

ROOM_CODE_ALPHABET = "abcdefghjkmnpqrstuvwxyz23456789"  # no 0/O, 1/l/i
MAX_ROOMS = 20
BOT_TYPES = ("dede", "doudou", "oracle_dd")

ROOMS = {}        # code -> Room
USER_ROOM = {}    # user_id -> code


# ===== Seat rotation =====

def disp_seat(phys, viewer):
    """Physical seat -> display seat (viewer always lands on 2 = South)."""
    return (phys - viewer + 2) % 4


def _phys_seat(display, viewer):
    return (display + viewer - 2) % 4


def _rot_seat_array(arr, viewer):
    return [arr[_phys_seat(d, viewer)] for d in range(4)]


def _rot_team_array(arr, viewer):
    return list(arr) if viewer % 2 == 0 else [arr[1], arr[0]]


def _rot_team(t, viewer):
    return t if viewer % 2 == 0 else 1 - t


def rotate_state(state, viewer):
    """Rotate a get_state() dict into the viewer's frame (viewer = South)."""
    s = dict(state)
    for key in ("hands", "current_trick", "last_trick"):
        if s.get(key):
            s[key] = _rot_seat_array(s[key], viewer)
    for key in ("current_player", "dealer", "trick_lead", "last_trick_winner"):
        if s.get(key) is not None:
            s[key] = disp_seat(s[key], viewer)
    for key in ("points", "tricks_won", "belote", "rewards"):
        if s.get(key):
            s[key] = _rot_team_array(s[key], viewer)
    if s.get("contract") and "team" in s["contract"]:
        contract = dict(s["contract"])
        contract["team"] = _rot_team(contract["team"], viewer)
        s["contract"] = contract
    if s.get("score_detail"):
        sd = dict(s["score_detail"])
        for key in ("trick_points", "belote", "final_scores"):
            if sd.get(key):
                sd[key] = _rot_team_array(sd[key], viewer)
        sd["contract_team"] = _rot_team(sd["contract_team"], viewer)
        s["score_detail"] = sd
    return s


def rotate_bid_history(bid_history, viewer):
    return [{**b, "player": disp_seat(b["player"], viewer)} for b in bid_history]


def rotate_tricks(tricks, viewer):
    out = []
    for t in tricks:
        out.append({
            "cards": _rot_seat_array(t["cards"], viewer),
            "winner": disp_seat(t["winner"], viewer),
            "points": t["points"],
            "lead": disp_seat(t["lead"], viewer) if t.get("lead") is not None else None,
        })
    return out


# ===== Room =====

class Room:
    def __init__(self, code, host_id, models):
        self.code = code
        self.host_id = host_id
        self.models = models          # dict: dmc, bid, belief model paths
        self.members = {}             # user_id -> {"username": str, "ws": ws|None}
        self.seats = [None] * 4       # physical seat -> user_id | None
        self.status = "lobby"         # lobby | playing | finished
        self.bot_type = "dede"
        self.move_delay = 2.0
        self.session = None
        self.game_id = None
        self.task = None
        self.action_queue = asyncio.Queue()
        self.waiting_for = None       # physical seat awaited, or None

    # ----- membership -----

    def seat_of(self, user_id):
        for i, uid in enumerate(self.seats):
            if uid == user_id:
                return i
        return None

    def username(self, user_id):
        m = self.members.get(user_id)
        return m["username"] if m else "?"

    def connected_members(self):
        return [m for m in self.members.values() if m["ws"] is not None]

    # ----- lobby state broadcast -----

    def _lobby_payload(self, for_user_id):
        seats = []
        for uid in self.seats:
            if uid is None:
                seats.append(None)
            else:
                seats.append({
                    "username": self.username(uid),
                    "connected": self.members.get(uid, {}).get("ws") is not None,
                    "is_host": uid == self.host_id,
                })
        return {
            "type": "room_state",
            "code": self.code,
            "status": self.status,
            "seats": seats,
            "you_seat": self.seat_of(for_user_id),
            "is_host": for_user_id == self.host_id,
            "bot_type": self.bot_type,
            "move_delay": self.move_delay,
            "members": [self.username(uid) for uid in self.members],
            "game_id": self.game_id,
        }

    async def _send(self, ws, msg):
        try:
            await ws.send_json(msg)
        except Exception:
            pass  # dead socket — disconnect handling will catch up

    async def broadcast_lobby(self):
        for uid, m in list(self.members.items()):
            if m["ws"] is not None:
                await self._send(m["ws"], self._lobby_payload(uid))

    # ----- game state broadcast -----

    def _viewer_state(self, viewer_seat, terminal=False):
        state = self.session.get_state(viewer_seat)
        # CFN embeds all four hands — never leak it mid-game in multiplayer.
        if not state["is_terminal"]:
            state.pop("cfn", None)
        # Trump suggestion is computed for the current player's decision;
        # only meaningful (and fair) when it's this viewer's turn.
        if state.get("current_player") != viewer_seat:
            state.pop("best_trump_suit", None)
            # Legal actions describe the current player's hand (e.g. which
            # cards can follow suit) — never share them with other seats.
            state["legal_actions"] = []
        return state

    def _seat_names(self, viewer_seat):
        """Display-ordered labels: usernames for humans, bot names for bots."""
        names = []
        for d in range(4):
            p = _phys_seat(d, viewer_seat)
            uid = self.seats[p]
            names.append(self.username(uid) if uid is not None else self.bot_type)
        return names

    async def broadcast_game_state(self, snapshot=False, extra=None):
        for p, uid in enumerate(self.seats):
            if uid is None:
                continue
            ws = self.members.get(uid, {}).get("ws")
            if ws is None:
                continue
            await self._send(ws, self._game_msg(p, snapshot=snapshot, extra=extra))

    def _game_msg(self, viewer_seat, snapshot=False, extra=None):
        state = self._viewer_state(viewer_seat)
        if snapshot:
            # Show the completed trick (4 cards) before it gets cleared:
            # same presentation trick as the solo flow in server.py.
            state = dict(state)
            state["current_trick"] = state["last_trick"]
            tw = list(state["tricks_won"])
            winner_team = state["last_trick_winner"] % 2
            tw[winner_team] = max(0, tw[winner_team] - 1)
            state["tricks_won"] = tw
        rotated = rotate_state(state, viewer_seat)
        msg = {
            "type": "room_game_state",
            "state": rotated,
            "you_seat": viewer_seat,
            "seat_names": self._seat_names(viewer_seat),
            "waiting_for": disp_seat(self.waiting_for, viewer_seat)
            if self.waiting_for is not None else None,
            "game_id": self.game_id,
        }
        if state["is_terminal"]:
            msg["initial_hands"] = _rot_seat_array(
                self.session.initial_hands, viewer_seat)
            msg["bid_history"] = rotate_bid_history(
                self.session.bid_history, viewer_seat)
            msg["completed_tricks"] = rotate_tricks(
                self.session.completed_tricks, viewer_seat)
        else:
            # Live bid history so rejoining clients can rebuild the panel.
            msg["bid_history"] = rotate_bid_history(
                self.session.bid_history, viewer_seat)
        if extra:
            msg.update(extra)
        return msg

    async def send_full_state(self, user_id):
        """Send lobby + (if playing) current game state to one member."""
        m = self.members.get(user_id)
        if not m or m["ws"] is None:
            return
        await self._send(m["ws"], self._lobby_payload(user_id))
        seat = self.seat_of(user_id)
        if self.session is not None and seat is not None and self.status != "lobby":
            await self._send(m["ws"], self._game_msg(seat))

    # ----- game driver -----

    async def start(self):
        humans = [uid for uid in self.seats if uid is not None]
        ai_types = {i: self.bot_type for i, uid in enumerate(self.seats) if uid is None}
        first_human_seat = next(i for i, uid in enumerate(self.seats) if uid is not None)

        loop = asyncio.get_event_loop()
        self.session = await loop.run_in_executor(None, lambda: PlaySession(
            ai_types=ai_types,
            human_seat=first_human_seat,
            dmc_model_path=self.models.get("dmc") if "doudou" in ai_types.values() else None,
            bid_model_path=self.models.get("bid"),
            belief_model_path=self.models.get("belief"),
            dede_time_ms=int(self.move_delay * 1000),
        ))

        agents_map = {}
        for i, uid in enumerate(self.seats):
            agents_map[str(i)] = self.username(uid) if uid is not None else self.bot_type
        self.game_id = await db.create_game(
            mode="multi",
            dealer=int(self.session.env.get_dealer()),
            hands=self.session.env.get_hands(),
            agents=agents_map,
            user_id=self.host_id,
        )
        for i, uid in enumerate(self.seats):
            if uid is not None:
                await db.add_game_player(self.game_id, i, uid)

        self.status = "playing"
        # Drain any stale actions from a previous game in this room
        while not self.action_queue.empty():
            self.action_queue.get_nowait()
        await self.broadcast_lobby()
        self.task = asyncio.create_task(self._drive())

    async def _drive(self):
        session = self.session
        loop = asyncio.get_event_loop()
        try:
            await self.broadcast_game_state()
            while not session.env.is_terminal():
                p = int(session.env.current_player())
                if self.seats[p] is None:
                    # Bot turn: pacing pause, then compute off the event loop
                    await asyncio.sleep(self.move_delay)
                    action, _name, _state = await loop.run_in_executor(
                        None, session.play_ai_turn)
                else:
                    # Human turn: wait for a valid action from that seat
                    self.waiting_for = p
                    action = await self._await_human_action(p)
                    self.waiting_for = None
                    session.play_action(action)
                await db.append_action(self.game_id, session.history[-1])
                if session.env.is_terminal():
                    # Flip status BEFORE the terminal broadcast so an instant
                    # "Revanche" (room_start) isn't rejected as still-playing.
                    points = list(session.env.get_points())
                    await db.complete_game(
                        self.game_id, points[0], points[1],
                        session.env.get_contract())
                    import colver.web.elo as elo
                    await elo.rate_game(self.game_id)
                    self.status = "finished"
                await self._after_action(p, action)

            await self.broadcast_lobby()
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"[room {self.code}] driver crashed: {e!r}")
            self.status = "finished"
            for m in self.connected_members():
                await self._send(m["ws"], {
                    "type": "room_error",
                    "msg": "La partie a rencontré une erreur et a été interrompue.",
                })
            await self.broadcast_lobby()

    async def _await_human_action(self, seat):
        while True:
            user_id, action = await self.action_queue.get()
            if self.seats[seat] != user_id:
                continue  # stale message from another seat
            if action in list(self.session.env.legal_actions()):
                return action
            ws = self.members.get(user_id, {}).get("ws")
            if ws is not None:
                await self._send(ws, {"type": "room_error", "msg": "Coup illégal"})

    async def _after_action(self, actor, action):
        session = self.session
        phase = session.history[-1]["phase"]
        belote = {}
        if session._belote_event:
            belote = {"belote_event": session._belote_event}
        # Move echo (per-viewer rotation of the actor seat)
        for p, uid in enumerate(self.seats):
            if uid is None:
                continue
            ws = self.members.get(uid, {}).get("ws")
            if ws is None:
                continue
            msg = {
                "type": "room_move",
                "player": disp_seat(actor, p),
                "action": int(action),
                "phase": int(phase),
                **belote,
            }
            if belote:
                msg["belote_player"] = disp_seat(session._belote_player, p)
            await self._send(ws, msg)

        if session.trick_just_completed:
            session.trick_just_completed = False
            await self.broadcast_game_state(snapshot=True)
            await asyncio.sleep(self.move_delay)
            await self.broadcast_game_state()
        else:
            await self.broadcast_game_state()

    def stop(self):
        if self.task and not self.task.done():
            self.task.cancel()


# ===== Manager API (called from the ws endpoint) =====

def _gen_code():
    return "".join(random.choice(ROOM_CODE_ALPHABET) for _ in range(4))


async def _leave_current_room(user_id):
    code = USER_ROOM.pop(user_id, None)
    if code is None or code not in ROOMS:
        return
    room = ROOMS[code]
    username = room.username(user_id)
    room.members.pop(user_id, None)
    seat = room.seat_of(user_id)
    if seat is not None:
        if room.status == "playing":
            # A seated player abandoning kills the game (no bot takeover yet).
            room.stop()
            room.status = "finished"
            for m in room.connected_members():
                await room._send(m["ws"], {
                    "type": "room_error",
                    "msg": f"{username} a quitté la partie — partie interrompue",
                })
        room.seats[seat] = None
    if not room.members:
        room.stop()
        del ROOMS[code]
        return
    if room.host_id == user_id:
        room.host_id = next(iter(room.members))
    await room.broadcast_lobby()


async def create_room(user, ws, models):
    if len(ROOMS) >= MAX_ROOMS:
        return None, "Trop de salons ouverts, réessayez plus tard"
    await _leave_current_room(user["id"])
    for _ in range(50):
        code = _gen_code()
        if code not in ROOMS:
            break
    room = Room(code, user["id"], models)
    room.members[user["id"]] = {"username": user["username"], "ws": ws}
    room.seats[2] = user["id"]  # host takes South by default
    ROOMS[code] = room
    USER_ROOM[user["id"]] = code
    await room.broadcast_lobby()
    return room, None


async def join_room(user, ws, code):
    room = ROOMS.get(code.strip().lower())
    if room is None:
        return None, "Salon introuvable"
    if user["id"] in room.members:
        # Reconnection: reattach the socket and resend everything.
        room.members[user["id"]]["ws"] = ws
        USER_ROOM[user["id"]] = room.code
        await room.broadcast_lobby()
        await room.send_full_state(user["id"])
        return room, None
    if room.status == "playing":
        return None, "La partie a déjà commencé"
    await _leave_current_room(user["id"])
    room.members[user["id"]] = {"username": user["username"], "ws": ws}
    USER_ROOM[user["id"]] = room.code
    # Auto-seat on the first free seat
    for i in range(4):
        if room.seats[i] is None:
            room.seats[i] = user["id"]
            break
    await room.broadcast_lobby()
    return room, None


def room_of(user_id):
    code = USER_ROOM.get(user_id)
    return ROOMS.get(code) if code else None


async def handle_message(user, ws, data, models):
    """Route a room_* ws message. Returns True if handled."""
    msg_type = data.get("type", "")
    if not msg_type.startswith("room_"):
        return False
    if user is None:
        await ws.send_json({"type": "room_error",
                            "msg": "Connectez-vous pour jouer en salon"})
        return True

    room = room_of(user["id"])
    # Self-healing: any room message from a member rebinds their socket
    # (covers ws reconnects without an explicit room_join).
    if room is not None and user["id"] in room.members:
        room.members[user["id"]]["ws"] = ws

    if msg_type == "room_status":
        if room is None:
            await ws.send_json({"type": "room_none"})
        else:
            await room.send_full_state(user["id"])
    elif msg_type == "room_create":
        _, err = await create_room(user, ws, models)
        if err:
            await ws.send_json({"type": "room_error", "msg": err})
    elif msg_type == "room_join":
        _, err = await join_room(user, ws, data.get("code", ""))
        if err:
            await ws.send_json({"type": "room_error", "msg": err})
    elif msg_type == "room_leave":
        await _leave_current_room(user["id"])
        await ws.send_json({"type": "room_left"})
    elif room is None:
        await ws.send_json({"type": "room_error", "msg": "Vous n'êtes dans aucun salon"})
    elif msg_type == "room_sit":
        seat = data.get("seat")
        if room.status != "playing" and seat in (0, 1, 2, 3) and room.seats[seat] is None:
            old = room.seat_of(user["id"])
            if old is not None:
                room.seats[old] = None
            room.seats[seat] = user["id"]
            await room.broadcast_lobby()
    elif msg_type == "room_stand":
        seat = room.seat_of(user["id"])
        if room.status != "playing" and seat is not None:
            room.seats[seat] = None
            await room.broadcast_lobby()
    elif msg_type == "room_config":
        if user["id"] == room.host_id and room.status != "playing":
            if data.get("bot_type") in BOT_TYPES:
                room.bot_type = data["bot_type"]
            if "move_delay" in data:
                room.move_delay = max(1.0, min(8.0, float(data["move_delay"])))
            await room.broadcast_lobby()
    elif msg_type == "room_start":
        if user["id"] != room.host_id:
            await ws.send_json({"type": "room_error",
                                "msg": "Seul l'hôte peut lancer la partie"})
        elif room.status == "playing":
            await ws.send_json({"type": "room_error", "msg": "Partie en cours"})
        elif room.seat_of(user["id"]) is None:
            await ws.send_json({"type": "room_error", "msg": "Prenez un siège d'abord"})
        else:
            await room.start()
    elif msg_type == "room_play":
        if room.status != "playing":
            await ws.send_json({"type": "room_error", "msg": "Pas de partie en cours"})
        else:
            try:
                action = int(data.get("action"))
            except (TypeError, ValueError):
                return True
            await room.action_queue.put((user["id"], action))
    return True


async def handle_disconnect(ws):
    """Mark any member using this socket as disconnected."""
    for room in list(ROOMS.values()):
        changed = False
        for uid, m in room.members.items():
            if m["ws"] is ws:
                m["ws"] = None
                changed = True
        if changed:
            if not room.connected_members():
                # Nobody left: drop lobby/finished rooms; keep live games
                # around so players can reconnect.
                if room.status != "playing":
                    room.stop()
                    for uid in list(room.members):
                        USER_ROOM.pop(uid, None)
                    del ROOMS[room.code]
                    continue
            await room.broadcast_lobby()
