"""Game session management for Colver web UI."""

import random
import time
from collections import OrderedDict
import colver

from colver.web import playgen_gpu
from colver.web import game_notation


# ---- Server-wide belief cache -------------------------------------------------
# Belief weights (NN + playgen) depend only on the game + position + observer, so
# they can be shared across connections/refreshes. Keyed by the full-game CFN
# (which now includes the auction, so `action_idx` is stable across viewers of
# the same game). LRU-capped by number of games; entries are small.
_BELIEF_CACHE_MAX_GAMES = 50
_belief_cache = OrderedDict()  # game_cfn -> {(kind, action_idx, observer): value}


def _belief_cache_get(game_cfn, kind, idx, observer):
    if game_cfn is None:
        return None
    g = _belief_cache.get(game_cfn)
    if g is None:
        return None
    _belief_cache.move_to_end(game_cfn)
    return g.get((kind, idx, observer))


def _belief_cache_put(game_cfn, kind, idx, observer, value):
    if game_cfn is None or value is None:
        return
    g = _belief_cache.get(game_cfn)
    if g is None:
        g = {}
        _belief_cache[game_cfn] = g
        while len(_belief_cache) > _BELIEF_CACHE_MAX_GAMES:
            _belief_cache.popitem(last=False)
    g[(kind, idx, observer)] = value
    _belief_cache.move_to_end(game_cfn)


def compute_game_cfn(dealer, initial_hands, action_ids) -> str:
    """Full-game CFN (auction + play) from a deal and a flat action-id list."""
    env = colver.Env.deal_with_hands(int(dealer), [list(h) for h in initial_hands])
    env.dede_init()
    bids = []
    for a in action_ids:
        a = int(a)
        if int(env.phase()) == 0:
            bids.append(a)
        env.dede_step(a)
    return game_notation.to_full_cfn(env.to_cfn(), bids)


def _safe_playgen(fn, *args, **kwargs):
    """Run a playgen inference call, swallowing Rust panics.

    The pure-Rust playgen sampler can panic on some mid-play positions
    (e.g. an index-out-of-bounds in `infer.rs`). PyO3 surfaces that as
    `pyo3_runtime.PanicException`, which subclasses BaseException — so an
    ordinary `except Exception` misses it and the panic tears down the whole
    WebSocket coroutine (observed as a reconnect loop on the Croyances page).
    Playgen is best-effort, so degrade to None on any failure instead.
    """
    try:
        return fn(*args, **kwargs)
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException as e:  # noqa: BLE001 — includes pyo3 PanicException
        print(f"[belief] playgen call failed ({type(e).__name__}): {e}")
        return None


def _inject_gpu_worlds(env, dealer, initial_hands, history, n_worlds=24):
    """Fetch playgen worlds from the GPU sidecar and inject them into the
    env's IS-DD search for the current player. Silent no-op (False) when the
    sidecar is disabled or unreachable — IS-DD then samples as usual (CPU).

    ``history``: list of {player, action} dicts (bid + play, in order).
    ``initial_hands`` must be the full pre-auction deal (32 cards)."""
    if not playgen_gpu.enabled():
        return False
    if initial_hands is None or sum(len(h) for h in initial_hands) != 32:
        return False
    actions = [(h["player"], h["action"]) for h in history]
    worlds = playgen_gpu.play_worlds(
        dealer, initial_hands, actions, int(env.current_player()), n_worlds=n_worlds
    )
    if not worlds:
        return False
    try:
        env.dede_inject_worlds(worlds)
        return True
    except Exception:
        return False


class TrickTracker:
    """Mixin for tracking completed tricks with point calculations."""

    # Card point values: [7, 8, 9, J, Q, K, 10, A]
    PLAIN_POINTS = [0, 0, 0, 2, 3, 4, 10, 11]
    TRUMP_POINTS = [0, 0, 14, 20, 3, 4, 10, 11]

    def _init_trick_tracking(self):
        self.last_trick = None
        self.last_trick_winner = None
        self.last_trick_points = 0
        self.last_trick_lead = None
        self.completed_tricks = []  # list of {cards, winner, points, lead}
        self._trick_just_completed = False
        self._current_trick_lead = None  # lead of the trick in progress
        self._belote_event = None  # "belote" or "rebelote" after an action
        self._belote_player = None  # seat that triggered it
        self.trick_just_completed = False  # set after each trick, cleared by caller

    def _card_points(self, card_idx, trump_suit):
        suit = card_idx >> 3
        rank = card_idx & 7
        if suit == trump_suit:
            return self.TRUMP_POINTS[rank]
        return self.PLAIN_POINTS[rank]

    def _trick_points(self, trick, trump_suit):
        return sum(self._card_points(c, trump_suit) for c in trick if 0 <= c < 32)

    def _check_trick_completion(self, action):
        self._trick_just_completed = False
        if self.env.phase() != 1:
            return
        trick = self.env.get_current_trick()
        filled = sum(1 for c in trick if c >= 0)
        if filled == 0:
            self._current_trick_lead = int(self.env.current_player())
        if filled == 3:
            player = int(self.env.current_player())
            trick[player] = action
            self.last_trick = trick
            self.last_trick_lead = self._current_trick_lead
            contract = self.env.get_contract()
            trump = contract.get("trump", 0)
            self.last_trick_points = self._trick_points(trick, trump)
            self._trick_just_completed = True

    def _detect_belote(self, player_before, belote_before):
        """Compare belote state before/after an action to detect announcements."""
        belote_after = list(self.env.get_belote())
        self._belote_event = None
        self._belote_player = None
        for team in range(2):
            if belote_after[team] != belote_before[team]:
                self._belote_player = int(player_before)
                if belote_after[team] == 1:
                    self._belote_event = "belote"
                elif belote_after[team] == 2:
                    self._belote_event = "rebelote"
                break

    def _finalize_trick_completion(self):
        if self._trick_just_completed:
            self.last_trick_winner = int(self.env.current_player())
            self.completed_tricks.append({
                "cards": self.last_trick[:],
                "winner": self.last_trick_winner,
                "points": self.last_trick_points,
                "lead": self.last_trick_lead,
            })
            self._trick_just_completed = False
            self.trick_just_completed = True


class PlaySession(TrickTracker):
    """Wraps a colver.Env for human vs AI play."""

    def __init__(self, ai_types=None, human_seat=2, dmc_model_path=None, bid_model_path=None, belief_model_path=None, dede_time_ms=None):
        # ai_types: dict mapping seat -> ai_type (for non-human seats)
        # If not provided, default all AI seats to "dede"
        self.human_seat = human_seat
        if ai_types is None:
            ai_types = {}
        self.ai_types = ai_types
        self.dede_time_ms = dede_time_ms or 1000
        self.env = colver.Env()
        self.history = []
        self.bid_history = []
        self._init_trick_tracking()
        self.env.reset()
        self.initial_hands = [list(h) for h in self.env.get_hands()]
        self.uses_dede = any(t == "dede" for t in self.ai_types.values())
        if belief_model_path and self.uses_dede:
            self.env.load_belief_net(belief_model_path)
        if self.uses_dede:
            self.env.dede_init()
        if dmc_model_path and any(t == "doudou" for t in self.ai_types.values()):
            self.env.load_dmc_model(dmc_model_path)
        if bid_model_path:
            self.env.load_bid_model(bid_model_path)

    def get_state(self, human_seat=2):
        phase = self.env.phase()
        hands = self.env.get_hands()
        hidden_hands = []
        for i, h in enumerate(hands):
            if i == human_seat:
                hidden_hands.append(h)
            else:
                hidden_hands.append([])
        state = {
            "phase": phase,
            "current_player": int(self.env.current_player()),
            "hands": hidden_hands,
            "current_trick": self.env.get_current_trick(),
            "contract": self.env.get_contract(),
            "points": list(self.env.get_points()),
            "tricks_won": list(self.env.get_tricks_won()),
            "legal_actions": list(self.env.legal_actions()) if not self.env.is_terminal() else [],
            "dealer": int(self.env.get_dealer()),
            "trick_lead": int(self.env.get_trick_lead()),
            "is_terminal": self.env.is_terminal(),
            "last_trick": self.last_trick,
            "last_trick_winner": self.last_trick_winner,
            "last_trick_points": self.last_trick_points,
            "belote": list(self.env.get_belote()),
            "cfn": self.env.to_cfn(),
            "rewards": list(self.env.rewards()) if self.env.is_terminal() else None,
        }
        if self.env.is_terminal():
            rewards = state["rewards"]
            contract = self.env.get_contract()
            points = list(self.env.get_points())
            belote = list(self.env.get_belote())
            contract_team = contract.get("team", 0)
            state["score_detail"] = {
                "trick_points": points,
                "belote": [20 if b == 2 else 0 for b in belote],
                "contract_value": contract.get("value", 0),
                "contract_team": contract_team,
                "contract_made": rewards[contract_team] > 0,
                "final_scores": rewards,
            }
        # During bidding, suggest the best trump suit for the human player
        if phase == 0:
            if self.env.has_bid_model():
                nn_result = self.env.action_bid_nn()
                best_action = int(nn_result["best_action"])
                # Extract best suit from NN's top bid action (actions 1-36: suit = (action-1)%4)
                if 1 <= best_action <= 40:
                    state["best_trump_suit"] = (best_action - 1) % 4
                else:
                    # Pass or coinche — fall back to heuristic
                    eval_result = self.env.evaluate_hand(human_seat)
                    state["best_trump_suit"] = int(eval_result["best_suit"])
            else:
                eval_result = self.env.evaluate_hand(human_seat)
                state["best_trump_suit"] = int(eval_result["best_suit"])
        return state

    def play_action(self, action):
        player = self.env.current_player()
        phase = self.env.phase()
        belote_before = list(self.env.get_belote())
        self.history.append({"player": int(player), "action": int(action), "phase": int(phase)})
        if phase == 0:
            name = colver.Env.action_name(int(action), int(phase))
            self.bid_history.append({"player": int(player), "action": int(action), "name": name})
        self._check_trick_completion(action)
        if self.uses_dede:
            self.env.dede_step(action)
        else:
            self.env.step(action)
        self._finalize_trick_completion()
        self._detect_belote(player, belote_before)
        return self.get_state()

    def get_ai_action(self):
        phase = self.env.phase()
        player = int(self.env.current_player())
        ai_type = self.ai_types.get(player, "dede")
        if phase == 0:
            return int(self.env.bid_a_dd())
        else:
            if ai_type == "doudou" and self.env.has_dmc_model():
                result = self.env.action_dmc_with_stats()
                return int(result["best_action"])
            elif ai_type == "dede":
                _inject_gpu_worlds(self.env, self.env.get_dealer(), self.initial_hands, self.history)
                return int(self.env.action_dede(self.dede_time_ms))
            elif ai_type == "oracle_dd":
                return int(self.env.action_oracle_dd())
            else:
                # Default fallback to doudou
                if self.env.has_dmc_model():
                    result = self.env.action_dmc_with_stats()
                    return int(result["best_action"])
                return int(self.env.action_dede(self.dede_time_ms))

    def play_ai_turn(self):
        player = self.env.current_player()
        phase = self.env.phase()
        belote_before = list(self.env.get_belote())
        action = self.get_ai_action()
        name = colver.Env.action_name(action, phase)
        self.history.append({"player": int(player), "action": action, "phase": int(phase)})
        if phase == 0:
            self.bid_history.append({"player": int(player), "action": action, "name": name})
        self._check_trick_completion(action)
        if self.uses_dede:
            self.env.dede_step(action)
        else:
            self.env.step(action)
        self._finalize_trick_completion()
        self._detect_belote(player, belote_before)
        return action, name, self.get_state()


AGENT_NAMES = {
    "dede": "Dédé (IS-DD)",
    "doudou": "DouDou50",
    "oracle_dd": "Oracle (DD)",
}

SEAT_NAMES = ["Nord", "Est", "Sud", "Ouest"]


class WatchSession(TrickTracker):
    """AI vs AI spectating with per-action thinking stats."""

    def __init__(self, agents, dmc_model_path=None, bid_model_path=None, belief_model_path=None, dealer=None, hands=None, env=None, dede_time_ms=5000):
        """
        agents: dict {0: "smart", 1: "naive", 2: "doudou", 3: "random"}
        dmc_model_path: path to .bin weights file for DouDou50 (Rust inference)
        bid_model_path: path to .bin weights file for Bid à DD (NN bidder)
        belief_model_path: path to .bin weights file for BeliefNet (NN card beliefs)
        dealer: optional dealer seat (for custom deals)
        hands: optional list of 4 hands (for custom deals)
        env: optional pre-built Env (e.g. from CFN), takes priority over dealer/hands
        dede_time_ms: per-move IS-DD search budget in ms (default 5000)
        """
        self.agents = agents
        self.dede_time_ms = int(dede_time_ms)
        self.history = []
        self.bid_history = []
        self._init_trick_tracking()
        if env is not None:
            self.env = env
        elif hands is not None:
            self.env = colver.Env.deal_with_hands(dealer if dealer is not None else 0, hands)
        else:
            self.env = colver.Env()
            self.env.reset()

        # Full pre-auction deal, for the GPU playgen sidecar. Only valid if
        # the session starts from a fresh deal (a CFN env may be mid-game).
        self.initial_hands = [list(h) for h in self.env.get_hands()]

        # Load DMC model if any seat uses DouDou50
        if dmc_model_path and any(a == "doudou" for a in agents.values()):
            self.env.load_dmc_model(dmc_model_path)

        # Load bid NN model (Bid à DD)
        if bid_model_path:
            self.env.load_bid_model(bid_model_path)

        # Initialize IS-DD if any seat uses Dédé
        self.uses_dede = any(a == "dede" for a in agents.values())
        if belief_model_path and self.uses_dede:
            self.env.load_belief_net(belief_model_path)
        if self.uses_dede:
            self.env.dede_init()

        # Compute DD oracle scores at deal start (all hands visible in watch mode)
        try:
            dd_result = self.env.solve_all_suits()
            self.dd_scores = dd_result["suits"]
            self.dd_elapsed_ms = round(dd_result["elapsed_ms"], 1)
        except Exception:
            self.dd_scores = None
            self.dd_elapsed_ms = None

    def get_state(self):
        """Full state with ALL hands visible."""
        state = {
            "phase": int(self.env.phase()),
            "current_player": int(self.env.current_player()),
            "hands": self.env.get_hands(),
            "current_trick": self.env.get_current_trick(),
            "contract": self.env.get_contract(),
            "points": list(self.env.get_points()),
            "tricks_won": list(self.env.get_tricks_won()),
            "legal_actions": list(self.env.legal_actions()) if not self.env.is_terminal() else [],
            "dealer": int(self.env.get_dealer()),
            "trick_lead": int(self.env.get_trick_lead()),
            "is_terminal": self.env.is_terminal(),
            "last_trick": self.last_trick,
            "last_trick_winner": self.last_trick_winner,
            "last_trick_points": self.last_trick_points,
            "belote": list(self.env.get_belote()),
            "cfn": self.env.to_cfn(),
        }
        if self.env.is_terminal():
            rewards = list(self.env.rewards())
            contract = self.env.get_contract()
            points = list(self.env.get_points())
            belote = list(self.env.get_belote())
            contract_team = contract.get("team", 0)
            state["rewards"] = rewards
            state["score_detail"] = {
                "trick_points": points,
                "belote": [20 if b == 2 else 0 for b in belote],
                "contract_value": contract.get("value", 0),
                "contract_team": contract_team,
                "contract_made": rewards[contract_team] > 0,
                "final_scores": rewards,
            }
        return state

    def compute_next_action(self):
        """Compute next action with thinking stats. Returns move dict."""
        player = int(self.env.current_player())
        phase = int(self.env.phase())
        agent_type = self.agents.get(player, "dede")

        # Bidding phase: all agents use bid_a_dd (NN if loaded, else improved_v2)
        if phase == 0:
            action = int(self.env.bid_a_dd())
            name = colver.Env.action_name(action, phase)
            stats = {
                "agent": agent_type,
                "agent_label": AGENT_NAMES.get(agent_type, agent_type),
            }
            # Show NN Q-values if bid model is loaded
            if self.env.has_bid_model():
                nn_result = self.env.action_bid_nn()
                stats["bid_nn"] = {
                    "q_values": [[int(a), round(float(q), 3)] for a, q in nn_result["q_values"]],
                    "best_action": int(nn_result["best_action"]),
                }
            return {"player": player, "action": action, "phase": phase, "name": name, "stats": stats}

        # Play phase: dispatch by agent type
        stats = {"agent": agent_type, "agent_label": AGENT_NAMES.get(agent_type, agent_type)}

        if agent_type == "dede":
            gpu = _inject_gpu_worlds(self.env, self.env.get_dealer(), self.initial_hands, self.history)
            result = self.env.action_dede_with_stats(self.dede_time_ms)
            action = int(result["best_action"])
            stats["card_scores"] = [[int(a), round(float(s), 1)] for a, s in result["card_scores"]]
            stats["determinizations"] = int(result["determinizations"])
            stats["elapsed_ms"] = round(result["elapsed_ms"], 1)
            if gpu:
                stats["worlds_source"] = "playgen-gpu"

        elif agent_type == "oracle_dd":
            t0 = time.monotonic()
            action = int(self.env.action_oracle_dd())
            stats["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)

        elif agent_type == "doudou" and self.env.has_dmc_model():
            result = self.env.action_dmc_with_stats()
            action = int(result["best_action"])
            stats["q_values"] = [[int(a), round(float(q), 4)] for a, q in result["q_values"]]
            stats["elapsed_ms"] = round(result["elapsed_ms"], 2)

        else:
            # Fallback to doudou or dede
            if self.env.has_dmc_model():
                result = self.env.action_dmc_with_stats()
                action = int(result["best_action"])
                stats["q_values"] = [[int(a), round(float(q), 4)] for a, q in result["q_values"]]
                stats["elapsed_ms"] = round(result["elapsed_ms"], 2)
            else:
                _inject_gpu_worlds(self.env, self.env.get_dealer(), self.initial_hands, self.history)
                result = self.env.action_dede_with_stats(self.dede_time_ms)
                action = int(result["best_action"])
                stats["card_scores"] = [[int(a), round(float(s), 1)] for a, s in result["card_scores"]]
                stats["determinizations"] = int(result["determinizations"])
                stats["elapsed_ms"] = round(result["elapsed_ms"], 1)

        name = colver.Env.action_name(action, phase)
        return {"player": player, "action": action, "phase": phase, "name": name, "stats": stats}

    def apply_action(self, action):
        """Apply an action, track trick completion and history."""
        player = int(self.env.current_player())
        phase = int(self.env.phase())
        belote_before = list(self.env.get_belote())
        name = colver.Env.action_name(action, phase)

        if phase == 0:
            self.bid_history.append({
                "player": player,
                "action": action,
                "name": name,
            })

        self.history.append({"player": player, "action": action, "phase": phase, "name": name})
        self._check_trick_completion(action)

        if self.uses_dede:
            self.env.dede_step(action)
        else:
            self.env.step(action)

        self._finalize_trick_completion()
        self._detect_belote(player, belote_before)

    def step(self):
        """Compute + apply one action. Returns (move_info, state, completed_tricks)."""
        move = self.compute_next_action()
        self.apply_action(move["action"])
        return move, self.get_state(), self.completed_tricks


class BidProblemSession:
    """Single-bid practice problem: run auction until South's turn, ask player to bid."""

    def __init__(self, bid_model_path=None, dmc_model_path=None):
        self.env = None
        self.hands = None
        self.bid_history = []
        self.bid_model_path = bid_model_path
        self.dmc_model_path = dmc_model_path

    def generate(self) -> dict:
        """Play a full auction (bid NN when available, heuristic fallback),
        then sample one of South's decision points uniformly and replay the
        same deal up to it. Retries only if South never got a turn (auction
        ended by surcoinche within the first 3 actions)."""
        for _ in range(20):
            env = colver.Env()
            env.reset()
            if self.bid_model_path:
                env.load_bid_model(self.bid_model_path)
            use_nn = env.has_bid_model()
            hands = [list(h) for h in env.get_hands()]
            dealer = int(env.get_dealer())
            actions = []
            south_turns = []  # indices into actions where South is to act
            while env.phase() == 0:
                player = int(env.current_player())
                if player == 2:
                    south_turns.append(len(actions))
                if use_nn:
                    action = int(env.action_bid_nn()["best_action"])
                else:
                    action = int(env.bid_improved())
                actions.append((player, action))
                env.step(action)
            if not south_turns:
                continue

            # Replay the same deal up to the sampled decision point
            cut = random.choice(south_turns)
            env.redeal_with_hands(dealer, hands)
            bid_history = []
            for player, action in actions[:cut]:
                bid_history.append({"player": player, "action": action,
                                     "name": colver.Env.action_name(action, 0)})
                env.step(action)

            self.env = env
            self.hands = hands
            self.bid_history = bid_history
            result = {
                "south_hand": hands[2],
                "bid_history": bid_history,
                "legal_actions": list(env.legal_actions()),
                "dealer": int(env.get_dealer()),
            }
            # Include NN Q-values for client-side XGB hint analysis
            if env.has_bid_model():
                nn_result = env.action_bid_nn()
                result["nn_q_values"] = [[int(a), round(float(q), 3)] for a, q in nn_result["q_values"]]
            return result
        raise RuntimeError("Could not generate bid problem")

    def evaluate(self, player_action: int) -> dict:
        """Evaluate without advancing env. Returns correction dict."""
        env = self.env
        nn_result = None
        if env.has_bid_model():
            nn_result = env.action_bid_nn()
        heuristic_action = int(env.bid_improved())
        dd_result = env.solve_all_suits()
        return {
            "player_action": player_action,
            "player_action_name": colver.Env.action_name(player_action, 0),
            "nn_action": int(nn_result["best_action"]) if nn_result else None,
            "nn_action_name": colver.Env.action_name(int(nn_result["best_action"]), 0) if nn_result else None,
            "nn_q_values": [[int(a), round(float(q), 3)] for a, q in nn_result["q_values"]] if nn_result else [],
            "heuristic_action": heuristic_action,
            "heuristic_action_name": colver.Env.action_name(heuristic_action, 0),
            "dd_suits": dd_result["suits"],
            "dd_elapsed_ms": round(dd_result["elapsed_ms"], 1),
        }


class PlayProblemSession:
    """Single-card practice problem: play heuristic until South's turn, ask player to choose."""

    def __init__(self, bid_model_path=None, dmc_model_path=None):
        self.env = None
        self.hands = None
        self.bid_history = []
        self.bid_model_path = bid_model_path
        self.dmc_model_path = dmc_model_path

    def generate(self) -> dict:
        """Bid with bid_improved(), then play heuristic until South's turn.
        Retry if trivial: <2 legal cards, or all choices collapse to one
        equivalence class (e.g. 7 and 8 of a plain suit with no outstanding
        card between them — picking either cannot affect the outcome)."""
        for _ in range(50):
            env = colver.Env()
            env.reset()
            if self.bid_model_path:
                env.load_bid_model(self.bid_model_path)
            if self.dmc_model_path:
                env.load_dmc_model(self.dmc_model_path)
            hands = [list(h) for h in env.get_hands()]
            bid_history = []
            env.dede_init()

            # Bidding phase
            void = False
            while env.phase() == 0:
                player = int(env.current_player())
                action = int(env.bid_improved())
                bid_history.append({"player": player, "action": action,
                                     "name": colver.Env.action_name(action, 0)})
                env.dede_step(action)
                if env.is_terminal():
                    void = True
                    break

            if void or env.phase() != 1:
                continue

            # Play phase: advance with heuristic until South's turn
            while env.phase() == 1 and not env.is_terminal() and int(env.current_player()) != 2:
                action = int(env.action_heuristic_play())
                env.dede_step(action)

            if env.is_terminal() or int(env.current_player()) != 2:
                continue
            if len(env.legal_actions_reduced()) < 2:
                continue

            self.env = env
            self.hands = hands
            self.bid_history = bid_history
            return {
                "south_hand": hands[2],
                "bid_history": bid_history,
                "contract": env.get_contract(),
                "current_trick": env.get_current_trick(),
                "tricks_won": list(env.get_tricks_won()),
                "points": list(env.get_points()),
                "legal_actions": list(env.legal_actions()),
                "dealer": int(env.get_dealer()),
                "trick_lead": int(env.get_trick_lead()),
            }
        raise RuntimeError("Could not generate play problem")

    def evaluate(self, player_action: int) -> dict:
        """Evaluate player's card. IS-DD beliefs are warm from generate()."""
        env = self.env
        t0 = time.monotonic()
        oracle_action = int(env.action_oracle_dd())
        oracle_elapsed = round((time.monotonic() - t0) * 1000, 1)

        dmc_result = None
        if env.has_dmc_model():
            dmc_result = env.action_dmc_with_stats()

        isdd_result = env.action_dede_with_stats(100)

        return {
            "player_action": player_action,
            "player_action_name": colver.Env.action_name(player_action, 1),
            "oracle_action": oracle_action,
            "oracle_action_name": colver.Env.action_name(oracle_action, 1),
            "oracle_elapsed_ms": oracle_elapsed,
            "dmc_action": int(dmc_result["best_action"]) if dmc_result else None,
            "dmc_action_name": colver.Env.action_name(int(dmc_result["best_action"]), 1) if dmc_result else None,
            "dmc_q_values": [[int(c), round(float(q), 4)] for c, q in dmc_result["q_values"]] if dmc_result else [],
            "isdd_action": int(isdd_result["best_action"]),
            "isdd_action_name": colver.Env.action_name(int(isdd_result["best_action"]), 1),
            "isdd_card_scores": [[int(c), round(float(s), 1)] for c, s in isdd_result["card_scores"]],
            "isdd_determinizations": int(isdd_result["determinizations"]),
            "isdd_elapsed_ms": round(isdd_result["elapsed_ms"], 1),
            "all_hands": self.hands,
        }


class BeliefSession:
    """Belief visualization: generate a game, step through it, query beliefs at each position."""

    def __init__(self, dmc_model_path=None, bid_model_path=None, belief_model_path=None, playgen_model_path=None):
        self.dmc_model_path = dmc_model_path
        self.bid_model_path = bid_model_path
        self.belief_model_path = belief_model_path
        self.playgen_model_path = playgen_model_path
        self.env = None
        self.initial_hands = None
        self.all_actions = []  # list of (player, action, phase)
        self.action_idx = 0
        self.dealer = 0
        self.num_bid_actions = 0
        # Full-game CFN (with auction) — identity for the shared belief cache.
        self.game_cfn = None
        # Precompute sweep state (dedicated env, advanced by precompute_step)
        self._sweep_env = None
        self._sweep_observer = 0
        self._sweep_i = 0
        self._sweep_done = 0
        self._sweep_total = 0

    # IS-DD time budget per play move when generating the demo game.
    GEN_PLAY_TIME_MS = 50

    def generate(self) -> dict:
        """Play a full game (NN bidder for the auction, IS-DD for the play),
        store all actions.

        The auction uses the NN bidder rather than the DD oracle: it is orders
        of magnitude faster and produces realistic signaling auctions (the DD
        oracle sees all hands, so it yields degenerate optimal-bid-then-pass
        auctions). The play uses IS-DD (belief-aware via the NN belief net)."""
        env = colver.Env()
        env.reset()
        if self.bid_model_path:
            env.load_bid_model(self.bid_model_path)
        if self.belief_model_path:
            env.load_belief_net(self.belief_model_path)
        env.dede_init()

        self.initial_hands = [list(h) for h in env.get_hands()]
        self.dealer = int(env.get_dealer())
        self.all_actions = []
        self.action_idx = 0

        # Play full game
        while not env.is_terminal():
            player = int(env.current_player())
            phase = int(env.phase())
            if phase == 0:
                if env.has_bid_model():
                    action = int(env.action_bid_nn()["best_action"])
                else:
                    action = int(env.bid_a_dd())
            else:
                action = int(env.action_dede(self.GEN_PLAY_TIME_MS))
            self.all_actions.append((player, action, phase))
            env.dede_step(action)

        return self._finalize()

    def import_cfn(self, cfn: str) -> dict:
        """Rebuild a session from a full-game CFN (auction + play).

        Accepts the extended 4-section CFN (with auction) or a plain 3-section
        core CFN. The engine's `from_cfn` reconstructs the deal + play order;
        the auction section supplies the bid actions."""
        core, bid_actions = game_notation.parse_full_cfn(cfn)
        src = colver.Env.from_cfn(core)
        self.dealer = int(src.get_dealer())
        self.initial_hands = [list(h) for h in src.get_initial_hands()]
        play_actions = [int(a) for a in src.get_play_order()]
        action_ids = [int(a) for a in bid_actions] + play_actions

        # Replay to recover (player, action, phase) for each step.
        env = colver.Env.deal_with_hands(self.dealer, self.initial_hands)
        env.dede_init()
        self.all_actions = []
        for a in action_ids:
            self.all_actions.append((int(env.current_player()), int(a), int(env.phase())))
            env.dede_step(a)
        if not env.is_terminal():
            raise ValueError("CFN ne décrit pas une partie complète")
        self.action_idx = 0
        return self._finalize()

    def _finalize(self) -> dict:
        """Set up the stepping env, compute the game CFN, return the payload.
        Shared by generate() and import_cfn()."""
        self.num_bid_actions = sum(1 for _, _, p in self.all_actions if p == 0)
        self.env = colver.Env.deal_with_hands(self.dealer, self.initial_hands)
        if self.bid_model_path:
            self.env.load_bid_model(self.bid_model_path)
        if self.belief_model_path:
            self.env.load_belief_net(self.belief_model_path)
        if self.playgen_model_path:
            self.env.load_playgen_model(self.playgen_model_path)
        self.env.dede_init()
        self._sweep_env = None
        self.game_cfn = self._compute_game_cfn()
        return {
            "initial_hands": self.initial_hands,
            "dealer": self.dealer,
            "total_actions": len(self.all_actions),
            "num_bid_actions": self.num_bid_actions,
            "game_cfn": self.game_cfn,
            "actions": [
                {"player": p, "action": a, "phase": ph}
                for p, a, ph in self.all_actions
            ],
        }

    def _compute_game_cfn(self) -> str:
        """Full-game CFN (with auction) for the current game."""
        return compute_game_cfn(
            self.dealer, self.initial_hands, [a for _p, a, _ph in self.all_actions])

    def restore(self, dealer, initial_hands, actions) -> dict:
        """Rebuild a session from a deal the client already holds (no new deal).

        Used when the WebSocket reconnects: the per-connection server session is
        gone, but the client still has the exact deal + action list, so we
        reconstruct it deterministically instead of generating a fresh game.
        """
        self.dealer = int(dealer)
        self.initial_hands = [list(h) for h in initial_hands]
        self.all_actions = [
            (int(a["player"]), int(a["action"]), int(a["phase"])) for a in actions
        ]
        self.action_idx = 0
        return self._finalize()

    def _get_state_info(self) -> dict:
        """Get current state info for the client."""
        env = self.env
        finished = self.action_idx >= len(self.all_actions)
        state = {
            "phase": int(env.phase()),
            "current_player": int(env.current_player()),
            "current_trick": env.get_current_trick(),
            "contract": env.get_contract(),
            "points": list(env.get_points()),
            "tricks_won": list(env.get_tricks_won()),
            "hands": env.get_hands(),
            "is_terminal": env.is_terminal(),
        }
        # Last action info
        last_action = None
        if self.action_idx > 0:
            player, action, phase = self.all_actions[self.action_idx - 1]
            last_action = {
                "player": player,
                "action": action,
                "phase": phase,
                "name": colver.Env.action_name(action, phase),
            }
        return {
            "action_idx": self.action_idx,
            "total_actions": len(self.all_actions),
            "state": state,
            "last_action": last_action,
            "finished": finished,
        }

    def step_forward(self) -> dict:
        """Apply next action and return state."""
        if self.action_idx >= len(self.all_actions):
            return self._get_state_info()
        player, action, phase = self.all_actions[self.action_idx]
        self.env.dede_step(action)
        self.action_idx += 1
        return self._get_state_info()

    def step_to(self, target: int) -> dict:
        """Jump to a specific action index. Resets and replays if backward."""
        target = max(0, min(target, len(self.all_actions)))
        if target < self.action_idx:
            # Reset from scratch
            self.env = colver.Env.deal_with_hands(self.dealer, self.initial_hands)
            if self.bid_model_path:
                self.env.load_bid_model(self.bid_model_path)
            if self.belief_model_path:
                self.env.load_belief_net(self.belief_model_path)
            if self.playgen_model_path:
                self.env.load_playgen_model(self.playgen_model_path)
            self.env.dede_init()
            self.action_idx = 0
        # Replay up to target
        while self.action_idx < target:
            player, action, phase = self.all_actions[self.action_idx]
            self.env.dede_step(action)
            self.action_idx += 1
        return self._get_state_info()

    # 30 worlds keeps the marginals readable while staying ~6-8s/position with
    # the v2 model (43MB, ~3-5x slower per world than v1).
    PLAYGEN_WORLDS = 30
    PLAYGEN_TEMP = 0.8
    # GPU sidecar: worlds are ~50x cheaper, so marginals get much smoother.
    PLAYGEN_WORLDS_GPU = 200

    def _playgen_marginals(self, env_actions_prefix: int, observer: int):
        """Playgen marginals at a position: GPU sidecar first (200 worlds),
        CPU PyO3 fallback (30 worlds). Returns None during bidding."""
        if playgen_gpu.enabled():
            actions = [(p, a) for p, a, _ in self.all_actions[:env_actions_prefix]]
            w = playgen_gpu.beliefs(
                self.dealer, self.initial_hands, actions, observer,
                n_worlds=self.PLAYGEN_WORLDS_GPU, temperature=self.PLAYGEN_TEMP,
            )
            if w is not None:
                return w
        return None

    def get_beliefs(self, observer: int, with_playgen: bool = False) -> dict:
        """Return NN + heuristic (+ playgen on demand) belief weights + ground truth hands.

        Playgen marginals cost ~1s of MC sampling, so they are only computed
        when the client displays them (with_playgen=True) and cached per
        (position, observer) — the precompute sweep fills the same cache.
        """
        idx = self.action_idx
        # NN + heuristic: cheap, but cache anyway so a shared game is instant.
        nnh = _belief_cache_get(self.game_cfn, "nn", idx, observer)
        if nnh is None:
            result = self.env.get_belief_weights(observer)
            nnh = {"nn": result["nn"], "heuristic": result["heuristic"]}
            _belief_cache_put(self.game_cfn, "nn", idx, observer, nnh)
        playgen = None
        if with_playgen and self.playgen_model_path:
            playgen = _belief_cache_get(self.game_cfn, "playgen", idx, observer)
            if playgen is None:
                # GPU sidecar (200 worlds) → CPU fallback (30 worlds);
                # None during bidding (contract unknown)
                playgen = self._playgen_marginals(idx, observer)
                if playgen is None:
                    playgen = _safe_playgen(
                        self.env.get_playgen_beliefs,
                        observer, n_worlds=self.PLAYGEN_WORLDS, temperature=self.PLAYGEN_TEMP
                    )
                _belief_cache_put(self.game_cfn, "playgen", idx, observer, playgen)
        return {
            "observer": observer,
            "nn": nnh["nn"],
            "heuristic": nnh["heuristic"],
            "playgen": playgen,
            "ground_truth": self.initial_hands,
        }

    def precompute_start(self, observer: int) -> int:
        """Prepare a playgen precompute sweep for `observer` on a dedicated env.

        Returns the number of play-phase positions to compute (0 if no playgen
        model or void deal).
        """
        self._sweep_observer = observer
        self._sweep_i = 0
        self._sweep_done = 0
        num_bid = sum(1 for _, _, p in self.all_actions if p == 0)
        self._sweep_total = max(0, len(self.all_actions) - num_bid)
        if not self.playgen_model_path or self._sweep_total == 0:
            self._sweep_env = None
            self._sweep_total = 0
            return 0
        env = colver.Env.deal_with_hands(self.dealer, self.initial_hands)
        if self.bid_model_path:
            env.load_bid_model(self.bid_model_path)
        if self.belief_model_path:
            env.load_belief_net(self.belief_model_path)
        env.load_playgen_model(self.playgen_model_path)
        env.dede_init()
        self._sweep_env = env
        return self._sweep_total

    def precompute_step(self):
        """Advance the sweep by one playgen position (~1s of MC sampling).

        Returns (done, total) after each computed position, or None when the
        sweep is finished. Designed to be called repeatedly from an executor
        so the caller can stream progress between steps.
        """
        if self._sweep_env is None:
            return None
        while self._sweep_i < len(self.all_actions):
            _, action, _ = self.all_actions[self._sweep_i]
            self._sweep_env.dede_step(action)
            self._sweep_i += 1
            idx = self._sweep_i
            if int(self._sweep_env.phase()) == 1 and not self._sweep_env.is_terminal():
                obs = self._sweep_observer
                if _belief_cache_get(self.game_cfn, "playgen", idx, obs) is None:
                    w = self._playgen_marginals(idx, obs)
                    if w is None:
                        w = _safe_playgen(
                            self._sweep_env.get_playgen_beliefs,
                            obs,
                            n_worlds=self.PLAYGEN_WORLDS,
                            temperature=self.PLAYGEN_TEMP,
                        )
                    _belief_cache_put(self.game_cfn, "playgen", idx, obs, w)
                self._sweep_done += 1
                return (self._sweep_done, self._sweep_total)
        self._sweep_env = None
        return None


class ReplaySession(TrickTracker):
    """Replay a stored game from the database step-by-step."""

    def __init__(self, game_data):
        self.actions = game_data["actions"]
        self.agents = game_data["agents"]
        self.action_idx = 0
        self.bid_history = []
        self._init_trick_tracking()
        self.env = colver.Env.deal_with_hands(game_data["dealer"], game_data["hands"])
        # Full-game CFN (auction + play) — copy-pasteable into the belief page.
        self.game_cfn = compute_game_cfn(
            game_data["dealer"], game_data["hands"],
            [e["action"] for e in self.actions])

    def get_state(self):
        """Full state with ALL hands visible (same as WatchSession)."""
        state = {
            "phase": int(self.env.phase()),
            "current_player": int(self.env.current_player()),
            "hands": self.env.get_hands(),
            "current_trick": self.env.get_current_trick(),
            "contract": self.env.get_contract(),
            "points": list(self.env.get_points()),
            "tricks_won": list(self.env.get_tricks_won()),
            "legal_actions": list(self.env.legal_actions()) if not self.env.is_terminal() else [],
            "dealer": int(self.env.get_dealer()),
            "trick_lead": int(self.env.get_trick_lead()),
            "is_terminal": self.env.is_terminal(),
            "last_trick": self.last_trick,
            "last_trick_winner": self.last_trick_winner,
            "last_trick_points": self.last_trick_points,
            "belote": list(self.env.get_belote()),
            "cfn": self.env.to_cfn(),
        }
        if self.env.is_terminal():
            rewards = list(self.env.rewards())
            contract = self.env.get_contract()
            points = list(self.env.get_points())
            belote = list(self.env.get_belote())
            contract_team = contract.get("team", 0)
            state["rewards"] = rewards
            state["score_detail"] = {
                "trick_points": points,
                "belote": [20 if b == 2 else 0 for b in belote],
                "contract_value": contract.get("value", 0),
                "contract_team": contract_team,
                "contract_made": rewards[contract_team] > 0,
                "final_scores": rewards,
            }
        return state

    def step(self):
        """Apply next stored action. Returns (move_info, state, completed_tricks, finished)."""
        if self.action_idx >= len(self.actions):
            return None, self.get_state(), self.completed_tricks, True

        entry = self.actions[self.action_idx]
        action = entry["action"]
        player = int(self.env.current_player())
        phase = int(self.env.phase())
        name = colver.Env.action_name(action, phase)

        if phase == 0:
            self.bid_history.append({
                "player": player,
                "action": action,
                "name": name,
            })

        belote_before = list(self.env.get_belote())
        self._check_trick_completion(action)
        self.env.step(action)
        self._finalize_trick_completion()
        self._detect_belote(player, belote_before)

        self.action_idx += 1

        agent_type = self.agents.get(str(player), self.agents.get(player, "?"))
        move = {
            "player": player,
            "action": action,
            "phase": phase,
            "name": name,
            "stats": {
                "agent": agent_type,
                "agent_label": AGENT_NAMES.get(agent_type, agent_type),
            },
        }

        finished = self.env.is_terminal() or self.action_idx >= len(self.actions)
        return move, self.get_state(), self.completed_tricks, finished


