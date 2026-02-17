"""Game session management for Colver web UI."""

import time
import colver


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


AI_TIME_MS = 100  # Fixed time budget for all search-based AIs

class PlaySession(TrickTracker):
    """Wraps a colver.Env for human vs AI play."""

    def __init__(self, ai_types=None, human_seat=2, dmc_model_path=None):
        # ai_types: dict mapping seat -> ai_type (for non-human seats)
        # If not provided, default all AI seats to "doudou"
        self.human_seat = human_seat
        if ai_types is None:
            ai_types = {}
        self.ai_types = ai_types
        self.env = colver.Env()
        self.history = []
        self._init_trick_tracking()
        self.env.reset()
        self.uses_smart = any(t == "smart" for t in self.ai_types.values())
        if self.uses_smart:
            self.env.smart_ismcts_init()
        if dmc_model_path and any(t == "doudou" for t in self.ai_types.values()):
            self.env.load_dmc_model(dmc_model_path)

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
        }
        # During bidding, suggest the best trump suit for the human player
        if phase == 0:
            eval_result = self.env.evaluate_hand(human_seat)
            state["best_trump_suit"] = int(eval_result["best_suit"])
        return state

    def play_action(self, action):
        player = self.env.current_player()
        phase = self.env.phase()
        belote_before = list(self.env.get_belote())
        self.history.append({"player": int(player), "action": int(action), "phase": int(phase)})
        self._check_trick_completion(action)
        if self.uses_smart:
            self.env.smart_ismcts_step(action)
        else:
            self.env.step(action)
        self._finalize_trick_completion()
        self._detect_belote(player, belote_before)
        return self.get_state()

    def get_ai_action(self):
        phase = self.env.phase()
        player = int(self.env.current_player())
        ai_type = self.ai_types.get(player, "doudou")
        if phase == 0:
            if ai_type == "maxi_bid":
                return int(self.env.bid_maxi())
            return int(self.env.bid_improved())
        else:
            if ai_type == "doudou" and self.env.has_dmc_model():
                result = self.env.action_dmc_with_stats()
                return int(result["best_action"])
            elif ai_type == "smart":
                return int(self.env.action_smart_ismcts(AI_TIME_MS))
            elif ai_type == "naive":
                return int(self.env.action_naive_ismcts(AI_TIME_MS))
            elif ai_type == "oracle":
                return int(self.env.action_oracle_mcts(AI_TIME_MS))
            elif ai_type == "maxi_bid":
                return int(self.env.action_maxi_play())
            else:
                return int(self.env.action_naive_ismcts(AI_TIME_MS))

    def play_ai_turn(self):
        player = self.env.current_player()
        phase = self.env.phase()
        belote_before = list(self.env.get_belote())
        action = self.get_ai_action()
        name = colver.Env.action_name(action, phase)
        self.history.append({"player": int(player), "action": action, "phase": int(phase)})
        self._check_trick_completion(action)
        if self.uses_smart:
            self.env.smart_ismcts_step(action)
        else:
            self.env.step(action)
        self._finalize_trick_completion()
        self._detect_belote(player, belote_before)
        return action, name, self.get_state()


AGENT_NAMES = {
    "smart": "Smart IS-MCTS",
    "naive": "Naive IS-MCTS",
    "doudou": "DouDou",
    "oracle": "Oracle (MCTS)",
    "heuristic": "Heuristique",
    "maxi_bid": "Maxi Bid",
    "random": "Aleatoire",
}

SEAT_NAMES = ["Nord", "Est", "Sud", "Ouest"]


class WatchSession(TrickTracker):
    """AI vs AI spectating with per-action thinking stats."""

    def __init__(self, agents, dmc_model_path=None, dealer=None, hands=None):
        """
        agents: dict {0: "smart", 1: "naive", 2: "doudou", 3: "random"}
        dmc_model_path: path to .bin weights file for DouDou (Rust inference)
        dealer: optional dealer seat (for custom deals)
        hands: optional list of 4 hands (for custom deals)
        """
        self.agents = agents
        self.env = colver.Env()
        self.history = []
        self.bid_history = []
        self._init_trick_tracking()
        if hands is not None:
            self.env = colver.Env.deal_with_hands(dealer if dealer is not None else 0, hands)
        else:
            self.env.reset()

        # Load DMC model if any seat uses DouDou
        if dmc_model_path and any(a == "doudou" for a in agents.values()):
            self.env.load_dmc_model(dmc_model_path)

        # Initialize Smart IS-MCTS if any seat uses it
        self.uses_smart = any(a == "smart" for a in agents.values())
        if self.uses_smart:
            self.env.smart_ismcts_init()

    def get_state(self):
        """Full state with ALL hands visible."""
        return {
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

    def compute_next_action(self):
        """Compute next action with thinking stats. Returns move dict."""
        player = int(self.env.current_player())
        phase = int(self.env.phase())
        agent_type = self.agents.get(player, "random")

        # Bidding phase: maxi uses its own bidding, others use improved_bid
        if phase == 0:
            if agent_type == "maxi_bid":
                action = int(self.env.bid_maxi())
            else:
                action = int(self.env.bid_improved())
            name = colver.Env.action_name(action, phase)
            bid_eval = self.env.evaluate_hand(player)
            stats = {
                "agent": agent_type,
                "agent_label": AGENT_NAMES.get(agent_type, agent_type),
                "bid_eval": {
                    "scores": list(bid_eval["scores"]),
                    "best_suit": int(bid_eval["best_suit"]),
                    "best_score": int(bid_eval["best_score"]),
                },
            }
            return {"player": player, "action": action, "phase": phase, "name": name, "stats": stats}

        # Play phase: dispatch by agent type
        stats = {"agent": agent_type, "agent_label": AGENT_NAMES.get(agent_type, agent_type)}

        if agent_type == "smart":
            result = self.env.action_smart_ismcts_with_stats(AI_TIME_MS)
            action = int(result["best_action"])
            stats["visit_counts"] = [[int(a), int(v)] for a, v in result["visit_counts"]]
            stats["root_visits"] = int(result["root_visits"])
            stats["elapsed_ms"] = round(result["elapsed_ms"], 1)

        elif agent_type == "naive":
            result = self.env.action_naive_ismcts_with_stats(AI_TIME_MS)
            action = int(result["best_action"])
            stats["visit_counts"] = [[int(a), int(v)] for a, v in result["visit_counts"]]
            stats["root_visits"] = int(result["root_visits"])
            stats["elapsed_ms"] = round(result["elapsed_ms"], 1)

        elif agent_type == "oracle":
            t0 = time.monotonic()
            action = int(self.env.action_oracle_mcts(AI_TIME_MS))
            stats["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 1)

        elif agent_type == "doudou" and self.env.has_dmc_model():
            result = self.env.action_dmc_with_stats()
            action = int(result["best_action"])
            stats["q_values"] = [[int(a), round(float(q), 4)] for a, q in result["q_values"]]
            stats["elapsed_ms"] = round(result["elapsed_ms"], 2)

        elif agent_type == "heuristic":
            t0 = time.monotonic()
            action = int(self.env.action_heuristic_play())
            stats["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 2)

        elif agent_type == "maxi_bid":
            t0 = time.monotonic()
            action = int(self.env.action_maxi_play())
            stats["elapsed_ms"] = round((time.monotonic() - t0) * 1000, 2)

        else:  # random or fallback
            action = int(self.env.action_random())
            stats["elapsed_ms"] = 0.0

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

        if self.uses_smart:
            self.env.smart_ismcts_step(action)
        else:
            self.env.step(action)

        self._finalize_trick_completion()
        self._detect_belote(player, belote_before)

    def step(self):
        """Compute + apply one action. Returns (move_info, state, completed_tricks)."""
        move = self.compute_next_action()
        self.apply_action(move["action"])
        return move, self.get_state(), self.completed_tricks


class ReplaySession(TrickTracker):
    """Replay a stored game from the database step-by-step."""

    def __init__(self, game_data):
        self.actions = game_data["actions"]
        self.agents = game_data["agents"]
        self.action_idx = 0
        self.bid_history = []
        self._init_trick_tracking()
        self.env = colver.Env.deal_with_hands(game_data["dealer"], game_data["hands"])

    def get_state(self):
        """Full state with ALL hands visible (same as WatchSession)."""
        return {
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


