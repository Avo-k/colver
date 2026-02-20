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


AI_TIME_MS = 100  # Fixed time budget for all search-based AIs

class PlaySession(TrickTracker):
    """Wraps a colver.Env for human vs AI play."""

    def __init__(self, ai_types=None, human_seat=2, dmc_model_path=None, bid_model_path=None):
        # ai_types: dict mapping seat -> ai_type (for non-human seats)
        # If not provided, default all AI seats to "doudou"
        self.human_seat = human_seat
        if ai_types is None:
            ai_types = {}
        self.ai_types = ai_types
        self.env = colver.Env()
        self.history = []
        self.bid_history = []
        self._init_trick_tracking()
        self.env.reset()
        self.initial_hands = [list(h) for h in self.env.get_hands()]
        self.uses_dede = any(t == "dede" for t in self.ai_types.values())
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
        ai_type = self.ai_types.get(player, "doudou")
        if phase == 0:
            return int(self.env.bid_a_dd())
        else:
            if ai_type == "doudou" and self.env.has_dmc_model():
                result = self.env.action_dmc_with_stats()
                return int(result["best_action"])
            elif ai_type == "dede":
                return int(self.env.action_dede(AI_TIME_MS))
            elif ai_type == "oracle_dd":
                return int(self.env.action_oracle_dd())
            else:
                # Default fallback to doudou
                if self.env.has_dmc_model():
                    result = self.env.action_dmc_with_stats()
                    return int(result["best_action"])
                return int(self.env.action_dede(AI_TIME_MS))

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
    "doudou": "DouDou35",
    "oracle_dd": "Oracle (DD)",
}

SEAT_NAMES = ["Nord", "Est", "Sud", "Ouest"]


class WatchSession(TrickTracker):
    """AI vs AI spectating with per-action thinking stats."""

    def __init__(self, agents, dmc_model_path=None, bid_model_path=None, dealer=None, hands=None, env=None):
        """
        agents: dict {0: "smart", 1: "naive", 2: "doudou", 3: "random"}
        dmc_model_path: path to .bin weights file for DouDou (Rust inference)
        bid_model_path: path to .bin weights file for Bid à DD (NN bidder)
        dealer: optional dealer seat (for custom deals)
        hands: optional list of 4 hands (for custom deals)
        env: optional pre-built Env (e.g. from CFN), takes priority over dealer/hands
        """
        self.agents = agents
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

        # Load DMC model if any seat uses DouDou
        if dmc_model_path and any(a == "doudou" for a in agents.values()):
            self.env.load_dmc_model(dmc_model_path)

        # Load bid NN model (Bid à DD)
        if bid_model_path:
            self.env.load_bid_model(bid_model_path)

        # Initialize IS-DD if any seat uses Dédé
        self.uses_dede = any(a == "dede" for a in agents.values())
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
        agent_type = self.agents.get(player, "doudou")

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
            result = self.env.action_dede_with_stats(AI_TIME_MS)
            action = int(result["best_action"])
            stats["card_scores"] = [[int(a), round(float(s), 1)] for a, s in result["card_scores"]]
            stats["determinizations"] = int(result["determinizations"])
            stats["elapsed_ms"] = round(result["elapsed_ms"], 1)

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
                result = self.env.action_dede_with_stats(AI_TIME_MS)
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


