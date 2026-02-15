"""Game session management for Colver web UI."""

import colver
import json
import time
import numpy as np


class PlaySession:
    """Wraps a colver.Env for human vs AI play."""

    def __init__(self, ai_type="smart", time_ms=50, nn_model=None, nn_device=None):
        self.ai_type = ai_type
        self.time_ms = time_ms
        self.nn_model = nn_model
        self.nn_device = nn_device
        self.env = colver.Env()
        self.history = []  # list of {player, action, phase}
        self.last_trick = None  # [card0, card1, card2, card3] or None
        self.last_trick_winner = None
        self.last_trick_points = 0
        obs, _ = self.env.reset()
        self._last_obs = obs  # track latest obs for NN agent
        if ai_type == "smart":
            self.env.smart_ismcts_init()

    # Card point values: [7, 8, 9, J, Q, K, 10, A]
    PLAIN_POINTS = [0, 0, 0, 2, 3, 4, 10, 11]
    TRUMP_POINTS = [0, 0, 14, 20, 3, 4, 10, 11]

    def _card_points(self, card_idx, trump_suit):
        """Point value of a single card given trump suit."""
        suit = card_idx >> 3
        rank = card_idx & 7
        if suit == trump_suit:
            return self.TRUMP_POINTS[rank]
        return self.PLAIN_POINTS[rank]

    def _trick_points(self, trick, trump_suit):
        """Total point value of a completed trick."""
        return sum(self._card_points(c, trump_suit) for c in trick if 0 <= c < 32)

    def _check_trick_completion(self, action):
        """Save the trick before step if this action completes it (4th card)."""
        self._trick_just_completed = False
        if self.env.phase() != 1:
            return
        trick = self.env.get_current_trick()
        filled = sum(1 for c in trick if c >= 0)
        if filled == 3:
            player = int(self.env.current_player())
            trick[player] = action
            self.last_trick = trick
            contract = self.env.get_contract()
            trump = contract.get("trump", 0)
            self.last_trick_points = self._trick_points(trick, trump)
            self._trick_just_completed = True

    def _finalize_trick_completion(self):
        """After step, record the trick winner if a trick just completed."""
        if self._trick_just_completed:
            self.last_trick_winner = int(self.env.current_player())
            self._trick_just_completed = False

    def get_state(self, human_seat=2):
        """Get state dict for frontend. Hides other hands in play mode."""
        phase = self.env.phase()
        hands = self.env.get_hands()
        # Only show human's hand in play mode
        hidden_hands = []
        for i, h in enumerate(hands):
            if i == human_seat:
                hidden_hands.append(h)
            else:
                hidden_hands.append([])  # hidden
        return {
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
        }

    def play_action(self, action):
        """Human plays an action. Returns updated state."""
        player = self.env.current_player()
        phase = self.env.phase()
        self.history.append({"player": int(player), "action": int(action), "phase": int(phase)})
        self._check_trick_completion(action)
        if self.ai_type == "smart":
            self.env.smart_ismcts_step(action)
        else:
            obs, _, _, _ = self.env.step(action)
            self._last_obs = obs
        self._finalize_trick_completion()
        return self.get_state()

    def get_ai_action(self):
        """Get AI's chosen action for current state."""
        import torch
        phase = self.env.phase()
        if phase == 0:
            # Bidding: use improved_bid for all AI types
            return int(self.env.bid_improved())
        else:
            # Playing
            if self.ai_type == "naive":
                return int(self.env.action_naive_ismcts(self.time_ms))
            elif self.ai_type == "smart":
                return int(self.env.action_smart_ismcts(self.time_ms))
            elif self.ai_type == "doudou" and self.nn_model is not None:
                # NN Q-network: single forward pass
                obs = np.array(self._last_obs, dtype=np.float32)
                mask = np.array(self.env.legal_action_mask(), dtype=np.float32)[:32]
                obs_t = torch.tensor(obs, device=self.nn_device).unsqueeze(0)
                mask_t = torch.tensor(mask, device=self.nn_device).unsqueeze(0)
                with torch.no_grad():
                    q = self.nn_model(obs_t)
                    q[mask_t == 0] = -1e9
                    action = q.argmax(dim=1).item()
                return int(action)
            else:
                return int(self.env.action_naive_ismcts(self.time_ms))

    def play_ai_turn(self):
        """AI plays its turn. Returns (action, action_name, state)."""
        player = self.env.current_player()
        phase = self.env.phase()
        action = self.get_ai_action()
        name = colver.Env.action_name(action, phase)
        self.history.append({"player": int(player), "action": action, "phase": int(phase)})
        self._check_trick_completion(action)
        if self.ai_type == "smart":
            self.env.smart_ismcts_step(action)
        else:
            obs, _, _, _ = self.env.step(action)
            self._last_obs = obs
        self._finalize_trick_completion()
        return action, name, self.get_state()


class ReplaySession:
    """Replay a recorded game step by step."""

    def __init__(self, log_data):
        self.log = log_data
        self.dealer = log_data["dealer"]
        self.hands = log_data["hands"]
        self.actions = log_data["actions"]
        self.states = []
        self._precompute_states()

    def _precompute_states(self):
        """Replay all actions and store states at each step."""
        env = colver.Env.deal_with_hands(self.dealer, self.hands)
        self.states.append(self._extract_state(env))
        for act in self.actions:
            env.step(act["action"])
            self.states.append(self._extract_state(env))

    def _extract_state(self, env):
        return {
            "phase": int(env.phase()),
            "current_player": int(env.current_player()),
            "hands": env.get_hands(),
            "current_trick": env.get_current_trick(),
            "contract": env.get_contract(),
            "points": list(env.get_points()),
            "tricks_won": list(env.get_tricks_won()),
            "legal_actions": list(env.legal_actions()) if not env.is_terminal() else [],
            "dealer": int(env.get_dealer()),
            "trick_lead": int(env.get_trick_lead()),
            "is_terminal": env.is_terminal(),
        }

    def get_state(self, step):
        step = max(0, min(step, len(self.states) - 1))
        return self.states[step]

    @property
    def total_steps(self):
        return len(self.states) - 1


class AnalysisSession:
    """Custom position setup and analysis."""

    def __init__(self):
        self.env = None

    def setup(self, dealer, hands, contract):
        """Set up a custom position."""
        self.env = colver.Env.deal_with_hands(dealer, hands)
        self.env.set_contract(
            contract["trump"],
            contract["value"],
            contract["team"],
            contract.get("coinche", 0),
        )
        self.env.set_phase_playing()
        return self._get_state()

    def _get_state(self):
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
        }

    def analyze(self, agent="naive", time_ms=200):
        """Run MCTS analysis on current position."""
        if self.env is None:
            return {"error": "No position set up"}

        legal = self.env.legal_actions()
        if not legal:
            return {"error": "No legal actions"}

        # Run multiple rollouts to get visit distribution
        results = {}
        phase = self.env.phase()

        if agent == "smart":
            self.env.smart_ismcts_init()
            action = self.env.action_smart_ismcts(time_ms)
        else:
            action = self.env.action_naive_ismcts(time_ms)

        # We can only get the best action, not full visit counts from Python API
        # So we return the best action
        return {
            "best": int(action),
            "name": colver.Env.action_name(action, phase),
            "legal_actions": [
                {"action": int(a), "name": colver.Env.action_name(a, phase)}
                for a in legal
            ],
        }
