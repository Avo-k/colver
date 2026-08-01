"""Game session management for Colver web UI."""

import logging
import os
import random
import threading
import time
from collections import OrderedDict
import colver

from colver.web import game_notation
from colver.web import playgen_gpu
from colver.web.agents import AGENT_NAMES, AgentTable, decision_stats

logger = logging.getLogger(__name__)


BID_PASS = 0


def only_pass_is_legal(env) -> bool:
    """Passing is the current player's only legal bid — nothing to decide.

    Rare, but real: once the opponents have coinched our bid and our partner
    has declined the surcoinche, we cannot surcoinche our own team and the
    coinche has frozen the contract, so pass is all that is left. Same over a
    partner's capot, which nobody can outbid. The server plays these itself
    rather than asking a human to click a single available button.
    """
    return (env.phase() == 0 and not env.is_terminal()
            and list(env.legal_actions()) == [BID_PASS])


LAST_TRICK = 7  # index 0-based de la 8e levée


def in_last_trick(env) -> bool:
    """La dernière levée est en cours : plus personne n'a de choix.

    Huit plis, huit cartes : sur le dernier chaque siège n'en a qu'une, donc
    aucun coup n'est une décision — ni pour un bot, ni pour un humain. Les
    pilotes déroulent ce pli tout seuls (cf. `pacing.last_trick_delay`), en
    laissant un joueur poser sa carte lui-même s'il est plus rapide.
    """
    return (env.phase() == 1 and not env.is_terminal()
            and sum(env.get_tricks_won()) == LAST_TRICK)


def cards_in_trick(env) -> int:
    """Cartes déjà posées sur le pli en cours (0 quand il n'est pas entamé)."""
    return sum(1 for c in env.get_current_trick() if c >= 0)


def trick_snapshot(state):
    """Image d'affichage d'un pli complet, avant que la table ne soit balayée.

    Le moteur résout le pli dès sa quatrième carte, donc l'état qu'il rend a
    déjà la table vide : on y réinjecte les quatre cartes, et on décrémente le
    compteur de plis du camp gagnant — sinon les mains adverses, comptées à
    `8 - plis joués`, perdraient une carte de trop.

    Sur la dernière levée l'état est terminal, et le panneau de fin recouvre la
    table : `deal_end_hold` dit au client de montrer le pli et d'attendre l'état
    terminal réel avant de l'afficher.
    """
    snap = dict(state)
    snap["current_trick"] = state["last_trick"]
    tricks_won = list(snap["tricks_won"])
    winner_team = state["last_trick_winner"] % 2
    tricks_won[winner_team] = max(0, tricks_won[winner_team] - 1)
    snap["tricks_won"] = tricks_won
    if state["is_terminal"]:
        snap["deal_end_hold"] = True
    return snap


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
    bids = []
    for a in action_ids:
        a = int(a)
        if int(env.phase()) == 0:
            bids.append(a)
        env.step(a)
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
        logger.warning("playgen call failed (%s): %s", type(e).__name__, e)
        return None


class TrickTracker:
    """Mixin for tracking completed tricks with point calculations."""

    # Card point values: [7, 8, 9, J, Q, K, 10, A]
    PLAIN_POINTS = [0, 0, 0, 2, 3, 4, 10, 11]
    TRUMP_POINTS = [0, 0, 14, 20, 3, 4, 10, 11]
    # Ordre de force à l'atout, indexé par rang — J > 9 > A > 10 > R > D > 8 > 7.
    # Copie de `card.rs::TRUMP_STRENGTH`, comme les deux barèmes ci-dessus.
    TRUMP_STRENGTH = [0, 1, 6, 7, 2, 3, 4, 5]

    def _init_trick_tracking(self):
        self.last_trick = None
        self.last_trick_winner = None
        self.last_trick_points = 0
        self.last_trick_lead = None
        self.completed_tricks = []  # list of {cards, winner, points, lead}
        self._trick_just_completed = False
        self._current_trick_lead = None  # lead of the trick in progress
        self._last_trick_trump = 0  # atout au moment où le pli s'est fermé
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

    def _trick_winner(self, trick, lead, trump_suit):
        """Le siège qui ramasse — recalculé, pas lu sur le moteur.

        `env.current_player()` after a `step` *is* the winner, but only for
        tricks 1 à 7 : `play.rs::resolve_trick` ne réaffecte `current_player`
        que dans la branche « il reste des plis ». Sur la 8e levée il garde donc
        le siège qui vient de poser la 4e carte, soit `(lead + 3) % 4` — mesuré
        faux sur 164 donnes /200 (127 fois même le camp était faux). Toute la
        chaîne d'affichage en héritait : vainqueur du dernier pli dans l'histo-
        rique, et `trick_snapshot` qui décrémentait le mauvais camp.

        Portage de `colver-core/src/engine/trick.rs::trick_winner`.
        """
        lead_card = trick[lead]
        lead_suit = lead_card >> 3
        best_trump_seat = None
        best_trump_strength = 0
        best_lead_seat = lead
        best_lead_rank = lead_card & 7
        if lead_suit == trump_suit:
            best_trump_seat = lead
            best_trump_strength = self.TRUMP_STRENGTH[lead_card & 7]
        for i in range(1, 4):
            seat = (lead + i) % 4
            card = trick[seat]
            suit, rank = card >> 3, card & 7
            if suit == trump_suit:
                strength = self.TRUMP_STRENGTH[rank]
                if best_trump_seat is None or strength > best_trump_strength:
                    best_trump_strength = strength
                    best_trump_seat = seat
            elif suit == lead_suit and rank > best_lead_rank:
                best_lead_rank = rank
                best_lead_seat = seat
        return best_trump_seat if best_trump_seat is not None else best_lead_seat

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
            self._last_trick_trump = trump
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
            self.last_trick_winner = self._trick_winner(
                self.last_trick, self.last_trick_lead, self._last_trick_trump)
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

    def __init__(self, ai_types=None, human_seat=2, dmc_model_path=None, bid_model_path=None, belief_model_path=None, dede_time_ms=None, dealer=None, scores=None):
        # ai_types: dict mapping seat -> ai_type (for non-human seats)
        # If not provided, default all AI seats to "dede"
        # dealer: siège qui donne (None = tirage au sort, comme une donne isolée)
        # scores: score cumulé de la partie [NS, EW], vu par les bots
        self.human_seat = human_seat
        if ai_types is None:
            ai_types = {}
        self.ai_types = ai_types
        self.dede_time_ms = dede_time_ms or 1000
        self.env = colver.Env()
        self.history = []
        self.bid_history = []
        self._init_trick_tracking()
        if dealer is None:
            self.env.reset()
        else:
            # `Env.reset()` tire le donneur au hasard ; en partie il tourne d'une
            # donne à l'autre, donc on distribue nous-mêmes (même loi uniforme).
            deck = list(range(32))
            random.shuffle(deck)
            self.env.redeal_with_hands(
                int(dealer) % 4, [deck[i * 8:(i + 1) * 8] for i in range(4)])
        self.initial_hands = [list(h) for h in self.env.get_hands()]
        if bid_model_path:
            self.env.load_bid_model(bid_model_path)
        # `ai_types` already names only the bot seats — in a salon room several
        # seats are human, so filtering by `human_seat` here would drop a bot.
        self.bots = AgentTable(
            self.ai_types,
            bid_model=bid_model_path,
            play_model=dmc_model_path,
            belief_model=belief_model_path,
            time_ms=self.dede_time_ms,
        )
        # Le score de la partie n'est pas décoratif : le bidder v6 lit une
        # observation score-aware, il annonce autrement à 900-200 qu'à 0-0.
        if scores:
            self.bots.set_scores(scores[0], scores[1])
        self.bots.init_deal(self.env)

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
        self._apply(action)
        self._detect_belote(player, belote_before)
        return self.get_state()

    def _apply(self, action):
        """Show the move to every bot, then advance the game."""
        self.bots.observe(self.env, action)
        self.env.step(action)
        self._finalize_trick_completion()

    def get_ai_action(self):
        player = int(self.env.current_player())
        if only_pass_is_legal(self.env):
            # Forced pass: no search worth running, for a bot or a human seat.
            return BID_PASS
        decision = self.bots.decide(self.env, player)
        if decision is None:
            # No bot seated here (or its models failed to load): fall back to
            # the rule-based bidder / a legal card rather than stalling.
            return int(self.env.bid_a_dd()) if self.env.phase() == 0 \
                else int(self.env.action_heuristic_play())
        return int(decision["action"])

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
        self._apply(action)
        self._detect_belote(player, belote_before)
        return action, name, self.get_state()



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

        # Bid NN, kept on the Env for the auction Q-value panel.
        if bid_model_path:
            self.env.load_bid_model(bid_model_path)

        self.bots = AgentTable(
            agents,
            bid_model=bid_model_path,
            play_model=dmc_model_path,
            belief_model=belief_model_path,
            time_ms=self.dede_time_ms,
        )
        self.bots.init_deal(self.env)

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
        """Compute the next action with the deciding bot's stats."""
        player = int(self.env.current_player())
        phase = int(self.env.phase())
        agent_type = self.bots.kind(player)

        decision = self.bots.decide(self.env, player)
        if decision is None:
            # Seat has no working bot: keep the game moving with a rule player.
            action = int(self.env.bid_a_dd()) if phase == 0 \
                else int(self.env.action_heuristic_play())
            stats = decision_stats(agent_type, None, error=self.bots.error(player))
        else:
            action = int(decision["action"])
            stats = decision_stats(agent_type, decision)

        # During the auction, always show the bid net's Q-values, whichever
        # bot is to speak — the panel is about the position, not the player.
        if phase == 0 and self.env.has_bid_model():
            nn_result = self.env.action_bid_nn()
            stats["bid_nn"] = {
                "q_values": [[int(a), round(float(q), 3)] for a, q in nn_result["q_values"]],
                "best_action": int(nn_result["best_action"]),
            }

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

        self.bots.observe(self.env, action)
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
                # A forced pass is no problem to pose — skip it as a cut point.
                if player == 2 and not only_pass_is_legal(env):
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

            # Bidding phase
            void = False
            while env.phase() == 0:
                player = int(env.current_player())
                action = int(env.bid_improved())
                bid_history.append({"player": player, "action": action,
                                     "name": colver.Env.action_name(action, 0)})
                env.step(action)
                if env.is_terminal():
                    void = True
                    break

            if void or env.phase() != 1:
                continue

            # Play phase: advance with heuristic until South's turn
            while env.phase() == 1 and not env.is_terminal() and int(env.current_player()) != 2:
                action = int(env.action_heuristic_play())
                env.step(action)

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

    # Short budget: the panel compares IS-DD against the oracle on one card,
    # it does not need production-strength search.
    PROBE_TIME_MS = 100

    def _isdd_probe(self) -> dict:
        """Ask a fresh IS-DD agent about the current position.

        Built per call and told the history is empty: the problem generator
        advanced the game with a heuristic player, so there is no belief state
        worth carrying, and a one-shot judgement is what the panel shows.
        """
        bots = AgentTable({2: "dede"}, time_ms=self.PROBE_TIME_MS)
        bots.init_deal(self.env)
        decision = bots.decide(self.env, int(self.env.current_player()))
        if decision is None:
            raise RuntimeError("IS-DD agent unavailable")
        return decision

    def evaluate(self, player_action: int) -> dict:
        """Evaluate player's card against the oracle, DouDou and IS-DD."""
        env = self.env
        t0 = time.monotonic()
        oracle_action = int(env.action_oracle_dd())
        oracle_elapsed = round((time.monotonic() - t0) * 1000, 1)

        dmc_result = None
        if env.has_dmc_model():
            dmc_result = env.action_dmc_with_stats()

        isdd_result = self._isdd_probe()

        return {
            "player_action": player_action,
            "player_action_name": colver.Env.action_name(player_action, 1),
            "oracle_action": oracle_action,
            "oracle_action_name": colver.Env.action_name(oracle_action, 1),
            "oracle_elapsed_ms": oracle_elapsed,
            "dmc_action": int(dmc_result["best_action"]) if dmc_result else None,
            "dmc_action_name": colver.Env.action_name(int(dmc_result["best_action"]), 1) if dmc_result else None,
            "dmc_q_values": [[int(c), round(float(q), 4)] for c, q in dmc_result["q_values"]] if dmc_result else [],
            "isdd_action": int(isdd_result["action"]),
            "isdd_action_name": colver.Env.action_name(int(isdd_result["action"]), 1),
            "isdd_card_scores": [[int(c), round(float(s), 1)] for c, s in isdd_result["candidates"]],
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

        bots = AgentTable(
            {seat: "dede" for seat in range(4)},
            bid_model=self.bid_model_path,
            belief_model=self.belief_model_path,
            time_ms=self.GEN_PLAY_TIME_MS,
        )
        bots.init_deal(env)

        self.initial_hands = [list(h) for h in env.get_hands()]
        self.dealer = int(env.get_dealer())
        self.all_actions = []
        self.action_idx = 0

        while not env.is_terminal():
            player = int(env.current_player())
            phase = int(env.phase())
            if phase == 0:
                # The NN bidder, not the DD oracle: the oracle sees all four
                # hands, so it produces degenerate optimal-bid-then-pass
                # auctions with none of the signaling the page is about.
                action = int(env.action_bid_nn()["best_action"]) if env.has_bid_model() \
                    else int(env.bid_a_dd())
            else:
                decision = bots.decide(env, player)
                action = int(decision["action"]) if decision \
                    else int(env.action_heuristic_play())
            self.all_actions.append((player, action, phase))
            bots.observe(env, action)
            env.step(action)

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
        self.all_actions = []
        for a in action_ids:
            self.all_actions.append((int(env.current_player()), int(a), int(env.phase())))
            env.step(a)
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
        self.env.step(action)
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
            self.action_idx = 0
        # Replay up to target
        while self.action_idx < target:
            player, action, phase = self.all_actions[self.action_idx]
            self.env.step(action)
            self.action_idx += 1
        return self._get_state_info()

    # 30 worlds keeps the marginals readable while staying ~6-8s/position with
    # the v2 model (43MB, ~3-5x slower per world than v1).
    PLAYGEN_WORLDS = 30
    PLAYGEN_TEMP = 0.8
    # GPU sidecar: worlds are ~50x cheaper, so marginals get much smoother.
    PLAYGEN_WORLDS_GPU = 200

    def _playgen_marginals(self, env_actions_prefix: int, observer: int):
        """Play-phase playgen marginals via the GPU sidecar (200 worlds).
        Returns None if the sidecar is disabled/unreachable (CPU fallback then)."""
        if playgen_gpu.enabled():
            actions = [(p, a) for p, a, _ in self.all_actions[:env_actions_prefix]]
            w = playgen_gpu.beliefs(
                self.dealer, self.initial_hands, actions, observer,
                n_worlds=self.PLAYGEN_WORLDS_GPU, temperature=self.PLAYGEN_TEMP,
            )
            if w is not None:
                return w
        return None

    def _analyst(self, idx: int, observer: int):
        """A playgen analyst replayed to position `idx` from `observer`'s seat.

        Rebuilt per query rather than kept live: the page jumps freely between
        positions and observers, and a replay costs a few milliseconds against
        the seconds a world sample takes.
        """
        if not self.playgen_model_path:
            return None
        actions = [a for _, a, _ in self.all_actions[:idx]]
        try:
            return colver.Analyst.replay(
                self.playgen_model_path, self.dealer, self.initial_hands, actions, observer,
            )
        except Exception as e:  # noqa: BLE001 — playgen is best-effort here
            logger.warning("playgen analyst unavailable: %s", e)
            return None

    @staticmethod
    def _marginals_from_deals(deals):
        """Aggregate sampled full deals (list of 4-hand worlds) into [4][32]
        marginals P(seat holds card)."""
        if not deals:
            return None
        counts = [[0.0] * 32 for _ in range(4)]
        n = 0
        for world in deals:
            for p in range(4):
                for c in world[p]:
                    counts[p][int(c)] += 1.0
            n += 1
        if n == 0:
            return None
        return [[counts[p][c] / n for c in range(32)] for p in range(4)]

    def _auction_marginals(self, idx: int, observer: int, env):
        """Playgen marginals during the auction (v2 model): sample deals
        conditioned on the bids so far, then aggregate. GPU sidecar first."""
        if playgen_gpu.enabled():
            actions = [(p, a) for p, a, _ in self.all_actions[:idx]]
            deals = playgen_gpu.auction_deals(
                self.dealer, self.initial_hands, actions, observer,
                n_worlds=self.PLAYGEN_WORLDS_GPU, temperature=self.PLAYGEN_TEMP,
            )
            m = self._marginals_from_deals(deals)
            if m is not None:
                return m
        analyst = self._analyst(idx, observer)
        if analyst is None:
            return None
        deals = _safe_playgen(
            analyst.auction_deals, env, self.PLAYGEN_WORLDS, self.PLAYGEN_TEMP,
        )
        return self._marginals_from_deals(deals)

    def _compute_playgen(self, idx: int, observer: int, env):
        """Playgen marginals at position `idx` for either phase (v2 model
        samples auction-conditioned deals during bidding). None if terminal."""
        if env.is_terminal():
            return None
        if int(env.phase()) == 0:
            return self._auction_marginals(idx, observer, env)
        w = self._playgen_marginals(idx, observer)
        if w is None:
            analyst = self._analyst(idx, observer)
            if analyst is None:
                return None
            w = _safe_playgen(
                analyst.marginals, env, self.PLAYGEN_WORLDS, self.PLAYGEN_TEMP,
            )
        return w

    def get_beliefs(self, observer: int) -> dict:
        """Return playgen marginals at the current position + ground truth hands.

        Playgen is the only source the page shows: the belief net and the
        heuristic were dropped because playgen dominates them (see the
        world-credibility benchmark). Marginals cost ~1s of MC sampling, so
        they are cached per (position, observer) — the precompute sweep fills
        the same cache.
        """
        idx = self.action_idx
        playgen = None
        if self.playgen_model_path:
            playgen = _belief_cache_get(self.game_cfn, "playgen", idx, observer)
            if playgen is None:
                playgen = self._compute_playgen(idx, observer, self.env)
                _belief_cache_put(self.game_cfn, "playgen", idx, observer, playgen)
        return {
            "observer": observer,
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
        # Every non-terminal position (auction + play) — the v2 model samples
        # auction-conditioned deals, so beliefs exist during the bids too.
        self._sweep_total = len(self.all_actions)
        if not self.playgen_model_path or self._sweep_total == 0:
            self._sweep_env = None
            self._sweep_total = 0
            return 0
        # A bare env: it only tracks the position, the analyst carries the model.
        self._sweep_env = colver.Env.deal_with_hands(self.dealer, self.initial_hands)
        return self._sweep_total

    def precompute_step(self):
        """Advance the sweep by one position (auction or play), filling the
        shared cache. Returns (done, total) per computed position, or None when
        the sweep is finished. Called repeatedly from an executor so the caller
        can stream progress between positions."""
        if self._sweep_env is None:
            return None
        env = self._sweep_env
        while self._sweep_i < len(self.all_actions):
            idx = self._sweep_i
            obs = self._sweep_observer
            computed = False
            if not env.is_terminal():
                if _belief_cache_get(self.game_cfn, "playgen", idx, obs) is None:
                    w = self._compute_playgen(idx, obs, env)
                    _belief_cache_put(self.game_cfn, "playgen", idx, obs, w)
                self._sweep_done += 1
                computed = True
            _, action, _ = self.all_actions[self._sweep_i]
            env.step(action)
            self._sweep_i += 1
            if computed:
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




_counting_tls = threading.local()


def _counting_env(bid_model_path, dmc_model_path, dealer, hands):
    """Un `Env` par thread, modèles chargés une fois pour toutes.

    Recharger DouDou50 (10 Mo) à chaque problème coûterait bien plus cher que
    de jouer la donne. Même remède que `server._get_doudou_env` :
    `redeal_with_hands` remet l'état à neuf sans toucher aux poids.
    """
    key = (bid_model_path, dmc_model_path)
    env = getattr(_counting_tls, "env", None)
    if env is not None and getattr(_counting_tls, "key", None) == key:
        env.redeal_with_hands(dealer, hands)
        return env
    env = colver.Env.deal_with_hands(dealer, hands)
    if bid_model_path:
        env.load_bid_model(bid_model_path)
    if dmc_model_path:
        env.load_dmc_model(dmc_model_path)
    _counting_tls.env = env
    _counting_tls.key = key
    return env


class CountingSession(TrickTracker):
    """Une donne réelle, découpée en plis, pour l'entraînement au comptage.

    Deux sources, un seul format de sortie. Une donne fraîche jouée par les
    bots (`generate`), ou une donne rejouée depuis la base (`from_game`) : dans
    les deux cas des plis *joués*, donc un ordre de cartes plausible et un
    atout qui est celui d'un vrai contrat — c'est toute la différence avec des
    plis tirés au hasard, où l'on compterait des situations qu'on ne verra
    jamais à une table.

    La génération est la source par défaut parce que la base est mince : une
    trentaine de donnes terminées, quand la page en consomme une toutes les
    vingt secondes. « Mes parties » reste proposé à qui est connecté, pour
    recompter ses propres donnes.

    Tout part au client d'un coup, la donne entière : la correction est alors
    instantanée et locale. Un curieux peut lire la réponse dans la console —
    c'est un exercice, la seule personne qu'il tromperait est lui-même.
    """

    # Une donne jouée à l'heuristique coûte des microsecondes ; le seul cas à
    # rejouer est l'enchère qui meurt sur quatre passes (aucun pli à compter).
    MAX_ATTEMPTS = 40

    def __init__(self, bid_model_path=None, dmc_model_path=None):
        self.bid_model_path = bid_model_path
        self.dmc_model_path = dmc_model_path
        self.env = None

    # ----- sources ---------------------------------------------------------

    def generate(self) -> dict:
        """Une donne neuve, enchérie puis jouée par les bots."""
        for _ in range(self.MAX_ATTEMPTS):
            deck = list(range(32))
            random.shuffle(deck)
            env = _counting_env(
                self.bid_model_path, self.dmc_model_path,
                random.randrange(4), [deck[i * 8:(i + 1) * 8] for i in range(4)])
            self.env = env
            self._init_trick_tracking()

            while env.phase() == 0 and not env.is_terminal():
                self._step(self._bid_action(env))
            if env.is_terminal() or env.phase() != 1:
                continue  # enchère morte sur quatre passes : rien à compter

            while not env.is_terminal():
                self._step(self._play_action(env))

            return self._payload(source="generee")
        raise RuntimeError("Aucune donne jouable générée")

    def from_game(self, game_data) -> dict:
        """Une donne enregistrée, rejouée coup par coup depuis ses actions."""
        self.env = colver.Env.deal_with_hands(
            game_data["dealer"], game_data["hands"])
        self._init_trick_tracking()
        for entry in game_data["actions"]:
            if self.env.is_terminal():
                break
            self._step(int(entry["action"]))
        if len(self.completed_tricks) != 8:
            raise RuntimeError("Donne enregistrée incomplète")
        return self._payload(source="base", game_id=game_data["id"])

    # ----- moteur ----------------------------------------------------------

    def _bid_action(self, env):
        if env.has_bid_model():
            return int(env.action_bid_nn()["best_action"])
        return int(env.bid_improved())

    def _play_action(self, env):
        if env.has_dmc_model():
            return int(env.action_dmc_with_stats()["best_action"])
        return int(env.action_heuristic_play())

    def _step(self, action):
        """Un coup, en tenant à jour les plis et les annonces de belote."""
        player = int(self.env.current_player())
        belote_before = list(self.env.get_belote())
        self._check_trick_completion(action)
        self.env.step(action)
        self._finalize_trick_completion()
        self._detect_belote(player, belote_before)
        if self._belote_event:
            self._announces.append({
                "trick": len(self.completed_tricks) - (1 if self.trick_just_completed else 0),
                "seat": self._belote_player,
                "card": action,
                "event": self._belote_event,
            })
        self.trick_just_completed = False

    def _init_trick_tracking(self):
        super()._init_trick_tracking()
        self._announces = []

    # ----- sortie ----------------------------------------------------------

    def _payload(self, source, game_id=None) -> dict:
        """La donne telle que la page la consomme : huit plis dans l'ordre.

        On envoie toujours les huit, jamais un prefixe : c'est le client qui
        décide combien il en montre selon le niveau, et la correction peut
        alors dérouler la donne entière une fois la réponse donnée.
        """
        contract = self.env.get_contract()
        trump = contract.get("trump", 0)
        tricks = []
        for i, t in enumerate(self.completed_tricks):
            tricks.append({
                "no": i + 1,
                "cards": list(t["cards"]),
                "lead": t["lead"],
                "winner": t["winner"],
                "points": t["points"],
                "announces": [a for a in self._announces if a["trick"] == i],
            })
        # Le dix de der, on le *lit* au lieu de le recoder : `resolve_trick` l'a
        # déjà versé dans `get_points()` (10, ou 100 si le camp a les 8 plis).
        # L'écart entre la somme des plis et le moteur est du même coup la
        # meilleure assertion possible sur le vainqueur du dernier pli — la
        # seule valeur que cette page ne peut pas se permettre de rater.
        cards_pts = [0, 0]
        for t in self.completed_tricks:
            cards_pts[t["winner"] % 2] += t["points"]
        engine_pts = list(self.env.get_points())
        der_team = self.completed_tricks[-1]["winner"] % 2
        der_value = engine_pts[der_team] - cards_pts[der_team]
        if (sum(cards_pts) != 152 or der_value not in (10, 100)
                or engine_pts[1 - der_team] != cards_pts[1 - der_team]):
            raise RuntimeError(
                f"décompte incohérent : plis={cards_pts} moteur={engine_pts}")

        belote = list(self.env.get_belote())
        return {
            "trump": trump,
            "contract": contract,
            "dealer": int(self.env.get_dealer()),
            "tricks": tricks,
            # Points *cartes* par camp, dix de der compris — la vérité du moteur.
            "points": engine_pts,
            "card_points": cards_pts,          # les mêmes, sans le dix de der
            "der": {"team": der_team, "value": der_value},
            # 2 = Roi et Dame d'atout tous deux joués, donc belote annoncée.
            "belote": [20 if b == 2 else 0 for b in belote],
            "source": source,
            "game_id": game_id,
        }
