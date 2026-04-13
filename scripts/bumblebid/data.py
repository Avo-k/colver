"""DD pool loader, tokenizer, and scoring for Bumblebid self-play training.

Tokenization:
  [CLS] [POS_x] [card_1 ... card_8] [bid_val bid_suit] [bid_val bid_suit] ...

Each bid round = 2 tokens. PASS/COINCHE/SURCOINCHE use SUIT_NULL as 2nd token.
Variable-length sequences, padded to MAX_SEQ_LEN for batching.
"""

import struct
from itertools import permutations

import numpy as np
import torch

try:
    from .model import (
        P_CLS, P_POS0, P_RANK0, P_VAL0, P_CAPOT,
        P_PASS, P_COINCHE, P_SURCOINCHE, P_NONE,
        S_NULL, NUM_BID_ACTIONS, MAX_SEQ_LEN,
    )
except ImportError:
    from model import (
        P_CLS, P_POS0, P_RANK0, P_VAL0, P_CAPOT,
        P_PASS, P_COINCHE, P_SURCOINCHE, P_NONE,
        S_NULL, NUM_BID_ACTIONS, MAX_SEQ_LEN,
    )

# All 24 suit permutations
SUIT_PERMS = list(permutations(range(4)))
SUIT_PERM_T = torch.tensor(SUIT_PERMS, dtype=torch.long)

# Action perm table: [24, 43] — maps old action to new action under suit perm
_apt = torch.arange(43).unsqueeze(0).expand(24, -1).clone()
for _pi, _perm in enumerate(SUIT_PERMS):
    for _a in range(1, 41):
        if _a <= 36:
            _i = _a - 1
            _apt[_pi, _a] = (_i // 4) * 4 + _perm[_i % 4] + 1
        else:
            _apt[_pi, _a] = 37 + _perm[_a - 37]
ACTION_PERM_TABLE = _apt

# Inverse: [24, 43] — given new action, what was the old action?
INV_ACTION_PERM_TABLE = torch.zeros_like(ACTION_PERM_TABLE)
for _pi in range(24):
    for _old in range(43):
        INV_ACTION_PERM_TABLE[_pi, ACTION_PERM_TABLE[_pi, _old]] = _old

# Bid action -> (value_token, suit_idx) for tokenization
BID_VALUES = [8, 9, 10, 11, 12, 13, 14, 15, 16]  # 80..160 in units of 10


# ---------------------------------------------------------------------------
# Pool loading
# ---------------------------------------------------------------------------
def load_dd_pool(path: str):
    """Load COLVDD01 or COLVDR01 binary pool.
    Returns (dealers[N], hands[N,4], dd_pts[N,4]).
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        count = struct.unpack("<Q", f.read(8))[0]
        record_size = 21 if magic == b"COLVDD01" else 25
        if magic not in (b"COLVDD01", b"COLVDR01"):
            raise ValueError(f"Unknown magic: {magic!r}")
        raw = np.frombuffer(f.read(count * record_size), dtype=np.uint8)

    raw = raw.reshape(count, record_size)
    dealers = raw[:, 0].copy()
    h = raw[:, 1:17].reshape(count, 4, 4).astype(np.uint32)
    hands = h[:, :, 0] | (h[:, :, 1] << 8) | (h[:, :, 2] << 16) | (h[:, :, 3] << 24)
    dd_pts = raw[:, 17:21].copy()
    print(f"Loaded {count:,} deals from {path}", flush=True)
    return dealers, hands, dd_pts


def load_bumblebid_pool(path: str):
    """Load COLVBB01/COLVBB02 pre-tokenized pool.
    Returns (dealers[N], hand_primary[N,4,10], hand_suits[N,4,10], dd_pts[N,4], real_pts_or_None).
    COLVBB02 includes real_pts[N,4] (DouDou50 play points); COLVBB01 returns None.
    """
    with open(path, "rb") as f:
        magic = f.read(8)
        if magic not in (b"COLVBB01", b"COLVBB02"):
            raise ValueError(f"Expected COLVBB01/02, got {magic!r}")
        enriched = (magic == b"COLVBB02")
        count = struct.unpack("<Q", f.read(8))[0]
        record_size = 89 if enriched else 85  # +4 for real_pts
        raw = np.frombuffer(f.read(count * record_size), dtype=np.uint8)

    raw = raw.reshape(count, record_size)
    dealers = raw[:, 0].copy()
    dd_pts = raw[:, 1:5].copy()
    if enriched:
        real_pts = raw[:, 5:9].copy()
        token_offset = 9
    else:
        real_pts = None
        token_offset = 5
    tokens = raw[:, token_offset:token_offset + 80].reshape(count, 4, 20)
    hand_primary = tokens[:, :, :10].copy()
    hand_suits = tokens[:, :, 10:20].copy()
    tag = "enriched " if enriched else ""
    print(f"Loaded {count:,} {tag}pre-tokenized deals from {path}", flush=True)
    return dealers, hand_primary, hand_suits, dd_pts, real_pts


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def _round10(x):
    return ((x + 5) // 10) * 10


def compute_deal_score(dd_pts, contract_value, trump_suit, taker_team):
    """Compute (NS_score, EW_score) given DD points and contract.

    Returns (0, 0) for void deals (contract_value=0).
    """
    if contract_value == 0:
        return (0, 0)

    ns_pts = int(dd_pts[trump_suit])
    ew_pts = (252 - ns_pts) if ns_pts in (0, 252) else (162 - ns_pts)
    taker_pts = ns_pts if taker_team == 0 else ew_pts
    defense_pts = ew_pts if taker_team == 0 else ns_pts

    is_capot = contract_value == 250

    if is_capot:
        if defense_pts == 0:
            t_sc, d_sc = 500, 0
        else:
            t_sc, d_sc = 0, 500
    else:
        if taker_pts >= contract_value:
            t_sc = _round10(taker_pts + contract_value)
            d_sc = _round10(defense_pts)
        else:
            t_sc = 0
            d_sc = _round10(160 + contract_value)

    return (t_sc, d_sc) if taker_team == 0 else (d_sc, t_sc)


# ---------------------------------------------------------------------------
# Vectorized scoring (numpy, batch)
# ---------------------------------------------------------------------------
def compute_deal_scores_batch(dd_pts, contract_values, trump_suits, taker_teams):
    """Vectorized scoring for N envs.

    Args:
        dd_pts: [N, 4] uint8
        contract_values: [N] int — 0 for void, 80-160 or 250
        trump_suits: [N] int
        taker_teams: [N] int (0=NS, 1=EW)

    Returns: (ns_scores[N], ew_scores[N]) as float32
    """
    N = len(contract_values)
    ns_sc = np.zeros(N, dtype=np.float32)
    ew_sc = np.zeros(N, dtype=np.float32)

    for i in range(N):
        cv = int(contract_values[i])
        if cv == 0:
            continue
        ns, ew = compute_deal_score(dd_pts[i], cv, int(trump_suits[i]), int(taker_teams[i]))
        ns_sc[i] = ns
        ew_sc[i] = ew

    return ns_sc, ew_sc


# ---------------------------------------------------------------------------
# Tokenizer (kept for evaluation / single-step use)
# ---------------------------------------------------------------------------
def extract_cards(hand):
    """Extract (rank, suit) pairs from 32-bit hand bitmask."""
    cards = []
    for bit in range(32):
        if hand & (1 << bit):
            cards.append((bit % 8, bit // 8))
    return cards


def action_to_tokens(action):
    """Convert bid action (0-42) to (primary_id, suit_id) token pair."""
    if action == 0:
        return (P_PASS, S_NULL)
    elif action == 41:
        return (P_COINCHE, S_NULL)
    elif action == 42:
        return (P_SURCOINCHE, S_NULL)
    elif action <= 36:
        idx = action - 1
        val_idx, suit_idx = idx // 4, idx % 4
        return (P_VAL0 + val_idx, suit_idx)
    else:  # 37-40: capot
        suit_idx = action - 37
        return (P_CAPOT, suit_idx)


def tokenize_state(hand, seat, dealer, bid_history):
    """Build (primary_ids, suit_ids, seq_len) for current auction state."""
    cards = extract_cards(hand)
    assert len(cards) == 8
    cards.sort(key=lambda c: c[1] * 8 + c[0])

    pos = (seat + 4 - dealer) % 4

    primary = [P_CLS, P_POS0 + pos]
    suits = [S_NULL, S_NULL]

    for rank, suit in cards:
        primary.append(P_RANK0 + rank)
        suits.append(suit)

    for _seat, action in bid_history:
        p, s = action_to_tokens(action)
        primary.append(p)
        suits.append(s)
        if action == 0 or action >= 41:
            primary.append(P_NONE)
            suits.append(S_NULL)
        else:
            primary.append(P_NONE)
            suits.append(s)

    seq_len = len(primary)

    while len(primary) < MAX_SEQ_LEN:
        primary.append(0)
        suits.append(0)

    return primary[:MAX_SEQ_LEN], suits[:MAX_SEQ_LEN], min(seq_len, MAX_SEQ_LEN)


# ---------------------------------------------------------------------------
# Bid action <-> token lookup tables (numpy, for vectorized env)
# ---------------------------------------------------------------------------
# action_primary[43]: primary token ID for each action
# action_suit[43]: suit token ID for each action
# action_suit2[43]: suit ID for the 2nd token (S_NULL for pass/coinche/surcoinche)
_action_primary = np.zeros(43, dtype=np.int64)
_action_suit = np.zeros(43, dtype=np.int64)
_action_suit2 = np.zeros(43, dtype=np.int64)  # suit of 2nd token

for _a in range(43):
    p, s = action_to_tokens(_a)
    _action_primary[_a] = p
    _action_suit[_a] = s
    if _a == 0 or _a >= 41:
        _action_suit2[_a] = S_NULL
    else:
        _action_suit2[_a] = s

ACTION_PRIMARY_LUT = _action_primary
ACTION_SUIT_LUT = _action_suit
ACTION_SUIT2_LUT = _action_suit2


# ---------------------------------------------------------------------------
# Bidding environment (Python, for single-game evaluation)
# ---------------------------------------------------------------------------
BID_PASS = 0
BID_COINCHE = 41
BID_SURCOINCHE = 42

BID_VALUES_ENC = [8, 9, 10, 11, 12, 13, 14, 15, 16, 25]  # 80-160 + capot


def decode_bid(action):
    """Decode action to (value_enc, suit_idx). value_enc*10 = contract value."""
    if action <= 36:
        idx = action - 1
        return BID_VALUES_ENC[idx // 4], idx % 4
    else:
        return 25, action - 37  # capot


class BiddingEnv:
    """Single bidding environment for evaluation."""

    def __init__(self):
        self.dealer = 0
        self.hands = [0, 0, 0, 0]
        self.dd_pts = [0, 0, 0, 0]
        self.bid_history = []
        self.current_player_seat = 0
        self.consecutive_passes = 0
        self.current_bid = None
        self.coinche_level = 0
        self.done = False

    def reset(self, dealer, hands, dd_pts):
        self.dealer = dealer
        self.hands = list(hands)
        self.dd_pts = list(dd_pts)
        self.bid_history = []
        self.current_player_seat = (dealer + 1) % 4
        self.consecutive_passes = 0
        self.current_bid = None
        self.coinche_level = 0
        self.done = False

    def current_player(self):
        return self.current_player_seat

    def current_team(self):
        return self.current_player_seat % 2

    def legal_actions_mask(self):
        mask = np.zeros(NUM_BID_ACTIONS, dtype=np.float32)
        mask[BID_PASS] = 1.0

        if self.current_bid is None:
            for a in range(1, 41):
                mask[a] = 1.0
        elif self.coinche_level == 0:
            cur_val, cur_suit, cur_team = self.current_bid
            for a in range(1, 41):
                val, suit = decode_bid(a)
                if val > cur_val:
                    mask[a] = 1.0
            if self.current_team() != cur_team:
                mask[BID_COINCHE] = 1.0
        elif self.coinche_level == 1:
            cur_val, cur_suit, cur_team = self.current_bid
            if self.current_team() == cur_team:
                mask[BID_SURCOINCHE] = 1.0

        return mask

    def step(self, action):
        seat = self.current_player_seat
        team = seat % 2
        self.bid_history.append((seat, action))

        if action == BID_PASS:
            self.consecutive_passes += 1
        elif action == BID_COINCHE:
            self.coinche_level = 1
            self.consecutive_passes = 0
        elif action == BID_SURCOINCHE:
            self.coinche_level = 2
            self.consecutive_passes = 0
            self.done = True
        else:
            val, suit = decode_bid(action)
            self.current_bid = (val, suit, team)
            self.coinche_level = 0
            self.consecutive_passes = 0

        if not self.done:
            if self.current_bid is None and self.consecutive_passes >= 4:
                self.done = True
            elif self.current_bid is not None and self.consecutive_passes >= 3:
                self.done = True

        self.current_player_seat = (seat + 1) % 4
        return self.done

    def compute_reward(self):
        if self.current_bid is None:
            return (0.0, 0.0)
        val_enc, suit, taker_team = self.current_bid
        contract_value = val_enc * 10
        ns, ew = compute_deal_score(self.dd_pts, contract_value, suit, taker_team)
        return ((ns - ew) / 500.0, (ew - ns) / 500.0)

    def get_tokens(self):
        hand = self.hands[self.current_player_seat]
        return tokenize_state(
            hand, self.current_player_seat, self.dealer, self.bid_history,
        )


# ---------------------------------------------------------------------------
# Replay buffer
# ---------------------------------------------------------------------------
class ReplayBuffer:
    """Simple circular replay buffer storing tokenized transitions."""

    def __init__(self, capacity):
        self.capacity = capacity
        self.primary = torch.zeros(capacity, MAX_SEQ_LEN, dtype=torch.long)
        self.suits = torch.zeros(capacity, MAX_SEQ_LEN, dtype=torch.long)
        self.seq_lens = torch.zeros(capacity, dtype=torch.long)
        self.masks = torch.zeros(capacity, NUM_BID_ACTIONS, dtype=torch.float32)
        self.actions = torch.zeros(capacity, dtype=torch.long)
        self.rewards = torch.zeros(capacity, dtype=torch.float32)
        self.size = 0
        self.pos = 0

    def push(self, primary, suit_ids, seq_len, mask, action, reward):
        i = self.pos
        self.primary[i, :len(primary)] = torch.tensor(primary, dtype=torch.long)
        self.suits[i, :len(suit_ids)] = torch.tensor(suit_ids, dtype=torch.long)
        self.seq_lens[i] = seq_len
        self.masks[i] = torch.tensor(mask, dtype=torch.float32)
        self.actions[i] = action
        self.rewards[i] = reward
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size, device):
        idx = torch.randint(self.size, (batch_size,))
        max_len = self.seq_lens[idx].max().item()
        return {
            "primary": self.primary[idx, :max_len].to(device),
            "suits": self.suits[idx, :max_len].to(device),
            "seq_lens": self.seq_lens[idx].to(device),
            "masks": self.masks[idx].to(device),
            "actions": self.actions[idx].to(device),
            "rewards": self.rewards[idx].to(device),
        }
