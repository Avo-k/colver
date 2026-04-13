"""Bid a Dede (v2) opponent for Bumblebid training.

Loads the raw .bin weights and runs inference in PyTorch.
Architecture: 108 → 512 (LN+ReLU) × 3 → Dueling DQN (V + A → 43 Q-values).

Obs builder reimplemented in numpy to match bid_obs.rs exactly.
"""

import struct

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

BID_OBS_DIM = 108
NUM_ACTIONS = 43


# ---------------------------------------------------------------------------
# Observation builder (matches colver-core/src/bid/bid_obs.rs)
# ---------------------------------------------------------------------------
def _decode_bid(action):
    """Decode action 1-40 to (val_enc, suit_idx)."""
    if action <= 36:
        idx = action - 1
        return idx // 4 + 8, idx % 4
    else:
        return 25, action - 37


def build_bid_obs_batch(hands, seats, dealers, bid_histories, n_bid_tokens):
    """Build [N, 108] obs for bid_v2 opponents.

    Args:
        hands: [N, 4] uint32 — raw hand bitmasks per seat
        seats: [N] int — current player seat
        dealers: [N] int — dealer seat
        bid_histories: list of N lists of (seat, action) pairs
        n_bid_tokens: not used here, kept for interface compat
    Returns: [N, 108] float32 numpy
    """
    N = len(seats)
    obs = np.zeros((N, BID_OBS_DIM), dtype=np.float32)

    for i in range(N):
        me = int(seats[i])
        dealer = int(dealers[i])
        hand = int(hands[i, me])

        pos = 0
        # Block 1: hand (32)
        for bit in range(32):
            if hand & (1 << bit):
                obs[i, pos + bit] = 1.0
        pos += 32

        # Block 2: bid history (72 = 12 slots × 6)
        first_bidder = (dealer + 1) % 4
        rel_offset = (first_bidder + 4 - me) % 4
        history = bid_histories[i]
        if len(history) > 12:
            history = history[-12:]
        for j, (seat_h, action) in enumerate(history):
            slot = rel_offset + j
            if slot >= 12:
                break
            base = pos + slot * 6
            if action == 0:  # PASS
                obs[i, base] = 0.2
            elif action == 41:  # COINCHE
                obs[i, base] = 0.8
            elif action == 42:  # SURCOINCHE
                obs[i, base] = 1.0
            elif 1 <= action <= 40:
                val_enc, suit_idx = _decode_bid(action)
                if val_enc == 25:  # capot
                    obs[i, base] = 0.6
                    obs[i, base + 1] = 1.0
                else:
                    obs[i, base] = 0.4
                    obs[i, base + 1] = (val_enc * 10.0) / 250.0
                obs[i, base + 2 + suit_idx] = 1.0
        pos += 72

        # Block 3: position (4)
        rel_pos = (me + 4 - dealer) % 4
        obs[i, pos + rel_pos] = 1.0
        pos += 4

    return obs


# ---------------------------------------------------------------------------
# Dueling DQN model (matches bid_net.rs)
# ---------------------------------------------------------------------------
class BidNetV2(nn.Module):
    """Bid a Dede: 108 → 512³ (LN+ReLU) → Dueling DQN → 43."""

    def __init__(self, obs_dim=108, hidden=512, n_layers=3):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.n_layers = n_layers

        # Hidden layers
        self.linears = nn.ModuleList()
        self.norms = nn.ModuleList()
        in_dim = obs_dim
        for _ in range(n_layers):
            self.linears.append(nn.Linear(in_dim, hidden))
            self.norms.append(nn.LayerNorm(hidden, eps=1e-5))
            in_dim = hidden

        # Dueling heads
        self.value_head = nn.Linear(hidden, 1)
        self.adv_head = nn.Linear(hidden, NUM_ACTIONS)

    def forward(self, x):
        for lin, norm in zip(self.linears, self.norms):
            x = F.relu(norm(lin(x)))

        value = self.value_head(x)  # [B, 1]
        adv = self.adv_head(x)      # [B, 43]
        q = value + adv - adv.mean(dim=-1, keepdim=True)
        return q

    @staticmethod
    def load_from_bin(path, hidden=512, n_layers=3, obs_dim=108):
        """Load from raw f32 binary (same format as Rust BidNet)."""
        with open(path, "rb") as f:
            data = f.read()
        floats = np.frombuffer(data, dtype=np.float32)

        model = BidNetV2(obs_dim=obs_dim, hidden=hidden, n_layers=n_layers)
        idx = 0
        h = hidden

        # Hidden layers: W (in×h), b (h), gamma (h), beta (h)
        in_dim = obs_dim
        for i in range(n_layers):
            # W stored as (out_dim × in_dim) row-major → reshape directly to [h, in_dim]
            w = floats[idx:idx + in_dim * h].reshape(h, in_dim)
            idx += in_dim * h
            b = floats[idx:idx + h]
            idx += h
            gamma = floats[idx:idx + h]
            idx += h
            beta = floats[idx:idx + h]
            idx += h

            model.linears[i].weight.data = torch.from_numpy(w.copy())
            model.linears[i].bias.data = torch.from_numpy(b.copy())
            model.norms[i].weight.data = torch.from_numpy(gamma.copy())
            model.norms[i].bias.data = torch.from_numpy(beta.copy())
            in_dim = h

        # Dueling: value head (h→1), advantage head (h→43)
        # Value: W (h,), b (1)
        w_val = floats[idx:idx + h].reshape(1, h)
        idx += h
        b_val = floats[idx:idx + 1]
        idx += 1

        model.value_head.weight.data = torch.from_numpy(w_val.copy())
        model.value_head.bias.data = torch.from_numpy(b_val.copy())

        # Advantage: W stored as (43 × h) row-major → reshape directly to [43, h]
        w_adv = floats[idx:idx + h * NUM_ACTIONS].reshape(NUM_ACTIONS, h)
        idx += h * NUM_ACTIONS
        b_adv = floats[idx:idx + NUM_ACTIONS]
        idx += NUM_ACTIONS

        model.adv_head.weight.data = torch.from_numpy(w_adv.copy())
        model.adv_head.bias.data = torch.from_numpy(b_adv.copy())

        assert idx == len(floats), f"Weight count mismatch: consumed {idx}, file has {len(floats)}"
        return model
