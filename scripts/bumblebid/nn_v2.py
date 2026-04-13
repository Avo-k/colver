"""nn_v2 (Bid a Dede) opponent model for Bumblebid training.

Loads the Rust .bin weights into a PyTorch DuelingDQN and provides
vectorized bid observation encoding + action selection.
"""

import struct

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# Dueling DQN matching Rust BidNet architecture
# ---------------------------------------------------------------------------
class DuelingBidNet(nn.Module):
    """Dueling DQN matching colver-core BidNet (bid_net.rs)."""

    def __init__(self, obs_dim=108, hidden=512, n_layers=3, n_actions=43):
        super().__init__()
        self.n_actions = n_actions
        layers = []
        in_dim = obs_dim
        for _ in range(n_layers):
            layers.append(nn.Linear(in_dim, hidden))
            layers.append(nn.LayerNorm(hidden))
            layers.append(nn.ReLU())
            in_dim = hidden
        self.trunk = nn.Sequential(*layers)
        self.value_head = nn.Linear(hidden, 1)
        self.advantage_head = nn.Linear(hidden, n_actions)

    def forward(self, x):
        h = self.trunk(x)
        v = self.value_head(h)
        a = self.advantage_head(h)
        return v + a - a.mean(dim=-1, keepdim=True)

    @staticmethod
    def load_from_bin(path, obs_dim=108, hidden=512, n_layers=3):
        """Load Rust BidNet .bin weights (raw f32 LE)."""
        with open(path, "rb") as f:
            data = f.read()
        floats = np.frombuffer(data, dtype=np.float32)
        pos = 0

        model = DuelingBidNet(obs_dim, hidden, n_layers)

        # Load trunk layers
        in_dim = obs_dim
        for i in range(n_layers):
            layer_idx = i * 3  # Linear, LayerNorm, ReLU
            linear = model.trunk[layer_idx]
            ln = model.trunk[layer_idx + 1]

            # Weight [in_dim, hidden] stored row-major in Rust = [in_dim * hidden]
            w = floats[pos:pos + in_dim * hidden].reshape(hidden, in_dim)
            pos += in_dim * hidden
            b = floats[pos:pos + hidden]
            pos += hidden
            gamma = floats[pos:pos + hidden]
            pos += hidden
            beta = floats[pos:pos + hidden]
            pos += hidden

            linear.weight.data = torch.from_numpy(w.copy())
            linear.bias.data = torch.from_numpy(b.copy())
            ln.weight.data = torch.from_numpy(gamma.copy())
            ln.bias.data = torch.from_numpy(beta.copy())

            in_dim = hidden

        # Dueling heads
        w_val = floats[pos:pos + hidden]
        pos += hidden
        b_val = floats[pos:pos + 1]
        pos += 1
        w_adv = floats[pos:pos + hidden * 43].reshape(43, hidden)
        pos += hidden * 43
        b_adv = floats[pos:pos + 43]
        pos += 43

        model.value_head.weight.data = torch.from_numpy(w_val.copy()).unsqueeze(0)
        model.value_head.bias.data = torch.from_numpy(b_val.copy())
        model.advantage_head.weight.data = torch.from_numpy(w_adv.copy())
        model.advantage_head.bias.data = torch.from_numpy(b_adv.copy())

        assert pos == len(floats), f"Weight mismatch: read {pos}, total {len(floats)}"
        model.eval()
        return model


# ---------------------------------------------------------------------------
# Vectorized 108-dim bid observation encoder (numpy)
# ---------------------------------------------------------------------------
# Action → (type_flag, val_scaled, suit_idx) lookup tables
_BID_TYPE = np.zeros(43, dtype=np.float32)
_BID_VAL = np.zeros(43, dtype=np.float32)
_BID_SUIT = np.full(43, -1, dtype=np.int32)

_BID_TYPE[0] = 0.2  # PASS
_BID_TYPE[41] = 0.8  # COINCHE
_BID_TYPE[42] = 1.0  # SURCOINCHE
for _a in range(1, 41):
    if _a <= 36:
        _idx = _a - 1
        _val_enc = _idx // 4 + 8
        _suit = _idx % 4
        _BID_TYPE[_a] = 0.4
        _BID_VAL[_a] = (_val_enc * 10.0) / 250.0
        _BID_SUIT[_a] = _suit
    else:
        _suit = _a - 37
        _BID_TYPE[_a] = 0.6
        _BID_VAL[_a] = 1.0
        _BID_SUIT[_a] = _suit


def encode_bid_obs_batch(hands_u32, dealers, cur_players, bid_actions_buf,
                         bid_seats_buf, n_bids):
    """Encode batch of bid observations (108-dim).

    Args:
        hands_u32: [N, 4] uint32 — raw hand bitmasks per player
        dealers: [N] int — dealer seat
        cur_players: [N] int — current player seat
        bid_actions_buf: [N, max_bids] int — action history
        bid_seats_buf: [N, max_bids] int — seat history
        n_bids: [N] int — number of bids so far

    Returns: [N, 108] float32 observation
    """
    N = len(cur_players)
    obs = np.zeros((N, 108), dtype=np.float32)

    # Block 1: Hand [0:32]
    for i in range(N):
        hand = int(hands_u32[i, cur_players[i]])
        for bit in range(32):
            if hand & (1 << bit):
                obs[i, bit] = 1.0

    # Block 2: Bid history [32:104] — 12 slots × 6
    for i in range(N):
        me = cur_players[i]
        dealer = dealers[i]
        first_bidder = (dealer + 1) % 4
        rel_offset = (first_bidder + 4 - me) % 4
        nb = min(n_bids[i], 12)
        start = max(0, n_bids[i] - 12)
        for j in range(nb):
            slot = rel_offset + j
            if slot >= 12:
                break
            action = bid_actions_buf[i, start + j]
            base = 32 + slot * 6
            obs[i, base] = _BID_TYPE[action]
            obs[i, base + 1] = _BID_VAL[action]
            suit = _BID_SUIT[action]
            if suit >= 0:
                obs[i, base + 2 + suit] = 1.0

    # Block 3: Position [104:108]
    for i in range(N):
        rel_pos = (cur_players[i] + 4 - dealers[i]) % 4
        obs[i, 104 + rel_pos] = 1.0

    return obs


@torch.no_grad()
def select_actions_nn_v2(model, obs_np, masks_np, device):
    """Run nn_v2 inference and return greedy actions.

    Args:
        model: DuelingBidNet on device
        obs_np: [N, 108] float32
        masks_np: [N, 43] float32

    Returns: [N] int32 actions
    """
    obs_t = torch.from_numpy(obs_np).to(device)
    masks_t = torch.from_numpy(masks_np).to(device)
    q_values = model(obs_t)
    q_values = q_values.float().masked_fill(masks_t == 0, -1e9)
    return q_values.argmax(dim=-1).cpu().numpy().astype(np.int32)
