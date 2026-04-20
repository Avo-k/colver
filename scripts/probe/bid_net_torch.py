"""Re-implement the v5 BidNet in PyTorch to extract hidden activations.

Weight file layout (see colver-core/src/bid/bid_net.rs):
  For each of `layers` layers:
    W[l]    : shape (hidden, in_dim_l), row-major f32       (in_dim_l * hidden floats)
    b[l]    : shape (hidden,)                                (hidden)
    gamma[l]: shape (hidden,)                                (hidden)
    beta[l] : shape (hidden,)                                (hidden)
  If dueling:
    w_value : shape (hidden,)                                (hidden)
    b_value : scalar                                         (1)
    w_adv   : shape (43, hidden), row-major                  (43 * hidden)
    b_adv   : shape (43,)                                    (43)

Forward pass:
  h0 = ReLU(LN(W0 @ obs + b0, gamma0, beta0))
  for l in 1..layers: h[l] = ReLU(LN(Wl @ h[l-1] + bl, gamma_l, beta_l))
  V = w_value · h[-1] + b_value         (scalar)
  A = w_adv @ h[-1] + b_adv             (43)
  Q = V + A - mean(A)                   (43)

LN: x - mean; var = mean((x-mean)^2); (x-mean) / sqrt(var + 1e-5) * gamma + beta
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

LN_EPS = 1e-5
NUM_ACTIONS = 43


class BidNetTorch(nn.Module):
    def __init__(self, obs_dim: int, hidden: int, layers: int, dueling: bool):
        super().__init__()
        self.obs_dim = obs_dim
        self.hidden = hidden
        self.layers = layers
        self.dueling = dueling
        # Linear layers (standard torch: weight shape (out, in))
        self.linears = nn.ModuleList()
        self.lns = nn.ModuleList()
        in_d = self.obs_dim
        for _ in range(self.layers):
            self.linears.append(nn.Linear(in_d, self.hidden, bias=True))
            # LayerNorm with elementwise affine (gamma, beta)
            self.lns.append(nn.LayerNorm(self.hidden, eps=LN_EPS, elementwise_affine=True))
            in_d = self.hidden
        if self.dueling:
            self.value_head = nn.Linear(self.hidden, 1, bias=True)
            self.adv_head = nn.Linear(self.hidden, NUM_ACTIONS, bias=True)
        else:
            self.out_head = nn.Linear(self.hidden, NUM_ACTIONS, bias=True)

    def forward(self, x: torch.Tensor, return_hidden: bool = False):
        """x: (B, obs_dim) → (Q: (B, 43), hidden: list of (B, hidden) per layer) if return_hidden else Q only."""
        acts = []
        for lin, ln in zip(self.linears, self.lns):
            x = lin(x)
            x = ln(x)
            x = torch.relu(x)
            if return_hidden:
                acts.append(x)
        if self.dueling:
            v = self.value_head(x).squeeze(-1)            # (B,)
            a = self.adv_head(x)                          # (B, 43)
            q = v.unsqueeze(-1) + a - a.mean(dim=-1, keepdim=True)
        else:
            q = self.out_head(x)
        return (q, acts) if return_hidden else q


def load_bid_net(path: str, hidden: int = 512) -> BidNetTorch:
    """Parse the raw f32 weight file and build a BidNetTorch."""
    data = np.fromfile(path, dtype="<f4")  # little-endian float32
    total = data.size
    dueling_tail = hidden + 1 + hidden * NUM_ACTIONS + NUM_ACTIONS
    standard_tail = hidden * NUM_ACTIONS + NUM_ACTIONS
    known = [108, 110, 113, 114]
    picked = None
    for layers in range(2, 5):
        trunk_fixed = (layers - 1) * (hidden * hidden + 3 * hidden) + 3 * hidden
        for tail, duel in [(dueling_tail, True), (standard_tail, False)]:
            fixed = trunk_fixed + tail
            if total > fixed and (total - fixed) % hidden == 0:
                obs_dim = (total - fixed) // hidden
                if obs_dim in known:
                    picked = (layers, duel, obs_dim)
                    break
        if picked:
            break
    if picked is None:
        raise RuntimeError(f"Cannot infer architecture from {total} floats, hidden={hidden}")
    layers, dueling, obs_dim = picked
    print(f"[load] arch: obs_dim={obs_dim}, hidden={hidden}, layers={layers}, dueling={dueling}")

    net = BidNetTorch(obs_dim=obs_dim, hidden=hidden, layers=layers, dueling=dueling)

    offset = 0
    in_dims = [obs_dim] + [hidden] * (layers - 1)
    with torch.no_grad():
        for l in range(layers):
            in_d = in_dims[l]
            w = data[offset:offset + in_d * hidden].reshape(hidden, in_d)
            offset += in_d * hidden
            b = data[offset:offset + hidden]
            offset += hidden
            gamma = data[offset:offset + hidden]
            offset += hidden
            beta = data[offset:offset + hidden]
            offset += hidden
            net.linears[l].weight.copy_(torch.from_numpy(w.copy()))
            net.linears[l].bias.copy_(torch.from_numpy(b.copy()))
            net.lns[l].weight.copy_(torch.from_numpy(gamma.copy()))
            net.lns[l].bias.copy_(torch.from_numpy(beta.copy()))

        if dueling:
            w_value = data[offset:offset + hidden]
            offset += hidden
            b_value = data[offset]
            offset += 1
            w_adv = data[offset:offset + NUM_ACTIONS * hidden].reshape(NUM_ACTIONS, hidden)
            offset += NUM_ACTIONS * hidden
            b_adv = data[offset:offset + NUM_ACTIONS]
            offset += NUM_ACTIONS
            net.value_head.weight.copy_(torch.from_numpy(w_value.reshape(1, hidden).copy()))
            net.value_head.bias.copy_(torch.tensor([b_value], dtype=torch.float32))
            net.adv_head.weight.copy_(torch.from_numpy(w_adv.copy()))
            net.adv_head.bias.copy_(torch.from_numpy(b_adv.copy()))
        else:
            w_out = data[offset:offset + NUM_ACTIONS * hidden].reshape(NUM_ACTIONS, hidden)
            offset += NUM_ACTIONS * hidden
            b_out = data[offset:offset + NUM_ACTIONS]
            offset += NUM_ACTIONS
            net.out_head.weight.copy_(torch.from_numpy(w_out.copy()))
            net.out_head.bias.copy_(torch.from_numpy(b_out.copy()))

    assert offset == total, f"used {offset}/{total} floats"
    net.eval()
    return net


if __name__ == "__main__":
    import sys
    path = sys.argv[1] if len(sys.argv) > 1 else "models/bid_v5_isdd/bid_nn_final.bin"
    net = load_bid_net(path)
    print(f"[ok] loaded {sum(p.numel() for p in net.parameters()):,} params")
    # Smoke test: evaluate a zero obs
    x = torch.zeros(1, net.obs_dim)
    with torch.no_grad():
        q, acts = net(x, return_hidden=True)
    print(f"Q shape: {q.shape}, hidden layers: {[a.shape for a in acts]}")
    print(f"Q-values on zero obs: {q.squeeze().tolist()[:10]}...")
