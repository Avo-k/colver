#!/usr/bin/env python3
"""Check if belote affects bid LEVELS (not just bid/pass)."""

import struct
from pathlib import Path
import numpy as np

R7, R8, R9, RJ, RQ, RK, R10, RA = 0, 1, 2, 3, 4, 5, 6, 7

def load_bid_net(path, hidden=512):
    data = Path(path).read_bytes()
    n = len(data) // 4
    floats = np.array(struct.unpack(f"<{n}f", data), dtype=np.float32)
    for layers in [3, 4, 2]:
        for dueling in [True, False]:
            trunk = (layers - 1) * (hidden * hidden + 3 * hidden) + 3 * hidden
            tail = (hidden + 1 + hidden * 43 + 43) if dueling else (hidden * 43 + 43)
            fixed = trunk + tail
            if n > fixed and (n - fixed) % hidden == 0:
                obs_dim = (n - fixed) // hidden
                if 0 < obs_dim <= 500:
                    return _parse(floats, obs_dim, hidden, layers, dueling)
    raise ValueError("Cannot detect architecture")

def _parse(floats, obs_dim, hidden, layers, dueling):
    off = 0
    net = {"obs_dim": obs_dim, "hidden": hidden, "layers": layers, "dueling": dueling,
           "w": [], "b": [], "gamma": [], "beta": []}
    in_dims = [obs_dim] + [hidden] * (layers - 1)
    for layer in range(layers):
        d = in_dims[layer]
        net["w"].append(floats[off:off+d*hidden].reshape(hidden, d)); off += d*hidden
        net["b"].append(floats[off:off+hidden].copy()); off += hidden
        net["gamma"].append(floats[off:off+hidden].copy()); off += hidden
        net["beta"].append(floats[off:off+hidden].copy()); off += hidden
    if dueling:
        net["w_value"] = floats[off:off+hidden].copy(); off += hidden
        net["b_value"] = floats[off]; off += 1
        net["w_adv"] = floats[off:off+hidden*43].reshape(43, hidden); off += hidden*43
        net["b_adv"] = floats[off:off+43].copy(); off += 43
    return net

def forward(net, obs):
    x = obs
    for layer in range(net["layers"]):
        x = x @ net["w"][layer].T + net["b"][layer]
        m = x.mean(axis=-1, keepdims=True)
        v = x.var(axis=-1, keepdims=True)
        x = net["gamma"][layer] * (x - m) / np.sqrt(v + 1e-5) + net["beta"][layer]
        x = np.maximum(x, 0)
    if net["dueling"]:
        val = x @ net["w_value"] + net["b_value"]
        adv = x @ net["w_adv"].T + net["b_adv"]
        return val[:, None] + adv - adv.mean(axis=1, keepdims=True)
    return x @ net["w_out"].T + net["b_out"]

def get_q(net, hands, trump=0):
    """Return full Q-values (N, 43) for opening pos."""
    N = hands.shape[0]
    obs = np.zeros((N, 108), dtype=np.float32)
    obs[:, :32] = hands
    obs[:, 105] = 1.0
    return forward(net, obs)

def make_hands(trump_must, trump_no, n, trump=0, seed=42):
    rng = np.random.RandomState(seed)
    hands = np.zeros((n, 32), dtype=np.float32)
    gen = 0
    for _ in range(n * 100):
        h = np.zeros(32, dtype=np.float32)
        for r in trump_must:
            h[trump * 8 + r] = 1.0
        placed = int(h.sum())
        avail = [c for c in range(32) if h[c] == 0 and not (c // 8 == trump and c % 8 in trump_no)]
        if len(avail) < 8 - placed:
            continue
        chosen = rng.choice(avail, size=8 - placed, replace=False)
        for c in chosen:
            h[c] = 1.0
        hands[gen] = h
        gen += 1
        if gen >= n:
            break
    return hands[:gen]

def decode_action(a):
    if a == 0: return "PASS"
    if a == 41: return "COINCHE"
    if a == 42: return "SURCOINCHE"
    if 37 <= a <= 40: return f"Capot"
    val_idx = (a - 1) // 4
    return f"{(val_idx + 8) * 10}"

net = load_bid_net("models/bid_v2/bid_nn_final.bin", 512)
N = 10000
TRUMP = 0  # spades

print("=" * 80)
print("  BELOTE EFFECT ON BID LEVELS")
print("=" * 80)

# Compare Q-values at each level for belote vs no belote
configs = [
    ("J9 + KQ (belote)", [RJ, R9, RK, RQ], [RA, R10]),
    ("J9 + K8",          [RJ, R9, RK, R8], [RA, R10, RQ]),
    ("J9 + Q8",          [RJ, R9, RQ, R8], [RA, R10, RK]),
    ("J9 + 87",          [RJ, R9, R8, R7], [RA, R10, RK, RQ]),
]

# action for 80♠=1, 90♠=5, 100♠=9, 110♠=13, 120♠=17, capot♠=37
levels = [
    ("80",  1),
    ("90",  5),
    ("100", 9),
    ("110", 13),
    ("120", 17),
    ("130", 21),
    ("140", 25),
    ("150", 29),
    ("160", 33),
    ("Capot", 37),
]

print("\n--- Q-values by bid level for spades ---")
header = f"{'Config':<20} {'PASS':>8}"
for lbl, _ in levels[:7]:
    header += f" {lbl:>8}"
print(header)
print("-" * len(header))

for label, must, no in configs:
    hands = make_hands(must, no, N, trump=TRUMP, seed=42)
    q = get_q(net, hands, trump=TRUMP)
    row = f"  {label:<18} {q[:, 0].mean():>+8.4f}"
    for lbl, act in levels[:7]:
        row += f" {q[:, act].mean():>+8.4f}"
    print(row)

# Now compare the NN's actual chosen action
print("\n--- Chosen bid level distribution ---")
for label, must, no in configs:
    hands = make_hands(must, no, N, trump=TRUMP, seed=42)
    q = get_q(net, hands, trump=TRUMP)
    # Best legal bid action (0-40, exclude coinche/surcoinche)
    q_legal = q[:, :41].copy()
    best = q_legal.argmax(axis=1)

    from collections import Counter
    dist = Counter()
    for a in best:
        dist[decode_action(a)] += 1

    total = len(best)
    print(f"\n  {label}:")
    for lbl in ["PASS", "80", "90", "100", "110", "120", "130", "140", "Capot"]:
        cnt = dist.get(lbl, 0)
        if cnt > 0:
            pct = cnt / total * 100
            bar = "█" * int(pct / 2)
            print(f"    {lbl:<8} {cnt:>6} ({pct:>5.1f}%) {bar}")

# -------------------------------------------------------
print("\n\n--- Belote effect: Q(level) difference ---")
print("  (belote Q - non-belote Q at each level)")

h_kq = make_hands([RJ, R9, RK, RQ], [RA, R10], N, trump=TRUMP, seed=42)
h_87 = make_hands([RJ, R9, R8, R7], [RA, R10, RK, RQ], N, trump=TRUMP, seed=42)

q_kq = get_q(net, h_kq, trump=TRUMP)
q_87 = get_q(net, h_87, trump=TRUMP)

print(f"\n  {'Level':<8} {'Q(KQ)':>8} {'Q(87)':>8} {'Delta':>8}")
print("  " + "-" * 36)
print(f"  {'PASS':<8} {q_kq[:, 0].mean():>+8.4f} {q_87[:, 0].mean():>+8.4f} {q_kq[:, 0].mean() - q_87[:, 0].mean():>+8.4f}")
for lbl, act in levels[:8]:
    d = q_kq[:, act].mean() - q_87[:, act].mean()
    marker = " ←" if abs(d) > 0.01 else ""
    print(f"  {lbl:<8} {q_kq[:, act].mean():>+8.4f} {q_87[:, act].mean():>+8.4f} {d:>+8.4f}{marker}")

# -------------------------------------------------------
print("\n\n--- Same for J alone (no 9): belote effect on levels ---")

h_jkq = make_hands([RJ, RK, RQ], [R9, RA, R10], N, trump=TRUMP, seed=42)
h_j87 = make_hands([RJ, R8, R7], [R9, RA, R10, RK, RQ], N, trump=TRUMP, seed=42)

q_jkq = get_q(net, h_jkq, trump=TRUMP)
q_j87 = get_q(net, h_j87, trump=TRUMP)

print(f"\n  {'Level':<8} {'Q(JKQ)':>8} {'Q(J87)':>8} {'Delta':>8}")
print("  " + "-" * 36)
print(f"  {'PASS':<8} {q_jkq[:, 0].mean():>+8.4f} {q_j87[:, 0].mean():>+8.4f} {q_jkq[:, 0].mean() - q_j87[:, 0].mean():>+8.4f}")
for lbl, act in levels[:8]:
    d = q_jkq[:, act].mean() - q_j87[:, act].mean()
    marker = " ←" if abs(d) > 0.01 else ""
    print(f"  {lbl:<8} {q_jkq[:, act].mean():>+8.4f} {q_j87[:, act].mean():>+8.4f} {d:>+8.4f}{marker}")

# -------------------------------------------------------
print("\n\n--- Bid level distribution comparison ---")
for label, h_set in [("J9+KQ", h_kq), ("J9+87", h_87), ("J+KQ", h_jkq), ("J+87", h_j87)]:
    q = get_q(net, h_set, trump=TRUMP)
    best = q[:, :41].argmax(axis=1)
    dist = Counter()
    for a in best:
        dist[decode_action(a)] += 1
    total = len(best)
    avg_level = 0
    n_bids = 0
    for a in best:
        if 1 <= a <= 36:
            val_idx = (a - 1) // 4
            avg_level += (val_idx + 8) * 10
            n_bids += 1
        elif 37 <= a <= 40:
            avg_level += 250
            n_bids += 1
    avg_level = avg_level / max(n_bids, 1)
    bid_pct = sum(1 for a in best if a >= 1) / total * 100
    print(f"  {label:<10} bid={bid_pct:.1f}%  avg_level={avg_level:.0f}  distribution: ", end="")
    for lbl in ["80", "90", "100", "110", "120"]:
        cnt = dist.get(lbl, 0)
        if cnt > 0:
            print(f"{lbl}={cnt/total*100:.1f}% ", end="")
    print()

print("\n" + "=" * 80)
