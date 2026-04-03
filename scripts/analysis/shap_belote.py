#!/usr/bin/env python3
"""Quick focused experiments on K, Q, and belote (K+Q) interactions."""

import struct
from pathlib import Path
import numpy as np

R7, R8, R9, RJ, RQ, RK, R10, RA = 0, 1, 2, 3, 4, 5, 6, 7
RANKS = ["7", "8", "9", "J", "Q", "K", "10", "A"]

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

def bid_adv(net, hands, trump=0):
    N = hands.shape[0]
    obs = np.zeros((N, 108), dtype=np.float32)
    obs[:, :32] = hands
    obs[:, 105] = 1.0
    q = forward(net, obs)
    return q[:, trump + 1] - q[:, 0]

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

def make_hands_constrained(trump_must, trump_no, side_aces=None, side_voids=None, n=3000, trump=0, seed=42):
    """Like make_hands but with side constraints (rejection sampling)."""
    rng = np.random.RandomState(seed)
    hands = np.zeros((n, 32), dtype=np.float32)
    gen = 0
    for _ in range(n * 200):
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
        # Check constraints
        if side_aces is not None:
            actual = sum(1 for s in range(4) if s != trump and h[s*8 + RA] > 0)
            if actual != side_aces:
                continue
        if side_voids is not None:
            actual = sum(1 for s in range(4) if s != trump and sum(h[s*8:s*8+8]) == 0)
            if actual != side_voids:
                continue
        hands[gen] = h
        gen += 1
        if gen >= n:
            break
    if gen < n:
        print(f"  (generated {gen}/{n})")
    return hands[:gen]


net = load_bid_net("models/bid_v2/bid_nn_final.bin", 512)
N = 5000

print("=" * 80)
print("  BELOTE & K/Q INTERACTIONS")
print("=" * 80)

# -------------------------------------------------------
print("\n--- Exp A: K+Q (belote) vs K seul vs Q seul vs neither ---")
print("  Base: J+9 + 1 other trump card (3 total)")
print()

combos = [
    ("J+9+K+Q (belote)", [RJ, R9, RK, RQ], [RA, R10]),
    ("J+9+K+8",          [RJ, R9, RK, R8], [RA, R10, RQ]),
    ("J+9+Q+8",          [RJ, R9, RQ, R8], [RA, R10, RK]),
    ("J+9+8+7",          [RJ, R9, R8, R7], [RA, R10, RK, RQ]),
]

for label, must, no in combos:
    h = make_hands(must, no, N, seed=42)
    a = bid_adv(net, h)
    print(f"  {label:<22} advantage={a.mean():>+.4f} (std={a.std():.4f})")

# -------------------------------------------------------
print("\n--- Exp B: With J only (no 9) — belote impact ---")
print("  3 trump: J + 2 others")
print()

combos = [
    ("J+K+Q (belote)", [RJ, RK, RQ], [R9, RA, R10]),
    ("J+K+8",          [RJ, RK, R8], [R9, RA, R10, RQ]),
    ("J+Q+8",          [RJ, RQ, R8], [R9, RA, R10, RK]),
    ("J+8+7",          [RJ, R8, R7], [R9, RA, R10, RK, RQ]),
]

for label, must, no in combos:
    h = make_hands(must, no, N, seed=42)
    a = bid_adv(net, h)
    print(f"  {label:<22} advantage={a.mean():>+.4f} (std={a.std():.4f})")

# -------------------------------------------------------
print("\n--- Exp C: Without J — belote impact ---")
print("  3 trump: no J, varied honors")
print()

combos = [
    ("9+K+Q (belote)",  [R9, RK, RQ], [RJ, RA, R10]),
    ("9+K+8",           [R9, RK, R8], [RJ, RA, R10, RQ]),
    ("9+Q+8",           [R9, RQ, R8], [RJ, RA, R10, RK]),
    ("K+Q+8 (no J/9)",  [RK, RQ, R8], [RJ, R9, RA, R10]),
    ("K+8+7 (no J/9)",  [RK, R8, R7], [RJ, R9, RA, R10, RQ]),
]

for label, must, no in combos:
    h = make_hands(must, no, N, seed=42)
    a = bid_adv(net, h)
    print(f"  {label:<22} advantage={a.mean():>+.4f} (std={a.std():.4f})")

# -------------------------------------------------------
print("\n--- Exp D: 4 trump with/without belote ---")
print()

combos = [
    ("J+9+K+Q",   [RJ, R9, RK, RQ], [RA, R10]),
    ("J+9+K+7",   [RJ, R9, RK, R7], [RA, R10, RQ]),
    ("J+9+Q+7",   [RJ, R9, RQ, R7], [RA, R10, RK]),
    ("J+9+8+7",   [RJ, R9, R8, R7], [RA, R10, RK, RQ]),
    ("J+K+Q+7",   [RJ, RK, RQ, R7], [R9, RA, R10]),
    ("J+K+8+7",   [RJ, RK, R8, R7], [R9, RA, R10, RQ]),
    ("9+K+Q+7",   [R9, RK, RQ, R7], [RJ, RA, R10]),
    ("9+K+8+7",   [R9, RK, R8, R7], [RJ, RA, R10, RQ]),
    ("K+Q+8+7",   [RK, RQ, R8, R7], [RJ, R9, RA, R10]),
]

results = []
for label, must, no in combos:
    h = make_hands(must, no, N, seed=42)
    a = bid_adv(net, h)
    results.append((label, a.mean(), a.std()))

results.sort(key=lambda x: -x[1])
for label, mean, std in results:
    print(f"  {label:<16} advantage={mean:>+.4f} (std={std:.4f})")

# -------------------------------------------------------
print("\n--- Exp E: Belote bonus isolé ---")
print("  Compare K+Q vs K+X vs Q+X (X=same rank) as 3rd+4th trump")
print("  Base: J+9 always present")
print()

# We want to measure: (J9KQ - J9K7) vs (J9K7 - J9_87)
# i.e. marginal value of Q when K is present, vs when K is absent

h_j9kq = make_hands([RJ, R9, RK, RQ], [RA, R10], N, seed=42)
h_j9k7 = make_hands([RJ, R9, RK, R7], [RA, R10, RQ], N, seed=42)
h_j9q7 = make_hands([RJ, R9, RQ, R7], [RA, R10, RK], N, seed=42)
h_j987 = make_hands([RJ, R9, R8, R7], [RA, R10, RK, RQ], N, seed=42)

a_kq = bid_adv(net, h_j9kq).mean()
a_k7 = bid_adv(net, h_j9k7).mean()
a_q7 = bid_adv(net, h_j9q7).mean()
a_87 = bid_adv(net, h_j987).mean()

print(f"  J9+KQ:  {a_kq:+.4f}")
print(f"  J9+K7:  {a_k7:+.4f}")
print(f"  J9+Q7:  {a_q7:+.4f}")
print(f"  J9+87:  {a_87:+.4f}")
print()
print(f"  Marginal Q when K present: {a_kq - a_k7:+.4f} (= belote synergy)")
print(f"  Marginal Q when K absent:  {a_q7 - a_87:+.4f} (= Q alone)")
print(f"  Marginal K when Q present: {a_kq - a_q7:+.4f}")
print(f"  Marginal K when Q absent:  {a_k7 - a_87:+.4f} (= K alone)")
print(f"  Belote synergy = (KQ - K - Q + base) = {a_kq - a_k7 - a_q7 + a_87:+.4f}")

# -------------------------------------------------------
print("\n--- Exp F: Same analysis for J alone (no 9) ---")
print()

h_jkq = make_hands([RJ, RK, RQ], [R9, RA, R10], N, seed=42)
h_jk8 = make_hands([RJ, RK, R8], [R9, RA, R10, RQ], N, seed=42)
h_jq8 = make_hands([RJ, RQ, R8], [R9, RA, R10, RK], N, seed=42)
h_j87 = make_hands([RJ, R8, R7], [R9, RA, R10, RK, RQ], N, seed=42)

a_kq = bid_adv(net, h_jkq).mean()
a_k8 = bid_adv(net, h_jk8).mean()
a_q8 = bid_adv(net, h_jq8).mean()
a_87 = bid_adv(net, h_j87).mean()

print(f"  J+KQ:  {a_kq:+.4f}")
print(f"  J+K8:  {a_k8:+.4f}")
print(f"  J+Q8:  {a_q8:+.4f}")
print(f"  J+87:  {a_87:+.4f}")
print()
print(f"  Marginal Q when K present: {a_kq - a_k8:+.4f} (belote synergy)")
print(f"  Marginal Q when K absent:  {a_q8 - a_87:+.4f} (Q alone)")
print(f"  Marginal K when Q present: {a_kq - a_q8:+.4f}")
print(f"  Marginal K when Q absent:  {a_k8 - a_87:+.4f} (K alone)")
print(f"  Belote synergy = {a_kq - a_k8 - a_q8 + a_87:+.4f}")

# -------------------------------------------------------
print("\n--- Exp G: Side suit K+Q combos ---")
print("  Trump: J+9+7 fixed. Side: does K+Q in same side suit help?")
print()

# Generate hands with J97 trump, and specific side cards
def count_side_pair(hands, trump, rank1, rank2):
    """Check how many hands have rank1+rank2 in the same non-trump suit."""
    cnt = 0
    for i in range(len(hands)):
        for s in range(4):
            if s == trump:
                continue
            if hands[i, s*8+rank1] > 0 and hands[i, s*8+rank2] > 0:
                cnt += 1
                break
    return cnt

# Just measure average advantage by whether hand has a side K+Q pair
h = make_hands([RJ, R9, R7], [RA, R10, RK, RQ], 20000, seed=42)
adv = bid_adv(net, h)

has_side_kq = []
no_side_kq = []
has_side_a = []
no_side_a = []
for i in range(len(h)):
    found_kq = False
    found_a = False
    for s in range(1, 4):  # skip trump suit 0
        if h[i, s*8+RK] > 0 and h[i, s*8+RQ] > 0:
            found_kq = True
        if h[i, s*8+RA] > 0:
            found_a = True
    if found_kq:
        has_side_kq.append(adv[i])
    else:
        no_side_kq.append(adv[i])
    if found_a:
        has_side_a.append(adv[i])
    else:
        no_side_a.append(adv[i])

print(f"  With side K+Q pair:    advantage={np.mean(has_side_kq):>+.4f} (N={len(has_side_kq)})")
print(f"  Without side K+Q pair: advantage={np.mean(no_side_kq):>+.4f} (N={len(no_side_kq)})")
print(f"  Delta (K+Q pair):      {np.mean(has_side_kq) - np.mean(no_side_kq):>+.4f}")
print()
print(f"  With side Ace:         advantage={np.mean(has_side_a):>+.4f} (N={len(has_side_a)})")
print(f"  Without side Ace:      advantage={np.mean(no_side_a):>+.4f} (N={len(no_side_a)})")
print(f"  Delta (side Ace):      {np.mean(has_side_a) - np.mean(no_side_a):>+.4f}")

# -------------------------------------------------------
print("\n--- Exp H: A+10 in same side suit (mariage d'As) ---")
print("  Trump: J+9+7 fixed.")
print()

h = make_hands([RJ, R9, R7], [RA, R10, RK, RQ], 20000, seed=142)
adv = bid_adv(net, h)

has_a10 = []
no_a10 = []
for i in range(len(h)):
    found = False
    for s in range(1, 4):
        if h[i, s*8+RA] > 0 and h[i, s*8+R10] > 0:
            found = True
    if found:
        has_a10.append(adv[i])
    else:
        no_a10.append(adv[i])

print(f"  With side A+10 pair:   advantage={np.mean(has_a10):>+.4f} (N={len(has_a10)})")
print(f"  Without side A+10:     advantage={np.mean(no_a10):>+.4f} (N={len(no_a10)})")
print(f"  Delta (A+10 pair):     {np.mean(has_a10) - np.mean(no_a10):>+.4f}")

print("\n" + "=" * 80)
print("Done!")
