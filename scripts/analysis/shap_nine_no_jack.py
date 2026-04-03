#!/usr/bin/env python3
"""
Deep dive: bidding with 9 but no Jack, across all positions.

Key questions:
- When does the NN bid with 9 alone? How many trump needed?
- How does position change the threshold?
- What side features tip the balance?
- Does the NN bid differently at pos3 (protective) vs pos1 (opening)?
- What levels does it choose?
- What's the role of the 9's companion cards?
"""

import struct
from pathlib import Path
from collections import Counter
import numpy as np

R7, R8, R9, RJ, RQ, RK, R10, RA = 0, 1, 2, 3, 4, 5, 6, 7
RANKS = ["7", "8", "9", "J", "Q", "K", "10", "A"]
SUITS = ["♠", "♥", "♦", "♣"]


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


def make_obs(hands, position, bid_history=None):
    """Build 108-dim obs for N hands.

    position: 1-4 (dealer-relative)
    bid_history: list of (relative_slot, action) for encoding
    """
    N = hands.shape[0]
    obs = np.zeros((N, 108), dtype=np.float32)
    obs[:, :32] = hands
    # Position one-hot at offset 104
    obs[:, 104 + (position - 1)] = 1.0

    # Bid history at offset 32, 12 slots × 6 floats
    if bid_history:
        for slot, action in bid_history:
            if slot >= 12:
                break
            base = 32 + slot * 6
            if action == 0:  # PASS
                obs[:, base] = 0.2
            elif action == 41:  # COINCHE
                obs[:, base] = 0.8
            elif 1 <= action <= 40:
                val_idx = (action - 1) // 4
                suit_idx = (action - 1) % 4
                val_enc = val_idx + 8
                if val_enc == 25:
                    obs[:, base] = 0.6
                    obs[:, base + 1] = 1.0
                else:
                    obs[:, base] = 0.4
                    obs[:, base + 1] = (val_enc * 10.0) / 250.0
                obs[:, base + 2 + suit_idx] = 1.0
    return obs


def get_q(net, hands, position, bid_history=None, trump=0):
    obs = make_obs(hands, position, bid_history)
    q = forward(net, obs)
    return q


def make_hands(trump_must, trump_no, n, trump=0, seed=42, side_aces=None, side_voids=None):
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
        ok = True
        if side_aces is not None:
            actual = sum(1 for s in range(4) if s != trump and h[s * 8 + RA] > 0)
            if actual != side_aces:
                ok = False
        if side_voids is not None:
            actual = sum(1 for s in range(4) if s != trump and sum(h[s * 8:s * 8 + 8]) == 0)
            if actual != side_voids:
                ok = False
        if ok:
            hands[gen] = h
            gen += 1
            if gen >= n:
                break
    if gen < n:
        print(f"  (only {gen}/{n} hands)")
    return hands[:gen]


def analyze_decisions(q, label=""):
    """Analyze bid decisions from Q-values."""
    best = q[:, :41].argmax(axis=1)
    n = len(best)
    n_pass = (best == 0).sum()
    n_bid = (best >= 1).sum()

    levels = Counter()
    total_level = 0
    for a in best:
        if a == 0:
            levels["PASS"] += 1
        elif 1 <= a <= 36:
            val = ((a - 1) // 4 + 8) * 10
            levels[str(val)] += 1
            total_level += val
        elif 37 <= a <= 40:
            levels["Capot"] += 1
            total_level += 250

    avg_level = total_level / max(n_bid, 1)
    bid_pct = n_bid / n * 100

    return bid_pct, avg_level, levels, n


def print_analysis(q, label, n):
    bid_pct, avg_level, levels, _ = analyze_decisions(q)
    parts = []
    for lbl in ["PASS", "80", "90", "100", "110", "120", "130", "Capot"]:
        cnt = levels.get(lbl, 0)
        if cnt > 0:
            parts.append(f"{lbl}={cnt/n*100:.1f}%")
    dist_str = " ".join(parts)
    print(f"  {label:<35} bid={bid_pct:>5.1f}%  avg={avg_level:>5.0f}  {dist_str}")


net = load_bid_net("models/bid_v2/bid_nn_final.bin", 512)
N = 8000
TRUMP = 0

# Position configs: (position, bid_history, label)
# Seat 0 always. Dealer chosen to get the right position.
# Position 1: dealer=3, first_bidder=0, no history
# Position 2: dealer=2, first_bidder=3, history: seat3 passes (slot 0)
# Position 3: dealer=1, first_bidder=2, history: seat2 passes (slot 0), seat3 passes (slot 1)
# Position 4: dealer=0, first_bidder=1, history: seat1,2,3 all pass (slots 0,1,2)

positions = [
    (1, None, "Pos1 (opening)"),
    (2, [(0, 0)], "Pos2 (1 pass)"),
    (3, [(0, 0), (1, 0)], "Pos3 (2 passes)"),
    (4, [(0, 0), (1, 0), (2, 0)], "Pos4 (3 passes)"),
]

print("=" * 90)
print("  9 WITHOUT JACK: DEEP DIVE BY POSITION")
print("=" * 90)

# ============================================================
print("\n" + "=" * 90)
print("  EXP 1: 9 + N small cards, by position")
print("=" * 90)

for n_extra in range(5):  # 0..4 extra = 1..5 total trump
    extra = [R7, R8, RK, RQ, R10][:n_extra]
    total = 1 + n_extra
    hands = make_hands([R9] + extra, [RJ, RA], N, seed=42 + n_extra)
    if len(hands) < 100:
        continue
    print(f"\n  --- 9 + {n_extra} small = {total} trump (no J, no A) ---")
    for pos, hist, plabel in positions:
        q = get_q(net, hands, pos, hist, trump=TRUMP)
        print_analysis(q, plabel, len(hands))

# ============================================================
print("\n\n" + "=" * 90)
print("  EXP 2: 9+A vs 9+7 vs 9+K by position (2 trump)")
print("=" * 90)

combos_2 = [
    ("9+A", [R9, RA], [RJ]),
    ("9+K", [R9, RK], [RJ, RA]),
    ("9+7", [R9, R7], [RJ, RA]),
    ("9+10", [R9, R10], [RJ, RA]),
]

for label, must, no in combos_2:
    hands = make_hands(must, no, N, seed=42)
    print(f"\n  --- {label} (2 trump) ---")
    for pos, hist, plabel in positions:
        q = get_q(net, hands, pos, hist, trump=TRUMP)
        print_analysis(q, plabel, len(hands))

# ============================================================
print("\n\n" + "=" * 90)
print("  EXP 3: 9 + 2 small, with side variations, by position")
print("  (3 trump: 9+7+8, varying sides)")
print("=" * 90)

side_configs = [
    ("0 aces, 0 voids", {"side_aces": 0, "side_voids": 0}),
    ("1 ace, 0 voids", {"side_aces": 1, "side_voids": 0}),
    ("0 aces, 1 void", {"side_aces": 0, "side_voids": 1}),
    ("1 ace, 1 void", {"side_aces": 1, "side_voids": 1}),
    ("2 aces, 0 voids", {"side_aces": 2, "side_voids": 0}),
    ("0 aces, 2 voids", {"side_aces": 0, "side_voids": 2}),
]

for slabel, skw in side_configs:
    hands = make_hands([R9, R7, R8], [RJ, RA], N, seed=42, **skw)
    if len(hands) < 50:
        print(f"\n  --- 9+7+8, {slabel} --- (too few hands)")
        continue
    print(f"\n  --- 9+7+8, {slabel} (N={len(hands)}) ---")
    for pos, hist, plabel in positions:
        q = get_q(net, hands, pos, hist, trump=TRUMP)
        print_analysis(q, plabel, len(hands))

# ============================================================
print("\n\n" + "=" * 90)
print("  EXP 4: Comparing 9-based vs J-based at same trump count")
print("  (How much worse is 9 vs J at each position?)")
print("=" * 90)

for total_trump in [2, 3, 4]:
    extra = [R7, R8, RK, RQ][:total_trump - 1]
    h_j = make_hands([RJ] + extra, [R9, RA], N, seed=42)
    h_9 = make_hands([R9] + extra, [RJ, RA], N, seed=42)
    h_none = make_hands(extra + ([R10] if total_trump == 1 else []), [RJ, R9, RA], N, seed=42)

    print(f"\n  --- {total_trump} trump ---")
    for pos, hist, plabel in positions:
        q_j = get_q(net, h_j, pos, hist, trump=TRUMP)
        q_9 = get_q(net, h_9, pos, hist, trump=TRUMP)
        q_none = get_q(net, h_none, pos, hist, trump=TRUMP) if len(h_none) > 0 else None

        bj, aj, _, _ = analyze_decisions(q_j)
        b9, a9, _, _ = analyze_decisions(q_9)
        line = f"  {plabel:<16} J: bid={bj:>5.1f}% avg={aj:>4.0f}  |  9: bid={b9:>5.1f}% avg={a9:>4.0f}  |  gap: {bj-b9:>+5.1f}pp"
        if q_none is not None:
            bn, an, _, _ = analyze_decisions(q_none)
            line += f"  |  no J/9: bid={bn:>5.1f}%"
        print(line)

# ============================================================
print("\n\n" + "=" * 90)
print("  EXP 5: Q-value advantage curves (9 without J)")
print("  Q(80♠) - Q(PASS) for 9 + N small, by position")
print("=" * 90)

print(f"\n  {'Config':<16}", end="")
for _, _, plabel in positions:
    print(f" {plabel:>16}", end="")
print()
print("  " + "-" * 80)

for n_extra in range(5):
    extra = [R7, R8, RK, RQ, R10][:n_extra]
    total = 1 + n_extra
    hands = make_hands([R9] + extra, [RJ, RA], N, seed=42 + n_extra)
    label = f"9+{n_extra}small={total}tr"
    print(f"  {label:<16}", end="")
    for pos, hist, plabel in positions:
        q = get_q(net, hands, pos, hist, trump=TRUMP)
        adv = (q[:, 1] - q[:, 0]).mean()  # Q(80♠) - Q(PASS)
        print(f" {adv:>+16.4f}", end="")
    print()

# Also show J for comparison
print()
for n_extra in range(5):
    extra = [R7, R8, RK, RQ, R10][:n_extra]
    total = 1 + n_extra
    hands = make_hands([RJ] + extra, [R9, RA], N, seed=42 + n_extra)
    label = f"J+{n_extra}small={total}tr"
    print(f"  {label:<16}", end="")
    for pos, hist, plabel in positions:
        q = get_q(net, hands, pos, hist, trump=TRUMP)
        adv = (q[:, 1] - q[:, 0]).mean()
        print(f" {adv:>+16.4f}", end="")
    print()

# ============================================================
print("\n\n" + "=" * 90)
print("  EXP 6: Position 3 deep dive — the 9 is king?")
print("  At pos3 (2 passes), does the 9 become almost as good as J?")
print("=" * 90)

for total in [2, 3, 4, 5]:
    extra = [R7, R8, RK, RQ][:total - 1]
    h_j = make_hands([RJ] + extra, [R9, RA], N, seed=42)
    h_9 = make_hands([R9] + extra, [RJ, RA], N, seed=42)

    # Position 3
    q_j = get_q(net, h_j, 3, [(0, 0), (1, 0)], trump=TRUMP)
    q_9 = get_q(net, h_9, 3, [(0, 0), (1, 0)], trump=TRUMP)

    bj, aj, lj, _ = analyze_decisions(q_j)
    b9, a9, l9, _ = analyze_decisions(q_9)

    print(f"\n  {total} trump at Pos3:")
    print(f"    J-based: bid={bj:.1f}%, avg level={aj:.0f}")
    print(f"    9-based: bid={b9:.1f}%, avg level={a9:.0f}")
    print(f"    Gap: {bj - b9:+.1f}pp bid rate, {aj - a9:+.0f} avg level")

    # Level distributions
    for lbl, lvls in [("J", lj), ("9", l9)]:
        parts = []
        for k in ["PASS", "80", "90", "100", "110", "120"]:
            c = lvls.get(k, 0)
            if c > 0:
                parts.append(f"{k}={c/N*100:.1f}%")
        print(f"    {lbl}: {' '.join(parts)}")

print("\n" + "=" * 90)
print("Done!")
