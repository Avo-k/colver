#!/usr/bin/env python3
"""
Card combination analysis on the bid NN V2.

Instead of individual card SHAP, analyzes groups and combos:
1. Trump honor combos: J+9, J+A, J+9+A, 9+A, etc.
2. Trump length with fixed honors: J+9+{1,2,3,4} small cards
3. Side card value: effect of side aces, side voids, side 10s
4. Full hand archetypes: "tricolore", "bicolore", etc.

Method: Monte Carlo controlled experiments. For each comparison,
generate thousands of random hands matching the constraint, measure
average Q(80_trump) - Q(PASS).

Usage:
    PYTHONPATH=scripts uv run python scripts/shap_combos.py
"""

import struct
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["7", "8", "9", "J", "Q", "K", "10", "A"]
CARD_NAMES = [f"{RANKS[r]}{SUITS[s]}" for s in range(4) for r in range(8)]

# Rank indices
R7, R8, R9, RJ, RQ, RK, R10, RA = 0, 1, 2, 3, 4, 5, 6, 7
SMALL = [R7, R8, RQ, RK]  # non-honor ranks (not J, 9, A, 10)
HONORS = {"J": RJ, "9": R9, "A": RA, "10": R10, "K": RK, "Q": RQ, "8": R8, "7": R7}


# ============================================================
# NN (reuse from shap_bid.py)
# ============================================================

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


def bid_advantage(net, hands, trump_suit=0):
    """Compute Q(80_trump) - Q(PASS) for a batch of hands."""
    N = hands.shape[0]
    obs = np.zeros((N, 108), dtype=np.float32)
    obs[:, :32] = hands
    obs[:, 105] = 1.0  # position 1 (dealer-relative index 1)
    q = forward(net, obs)
    return q[:, trump_suit + 1] - q[:, 0]  # Q(80_suit) - Q(PASS)


def best_bid_advantage(net, hands):
    """max over suits of Q(80_suit) - Q(PASS)."""
    N = hands.shape[0]
    obs = np.zeros((N, 108), dtype=np.float32)
    obs[:, :32] = hands
    obs[:, 105] = 1.0
    q = forward(net, obs)
    q80 = np.stack([q[:, s+1] for s in range(4)], axis=1)
    return q80.max(axis=1) - q[:, 0]


# ============================================================
# Hand generation with constraints
# ============================================================

def make_hands(trump_ranks, side_constraints, n, trump_suit=0, seed=42):
    """Generate n random hands with specific trump cards and side constraints.

    trump_ranks: list of rank indices that MUST be in the trump suit
    side_constraints: dict with optional keys:
        'side_aces': int — exact number of aces in non-trump suits
        'side_voids': int — exact number of void non-trump suits
        'side_tens': int — exact number of 10s in non-trump suits
        'extra_trump': list of rank indices — additional trump cards to include
        'no_trump': list of rank indices — trump ranks to exclude
    """
    rng = np.random.RandomState(seed)
    hands = np.zeros((n, 32), dtype=np.float32)
    generated = 0
    attempts = 0

    while generated < n and attempts < n * 200:
        attempts += 1
        hand = np.zeros(32, dtype=np.float32)

        # Place required trump cards
        for r in trump_ranks:
            hand[trump_suit * 8 + r] = 1.0
        # Place extra trump if specified
        extra = side_constraints.get("extra_trump", [])
        for r in extra:
            hand[trump_suit * 8 + r] = 1.0

        no_trump = side_constraints.get("no_trump", [])
        for r in no_trump:
            if hand[trump_suit * 8 + r] > 0:
                continue  # conflict — skip this constraint

        placed = int(hand.sum())
        remaining_slots = 8 - placed

        # Available cards (not yet placed, not excluded trump)
        available = []
        for c in range(32):
            if hand[c] > 0:
                continue
            s, r = c // 8, c % 8
            if s == trump_suit and r in no_trump:
                continue
            available.append(c)

        if len(available) < remaining_slots:
            continue

        # If we have side constraints, try to satisfy them
        target_aces = side_constraints.get("side_aces", None)
        target_voids = side_constraints.get("side_voids", None)
        target_tens = side_constraints.get("side_tens", None)

        # Simple rejection sampling: place remaining cards randomly, check constraints
        chosen = rng.choice(available, size=remaining_slots, replace=False)
        for c in chosen:
            hand[c] = 1.0

        # Check constraints
        ok = True
        if target_aces is not None:
            actual = sum(1 for s in range(4) if s != trump_suit and hand[s*8 + RA] > 0)
            if actual != target_aces:
                ok = False
        if target_voids is not None:
            actual = sum(1 for s in range(4) if s != trump_suit and sum(hand[s*8:s*8+8]) == 0)
            if actual != target_voids:
                ok = False
        if target_tens is not None:
            actual = sum(1 for s in range(4) if s != trump_suit and hand[s*8 + R10] > 0)
            if actual != target_tens:
                ok = False

        if ok:
            hands[generated] = hand
            generated += 1

    if generated < n:
        print(f"  Warning: only generated {generated}/{n} hands (constraints too tight)")
    return hands[:generated]


def trump_description(trump_ranks):
    """Human name for a set of trump ranks."""
    return "".join(RANKS[r] for r in sorted(trump_ranks, key=lambda r: -HONORS.get(RANKS[r], r)))


# ============================================================
# Experiments
# ============================================================

def exp_honor_combos(net, out):
    """Compare trump honor combinations with same total count."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("  EXPERIMENT 1: Trump Honor Combos (3 trump cards, varied honors)\n")
    out.write("=" * 80 + "\n")
    out.write("  Fixed: 3 trump cards total. Compare which honors matter.\n\n")

    N = 5000
    combos = [
        ("J+9+7", [RJ, R9, R7]),
        ("J+9+8", [RJ, R9, R8]),
        ("J+9+A", [RJ, R9, RA]),
        ("J+9+10", [RJ, R9, R10]),
        ("J+9+K", [RJ, R9, RK]),
        ("J+A+7", [RJ, RA, R7]),
        ("J+A+10", [RJ, RA, R10]),
        ("J+A+K", [RJ, RA, RK]),
        ("J+7+8", [RJ, R7, R8]),
        ("J+10+K", [RJ, R10, RK]),
        ("J+K+Q", [RJ, RK, RQ]),
        ("9+A+10", [R9, RA, R10]),
        ("9+A+7", [R9, RA, R7]),
        ("9+7+8", [R9, R7, R8]),
        ("9+K+Q", [R9, RK, RQ]),
        ("A+10+K", [RA, R10, RK]),
        ("A+K+Q", [RA, RK, RQ]),
        ("K+Q+7", [RK, RQ, R7]),
        ("10+K+Q", [R10, RK, RQ]),
        ("7+8+Q", [R7, R8, RQ]),
    ]

    results = []
    for label, ranks in combos:
        hands = make_hands(ranks, {}, N, seed=42)
        if len(hands) < 100:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        results.append((label, adv.mean(), adv.std(), len(hands)))

    results.sort(key=lambda x: -x[1])
    out.write(f"  {'Combo':<14} {'Advantage':>10} {'Std':>8} {'N':>6}\n")
    out.write("  " + "-" * 42 + "\n")
    for label, mean, std, n in results:
        bar = "+" * int(max(0, mean) * 60) + "-" * int(max(0, -mean) * 60)
        out.write(f"  {label:<14} {mean:>+10.4f} {std:>8.4f} {n:>6}  {bar}\n")


def exp_honor_combos_4(net, out):
    """Compare 4-card trump combos."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("  EXPERIMENT 2: Trump Honor Combos (4 trump cards)\n")
    out.write("=" * 80 + "\n\n")

    N = 5000
    combos = [
        ("J+9+A+10", [RJ, R9, RA, R10]),
        ("J+9+A+7", [RJ, R9, RA, R7]),
        ("J+9+A+K", [RJ, R9, RA, RK]),
        ("J+9+10+7", [RJ, R9, R10, R7]),
        ("J+9+K+Q", [RJ, R9, RK, RQ]),
        ("J+9+7+8", [RJ, R9, R7, R8]),
        ("J+A+10+K", [RJ, RA, R10, RK]),
        ("J+A+7+8", [RJ, RA, R7, R8]),
        ("J+A+K+Q", [RJ, RA, RK, RQ]),
        ("J+10+K+Q", [RJ, R10, RK, RQ]),
        ("J+7+8+Q", [RJ, R7, R8, RQ]),
        ("9+A+10+K", [R9, RA, R10, RK]),
        ("9+A+7+8", [R9, RA, R7, R8]),
        ("9+7+8+Q", [R9, R7, R8, RQ]),
        ("A+10+K+Q", [RA, R10, RK, RQ]),
        ("7+8+K+Q", [R7, R8, RK, RQ]),
    ]

    results = []
    for label, ranks in combos:
        hands = make_hands(ranks, {}, N, seed=42)
        if len(hands) < 100:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        results.append((label, adv.mean(), adv.std(), len(hands)))

    results.sort(key=lambda x: -x[1])
    out.write(f"  {'Combo':<14} {'Advantage':>10} {'Std':>8} {'N':>6}\n")
    out.write("  " + "-" * 42 + "\n")
    for label, mean, std, n in results:
        bar = "+" * int(max(0, mean) * 40)
        out.write(f"  {label:<14} {mean:>+10.4f} {std:>8.4f} {n:>6}  {bar}\n")


def exp_trump_length(net, out):
    """Effect of trump length with fixed J+9 core."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("  EXPERIMENT 3: Trump Length (J+9 + N small cards)\n")
    out.write("=" * 80 + "\n\n")

    N = 5000
    for n_extra in range(5):  # 0 to 4 extra = 2 to 6 total
        extra = [R7, R8, RK, RQ, R10][:n_extra]
        total = 2 + n_extra
        hands = make_hands([RJ, R9] + extra, {"no_trump": [RA]}, N, seed=42 + n_extra)
        if len(hands) < 100:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        out.write(f"  J9 + {n_extra} small = {total} trump: advantage={adv.mean():>+.4f} (std={adv.std():.4f}, N={len(hands)})\n")

    out.write("\n  Same with J alone:\n")
    for n_extra in range(5):
        extra = [R7, R8, RK, RQ, R10][:n_extra]
        total = 1 + n_extra
        if total < 1:
            continue
        hands = make_hands([RJ] + extra, {"no_trump": [R9, RA]}, N, seed=142 + n_extra)
        if len(hands) < 100:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        out.write(f"  J + {n_extra} small = {total} trump:  advantage={adv.mean():>+.4f} (std={adv.std():.4f}, N={len(hands)})\n")

    out.write("\n  Same with 9 alone:\n")
    for n_extra in range(5):
        extra = [R7, R8, RK, RQ, R10][:n_extra]
        total = 1 + n_extra
        hands = make_hands([R9] + extra, {"no_trump": [RJ, RA]}, N, seed=242 + n_extra)
        if len(hands) < 100:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        out.write(f"  9 + {n_extra} small = {total} trump:  advantage={adv.mean():>+.4f} (std={adv.std():.4f}, N={len(hands)})\n")

    out.write("\n  No J, no 9 (small cards only):\n")
    for n_cards in range(2, 7):
        extra = [R7, R8, RK, RQ, R10, RA][:n_cards]
        hands = make_hands(extra, {"no_trump": [RJ, R9]}, N, seed=342 + n_cards)
        if len(hands) < 100:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        out.write(f"  {n_cards} small (no J/9): advantage={adv.mean():>+.4f} (N={len(hands)})\n")


def exp_side_impact(net, out):
    """Effect of side cards with fixed trump J+9+7 (3 trump)."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("  EXPERIMENT 4: Side Card Impact (fixed trump: J+9+7)\n")
    out.write("=" * 80 + "\n\n")

    N = 3000

    # Side aces
    out.write("  --- Side aces ---\n")
    for n_aces in range(4):
        hands = make_hands([RJ, R9, R7], {"side_aces": n_aces}, N, seed=42 + n_aces)
        if len(hands) < 50:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        out.write(f"  {n_aces} side aces: advantage={adv.mean():>+.4f} (N={len(hands)})\n")

    # Side voids
    out.write("\n  --- Side voids ---\n")
    for n_voids in range(4):
        hands = make_hands([RJ, R9, R7], {"side_voids": n_voids}, N, seed=142 + n_voids)
        if len(hands) < 50:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        out.write(f"  {n_voids} side voids: advantage={adv.mean():>+.4f} (N={len(hands)})\n")

    # Side tens
    out.write("\n  --- Side 10s ---\n")
    for n_tens in range(4):
        hands = make_hands([RJ, R9, R7], {"side_tens": n_tens}, N, seed=242 + n_tens)
        if len(hands) < 50:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        out.write(f"  {n_tens} side 10s: advantage={adv.mean():>+.4f} (N={len(hands)})\n")

    # Cross: aces vs voids (which matters more?)
    out.write("\n  --- Aces vs Voids (head to head) ---\n")
    for aces, voids in [(1, 0), (0, 1), (2, 0), (0, 2), (1, 1)]:
        hands = make_hands([RJ, R9, R7], {"side_aces": aces, "side_voids": voids}, N, seed=342 + aces * 10 + voids)
        if len(hands) < 50:
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        out.write(f"  {aces} aces + {voids} voids: advantage={adv.mean():>+.4f} (N={len(hands)})\n")


def exp_j_plus_what(net, out):
    """The key question: J + which card is the best 2nd card?"""
    out.write("\n" + "=" * 80 + "\n")
    out.write("  EXPERIMENT 5: J + X — What's the best 2nd trump card?\n")
    out.write("  (2 trump cards: J + one other, rest random sides)\n")
    out.write("=" * 80 + "\n\n")

    N = 5000
    partners = [
        ("J+9", [RJ, R9]),
        ("J+A", [RJ, RA]),
        ("J+10", [RJ, R10]),
        ("J+K", [RJ, RK]),
        ("J+Q", [RJ, RQ]),
        ("J+8", [RJ, R8]),
        ("J+7", [RJ, R7]),
    ]

    results = []
    for label, ranks in partners:
        exclude = [r for r in range(8) if r not in ranks]
        hands = make_hands(ranks, {"no_trump": exclude}, N, seed=42)
        if len(hands) < 100:
            # Fallback: don't exclude other trump
            hands = make_hands(ranks, {}, N, seed=42)
        adv = bid_advantage(net, hands, trump_suit=0)
        results.append((label, adv.mean(), adv.std(), len(hands)))

    results.sort(key=lambda x: -x[1])
    out.write(f"  {'Combo':<10} {'Advantage':>10} {'Std':>8}\n")
    out.write("  " + "-" * 32 + "\n")
    for label, mean, std, n in results:
        bar = "+" * int(max(0, mean + 0.05) * 50)
        out.write(f"  {label:<10} {mean:>+10.4f} {std:>8.4f}  {bar}\n")


def exp_9_plus_what(net, out):
    """Same for 9 + X."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("  EXPERIMENT 6: 9 + X — Best 2nd card when no Jack?\n")
    out.write("  (2 trump cards: 9 + one other, no J in trump)\n")
    out.write("=" * 80 + "\n\n")

    N = 5000
    partners = [
        ("9+A", [R9, RA]),
        ("9+10", [R9, R10]),
        ("9+K", [R9, RK]),
        ("9+Q", [R9, RQ]),
        ("9+8", [R9, R8]),
        ("9+7", [R9, R7]),
    ]

    results = []
    for label, ranks in partners:
        hands = make_hands(ranks, {"no_trump": [RJ]}, N, seed=42)
        adv = bid_advantage(net, hands, trump_suit=0)
        results.append((label, adv.mean(), adv.std(), len(hands)))

    results.sort(key=lambda x: -x[1])
    out.write(f"  {'Combo':<10} {'Advantage':>10} {'Std':>8}\n")
    out.write("  " + "-" * 32 + "\n")
    for label, mean, std, n in results:
        out.write(f"  {label:<10} {mean:>+10.4f} {std:>8.4f}\n")


def exp_archetypes(net, out):
    """Classic hand archetypes."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("  EXPERIMENT 7: Hand Archetypes\n")
    out.write("=" * 80 + "\n\n")

    N = 3000

    archetypes = [
        # (label, trump_ranks, side_constraints, description)
        ("Monster J9A10", [RJ, R9, RA, R10], {}, "4 top trump"),
        ("J9 + belote", [RJ, R9, RK, RQ], {}, "J9KQ"),
        ("J9 + 2 small", [RJ, R9, R7, R8], {}, "J9 + garbage trump"),
        ("J alone", [RJ], {"no_trump": [R9, RA]}, "J solitaire"),
        ("9 alone 3 trump", [R9, R7, R8], {"no_trump": [RJ, RA]}, "9 + 2 small"),
        ("J9 + 1 void", [RJ, R9, R7], {"side_voids": 1}, "short, with cut"),
        ("J9 + 2 voids", [RJ, R9, R7], {"side_voids": 2}, "very short"),
        ("J9 + 0 void + 2 aces", [RJ, R9, R7], {"side_voids": 0, "side_aces": 2}, "long sides + aces"),
        ("5 trump J9", [RJ, R9, R7, R8, RQ], {}, "5 trump with J9"),
        ("5 trump no J/9", [R7, R8, RK, RQ, R10], {"no_trump": [RJ, R9]}, "5 trump, no honors"),
        ("6 trump J9", [RJ, R9, R7, R8, RK, RQ], {}, "6 trump J9"),
        ("3 side aces no J", [R7, R8, RK], {"side_aces": 3, "no_trump": [RJ, R9]}, "aux as"),
    ]

    results = []
    for label, trump, side, desc in archetypes:
        hands = make_hands(trump, side, N, seed=42)
        if len(hands) < 30:
            out.write(f"  {label:<25} — too few hands ({len(hands)})\n")
            continue
        adv = bid_advantage(net, hands, trump_suit=0)
        results.append((label, adv.mean(), adv.std(), len(hands), desc))

    results.sort(key=lambda x: -x[1])
    out.write(f"  {'Archetype':<25} {'Advantage':>10} {'Std':>8} {'N':>5}  Description\n")
    out.write("  " + "-" * 75 + "\n")
    for label, mean, std, n, desc in results:
        bar = "+" * int(max(0, mean + 0.02) * 30)
        out.write(f"  {label:<25} {mean:>+10.4f} {std:>8.4f} {n:>5}  {desc}  {bar}\n")


def exp_added_value(net, out):
    """Marginal value of adding a specific card to J+9+7 base."""
    out.write("\n" + "=" * 80 + "\n")
    out.write("  EXPERIMENT 8: Adding a 4th trump to J+9+7\n")
    out.write("  (What's the marginal value of each additional trump card?)\n")
    out.write("=" * 80 + "\n\n")

    N = 5000
    base = [RJ, R9, R7]
    base_hands = make_hands(base, {"no_trump": [r for r in range(8) if r not in base]}, N, seed=42)
    base_adv = bid_advantage(net, base_hands, trump_suit=0).mean()
    out.write(f"  Base (J+9+7, 3 trump): advantage = {base_adv:+.4f}\n\n")

    additions = [
        ("+ A", RA), ("+ 10", R10), ("+ K", RK), ("+ Q", RQ), ("+ 8", R8),
    ]

    results = []
    for label, rank in additions:
        hands = make_hands(base + [rank], {}, N, seed=42)
        adv = bid_advantage(net, hands, trump_suit=0).mean()
        delta = adv - base_adv
        results.append((label, adv, delta))

    results.sort(key=lambda x: -x[2])
    out.write(f"  {'Added card':<12} {'Total':>8} {'Delta':>8}\n")
    out.write("  " + "-" * 32 + "\n")
    for label, total, delta in results:
        bar = "+" * int(max(0, delta) * 100)
        out.write(f"  {label:<12} {total:>+8.4f} {delta:>+8.4f}  {bar}\n")


# ============================================================
# Main
# ============================================================

def main():
    import sys

    model_path = "models/bid_v2/bid_nn_final.bin"
    net = load_bid_net(model_path, hidden=512)
    print(f"Loaded NN (obs={net['obs_dim']}, hidden={net['hidden']}, layers={net['layers']})")

    log_path = "data/shap/shap_combos.log"
    log_file = open(log_path, "w")

    class Tee:
        def __init__(self, *w):
            self.writers = w
        def write(self, s):
            for w in self.writers:
                w.write(s)
        def flush(self):
            for w in self.writers:
                w.flush()

    out = Tee(sys.stdout, log_file)

    exp_j_plus_what(net, out)
    exp_9_plus_what(net, out)
    exp_honor_combos(net, out)
    exp_honor_combos_4(net, out)
    exp_trump_length(net, out)
    exp_side_impact(net, out)
    exp_added_value(net, out)
    exp_archetypes(net, out)

    out.write(f"\n{'='*80}\n  Full log: {log_path}\n{'='*80}\n")
    log_file.close()
    print(f"\nDone!")


if __name__ == "__main__":
    main()
