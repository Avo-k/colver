#!/usr/bin/env python3
"""
Test the theory: announcing a suit DOWNGRADES the Ace of that suit.

The Ace is the master card in a side suit (11 pts, strongest).
But as trump, it's only 3rd strongest (behind J=20, 9=14).
So when you announce a suit, you're neutralizing an opponent's Ace
in that suit — turning it from a sure winner into a vulnerable card.

Experiments:
1. Defender's perspective: when opp bids 80♠, how much does A♠ help me vs A♥?
2. Announcer's perspective: does not having the Ace correlate with opponent
   holding a devalued card?
3. DD-based: actual trick/score impact of Ace placement
4. Counterfactual: same hand, announce ♠ vs ♥ — what happens to opponent's Ace value?
"""

import struct
from pathlib import Path
import numpy as np
from collections import Counter

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
        net["w"].append(floats[off:off + d * hidden].reshape(hidden, d)); off += d * hidden
        net["b"].append(floats[off:off + hidden].copy()); off += hidden
        net["gamma"].append(floats[off:off + hidden].copy()); off += hidden
        net["beta"].append(floats[off:off + hidden].copy()); off += hidden
    if dueling:
        net["w_value"] = floats[off:off + hidden].copy(); off += hidden
        net["b_value"] = floats[off]; off += 1
        net["w_adv"] = floats[off:off + hidden * 43].reshape(43, hidden); off += hidden * 43
        net["b_adv"] = floats[off:off + 43].copy(); off += 43
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


def make_obs_opp80(hands, opp_suit, position=2):
    """Build obs for a defender facing opponent's 80 bid.

    Defender is seat 0, position=2 (dealer=2, seat3 bids 80).
    opp_suit: which suit the opponent bid (0-3).
    """
    N = hands.shape[0]
    obs = np.zeros((N, 108), dtype=np.float32)
    obs[:, :32] = hands
    # Position 2: dealer-relative = (0 + 4 - 2) % 4 = 2
    obs[:, 104 + 1] = 1.0  # position index 1 (pos2)

    # Bid history: seat3 bid 80 in opp_suit
    # rel_offset: first_bidder = (dealer+1)%4 = 3, me = 0
    # rel_offset = (3 + 4 - 0) % 4 = 3
    # So seat3's bid is at slot 3... wait let me recalculate
    # Actually, in the obs encoding, slot 0 is the first bidder relative to me.
    # dealer=2, first_bidder=3. me=0.
    # rel_offset = (first_bidder + 4 - me) % 4 = (3 + 4 - 0) % 4 = 3
    # seat3 is the first bidder, so their action goes to slot rel_offset + 0 = 3
    # Hmm, that seems off. Let me re-read the code.
    #
    # From bid_obs.rs: rel_offset = (first_bidder + 4 - me) % 4
    # history[(seat, action)] is enumerated, slot = rel_offset + i
    # With dealer=2, first_bidder=3, me=0: rel_offset = (3+4-0)%4 = 3
    # seat3 bids (i=0): slot = 3+0 = 3
    #
    # But actually for pos2 scenario: dealer=2, first_bidder=3
    # seat3 (opp) bids 80♠. Then it's seat0's turn (me).
    # In history, there's 1 entry: (seat3, bid_action)
    # slot = 3 + 0 = 3... let me just put it at slot 0 like the Rust code does
    # Actually the Rust distill_bid.rs passes (seat, action) pairs and the
    # encode function handles the mapping. Here I need to replicate that.

    # Simpler: just encode the bid at the correct relative slot.
    # The bid is from the player before me. In a 4-player sequence:
    # first_bidder bids, then next, etc. If I'm pos2, one person bid before me.
    # That person is at relative position 0 from first_bidder.
    # rel_offset maps first_bidder's position relative to me.

    # Let me just use slot 0 for the opponent's bid (it's the first action)
    # and correct the relative offset.
    # With dealer=2: first_bidder=3. Me=0.
    # rel_offset = (3 + 4 - 0) % 4 = 3
    # The opponent (seat 3) is the first bidder, their action index in history = 0
    # slot = rel_offset + 0 = 3

    slot = 3  # For pos2 with dealer=2
    bid_action = 1 + opp_suit  # 80 in suit = value_idx=0, action = 0*4 + suit + 1
    base = 32 + slot * 6
    val_enc = 8  # 80 = 8*10
    obs[:, base] = 0.4  # regular bid marker
    obs[:, base + 1] = (val_enc * 10.0) / 250.0
    obs[:, base + 2 + opp_suit] = 1.0

    return obs


def make_obs_opening(hands, position=1):
    N = hands.shape[0]
    obs = np.zeros((N, 108), dtype=np.float32)
    obs[:, :32] = hands
    obs[:, 104 + (position - 1)] = 1.0
    return obs


def random_hands_with_constraint(n, must_have=None, must_not_have=None, seed=42):
    """Generate random 8-card hands with card constraints."""
    rng = np.random.RandomState(seed)
    must_have = must_have or []
    must_not_have = must_not_have or []
    hands = np.zeros((n, 32), dtype=np.float32)
    gen = 0
    for _ in range(n * 50):
        h = np.zeros(32)
        for c in must_have:
            h[c] = 1.0
        placed = len(must_have)
        avail = [c for c in range(32) if h[c] == 0 and c not in must_not_have]
        if len(avail) < 8 - placed:
            continue
        chosen = rng.choice(avail, size=8 - placed, replace=False)
        for c in chosen:
            h[c] = 1.0
        hands[gen] = h
        gen += 1
        if gen >= n:
            break
    return hands[:gen].astype(np.float32)


net = load_bid_net("models/bid_v2/bid_nn_final.bin", 512)
N = 10000

print("=" * 80)
print("  THE ACE DEGRADATION THEORY")
print("  'Announcing a suit downgrades the Ace of that suit'")
print("=" * 80)

# ================================================================
print("\n" + "=" * 80)
print("  EXP 1: ANNOUNCER'S VIEW")
print("  How does having the Ace of trump vs Ace of a side suit affect Q-values?")
print("=" * 80)

# Generate hands with J+9+7 in spades, compare:
# a) Also have A♠ (ace of trump)
# b) Have A♥ instead (ace of side suit)
# c) No ace at all

print("\n  Base: J+9+7♠ (3 trump). Where is the Ace?")

# A♠ = card index 7 (suit 0, rank 7)
# A♥ = card index 15 (suit 1, rank 7)
# A♦ = card index 23 (suit 2, rank 7)

configs = [
    ("A in trump (A♠)", [7], [15, 23, 31]),    # must have A♠, no other aces
    ("A in side (A♥)",  [15], [7, 23, 31]),     # must have A♥, no A♠
    ("A in side (A♦)",  [23], [7, 15, 31]),     # must have A♦
    ("No Ace at all",   [], [7, 15, 23, 31]),   # no aces
    ("2 side Aces",     [15, 23], [7, 31]),     # A♥+A♦, no A♠
    ("A♠ + A♥",         [7, 15], [23, 31]),     # trump ace + side ace
]

# Base trump cards: J♠=3, 9♠=2, 7♠=0
base_trump = [0*8+RJ, 0*8+R9, 0*8+R7]  # spades J, 9, 7

for label, must_aces, must_not_aces in configs:
    must = base_trump + must_aces
    must_not = must_not_aces + [0*8+RA if 0*8+RA not in must else -1,
                                 0*8+R10]  # also exclude 10♠ to keep it clean
    must_not = [x for x in must_not if x >= 0 and x not in must]
    hands = random_hands_with_constraint(N, must_have=must, must_not_have=must_not, seed=42)

    obs = make_obs_opening(hands, position=1)
    q = forward(net, obs)

    q_80s = q[:, 1]  # Q(80♠)
    q_pass = q[:, 0]
    adv = (q_80s - q_pass).mean()

    # Also get best bid
    best = q[:, :41].argmax(axis=1)
    bid_rate = (best > 0).mean()
    avg_level = 0
    n_bids = 0
    for a in best:
        if 1 <= a <= 36:
            avg_level += ((a-1)//4 + 8) * 10
            n_bids += 1
        elif 37 <= a <= 40:
            avg_level += 250
            n_bids += 1
    avg_level = avg_level / max(n_bids, 1)

    print(f"  {label:<22} Q(80♠)-Q(PASS)={adv:>+.4f}  bid={bid_rate:.1%}  avg_lvl={avg_level:.0f}  (N={len(hands)})")


# ================================================================
print("\n\n" + "=" * 80)
print("  EXP 2: DEFENDER'S VIEW")
print("  Opponent bid 80♠. How much does MY Ace help me?")
print("  Compare: A♠ (in their trump) vs A♥ (in a side suit)")
print("=" * 80)

# From defender's perspective: opponent bid 80♠
# I'm deciding whether to bid or pass
# Compare having A♠ vs A♥ vs A♦ vs no ace

print("\n  Opponent bid 80♠. My random hand, but I have...")

for label, must, must_not in [
    ("A♠ (their trump)", [0*8+RA], []),
    ("A♥ (side suit)",   [1*8+RA], [0*8+RA]),
    ("A♦ (side suit)",   [2*8+RA], [0*8+RA]),
    ("A♣ (side suit)",   [3*8+RA], [0*8+RA]),
    ("No Ace",           [], [0*8+RA, 1*8+RA, 2*8+RA, 3*8+RA]),
    ("A♠ + A♥",          [0*8+RA, 1*8+RA], []),
    ("A♥ + A♦",          [1*8+RA, 2*8+RA], [0*8+RA]),
]:
    hands = random_hands_with_constraint(N, must_have=must, must_not_have=must_not, seed=42)
    obs = make_obs_opp80(hands, opp_suit=0, position=2)
    q = forward(net, obs)

    # As defender, my options: pass, bid another suit, or coinche
    q_pass = q[:, 0].mean()
    q_coinche = q[:, 41].mean()
    # Best bid in non-spade suits
    q_90h = q[:, 6].mean()  # 90♥ = action (1*4 + 1 + 1) = 6...
    # Actually: action = (val_idx * 4) + suit_idx + 1
    # 90♥ = val_idx=1, suit=1 → 1*4+1+1 = 6
    # 90♦ = 1*4+2+1 = 7
    # 90♣ = 1*4+3+1 = 8

    best = q[:, :42].argmax(axis=1)  # include coinche
    bid_rate = ((best >= 1) & (best <= 40)).mean()
    coinche_rate = (best == 41).mean()
    pass_rate = (best == 0).mean()

    print(f"  {label:<18} pass={pass_rate:.1%} bid={bid_rate:.1%} coinche={coinche_rate:.1%}  Q(pass)={q_pass:+.3f} Q(coinche)={q_coinche:+.3f}")


# ================================================================
print("\n\n" + "=" * 80)
print("  EXP 3: THE KEY TEST — Ace value from OPPONENT'S perspective")
print("  Does the Ace of the announced suit lose value for the opponent?")
print("=" * 80)

# Controlled experiment:
# Fix opponent's hand EXCEPT one card slot.
# Scenario A: that slot has A♠ (ace of announced trump)
# Scenario B: that slot has A♥ (ace of side suit)
# Everything else identical. Measure how much the Ace helps.

print("\n  Opponent announced 80♠. I have a medium hand with 3♥ as trump candidate.")
print("  I swap ONE card: A♠ (their trump) vs A♥ (my side suit) vs 7♣ (nothing)")
print()

# My base hand: 9♥, K♥, 8♥ (3 hearts), Q♦, 10♦ (2 diamonds), 8♣ (1 club) = 7 cards + 1 slot
base_cards = [1*8+R9, 1*8+RK, 1*8+R8,  # 9♥, K♥, 8♥
              2*8+RQ, 2*8+R10,           # Q♦, 10♦
              3*8+R8, 3*8+R7]            # 8♣, 7♣

# Note: this is a fixed hand, not random — very controlled
for swap_label, swap_card in [
    ("+ A♠ (their trump ace)", 0*8+RA),
    ("+ A♥ (my side ace)",     1*8+RA),
    ("+ A♦ (side ace)",        2*8+RA),
    ("+ A♣ (side ace)",        3*8+RA),
    ("+ 7♠ (trump small)",     0*8+R7),
    ("+ 7♦ (side small)",      2*8+R7),
    ("+ 10♣ (side 10)",        3*8+R10),
]:
    hand = np.zeros((1, 32), dtype=np.float32)
    for c in base_cards:
        hand[0, c] = 1.0
    hand[0, swap_card] = 1.0

    # Check we have exactly 8 cards
    assert hand.sum() == 8, f"Hand has {int(hand.sum())} cards, need 8"

    obs = make_obs_opp80(hand, opp_suit=0, position=2)
    q = forward(net, obs)

    best = q[0, :42].argmax()
    q_pass = q[0, 0]
    q_coinche = q[0, 41]
    q_best = q[0, best]

    action_str = "PASS" if best == 0 else "COINCHE" if best == 41 else f"{((best-1)//4+8)*10}{SUITS[(best-1)%4]}"

    print(f"  {swap_label:<28} → {action_str:<10} Q(best)={q_best:+.3f} Q(pass)={q_pass:+.3f} Q(coinche)={q_coinche:+.3f}")


# ================================================================
print("\n\n" + "=" * 80)
print("  EXP 4: STATISTICAL — Ace value by position (trump vs side)")
print("  Large-scale: random hands, measure Ace contribution")
print("=" * 80)

# For 10k random hands, measure:
# If I have A♠ and opp announces ♠ → how much does A♠ help my Q?
# If I have A♠ and opp announces ♥ → how much does A♠ help my Q?

rng = np.random.RandomState(42)

hands_with_As = random_hands_with_constraint(N, must_have=[0*8+RA], seed=42)
hands_without_As = random_hands_with_constraint(N, must_not_have=[0*8+RA], seed=42)

print("\n  I have A♠. Opponent announces...")
for opp_suit in range(4):
    obs_with = make_obs_opp80(hands_with_As, opp_suit=opp_suit, position=2)
    obs_without = make_obs_opp80(hands_without_As, opp_suit=opp_suit, position=2)

    q_with = forward(net, obs_with)
    q_without = forward(net, obs_without)

    # Best legal Q (excluding coinche for cleanliness)
    best_with = q_with[:, :41].max(axis=1).mean()
    best_without = q_without[:, :41].max(axis=1).mean()
    delta = best_with - best_without

    # Also check: does A♠ help me coinche more when opp announces ♠?
    coinche_rate_with = (q_with[:, :42].argmax(axis=1) == 41).mean()
    coinche_rate_without = (q_without[:, :42].argmax(axis=1) == 41).mean()

    trump_label = "♠ (A is in THEIR trump)" if opp_suit == 0 else f"{SUITS[opp_suit]} (A♠ is side)"
    print(f"  Opp bids 80{SUITS[opp_suit]:} {trump_label:<30}  "
          f"A♠ value={delta:>+.4f}  "
          f"coinche: {coinche_rate_with:.1%} vs {coinche_rate_without:.1%}")


# ================================================================
print("\n\n" + "=" * 80)
print("  EXP 5: SYMMETRIC TEST — Same Ace, different announcements")
print("  I have A♠. How valuable is it depending on what suit is announced?")
print("=" * 80)

# Same hands_with_As. Try each opp suit.
# Measure: Q(best response) - Q(PASS). Higher = my hand is better positioned.

print("\n  Same hands with A♠. How good is my position when opponent announces:")
for opp_suit in range(4):
    obs = make_obs_opp80(hands_with_As, opp_suit=opp_suit, position=2)
    q = forward(net, obs)

    q_pass = q[:, 0].mean()
    q_best_bid = q[:, 1:41].max(axis=1).mean()
    q_coinche = q[:, 41].mean()
    best = q[:, :42].argmax(axis=1)
    bid_rate = ((best >= 1) & (best <= 40)).mean()
    coinche_rate = (best == 41).mean()

    label = "★ A♠ is in their trump!" if opp_suit == 0 else f"  A♠ is a side ace"
    print(f"  80{SUITS[opp_suit]}  bid={bid_rate:.1%} coinche={coinche_rate:.1%}  "
          f"Q(best_bid)={q_best_bid:+.4f} Q(coinche)={q_coinche:+.4f}  {label}")


# ================================================================
print("\n\n" + "=" * 80)
print("  EXP 6: THE ANNOUNCER'S STRATEGIC BENEFIT")
print("  By announcing ♠, I turn the opponent's A♠ from a winner to a loser.")
print("  Quantify: how much does the opponent LOSE when I announce their Ace's suit?")
print("=" * 80)

# Generate opponent hands that have A♠
# Compare: the opponent's expected value when I announce ♠ vs when I announce ♥
# If theory is right: opponent does WORSE when I announce ♠ (their Ace suit)

print("\n  Opponent has A♠. My announcement changes their situation:")
print("  (Measuring opponent's Q-value = how good THEY feel)")
print()

# This requires seeing it from the opponent's side
# When I (seat 0) announce 80♠, seat 1 (opponent with A♠) faces my bid
# When I announce 80♥, seat 1 faces that instead
# The opponent's Q should be worse when I announce ♠ (degrading their Ace)

opp_hands = random_hands_with_constraint(N, must_have=[0*8+RA], seed=142)

# When I announce 80♠ — opponent faces 80♠ and has A♠
obs_announce_s = make_obs_opp80(opp_hands, opp_suit=0, position=2)
q_s = forward(net, obs_announce_s)

# When I announce 80♥ — opponent faces 80♥ and has A♠ (now just a side ace)
obs_announce_h = make_obs_opp80(opp_hands, opp_suit=1, position=2)
q_h = forward(net, obs_announce_h)

# When I announce 80♦
obs_announce_d = make_obs_opp80(opp_hands, opp_suit=2, position=2)
q_d = forward(net, obs_announce_d)

for label, q_vals, opp_suit in [
    ("I announce 80♠ (their A♠ is trump)", q_s, 0),
    ("I announce 80♥ (their A♠ is side)", q_h, 1),
    ("I announce 80♦ (their A♠ is side)", q_d, 2),
]:
    q_pass = q_vals[:, 0].mean()
    q_best = q_vals[:, :42].max(axis=1).mean()
    best = q_vals[:, :42].argmax(axis=1)
    bid_rate = ((best >= 1) & (best <= 40)).mean()
    coinche_rate = (best == 41).mean()
    pass_rate = (best == 0).mean()

    print(f"  {label}")
    print(f"    Opp response: pass={pass_rate:.1%} bid={bid_rate:.1%} coinche={coinche_rate:.1%}")
    print(f"    Q(pass)={q_pass:+.4f}  Q(best)={q_best:+.4f}")
    print()

print("  If the theory is right, the opponent should be WORSE OFF (lower Q, more passes)")
print("  when we announce the suit of their Ace.")

print("\n" + "=" * 80)
print("Done!")
