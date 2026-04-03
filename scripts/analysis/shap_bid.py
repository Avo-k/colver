#!/usr/bin/env python3
"""
SHAP analysis of the bid NN V2.

1. SHAP on XGBoost (17 features) — TreeExplainer, instant
2. Direct card-level analysis on the NN — marginal contribution of each card
3. Gradient-based saliency on the NN — which input bits the NN is sensitive to

Usage:
    PYTHONPATH=scripts uv run python scripts/shap_bid.py
"""

import struct
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import shap
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

SUITS = ["♠", "♥", "♦", "♣"]
RANKS = ["7", "8", "9", "J", "Q", "K", "10", "A"]
CARD_NAMES = [f"{RANKS[r]}{SUITS[s]}" for s in range(4) for r in range(8)]
TRUMP_POINTS = [0, 0, 14, 20, 3, 4, 10, 11]  # per rank as trump

# Suit colors for plots
SUIT_COLORS = {"♠": "#1a5276", "♥": "#c0392b", "♦": "#e67e22", "♣": "#27ae60"}


# ============================================================
# NN loading and forward pass (numpy)
# ============================================================

def load_bid_net(path: str, hidden: int = 512):
    data = Path(path).read_bytes()
    n_floats = len(data) // 4
    floats = np.array(struct.unpack(f"<{n_floats}f", data), dtype=np.float32)

    num_actions = 43
    for layers in [3, 4, 2]:
        for dueling in [True, False]:
            trunk_fixed = (layers - 1) * (hidden * hidden + 3 * hidden) + 3 * hidden
            tail = (hidden + 1 + hidden * num_actions + num_actions) if dueling else (hidden * num_actions + num_actions)
            fixed = trunk_fixed + tail
            if n_floats > fixed and (n_floats - fixed) % hidden == 0:
                obs_dim = (n_floats - fixed) // hidden
                if 0 < obs_dim <= 500:
                    print(f"NN: obs_dim={obs_dim}, hidden={hidden}, layers={layers}, dueling={dueling}")
                    return _parse_weights(floats, obs_dim, hidden, layers, dueling, num_actions)
    raise ValueError(f"Cannot detect architecture")


def _parse_weights(floats, obs_dim, hidden, layers, dueling, num_actions):
    off = 0
    net = {"obs_dim": obs_dim, "hidden": hidden, "layers": layers, "dueling": dueling}
    net["w"], net["b"], net["gamma"], net["beta"] = [], [], [], []
    in_dims = [obs_dim] + [hidden] * (layers - 1)
    for layer in range(layers):
        in_d = in_dims[layer]
        net["w"].append(floats[off:off + in_d * hidden].reshape(hidden, in_d)); off += in_d * hidden
        net["b"].append(floats[off:off + hidden].copy()); off += hidden
        net["gamma"].append(floats[off:off + hidden].copy()); off += hidden
        net["beta"].append(floats[off:off + hidden].copy()); off += hidden
    if dueling:
        net["w_value"] = floats[off:off + hidden].copy(); off += hidden
        net["b_value"] = floats[off]; off += 1
        net["w_adv"] = floats[off:off + hidden * num_actions].reshape(num_actions, hidden); off += hidden * num_actions
        net["b_adv"] = floats[off:off + num_actions].copy(); off += num_actions
    else:
        net["w_out"] = floats[off:off + hidden * num_actions].reshape(num_actions, hidden); off += hidden * num_actions
        net["b_out"] = floats[off:off + num_actions].copy(); off += num_actions
    assert off == len(floats)
    return net


def forward(net, obs_batch):
    x = obs_batch
    for layer in range(net["layers"]):
        x = x @ net["w"][layer].T + net["b"][layer]
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x = net["gamma"][layer] * (x - mean) / np.sqrt(var + 1e-5) + net["beta"][layer]
        x = np.maximum(x, 0)
    if net["dueling"]:
        v = x @ net["w_value"] + net["b_value"]
        adv = x @ net["w_adv"].T + net["b_adv"]
        return v[:, None] + adv - adv.mean(axis=1, keepdims=True)
    return x @ net["w_out"].T + net["b_out"]


def make_opening_obs(hand_bits):
    """Build 108-dim obs for opening (pos1, empty history)."""
    N = hand_bits.shape[0]
    obs = np.zeros((N, 108), dtype=np.float32)
    obs[:, :32] = hand_bits
    obs[:, 104 + 1] = 1.0  # position 1
    return obs


def random_hands(n, seed=42):
    rng = np.random.RandomState(seed)
    hands = np.zeros((n, 32), dtype=np.float32)
    for i in range(n):
        cards = rng.choice(32, size=8, replace=False)
        hands[i, cards] = 1.0
    return hands


# ============================================================
# Card-level marginal contribution analysis
# ============================================================

def marginal_contribution(net, n_deals=20000, seed=42):
    """For each card, compute its average marginal contribution to Q(80_best_suit) - Q(PASS).

    Method: for each random hand, for each card in the hand, compute:
      contribution[card] = Q_with_card - Q_without_card (replaced by random other card)
    This is a Monte Carlo estimate of the Shapley value marginal.
    """
    print("\n" + "=" * 80)
    print("  MARGINAL CARD CONTRIBUTIONS (Monte Carlo, N={})".format(n_deals))
    print("=" * 80)

    rng = np.random.RandomState(seed)
    # Accumulate per-card stats
    card_contrib_sum = np.zeros(32, dtype=np.float64)  # sum of contributions
    card_contrib_sq = np.zeros(32, dtype=np.float64)   # sum of squared contributions
    card_count = np.zeros(32, dtype=np.int64)           # how many times card appeared

    # Per-suit contributions (card's marginal to Q(80_suit) for the suit it belongs to)
    card_trump_contrib_sum = np.zeros(32, dtype=np.float64)
    card_trump_count = np.zeros(32, dtype=np.int64)

    batch_size = 500
    for batch_start in range(0, n_deals, batch_size):
        bs = min(batch_size, n_deals - batch_start)
        hands = random_hands(bs, seed=seed + batch_start)

        # Get baseline Q-values
        obs = make_opening_obs(hands)
        q = forward(net, obs)
        q_pass = q[:, 0]
        # Best Q(80) across suits
        q_80_suits = np.stack([q[:, s + 1] for s in range(4)], axis=1)
        best_suit = q_80_suits.argmax(axis=1)
        q_80_best = q_80_suits[np.arange(bs), best_suit]
        advantage = q_80_best - q_pass

        for i in range(bs):
            hand_cards = np.where(hands[i] > 0.5)[0]
            non_hand = np.where(hands[i] < 0.5)[0]

            for card in hand_cards:
                # Remove this card, add a random replacement
                replacement = rng.choice(non_hand)
                modified = hands[i].copy()
                modified[card] = 0.0
                modified[replacement] = 1.0

                obs_mod = make_opening_obs(modified[np.newaxis])
                q_mod = forward(net, obs_mod)
                q_pass_mod = q_mod[0, 0]
                q_80_mod = np.array([q_mod[0, s + 1] for s in range(4)])
                q_80_best_mod = q_80_mod.max()
                advantage_mod = q_80_best_mod - q_pass_mod

                contrib = advantage[i] - advantage_mod
                card_contrib_sum[card] += contrib
                card_contrib_sq[card] += contrib ** 2
                card_count[card] += 1

                # Also track as trump contribution (in its own suit)
                card_suit = card // 8
                q_trump = q[i, card_suit + 1] - q_pass[i]
                q_trump_mod = q_mod[0, card_suit + 1] - q_pass_mod
                trump_contrib = q_trump - q_trump_mod
                card_trump_contrib_sum[card] += trump_contrib
                card_trump_count[card] += 1

        if batch_start % 2000 == 0:
            print(f"  {batch_start}/{n_deals}...")

    # Compute means
    card_mean = np.where(card_count > 0, card_contrib_sum / card_count, 0)
    card_std = np.where(card_count > 0,
                        np.sqrt(card_contrib_sq / card_count - card_mean ** 2), 0)
    card_trump_mean = np.where(card_trump_count > 0,
                               card_trump_contrib_sum / card_trump_count, 0)

    # === Print results ===
    print("\n--- Overall bid advantage marginal contribution ---")
    print("  (How much does having this card increase max Q(80) - Q(PASS)?)")
    print(f"  {'Card':<8} {'Mean':>8} {'Std':>8} {'Count':>7}")
    print("  " + "-" * 35)
    ranked = sorted(range(32), key=lambda i: -card_mean[i])
    for idx in ranked:
        if card_count[idx] < 100:
            continue
        bar = "+" * int(max(0, card_mean[idx]) * 100) + "-" * int(max(0, -card_mean[idx]) * 100)
        print(f"  {CARD_NAMES[idx]:<8} {card_mean[idx]:>+8.4f} {card_std[idx]:>8.4f} {card_count[idx]:>7}  {bar}")

    # Group by rank
    print("\n--- By rank (averaged across suits) ---")
    for r in range(8):
        indices = [s * 8 + r for s in range(4)]
        total_count = sum(card_count[i] for i in indices)
        if total_count == 0:
            continue
        avg = sum(card_contrib_sum[i] for i in indices) / total_count
        avg_trump = sum(card_trump_contrib_sum[i] for i in indices) / sum(max(1, card_trump_count[i]) for i in indices)
        print(f"  {RANKS[r]:<4}  overall={avg:>+.4f}  as_trump={avg_trump:>+.4f}")

    # === As trump only ===
    print("\n--- As trump: marginal contribution to Q(80_own_suit) ---")
    print(f"  {'Card':<8} {'Mean':>8}")
    print("  " + "-" * 20)
    ranked_trump = sorted(range(32), key=lambda i: -card_trump_mean[i])
    for idx in ranked_trump:
        if card_trump_count[idx] < 100:
            continue
        print(f"  {CARD_NAMES[idx]:<8} {card_trump_mean[idx]:>+8.4f}")

    # === Plot ===
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot 1: Overall contribution by card
    ax = axes[0]
    names = [CARD_NAMES[i] for i in ranked]
    vals = [card_mean[i] for i in ranked]
    colors = [SUIT_COLORS[CARD_NAMES[i][-1]] for i in ranked]
    ax.barh(range(32), list(reversed(vals)),
            tick_label=list(reversed(names)),
            color=list(reversed(colors)))
    ax.set_xlabel("Mean marginal contribution to bid advantage")
    ax.set_title("Card importance (overall)")
    ax.axvline(x=0, color="black", linewidth=0.5)

    # Plot 2: As-trump contribution
    ax = axes[1]
    names_t = [CARD_NAMES[i] for i in ranked_trump]
    vals_t = [card_trump_mean[i] for i in ranked_trump]
    colors_t = [SUIT_COLORS[CARD_NAMES[i][-1]] for i in ranked_trump]
    ax.barh(range(32), list(reversed(vals_t)),
            tick_label=list(reversed(names_t)),
            color=list(reversed(colors_t)))
    ax.set_xlabel("Mean marginal contribution to Q(80_own_suit)")
    ax.set_title("Card importance (as trump)")
    ax.axvline(x=0, color="black", linewidth=0.5)

    plt.tight_layout()
    plt.savefig("data/shap/shap_card_contributions.png", dpi=150)
    print("\nSaved: data/shap/shap_card_contributions.png")

    # === Rank × position heatmap ===
    fig, ax = plt.subplots(figsize=(10, 5))
    heatmap = np.zeros((4, 8))  # suits × ranks
    for s in range(4):
        for r in range(8):
            idx = s * 8 + r
            heatmap[s, r] = card_trump_mean[idx]

    im = ax.imshow(heatmap, cmap="RdYlGn", aspect="auto")
    ax.set_xticks(range(8))
    ax.set_xticklabels(RANKS)
    ax.set_yticks(range(4))
    ax.set_yticklabels(SUITS, fontsize=16)
    ax.set_title("Marginal contribution as trump by card")
    for s in range(4):
        for r in range(8):
            ax.text(r, s, f"{heatmap[s, r]:+.3f}", ha="center", va="center", fontsize=9)
    plt.colorbar(im, ax=ax)
    plt.tight_layout()
    plt.savefig("data/shap/shap_card_heatmap.png", dpi=150)
    print("Saved: data/shap/shap_card_heatmap.png")

    return card_mean, card_trump_mean


# ============================================================
# SHAP on XGBoost
# ============================================================

def shap_xgboost(csv_path: str):
    print("\n" + "=" * 80)
    print("  SHAP on XGBoost (17 engineered features)")
    print("=" * 80)

    df = pd.read_csv(csv_path)
    df = df[df["scenario"] == "pos1_open"]
    print(f"Loaded {len(df)} rows (pos1_open)")

    features = [
        "trump_count", "has_jack", "has_nine", "has_ace", "has_ten",
        "has_king", "has_queen", "trump_points", "trump_score", "has_belote",
        "side_aces", "side_tens", "side_voids", "side_singletons",
        "side_doubletons", "total_aces", "best_side_length",
    ]

    X = df[features].values
    y = df["nn_bids_this_suit"].values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    from xgboost import XGBClassifier
    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=scale, random_state=42, verbosity=0,
    )
    xgb.fit(X_train, y_train)

    print("Computing SHAP (TreeExplainer)...")
    explainer = shap.TreeExplainer(xgb)
    sample = X_test[:5000]
    sv = explainer.shap_values(sample)

    # Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(sv, sample, feature_names=features, show=False)
    plt.tight_layout()
    plt.savefig("data/shap/shap_xgb_summary.png", dpi=150, bbox_inches="tight")
    print("Saved: data/shap/shap_xgb_summary.png")

    # Bar plot
    plt.figure(figsize=(10, 6))
    shap.summary_plot(sv, sample, feature_names=features, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig("data/shap/shap_xgb_bar.png", dpi=150, bbox_inches="tight")
    print("Saved: data/shap/shap_xgb_bar.png")

    # Dependence plots
    for feat_idx, feat_name in [(1, "has_jack"), (2, "has_nine"), (0, "trump_count"),
                                 (3, "has_ace"), (10, "side_voids")]:
        plt.figure(figsize=(8, 5))
        shap.dependence_plot(feat_idx, sv, sample, feature_names=features, show=False)
        plt.tight_layout()
        plt.savefig(f"data/shap/shap_xgb_dep_{feat_name}.png", dpi=150)
    print("Saved: data/shap/shap_xgb_dep_*.png")

    # SHAP interaction
    print("\n--- Mean |SHAP| per feature ---")
    mean_abs = np.abs(sv).mean(axis=0)
    ranked = sorted(zip(features, mean_abs), key=lambda x: -x[1])
    for feat, val in ranked:
        bar = "█" * int(val * 30)
        print(f"  {feat:<22} {val:.4f} {bar}")

    # Direction analysis
    print("\n--- SHAP direction (positive = helps bid, negative = helps pass) ---")
    for feat_idx, feat_name in enumerate(features):
        # Mean SHAP when feature is high vs low
        median = np.median(sample[:, feat_idx])
        high_mask = sample[:, feat_idx] > median
        low_mask = ~high_mask
        if high_mask.sum() > 0 and low_mask.sum() > 0:
            mean_high = sv[high_mask, feat_idx].mean()
            mean_low = sv[low_mask, feat_idx].mean()
            direction = "+" if mean_high > mean_low else "-"
            print(f"  {feat_name:<22} high→{mean_high:>+.3f}  low→{mean_low:>+.3f}  ({direction})")


# ============================================================

def main():
    model_path = "models/bid_v2/bid_nn_final.bin"
    csv_path = "data/distill/bid_distill.csv"

    # 1. SHAP on XGBoost
    if Path(csv_path).exists():
        shap_xgboost(csv_path)

    # 2. Card-level marginal contributions on NN
    if Path(model_path).exists():
        net = load_bid_net(model_path, hidden=512)

        # Quick validation
        test_hand = np.zeros((1, 32), dtype=np.float32)
        for c in [3, 2, 7, 6, 15, 13, 17, 24]:  # J9A10♠ AK♥ 8♦ 7♣
            test_hand[0, c] = 1.0
        q = forward(net, make_opening_obs(test_hand))
        print(f"\nValidation: J9A10♠ AK♥ 8♦ 7♣")
        print(f"  Q(PASS)={q[0,0]:.4f}  Q(80♠)={q[0,1]:.4f}  Q(80♥)={q[0,2]:.4f}")
        best_legal = max(range(41), key=lambda i: q[0, i])  # exclude coinche/surcoinche
        print(f"  Best legal bid: action {best_legal} (Q={q[0, best_legal]:.4f})")

        marginal_contribution(net, n_deals=20000)

    print("\n" + "=" * 80)
    print("  All plots saved to data/shap/shap_*.png")
    print("=" * 80)


if __name__ == "__main__":
    main()
