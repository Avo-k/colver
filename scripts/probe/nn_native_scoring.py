"""Extract NN-native card weights via linear regression on the Q-gap.

Target: q[bid_80_in_suit_X] − q[pass], computed per (sample, suit_idx).
  → positive = NN wants to bid 80 in suit X
  → negative = NN wants to pass

Features (~26 per (sample, suit)):
  TRUMP_side (suit X interpreted as trump):
    trump_has_R ∈ {0,1} for R ∈ {7, 8, 9, J, Q, K, 10, A}  (8 binary)
    trump_count                                             (1 int)
  SIDE_side (the 3 other suits):
    side_has_R_count ∈ {0..3} for each rank R                (8 int)
    shape_l1, shape_l2, shape_l3 (sorted side lengths desc)  (3 int)
  INTERACTIONS:
    has_J × has_9              (presence of belote-style duo on trump)
    has_J × trump_count        (J + length)
    has_A × has_J              (toxic-ace interaction?)
    n_side_voids × has_J       (J + coupe)

Fit Ridge regression per scenario → coefficients are "implicit points" the NN
assigns to each card / pattern. Rounded to integers/half-points, we get an
NN-native replacement for the hand-crafted `trump_score`.

Usage: PYTHONPATH=scripts/probe uv run python scripts/probe/nn_native_scoring.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.metrics import r2_score

from bid_net_torch import load_bid_net

ACT_PATH = "/tmp/probe_activations.npz"
MODEL_PATH = "models/bid_v5_isdd/bid_nn_final.bin"


RANK_NAMES = ["7", "8", "9", "J", "Q", "K", "10", "A"]
SCEN_NAMES = {
    0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p",
    4: "pos3_partner80", 5: "pos4_partner80", 6: "pos2_opp80", 7: "pos3_opp80",
    8: "pos4_opp80",
}

# Action index mapping (from bidding.rs): bid = (value-8)*4 + suit + 1, for value 80-130 → 8..13
# bid_80_in_suit[s] = 1 + s (s=0..3)
ACTION_BID_80 = [1, 2, 3, 4]
ACTION_PASS = 0


def build_features_per_suit(hand_bits: np.ndarray, trump_suit: int) -> np.ndarray:
    """hand_bits: (N, 32). Return (N, F) feature matrix for trump_suit."""
    N = hand_bits.shape[0]
    feats = []
    names = []

    # Trump ranks: 8 binary
    for r in range(8):
        feats.append(hand_bits[:, trump_suit * 8 + r].astype(np.int8))
        names.append(f"trump_{RANK_NAMES[r]}")
    # trump_count
    trump_count = hand_bits[:, trump_suit * 8:trump_suit * 8 + 8].sum(axis=1).astype(np.int8)
    feats.append(trump_count)
    names.append("trump_count")

    # Side suit rank counts (0-3 across the 3 side suits)
    side_counts_per_rank = np.zeros((N, 8), dtype=np.int8)
    side_lengths = []
    for s in range(4):
        if s == trump_suit:
            continue
        side_suit = hand_bits[:, s * 8:s * 8 + 8]
        side_counts_per_rank += side_suit.astype(np.int8)
        side_lengths.append(side_suit.sum(axis=1))
    side_lengths = np.stack(side_lengths, axis=1)  # (N, 3)
    side_lengths_sorted = np.sort(side_lengths, axis=1)[:, ::-1]

    for r in range(8):
        feats.append(side_counts_per_rank[:, r])
        names.append(f"side_{RANK_NAMES[r]}")

    # Shape of side suits (sorted desc)
    feats.append(side_lengths_sorted[:, 0].astype(np.int8))
    names.append("side_l1")
    feats.append(side_lengths_sorted[:, 1].astype(np.int8))
    names.append("side_l2")
    feats.append(side_lengths_sorted[:, 2].astype(np.int8))
    names.append("side_l3")
    # side voids/singletons/doubletons
    n_voids = (side_lengths == 0).sum(axis=1).astype(np.int8)
    n_singletons = (side_lengths == 1).sum(axis=1).astype(np.int8)
    feats.append(n_voids)
    names.append("side_n_voids")
    feats.append(n_singletons)
    names.append("side_n_singletons")

    # Interactions
    trump_J = hand_bits[:, trump_suit * 8 + 3]
    trump_9 = hand_bits[:, trump_suit * 8 + 2]
    trump_A = hand_bits[:, trump_suit * 8 + 7]
    trump_10 = hand_bits[:, trump_suit * 8 + 6]
    feats.append((trump_J & trump_9).astype(np.int8))
    names.append("trump_J×9")
    feats.append((trump_J & trump_A).astype(np.int8))
    names.append("trump_J×A")
    feats.append((trump_J & trump_10).astype(np.int8))
    names.append("trump_J×10")
    feats.append((trump_J * trump_count).astype(np.int8))
    names.append("trump_J×count")
    feats.append((trump_9 * trump_count).astype(np.int8))
    names.append("trump_9×count")
    feats.append((trump_J * n_voids).astype(np.int8))
    names.append("trump_J×voids")

    X = np.stack(feats, axis=1).astype(np.float32)
    return X, names


def main():
    print(f"Loading probe data...")
    d = np.load(ACT_PATH)
    obs = d["obs"]
    scenario_id = d["scenario_id"]
    N = len(obs)
    print(f"  {N:,} samples")

    # Forward pass to get Q-values for all samples
    print(f"Loading NN → forward pass for Q-values...")
    net = load_bid_net(MODEL_PATH).cuda().eval()
    q_all = np.empty((N, 43), dtype=np.float32)
    batch = 16384
    t0 = time.time()
    with torch.no_grad():
        for s in range(0, N, batch):
            e = min(s + batch, N)
            x = torch.from_numpy(obs[s:e].copy()).cuda()
            q = net(x).cpu().numpy()
            q_all[s:e] = q
    print(f"  Q-values: {time.time()-t0:.1f}s")

    hand_bits = (obs[:, :32] > 0.5).astype(np.int8)

    all_results = {}

    for scen_id, scen_name in SCEN_NAMES.items():
        mask = scenario_id == scen_id
        if mask.sum() < 1000:
            continue

        print(f"\n=== {scen_name} (n={mask.sum():,}) ===")

        # Build per-suit dataset: for each sample, 4 rows (one per potential trump suit)
        X_all, y_all = [], []
        sub_obs = obs[mask]
        sub_hand = hand_bits[mask]
        sub_q = q_all[mask]
        for suit_idx in range(4):
            X, names = build_features_per_suit(sub_hand, suit_idx)
            # q_gap for this suit: q[bid_80_suit] - q[pass]
            y = sub_q[:, ACTION_BID_80[suit_idx]] - sub_q[:, ACTION_PASS]
            X_all.append(X)
            y_all.append(y)
        X_all = np.concatenate(X_all, axis=0)
        y_all = np.concatenate(y_all, axis=0)

        # Ridge regression (handles collinearity between trump_count and rank sums)
        model = Ridge(alpha=1.0)
        model.fit(X_all, y_all)
        y_pred = model.predict(X_all)
        r2 = r2_score(y_all, y_pred)

        # Rescale so that J coefficient ≈ 8 (matching hand-crafted).
        # Find trump_J coefficient index
        j_idx = names.index("trump_J")
        j_coef = model.coef_[j_idx]
        if abs(j_coef) < 1e-6:
            scale = 1.0
        else:
            scale = 8.0 / j_coef  # so trump_J becomes 8

        print(f"  R²={r2:.3f}   trump_J raw coef={j_coef:.3f}   scale={scale:.3f}")
        print(f"  Intercept (raw) = {model.intercept_:.3f}  (×scale = {model.intercept_*scale:.2f})")
        print(f"\n  {'feature':<20} {'raw coef':>10} {'scaled':>10} {'int':>6}")
        coefs = []
        for n, c in zip(names, model.coef_):
            sc = c * scale
            coefs.append((n, float(c), float(sc), round(sc)))
            print(f"  {n:<20} {c:>+10.4f} {sc:>+10.2f} {round(sc):>+6d}")

        all_results[scen_name] = {
            "r2": float(r2),
            "scale": float(scale),
            "intercept_scaled": float(model.intercept_ * scale),
            "coefs": coefs,
        }

    # Consolidated view: average coefs across pos1/pos2/pos3/pos4 (opening-ish scenarios)
    print("\n\n=== CONSOLIDATED CARD WEIGHTS (average over pos1/pos2/pos3/pos4) ===")
    primary_scens = ["pos1_open", "pos2_after_pass", "pos3_after_2p", "pos4_after_3p"]
    avg_coefs = {}
    for scen in primary_scens:
        for n, _, sc, _ in all_results[scen]["coefs"]:
            avg_coefs.setdefault(n, []).append(sc)
    print(f"  {'feature':<20} {'mean':>10} {'std':>10} {'int':>6}")
    summary = []
    for n in names:
        if n in avg_coefs:
            arr = np.array(avg_coefs[n])
            m, s = arr.mean(), arr.std()
            print(f"  {n:<20} {m:>+10.2f} {s:>+10.2f} {round(m):>+6d}")
            summary.append((n, float(m), float(s), round(m)))
    all_results["_consolidated"] = {"coefs": summary}

    with open("/tmp/nn_native_scoring.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\n[saved] /tmp/nn_native_scoring.json")


if __name__ == "__main__":
    main()
