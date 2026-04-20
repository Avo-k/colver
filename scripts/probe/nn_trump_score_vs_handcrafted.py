"""Compare hand-crafted trump_score vs NN-native trump_score as a bid predictor.

Both scores are used via a SIMPLE rule:
  "bid if score ≥ threshold"

We sweep thresholds, find the best one per score, compare accuracies.
"""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score

from bid_net_torch import load_bid_net

ACT_PATH = "/tmp/probe_activations.npz"
MODEL_PATH = "models/bid_v5_isdd/bid_nn_final.bin"

# ---------- Hand-crafted trump_score (from Rust: evaluate_for_trump) ----------
# rank indexing: 7=0, 8=1, 9=2, J=3, Q=4, K=5, 10=6, A=7
TRUMP_POINTS = np.array([0, 0, 6, 8, 1, 1, 3, 4], dtype=np.int8)  # Rust ranks

def handcrafted_trump_score(hand_bits: np.ndarray, trump_suit: int) -> np.ndarray:
    """Reimplements bid_eval::evaluate_for_trump."""
    N = hand_bits.shape[0]
    trump = hand_bits[:, trump_suit * 8:trump_suit * 8 + 8]
    honours = (trump * TRUMP_POINTS).sum(axis=1).astype(np.int32)
    length_bonus = np.maximum(0, trump.sum(axis=1) - 2).astype(np.int32) * 2

    side_aces = np.zeros(N, dtype=np.int32)
    side_voids = np.zeros(N, dtype=np.int32)
    side_singletons = np.zeros(N, dtype=np.int32)
    for s in range(4):
        if s == trump_suit:
            continue
        ss = hand_bits[:, s * 8:s * 8 + 8]
        count = ss.sum(axis=1)
        side_aces += ss[:, 7] * 3
        side_voids += (count == 0).astype(np.int32) * 3
        side_singletons += (count == 1).astype(np.int32) * 1

    return honours + length_bonus + side_aces + side_voids + side_singletons


# ---------- NN-native trump_score (learned weights, rounded) ----------
# Derived from consolidated regression output. See nn_native_scoring.py.
# Formula: sum of per-card weights + shape + interactions, all integers.
NN_TRUMP_POINTS = np.array([
    # rank index: 7=0, 8=1, 9=2, J=3, Q=4, K=5, 10=6, A=7
    -1,  # 7
    -1,  # 8
    +1,  # 9
    +8,  # J
    -1,  # Q
    -1,  # K
    0,   # 10
    -2,  # A (toxic!)
], dtype=np.int32)

NN_SIDE_POINTS = np.array([
    0,   # 7
    0,   # 8
    -1,  # 9
    -1,  # J  (having the J in a non-trump suit = slightly bad)
    0,   # Q
    0,   # K
    0,   # 10
    +1,  # A  (side A = +1, not +3 like hand-crafted)
], dtype=np.int32)


def nn_trump_score(hand_bits: np.ndarray, trump_suit: int) -> np.ndarray:
    """NN-learned equivalent."""
    N = hand_bits.shape[0]
    trump = hand_bits[:, trump_suit * 8:trump_suit * 8 + 8]
    # per-rank trump points
    honours = (trump * NN_TRUMP_POINTS).sum(axis=1).astype(np.int32)
    # length (3 per trump, uniform — baseline)
    length = trump.sum(axis=1).astype(np.int32) * 3

    # side suit per-rank
    side_rank_sum = np.zeros(N, dtype=np.int32)
    side_lengths = []
    for s in range(4):
        if s == trump_suit:
            continue
        ss = hand_bits[:, s * 8:s * 8 + 8]
        side_rank_sum += (ss * NN_SIDE_POINTS).sum(axis=1)
        side_lengths.append(ss.sum(axis=1))
    side_lengths = np.stack(side_lengths, axis=1)
    # side shape penalties: each card in a long side suit is slightly bad
    # use -1 per side card as proxy
    side_length_penalty = -(side_lengths.sum(axis=1))

    # distribution bonuses
    n_voids = (side_lengths == 0).sum(axis=1).astype(np.int32) * 2
    n_singletons = (side_lengths == 1).sum(axis=1).astype(np.int32) * 1

    # interaction: J × 9 = -2 (anti-synergy)
    j_and_9 = (trump[:, 3] * trump[:, 2]).astype(np.int32) * (-2)
    # interaction: J × A = +1 (slight positive)
    j_and_a = (trump[:, 3] * trump[:, 7]).astype(np.int32) * 1

    return honours + length + side_rank_sum + side_length_penalty + n_voids + n_singletons + j_and_9 + j_and_a


def evaluate_score(score_fn, hand_bits, q_bid_80, q_pass, label):
    """For each sample, compute score per suit, find best suit, check if score >= threshold
    agrees with the NN's (q_80_best >= q_pass) decision."""
    N = hand_bits.shape[0]
    scores = np.stack([score_fn(hand_bits, s) for s in range(4)], axis=1)  # (N, 4)
    best_score = scores.max(axis=1)
    best_suit = scores.argmax(axis=1)
    # NN's decision at 80-level: max over 4 suits of q_bid_80 > q_pass
    q_best_80 = np.take_along_axis(q_bid_80, best_suit[:, None], axis=1).squeeze(1)
    nn_wants_bid = q_best_80 > q_pass

    # Sweep thresholds
    best_acc = 0.0
    best_thr = None
    for thr in range(0, 40):
        pred = best_score >= thr
        acc = (pred == nn_wants_bid).mean()
        if acc > best_acc:
            best_acc = acc
            best_thr = thr
    print(f"  {label:<30}  best threshold={best_thr}, accuracy={best_acc:.4f}")
    return best_thr, best_acc, scores


def main():
    d = np.load(ACT_PATH)
    obs = d["obs"]
    scenario_id = d["scenario_id"]
    hand_bits = (obs[:, :32] > 0.5).astype(np.int8)

    # Forward pass
    print("Forward pass...")
    net = load_bid_net(MODEL_PATH).cuda().eval()
    q_all = np.empty((len(obs), 43), dtype=np.float32)
    batch = 16384
    with torch.no_grad():
        for s in range(0, len(obs), batch):
            e = min(s + batch, len(obs))
            x = torch.from_numpy(obs[s:e].copy()).cuda()
            q_all[s:e] = net(x).cpu().numpy()

    q_bid_80 = q_all[:, 1:5]  # 4 suits at level 80
    q_pass = q_all[:, 0]

    SCEN_NAMES = {
        0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p",
    }
    for scen_id, name in SCEN_NAMES.items():
        mask = scenario_id == scen_id
        if mask.sum() < 1000:
            continue
        print(f"\n=== {name} (n={mask.sum():,}) ===")
        evaluate_score(handcrafted_trump_score, hand_bits[mask], q_bid_80[mask], q_pass[mask],
                       "hand-crafted trump_score")
        evaluate_score(nn_trump_score, hand_bits[mask], q_bid_80[mask], q_pass[mask],
                       "NN-native trump_score")


if __name__ == "__main__":
    main()
