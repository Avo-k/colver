"""Full rule set using NN-native trump_score.

Test: does replacing hand-crafted trump_score with NN-learned trump_score
improve human rule accuracy?
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from bid_net_torch import load_bid_net
from nn_trump_score_vs_handcrafted import handcrafted_trump_score, nn_trump_score

ACT_PATH = "/tmp/probe_activations.npz"
MODEL_PATH = "models/bid_v5_isdd/bid_nn_final.bin"


def extract_opp_suit(obs: np.ndarray) -> np.ndarray:
    opp_suit = np.full(len(obs), -1, dtype=np.int8)
    for i in range(len(obs)):
        hist = obs[i, 32:104].reshape(12, 6)
        for slot in range(12):
            if abs(hist[slot, 0] - 0.4) < 1e-3:
                opp_suit[i] = int(hist[slot, 2:6].argmax())
                break
    return opp_suit


def rule_opening(ts: np.ndarray, tc: np.ndarray, has_j: np.ndarray, voids: np.ndarray, thr: tuple) -> np.ndarray:
    # thr = (primary, length_2nd)
    t_primary, t_length = thr
    return (
        (ts >= t_primary)
        | ((ts >= t_length) & (tc >= 3))
        | ((has_j == 1) & (voids >= 1) & (tc >= 3))
        | (tc >= 5)
    )


def main():
    d = np.load(ACT_PATH)
    obs = d["obs"]
    scenario_id = d["scenario_id"]
    hand_bits = (obs[:, :32] > 0.5).astype(np.int8)

    # Forward pass
    net = load_bid_net(MODEL_PATH).cuda().eval()
    q_all = np.empty((len(obs), 43), dtype=np.float32)
    with torch.no_grad():
        for s in range(0, len(obs), 16384):
            e = min(s + 16384, len(obs))
            x = torch.from_numpy(obs[s:e].copy()).cuda()
            q_all[s:e] = net(x).cpu().numpy()

    q_bid_80 = q_all[:, 1:5]
    q_pass = q_all[:, 0]

    # Compute per-suit ts (both scorings)
    hc_scores = np.stack([handcrafted_trump_score(hand_bits, s) for s in range(4)], axis=1)
    nn_scores = np.stack([nn_trump_score(hand_bits, s) for s in range(4)], axis=1)

    hc_best = hc_scores.max(axis=1)
    nn_best = nn_scores.max(axis=1)
    hc_suit = hc_scores.argmax(axis=1)
    nn_suit = nn_scores.argmax(axis=1)

    # Trump count + J of best suit
    def trump_count_of(hand, suit_arr):
        N = len(hand)
        out = np.zeros(N, dtype=np.int32)
        for i in range(N):
            s = suit_arr[i]
            out[i] = hand[i, s*8:s*8+8].sum()
        return out

    hc_tc = trump_count_of(hand_bits, hc_suit)
    nn_tc = trump_count_of(hand_bits, nn_suit)
    # J of best suit
    hc_hasj = np.array([hand_bits[i, hc_suit[i]*8 + 3] for i in range(len(hand_bits))], dtype=np.int8)
    nn_hasj = np.array([hand_bits[i, nn_suit[i]*8 + 3] for i in range(len(hand_bits))], dtype=np.int8)

    # Side voids (wrt chosen suit)
    def compute_voids(hand, suit_arr):
        N = len(hand)
        out = np.zeros(N, dtype=np.int32)
        for i in range(N):
            s = suit_arr[i]
            for ss in range(4):
                if ss == s:
                    continue
                if hand[i, ss*8:ss*8+8].sum() == 0:
                    out[i] += 1
        return out
    hc_voids = compute_voids(hand_bits, hc_suit)
    nn_voids = compute_voids(hand_bits, nn_suit)

    # NN "wants to bid" truth
    best_q_bid = q_bid_80.max(axis=1)
    nn_wants = best_q_bid > q_pass

    SCEN = {0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p"}
    print(f"\n{'scenario':<22} {'hc rule':>12} {'nn rule':>12} {'NN bid %':>10}")
    print("-" * 70)
    for scen_id, name in SCEN.items():
        mask = scenario_id == scen_id
        y = nn_wants[mask]

        # Sweep thresholds for each
        best_hc = 0
        best_thr_hc = None
        for t1 in range(10, 20):
            for t2 in range(8, 18):
                pred = rule_opening(hc_best[mask], hc_tc[mask], hc_hasj[mask], hc_voids[mask], (t1, t2))
                acc = (pred == y).mean()
                if acc > best_hc:
                    best_hc = acc
                    best_thr_hc = (t1, t2)

        best_nn = 0
        best_thr_nn = None
        for t1 in range(0, 15):
            for t2 in range(-5, 12):
                pred = rule_opening(nn_best[mask], nn_tc[mask], nn_hasj[mask], nn_voids[mask], (t1, t2))
                acc = (pred == y).mean()
                if acc > best_nn:
                    best_nn = acc
                    best_thr_nn = (t1, t2)

        print(f"{name:<22} {best_hc*100:>10.1f}%{'':3}[{best_thr_hc[0]:>2},{best_thr_hc[1]:>2}]  "
              f"{best_nn*100:>6.1f}%{'':3}[{best_thr_nn[0]:>2},{best_thr_nn[1]:>2}]  "
              f"{y.mean()*100:>8.1f}%")


if __name__ == "__main__":
    main()
