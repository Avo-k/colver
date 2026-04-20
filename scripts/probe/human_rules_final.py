"""Final human rules: NN-native trump_score + opp_best_other for defense.

Combines every discovery of the night:
  - NN-learned per-card weights (J=11, 9=4, A=1, 10=3, small=2; side A=+1, side J/9=−1)
  - Interaction corrections (J×9=−2, J×A=+1)
  - Side distribution (void +2, singleton +1)
  - "Mirror rule" for defense: use NN-native score excluding opp's suit

Measured on the 720k probe dataset.
"""
from __future__ import annotations

import numpy as np
import torch

from bid_net_torch import load_bid_net
from nn_trump_score_vs_handcrafted import nn_trump_score, handcrafted_trump_score

ACT_PATH = "/tmp/probe_activations.npz"
MODEL_PATH = "models/bid_v5_isdd/bid_nn_final.bin"


def extract_opp_suit(obs: np.ndarray) -> np.ndarray:
    """Find opp's bid suit from bid history, or -1 if no bid."""
    opp_suit = np.full(len(obs), -1, dtype=np.int8)
    for i in range(len(obs)):
        hist = obs[i, 32:104].reshape(12, 6)
        for slot in range(12):
            if abs(hist[slot, 0] - 0.4) < 1e-3:
                opp_suit[i] = int(hist[slot, 2:6].argmax())
                break
    return opp_suit


def extract_partner_suit_from_history(obs: np.ndarray) -> np.ndarray:
    """In partner80 scenarios, partner is seat 2. Find the suit they bid."""
    # For this probe we use scenario_id to distinguish partner vs opp; bid suit is same parser.
    return extract_opp_suit(obs)  # partner/opp are both "a prior bid" — parser is same


def count_J_in_hand(hand_bits: np.ndarray) -> np.ndarray:
    """Number of suits where we have the J (0..4)."""
    return np.stack([hand_bits[:, s*8 + 3] for s in range(4)], axis=1).sum(axis=1)


def compute_side_features(hand_bits: np.ndarray, trump_suit: np.ndarray) -> dict:
    """Given chosen trump per sample, compute distribution-level features."""
    N = len(hand_bits)
    n_voids = np.zeros(N, dtype=np.int32)
    n_singletons = np.zeros(N, dtype=np.int32)
    has_jack = np.zeros(N, dtype=np.int8)
    trump_count = np.zeros(N, dtype=np.int32)
    for i in range(N):
        t = trump_suit[i]
        trump_count[i] = hand_bits[i, t*8:t*8+8].sum()
        has_jack[i] = hand_bits[i, t*8 + 3]
        for s in range(4):
            if s == t:
                continue
            c = hand_bits[i, s*8:s*8+8].sum()
            if c == 0:
                n_voids[i] += 1
            elif c == 1:
                n_singletons[i] += 1
    return dict(trump_count=trump_count, has_jack=has_jack,
                n_voids=n_voids, n_singletons=n_singletons)


def compute_opp_suit_cards(hand_bits: np.ndarray, opp_suit: np.ndarray) -> np.ndarray:
    """Cards in opp's bid suit, or -1 if no opp bid."""
    N = len(hand_bits)
    out = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        if opp_suit[i] >= 0:
            out[i] = hand_bits[i, opp_suit[i]*8:opp_suit[i]*8+8].sum()
    return out


def compute_partner_support(hand_bits: np.ndarray, partner_suit: np.ndarray) -> np.ndarray:
    N = len(hand_bits)
    out = np.full(N, -1, dtype=np.int32)
    for i in range(N):
        if partner_suit[i] >= 0:
            out[i] = hand_bits[i, partner_suit[i]*8:partner_suit[i]*8+8].sum()
    return out


# ===========================================================
# Final rules (uses NN-native score everywhere)
# ===========================================================

def rule_opening(best_score, tc, j, vd, position):
    """pos1-pos4 after-pass rules."""
    if position == 1:   # opening
        t_primary, t_length = 7, 5
    elif position == 2:
        t_primary, t_length = 8, 5
    elif position == 3:
        t_primary, t_length = 6, 4
    else:   # pos4
        t_primary, t_length = 7, 5
    return (
        (best_score >= t_primary)
        | ((best_score >= t_length) & (tc >= 3))
        | ((j == 1) & (vd >= 1) & (tc >= 2))
        | (tc >= 5)
    ).astype(int)


def rule_partner80(partner_support, best_score, tc):
    """Respond unless truly dead (NN-scale thresholds)."""
    pass_hand = (partner_support <= 1) & (best_score < 5) & (tc < 3)
    return (~pass_hand).astype(int)


def rule_opp80_active(best_other_score, tc_other, opp_suit_cards):
    """Active = counter-bid OR coinche.
    KEY: use best_other_score (NN-native trump_score excluding opp's suit).
    NN-native thresholds are lower than classical.
    """
    active = (
        (best_other_score >= 6)                   # strong other suit
        | ((best_other_score >= 4) & (tc_other >= 3))  # decent + length
        | (opp_suit_cards >= 4)                   # coinche territory
    )
    return active.astype(int)


def rule_opp80_coinche(opp_suit_cards, best_other_score):
    """Coinche when holding opp's trump AND no strong alternative."""
    return ((opp_suit_cards >= 4) | ((opp_suit_cards >= 3) & (best_other_score < 5))).astype(int)


# ===========================================================
# Main
# ===========================================================
def main():
    d = np.load(ACT_PATH)
    obs = d["obs"]
    scenario_id = d["scenario_id"]
    nn_bids_truth = d["nn_bids"]
    nn_action = d["nn_action"]
    nn_active_truth = ((nn_bids_truth == 1) | (nn_action == 41)).astype(np.int32)
    nn_coinche_truth = (nn_action == 41).astype(np.int32)
    hand_bits = (obs[:, :32] > 0.5).astype(np.int8)

    # Compute NN-native score per suit
    print("Computing NN-native scores for all 4 suits...")
    nn_scores = np.stack([nn_trump_score(hand_bits, s) for s in range(4)], axis=1)
    nn_best = nn_scores.max(axis=1)
    nn_best_suit = nn_scores.argmax(axis=1)

    hc_scores = np.stack([handcrafted_trump_score(hand_bits, s) for s in range(4)], axis=1)
    hc_best = hc_scores.max(axis=1)
    hc_best_suit = hc_scores.argmax(axis=1)

    # Opp suit extraction (works for both partner80 and opp80 scenarios)
    bid_suit = extract_opp_suit(obs)
    # For pos1-4 "after passes", bid_suit stays -1 (no prior bid). Good.

    # For opp80 scenarios (scen_id 6, 7, 8): best-other = max score excluding opp's suit
    # For partner80 scenarios (scen_id 4, 5): partner_support = cards in partner's suit
    nn_best_other = np.copy(nn_best)
    nn_best_other_suit = np.copy(nn_best_suit)
    for i in range(len(obs)):
        if scenario_id[i] in [6, 7, 8] and bid_suit[i] >= 0:
            # Mask out opp's suit, take max of others
            masked = np.copy(nn_scores[i])
            masked[bid_suit[i]] = -999
            nn_best_other[i] = masked.max()
            nn_best_other_suit[i] = masked.argmax()

    # Compute side features based on chosen trump suit
    sf_nn = compute_side_features(hand_bits, nn_best_suit)
    sf_hc = compute_side_features(hand_bits, hc_best_suit)
    sf_nn_other = compute_side_features(hand_bits, nn_best_other_suit)

    opp_suit_cards = compute_opp_suit_cards(hand_bits, bid_suit)
    partner_support = compute_partner_support(hand_bits, bid_suit)

    SCEN = {
        0: ("pos1_open", 1, "nn_bids"),
        1: ("pos2_after_pass", 2, "nn_bids"),
        2: ("pos3_after_2p", 3, "nn_bids"),
        3: ("pos4_after_3p", 4, "nn_bids"),
    }
    print(f"\n{'scenario':<22} {'HC v1':>8} {'NN final':>10} {'Δ':>7} {'NN rate':>10}")
    print("-" * 70)
    for scen_id, (name, pos, truth_col) in SCEN.items():
        mask = scenario_id == scen_id
        truth = nn_bids_truth[mask]

        # HC baseline (old rules)
        hc_pred = rule_opening(hc_best[mask], sf_hc["trump_count"][mask],
                                sf_hc["has_jack"][mask], sf_hc["n_voids"][mask], pos)
        # Sweep thresholds for HC
        best_hc = 0
        for t1 in range(10, 20):
            for t2 in range(8, 18):
                def r(bs, tc, j, v):
                    return (((bs >= t1) | ((bs >= t2) & (tc >= 3)) | ((j == 1) & (v >= 1) & (tc >= 2)) | (tc >= 5))).astype(int)
                p = r(hc_best[mask], sf_hc["trump_count"][mask], sf_hc["has_jack"][mask], sf_hc["n_voids"][mask])
                a = (p == truth).mean()
                if a > best_hc:
                    best_hc = a

        # NN final
        nn_pred = rule_opening(nn_best[mask], sf_nn["trump_count"][mask],
                                sf_nn["has_jack"][mask], sf_nn["n_voids"][mask], pos)
        acc_nn = (nn_pred == truth).mean()
        print(f"{name:<22} {best_hc*100:>7.1f}% {acc_nn*100:>9.1f}% {(acc_nn-best_hc)*100:>+6.1f}pp {truth.mean()*100:>8.1f}%")

    # Partner 80
    print()
    for scen_id in [4, 5]:
        name = "pos3_partner80" if scen_id == 4 else "pos4_partner80"
        mask = scenario_id == scen_id
        truth = nn_bids_truth[mask]
        pred = rule_partner80(partner_support[mask], nn_best[mask], sf_nn["trump_count"][mask])
        acc = (pred == truth).mean()
        print(f"{name:<22} partner_support rule: acc={acc*100:.1f}%  NN rate={truth.mean()*100:.1f}%")

    # Opp 80 ACTIVE (bid or coinche)
    print()
    for scen_id in [6, 7, 8]:
        name = f"pos{scen_id-4}_opp80"
        mask = scenario_id == scen_id
        truth_active = nn_active_truth[mask]
        truth_coinche = nn_coinche_truth[mask]

        # OLD rule: nn_best (our max) + opp_suit_cards
        old_pred = rule_opp80_active(nn_best[mask], sf_nn["trump_count"][mask], opp_suit_cards[mask])
        acc_old = (old_pred == truth_active).mean()

        # NEW rule: nn_best_other (excludes opp's suit) + tc of that suit
        new_pred = rule_opp80_active(nn_best_other[mask], sf_nn_other["trump_count"][mask], opp_suit_cards[mask])
        acc_new = (new_pred == truth_active).mean()

        # Coinche
        coinche_pred = rule_opp80_coinche(opp_suit_cards[mask], nn_best_other[mask])
        acc_coinche = (coinche_pred == truth_coinche).mean()

        print(f"{name:<22} ACTIVE: old={acc_old*100:.1f}%  new(opp_best_other)={acc_new*100:.1f}%  "
              f"Δ={(acc_new-acc_old)*100:+.1f}pp  (NN rate={truth_active.mean()*100:.1f}%)")
        print(f"{'':<22}   COINCHE: acc={acc_coinche*100:.1f}%  (NN rate={truth_coinche.mean()*100:.1f}%)")

    # SUMMARY
    print("\n=== FINAL SUMMARY: best human rule, all discoveries combined ===\n")
    print(f"{'scenario':<24} {'NN-final rule':>14} {'vs v1 rule':>12} {'vs probe ceiling':>18}")
    print("-" * 75)

    CEILING = {
        "pos1_open": 0.999, "pos2_after_pass": 0.999, "pos3_after_2p": 0.998,
        "pos4_after_3p": 0.999, "pos3_partner80": 0.988, "pos4_partner80": 0.982,
        "pos2_opp80": 0.968, "pos3_opp80": 0.973, "pos4_opp80": 0.972,
    }
    V1_RULES = {
        "pos1_open": 0.824, "pos2_after_pass": 0.866, "pos3_after_2p": 0.910,
        "pos4_after_3p": 0.887, "pos3_partner80": 0.879, "pos4_partner80": 0.879,
        "pos2_opp80": 0.797, "pos3_opp80": 0.826, "pos4_opp80": 0.836,
    }

    for scen_id in range(9):
        name = {0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p",
                4: "pos3_partner80", 5: "pos4_partner80", 6: "pos2_opp80", 7: "pos3_opp80",
                8: "pos4_opp80"}[scen_id]
        mask = scenario_id == scen_id

        if scen_id in [0, 1, 2, 3]:
            pos = scen_id + 1
            pred = rule_opening(nn_best[mask], sf_nn["trump_count"][mask],
                                 sf_nn["has_jack"][mask], sf_nn["n_voids"][mask], pos)
            truth = nn_bids_truth[mask]
        elif scen_id in [4, 5]:
            pred = rule_partner80(partner_support[mask], nn_best[mask], sf_nn["trump_count"][mask])
            truth = nn_bids_truth[mask]
        else:
            pred = rule_opp80_active(nn_best_other[mask], sf_nn_other["trump_count"][mask], opp_suit_cards[mask])
            truth = nn_active_truth[mask]

        acc = (pred == truth).mean()
        v1 = V1_RULES.get(name, 0)
        ceil = CEILING.get(name, 1.0)
        delta_v1 = (acc - v1) * 100
        gap = (ceil - acc) * 100
        print(f"{name:<24} {acc*100:>13.1f}%  {delta_v1:>+11.1f}pp  {gap:>+16.1f}pp")


if __name__ == "__main__":
    main()
