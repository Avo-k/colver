"""Human rules v2: using the newly discovered features.

Key additions vs v1:
  - ts_2nd : trump_score of my 2nd-best suit (fallback option)
  - n_suits_ge_14: how many decent-as-trump suits I have
  - opp_best_other_ts: my best non-opp suit (defense)
  - Per-suit J/9 awareness (via n_suits_with_J, etc.)

Goal: push human-usable rules from 82-91% to 90%+ across all scenarios.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from discover_features import engineer_candidate_features, BASE

ACT_PATH = "/tmp/probe_activations.npz"


def compute_per_deal_ts(obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Compute trump_score per suit (N, 4) and identify opp_suit per sample."""
    hand_bits = (obs[:, :32] > 0.5).astype(np.int8)
    TRUMP_POINTS = np.array([0, 0, 0, 8, 1, 1, 3, 4], dtype=np.int8)

    def trump_score_for(suit_idx: int) -> np.ndarray:
        base = suit_idx * 8
        suit = hand_bits[:, base:base + 8]
        honours = (suit * TRUMP_POINTS).sum(axis=1).astype(np.int32)
        side_aces = np.zeros(len(hand_bits), dtype=np.int32)
        side_voids = np.zeros(len(hand_bits), dtype=np.int32)
        side_singletons = np.zeros(len(hand_bits), dtype=np.int32)
        length_bonus = np.maximum(0, suit.sum(axis=1) - 2).astype(np.int32) * 2
        for s in range(4):
            if s == suit_idx:
                continue
            sbase = s * 8
            other = hand_bits[:, sbase:sbase + 8]
            count = other.sum(axis=1)
            side_aces += other[:, 7] * 3
            side_voids += (count == 0).astype(np.int32) * 3
            side_singletons += (count == 1).astype(np.int32) * 1
        return honours + side_aces + side_voids + side_singletons + length_bonus

    ts_per_suit = np.stack([trump_score_for(s) for s in range(4)], axis=1)
    return ts_per_suit


def extract_opp_suit(obs: np.ndarray) -> np.ndarray:
    """Parse bid history to find opp's bid suit, or -1 if none."""
    opp_suit = np.full(len(obs), -1, dtype=np.int8)
    for i in range(len(obs)):
        hist = obs[i, 32:104].reshape(12, 6)
        for slot in range(12):
            if abs(hist[slot, 0] - 0.4) < 1e-3:
                opp_suit[i] = int(hist[slot, 2:6].argmax())
                break
    return opp_suit


def agreement(pred, truth, label):
    acc = (pred == truth).mean()
    n = len(truth)
    miss = ((pred == 0) & (truth == 1)).sum() / n
    fa = ((pred == 1) & (truth == 0)).sum() / n
    print(f"  [{label}] n={n:,}  acc={acc:.1%}  miss={miss:.1%}  FA={fa:.1%}  rule_rate={pred.mean():.1%}")


# =========================================================
# RULES v2
# =========================================================

def rule_opening(df: pd.DataFrame) -> np.ndarray:
    """pos1. trump_score_best: strongest suit; n_ge_14: count of decent suits."""
    ts = df["ts_best"]
    tc = df["tc_best"]
    n14 = df["n_ge_14"]
    j = df["best_has_jack"]
    vd = df["side_voids"]

    bid = (
        (ts >= 17)
        | ((ts >= 14) & (tc >= 3))
        | ((ts >= 15) & (n14 >= 2))
        | ((j == 1) & (vd >= 1) & (tc >= 3))
        | (tc >= 5)
    )
    return bid.astype(int).values


def rule_after_pass(df: pd.DataFrame, position: int) -> np.ndarray:
    ts = df["ts_best"]
    tc = df["tc_best"]
    n14 = df["n_ge_14"]
    j = df["best_has_jack"]
    vd = df["side_voids"]

    if position == 2:
        bid = (
            (ts >= 16)
            | ((ts >= 13) & (tc >= 3))
            | ((ts >= 14) & (n14 >= 2))
            | ((j == 1) & (vd >= 1) & (tc >= 3))
            | (tc >= 4)
        )
    elif position == 3:
        bid = (
            (ts >= 12)
            | ((ts >= 10) & (tc >= 3))
            | ((j == 1) & (tc >= 2))
            | ((vd >= 1) & (ts >= 9))
            | (n14 >= 1)
        )
    else:  # pos4
        bid = (
            (ts >= 14)
            | ((ts >= 11) & (tc >= 3))
            | ((j == 1) & (vd >= 1) & (tc >= 2))
            | (tc >= 4)
            | ((ts >= 12) & (n14 >= 2))
        )
    return bid.astype(int).values


def rule_partner80(df: pd.DataFrame) -> np.ndarray:
    """Partner bid 80 — almost always respond, pass only on dead hands."""
    ps = df["partner_support"]
    ts = df["ts_best"]
    tc = df["tc_best"]
    pass_hand = (ps <= 1) & (ts < 12) & (tc < 3)
    return (~pass_hand).astype(int).values


def rule_opp80_active(df: pd.DataFrame) -> np.ndarray:
    """KEY DISCOVERY: use opp_best_other_ts = my best non-opp suit.
    Be active if we have a real option besides opp's suit."""
    ts_other = df["opp_best_other_ts"]
    tc = df["tc_best"]
    osc = df["opp_suit_cards"]
    ts_best = df["ts_best"]
    is_opp = df["best_is_opp_suit"]

    active = (
        # Own strong non-opp suit
        (ts_other >= 14)
        # Medium suit + length
        | ((ts_other >= 11) & (tc >= 3))
        # Coinche territory: lots of cards in opp's trump
        | (osc >= 4)
        # Own best is the opp's suit but VERY strong
        | ((is_opp == 1) & (ts_best >= 20))
    )
    return active.astype(int).values


def rule_opp80_coinche(df: pd.DataFrame) -> np.ndarray:
    """Coinche: we hold opp's trump. Own alternate suit must be weak."""
    osc = df["opp_suit_cards"]
    ts_other = df["opp_best_other_ts"]
    return ((osc >= 4) | ((osc >= 3) & (ts_other < 11))).astype(int).values


def main():
    d = np.load(ACT_PATH)
    scenario_id = d["scenario_id"]
    features = d["features"].astype(np.float32)
    obs = d["obs"]
    nn_bids = d["nn_bids"]
    nn_action = d["nn_action"]

    ts_per_suit = compute_per_deal_ts(obs)
    ts_sorted = np.sort(ts_per_suit, axis=1)[:, ::-1]
    n_ge_14 = (ts_per_suit >= 14).sum(axis=1)
    opp_suit = extract_opp_suit(obs)

    opp_best_other = np.zeros(len(obs), dtype=np.int32)
    for i in range(len(obs)):
        if opp_suit[i] >= 0:
            others = np.delete(ts_per_suit[i], opp_suit[i])
            opp_best_other[i] = others.max()
        else:
            opp_best_other[i] = ts_sorted[i, 0]

    # Recompute per-deal base features from features array (since features[] is for the "reference" suit)
    # For per-deal rules we use the best suit (max ts per row).
    best_suit_idx = ts_per_suit.argmax(axis=1)
    hand_bits = (obs[:, :32] > 0.5).astype(np.int8)
    tc_best = np.array([hand_bits[i, best_suit_idx[i]*8:best_suit_idx[i]*8+8].sum() for i in range(len(obs))], dtype=np.int32)
    # J, 9 in best suit
    best_has_jack = np.array([hand_bits[i, best_suit_idx[i]*8+3] for i in range(len(obs))], dtype=np.int8)

    # side_voids (across all 3 non-best-trump suits)
    side_voids = np.zeros(len(obs), dtype=np.int32)
    for i in range(len(obs)):
        bs = best_suit_idx[i]
        for s in range(4):
            if s == bs:
                continue
            c = hand_bits[i, s*8:s*8+8].sum()
            if c == 0:
                side_voids[i] += 1

    # partner_support / opp_suit_cards: derive from bid history + hand.
    # Partner of seat 0 is seat 2. Opp = seat 1 or 3.
    partner_suit = np.full(len(obs), -1, dtype=np.int8)
    for i in range(len(obs)):
        hist = obs[i, 32:104].reshape(12, 6)
        # Scan history. The dealer info is not in obs, but the scenario pattern is:
        # partner80 scenarios → the (seat 2) bid appears as the 'partner' bid.
        # For simplicity, use scenario id: partner80 = id 4-5, opp80 = id 6-8.
        if scenario_id[i] in (4, 5):
            # In partner80, partner bid at some slot. Find it (any bid).
            for slot in range(12):
                if abs(hist[slot, 0] - 0.4) < 1e-3:
                    partner_suit[i] = int(hist[slot, 2:6].argmax())
                    break

    partner_support = np.zeros(len(obs), dtype=np.int32)
    opp_suit_cards = np.zeros(len(obs), dtype=np.int32)
    for i in range(len(obs)):
        if partner_suit[i] >= 0:
            partner_support[i] = hand_bits[i, partner_suit[i]*8:partner_suit[i]*8+8].sum()
        else:
            partner_support[i] = -1
        if opp_suit[i] >= 0:
            opp_suit_cards[i] = hand_bits[i, opp_suit[i]*8:opp_suit[i]*8+8].sum()
        else:
            opp_suit_cards[i] = -1

    best_is_opp_suit = (best_suit_idx == opp_suit).astype(np.int8)

    deal_df = pd.DataFrame({
        "ts_best": ts_sorted[:, 0],
        "ts_2nd": ts_sorted[:, 1],
        "tc_best": tc_best,
        "n_ge_14": n_ge_14,
        "side_voids": side_voids,
        "best_has_jack": best_has_jack,
        "best_is_opp_suit": best_is_opp_suit,
        "partner_support": partner_support,
        "opp_suit_cards": opp_suit_cards,
        "opp_best_other_ts": opp_best_other,
        "nn_bids": nn_bids,
        "nn_coinche": (nn_action == 41).astype(np.int32),
        "nn_active": ((nn_bids == 1) | (nn_action == 41)).astype(np.int32),
        "scenario_id": scenario_id,
    })

    SCEN = {
        0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p",
        4: "pos3_partner80", 5: "pos4_partner80", 6: "pos2_opp80", 7: "pos3_opp80",
        8: "pos4_opp80",
    }

    print("=== RULES V2 (with discovered features) ===\n")
    print(f"{'scenario':<22} {'NN rate':>8} {'acc':>7} {'miss':>7} {'FA':>7} {'rule rate':>10}")

    results = []
    def run(scen_ids, rule_fn, target, label):
        mask = np.isin(scenario_id, scen_ids)
        sub = deal_df[mask]
        y = sub[target].values
        pred = rule_fn(sub)
        acc = (pred == y).mean()
        miss = ((pred == 0) & (y == 1)).sum() / len(y)
        fa = ((pred == 1) & (y == 0)).sum() / len(y)
        print(f"{label:<22} {y.mean():>8.1%} {acc:>7.1%} {miss:>7.1%} {fa:>7.1%} {pred.mean():>10.1%}")
        results.append((label, len(y), float(acc)))

    run([0], rule_opening, "nn_bids", "Opening (pos1)")
    run([1], lambda df: rule_after_pass(df, 2), "nn_bids", "Pos 2 after 1 pass")
    run([2], lambda df: rule_after_pass(df, 3), "nn_bids", "Pos 3 after 2 passes")
    run([3], lambda df: rule_after_pass(df, 4), "nn_bids", "Pos 4 after 3 passes")
    run([4, 5], rule_partner80, "nn_bids", "Partner bid 80")
    run([6], rule_opp80_active, "nn_active", "Opp 80 — active pos2")
    run([7], rule_opp80_active, "nn_active", "Opp 80 — active pos3")
    run([8], rule_opp80_active, "nn_active", "Opp 80 — active pos4")
    run([6, 7, 8], rule_opp80_coinche, "nn_coinche", "Opp 80 — coinche")

    print("\n=== SUMMARY ===")
    for label, n, acc in results:
        print(f"  {label:<30}  n={n:,}  acc={acc:.2%}")


if __name__ == "__main__":
    main()
