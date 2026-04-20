"""Why does XGBoost plateau on opp80? Test specific hypotheses:
  - H1: second_trump_score gives the gain (alt-suit availability)
  - H2: second suit per-detail (has_J, has_9 of second-best suit)
  - H3: "my trump suit vs opp suit" gap — diff(ts_mine, ts_opp)
  - H4: interaction bid-history × hand: opp_suit_cards * trump_count
  - H5: full per-suit evaluation (trump_score per suit, not just best + counts)
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from discover_features import engineer_candidate_features, BASE

ACT_PATH = "/tmp/probe_activations.npz"


def fit(X, y):
    if y.mean() < 0.02 or y.mean() > 0.98:
        return float("nan")
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=(yt == 0).sum() / max((yt == 1).sum(), 1),
        random_state=42, verbosity=0, n_jobs=-1,
    )
    m.fit(Xt, yt)
    return accuracy_score(yv, m.predict(Xv)), m


def top_imp(m, names, k=12):
    imp = sorted(zip(names, m.feature_importances_), key=lambda t: -t[1])
    return [(n, float(i)) for n, i in imp[:k] if i > 0.002]


def main():
    d = np.load(ACT_PATH)
    scenario_id = d["scenario_id"]
    features = d["features"].astype(np.float32)
    obs = d["obs"]
    nn_bids = d["nn_bids"]
    nn_action = d["nn_action"]

    # Compute trump_score for each of the 4 suits per hand (from obs bitmask).
    # Use the same Rust evaluate_for_trump logic: for each suit, trump score =
    #   honours (J=8, 9=6, A=4, 10=3, K=1, Q=1) + side aces + voids/singletons + length bonus.
    # This is expensive in pure Python so let's do a vectorized version.
    hand_bits = (obs[:, :32] > 0.5).astype(np.int8)  # (N, 32)
    TRUMP_POINTS = np.array([0, 0, 0, 8, 1, 1, 3, 4], dtype=np.int8)  # rank 0..7
    SIDE_POINTS = np.array([0, 0, 0, 0, 0, 0, 0, 3], dtype=np.int8)  # side ace = 3

    def trump_score_for(suit_idx: int) -> np.ndarray:
        base = suit_idx * 8
        suit = hand_bits[:, base:base + 8]
        # honours
        honours = (suit * TRUMP_POINTS).sum(axis=1).astype(np.int32)
        # side aces (aces in OTHER suits) + voids + singletons
        side_aces = 0
        side_voids = 0
        side_singletons = 0
        length_bonus = np.maximum(0, suit.sum(axis=1) - 2).astype(np.int32) * 2
        for s in range(4):
            if s == suit_idx:
                continue
            sbase = s * 8
            other = hand_bits[:, sbase:sbase + 8]
            count = other.sum(axis=1)
            side_aces = side_aces + other[:, 7] * 3  # ace of side suit
            side_voids = side_voids + (count == 0).astype(np.int32) * 3
            side_singletons = side_singletons + (count == 1).astype(np.int32) * 1
        return honours + side_aces + side_voids + side_singletons + length_bonus

    ts_per_suit = np.stack([trump_score_for(s) for s in range(4)], axis=1)  # (N, 4)
    ts_sorted = np.sort(ts_per_suit, axis=1)[:, ::-1]

    # Per-suit detailed features
    per_suit = pd.DataFrame({
        "ts_best": ts_sorted[:, 0],
        "ts_2nd": ts_sorted[:, 1],
        "ts_3rd": ts_sorted[:, 2],
        "ts_4th": ts_sorted[:, 3],
        "ts_gap": ts_sorted[:, 0] - ts_sorted[:, 1],
        "ts_sum": ts_per_suit.sum(axis=1),
        "ts_max_minus_min": ts_sorted[:, 0] - ts_sorted[:, 3],
        "n_suits_ge_14": (ts_per_suit >= 14).sum(axis=1),
        "n_suits_ge_10": (ts_per_suit >= 10).sum(axis=1),
    })

    # Engineered shape features
    eng = engineer_candidate_features(features, obs)
    # combine
    all_df = pd.concat([eng.reset_index(drop=True), per_suit], axis=1)

    # Also extract `opp_suit_ts` — trump_score that OPP would play as trump, for opp80 scenarios.
    # The opp's bid suit is encoded in the bid history. Parse it.
    # Bid history is at obs[32:104], 12 rows × 6 features. Suit = obs[base+2..base+6] (one-hot).
    opp_ts_arr = np.zeros(len(features), dtype=np.int32)
    for scen_id in [6, 7, 8]:  # opp80
        mask = scenario_id == scen_id
        # In opp80 scenarios, the opp's bid suit is encoded in one of the history slots
        # (the last non-pass entry). Easier: just scan the bid history for a bid entry.
        # Actually the distill_bid placed the opp's bid as entry in prior_template — position
        # in history varies by scenario. Scan all 12 slots for a non-zero action marker.
        sub_hist = obs[mask, 32:104].reshape(-1, 12, 6)  # (n, 12, 6)
        # A bid has hist[i, 0] = 0.4 (see bid_obs.rs encode_bid_history).
        # Suit: index of max in hist[i, 2:6].
        for i_row in np.where(mask)[0]:
            hist = obs[i_row, 32:104].reshape(12, 6)
            for slot in range(12):
                if abs(hist[slot, 0] - 0.4) < 1e-3:
                    suit = int(hist[slot, 2:6].argmax())
                    opp_ts_arr[i_row] = int(ts_per_suit[i_row, suit])
                    break
    all_df["opp_trump_score"] = opp_ts_arr

    # Best non-opp-suit trump_score
    def compute_best_nonopp(row_idx, opp_ts, ts_row):
        if opp_ts <= 0:
            return int(ts_row.max())
        # filter out the suit with opp's ts
        mask = ts_row != opp_ts  # may match multiple if tied; OK
        return int(ts_row[mask].max()) if mask.any() else 0

    # Actually simpler: find opp suit index from obs hist, then ts of other 3
    opp_best_other = np.zeros(len(features), dtype=np.int32)
    opp_second_other = np.zeros(len(features), dtype=np.int32)
    for i_row in range(len(features)):
        s_id = scenario_id[i_row]
        if s_id not in [6, 7, 8]:
            continue
        hist = obs[i_row, 32:104].reshape(12, 6)
        opp_suit = -1
        for slot in range(12):
            if abs(hist[slot, 0] - 0.4) < 1e-3:
                opp_suit = int(hist[slot, 2:6].argmax())
                break
        if opp_suit >= 0:
            others = np.delete(ts_per_suit[i_row], opp_suit)
            others_sorted = np.sort(others)[::-1]
            opp_best_other[i_row] = int(others_sorted[0])
            opp_second_other[i_row] = int(others_sorted[1])
    all_df["opp_best_other_suit_ts"] = opp_best_other
    all_df["opp_second_other_suit_ts"] = opp_second_other

    # Per-suit J/9/count from engineered
    per_suit_J9_count = ["sS_has_J", "sH_has_J", "sD_has_J", "sC_has_J",
                        "sS_has_9", "sH_has_9", "sD_has_9", "sC_has_9",
                        "sS_count", "sH_count", "sD_count", "sC_count"]

    # Feature sets
    sets = {
        "A. base (17)": BASE,
        "B. +per_suit_J9_count (12)": BASE + per_suit_J9_count,
        "C. B+second_ts (3)": BASE + per_suit_J9_count + ["ts_2nd", "ts_gap", "n_suits_ge_14"],
        "D. C+opp_best_other (2)": BASE + per_suit_J9_count + ["ts_2nd", "ts_gap", "n_suits_ge_14"]
                                    + ["opp_best_other_suit_ts", "opp_second_other_suit_ts"],
        "E. D+ts_per_suit (4)": BASE + per_suit_J9_count + ["ts_2nd", "ts_gap", "n_suits_ge_14"]
                                + ["opp_best_other_suit_ts", "opp_second_other_suit_ts"]
                                + ["ts_best", "ts_3rd", "ts_4th", "ts_sum"],
    }

    SCEN_NAMES = {
        0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p",
        4: "pos3_partner80", 5: "pos4_partner80", 6: "pos2_opp80", 7: "pos3_opp80",
        8: "pos4_opp80",
    }
    print("=== XGB per-deal accuracy by feature set ===\n")
    header = f"{'scenario':<22}"
    for k in sets.keys():
        header += f" {k[:18]:>20}"
    print(header)

    importances_by_scenario = {}
    for scen_id, name in SCEN_NAMES.items():
        mask = scenario_id == scen_id
        if mask.sum() < 1000:
            continue
        y = nn_bids[mask]
        if y.mean() < 0.02 or y.mean() > 0.98:
            continue
        sub = all_df.iloc[mask.nonzero()[0]].reset_index(drop=True)

        row = f"{name:<22}"
        for label, cols in sets.items():
            acc, m = fit(sub[cols].values, y)
            row += f" {acc:>20.4f}"
            if label == "E. D+ts_per_suit (4)":
                importances_by_scenario[name] = top_imp(m, cols)
        print(row)

    print("\n=== Top features in set E per scenario (opp80 focus) ===")
    for name in ["pos2_opp80", "pos3_opp80", "pos4_opp80"]:
        if name not in importances_by_scenario:
            continue
        print(f"\n{name}:")
        for f, i in importances_by_scenario[name]:
            print(f"  {f:<32}  {i:.3f}")


if __name__ == "__main__":
    main()
