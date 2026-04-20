#!/usr/bin/env python3
"""
Feature discovery: can we crack 90% by engineering new features?

Approach:
  1. Baseline = XGBoost on the 17-feature BASE set (what distill_bid exports).
  2. Expanded = baseline + engineered features (interactions, side-suit granularity).
  3. Q-signal = expanded + features derived from the NN's own Q-values.
  4. For each setup, measure per-deal accuracy on each scenario group.
  5. Look at XGB importances to identify which new features actually help.

Usage: uv run python scripts/analysis/bid_v5_feature_discovery.py
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

CSV = "data/distill/bid_v5_distill.csv"
SAMPLE = "/tmp/bid_v5_sampled.csv"


BASE_FEATURES = [
    "trump_count", "has_jack", "has_nine", "has_ace", "has_ten",
    "has_king", "has_queen", "trump_points", "trump_score", "has_belote",
    "side_aces", "side_tens", "side_voids", "side_singletons",
    "side_doubletons", "total_aces", "best_side_length",
]

# Context features (present only in response scenarios)
RESPONSE_FEATURES = ["partner_support", "is_partner_suit", "opp_suit_cards", "is_opp_suit"]


def presample():
    """Keep every 10th deal (4 rows/deal) for speed."""
    if Path(SAMPLE).exists():
        return
    print(f"Pre-sampling {CSV} -> {SAMPLE} (every 10th deal, ~720k rows)...")
    subprocess.run(
        f"awk 'NR==1 || ((NR-2)%40 < 4)' '{CSV}' > '{SAMPLE}'",
        shell=True, check=True,
    )


def build_deal_df(gdf: pd.DataFrame) -> pd.DataFrame:
    """Per-deal view: best suit by q_80 + second-best features."""
    s = gdf.sort_values(["deal_id", "q_80"], ascending=[True, False]).reset_index(drop=True)
    n = len(s)
    best_mask = np.zeros(n, dtype=bool)
    second_mask = np.zeros(n, dtype=bool)
    for i in range(0, n, 4):
        best_mask[i] = True
        if i + 1 < n:
            second_mask[i + 1] = True
    best = s[best_mask].reset_index(drop=True)
    second = s[second_mask].reset_index(drop=True)

    best["second_trump_score"] = second["trump_score"].values
    best["second_trump_count"] = second["trump_count"].values
    best["second_has_jack"] = second["has_jack"].values
    best["nn_bids"] = ((best["nn_action"] >= 1) & (best["nn_action"] <= 40)).astype(int)
    return best


def engineer_interactions(df: pd.DataFrame) -> pd.DataFrame:
    """Explicit interaction & combined features."""
    df = df.copy()
    # Interactions
    df["jack_x_count"] = df["has_jack"] * df["trump_count"]
    df["nine_x_count"] = df["has_nine"] * df["trump_count"]
    df["ts_x_voids"] = df["trump_score"] * df["side_voids"]
    df["ts_minus_aces"] = df["trump_score"] - 3 * df["side_aces"]
    # "Real trump strength" — remove penalty of ace-of-trump
    df["trump_strength_noace"] = df["trump_score"] - 2 * df["has_ace"]
    # Length premium
    df["length_bonus"] = np.maximum(0, df["trump_count"] - 2) ** 2
    # J-in-partner-suit & is_partner_suit
    if "is_partner_suit" in df.columns:
        df["jack_in_partner_suit"] = df["has_jack"] * df["is_partner_suit"]
        df["support_x_score"] = df["partner_support"].clip(lower=0) * df["trump_score"]
    if "is_opp_suit" in df.columns:
        df["opp_cards_x_ts"] = df["opp_suit_cards"].clip(lower=0) * (30 - df["trump_score"])
        df["coinche_signal"] = df["opp_suit_cards"].clip(lower=0) * (df["trump_count"] <= 2).astype(int)
    # Second suit gap
    df["gap_best_second"] = df["trump_score"] - df["second_trump_score"]
    return df


def engineer_qvalues(df: pd.DataFrame) -> pd.DataFrame:
    """Features derived from the NN's own Q-values — this is the NN speaking about itself."""
    df = df.copy()
    df["q_gap_80_pass"] = df["q_80"] - df["q_pass"]
    df["q_gap_90_80"] = df["q_90"] - df["q_80"]
    df["q_gap_100_90"] = df["q_100"] - df["q_90"]
    df["q_max_bid"] = df[["q_80", "q_90", "q_100", "q_110", "q_120"]].max(axis=1)
    df["q_bid_vs_pass"] = df["q_max_bid"] - df["q_pass"]
    # Replace -inf (capot/coinche) with NaN then fill
    for c in ["q_80", "q_90", "q_100", "q_110", "q_120", "q_capot", "q_pass", "q_coinche",
              "q_gap_80_pass", "q_gap_90_80", "q_gap_100_90", "q_max_bid", "q_bid_vs_pass"]:
        df[c] = df[c].replace([-np.inf, np.inf], np.nan).fillna(-10.0)
    return df


def fit_xgb(X, y, label):
    from xgboost import XGBClassifier
    if y.mean() < 0.01 or y.mean() > 0.99:
        return None, 0.0
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=((yt == 0).sum() / max((yt == 1).sum(), 1)),
        random_state=42, verbosity=0, n_jobs=-1,
    )
    m.fit(Xt, yt)
    acc = accuracy_score(yv, m.predict(Xv))
    return m, acc


def importance_top(m, feats, k=15):
    imp = sorted(zip(feats, m.feature_importances_), key=lambda x: -x[1])
    return [(f, float(i)) for f, i in imp[:k]]


def analyze(scen_filter: list[str], label: str, include_response: bool):
    df = pd.read_csv(SAMPLE)
    gdf = df[df["scenario"].isin(scen_filter)].copy()
    if len(gdf) == 0:
        print(f"[{label}] empty"); return
    deal = build_deal_df(gdf)

    base_feats = BASE_FEATURES + ["second_trump_score", "second_trump_count", "second_has_jack"]
    if include_response:
        base_feats += RESPONSE_FEATURES
    base_feats = [f for f in base_feats if f in deal.columns]

    y = deal["nn_bids"].values

    print(f"\n{'='*72}\n  {label}\n  n={len(deal):,}, NN bid_rate={y.mean():.1%}")
    print(f"{'='*72}")

    # ------- 1. Baseline -------
    X = deal[base_feats].values
    m1, acc1 = fit_xgb(X, y, "baseline")
    print(f"  [1] Baseline (n_feats={len(base_feats)}):           acc={acc1:.4f}")

    # ------- 2. + Interactions -------
    deal_i = engineer_interactions(deal)
    int_feats = [c for c in deal_i.columns if c not in deal.columns]
    X = deal_i[base_feats + int_feats].values
    m2, acc2 = fit_xgb(X, y, "+interactions")
    print(f"  [2] + interactions  (+{len(int_feats)} feats):      acc={acc2:.4f}  (Δ={acc2-acc1:+.4f})")
    if m2:
        print("      Top new features:")
        for f, i in importance_top(m2, base_feats + int_feats, k=20):
            if f in int_feats and i > 0.005:
                print(f"        {f:<25} {i:.3f}")

    # ------- 3. + Q-values -------
    deal_q = engineer_qvalues(deal_i)
    q_feats = ["q_80", "q_90", "q_100", "q_pass", "q_gap_80_pass", "q_gap_90_80",
               "q_max_bid", "q_bid_vs_pass"]
    q_feats = [f for f in q_feats if f in deal_q.columns]
    X = deal_q[base_feats + int_feats + q_feats].values
    m3, acc3 = fit_xgb(X, y, "+q_values")
    print(f"  [3] + Q-values      (+{len(q_feats)} feats):         acc={acc3:.4f}  (Δ={acc3-acc1:+.4f})")
    if m3:
        print("      Top Q-derived features:")
        for f, i in importance_top(m3, base_feats + int_feats + q_feats, k=25):
            if f in q_feats and i > 0.005:
                print(f"        {f:<25} {i:.3f}")

    return {"label": label, "baseline": acc1, "interactions": acc2, "qvalues": acc3}


def main():
    presample()

    results = []
    results.append(analyze(["pos1_open"], "OPENING (pos1)", include_response=False))
    results.append(analyze(["pos2_after_pass"], "POS 2 (after 1 pass)", include_response=False))
    results.append(analyze(["pos3_after_2p"], "POS 3 (after 2 passes)", include_response=False))
    results.append(analyze(["pos4_after_3p"], "POS 4 (after 3 passes)", include_response=False))
    partner_scen = [f"pos3_partner80_{s}" for s in "shdc"] + [f"pos4_partner80_{s}" for s in "shdc"]
    results.append(analyze(partner_scen, "PARTNER 80", include_response=True))
    opp_scen = (
        [f"pos2_opp80_{s}" for s in "shdc"] +
        [f"pos3_opp80_{s}" for s in "shdc"] +
        [f"pos4_opp80_{s}" for s in "shdc"]
    )
    results.append(analyze(opp_scen, "OPP 80", include_response=True))

    print("\n\n=== FINAL SUMMARY ===")
    print(f"{'Scenario':<28}  {'base':>8} {'+inter':>8} {'+qvals':>8} {'Δ total':>9}")
    print("-" * 70)
    for r in results:
        if r:
            delta = r["qvalues"] - r["baseline"]
            print(f"{r['label']:<28}  {r['baseline']:>8.4f} {r['interactions']:>8.4f} {r['qvalues']:>8.4f} {delta:>+9.4f}")


if __name__ == "__main__":
    main()
