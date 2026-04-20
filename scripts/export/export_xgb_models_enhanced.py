#!/usr/bin/env python3
"""Enhanced XGBoost export with features discovered via hidden-layer probe.

Adds to the base 17 features:
  - second_trump_score, second_trump_count (per-deal, 2nd-best suit)
  - third_trump_score, fourth_trump_score (full shape)
  - n_suits_ge_14 (how many decent trump candidates)
  - s{S,H,D,C}_has_J / _has_9 (per-suit J/9 flags)
  - s{S,H,D,C}_count (per-suit length)
  - opp_best_other_ts (MAX trump_score excluding opp's suit — for opp80 scenarios)

Writes to: python/colver/web/static/data/xgb_models_enhanced.json
(Does NOT overwrite the live xgb_models.json — frontend JS would need updating.)

Usage:
    PYTHONPATH=scripts/probe uv run python scripts/export/export_xgb_models_enhanced.py \\
        data/distill/bid_v5_distill.csv
"""
from __future__ import annotations

import json
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


BASE = [
    "trump_count", "has_jack", "has_nine", "has_ace", "has_ten",
    "has_king", "has_queen", "trump_points", "trump_score", "has_belote",
    "side_aces", "side_tens", "side_voids", "side_singletons",
    "side_doubletons", "total_aces", "best_side_length",
]

RESPONSE = BASE + ["partner_support", "is_partner_suit", "opp_suit_cards", "is_opp_suit"]

# New per-deal enhancements
DEAL_EXTRA = [
    "second_trump_score", "second_trump_count",
    "third_trump_score", "fourth_trump_score",
    "n_suits_ge_14", "n_suits_ge_11",
]


def build_deal_df(gdf: pd.DataFrame, include_opp_best_other: bool = False) -> pd.DataFrame:
    """Per-deal view with rich alt-suit features.

    Each deal has 4 rows (one per suit). We sort by q_80 desc and take #1 as "best"
    (the NN's preferred suit). Then compute features from all 4 rows.
    """
    gdf_sorted = gdf.sort_values(["deal_id", "q_80"], ascending=[True, False]).reset_index(drop=True)
    n = len(gdf_sorted)

    # Reshape to (n_deals, 4, n_cols). Safer: use grouping.
    ts_per_deal = gdf.groupby("deal_id")["trump_score"].apply(list)
    tc_per_deal = gdf.groupby("deal_id")["trump_count"].apply(list)
    jack_per_deal = gdf.groupby("deal_id")["has_jack"].apply(list)
    nine_per_deal = gdf.groupby("deal_id")["has_nine"].apply(list)
    count_per_deal = gdf.groupby("deal_id")["trump_count"].apply(list)
    suit_per_deal = gdf.groupby("deal_id")["suit"].apply(list)
    if include_opp_best_other:
        opp_per_deal = gdf.groupby("deal_id")["is_opp_suit"].apply(list)

    # Mask: in-sorted order, every 4th row is the "best" per deal
    best_mask = np.zeros(n, dtype=bool)
    for i in range(0, n, 4):
        best_mask[i] = True
    best = gdf_sorted[best_mask].copy().reset_index(drop=True)

    # Compute per-deal extras
    ts_sorted = np.array([sorted(v, reverse=True) for v in ts_per_deal.values])  # (n_deals, 4)
    tc_sorted = np.array([[tc_per_deal[d][sorted(range(4), key=lambda k: -ts_per_deal[d][k])[r]] for r in range(4)] for d in ts_per_deal.index])

    best["second_trump_score"] = ts_sorted[:, 1]
    best["third_trump_score"] = ts_sorted[:, 2]
    best["fourth_trump_score"] = ts_sorted[:, 3]
    best["second_trump_count"] = tc_sorted[:, 1]
    best["n_suits_ge_14"] = (ts_sorted >= 14).sum(axis=1)
    best["n_suits_ge_11"] = (ts_sorted >= 11).sum(axis=1)

    # Per-suit J / 9 / count (by physical suit index 0-3)
    # The CSV has suit column per row. We need to attribute J/9/count back to suits.
    def suit_feature_per_deal(per_deal: pd.Series, suits: pd.Series, idx: int):
        # For each deal, find the row with suit==idx, return its value
        out = np.zeros(len(per_deal), dtype=np.int8)
        for i, (vals, ss) in enumerate(zip(per_deal.values, suits.values)):
            try:
                pos = ss.index(idx)
                out[i] = int(vals[pos])
            except (ValueError, IndexError):
                out[i] = 0
        return out

    for s_idx, s_name in enumerate(["S", "H", "D", "C"]):
        best[f"s{s_name}_has_J"] = suit_feature_per_deal(jack_per_deal, suit_per_deal, s_idx)
        best[f"s{s_name}_has_9"] = suit_feature_per_deal(nine_per_deal, suit_per_deal, s_idx)
        best[f"s{s_name}_count"] = suit_feature_per_deal(count_per_deal, suit_per_deal, s_idx)

    # opp_best_other_ts: for opp80 scenarios only, max ts among rows where is_opp_suit==0
    if include_opp_best_other:
        opp_best_other = np.zeros(len(best), dtype=np.int32)
        opp_second_other = np.zeros(len(best), dtype=np.int32)
        for i, (tss, ops) in enumerate(zip(ts_per_deal.values, opp_per_deal.values)):
            non_opp_ts = [tss[k] for k in range(4) if ops[k] == 0]
            if len(non_opp_ts) >= 1:
                sorted_other = sorted(non_opp_ts, reverse=True)
                opp_best_other[i] = sorted_other[0]
                opp_second_other[i] = sorted_other[1] if len(sorted_other) >= 2 else 0
            else:
                opp_best_other[i] = best.iloc[i]["trump_score"]
                opp_second_other[i] = 0
        best["opp_best_other_ts"] = opp_best_other
        best["opp_second_other_ts"] = opp_second_other

    best["nn_bids"] = ((best["nn_action"] >= 1) & (best["nn_action"] <= 40)).astype(int)
    return best


def export_all_trees(booster):
    dump = booster.get_dump(dump_format="json", with_stats=True)

    def parse_node(node):
        if "leaf" in node:
            return {"v": round(node["leaf"], 6), "c": round(node.get("cover", 0), 1)}
        left = parse_node(node["children"][0])
        right = parse_node(node["children"][1])
        lc, rc = left["c"], right["c"]
        total_c = lc + rc
        node_val = (left["v"] * lc + right["v"] * rc) / total_c if total_c > 0 else 0
        return {
            "f": int(node["split"].replace("f", "")),
            "t": round(node["split_condition"], 6),
            "l": left, "r": right,
            "v": round(node_val, 6), "c": round(total_c, 1),
        }

    return [parse_node(json.loads(t)) for t in dump]


def train_and_export(X, y, feature_names, label):
    from xgboost import XGBClassifier
    if y.mean() < 0.005 or y.mean() > 0.995:
        print(f"  [{label}] skip (class balance {y.mean():.3f})")
        return None
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scale = (yt == 0).sum() / max((yt == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=100, max_depth=4, learning_rate=0.15,
        scale_pos_weight=scale, random_state=42, verbosity=0, n_jobs=-1,
    )
    xgb.fit(Xt, yt)
    acc = accuracy_score(yv, xgb.predict(Xv))
    print(f"  [{label}] acc={acc:.4f}  bid_rate={y.mean():.3f}  n={len(y):,}  n_feat={len(feature_names)}")
    booster = xgb.get_booster()
    trees = export_all_trees(booster)
    base_score = xgb.get_params().get("base_score", 0.5) or 0.5
    base_score = float(base_score)
    base_logit = math.log(base_score / (1 - base_score)) if 0 < base_score < 1 else 0.0
    return {
        "trees": trees, "base_score": round(base_logit, 6),
        "features": feature_names, "accuracy": round(acc, 3),
        "bid_rate": round(float(y.mean()), 3), "n_samples": int(len(y)),
    }


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/distill/bid_v5_distill.csv"
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    needed_cols = ["deal_id", "scenario", "suit", "nn_action", "nn_bids_this_suit", "q_80"] + RESPONSE
    sampled_path = Path("/tmp/bid_v5_enhanced_sampled.csv")
    print(f"Pre-sampling {csv_path} → {sampled_path} (every 10th deal)...")
    subprocess.run(
        f"awk 'NR==1 || ((NR-2)%40 < 4)' '{csv_path}' > '{sampled_path}'",
        shell=True, check=True,
    )
    df = pd.read_csv(sampled_path, usecols=needed_cols)
    sampled_path.unlink(missing_ok=True)
    print(f"Loaded {len(df):,} rows")

    groups = [
        ("opening", "Opening (pos 1)", ["pos1_open"], False),
        ("pos2_pass", "After 1 pass (pos 2)", ["pos2_after_pass"], False),
        ("pos3_pass", "After 2 passes (pos 3)", ["pos3_after_2p"], False),
        ("pos4_pass", "After 3 passes (pos 4)", ["pos4_after_3p"], False),
        ("partner80", "Partner bid 80",
         [f"pos3_partner80_{s}" for s in "shdc"] + [f"pos4_partner80_{s}" for s in "shdc"], False),
        ("opp80", "Opponent bid 80",
         [f"pos2_opp80_{s}" for s in "shdc"] + [f"pos3_opp80_{s}" for s in "shdc"]
         + [f"pos4_opp80_{s}" for s in "shdc"], True),
    ]

    per_suit_cols = [f"s{s}_has_J" for s in "SHDC"] + [f"s{s}_has_9" for s in "SHDC"] + [f"s{s}_count" for s in "SHDC"]

    models = {}
    for gname, title, scens, include_opp in groups:
        print(f"\n=== {title} ({gname}) ===")
        mask = df["scenario"].isin(scens)
        gdf = df[mask]
        if len(gdf) == 0:
            continue
        # Keep only deals that have exactly 4 rows
        counts = gdf.groupby("deal_id").size()
        full_deals = counts[counts == 4].index
        gdf = gdf[gdf["deal_id"].isin(full_deals)]
        if len(gdf) == 0:
            continue
        deal_df = build_deal_df(gdf, include_opp_best_other=include_opp)

        # Feature set depends on scenario
        feat_base = BASE + DEAL_EXTRA + per_suit_cols
        if any("partner" in s or "opp" in s for s in scens):
            feat_base = RESPONSE + DEAL_EXTRA + per_suit_cols
        if include_opp:
            feat_base = feat_base + ["opp_best_other_ts", "opp_second_other_ts"]
        feat = [f for f in feat_base if f in deal_df.columns]

        X_deal = deal_df[feat].values
        y_deal = deal_df["nn_bids"].values
        res = train_and_export(X_deal, y_deal, feat, f"{gname}/per_deal_enhanced")
        if res:
            res["type"] = "per_deal_enhanced"
            res["title"] = title
            models[f"{gname}_deal_enhanced"] = res

    out_path = Path("python/colver/web/static/data/xgb_models_enhanced.json")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    raw = json.dumps(models)
    print(f"\nRaw size: {len(raw)/1024:.0f} KB")
    with open(out_path, "w") as f:
        json.dump(models, f, separators=(",", ":"))
    print(f"Saved: {out_path} ({out_path.stat().st_size/1024:.0f} KB)")


if __name__ == "__main__":
    main()
