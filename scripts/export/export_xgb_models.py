#!/usr/bin/env python3
"""
Train XGBoost models per scenario group and export tree structures as JSON
for browser-side interpretability (Saabas path-based feature attribution).

Produces a single JSON file with all models:
  python/colver/web/static/data/xgb_models.json

Each model contains:
  - trees: list of decision trees (nodes with feature, threshold, left, right, value, cover)
  - base_score: the model's bias term
  - features: list of feature names
  - type: "per_suit" or "per_deal"
  - scenario: group name

Usage:
    # 1. Generate distillation CSV (if not already done):
    cargo run -p colver-core --bin distill_bid --release

    # 2. Export XGBoost models:
    PYTHONPATH=scripts uv run python scripts/export_xgb_models.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

BASE_FEATURES = [
    "trump_count", "has_jack", "has_nine", "has_ace", "has_ten",
    "has_king", "has_queen", "trump_points", "trump_score", "has_belote",
    "side_aces", "side_tens", "side_voids", "side_singletons",
    "side_doubletons", "total_aces", "best_side_length",
]

RESPONSE_FEATURES = BASE_FEATURES + [
    "partner_support", "is_partner_suit", "opp_suit_cards", "is_opp_suit",
]

DEAL_EXTRA_FEATURES = ["second_trump_score", "second_trump_count"]


def build_deal_df(gdf):
    """Build one row per deal from the 4-suits-per-deal data."""
    gdf_sorted = gdf.sort_values(
        ["deal_id", "q_80"], ascending=[True, False]
    ).reset_index(drop=True)
    n = len(gdf_sorted)
    best_mask = np.zeros(n, dtype=bool)
    second_mask = np.zeros(n, dtype=bool)
    for i in range(0, n, 4):
        best_mask[i] = True
        if i + 1 < n:
            second_mask[i + 1] = True

    best = gdf_sorted[best_mask].copy().reset_index(drop=True)
    second = gdf_sorted[second_mask].reset_index(drop=True)
    best["second_trump_score"] = second["trump_score"].values
    best["second_trump_count"] = second["trump_count"].values
    best["nn_bids"] = (
        (best["nn_action"] >= 1) & (best["nn_action"] <= 40)
    ).astype(int)
    return best


def export_all_trees(booster):
    """Convert all XGBoost trees to nested dict structures.

    Returns a list of trees where each node is:
      Internal: {f: feature_idx, t: threshold, l: left, r: right, v: node_value, c: cover}
      Leaf:     {v: leaf_value, c: cover}

    node_value is the cover-weighted mean prediction of all leaves in the subtree.
    """
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
            "l": left,
            "r": right,
            "v": round(node_val, 6),
            "c": round(total_c, 1),
        }

    return [parse_node(json.loads(t)) for t in dump]


def train_and_export(X, y, features, label):
    """Train XGBoost and return exportable model dict."""
    from xgboost import XGBClassifier

    if y.mean() < 0.005 or y.mean() > 0.995:
        print(f"  [{label}] Skipping — extreme class balance ({y.mean():.3f})")
        return None

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    scale = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    xgb = XGBClassifier(
        n_estimators=100,
        max_depth=4,
        learning_rate=0.15,
        scale_pos_weight=scale,
        random_state=42,
        verbosity=0,
        n_jobs=-1,
    )
    xgb.fit(X_train, y_train)

    y_pred = xgb.predict(X_test)
    from sklearn.metrics import accuracy_score
    acc = accuracy_score(y_test, y_pred)
    print(f"  [{label}] accuracy={acc:.3f}, bid_rate={y.mean():.3f}, n={len(y)}")

    # Export trees
    booster = xgb.get_booster()
    trees = export_all_trees(booster)

    # Base score (log-odds of the intercept)
    import math
    base_score = xgb.get_params().get("base_score", None)
    if base_score is None:
        base_score = 0.5
    base_score = float(base_score)
    base_logit = math.log(base_score / (1 - base_score)) if 0 < base_score < 1 else 0.0

    return {
        "trees": trees,
        "base_score": round(base_logit, 6),
        "features": features,
        "accuracy": round(acc, 3),
        "bid_rate": round(float(y.mean()), 3),
        "n_samples": int(len(y)),
    }


def main():
    csv_path = sys.argv[1] if len(sys.argv) > 1 else "data/bid_distill.csv"
    if not Path(csv_path).exists():
        print(f"Error: {csv_path} not found.")
        print("Generate: cargo run -p colver-core --bin distill_bid --release")
        sys.exit(1)

    # Pre-sample with awk for speed (rows come in groups of 4 per deal)
    import subprocess, tempfile
    needed_cols = (
        ["deal_id", "scenario", "nn_action", "nn_bids_this_suit", "q_80"]
        + BASE_FEATURES + ["partner_support", "is_partner_suit", "opp_suit_cards", "is_opp_suit"]
    )
    sampled_path = Path("/tmp/bid_distill_sampled.csv")
    print(f"Pre-sampling {csv_path} with awk (every 10th deal)...")
    # Keep header + every 10th group of 4 rows
    subprocess.run(
        f"awk 'NR==1 || ((NR-2)%40 < 4)' '{csv_path}' > '{sampled_path}'",
        shell=True, check=True,
    )
    sample_size = sampled_path.stat().st_size / (1024 * 1024)
    print(f"Sampled file: {sample_size:.0f} MB")
    df = pd.read_csv(sampled_path, usecols=needed_cols)
    sampled_path.unlink(missing_ok=True)
    print(f"Loaded {len(df)} rows")

    groups = [
        ("opening", "Opening (pos 1)",
         ["pos1_open"], BASE_FEATURES),
        ("pos2_pass", "After 1 pass (pos 2)",
         ["pos2_after_pass"], BASE_FEATURES),
        ("pos3_pass", "After 2 passes (pos 3)",
         ["pos3_after_2p"], BASE_FEATURES),
        ("pos4_pass", "After 3 passes (pos 4)",
         ["pos4_after_3p"], BASE_FEATURES),
        ("partner80", "Partner bid 80",
         [f"pos3_partner80_{s}" for s in "shdc"] +
         [f"pos4_partner80_{s}" for s in "shdc"],
         RESPONSE_FEATURES),
        ("opp80", "Opponent bid 80",
         [f"pos2_opp80_{s}" for s in "shdc"] +
         [f"pos3_opp80_{s}" for s in "shdc"] +
         [f"pos4_opp80_{s}" for s in "shdc"],
         RESPONSE_FEATURES),
    ]

    models = {}

    for gname, title, scens, feats in groups:
        print(f"\n=== {title} ({gname}) ===")
        mask = df["scenario"].isin(scens)
        gdf = df[mask]
        if len(gdf) == 0:
            print(f"  No data for {gname}")
            continue

        valid_features = [f for f in feats if f in gdf.columns]

        # Per-suit model: "will NN bid THIS suit?"
        X = gdf[valid_features].values
        y = gdf["nn_bids_this_suit"].values
        result = train_and_export(X, y, valid_features, f"{gname}/per_suit")
        if result:
            result["type"] = "per_suit"
            result["title"] = title
            models[f"{gname}_suit"] = result

        # Per-deal model: "will NN bid at all?" (best suit features)
        deal_df = build_deal_df(gdf)
        deal_feats = valid_features + [
            f for f in DEAL_EXTRA_FEATURES if f not in valid_features
        ]
        if "partner_support" in deal_df.columns:
            deal_feats = [f for f in deal_feats if f in deal_df.columns]
            for extra in ["partner_support", "is_partner_suit", "opp_suit_cards", "is_opp_suit"]:
                if extra in deal_df.columns and extra not in deal_feats:
                    deal_feats.append(extra)

        deal_feats = [f for f in deal_feats if f in deal_df.columns]
        X_deal = deal_df[deal_feats].values
        y_deal = deal_df["nn_bids"].values
        result = train_and_export(X_deal, y_deal, deal_feats, f"{gname}/per_deal")
        if result:
            result["type"] = "per_deal"
            result["title"] = title
            models[f"{gname}_deal"] = result

    # Save
    out_dir = Path("python/colver/web/static/data")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "xgb_models.json"

    # Compute total size
    raw = json.dumps(models)
    print(f"\nRaw JSON size: {len(raw) / 1024:.0f} KB")

    with open(out_path, "w") as f:
        json.dump(models, f, separators=(",", ":"))

    compressed_size = out_path.stat().st_size
    print(f"Saved: {out_path} ({compressed_size / 1024:.0f} KB)")
    print(f"Models: {list(models.keys())}")


if __name__ == "__main__":
    main()
