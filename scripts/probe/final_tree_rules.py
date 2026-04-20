"""Extract the FINAL ultra-simple tree rules (depth-3 single tree) per scenario.

Gives human-readable if-else chains that hit 91-93% agreement with the NN,
using only 3-4 features per scenario.
"""
from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text

from simplify_xgb import compute_features

ACT_PATH = "/tmp/probe_activations.npz"


def tree_to_if_else(tree, feature_names, indent=0):
    """Convert sklearn tree to readable if-else chain."""
    tree_ = tree.tree_

    def recurse(node, depth):
        # Leaf: children_left == children_right == -1
        if tree_.children_left[node] == -1:
            samples = tree_.value[node][0]
            cls = int(np.argmax(samples))
            n = int(samples.sum())
            label = "ANNONCE" if cls == 1 else "PASSE"
            pct = samples[1] / max(n, 1) * 100
            return ["  " * depth + f"→ {label}  (n={n}, {pct:.0f}% annoncent)"]
        feat_name = feature_names[tree_.feature[node]]
        thr = tree_.threshold[node]
        lines = []
        lines.append("  " * depth + f"si {feat_name} ≤ {thr:.1f}:")
        lines.extend(recurse(tree_.children_left[node], depth + 1))
        lines.append("  " * depth + f"sinon ({feat_name} > {thr:.1f}):")
        lines.extend(recurse(tree_.children_right[node], depth + 1))
        return lines

    return "\n".join(recurse(0, indent))


def main():
    d = np.load(ACT_PATH)
    obs = d["obs"]
    scenario_id = d["scenario_id"]
    nn_bids = d["nn_bids"]
    hand_bits = (obs[:, :32] > 0.5).astype(np.int8)

    print("Computing feature set...")
    X_df = compute_features(hand_bits, obs)

    # Use only the top-5 most reliable features
    MINIMAL = ["nn_best", "hc_best", "tc_best", "has_jack", "n_voids"]
    for col in MINIMAL:
        assert col in X_df.columns, f"{col} missing"

    SCEN = {0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p"}

    print("\n" + "="*72)
    print("  FINAL RULES : arbre depth-3 par scénario (5 features)")
    print("="*72)
    print(f"  Features utilisées : {MINIMAL}\n")

    for scen_id, name in SCEN.items():
        mask = scenario_id == scen_id
        if mask.sum() < 1000:
            continue
        y = nn_bids[mask]
        X = X_df.loc[mask.nonzero()[0], MINIMAL].values

        Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        tree = DecisionTreeClassifier(
            max_depth=3, min_samples_leaf=500,
            class_weight="balanced", random_state=42,
        )
        tree.fit(Xt, yt)
        acc = accuracy_score(yv, tree.predict(Xv))

        # Also try depth 4 for comparison
        tree4 = DecisionTreeClassifier(
            max_depth=4, min_samples_leaf=300,
            class_weight="balanced", random_state=42,
        )
        tree4.fit(Xt, yt)
        acc4 = accuracy_score(yv, tree4.predict(Xv))

        print(f"\n--- {name.upper()}  (NN bid rate = {y.mean()*100:.0f}%) ---")
        print(f"    depth-3: acc = {acc*100:.1f}%")
        print(f"    depth-4: acc = {acc4*100:.1f}%\n")

        print("  Règle depth-3 :")
        print(tree_to_if_else(tree, MINIMAL, indent=1))
        print()


if __name__ == "__main__":
    main()
