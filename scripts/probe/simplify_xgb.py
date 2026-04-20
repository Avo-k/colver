"""Simplify the XGBoost model — find the smallest (features, depth, n_trees) that
still hits ≥95% accuracy, then extract as human-readable rules.

Experiments:
  E1. Backward feature elimination (start with all, remove least important)
  E2. Depth × n_estimators sweep (depth 1-5, n_estimators 1, 5, 10, 50, 100, 300)
  E3. Single best depth-3 tree → if-else extraction
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, export_text
from xgboost import XGBClassifier

from bid_net_torch import load_bid_net
from nn_trump_score_vs_handcrafted import nn_trump_score, handcrafted_trump_score

ACT_PATH = "/tmp/probe_activations.npz"
MODEL_PATH = "models/bid_v5_isdd/bid_nn_final.bin"


def compute_features(hand_bits: np.ndarray, obs: np.ndarray) -> pd.DataFrame:
    """Rich feature set per deal (no context; scenario fixes position/scenario_id)."""
    N = len(hand_bits)

    # NN-native score per suit (4 suits)
    nn_scores = np.stack([nn_trump_score(hand_bits, s) for s in range(4)], axis=1)
    nn_scores_sorted = np.sort(nn_scores, axis=1)[:, ::-1]
    nn_best_suit = nn_scores.argmax(axis=1)

    # Hand-crafted score
    hc_scores = np.stack([handcrafted_trump_score(hand_bits, s) for s in range(4)], axis=1)
    hc_scores_sorted = np.sort(hc_scores, axis=1)[:, ::-1]

    # Trump count per suit (picked)
    tc_best = np.array([hand_bits[i, nn_best_suit[i]*8:nn_best_suit[i]*8+8].sum() for i in range(N)], dtype=np.int8)
    has_jack = np.array([hand_bits[i, nn_best_suit[i]*8 + 3] for i in range(N)], dtype=np.int8)
    has_nine = np.array([hand_bits[i, nn_best_suit[i]*8 + 2] for i in range(N)], dtype=np.int8)
    has_ace = np.array([hand_bits[i, nn_best_suit[i]*8 + 7] for i in range(N)], dtype=np.int8)

    # Side voids / singletons (w.r.t. best suit)
    n_voids = np.zeros(N, dtype=np.int8)
    n_singletons = np.zeros(N, dtype=np.int8)
    for i in range(N):
        t = nn_best_suit[i]
        for s in range(4):
            if s == t:
                continue
            c = hand_bits[i, s*8:s*8+8].sum()
            if c == 0:
                n_voids[i] += 1
            elif c == 1:
                n_singletons[i] += 1

    # Count of suits where I have J / 9 (0..4)
    n_J = np.stack([hand_bits[:, s*8 + 3] for s in range(4)], axis=1).sum(axis=1)
    n_9 = np.stack([hand_bits[:, s*8 + 2] for s in range(4)], axis=1).sum(axis=1)

    # Number of trump-worthy suits
    n_suits_ge_5 = (nn_scores >= 5).sum(axis=1)
    n_suits_ge_8 = (nn_scores >= 8).sum(axis=1)

    # Shape
    lengths = np.stack([hand_bits[:, s*8:s*8+8].sum(axis=1) for s in range(4)], axis=1)
    shape_sorted = np.sort(lengths, axis=1)[:, ::-1]

    df = pd.DataFrame({
        # NN-native score features
        "nn_best": nn_scores_sorted[:, 0],
        "nn_2nd": nn_scores_sorted[:, 1],
        "nn_3rd": nn_scores_sorted[:, 2],
        # Hand-crafted for comparison
        "hc_best": hc_scores_sorted[:, 0],
        "hc_2nd": hc_scores_sorted[:, 1],
        # Counts
        "tc_best": tc_best,
        "has_jack": has_jack,
        "has_nine": has_nine,
        "has_ace": has_ace,
        "n_voids": n_voids,
        "n_singletons": n_singletons,
        # Flexibility
        "n_J_in_hand": n_J.astype(np.int8),
        "n_9_in_hand": n_9.astype(np.int8),
        "n_suits_ge_5": n_suits_ge_5.astype(np.int8),
        "n_suits_ge_8": n_suits_ge_8.astype(np.int8),
        # Shape
        "shape_l1": shape_sorted[:, 0],
        "shape_l4": shape_sorted[:, 3],
    })
    return df


def fit_xgb(X, y, max_depth=5, n_estimators=300, seed=42):
    if y.mean() < 0.02 or y.mean() > 0.98:
        return None, float("nan")
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    m = XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth, learning_rate=0.1,
        scale_pos_weight=(yt == 0).sum() / max((yt == 1).sum(), 1),
        random_state=seed, verbosity=0, n_jobs=-1,
    )
    m.fit(Xt, yt)
    return m, accuracy_score(yv, m.predict(Xv))


def fit_tree(X, y, max_depth=3, min_leaf=300, seed=42):
    if y.mean() < 0.02 or y.mean() > 0.98:
        return None, float("nan")
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    m = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_leaf,
                                class_weight="balanced", random_state=seed)
    m.fit(Xt, yt)
    return m, accuracy_score(yv, m.predict(Xv))


def backward_eliminate(X_df, y, max_depth=5, n_est=300, min_features=3, verbose=True):
    """Iteratively remove least-important feature while keeping accuracy above floor."""
    features = list(X_df.columns)
    history = []
    m, acc_full = fit_xgb(X_df.values, y, max_depth, n_est)
    if m is None:
        return features, [], 0
    history.append((list(features), float(acc_full)))
    current_acc = acc_full
    while len(features) > min_features:
        imp = sorted(zip(features, m.feature_importances_), key=lambda t: t[1])
        worst_feat = imp[0][0]
        new_feats = [f for f in features if f != worst_feat]
        m_new, acc_new = fit_xgb(X_df[new_feats].values, y, max_depth, n_est)
        if m_new is None:
            break
        history.append((list(new_feats), float(acc_new)))
        if verbose:
            print(f"    remove '{worst_feat}' (imp={imp[0][1]:.3f}) → acc={acc_new:.4f}  (Δ={acc_new-current_acc:+.4f})")
        features = new_feats
        m = m_new
        current_acc = acc_new
    return features, history, acc_full


def main():
    d = np.load(ACT_PATH)
    obs = d["obs"]
    scenario_id = d["scenario_id"]
    nn_bids = d["nn_bids"]
    hand_bits = (obs[:, :32] > 0.5).astype(np.int8)

    print("Computing rich feature set...")
    X_df = compute_features(hand_bits, obs)
    print(f"  {len(X_df.columns)} features: {list(X_df.columns)}")

    SCEN = {0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p"}

    results = {}
    for scen_id, name in SCEN.items():
        mask = scenario_id == scen_id
        if mask.sum() < 1000:
            continue
        y = nn_bids[mask]
        X_sub = X_df.iloc[mask.nonzero()[0]].reset_index(drop=True)
        print(f"\n{'='*70}")
        print(f"  {name}  (n={mask.sum():,}, NN bid_rate={y.mean():.1%})")
        print(f"{'='*70}")

        # --- E1. Backward elimination (XGBoost depth=5, n_est=300) ---
        print(f"\n  E1. Backward feature elimination (XGB depth=5, n_est=300):")
        _, history, acc_full = backward_eliminate(X_sub, y, max_depth=5, n_est=300, min_features=3)

        # Find the smallest set that stays within 0.5pp of full
        floor = acc_full - 0.005
        smallest_set = history[0][0]
        smallest_acc = history[0][1]
        for feats, acc in history:
            if acc >= floor and len(feats) < len(smallest_set):
                smallest_set = feats
                smallest_acc = acc
        print(f"\n    → full ({len(history[0][0])} feats): {acc_full:.4f}")
        print(f"    → smallest set within 0.5pp floor ({len(smallest_set)} feats): "
              f"{smallest_acc:.4f}  feats={smallest_set}")

        # --- E2. Depth × n_estimators sweep ---
        print(f"\n  E2. Depth × n_estimators (using simplest useful feature set):")
        X_small = X_sub[smallest_set].values
        print(f"    {'d\\n_est':>10} {'1':>7} {'5':>7} {'10':>7} {'50':>7} {'100':>7} {'300':>7}")
        sweep_results = {}
        for depth in [1, 2, 3, 4, 5]:
            row = f"    depth={depth:<5}"
            sweep_results[depth] = {}
            for n_est in [1, 5, 10, 50, 100, 300]:
                _, a = fit_xgb(X_small, y, max_depth=depth, n_estimators=n_est)
                row += f" {a:>7.4f}"
                sweep_results[depth][n_est] = float(a) if not np.isnan(a) else None
            print(row)

        # --- E3. Single tree depth-3/4/5 as human-readable ---
        print(f"\n  E3. Single tree at different depths:")
        best_tree = None
        best_tree_acc = 0
        for depth in [2, 3, 4]:
            t, a = fit_tree(X_small, y, max_depth=depth, min_leaf=500)
            print(f"    tree depth={depth}: acc={a:.4f}")
            if a > best_tree_acc and depth <= 3:
                best_tree_acc = a
                best_tree = (t, depth)
        if best_tree is not None:
            tree, depth = best_tree
            print(f"\n    Best depth-{depth} tree (acc={best_tree_acc:.4f}):")
            print(export_text(tree, feature_names=smallest_set, max_depth=depth))

        results[name] = {
            "acc_full": float(acc_full),
            "smallest_set": smallest_set,
            "smallest_acc": float(smallest_acc),
            "sweep": sweep_results,
            "best_tree_depth": best_tree[1] if best_tree else None,
            "best_tree_acc": float(best_tree_acc),
        }

    with open("/tmp/xgb_simplify.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[saved] /tmp/xgb_simplify.json")

    # Final summary table
    print("\n\n=== FINAL ACCURACY TABLE ===")
    print(f"{'scenario':<22} {'full XGB':>10} {'minimal':>10} {'depth=2 XGB 50':>16} {'depth=2 XGB 10':>16} {'single tree d=3':>17}")
    for name, r in results.items():
        d2_50 = r["sweep"][2][50] if r["sweep"].get(2) else None
        d2_10 = r["sweep"][2][10] if r["sweep"].get(2) else None
        print(f"{name:<22} {r['acc_full']:>10.4f} {r['smallest_acc']:>10.4f} "
              f"{d2_50 if d2_50 else 'nan':>16}   {d2_10 if d2_10 else 'nan':>16}   {r['best_tree_acc']:>17.4f}")


if __name__ == "__main__":
    main()
