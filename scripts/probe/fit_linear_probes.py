"""Linear probes on hidden activations.

For each scenario group, fit a logistic regression on each of 3 hidden layers
to predict `nn_bids` from the 512 activations. Compare accuracy vs
baseline (features only). Identify top neurons per layer.

Outputs:
  /tmp/probe_results.json — accuracy table + top neuron IDs
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split

ACT_PATH = "/tmp/probe_activations.npz"

SCEN_NAMES = {
    0: "pos1_open",
    1: "pos2_after_pass",
    2: "pos3_after_2p",
    3: "pos4_after_3p",
    4: "pos3_partner80",
    5: "pos4_partner80",
    6: "pos2_opp80",
    7: "pos3_opp80",
    8: "pos4_opp80",
}

BASE_FEATURE_NAMES = [
    "trump_count", "has_jack", "has_nine", "has_ace", "has_ten",
    "has_king", "has_queen", "trump_points", "trump_score", "has_belote",
    "side_aces", "side_tens", "side_voids", "side_singletons",
    "side_doubletons", "total_aces", "best_side_length",
]


def fit(X, y, label, C=1.0):
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = LogisticRegression(max_iter=2000, C=C, solver="lbfgs", n_jobs=-1)
    m.fit(Xt, yt)
    yp = m.predict(Xv)
    yp_proba = m.predict_proba(Xv)[:, 1]
    acc = accuracy_score(yv, yp)
    try:
        ll = log_loss(yv, yp_proba)
    except Exception:
        ll = float("nan")
    return m, acc, ll


def run_scenario(scen_id: int, acts_by_layer: list, features: np.ndarray, nn_bids: np.ndarray,
                 scen_mask: np.ndarray, results: dict):
    scen_name = SCEN_NAMES[scen_id]
    n = scen_mask.sum()
    y = nn_bids[scen_mask]
    if n < 1000 or y.mean() < 0.02 or y.mean() > 0.98:
        print(f"  {scen_name}: skip (n={n}, bid_rate={y.mean():.1%})")
        return

    print(f"\n=== {scen_name} (n={n:,}, bid_rate={y.mean():.1%}) ===")
    entry = {"n": int(n), "bid_rate": float(y.mean()), "layers": {}}

    # Baseline: features only
    t0 = time.time()
    X_feat = features[scen_mask]
    _, acc_feat, _ = fit(X_feat, y, "features")
    print(f"  features baseline: acc={acc_feat:.4f}  ({time.time()-t0:.1f}s)")
    entry["features_acc"] = float(acc_feat)

    # Per-layer probes
    for layer_idx in range(3):
        t0 = time.time()
        X_act = acts_by_layer[layer_idx][scen_mask].astype(np.float32)
        m, acc, ll = fit(X_act, y, f"h{layer_idx}", C=0.5)
        dt = time.time() - t0
        # Top-k neurons by |coef|
        coef = m.coef_[0]
        order = np.argsort(-np.abs(coef))
        top20 = order[:20].tolist()
        print(f"  h{layer_idx}: acc={acc:.4f}  ll={ll:.4f}  ({dt:.1f}s)  top_coef=[{coef[top20[0]]:+.3f},{coef[top20[1]]:+.3f},...]")
        entry["layers"][f"h{layer_idx}"] = {
            "acc": float(acc), "log_loss": float(ll),
            "top_neurons": top20,
            "top_coefs": [float(coef[i]) for i in top20],
        }

        # Combined: features + layer
        X_combined = np.concatenate([X_feat, X_act], axis=1)
        t0 = time.time()
        _, acc_combined, _ = fit(X_combined, y, f"h{layer_idx}+feats", C=0.5)
        print(f"  h{layer_idx}+features combined: acc={acc_combined:.4f}  ({time.time()-t0:.1f}s)")
        entry["layers"][f"h{layer_idx}"]["combined_acc"] = float(acc_combined)

    results[scen_name] = entry


def main():
    print(f"Loading {ACT_PATH}...")
    d = np.load(ACT_PATH)
    scenario_id = d["scenario_id"]
    nn_bids = d["nn_bids"]
    features = d["features"]
    acts = [d["h0"], d["h1"], d["h2"]]
    print(f"  {len(scenario_id):,} samples, {acts[0].shape[1]} neurons/layer")

    results = {}
    for scen_id in range(9):
        mask = scenario_id == scen_id
        run_scenario(scen_id, acts, features, nn_bids, mask, results)

    # Summary
    print("\n\n=== SUMMARY ===")
    print(f"{'scenario':<22} {'feat':>8} {'h0':>8} {'h1':>8} {'h2':>8} {'h0+f':>8} {'h2+f':>8}")
    for name, e in results.items():
        row = f"{name:<22} {e['features_acc']:>8.4f}"
        for l in ["h0", "h1", "h2"]:
            row += f" {e['layers'][l]['acc']:>8.4f}"
        row += f" {e['layers']['h0']['combined_acc']:>8.4f} {e['layers']['h2']['combined_acc']:>8.4f}"
        print(row)

    with open("/tmp/probe_results.json", "w") as f:
        json.dump(results, f, indent=2)
    print("\n[saved] /tmp/probe_results.json")


if __name__ == "__main__":
    main()
