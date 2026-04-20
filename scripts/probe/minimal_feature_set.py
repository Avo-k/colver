"""Incrementally test small additions to the 17-feature baseline, to find
the CHEAPEST set of new features that closes the gap.

We go from baseline (17) → baseline + K new features, trying different
combinations of:
  - second_trump_score / second_trump_count (per-deal, requires 2 rows)
  - shape_entropy
  - sX_has_J / sX_has_9 (per-suit honours)
  - n_strong_suits

Report accuracy per scenario per feature-set addition.
"""
from __future__ import annotations

import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from discover_features import engineer_candidate_features, BASE, SCEN

ACT_PATH = "/tmp/probe_activations.npz"


def fit(X, y):
    if y.mean() < 0.02 or y.mean() > 0.98:
        return float("nan")
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    m = XGBClassifier(
        n_estimators=200, max_depth=5, learning_rate=0.1,
        scale_pos_weight=(yt == 0).sum() / max((yt == 1).sum(), 1),
        random_state=42, verbosity=0, n_jobs=-1,
    )
    m.fit(Xt, yt)
    return accuracy_score(yv, m.predict(Xv))


def main():
    d = np.load(ACT_PATH)
    scenario_id = d["scenario_id"]
    features = d["features"].astype(np.float32)
    obs = d["obs"]
    nn_bids = d["nn_bids"]

    print(f"Engineering...")
    eng = engineer_candidate_features(features, obs)

    # Build feature sets by incremental addition
    sets = {
        "A. base (17)": BASE,
        "B. +shape (3)": BASE + ["shape_entropy", "shape_l1", "shape_l4"],
        "C. +per_suit_J9 (11)": BASE + ["shape_entropy", "shape_l1", "shape_l4"]
                              + ["sS_has_J", "sH_has_J", "sD_has_J", "sC_has_J",
                                 "sS_has_9", "sH_has_9", "sD_has_9", "sC_has_9"],
        "D. +per_suit_count (4)": BASE + ["shape_entropy", "shape_l1", "shape_l4"]
                                + ["sS_has_J", "sH_has_J", "sD_has_J", "sC_has_J",
                                   "sS_has_9", "sH_has_9", "sD_has_9", "sC_has_9"]
                                + ["sS_count", "sH_count", "sD_count", "sC_count"],
        "E. +n_strong (1)": BASE + ["shape_entropy", "shape_l1", "shape_l4"]
                          + ["sS_has_J", "sH_has_J", "sD_has_J", "sC_has_J",
                             "sS_has_9", "sH_has_9", "sD_has_9", "sC_has_9"]
                          + ["sS_count", "sH_count", "sD_count", "sC_count"]
                          + ["n_strong_suits"],
    }

    header = f"{'scenario':<22}"
    for k in sets.keys():
        header += f" {k:>20}"
    print(header)

    rows = []
    for scen_id, scen_name in SCEN.items():
        mask = scenario_id == scen_id
        if mask.sum() < 1000:
            continue
        y = nn_bids[mask]
        if y.mean() < 0.02 or y.mean() > 0.98:
            continue
        sub = eng.iloc[mask.nonzero()[0]].reset_index(drop=True)

        row_str = f"{scen_name:<22}"
        accs = []
        t0 = time.time()
        for label, cols in sets.items():
            a = fit(sub[cols].values, y)
            row_str += f" {a:>20.4f}"
            accs.append(a)
        print(row_str + f"   ({time.time()-t0:.1f}s)")
        rows.append([scen_name] + accs)

    import json
    with open("/tmp/minimal_sets_results.json", "w") as f:
        json.dump({"sets": list(sets.keys()), "rows": rows}, f, indent=2)

    # Deltas vs A
    print("\nDeltas vs baseline (A):")
    hdr = f"{'scenario':<22}"
    for k in list(sets.keys())[1:]:
        hdr += f" {k:>20}"
    print(hdr)
    for row in rows:
        name = row[0]
        a0 = row[1]
        s = f"{name:<22}"
        for a in row[2:]:
            s += f" {a-a0:>+20.4f}"
        print(s)


if __name__ == "__main__":
    main()
