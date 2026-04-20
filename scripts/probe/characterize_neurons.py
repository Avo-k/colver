"""Characterize what the top neurons from the probes respond to.

For each of the top-N bid-predictive neurons (h2 layer, last hidden), we:
  1. Fit a depth-3 decision tree on (features) → (neuron activation)
  2. Report the tree structure as a human-readable rule
  3. Report correlation with each base feature

This reveals what "concept" each neuron has learned.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor, export_text

ACT_PATH = "/tmp/probe_activations.npz"
PROBE_JSON = "/tmp/probe_results.json"
OUT = Path("/tmp/probe_neuron_concepts.md")

BASE_FEATURE_NAMES = [
    "trump_count", "has_jack", "has_nine", "has_ace", "has_ten",
    "has_king", "has_queen", "trump_points", "trump_score", "has_belote",
    "side_aces", "side_tens", "side_voids", "side_singletons",
    "side_doubletons", "total_aces", "best_side_length",
]

SCEN_NAMES = {
    0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p",
    4: "pos3_partner80", 5: "pos4_partner80", 6: "pos2_opp80", 7: "pos3_opp80",
    8: "pos4_opp80",
}


def characterize_neuron(feats: np.ndarray, activation: np.ndarray, max_depth: int = 3):
    """Fit a tree to predict activation from hand features. Return tree text + R²."""
    tree = DecisionTreeRegressor(max_depth=max_depth, min_samples_leaf=500, random_state=42)
    tree.fit(feats, activation)
    pred = tree.predict(feats)
    ss_res = ((activation - pred) ** 2).sum()
    ss_tot = ((activation - activation.mean()) ** 2).sum()
    r2 = 1 - ss_res / max(ss_tot, 1e-10)
    text = export_text(tree, feature_names=BASE_FEATURE_NAMES, max_depth=max_depth, decimals=2)
    return text, r2


def main():
    d = np.load(ACT_PATH)
    scenario_id = d["scenario_id"]
    features = d["features"].astype(np.float32)
    h2 = d["h2"].astype(np.float32)

    with open(PROBE_JSON) as f:
        probe = json.load(f)

    lines = ["# Neuron concept map — bid NN v5 layer 2 (h2)\n"]
    lines.append("For each scenario, we take the top-8 neurons (by |probe coefficient|) and\n")
    lines.append("fit a depth-3 tree on hand features to predict the neuron's activation.\n")
    lines.append("The R² tells us how well 17 hand features capture what the neuron encodes.\n")

    for scen_id, scen_name in SCEN_NAMES.items():
        if scen_name not in probe:
            continue
        entry = probe[scen_name]
        mask = scenario_id == scen_id
        sub_feats = features[mask]
        sub_h2 = h2[mask]
        sub_bids = d["nn_bids"][mask]
        if sub_feats.shape[0] < 500:
            continue

        top_h2 = entry["layers"]["h2"]["top_neurons"][:8]
        top_coefs = entry["layers"]["h2"]["top_coefs"][:8]

        lines.append(f"\n## {scen_name} (n={sub_feats.shape[0]:,}, bid_rate={sub_bids.mean():.1%})\n")
        lines.append(f"Layer-2 probe accuracy: {entry['layers']['h2']['acc']:.4f}\n")

        for rank, (nid, coef) in enumerate(zip(top_h2, top_coefs), 1):
            act = sub_h2[:, nid]
            text, r2 = characterize_neuron(sub_feats, act)

            # Also compute correlation w/ each base feature
            corrs = []
            for i, fname in enumerate(BASE_FEATURE_NAMES):
                if sub_feats[:, i].std() > 0:
                    c = np.corrcoef(sub_feats[:, i], act)[0, 1]
                    if not np.isnan(c):
                        corrs.append((fname, float(c)))
            corrs.sort(key=lambda t: -abs(t[1]))
            top_corrs = ", ".join(f"{n}={c:+.2f}" for n, c in corrs[:5])

            direction = "+→bid" if coef > 0 else "−→bid"
            lines.append(f"\n### Neuron h2[{nid}] (probe coef={coef:+.3f}, {direction}, tree R²={r2:.2f})\n")
            lines.append(f"Top correlations: {top_corrs}\n")
            lines.append(f"\n```\n{text.strip()}\n```\n")

    OUT.write_text("\n".join(lines))
    print(f"[wrote] {OUT}")
    print(f"Total lines: {len(lines)}")


if __name__ == "__main__":
    main()
