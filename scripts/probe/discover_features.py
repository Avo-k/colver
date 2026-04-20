"""From the top-neuron tree rules, derive candidate NEW interpretable features
and measure accuracy gain when added to baseline XGBoost.

Strategy:
  - Scan all scenarios × top-12 neurons.
  - For each neuron, try depth-3 tree on hand features. Keep if R² > 0.5
    (meaning the neuron IS largely captured by hand features — rejection
    criterion: if R² is low, the neuron encodes something NOT in our 17 features).
  - Conversely, neurons with R² < 0.3 are the INTERESTING ones — they encode
    info that 17 features can't explain.
  - For those "mysterious" neurons, correlate with 64 engineered shape/side
    candidates and report which one is the best proxy.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

ACT_PATH = "/tmp/probe_activations.npz"
PROBE_JSON = "/tmp/probe_results.json"

BASE = [
    "trump_count", "has_jack", "has_nine", "has_ace", "has_ten",
    "has_king", "has_queen", "trump_points", "trump_score", "has_belote",
    "side_aces", "side_tens", "side_voids", "side_singletons",
    "side_doubletons", "total_aces", "best_side_length",
]

SCEN = {
    0: "pos1_open", 1: "pos2_after_pass", 2: "pos3_after_2p", 3: "pos4_after_3p",
    4: "pos3_partner80", 5: "pos4_partner80", 6: "pos2_opp80", 7: "pos3_opp80",
    8: "pos4_opp80",
}


def engineer_candidate_features(feats: np.ndarray, obs: np.ndarray) -> pd.DataFrame:
    """Derive a wide set of candidate features from the 17 base + raw obs.
    Obs layout (113-dim, see bid_obs.rs):
      [0:32]   hand bitmask (floats)     — S[0:8] H[8:16] D[16:24] C[24:32]
      [32:104] bid history 12×6
      [104:108] position
      [108:113] score features
    """
    df = pd.DataFrame(feats, columns=BASE)

    # Hand bitmask: obs[0:32] are 32 floats, 0 or 1. Decode per-suit.
    hand_bits = obs[:, :32] > 0.5  # (N, 32)

    # Per-suit counts & specific cards
    for sidx, sname in enumerate(["S", "H", "D", "C"]):
        base_bit = sidx * 8
        df[f"s{sname}_count"] = hand_bits[:, base_bit:base_bit + 8].sum(axis=1).astype(np.int32)
        # Rank bits: 7=0, 8=1, 9=2, J=3, Q=4, K=5, 10=6, A=7 (see card.rs)
        df[f"s{sname}_has_J"] = hand_bits[:, base_bit + 3].astype(np.int8)
        df[f"s{sname}_has_9"] = hand_bits[:, base_bit + 2].astype(np.int8)
        df[f"s{sname}_has_A"] = hand_bits[:, base_bit + 7].astype(np.int8)
        df[f"s{sname}_has_K"] = hand_bits[:, base_bit + 5].astype(np.int8)
        df[f"s{sname}_has_Q"] = hand_bits[:, base_bit + 4].astype(np.int8)
        df[f"s{sname}_has_T"] = hand_bits[:, base_bit + 6].astype(np.int8)
        # Top-2 (A+K) together, guard
        df[f"s{sname}_AK"] = (hand_bits[:, base_bit + 7] & hand_bits[:, base_bit + 5]).astype(np.int8)
        df[f"s{sname}_KQ"] = (hand_bits[:, base_bit + 5] & hand_bits[:, base_bit + 4]).astype(np.int8)

    # Shape features (sorted lengths across 4 suits)
    lengths = np.stack([df["sS_count"], df["sH_count"], df["sD_count"], df["sC_count"]], axis=1)
    lengths_sorted = np.sort(lengths, axis=1)[:, ::-1]  # desc
    df["shape_l1"] = lengths_sorted[:, 0]
    df["shape_l2"] = lengths_sorted[:, 1]
    df["shape_l3"] = lengths_sorted[:, 2]
    df["shape_l4"] = lengths_sorted[:, 3]
    # Binary shape encodings
    df["is_flat_4333"] = ((lengths_sorted[:, 0] == 4) & (lengths_sorted[:, 1] == 3) &
                          (lengths_sorted[:, 2] == 3) & (lengths_sorted[:, 3] == 3)).astype(np.int8)
    df["is_5332"] = ((lengths_sorted[:, 0] == 5) & (lengths_sorted[:, 1] == 3) &
                     (lengths_sorted[:, 2] == 3) & (lengths_sorted[:, 3] == 2)).astype(np.int8)
    df["is_5431"] = ((lengths_sorted[:, 0] == 5) & (lengths_sorted[:, 1] == 4) &
                     (lengths_sorted[:, 2] == 3) & (lengths_sorted[:, 3] == 1)).astype(np.int8)
    df["is_5440"] = ((lengths_sorted[:, 0] == 5) & (lengths_sorted[:, 1] == 4) &
                     (lengths_sorted[:, 2] == 4) & (lengths_sorted[:, 3] == 0)).astype(np.int8)
    df["is_6322"] = ((lengths_sorted[:, 0] == 6) & (lengths_sorted[:, 1] == 3) &
                     (lengths_sorted[:, 2] == 2) & (lengths_sorted[:, 3] == 2)).astype(np.int8)
    df["has_void"] = (lengths_sorted[:, 3] == 0).astype(np.int8)
    df["has_singleton"] = (lengths_sorted[:, 3] == 1).astype(np.int8)
    df["has_6plus"] = (lengths_sorted[:, 0] >= 6).astype(np.int8)

    # Shape entropy (distribution diversity)
    probs = lengths / 8.0
    with np.errstate(divide="ignore", invalid="ignore"):
        log_p = np.where(probs > 0, np.log(probs), 0)
        entropy = -(probs * log_p).sum(axis=1)
    df["shape_entropy"] = entropy.astype(np.float32)

    # Number of "strong" side suits: suits with A or K+Q
    strong = np.zeros(len(df), dtype=np.int8)
    for sname in ["S", "H", "D", "C"]:
        has_strong = (df[f"s{sname}_has_A"] | df[f"s{sname}_KQ"]).values
        strong += has_strong.astype(np.int8)
    df["n_strong_suits"] = strong

    # Total high cards (A + K + Q + J aggregated)
    hc = np.zeros(len(df), dtype=np.int16)
    for sname in ["S", "H", "D", "C"]:
        hc += df[f"s{sname}_has_A"] + df[f"s{sname}_has_K"] + df[f"s{sname}_has_Q"] + df[f"s{sname}_has_J"]
    df["total_honours"] = hc

    return df


def main():
    d = np.load(ACT_PATH)
    scenario_id = d["scenario_id"]
    features = d["features"].astype(np.float32)
    obs = d["obs"]
    h2 = d["h2"].astype(np.float32)

    with open(PROBE_JSON) as f:
        probe = json.load(f)

    # Engineer candidates once on the whole dataset
    eng = engineer_candidate_features(features, obs)
    candidate_cols = [c for c in eng.columns if c not in BASE]
    print(f"Engineered {len(candidate_cols)} candidate features")

    results = []
    for scen_id, scen_name in SCEN.items():
        if scen_name not in probe:
            continue
        mask = scenario_id == scen_id
        sub_feats = features[mask]
        sub_eng = eng.iloc[mask.nonzero()[0]].reset_index(drop=True)
        sub_h2 = h2[mask]
        if sub_feats.shape[0] < 500:
            continue

        print(f"\n=== {scen_name} (n={sub_feats.shape[0]:,}) ===")

        top_h2 = probe[scen_name]["layers"]["h2"]["top_neurons"][:12]
        top_coefs = probe[scen_name]["layers"]["h2"]["top_coefs"][:12]

        for rank, (nid, coef) in enumerate(zip(top_h2, top_coefs), 1):
            act = sub_h2[:, nid]

            # How well does the 17-feature tree explain the neuron?
            t = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=42)
            t.fit(sub_feats, act)
            base_r2 = max(0.0, 1 - ((act - t.predict(sub_feats))**2).sum() / max(((act - act.mean())**2).sum(), 1e-10))

            # How well does extended feature set explain it?
            t_ext = DecisionTreeRegressor(max_depth=3, min_samples_leaf=500, random_state=42)
            t_ext.fit(sub_eng.values, act)
            ext_r2 = max(0.0, 1 - ((act - t_ext.predict(sub_eng.values))**2).sum() / max(((act - act.mean())**2).sum(), 1e-10))

            # Top candidate correlations (only extended, not base)
            ext_corrs = []
            for c in candidate_cols:
                v = sub_eng[c].values.astype(np.float32)
                if v.std() > 0:
                    cc = np.corrcoef(v, act)[0, 1]
                    if not np.isnan(cc):
                        ext_corrs.append((c, float(cc)))
            ext_corrs.sort(key=lambda t: -abs(t[1]))

            is_mysterious = base_r2 < 0.3
            marker = " [MYSTERY]" if is_mysterious else ("" if ext_r2 - base_r2 < 0.05 else " [EXT_GAIN]")
            top3 = ", ".join(f"{n}={c:+.2f}" for n, c in ext_corrs[:3])
            print(f"  h2[{nid}] coef={coef:+.3f}  base_R²={base_r2:.2f}  ext_R²={ext_r2:.2f}{marker}  top_ext: {top3}")

            results.append({
                "scenario": scen_name,
                "neuron": int(nid),
                "probe_coef": float(coef),
                "base_r2": float(base_r2),
                "ext_r2": float(ext_r2),
                "r2_gain": float(ext_r2 - base_r2),
                "top_candidates": [{"name": n, "corr": c} for n, c in ext_corrs[:8]],
            })

    # Summary
    print("\n\n=== FEATURES MOST FREQUENTLY TOP-CORRELATED WITH MYSTERY NEURONS ===")
    from collections import Counter
    mystery = [r for r in results if r["base_r2"] < 0.3]
    counter = Counter()
    for r in mystery:
        for c in r["top_candidates"][:3]:
            counter[c["name"]] += 1
    print(f"Mystery neurons: {len(mystery)} / {len(results)}")
    for name, n in counter.most_common(20):
        print(f"  {name:<25}  {n:>4}")

    with open("/tmp/probe_discovered_features.json", "w") as f:
        json.dump({"per_neuron": results, "mystery_top_corrs": counter.most_common(20)}, f, indent=2)
    print("\n[saved] /tmp/probe_discovered_features.json")


if __name__ == "__main__":
    main()
