"""Final step: does adding the discovered features actually improve XGBoost?

Compares:
  A. 17 base features (what distill_bid exports today)
  B. 17 base + discovered interpretable features (shape, per-suit cards)
  C. Same as B + Q-values (plafond absolu)

Measures per-deal accuracy per scenario.
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier

from discover_features import engineer_candidate_features, BASE, SCEN

ACT_PATH = "/tmp/probe_activations.npz"


def fit_xgb(X, y, seed=42):
    if y.mean() < 0.02 or y.mean() > 0.98:
        return None, float("nan")
    Xt, Xv, yt, yv = train_test_split(X, y, test_size=0.2, random_state=seed, stratify=y)
    m = XGBClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.1,
        scale_pos_weight=(yt == 0).sum() / max((yt == 1).sum(), 1),
        random_state=seed, verbosity=0, n_jobs=-1,
    )
    m.fit(Xt, yt)
    return m, accuracy_score(yv, m.predict(Xv))


def top_importance(m, names, k=15):
    imp = sorted(zip(names, m.feature_importances_), key=lambda t: -t[1])
    return [(n, float(i)) for n, i in imp[:k]]


def main():
    d = np.load(ACT_PATH)
    scenario_id = d["scenario_id"]
    features = d["features"].astype(np.float32)
    obs = d["obs"]
    nn_bids = d["nn_bids"]

    print(f"Engineering candidates on {len(features):,} samples...")
    eng = engineer_candidate_features(features, obs)
    base_cols = BASE
    all_cols = list(eng.columns)
    extra_cols = [c for c in all_cols if c not in base_cols]
    print(f"  base features: {len(base_cols)}, extra: {len(extra_cols)}")

    # Keep Q-like columns from obs? Actually we don't have Q here. Skip C.
    results = []
    for scen_id, scen_name in SCEN.items():
        mask = scenario_id == scen_id
        n = mask.sum()
        if n < 1000:
            continue
        y = nn_bids[mask]
        if y.mean() < 0.02 or y.mean() > 0.98:
            continue
        sub_eng = eng.iloc[mask.nonzero()[0]].reset_index(drop=True)

        # A. Base only
        t0 = time.time()
        _, acc_A = fit_xgb(sub_eng[base_cols].values, y)

        # B. Base + extra
        m_B, acc_B = fit_xgb(sub_eng[all_cols].values, y)
        dt = time.time() - t0

        # Which extras matter?
        extra_top = []
        if m_B is not None:
            imp = top_importance(m_B, all_cols, k=25)
            for name, i in imp:
                if name in extra_cols and i > 0.003:
                    extra_top.append((name, i))

        print(f"\n=== {scen_name} (n={n:,}, bid_rate={y.mean():.1%}) ===")
        print(f"  A. base (17):             acc={acc_A:.4f}")
        print(f"  B. base + extras (+{len(extra_cols)}):   acc={acc_B:.4f}  Δ={acc_B-acc_A:+.4f}")
        if extra_top:
            print("  Most useful extras:")
            for name, i in extra_top[:10]:
                print(f"    {name:<25}  imp={i:.3f}")
        print(f"  ({dt:.1f}s)")

        results.append({
            "scenario": scen_name,
            "n": int(n),
            "bid_rate": float(y.mean()),
            "acc_base": float(acc_A),
            "acc_extended": float(acc_B),
            "delta": float(acc_B - acc_A),
            "useful_extras": extra_top[:15],
        })

    print("\n\n=== SUMMARY ===")
    print(f"{'scenario':<22} {'base':>8} {'+extras':>8} {'Δ':>7}")
    for r in results:
        print(f"{r['scenario']:<22} {r['acc_base']:>8.4f} {r['acc_extended']:>8.4f} {r['delta']:>+7.4f}")

    # Aggregate: extras that were important most often
    from collections import Counter
    global_counter = Counter()
    for r in results:
        for name, _ in r["useful_extras"]:
            global_counter[name] += 1
    print("\n=== EXTRA FEATURES MOST FREQUENTLY IN TOP-15 ACROSS SCENARIOS ===")
    for name, n in global_counter.most_common(20):
        print(f"  {name:<25}  {n:>4} / {len(results)}")

    with open("/tmp/probe_final_results.json", "w") as f:
        json.dump({"per_scenario": results, "global_counter": global_counter.most_common(25)}, f, indent=2)
    print("\n[saved] /tmp/probe_final_results.json")


if __name__ == "__main__":
    main()
