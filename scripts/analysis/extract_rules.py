"""
Extract human-readable rule cards for IS-DD play decisions.

For each (scenario, target):
  - Top features by XGBoost importance (top 5)
  - Conditional probability tables for top features (1D + 2D cross)
  - Decision trees at depth 2 and 3 (sklearn, balanced classes)
  - Compact rule summary (top split + leaf rates)

Usage:
  python extract_rules.py --scenario dt_lead_mid --target led_trump
  python extract_rules.py --batch     # run on a curated list of high-value combos
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
import play_lab as L


def fmt_pct(x): return f"{x*100:5.1f}%"


def cond_prob_table_1d(y, x, name, max_buckets=10):
    """Print P(y=1 | x=v) for each value of x (or for binned ranges)."""
    print(f"\n  P(target | {name}):")
    if x.dtype.kind in "f":
        # bin floats
        edges = np.percentile(x, np.linspace(0, 100, 11))
        edges = np.unique(edges)
        bins = np.digitize(x, edges[1:-1])
        labels = [f"[{edges[i]:.1f}, {edges[i+1]:.1f}]" for i in range(len(edges)-1)]
        for b, lab in enumerate(labels):
            sel = bins == b
            n = sel.sum()
            if n < 50: continue
            p = y[sel].mean()
            bar = "█" * int(p * 30)
            print(f"    {lab:<24} n={n:>7,} {fmt_pct(p)} {bar}")
    else:
        vals, counts = np.unique(x, return_counts=True)
        for v, c in zip(vals, counts):
            if c < 50: continue
            sel = x == v
            p = y[sel].mean()
            bar = "█" * int(p * 30)
            print(f"    {name}={v:>3}  n={c:>7,}  {fmt_pct(p)}  {bar}")


def cond_prob_table_2d(y, x1, x2, name1, name2, min_n=200):
    """Print P(y=1 | x1, x2) cross-tab."""
    print(f"\n  P(target | {name1} × {name2}):")
    v1s = sorted(np.unique(x1).tolist())
    v2s = sorted(np.unique(x2).tolist())
    if len(v1s) > 8 or len(v2s) > 8:
        # too many — collapse via percentile
        return None
    hdr = f"    {name1:>12}\\{name2:<10}  "
    for v2 in v2s:
        hdr += f"{str(v2):>8}  "
    print(hdr)
    for v1 in v1s:
        row = f"    {str(v1):>22}  "
        for v2 in v2s:
            sel = (x1 == v1) & (x2 == v2)
            n = sel.sum()
            if n < min_n:
                row += f"{'---':>8}  "
            else:
                p = y[sel].mean()
                row += f"{fmt_pct(p):>8}  "
        print(row)


def fit_xgb(Xtr, ytr, Xval, yval, **kw):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=kw.get("n_estimators", 200),
        max_depth=kw.get("max_depth", 5),
        learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=0,
        eval_metric="logloss",
    )
    clf.fit(Xtr, ytr)
    p = clf.predict_proba(Xval)[:, 1]
    return clf, roc_auc_score(yval, p) if len(np.unique(yval)) > 1 else float("nan")


def fit_dt(Xtr, ytr, Xval, yval, max_depth=3, min_samples_leaf=1000):
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf,
                                class_weight="balanced", random_state=0)
    dt.fit(Xtr, ytr)
    p = dt.predict_proba(Xval)[:, 1]
    return dt, roc_auc_score(yval, p) if len(np.unique(yval)) > 1 else float("nan")


def extract_rules(d, scenario_name, target_name, *, sample_n=80_000):
    sc_fn = L.SCENARIOS[scenario_name]
    tg_fn = L.TARGETS[(scenario_name, target_name)]
    mask = sc_fn(d)
    n = int(mask.sum())
    if n < 1000:
        print(f"  [{scenario_name}/{target_name}] only {n} rows — skipping")
        return
    y = tg_fn(d, mask)
    pos = y.mean()

    print()
    print("=" * 78)
    print(f"  {scenario_name}  →  {target_name}")
    print("=" * 78)
    print(f"  rows: {n:,}    positive rate: {fmt_pct(pos)}")

    cols = L.get_predictor_columns(d)
    X = L.make_X(d, mask, cols)

    # Subsample for speed
    if sample_n and len(y) > sample_n:
        rng = np.random.default_rng(0)
        idx = rng.choice(len(y), sample_n, replace=False)
        X, y = X[idx], y[idx]
    Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)

    # XGBoost feature importance (full features, depth-5)
    clf, auc = fit_xgb(Xtr, ytr, Xv, yv, n_estimators=200, max_depth=5)
    print(f"  XGBoost (all features, depth=5): val_AUC={auc:.4f}")
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    top_k = 5
    print(f"  Top {top_k} features by importance:")
    top_features = []
    for i in order[:top_k]:
        if imp[i] < 0.005:
            break
        top_features.append(cols[i])
        bar = "█" * int(imp[i] * 60)
        print(f"    {cols[i]:<28} {imp[i]*100:5.1f}%  {bar}")

    # 1D conditional prob tables for top 2 features
    for f in top_features[:2]:
        cond_prob_table_1d(y, X[:, cols.index(f)], f)

    # 2D cross
    if len(top_features) >= 2:
        cond_prob_table_2d(y, X[:, cols.index(top_features[0])],
                           X[:, cols.index(top_features[1])],
                           top_features[0], top_features[1])

    # Depth-2 + Depth-3 trees on top features
    for depth in (2, 3):
        sub_cols = top_features[:min(len(top_features), 5)]
        sub_idx = [cols.index(f) for f in sub_cols]
        dt, dt_auc = fit_dt(Xtr[:, sub_idx], ytr, Xv[:, sub_idx], yv,
                            max_depth=depth, min_samples_leaf=max(500, len(ytr)//200))
        print(f"\n  Tree depth={depth} on {sub_cols} → val_AUC={dt_auc:.4f}")
        text = export_text(dt, feature_names=sub_cols, max_depth=depth, show_weights=True)
        for line in text.splitlines():
            print(f"    {line}")


# Curated batch — high-value (scenario, target) combos
BATCH = [
    ("dt_lead_opening", "led_trump"),
    ("dt_lead_opening", "led_ace_offsuit"),
    ("dt_lead_opening", "led_master"),
    ("dt_lead_opening", "led_low"),

    ("dt_lead_mid",     "led_trump"),
    ("dt_lead_mid",     "led_ace_offsuit"),
    ("dt_lead_mid",     "led_master"),
    ("dt_lead_mid",     "led_low"),

    ("dt_trump_follow", "played_max_trump"),
    ("dt_trump_follow", "played_jack_or_nine"),

    ("dt_follow_opp_takeable", "played_master"),
    ("dt_follow_opp_takeable", "played_low"),
    ("dt_follow_opp_duck",     "played_high"),
    ("dt_follow_opp_duck",     "played_low"),
    ("dt_follow_partner_wins", "played_high"),
    ("dt_follow_partner_wins", "played_low"),

    ("dt_cut_or_duck_opp",     "did_cut"),
    ("dt_cut_or_duck_partner", "did_cut"),

    ("dt_discard_no_trump",    "discarded_low"),
    ("dt_discard_no_trump",    "discarded_ace_or_ten"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/distill/play_features_real.npz")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--batch", action="store_true")
    ap.add_argument("--sample-n", type=int, default=80_000)
    args = ap.parse_args()

    print(f"Loading {args.input}...")
    d = dict(np.load(args.input))
    print(f"  {len(d['deal_id']):,} rows, {len(d)} columns")

    if args.batch:
        for sc, tg in BATCH:
            extract_rules(d, sc, tg, sample_n=args.sample_n)
    else:
        if not args.scenario or not args.target:
            print("Use --scenario X --target Y, or --batch", file=sys.stderr)
            return 2
        extract_rules(d, args.scenario, args.target, sample_n=args.sample_n)


if __name__ == "__main__":
    sys.exit(main())
