"""
First-pass interpretability for IS-DD play: declarer opening lead (trick 0).

Scenario: trick_idx == 0, play_idx == 0, is_declarer_team == 1
  → 200k deals × 4 trumps × ~2 NS seats = ~400k rows
    (only the seat that leads first; depends on dealer)

Targets analyzed (binary):
  1. led_trump            — does declarer lead a trump?
  2. led_ace_offsuit      — does declarer lead an off-suit Ace?
  3. led_master_of_suit   — does declarer lead the master of the suit they led?

Pipeline mirrors scripts/analysis/distill_bid.py:
  - Train depth-5 DecisionTree (human-readable rules)
  - Train XGBoost (300 trees, depth 5) for stronger feature importance
  - Print feature-importance bars + extracted rules + statistical bid tables.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.tree import DecisionTreeClassifier, export_text


def load(path: Path):
    print(f"Loading {path}...")
    d = dict(np.load(path))
    n = len(d["deal_id"])
    print(f"  {n:,} rows, {len(d)} columns")
    return d


def filter_declarer_opening_lead(d: dict) -> np.ndarray:
    return (d["trick_idx"] == 0) & (d["play_idx"] == 0) & (d["is_declarer_team"] == 1)


def filter_defender_opening_lead(d: dict) -> np.ndarray:
    return (d["trick_idx"] == 0) & (d["play_idx"] == 0) & (d["is_declarer_team"] == 0)


def card_suit(c: np.ndarray) -> np.ndarray:
    return (c // 8).astype(np.int8)


def card_rank(c: np.ndarray) -> np.ndarray:
    return (c % 8).astype(np.int8)


def is_master_of_suit(card: np.ndarray, played_cards: np.ndarray, hand: np.ndarray) -> np.ndarray:
    """True if card is the highest remaining card (in plain order) in its suit,
       considering all cards still outstanding (not yet played)."""
    suit = card_suit(card)
    rank = card_rank(card)
    out = np.zeros(len(card), dtype=bool)
    outstanding = (~played_cards) & 0xFFFFFFFF
    suit_outstanding = ((outstanding >> (suit.astype(np.uint32) * 8)) & 0xFF).astype(np.uint8)
    # is highest remaining bit in this suit equal to our rank?
    # find highest bit
    for r in range(7, -1, -1):
        is_present = (suit_outstanding >> r) & 1 == 1
        not_yet = ~out
        # First time we find a present rank, check if it equals card's rank
        match = is_present & (rank == r) & not_yet
        higher = is_present & (rank < r) & not_yet
        out[match] = True
        # Once we hit a higher present rank that isn't ours, mark not-master
        # (we need to short-circuit). Use a "found" tracker.
    # The naive loop above is buggy for "found higher first"; do it differently:
    out = np.zeros(len(card), dtype=bool)
    for s in range(4):
        sel = suit == s
        if not sel.any():
            continue
        outsuit = ((outstanding >> (s * 8)) & 0xFF).astype(np.uint8)
        # highest set bit in outsuit
        highest = np.full(len(card), -1, dtype=np.int8)
        for r in range(7, -1, -1):
            mask_present = ((outsuit >> r) & 1 == 1) & sel & (highest == -1)
            highest[mask_present] = r
        out[sel] = (rank[sel] == highest[sel])
    return out


# Feature columns to use as predictors (exclude the action itself).
PREDICTOR_COLS = [
    "trump_count",
    "has_trump_seven", "has_trump_eight", "has_trump_nine", "has_trump_jack",
    "has_trump_queen", "has_trump_king", "has_trump_ten", "has_trump_ace",
    "trump_strength_max", "trump_points_in_hand",
    "side_aces", "side_tens", "side_voids", "side_singletons", "side_doubletons",
    "best_side_length",
    "holds_master_trump", "holds_master_side_count",
    "trumps_remaining_outside",
    # opening-lead-specific: nothing in the trick yet so trick_state features are degenerate
]


def make_X(d: dict, mask: np.ndarray) -> tuple[np.ndarray, list[str]]:
    cols = []
    arrs = []
    for c in PREDICTOR_COLS:
        if c not in d:
            continue
        cols.append(c)
        arrs.append(d[c][mask].astype(np.float32))
    return np.stack(arrs, axis=1), cols


def fit_tree(X, y, feature_names, max_depth=5, min_samples_leaf=2000):
    dt = DecisionTreeClassifier(max_depth=max_depth, min_samples_leaf=min_samples_leaf, random_state=0)
    dt.fit(X, y)
    print(f"\n  Decision tree (depth={max_depth}, min_leaf={min_samples_leaf}):")
    print(f"    train acc: {dt.score(X, y):.3f}")
    text = export_text(dt, feature_names=feature_names, max_depth=max_depth)
    # Indent for readability
    for line in text.splitlines():
        print(f"    {line}")
    return dt


def fit_xgb(X, y, feature_names, n_estimators=300, max_depth=5):
    try:
        import xgboost as xgb
    except ImportError:
        print("  xgboost not available, skipping")
        return None
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=0,
        eval_metric="logloss",
    )
    clf.fit(X, y)
    acc = clf.score(X, y)
    print(f"\n  XGBoost (n={n_estimators}, depth={max_depth}): train acc {acc:.3f}")
    imp = clf.feature_importances_
    order = np.argsort(imp)[::-1]
    print(f"  Feature importance:")
    width = max(len(f) for f in feature_names)
    for i in order:
        if imp[i] < 0.001:
            break
        bar = "█" * int(imp[i] * 80)
        print(f"    {feature_names[i]:<{width}} {imp[i]*100:5.1f}% {bar}")
    return clf


def stats_table(name: str, y: np.ndarray, X: np.ndarray, feature_names: list[str], by: str | list[str]):
    """Print a table of mean(y) bucketed by one or two integer-ish features."""
    if isinstance(by, str):
        by = [by]
    idx = [feature_names.index(b) for b in by]
    cols = [X[:, i].astype(np.int8) for i in idx]
    print(f"\n  {name}: P({name} | {' × '.join(by)}):")
    if len(by) == 1:
        col = cols[0]
        for v in sorted(np.unique(col)):
            sel = col == v
            n = sel.sum()
            if n < 50:
                continue
            p = y[sel].mean()
            print(f"    {by[0]}={v:>2}  n={n:>7,}  p={p*100:5.1f}%")
    elif len(by) == 2:
        c0, c1 = cols
        u0 = sorted(np.unique(c0))
        u1 = sorted(np.unique(c1))
        # Header
        hdr = f"    {by[0]}\\{by[1]}    "
        for v1 in u1:
            hdr += f"  {v1:>2}    "
        print(hdr)
        for v0 in u0:
            row = f"    {v0:>10}      "
            for v1 in u1:
                sel = (c0 == v0) & (c1 == v1)
                n = sel.sum()
                if n < 50:
                    row += "  ---    "
                else:
                    p = y[sel].mean()
                    row += f"  {p*100:5.1f}% "
            print(row)


def analyze_target(d: dict, mask: np.ndarray, target_y: np.ndarray, target_name: str):
    print(f"\n{'='*70}\n  TARGET: {target_name}\n{'='*70}")
    print(f"  rows: {mask.sum():,}  positive rate: {target_y.mean()*100:.1f}%")
    X, cols = make_X(d, mask)
    print(f"  features: {len(cols)}")

    fit_tree(X, target_y, cols, max_depth=5)
    fit_xgb(X, target_y, cols)

    # Statistical breakdowns by key features
    if "trump_count" in cols and "trump_strength_max" in cols:
        stats_table(target_name, target_y, X, cols, ["trump_count"])
    if "has_trump_jack" in cols and "trump_count" in cols:
        stats_table(target_name, target_y, X, cols, ["trump_count", "has_trump_jack"])
    if "side_aces" in cols:
        stats_table(target_name, target_y, X, cols, ["side_aces"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/distill/play_features.npz")
    ap.add_argument("--side", default="declarer", choices=["declarer", "defender"])
    args = ap.parse_args()

    d = load(Path(args.input))

    if args.side == "declarer":
        mask = filter_declarer_opening_lead(d)
        side_label = "DECLARER"
    else:
        mask = filter_defender_opening_lead(d)
        side_label = "DEFENDER"
    print(f"\n=== {side_label} OPENING LEAD ({mask.sum():,} rows) ===")

    chosen = d["chosen"][mask]
    trump = d["forced_suit"][mask]
    played = d["played_cards"][mask]
    hand = d["hand"][mask]

    chosen_suit = card_suit(chosen)
    chosen_rank = card_rank(chosen)

    # Targets
    y_trump = (chosen_suit == trump).astype(np.uint8)
    y_ace_offsuit = ((chosen_suit != trump) & (chosen_rank == 7)).astype(np.uint8)
    y_master = is_master_of_suit(chosen, played, hand).astype(np.uint8)

    analyze_target(d, mask, y_trump, "led_trump")
    analyze_target(d, mask, y_ace_offsuit, "led_ace_offsuit")
    analyze_target(d, mask, y_master, "led_master_of_suit")

    # Quick raw distribution: which suit relative to trump does declarer/defender lead?
    print(f"\n=== Raw lead-suit distribution ({side_label}) ===")
    rel = np.where(chosen_suit == trump, 0,
          np.where(chosen_suit < trump, chosen_suit + 1, chosen_suit)).astype(np.int8)
    # Simpler: just show is_trump and counts
    print(f"  led trump:        {y_trump.mean()*100:5.1f}%")
    print(f"  led off-suit ace: {y_ace_offsuit.mean()*100:5.1f}%")
    print(f"  led master:       {y_master.mean()*100:5.1f}%")
    # Rank distribution
    ranks_label = ["7", "8", "9", "J", "Q", "K", "10", "A"]
    print(f"\n  Lead-card rank distribution:")
    for r in range(8):
        p = (chosen_rank == r).mean()
        print(f"    {ranks_label[r]:>2}: {p*100:5.1f}%")


if __name__ == "__main__":
    sys.exit(main())
