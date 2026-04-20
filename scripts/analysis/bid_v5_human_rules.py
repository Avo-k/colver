#!/usr/bin/env python3
"""
Build human-usable bid rules for v5 and validate against the real NN.

Workflow:
  1. Load distill CSV (7.2M rows, 9 scenarios × 200k deals × 4 suits).
  2. Collapse to per-deal view: one row per (scenario, deal_id) with the
     feature values of the BEST suit (highest q_80) plus the NN's chosen
     action (pass / bid / coinche).
  3. For each scenario group, apply a proposed simple rule and measure
     agreement with the NN's actual decision.
  4. Report accuracy per rule, confusion matrix, and disagreement examples.

Usage: uv run python scripts/analysis/bid_v5_human_rules.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

CSV = "data/distill/bid_v5_distill.csv"

COLS = [
    "deal_id", "scenario", "position", "suit",
    "trump_count", "has_jack", "has_nine", "has_ace",
    "has_ten", "has_king", "has_queen",
    "trump_points", "trump_score", "has_belote",
    "side_aces", "side_tens", "side_voids", "side_singletons",
    "side_doubletons", "total_aces", "best_side_length",
    "partner_support", "opp_suit_cards",
    "is_partner_suit", "is_opp_suit",
    "q_80", "nn_action", "nn_bids_this_suit",
]


def load_per_deal(csv: str) -> pd.DataFrame:
    """One row per (scenario, deal_id) with per-deal features:
       best_trump_score    — max trump_score across the 4 suits
       second_trump_score  — 2nd-highest trump_score
       best_trump_count    — trump_count of the best-score suit
       best_is_opp_suit    — 1 if best suit is the opponent's bid suit
       best_is_partner     — 1 if best suit is partner's suit
       best_has_jack       — has_jack of best suit
       best_has_nine       — has_nine of best suit
       best_side_length    — carried over (same for all suits of a deal)
       side_voids          — idem
       total_aces          — idem
       partner_support     — cards in partner's suit (or -1)
       opp_suit_cards      — cards in opp's suit (or -1)
       nn_action           — NN's decision
    """
    print(f"Loading {csv} ...", flush=True)
    df = pd.read_csv(csv, usecols=COLS)
    print(f"  {len(df):,} rows, {df['scenario'].nunique()} scenarios", flush=True)

    # Sort by trump_score desc within each deal
    df_sorted = df.sort_values(["scenario", "deal_id", "trump_score"],
                                ascending=[True, True, False]).reset_index(drop=True)

    # Pick top-1 and top-2 rows per deal (each deal has exactly 4 rows)
    n = len(df_sorted)
    best_mask = np.zeros(n, dtype=bool)
    second_mask = np.zeros(n, dtype=bool)
    for i in range(0, n, 4):
        best_mask[i] = True
        if i + 1 < n:
            second_mask[i + 1] = True

    best = df_sorted[best_mask].reset_index(drop=True)
    second = df_sorted[second_mask].reset_index(drop=True)

    out = pd.DataFrame({
        "scenario": best["scenario"].values,
        "deal_id": best["deal_id"].values,
        # Best-suit features
        "best_trump_score": best["trump_score"].values,
        "best_trump_count": best["trump_count"].values,
        "best_has_jack": best["has_jack"].values,
        "best_has_nine": best["has_nine"].values,
        "best_has_ace": best["has_ace"].values,
        "best_total_aces": best["total_aces"].values,
        "best_is_partner_suit": best["is_partner_suit"].values,
        "best_is_opp_suit": best["is_opp_suit"].values,
        # Second-suit features
        "second_trump_score": second["trump_score"].values,
        "second_trump_count": second["trump_count"].values,
        # Global (same across 4 rows of a deal)
        "side_voids": best["side_voids"].values,  # computed as "voids in side suits" but still OK
        "best_side_length": best["best_side_length"].values,
        "total_aces": best["total_aces"].values,
        "partner_support": best["partner_support"].values,
        "opp_suit_cards": best["opp_suit_cards"].values,
        # Decision
        "nn_action": best["nn_action"].values,
    })
    out["nn_bids"] = ((out["nn_action"] >= 1) & (out["nn_action"] <= 40)).astype(int)
    out["nn_coinche"] = (out["nn_action"] == 41).astype(int)
    out["nn_active"] = (out["nn_bids"] | out["nn_coinche"]).astype(int)
    print(f"  per-deal view: {len(out):,} rows", flush=True)
    return out


def agreement(pred: np.ndarray, truth: np.ndarray, label: str) -> dict:
    acc = (pred == truth).mean()
    n = len(truth)
    miss = ((pred == 0) & (truth == 1)).sum() / n
    fa = ((pred == 1) & (truth == 0)).sum() / n
    pred_rate = pred.mean()
    print(f"  [{label}] n={n:,}  acc={acc:.1%}  miss={miss:.1%}  FA={fa:.1%}  rule_rate={pred_rate:.1%}", flush=True)
    return {"label": label, "n": int(n), "acc": float(acc), "miss": float(miss), "fa": float(fa)}


def show_disagreement_samples(df: pd.DataFrame, pred: np.ndarray, truth: np.ndarray,
                               feats: list[str], label: str, n_each: int = 5):
    miss_idx = np.where((pred == 0) & (truth == 1))[0]
    fa_idx = np.where((pred == 1) & (truth == 0))[0]
    if len(miss_idx) > 0:
        print(f"\n  [{label}] MISS examples (rule=pass, NN=bid):")
        for i in miss_idx[:n_each]:
            row = df.iloc[i]
            vals = {f: row[f] for f in feats if f in row}
            print(f"    {vals}")
    if len(fa_idx) > 0:
        print(f"  [{label}] FALSE-ALARM examples (rule=bid, NN=pass):")
        for i in fa_idx[:n_each]:
            row = df.iloc[i]
            vals = {f: row[f] for f in feats if f in row}
            print(f"    {vals}")


# ============================================================
# All rules below use per-deal features:
#   best_trump_score    = strength of best-suit-as-trump
#   best_trump_count    = number of cards in that best suit
#   best_has_jack/nine  = J/9 presence in best suit
#   second_trump_score  = strength of second-best suit
#   best_side_length    = length of longest suit (may equal best_trump_count)
#   total_aces, side_voids, partner_support, opp_suit_cards
# ============================================================

def rule_opening(df: pd.DataFrame) -> np.ndarray:
    """Opening (pos1). Target: 75.8% bid rate."""
    ts = df["best_trump_score"]
    tc = df["best_trump_count"]
    j = df["best_has_jack"]
    vd = df["side_voids"]

    bid = (
        (ts >= 15)
        | ((ts >= 13) & (tc >= 3))
        | ((j == 1) & (vd >= 1) & (tc >= 3))
        | (tc >= 5)
    )
    return bid.astype(int).values


def rule_after_pass(df: pd.DataFrame, position: int) -> np.ndarray:
    """Pos 2-4 after passes. Target: 74% (p2), 87% (p3), 81% (p4)."""
    ts = df["best_trump_score"]
    tc = df["best_trump_count"]
    j = df["best_has_jack"]
    vd = df["side_voids"]

    if position == 2:
        # Pos 2: ~= opening but slightly more permissive
        bid = (
            (ts >= 14)
            | ((ts >= 12) & (tc >= 3))
            | ((j == 1) & (vd >= 1) & (tc >= 3))
            | (tc >= 4)
        )
    elif position == 3:
        # Pos 3: protection, lower bar
        bid = (
            (ts >= 11)
            | ((ts >= 9) & (tc >= 3))
            | ((j == 1) & (tc >= 2))
            | ((vd >= 1) & (ts >= 8))
        )
    else:  # pos4
        bid = (
            (ts >= 13)
            | ((ts >= 10) & (tc >= 3))
            | ((j == 1) & (vd >= 1) & (tc >= 2))
            | (tc >= 4)
        )
    return bid.astype(int).values


def rule_partner80(df: pd.DataFrame) -> np.ndarray:
    """Partner bid 80 — respond unless truly dead."""
    ps = df["partner_support"]
    ts = df["best_trump_score"]
    tc = df["best_trump_count"]

    # Pass only if: ≤1 card in partner's suit AND score < 12 AND length < 3
    pass_hand = (ps <= 1) & (ts < 12) & (tc < 3)
    return (~pass_hand).astype(int).values


def rule_opp80_active(df: pd.DataFrame) -> np.ndarray:
    """Opp bid 80 — be active if we have something tangible."""
    ts = df["best_trump_score"]
    tc = df["best_trump_count"]
    is_opp = df["best_is_opp_suit"]
    osc = df["opp_suit_cards"]
    second_ts = df["second_trump_score"]
    sl = df["best_side_length"]

    active = (
        # Own good non-opp suit
        ((ts >= 14) & (is_opp == 0))
        # Our best IS opp's: look at second
        | ((is_opp == 1) & (second_ts >= 14))
        # Long side suit (distribution)
        | (sl >= 4)
        # Coinche territory: lots of cards in opp's trump
        | (osc >= 4)
    )
    return active.astype(int).values


def rule_opp80_coinche(df: pd.DataFrame) -> np.ndarray:
    """Coinche: we hold opp's trump cards. Target: 14.9% firing rate."""
    osc = df["opp_suit_cards"]
    ts = df["best_trump_score"]

    # Coinche when many cards in opp's trump AND our own trump_score is low
    coinche = (osc >= 4) | ((osc >= 3) & (ts < 12))
    return coinche.astype(int).values


# ============================================================
# Runner
# ============================================================
def run_group(best: pd.DataFrame, scenarios: list[str], rule_fn, target_col: str, label: str,
              show_samples: bool = False):
    mask = best["scenario"].isin(scenarios)
    g = best[mask].reset_index(drop=True)
    if len(g) == 0:
        print(f"[{label}] empty group, skip")
        return None
    truth = g[target_col].values
    pred = rule_fn(g)
    print(f"\n=== {label} ===", flush=True)
    print(f"  scenarios={sorted(g['scenario'].unique())}", flush=True)
    print(f"  NN {target_col} rate: {truth.mean():.1%}", flush=True)
    res = agreement(pred, truth, label)
    if show_samples:
        feats = ["scenario", "best_trump_score", "best_trump_count", "best_has_jack",
                 "best_is_opp_suit", "second_trump_score", "second_trump_count",
                 "side_voids", "total_aces", "partner_support", "opp_suit_cards",
                 "best_side_length", "nn_action"]
        show_disagreement_samples(g, pred, truth, feats, label)
    return res


def main():
    best = load_per_deal(CSV)

    results = []

    # Opening
    results.append(run_group(
        best, ["pos1_open"], rule_opening, "nn_bids",
        "Opening (pos1)"
    ))

    # Pos 2
    results.append(run_group(
        best, ["pos2_after_pass"],
        lambda d: rule_after_pass(d, 2), "nn_bids",
        "Pos 2 after 1 pass"
    ))

    # Pos 3
    results.append(run_group(
        best, ["pos3_after_2p"],
        lambda d: rule_after_pass(d, 3), "nn_bids",
        "Pos 3 after 2 passes"
    ))

    # Pos 4
    results.append(run_group(
        best, ["pos4_after_3p"],
        lambda d: rule_after_pass(d, 4), "nn_bids",
        "Pos 4 after 3 passes"
    ))

    # Partner 80
    p80_scen = [f"pos3_partner80_{s}" for s in "shdc"] + [f"pos4_partner80_{s}" for s in "shdc"]
    results.append(run_group(
        best, p80_scen, rule_partner80, "nn_bids",
        "Partner bid 80"
    ))

    # Opp 80 active — broken down by position
    for pos in [2, 3, 4]:
        o80_pos_scen = [f"pos{pos}_opp80_{s}" for s in "shdc"]
        results.append(run_group(
            best, o80_pos_scen, rule_opp80_active, "nn_active",
            f"Opp 80 — active, pos{pos}",
            show_samples=(pos == 2)
        ))

    # Opp 80 coinche aggregated
    o80_scen = (
        [f"pos2_opp80_{s}" for s in "shdc"]
        + [f"pos3_opp80_{s}" for s in "shdc"]
        + [f"pos4_opp80_{s}" for s in "shdc"]
    )
    results.append(run_group(
        best, o80_scen, rule_opp80_coinche, "nn_coinche",
        "Opp bid 80 — coinche (subset)"
    ))

    print("\n\n=== SUMMARY ===")
    print(f"{'Rule':<40} {'n':>10} {'acc':>8} {'miss':>8} {'FA':>8}")
    print("-" * 78)
    for r in results:
        if r:
            print(f"{r['label']:<40} {r['n']:>10,} {r['acc']:>8.1%} {r['miss']:>8.1%} {r['fa']:>8.1%}")


if __name__ == "__main__":
    main()
