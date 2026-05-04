"""
play_lab.py — modular iteration loop for IS-DD play interpretability.

Goal of each run: find the SMALLEST feature set + SHALLOWEST tree that comes
within `--eps` of the full-feature XGBoost baseline accuracy. The algorithm
drives the choices, not human judgment.

Pipeline (per (scenario, target)):
  1. Train baseline XGBoost on ALL features → accuracy A0 (val split)
  2. Forward greedy feature selection: at step k, add the candidate feature
     that maximizes validation accuracy on top of the running set. Stop when
     adding any feature gives < `--gain-eps` improvement.
  3. With the minimal set found in (2), sweep max_depth ∈ {1,2,3,5,8} for both
     XGBoost and a single DecisionTreeClassifier. Report the smallest depth
     within `--eps` of A0.
  4. Print the best small tree as human-readable text.

Add a new scenario:    @scenario("name") def name(d): return mask
Add a new target:      @target("name", scenario="...") def name(d, mask): return y
Add a new feature:     register in build_feature_columns() (or already in .npz)

CLI:
  python play_lab.py --list                # list scenarios & targets
  python play_lab.py --scenario declarer_lead --target led_trump
  python play_lab.py --scenario declarer_lead --target led_trump --eps 0.005
  python play_lab.py --all                 # run every (scenario,target) combo
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split

# ============================================================================
# Registries
# ============================================================================

ScenarioFn = Callable[[Dict[str, np.ndarray]], np.ndarray]
TargetFn = Callable[[Dict[str, np.ndarray], np.ndarray], np.ndarray]

SCENARIOS: Dict[str, ScenarioFn] = {}
TARGETS: Dict[Tuple[str, str], TargetFn] = {}  # (scenario, target) -> fn


def scenario(name: str):
    def deco(fn: ScenarioFn):
        SCENARIOS[name] = fn
        return fn
    return deco


def target(name: str, *, scenario_name: str):
    def deco(fn: TargetFn):
        TARGETS[(scenario_name, name)] = fn
        return fn
    return deco


# ============================================================================
# Scenario filters
# ============================================================================

@scenario("declarer_lead")
def _s1(d):
    return (d["trick_idx"] == 0) & (d["play_idx"] == 0) & (d["is_declarer_team"] == 1)


@scenario("defender_lead")
def _s2(d):
    return (d["trick_idx"] == 0) & (d["play_idx"] == 0) & (d["is_declarer_team"] == 0)


@scenario("any_lead_t0")
def _s3(d):
    return (d["trick_idx"] == 0) & (d["play_idx"] == 0)


@scenario("can_cut")
def _s4(d):
    # void in lead suit, lead != trump, holds trump
    return (d["can_cut"] == 1) & (d["play_idx"] > 0)


@scenario("could_overcut")
def _s5(d):
    # trick has trump and we have higher trump than current best on table
    return (d["has_higher_trump_than_trick"] == 1) & (d["play_idx"] > 0)


@scenario("declarer_lead_mid")
def _s6(d):
    # declarer leading mid-game (trick 1..4)
    return (d["play_idx"] == 0) & (d["is_declarer_team"] == 1) & \
           (d["trick_idx"] >= 1) & (d["trick_idx"] <= 4)


@scenario("late_game")
def _s7(d):
    # any decision in tricks 5-7 (endgame)
    return d["trick_idx"] >= 5


# Per decision_type scenarios (require decision_types.py to have run)
def _dt_eq(d, k):
    if "decision_type" not in d:
        return np.zeros(len(d["deal_id"]), dtype=bool)
    return d["decision_type"] == k

@scenario("dt_lead_opening")
def _dt0(d): return _dt_eq(d, 0)

@scenario("dt_lead_mid")
def _dt1(d): return _dt_eq(d, 1)

@scenario("dt_follow_partner_wins")
def _dt2(d): return _dt_eq(d, 2)

@scenario("dt_follow_opp_takeable")
def _dt3(d): return _dt_eq(d, 3)

@scenario("dt_follow_opp_duck")
def _dt4(d): return _dt_eq(d, 4)

@scenario("dt_trump_follow")
def _dt5(d): return _dt_eq(d, 5)

@scenario("dt_cut_or_duck_opp")
def _dt6(d): return _dt_eq(d, 6)

@scenario("dt_cut_or_duck_partner")
def _dt7(d): return _dt_eq(d, 7)

@scenario("dt_discard_no_trump")
def _dt8(d): return _dt_eq(d, 8)


# ============================================================================
# Target functions (binary)
# ============================================================================

def _card_suit(c): return (c // 8).astype(np.int8)
def _card_rank(c): return (c % 8).astype(np.int8)


@target("led_trump", scenario_name="declarer_lead")
def _t_led_trump_decl(d, mask):
    return (_card_suit(d["chosen"][mask]) == d["forced_suit"][mask]).astype(np.uint8)


@target("led_trump", scenario_name="defender_lead")
def _t_led_trump_def(d, mask):
    return (_card_suit(d["chosen"][mask]) == d["forced_suit"][mask]).astype(np.uint8)


@target("led_trump", scenario_name="any_lead_t0")
def _t_led_trump_any(d, mask):
    return (_card_suit(d["chosen"][mask]) == d["forced_suit"][mask]).astype(np.uint8)


@target("led_trump", scenario_name="declarer_lead_mid")
def _t_led_trump_mid(d, mask):
    return (_card_suit(d["chosen"][mask]) == d["forced_suit"][mask]).astype(np.uint8)


@target("led_ace_offsuit", scenario_name="declarer_lead")
def _t_ace_off_decl(d, mask):
    cs = _card_suit(d["chosen"][mask])
    cr = _card_rank(d["chosen"][mask])
    return ((cs != d["forced_suit"][mask]) & (cr == 7)).astype(np.uint8)


@target("led_ace_offsuit", scenario_name="defender_lead")
def _t_ace_off_def(d, mask):
    cs = _card_suit(d["chosen"][mask])
    cr = _card_rank(d["chosen"][mask])
    return ((cs != d["forced_suit"][mask]) & (cr == 7)).astype(np.uint8)


@target("led_low_seven", scenario_name="declarer_lead")
def _t_seven_decl(d, mask):
    return (_card_rank(d["chosen"][mask]) == 0).astype(np.uint8)


@target("led_low_seven", scenario_name="defender_lead")
def _t_seven_def(d, mask):
    return (_card_rank(d["chosen"][mask]) == 0).astype(np.uint8)


@target("did_cut", scenario_name="can_cut")
def _t_did_cut(d, mask):
    chosen = d["chosen"][mask]
    return (_card_suit(chosen) == d["forced_suit"][mask]).astype(np.uint8)


@target("did_overcut", scenario_name="could_overcut")
def _t_did_overcut(d, mask):
    """Did the player play a trump (which by has_higher_trump precondition will
    overtake the trick) — vs. discarding low trump or off-suit junk."""
    chosen = d["chosen"][mask]
    return (_card_suit(chosen) == d["forced_suit"][mask]).astype(np.uint8)


@target("led_trump_jack", scenario_name="declarer_lead_mid")
def _t_lead_jack_mid(d, mask):
    cs = _card_suit(d["chosen"][mask])
    cr = _card_rank(d["chosen"][mask])
    return ((cs == d["forced_suit"][mask]) & (cr == 3)).astype(np.uint8)


# ============================================================================
# Type-specific targets (per decision_type scenario)
# ============================================================================

def _is_played_trump(d, mask):
    return (_card_suit(d["chosen"][mask]) == d["forced_suit"][mask])

def _is_played_master_of_chosen_suit(d, mask):
    """Chosen card is the highest remaining (in plain rank) in its suit,
    relative to outstanding (= ~played_cards)."""
    chosen = d["chosen"][mask]
    played = d["played_cards"][mask]
    suit = _card_suit(chosen).astype(np.uint8)
    rank = _card_rank(chosen).astype(np.int8)
    outstanding = (~played) & 0xFFFFFFFF
    suit_outs = ((outstanding >> (suit.astype(np.uint32) * 8)) & 0xFF).astype(np.uint8)
    out = np.zeros(len(chosen), dtype=bool)
    for r in range(7, -1, -1):
        is_present = (suit_outs >> r) & 1 == 1
        match = is_present & (rank == r) & (~out)
        higher = is_present & (rank < r) & (~out)
        out[match] = True  # found ourself as highest
        out[higher] = False  # higher exists, we're not master (but mark "found")
        # We need to short-circuit. Rebuild: highest_present_rank
    # Simpler: compute highest-present rank explicitly
    out2 = np.zeros(len(chosen), dtype=bool)
    highest_rank = np.full(len(chosen), -1, dtype=np.int8)
    for r in range(7, -1, -1):
        is_present = (suit_outs >> r) & 1 == 1
        new_find = is_present & (highest_rank == -1)
        highest_rank[new_find] = r
    out2 = (rank == highest_rank)
    return out2.astype(np.uint8)

# --- LEAD targets (apply to LEAD_OPENING and LEAD_MID) ---

for sc_name in ["dt_lead_opening", "dt_lead_mid"]:
    @target("led_trump", scenario_name=sc_name)
    def _t(d, mask, _bind=sc_name):
        return _is_played_trump(d, mask).astype(np.uint8)

    @target("led_ace_offsuit", scenario_name=sc_name)
    def _t(d, mask, _bind=sc_name):
        cs = _card_suit(d["chosen"][mask])
        cr = _card_rank(d["chosen"][mask])
        return ((cs != d["forced_suit"][mask]) & (cr == 7)).astype(np.uint8)

    @target("led_master", scenario_name=sc_name)
    def _t(d, mask, _bind=sc_name):
        return _is_played_master_of_chosen_suit(d, mask)

    @target("led_low", scenario_name=sc_name)
    def _t(d, mask, _bind=sc_name):
        return (_card_rank(d["chosen"][mask]) <= 1).astype(np.uint8)


# --- FOLLOW_* targets ---

for sc_name in ["dt_follow_partner_wins", "dt_follow_opp_takeable", "dt_follow_opp_duck"]:
    @target("played_high", scenario_name=sc_name)
    def _t(d, mask, _bind=sc_name):
        return (_card_rank(d["chosen"][mask]) >= 6).astype(np.uint8)

    @target("played_low", scenario_name=sc_name)
    def _t(d, mask, _bind=sc_name):
        return (_card_rank(d["chosen"][mask]) <= 1).astype(np.uint8)

# Master target only for FOLLOW_OPP_TAKEABLE (where it's a real choice)
@target("played_master", scenario_name="dt_follow_opp_takeable")
def _t_follow_master(d, mask):
    return _is_played_master_of_chosen_suit(d, mask)


# --- TRUMP_FOLLOW targets ---

@target("played_jack_or_nine", scenario_name="dt_trump_follow")
def _t_jack_nine(d, mask):
    cr = _card_rank(d["chosen"][mask])
    return ((cr == 3) | (cr == 2)).astype(np.uint8)

@target("played_max_trump", scenario_name="dt_trump_follow")
def _t_max_trump(d, mask):
    """Highest trump (in trump-strength order) among legal cards == chosen?"""
    # We don't have legal cards for each row materialized as a list; use legal mask
    chosen = d["chosen"][mask]
    legal = d["legal"][mask]
    trump = d["forced_suit"][mask]
    # Trump strength order: rank 3>2>7>6>5>4>1>0
    trump_strength_by_rank = np.array([0, 1, 6, 7, 2, 3, 4, 5], dtype=np.int8)
    # For each row: find max trump strength among legal trump cards
    n = len(chosen)
    legal_trump = ((legal >> (trump.astype(np.uint32) * 8)) & 0xFF).astype(np.uint8)
    max_str = np.full(n, -1, dtype=np.int8)
    best_rank = np.full(n, -1, dtype=np.int8)
    for r in [3, 2, 7, 6, 5, 4, 1, 0]:
        present = (legal_trump >> r) & 1 == 1
        sel = present & (max_str == -1)
        max_str[sel] = trump_strength_by_rank[r]
        best_rank[sel] = r
    chosen_suit = _card_suit(chosen)
    chosen_rank = _card_rank(chosen)
    is_max = (chosen_suit == trump) & (chosen_rank == best_rank)
    return is_max.astype(np.uint8)


# --- CUT_OR_DUCK_* targets ---

@target("did_cut", scenario_name="dt_cut_or_duck_opp")
def _t_cut_opp(d, mask):
    return _is_played_trump(d, mask).astype(np.uint8)

@target("did_cut", scenario_name="dt_cut_or_duck_partner")
def _t_cut_partner(d, mask):
    return _is_played_trump(d, mask).astype(np.uint8)


# --- DISCARD_NO_TRUMP targets ---

@target("discarded_low", scenario_name="dt_discard_no_trump")
def _t_disc_low(d, mask):
    return (_card_rank(d["chosen"][mask]) <= 1).astype(np.uint8)

@target("discarded_ace_or_ten", scenario_name="dt_discard_no_trump")
def _t_disc_high(d, mask):
    return (_card_rank(d["chosen"][mask]) >= 6).astype(np.uint8)


# ============================================================================
# Feature universe
# ============================================================================

# Default predictor pool: every numeric column from the .npz that is NOT an
# action / outcome / identifier.
EXCLUDED_FROM_PREDICTORS = {
    "deal_id", "forced_suit", "dealer", "trick_idx", "play_idx", "seat",
    "trick_lead", "chosen", "n_legal", "final_ns_pts",
    "hand", "legal", "played_cards", "trick_packed", "voids_packed",
    "partner_seat", "trick_winner_so_far_seat",
    # Q-summary leaks the action — exclude
    "q_chosen", "q_max", "q_min", "q_2nd_max", "q_2nd_min",
    "q_margin", "q_chosen_vs_best",
    # decision_type is the scenario filter itself
    "decision_type",
}


def get_predictor_columns(d: Dict[str, np.ndarray]) -> List[str]:
    cols = []
    for k, v in d.items():
        if k in EXCLUDED_FROM_PREDICTORS:
            continue
        if v.ndim != 1:
            continue
        if v.dtype.kind not in "uifb":
            continue
        cols.append(k)
    return sorted(cols)


def make_X(d: Dict[str, np.ndarray], mask: np.ndarray, cols: List[str]) -> np.ndarray:
    arrs = [d[c][mask].astype(np.float32) for c in cols]
    return np.stack(arrs, axis=1)


# ============================================================================
# Model fits
# ============================================================================

def _auc(clf, X, y):
    """AUC-ROC on validation. Robust to single-class predictions."""
    if hasattr(clf, "predict_proba"):
        p = clf.predict_proba(X)[:, 1]
    else:
        p = clf.predict(X).astype(float)
    if len(np.unique(y)) < 2:
        return float("nan")
    return float(roc_auc_score(y, p))


def fit_xgb(Xtr, ytr, Xval, yval, n_estimators=200, max_depth=5):
    import xgboost as xgb
    clf = xgb.XGBClassifier(
        n_estimators=n_estimators, max_depth=max_depth,
        learning_rate=0.1, tree_method="hist", n_jobs=-1, random_state=0,
        eval_metric="logloss",
    )
    clf.fit(Xtr, ytr)
    return clf, _auc(clf, Xval, yval)


def fit_dt(Xtr, ytr, Xval, yval, max_depth=3, min_samples_leaf=1000):
    dt = DecisionTreeClassifier(
        max_depth=max_depth, min_samples_leaf=min_samples_leaf,
        class_weight="balanced", random_state=0,
    )
    dt.fit(Xtr, ytr)
    return dt, _auc(dt, Xval, yval)


# ============================================================================
# Forward feature selection
# ============================================================================

def forward_select(d, mask, target_y, all_cols, *, gain_eps=0.001, baseline_acc=None,
                   max_features=8, val_size=0.2, sample_n=None, seed=0):
    """Greedy forward selection. Adds one feature at a time by validation
    accuracy. Stops when no candidate improves by >= gain_eps OR max_features
    reached. Returns ordered list of (k, added_feature, val_acc, full_set)."""
    Xall = make_X(d, mask, all_cols)
    y = target_y
    if sample_n is not None and len(y) > sample_n:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(y), sample_n, replace=False)
        Xall, y = Xall[idx], y[idx]
        print(f"  (subsampled to {sample_n:,} for forward selection)")

    Xtr, Xval, ytr, yval = train_test_split(Xall, y, test_size=val_size, random_state=seed, stratify=y)

    selected: List[int] = []
    history: List[Tuple[int, str, float]] = []
    last_score = 0.5

    while len(selected) < max_features and len(selected) < len(all_cols):
        candidates = [i for i in range(len(all_cols)) if i not in selected]
        best_score = -1.0
        best_idx = -1
        # Feasibility: 80-tree XGB × n_candidates × ~0.4s ≈ 15s/round
        for c in candidates:
            cols_idx = selected + [c]
            _, sc = fit_xgb(Xtr[:, cols_idx], ytr, Xval[:, cols_idx], yval,
                            n_estimators=80, max_depth=4)
            if sc > best_score:
                best_score = sc
                best_idx = c
        gain = best_score - last_score
        if len(selected) > 0 and gain < gain_eps:
            print(f"    plateau at k={len(selected)} (next would gain {gain:+.4f}, eps={gain_eps})")
            break
        selected.append(best_idx)
        last_score = best_score
        added = all_cols[best_idx]
        history.append((len(selected), added, best_score))
        print(f"    k={len(selected):>2}  + {added:<30}  val_auc={best_score:.4f}  "
              f"({100*(best_score - (baseline_acc or 0)):+.2f}pp vs baseline)")
    return [all_cols[i] for i in selected], history


# ============================================================================
# Main run for one (scenario, target)
# ============================================================================

def run(d, scenario_name: str, target_name: str, *, eps=0.005, gain_eps=0.001,
        max_features=8, val_size=0.2, sample_n=None):
    sc_fn = SCENARIOS[scenario_name]
    key = (scenario_name, target_name)
    if key not in TARGETS:
        raise ValueError(f"No target '{target_name}' registered for scenario '{scenario_name}'. "
                         f"Available: {sorted(TARGETS.keys())}")
    tg_fn = TARGETS[key]

    print("=" * 72)
    print(f"  SCENARIO: {scenario_name}    TARGET: {target_name}")
    print("=" * 72)

    mask = sc_fn(d)
    n = int(mask.sum())
    if n < 1000:
        print(f"  too few rows ({n}) — skipping")
        return None
    y = tg_fn(d, mask)
    pos_rate = y.mean()
    print(f"  rows: {n:,}    positive rate: {pos_rate*100:.2f}%")

    all_cols = get_predictor_columns(d)
    print(f"  feature pool: {len(all_cols)} columns")

    Xall = make_X(d, mask, all_cols)
    Xtr, Xval, ytr, yval = train_test_split(Xall, y, test_size=val_size, random_state=0, stratify=y)

    # 1. Baseline (AUC; majority-class AUC = 0.5 by construction)
    print("\n  [1] BASELINE (all features, XGBoost depth=5, n=200)")
    t0 = time.time()
    _, baseline_acc = fit_xgb(Xtr, ytr, Xval, yval, n_estimators=200, max_depth=5)
    print(f"      val AUC = {baseline_acc:.4f}   (random baseline = 0.5)")
    print(f"      fit time {time.time()-t0:.1f}s")

    # 2. Forward selection
    print(f"\n  [2] FORWARD SELECTION (gain_eps={gain_eps}, max_k={max_features})")
    selected, history = forward_select(
        d, mask, y, all_cols, gain_eps=gain_eps, baseline_acc=baseline_acc,
        max_features=max_features, val_size=val_size, sample_n=sample_n,
    )

    # 3. Depth sweep on minimal set
    print(f"\n  [3] DEPTH SWEEP on {len(selected)} features = {selected}")
    Xs_tr = make_X(d, mask, selected)
    # Re-split with same seed
    Xs_tr_, Xs_val_, ys_tr_, ys_val_ = train_test_split(Xs_tr, y, test_size=val_size, random_state=0, stratify=y)
    print(f"  {'depth':>5}  {'XGB AUC':>9}  {'DT AUC':>9}  {'within eps':>11}")
    print(f"  {'-'*5}  {'-'*9}  {'-'*9}  {'-'*11}")
    rows_complexity = []
    for depth in [1, 2, 3, 5, 8]:
        _, axgb = fit_xgb(Xs_tr_, ys_tr_, Xs_val_, ys_val_, n_estimators=200, max_depth=depth)
        _, adt = fit_dt(Xs_tr_, ys_tr_, Xs_val_, ys_val_, max_depth=depth, min_samples_leaf=1000)
        gap = baseline_acc - axgb
        flag = "✓" if gap <= eps else " "
        print(f"  {depth:>5}  {axgb:>9.4f}  {adt:>9.4f}  {flag:>11}")
        rows_complexity.append((depth, axgb, adt, gap))

    # 4. Print human tree at smallest depth that's within eps of baseline
    smallest_ok = None
    for depth, axgb, adt, gap in rows_complexity:
        if gap <= eps:
            smallest_ok = depth
            break
    if smallest_ok is None:
        smallest_ok = 5
        print(f"\n  No depth reached baseline-{eps:.3f}; using depth=5 for tree dump")

    dt, dt_acc = fit_dt(Xs_tr_, ys_tr_, Xs_val_, ys_val_, max_depth=smallest_ok, min_samples_leaf=1000)
    print(f"\n  [4] DECISION TREE (depth={smallest_ok}, {len(selected)} features) val_AUC={dt_acc:.4f}")
    text = export_text(dt, feature_names=selected, max_depth=smallest_ok, show_weights=True)
    for line in text.splitlines():
        print(f"    {line}")

    return {
        "scenario": scenario_name,
        "target": target_name,
        "n": n,
        "pos_rate": pos_rate,
        "baseline_acc": baseline_acc,
        "selected": selected,
        "history": history,
        "complexity": rows_complexity,
        "best_depth": smallest_ok,
        "dt_acc": dt_acc,
    }


# ============================================================================
# CLI
# ============================================================================

def list_combos():
    print("Scenarios:")
    for s in sorted(SCENARIOS):
        print(f"  - {s}")
    print("\nTargets (scenario :: target):")
    for (s, t) in sorted(TARGETS.keys()):
        print(f"  - {s} :: {t}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/distill/play_features.npz")
    ap.add_argument("--scenario", default=None)
    ap.add_argument("--target", default=None)
    ap.add_argument("--all", action="store_true",
                    help="Run every registered (scenario, target) combo")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--eps", type=float, default=0.005,
                    help="Acceptable gap from baseline acc (default 0.005)")
    ap.add_argument("--gain-eps", type=float, default=0.001,
                    help="Min forward-selection gain to keep adding (default 0.001)")
    ap.add_argument("--max-features", type=int, default=8)
    ap.add_argument("--sample-n", type=int, default=200_000,
                    help="Subsample size for forward selection (None = full)")
    args = ap.parse_args()

    if args.list:
        list_combos()
        return 0

    print(f"Loading {args.input}...")
    d = dict(np.load(args.input))
    print(f"  {len(d['deal_id']):,} rows, {len(d)} columns")

    runs = []
    if args.all:
        runs = sorted(TARGETS.keys())
    else:
        if not args.scenario or not args.target:
            print("Specify --scenario AND --target, or --all, or --list", file=sys.stderr)
            return 2
        runs = [(args.scenario, args.target)]

    summary = []
    for sc, tg in runs:
        try:
            res = run(d, sc, tg,
                      eps=args.eps, gain_eps=args.gain_eps,
                      max_features=args.max_features,
                      sample_n=args.sample_n)
            if res:
                summary.append(res)
        except Exception as e:
            import traceback
            print(f"  ERROR on {sc}/{tg}: {e}")
            traceback.print_exc()

    if len(summary) > 1:
        print("\n" + "=" * 72)
        print("  SUMMARY")
        print("=" * 72)
        print(f"  {'scenario':<22} {'target':<22} {'n':>8} {'baseline':>9} "
              f"{'k':>3} {'depth':>5} {'dt_acc':>7}")
        for r in summary:
            print(f"  {r['scenario']:<22} {r['target']:<22} {r['n']:>8,} "
                  f"{r['baseline_acc']:>9.4f} {len(r['selected']):>3} "
                  f"{r['best_depth']:>5} {r['dt_acc']:>7.4f}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
