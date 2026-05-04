"""
Categorize each decision into a decision-type taxonomy and report frequencies.

Mutually exclusive types, derived from per-row state:

  LEAD_OPENING        play_idx=0, trick_idx=0
  LEAD_MID            play_idx=0, trick_idx>0
  FOLLOW_PARTNER_WINS play_idx>0, can_follow_lead=1, partner_winning=1
  FOLLOW_OPP_TAKEABLE play_idx>0, can_follow_lead=1, partner_winning=0, holds_master_lead=1
  FOLLOW_OPP_DUCK     play_idx>0, can_follow_lead=1, partner_winning=0, holds_master_lead=0
  TRUMP_FOLLOW        play_idx>0, lead_suit==trump, can_follow_lead=1
                       (subset of FOLLOW_*; we compute it separately and prefer it)
  CUT_OR_DUCK_OPP     play_idx>0, can_follow_lead=0, lead_suit≠trump,
                        partner_winning=0, has_trump
  CUT_OR_DUCK_PARTNER play_idx>0, can_follow_lead=0, lead_suit≠trump,
                        partner_winning=1, has_trump
  DISCARD_NO_TRUMP    play_idx>0, can_follow_lead=0, no trump in hand
  OTHER               unclassified

Adds a `decision_type` int8 column (0..N-1) and saves the enriched npz.
"""

import argparse
import sys
from pathlib import Path

import numpy as np


TYPE_NAMES = [
    "LEAD_OPENING",       # 0
    "LEAD_MID",           # 1
    "FOLLOW_PARTNER_WINS", # 2
    "FOLLOW_OPP_TAKEABLE", # 3
    "FOLLOW_OPP_DUCK",     # 4
    "TRUMP_FOLLOW",        # 5
    "CUT_OR_DUCK_OPP",     # 6
    "CUT_OR_DUCK_PARTNER", # 7
    "DISCARD_NO_TRUMP",    # 8
    "OTHER",               # 9
]


def categorize(d):
    n = len(d["deal_id"])
    play_idx = d["play_idx"]
    trick_idx = d["trick_idx"]
    can_follow = d["can_follow_lead"].astype(bool)
    is_trump_led = d["is_trump_led"].astype(bool)
    partner_winning = d["partner_winning"].astype(bool)
    holds_master_lead = d["holds_master_lead"].astype(bool)
    trump_count = d["trump_count"]
    can_cut = d["can_cut"].astype(bool)
    is_lead = play_idx == 0

    out = np.full(n, 9, dtype=np.int8)  # OTHER

    # Lead categories first (they are mutually exclusive with everything else)
    out[is_lead & (trick_idx == 0)] = 0  # LEAD_OPENING
    out[is_lead & (trick_idx > 0)] = 1   # LEAD_MID

    # Following suit (incl. trump-led — special-cased to TRUMP_FOLLOW)
    follow_mask = (~is_lead) & can_follow
    trump_follow = follow_mask & is_trump_led
    out[trump_follow] = 5  # TRUMP_FOLLOW

    follow_nontrump = follow_mask & (~is_trump_led)
    out[follow_nontrump & partner_winning] = 2                       # FOLLOW_PARTNER_WINS
    out[follow_nontrump & (~partner_winning) & holds_master_lead] = 3  # FOLLOW_OPP_TAKEABLE
    out[follow_nontrump & (~partner_winning) & (~holds_master_lead)] = 4  # FOLLOW_OPP_DUCK

    # Void in lead suit. Includes both "side led + I'm void" and
    # "trump led + I'm void in trump" — the latter is a pure discard situation.
    void_mask = (~is_lead) & (~can_follow)
    has_trump = trump_count > 0
    # CUT_OR_DUCK_* only apply when lead is a side suit (you're cutting *something*).
    # When trump itself is led and you can't follow, it's a discard.
    side_led_void = void_mask & (~is_trump_led)
    out[side_led_void & has_trump & (~partner_winning)] = 6   # CUT_OR_DUCK_OPP
    out[side_led_void & has_trump & partner_winning] = 7      # CUT_OR_DUCK_PARTNER
    out[void_mask & (~has_trump)] = 8                          # DISCARD_NO_TRUMP
    # Edge case: side_led_void with has_trump=False is already DISCARD_NO_TRUMP above.
    # Edge case: trump_led_void with has_trump=True is impossible (if trump led
    # and you can't follow, you have no trump).

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="data/distill/play_features_real.npz")
    ap.add_argument("--output", default=None,
                    help="Default: in-place; writes to <input> with decision_type added")
    args = ap.parse_args()

    in_path = Path(args.input)
    out_path = Path(args.output) if args.output else in_path

    print(f"Loading {in_path}...")
    d = dict(np.load(in_path))
    n = len(d["deal_id"])
    print(f"  {n:,} rows, {len(d)} columns")

    print("Categorizing...")
    dt = categorize(d)
    d["decision_type"] = dt

    # Frequency report
    counts = np.bincount(dt, minlength=len(TYPE_NAMES))
    print(f"\n=== Decision type frequencies ({n:,} rows) ===")
    print(f"  {'#':>3}  {'type':<22}  {'n':>10}   {'%':>6}  {'avg n_legal':>11}")
    for i, name in enumerate(TYPE_NAMES):
        c = int(counts[i])
        if c == 0:
            continue
        pct = c / n * 100
        avg_legal = float(d["n_legal"][dt == i].mean()) if c else 0
        bar = "█" * int(pct * 1.2)
        print(f"  {i:>3}  {name:<22}  {c:>10,}   {pct:>5.1f}%  {avg_legal:>10.2f}  {bar}")

    print(f"\nSaving to {out_path}...")
    np.savez_compressed(out_path, **d)
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"  saved ({sz:.1f} MB)")


if __name__ == "__main__":
    sys.exit(main())
