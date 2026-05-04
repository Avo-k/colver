"""
Filter play_features.npz to (deal, suit) pairs that the NN bidder would have
actually taken, with NS as declarer (since our setup_dd hard-codes team=0 NS).

Optional: only keep made contracts (final_ns_pts >= contract_value * 10).

Output: data/distill/play_features_real.npz  (subset of input)
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/distill/play_features.npz")
    ap.add_argument("--bids", default="data/distill/bid_pool_200k.csv")
    ap.add_argument("--output", default="data/distill/play_features_real.npz")
    ap.add_argument("--made-only", action="store_true",
                    help="Keep only deals where contract was made (NS pts >= contract value)")
    ap.add_argument("--ns-only", action="store_true", default=True,
                    help="Keep only deals where NS is declarer (default; required to match setup_dd)")
    ap.add_argument("--all-teams", action="store_true",
                    help="Keep both NS and EW declarer deals (advanced; unflips not implemented)")
    args = ap.parse_args()

    print(f"Loading {args.features}...")
    d = dict(np.load(args.features))
    n = len(d["deal_id"])
    print(f"  {n:,} rows")

    print(f"Loading {args.bids}...")
    bids = pd.read_csv(args.bids)
    print(f"  {len(bids):,} bid records")
    print(f"  passed: {(bids['passed']==1).sum():,}")
    print(f"  NS declarer: {(bids['declarer_seat']%2 == 0).sum():,}")
    print(f"  EW declarer: {(bids['declarer_seat']%2 == 1).sum():,}")

    # Build a map deal_id -> (chosen_suit, value, declarer_seat, ns_declarer)
    bids = bids[bids["passed"] == 0].copy()
    if not args.all_teams:
        bids = bids[bids["declarer_seat"] % 2 == 0].copy()
        print(f"  filtered to NS-declarer: {len(bids):,}")

    deal_to_suit = dict(zip(bids["deal_id"].astype(int), bids["trump_suit"].astype(int)))
    deal_to_value = dict(zip(bids["deal_id"].astype(int), bids["value"].astype(int)))

    # Build per-row keep mask
    deal_id = d["deal_id"]
    forced_suit = d["forced_suit"]
    final_ns_pts = d["final_ns_pts"]

    print("Building keep mask...")
    chosen_suit_per_row = np.full(n, -1, dtype=np.int8)
    contract_value_per_row = np.full(n, 0, dtype=np.int16)
    # Vectorize via mapping
    unique_deals = np.unique(deal_id)
    print(f"  unique deals in features: {len(unique_deals):,}")
    # Build lookup arrays sized to max deal_id+1
    max_deal = int(deal_id.max()) + 1
    suit_lookup = np.full(max_deal, -1, dtype=np.int8)
    value_lookup = np.zeros(max_deal, dtype=np.int16)
    for did, s in deal_to_suit.items():
        if did < max_deal:
            suit_lookup[did] = s
            value_lookup[did] = deal_to_value[did]
    chosen_suit_per_row = suit_lookup[deal_id]
    contract_value_per_row = value_lookup[deal_id]

    keep = (chosen_suit_per_row == forced_suit.astype(np.int8)) & (chosen_suit_per_row >= 0)
    print(f"  rows matching (deal, suit) bidder pick + NS-declarer: {keep.sum():,}")

    if args.made_only:
        contract_pts_needed = (contract_value_per_row.astype(np.int32) * 10).clip(min=82)
        # 250 (capot) needs 252 (with der). For simplicity just *10. Refine later.
        made = final_ns_pts.astype(np.int32) >= contract_pts_needed
        print(f"  made contracts (NS pts >= contract value): {(keep & made).sum():,}")
        keep = keep & made

    print(f"\nKeeping {keep.sum():,} / {n:,} rows ({keep.sum()/n*100:.1f}%)")

    out = {k: v[keep] for k, v in d.items()}
    # Add contract_value as a new feature column
    out["contract_value"] = contract_value_per_row[keep]

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_path, **out)
    sz = out_path.stat().st_size / 1024 / 1024
    print(f"Saved {out_path} ({sz:.1f} MB)")

    # Quick summary on the filtered set
    print(f"\n=== Filtered subset summary ===")
    print(f"  unique games kept: {len(np.unique(deal_id[keep] * 4 + forced_suit[keep])):,}")
    print(f"  decisions/game: {keep.sum() / max(1, len(np.unique(deal_id[keep] * 4 + forced_suit[keep]))):.1f}")
    cv = contract_value_per_row[keep]
    print(f"  contract values in kept set: ", dict(zip(*np.unique(cv, return_counts=True))))


if __name__ == "__main__":
    sys.exit(main())
