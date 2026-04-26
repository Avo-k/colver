"""Analyse the audit_bid_v5 CSV and print a markdown report.

Usage:
    uv run python scripts/analysis/audit_bid_v5.py data/audit/audit_v5_100k.csv
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SUITS = ["S", "H", "D", "C"]
VALUE_LABEL = {i: f"{i*10}" for i in range(8, 17)}
VALUE_LABEL[25] = "capot"


def bucket_value(v: int) -> str:
    if v == 25:
        return "capot"
    if v <= 9:
        return "80-90"
    if v <= 11:
        return "100-110"
    if v <= 13:
        return "120-130"
    return "140-160"


def bucket_hand_strength(s: int) -> str:
    if s < 12:
        return "<12"
    if s < 16:
        return "12-15"
    if s < 20:
        return "16-19"
    if s < 25:
        return "20-24"
    return "25+"


def bucket_trump_count(n: int) -> str:
    if n <= 2:
        return "≤2"
    if n == 3:
        return "3"
    if n == 4:
        return "4"
    if n == 5:
        return "5"
    return "6+"


def fmt_row(label, n, regret, chute, made, extra=""):
    return f"| {label:<14} | {n:>7} | {regret:>6.1f} | {chute*100:>5.1f}% | {made*100:>5.1f}% | {extra}"


def main():
    csv = Path(sys.argv[1] if len(sys.argv) > 1 else "data/audit/audit_v5_100k.csv")
    df = pd.read_csv(csv)
    n = len(df)
    played = df[df.passed_out == 0].copy()
    n_played = len(played)

    print(f"# Audit v5_isdd (final = 25M) — {csv.name}")
    print()
    print(f"- deals audited: **{n:,}** (held-out slice, no training exposure)")
    print(f"- pass-outs: **{n - n_played}** ({(n-n_played)/n*100:.2f}%)")
    print(f"- made: **{played.made.sum():,}** ({played.made.mean()*100:.1f}%)")
    print(f"- chute: **{(~played.made.astype(bool)).sum():,}** ({(1-played.made.mean())*100:.1f}%)")
    print(f"- avg regret (declarer team): **{played.regret.mean():.1f}** pts")
    print(f"- median regret: {played.regret.median():.0f} pts")
    print(f"- % deals with regret = 0 (optimal): **{(played.regret == 0).mean()*100:.1f}%**")
    print(f"- % deals with regret ≥ 150: {(played.regret >= 150).mean()*100:.1f}%")
    print(f"- % deals with regret ≥ 300: {(played.regret >= 300).mean()*100:.1f}%")
    print()

    # === By position ===
    print("## Regret by declarer position")
    print("(position is the declarer's order in the auction: pos1 = first bidder after dealer)")
    print()
    print("| position | n | avg_regret | chute % | made % | avg_val |")
    print("|---|---|---|---|---|---|")
    for pos in sorted(played.decl_pos.unique()):
        g = played[played.decl_pos == pos]
        avg_v = (g.value.where(g.value != 25, 25) * (g.value != 25).astype(int) * 10 + (g.value == 25).astype(int) * 250).mean()
        print(f"| pos{pos} | {len(g):,} | {g.regret.mean():.1f} | {(1-g.made.mean())*100:.1f}% | {g.made.mean()*100:.1f}% | {avg_v:.0f} |")
    print()

    # === By chosen contract value ===
    print("## Regret by chosen contract value")
    print()
    print("| value_bucket | n | avg_regret | chute % | made % |")
    print("|---|---|---|---|---|")
    played["v_bucket"] = played.value.apply(bucket_value)
    order = ["80-90", "100-110", "120-130", "140-160", "capot"]
    for b in order:
        g = played[played.v_bucket == b]
        if len(g) == 0:
            continue
        print(f"| {b} | {len(g):,} | {g.regret.mean():.1f} | {(1-g.made.mean())*100:.1f}% | {g.made.mean()*100:.1f}% |")
    print()

    # === Chute rate by value ===
    print("## Chute rate by exact contract value")
    print()
    print("| value | n | chute % | made % | avg regret on chute |")
    print("|---|---|---|---|---|")
    for v in sorted(played.value.unique()):
        g = played[played.value == v]
        ch = g[g.made == 0]
        label = "capot" if v == 25 else str(v * 10)
        print(f"| {label} | {len(g):,} | {(1-g.made.mean())*100:.1f}% | {g.made.mean()*100:.1f}% | {ch.regret.mean() if len(ch) else 0:.0f} |")
    print()

    # === By hand strength of declarer ===
    print("## Regret by declarer hand strength (evaluate_for_trump on contract suit)")
    print()
    print("| strength | n | avg_regret | chute % | avg_value |")
    print("|---|---|---|---|---|")
    played["hs_bucket"] = played.decl_hand_strength.apply(bucket_hand_strength)
    for b in ["<12", "12-15", "16-19", "20-24", "25+"]:
        g = played[played.hs_bucket == b]
        if len(g) == 0:
            continue
        avg_v = (g.value.where(g.value != 25, 25) * (g.value != 25).astype(int) * 10 + (g.value == 25).astype(int) * 250).mean()
        print(f"| {b} | {len(g):,} | {g.regret.mean():.1f} | {(1-g.made.mean())*100:.1f}% | {avg_v:.0f} |")
    print()

    # === By trump count ===
    print("## Regret by declarer trump count (trump cards held)")
    print()
    print("| trump_count | n | avg_regret | chute % | avg_value |")
    print("|---|---|---|---|---|")
    played["tc_bucket"] = played.decl_trump_count.apply(bucket_trump_count)
    for b in ["≤2", "3", "4", "5", "6+"]:
        g = played[played.tc_bucket == b]
        if len(g) == 0:
            continue
        avg_v = (g.value.where(g.value != 25, 25) * (g.value != 25).astype(int) * 10 + (g.value == 25).astype(int) * 250).mean()
        print(f"| {b} | {len(g):,} | {g.regret.mean():.1f} | {(1-g.made.mean())*100:.1f}% | {avg_v:.0f} |")
    print()

    # === Suit mismatch: did model pick the "oracle best" suit? ===
    # For deals where declarer = NS, compare contract suit vs ns_best_suit.
    print("## Suit-choice accuracy")
    print("(When the model's team declares, how often does the chosen trump match the oracle's best suit for that team?)")
    print()
    ns_decl = played[played.decl_team == 0].copy()
    ew_decl = played[played.decl_team == 1].copy()
    ns_match = (ns_decl.trump == ns_decl.ns_best_suit).mean()
    ew_match = (ew_decl.trump == ew_decl.ew_best_suit).mean()
    print(f"- NS declarer: suit match rate = **{ns_match*100:.1f}%** ({len(ns_decl):,} deals)")
    print(f"- EW declarer: suit match rate = **{ew_match*100:.1f}%** ({len(ew_decl):,} deals)")
    print()

    # Among mismatches, what's the regret vs matches?
    ns_m = ns_decl[ns_decl.trump == ns_decl.ns_best_suit]
    ns_n = ns_decl[ns_decl.trump != ns_decl.ns_best_suit]
    print(f"- NS match avg_regret: {ns_m.regret.mean():.1f} (n={len(ns_m):,})")
    print(f"- NS mismatch avg_regret: {ns_n.regret.mean():.1f} (n={len(ns_n):,})")
    print()

    # === Wrong-team contracts: did NS declare when EW had a strictly better contract? ===
    print("## Wrong-team contracts")
    print("(Did the declaring team have a *strictly worse* best-feasible than the other team?)")
    print()
    # Compare ns_best_pts vs ew_best_pts (both from each team's own perspective; larger = better for that team).
    # Wrong team = declarer is the team with smaller own-best-pts.
    played["ns_own"] = played.ns_best_pts.astype(int)
    played["ew_own"] = played.ew_best_pts.astype(int)
    # For each deal, which team has the higher "own best" — that's the team that *could* have declared best.
    better_team = np.where(played.ns_own > played.ew_own, 0, 1)
    wrong_team = (better_team != played.decl_team).astype(int)
    print(f"- % deals where the other team had a higher own-best: **{wrong_team.mean()*100:.1f}%**")
    print(f"- avg_regret on wrong-team deals: {played.regret[wrong_team == 1].mean():.1f}")
    print(f"- avg_regret on right-team deals: {played.regret[wrong_team == 0].mean():.1f}")
    print()

    # === Coinche stats ===
    print("## Coinche / surcoinche")
    print()
    print(f"- coinche rate: {(played.coinche == 1).mean()*100:.2f}%")
    print(f"- surcoinche rate: {(played.coinche == 2).mean()*100:.2f}%")
    if (played.coinche > 0).any():
        print(f"- avg_regret on coinched: {played.regret[played.coinche > 0].mean():.1f}")
    print()

    # === Top archetypes by total regret ===
    print("## Top-10 archetypes by TOTAL regret contribution")
    print("(position × value_bucket × trump_suit — ranked by sum of regret across deals)")
    print()
    played["suit_name"] = played.trump.map(dict(enumerate(SUITS)))
    agg = (
        played.groupby(["decl_pos", "v_bucket", "suit_name"])
        .agg(n=("regret", "size"), total_regret=("regret", "sum"), avg_regret=("regret", "mean"), chute=("made", lambda x: 1 - x.mean()))
        .sort_values("total_regret", ascending=False)
        .head(10)
    )
    print("| position | value | suit | n | total_regret | avg_regret | chute % |")
    print("|---|---|---|---|---|---|---|")
    for (pos, vb, s), row in agg.iterrows():
        print(f"| pos{pos} | {vb} | {s} | {int(row.n):,} | {int(row.total_regret):,} | {row.avg_regret:.0f} | {row.chute*100:.1f}% |")
    print()

    # === Top archetypes by AVG regret (min 200 deals) ===
    print("## Top-10 archetypes by AVG regret (min 200 deals)")
    print()
    agg2 = (
        played.groupby(["decl_pos", "v_bucket", "suit_name"])
        .agg(n=("regret", "size"), avg_regret=("regret", "mean"), chute=("made", lambda x: 1 - x.mean()))
        .reset_index()
    )
    agg2 = agg2[agg2.n >= 200].sort_values("avg_regret", ascending=False).head(10)
    print("| position | value | suit | n | avg_regret | chute % |")
    print("|---|---|---|---|---|---|")
    for _, row in agg2.iterrows():
        print(f"| pos{int(row.decl_pos)} | {row.v_bucket} | {row.suit_name} | {int(row.n):,} | {row.avg_regret:.0f} | {row.chute*100:.1f}% |")
    print()

    # === Chute vs made: where are the chutes concentrated? ===
    print("## Chute concentration by position × value")
    print()
    chute = played[played.made == 0]
    print(f"total chutes: {len(chute):,}")
    print()
    cross = pd.crosstab(chute.decl_pos, chute.v_bucket)
    for col_order in order:
        if col_order not in cross.columns:
            cross[col_order] = 0
    cross = cross[[c for c in order if c in cross.columns]]
    header = "| pos | " + " | ".join(cross.columns) + " |"
    sep = "|---" * (len(cross.columns) + 1) + "|"
    print(header)
    print(sep)
    for idx, row in cross.iterrows():
        print(f"| pos{idx} | " + " | ".join(str(int(v)) for v in row.values) + " |")
    print()

    # === Oracle-ceiling distribution ===
    print("## Oracle ceiling distribution")
    print("(Max achievable pts for best team under the 'no overbid' assumption.)")
    print()
    best_of_teams = played[["ns_best_pts", "ew_best_pts"]].max(axis=1)
    print(f"- mean best (either team): {best_of_teams.mean():.1f}")
    print(f"- median best: {best_of_teams.median():.0f}")
    print(f"- % deals where best >= 200 (capot or high scorers): {(best_of_teams >= 200).mean()*100:.1f}%")
    print(f"- % deals where both teams have best >= 110: {((played.ns_best_pts >= 110) & (played.ew_best_pts >= 110)).mean()*100:.1f}%")
    print()

    # === Did the model find the chutes that never should have been taken? ===
    # For chutes, what was the oracle ceiling?
    print("## What do our chutes look like?")
    print()
    chute = chute.copy()
    chute["regret_bucket"] = pd.cut(chute.regret, bins=[-1, 50, 100, 200, 300, 500, 10000], labels=["0-50", "50-100", "100-200", "200-300", "300-500", "500+"])
    vc = chute.regret_bucket.value_counts().sort_index()
    print("| regret_bucket | count |")
    print("|---|---|")
    for k, v in vc.items():
        print(f"| {k} | {int(v):,} |")
    print()


if __name__ == "__main__":
    main()
