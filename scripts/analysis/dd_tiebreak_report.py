#!/usr/bin/env python3
"""Read the tie-break arena sweep and say whether the choice among DD-equal cards matters.

The metric is the average match margin in points, not the win rate: against the weaker bots the
oracle wins ~95 % of matches whatever it plays, so the rate saturates and carries no signal.

Three seeds per cell give the error bar. Without one, a 60-point difference between two
tie-breaks is unreadable — the seed-to-seed spread on this metric is itself tens of points.
"""
import csv
import statistics
import sys

path = sys.argv[1] if len(sys.argv) > 1 else "/tmp/tiebreak_arena.csv"
rows = list(csv.DictReader(open(path)))

cells = {}
for r in rows:
    cells.setdefault((r["opponent"], r["tiebreak"]), []).append(
        (float(r["margin"]), float(r["win_pct"]))
    )

# Weakest opponent first: the hypothesis is that the effect grows with the opponent's
# imperfection, since a DD-equal card can only differentiate itself by exploiting mistakes.
OPPONENTS = ["heuristic", "rule", "ismcts", "dmc50"]
TIEBREAKS = ["order", "lowest", "highest", "cheapest", "dearest"]

print(f"{'adversaire':>11} {'départage':>10} {'marge moy.':>12} {'écart-type':>11} "
      f"{'vs order':>10} {'% victoires':>12}")
for opp in OPPONENTS:
    base = cells.get((opp, "order"))
    if not base:
        continue
    base_m = statistics.mean(m for m, _ in base)
    # Pooled spread across seeds, used as the yardstick for "is this difference real?"
    spreads = [
        statistics.pstdev([m for m, _ in v])
        for k, v in cells.items()
        if k[0] == opp and len(v) > 1
    ]
    noise = statistics.mean(spreads) if spreads else 0.0
    for tb in TIEBREAKS:
        v = cells.get((opp, tb))
        if not v:
            continue
        ms = [m for m, _ in v]
        ws = [w for _, w in v]
        d = statistics.mean(ms) - base_m
        # Two pooled standard deviations is the bar a difference has to clear to be worth a
        # sentence; anything under it is a seed, not an effect.
        mark = "" if tb == "order" else ("  <-- réel" if abs(d) > 2 * noise else "")
        print(f"{opp:>11} {tb:>10} {statistics.mean(ms):>12.0f} "
              f"{statistics.pstdev(ms):>11.0f} {d:>+10.0f} {statistics.mean(ws):>11.1f}%{mark}")
    print(f"{'':>11} {'(bruit graine à graine ≈ ' + format(noise, '.0f') + ' pts)':>40}")
    print()

print("Lecture : « vs order » est l'écart de marge moyenne contre le départage actuel.")
print("Un écart inférieur à deux fois le bruit graine à graine n'est pas un effet.")
