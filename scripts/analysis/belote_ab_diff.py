#!/usr/bin/env python3
"""A/B apparié de la déduction de belote : les deux bras de `bench_belote_ab`.

Chaque bras rend une ligne `donne:score_NS:score_EO`, pour **les mêmes donnes** —
c'est tout l'intérêt du montage : le RNG de distribution est dédié, donc une donne
où le jeu n'a pas divergé rend exactement le même score des deux côtés et sort du
test toute seule. Le bruit de donne, qui écrase n'importe quel effet dans une
comparaison non appariée, ne compte donc que sur les donnes où quelque chose a
réellement changé.

On rapporte trois choses, dans cet ordre d'importance :

  1. la fraction de donnes qui **divergent** — c'est la taille de la prise, et elle
     se lit toute seule ;
  2. l'écart moyen par donne sur l'ensemble, avec son erreur-type appariée ;
  3. le même écart restreint aux donnes divergentes, qui dit ce que la déduction
     vaut *quand elle sert*.

    python3 scripts/analysis/belote_ab_diff.py on.txt off.txt --tag isdd60_3000
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runlog  # noqa: E402


def load(path):
    out = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            k, ns, ew = line.split(":")
            out[int(k)] = (int(ns), int(ew))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("on", help="bras avec la déduction")
    ap.add_argument("off", help="bras COLVER_NO_BELOTE_FACTS=1")
    ap.add_argument("--tag", default="belote_ab")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    on, off = load(args.on), load(args.off)
    keys = sorted(set(on) & set(off))
    if not keys:
        print("aucune donne commune", file=sys.stderr)
        return 1
    if len(keys) != len(on) or len(keys) != len(off):
        print(f"attention : {len(on)} / {len(off)} donnes, {len(keys)} en commun",
              file=sys.stderr)

    # Écart du camp qui porte la déduction (N-S), en points marqués.
    diffs = [(on[k][0] - on[k][1]) - (off[k][0] - off[k][1]) for k in keys]
    diverged = [d for d in diffs if d != 0]
    n, nd = len(diffs), len(diverged)

    mean = statistics.fmean(diffs)
    sd = statistics.stdev(diffs) if n > 1 else 0.0
    se = sd / math.sqrt(n) if n else 0.0
    z = mean / se if se else 0.0

    print(f"donnes appariées      : {n}")
    print(f"donnes divergentes    : {nd} ({100 * nd / n:.1f} %)")
    print(f"écart N-S moyen       : {mean:+.2f} pts/donne  (±{1.96 * se:.2f} à 95 %, z = {z:+.2f})")
    if nd:
        md = statistics.fmean(diverged)
        sdd = statistics.stdev(diverged) if nd > 1 else 0.0
        print(f"  … sur les divergentes : {md:+.2f} pts/donne (σ = {sdd:.0f})")
        wins = sum(1 for d in diverged if d > 0)
        print(f"  … dont {wins} en faveur de la déduction, {nd - wins} contre "
              f"({100 * wins / nd:.1f} %)")
    print()
    print("Un z sous 2 ne dit pas que l'effet est nul : il dit que ce montage-là ne "
          "le résout pas.")

    if not args.no_log:
        summary = {
            "deals": n,
            "diverged": nd,
            "diverged_pct": round(100 * nd / n, 2),
            "mean_diff_ns": round(mean, 3),
            "se": round(se, 3),
            "z": round(z, 2),
        }
        if nd:
            summary["mean_diff_diverged"] = round(statistics.fmean(diverged), 2)
        path = runlog.save(
            "belote_ab_diff", args.tag,
            {"on": args.on, "off": args.off},
            summary,
            payload={"keys": keys, "diffs": diffs},
        )
        print(f"\njournalisé : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
