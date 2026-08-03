#!/usr/bin/env python3
"""Regret de la carte choisie, avec et sans la déduction de belote.

L'A/B apparié par donne (`belote_ab_diff.py`) est l'instrument honnête mais peu
sensible : le bruit de donne est énorme devant l'effet. Ici on mesure la même
chose une couche plus bas, là où le bruit est petit — **la décision**.

Protocole (`bench_belote_facts --isdd`, deux exécutions) :

  * chaque position contrainte est résolue à 60 mondes par les deux bras (le
    second sous `COLVER_NO_BELOTE_FACTS=1`) ;
  * le **juge** est la même position résolue à 400 mondes *avec* la déduction,
    graine distincte pour qu'aucun bras n'hérite des mondes de son juge ;
  * regret d'un bras = EV(meilleure carte) − EV(carte choisie), lue chez le juge.

Le juge n'est pas arbitraire : la belote est une **règle**, donc la distribution
qui l'honore est la bonne, et celle qui l'ignore contient des mondes impossibles.
La seule chose qu'on lui demande, c'est d'avoir assez de mondes pour que son
classement soit stable — d'où le 400 contre 60.

    python3 scripts/analysis/belote_regret.py reg_on.txt reg_off.txt
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
    rows = {}
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            parts = line.split(":")
            key = (int(parts[0]), int(parts[1]))
            chosen = int(parts[2])
            table = None
            if len(parts) > 3 and parts[3]:
                table = {}
                for item in parts[3].split(";"):
                    c, v = item.split("=")
                    table[int(c)] = float(v)
            rows[key] = (chosen, table)
    return rows


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("on", help="bras avec la déduction (porte la table de référence)")
    ap.add_argument("off", help="bras COLVER_NO_BELOTE_FACTS=1")
    ap.add_argument("--tag", default="isdd60_ref400")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    on, off = load(args.on), load(args.off)
    keys = [k for k in sorted(set(on) & set(off)) if on[k][1]]
    if not keys:
        print("aucune position commune avec table de référence", file=sys.stderr)
        return 1

    flips, reg_on, reg_off, diffs = 0, [], [], []
    for k in keys:
        c_on, table = on[k]
        c_off, _ = off[k]
        if c_on != c_off:
            flips += 1
        best = max(table.values())
        # Une carte absente de la table du juge n'est pas notable : ça n'arrive
        # que si les deux recherches n'ont pas vu le même ensemble légal, ce qui
        # signalerait un bug de rejeu plutôt qu'une différence de politique.
        if c_on not in table or c_off not in table:
            continue
        r_on, r_off = best - table[c_on], best - table[c_off]
        reg_on.append(r_on)
        reg_off.append(r_off)
        diffs.append(r_off - r_on)  # > 0 = la déduction fait mieux

    n = len(diffs)
    mean = statistics.fmean(diffs)
    se = statistics.stdev(diffs) / math.sqrt(n) if n > 1 else 0.0
    z = mean / se if se else 0.0
    nz = [d for d in diffs if d != 0]

    print(f"positions comparées   : {n}")
    print(f"cartes différentes    : {flips} ({100 * flips / len(keys):.1f} %)")
    print(f"regret moyen — avec   : {statistics.fmean(reg_on):.3f} pts DD")
    print(f"regret moyen — sans   : {statistics.fmean(reg_off):.3f} pts DD")
    print(f"gain de la déduction  : {mean:+.3f} pts DD/décision "
          f"(±{1.96 * se:.3f} à 95 %, z = {z:+.2f})")
    if nz:
        wins = sum(1 for d in nz if d > 0)
        print(f"  sur les {len(nz)} décisions qui diffèrent : {wins} en faveur "
              f"({100 * wins / len(nz):.1f} %), gain moyen {statistics.fmean(nz):+.2f}")

    if not args.no_log:
        summary = {
            "positions": n,
            "flip_pct": round(100 * flips / len(keys), 2),
            "regret_with": round(statistics.fmean(reg_on), 3),
            "regret_without": round(statistics.fmean(reg_off), 3),
            "gain_per_decision": round(mean, 3),
            "se": round(se, 3),
            "z": round(z, 2),
        }
        path = runlog.save("belote_regret", args.tag,
                           {"on": args.on, "off": args.off}, summary,
                           payload={"keys": [list(k) for k in keys], "diffs": diffs})
        print(f"\njournalisé : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
