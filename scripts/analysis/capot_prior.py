#!/usr/bin/env python3
"""Mesure C : `P(capot | mes 8 cartes)` — la conditionnelle qui dimensionne `tail_100k`.

Enveloppe de `bench_capot_prior`. Sans GPU : c'est le solveur DD qu'on interroge, donc
elle tourne sur les cœurs libres pendant qu'une mesure GPU occupe la carte.

**Le chiffre qu'elle remplace.** `base_5M.bin` dit qu'un capot N-S est atteignable en DD
dans 16,08 % des donnes. Ce nombre est une **marginale vue des quatre mains** ; s'en
servir pour dimensionner une strate d'entraînement est une erreur de conditionnement,
puisqu'un bidder ne voit que la sienne. Et il n'annonce **qu'une** couleur, pas
« l'une des quatre » — donc la bonne quantité est `P(capot au meilleur atout | main)`.

Usage :
    uv run python scripts/analysis/capot_prior.py --hands 1200 --worlds 80 --tag couche-v2
    uv run python scripts/analysis/capot_prior.py --reuse <fichier.json>
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, "target/release/bench_capot_prior")
THRESHOLDS = [0.01, 0.05, 0.10, 0.25, 0.50, 0.75]
BANDS = [(5, 10), (10, 15), (15, 20), (20, 25), (25, 30), (30, 35), (35, 40)]


def run(hands, worlds, threads, seed, out_json):
    cmd = [BIN, "--hands", str(hands), "--worlds", str(worlds),
           "--threads", str(threads), "--seed", str(seed), "--json", out_json]
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, check=True)
    with open(out_json, encoding="utf-8") as fh:
        return json.load(fh)


def digest(raw):
    # rows : [hand, p_any, p_best, mean_best, eval_max]
    rows = raw["rows"]
    n = len(rows)
    p_best = [r[2] for r in rows]
    out = {
        "hands": n,
        "worlds": raw["worlds"],
        "secs": round(raw["secs"], 1),
        "mean_p_any": round(raw["mean_p_any"], 5),
        "mean_p_best": round(raw["mean_p_best"], 5),
        "tail_pct": {f"ge_{int(100 * t)}": round(100 * sum(p >= t for p in p_best) / n, 3)
                     for t in THRESHOLDS},
    }
    # La strate ne vaut que si elle est reconnaissable sans simuler.
    bands = {}
    for lo, hi in BANDS:
        sel = [r for r in rows if lo <= r[4] < hi]
        if len(sel) < 5:
            continue
        bands[f"{lo}-{hi}"] = {
            "n": len(sel),
            "p_best_pct": round(100 * sum(r[2] for r in sel) / len(sel), 3),
            "mean_pts": round(sum(r[3] for r in sel) / len(sel), 1),
        }
    out["by_eval_max"] = bands
    return out


def show(d):
    print(f"\n=== P(capot | main) — {d['hands']} mains × {d['worlds']} complétions, "
          f"{d['secs']:.0f} s ===")
    print(f"  un atout quelconque      : {100 * d['mean_p_any']:.2f} %")
    print(f"  au meilleur atout        : {100 * d['mean_p_best']:.2f} %   ← celle qui compte")
    print("  (16,08 % = la marginale de base_5M, vue des QUATRE mains — autre question)")
    print("\n  queue, au meilleur atout :")
    for t in THRESHOLDS:
        print(f"    P ≥ {100 * t:>4.0f} % : {d['tail_pct'][f'ge_{int(100 * t)}']:>7.3f} % des mains")
    print("\n  reconnaissable sans simuler ? P(capot) par tranche d'eval_max :")
    for band, v in d["by_eval_max"].items():
        print(f"    [{band:<6}) n={v['n']:>5}  P={v['p_best_pct']:>6.2f} %  "
              f"points moyens {v['mean_pts']:.0f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=1200)
    ap.add_argument("--worlds", type=int, default=80)
    ap.add_argument("--threads", type=int, default=16)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="capot-prior")
    ap.add_argument("--reuse")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    with runlog.Timer() as t:
        if args.reuse:
            with open(args.reuse, encoding="utf-8") as fh:
                raw = json.load(fh)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                raw = run(args.hands, args.worlds, args.threads, args.seed,
                          os.path.join(tmp, "c.json"))

    d = digest(raw)
    show(d)

    if not args.no_log:
        runlog.save(
            script="capot_prior",
            tag=args.tag,
            params={"hands": args.hands, "worlds": args.worlds, "seed": args.seed,
                    "reused": bool(args.reuse)},
            summary=d,
            payload=raw,   # `rows` porte P(capot) main par main
            models=(),     # aucun poids consulté : DD pur
            took_s=t.s,
        )


if __name__ == "__main__":
    main()
