#!/usr/bin/env python3
"""Journalise la mesure « taille du sampler playgen → débit de génération ».

Le banc lui-même est `playgen_size_bench.sh` : il alterne les trois modèles et
écrit un TSV. Ce script en fait une entrée de `docs/measurements/index.jsonl`,
avec l'empreinte des trois checkpoints — sans elle, « v2-belote-large » ne veut rien
dire dans six mois (trois répertoires `playgen_v2belote_large*` portent des poids
différents sous des noms voisins).

Usage :
  python scripts/analysis/playgen_size_bench_log.py /tmp/playgen_size_bench/results.tsv \
      --tag moxxi-3090-400x3
"""
import argparse
import csv
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

# Les checkpoints tels qu'ils ont été servis, côté local (identiques par sha256
# à ceux copiés sur l'hôte GPU — vérifié au moment de la copie).
MODELS = {
    "v2belote_small": "models/playgen_v2belote_small/v2belote_small_120000.bin",
    "v2": "models/playgen/playgen_v2_final.bin",
    "v2belote_large": "models/playgen_v2belote_large2/v2belote_large2_80000.bin",
}
ARCH = {"v2belote_small": "d=256 L=4", "v2": "d=384 L=6", "v2belote_large": "d=512 L=8"}
PARAMS_M = {"v2belote_small": 3.22, "v2": 10.74, "v2belote_large": 25.3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tsv")
    ap.add_argument("--tag", default="playgen-size")
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--dets", type=int, default=40)
    ap.add_argument("--threads", type=int, default=256)
    ap.add_argument("--lanes", type=int, default=512)
    ap.add_argument("--no-log", action="store_true")
    a = ap.parse_args()

    rows = list(csv.DictReader(open(a.tsv), delimiter="\t"))
    for r in rows:
        for k in ("deals_s", "actions_s", "wall_s", "wait_pct", "solve_pct"):
            r[k] = float(r[k]) if r[k] else None

    by_model = {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(r)

    summary = {}
    for name, rs in by_model.items():
        dps = sorted(x["deals_s"] for x in rs)
        summary[name] = {
            "arch": ARCH.get(name),
            "params_M": PARAMS_M.get(name),
            "runs": len(dps),
            "deals_s": [round(x, 3) for x in dps],
            # La médiane, pas la moyenne : trois points dont un ralenti par une
            # charge extérieure tirent une moyenne et pas une médiane.
            "deals_s_median": round(statistics.median(dps), 3),
            "deals_s_spread_pct": round(100 * (dps[-1] - dps[0]) / dps[0], 1) if dps[0] else None,
            "wait_sidecar_pct_median": round(
                statistics.median([x["wait_pct"] for x in rs if x["wait_pct"] is not None]), 1
            ) if any(x["wait_pct"] is not None for x in rs) else None,
        }
    if "v2" in summary:
        base = summary["v2"]["deals_s_median"]
        for name in summary:
            summary[name]["vs_v2"] = round(summary[name]["deals_s_median"] / base, 3)

    print(f"{'modèle':<10} {'arch':<10} {'M par.':>7} {'donnes/s':>9} {'×v2':>6} {'attente GPU':>12}")
    for name in ("v2belote_small", "v2", "v2belote_large"):
        s = summary.get(name)
        if not s:
            continue
        print(f"{name:<10} {s['arch']:<10} {s['params_M']:>7} "
              f"{s['deals_s_median']:>9.2f} {s.get('vs_v2', 0):>6.2f} "
              f"{s['wait_sidecar_pct_median']:>11.1f} %")

    if a.no_log:
        return
    runlog.save(
        script="playgen_size_bench",
        tag=a.tag,
        params={"deals_per_run": a.deals, "dets": a.dets, "threads": a.threads,
                "lane_budget": a.lanes, "match_mode": True,
                "gpu": "moxxi RTX 3090 (partagée avec le sidecar de prod)",
                "client": "machine de dev, 32 fils, nice 10",
                "rounds": max(int(r["round"]) for r in rows)},
        summary=summary,
        payload={"rows": rows},
        models=[MODELS[m] for m in MODELS],
    )


if __name__ == "__main__":
    main()
