#!/usr/bin/env python3
"""Journalise une mesure du solveur DD (`bench_dd`) dans le registre.

`bench_dd` est un binaire Rust : il ne peut pas appeler `runlog` lui-même. Ce script
l'exécute, parse son tableau et enregistre le run — sans quoi chaque mesure du solveur
serait à repayer, ce qui est exactement ce que la règle « toute mesure se journalise »
existe pour éviter. Il n'y avait aucune entrée solveur dans le registre avant celui-ci,
et c'est pour ça que quatre documents citent quatre chiffres différents (13,5 / 14,9 /
28 / 77 ms) pour des choses toutes appelées « un solve ».

Provenance particulière ici : il n'y a aucun modèle à hacher, mais le binaire dépend des
drapeaux de compilation (`target-cpu`, features) autant que du code. On enregistre donc
les features et le RUSTFLAGS effectif — une mesure sans eux n'est pas comparable.

Usage :
  uv run python scripts/analysis/dd_solver_bench.py --tag epoch-tt --ab --repeats 3
  uv run python scripts/analysis/dd_solver_bench.py --tag baseline --values out.vals
"""

import argparse
import os
import re
import subprocess
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, "target", "release", "bench_dd")

# "    full     800      1448045       38326.6       38904.3     1.02x"
AB_ROW = re.compile(
    r"^\s*(full|mid|end|worlds|ALL)\s+(\d+)\s+(\d*)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)x\s*$"
)
# "    full     800       1158.4      1448045      45830.7       32.0"
RUN_ROW = re.compile(
    r"^\s*(full|mid|end|worlds|ALL)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)(?:\s+([\d.]+))?\s*$"
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True, help="nom du run (ex. epoch-tt, compact-state)")
    ap.add_argument("--corpus", default="data/analysis/dd_corpus_v1.bin")
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--ab", action="store_true", help="A/B entrelacé contre le memset")
    ap.add_argument("--values", default=None, help="fichier de valeurs à écrire")
    ap.add_argument("--note", default="", help="ce qui change dans cette version")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(BIN):
        sys.exit(
            f"{BIN} absent — construire d'abord :\n"
            '  cargo build --release '
            '--features "parallel solver_stats" --bin bench_dd'
        )

    cmd = [BIN, "run", "--corpus", args.corpus, "--threads", str(args.threads),
           "--repeats", str(args.repeats)]
    if args.ab:
        cmd += ["--ab"]
    if args.values:
        cmd += ["--values", args.values]

    t0 = time.time()
    proc = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True)
    took = time.time() - t0
    sys.stderr.write(proc.stderr)
    print(proc.stdout)
    if proc.returncode != 0:
        sys.exit(f"bench_dd a échoué ({proc.returncode})")

    shapes = {}
    checksum = None
    for line in proc.stdout.splitlines():
        if "checksum" in line:
            m = re.search(r"(-?\d+)\s*$", line)
            if m:
                checksum = int(m.group(1))
            continue
        if args.ab:
            m = AB_ROW.match(line)
            if m:
                shapes[m.group(1)] = {
                    "n": int(m.group(2)),
                    "nodes_per_pos": int(m.group(3)) if m.group(3) else None,
                    "us_per_pos": float(m.group(4)),
                    "us_per_pos_legacy_clear": float(m.group(5)),
                    "speedup": float(m.group(6)),
                }
        else:
            m = RUN_ROW.match(line)
            if m:
                shapes[m.group(1)] = {
                    "n": int(m.group(2)),
                    "nodes_M": float(m.group(3)),
                    "nodes_per_pos": int(m.group(4)),
                    "us_per_pos": float(m.group(5)),
                    "cards_left": float(m.group(6)) if m.group(6) else None,
                }

    if not shapes:
        sys.exit("aucune ligne de résultat reconnue — le format de bench_dd a changé ?")

    summary = {"note": args.note, "checksum": checksum, "shapes": shapes}
    params = {
        "corpus": args.corpus,
        "threads": args.threads,
        "repeats": args.repeats,
        "ab": args.ab,
        # Les drapeaux de compilation font partie du résultat autant que le code.
        "rustflags": os.environ.get("RUSTFLAGS", ""),
        "bin_mtime": time.strftime("%Y-%m-%dT%H:%M:%S", time.localtime(os.path.getmtime(BIN))),
        "loadavg": os.getloadavg(),
    }

    if args.no_log:
        print("[--no-log] rien enregistré", file=sys.stderr)
        return
    runlog.save("dd_solver_bench", args.tag, params, summary, took_s=took)


if __name__ == "__main__":
    main()
