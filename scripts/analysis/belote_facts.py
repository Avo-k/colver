#!/usr/bin/env python3
"""Ce que la belote annoncée contraint, et ce que coûtait de l'ignorer.

Enveloppe journalisée de `bench_belote_facts` (Rust) : le binaire fait la mesure,
ce script garde la trace — sans quoi il faudrait relire 50 000 donnes et refaire
6 400 mondes GPU à chaque fois qu'on repose la question.

Deux volets, cf. l'en-tête du binaire :
  * fréquence sur donnes réellement jouées (corpus COLVGM01) ;
  * fraction de mondes impossibles, analytique pour un tirage uniforme aveugle,
    **empirique** pour playgen si `--sidecar` est donné.

    uv run python scripts/analysis/belote_facts.py --deals 50000 \\
        --sidecar "$COLVER_PLAYGEN_GPU_URL" --positions 200 --worlds 32
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import runlog  # noqa: E402

BIN = "target/release/bench_belote_facts"
PLAYGEN = "models/playgen/playgen_v2_final.bin"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--corpus", default="data/training/games_500k.bin")
    ap.add_argument("--deals", type=int, default=50000)
    ap.add_argument("--sidecar", default="")
    ap.add_argument("--positions", type=int, default=200)
    ap.add_argument("--worlds", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="corpus")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    if not os.path.exists(os.path.join(runlog.ROOT, BIN)):
        print(f"{BIN} absent — cargo build --release --bin bench_belote_facts",
              file=sys.stderr)
        return 1

    cmd = [BIN, "--corpus", args.corpus, "--deals", str(args.deals),
           "--seed", str(args.seed), "--json"]
    if args.sidecar:
        cmd += ["--sidecar", args.sidecar, "--positions", str(args.positions),
                "--worlds", str(args.worlds)]

    with runlog.Timer() as t:
        proc = subprocess.run(cmd, cwd=runlog.ROOT, capture_output=True, text=True)
    sys.stderr.write(proc.stderr)
    if proc.returncode != 0:
        return proc.returncode

    data = json.loads(proc.stdout.strip().splitlines()[-1])
    pg = data.get("playgen")
    summary = {
        "positions": data["positions"],
        "constrained_pct": round(100 * data["constrained"] / data["positions"], 2),
        "with_held_pct": round(100 * data["with_held"] / data["positions"], 2),
        "banned_only_pct": round(100 * data["banned_only"] / data["positions"], 2),
        "deals_with_announcement_pct": round(
            100 * data["deals_with_announcement"] / data["deals"], 2),
        "impossible_uniform_at_constrained_pct": round(
            100 * data["p_impossible_uniform_constrained"], 2),
        "impossible_uniform_overall_pct": round(
            100 * data["p_impossible_uniform_all"], 2),
    }
    if pg and pg["returned"]:
        summary["impossible_playgen_at_constrained_pct"] = round(
            100 * pg["belote_violations"] / pg["returned"], 2)
        summary["playgen_void_violations"] = pg["void_violations"]

    if not args.no_log:
        runlog.save("belote_facts", args.tag,
                    {k: v for k, v in vars(args).items() if k not in ("tag", "no_log")},
                    summary, payload=data,
                    models=[PLAYGEN] if args.sidecar else (),
                    took_s=t.s)
    print(json.dumps(summary, indent=1, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
