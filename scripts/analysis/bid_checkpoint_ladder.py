#!/usr/bin/env python3
"""À partir de quel budget de pas une sonde classe-t-elle deux bidders ?

**C'est une calibration de puissance, pas une mesure de v6.** Avant de dépenser N runs
d'ablation à budget réduit, il faut savoir si un budget réduit *décide* — si les métriques
d'un checkpoint à 10 M de pas rangent les bras dans le même ordre qu'à 30 M, ou si elles
bougent encore. La réponse s'achète gratuitement : l'échelle de checkpoints de v6 est déjà
sur le disque.

Deux métriques par checkpoint, toutes deux sans donne jouée ni monde échantillonné :

* **taux de bascule sous renommage de couleur** (`bid_equivariance`) — la cible directe de
  la canonicalisation, §3.1, donc le signal qu'une ablation « canonique vs physique » doit
  lire ;
* **profil de la sonde stratifiée** (`bid_probes`) — la distance à la référence v6 finale,
  famille par famille et régime par régime, §3.2.

Ce que le résultat autorise : si les deux ont plafonné avant 10 M, une ablation à 10 M est
lisible et les bras coûtent ~6 h au lieu de ~48. Si elles bougent encore, un budget réduit
ne peut pas attribuer et il faut soit payer plein tarif, soit changer d'instrument.

    uv run python scripts/analysis/bid_checkpoint_ladder.py --dir models/bid_v6_isdd_resume
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

REGIMES = {"ouverture": "", "contestation": "100C", "soutien": "100C P"}


def checkpoints(directory: Path) -> list[tuple[int, Path]]:
    """(pas, chemin) triés. `bid_nn_final` est rangé après le dernier numéroté."""
    out: list[tuple[int, Path]] = []
    for p in sorted(directory.glob("bid_nn_*.bin")):
        m = re.fullmatch(r"bid_nn_(\d+)\.bin", p.name)
        if m:
            out.append((int(m.group(1)), p))
    out.sort()
    final = directory / "bid_nn_final.bin"
    if final.exists() and out:
        out.append((out[-1][0] + 1, final))
    return out


def flip_rate(model: Path, prior: str, deals: int) -> float | None:
    """Taux de bascule d'annonce sous les 23 renommages non triviaux."""
    cmd = [
        "uv", "run", "python", str(ROOT / "scripts/analysis/bid_equivariance.py"),
        "--bid-model", str(model), "--deals", str(deals), "--no-log",
    ]
    if prior:
        cmd += ["--prior", prior]
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=ROOT)
    if r.returncode != 0:
        print(r.stderr[-800:], file=sys.stderr)
        return None
    # Viser la ligne du réseau, pas la dernière du rapport : le script imprime ensuite
    # l'échelle des Q, dont les pourcentages n'ont rien à voir.
    m = re.search(r"bid v6\s+[\d,]+/[\d,]+\s*=\s*([\d.]+)\s*%", r.stdout)
    return float(m.group(1)) if m else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="models/bid_v6_isdd_resume")
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--regimes", default="ouverture,contestation,soutien")
    ap.add_argument("--tag", default="v6_ladder")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    directory = ROOT / args.dir
    ckpts = checkpoints(directory)
    if not ckpts:
        print(f"aucun checkpoint dans {directory}", file=sys.stderr)
        return 1
    regimes = [r for r in args.regimes.split(",") if r in REGIMES]

    print(f"{len(ckpts)} checkpoints, {len(regimes)} régimes, {args.deals} donnes\n")
    rows = []
    for steps, path in ckpts:
        label = "final" if path.name.endswith("final.bin") else f"{steps / 1e6:.1f}M"
        row = {"steps": steps, "label": label, "model": path.name}
        for reg in regimes:
            row[reg] = flip_rate(path, REGIMES[reg], args.deals)
        rows.append(row)
        cells = "  ".join(
            f"{reg[:5]}={row[reg]:5.1f}%" if row[reg] is not None else f"{reg[:5]}=  n/a"
            for reg in regimes
        )
        print(f"  {label:>6}  {cells}")

    print("\nÉcart entre le dernier checkpoint et chaque budget antérieur :")
    last = rows[-1]
    for row in rows[:-1]:
        deltas = "  ".join(
            f"{reg[:5]}={row[reg] - last[reg]:+5.1f}"
            for reg in regimes
            if row[reg] is not None and last[reg] is not None
        )
        print(f"  {row['label']:>6}  {deltas}")

    if not args.no_log:
        import runlog

        runlog.save(
            script="bid_checkpoint_ladder",
            tag=args.tag,
            params={"dir": args.dir, "deals": args.deals, "regimes": regimes},
            summary={
                f"final_{reg}": last[reg] for reg in regimes if last[reg] is not None
            },
            payload={"rows": rows},
            models=[str(directory / r["model"]) for r in rows],
        )
    print(json.dumps({"rows": rows}, ensure_ascii=False)[:0])  # payload gardé par runlog
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
