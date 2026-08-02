#!/usr/bin/env python3
"""Enregistre une comparaison de couches de labels IS-DD dans le journal de mesures.

`relabel_isdd` (Rust) produit deux choses : une couche COLVSC01 de labels, et la
comparaison appariée de cette couche contre une autre. Les deux vivent sous `data/`,
qui est gitignoré — donc sans ce pont, la trace d'une mesure qui a coûté une heure de
GPU disparaît au premier `git clean` ou changement de machine.

On enregistre donc **la comparaison** (petite, c'est le résultat) dans
`docs/measurements/index.jsonl`, avec la provenance des deux couches comparées : leur
sha256, leur taille, et les paramètres du run. Les `.sc` eux-mêmes restent sous `data/` ;
s'ils sont perdus, l'index dit au moins ce qu'ils valaient et comment les refaire.

Usage :
  # enregistre une comparaison déjà calculée
  python scripts/analysis/relabel_log.py --json /tmp/cmp.json --tag c_vs_b \\
      --params '{"worlds":"sidecar","auction":"synthetic","dets":20,"seed":42}'

  # ou calcule et enregistre d'un coup
  python scripts/analysis/relabel_log.py --compare data/deals/relabel/c_playgen_synth.sc \\
      --baseline data/deals/relabel/b_uniform_synth.sc --tag c_vs_b \\
      --params '{"worlds":"sidecar","auction":"synthetic","dets":20,"seed":42}'
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = runlog.ROOT
BIN = os.path.join(ROOT, "target", "release", "relabel_isdd")


def compute(compare_path, baseline_path):
    """Lance `relabel_isdd --compare-only` et rend le dict de stats."""
    if not os.path.exists(BIN):
        sys.exit(f"binaire absent : {BIN}\n"
                 "  cargo build --release --features parallel --bin relabel_isdd")
    with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as fh:
        out = fh.name
    try:
        r = subprocess.run(
            [BIN, "--compare-only", compare_path, "--baseline", baseline_path,
             "--json", out],
            cwd=ROOT, capture_output=True, text=True, timeout=600,
        )
        if r.returncode != 0:
            sys.exit(f"relabel_isdd a échoué :\n{r.stderr}")
        with open(out, encoding="utf-8") as fh:
            return json.load(fh)
    finally:
        os.unlink(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="stats déjà produites par relabel_isdd --json")
    ap.add_argument("--compare", help="couche .sc à comparer (calcule les stats)")
    ap.add_argument("--baseline", help="couche .sc de référence")
    ap.add_argument("--tag", required=True, help="nom court de la comparaison")
    ap.add_argument("--params", default="{}", help="paramètres du run, en JSON")
    ap.add_argument("--note", default="", help="ce que la comparaison isole")
    a = ap.parse_args()

    if a.json:
        with open(a.json, encoding="utf-8") as fh:
            stats = json.load(fh)
    elif a.compare and a.baseline:
        stats = compute(a.compare, a.baseline)
    else:
        sys.exit("il faut --json, ou --compare avec --baseline")

    params = json.loads(a.params)
    if a.note:
        params["isole"] = a.note

    # La provenance est la moitié de l'intérêt : les .sc ne sont pas versionnés
    # et deux fichiers homonymes peuvent porter des labels différents.
    layers = [p for p in (a.compare, a.baseline) if p]

    runlog.save(
        script="relabel_isdd",
        tag=a.tag,
        params=params,
        summary=stats,
        models=layers,
    )


if __name__ == "__main__":
    main()
