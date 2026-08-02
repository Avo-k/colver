#!/usr/bin/env python3
"""Journal des mesures d'analyse — pour ne pas repayer deux fois une heure de calcul.

Le 2026-08-02, les trois régimes de §1.7 ont coûté ~50 min de GPU et **rien n'a été
conservé** : les scripts n'écrivaient que sur stdout, et l'affichage tronquait à 25
mains sur 120. Reposer la moindre question sur ces données — une médiane au lieu d'une
moyenne, un intervalle bootstrap, un sous-ensemble par famille de mains — imposait de
tout relancer. D'où ce module.

Deux sorties par run :

  - `data/analysis/<script>/<horodatage>__<tag>.json` — **tout**, données brutes
    comprises (les écarts monde par monde, pas seulement leur moyenne). C'est ce qui
    permet de ré-agréger sans recalculer. Volumineux, et `data/` est gitignoré.
  - une ligne dans `docs/measurements/index.jsonl` — métadonnées et agrégats seulement.
    Minuscule, et **versionné** : la trace d'une mesure doit survivre à un
    `rm -rf data/`, à un changement de machine et à l'oubli. Il vit sous `docs/` et non
    sous `data/` pour être versionné sans avoir à percer une exception dans les règles
    d'ignore de `data/` — et parce qu'il est le pendant des documents qui citent ces
    chiffres.

**La provenance est la moitié de l'intérêt.** Un écart de points ne veut rien dire sans
le modèle qui l'a produit, or les `.bin` de poids ne sont pas dans git et changent sans
prévenir. On enregistre donc un sha256 de chaque fichier de poids consulté, le SHA du
dépôt, et s'il était sale — une mesure faite sur un arbre modifié n'est pas reproductible
et doit le dire.
"""

import hashlib
import json
import os
import subprocess
import sys
import time
from datetime import datetime

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(ROOT, "data", "analysis")          # brut, gitignoré
INDEX = os.path.join(ROOT, "docs", "measurements", "index.jsonl")  # registre, versionné


def _git():
    def run(*a):
        try:
            return subprocess.run(a, cwd=ROOT, capture_output=True, text=True,
                                  timeout=10).stdout.strip()
        except Exception:
            return ""
    sha = run("git", "rev-parse", "HEAD")
    dirty = bool(run("git", "status", "--porcelain"))
    return {"sha": sha[:12], "dirty": dirty}


def file_id(path):
    """Identité d'un fichier de poids : chemin, taille, sha256 court.

    Le hash coûte ~0,2 s pour 55 Mo de modèles, négligeable devant les runs qu'il
    documente — et c'est la seule chose qui distingue deux checkpoints homonymes.
    """
    try:
        st = os.stat(path)
    except OSError:
        return {"path": str(path), "missing": True}
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return {"path": os.path.relpath(path, ROOT) if os.path.isabs(path) else str(path),
            "bytes": st.st_size, "sha256": h.hexdigest()[:16]}


def save(script, tag, params, summary, payload=None, models=(), took_s=None):
    """Écrit le run complet + une ligne d'index. Retourne le chemin du fichier complet.

    `summary` doit rester petit : c'est lui qui part dans l'index versionné, donc c'est
    lui qu'on relira dans six mois. `payload` porte le brut (listes par monde, lignes
    par main) et n'est écrit que dans le fichier local.
    """
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in tag) or "run"
    outdir = os.path.join(STORE, script)
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, f"{stamp}__{safe}.json")

    meta = {
        "script": script, "tag": tag, "when": datetime.now().isoformat(timespec="seconds"),
        "took_s": round(took_s, 1) if took_s is not None else None,
        "git": _git(), "argv": sys.argv[1:],
        "models": [file_id(m) for m in models if m],
        "params": params,
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({**meta, "summary": summary, "payload": payload}, fh,
                  ensure_ascii=False, indent=1)

    os.makedirs(os.path.dirname(INDEX), exist_ok=True)
    with open(INDEX, "a", encoding="utf-8") as fh:
        fh.write(json.dumps({**meta, "summary": summary,
                             "file": os.path.relpath(path, ROOT)},
                            ensure_ascii=False) + "\n")
    print(f"\n[runlog] {os.path.relpath(path, ROOT)}"
          f"  (+ index.jsonl)", file=sys.stderr)
    return path


class Timer:
    def __enter__(self):
        self.t0 = time.monotonic()
        return self

    def __exit__(self, *a):
        self.s = time.monotonic() - self.t0

    @property
    def elapsed(self):
        return time.monotonic() - self.t0
