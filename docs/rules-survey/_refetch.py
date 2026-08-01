#!/usr/bin/env python3
"""Reconstitue le corpus brut (data/rules-corpus/) à partir de `_sources.tsv`.

Le corpus vit dans data/rules-corpus/, qui n'est pas versionné (data/ est gitignoré).
Ce script le retélécharge.

    uv run --with pypdf --no-project python _refetch.py            # tout
    uv run --with pypdf --no-project python _refetch.py federations tournois

Ce qui ne se reconstitue pas :
  - les pages disparues depuis la collecte (le web pourrit) — l'échec est signalé, pas masqué ;
  - le contenu exact d'une page dynamique, qui aura changé ;
  - les fichiers `open-source/` récupérés en direct sur raw.githubusercontent : le manifeste
    ne garde que le dépôt, la conversion `/` → `_` du chemin n'étant pas réversible.
Voir METHODE.md.
"""
import csv
import os
import sys

from _fetch import fetch

ROOT = os.path.dirname(os.path.abspath(__file__))
CORPUS = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data", "rules-corpus")


def main():
    wanted = set(sys.argv[1:])
    todo, skipped = [], []
    with open(os.path.join(ROOT, "_sources.tsv"), encoding="utf-8") as fh:
        for row in csv.DictReader(fh, delimiter="\t"):
            if wanted and row["dossier"] not in wanted:
                continue
            if row["mode"] == "raw-github" or not row["url"].startswith("http"):
                skipped.append((row["dossier"], row["nom"], row["url"]))
                continue
            todo.append((row["dossier"], row["url"]))

    print(f"{len(todo)} à télécharger, {len(skipped)} non reconstituables automatiquement\n")
    ok = 0
    for d, url in todo:
        before = set(os.listdir(os.path.join(CORPUS, d))) if os.path.isdir(os.path.join(CORPUS, d)) else set()
        fetch(os.path.join(CORPUS, d), url)
        after = set(os.listdir(os.path.join(CORPUS, d)))
        ok += bool(after - before) or True  # fetch() imprime déjà son code HTTP

    if skipped:
        print("\nÀ récupérer à la main :")
        for d, name, url in skipped:
            print("  {}/{}  <-  {}".format(d, name, url or "(pas d'URL)"))


if __name__ == "__main__":
    main()
