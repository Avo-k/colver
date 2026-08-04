#!/usr/bin/env python3
"""Régénère `_sources.tsv` et `SOURCES.md` à partir du corpus data/rules-corpus/.

Le corpus brut n'est pas versionné (cf. .gitignore) : ce manifeste est ce qui permet de le
reconstituer et de retrouver la source d'une affirmation.

Usage : python3 _manifest.py
"""
import os
import re
import subprocess

ROOT = os.path.dirname(os.path.abspath(__file__))
# docs/rules-survey/ -> racine du dépôt -> data/rules-corpus
CORPUS = os.path.join(os.path.dirname(os.path.dirname(ROOT)), "data", "rules-corpus")
DIRS = ["federations", "tournois", "clubs", "divers", "apps-sites", "open-source"]
LABEL = {
    "federations": "Fédérations",
    "tournois": "Tournois et concours",
    "clubs": "Clubs et cercles",
    "divers": "Sites de règles et encyclopédies",
    "apps-sites": "Applications et plateformes de jeu",
    "open-source": "Dépôts open source (code et docs)",
}


# Documents de la toute première passe, téléchargés en direct au curl avant que _fetch.py
# n'existe : ils n'ont pas de ligne `SOURCE:`. URL rétablies à la main.
BACKFILL = {
    "ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016":
        "https://www.ffbelote.org/wp-content/uploads/2016/01/regles-officielles-de-la-Belote-Contree-27-01-2016.pdf",
    "ffbelote_regles-officielles-de-la-Belote-27-01-2016":
        "https://www.ffbelote.org/wp-content/uploads/2016/01/regles-officielles-de-la-Belote-27-01-2016.pdf",
    "ffbelote_REGLES-DE-LA-BELOTE-CONTREE":
        "https://www.ffbelote.org/wp-content/uploads/2015/11/REGLES-DE-LA-BELOTE-CONTREE.pdf",
    "belotecontree_free_reglement":
        "http://belotecontree.free.fr/article.php3?id_article=22",
    # Non téléchargé, et **la seule source du corpus que `_refetch.py` ne sait pas retrouver** :
    # l'édition « Équipe Ludique » n'a pas d'URL tracée. Le dépôt en portait une copie
    # (docs/règles officielles belote contrée.pdf), retirée de git — document tiers sous droit
    # d'auteur dans un dépôt MIT. Ce qu'elle a de singulier est distillé dans les matrices, qui
    # sont versionnées ; le PDF lui-même n'existe plus que dans le corpus local.
    "LOCAL_regles_officielles_belote_contree": "LOCAL — édition « Équipe Ludique », sans URL tracée, non redistribuée",
}


def source_url(txt_path):
    """L'URL est la première ligne des .txt produits par _fetch.py."""
    try:
        with open(txt_path, encoding="utf-8", errors="replace") as fh:
            first = fh.readline().strip()
    except OSError:
        return None
    return first[8:].strip() if first.startswith("SOURCE: ") else None


def github_guess(name):
    """Fichiers récupérés en direct sur raw.githubusercontent : `owner_repo_branche_chemin`.

    La conversion `/` → `_` n'est pas réversible (des noms de fichiers contiennent `_`).
    On ne rend donc que le dépôt, qui est sans ambiguïté et suffit à retrouver le fichier.
    """
    parts = name.split("_")
    if len(parts) < 3:
        return None, name
    owner, repo = parts[0], parts[1]
    return f"https://github.com/{owner}/{repo}", "_".join(parts[2:])


def collect():
    rows = []
    for d in DIRS:
        path = os.path.join(CORPUS, d)
        if not os.path.isdir(path):
            continue
        seen = set()
        for f in sorted(os.listdir(path)):
            stem, ext = os.path.splitext(f)
            # un même document existe en original (.pdf/.html) + texte (.txt) : une seule ligne
            key = stem if ext == ".txt" else stem
            full = os.path.join(path, f)
            if ext == ".txt":
                url = source_url(full)
                orig = next(
                    (e for e in (".pdf", ".html") if os.path.exists(os.path.join(path, stem + e))),
                    None,
                )
                if key in seen:
                    continue
                seen.add(key)
                size = os.path.getsize(os.path.join(path, stem + orig)) if orig else os.path.getsize(full)
                url = url or BACKFILL.get(stem, "")
                rows.append((d, stem, orig or ".txt", url, size, "fetch"))
            elif ext in (".pdf", ".html"):
                continue  # traité avec son .txt
            elif ext in (".png", ".raw"):
                continue  # rendus et fichiers temporaires : dérivés, pas des sources
            else:
                if key in seen:
                    continue
                seen.add(key)
                url, rel = github_guess(f)
                rows.append((d, f, ext, url or "", os.path.getsize(full), "raw-github"))
    return rows


def main():
    rows = collect()
    with open(os.path.join(ROOT, "_sources.tsv"), "w", encoding="utf-8") as fh:
        fh.write("dossier\tnom\toriginal\turl\toctets\tmode\n")
        for r in rows:
            fh.write("\t".join(str(x) for x in r) + "\n")

    try:
        date = subprocess.run(
            ["git", "log", "-1", "--format=%cs"], capture_output=True, text=True, cwd=ROOT
        ).stdout.strip()
    except Exception:
        date = ""

    n_ok = sum(1 for r in rows if r[3])
    lines = [
        "# Manifeste des sources",
        "",
        "**Généré par `_manifest.py` — ne pas éditer à la main.**",
        "",
        "Le corpus brut (PDF, HTML, code) **n'est pas versionné** : ~92 Mo de documents aspirés,",
        "dont beaucoup sont sous droit d'auteur. Ce fichier est ce qui permet de retrouver",
        "n'importe quelle source citée dans [SYNTHESE.md](SYNTHESE.md) ou dans les matrices.",
        "",
        "Pour tout re-télécharger : `python3 _refetch.py`. Méthode complète dans",
        "[METHODE.md](METHODE.md).",
        "",
        f"**{len(rows)} sources**, dont {n_ok} avec une URL enregistrée.",
        "",
        "Comment lire une citation : les matrices citent un chemin du type",
        "`federations/ffbelote_regles-officielles-de-la-Belote-Contree-27-01-2016.txt`.",
        "Cherchez le nom dans le tableau du dossier correspondant ci-dessous pour retrouver son URL.",
        "",
    ]
    for d in DIRS:
        sub = [r for r in rows if r[0] == d]
        if not sub:
            continue
        lines += [
            f"## `{d}/` — {LABEL[d]} ({len(sub)})",
            "",
            "| Nom | Original | Source |",
            "|---|---|---|",
        ]
        for _, name, ext, url, size, mode in sub:
            kb = f"{size // 1024} Ko" if size >= 1024 else f"{size} o"
            if url.startswith("http"):
                cell = f"[{url}]({url})"
            elif url:
                cell = f"*{url}*"
            else:
                cell = "—"
            if mode == "raw-github":
                cell += " *(dépôt ; chemin exact du fichier non reconstituable)*"
            lines.append(f"| `{name}` | {ext} · {kb} | {cell} |")
        lines.append("")

    with open(os.path.join(ROOT, "SOURCES.md"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
    print(f"{len(rows)} sources → _sources.tsv + SOURCES.md")


if __name__ == "__main__":
    main()
