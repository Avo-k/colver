#!/usr/bin/env python3
"""Télécharge une liste d'URL et en extrait le texte dans le corpus data/rules-corpus/.

Usage: uv run --with pypdf --no-project python _fetch.py <dossier> <url> [<url> ...]
Écrit <corpus>/<dossier>/<slug>.html|.pdf (brut) + <corpus>/<dossier>/<slug>.txt (texte).

Le corpus n'est pas versionné (data/ est gitignoré) ; l'analyse, elle, vit ici.
Cf. METHODE.md.
"""
import html
import os
import re
import subprocess
import sys
import urllib.parse

UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126 Safari/537.36"

# docs/rules-survey/_fetch.py -> racine du dépôt -> data/rules-corpus
CORPUS = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
    "data", "rules-corpus",
)


def slug(url):
    u = urllib.parse.urlparse(url)
    s = (u.netloc + u.path).replace("www.", "")
    s = re.sub(r"[^A-Za-z0-9]+", "_", s).strip("_")
    if u.query:
        s += "_" + re.sub(r"[^A-Za-z0-9]+", "_", u.query)[:40]
    return s[:110]


def html_to_text(raw):
    s = raw
    s = re.sub(r"(?is)<(script|style|nav|footer|svg).*?</\1>", " ", s)
    s = re.sub(r"(?is)<br\s*/?>", "\n", s)
    s = re.sub(r"(?is)</(p|div|li|tr|h[1-6]|table)>", "\n", s)
    s = re.sub(r"(?is)</t[dh]>", " | ", s)
    s = re.sub(r"(?s)<[^>]+>", " ", s)
    s = html.unescape(s)
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n\s*\n+", "\n", s)
    return s.strip()


def fetch(dest, url):
    os.makedirs(dest, exist_ok=True)
    base = os.path.join(dest, slug(url))
    tmp = base + ".raw"
    r = subprocess.run(
        ["curl", "-sL", "--max-time", "45", "-A", UA, "-w", "%{http_code} %{content_type}", url, "-o", tmp],
        capture_output=True, text=True,
    )
    info = r.stdout.strip()
    code = info.split(" ")[0] if info else "ERR"
    ctype = info.split(" ", 1)[1] if " " in info else ""
    if code != "200" or not os.path.exists(tmp):
        print(f"{code:>4}  {url}")
        return
    data = open(tmp, "rb").read()
    if data[:4] == b"%PDF":
        os.rename(tmp, base + ".pdf")
        try:
            import pypdf
            t = "\n".join((p.extract_text() or "") for p in pypdf.PdfReader(base + ".pdf").pages)
        except Exception as e:
            t = f"[extraction PDF échouée: {e}]"
        open(base + ".txt", "w", encoding="utf-8").write(f"SOURCE: {url}\n\n{t}")
        print(f"{code:>4}  PDF {len(t):>7}  {url}")
        return
    enc = "utf-8"
    m = re.search(rb'charset=["\']?([\w-]+)', data[:4000], re.I)
    if m:
        enc = m.group(1).decode("ascii", "ignore")
    txt = data.decode(enc, errors="replace")
    os.rename(tmp, base + ".html")
    body = html_to_text(txt)
    open(base + ".txt", "w", encoding="utf-8").write(f"SOURCE: {url}\n\n{body}")
    print(f"{code:>4}  {len(body):>7}  {url}")


if __name__ == "__main__":
    # un nom simple ("tournois") est résolu dans le corpus ; un chemin explicite est respecté
    arg = sys.argv[1]
    dest = arg if os.path.sep in arg else os.path.join(CORPUS, arg)
    for u in sys.argv[2:]:
        fetch(dest, u)
