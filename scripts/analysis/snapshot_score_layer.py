#!/usr/bin/env python3
"""Copie une couche de scores **en cours de génération**, et refuse une copie déchirée.

**Pourquoi ce script existe.** `gen_score_layer` réécrit sa couche à chaque checkpoint
avec `File::create`, qui **tronque en place** — pas de fichier temporaire, pas de
`rename`. Entre la troncature et le `flush`, le fichier sur disque est court. La fenêtre
est de quelques millisecondes toutes les six minutes, mais son mode de panne est
silencieux des deux côtés :

- côté Python, `check_score_layer.read_layer` construit `cnt` lignes depuis un fichier
  trop court et rend des tuples **vides** sans lever ;
- côté trainer, `RewardMode::RealOnly` fait `unwrap_or(ns_dd_pts)` — une donne que la
  couche ne couvre plus retombe sur la **valeur DD périmée**, sans un mot. C'est le
  défaut que §9 de `docs/data_gen/isdd_score_layer_v2.md` décrit pour la couverture
  partielle, et il arriverait ici par accident de lecture plutôt que par omission.

Le contrôle est arithmétique et suffit : l'en-tête `COLVSC01` annonce `count`, et la
taille du fichier **doit** valoir exactement `10 + len(nom) + 8 + 4 × count`. Un fichier
déchiré rate cette égalité. On recopie jusqu'à ce qu'elle tienne.

Le fichier de rangs (`.ranks`, COLVRK01) est écrit juste après, donc il peut être en
retard d'un checkpoint sur la couche. On le tronque au `count` de la couche plutôt que
d'échouer : les rangs sont un fichier d'accompagnement, pas une source de vérité, et un
préfixe cohérent vaut mieux qu'un refus.

    uv run python scripts/analysis/snapshot_score_layer.py \\
        data/deals/scores_isdd_v2.sc /tmp/snap.sc
"""

import argparse
import shutil
import struct
import sys
import time


def header(path):
    """(nom, count, offset, taille attendue) — lit les 32 premiers octets seulement."""
    with open(path, "rb") as f:
        d = f.read(32)
    if d[:8] != b"COLVSC01":
        sys.exit(f"{path} : magic {d[:8]!r}, ce n'est pas une couche COLVSC01")
    nl = struct.unpack("<H", d[8:10])[0]
    name = d[10:10 + nl].decode()
    cnt, off = struct.unpack("<II", d[10 + nl:18 + nl])
    return name, cnt, off, 18 + nl + 4 * cnt


def snapshot(src, dst, tries=10, pause=2.0):
    for k in range(tries):
        shutil.copyfile(src, dst)
        name, cnt, off, expect = header(dst)
        import os
        got = os.path.getsize(dst)
        if got == expect:
            return name, cnt, off
        print(f"  tentative {k + 1} : {got} octets pour {expect} attendus "
              f"({cnt} donnes) — checkpoint en cours, on repasse", file=sys.stderr)
        time.sleep(pause)
    sys.exit(f"{src} : {tries} copies déchirées d'affilée — le générateur écrit-il "
             f"en boucle ? Vérifier --checkpoint.")


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("src")
    p.add_argument("dst")
    p.add_argument("--ranks", action="store_true",
                   help="copie aussi <src>.ranks, tronqué au count de la couche")
    a = p.parse_args()

    name, cnt, off = snapshot(a.src, a.dst)
    print(f"{a.dst} : couche « {name} », offset {off}, {cnt} donnes — cohérente")

    if a.ranks:
        import os
        rsrc, rdst = a.src + ".ranks", a.dst + ".ranks"
        if not os.path.exists(rsrc):
            sys.exit(f"{rsrc} absent")
        d = open(rsrc, "rb").read()
        if d[:8] != b"COLVRK01":
            sys.exit(f"{rsrc} : magic {d[:8]!r}")
        n = struct.unpack("<I", d[8:12])[0]
        # Le fichier de rangs est écrit après la couche : il peut avoir un checkpoint
        # de retard (n < cnt) ou d'avance (n > cnt) selon l'instant de la copie.
        keep = min(n, cnt, (len(d) - 12) // 4)
        with open(rdst, "wb") as f:
            f.write(b"COLVRK01")
            f.write(struct.pack("<I", keep))
            f.write(d[12:12 + 4 * keep])
        note = "" if keep == cnt else f"  ⚠️ tronqué de {n} à {keep} pour suivre la couche"
        print(f"{rdst} : {keep} lignes de rangs{note}")


if __name__ == "__main__":
    main()
