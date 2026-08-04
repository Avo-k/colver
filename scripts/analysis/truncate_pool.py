#!/usr/bin/env python3
"""Tronque un pool `COLVDD01` à ses N premières donnes.

**Pourquoi ça existe.** Une couche de scores partielle ne s'entraîne pas sur le pool
entier : `RewardMode::RealOnly` retombe silencieusement sur `dd_pts` pour toute donne
que la couche ne couvre pas, et `DealPool::load_or_generate` ne tronque pas. On tirerait
donc l'immense majorité des épisodes avec une étiquette DD **périmée** mélangée à une
minorité d'étiquettes IS-DD fraîches — deux échelles dans une même reward, exactement ce
que le plan refuse par ailleurs. Détail : docs/data_gen/isdd_score_layer_v2.md §9.

Format : magic "COLVDD01"[8] + count u64[8] + par donne (dealer[1] + hands[16] +
dd_pts[4]) = 21 o.

    uv run python scripts/analysis/truncate_pool.py data/deals/base_5M.bin 30000 \\
        data/deals/base_30k.bin
"""

import struct
import sys

REC = 21
HDR = 16


def main():
    if len(sys.argv) != 4:
        print(__doc__)
        sys.exit(1)
    src, n, dst = sys.argv[1], int(sys.argv[2]), sys.argv[3]

    with open(src, "rb") as fh:
        head = fh.read(HDR)
        if head[:8] != b"COLVDD01":
            sys.exit(f"{src} : magic {head[:8]!r}, attendu COLVDD01")
        total = struct.unpack("<Q", head[8:16])[0]
        if n > total:
            sys.exit(f"{src} porte {total} donnes, {n} demandées")
        body = fh.read(REC * n)
    if len(body) != REC * n:
        sys.exit(f"lecture courte : {len(body)} o au lieu de {REC * n}")

    with open(dst, "wb") as fh:
        fh.write(b"COLVDD01")
        fh.write(struct.pack("<Q", n))
        fh.write(body)
    print(f"{dst} : {n} donnes ({HDR + REC * n} o), tronqué de {src} ({total} donnes)")


if __name__ == "__main__":
    main()
