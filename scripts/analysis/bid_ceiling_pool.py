#!/usr/bin/env python3
"""Même mesure que `bid_ceiling_170_180.py`, mais sur le pool pré-résolu COLVDD01 (grand N).

Le pool ne stocke que les points **NS** par couleur (`dd_pts: [u8; 4]`), donc on ne mesure que
le camp NS — ce qui suffit : les donnes sont symétriques en distribution.

    uv run --with numpy --no-project python scripts/analysis/bid_ceiling_pool.py data/deals/base_5M.bin

ATTENTION : `base_5M.bin` est antérieur au correctif `quick_tricks` (2026-07-23) et au correctif
d'atout (2026-08-02). Ses valeurs DD sont donc approximatives. À lire comme une confirmation
d'ordre de grandeur d'une mesure faite à neuf, pas comme une référence.
"""
import sys

import numpy as np

QUEEN, KING = 4, 5
CAPOT = 252


def load_layer(path, n):
    """Couche de scores COLVSC01 : magic(8) + name_len(u16) + name + count(u32) + offset(u32) + n×[u8;4]."""
    raw = np.fromfile(path, dtype=np.uint8)
    assert raw[:8].tobytes() == b"COLVSC01", raw[:8].tobytes()
    nl = int(np.frombuffer(raw[8:10].tobytes(), dtype="<u2")[0])
    name = raw[10:10 + nl].tobytes().decode()
    o = 10 + nl
    count = int(np.frombuffer(raw[o:o + 4].tobytes(), dtype="<u4")[0])
    off = int(np.frombuffer(raw[o + 4:o + 8].tobytes(), dtype="<u4")[0])
    body = raw[o + 8:o + 8 + count * 4].reshape(count, 4).astype(np.int32)
    full = np.full((n, 4), -1, dtype=np.int32)
    full[off:off + count] = body
    return name, full, off, count


def report(label, hands, pts, n):
    totals = np.empty((n, 4), dtype=np.int32)
    for s in range(4):
        m = np.uint32((1 << (s * 8 + QUEEN)) | (1 << (s * 8 + KING)))
        bel = ((hands[:, 0] & m) == m) | ((hands[:, 2] & m) == m)
        totals[:, s] = pts[:, s] + 20 * bel
    capot_any = (pts == CAPOT).any(axis=1)
    best_nc = np.where(pts == CAPOT, -1, totals).max(axis=1)
    rows = {
        "capot": capot_any,
        "180+, pas capot": ~capot_any & (best_nc >= 180),
        "170-179, pas capot": ~capot_any & (best_nc >= 170) & (best_nc < 180),
        "160-169": ~capot_any & (best_nc >= 160) & (best_nc < 170),
    }
    out = {k: 100 * float(v.sum()) / n for k, v in rows.items()}
    out["170+180"] = out["180+, pas capot"] + out["170-179, pas capot"]
    print(f"{label:<14} " + "  ".join(f"{out[k]:>8.4f}%" for k in
          ("capot", "180+, pas capot", "170-179, pas capot", "160-169", "170+180")))
    return out


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "data/deals/base_5M.bin"
    layers = sys.argv[2:]
    raw = np.fromfile(path, dtype=np.uint8)
    assert raw[:8].tobytes() == b"COLVDD01", raw[:8].tobytes()
    n = int(np.frombuffer(raw[8:16].tobytes(), dtype="<u8")[0])
    rec = raw[16:16 + n * 21].reshape(n, 21)
    hands = rec[:, 1:17].copy().view("<u4").reshape(n, 4)
    dd = rec[:, 17:21].astype(np.int32)
    print(f"{path} — {n:,} donnes\n")

    if layers:
        # comparaison DD / jeu réel : même pool, mêmes mains, mêmes couleurs
        print("Camp NS, meilleure couleur — % de donnes par catégorie\n")
        print(f"{'source':<14} {'capot':>9}  {'180+':>9}  {'170-179':>9}  {'160-169':>9}  {'170+180':>9}")
        print("-" * 74)
        report("DD (parfait)", hands, dd, n)
        for lp in layers:
            name, pts, off, count = load_layer(lp, n)
            report(name, hands, pts, n)
        print("\nDD = jeu parfait des deux côtés. Les autres couches sont du jeu réel : un camp")
        print("peut y dépasser sa valeur DD quand la défense adverse joue mal.")
        return

    # Le pool contient quelques valeurs impossibles (162 < v < 252) — signature du bug
    # `quick_tricks` corrigé le 2026-07-23, qui rendait une borne au lieu d'une valeur.
    # On écarte les donnes concernées plutôt que de les compter comme des 180 réalisables.
    bad = ((dd > 162) & (dd < CAPOT)).any(axis=1)
    if bad.any():
        print(f"  {int(bad.sum()):,} donnes écartées : valeur DD impossible "
              f"(162 < v < 252), reliquat du bug quick_tricks\n")
        keep = ~bad
        hands, dd, n = hands[keep], dd[keep], int(keep.sum())

    totals = np.empty((n, 4), dtype=np.int32)
    for s in range(4):
        m = np.uint32((1 << (s * 8 + QUEEN)) | (1 << (s * 8 + KING)))
        # belote = Dame ET Roi d'atout dans la MÊME main, côté NS (sièges 0 et 2)
        bel = ((hands[:, 0] & m) == m) | ((hands[:, 2] & m) == m)
        totals[:, s] = dd[:, s] + 20 * bel

    capot_any = (dd == CAPOT).any(axis=1)
    # meilleur total atteignable en excluant les couleurs où c'est un capot
    masked = np.where(dd == CAPOT, -1, totals)
    best_nc = masked.max(axis=1)

    cats = {
        "capot réalisable": capot_any,
        "180 réalisable, pas capot": ~capot_any & (best_nc >= 180),
        "170 réalisable, pas capot": ~capot_any & (best_nc >= 170) & (best_nc < 180),
        "160-169": ~capot_any & (best_nc >= 160) & (best_nc < 170),
        "< 160": ~capot_any & (best_nc < 160),
    }
    print(f"{'catégorie':<32} {'n':>10} {'%':>9}")
    print("-" * 54)
    for k, v in cats.items():
        c = int(v.sum())
        print(f"{k:<32} {c:>10,} {100 * c / n:>8.4f}%")

    add = int(cats["170 réalisable, pas capot"].sum()) + int(cats["180 réalisable, pas capot"].sum())
    print("-" * 54)
    print(f"{'ce qu''ajouteraient 170+180':<32} {add:>10,} {100 * add / n:>8.4f}%")

    print("\nDétail au-dessus de 160 (hors capot) :")
    for lo in range(160, 200, 10):
        c = int((~capot_any & (best_nc >= lo) & (best_nc < lo + 10)).sum())
        print(f"  {lo}-{lo + 9}: {c:>9,}  {100 * c / n:>7.4f}%")


if __name__ == "__main__":
    main()
