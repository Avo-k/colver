#!/usr/bin/env python3
"""Vérifie une couche de scores en cours de génération, et la compare à l'ancienne.

**Pourquoi pendant et pas après.** Une couche se produit en jours. Un défaut systématique
— une étiquette dans la mauvaise case, un camp inversé, un rang mal attribué — ne se voit
pas dans un run qui n'imprime que son débit, et le découvrir à 100 000 donnes coûte tout
ce qui a été calculé. Ce script se lance à tout moment sur le fichier partiel.

Cinq contrôles, du plus mécanique au plus révélateur :

1. **arithmétique** — les points cartes N-S valent 0-162, ou 0/252 sur un capot. Les
   valeurs entre 163 et 251 sont *impossibles* : c'est le résidu qui avait trahi le bug
   `quick_tricks` dans l'ancienne couche (58 cases sur 20 M, toutes juste sous le capot).
2. **couverture** — la couche, le fichier de rangs et le pool doivent annoncer le même
   nombre de donnes, et les rangs doivent tous être renseignés.
3. **rangs** — la répartition or/argent/bronze/fer doit retomber sur ce que la mesure A
   prédit : ~1 case or par donne, ~2,37 cases avec un vrai préfixe.
4. **contre l'ancienne couche**, sur les mêmes donnes — c'est l'A/B que le plan promet.
   Compté **du côté du preneur** : en points N-S bruts l'effet s'annule à moitié, parce
   qu'il change de signe selon le camp qui prend. Le taux d'accord sur le meilleur atout
   est accompagné de son **plancher simulé** : deux étiquetages du même procédé sont déjà
   en désaccord ~30 % du temps, donc le taux brut ne veut rien dire seul.
5. **contre la valeur DD** — un jeu à information incomplète rend moins que le double
   dummy au preneur. Un écart nul ou négatif signalerait une étiquette qui n'est pas ce
   qu'on croit.

    uv run python scripts/analysis/check_score_layer.py data/deals/scores_isdd_v2.sc
"""

import argparse
import struct
import sys

POOL = "data/deals/base_5M.bin"
OLD = "data/deals/scores_isdd_5M.sc"
RANK_NAMES = ["or", "argent", "bronze", "fer"]


def read_layer(path):
    d = open(path, "rb").read()
    if d[:8] != b"COLVSC01":
        sys.exit(f"{path} : magic {d[:8]!r}")
    nl = struct.unpack("<H", d[8:10])[0]
    name = d[10:10 + nl].decode()
    cnt, off = struct.unpack("<II", d[10 + nl:18 + nl])
    h = 18 + nl
    rows = [tuple(d[h + 4 * k:h + 4 * k + 4]) for k in range(cnt)]
    return name, off, rows


def read_ranks(path):
    try:
        d = open(path, "rb").read()
    except OSError:
        return None
    if d[:8] != b"COLVRK01":
        return None
    n = struct.unpack("<I", d[8:12])[0]
    return [tuple(d[12 + 4 * k:12 + 4 * k + 4]) for k in range(n)]


def read_pool_dd(path, offset, count):
    """dd_pts du pool. ⚠️ Périmées (antérieures au retrait de `quick_tricks`) — elles ne
    servent ici que de repère d'ordre de grandeur, jamais de vérité."""
    with open(path, "rb") as fh:
        head = fh.read(16)
        if head[:8] != b"COLVDD01":
            sys.exit(f"{path} : magic {head[:8]!r}")
        fh.seek(16 + 21 * offset)
        raw = fh.read(21 * count)
    return [tuple(raw[21 * k + 17:21 * k + 21]) for k in range(count)]


def taker_side(ns_dd):
    """0 = N-S peut tenir le contrat à cet atout (points cartes à somme constante)."""
    return 0 if ns_dd > 81 else 1


def as_taker(ns_pts, side):
    if ns_pts in (0, 252):
        return ns_pts if side == 0 else 252 - ns_pts
    return ns_pts if side == 0 else 162 - ns_pts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("layer", nargs="?", default="data/deals/scores_isdd_v2.sc")
    ap.add_argument("--pool", default=POOL)
    ap.add_argument("--old", default=OLD)
    # Écart-type apparié de deux étiquetages du même procédé, mesuré par la mesure B
    # (`prefix_label`, 1 985 donnes). C'est le plancher contre lequel tout se lit.
    ap.add_argument("--control-sd", type=float, default=24.37)
    ap.add_argument("--sim-reps", type=int, default=20)
    args = ap.parse_args()

    name, off, rows = read_layer(args.layer)
    n = len(rows)
    print(f"=== {args.layer} — couche « {name} », offset {off}, {n} donnes ===")
    if n == 0:
        sys.exit("vide")

    # 1. arithmétique
    bad = [(k, t, v) for k, r in enumerate(rows) for t, v in enumerate(r) if 162 < v < 252]
    zero = sum(1 for r in rows for v in r if v == 0)
    capot = sum(1 for r in rows for v in r if v == 252)
    print(f"\n1. arithmétique : {len(bad)} valeurs impossibles (163-251)"
          + (f"  ⚠️  ex. {bad[:3]}" if bad else "  ✔"))
    print(f"   capots : {capot} cases à 252 ({100 * capot / (4 * n):.2f} %), "
          f"{zero} à 0 ({100 * zero / (4 * n):.2f} %)")

    # 2. couverture
    ranks = read_ranks(args.layer + ".ranks")
    if ranks is None:
        print("\n2. couverture : pas de fichier .ranks  ⚠️")
    else:
        m = min(n, len(ranks))
        unset = sum(1 for r in ranks[:m] for v in r if v > 3)
        print(f"\n2. couverture : {len(ranks)} lignes de rangs pour {n} de scores"
              + (f", {unset} cases sans rang  ⚠️" if unset else "  ✔"))

        # 3. rangs
        cnt = [0, 0, 0, 0]
        for r in ranks[:m]:
            for v in r:
                if v <= 3:
                    cnt[v] += 1
        tot = sum(cnt) or 1
        print("\n3. rangs de préfixe (mesure A prédit ~1,00 or et ~2,37 cases réelles) :")
        for i, c in enumerate(cnt):
            print(f"   {RANK_NAMES[i]:<7} {c:>8}  {100 * c / tot:>5.1f} %  "
                  f"{c / m:>4.2f} case/donne")
        print(f"   → {sum(cnt[:3]) / m:.2f} case(s) par donne avec un VRAI préfixe")

    dd = read_pool_dd(args.pool, off, n)

    # 3bis. La mesure B sur données de production. Deux cases de rangs différents sont
    # des ATOUTS différents, donc leurs étiquettes ne se comparent pas entre elles ; ce
    # qui se compare, c'est leur écart à une référence par case — la valeur DD. Si le
    # préfixe fait bien jouer le preneur ~4 pt mieux, les cases « or » doivent montrer un
    # déficit plus petit que les « fer ».
    #
    # ⚠️ Les dd_pts du pool sont périmées. Le NIVEAU de ces écarts n'a donc pas de sens
    # absolu ; leur ORDRE entre rangs en a un, la péremption frappant les quatre pareil.
    if ranks:
        import statistics as st
        by_rank = {r: [] for r in range(4)}
        for k in range(min(n, len(ranks))):
            for t in range(4):
                r = ranks[k][t]
                if r <= 3:
                    side = taker_side(dd[k][t])
                    by_rank[r].append(as_taker(rows[k][t], side) - as_taker(dd[k][t], side))
        print("\n3bis. écart à la valeur DD (orienté preneur), PAR RANG DE PRÉFIXE —")
        print("      la mesure B prédit or > argent ≈ or > bronze > fer")
        for r in range(4):
            xs = by_rank[r]
            if len(xs) < 30:
                continue
            m = sum(xs) / len(xs)
            se = st.stdev(xs) / len(xs) ** 0.5
            print(f"   {RANK_NAMES[r]:<7} n={len(xs):>7}  {m:+7.2f} ±{se:.2f}")
        if len(by_rank[0]) > 30 and len(by_rank[3]) > 30:
            g = sum(by_rank[0]) / len(by_rank[0]) - sum(by_rank[3]) / len(by_rank[3])
            gse = (st.stdev(by_rank[0]) ** 2 / len(by_rank[0])
                   + st.stdev(by_rank[3]) ** 2 / len(by_rank[3])) ** 0.5
            print(f"   or − fer : {g:+.2f} ±{2 * gse:.2f} (2σ)   "
                  f"[mesure B, en appairé : +4,36 ±1,30]")
            print("   ⚠️ CONFONDU, et il ne faut pas citer cet écart comme l'effet du")
            print("      préfixe : les cases « or » sont les atouts que v6 annonce, donc")
            print("      des donnes où le contrat est net — elles seraient peut-être plus")
            print("      faciles à jouer quel que soit le préfixe. Seule la mesure B, qui")
            print("      compare la MÊME case sous deux préfixes, sépare les deux. Ce qui")
            print("      vaut ici est l'ORDRE des quatre rangs, qui se reproduit.")

    # 5. contre la valeur DD, orienté preneur
    diffs = [as_taker(rows[k][t], taker_side(dd[k][t])) - as_taker(dd[k][t], taker_side(dd[k][t]))
             for k in range(n) for t in range(4)]
    mean = sum(diffs) / len(diffs)
    print(f"\n5. contre la valeur DD du pool, orienté preneur : {mean:+.2f} pt en moyenne")
    print("   (attendu NÉGATIF : à information incomplète le preneur rend moins que le")
    print("    double dummy. Les dd_pts du pool sont périmées, donc c'est un ordre de")
    print("    grandeur — un signe positif, lui, serait un vrai signal d'alarme.)")

    # 4. contre l'ancienne couche
    try:
        oname, ooff, orows = read_layer(args.old)
    except (OSError, SystemExit):
        print(f"\n4. ancienne couche {args.old} illisible — comparaison sautée")
        return
    lo, hi = max(off, ooff), min(off + n, ooff + len(orows))
    if hi <= lo:
        print("\n4. aucune donne commune avec l'ancienne couche")
        return
    d_taker, d_ns = [], []
    for g in range(lo, hi):
        a, b = rows[g - off], orows[g - ooff]
        for t in range(4):
            s = taker_side(dd[g - off][t])
            d_taker.append(as_taker(a[t], s) - as_taker(b[t], s))
            d_ns.append(a[t] - b[t])
    import statistics as st
    mt = sum(d_taker) / len(d_taker)
    mn = sum(d_ns) / len(d_ns)
    sd = st.stdev(d_taker)
    same = sum(1 for x in d_ns if x == 0)
    print(f"\n4. contre « {oname} » sur {hi - lo} donnes communes ({len(d_taker)} cases) :")
    print(f"   orienté preneur : {mt:+.2f} pt (écart-type {sd:.1f})   |   N-S brut : {mn:+.2f} pt")
    print(f"   étiquettes identiques : {100 * same / len(d_ns):.1f} %")

    # LA question. Une couche ne sert pas à donner un niveau, elle sert à faire CHOISIR
    # un atout : le bidder compare les quatre cases d'une même donne. Si les deux couches
    # désignent le même meilleur atout, régénérer n'achète presque rien de décisionnel,
    # quel que soit l'écart par case — c'est la leçon de pool_staleness.md, où 87 % de
    # l'écart mesuré était du bruit d'échantillonnage.
    agree, agree_top2, tot_d = 0, 0, 0
    for g in range(lo, hi):
        a, b, dg = rows[g - off], orows[g - ooff], dd[g - off]
        for side in (0, 1):
            # Les cases où CE camp peut tenir le contrat. Comparer sur les quatre
            # couleurs mélangerait les deux camps, qui n'ont pas les mêmes options.
            cells = [t for t in range(4) if taker_side(dg[t]) == side]
            if len(cells) < 2:
                continue
            ka = max(cells, key=lambda t: as_taker(a[t], side))
            kb = max(cells, key=lambda t: as_taker(b[t], side))
            tot_d += 1
            agree += ka == kb
            top2a = sorted(cells, key=lambda t: -as_taker(a[t], side))[:2]
            agree_top2 += kb in top2a
    if tot_d:
        print(f"\n   MÊME MEILLEUR ATOUT ? sur {tot_d} (donne, camp) à ≥2 options :")
        print(f"     même argmax        : {100 * agree / tot_d:.1f} %")
        print(f"     l'ancien argmax est dans le top-2 du nouveau : {100 * agree_top2 / tot_d:.1f} %")
        print("     (c'est la quantité décisionnelle : le bidder compare des cases entre")
        print("      elles, il ne lit pas un niveau absolu)")

        # ⚠️ Ce taux ne se lit PAS dans l'absolu. Deux étiquetages du MÊME procédé sont
        # déjà en désaccord, parce qu'un étiquetage IS-DD est bruité : la mesure B donne
        # un écart-type apparié de 24,37 points, soit ~17,2 par étiquetage. On simule ce
        # plancher en re-bruitant la couche neuve contre elle-même. Si l'accord simulé
        # vaut l'accord mesuré, l'écart ancien/nouveau n'est QUE du bruit — c'est
        # exactement la conclusion de pool_staleness.md, et elle retirerait sa raison
        # d'être à la regénération.
        import random
        rng = random.Random(7)
        sigma = args.control_sd / (2 ** 0.5)
        sim, sim_tot = 0, 0
        for _ in range(args.sim_reps):
            for g in range(lo, hi):
                a, dg = rows[g - off], dd[g - off]
                for side in (0, 1):
                    cells = [t for t in range(4) if taker_side(dg[t]) == side]
                    if len(cells) < 2:
                        continue
                    base = {t: as_taker(a[t], side) for t in cells}
                    k1 = max(cells, key=lambda t: base[t] + rng.gauss(0, sigma))
                    k2 = max(cells, key=lambda t: base[t] + rng.gauss(0, sigma))
                    sim_tot += 1
                    sim += k1 == k2
        floor = 100 * sim / sim_tot
        meas = 100 * agree / tot_d
        print(f"\n     PLANCHER simulé (même procédé, bruit σ={sigma:.1f}/étiquetage) : "
              f"{floor:.1f} %")
        print(f"     mesuré ancien/nouveau : {meas:.1f} %  →  écart {meas - floor:+.1f} pp")
        if meas >= floor - 2:
            # ⚠️ Ne PAS lire « les deux couches sont équivalentes ». Le plancher étant
            # au niveau du mesuré, ce test n'a tout simplement pas la puissance de les
            # distinguer : l'argmax d'une case est dominé par le bruit d'étiquetage,
            # pour l'ANCIENNE comme pour la NOUVELLE. Un test sans puissance rend un
            # null, pas une équivalence — et conclure l'équivalence ici serait la même
            # erreur que lire « pas d'effet » dans un h2h d'arène trop court.
            print("     ⇒ ce test N'A PAS LA PUISSANCE de distinguer les deux couches :")
            print("       le bruit d'étiquetage domine l'argmax, des deux côtés. Ce n'est")
            print("       pas une équivalence, c'est un null sans puissance.")
            print("       Ce qui reste détectable est le décalage systématique ci-dessus,")
            print("       qui lui ne se moyenne pas.")
        else:
            print("     ⇒ les deux couches diffèrent PLUS que le bruit ne l'explique :")
            print("       la regénération change bien le choix d'atout.")


if __name__ == "__main__":
    main()
