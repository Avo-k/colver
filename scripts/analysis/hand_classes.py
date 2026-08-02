#!/usr/bin/env python3
"""Combien de mains distinctes existe-t-il, et combien de codes pour les décrire ?

Les quatre couleurs sont interchangeables tant qu'aucun atout n'est nommé, donc une
main et la même main couleurs échangées sont la même position. Ce script produit les
deux chiffres que cite `colver-core/src/hand_class.rs`, et que le module ne calcule
pas lui-même : la **vérification** des dénombrements par force brute, et la
**concentration** des codes (quelle part des mains vit dans combien de codes).

Le nombre de codes, lui, est vérifié côté Rust :
`cargo test -p colver-core --release --lib hand_class -- --ignored`.

Deux modes.

`--verify` — dénombrement indépendant du moteur :
  * Burnside sur S₄ pour une main de 8 cartes (472 579) et pour une donne 4×8
    (4 148 577 738 928 080), **pas une division par 24** : 7,5 % des mains ont une
    symétrie de couleur, et 10 518 300/24 n'est même pas entier ;
  * la formule de Burnside pour les donnes est validée par énumération exhaustive sur
    des jeux réduits (R rangs × 4 couleurs), seul régime où la force brute est
    possible ;
  * les 472 579 classes de mains sont recomptées par énumération exhaustive des
    10 518 300 mains, canonisées par tri des quatre masques de rangs.

  ⚠️ Piège trouvé en écrivant ce script : pour canoniser une **donne**, il faut
  appliquer *une seule* permutation à tous les joueurs à la fois. Trier les masques
  couleur par joueur revient à autoriser une permutation différente par joueur, et
  sous-compte (66 au lieu de 126 sur R=2). Le symptôme qui l'attrape : le résultat
  tombe sous `brut/24`, ce qu'aucun quotient ne peut faire.

Mode par défaut — concentration des codes, calculée en interrogeant le **binding
Rust** (`colver.hand_code`), donc sans réimplémenter la logique du code en Python.
On parcourt les 1 820 803 classes à atout désigné et on pondère chacune par sa taille
d'orbite ; la somme doit retomber sur 10 518 300, ce qui sert d'assertion.

Référence (2026-08-02) :

    niveau     codes   50 % des mains dans   90 % dans
    length         9                     2           4
    trump         80                     8          28
    shape        339                    28         122
    tops       5 277                   388       1 927
    full       6 654                   420       2 281

    uv run python scripts/analysis/hand_classes.py
    uv run python scripts/analysis/hand_classes.py --verify
"""

import argparse
import os
import sys
from collections import Counter, defaultdict
from itertools import permutations, product
from math import comb, factorial

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

LEVELS = ["length", "trump", "shape", "tops", "full"]


# --------------------------------------------------------------------------
# --verify : dénombrements indépendants du moteur
# --------------------------------------------------------------------------

def _cycle_type(p):
    seen = [False] * len(p)
    ct = []
    for i in range(len(p)):
        if not seen[i]:
            length, j = 0, i
            while not seen[j]:
                seen[j] = True
                j = p[j]
                length += 1
            ct.append(length)
    return tuple(sorted(ct))


def _types(n_suits=4):
    t = Counter()
    for p in permutations(range(n_suits)):
        t[_cycle_type(p)] += 1
    return t


def burnside_hand(ranks=8):
    """Orbites des mains de `ranks` cartes parmi 4×`ranks`, sous S₄."""
    total = 0
    for ct, mult in _types().items():
        fix = 0
        for ks in product(range(ranks + 1), repeat=len(ct)):
            if sum(length * k for length, k in zip(ct, ks, strict=True)) == ranks:
                prod = 1
                for k in ks:
                    prod *= comb(ranks, k)
                fix += prod
        total += mult * fix
    assert total % 24 == 0
    return total // 24


def burnside_deal(ranks=8):
    """Orbites des donnes (4×`ranks` cartes → 4 joueurs étiquetés), sous S₄."""
    def cycle_dist(length):
        d = defaultdict(int)
        for comp in product(range(ranks + 1), repeat=4):
            if sum(comp) != ranks:
                continue
            ways = factorial(ranks)
            for c in comp:
                ways //= factorial(c)
            d[tuple(length * c for c in comp)] += ways
        return d

    cache = {length: cycle_dist(length) for length in range(1, 5)}
    total = 0
    for ct, mult in _types().items():
        acc = {(0, 0, 0, 0): 1}
        for length in ct:
            new = defaultdict(int)
            for v, w in acc.items():
                for v2, w2 in cache[length].items():
                    s = tuple(a + b for a, b in zip(v, v2, strict=True))
                    if all(x <= ranks for x in s):
                        new[s] += w * w2
            acc = new
        total += mult * acc.get((ranks,) * 4, 0)
    assert total % 24 == 0
    return total // 24


def brute_deal(ranks):
    """Énumération exhaustive des donnes, pour valider `burnside_deal` en petit."""
    cards = [(s, r) for s in range(4) for r in range(ranks)]
    seen = set()

    def rec(i, hands):
        if i == len(cards):
            # UNE permutation, appliquée aux quatre joueurs — cf. le piège en tête.
            seen.add(min(
                tuple(tuple(hands[p][sigma[s]] for s in range(4)) for p in range(4))
                for sigma in permutations(range(4))
            ))
            return
        s, r = cards[i]
        for p in range(4):
            if sum(bin(x).count("1") for x in hands[p]) < ranks:
                hands[p][s] |= 1 << r
                rec(i + 1, hands)
                hands[p][s] &= ~(1 << r)

    rec(0, [[0] * 4 for _ in range(4)])
    return len(seen)


def brute_hand_classes():
    """Recompte les 472 579 classes par énumération des 10 518 300 mains."""
    by_pc = defaultdict(list)
    for m in range(256):
        by_pc[bin(m).count("1")].append(m)
    seen = set()
    for k in product(range(9), repeat=4):
        if sum(k) != 8:
            continue
        for m0 in by_pc[k[0]]:
            for m1 in by_pc[k[1]]:
                for m2 in by_pc[k[2]]:
                    for m3 in by_pc[k[3]]:
                        seen.add(tuple(sorted((m0, m1, m2, m3))))
    return len(seen)


def verify():
    print("Burnside — une main de 8 cartes")
    h = burnside_hand()
    print(f"   brut {comb(32, 8):,}  →  {h:,} classes   (brut/24 = {comb(32,8)/24:,.1f},"
          " pas entier)")
    assert h == colver.NUM_HAND_CLASSES == 472_579

    print("\nBurnside — une donne complète 4×8")
    d = burnside_deal()
    raw = factorial(32) // factorial(8) ** 4
    print(f"   brut {raw:,}  →  {d:,} classes")
    assert d == 4_148_577_738_928_080

    print("\nValidation de la formule sur jeux réduits (force brute exhaustive)")
    for r in (1, 2, 3):
        b, f = burnside_deal(r), brute_deal(r)
        print(f"   R={r} : burnside {b:,}  force brute {f:,}  {'ok' if b == f else 'ÉCART'}")
        assert b == f

    print("\nRecomptage exhaustif des classes de mains (10 518 300 mains)")
    n = brute_hand_classes()
    print(f"   {n:,}  {'ok' if n == h else 'ÉCART'}")
    assert n == h
    print("\nTout concorde.")


# --------------------------------------------------------------------------
# Mode par défaut : concentration des codes
# --------------------------------------------------------------------------

def concentration():
    total_classes = colver.NUM_HAND_CLASSES_TRUMP
    print(f"Parcours des {total_classes:,} classes à atout désigné"
          " (codes lus depuis le binding Rust)…")
    acc = [Counter() for _ in LEVELS]
    total = 0
    for cid in range(total_classes):
        cards = colver.hand_from_class_id_trump(cid)
        # Poids = taille de l'orbite, soit le nombre d'assignations distinctes des
        # 3 masques de côté aux 3 couleurs de côté.
        masks = [0, 0, 0, 0]
        for c in cards:
            masks[c // 8] |= 1 << (c % 8)
        mult = Counter(masks[1:])
        weight = 6
        for v in mult.values():
            weight //= (1, 1, 2, 6)[v]
        total += weight
        for i, lvl in enumerate(LEVELS):
            acc[i][colver.hand_code(cards, 0, lvl)] += weight

    assert total == comb(32, 8), f"somme des poids {total:,} ≠ {comb(32,8):,}"
    print(f"somme des poids = {total:,} = C(32,8) ✓\n")

    print(f"{'niveau':8s} {'codes':>8s} {'50 % des mains':>16s} {'90 %':>10s}"
          f" {'plus gros':>11s}")
    for i, lvl in enumerate(LEVELS):
        vals = sorted(acc[i].values(), reverse=True)
        cum = n50 = n90 = 0
        for j, v in enumerate(vals, 1):
            cum += v
            if not n50 and cum >= 0.5 * total:
                n50 = j
            if not n90 and cum >= 0.9 * total:
                n90 = j
        print(f"{lvl:8s} {len(acc[i]):8,} {n50:16,} {n90:10,}"
              f" {100*vals[0]/total:10.2f}%")

    print("\nLes 15 codes les plus fréquents au niveau « tops » :")
    for code, w in acc[LEVELS.index("tops")].most_common(15):
        print(f"   {code:<24s} {100*w/total:5.2f}%")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", action="store_true",
                    help="re-dénombre tout indépendamment du moteur (quelques minutes)")
    args = ap.parse_args()
    verify() if args.verify else concentration()


if __name__ == "__main__":
    main()
