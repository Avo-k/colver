#!/usr/bin/env python3
"""Combien vaut une carte ? Mesure appariée sur le solveur DD.

Sert de justification chiffrée au contenu de `HandCode` (colver-core/src/hand_class.rs) :
on ne code que ce qui pèse, et « ce qui pèse » se mesure au lieu de se postuler.

Protocole. Donne aléatoire, siège 0 = « nous ». Pour chaque carte `c` de notre main,
on l'échange contre la plus faible carte de la même couleur détenue par un adversaire,
et on re-résout. Le reste de la donne est **identique** entre les deux solves, donc la
variance de la donne s'annule — c'est ce qui rend la mesure lisible sur quelques
centaines de donnes seulement.

⚠️ **Le piège, et il n'est pas anodin.** « La plus faible carte de la couleur » n'est
pas la même selon le rôle de la couleur : à l'atout l'ordre est J 9 A 10 K Q 8 7, et
le Valet a un indice de rang *bas* (3) tout en étant la carte la plus forte. Choisir
le partenaire d'échange par l'ordre naturel donne donc parfois « on donne son Roi et
on reçoit le Valet » — un échange qui *améliore* la main. Symptôme observé : les lignes
K et Q de l'atout ressortaient **négatives**. D'où deux passes séparées, chacune avec
son ordre, chacune ne lisant que le solve où la couleur a le bon rôle.

Résultat de référence (2026-08-02, 600 donnes atout / 400 côté), perte moyenne en
points DD, échelle 0-252, IC 95 % ≈ ±1 à ±2 sur les grosses lignes :

    ATOUT   J +49,2  9 +18,9  A +9,5  10 +5,6  K +1,8  Q +0,9  8 +0,4
    CÔTÉ    A +26,0  10 +6,3  K +1,5  Q −0,1  9 0,0  8 0,0  J −0,5

Deux cartes d'atout portent 68 des ~86 points d'importance de la couleur. À côté, tout
ce qui est sous le 10 est du bruit statistique. Le Valet de côté est *négatif* : le
donner à l'adversaire rapporte un demi-point, ce sont 2 points de mangeaille qu'on
récupère au pli plutôt que de les défausser soi-même.

Portée. Valeur **marginale sur fond aléatoire, en jeu parfait**, dealer fixé. Ce n'est
pas une valeur d'enchère : le J d'atout ne vaut pas 49 dans toutes les mains, il vaut
ça en moyenne contre un remplaçant faible.

    uv run python scripts/analysis/card_importance.py --deals 600
"""

import argparse
import os
import sys
import threading
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

NAME = {0: "7", 1: "8", 2: "9", 3: "J", 4: "Q", 5: "K", 6: "T", 7: "A"}
# Force à l'atout, indexée par bit de rang — miroir de card.rs::TRUMP_STRENGTH.
TRUMP_STRENGTH = {0: 0, 1: 1, 2: 6, 3: 7, 4: 2, 5: 3, 6: 4, 7: 5}
# À côté, la force suit l'ordre naturel des bits.
PLAIN_STRENGTH = {r: r for r in range(8)}
DEALER = 0

_tl = threading.local()


def _env():
    if not hasattr(_tl, "e"):
        _tl.e = colver.Env()
        _tl.e.reset()
    return _tl.e


def _solve(hands):
    """Points N-S en double-mort, pour chacun des 4 atouts."""
    e = _env()
    e.redeal_with_hands(DEALER, [sorted(h) for h in hands])
    return [s[0] for s in e.solve_all_suits()["suits"]]


def _one_deal(seed, strength, trump_role):
    """Une donne : échange chaque carte de la main 0 et mesure la perte DD.

    `strength` ordonne les cartes pour choisir le remplaçant ; `trump_role` dit si
    l'on ne retient que le solve où la couleur est l'atout (passe atout) ou tous les
    autres (passe côté).
    """
    rng = __import__("random").Random(seed)
    deck = list(range(32))
    rng.shuffle(deck)
    hands = [deck[i * 8 : (i + 1) * 8] for i in range(4)]
    base = _solve(hands)
    out = []
    for c in hands[0]:
        suit = c // 8
        # La plus faible carte de la couleur chez un adversaire (sièges 1 et 3).
        cands = [
            (strength[d % 8], d, p) for p in (1, 3) for d in hands[p] if d // 8 == suit
        ]
        if not cands:
            continue
        _, d, p = min(cands)
        if strength[d % 8] > strength[c % 8]:
            continue  # on ne mesure que des dégradations
        h2 = [list(h) for h in hands]
        h2[0].remove(c)
        h2[0].append(d)
        h2[p].remove(d)
        h2[p].append(c)
        swapped = _solve(h2)
        trumps = [suit] if trump_role else [t for t in range(4) if t != suit]
        for t in trumps:
            out.append((c % 8, base[t] - swapped[t]))
    return out


def _pass(label, deals, workers, strength, trump_role, order):
    with ThreadPoolExecutor(max_workers=workers) as ex:
        batches = list(
            ex.map(lambda s: _one_deal(s, strength, trump_role), range(deals))
        )
    agg = defaultdict(list)
    for batch in batches:
        for rank, delta in batch:
            agg[rank].append(delta)

    print(f"\n--- {label} — perte DD moyenne quand la carte est remplacée")
    print("    par la plus faible de la couleur détenue par un adversaire")
    rows = []
    for r in range(8):
        v = agg.get(r, [])
        if len(v) < 20:
            continue
        mean = sum(v) / len(v)
        se = (sum((x - mean) ** 2 for x in v) / len(v) / len(v)) ** 0.5
        rows.append((order[r], NAME[r], mean, 1.96 * se, len(v)))
    for _, name, mean, ci, n in sorted(rows, key=lambda x: -x[2]):
        print(f"    {name}  {mean:+7.2f} pts  (±{ci:.2f})   n={n}")
    return agg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=600, help="donnes par passe")
    ap.add_argument("--workers", type=int, default=12,
                    help="solve_all_suits relâche le GIL, les threads travaillent vraiment")
    args = ap.parse_args()

    print(f"{args.deals} donnes par passe, solves appariés (~310 ms les 4 couleurs)")
    _pass("ATOUT", args.deals, args.workers, TRUMP_STRENGTH, True, TRUMP_STRENGTH)
    _pass("CÔTÉ", args.deals, args.workers, PLAIN_STRENGTH, False, PLAIN_STRENGTH)
    print(
        "\nContrôle de cohérence : chaque colonne doit être monotone dans l'ordre de\n"
        "force de son rôle. Une ligne négative en haut de tableau signale que le\n"
        "remplaçant a été choisi avec le mauvais ordre."
    )


if __name__ == "__main__":
    main()
