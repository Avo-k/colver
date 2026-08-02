#!/usr/bin/env python3
"""À quelle fréquence les paliers d'enchère 170 / 180 sont-ils réalisables ?

Question : faut-il ouvrir l'échelle d'enchères au-dessus de 160 ? Les paliers 170 et 180 ne
sont atteignables qu'avec la belote (le maximum sans capot est 162 points de plis), et tout
palier ≥ 190 exige le capot (252). On mesure donc, en double-mort (DD, jeu parfait des deux
côtés), la fréquence des mains où un camp peut faire 170 ou 180 **sans** pouvoir faire capot :
c'est exactement le domaine que ces paliers ajouteraient au jeu.

    env -u CONDA_PREFIX uv run python scripts/analysis/bid_ceiling_170_180.py --deals 20000

Résultat commenté : docs/rules-survey/matrices/encheres.md, §2 « Plafond ».
"""
import argparse
import random
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import colver

# card.rs : Piques 0-7, Cœurs 8-15, Carreaux 16-23, Trèfles 24-31
# rangs : 7=0, 8=1, 9=2, V=3, D=4, R=5, 10=6, A=7
QUEEN, KING = 4, 5
CAPOT_PTS = 252


def deal(rng):
    cards = list(range(32))
    rng.shuffle(cards)
    return [cards[i * 8:(i + 1) * 8] for i in range(4)]


def has_belote(hand_set, suit):
    """Belote = Dame ET Roi d'atout dans la MÊME main (cf. RULES.md)."""
    return (suit * 8 + QUEEN) in hand_set and (suit * 8 + KING) in hand_set


def analyse_one(args):
    dealer, hands = args
    env = colver.Env.deal_with_hands(dealer, hands)
    suits = env.solve_all_suits()["suits"]
    sets = [set(h) for h in hands]
    out = []
    for side in (0, 1):  # 0 = NS (joueurs 0,2), 1 = EO (joueurs 1,3)
        seats = (0, 2) if side == 0 else (1, 3)
        rows = []
        for suit in range(4):
            pts = suits[suit][side]
            belote = 20 if any(has_belote(sets[s], suit) for s in seats) else 0
            rows.append((pts, belote, pts + belote))
        out.append(rows)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=8)
    a = ap.parse_args()

    rng = random.Random(a.seed)
    jobs = [(rng.randrange(4), deal(rng)) for _ in range(a.deals)]

    cat = Counter()
    # détail des cas intéressants : le meilleur total atteignable hors capot
    best_nocapot = Counter()
    n_sides = 0
    with ThreadPoolExecutor(max_workers=a.threads) as ex:
        for res in ex.map(analyse_one, jobs, chunksize=16):
            for rows in res:  # une entrée par camp
                n_sides += 1
                capot = any(p == CAPOT_PTS for p, _, _ in rows)
                best_t = max(t for _, _, t in rows)
                best_t_nc = max((t for p, _, t in rows if p < CAPOT_PTS), default=0)
                if capot:
                    cat["capot réalisable"] += 1
                elif best_t >= 180:
                    cat["180 réalisable, pas capot"] += 1
                elif best_t >= 170:
                    cat["170 réalisable, pas capot"] += 1
                elif best_t >= 160:
                    cat["160-169"] += 1
                else:
                    cat["< 160"] += 1
                if not capot:
                    best_nocapot[min(best_t_nc // 10 * 10, 190)] += 1

    print(f"\n{a.deals} donnes, graine {a.seed} — {n_sides} couples (donne, camp)\n")
    print(f"{'catégorie':<32} {'n':>8} {'%':>8}")
    print("-" * 50)
    for k in ("capot réalisable", "180 réalisable, pas capot", "170 réalisable, pas capot",
              "160-169", "< 160"):
        n = cat[k]
        print(f"{k:<32} {n:>8} {100 * n / n_sides:>7.3f}%")

    add = cat["170 réalisable, pas capot"] + cat["180 réalisable, pas capot"]
    print("-" * 50)
    print(f"{'ce qu''ajouteraient 170+180':<32} {add:>8} {100 * add / n_sides:>7.3f}%")
    print(f"{'(pour comparaison) capot':<32} {cat['capot réalisable']:>8} "
          f"{100 * cat['capot réalisable'] / n_sides:>7.3f}%")

    print("\nDistribution du meilleur total atteignable hors capot (par dizaine) :")
    for k in sorted(best_nocapot):
        n = best_nocapot[k]
        label = f"{k}+" if k >= 190 else f"{k}-{k + 9}"
        print(f"  {label:>10} {n:>8} {100 * n / n_sides:>7.3f}%")


if __name__ == "__main__":
    main()
