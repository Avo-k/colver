#!/usr/bin/env python3
"""Le capot est-il une action morte chez v6 ?

Question §1.3 de docs/bid/bid_v7_plan.md. Trois mesures indépendantes, de la plus
agrégée à la plus spécifique.

  1. **Fréquence** : on fait jouer N enchères complètes par v6 aux quatre sièges et
     on regarde la distribution des contrats. Référence : **0 capot sur 3000**, et
     0,47 % de contrats à 160 (le plafond non-capot).

  2. **Sondes sur mains capot forcé** : des mains qui prennent les huit levées *quelle
     que soit* la répartition des 24 autres cartes, donc sans qu'aucun échantillonnage
     de mondes ne soit nécessaire pour trancher. Trois familles construites à la main,
     avec l'argument qui les rend forcées dans le code.

  3. **Rareté** : à quelle fréquence ces mains apparaissent dans un pool, pour
     distinguer « le signal manque » de « le signal est là mais n'est jamais visité ».

Le résultat le plus net est la troisième sonde : sur **les huit cartes d'une même
couleur** — le capot le plus trivial du jeu — v6 annonce 140 avec un Q supérieur de
0,20 à celui du capot, sur une étendue de 0,79. Ce n'est pas un ex æquo, c'est une
réponse fausse et confiante. Sur les deux autres familles il sature le plafond (160),
signature d'une action jamais explorée plutôt que d'une évaluation prudente.

La rareté (~10⁻⁴) explique pourquoi : le modèle joue ses propres enchères, il
n'annonce jamais capot, donc l'action ne reçoit jamais de gradient. Boucle fermée que
ni plus de données ni plus de pas ne rouvriront — cf. §3.3.

Pour chiffrer ce que coûte l'erreur (585 pts/donne), voir bid_candidates.py, qui force
chaque candidate et laisse l'enchère se poursuivre.

    uv run python scripts/analysis/bid_capot_probe.py --auctions 3000
"""

import argparse
import os
import random
import sys
from collections import Counter
from math import comb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

BID_MODEL = "models/bid_v6_isdd_resume/bid_nn_final.bin"
NAME = {0: "7", 1: "8", 2: "9", 3: "J", 4: "Q", 5: "K", 6: "T", 7: "A"}
SUIT = ["♠", "♥", "♦", "♣"]
# J 9 A T K Q : les six plus fortes à l'atout.
TOP6 = [3, 2, 7, 6, 5, 4]


def show(hand):
    return " ".join(
        f"{NAME[c % 8]}{SUIT[c // 8]}"
        for c in sorted(hand, key=lambda c: (c // 8, -(c % 8)))
    )


def frequency(env, n, seed):
    rng = random.Random(seed)
    values = Counter()
    capot = void = 0
    for _ in range(n):
        deck = list(range(32))
        rng.shuffle(deck)
        env.redeal_with_hands(
            rng.randrange(4), [sorted(deck[i * 8 : (i + 1) * 8]) for i in range(4)]
        )
        top = None
        while env.phase() == 0 and not env.is_terminal():
            a = env.action_bid_nn()["best_action"]
            if 1 <= a <= 36:
                top = 80 + 10 * ((a - 1) // 4)
            elif 37 <= a <= 40:
                top = "capot"
            env.step(a)
        if top is None:
            void += 1
        elif top == "capot":
            capot += 1
        else:
            values[top] += 1

    print(f"{n:,} enchères jouées par v6 aux quatre sièges")
    print(f"   contrats capot           : {capot} ({100*capot/n:.2f} %)")
    print(f"   contrats à 160 (plafond) : {values[160]} ({100*values[160]/n:.2f} %)")
    print(f"   donnes passées           : {void} ({100*void/n:.1f} %)")
    print("   distribution :",
          {k: f"{100*v/n:.1f}%" for k, v in sorted(values.items())})


def probe(env, hand, label, rng, forced=True, trials=6):
    """Fait parler le porteur en premier, sur `trials` répartitions du reste."""
    seen = set()
    for t in range(trials):
        rest = [c for c in range(32) if c not in hand]
        rng.shuffle(rest)
        dealer = t % 4
        holder = (dealer + 1) % 4  # le premier à parler
        hands, k = [None] * 4, 0
        hands[holder] = sorted(hand)
        for p in range(4):
            if p != holder:
                hands[p] = sorted(rest[k : k + 8])
                k += 8
        env.redeal_with_hands(dealer, hands)
        assert env.current_player() == holder, "le porteur doit parler en premier"
        d = env.action_bid_nn()
        q = dict(d["q_values"])
        best_capot = max((q[x] for x in q if 37 <= x <= 40), default=float("nan"))
        seen.add((colver.Env.action_name(d["best_action"], 0),
                  round(q[d["best_action"]], 4), round(best_capot, 4)))

    print(f"\n{label}\n   {show(hand)}")
    for annonce, qa, qc in sorted(seen):
        flag = ""
        if forced and not annonce.startswith("CAPOT"):
            flag = "   ← rate le capot"
        print(f"   annonce {annonce:<7s} Q={qa:+.4f}   meilleur Q capot={qc:+.4f}{flag}")
    if len(seen) > 1:
        print("   (plusieurs réponses selon la répartition du reste — l'obs ne voit"
              " pourtant que la main)")


def probes(env, seed):
    rng = random.Random(seed)
    # Six plus gros atouts + As-Dix ailleurs : on tire deux fois atout pour épuiser
    # les deux atouts adverses, on encaisse les quatre atouts maîtres restants, puis
    # l'As et le Dix passent puisque plus personne n'a d'atout. Huit levées.
    probe(env, [0 * 8 + r for r in TOP6] + [1 * 8 + 7, 1 * 8 + 6],
          "CAPOT FORCÉ — six gros atouts ♠ + As-Dix ♥", rng)
    probe(env, [1 * 8 + r for r in TOP6] + [3 * 8 + 7, 3 * 8 + 6],
          "CAPOT FORCÉ — six gros atouts ♥ + As-Dix ♣", rng)
    # Les huit cartes d'une couleur : personne d'autre n'en a, tout est maître.
    probe(env, list(range(16, 24)),
          "CAPOT FORCÉ — les huit carreaux (le cas le plus trivial du jeu)", rng)
    # Contrôle négatif : très forte, mais pas forcée.
    probe(env, [0 * 8 + r for r in TOP6[:5]] + [1 * 8 + 7, 2 * 8 + 7, 3 * 8 + 0],
          "CONTRÔLE — très forte mais capot NON forcé (5 gros atouts + 2 As + un 7)",
          rng, forced=False)


def rarity(pool_deals):
    hands = 4 * pool_deals
    total = comb(32, 8)
    print(f"\nRareté, rapportée à un pool de {pool_deals:,} donnes = {hands:,} mains")
    for label, count in (
        ("main contenant les six gros atouts d'une couleur", 4 * comb(26, 2)),
        ("main monocolore (huit cartes d'une couleur)", 4),
    ):
        p = count / total
        print(f"   {label:50s} {p:.2e}   ~{p*hands:,.0f} occurrences")
    print("   → le signal existe dans les données ; c'est la trajectoire d'enchère"
          " qui ne l'atteint jamais.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--auctions", type=int, default=3000)
    ap.add_argument("--pool-deals", type=int, default=5_000_000)
    ap.add_argument("--bid-model", default=BID_MODEL)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=2024)
    args = ap.parse_args()

    env = colver.Env()
    env.reset()
    env.load_bid_model(args.bid_model, args.hidden)

    frequency(env, args.auctions, args.seed)
    probes(env, args.seed + 1)
    rarity(args.pool_deals)


if __name__ == "__main__":
    main()
