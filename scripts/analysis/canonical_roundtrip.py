#!/usr/bin/env python3
"""Ce que coûte une erreur sur `canonical = true`, et le contrôle de légalité d'un v7.

**Le défaut que ce script rend visible.** Un réseau d'annonce entraîné sur l'ordre
canonique des couleurs a **exactement la même taille de fichier** qu'un réseau entraîné
sur l'ordre physique : la largeur d'obs est la même (117), le nombre de couches aussi.
`AgentSpec` ne peut donc pas le deviner — d'où `canonical = true` dans le TOML, à écrire
à la main. Se tromper ne produit **ni erreur, ni avertissement, ni annonce illégale** :
le masque légal est bien appliqué, simplement dans le mauvais repère, et le réseau
annonce une couleur qui n'est pas celle qu'il voulait.

Deux mesures, et la première ne dépend pas d'un entraînement :

1. **Le coût de l'erreur**, mesuré sur un réseau *déjà entraîné* — v6, qui est physique.
   On le lit une fois correctement (`canonical = false`) et une fois à tort
   (`canonical = true`), sur les mêmes positions. Le taux de désaccord est ce qu'une
   faute de frappe dans un TOML coûterait. Utiliser v6 plutôt qu'un v7 tout frais est
   délibéré : un réseau à quelques dizaines de milliers de pas a des Q quasi plates,
   donc son argmax est du bruit et le désaccord mesuré ne voudrait rien dire.

2. **La légalité** du réseau à tester, dans les deux lectures. Elle doit tenir des deux
   côtés — c'est bien le problème : l'illégalité serait un garde-fou, elle n'existe pas.

    uv run python scripts/analysis/canonical_roundtrip.py \\
        --net models/bid_v6_isdd_resume/bid_nn_final.bin --deals 300
"""

import argparse
import random

import colver

SPEC = """[bid]
strategy = "nn"
model = "{net}"
hidden = 512
score_aware = true
canonical = {canon}

[play]
method = "heuristic"
"""


def agents(net, canon, seed):
    spec = SPEC.format(net=net, canon="true" if canon else "false")
    return [colver.Agent(spec, seat, seed) for seat in range(4)]


def run(net, deals, seed):
    rng = random.Random(seed)
    a_phys = agents(net, False, seed)
    a_canon = agents(net, True, seed)

    n = 0
    disagree = 0
    illegal_phys = 0
    illegal_canon = 0
    suit_changed = 0
    pass_flip = 0

    for d in range(deals):
        # Distribution faite ici plutôt que par `Env.reset()` : celui-ci tire sur un RNG
        # que le script ne contrôle pas, donc le run ne serait pas reproductible.
        cards = list(range(32))
        rng.shuffle(cards)
        hands = [sorted(cards[8 * s:8 * s + 8]) for s in range(4)]
        env = colver.Env.deal_with_hands(rng.randrange(4), hands)
        for ag in a_phys + a_canon:
            ag.init_deal(env)

        # L'enchère est jouée par la lecture CORRECTE ; la fautive est interrogée à
        # chaque position sans jamais agir. Laisser la fautive conduire produirait deux
        # enchères différentes, donc des positions non comparables — le même piège que
        # « ne jamais tirer les questions du flux que le testé consomme ».
        while env.phase() == 0 and not env.is_terminal():
            seat = env.current_player()
            legal = set(env.legal_actions())
            ref = a_phys[seat].action(env)
            alt = a_canon[seat].action(env)

            n += 1
            if ref not in legal:
                illegal_phys += 1
            if alt not in legal:
                illegal_canon += 1
            if ref != alt:
                disagree += 1
                # 0 = PASS ; 1..36 = value_idx*4 + suit_idx + 1 ; 37..40 = capot
                if ref == 0 or alt == 0:
                    pass_flip += 1
                elif 1 <= ref <= 36 and 1 <= alt <= 36:
                    if (ref - 1) % 4 != (alt - 1) % 4:
                        suit_changed += 1

            for ag in a_phys + a_canon:
                ag.observe(env, ref)
            env.step(ref)

    return dict(positions=n, disagree=disagree, illegal_phys=illegal_phys,
                illegal_canon=illegal_canon, suit_changed=suit_changed,
                pass_flip=pass_flip)


def cross_check(net, canonical, deals, seed):
    """Les deux chemins canoniques indépendants doivent rendre la MÊME annonce.

    `colver.Agent` passe par `AgentSpec` → `BidNetPolicy` (colver-core), `Env` passe par
    `load_bid_model` → `bid_net_answer` (colver-py). Ce sont deux implémentations de la
    même permutation, écrites séparément — c'est justement pourquoi les comparer vaut
    quelque chose. Un désaccord non nul est un bug de câblage dans l'une des deux, pas
    une question de force de jeu.
    """
    rng = random.Random(seed)
    ags = agents(net, canonical, seed)

    n = disagree = 0
    for _ in range(deals):
        cards = list(range(32))
        rng.shuffle(cards)
        hands = [sorted(cards[8 * s:8 * s + 8]) for s in range(4)]
        env = colver.Env.deal_with_hands(rng.randrange(4), hands)
        env.load_bid_model(net, 512, canonical)
        for ag in ags:
            ag.init_deal(env)

        while env.phase() == 0 and not env.is_terminal():
            seat = env.current_player()
            a_agent = ags[seat].action(env)
            a_env = int(env.action_bid_nn()["best_action"])
            n += 1
            disagree += a_agent != a_env
            for ag in ags:
                ag.observe(env, a_agent)
            env.step(a_agent)
    return n, disagree


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--net", default="models/bid_v6_isdd_resume/bid_nn_final.bin")
    p.add_argument("--deals", type=int, default=300)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--cross-check", action="store_true",
                   help="compare colver.Agent et Env.action_bid_nn dans les DEUX lectures")
    a = p.parse_args()

    if a.cross_check:
        print(f"\n{a.net} — Agent (colver-core) contre Env.action_bid_nn (colver-py)\n")
        bad = 0
        for canon in (False, True):
            n, d = cross_check(a.net, canon, a.deals, a.seed)
            tag = "canonical=True " if canon else "canonical=False"
            print(f"  {tag} : {d}/{n} désaccords" + ("   ✔" if d == 0 else "   ✗ CÂBLAGE"))
            bad += d
        print()
        print("  ⇒ les deux chemins concordent" if bad == 0 else
              "  ⇒ ÉCHEC : les deux implémentations de la permutation divergent")
        return

    r = run(a.net, a.deals, a.seed)
    n = r["positions"]
    print(f"\n{a.net} — {a.deals} donnes, {n} positions d'annonce\n")
    print(f"  annonces illégales, lecture correcte : {r['illegal_phys']}")
    print(f"  annonces illégales, lecture fautive  : {r['illegal_canon']}")
    print(f"    (les deux DOIVENT être 0 : le masque est appliqué des deux côtés,")
    print(f"     c'est précisément pourquoi l'erreur est silencieuse)\n")
    d = r["disagree"]
    print(f"  désaccord entre les deux lectures : {d}/{n} = {100 * d / max(n, 1):.1f} %")
    print(f"    dont couleur changée à valeur égale : {r['suit_changed']}")
    print(f"    dont passe ↔ annonce               : {r['pass_flip']}")
    print()
    if d == 0:
        print("  ⚠️ ZÉRO désaccord : soit le réseau est plat (pas assez entraîné),")
        print("     soit la lecture canonique n'est pas branchée. À élucider —")
        print("     ce n'est PAS un résultat rassurant.")
    else:
        print(f"  ⇒ une faute de frappe sur `canonical` change {100 * d / n:.0f} % des")
        print("     annonces, sans un seul signe extérieur.")


if __name__ == "__main__":
    main()
