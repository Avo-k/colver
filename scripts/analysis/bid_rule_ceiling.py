#!/usr/bin/env python3
"""Jusqu'où une règle humaine peut-elle imiter le bidder — et où bute-t-elle ?

Question ouverte par le travail d'interprétabilité
([docs/bid/interpretability/bid_rules_xgb_v2.md](../../docs/bid/interpretability/bid_rules_xgb_v2.md)) :
l'arbre distillé est d'accord avec le réseau à ~93 %, et **personne ne sait où vivent
les 7 % restants**. Ce script répond aux deux moitiés de la question.

## 1. Le plafond

Une règle écrite pour un humain ne connaît pas le nom des couleurs : « J + 2 atouts →
annonce » vaut pique comme trèfle. Elle est donc **équivariante** par construction, tout
comme les features agrégées de l'arbre (`trump_count`, `has_jack`, `side_voids`…).

Or un réseau non équivariant donne jusqu'à 24 réponses pour la même main renommée
([bid_v7_plan.md](../../docs/bid/bid_v7_plan.md) §1.1 : 24,6 % de bascules sur v6). Sur
l'orbite d'une main, une règle équivariante ne peut sortir qu'**une** réponse ; son
meilleur choix est le mode. D'où :

    plafond = moyenne sur les mains de (effectif du mode / 24)

C'est une borne supérieure **sur toute règle équivariante**, aussi profonde soit-elle,
et elle ne dépend d'aucun choix de features. Un arbre qui l'atteint n'est pas
perfectible : ce qui lui reste à expliquer est le bruit de symétrie du réseau.

Deux granularités, parce que les règles publiées portent sur les deux :
  * **exacte** — l'action entière (valeur × couleur, ou passe) ;
  * **binaire** — annoncer ou passer, la décision des tables de `bid_rules_xgb_v2` §1.

## 2. Le lieu

Le plafond global ne dit pas *où* ça casse. On regroupe donc par famille de mains
(`HandCode`, [hand_classification.md](../../docs/bid/interpretability/hand_classification.md)),
dont le contenu est mesuré au solve apparié plutôt que deviné — 80 codes au niveau
`trump`, dont 28 couvrent 90 % des mains.

**L'ancrage de la famille doit être stable sous renommage**, sinon on regroupe des
choses différentes à chaque permutation. `hand_code(main, atout)` est invariant par
renommage *simultané* de la main et de l'atout, donc il suffit de choisir l'atout par une
fonction pure de la main : **la couleur qu'on envisagerait**, `argmax evaluate_for_trump`,
départagée par l'ordre d'atout. Les ex æquo sont des couleurs de rangs identiques —
réellement interchangeables — donc leur code coïncide et l'ambiguïté est vide.

Sous préfixe d'enchère il n'y a plus rien à choisir : l'atout est **donné** par l'annonce
en cours, et c'est lui qui ancre. `hand_code(main, couleur annoncée)` répond alors
exactement à la question du siège — « que vaut ma main *contre* / *pour* ce contrat ? ».

Référence (2026-08-03) : voir `docs/measurements/index.jsonl`.

    uv run python scripts/analysis/bid_rule_ceiling.py --deals 120000 --tag v6-opening
    uv run python scripts/analysis/bid_rule_ceiling.py --deals 120000 --prior 80C \
        --tag v6-defense
"""

import argparse
import os
import random
import sys
import time
from collections import Counter, defaultdict
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

import runlog  # noqa: E402
# `apply_prior` rejoue le préfixe **permuté par σ** : une permutation agit sur la
# position entière, historique compris. Rejouer 100♣ tel quel sous σ comparerait deux
# positions différentes. Partagé avec bid_equivariance.py plutôt que recopié.
from bid_equivariance import IDENTITY, apply_prior, parse_action  # noqa: E402

BID_MODEL = "models/bid_v6_isdd_resume/bid_nn_final.bin"
ALL_PERMS = list(permutations(range(4)))

# Force d'atout par bit de rang : 7=0 8=1 9=2 J=3 Q=4 K=5 10=6 A=7 (card.rs)
TRUMP_STRENGTH = {3: 7, 2: 6, 7: 5, 6: 4, 5: 3, 4: 2, 1: 1, 0: 0}


def perm_card(c, sigma):
    return sigma[c // 8] * 8 + (c % 8)


def perm_action(a, sigma):
    """0 = PASS ; 1-36 = valeur×4 + couleur ; 37-40 = capot×couleur ; 41-42 = coinches."""
    if a == 0:
        return 0
    if 1 <= a <= 36:
        value, suit = divmod(a - 1, 4)
        return value * 4 + sigma[suit] + 1
    if 37 <= a <= 40:
        return 37 + sigma[a - 37]
    return a


def inverse(sigma):
    inv = [0] * 4
    for i, s in enumerate(sigma):
        inv[s] = i
    return tuple(inv)


def anchor_suit(hand, scores=None):
    """La couleur qu'on **envisagerait** comme atout — fonction pure de la main.

    C'est l'étiquette de la famille, donc elle doit nommer la question que le joueur se
    pose. `evaluate_for_trump` (fourni par `scores`) est ce critère : il pèse le Valet,
    le 9 *et* la longueur. Départage par `suit_key`, un ordre total sur l'ensemble des
    rangs, donc les ex æquo sont des couleurs réellement interchangeables — leur
    `hand_code` coïncide et le départage est sans effet.

    Sans `scores` on retombe sur `suit_key` seul, l'ordre d'atout lexicographique. Il est
    tout aussi stable, mais il **sur-pondère la plus haute carte** : un Valet sec y bat
    une couleur de quatre cartes à l'As, si bien qu'une famille étiquetée « `T1.J` »
    regroupe des mains que v6 annonce — dans l'autre couleur. Un plafond juste sous une
    étiquette qui décrit autre chose est le pire des deux mondes. Ne s'utilise que là où
    aucun `Env` n'est disponible (le contrôle d'invariance).

    ⚠️ Dans `suit_key`, le bit d'une carte est sa **force**, pas son complément : le
    Valet (force 7) pèse 128, pas 1. Inversé, la clé reste un ordre total stable au
    renommage — donc le contrôle passe et les plafonds restent justes — mais elle
    désigne la couleur la plus *pauvre* de la main.
    """
    if scores is None:
        return max(range(4), key=lambda s: suit_key(hand, s))
    return max(range(4), key=lambda s: (scores[s], suit_key(hand, s)))


def suit_key(hand, s):
    """Masque des rangs de `s` réécrit dans l'ordre de force d'atout. Voir `anchor_suit`."""
    k = 0
    for c in hand:
        if c // 8 == s:
            k |= 1 << TRUMP_STRENGTH[c % 8]
    return k


def deals(rng, n):
    for _ in range(n):
        deck = list(range(32))
        rng.shuffle(deck)
        yield rng.randrange(4), [sorted(deck[i * 8 : (i + 1) * 8]) for i in range(4)]


def check_anchor_stability(env, rng, n=200):
    """Le **code** de la main ancrée doit survivre au renommage.

    On ne teste pas `anchor(σ·main) == σ(anchor(main))` : c'est faux quand deux couleurs
    ont la même clé, et sans conséquence — ces couleurs ont le même ensemble de rangs,
    donc désigner l'une ou l'autre laisse le même code (les couleurs de côté ne diffèrent
    que par un échange entre elles, que le code trie). C'est l'égalité des codes qui est
    la propriété dont dépend le regroupement, et c'est elle qu'on vérifie.

    Contrôle gratuit qui vient avec : l'ancre lit `evaluate_for_trump` **du moteur**, donc
    la boucle vérifie du même coup que cette fonction est équivariante. Un biais d'indice
    de couleur y passerait autrement inaperçu et déplacerait des mains de famille.
    """
    ties = 0
    for _ in range(n):
        deck = list(range(32))
        rng.shuffle(deck)
        dealer = rng.randrange(4)
        hands = [sorted(deck[i * 8 : (i + 1) * 8]) for i in range(4)]
        env.redeal_with_hands(dealer, hands)
        seat = env.current_player()
        a = anchor_suit(hands[seat], env.evaluate_hand(seat)["scores"])
        ref = colver.hand_code(hands[seat], a, "full")
        for sigma in ALL_PERMS:
            ph = [sorted(perm_card(c, sigma) for c in h) for h in hands]
            env.redeal_with_hands(dealer, ph)
            a2 = anchor_suit(ph[seat], env.evaluate_hand(seat)["scores"])
            if a2 != sigma[a]:
                ties += 1
            if colver.hand_code(ph[seat], a2, "full") != ref:
                raise SystemExit(f"code non invariant : {hands[seat]} σ={sigma}")
    return ties


LEVELS = ["length", "trump", "shape", "tops", "full"]
LUT_LEVELS = LEVELS + ["trump+2e", "full+2e"]


def prior_suit(prior):
    """La couleur nommée par le préfixe d'enchère, ou None si le préfixe est vide/passes.

    En défense comme en soutien, l'atout n'est plus à choisir : il est **donné** par
    l'annonce en cours. C'est donc lui qui ancre la famille, et non plus la meilleure
    couleur de la main — `hand_code(main, couleur annoncée)` répond exactement à la
    question qu'on se pose (« que vaut ma main *contre* / *pour* ce contrat ? »).
    """
    for a in reversed(prior):
        if 1 <= a <= 36:
            return (a - 1) % 4
        if 37 <= a <= 40:
            return a - 37
    return None


def collect(env, decide, n, seed, level, prior=()):
    """Pour chaque main : les 24 réponses ramenées dans l'espace identité, + ses codes."""
    rng = random.Random(seed)
    rows = []
    named = prior_suit(prior)
    for dealer, hands in deals(rng, n):
        env.redeal_with_hands(dealer, hands)
        apply_prior(env, prior, IDENTITY)
        seat = env.current_player()
        hand = hands[seat]
        anchor = named if named is not None else anchor_suit(
            hand, env.evaluate_hand(seat)["scores"])
        answers = []
        for sigma in ALL_PERMS:
            env.redeal_with_hands(
                dealer, [sorted(perm_card(c, sigma) for c in h) for h in hands]
            )
            apply_prior(env, prior, sigma)
            answers.append(perm_action(decide(env), inverse(sigma)))
        codes = {lv: colver.hand_code(hand, anchor, lv) for lv in LEVELS}
        # `HandCode` décrit la main **avec un atout désigné** : les couleurs de côté n'y
        # gardent que As / Dix / longueur, donc le J et le 9 d'une *autre* couleur —
        # ce qui déciderait d'y jouer l'atout — sont perdus. On garde le descripteur de
        # la deuxième meilleure couleur pour pouvoir mesurer ce que ça coûte.
        second = max((s for s in range(4) if s != anchor), key=lambda s: suit_key(hand, s))
        ranks = {c % 8 for c in hand if c // 8 == second}
        codes["full+2e"] = codes["full"] + "|" + str(len(ranks)) + "".join(
            n for r, n in ((3, "J"), (2, "9"), (7, "A"), (6, "T")) if r in ranks)
        codes["trump+2e"] = codes["trump"] + "|" + str(len(ranks)) + "".join(
            n for r, n in ((3, "J"), (2, "9"), (7, "A"), (6, "T")) if r in ranks)
        rows.append({
            "code": codes[level],
            "coarse": codes["length"],
            "codes": codes,
            "answers": answers,
        })
    return rows


def action_value(a):
    """Classe de niveau annoncé : 0 = passe, 8..16 = 80..160, 25 = capot."""
    if a == 0 or a >= 41:
        return a
    if 37 <= a <= 40:
        return 25
    return (a - 1) // 4 + 8


def lookup_by_level(rows, seed=0):
    """« Une réponse par code » — la règle la plus littérale qu'un humain puisse tenir.

    Mesurée **hors échantillon** : à 6 654 codes pour `full`, une table apprise et lue
    sur les mêmes mains se note elle-même. Un code jamais vu à l'apprentissage retombe
    sur la réponse majoritaire globale, ce qui est exactement ce que ferait un joueur
    devant une main dont la règle ne parle pas.

    La cible d'apprentissage est le **mode de l'orbite** : c'est le meilleur objet
    équivariant que le réseau contienne, donc la seule cible qu'une table par code puisse
    viser.

    ⚠️ **La note n'est pas l'accord avec cette cible, c'est l'accord avec l'orbite** — la
    fraction des 24 réponses du réseau que la règle retrouve. Les deux échelles diffèrent
    et les confondre casse la comparaison : une règle notée contre le réseau *symétrisé*
    vise 100 %, tandis que le plafond borne l'accord avec le réseau *tel qu'il répond*.
    Le symptôme qui l'attrape : une table qui « dépasse » son plafond (vu sur v2, 97,2 %
    contre 96,8 %).
    """
    idx = list(range(len(rows)))
    random.Random(seed).shuffle(idx)
    cut = int(0.7 * len(rows))
    tr, te = idx[:cut], idx[cut:]

    def mode(vals):
        return Counter(vals).most_common(1)[0][0]

    out = []
    for lv in LUT_LEVELS:
        for name, fn in (("annonce", lambda a: int(a != 0)),
                         ("niveau", action_value)):
            orbits = [Counter(fn(a) for a in r["answers"]) for r in rows]
            target = [c.most_common(1)[0][0] for c in orbits]
            groups = defaultdict(list)
            for i in tr:
                groups[rows[i]["codes"][lv]].append(target[i])
            table = {c: mode(v) for c, v in groups.items()}
            fallback = mode([target[i] for i in tr])
            n24 = len(rows[0]["answers"])
            agree = sum(orbits[i][table.get(rows[i]["codes"][lv], fallback)]
                        for i in te) / (len(te) * n24)
            ceil = sum(orbits[i].most_common(1)[0][1] for i in te) / (len(te) * n24)
            out.append({"level": lv, "target": name, "codes_seen": len(table),
                        "acc": 100 * agree, "ceiling": 100 * ceil})
    return out


def ceilings(rows):
    """Plafond d'accord de toute règle équivariante : le mode, par main."""
    exact = binary = 0.0
    for r in rows:
        a = r["answers"]
        exact += Counter(a).most_common(1)[0][1] / len(a)
        nb_pass = sum(1 for x in a if x == 0)
        binary += max(nb_pass, len(a) - nb_pass) / len(a)
    return {"exact": 100 * exact / len(rows), "binary": 100 * binary / len(rows)}


def per_family(rows, key="code"):
    groups = defaultdict(list)
    for r in rows:
        groups[r[key]].append(r)
    out = []
    for code, rs in groups.items():
        c = ceilings(rs)
        distinct = sum(len(set(r["answers"])) for r in rs) / len(rs)
        pass_rate = 100 * sum(
            1 for r in rs for x in r["answers"] if x == 0
        ) / (len(rs) * len(rs[0]["answers"]))
        out.append({"code": code, "n": len(rs), "ceiling_exact": c["exact"],
                    "ceiling_binary": c["binary"], "distinct": distinct,
                    "pass_pct": pass_rate})
    out.sort(key=lambda d: -d["n"])
    return out


def show(title, fams, total, limit=None):
    print(f"\n{title}")
    print(f"   {'code':22s} {'part':>6s} {'plafond ex.':>11s} "
          f"{'plafond b/p':>11s} {'réponses':>9s} {'passe':>7s}")
    rows = fams if limit is None else fams[:limit]
    for f in rows:
        print(f"   {f['code']:22s} {100*f['n']/total:5.1f}% "
              f"{f['ceiling_exact']:10.1f}% {f['ceiling_binary']:10.1f}% "
              f"{f['distinct']:8.1f} {f['pass_pct']:6.1f}%")
    if limit is not None and len(fams) > limit:
        print(f"   … {len(fams)-limit} codes de plus")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=5000)
    ap.add_argument("--bid-model", default=BID_MODEL)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--level", default="trump", help="niveau de HandCode pour les familles")
    ap.add_argument("--prior", default="",
                    help="préfixe d'enchère, p.ex. '100C' (l'adversaire ouvre) ou "
                         "'100C P' (le partenaire ouvre). Vide = ouverture.")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    prior = [parse_action(t) for t in args.prior.replace(",", " ").split()]
    # Le siège qui décide est `len(prior)` crans après le premier parleur, donc la
    # longueur du préfixe nomme le régime (même convention que bid_equivariance.py).
    regime = {0: "ouverture", 1: "après l'adversaire", 2: "après le partenaire"}.get(
        len(prior), f"préfixe de {len(prior)} actions")
    print(f"Régime : {regime}" + (f"   (préfixe : {args.prior})" if prior else ""))
    if prior_suit(prior) is not None:
        print("   familles ancrées sur la couleur annoncée, pas sur la meilleure de la main")

    env = colver.Env()
    env.reset()
    env.load_bid_model(args.bid_model, args.hidden)

    ties = check_anchor_stability(env, random.Random(7))
    print(f"Contrôle d'ancrage : OK (200 mains × 24 permutations, "
          f"{ties} ex æquo absorbés par l'égalité des codes)")

    t0 = time.monotonic()
    rows = collect(env, lambda e: e.action_bid_nn()["best_action"], args.deals, args.seed,
                   args.level, prior)
    took = time.monotonic() - t0

    glob = ceilings(rows)
    distinct = sum(len(set(r["answers"])) for r in rows) / len(rows)
    never_flips = 100 * sum(1 for r in rows if len(set(r["answers"])) == 1) / len(rows)

    print(f"\nModèle : {args.bid_model}")
    print(f"{args.deals:,} mains × 24 permutations = {24*args.deals:,} annonces"
          f"   ({took:.1f} s)")
    print("\nPlafond d'accord de toute règle équivariante")
    print(f"   action exacte      : {glob['exact']:5.1f} %")
    print(f"   annoncer / passer  : {glob['binary']:5.1f} %")
    print(f"   réponses distinctes par main : {distinct:.2f} / 24"
          f"   ({never_flips:.1f} % des mains sont stables)")

    fam = per_family(rows, "code")
    coarse = per_family(rows, "coarse")
    show(f"Par famille — niveau '{args.level}' ({len(fam)} codes vus), les 20 plus peuplées",
         fam, len(rows), 20)
    worst = sorted([f for f in fam if f["n"] >= max(20, args.deals // 500)],
                   key=lambda d: d["ceiling_exact"])[:12]
    show("Les 12 familles où le plafond est le plus bas (n ≥ seuil)", worst, len(rows))
    show(f"Par famille — niveau 'length' ({len(coarse)} codes)", coarse, len(rows))

    lut = lookup_by_level(rows, args.seed)
    print("\nTable de correspondance « une réponse par code », hors échantillon")
    print(f"   {'niveau':10s} {'codes vus':>10s}  {'annoncer/passer':>22s}"
          f"  {'niveau annoncé':>22s}")
    for lv in LUT_LEVELS:
        a = next(r for r in lut if r["level"] == lv and r["target"] == "annonce")
        b = next(r for r in lut if r["level"] == lv and r["target"] == "niveau")
        print(f"   {lv:10s} {a['codes_seen']:10d}  "
              f"{a['acc']:9.1f} % / {a['ceiling']:5.1f}  "
              f"{b['acc']:14.1f} % / {b['ceiling']:5.1f}")
    print("   (accord avec l'orbite / plafond équivariant ; table apprise sur le mode)")

    if not args.no_log:
        p = runlog.save(
            "bid_rule_ceiling", args.tag or "run",
            {"deals": args.deals, "seed": args.seed, "level": args.level,
             "prior": args.prior, "regime": regime, "model": args.bid_model},
            {"ceiling_exact": glob["exact"], "ceiling_binary": glob["binary"],
             "distinct_mean": distinct, "stable_hands_pct": never_flips,
             "n_codes": len(fam), "took_s": took,
             "families": fam, "coarse": coarse, "lookup_by_level": lut},
            payload={"rows": rows},
            models=[args.bid_model], took_s=took)
        print(f"\nJournalisé → {p}")


if __name__ == "__main__":
    main()
