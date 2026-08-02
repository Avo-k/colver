#!/usr/bin/env python3
"""Le déficit de deal-EV de v6 contre v5 dépend-il du score de partie ? (§2.6 du plan v7)

v6 gagne 55-57 % des matchs contre v5 mais perd 16 à 26 pts/donne à chaque sonde de
score. Le dossier appelle ça le paradoxe donne/match et l'attribue à la reward
Δ-winprob, sans l'avoir testé. Cette mesure le teste, et la prédiction est nette :

    **À 0-0, la sigmoïde de win_probability est localement linéaire**, donc Δ-winprob y
    est proportionnel à Δ-points et les deux objectifs **coïncident**. Si v6 sacrifie de
    l'EV pour gérer la variance, son déficit doit donc **s'annuler à 0-0** et grandir
    avec l'asymétrie du score.

    - déficit ~0 à 0-0, croissant avec l'asymétrie → gestion de variance confirmée, le
      paradoxe est l'objectif qui fonctionne, v7 garde la reward ;
    - déficit qui **persiste à 0-0** → v6 est réellement plus faible sur l'annonce et
      son avantage en matchs vient d'ailleurs. C'est le design de v7 qui change.

Méthode. Les deux specs ne diffèrent que par le modèle d'annonce (même DouDou50 au jeu,
cf. arena/bots/v6_isdd_75M.toml et v5_isdd_25M.toml), donc l'écart mesuré est bien celui
du bidder. Chaque donne est jouée **deux fois, v6 dans un camp puis dans l'autre**, à
score miroir pour que *la situation de v6 reste la même* — c'est un appariement, il
annule l'asymétrie de siège et de donneur, et il divise la variance.

    uv run python scripts/analysis/bid_ev_by_score.py --deals 800
"""

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

import runlog  # noqa: E402

V6 = "arena/bots/v6_isdd_75M.toml"
V5 = "arena/bots/v5_isdd_25M.toml"

# (score de v6, score de l'adversaire). 0-0 est la sonde qui décide.
SCORES = [
    (0, 0),
    (900, 200),
    (200, 900),
    (1500, 1000),
    (1000, 1500),
    (1700, 1700),
]


def stderr_of(xs):
    return statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else float("nan")


def play(env, dealer, hands, seats, ns_score, ew_score):
    """Joue une donne complète. `seats[i]` est l'agent du siège i."""
    env.redeal_with_hands(dealer, hands)
    for ag in seats:
        ag.set_scores(ns_score, ew_score)
        ag.init_deal(env)
    guard = 0
    while not env.is_terminal() and guard < 200:
        a = int(seats[int(env.current_player())].action(env))
        for ag in seats:            # `observe` attend l'env *avant* le coup
            ag.observe(env, a)
        env.step(a)
        guard += 1
    r = env.rewards()
    return float(r[0]), float(r[1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=800)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    env = colver.Env()
    env.reset()
    a6 = [colver.Agent.from_file(V6, s, args.seed) for s in range(4)]
    a5 = [colver.Agent.from_file(V5, s, args.seed) for s in range(4)]
    # v6 en Nord-Sud (sièges 0 et 2) / v6 en Est-Ouest (sièges 1 et 3)
    v6_ns = [a6[0], a5[1], a6[2], a5[3]]
    v6_ew = [a5[0], a6[1], a5[2], a6[3]]

    print(f"v6 = {a6[0].label}\nv5 = {a5[0].label}")
    print(f"{args.deals} donnes par sonde, chacune jouée dans les deux sens (appariée)\n")

    t0 = time.monotonic()
    rows = []
    for s6, sopp in SCORES:
        deltas = []
        for _ in range(args.deals):
            deck = list(range(32))
            rng.shuffle(deck)
            hands = [sorted(deck[i * 8:(i + 1) * 8]) for i in range(4)]
            dealer = rng.randrange(4)
            # sens 1 : v6 en NS, donc NS porte le score de v6
            ns1, ew1 = play(env, dealer, hands, v6_ns, s6, sopp)
            # sens 2 : v6 en EW, score miroir pour que la *situation de v6* soit la même
            ns2, ew2 = play(env, dealer, hands, v6_ew, sopp, s6)
            deltas.append(((ns1 - ew1) + (ew2 - ns2)) / 2)
        m, se = statistics.fmean(deltas), stderr_of(deltas)
        rows.append({"v6_score": s6, "opp_score": sopp, "delta": m, "se": se,
                     "n": len(deltas), "sigma": statistics.stdev(deltas)})
        tag = "0-0 ← LA SONDE QUI DÉCIDE" if (s6, sopp) == (0, 0) else ""
        print(f"  v6 {s6:>4d} - {sopp:<4d}   Δ deal-EV = {m:+7.2f} ± {se:5.2f}"
              f"   ({m/se:+5.1f} σ)  {tag}")
        print(f"\r    ({time.monotonic() - t0:.0f}s)", end="", file=sys.stderr)
    print(file=sys.stderr)

    z0 = rows[0]["delta"] / rows[0]["se"]
    print("\n--- lecture ---")
    print(f"À 0-0 : {rows[0]['delta']:+.2f} ± {rows[0]['se']:.2f} pts/donne ({z0:+.1f} σ)")
    n_neg = sum(1 for r in rows if r["delta"] < 0)
    if z0 > 2 and n_neg == 0:
        print("  → v6 est MEILLEUR en deal-EV, à tous les scores. Il n'y a donc aucun")
        print("    paradoxe à expliquer : son avantage en matchs est simplement celui de")
        print("    son avantage par donne. Contrôle à faire — cet écart doit prédire le")
        print("    taux de matchs observé, via la courbe de arena_power (§1.8).")
    elif abs(z0) < 2:
        print("  → compatible avec ZÉRO : les deux objectifs coïncident là où la théorie")
        print("    le prédit. L'écart global viendrait alors de l'asymétrie de score,")
        print("    c'est-à-dire de la gestion de variance.")
    else:
        print("  → écart significatif à 0-0, or Δ-winprob y est ∝ Δ-points. La gestion de")
        print("    variance ne peut donc pas l'expliquer ; chercher ailleurs.")

    if not args.no_log:
        runlog.save("bid_ev_by_score", args.tag or "v6_vs_v5",
                    params={"deals": args.deals, "seed": args.seed,
                            "v6_spec": V6, "v5_spec": V5, "scores": SCORES},
                    summary={"rows": rows, "z_at_0_0": z0},
                    payload=None,
                    models=["models/bid_v6_isdd_resume/bid_nn_final.bin",
                            "models/bid_v5_isdd/bid_nn_final.bin",
                            "models/play_v2/play_final.bin"],
                    took_s=time.monotonic() - t0)


if __name__ == "__main__":
    main()
