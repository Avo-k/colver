#!/usr/bin/env python3
"""Cartographie de la marge de décision de v6 sur toute l'enchère.

Exploite une **asymétrie de coût** découverte le 2026-08-02 : mesurer la *marge* et le
*taux de bascule* coûte 0,8 s par régime (bid_equivariance, inférence pure), mesurer le
*coût* d'une bascule en coûte 13 à 25 min (bid_q_flatness, 108 000 déroulements). On
peut donc balayer des dizaines de préfixes pour presque rien, et ne dépenser le GPU que
là où la carte dit que ça vaut le coup.

§1.7 n'a sondé qu'un seul préfixe (100♣) et en a tiré trois régimes. Ce balayage teste
si ces trois régimes sont des points isolés ou des plateaux : la marge dépend-elle du
*niveau* de l'enchère ? de la *couleur* ? du fait que le partenaire ait parlé ?

Le siège qui décide est toujours `len(prior)` crans après le premier parleur, donc la
longueur du préfixe fixe qui a dit quoi (SEAT = 2 = Sud) :

    len 1 : [Est=adv]
    len 2 : [Nord=PARTENAIRE, Est=adv]
    len 3 : [Ouest=adv, Nord=PARTENAIRE, Est=adv]
    len 4 : [Sud=NOUS, Ouest=adv, Nord=PARTENAIRE, Est=adv]   (2e tour)

    uv run python scripts/analysis/bid_margin_sweep.py --deals 400
"""

import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

import bid_equivariance as be  # noqa: E402
import runlog  # noqa: E402

# (préfixe, description du régime). L'ordre est celui du tableau de sortie.
CASES = [
    ("",                "ouverture, 1er de parole"),
    ("P",               "ouverture, 2e (adv a passé)"),
    ("P P",             "ouverture, 3e (part. + adv ont passé)"),
    ("P P P",           "ouverture, 4e (dernier mot)"),

    ("100C",            "contestation — adv ouvre 100"),
    ("110C",            "contestation — adv ouvre 110"),
    ("130C",            "contestation — adv ouvre 130"),
    ("150C",            "contestation — adv ouvre 150"),
    ("100S",            "contestation — adv ouvre 100 (pique)"),
    ("100H",            "contestation — adv ouvre 100 (coeur)"),
    ("100D",            "contestation — adv ouvre 100 (carreau)"),
    ("P 100C",          "contestation — part. muet, adv ouvre 100"),

    ("100C P",          "soutien — part. ouvre 100"),
    ("130C P",          "soutien — part. ouvre 130"),
    ("P 100C P",        "soutien — part. ouvre 100 (3e tour de parole)"),
    ("100C 110D P",     "soutien — part. surenchérit 110 sur 100 adv"),

    ("100C 110D",       "contestation — part. a ouvert, adv monte à 110"),
    ("100C P 110D",     "contestation — deux adv actifs (100 puis 110)"),

    ("100C P P 110D",   "2e tour — notre 100 contré par 110 adv"),
    ("100C P 130C P",   "2e tour — part. nous relance à 130"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--bid-model", default=be.BID_MODEL)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--tag", default="sweep")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    env = colver.Env()
    env.reset()
    env.load_bid_model(args.bid_model, args.hidden)

    t0 = time.monotonic()
    rows = []
    for prefix, desc in CASES:
        prior = [be.parse_action(t) for t in prefix.split()]
        try:
            v6 = be.flip_rate(env, "v6", lambda e: e.action_bid_nn()["best_action"],
                              args.deals, args.seed, prior)
            ctrl = be.flip_rate(env, "ctrl", lambda e: e.bid_improved_v2(),
                                args.deals, args.seed, prior)
            scale = be.q_scale(env, args.deals, args.seed + 1, prior)
            equiv = be.q_equivariance(env, min(args.deals, 200), args.seed + 2,
                                      scale["spread_median"], scale["gap_median"], prior)
        except SystemExit as e:      # préfixe illégal sous une permutation : on saute
            print(f"  [SAUTÉ] {prefix!r} — {e}", file=sys.stderr)
            continue
        rows.append({
            "prefix": prefix, "desc": desc, "n_actions": len(prior),
            "flip_v6_pct": v6["pct"], "flip_ctrl_pct": ctrl["pct"],
            "gap_median": scale["gap_median"], "spread_median": scale["spread_median"],
            "under_003_pct": scale["under_003_pct"],
            "equiv_median": equiv["median"], "ratio_to_margin": equiv["ratio_to_margin"],
        })
        print(f"\r  {len(rows)}/{len(CASES)}  {prefix or '(ouverture)':<16s}",
              end="", file=sys.stderr)
    print(file=sys.stderr)

    print(f"\n{len(rows)} régimes × {args.deals} donnes × {len(be.PERMS)} permutations "
          f"en {time.monotonic() - t0:.0f}s\n")
    print(f"{'préfixe':>15s}  {'régime':<44s} {'bascules':>9s} {'ctrl':>6s} "
          f"{'marge':>8s} {'bruit/marge':>12s}")
    print("-" * 104)
    last = None
    for r in rows:
        if last is not None and r["n_actions"] != last:
            print("-" * 104)
        last = r["n_actions"]
        print(f"{r['prefix'] or '(vide)':>15s}  {r['desc']:<44s} "
              f"{r['flip_v6_pct']:8.2f}% {r['flip_ctrl_pct']:5.2f}% "
              f"{r['gap_median']:8.4f} {r['ratio_to_margin']:11.1f}×")

    print("\nLecture : `bascules` = % d'annonces qui changent sous renommage de couleurs.")
    print("`marge` = écart top1−top2 médian. `bruit/marge` = erreur d'équivariance ÷ marge :")
    print("au-dessus de 1, le bruit de symétrie franchit couramment la marge de décision.")
    print("`ctrl` (improved_v2) valide l'arithmétique de permutation, préfixe compris.")

    if not args.no_log:
        runlog.save("bid_margin_sweep", args.tag,
                    params={"deals": args.deals, "seed": args.seed,
                            "n_regimes": len(rows), "perms": len(be.PERMS)},
                    summary={"rows": rows},
                    payload=None, models=[args.bid_model],
                    took_s=time.monotonic() - t0)


if __name__ == "__main__":
    main()
