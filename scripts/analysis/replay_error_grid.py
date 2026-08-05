#!/usr/bin/env python3
"""La grille erreur / malchance / coup heureux, remplie sur de vraies donnes.

Rejouer classe un coup en croisant deux avis qui ne voient pas la même chose :

- **l'Oracle (DD)** voit les quatre mains. C'est un plafond, pas un joueur : un
  coup qu'il désapprouve peut avoir été le meilleur choix disponible.
- **Dédé (IS-DD)** ne voit que l'information du siège. Quand il approuve un coup
  que l'Oracle condamne, l'écart est de l'information, pas une erreur.

    coût DD > 0, coût IS-DD > 0  →  erreur (visible depuis ce siège)
    coût DD > 0, coût IS-DD = 0  →  malchance
    coût DD = 0, coût IS-DD > 0  →  coup heureux

**Ce que ce script décide** : si presque tous les écarts DD sont aussi des
écarts IS-DD, le filtre ne sert à rien et le compteur peut rester sur DD seul.
C'était la croyance n°1 de `docs/idees/rejouer_analyse_erreurs.md`.

Il passe par `analysis._analyze_sync` et `agent_review._Runner`, donc par le
code de production et non par une réimplémentation.

⚠️ **Demande le sidecar playgen** (`playgen-up`), et il faut le redescendre
après : il garde ~5 Go de VRAM tant qu'il tourne.

    uv run python scripts/analysis/replay_error_grid.py --deals 20
"""

import argparse
import os
import random
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402
import runlog  # noqa: E402
from replay_error_scale import _models, med, play_one_deal  # noqa: E402

# Le seuil sous lequel on considère que Dédé approuve. IS-DD est une moyenne sur
# des mondes échantillonnés : il ne rend presque jamais exactement 0, même quand
# il tient deux cartes pour équivalentes. Doit rester aligné sur `ISDD_NOISE`
# dans `replay.js`.
NOISE = 1.0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--deals", type=int, default=20)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--isdd-ms", type=int, default=None,
                   help="budget IS-DD par carte (défaut : celui d'agent_review)")
    p.add_argument("--tag", default=None)
    p.add_argument("--no-log", action="store_true")
    args = p.parse_args()

    if args.isdd_ms:
        os.environ["COLVER_REVIEW_ISDD_MS"] = str(args.isdd_ms)
    # Importé après la variable d'environnement : `ISDD_MS` est lu au chargement.
    from colver.web import agent_review, analysis

    bid_path, dmc_path = _models()
    belief_path = str(colver.belief_model_path() or "") or None
    rng = random.Random(args.seed)

    grid = {"erreur": 0, "malchance": 0, "aubaine": 0, "rien": 0}
    swing_split = {"imprecision": 0, "decisive": 0}
    dd_costs, isdd_costs = [], []
    unlucky_dd, error_dd = [], []
    no_isdd = 0

    with runlog.Timer() as t:
        for d in range(args.deals):
            dealer, hands, actions, _ = play_one_deal(bid_path, dmc_path, rng)
            game = {"dealer": dealer, "hands": hands,
                    "actions": [{"action": int(a)} for a in actions]}

            an = analysis._analyze_sync(game, bid_model_path=bid_path)
            by_idx = {m["idx"]: m for m in an["moves"]}

            runner = agent_review._Runner(
                game, play_model=dmc_path, belief_model=belief_path)
            runner.start()
            while runner.step() is not None:
                pass
            review = {m["idx"]: m for m in runner.finish()["moves"]}

            for idx, m in by_idx.items():
                if m.get("forced"):
                    continue
                r = review.get(idx) or {}
                isdd = r.get("isdd_cost")
                if isdd is None:
                    no_isdd += 1
                    continue
                dd = m["cost_score"]
                dd_costs.append(dd)
                isdd_costs.append(isdd)
                oracle = dd > 0
                dede = isdd > NOISE
                if oracle and dede:
                    grid["erreur"] += 1
                    error_dd.append(dd)
                    swing_split[m["category"]] = swing_split.get(m["category"], 0) + 1
                elif oracle:
                    grid["malchance"] += 1
                    unlucky_dd.append(dd)
                elif dede:
                    grid["aubaine"] += 1
                else:
                    grid["rien"] += 1
            print(f"\rdonne {d + 1}/{args.deals} — {sum(grid.values())} décisions",
                  end="", flush=True)

    n = sum(grid.values())
    if not n:
        print("\naucune décision évaluée — le sidecar répond-il ?")
        return
    print(f"\n\n{n} décisions sur {args.deals} donnes"
          + (f"  ({no_isdd} sans avis de Dédé)" if no_isdd else ""))
    print(f"budget IS-DD : {agent_review.ISDD_MS} ms/carte\n")

    blamed = grid["erreur"] + grid["malchance"]
    print(f"{'erreur (les deux désapprouvent)':<34} {grid['erreur']:>5}")
    print(f"{'malchance (Oracle seul)':<34} {grid['malchance']:>5}")
    print(f"{'coup heureux (Dédé seul)':<34} {grid['aubaine']:>5}")
    print(f"{'rien à signaler':<34} {grid['rien']:>5}")
    if blamed:
        share = 100 * grid["malchance"] / blamed
        print(f"\n**{share:.1f} % des écarts DD sont de la malchance** "
              f"({grid['malchance']}/{blamed})")
        print("   → le filtre change le compteur" if share >= 10
              else "   → le filtre ne change presque rien, DD seul suffirait")
    if grid["erreur"]:
        print(f"\nparmi les erreurs : {swing_split.get('imprecision', 0)} imprécisions, "
              f"{swing_split.get('decisive', 0)} fautes décisives")
    if unlucky_dd and error_dd:
        print(f"\ncoût DD médian — erreurs {med(error_dd)}, "
              f"malchance {med(unlucky_dd)}")

    if not args.no_log:
        runlog.save(
            "replay_error_grid", args.tag or "grille",
            {"deals": args.deals, "seed": args.seed,
             "isdd_ms": agent_review.ISDD_MS, "noise": NOISE},
            {"grid": grid, "swing_split": swing_split, "decisions": n,
             "no_isdd": no_isdd,
             "unlucky_share_of_blamed": (round(100 * grid["malchance"] / blamed, 1)
                                         if blamed else None),
             "median_dd_cost_error": med(error_dd) if error_dd else None,
             "median_dd_cost_unlucky": med(unlucky_dd) if unlucky_dd else None},
            payload={"dd_costs": dd_costs, "isdd_costs": isdd_costs},
            models=[bid_path, dmc_path, belief_path], took_s=t.s)


if __name__ == "__main__":
    main()
