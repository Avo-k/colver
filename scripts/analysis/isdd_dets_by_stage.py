#!/usr/bin/env python3
"""À partir de combien de mondes la réponse d'IS-DD cesse de bouger — par stade de donne ?

Le det-sweep du dépôt (2026-07-24) donne un plateau vers 240 mondes, mais il a été mesuré
en **force de jeu** (marge sur 80 matchs, IC ±10,7 pp) et avec un compte *uniforme sur
toute la donne*. Or on sait maintenant que Dédé en traverse de 256 à 697 selon le stade,
et qu'un solve coûte quatre ordres de grandeur de moins en finale qu'à l'entame
(docs/play/dd_solver.md#performance). Un plateau unique pour toute la donne est donc au
mieux une moyenne, au pire un contresens.

## Ce qui est mesuré

Pour chaque position de jeu, la carte choisie avec N mondes est comparée à celle choisie
avec un budget de **référence** bien plus grand, et on relève le **regret** : ce que la
référence pense qu'on perd en jouant la carte de N plutôt que la sienne, en points DD.

Le regret, pas le taux de désaccord : **57,8 % des positions ont plusieurs cartes
DD-optimales** (CLAUDE.md), donc « une autre carte » n'est très souvent pas « une moins
bonne carte ». Un taux de désaccord surestimerait massivement le besoin en mondes.

Regret nul = le petit budget choisit une carte que la référence juge optimale. Le stade
où le regret tombe à zéro *est* le plateau de ce stade.

## ⚠️ Ce que le premier run (2026-08-03, 250 positions) permet et ne permet pas de dire

Le message du commit `bdc6611` annonce « le plateau est vers 60, pas 240 ». **C'est plus
affirmatif que les données.** Avec les barres d'erreur, à l'entame :

    budget      15       30       60      120      240
    moyenne  0,800    0,091    0,066    0,115    0,096
    IC95 ±   0,859    0,100    0,072    0,135    0,177

**Tous les intervalles se recouvrent, y compris celui de 15 mondes**, et les moyennes ne
sont même pas monotones. À n = 33 par cellule, cette mesure ne distingue pas 15 de 240 ;
elle dit seulement que tous sont petits en valeur absolue.

Ce qui tient mieux, c'est la **queue** — la moyenne est dominée par de rares grosses
erreurs. Regret maximum à 8 cartes : 13,3 (15) → 1,4 (30) → 1,1 (60) → 1,9 (120) →
3,0 (240). Là il y a un vrai signal en dessous de 30, et plus aucune tendance au-dessus
de 60. C'est ça, et seulement ça, qui fonde « autour de 60 ».

Et il y a une **contradiction ouverte** avec le det-sweep de 2026-07-24, qui mesurait la
force de jeu et non l'accord : marge −101 (20) → +29 (60) → +72 (120) → +227 (240) →
+134 (512), soit 240 très au-dessus de 60. Aucune des deux mesures n'est solide (l'autre
a des IC de ±10,7 pp et son auteur note qu'aucune différence deux à deux n'est
significative ; celle-ci mesure l'accord avec une référence IS-DD, pas la force). **Ne
pas descendre `max_worlds` sur la seule foi de ce script** — il faut un h2h de force.

## Ce qui n'est pas mesuré

La force de jeu. Le regret est mesuré contre une référence IS-DD, pas contre la vérité :
si 3000 mondes se trompent, 240 se trompent pareil et le regret est nul. C'est la bonne
question pour « faut-il en demander plus », pas pour « IS-DD joue-t-il bien ».

## Protocole

Les positions viennent de donnes **réellement jouées** — enchère par le NN v6, cartes par
DouDou50 — et non de distributions au hasard : c'est la règle du dépôt, et une position
de milieu de donne issue d'un jeu aléatoire n'a pas la même structure de coupes.

Mode **compte** (`time_ms = 0`), donc N est bien le nombre de mondes et non une échéance.
Le garde-fou de refill n'agit que sous échéance : il est inactif ici, la mesure n'est pas
contaminée par lui.

Pas de section `[belief]` : on vient de mesurer qu'avec le sidecar la file playgen ne
s'assèche pas et que le belief net n'est jamais consulté. L'enlever ne change donc rien
au résultat et divise par deux la mémoire (un réseau de 2 Mo par agent, et il y a
4 sièges × (1 + len(budgets)) agents).

    playgen-up
    uv run python scripts/analysis/isdd_dets_by_stage.py --deals 20 --ref 3000
    playgen-down
"""

from __future__ import annotations

import argparse
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

import colver

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

BID_MODEL = "models/bid_v6_isdd.bin"
PLAY_MODEL = "models/dmc_50.bin"


def isdd_spec(dets: int, url: str) -> str:
    """Agent IS-DD en mode compte, mondes playgen — la source de la prod.

    `objective` est épinglé sur les points cartes, *pas* laissé au défaut
    (`deal_score` depuis le 2026-08-03). Ce script chiffre un regret « en points
    DD » — c'est lui qui a fixé le plafond `max_worlds` de la prod (sous 0,10 pt
    dès 60 mondes) — et sur l'échelle du score de donne les mêmes écarts sont
    d'un autre ordre de grandeur : un re-run se comparerait silencieusement aux
    chiffres publiés. Pour mesurer le plateau du bot *actuel*, changer la valeur
    ici et rejouer les deux bras.
    """
    return f"""
[bid]
strategy = "nn"
model = "{BID_MODEL}"
hidden = 512

[play]
method = "isdd"
time_ms = 0
determinizations = {dets}
objective = "card_points"
parallel = true

[worlds]
source = "sidecar"
url = "{url}"
batch = 256
fallback = "strict"
"""


def driver_spec() -> str:
    """Table qui joue réellement la donne : rapide et réaliste."""
    return f"""
[bid]
strategy = "nn"
model = "{BID_MODEL}"
hidden = 512

[play]
method = "dmc"
model = "{PLAY_MODEL}"
residual = true
"""


def scores_map(decision) -> dict[int, float]:
    """Scores DD par carte d'une décision IS-DD brute.

    La clé est `candidates`, pas `card_scores` : ce dernier n'existe qu'après le
    remodelage de `web/agents.py::decision_stats`. Se tromper est silencieux —
    on récolte un dictionnaire vide et zéro position mesurée, pas une erreur.
    """
    return {int(c): float(s) for c, s in decision.get("candidates", [])}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=20)
    ap.add_argument("--ref", type=int, default=3000, help="mondes du budget de référence")
    ap.add_argument("--budgets", default="30,60,120,240,480")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--min-cards", type=int, default=2,
                    help="ne pas évaluer en dessous de ce nombre de cartes restantes")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    url = os.environ.get("COLVER_PLAYGEN_GPU_URL", "").rstrip("/")
    if not url:
        print("COLVER_PLAYGEN_GPU_URL vide — ce sweep veut les mondes playgen de la prod",
              file=sys.stderr)
        return 1

    budgets = sorted(int(b) for b in args.budgets.split(","))
    if args.ref <= max(budgets) * 2:
        print(f"⚠ référence {args.ref} trop proche du plus gros budget {max(budgets)} : "
              f"le regret plancherait sur le bruit de la référence elle-même",
              file=sys.stderr)

    rng = random.Random(args.seed)
    # regret[(cards_left, budget)] -> liste de points DD perdus
    regret: dict[tuple[int, int], list[float]] = defaultdict(list)
    # combien de positions par stade
    seen: dict[int, int] = defaultdict(int)
    t0 = time.time()

    for deal_no in range(args.deals):
        env = colver.Env()
        env.reset()

        driver = [colver.Agent(driver_spec(), s, rng.randrange(1 << 30)) for s in range(4)]
        # IS-DD est lié au siège (il ne voit que sa main), donc une instance par siège et
        # par budget, toutes nourries de **toutes** les actions.
        probes = {
            d: [colver.Agent(isdd_spec(d, url), s, rng.randrange(1 << 30)) for s in range(4)]
            for d in [args.ref] + budgets
        }
        for a in driver:
            a.init_deal(env)
        for agents in probes.values():
            for a in agents:
                a.init_deal(env)

        while not env.is_terminal():
            seat = env.current_player()
            phase = env.phase()

            if phase == 1:
                cards_left = len(env.get_hands()[seat])
                # Une seule carte légale : aucune décision, IS-DD sort avant d'échantillonner.
                decisive = len(env.legal_actions()) >= 2
                if decisive and cards_left >= args.min_cards:
                    ref = probes[args.ref][seat].decide(env)
                    ref_scores = scores_map(ref)
                    if ref_scores:
                        # Le solveur rend des points N-S ; le siège qui joue maximise
                        # pour son camp.
                        sign = 1.0 if seat % 2 == 0 else -1.0
                        best = max(sign * v for v in ref_scores.values())
                        for d in budgets:
                            got = probes[d][seat].decide(env)
                            card = int(got["action"])
                            if card in ref_scores:
                                regret[(cards_left, d)].append(best - sign * ref_scores[card])
                        seen[cards_left] += 1

            out = driver[seat].decide(env)
            action = int(out["action"])
            for a in driver:
                a.observe(env, action)
            for agents in probes.values():
                for a in agents:
                    a.observe(env, action)
            env.step(action)

        done = deal_no + 1
        print(f"  donne {done}/{args.deals} — {time.time() - t0:.0f}s écoulées", flush=True)

    took = time.time() - t0

    stages = sorted({c for c, _ in regret}, reverse=True)
    print(f"\nRegret moyen en points DD contre une référence à {args.ref} mondes")
    print("(0 = le petit budget choisit une carte que la référence juge optimale)\n")
    header = f"{'cartes':>7} {'positions':>10}" + "".join(f"{b:>9}" for b in budgets)
    print(header)
    print("-" * len(header))
    for c in stages:
        row = f"{c:>7} {seen[c]:>10}"
        for b in budgets:
            vals = regret.get((c, b), [])
            row += f"{statistics.fmean(vals):>9.2f}" if vals else f"{'—':>9}"
        print(row)

    print(f"\n{'':>7} {'part de regret nul':>10}")
    for c in stages:
        row = f"{c:>7} {'':>10}"
        for b in budgets:
            vals = regret.get((c, b), [])
            zero = sum(1 for v in vals if v < 1e-6) / len(vals) if vals else 0.0
            row += f"{100 * zero:>8.0f}%" if vals else f"{'—':>9}"
        print(row)

    print(f"\n{args.deals} donnes, {sum(seen.values())} positions évaluées, {took:.0f}s")

    if not args.no_log:
        import runlog

        summary = {
            f"c{c}_b{b}_mean_regret": round(statistics.fmean(regret[(c, b)]), 3)
            for c in stages for b in budgets if regret.get((c, b))
        }
        summary["positions"] = sum(seen.values())
        runlog.save(
            script="isdd_dets_by_stage",
            tag=args.tag,
            params={"deals": args.deals, "ref": args.ref, "budgets": budgets,
                    "seed": args.seed, "worlds": "sidecar-playgen",
                    "min_cards": args.min_cards},
            summary=summary,
            payload={f"{c}|{b}": regret[(c, b)] for c in stages for b in budgets
                     if regret.get((c, b))},
            models=[BID_MODEL, PLAY_MODEL],
            took_s=took,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
