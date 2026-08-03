#!/usr/bin/env python3
"""Quelle est la dispersion de la marge d'une donne — l'échelle que R3 doit poser ?

`elo.py` note aujourd'hui une donne en binaire (`1 / 0`), ce qui jette la marge. Or c'est
dans la marge que vit l'écart entre deux joueurs : mesuré le 2026-08-03, Dédé bat DouDou
55,4 % des donnes (soit +37 Elo) mais 72 % des **matchs** (+164 Elo), parce qu'il gagne
+60 points de marge par donne et que ça se cumule sur une dizaine de donnes.

Passer à `s = σ(marge / échelle)` demande de fixer `échelle`, et ce n'est pas un réglage
libre : c'est lui qui définit ce que vaut « 1 Elo ». Trop petit, tout sature à 0/1 et on
revient au binaire ; trop grand, tout vaut 0,5 et plus rien ne bouge.

Le choix naturel est l'**écart-type de la marge**, qui rend `s` comparable d'un jeu à
l'autre : `σ(m/sd)` place une donne moyenne à ~0,5 et une donne exceptionnelle aux bords.

## Ce que ce script mesure, et ce qu'il ne mesure pas

Il joue des donnes complètes avec une table réaliste (enchère par le NN v6, cartes par
DouDou50) et relève `env.rewards()`, c'est-à-dire les **points marqués** — pas les points
cartes. La distribution n'est pas gaussienne : le barème a une marche au seuil du contrat,
donc les marges sont bimodales (contrat réussi contre chute). L'écart-type reste la bonne
échelle de travail, mais les quantiles disent mieux à quoi ressemble la loi.

⚠️ Mesuré **bot contre bot**. Les donnes de la prod sont humain-contre-bots, donc leur
dispersion peut différer — surtout si les humains chutent plus souvent. À recouper sur la
base quand on pourra la lire.

    uv run python scripts/analysis/deal_margin_scale.py --deals 3000
"""

from __future__ import annotations

import argparse
import random
import statistics
import sys
from pathlib import Path

import colver

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

BID_MODEL = "models/bid_v6_isdd.bin"
PLAY_MODEL = "models/dmc_50.bin"

SPEC = f"""
[bid]
strategy = "nn"
model = "{BID_MODEL}"
hidden = 512

[play]
method = "dmc"
model = "{PLAY_MODEL}"
residual = true
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=3000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    margins: list[float] = []
    void = 0

    agents = [colver.Agent(SPEC, s, rng.randrange(1 << 30)) for s in range(4)]
    for _ in range(args.deals):
        env = colver.Env()
        env.reset()
        for a in agents:
            a.init_deal(env)
        while not env.is_terminal():
            seat = env.current_player()
            out = agents[seat].decide(env)
            action = int(out["action"])
            for a in agents:
                a.observe(env, action)
            env.step(action)
        if not env.get_contract().get("value"):
            void += 1
            continue  # donne passée : elle ne marque rien, elle n'est pas notée
        rw = env.rewards()
        margins.append(float(rw[0]) - float(rw[1]))

    n = len(margins)
    absm = [abs(m) for m in margins]
    sd = statistics.pstdev(margins)
    q = statistics.quantiles(absm, n=20)

    print(f"\n{n} donnes notables ({void} passées, écartées)\n")
    print(f"  écart-type de la marge   : {sd:8.1f}")
    print(f"  moyenne de |marge|       : {statistics.fmean(absm):8.1f}")
    print(f"  médiane de |marge|       : {statistics.median(absm):8.1f}")
    print(f"  p75 / p90 / p95 de |marge| : {q[14]:.0f} / {q[17]:.0f} / {q[18]:.0f}")
    print(f"  min / max                : {min(margins):.0f} / {max(margins):.0f}")

    # Ce que l'échelle donne concrètement : à quoi ressemble `s` pour une marge donnée.
    print("\n  s = 1/(1+10^(-marge/échelle)), échelle = écart-type")
    for m in (10, 50, 100, 200, 400, 800):
        s = 1.0 / (1.0 + 10 ** (-m / sd))
        print(f"    marge {m:>4} → s = {s:.3f}")

    # Part des donnes que le binaire et la marge classent « pareil » : si la marge
    # médiane est déjà énorme, l'écrasement rend la note quasi binaire et R3 n'apporte
    # rien. C'est le contrôle qui dit si le changement vaut la peine.
    near = sum(1 for m in absm if m < sd / 2)
    print(f"\n  donnes à |marge| < écart-type/2 : {100*near/n:.0f} % "
          f"(celles où la marge change vraiment la note)")

    if not args.no_log:
        import runlog

        runlog.save(
            script="deal_margin_scale",
            tag=args.tag,
            params={"deals": args.deals, "seed": args.seed, "table": "bid_v6 + DouDou50"},
            summary={"n": n, "void": void, "sd": round(sd, 1),
                     "mean_abs": round(statistics.fmean(absm), 1),
                     "median_abs": round(statistics.median(absm), 1)},
            payload={"margins": margins},
            models=[BID_MODEL, PLAY_MODEL],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
