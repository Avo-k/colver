#!/usr/bin/env python3
"""Combien de mondes une recherche IS-DD traverse-t-elle réellement, par pli ?

`determinizations` dans un TOML de bot ou dans `IsDdConfig` **n'est pas** le nombre de
mondes dès que `time_limit_ms` est posé : la boucle d'`is_dd.rs` ne sort alors que sur
l'échéance, et la branche du compte est inatteignable. Le budget est donc une échéance,
et le nombre de mondes est ce qui tient dedans — variable selon le pli, puisqu'un solve
coûte quatre ordres de grandeur de moins en fin de donne qu'à l'entame
(docs/play/dd_solver.md#performance).

Ce script mesure ce nombre. Deux usages :

  * calibrer le `k` réel des labels du pool (`enrich_pool_isdd` tourne à 20 ms/coup) et
    des bots d'arène de référence (50 ms) — le plan v7 §2.8 indexe toute son arithmétique
    de bruit sur « k = 20 » ;
  * garder une trace de ce que valent les optimisations du solveur *pour un appelant à
    échéance*, où elles ne rendent pas du temps mais des mondes.

Mondes uniformes (`source = "uniform"`), comme `enrich_pool_isdd` : pas de sidecar, donc
la mesure ne dépend pas de la charge GPU ni du réseau.

    uv run python scripts/analysis/isdd_worlds_per_budget.py --deals 20 --budgets 20,50
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


def spec(budget_ms: int, parallel: bool) -> str:
    return f"""
[bid]
strategy = "improved_v2"

[play]
method = "isdd"
time_ms = {budget_ms}
determinizations = 20
parallel = {str(parallel).lower()}

[worlds]
source = "uniform"
"""


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=20)
    ap.add_argument("--budgets", default="20,50", help="échéances en ms, séparées par des virgules")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--parallel",
        action="store_true",
        help="chemin arène/web (chunk = nb de threads). Sans le drapeau : chemin du pool "
             "(`IsDdConfig::default()`, chunk 1, échéance respectée au coup près).",
    )
    ap.add_argument("--tag", default="default")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    budgets = [int(b) for b in args.budgets.split(",")]
    rows = []

    for budget in budgets:
        cfg = spec(budget, args.parallel)
        # Un pli = 4 cartes. On indexe par pli pour rester comparable au tableau de
        # coûts du solveur, qui est indexé par cartes restantes.
        by_trick: dict[int, list[int]] = {t: [] for t in range(8)}
        rng = random.Random(args.seed)

        for _d in range(args.deals):
            env = colver.Env()
            env.reset()
            agents = [colver.Agent(cfg, seat, rng.randrange(1 << 30)) for seat in range(4)]
            for a in agents:
                a.init_deal(env)

            cards_played = 0
            while not env.is_terminal():
                seat = env.current_player()
                phase = env.phase()
                out = agents[seat].decide(env)
                if phase == 1:  # jeu
                    by_trick[cards_played // 4].append(out["determinizations"])
                    cards_played += 1
                for a in agents:
                    a.observe(env, out["action"])
                env.step(out["action"])

        for trick, counts in sorted(by_trick.items()):
            if not counts:
                continue
            rows.append(
                {
                    "budget_ms": budget,
                    "trick": trick + 1,
                    "n": len(counts),
                    "median": statistics.median(counts),
                    "mean": round(statistics.fmean(counts), 1),
                    "p10": sorted(counts)[len(counts) // 10],
                    "max": max(counts),
                }
            )

    width = max(len(str(r["budget_ms"])) for r in rows)
    print(f"\n{'budget':>{width + 3}} {'pli':>4} {'n':>6} {'médiane':>8} {'moyenne':>8} {'p10':>6} {'max':>6}")
    for r in rows:
        print(
            f"{r['budget_ms']:>{width + 1}}ms {r['trick']:>4} {r['n']:>6} "
            f"{r['median']:>8.0f} {r['mean']:>8.1f} {r['p10']:>6} {r['max']:>6}"
        )

    for budget in budgets:
        sub = [r for r in rows if r["budget_ms"] == budget]
        total = sum(r["mean"] * r["n"] for r in sub) / sum(r["n"] for r in sub)
        print(f"\n{budget} ms — mondes par coup, moyenné sur la donne : {total:.1f}")

    if not args.no_log:
        import runlog

        runlog.save(
            script="isdd_worlds_per_budget",
            tag=args.tag,
            params={"deals": args.deals, "budgets": budgets, "seed": args.seed,
                    "worlds": "uniform", "determinizations_nominal": 20,
                    "parallel": args.parallel},
            summary={
                f"{b}ms_mean_worlds_per_move": round(
                    sum(r["mean"] * r["n"] for r in rows if r["budget_ms"] == b)
                    / sum(r["n"] for r in rows if r["budget_ms"] == b),
                    1,
                )
                for b in budgets
            },
            payload={"rows": rows},
            models=[],
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
