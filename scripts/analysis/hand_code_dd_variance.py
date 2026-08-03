#!/usr/bin/env python3
"""À quel point un code de main prédit-il la valeur DD ? Et quel est son plafond ?

Ferme la moitié restante du §6 de
[hand_classification.md](../../docs/bid/interpretability/hand_classification.md) : « le
niveau de code à choisir n'est pas tranché ». Le critère y est posé — celui du poker :
un bucket vaut ce qu'il prédit de l'issue — mais jamais mesuré contre le DD.
[bid_rule_ceiling.py](bid_rule_ceiling.py) a répondu contre la *politique de v6* ; ici
c'est contre la **valeur**, qui n'est pas la même cible : imiter un bidder imparfait et
prédire ce que la main vaut sont deux questions.

## Le plafond, encore, et il est indispensable ici

Une main ne détermine pas sa valeur DD : les 24 autres cartes en décident pour une large
part. Aucun code ne peut donc expliquer 100 % de la variance — pas même le code exact.
Le plafond est la part de variance qu'explique **la main elle-même**, et il s'estime en
résolvant chaque main sous plusieurs répartitions des 24 autres cartes.

D'où le plan : `n` mains × `m` répartitions, une ANOVA à un facteur, et l'ICC comme
mesure de variance expliquée. Sans les `m` répétitions on ne pourrait pas séparer « le
code est grossier » de « la donne est bruitée », et on lirait un R² faible comme un
défaut du code alors que c'est la nature du jeu.

⚠️ **L'ICC, pas le R² brut.** Un R² empirique par groupe monte mécaniquement avec le
nombre de groupes : `full` a 6 654 codes, il « expliquerait » plus que `trump` même sur
du bruit pur. L'estimateur ANOVA corrige ce biais (il retranche la variance intra), donc
un niveau plus fin peut, à raison, ressortir *moins* bon.

## Deux cibles

* **atout ancré** — la valeur DD dans la couleur qu'on envisagerait. C'est la cible que
  `hand_code(main, atout)` décrit littéralement.
* **deuxième couleur** — la valeur DD dans la couleur qu'on envisagerait *en second*.
  C'est la coordonnée que `hand_code` ne décrit presque pas : il y réduit la couleur à
  « As / Dix / longueur » et jette son Valet et son 9. La comparer à la première chiffre
  l'angle mort **en points DD** plutôt qu'en accord de politique.
* **meilleur atout** — le max sur les quatre couleurs. C'est ce qui décide d'annoncer,
  et c'est là que le code perd la qualité d'atout des *autres* couleurs.

Référence (2026-08-03, 720 000 solves, ~31 min sur 8 threads) :
`docs/measurements/index.jsonl`, tag `dd-variance`. La main explique **23,5 %** de la
variance de sa valeur DD à l'atout ancré ; `tops` sature ce plafond, `full` n'ajoute
rien — ce qui est cohérent avec sa conception, `full` ne rajoutant que la belote, qui
vaut zéro point carte.

    uv run python scripts/analysis/hand_code_dd_variance.py --hands 60000 --reps 3 \\
        --tag dd-variance
"""

import argparse
import os
import random
import sys
import threading
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

import runlog  # noqa: E402
from bid_rule_ceiling import LEVELS, suit_key  # noqa: E402

_local = threading.local()


def env():
    """Un `Env` par thread : le solveur relâche le GIL, l'objet n'est pas partageable."""
    if not hasattr(_local, "env"):
        _local.env = colver.Env()
        _local.env.reset()
    return _local.env


def one_hand(args):
    """Résout `reps` répartitions des 24 autres cartes pour une main donnée.

    Le donneur est fixé à 3 pour que le siège 0 (Nord, camp N-S) parle le premier : la
    main étudiée est donc toujours celle dont on lit les points, et `evaluate_hand(0)`
    voit bien cette main-là.
    """
    hand, seed, reps = args
    rng = random.Random(seed)
    rest = [c for c in range(32) if c not in hand]
    e = env()
    out = []
    for _ in range(reps):
        rng.shuffle(rest)
        others = [sorted(rest[i * 8:(i + 1) * 8]) for i in range(3)]
        e.redeal_with_hands(3, [sorted(hand), others[0], others[1], others[2]])
        suits = e.solve_all_suits()["suits"]
        out.append([s[0] for s in suits])          # points N-S par couleur d'atout
    scores = e.evaluate_hand(0)["scores"]
    # Ordre des couleurs par qualité d'atout décroissante, départagé par `suit_key` —
    # un ordre total stable au renommage, donc la coordonnée `k` du vecteur veut dire la
    # même chose d'une main à l'autre. Sans ça les quatre valeurs sont indexées par
    # couleur physique et ne sont comparables entre mains d'aucune façon.
    h = sorted(hand)
    order = sorted(range(4), key=lambda s: (-scores[s], -suit_key(h, s)))
    return hand, order, out


def icc(groups):
    """Part de variance expliquée par le facteur, estimée par ANOVA à un facteur.

    `groups` : liste de listes d'observations. Retourne l'ICC, borné à [0, 1] — négatif
    signifierait « moins que le hasard », qu'on lit comme zéro.
    """
    groups = [g for g in groups if g]
    k = len(groups)
    n = [len(g) for g in groups]
    total = sum(n)
    if k < 2 or total <= k:
        return float("nan")
    grand = sum(sum(g) for g in groups) / total
    means = [sum(g) / len(g) for g in groups]
    ss_b = sum(ni * (m - grand) ** 2 for ni, m in zip(n, means, strict=True))
    ss_w = sum((x - m) ** 2 for g, m in zip(groups, means, strict=True) for x in g)
    ms_b = ss_b / (k - 1)
    ms_w = ss_w / (total - k)
    # n₀ : taille de groupe « effective » pour des groupes de tailles inégales.
    n0 = (total - sum(ni * ni for ni in n) / total) / (k - 1)
    var_b = max(0.0, (ms_b - ms_w) / n0)
    return var_b / (var_b + ms_w) if var_b + ms_w > 0 else float("nan")


def code_diagnostics(rows, level, top=10):
    """Les deux sens du contrôle : le code sépare-t-il ce que la valeur sépare ?

    * **Fusions manquées** — deux codes dont les vecteurs DD moyens sont à moins d'un
      point l'un de l'autre : le code distingue des mains que la valeur ne distingue pas.
      C'est du raffinement gratuit qu'on paie en nombre de familles.
    * **Séparations manquées** — un code dont les mains ont des vecteurs très dispersés :
      il fusionne des mains que la valeur sépare. C'est là qu'une composante manque.

    La dispersion intra-code se lit **sur les moyennes par main**, pas sur les
    observations : sinon elle mesurerait surtout le bruit de la donne, qui vaut trois
    fois la variance de la main et écraserait tout.
    """
    g = defaultdict(list)
    for r in rows:
        m = [sum(c) / len(c) for c in zip(*r["vec"], strict=True)]
        g[r["codes"][level]].append(m)
    stats = {}
    for c, ms in g.items():
        if len(ms) < 30:
            continue
        mean = [sum(x) / len(ms) for x in zip(*ms, strict=True)]
        sd = [(sum((v - mu) ** 2 for v in x) / (len(ms) - 1)) ** 0.5
              for x, mu in zip(zip(*ms, strict=True), mean, strict=True)]
        stats[c] = {"n": len(ms), "mean": mean, "sd": sd}

    print(f"\nDiagnostic du code au niveau '{level}' ({len(stats)} codes ≥ 30 mains)")
    pairs = []
    keys = list(stats)
    for i, a in enumerate(keys):
        for b in keys[i + 1:]:
            d = max(abs(x - y) for x, y in
                    zip(stats[a]["mean"], stats[b]["mean"], strict=True))
            pairs.append((d, a, b))
    pairs.sort()
    print("   Fusions manquées — vecteurs DD moyens les plus proches (écart max, pts) :")
    for d, a, b in pairs[:top]:
        print(f"      {d:5.1f}   {a:22s} ≈ {b}")
    print("   Séparations manquées — codes dont les mains sont les plus dispersées "
          "(σ de la 1re coordonnée) :")
    for c, s in sorted(stats.items(), key=lambda kv: -kv[1]["sd"][0])[:top]:
        print(f"      {s['sd'][0]:5.1f}   {c:22s} (n={s['n']}, "
              f"moyenne {s['mean'][0]:.0f} / {s['mean'][1]:.0f} pts)")
    return {"level": level, "closest_pairs": [(d, a, b) for d, a, b in pairs[:top]],
            "most_spread": sorted(
                ({"code": c, **s} for c, s in stats.items()),
                key=lambda s: -s["sd"][0])[:top]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=8000)
    ap.add_argument("--reps", type=int, default=2,
                    help="répartitions des 24 autres cartes par main ; ≥2 est requis "
                         "pour estimer le plafond")
    ap.add_argument("--threads", type=int, default=8)
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--from-run", default="",
                    help="ré-agréger un payload déjà journalisé au lieu de re-résoudre")
    ap.add_argument("--diag-level", default="trump",
                    help="niveau du diagnostic de code (fusions/séparations manquées)")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()
    if args.from_run:
        import json
        rows = json.load(open(args.from_run))["payload"]["rows"]
        if "vec" not in rows[0]:
            raise SystemExit(
                "ce payload est antérieur au vecteur DD réordonné (tag `dd-variance`, "
                "2026-08-03T00:58) : il ne porte que les valeurs à l'atout ancré et au "
                "meilleur atout, donc ni la cible « second » ni le diagnostic de code "
                "ne s'en déduisent. Relancer le calcul.")
        print(f"{len(rows):,} mains relues de {args.from_run}")
        report(rows)
        code_diagnostics(rows, args.diag_level)
        return
    if args.reps < 2:
        raise SystemExit("--reps doit valoir au moins 2 : sans répétition, la part de "
                         "variance due à la main est inséparable de celle du code")

    rng = random.Random(args.seed)
    jobs = []
    for i in range(args.hands):
        deck = list(range(32))
        rng.shuffle(deck)
        jobs.append((tuple(sorted(deck[:8])), args.seed * 1_000_003 + i, args.reps))

    t0 = time.monotonic()
    with ThreadPoolExecutor(max_workers=args.threads) as pool:
        results = list(pool.map(one_hand, jobs))
    took = time.monotonic() - t0
    print(f"{args.hands:,} mains × {args.reps} répartitions × 4 atouts "
          f"= {args.hands * args.reps * 4:,} solves   ({took:.0f} s, "
          f"{args.threads} threads)")

    rows = []
    for hand, order, per_rep in results:
        anchor = order[0]
        codes = {lv: colver.hand_code(list(hand), anchor, lv) for lv in LEVELS}
        rows.append({
            "hand": list(hand), "order": order, "codes": codes,
            # Vecteur DD réordonné : coordonnée 0 = la couleur qu'on envisagerait,
            # 1 = la deuxième, etc. C'est la forme qui rend les mains comparables.
            "vec": [[r[s] for s in order] for r in per_rep],
            "anchored": [r[order[0]] for r in per_rep],
            "second": [r[order[1]] for r in per_rep],
            "best": [max(r) for r in per_rep],
        })

    summary = report(rows)

    code_diagnostics(rows, args.diag_level)

    if not args.no_log:
        p = runlog.save(
            "hand_code_dd_variance", args.tag or "run",
            {"hands": args.hands, "reps": args.reps, "seed": args.seed},
            {"took_s": took, **summary},
            payload={"rows": rows}, took_s=took)
        print(f"\nJournalisé → {p}")


def report(rows):
    summary = {}
    for target in ("anchored", "second", "best"):
        vals = [x for r in rows for x in r[target]]
        mean = sum(vals) / len(vals)
        sd = (sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)) ** 0.5
        ceiling = icc([r[target] for r in rows])   # une main = un groupe
        print(f"\nCible « {target} » — moyenne {mean:.1f} pts, écart-type {sd:.1f}")
        print(f"   {'niveau':10s} {'codes':>7s} {'variance expliquée':>20s}")
        print(f"   {'la main':10s} {len(rows):7d} {100*ceiling:19.1f} %   ← plafond")
        lvls = {}
        for lv in LEVELS:
            g = defaultdict(list)
            for r in rows:
                g[r["codes"][lv]].extend(r[target])
            v = icc(list(g.values()))
            lvls[lv] = {"codes": len(g), "explained": 100 * v}
            print(f"   {lv:10s} {len(g):7d} {100*v:19.1f} %"
                  f"   ({100*v/(100*ceiling):.0%} du plafond)")
        summary[target] = {"mean": mean, "sd": sd, "ceiling": 100 * ceiling,
                           "levels": lvls}
    return summary


if __name__ == "__main__":
    main()
