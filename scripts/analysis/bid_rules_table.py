#!/usr/bin/env python3
"""Écrit la politique d'annonce en familles de mains lisibles, avec sa fiabilité.

Point d'arrivée de [bid_rule_ceiling.py](bid_rule_ceiling.py) et
[bid_rules_by_family.py](bid_rules_by_family.py) : ceux-là mesurent *jusqu'où* une règle
peut aller et *où* elle bute ; celui-ci **écrit la règle**.

Une ligne = une famille `HandCode`. Pour chacune : ce que le bidder répond le plus
souvent, à quelle fréquence, et le plafond de la famille. Trois colonnes valent d'être
lues ensemble :

* **accord** — combien des 24 réponses du réseau la règle retrouve. C'est la fiabilité
  de la ligne, pas une moyenne globale.
* **plafond** — ce qu'aucune règle équivariante ne peut dépasser sur cette famille. Une
  ligne à 88 % d'accord sous un plafond de 89 % est *finie* ; la même sous un plafond de
  99 % est un chantier.
* **couverture cumulée** — où s'arrêter de lire. 28 familles couvrent 90 % des mains.

Ne recalcule rien : lit les payloads déjà écrits par `runlog` (les 24 réponses par main
y sont). C'est précisément ce pour quoi le journal existe — poser une question de plus
sur une mesure ne doit pas coûter la mesure.

    uv run python scripts/analysis/bid_rules_table.py --tag v6-opening
    uv run python scripts/analysis/bid_rules_table.py --tag v6-defense --level shape
    uv run python scripts/analysis/bid_rules_table.py --tag v6-opening --out docs/…/x.md
"""

import argparse
import glob
import json
import os
import sys
from collections import Counter, defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
STORE = os.path.join(ROOT, "data", "analysis", "bid_rule_ceiling")

SUITS = "♠♥♦♣"


def action_name(a):
    """Nom lisible d'une action d'enchère, **couleur comprise quand elle est nommée**."""
    if a == 0:
        return "passe"
    if a == 41:
        return "coinche"
    if a == 42:
        return "surcoinche"
    if 37 <= a <= 40:
        return f"capot {SUITS[a - 37]}"
    v, s = divmod(a - 1, 4)
    return f"{80 + 10 * v} {SUITS[s]}"


def value_name(a):
    """Idem, mais sans la couleur : c'est ce qu'une règle humaine peut promettre.

    La couleur d'une annonce se lit sur la main (« ma meilleure couleur ») ; c'est le
    *niveau* qui est la décision. Séparer les deux évite d'écrire une règle qui nomme
    pique alors que la famille est insensible aux couleurs par construction.
    """
    if a == 0:
        return "passe"
    if a >= 41:
        return "coinche"
    if 37 <= a <= 40:
        return "capot"
    return str(80 + 10 * ((a - 1) // 4))


def latest(tag):
    hits = sorted(glob.glob(os.path.join(STORE, f"*__{tag}.json")))
    if not hits:
        raise SystemExit(f"aucun run journalisé pour --tag {tag} dans {STORE}")
    return hits[-1]


def _rule_and_scores(rs, fn):
    """(réponse de la règle, accord, plafond, 2ᵉ réponse) pour une projection `fn`.

    La règle est le **mode des modes** : mode par main d'abord, puis mode sur les mains.
    Sans le premier passage, une famille se ferait décider par les quelques mains les
    plus instables — celles qui pèsent 24 réponses éparpillées au lieu d'une répétée.
    """
    tot = sum(len(r["answers"]) for r in rs)
    per_hand = [Counter(fn(a) for a in r["answers"]).most_common(1)[0][0] for r in rs]
    rule = Counter(per_hand).most_common(1)[0][0]
    agree = sum(1 for r in rs for a in r["answers"] if fn(a) == rule) / tot
    ceil = sum(Counter(fn(a) for a in r["answers"]).most_common(1)[0][1]
               for r in rs) / tot
    flat = Counter(fn(a) for r in rs for a in r["answers"])
    others = [k for k, _ in flat.most_common() if k != rule]
    return rule, 100 * agree, 100 * ceil, (others[0] if others else "—")


def build(rows, level):
    groups = defaultdict(list)
    for r in rows:
        groups[r["codes"][level]].append(r)
    out = []
    for code, rs in groups.items():
        lvl, l_agree, l_ceil, l_second = _rule_and_scores(rs, value_name)
        dec, d_agree, d_ceil, _ = _rule_and_scores(
            rs, lambda a: "passe" if a == 0 else "annonce")
        out.append({"code": code, "n": len(rs), "decision": dec,
                    "d_agree": d_agree, "d_ceiling": d_ceil,
                    "rule": lvl, "second": l_second,
                    "agree": l_agree, "ceiling": l_ceil})
    out.sort(key=lambda d: -d["n"])
    return out


def render(fams, total, meta, level, limit):
    lines = []
    a = meta["argv"]
    prior = a[a.index("--prior") + 1] if "--prior" in a else ""
    regime = {"": "ouverture (premier à parler)"}.get(prior, f"après « {prior} »")
    lines.append(f"### {regime} — niveau `{level}`\n")
    lines.append(f"*{total:,} mains × 24 permutations, modèle "
                 f"`{meta['models'][0]['path']}` (sha256 "
                 f"{meta['models'][0]['sha256']}), run {meta['when']}.*\n")
    lines.append("| # | famille | part | cum. | décision | accord | plaf. "
                 "| niveau | 2ᵉ | accord | plaf. |")
    lines.append("|--:|---|--:|--:|---|--:|--:|---|---|--:|--:|")
    cum = 0.0
    for i, f in enumerate(fams[:limit], 1):
        cum += 100 * f["n"] / total
        # Le séparateur `|` des codes « + 2ᵉ couleur » est aussi celui des colonnes d'un
        # tableau Markdown, et les backticks ne le protègent pas : sans l'échappement la
        # ligne se scinde et le tableau se décale à partir de la première de ces familles.
        code = f["code"].replace("|", "\\|")
        lines.append(
            f"| {i} | `{code}` | {100*f['n']/total:.1f} % | {cum:.0f} % | "
            f"**{f['decision']}** | {f['d_agree']:.0f} % | {f['d_ceiling']:.0f} % | "
            f"**{f['rule']}** | {f['second']} | {f['agree']:.0f} % | {f['ceiling']:.0f} % |")
    if len(fams) > limit:
        rest = sum(f["n"] for f in fams[limit:])
        lines.append(f"| | *{len(fams)-limit} familles de plus* | "
                     f"{100*rest/total:.1f} % | 100 % | | | | | | | |")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="v6-opening")
    ap.add_argument("--file", default="", help="chemin d'un payload précis")
    ap.add_argument("--level", default="trump", choices=[
        "length", "trump", "shape", "tops", "full", "trump+2e", "full+2e"])
    ap.add_argument("--limit", type=int, default=30)
    ap.add_argument("--refine", default="",
                    help="détailler une famille au niveau suivant, p.ex. 'T1.J.-/-/-'. "
                         "À utiliser sur les familles où l'accord est loin du plafond : "
                         "c'est là que le code jette ce qui décide.")
    ap.add_argument("--refine-level", default="tops")
    ap.add_argument("--weak", action="store_true",
                    help="lister les familles où l'accord est le plus loin du plafond "
                         "— celles pour lesquelles le niveau de code est trop grossier")
    ap.add_argument("--out", default="", help="fichier Markdown ; sinon stdout")
    args = ap.parse_args()

    path = args.file or latest(args.tag)
    print(f"lecture : {os.path.relpath(path, ROOT)}", file=sys.stderr)
    d = json.load(open(path))
    rows = d["payload"]["rows"]
    if args.level not in rows[0]["codes"]:
        raise SystemExit(f"ce payload ne porte pas le niveau '{args.level}' "
                         f"(disponibles : {', '.join(rows[0]['codes'])})")
    if args.refine:
        rows = [r for r in rows if r["codes"][args.level] == args.refine]
        if not rows:
            raise SystemExit(f"aucune main dans la famille {args.refine}")
        fams = build(rows, args.refine_level)
        md = (f"### `{args.refine}` détaillée au niveau `{args.refine_level}`\n\n"
              f"*{len(rows):,} mains de cette famille.*\n\n"
              + "\n".join(render(fams, len(rows), d, args.refine_level,
                                 args.limit).split("\n")[3:]))
        print(md)
        return

    fams = build(rows, args.level)
    if args.weak:
        weak = sorted(fams, key=lambda f: -(f["d_ceiling"] - f["d_agree"]))
        print("Familles où le code est trop grossier (accord loin du plafond, "
              "décision annoncer/passer)\n")
        print(f"   {'famille':22s} {'part':>6s} {'accord':>7s} {'plafond':>8s} {'manque':>7s}")
        for f in weak[:15]:
            if f["n"] < len(rows) / 1000:
                continue
            print(f"   {f['code']:22s} {100*f['n']/len(rows):5.1f}% "
                  f"{f['d_agree']:6.0f}% {f['d_ceiling']:7.0f}% "
                  f"{f['d_ceiling']-f['d_agree']:6.0f} pt")
        print()
    md = render(fams, len(rows), d, args.level, args.limit)
    if args.out:
        with open(args.out, "a" if os.path.exists(args.out) else "w") as fh:
            fh.write("\n" + md)
        print(f"écrit → {args.out}", file=sys.stderr)
    else:
        print(md)


if __name__ == "__main__":
    main()
