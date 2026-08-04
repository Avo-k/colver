#!/usr/bin/env python3
"""Mesure A : l'enchère synthétique du générateur de couche est-elle dans la
distribution des enchères réelles ?

Le plan de [docs/data_gen/isdd_score_layer_v2.md](../../docs/data_gen/isdd_score_layer_v2.md)
§4 fabrique une enchère pour chaque case `(donne, atout)`, parce que playgen ne peut pas
échantillonner un monde sans préfixe d'enchère. Si cette enchère ne ressemble pas à
celles qu'il a vues à l'entraînement, il échantillonne pour une table qui n'existe pas —
sans erreur et sans signal.

**Deux corpus, et le second est le vrai juge.** `isdd_games_v1.bin` porte des enchères
de bid v6 avec IS-DD au jeu ; `playgen_games_9M.bin` est le corpus sur lequel playgen v2
a *appris*. « Dans la distribution » se dit par rapport au second. Les faire tourner tous
les deux dit au passage si le bidder se comporte pareil derrière deux joueurs de cartes
différents (IS-DD contre DouDou50).

Usage :
    uv run python scripts/analysis/taker_position.py --tag couche-v2
    uv run python scripts/analysis/taker_position.py --reuse <dir>   # journaliser sans recalculer

Le binaire doit être construit :
    cargo build -p colver-core --release --bin bench_taker_position
"""

import argparse
import json
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, "target/release/bench_taker_position")

CORPORA = {
    # nom → (chemin, limite)
    "isdd_v1": ("data/training/isdd_games_v1.bin", None),
    "playgen_9M": ("data/training/playgen_games_9M.bin", 43076),
}

POS = ["pos 0 (premier)", "pos 1", "pos 2", "pos 3 (donneur)"]
VALUES = ["80", "90", "100", "110", "120", "130", "140", "150", "160", "capot"]


def run(path, limit, out_json):
    cmd = [BIN, "--games", path, "--json", out_json]
    if limit:
        cmd += ["--limit", str(limit)]
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    with open(out_json, encoding="utf-8") as fh:
        return json.load(fh)


def digest(raw):
    """Ce qui part dans l'index versionné : les chiffres qui décident, pas les histogrammes."""
    c = raw["contracts"]
    nb = raw["real_nbids"]
    return {
        "contracts": c,
        "voids": raw["voids"],
        # 1. position du preneur
        "real_pos_pct": raw["real_pos_pct"],
        "cons_pos_at_real_pct": raw["cons_pos_at_real_pct"],
        "cons_pos_all_pct": raw["cons_pos_all_pct"],
        "tvd_pp": raw["tvd_pp"],
        # 2. accord apparié de la construction
        "side_agree_pct": raw["side_agree_pct"],
        "seat_agree_pct": raw["seat_agree_pct"],
        "real_seat_is_argmax_pct": raw["real_seat_is_argmax_pct"],
        # 3. forme du préfixe — la construction fait toujours k·P + 1 annonce + 3·P
        "mean_prefix_len": raw["mean_prefix_len"],
        "single_bid_pct": round(100 * nb[1] / c, 2),
        "contested_pct": raw["contested_pct"],
        "coinched_pct": raw["coinched_pct"],
        "first_bid_pos_pct": raw["real_first_bid_pos_pct"],
        # 4. valeur
        "value_pct": [round(100 * x / c, 2) for x in raw["real_value"]],
    }


def show(name, d):
    print(f"\n=== {name} — {d['contracts']:,} contrats, {d['voids']} donnes passées ===")
    print("  position du preneur dans l'ordre de parole :")
    print(f"    {'':<26}" + "".join(f"{p:>16}" for p in POS))
    for lbl, key in [("enchères réelles", "real_pos_pct"),
                     ("construction (atout réel)", "cons_pos_at_real_pct"),
                     ("construction (4 atouts)", "cons_pos_all_pct")]:
        print(f"    {lbl:<26}" + "".join(f"{v:>15.2f}%" for v in d[key]))
    print(f"    distance en variation totale : {d['tvd_pp']:.2f} pp")
    print(f"  accord apparié — camp {d['side_agree_pct']:.2f} %, "
          f"siège {d['seat_agree_pct']:.2f} % "
          f"(argmax dans le bon camp : {d['real_seat_is_argmax_pct']:.2f} %)")
    print(f"  forme du préfixe : {d['mean_prefix_len']:.2f} jetons en moyenne, "
          f"une seule annonce {d['single_bid_pct']:.2f} %, "
          f"contestée {d['contested_pct']:.2f} %, coinchée {d['coinched_pct']:.2f} %")
    print("    première annonce, position : "
          + "  ".join(f"{i}:{v:.1f}%" for i, v in enumerate(d["first_bid_pos_pct"])))
    print("  valeur : "
          + "  ".join(f"{lbl}:{v:.1f}%"
                      for lbl, v in zip(VALUES, d["value_pct"], strict=True) if v >= 0.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taker-position")
    ap.add_argument("--reuse", help="répertoire contenant <nom>.json déjà produits")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    raws = {}
    with runlog.Timer() as t:
        with tempfile.TemporaryDirectory() as tmp:
            for name, (path, limit) in CORPORA.items():
                if args.reuse:
                    with open(os.path.join(args.reuse, f"{name}.json"), encoding="utf-8") as fh:
                        raws[name] = json.load(fh)
                else:
                    raws[name] = run(path, limit, os.path.join(tmp, f"{name}.json"))

    digests = {k: digest(v) for k, v in raws.items()}
    for name, d in digests.items():
        show(name, d)

    if not args.no_log:
        runlog.save(
            script="taker_position",
            tag=args.tag,
            params={"corpora": {k: {"path": p, "limit": lim}
                                for k, (p, lim) in CORPORA.items()},
                    "reused": bool(args.reuse)},
            summary=digests,
            payload=raws,
            # Les corpus jouent ici le rôle des poids : ils portent la politique
            # d'enchère qui sert de référence, et ils ne sont pas dans git.
            models=[os.path.join(ROOT, p) for p, _ in CORPORA.values()],
            took_s=t.s,
        )


if __name__ == "__main__":
    main()
