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

# Le modèle qui a produit `isdd_games_v1.bin`. Il sert à deux choses : dérouler les
# variantes d'enchère candidates, et **se rejouer lui-même sans masque** sur les donnes
# du corpus — un témoin qui doit rendre les enchères à l'identique. Sans ce témoin, un
# défaut du pilote (historique mal suivi, mauvais score, pénalité oubliée) se lirait
# comme une propriété de la variante.
BID_MODEL = "models/bid_v6_isdd_resume/bid_nn_final.bin"

CORPORA = {
    # nom → (chemin, limite, variantes d'enchère)
    "isdd_v1": ("data/training/isdd_games_v1.bin", None, True),
    "playgen_9M": ("data/training/playgen_games_9M.bin", 43076, False),
}

POS = ["pos 0 (premier)", "pos 1", "pos 2", "pos 3 (donneur)"]
VALUES = ["80", "90", "100", "110", "120", "130", "140", "150", "160", "capot"]


def run(path, limit, variants, out_json):
    cmd = [BIN, "--games", path, "--json", out_json]
    if limit:
        cmd += ["--limit", str(limit)]
    if variants:
        cmd += ["--bid-model", BID_MODEL]
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    # Surtout pas de `head` ni de pipe tronquant ici : le SIGPIPE tuerait le processus
    # avant l'écriture du JSON, et le run aurait l'air d'avoir réussi.
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
        # 5. les variantes candidates, aux mêmes cibles (absentes sans --bid-model)
        "free_v6": raw.get("free_v6"),
        "masked_v6": raw.get("masked_v6"),
        "peel": raw.get("peel"),
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
    if not d.get("free_v6"):
        return
    u, m = d["free_v6"], d["masked_v6"]
    print(f"  TÉMOIN — v6 rejoué sans masque reproduit le corpus : "
          f"{u['exact_match_pct']:.2f} % d'enchères identiques")
    print(f"    {'':<42}{'corpus':>9}{'v6 libre':>11}{'v6 masqué':>11}")
    for lbl, key, ref in [("1re annonce par le premier parleur", "first_bid_pos0_pct",
                           d["first_bid_pos_pct"][0]),
                          ("une seule annonce", "single_bid_pct", d["single_bid_pct"]),
                          ("contestée", "contested_pct", d["contested_pct"]),
                          ("coinchée", "coinched_pct", d["coinched_pct"]),
                          ("longueur du préfixe", "mean_prefix_len", d["mean_prefix_len"])]:
        print(f"    {lbl:<42}{ref:>9.2f}{u[key]:>11.2f}{m[key]:>11.2f}")
    print(f"    {'cases sans aucune enchère':<42}{'—':>9}{'—':>11}{m['void_pct']:>10.2f}%")
    print("  rang de l'atout que v6 choisit librement, vu de son camp : "
          + "  ".join(f"{i}:{p:.1f}%" for i, p in enumerate(u["rank_pct"])))
    q = d.get("peel")
    if not q:
        return
    print(f"  ÉPLUCHAGE — une vraie enchère nomme {q['mean_real_suits']:.2f} couleurs "
          f"distinctes (plafond)")
    print(f"    {'niveau':<8}{'cases':>9}{'sans ench.':>12}{'atout neuf':>12}"
          f"{'contestée':>11}{'= v6 redemandé':>16}")
    for k, lv in enumerate(q["levels"]):
        name = "or" if k == 0 else f"−{k}"
        agree = "—" if k == 0 else f"{lv['agree_free_pct']:.1f}%"
        print(f"    {name:<8}{lv['cells']:>9}{lv['void_pct']:>11.1f}%"
              f"{lv['fresh_suit_pct']:>11.1f}%{lv['contested_pct']:>10.1f}%{agree:>16}")
    print(f"    couleurs couvertes : troncature {q['mean_covered']:.2f}, "
          f"v6 redemandé {q['mean_covered_free']:.2f}, "
          f"sans jamais rendre un siège muet {q['mean_covered_safe']:.2f} — sur 4")
    print("    annonce retirée : "
          + "  ".join(
              f"−{k} relance {100 * q['peel_raise'][k] / max(1, q['peel_raise'][k] + q['peel_open'][k]):.1f}%"
              for k in range(1, 4)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="taker-position")
    ap.add_argument("--reuse", help="répertoire contenant <nom>.json déjà produits")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    raws = {}
    with runlog.Timer() as t:
        with tempfile.TemporaryDirectory() as tmp:
            for name, (path, limit, variants) in CORPORA.items():
                if args.reuse:
                    with open(os.path.join(args.reuse, f"{name}.json"), encoding="utf-8") as fh:
                        raws[name] = json.load(fh)
                else:
                    raws[name] = run(path, limit, variants,
                                     os.path.join(tmp, f"{name}.json"))

    digests = {k: digest(v) for k, v in raws.items()}
    for name, d in digests.items():
        show(name, d)

    if not args.no_log:
        runlog.save(
            script="taker_position",
            tag=args.tag,
            params={"corpora": {k: {"path": p, "limit": lim, "variants": v}
                                for k, (p, lim, v) in CORPORA.items()},
                    "bid_model": BID_MODEL, "reused": bool(args.reuse)},
            summary=digests,
            payload=raws,
            # Les corpus jouent ici le rôle des poids : ils portent la politique
            # d'enchère qui sert de référence, et ils ne sont pas dans git.
            models=[os.path.join(ROOT, BID_MODEL)]
                   + [os.path.join(ROOT, p) for p, _, _ in CORPORA.values()],
            took_s=t.s,
        )


if __name__ == "__main__":
    main()
