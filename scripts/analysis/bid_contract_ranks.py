#!/usr/bin/env python3
"""Quels contrats la boucle d'entraînement du bidder atteint-elle réellement ?

La reward ne lit qu'**une** case `[atout]` de la couche de scores par épisode
(`bid_train_env::compute_scores`). Étiqueter les quatre atouts au même prix n'a donc
de sens que si les quatre sont atteints — et `dd_pts` seul ne peut pas le dire : c'est
l'ε-greedy et un réseau qui débute qui décident, pas l'argmax d'un bidder entraîné.

Le chiffre qui compte est le **rang de l'atout contracté du point de vue du camp qui
l'a pris**. Le rang « de la donne » — les quatre atouts classés en mélangeant les deux
camps — est un piège : sur une donne où N-S est fort à ♠ et E-O à ♥, les deux couleurs
sortent en tête, mais le camp qui gagne l'enchère n'a le choix qu'entre *ses* quatre
valeurs. Mesuré, l'écart entre les deux lectures est de 20 points au rang 0.

Deux régimes, parce que la réponse en dépend entièrement :
  * tardif — poids de v6, ε = 0,02 : ce que consulte une politique entraînée ;
  * début — init aléatoire, ε = 0,30 : ce que consulte une politique qui apprend.

Usage :
    uv run python scripts/analysis/bid_contract_ranks.py --tag budget-couche-v2
    uv run python scripts/analysis/bid_contract_ranks.py --reuse <dir>   # journaliser sans recalculer

Le binaire doit être construit :
    cargo build -p colver-core --bin train_bid_nn --features dmc_train --release
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
BIN = os.path.join(ROOT, "target/release/train_bid_nn")
POOL = "data/deals/base_5M.bin"
SCORES = "data/deals/scores_isdd_5M.sc"
CKPT = "models/bid_v6_isdd_resume/bid_nn_final.safetensors"

RANK_LABELS = ["rang 0", "rang 1", "rang 2", "rang 3"]


def run_regime(name, out_json, save_dir, steps, eps, resume):
    """Un régime = un court entraînement dont on ne garde que l'histogramme."""
    cmd = [
        BIN,
        "--num-envs", "256", "--hidden", "512", "--layers", "3",
        "--eps-start", str(eps), "--eps-end", str(eps), "--eps-decay-steps", "1",
        "--steps", str(steps),
        "--reward", "real", "--score-aware", "--sa-features-v3", "--match-sim",
        "--reward-clip", "1.0",
        "--pool-file", POOL, "--scores", SCORES,
        # L'éval coûte 1000 matchs et ne sert à rien ici ; la repousser hors d'atteinte
        # est le seul moyen de mesurer 30k pas en 5 min.
        "--eval-freq", "999000000", "--save-freq", "999000000",
        "--save-dir", save_dir,
        "--log-contract-ranks", out_json,
    ]
    if resume:
        cmd += ["--resume", resume]
    print(f"[{name}] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, check=True, stdout=subprocess.DEVNULL)
    with open(out_json, encoding="utf-8") as fh:
        return json.load(fh)


def digest(raw):
    """Agrégats lisibles. Reste petit : c'est ce qui part dans l'index versionné."""
    n = raw["contracts"]
    pct = lambda xs: [round(100 * x / n, 2) for x in xs]  # noqa: E731
    taker = raw["by_rank_taker"]
    capot = raw["capot_bids"]
    return {
        "contracts": n,
        "voids": raw["voids"],
        "by_rank_taker_pct": pct(taker),
        "top2_pct": round(100 * (taker[0] + taker[1]) / n, 2),
        "top3_pct": round(100 * (taker[0] + taker[1] + taker[2]) / n, 2),
        # Lecture « de la donne », gardée pour montrer l'écart avec la bonne.
        "by_rank_deal_pct": pct(raw["by_rank"]),
        # Part des contrats pris par le camp que dd_pts désigne : le complément
        # borne la population dont les croyances d'étiquetage seraient décalées
        # si la couche reste en [u8;4] au lieu de [u8;8].
        "taker_is_dd_side_pct": round(100 * sum(raw["by_rank_matched"]) / n, 2),
        # Camp preneur sous 80 points DD : il ne peut pas tenir le contrat minimum.
        "taker_below_80_pct": round(100 * raw["taker_pts_bucket"][0] / n, 2),
        "taker_dd_capot_pct": round(100 * raw["taker_pts_bucket"][8] / n, 2),
        "capot_contracts_pct": round(100 * capot / n, 2),
        "capot_sound_pct": round(100 * raw["capot_bids_sound"] / capot, 2) if capot else None,
    }


def show(name, d):
    print(f"\n=== {name} — {d['contracts']:,} contrats, {d['voids']:,} donnes passées ===")
    print("  rang de l'atout contracté, vu du camp preneur :")
    for lbl, p in zip(RANK_LABELS, d["by_rank_taker_pct"]):
        print(f"    {lbl} : {p:5.2f} %")
    print(f"    top-2 : {d['top2_pct']:.2f} %   top-3 : {d['top3_pct']:.2f} %")
    print(f"  preneur = camp désigné par dd_pts : {d['taker_is_dd_side_pct']:.2f} %")
    print(f"  preneur sous 80 pts DD (sur-annonce) : {d['taker_below_80_pct']:.2f} %")
    print(f"  capot annoncé : {d['capot_contracts_pct']:.2f} % des contrats, "
          f"dont {d['capot_sound_pct']} % sur une main qui a vraiment le capot")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=30000)
    ap.add_argument("--tag", default="contract-ranks")
    ap.add_argument("--reuse", help="répertoire contenant final.json / early2.json déjà produits")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    with runlog.Timer() as t:
        if args.reuse:
            with open(os.path.join(args.reuse, "final.json"), encoding="utf-8") as fh:
                late = json.load(fh)
            with open(os.path.join(args.reuse, "early2.json"), encoding="utf-8") as fh:
                early = json.load(fh)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                late = run_regime("tardif", os.path.join(tmp, "late.json"),
                                  os.path.join(tmp, "ck"), args.steps, 0.02,
                                  os.path.join(ROOT, CKPT))
                early = run_regime("début", os.path.join(tmp, "early.json"),
                                   os.path.join(tmp, "ck2"), args.steps, 0.30, None)

    d_late, d_early = digest(late), digest(early)
    show("TARDIF (v6, ε=0,02)", d_late)
    show("DÉBUT (init aléatoire, ε=0,30)", d_early)

    if not args.no_log:
        runlog.save(
            script="bid_contract_ranks",
            tag=args.tag,
            params={"steps": args.steps, "num_envs": 256, "pool": POOL, "scores": SCORES,
                    "match_sim": True, "sa_features": "v3", "reward": "real",
                    "reused": bool(args.reuse)},
            summary={"late": d_late, "early": d_early},
            payload={"late_raw": late, "early_raw": early},
            models=[os.path.join(ROOT, CKPT), os.path.join(ROOT, POOL),
                    os.path.join(ROOT, SCORES)],
            took_s=t.s,
        )


if __name__ == "__main__":
    main()
