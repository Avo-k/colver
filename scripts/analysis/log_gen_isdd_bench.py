#!/usr/bin/env python3
"""Verse au registre les mesures d'optimisation de `gen_games_isdd`.

Les chiffres ont été produits par des A/B alternés à la main (deux sidecars
vivants, la charge passant de l'un à l'autre) et non par un script : les
reprendre ici est ce qui les rend relisables dans six mois, conformément à
`docs/measurements/README.md`. Un run d'analyse qui n'écrit que sur stdout se
repaie à chaque question posée.

Toutes les valeurs sont en **donnes par seconde**, corpus = donnes tirées au
hasard, joueur = `arena/bots/gen_isdd.toml` (bid v6 + IS-DD, mondes playgen),
sidecar sur la 3090 de moxxi, client sur 32 cœurs.

    uv run python scripts/analysis/log_gen_isdd_bench.py
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

MODEL = "models/playgen/playgen_v2_final.bin"

# Chaque entrée : (étiquette, [valeurs observées]). Plusieurs valeurs = plusieurs
# tours alternés ; on garde la liste et pas seulement la moyenne, parce que la
# dispersion est ce qui dit si un écart tient.
RUNS = {
    # Concurrence, sidecar d'origine
    "baseline_32t": [1.05],
    "baseline_96t": [1.48],
    # Préfixe groupé seul, 64 threads, client sans sur-commande
    "prefill_off_64t": [1.08, 1.19, 1.22],
    "prefill_on_64t": [1.75, 1.67, 1.70],
    # Préfixe + retrait de lanes, 256 threads, client identique des deux côtés
    "gpu_orig_256t": [1.73, 1.86, 1.69],
    "gpu_opt_256t": [2.57, 2.71, 2.58],
    # Ablation du retrait de lanes, MÊME binaire (COLVER_PLAYGEN_NO_RETIRE)
    "retire_on": [2.48, 2.20, 2.29],
    "retire_off": [1.27, 1.95, 1.87],
    # Modèle réduit v3-small (d=256 L=4) — autre échantillonneur, pas une optim
    "v3_small": [4.27, 4.36],
    "v2_final": [2.57, 2.68],
    # TF32 : régression franche
    "tf32_on": [0.83, 0.44],
    "tf32_off": [2.28, 2.18],
    # Mondes par décision, VRAM propre
    "dets20": [3.93],
    "dets40": [2.34],
    "dets60": [1.70],
    # Calendrier par stade, à total de mondes égal puis décroissant
    "sched_up_280": [1.90, 1.35],
    "flat40_vs_up": [2.15, 2.34],
    "sched_down_200": [2.56, 2.44],
    "flat40_vs_down": [1.95, 2.08],
}


def med(v):
    s = sorted(v)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def main():
    m = {k: med(v) for k, v in RUNS.items()}
    summary = {
        "unite": "donnes/s",
        "medianes": {k: round(x, 3) for k, x in m.items()},
        "rapports": {
            "prefixe_groupe_64t": round(m["prefill_on_64t"] / m["prefill_off_64t"], 3),
            "gpu_total_256t": round(m["gpu_opt_256t"] / m["gpu_orig_256t"], 3),
            "retrait_de_lanes": round(m["retire_on"] / m["retire_off"], 3),
            "v3_small_vs_v2": round(m["v3_small"] / m["v2_final"], 3),
            "tf32": round(m["tf32_on"] / m["tf32_off"], 3),
            "calendrier_montant_a_total_egal": round(
                m["sched_up_280"] / m["flat40_vs_up"], 3),
            "calendrier_decroissant": round(
                m["sched_down_200"] / m["flat40_vs_down"], 3),
            "bout_en_bout": round(m["gpu_opt_256t"] / m["baseline_32t"], 3),
        },
        "notes": [
            "A/B alternés, jamais deux exécutions séquentielles : la charge dérive "
            "de ~20 % sur cette machine.",
            "Le retrait de lanes est ablaté dans le MÊME binaire "
            "(COLVER_PLAYGEN_NO_RETIRE=1), pas entre deux compilations.",
            "v3-small n'est pas une optimisation : ses mondes sont à 2,09× le bruit "
            "d'échantillonnage de ceux de v2 (bench_prefill_eq). C'est un autre joueur.",
            "Le calendrier montant est plus LENT à total de mondes égal : un monde "
            "d'entame demande 24 cartes cachées (48 pas) contre 6 (12 pas) en finale.",
            "Une série entière a été faussée de ~30 % par trois sidecars OISIFS "
            "occupant 21 Go des 24 de la carte. Vérifier nvidia-smi avant de croire.",
        ],
    }
    runlog.save(
        script="gen_isdd_bench",
        tag="prefill-retire-v3small",
        params={
            "bot": "arena/bots/gen_isdd.toml",
            "dets": 40,
            "threads": 256,
            "sidecar": "moxxi RTX 3090",
            "client": "32 coeurs",
            "deals_par_run": "120-300",
        },
        summary=summary,
        payload={"runs_bruts": RUNS},
        models=[MODEL],
    )


if __name__ == "__main__":
    main()
