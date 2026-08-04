#!/usr/bin/env python3
"""Distribution des contrats annoncés et des points réellement faits, sur le corpus de parties.

Lit un COLVGM01 (`dealer`, `hands`, `actions`), rejoue chaque donne et relève :
  - la valeur du **contrat annoncé** (dernière enchère du tour d'enchères) ;
  - les **points faits par le preneur** (plis + dix de der, hors belote).

    env -u CONDA_PREFIX uv run --with numpy --no-project python \\
        scripts/analysis/bid_distribution.py data/training/playgen_games_9M.bin --games 200000

Sort un CSV sur stdout, consommé par `bid_curves_chart.py`.
"""
import argparse
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import numpy as np

import colver

CAPOT_BID = 250


def bits_to_cards(mask):
    return [i for i in range(32) if mask >> i & 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--games", type=int, default=200000)
    a = ap.parse_args()

    raw = np.memmap(a.path, dtype=np.uint8, mode="r")
    magic = raw[:8].tobytes()
    # COLVGM02 insère `score_ns`/`score_ew` (2×u16) entre les mains et le compte
    # d'actions. Lire un v2 avec le pas de v1 ne lèverait aucune erreur — ça
    # décalerait simplement chaque donne de 4 octets et rendrait des statistiques
    # d'enchères entièrement fausses. D'où le pas dérivé du magic, pas constant.
    assert magic in (b"COLVGM01", b"COLVGM02"), magic
    score_len = 4 if magic == b"COLVGM02" else 0
    total = int(np.frombuffer(raw[8:16].tobytes(), dtype="<u8")[0])
    n = min(a.games, total)

    bid_hist, made_hist, dd_hist = Counter(), Counter(), Counter()
    kept = passed = 0
    p = 16
    jobs = []
    for _ in range(n):
        dealer = int(raw[p])
        hands = [bits_to_cards(int(h)) for h in
                 np.frombuffer(raw[p + 1:p + 17].tobytes(), dtype="<u4")]
        head = p + 17 + score_len
        na = int(raw[head])
        acts = raw[head + 1:head + 1 + na].tolist()
        p = head + 1 + na

        jobs.append((dealer, hands, acts))

    def one(job):
        """Rejoue la donne, puis résout le même atout en double-mort.

        Le contrat ne se décode PAS depuis le flux d'actions : les cartes (0-31) et les
        enchères (1-36) partagent le même encodage. On rejoue et on interroge le moteur.
        """
        dealer, hands, acts = job
        env = colver.Env.deal_with_hands(dealer, hands)
        try:
            for act in acts:
                env.step(act)
        except Exception:
            return None
        if not env.is_terminal():
            return None
        c = env.get_contract()
        if not c["value"]:
            return "pass"
        taker = c["team"]
        # même donne, même atout, jeu parfait des deux côtés
        fresh = colver.Env.deal_with_hands(dealer, hands)
        dd = fresh.solve_all_suits()["suits"][c["trump"]][taker]
        return c["value"], env.get_points()[taker], dd

    with ThreadPoolExecutor(max_workers=8) as ex:
        for r in ex.map(one, jobs, chunksize=32):
            if r is None:
                continue
            if r == "pass":
                passed += 1
                continue
            bid, made, dd = r
            bid_hist[bid] += 1
            made_hist[min(made, 260)] += 1
            dd_hist[min(dd, 260)] += 1
            kept += 1

    print(f"# {a.path} — {kept:,} donnes retenues sur {n:,} ({passed:,} passées)",
          file=sys.stderr)
    print("bucket,annonce,points_faits,dd_atteignable")
    lo, hi = 60, 260
    for b in range(lo, hi + 1, 10):
        nb = sum(v for k, v in bid_hist.items() if b <= k < b + 10)
        nm = sum(v for k, v in made_hist.items() if b <= k < b + 10)
        nd = sum(v for k, v in dd_hist.items() if b <= k < b + 10)
        print(f"{b},{100 * nb / kept:.4f},{100 * nm / kept:.4f},{100 * nd / kept:.4f}")


if __name__ == "__main__":
    main()
