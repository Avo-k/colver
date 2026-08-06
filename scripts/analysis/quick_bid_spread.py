#!/usr/bin/env python3
"""Que vaut le « +83 pts » de l'analyse rapide d'une annonce, et que vaut sa dispersion ?

La page Annonces et le panneau « Analyse rapide » de Rejouer affichent tous deux une
**espérance d'écart de score** (Nord-Sud − Est-Ouest) sur des donnes jouées par
DouDou50 depuis une annonce forcée. La question qui a déclenché cette mesure : faut-il
lui adjoindre un écart type, ou un pire/meilleur cas ?

Ce script rend les trois chiffres qui répondent, et ils ne disent pas la même chose :

1. **La dispersion brute** (`spread`) — moyenne, écart type, quantiles, extrêmes de
   l'écart de score sur *n* simulations d'une annonce donnée.
2. **La dispersion à l'intérieur de chaque issue** (`cond`) — le même écart, découpé
   selon qui prend le contrat et s'il le réussit. C'est le test de l'hypothèse « la
   distribution n'est pas un nuage mais quelques paquets ».
3. **L'appariement** (`paired`) — les deux annonces d'un panneau rejouées sur les
   **mêmes** mondes, pour savoir si partager les tirages réduirait assez le bruit pour
   que les deux lignes deviennent comparables.

## Ce que la mesure a répondu le 2026-08-06

3 mains, 600 sims par annonce, 400 paires appariées
(`docs/measurements/index.jsonl`, tag `design-2026-08-06`) :

- σ ≈ **310 à 370 points** pour une moyenne comprise entre −84 et −26 — l'écart type
  vaut plusieurs fois la moyenne et il est **le même pour toutes les annonces**, donc
  il ne distingue rien. La distribution est à deux bosses (p25 ≈ −370, p75 ≈ +290) et
  la moyenne tombe **dans le creux**, là où il n'y a presque pas de donnes : un « ± »
  suppose un nuage centré, il n'y en a pas.
- Les extrêmes observés (−662 / +752) sont des maxima sur *n* tirages : ils grandissent
  avec le budget de simulation. Un « pire cas » afficherait la taille de l'échantillon.
- **91,2 % de la variance est *entre* les quatre issues** (σ dans une case : 65 à 125
  points). C'est ça, la dispersion à montrer — « on prend et ça passe → +281 ; on prend
  et ça chute → −400 » — et la moyenne s'en déduit.
- Ce qui manquait vraiment est l'incertitude **sur la moyenne** : ±54 points à 160 sims
  (Rejouer), ±22 à 1000 (Annonces). À l'écran, +83 contre +69 est du bruit.
- **Apparier ne sauve pas** : ρ = 0,36 à 0,48 seulement (10 à 18 % des mondes finissent
  à l'identique — forcer une autre annonce change toute l'enchère), soit 1,25 à 1,38×
  sur l'écart type de la différence, qui reste à ±51-65 points à 160 sims contre
  ±71-82 sans appariement. Et l'écart *vrai* entre deux annonces voisines vaut quelques
  points (+22,9 ± 41,4 · +7,1 ± 37,3 · +13,8 ± 32,4, les trois compatibles avec zéro) :
  il n'y a rien à séparer, donc aucun budget ne les séparera.

Réserve : trois mains, et des annonces voisines (X contre X+10 dans la même couleur).
« passe » contre « 100♥ » se sépare sans doute mieux — ce qui ne change rien à la
conclusion sur l'écart type, qui ne dépend pas du couple comparé.

Conséquences dans le code : `_quick_bid_readout` arrondit l'espérance à la dizaine,
`_doudou_new_stats` accumule `pts_gap_sq_sum` (l'intervalle) et `outcomes` (les quatre
paquets), et la page Annonces affiche les deux.

    uv run python scripts/analysis/quick_bid_spread.py --sims 600 --pairs 400 --hands 3
"""

import argparse
import json
import os
import random
import statistics as st
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

SUITS = "SHDC"
_env = None
# Sud parle en premier, comme dans `_quick_bid_one` sans historique d'enchères.
DEALER = 1


def bid_label(a):
    return f"{80 + (a - 1) // 4 * 10}{SUITS[(a - 1) % 4]}" if 1 <= a <= 36 else str(a)


def get_env(bid_path, dmc_path, hands):
    """Un seul `Env` réutilisé : le charger coûte plus que la donne qu'il joue."""
    global _env
    if _env is None:
        _env = colver.Env.deal_with_hands(DEALER, hands)
        _env.load_bid_model(bid_path)
        _env.load_dmc_model(dmc_path)
    else:
        _env.redeal_with_hands(DEALER, hands)
    return _env


def deal_hands(hand, rest):
    """Mélange uniforme des 24 cartes restantes — ce que fait `_run_single_doudou_sim`."""
    random.shuffle(rest)
    hands = [None] * 4
    hands[2] = sorted(hand)
    for i, p in enumerate([0, 1, 3]):
        hands[p] = sorted(rest[i * 8:(i + 1) * 8])
    return hands


def play(hands, forced, bid_path, dmc_path):
    """Une donne : annonce forcée, enchère au NN, jeu par DouDou50.

    Rend (écart Nord-Sud − Est-Ouest, issue). Réplique de `_run_single_doudou_sim`.
    """
    env = get_env(bid_path, dmc_path, hands)
    if forced is not None and env.phase() == 0 and not env.is_terminal():
        env.step(forced)
    n = 0
    while env.phase() == 0 and not env.is_terminal() and n < 50:
        env.step(int(env.bid_a_dd()))
        n += 1
    contract = env.get_contract()
    if not contract:
        return 0.0, "passee"
    while not env.is_terminal():
        env.step(int(env.action_dmc_with_stats()["best_action"]))
    r = env.rewards()
    ours, made = contract["team"] == 0, r[contract["team"]] > 0
    kind = ("ns_made" if made else "ns_set") if ours else ("ew_made" if made else "ew_set")
    return float(r[0]) - float(r[1]), kind


def describe(gaps):
    q = st.quantiles(gaps, n=20)
    sd = st.stdev(gaps)
    return {"n": len(gaps), "mean": round(st.fmean(gaps), 1), "sd": round(sd, 1),
            "ci95": round(1.96 * sd / len(gaps) ** 0.5, 1),
            "min": min(gaps), "max": max(gaps),
            "p5": q[0], "p25": q[4], "p50": q[9], "p75": q[14], "p95": q[18]}


def pick_hands(n_hands, bid_path, dmc_path):
    """Des mains sur lesquelles le NN ouvre par une annonce, et un cran au-dessus.

    C'est la situation du panneau : « ce qui a été joué » contre « ce que v6 dirait ».
    On s'arrête à 32 pour que l'annonce du dessus existe dans la même couleur.
    """
    out, tried = [], 0
    while len(out) < n_hands and tried < 200:
        tried += 1
        deck = list(range(32))
        random.shuffle(deck)
        hand = sorted(deck[:8])
        rest = [c for c in range(32) if c not in hand]
        env = get_env(bid_path, dmc_path, deal_hands(hand, rest))
        a = int(env.bid_a_dd())
        if 1 <= a <= 32:
            out.append((hand, a, a + 4))   # même couleur, +10
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sims", type=int, default=600, help="simulations par annonce")
    ap.add_argument("--pairs", type=int, default=400, help="mondes partagés par les deux")
    ap.add_argument("--hands", type=int, default=3)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--tag", default="default")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    bid_path = str(colver.bid_model_path() or colver.download_bid_model())
    dmc_path = str(colver.model_path() or colver.download_model())

    random.seed(args.seed)
    rows = []
    for hand, a, b in pick_hands(args.hands, bid_path, dmc_path):
        rest = [c for c in range(32) if c not in hand]
        row = {"hand": hand, "bids": [bid_label(a), bid_label(b)]}

        # (1) + (2) : dispersion brute et par issue, sur l'annonce de gauche.
        gaps, kinds = [], []
        for _ in range(args.sims):
            g, k = play(deal_hands(hand, rest), a, bid_path, dmc_path)
            gaps.append(g)
            kinds.append(k)
        row["spread"] = describe(gaps)
        row["by_outcome"] = {}
        for k in sorted(set(kinds)):
            xs = [g for g, kk in zip(gaps, kinds, strict=True) if kk == k]
            row["by_outcome"][k] = {
                "n": len(xs), "mean": round(st.fmean(xs), 1),
                "sd": round(st.stdev(xs), 1) if len(xs) > 1 else None}
        # Part de la variance expliquée par l'issue : c'est ce chiffre qui dit
        # qu'un écart type global est le mauvais résumé.
        within = sum(o["n"] * (o["sd"] or 0) ** 2 for o in row["by_outcome"].values())
        within /= max(1, sum(o["n"] for o in row["by_outcome"].values()))
        row["variance_explained_by_outcome"] = round(1 - within / row["spread"]["sd"] ** 2, 3)

        # (3) : les deux annonces sur les mêmes mondes.
        ga, gb = [], []
        for _ in range(args.pairs):
            hands = deal_hands(hand, rest)
            ga.append(play([list(h) for h in hands], a, bid_path, dmc_path)[0])
            gb.append(play([list(h) for h in hands], b, bid_path, dmc_path)[0])
        d = [x - y for x, y in zip(ga, gb, strict=True)]
        sd_a, sd_b, sd_d = st.stdev(ga), st.stdev(gb), st.stdev(d)
        rho = (sd_a ** 2 + sd_b ** 2 - sd_d ** 2) / (2 * sd_a * sd_b)
        row["paired"] = {
            "pairs": args.pairs, "delta": round(st.fmean(d), 1),
            "delta_ci95": round(1.96 * sd_d / args.pairs ** 0.5, 1),
            "sd_delta_paired": round(sd_d, 1),
            "sd_delta_unpaired": round((sd_a ** 2 + sd_b ** 2) ** 0.5, 1),
            "rho": round(rho, 3),
            "gain": round((sd_a ** 2 + sd_b ** 2) ** 0.5 / sd_d, 2),
            "identical_worlds_pct": round(100 * sum(1 for x in d if x == 0) / len(d)),
        }
        rows.append({**row, "_gaps": gaps, "_kinds": kinds, "_ga": ga, "_gb": gb})
        print(json.dumps(row, ensure_ascii=False), flush=True)

    # Ce que ça donne au budget réel des deux écrans.
    sds = [r["spread"]["sd"] for r in rows]
    at = lambda n: round(1.96 * st.fmean(sds) / n ** 0.5)          # noqa: E731
    print(f"\nIC 95 % sur l'espérance : ±{at(160)} pts à 160 sims (Rejouer), "
          f"±{at(1000)} à 1000 (Annonces)")
    print(f"Variance expliquée par l'issue : "
          f"{st.fmean(r['variance_explained_by_outcome'] for r in rows):.1%}")

    if not args.no_log:
        import runlog

        runlog.save(
            script="quick_bid_spread",
            tag=args.tag,
            params={"sims": args.sims, "pairs": args.pairs, "hands": args.hands,
                    "seed": args.seed, "worlds": "uniform", "play": "doudou50",
                    "bid": "nn (bid_a_dd)"},
            summary={
                "sd_gap": [r["spread"]["sd"] for r in rows],
                "mean_gap": [r["spread"]["mean"] for r in rows],
                "ci95_at_160_sims": at(160),
                "ci95_at_1000_sims": at(1000),
                "variance_explained_by_outcome": [
                    r["variance_explained_by_outcome"] for r in rows],
                "paired_rho": [r["paired"]["rho"] for r in rows],
                "paired_gain": [r["paired"]["gain"] for r in rows],
                "true_delta_between_neighbouring_bids": [
                    [r["paired"]["delta"], r["paired"]["delta_ci95"]] for r in rows],
            },
            payload={"rows": rows},
            models=[bid_path, dmc_path],
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
