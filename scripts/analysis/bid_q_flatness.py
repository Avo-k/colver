#!/usr/bin/env python3
"""Le Q plat de v6 reflète-t-il un jeu réellement plat, ou une sous-discrimination ?

Question §2.2 de docs/bid/bid_v7_plan.md. v6 sépare ses deux meilleures annonces par
0,0042 sur une étendue de Q de 0,79. Deux lectures possibles, opposées :

  (a) le jeu *est* plat au sommet — plusieurs annonces sont vraiment équivalentes,
      et le Q est bien calibré ;
  (b) v6 sous-discrimine — les annonces diffèrent réellement, mais il ne le voit pas.

On tranche en mesurant la valeur **réelle** (continuation d'enchère sur mondes
playgen, cf. bid_candidates.py) de top1, top2 et d'un contrôle positif « loin dans
le classement », sur le même pool de mondes par main.

Trois sorties :
  - Δréel(top1, top2) moyenné sur les mains → (a) si ~0, (b) si large ;
  - Δréel(top1, loin) → **contrôle positif** : si lui aussi est ~0, la mesure n'a
    aucune puissance et les deux autres colonnes ne veulent rien dire ;
  - corrélation ΔQ ↔ Δréel sur les paires (top1, top2).

    export COLVER_PLAYGEN_GPU_URL=http://localhost:8003
    uv run python scripts/analysis/bid_q_flatness.py --hands 40 --worlds 300
"""

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

import bid_candidates as bc  # noqa: E402
import runlog  # noqa: E402


def pearson(xs, ys):
    n = len(xs)
    if n < 3:
        return float("nan")
    mx, my = statistics.fmean(xs), statistics.fmean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys, strict=True))
    dx = sum((x - mx) ** 2 for x in xs) ** 0.5
    dy = sum((y - my) ** 2 for y in ys) ** 0.5
    return num / (dx * dy) if dx and dy else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=40)
    ap.add_argument("--worlds", type=int, default=300)
    ap.add_argument("--far-rank", type=int, default=8,
                    help="rang de l'annonce de contrôle positif (défaut 8e)")
    ap.add_argument("--prior", default="", help="préfixe d'enchère commun (optionnel)")
    ap.add_argument("--bid-model", default=bc.BID_MODEL)
    ap.add_argument("--play-model", default=bc.PLAY_MODEL)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--tag", default="", help="étiquette du run dans data/analysis/")
    ap.add_argument("--no-log", action="store_true",
                    help="ne rien écrire dans data/analysis/ (essais jetables)")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    prior = [bc.parse_action(t) for t in args.prior.replace(",", " ").split()] if args.prior else []
    dealer = (bc.SEAT - 1 - len(prior)) % 4
    team = bc.SEAT % 2

    env = colver.Env.deal_with_hands(dealer, bc.uniform_world(list(range(8)), rng))
    env.load_bid_model(args.bid_model)
    env.load_dmc_model(args.play_model)

    rows = []
    t0 = time.monotonic()
    for hi in range(args.hands):
        deck = list(range(32))
        rng.shuffle(deck)
        hand = sorted(deck[:8])

        env.redeal_with_hands(dealer, bc.uniform_world(hand, rng))
        for a in prior:
            env.step(int(a))
        q = dict(env.action_bid_nn()["q_values"])
        ranked = sorted(q.items(), key=lambda kv: -kv[1])
        if len(ranked) < args.far_rank + 1:
            continue
        top1, q1 = ranked[0]
        top2, q2 = ranked[1]
        far, qf = ranked[min(args.far_rank, len(ranked) - 1)]

        worlds, source = bc.sample_worlds(hand, dealer, prior, args.worlds, rng, verbose=False)
        if len(worlds) < args.worlds:
            print(f"[main {hi}] mondes insuffisants — ignorée", file=sys.stderr)
            continue

        diffs = {}
        for cand in (top1, top2, far):
            vals = []
            for w in worlds:
                r = bc.rollout(env, dealer, w, prior, cand)
                vals.append(r["scores"][team] - r["scores"][1 - team])
            diffs[cand] = vals

        d21 = [a - b for a, b in zip(diffs[top2], diffs[top1], strict=True)]
        df1 = [a - b for a, b in zip(diffs[far], diffs[top1], strict=True)]
        rows.append({
            "hand": " ".join(bc.card_name(c) for c in hand),
            "hand_cards": hand,
            "top1": bc.action_label(top1), "top2": bc.action_label(top2),
            "far": bc.action_label(far),
            "actions": {"top1": int(top1), "top2": int(top2), "far": int(far)},
            "q": {"top1": q1, "top2": q2, "far": qf},
            "dq_21": q1 - q2, "dq_f1": q1 - qf,
            "d_21": statistics.fmean(d21), "se_21": bc.stderr_of(d21),
            "d_f1": statistics.fmean(df1), "se_f1": bc.stderr_of(df1),
            # Le brut, monde par monde : c'est lui qui permet de ré-agréger autrement
            # (médiane, bootstrap, sous-ensemble) sans repayer les déroulements.
            "raw": {"top1": diffs[top1], "top2": diffs[top2], "far": diffs[far]},
            "source": source,
        })
        print(f"\r  {len(rows)}/{args.hands} mains  "
              f"({time.monotonic() - t0:.0f}s)", end="", file=sys.stderr)
    print(file=sys.stderr)

    if not rows:
        raise SystemExit("aucune main exploitable")

    print(f"\n{len(rows)} mains × {args.worlds} mondes × 3 candidates "
          f"= {len(rows) * args.worlds * 3} déroulements en {time.monotonic() - t0:.0f}s")
    print(f"Mondes : {rows[0]['source']}"
          + (f"   préfixe : {' '.join(bc.action_label(a) for a in prior)}" if prior else ""))

    print(f"\n{'main':>26s} {'top1':>7s} {'top2':>7s} {'ΔQ':>7s} "
          f"{'Δréel(2−1)':>13s} {'loin':>7s} {'Δréel(loin−1)':>15s}")
    print("-" * 96)
    for r in rows[:25]:
        print(f"{r['hand']:>26s} {r['top1']:>7s} {r['top2']:>7s} {r['dq_21']:7.4f} "
              f"{r['d_21']:+8.1f}±{r['se_21']:4.1f} {r['far']:>7s} "
              f"{r['d_f1']:+9.1f}±{r['se_f1']:4.1f}")
    if len(rows) > 25:
        print(f"  … {len(rows) - 25} mains de plus")

    d21 = [r["d_21"] for r in rows]
    df1 = [r["d_f1"] for r in rows]
    dq21 = [r["dq_21"] for r in rows]

    print("\n--- agrégat sur les mains (erreur type entre mains) ---")
    print(f"  Δréel(top2 − top1)  = {statistics.fmean(d21):+7.2f} ± {bc.stderr_of(d21):.2f}"
          f"   (négatif = top1 vraiment meilleur)")
    print(f"  Δréel(loin − top1)  = {statistics.fmean(df1):+7.2f} ± {bc.stderr_of(df1):.2f}"
          f"   ← CONTRÔLE POSITIF, doit être nettement négatif")
    print(f"  mains où top1 > top2 en réel : "
          f"{100 * sum(1 for x in d21 if x < 0) / len(d21):.0f}%")
    print(f"  corrélation ΔQ(1,2) ↔ Δréel(1,2) : r = {pearson(dq21, [-x for x in d21]):+.3f}"
          f"   (positif = le Q ordonne juste)")
    print(f"  ΔQ médian sur ces paires : {statistics.median(dq21):.4f}")

    if not args.no_log:
        runlog.save(
            "bid_q_flatness",
            args.tag or (args.prior.replace(" ", "_") if args.prior else "ouverture"),
            params={"hands": args.hands, "worlds": args.worlds, "seed": args.seed,
                    "prior": args.prior, "far_rank": args.far_rank,
                    "world_source": rows[0]["source"], "seat": bc.SEAT, "dealer": dealer},
            summary={"n_hands": len(rows), "rollouts": len(rows) * args.worlds * 3,
                     "d_21": statistics.fmean(d21), "se_21": bc.stderr_of(d21),
                     "d_f1": statistics.fmean(df1), "se_f1": bc.stderr_of(df1),
                     "top1_wins_pct": 100 * sum(1 for x in d21 if x < 0) / len(d21),
                     "r_dq_dreal": pearson(dq21, [-x for x in d21]),
                     "dq_median": statistics.median(dq21)},
            payload={"rows": rows},
            models=[args.bid_model, args.play_model],
            took_s=time.monotonic() - t0,
        )


if __name__ == "__main__":
    main()
