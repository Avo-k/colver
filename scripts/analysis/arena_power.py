#!/usr/bin/env python3
"""Que peut réellement résoudre l'arène, et à partir de quel écart ? (§2.9 du plan v7)

Plusieurs questions du plan posent un critère en % de matchs sans avoir vérifié que
l'instrument peut l'atteindre. §2.1 exigeait « ≥ 52 % sur 1000 matchs » pour la
canonicalisation, alors que §1.7 chiffre son effet à quelques points par décision. Un
critère hors de portée de la mesure teste la puissance de l'arène, pas l'hypothèse.

Trois mesures en une passe :

  1. **Fréquence des régimes d'enchère.** §1.7 donne un coût par *décision* (ouverture
     2,2 pts, contestation 3,9, soutien 0,2) ; sans savoir à quelle fréquence chaque
     régime se présente, on ne peut pas passer au coût par *donne*. On classe donc
     chaque décision réelle (≥ 2 actions légales) d'une enchère jouée par v6.

  2. **Distribution du score marqué par donne**, deux bots identiques. Sa dispersion
     est ce qui décide de tout : un avantage systématique δ doit émerger de ce bruit.

  3. **Courbe δ → % de matchs**, par simulation depuis la distribution empirique, et
     seuil de détectabilité aux effectifs usuels de l'arène.

Sur le duplicate matching : l'arène joue chaque paire dans les deux sens, ce qui annule
une partie de la chance de donne. On ne peut pas le mesurer ici (il faudrait deux bots
réellement différents), donc on **encadre** : σ mesuré = borne pessimiste, σ/2 = borne
optimiste. Si les deux bornes disent la même chose, la question est tranchée.

    uv run python scripts/analysis/arena_power.py --deals 2000
"""

import argparse
import os
import random
import statistics
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

import runlog  # noqa: E402

BID_MODEL = "models/bid_v6_isdd_resume/bid_nn_final.bin"
PLAY_MODEL = str(colver.model_path() or "models/play_v2/play_final.bin")


def regime_of(highest_seat, seat):
    """Classe une décision d'enchère du point de vue du siège qui parle."""
    if highest_seat is None:
        return "ouverture"
    if highest_seat == seat:
        return "notre enchere"
    if highest_seat == seat ^ 2:
        return "soutien"
    return "contestation"


def play_deals(env, n, rng, seed):
    """Joue n donnes complètes (enchère v6, jeu DouDou50). Rend scores + régimes."""
    scores, regimes, levels, void = [], {}, {}, 0
    for i in range(n):
        env.reset()
        highest_seat, highest_val = None, 0
        safety = 0
        while env.phase() == 0 and not env.is_terminal() and safety < 60:
            seat = int(env.current_player())
            legal = env.legal_actions()
            if len(legal) >= 2:                    # une passe forcée n'est pas une décision
                r = regime_of(highest_seat, seat)
                regimes[r] = regimes.get(r, 0) + 1
                if r == "contestation":
                    levels[highest_val] = levels.get(highest_val, 0) + 1
            a = int(env.bid_a_dd())
            if 1 <= a <= 40:                       # une annonce (pas passe/coinche)
                highest_seat = seat
                highest_val = 80 + 10 * ((a - 1) // 4) if a <= 36 else 250
            env.step(a)
            safety += 1

        if env.is_terminal():                      # 4 passes : donne nulle
            void += 1
            scores.append((0.0, 0.0))
            continue
        while not env.is_terminal():
            env.step(int(env.action_dmc_with_stats()["best_action"]))
        r = env.rewards()
        scores.append((float(r[0]), float(r[1])))
        if (i + 1) % 100 == 0:
            print(f"\r  {i + 1}/{n} donnes", end="", file=sys.stderr)
    print(file=sys.stderr)
    return scores, regimes, levels, void


def simulate(scores, delta, target, n_matches, rng, sigma_scale=1.0):
    """% de matchs gagnés par NS avec +delta pts/donne, cible `target`.

    On rejoue les **deux** scores de chaque donne, pas leur écart : en contrée les deux
    camps marquent (la défense encaisse ses points de cartes), donc la course à `target`
    ne se déduit pas de la seule différence. Une première version reversait tout l'écart
    au gagnant de la donne et rendait 58 % à δ = 0 — d'où le contrôle : **à δ = 0 le
    taux doit valoir 50 %**, sans quoi le reste de la courbe ne veut rien dire.

    `sigma_scale` rétrécit la chance de donne (l'écart) **à somme constante**, ce qui
    laisse la durée d'une partie réaliste : c'est le modèle du duplicate matching, où
    jouer la même donne dans les deux sens annule une partie de cette chance.
    """
    # **Pool symétrisé** : chaque donne entre aussi retournée. Avec deux bots identiques
    # et un donneur tiré au sort, la vraie distribution est symétrique ; toute asymétrie
    # d'un échantillon fini est du bruit, et elle se propageait directement dans la
    # courbe (53,7 % à δ = 0 sur un pool où NS gagnait 55 % des donnes). C'est aussi le
    # modèle fidèle du duplicate matching, où chaque donne est jouée dans les deux sens.
    base = list(scores) + [(ew, ns) for ns, ew in scores]
    if sigma_scale != 1.0:
        pool = []
        for ns, ew in base:
            s, d = ns + ew, (ns - ew) * sigma_scale
            pool.append(((s + d) / 2, (s - d) / 2))
    else:
        pool = base

    wins = ties = 0
    lengths = []
    k = len(pool)
    for _ in range(n_matches):
        ns = ew = 0.0
        nd = 0
        while nd < 200:
            a, b = pool[rng.randrange(k)]
            ns += a + delta
            ew += b
            nd += 1
            if (ns >= target or ew >= target) and ns != ew:
                break
        lengths.append(nd)
        if ns > ew:
            wins += 1
        elif ns == ew:
            ties += 1
    return 100 * wins / n_matches, statistics.fmean(lengths), ties


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--target", type=int, default=2000, help="points de la partie")
    ap.add_argument("--sims", type=int, default=20000, help="matchs simulés par δ")
    ap.add_argument("--seed", type=int, default=99)
    ap.add_argument("--bid-model", default=BID_MODEL)
    ap.add_argument("--play-model", default=PLAY_MODEL)
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    env = colver.Env()
    env.reset()
    env.load_bid_model(args.bid_model)
    env.load_dmc_model(args.play_model)

    t0 = time.monotonic()
    print(f"1. {args.deals} donnes jouées (enchère v6, jeu DouDou50)")
    scores, regimes, levels, void = play_deals(env, args.deals, rng, args.seed)

    tot = sum(regimes.values())
    print(f"\n2. Fréquence des régimes — {tot:,} décisions réelles "
          f"({tot / args.deals:.1f} par donne, passes forcées exclues)")
    freq = {}
    for r, c in sorted(regimes.items(), key=lambda kv: -kv[1]):
        freq[r] = 100 * c / tot
        print(f"   {r:16s} {c:7,}  {freq[r]:5.1f} %   ({c / args.deals:.2f}/donne)")
    if levels:
        top = sorted(levels.items(), key=lambda kv: -kv[1])[:5]
        print("   contestation par niveau : "
              + "  ".join(f"{v}→{100*c/sum(levels.values()):.0f}%" for v, c in top))

    diffs = [ns - ew for ns, ew in scores]
    sigma = statistics.stdev(diffs)
    ns_share = 100 * sum(1 for d in diffs if d > 0) / len(diffs)
    print("\n3. Score marqué par donne — écart NS−EW")
    print(f"   moyenne {statistics.fmean(diffs):+.1f}   médiane {statistics.median(diffs):+.1f}"
          f"   **σ = {sigma:.1f}**   donnes nulles {100*void/args.deals:.1f} %")
    print(f"   donnes gagnées par NS : {ns_share:.1f} %   (contrôle de symétrie : "
          f"doit être ~50 %, le donneur est tiré au sort)")

    print(f"\n4. Courbe δ → % de matchs (cible {args.target}, {args.sims:,} matchs par point)")
    deltas = [0, 1, 2, 4, 6, 10, 15, 25, 40]
    curve = []
    print(f"   {'δ (pts/donne)':>14s} {'σ mesuré':>12s} {'σ/2 (duplicate)':>17s}")
    for d in deltas:
        w1, ln, _ = simulate(scores, d, args.target, args.sims, random.Random(args.seed + d))
        w2, _, _ = simulate(scores, d, args.target, args.sims,
                            random.Random(args.seed + d), sigma_scale=0.5)
        curve.append({"delta": d, "win_pct": w1, "win_pct_halfsigma": w2, "deals": ln})
        print(f"   {d:>14d} {w1:11.2f}% {w2:16.2f}%"
              + ("   ← contrôle : doit valoir 50 %" if d == 0 else ""))
    print(f"   (une partie dure {curve[0]['deals']:.1f} donnes en moyenne)")
    if abs(curve[0]["win_pct"] - 50) > 2.0:
        print(f"   ⚠️  contrôle en échec ({curve[0]['win_pct']:.2f} % à δ=0) : "
              f"la courbe est biaisée, ne pas l'utiliser", file=sys.stderr)

    print("\n5. Seuil de détectabilité — δ minimal pour sortir du bruit à 95 %")
    thresholds = []
    for n in (1000, 2000, 5000, 10000):
        need = 50 + 100 * 1.96 * (0.25 / n) ** 0.5
        got = [c for c in curve if c["win_pct"] >= need]
        got2 = [c for c in curve if c["win_pct_halfsigma"] >= need]
        lo = f"{got[0]['delta']}" if got else f">{deltas[-1]}"
        hi = f"{got2[0]['delta']}" if got2 else f">{deltas[-1]}"
        thresholds.append({"matches": n, "need_win_pct": need, "delta_sigma": lo,
                           "delta_halfsigma": hi})
        print(f"   {n:>6,} matchs : il faut ≥ {need:.2f} %  →  "
              f"δ ≈ {hi} à {lo} pts/donne")

    if not args.no_log:
        runlog.save("arena_power", args.tag or f"target{args.target}",
                    params={"deals": args.deals, "target": args.target,
                            "sims": args.sims, "seed": args.seed},
                    summary={"regime_freq_pct": freq, "decisions_per_deal": tot / args.deals,
                             "sigma_deal": sigma, "void_pct": 100 * void / args.deals,
                             "mean_deals_per_match": curve[0]["deals"],
                             "curve": curve, "thresholds": thresholds},
                    payload={"scores": scores, "regimes": regimes, "levels": levels},
                    models=[args.bid_model, args.play_model],
                    took_s=time.monotonic() - t0)


if __name__ == "__main__":
    main()
