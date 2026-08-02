#!/usr/bin/env python3
"""Bid v6 est-il équivariant aux permutations de couleurs ?

Question §1.1 de docs/bid/bid_v7_plan.md. Une main et la même main couleurs échangées
sont la même position — mais l'obs d'annonce porte la main en **bits bruts**
(`[0:32]`, colver-core/src/bid/bid_obs.rs) et rien dans un MLP n'impose que les deux
donnent la même réponse.

Trois mesures, dans cet ordre, parce que la première seule est ininterprétable :

  1. **Taux de bascule de l'annonce** sous les 23 permutations non triviales, avec
     deux bidders heuristiques en **contrôle**. Les contrôles ne sont pas nuls (ils
     départagent les ex æquo par indice de couleur, comportement déterministe et
     légitime) mais restent bas ; c'est ce qui valide l'arithmétique de permutation.
     Une erreur de mapping donnerait ~75 % partout, y compris sur les contrôles.

  2. **Échelle des Q**, sans quoi le point 3 ne veut rien dire : étendue max−min, et
     surtout l'écart top1−top2, la marge qui décide réellement de l'annonce.

  3. **Erreur d'équivariance du vecteur Q** — max sur les actions de |Q_σ(σ(a)) − Q(a)|.
     C'est la mesure propre : contrairement au taux de bascule, elle ne dépend pas de
     la platitude du sommet.

Le résultat tient dans un rapport, pas dans un pourcentage : l'erreur de symétrie est
petite en absolu (4,7 % de l'étendue) mais vaut **8,8× la marge de décision**. C'est
la cause mécanique des 24,6 %.

Référence (2026-08-02, 400 donnes × 23 permutations) :

    bascules   improved_v2 3,1 %   roro 0,9 %   **v6 24,6 %**
    échelle    étendue 0,79   top1−top2 médian 0,0042   (97,8 % des positions < 0,03)
    Q vectoriel  médiane 0,037   p90 0,054   p99 0,070   → 4,7 % de l'étendue

À lire avec §1.7 : les options que le bruit permute ne valent qu'environ 8 points
d'écart, donc le coût attendu est de ~2 pts/donne, pas des dizaines.

    uv run python scripts/analysis/bid_equivariance.py --deals 400
"""

import argparse
import os
import random
import sys
import time
from itertools import permutations

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

import runlog  # noqa: E402

BID_MODEL = "models/bid_v6_isdd_resume/bid_nn_final.bin"
PERMS = [p for p in permutations(range(4)) if p != (0, 1, 2, 3)]


def perm_card(c, sigma):
    return sigma[c // 8] * 8 + (c % 8)


def perm_action(a, sigma):
    """0 = PASS ; 1-36 = valeur×4 + couleur ; 37-40 = capot×couleur ; 41-42 = coinches."""
    if a == 0:
        return 0
    if 1 <= a <= 36:
        value, suit = divmod(a - 1, 4)
        return value * 4 + sigma[suit] + 1
    if 37 <= a <= 40:
        return 37 + sigma[a - 37]
    return a


IDENTITY = (0, 1, 2, 3)
SUIT_IDX = {"S": 0, "H": 1, "D": 2, "C": 3}


def parse_action(tok):
    """'P' -> passe ; '100C' -> 100 trèfle ; 'CS' / 'CAPOTS' -> capot pique."""
    t = tok.strip().upper()
    if t in ("P", "PASS", "PASSE"):
        return 0
    if t.startswith("CAPOT") or (t.startswith("C") and len(t) == 2):
        return 37 + SUIT_IDX[t[-1]]
    value, suit = int(t[:-1]), SUIT_IDX[t[-1]]
    if not (80 <= value <= 160) or value % 10:
        raise ValueError(f"valeur d'annonce invalide : {value}")
    return (value - 80) // 10 * 4 + suit + 1


def apply_prior(env, prior, sigma):
    """Rejoue le préfixe d'enchère **permuté par sigma**.

    C'est le point qui décide de la validité de toute la mesure sous préfixe : une
    permutation de couleurs agit sur la position *entière*, historique d'enchères
    compris. Rejouer 100♣ tel quel sous σ comparerait deux positions différentes, et
    le taux de bascule mesurerait alors ce décalage plutôt que la non-équivariance.

    `env.step()` ne valide pas la légalité (contrat d'un moteur RL), d'où le garde-fou :
    un préfixe qui devient illégal sous σ produirait des chiffres plausibles et faux.
    """
    for a in prior:
        pa = perm_action(a, sigma)
        if pa not in env.legal_actions():
            raise SystemExit(f"préfixe illégal sous σ={sigma} : action {pa}")
        env.step(pa)


def deals(rng, n):
    for _ in range(n):
        deck = list(range(32))
        rng.shuffle(deck)
        yield rng.randrange(4), [sorted(deck[i * 8 : (i + 1) * 8]) for i in range(4)]


def flip_rate(env, label, decide, n, seed, prior=()):
    rng = random.Random(seed)
    bad = total = 0
    for dealer, hands in deals(rng, n):
        env.redeal_with_hands(dealer, hands)
        apply_prior(env, prior, IDENTITY)
        ref = decide(env)
        for sigma in PERMS:
            env.redeal_with_hands(
                dealer, [sorted(perm_card(c, sigma) for c in h) for h in hands]
            )
            apply_prior(env, prior, sigma)
            total += 1
            if decide(env) != perm_action(ref, sigma):
                bad += 1
    print(f"   {label:18s} {bad:6,}/{total:,} = {100*bad/total:5.2f} %")
    return {"label": label, "flips": bad, "total": total, "pct": 100 * bad / total}


def q_scale(env, n, seed, prior=()):
    rng = random.Random(seed)
    spreads, gaps = [], []
    for dealer, hands in deals(rng, n):
        env.redeal_with_hands(dealer, hands)
        apply_prior(env, prior, IDENTITY)
        q = sorted((v for _, v in env.action_bid_nn()["q_values"]), reverse=True)
        spreads.append(q[0] - q[-1])
        if len(q) > 1:
            gaps.append(q[0] - q[1])
    spreads.sort()
    gaps.sort()
    pct = lambda a, p: a[int(p * (len(a) - 1))]  # noqa: E731
    print(f"   étendue des Q (max−min) : médiane {pct(spreads,.5):.2f}"
          f"   p10 {pct(spreads,.1):.2f}   p90 {pct(spreads,.9):.2f}")
    print(f"   écart top1−top2         : médiane {pct(gaps,.5):.4f}"
          f"   p25 {pct(gaps,.25):.4f}   p75 {pct(gaps,.75):.4f}")
    frac = 100 * sum(1 for g in gaps if g < 0.03) / len(gaps)
    print(f"   positions dont le top-2 tient sous 0,03 : {frac:.1f} %")
    return {"spread_median": pct(spreads, .5), "spread_p10": pct(spreads, .1),
            "spread_p90": pct(spreads, .9), "gap_median": pct(gaps, .5),
            "gap_p25": pct(gaps, .25), "gap_p75": pct(gaps, .75),
            "under_003_pct": frac, "raw_spreads": spreads, "raw_gaps": gaps}


def q_equivariance(env, n, seed, spread, margin, prior=()):
    rng = random.Random(seed)
    errs = []
    for dealer, hands in deals(rng, n):
        env.redeal_with_hands(dealer, hands)
        apply_prior(env, prior, IDENTITY)
        q0 = dict(env.action_bid_nn()["q_values"])
        for sigma in PERMS:
            env.redeal_with_hands(
                dealer, [sorted(perm_card(c, sigma) for c in h) for h in hands]
            )
            apply_prior(env, prior, sigma)
            q1 = dict(env.action_bid_nn()["q_values"])
            errs.append(max(
                abs(q1[perm_action(a, sigma)] - v)
                for a, v in q0.items()
                if perm_action(a, sigma) in q1
            ))
    errs.sort()
    pct = lambda p: errs[int(p * (len(errs) - 1))]  # noqa: E731
    print(f"   médiane {pct(.5):.4f}   p90 {pct(.9):.4f}   p99 {pct(.99):.4f}"
          f"   max {errs[-1]:.4f}")
    print(f"   rapportée à l'étendue des Q ({spread:.2f})     : "
          f"médiane {100*pct(.5)/spread:.2f} %")
    print(f"   rapportée à la marge de décision ({margin:.4f}) : "
          f"**{pct(.5)/margin:.1f}×**")
    return {"median": pct(.5), "p90": pct(.9), "p99": pct(.99), "max": errs[-1],
            "pct_of_spread": 100 * pct(.5) / spread,
            "ratio_to_margin": pct(.5) / margin, "raw_errs": errs}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--bid-model", default=BID_MODEL)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--prior", default="",
                    help="préfixe d'enchère, p.ex. '100C' (adversaire ouvre) "
                         "ou '100C P' (partenaire ouvre). Vide = ouverture.")
    ap.add_argument("--tag", default="", help="étiquette du run dans data/analysis/")
    ap.add_argument("--no-log", action="store_true",
                    help="ne rien écrire dans data/analysis/ (essais jetables)")
    args = ap.parse_args()

    prior = [parse_action(t) for t in args.prior.replace(",", " ").split()]

    env = colver.Env()
    env.reset()
    env.load_bid_model(args.bid_model, args.hidden)

    # Le siège qui décide est toujours `len(prior)` crans après le premier parleur,
    # donc la longueur du préfixe suffit à nommer le régime — comme dans
    # bid_candidates.py : 1 action = un adversaire a parlé, 2 = le partenaire.
    regime = {0: "ouverture", 1: "après l'adversaire", 2: "après le partenaire"}.get(
        len(prior), f"préfixe de {len(prior)} actions")
    print(f"Régime : {regime}" + (f"   (préfixe : {args.prior})" if prior else ""))

    t0 = time.monotonic()
    print(f"\n1. Bascules de l'annonce — {args.deals} donnes × {len(PERMS)} permutations")
    flips = [
        flip_rate(env, "improved_v2 (ctrl)", lambda e: e.bid_improved_v2(), args.deals, args.seed, prior),
        flip_rate(env, "roro (ctrl)", lambda e: e.bid_roro(), args.deals, args.seed, prior),
        flip_rate(env, "bid v6", lambda e: e.action_bid_nn()["best_action"], args.deals, args.seed, prior),
    ]

    print("\n2. Échelle des Q")
    scale = q_scale(env, args.deals, args.seed + 1, prior)

    print("\n3. Erreur d'équivariance du vecteur Q (max sur les actions)")
    equiv = q_equivariance(env, min(args.deals, 300), args.seed + 2,
                           scale["spread_median"], scale["gap_median"], prior)

    if not args.no_log:
        light = lambda d: {k: v for k, v in d.items() if not k.startswith("raw_")}  # noqa: E731
        runlog.save(
            "bid_equivariance", args.tag or (args.prior.replace(" ", "_") or "ouverture"),
            params={"deals": args.deals, "seed": args.seed, "prior": args.prior,
                    "regime": regime, "perms": len(PERMS)},
            summary={"flips": flips, "scale": light(scale), "equivariance": light(equiv)},
            payload={"scale_raw": {k: v for k, v in scale.items() if k.startswith("raw_")},
                     "equivariance_raw": equiv["raw_errs"]},
            models=[args.bid_model], took_s=time.monotonic() - t0,
        )


if __name__ == "__main__":
    main()
