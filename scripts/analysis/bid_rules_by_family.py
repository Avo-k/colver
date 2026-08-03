#!/usr/bin/env python3
"""Où les règles distillées butent-elles, et est-ce leur faute ?

Suite de [bid_rule_ceiling.py](bid_rule_ceiling.py), qui établit qu'aucune règle
équivariante ne peut dépasser un certain accord avec le réseau. Ce script mesure
l'écart **effectivement** atteint par un arbre / un XGBoost, famille de mains par
famille de mains, et le pose à côté du plafond.

Trois choses que le protocole publié
([bid_rules_xgb_v2.md](../../docs/bid/interpretability/bid_rules_xgb_v2.md)) ne pouvait
pas faire :

1. **Localiser le résidu.** « 92,8 % d'accord » est un scalaire ; découpé par `HandCode`
   il devient une liste de familles où la règle tombe, donc un domaine de validité.
2. **Distinguer l'erreur de l'arbre du bruit du réseau.** Une famille où le réseau se
   contredit sous renommage plafonne bas *pour tout le monde*. Sans le plafond, on
   attribue à l'arbre ce qui appartient au réseau.
3. **Apprendre la bonne cible.** Le réseau donne jusqu'à 24 réponses par main ; en
   prendre une au hasard (celle de l'identité) fait apprendre du bruit à l'arbre. On
   entraîne donc aussi sur le **mode de l'orbite** — le réseau symétrisé, qui est le
   meilleur objet équivariant qu'on puisse extraire de lui — et on mesure les deux.

⚠️ **La note est l'accord avec l'orbite** — la fraction des 24 réponses du réseau que la
règle retrouve — et non l'accord avec la cible apprise. Sans quoi les quatre lignes ne
sont pas sur la même échelle : noter contre le mode revient à noter contre le réseau
*symétrisé*, qui vise 100 %, alors que le plafond borne l'accord avec le réseau *tel
qu'il répond*. Le symptôme qui l'attrape est visible : une règle qui « dépasse » son
plafond.

Le CSV de `distill_bid` ne peut pas servir : il ne porte que les features agrégées, pas
la main, donc aucune colonne `HandCode` ne s'y greffe. On regénère ici, en Python — le
binding rend ~40 µs par annonce, donc 24 permutations × 60 000 mains coûtent ~1 min.

    uv run python scripts/analysis/bid_rules_by_family.py --deals 60000
"""

import argparse
import os
import random
import sys
import time
from collections import Counter, defaultdict

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

import runlog  # noqa: E402
from bid_equivariance import IDENTITY, apply_prior, parse_action  # noqa: E402
from bid_rule_ceiling import (  # noqa: E402
    ALL_PERMS,
    anchor_suit,
    deals,
    inverse,
    perm_action,
    perm_card,
    prior_suit,
    suit_key,
)

BID_MODEL = "models/bid_v6_isdd_resume/bid_nn_final.bin"

# Mêmes features que le protocole publié (distill_bid.py::BASE_FEATURES), calculées
# sur la couleur qu'on annoncerait, plus un rappel de la deuxième.
FEATURES = [
    "trump_count", "has_jack", "has_nine", "has_ace", "has_ten",
    "has_king", "has_queen", "trump_points", "trump_score", "has_belote",
    "side_aces", "side_tens", "side_voids", "side_singletons",
    "side_doubletons", "total_aces", "best_side_length",
    "second_trump_score", "second_trump_count",
]

# Sous préfixe d'enchère, les features qui décrivent ma main **relativement à la couleur
# annoncée**. Les trois premières sont celles du protocole publié (RESPONSE_FEATURES) ;
# `best_other_ts` est la découverte du probe — le meilleur trump_score de mes couleurs
# *en ignorant* celle qu'on vient d'annoncer, une interaction main × historique que les
# features agrégées ne portaient pas ([probe_morning_report.md](
# ../../docs/bid/interpretability/probe_morning_report.md)).
NAMED_FEATURES = ["named_suit_cards", "named_is_my_best", "named_ts"]
PROBE_FEATURE = "best_other_ts"
SECOND_FEATURES = ["second_trump_score", "second_trump_count"]

# Les 17 du protocole publié, sans les deux de la 2e couleur.
PUBLISHED = [f for f in FEATURES if f not in SECOND_FEATURES]


def feature_sets(has_prior):
    """Les jeux de features à comparer, du plus pauvre au plus riche.

    L'enjeu est de savoir si `best_other_ts` apporte de l'**information** ou seulement
    un **biais inductif**. Il vaut `trump_score` quand ma meilleure couleur n'est pas
    celle qu'on annonce, et `second_trump_score` sinon : il est donc exactement
    reconstructible dès que les deux sont là. Le tester contre un jeu qui ne porte que
    l'un des deux ne mesure pas ce qu'on croit.
    """
    base = PUBLISHED + (NAMED_FEATURES if has_prior else [])
    sets = {"publié (17)": base, "+ 2e couleur": base + SECOND_FEATURES}
    if has_prior:
        sets["+ probe"] = base + [PROBE_FEATURE]
        sets["tout"] = base + SECOND_FEATURES + [PROBE_FEATURE]
    return sets

RANK_J, RANK_9, RANK_A, RANK_T, RANK_K, RANK_Q = 3, 2, 7, 6, 5, 4
TRUMP_POINTS = {RANK_J: 20, RANK_9: 14, RANK_A: 11, RANK_T: 10, RANK_K: 4, RANK_Q: 3}


def hand_features(hand, scores, trump):
    """Les 19 features, pour `trump` désigné. Fonction pure de (main, couleur)."""
    by_suit = [[c % 8 for c in hand if c // 8 == s] for s in range(4)]
    tr = by_suit[trump]
    sides = [by_suit[s] for s in range(4) if s != trump]
    order = sorted(range(4), key=lambda s: (-scores[s], -suit_key(hand, s)))
    second = order[1] if order[0] == trump else order[0]
    return {
        "trump_count": len(tr),
        "has_jack": int(RANK_J in tr), "has_nine": int(RANK_9 in tr),
        "has_ace": int(RANK_A in tr), "has_ten": int(RANK_T in tr),
        "has_king": int(RANK_K in tr), "has_queen": int(RANK_Q in tr),
        "trump_points": sum(TRUMP_POINTS.get(r, 0) for r in tr),
        "trump_score": scores[trump],
        "has_belote": int(RANK_K in tr and RANK_Q in tr),
        "side_aces": sum(1 for s in sides if RANK_A in s),
        "side_tens": sum(1 for s in sides if RANK_T in s),
        "side_voids": sum(1 for s in sides if not s),
        "side_singletons": sum(1 for s in sides if len(s) == 1),
        "side_doubletons": sum(1 for s in sides if len(s) == 2),
        "total_aces": sum(1 for s in range(4) if RANK_A in by_suit[s]),
        "best_side_length": max((len(s) for s in sides), default=0),
        "second_trump_score": scores[second],
        "second_trump_count": len(by_suit[second]),
    }


def action_value(a):
    """Classe de niveau : 0 = passe, 8..16 = 80..160, 25 = capot, 41/42 = coinches."""
    if a == 0 or a >= 41:
        return a
    if 37 <= a <= 40:
        return 25
    return (a - 1) // 4 + 8


def collect(env, n, seed, prior=()):
    rng = random.Random(seed)
    rows = []
    called = prior_suit(prior)
    for dealer, hands in deals(rng, n):
        env.redeal_with_hands(dealer, hands)
        apply_prior(env, prior, IDENTITY)
        seat = env.current_player()
        hand = hands[seat]
        scores = env.evaluate_hand(seat)["scores"]
        # Couleur pour laquelle on calcule les features : la meilleure de la main,
        # départagée par l'ordre d'atout — un ordre total stable au renommage, donc les
        # features aussi. La **famille**, elle, s'ancre sur la couleur annoncée dès qu'il
        # y en a une : c'est la question que le siège se pose.
        best = anchor_suit(hand, scores)
        anchor = best if called is None else called
        extra = {}
        if called is not None:
            extra = {
                "named_suit_cards": sum(1 for c in hand if c // 8 == called),
                "named_is_my_best": int(best == called),
                "named_ts": scores[called],
                "best_other_ts": max(scores[s] for s in range(4) if s != called),
            }
        answers = []
        for sigma in ALL_PERMS:
            env.redeal_with_hands(
                dealer, [sorted(perm_card(c, sigma) for c in h) for h in hands]
            )
            apply_prior(env, prior, sigma)
            answers.append(perm_action(env.action_bid_nn()["best_action"], inverse(sigma)))
        raw = answers[ALL_PERMS.index((0, 1, 2, 3))]
        mode = Counter(answers).most_common(1)[0][0]
        nb_pass = sum(1 for x in answers if x == 0)
        vals = Counter(action_value(a) for a in answers)
        rows.append({
            "code": colver.hand_code(hand, anchor, "trump"),
            "feat": {**hand_features(hand, scores, best), **extra},
            "raw_bid": int(raw != 0), "mode_bid": int(nb_pass * 2 < len(answers)),
            "raw_val": action_value(raw), "mode_val": action_value(mode),
            # Effectifs de l'orbite : de quoi noter n'importe quelle prédiction par son
            # accord avec le réseau *tel qu'il répond*, et non avec la cible apprise.
            "orb_bid": {0: nb_pass, 1: len(answers) - nb_pass},
            "orb_val": dict(vals),
            "n24": len(answers),
            "ceil_bin": max(nb_pass, len(answers) - nb_pass) / len(answers),
            "ceil_val": vals.most_common(1)[0][1] / len(answers),
        })
    return rows


def fit_and_score(rows, target, depth, use_xgb, feats):
    from sklearn.tree import DecisionTreeClassifier

    X = np.array([[r["feat"][f] for f in feats] for r in rows], dtype=np.float32)
    y = np.array([r[target] for r in rows])
    n_tr = int(0.7 * len(rows))
    idx = np.random.RandomState(0).permutation(len(rows))
    tr, te = idx[:n_tr], idx[n_tr:]

    models = {f"arbre d{depth}": DecisionTreeClassifier(max_depth=depth, random_state=0)}
    if use_xgb:
        import xgboost as xgb
        # Les classes se lisent sur le **train** seul. Prises sur tout le jeu, une classe
        # qui n'apparaît qu'au test laisse un trou dans les étiquettes réencodées et
        # XGBoost refuse d'ajuster (« Expected: [0..8], got [0..7, 9] »). Le cas se
        # produit dès qu'un niveau d'enchère est très rare — la défense en a un.
        classes = sorted(set(y[tr].tolist()))
        remap = {c: i for i, c in enumerate(classes)}
        models["xgboost"] = ("xgb", classes, remap,
                             xgb.XGBClassifier(n_estimators=200, max_depth=6,
                                               learning_rate=0.1, verbosity=0,
                                               tree_method="hist", n_jobs=8))
    out = {}
    for name, m in models.items():
        if isinstance(m, tuple):
            _, classes, remap, clf = m
            clf.fit(X[tr], np.array([remap[v] for v in y[tr]]))
            pred_all = np.array([classes[i] for i in clf.predict(X)])
        else:
            m.fit(X[tr], y[tr])
            pred_all = m.predict(X)
        out[name] = pred_all
    return out, y, te


def orbit_agreement(rows, ii, pred, orb_key):
    """Fraction des 24 réponses du réseau que la prédiction retrouve.

    **La seule note comparable au plafond**, quelle que soit la cible d'apprentissage :
    noter contre le mode reviendrait à noter contre le réseau symétrisé, dont le plafond
    n'est pas 100 %. Confondre les deux laisse une règle « dépasser » son plafond.
    """
    return 100 * sum(rows[i][orb_key].get(pred[i], 0) for i in ii) / sum(
        rows[i]["n24"] for i in ii)


def report(rows, target, ceil_key, orb_key, preds, y, te, label):
    te_set = set(te.tolist())
    print(f"\n{label}")
    ceil = 100 * np.mean([rows[i][ceil_key] for i in te])
    print(f"   plafond équivariant                     : {ceil:5.1f} %")
    for name, pred in preds.items():
        acc = orbit_agreement(rows, te, pred, orb_key)
        print(f"   {name:20s} accord à l'orbite  : {acc:5.1f} %"
              f"   (reste {ceil-acc:5.1f} pt sous le plafond)")
    # Par famille, sur le seul jeu de test.
    fams = defaultdict(list)
    for i, r in enumerate(rows):
        if i in te_set:
            fams[r["code"]].append(i)
    best = list(preds)[-1]
    table = []
    for code, ii in fams.items():
        if len(ii) < 60:
            continue
        c = 100 * np.mean([rows[i][ceil_key] for i in ii])
        a = orbit_agreement(rows, ii, preds[best], orb_key)
        table.append({"code": code, "n": len(ii), "ceiling": c, "acc": a, "gap": c - a})
    return table


def show(title, table, limit=12):
    print(f"\n   {title}")
    print(f"      {'code':22s} {'n':>6s} {'plafond':>8s} {'atteint':>8s} {'manque':>7s}")
    for t in table[:limit]:
        print(f"      {t['code']:22s} {t['n']:6d} {t['ceiling']:7.1f}% "
              f"{t['acc']:7.1f}% {t['gap']:6.1f} pt")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=60000)
    ap.add_argument("--bid-model", default=BID_MODEL)
    ap.add_argument("--hidden", type=int, default=512)
    ap.add_argument("--seed", type=int, default=4242)
    ap.add_argument("--depth", type=int, default=5)
    ap.add_argument("--prior", default="",
                    help="préfixe d'enchère, p.ex. '100C' (l'adversaire ouvre). "
                         "Vide = ouverture.")
    ap.add_argument("--no-feature-sets", action="store_true",
                    help="sauter la comparaison des jeux de features (plus rapide)")
    ap.add_argument("--no-xgb", action="store_true")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    prior = [parse_action(t) for t in args.prior.replace(",", " ").split()]
    regime = {0: "ouverture", 1: "après l'adversaire", 2: "après le partenaire"}.get(
        len(prior), f"préfixe de {len(prior)} actions")
    has_prior = prior_suit(prior) is not None

    env = colver.Env()
    env.reset()
    env.load_bid_model(args.bid_model, args.hidden)

    t0 = time.monotonic()
    rows = collect(env, args.deals, args.seed, prior)
    took = time.monotonic() - t0
    sets = feature_sets(has_prior)
    feats = sets["tout" if has_prior else "+ 2e couleur"]
    print(f"Modèle : {args.bid_model}")
    print(f"Régime : {regime}" + (f"   (préfixe : {args.prior})" if prior else ""))
    print(f"{args.deals:,} mains × 24 permutations   ({took:.0f} s)"
          f"   {len(feats)} features")

    summary = {}
    for target, ceil_key, orb_key, label in [
        ("raw_bid", "ceil_bin", "orb_bid", "A. Annoncer / passer — cible = réponse brute (protocole publié)"),
        ("mode_bid", "ceil_bin", "orb_bid", "B. Annoncer / passer — cible = mode de l'orbite (réseau symétrisé)"),
        ("raw_val", "ceil_val", "orb_val", "C. Niveau annoncé — cible = réponse brute"),
        ("mode_val", "ceil_val", "orb_val", "D. Niveau annoncé — cible = mode de l'orbite"),
    ]:
        preds, y, te = fit_and_score(rows, target, args.depth, not args.no_xgb, feats)
        table = report(rows, target, ceil_key, orb_key, preds, y, te, label)
        table.sort(key=lambda t: -t["gap"])
        show("Familles où la règle manque le plus son plafond", table)
        summary[target] = {
            "ceiling": 100 * float(np.mean([rows[i][ceil_key] for i in te])),
            "acc": {k: orbit_agreement(rows, te, v, orb_key) for k, v in preds.items()},
            "families": sorted(table, key=lambda t: -t["n"]),
        }

    ablation = {}
    if not args.no_feature_sets:
        print("\n─── Jeux de features (cible = mode de l'orbite, accord à l'orbite) ───")
        for target, ceil_key, orb_key, what in [
            ("mode_bid", "ceil_bin", "orb_bid", "annoncer / passer"),
            ("mode_val", "ceil_val", "orb_val", "niveau annoncé"),
        ]:
            print(f"\n   {what}")
            for sname, sfeats in sets.items():
                preds, y, te = fit_and_score(rows, target, args.depth,
                                             not args.no_xgb, sfeats)
                cells = {k: orbit_agreement(rows, te, v, orb_key)
                         for k, v in preds.items()}
                ablation[f"{target}/{sname}"] = {"n_features": len(sfeats), **cells}
                print(f"      {sname:14s} ({len(sfeats):2d} feat.)  "
                      + "   ".join(f"{k} {v:5.1f} %" for k, v in cells.items()))
            print(f"      {'plafond':14s}            "
                  f"{100*np.mean([rows[i][ceil_key] for i in te]):24.1f} %")

    if not args.no_log:
        p = runlog.save(
            "bid_rules_by_family", args.tag or "run",
            {"deals": args.deals, "seed": args.seed, "depth": args.depth,
             "prior": args.prior, "regime": regime, "model": args.bid_model},
            {"took_s": took, "feature_sets": ablation, **summary},
            payload={"rows": rows},
            models=[args.bid_model], took_s=took)
        print(f"\nJournalisé → {p}")


if __name__ == "__main__":
    main()
