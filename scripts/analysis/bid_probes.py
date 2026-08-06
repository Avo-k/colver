#!/usr/bin/env python3
"""Suite de sondes stratifiée — ce qu'un bidder annonce sur des familles de mains choisies.

Question §3.2 de docs/bid/bid_v7_plan.md. Coût nul, risque nul : aucune donne n'est
jouée, aucun monde n'est échantillonné, on lit seulement la sortie du réseau sur des
mains **construites**. C'est l'instrument qui manquait — il aurait attrapé §1.3 (le
capot est une action morte) des mois avant qu'on le cherche.

## Pourquoi construire les mains plutôt que les tirer

Les familles qui décident sont rares : un capot forcé est à ~10⁻⁴, une main de huit
atouts à 4/10 518 300. Les tirer au hasard, c'est ne jamais les voir. On les construit
donc, ce qui a un effet secondaire utile — la sonde reste *exactement* la même d'un
checkpoint à l'autre, donc deux runs sont directement comparables.

## Pourquoi par régime

§1.7 : le comportement de v6 dépend beaucoup plus du **type de décision** que de la
main. 24,6 % d'annonces qui basculent sous renommage de couleur à l'ouverture, 0 à 10 %
dès qu'une annonce est sur la table. Une moyenne sur les régimes cache l'essentiel, donc
chaque famille est jouée dans les trois :

    ouverture      --prior ""        on parle en premier
    contestation   --prior "100C"    un adversaire a annoncé 100♣
    soutien        --prior "100C P"  le partenaire a annoncé 100♣, un adversaire a passé

Ce sont les **mêmes** chaînes de préfixe que bid_q_flatness.py et bid_equivariance.py,
pour que les trois sondes parlent des mêmes régimes.

## Ce qui est une assertion et ce qui est une observation

Une seule famille a une réponse **prouvable** : huit cartes d'une même couleur prennent
les huit levées quelle que soit la répartition des 24 autres, et sous le barème du dépôt
un capot réussi marque 502 contre 412 pour un 160 tous plis. `check` la vérifie et le
run sort en erreur si elle tombe. Tout le reste est *observé* et comparé à la référence :
on ne prétend pas connaître la bonne annonce sur une coupe franche.

⚠️ **v6 échoue l'assertion dure, à 100 %, dans les trois régimes** — c'est l'état
documenté, pas une panne de la sonde. Le code de sortie 1 est donc attendu tant que le
capot n'est pas réveillé (§3.3). Il n'y a volontairement pas de drapeau pour le taire.

    uv run python scripts/analysis/bid_probes.py                       # v6, 200 mains/famille
    uv run python scripts/analysis/bid_probes.py --bid-model <ckpt.bin> --baseline ref.json
    uv run python scripts/analysis/bid_probes.py --save-baseline docs/measurements/bid_probes_v6.json
"""

import argparse
import json
import os
import random
import statistics
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402

import bid_candidates as bc  # noqa: E402
import runlog  # noqa: E402

# Rangs, en bits de carte. L'ordre de force à l'atout est J 9 A T K Q 8 7.
R7, R8, R9, RJ, RQ, RK, RT, RA = 0, 1, 2, 3, 4, 5, 6, 7
TOP6 = [RJ, R9, RA, RT, RK, RQ]  # les six maîtres d'une couleur d'atout
LOW3 = [R7, R8, R9]

REGIMES = [
    ("ouverture", ""),
    ("contestation", "100C"),
    ("soutien", "100C P"),
]


# --------------------------------------------------------------------- familles

def _pick(rng, pool, k):
    return rng.sample(pool, k)


def f_huit_atouts(rng):
    """Les 8 cartes d'une même couleur. Huit levées par construction."""
    s = rng.randrange(4)
    return [s * 8 + r for r in range(8)]


def f_sept_atouts(rng):
    """7 cartes d'une couleur + 1 ailleurs."""
    s = rng.randrange(4)
    missing = rng.randrange(8)
    hand = [s * 8 + r for r in range(8) if r != missing]
    hand.append(rng.choice([c for c in range(32) if c // 8 != s]))
    return hand


def f_six_maitres(rng):
    """J 9 A T K Q d'une couleur + 2 cartes ailleurs. Six levées d'atout sûres."""
    s = rng.randrange(4)
    hand = [s * 8 + r for r in TOP6]
    hand += _pick(rng, [c for c in range(32) if c // 8 != s], 2)
    return hand


def f_cinq_maitres(rng):
    """J 9 A + 2 autres de la couleur + 3 ailleurs — la main de prise ordinaire."""
    s = rng.randrange(4)
    hand = [s * 8 + r for r in (RJ, R9, RA)]
    hand += _pick(rng, [s * 8 + r for r in (R7, R8, RQ, RK, RT)], 2)
    hand += _pick(rng, [c for c in range(32) if c // 8 != s], 3)
    return hand


def f_coupe_longue(rng):
    """Coupe franche dans une couleur, longue de 5 dans une autre."""
    void, long_s = _pick(rng, list(range(4)), 2)
    hand = _pick(rng, [long_s * 8 + r for r in range(8)], 5)
    rest = [c for c in range(32) if c // 8 not in (void, long_s)]
    hand += _pick(rng, rest, 3)
    return hand


def f_belote_seche(rng):
    """Dame + Roi d'une couleur (la belote), et rien d'autre dans cette couleur.

    La belote vaut 20 points *si on prend dans cette couleur*. Le piège de cette
    famille est là : le bonus n'existe qu'en atout, et il ne rachète pas une main
    faible. C'est exactement le genre de raisonnement qu'on veut voir dans la sortie.
    """
    s = rng.randrange(4)
    hand = [s * 8 + RQ, s * 8 + RK]
    weak = [c for c in range(32) if c // 8 != s and c % 8 in LOW3]
    hand += _pick(rng, weak, 6)
    return hand


def f_main_pauvre(rng):
    """8 cartes prises parmi les douze 7-8-9 du paquet. Aucune carte à points."""
    return _pick(rng, [c for c in range(32) if c % 8 in LOW3], 8)


def f_quatre_as(rng):
    """Les quatre as + 4 basses. Beaucoup de points, aucune longue."""
    hand = [s * 8 + RA for s in range(4)]
    hand += _pick(rng, [c for c in range(32) if c % 8 in LOW3], 4)
    return hand


def f_aleatoire(rng):
    """Témoin uniforme — sans lui on ne sait pas si un écart est propre à une famille."""
    return _pick(rng, list(range(32)), 8)


def is_capot(a):
    return 37 <= a <= 40


def bid_value(a):
    """Valeur d'annonce en points, 0 pour Passe, 250 pour un capot."""
    if a == 0 or a > 40:
        return 0
    return 250 if is_capot(a) else 80 + 10 * ((a - 1) // 4)


def perm_action(a, k):
    """Décale la couleur d'une action de `k`. Passe / Coinche sont invariants."""
    if 1 <= a <= 36:
        v, su = divmod(a - 1, 4)
        return v * 4 + (su + k) % 4 + 1
    if is_capot(a):
        return 37 + (a - 37 + k) % 4
    return a


FAMILIES = [
    # (nom, générateur, attente lisible, check dur ou None)
    ("huit_atouts", f_huit_atouts, "capot (prouvable)", is_capot),
    ("sept_atouts", f_sept_atouts, "capot ou 160", None),
    ("six_maitres", f_six_maitres, "haut", None),
    ("cinq_maitres", f_cinq_maitres, "prise ordinaire", None),
    ("coupe_longue", f_coupe_longue, "annonce dans la longue", None),
    ("belote_seche", f_belote_seche, "modéré, pas de sur-annonce", None),
    ("main_pauvre", f_main_pauvre, "passe", None),
    ("quatre_as", f_quatre_as, "modéré", None),
    ("aleatoire", f_aleatoire, "témoin", None),
]


# --------------------------------------------------------------------- mesure

def probe_family(env, dealer, prior, gen, n, rng):
    """Retourne les lignes brutes pour une famille dans un régime."""
    rows, seen = [], set()
    tries = 0
    while len(rows) < n and tries < n * 20:
        tries += 1
        hand = sorted(gen(rng))
        if len(set(hand)) != 8:
            continue
        key = tuple(hand)
        if key in seen:
            continue  # certaines familles sont minuscules (4 mains de huit atouts)
        seen.add(key)

        env.redeal_with_hands(dealer, bc.uniform_world(hand, rng))
        for a in prior:
            env.step(int(a))
        out = env.action_bid_nn()
        q = sorted(out["q_values"], key=lambda kv: -kv[1])
        best = int(out["best_action"])
        margin = q[0][1] - q[1][1] if len(q) > 1 else float("nan")
        rows.append({
            "hand": [int(c) for c in hand],
            "hand_str": " ".join(bc.card_name(c) for c in hand),
            "action": best,
            "label": bc.action_label(best),
            "q_top1": q[0][1],
            "margin": margin,
        })
    return rows


def summarise(rows, check):
    acts = Counter(r["label"] for r in rows)
    fails = [r for r in rows if check and not check(r["action"])]
    return {
        "n": len(rows),
        "pass_pct": 100 * sum(1 for r in rows if r["action"] == 0) / len(rows),
        "capot_pct": 100 * sum(1 for r in rows if is_capot(r["action"])) / len(rows),
        "top": acts.most_common(3),
        "q_top1": statistics.fmean(r["q_top1"] for r in rows),
        "margin_median": statistics.median(r["margin"] for r in rows),
        "check_fail_pct": 100 * len(fails) / len(rows) if check else None,
        "check_fail_example": fails[0]["hand_str"] + " → " + fails[0]["label"] if fails else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hands", type=int, default=200, help="mains par famille et par régime")
    ap.add_argument("--bid-model", default=bc.BID_MODEL)
    ap.add_argument("--canonical", action="store_true",
                    help="réseau entraîné sur l'ordre canonique des couleurs (v7+). Indétectable depuis le fichier — même taille qu'un réseau physique — et l'oublier rend une annonce légale dans la mauvaise couleur, sans erreur.")
    ap.add_argument("--seed", type=int, default=20260802)
    ap.add_argument("--regimes", default="", help="sous-ensemble, p.ex. 'ouverture,soutien'")
    ap.add_argument("--baseline", default="", help="JSON de référence à comparer")
    ap.add_argument("--save-baseline", default="", help="écrire la référence dans ce JSON")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    wanted = [r.strip() for r in args.regimes.split(",") if r.strip()]
    regimes = [r for r in REGIMES if not wanted or r[0] in wanted]

    t0 = time.monotonic()
    env = colver.Env.deal_with_hands(0, bc.uniform_world(list(range(8)), random.Random(0)))
    env.load_bid_model(args.bid_model, None, args.canonical)

    result, raw, hard_fail = {}, {}, []
    for reg_name, prior_str in regimes:
        prior = [bc.parse_action(t) for t in prior_str.split()] if prior_str else []
        # Même convention que bid_q_flatness : le siège sondé parle juste après le préfixe.
        dealer = (bc.SEAT - 1 - len(prior)) % 4
        result[reg_name], raw[reg_name] = {}, {}
        # Une graine par régime, dérivée de la même racine : les familles voient donc
        # les mêmes mains dans les trois régimes, ce qui rend les colonnes comparables.
        for fam, gen, expect, check in FAMILIES:
            rng = random.Random(f"{args.seed}:{fam}")
            rows = probe_family(env, dealer, prior, gen, args.hands, rng)
            if not rows:
                continue
            s = summarise(rows, check)
            s["expect"] = expect
            result[reg_name][fam] = s
            raw[reg_name][fam] = rows
            if check and s["check_fail_pct"]:
                hard_fail.append(f"{reg_name}/{fam}: {s['check_fail_pct']:.0f}% "
                                 f"(ex. {s['check_fail_example']})")
            print(f"\r  {reg_name}/{fam}  ", end="", file=sys.stderr)
    print(file=sys.stderr)

    # ------------------------------------------------------------------ affichage
    print(f"\nmodèle : {args.bid_model}")
    print(f"{args.hands} mains demandées par famille × {len(regimes)} régimes "
          f"en {time.monotonic() - t0:.1f}s")

    for reg_name, _ in regimes:
        print(f"\n=== {reg_name} ===")
        print(f"{'famille':>15s} {'n':>4s} {'passe':>7s} {'capot':>7s} "
              f"{'Q top1':>7s} {'marge':>7s}  annonces les plus fréquentes")
        print("-" * 100)
        for fam in result[reg_name]:
            s = result[reg_name][fam]
            top = "  ".join(f"{lab} {100 * c / s['n']:.0f}%" for lab, c in s["top"])
            print(f"{fam:>15s} {s['n']:4d} {s['pass_pct']:6.1f}% {s['capot_pct']:6.1f}% "
                  f"{s['q_top1']:7.3f} {s['margin_median']:7.4f}  {top}")

    # ---------------------------------------------------- équivariance, cas exact
    # Les quatre mains de huit atouts sont des permutations de couleur exactes les
    # unes des autres, donc ce que le modèle en dit *est* une mesure d'équivariance,
    # sans échantillonnage et sans marge d'erreur.
    #
    # ⚠️ Le préfixe doit être permuté avec la main. Sinon, en contestation, les quatre
    # positions comparées sont « j'ai huit ♠ et l'adversaire a dit 100♣ », « j'ai huit
    # ♥ et l'adversaire a dit 100♣ »… qui ne sont pas la même position : la main ♣ est
    # la seule dont la longue est la couleur annoncée. C'est le piège qu'`apply_prior`
    # évite déjà dans bid_equivariance.py, et il gonflait ici l'étendue à 160 pts.
    print("\n--- équivariance sur les 4 mains de huit atouts (permutations exactes) ---")
    print("    (main ET préfixe permutés ensemble ; l'annonce est ramenée dans le repère"
          " de référence, donc les 4 colonnes doivent être identiques)")
    equiv = {}
    for reg_name, prior_str in regimes:
        base = [bc.parse_action(t) for t in prior_str.split()] if prior_str else []
        dealer = (bc.SEAT - 1 - len(base)) % 4
        labels, values = [], []
        for k in range(4):
            hand = sorted((0 + k) % 4 * 8 + r for r in range(8))
            env.redeal_with_hands(dealer, bc.uniform_world(hand, random.Random(k)))
            for a in base:
                env.step(int(perm_action(a, k)))
            got = int(env.action_bid_nn()["best_action"])
            back = perm_action(got, -k)  # ramené dans le repère k=0
            labels.append(bc.action_label(back))
            values.append(bid_value(back))
        spread = max(values) - min(values)
        equiv[reg_name] = {"labels": labels, "values": values, "spread": spread,
                           "equivariant": len(set(labels)) == 1}
        flag = "" if len(set(labels)) == 1 else "   ← NON ÉQUIVARIANT"
        print(f"  {reg_name:>13s} : {'  '.join(f'{lb:>7s}' for lb in labels)}"
              f"    étendue {spread:3d} pts{flag}")

    # ------------------------------------------------------------------ régression
    if args.baseline:
        with open(args.baseline) as fh:
            ref = json.load(fh)["result"]
        print("\n--- écarts contre la référence ---")
        moved = 0
        for reg_name in result:
            for fam, s in result[reg_name].items():
                r = ref.get(reg_name, {}).get(fam)
                if not r:
                    continue
                d_pass = s["pass_pct"] - r["pass_pct"]
                same_top = s["top"][0][0] == r["top"][0][0]
                if abs(d_pass) >= 2.0 or not same_top:
                    moved += 1
                    print(f"  {reg_name}/{fam}: passe {r['pass_pct']:.1f}% → "
                          f"{s['pass_pct']:.1f}% ({d_pass:+.1f}), "
                          f"annonce dominante {r['top'][0][0]} → {s['top'][0][0]}")
        print(f"  {moved} cellule(s) bougée(s) sur {sum(len(v) for v in result.values())}")

    if args.save_baseline:
        with open(args.save_baseline, "w") as fh:
            json.dump({"bid_model": args.bid_model, "hands": args.hands,
                       "seed": args.seed, "result": result}, fh, indent=1, ensure_ascii=False)
        print(f"\nréférence écrite : {args.save_baseline}")

    if not args.no_log:
        runlog.save(
            "bid_probes",
            args.tag or "suite",
            params={"hands": args.hands, "seed": args.seed,
                    "regimes": [r[0] for r in regimes],
                    "families": [f[0] for f in FAMILIES], "seat": bc.SEAT},
            summary={"familles": result, "equivariance_huit_atouts": equiv},
            payload={"rows": raw},
            models=[args.bid_model],
            took_s=time.monotonic() - t0,
        )

    if hard_fail:
        # Une seule famille a une réponse prouvable ; si elle tombe, c'est un défaut
        # du modèle, pas du bruit de mesure. Le code de sortie le dit.
        print("\nÉCHEC des assertions dures :", file=sys.stderr)
        for line in hard_fail:
            print("  " + line, file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    main()
