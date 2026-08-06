#!/usr/bin/env python3
"""Valeur réelle des annonces candidates, par **continuation d'enchère**.

Pour une main + un préfixe d'enchère, on force chaque annonce candidate, puis on
laisse l'enchère se poursuivre normalement (NN aux quatre sièges) et la donne se
jouer (DouDou50), sur un pool de mondes playgen conditionné par le préfixe. On lit
le score **marqué** du camp du siège.

C'est la cible de label validée par docs/bid/experiments/auction_conditioned_labels.md :
« valeur de cette *action* dans la suite de l'enchère », et non « valeur de jouer ce
contrat » — cette dernière fait sur-annoncer.

Pas d'Oracle DD ici, volontairement : un solve est un *majorant* par monde (il voit
les quatre mains), donc moyenner mille solves donne un majorant, pas une valeur. Le
biais ne se moyenne pas. Cf. docs/bid/bid_v7_plan.md §4.

Le pool de mondes est **partagé** par toutes les candidates : la comparaison est donc
appariée, et c'est l'écart apparié (± son erreur type) qui est lisible, pas l'écart
des moyennes brutes.

Exemples
--------
    export COLVER_PLAYGEN_GPU_URL=http://localhost:8003   # avant l'import !
    uv run python scripts/analysis/bid_candidates.py --hand "AD TD KD QD JD 9D 8D 7D"
    uv run python scripts/analysis/bid_candidates.py --hand "AS TS KS QS JS 9S AH TH" \
        --worlds 400 --top 5
    uv run python scripts/analysis/bid_candidates.py --hand "..." --prior "P,100H,P"
"""

import argparse
import json
import os
import random
import statistics
import sys
import time

import colver

BID_MODEL = "models/bid_v6_isdd_resume/bid_nn_final.bin"
PLAYGEN_MODEL = "models/playgen/playgen_v2_final.bin"
# DouDou50 : le même fichier que celui que sert le web (`colver.model_path()`),
# pour que ces chiffres décrivent le joueur de prod et pas un checkpoint voisin.
PLAY_MODEL = str(colver.model_path() or "models/play_v2/play_final.bin")

RANK_BIT = {"7": 0, "8": 1, "9": 2, "J": 3, "Q": 4, "K": 5, "T": 6, "A": 7}
BIT_RANK = {v: k for k, v in RANK_BIT.items()}
SUIT_IDX = {"S": 0, "H": 1, "D": 2, "C": 3}
SUIT_SYM = ["♠", "♥", "♦", "♣"]

SEAT = 2  # Sud, comme la page Annonces
GPU_CHUNK = 512  # `max_worlds` du sidecar — au-delà la réponse revient tronquée


# --------------------------------------------------------------------------- cartes

def parse_card(tok):
    tok = tok.strip().upper().replace("10", "T")
    if len(tok) != 2 or tok[0] not in RANK_BIT or tok[1] not in SUIT_IDX:
        raise ValueError(f"carte illisible : {tok!r} (attendu p.ex. AS, TD, 9H)")
    return SUIT_IDX[tok[1]] * 8 + RANK_BIT[tok[0]]


def card_name(c):
    return f"{BIT_RANK[c % 8]}{SUIT_SYM[c // 8]}"


def parse_hand(s):
    cards = [parse_card(t) for t in s.replace(",", " ").split()]
    if len(cards) != 8:
        raise ValueError(f"{len(cards)} cartes, il en faut 8")
    if len(set(cards)) != 8:
        raise ValueError("carte en double")
    return sorted(cards)


def parse_action(tok):
    """'P'/'PASS' -> 0 ; '100H' -> action ; 'CAPOTS' / 'CS' -> capot pique."""
    t = tok.strip().upper()
    if t in ("P", "PASS", "PASSE"):
        return 0
    if t.startswith("CAPOT") or (t.startswith("C") and len(t) == 2):
        suit = t[-1]
        if suit not in SUIT_IDX:
            raise ValueError(f"capot illisible : {tok!r}")
        return 37 + SUIT_IDX[suit]
    suit = t[-1]
    if suit not in SUIT_IDX:
        raise ValueError(f"annonce illisible : {tok!r}")
    value = int(t[:-1])
    if not (80 <= value <= 160) or value % 10:
        raise ValueError(f"valeur d'annonce invalide : {value}")
    return (value - 80) // 10 * 4 + SUIT_IDX[suit] + 1


def contract_label(value, trump, coinche=0):
    """Contrat *résolu* -> étiquette. `value` vaut 250 pour un capot : le
    reconvertir en action d'annonce déborde de la plage des valeurs (80-160)."""
    base = f"CAPOT{SUIT_SYM[trump]}" if value >= 250 else f"{value}{SUIT_SYM[trump]}"
    return base + ("x" * min(int(coinche), 2))


def action_label(a):
    if a == 0:
        return "Passe"
    if 1 <= a <= 36:
        v, su = divmod(a - 1, 4)
        return f"{80 + 10 * v}{SUIT_SYM[su]}"
    if 37 <= a <= 40:
        return f"CAPOT{SUIT_SYM[a - 37]}"
    return {41: "Coinche", 42: "Surcoinche"}.get(a, str(a))


# --------------------------------------------------------------------------- mondes

def uniform_world(hand, rng):
    rest = [c for c in range(32) if c not in hand]
    rng.shuffle(rest)
    others = [s for s in range(4) if s != SEAT]
    w = [None] * 4
    w[SEAT] = sorted(hand)
    for j, p in enumerate(others):
        w[p] = sorted(rest[j * 8:(j + 1) * 8])
    return w


def sample_worlds(hand, dealer, prior, n, rng, verbose=True):
    """Retourne (mondes, source). Sidecar GPU > playgen CPU > uniforme."""
    seed = uniform_world(hand, rng)
    pairs = []
    e = colver.Env.deal_with_hands(dealer, seed)
    for a in prior:
        pairs.append((int(e.current_player()), int(a)))
        e.step(a)

    try:
        from colver.web import playgen_gpu
    except Exception:
        playgen_gpu = None

    if playgen_gpu is not None and playgen_gpu.enabled():
        # Le sidecar plafonne à `max_worlds` par requête (512 sur ce modèle) :
        # au-delà il faut découper, sinon la réponse revient tronquée sans le dire.
        t0 = time.monotonic()
        deals = []
        while len(deals) < n:
            chunk = playgen_gpu.auction_deals(
                dealer, seed, pairs, SEAT, min(GPU_CHUNK, n - len(deals)), 1.0)
            if not chunk:
                break
            deals.extend(chunk)
        if len(deals) >= n:
            if verbose:
                print(f"[mondes] sidecar GPU : {len(deals)} mondes en "
                      f"{time.monotonic() - t0:.1f}s", file=sys.stderr)
            return deals[:n], "playgen (sidecar GPU)"
        print(f"[mondes] sidecar configuré mais n'a rendu que {len(deals)}/{n} "
              "— repli", file=sys.stderr)

    if os.path.exists(PLAYGEN_MODEL):
        try:
            t0 = time.monotonic()
            analyst = colver.Analyst.replay(PLAYGEN_MODEL, dealer, seed,
                                            [int(a) for a in prior], SEAT)
            deals = analyst.auction_deals(e, n, 1.0)
            if deals:
                if verbose:
                    print(f"[mondes] playgen CPU : {len(deals)} mondes en "
                          f"{time.monotonic() - t0:.1f}s", file=sys.stderr)
                return deals, "playgen (CPU)"
        except Exception as exc:
            print(f"[mondes] playgen CPU indisponible ({exc}) — repli", file=sys.stderr)

    print("[mondes] REPLI UNIFORME : les mondes ignorent ce que l'enchère "
          "révèle, les chiffres sont dégradés", file=sys.stderr)
    return [uniform_world(hand, rng) for _ in range(n)], "uniforme (dégradé)"


# --------------------------------------------------------------------- déroulements

def rollout(env, dealer, hands, prior, forced):
    """Rejoue le préfixe, force `forced`, continue l'enchère au NN, joue la donne.

    Retourne le score marqué des deux camps + le contrat final.
    """
    env.redeal_with_hands(dealer, hands)
    for a in prior:
        env.step(int(a))
    env.step(int(forced))

    safety = 0
    while env.phase() == 0 and not env.is_terminal() and safety < 50:
        env.step(int(env.bid_a_dd()))
        safety += 1

    contract = env.get_contract()
    if not contract or not contract.get("value"):
        return {"void": True, "scores": (0.0, 0.0)}

    while not env.is_terminal():
        env.step(int(env.action_dmc_with_stats()["best_action"]))

    r = env.rewards()
    taker = contract["team"]
    return {
        "void": False,
        "scores": (float(r[0]), float(r[1])),
        "taker": taker,
        "value": contract["value"],
        "trump": contract["trump"],
        "coinche": contract["coinche"],
        "achieved": float(r[taker]) > 0,
    }


def stderr_of(xs):
    return statistics.stdev(xs) / len(xs) ** 0.5 if len(xs) > 1 else 0.0


# ---------------------------------------------------------------------------- main

def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--hand", required=True, help='8 cartes, p.ex. "AS TS KS QS JS 9S AH TH"')
    ap.add_argument("--prior", default="", help='préfixe d\'enchère, p.ex. "P,100H,P"')
    ap.add_argument("--worlds", type=int, default=200, help="mondes (défaut 200)")
    ap.add_argument("--top", type=int, default=4, help="candidates issues du top-Q de v6")
    ap.add_argument("--add", default="", help='candidates forcées en plus, p.ex. "CAPOTD,160D"')
    ap.add_argument("--bid-model", default=BID_MODEL)
    ap.add_argument("--canonical", action="store_true",
                    help="réseau entraîné sur l'ordre canonique des couleurs (v7+). Indétectable depuis le fichier — même taille qu'un réseau physique — et l'oublier rend une annonce légale dans la mauvaise couleur, sans erreur.")
    ap.add_argument("--play-model", default=PLAY_MODEL)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--json", action="store_true", help="sortie JSON sur stdout")
    args = ap.parse_args()

    hand = parse_hand(args.hand)
    prior = [parse_action(t) for t in args.prior.replace(",", " ").split()] if args.prior else []
    rng = random.Random(args.seed)
    verbose = not args.json

    # Le donneur est choisi pour que ce soit à nous de parler après le préfixe.
    dealer = (SEAT - 1 - len(prior)) % 4

    env = colver.Env.deal_with_hands(dealer, uniform_world(hand, rng))
    env.load_bid_model(args.bid_model, None, args.canonical)
    env.load_dmc_model(args.play_model)

    # Position de décision : Q de v6 et actions légales.
    env.redeal_with_hands(dealer, uniform_world(hand, rng))
    for a in prior:
        env.step(int(a))
    if env.current_player() != SEAT:
        raise SystemExit(f"préfixe incohérent : c'est au siège {env.current_player()} de parler")
    q = dict(env.action_bid_nn()["q_values"])
    ranked = sorted(q.items(), key=lambda kv: -kv[1])

    cands = [a for a, _ in ranked[:args.top]]
    best_capot = max((a for a in q if 37 <= a <= 40), key=lambda a: q[a], default=None)
    if best_capot is not None and best_capot not in cands:
        cands.append(best_capot)          # l'action morte, toujours dans le tableau
    if 0 in q and 0 not in cands:
        cands.append(0)                   # Passe, référence basse
    for tok in args.add.replace(",", " ").split():
        a = parse_action(tok)
        if a not in q:
            print(f"[candidates] {tok} n'est pas légale ici — ignorée", file=sys.stderr)
        elif a not in cands:
            cands.append(a)

    worlds, source = sample_worlds(hand, dealer, prior, args.worlds, rng, verbose)
    bad = sum(1 for w in worlds if sorted(w[SEAT]) != hand)
    if bad:
        raise SystemExit(f"{bad} mondes ne contiennent pas la main du siège — "
                         "échantillonneur incohérent, on s'arrête")

    if verbose:
        print(f"\nMain   {' '.join(card_name(c) for c in hand)}"
              f"    siège {SEAT} (Sud), donneur {dealer}")
        if prior:
            print(f"Avant  {' '.join(action_label(a) for a in prior)}")
        print(f"Mondes {len(worlds)} — {source}")
        print(f"Candidates : {', '.join(action_label(a) for a in cands)}\n")

    team = SEAT % 2  # 0 = NS
    results = {}
    t0 = time.monotonic()
    for ci, cand in enumerate(cands):
        per_world = []
        took = made = void = 0
        finals = {}
        for wi, w in enumerate(worlds):
            r = rollout(env, dealer, w, prior, cand)
            per_world.append(r["scores"][team] - r["scores"][1 - team])
            if r["void"]:
                void += 1
                continue
            if r["taker"] == team:
                took += 1
                if r["achieved"]:
                    made += 1
            k = contract_label(r["value"], r["trump"], r["coinche"])
            finals[k] = finals.get(k, 0) + 1
            if verbose and (wi + 1) % 50 == 0:
                print(f"\r  {action_label(cand):>8s}  {wi + 1}/{len(worlds)}",
                      end="", file=sys.stderr)
        results[cand] = {
            "action": action_label(cand), "q": q[cand], "diffs": per_world,
            "took": took, "made": made, "void": void, "finals": finals,
        }
        if verbose:
            print(f"\r  {action_label(cand):>8s}  {len(worlds)}/{len(worlds)}  "
                  f"({ci + 1}/{len(cands)})", file=sys.stderr)

    ref = cands[0]
    ref_diffs = results[ref]["diffs"]

    rows = []
    for cand in cands:
        r = results[cand]
        d = r["diffs"]
        paired = [x - y for x, y in zip(d, ref_diffs, strict=True)]
        rows.append({
            "action": r["action"], "q": r["q"],
            "mean": statistics.fmean(d), "se": stderr_of(d),
            "median": statistics.median(d),
            "vs_ref": statistics.fmean(paired), "vs_ref_se": stderr_of(paired),
            "took_pct": 100 * r["took"] / len(worlds),
            "made_pct": 100 * r["made"] / r["took"] if r["took"] else None,
            "void_pct": 100 * r["void"] / len(worlds),
            "finals": dict(sorted(r["finals"].items(), key=lambda kv: -kv[1])[:3]),
        })

    if args.json:
        print(json.dumps({"hand": [card_name(c) for c in hand], "prior": prior,
                          "worlds": len(worlds), "source": source,
                          "rows": rows}, ensure_ascii=False, indent=2))
        return

    print(f"\n{'annonce':>9s} {'Q v6':>7s} {'écart N-S':>12s} {'médiane':>8s} "
          f"{'vs réf':>13s} {'prend':>7s} {'réussi':>7s}  contrat final le + fréquent")
    print("-" * 104)
    for row in rows:
        vs = "     (réf)   " if row["action"] == rows[0]["action"] else \
             f"{row['vs_ref']:+7.1f}±{row['vs_ref_se']:4.1f}"
        made = f"{row['made_pct']:5.0f}%" if row["made_pct"] is not None else "    —"
        top = ", ".join(f"{k} {v}" for k, v in list(row["finals"].items())[:2])
        print(f"{row['action']:>9s} {row['q']:7.3f} {row['mean']:+7.1f}±{row['se']:4.1f} "
              f"{row['median']:+8.0f} {vs:>13s} {row['took_pct']:6.0f}% {made:>7s}  {top}")
    print("\nÉcart = points marqués N-S − E-O, moyenne sur les mondes ± erreur type.")
    print("« vs réf » est apparié sur le même pool de mondes. Le gain sur l'erreur "
          "type est réel mais modéré : forcer une annonce différente fait diverger\n"
          "l'enchère, donc une partie de la corrélation entre mondes est perdue. "
          "Il est maximal entre annonces voisines, minimal entre annonces qui\nmènent "
          "à des contrats différents.")
    print(f"Total {len(cands) * len(worlds)} déroulements en "
          f"{time.monotonic() - t0:.0f}s.")


if __name__ == "__main__":
    main()
