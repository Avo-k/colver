#!/usr/bin/env python3
"""Ce que « une erreur » veut dire dans Rejouer, et ce que coûterait de le dire mieux.

Quatre mesures, toutes sur des donnes **jouées** (enchère par bid v6, jeu par
DouDou50) et non sur des positions tirées au hasard :

  scale     l'échelle. `analysis.py` note chaque carte par l'écart DD en points
            cartes ; le score de donne est une fonction en escalier de ces
            points. Combien de décisions les deux échelles classent-elles
            différemment, et dans quel sens ?
  example   une donne complète portant les deux désaccords, à lire.
  variation ce que coûte de dérouler une ligne DD jusqu'au 8e pli, par stade.
  rollout   un déroulé DouDou50 contre un solve DD à la même position.
  errors    combien de décisions coûtent réellement quelque chose, par donne.

Toutes écrivent dans le journal (`runlog`) sauf `example`, qui ne produit pas
d'agrégat. `--no-log` pour un essai jetable.

    uv run python scripts/analysis/replay_error_scale.py scale --deals 60
    uv run python scripts/analysis/replay_error_scale.py example --tries 400
    uv run python scripts/analysis/replay_error_scale.py rollout --deals 8
    uv run python scripts/analysis/replay_error_scale.py errors --deals 40

**Le barème est réimplémenté ici**, alors que `scoring.rs` le porte déjà. C'est
une entorse assumée à « une seule implémentation du barème », et elle n'est
tenable que parce que `scale` le **valide contre `env.rewards()` sur chaque
donne** et refuse de conclure si un seul écart apparaît. Du code de production
devrait passer par un binding, pas par cette copie — c'est précisément la
recommandation de la fiche que ce script alimente.
"""

import argparse
import os
import random
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import colver  # noqa: E402
import runlog  # noqa: E402

TOTAL_PTS = 162
CAPOT_PTS = 252

RANKS = ["7", "8", "9", "V", "D", "R", "10", "A"]
SUITS = ["♠", "♥", "♦", "♣"]
SEATS = ["Nord", "Est", "Sud", "Ouest"]
TEAMS = ["N-S", "E-O"]

# Les seuils de `analysis.CATEGORIES`, en points cartes.
CATEGORIES = [(0, "parfait"), (4, "bon"), (14, "imprecision"),
              (29, "erreur"), (10 ** 9, "faute")]


# ── barème (copie validée de scoring.rs) ──

def deal_score_from_card_points(contract, card_pts, belote, capot_realise):
    """Port de `scoring.rs::deal_score_from_card_points`. Rend [N-S, E-O]."""
    taker = contract["team"]
    defense = 1 - taker
    total_belote = belote[0] + belote[1]
    value = contract["value"]
    coinche = contract["coinche"]
    scores = [0, 0]

    if value == 250:  # capot annoncé
        if capot_realise:
            if coinche == 0:
                scores[taker] = card_pts[taker] + value + belote[taker]
                scores[defense] = belote[defense]
            else:
                scores[taker] = CAPOT_PTS + value * (coinche + 1) + total_belote
        else:
            scores[defense] = TOTAL_PTS + value * (coinche + 1) + total_belote
        return scores

    if card_pts[taker] + belote[taker] >= value:  # contrat tenu
        if coinche == 0:
            scores[taker] = card_pts[taker] + value + belote[taker]
            scores[defense] = card_pts[defense] + belote[defense]
        else:
            base = CAPOT_PTS if capot_realise else TOTAL_PTS
            scores[taker] = base + value * (coinche + 1) + total_belote
    else:  # chute
        scores[defense] = TOTAL_PTS + value * (coinche + 1) + total_belote
    return scores


def world_belote(initial_hands, trump):
    """Belote finale par camp : acquise dès qu'une main tient Dame **et** Roi
    d'atout. `state.belote` ne compte que ce qui a déjà été joué, donc il
    sous-estime en cours de donne — même raisonnement que `is_dd::world_belote`.
    """
    need = {trump * 8 + 4, trump * 8 + 5}
    return [20 if any(need <= set(initial_hands[p]) for p in (t, t + 2)) else 0
            for t in (0, 1)]


def ns_delta(ns_pts, contract, belote, ns_tricks_so_far):
    """Écart de score signé N-S − E-O pour une valeur DD en points cartes N-S."""
    if ns_pts == CAPOT_PTS or (ns_pts == 0 and ns_tricks_so_far == 0):
        total = CAPOT_PTS
    else:
        total = TOTAL_PTS
    card = [ns_pts, total - ns_pts]
    s = deal_score_from_card_points(
        contract, card, belote, card[contract["team"]] == CAPOT_PTS)
    return s[0] - s[1]


def categorize(cost):
    for threshold, label in CATEGORIES:
        if cost <= threshold:
            return label
    return "faute"


# ── donnes ──

def _models():
    return str(colver.bid_model_path()), str(colver.model_path())


def make_env(bid_path, dmc_path, dealer, hands):
    env = colver.Env.deal_with_hands(dealer, hands)
    env.load_bid_model(bid_path)
    env.load_dmc_model(dmc_path)
    return env


def play_one_deal(bid_path, dmc_path, rng, weak_seat=None):
    """Une donne enchérie et jouée. `weak_seat` joue à l'heuristique.

    Rend `(dealer, hands, actions, env_final)`. Une enchère morte sur quatre
    passes est rejouée : il n'y a rien à analyser.
    """
    for _ in range(40):
        deck = list(range(32))
        rng.shuffle(deck)
        hands = [deck[i * 8:(i + 1) * 8] for i in range(4)]
        dealer = rng.randrange(4)
        env = make_env(bid_path, dmc_path, dealer, hands)
        actions = []
        while env.phase() == 0 and not env.is_terminal():
            a = int(env.action_bid_nn()["best_action"])
            actions.append(a)
            env.step(a)
        if env.is_terminal() or env.phase() != 1:
            continue
        while not env.is_terminal():
            if weak_seat is not None and int(env.current_player()) == weak_seat:
                a = int(env.action_heuristic_play())
            else:
                a = int(env.action_dmc_with_stats()["best_action"])
            actions.append(a)
            env.step(a)
        return dealer, hands, actions, env
    raise RuntimeError("aucune donne jouable en 40 tirages")


def decisions_of(dealer, hands, actions, want_tables=False):
    """Chaque décision de jeu non forcée, dans les deux échelles."""
    env = colver.Env.deal_with_hands(dealer, hands)
    out = []
    for i, a in enumerate(actions):
        if env.is_terminal():
            break
        if int(env.phase()) == 1:
            legals = list(env.legal_actions())
            if len(legals) > 1:
                player = int(env.current_player())
                team = player % 2
                contract = env.get_contract()
                bel = world_belote(hands, contract["trump"])
                nst = int(env.get_tricks_won()[0])
                sc = {int(c): int(v) for c, v in env.solve_scores()["scores"]}
                dl = {c: ns_delta(v, contract, bel, nst) for c, v in sc.items()}

                best_cp = max(sc.values()) if team == 0 else min(sc.values())
                cost_cp = ((best_cp - sc[int(a)]) if team == 0
                           else (sc[int(a)] - best_cp))
                best_ds = max(dl.values()) if team == 0 else min(dl.values())
                cost_ds = ((best_ds - dl[int(a)]) if team == 0
                           else (dl[int(a)] - best_ds))

                row = {
                    "idx": i, "player": player, "team": team, "played": int(a),
                    "cost_cp": int(cost_cp), "cost_ds": int(cost_ds),
                    "trick": len(env.get_played_cards()) // 4 + 1,
                    "cards_left": len(env.get_hands()[player]),
                }
                if want_tables:
                    row.update({"scores": sc, "deltas": dl,
                                "contract": dict(contract), "belote": bel})
                out.append(row)
        env.step(int(a))
    return out


def check_scoring(hands, final):
    """La copie du barème retrouve-t-elle `rewards()` sur cette donne ?"""
    contract = final.get_contract()
    bel = world_belote(hands, contract["trump"])
    mine = deal_score_from_card_points(
        contract, list(final.get_points()), bel,
        max(final.get_tricks_won()) == 8)
    return tuple(mine) == tuple(int(x) for x in final.rewards())


def med(v):
    v = sorted(v)
    return v[len(v) // 2]


# ── affichage ──

def cname(c):
    return f"{RANKS[c % 8]}{SUITS[c // 8]}"


def hand_str(cards):
    by_suit = {s: [] for s in range(4)}
    for c in sorted(cards, key=lambda c: (c // 8, -(c % 8))):
        by_suit[c // 8].append(RANKS[c % 8])
    return "   ".join(f"{SUITS[s]} " + (" ".join(by_suit[s]) or "—")
                      for s in range(4))


def bid_name(a):
    if a == 0:
        return "passe"
    if a == 41:
        return "COINCHE"
    if a == 42:
        return "SURCOINCHE"
    if 37 <= a <= 40:
        return f"CAPOT{SUITS[a - 37]}"
    return f"{80 + (a - 1) // 4 * 10}{SUITS[(a - 1) % 4]}"


# ── mesures ──

def cmd_scale(args):
    """Les deux échelles sur les mêmes décisions."""
    bid_path, dmc_path = _models()
    rng = random.Random(args.seed)
    rows, checked = [], 0
    with runlog.Timer() as t:
        for d in range(args.deals):
            dealer, hands, actions, final = play_one_deal(bid_path, dmc_path, rng)
            checked += check_scoring(hands, final)
            rows += decisions_of(dealer, hands, actions)
            print(f"\rdonne {d + 1}/{args.deals} — {len(rows)} décisions",
                  end="", flush=True)

    if checked != args.deals:
        print(f"\n\nBARÈME FAUX sur {args.deals - checked} donnes — "
              f"rien à conclure, la copie de scoring.rs a divergé.")
        return
    print(f"\n\nbarème vérifié sur {checked}/{args.deals} donnes")

    n = len(rows)
    zero_cp = sum(1 for r in rows if r["cost_cp"] == 0)
    zero_ds = sum(1 for r in rows if r["cost_ds"] == 0)
    gros_cp = [r for r in rows if r["cost_cp"] > 29]
    faute_gratuite = [r for r in gros_cp if r["cost_ds"] == 0]
    petit_cp = [r for r in rows if r["cost_cp"] <= 4]
    faux_bon = [r for r in petit_cp if r["cost_ds"] > 0]

    cats_cp = {}
    for r in rows:
        lab = categorize(r["cost_cp"])
        cats_cp[lab] = cats_cp.get(lab, 0) + 1

    print(f"\n{n} décisions non forcées sur {args.deals} donnes\n")
    print(f"coût nul en points cartes  : {zero_cp:5d} ({100 * zero_cp / n:.1f} %)")
    print(f"coût nul en score de donne : {zero_ds:5d} ({100 * zero_ds / n:.1f} %)")
    print(f"\n« faute » (>29 pts cartes) qui ne coûte rien au score : "
          f"{len(faute_gratuite)}/{len(gros_cp)}")
    print(f"« bon coup » (≤4 pts cartes) qui coûte au score       : "
          f"{len(faux_bon)}/{len(petit_cp)}")
    if faux_bon:
        worst = sorted(faux_bon, key=lambda r: -r["cost_ds"])[:8]
        print("   pires : " + ", ".join(
            f"{r['cost_cp']} pts → {r['cost_ds']} score" for r in worst))
    print(f"\ncatégories en points cartes : {cats_cp}")

    if not args.no_log:
        runlog.save(
            "replay_error_scale", args.tag or "scale",
            {"deals": args.deals, "seed": args.seed,
             "bid_model": bid_path, "play_model": dmc_path},
            {"decisions": n, "scoring_checked": checked,
             "zero_cost_card_points": zero_cp, "zero_cost_deal_score": zero_ds,
             "blunders_card_points": len(gros_cp),
             "blunders_free_in_deal_score": len(faute_gratuite),
             "good_card_points": len(petit_cp),
             "good_but_costly_in_deal_score": len(faux_bon),
             "worst_disagreements": [[r["cost_cp"], r["cost_ds"]]
                                     for r in sorted(faux_bon,
                                                     key=lambda r: -r["cost_ds"])[:20]],
             "categories_card_points": cats_cp},
            payload={"rows": rows},
            models=[bid_path, dmc_path], took_s=t.s)


def cmd_example(args):
    """Une donne complète portant les deux désaccords, déroulée et commentée."""
    bid_path, dmc_path = _models()
    rng = random.Random(args.seed)

    def wanted(rows):
        fb = [r for r in rows if r["cost_cp"] <= 4 and r["cost_ds"] > 100]
        ff = [r for r in rows if r["cost_cp"] >= 20 and r["cost_ds"] == 0]
        return fb, ff

    found = None
    for t in range(args.tries):
        dealer, hands, actions, final = play_one_deal(bid_path, dmc_path, rng)
        rows = decisions_of(dealer, hands, actions, want_tables=True)
        fb, ff = wanted(rows)
        hit = {"both": fb and ff, "bon": fb, "faute": ff}[args.want]
        if hit:
            found = (dealer, hands, actions, final, rows, fb, ff)
            print(f"trouvé au tirage {t + 1}\n")
            break
        print(f"\r{t + 1}/{args.tries}", end="", flush=True)
    if found is None:
        print(f"\nrien trouvé en {args.tries} tirages")
        return

    dealer, hands, actions, final, rows, fb, ff = found
    contract = final.get_contract()
    bel = world_belote(hands, contract["trump"])

    print("=" * 78)
    print(f"DONNE — donneur {SEATS[dealer]}")
    for s in range(4):
        print(f"  {SEATS[s]:<6} {hand_str(hands[s])}")

    env = colver.Env.deal_with_hands(dealer, hands)
    auction = []
    for a in actions:
        if int(env.phase()) != 0:
            break
        auction.append(f"{SEATS[int(env.current_player())][0]} {bid_name(int(a))}")
        env.step(int(a))
    print("\nENCHÈRE : " + " · ".join(auction))
    print(f"CONTRAT : {contract['value']}{SUITS[contract['trump']]} par "
          f"{TEAMS[contract['team']]}"
          + (f"  (coinche ×{contract['coinche'] + 1})" if contract["coinche"] else "")
          + (f"  · belote {TEAMS[0] if bel[0] else TEAMS[1]}" if any(bel) else ""))
    pts, real = final.get_points(), final.rewards()
    print(f"RÉSULTAT : {pts[0]}-{pts[1]} points cartes → "
          f"marqué {int(real[0])}-{int(real[1])}\n")

    print("DÉROULÉ (décisions non forcées)")
    print(f"  {'pli':>3} {'siège':<6} {'carte':>5} {'pts cartes':>11} {'score':>8}")
    for r in rows:
        flag = ""
        if r["cost_cp"] <= 4 and r["cost_ds"] > 100:
            flag = "   <<< noté bon, perd la donne"
        elif r["cost_cp"] >= 20 and r["cost_ds"] == 0:
            flag = "   <<< noté faute, gratuit"
        print(f"  {r['trick']:>3} {SEATS[r['player']]:<6} "
              f"{cname(r['played']):>5} {r['cost_cp']:>11} {r['cost_ds']:>8}{flag}")

    for label, group in (("NOTÉ BON, PERD LA DONNE", fb),
                         ("NOTÉ FAUTE, NE COÛTE RIEN", ff)):
        for r in group:
            c, b = r["contract"], r["belote"]
            taker = c["team"]
            print(f"\n{'=' * 78}\n{label} — pli {r['trick']}, "
                  f"{SEATS[r['player']]} joue {cname(r['played'])}")
            print(f"  points cartes −{r['cost_cp']}   "
                  f"score de donne −{r['cost_ds']}")
            print(f"     {'carte':>6} {'pts cartes preneur':>20} {'':>7} "
                  f"{'score N-S − E-O':>16}")
            order = sorted(r["scores"], key=lambda k: (-r["deltas"][k]
                                                       if r["team"] == 0
                                                       else r["deltas"][k]))
            for card in order:
                ns = r["scores"][card]
                total = CAPOT_PTS if ns in (0, CAPOT_PTS) else TOTAL_PTS
                tk = ns if taker == 0 else total - ns
                ok = "tenu " if tk + b[taker] >= c["value"] else "CHUTE"
                mark = " ← joué" if card == r["played"] else ""
                print(f"     {cname(card):>6} {tk:>14} +bel {b[taker]:<2} "
                      f"{ok:>7} {r['deltas'][card]:>13}{mark}")
            print(f"     seuil du contrat : {c['value']}")


def cmd_variation(args):
    """Dérouler une ligne DD jusqu'au 8e pli : combien de temps, par stade."""
    bid_path, dmc_path = _models()
    rng = random.Random(args.seed)
    by_left, per_deal = {}, []
    with runlog.Timer() as t:
        for d in range(args.deals):
            dealer, hands, actions, _ = play_one_deal(bid_path, dmc_path, rng)
            env = colver.Env.deal_with_hands(dealer, hands)
            total = 0.0
            for a in actions:
                if env.is_terminal():
                    break
                if int(env.phase()) == 1 and len(list(env.legal_actions())) > 1:
                    left = len(env.get_hands()[int(env.current_player())])
                    probe = colver.Env.from_cfn(env.to_cfn())
                    probe.step(int(probe.action_oracle_dd()))
                    t0 = time.perf_counter()
                    while not probe.is_terminal():
                        probe.step(int(probe.action_oracle_dd()))
                    dt = (time.perf_counter() - t0) * 1000
                    by_left.setdefault(left, []).append(dt)
                    total += dt
                env.step(int(a))
            per_deal.append(total)
            print(f"\rdonne {d + 1}/{args.deals}", end="", flush=True)

    print("\n\ncoût d'une variante déroulée jusqu'à la fin, par stade")
    print(f"{'cartes en main':>14} {'n':>5} {'médiane':>10} {'p90':>10} {'max':>10}")
    summary = {}
    for k in sorted(by_left, reverse=True):
        v = sorted(by_left[k])
        summary[k] = {"n": len(v), "median_ms": round(med(v), 2),
                      "p90_ms": round(v[int(len(v) * 0.9)], 2),
                      "max_ms": round(v[-1], 2)}
        print(f"{k:>14} {len(v):>5} {med(v):>9.1f}ms "
              f"{v[int(len(v) * 0.9)]:>9.1f}ms {v[-1]:>9.1f}ms")
    print(f"\ntoutes les variantes d'une donne : médiane "
          f"{med(per_deal) / 1000:.2f}s, max {max(per_deal) / 1000:.2f}s")

    if not args.no_log:
        runlog.save(
            "replay_error_scale", args.tag or "variation",
            {"deals": args.deals, "seed": args.seed},
            {"by_cards_left": summary,
             "whole_deal_median_s": round(med(per_deal) / 1000, 3),
             "whole_deal_max_s": round(max(per_deal) / 1000, 3)},
            payload={"per_deal_ms": per_deal},
            models=[bid_path, dmc_path], took_s=t.s)


def cmd_rollout(args):
    """Un déroulé DouDou50 complet contre un solve DD, à la même position."""
    bid_path, dmc_path = _models()
    rng = random.Random(args.seed)
    roll, solve = {}, {}
    with runlog.Timer() as t:
        for d in range(args.deals):
            dealer, hands, actions, _ = play_one_deal(bid_path, dmc_path, rng)
            env = make_env(bid_path, dmc_path, dealer, hands)
            for a in actions:
                if env.is_terminal():
                    break
                if int(env.phase()) == 1 and len(list(env.legal_actions())) > 1:
                    left = len(env.get_hands()[int(env.current_player())])
                    t0 = time.perf_counter()
                    env.solve_scores()
                    solve.setdefault(left, []).append(
                        (time.perf_counter() - t0) * 1000)
                    # Reconstruction de la position hors mesure.
                    probe = colver.Env.from_cfn(env.to_cfn())
                    probe.load_dmc_model(dmc_path)
                    t0 = time.perf_counter()
                    while not probe.is_terminal():
                        probe.step(int(probe.action_dmc_with_stats()["best_action"]))
                    roll.setdefault(left, []).append(
                        (time.perf_counter() - t0) * 1000)
                env.step(int(a))
            print(f"\rdonne {d + 1}/{args.deals}", end="", flush=True)

    print("\n\nun déroulé DouDou50 contre un solve DD, même position")
    print(f"{'cartes en main':>14} {'n':>5} {'DouDou50':>12} "
          f"{'solve DD':>12} {'rapport':>9}")
    summary = {}
    for k in sorted(roll, reverse=True):
        r, s = med(roll[k]), med(solve[k])
        summary[k] = {"n": len(roll[k]), "doudou_ms": round(r, 3),
                      "solve_ms": round(s, 4), "ratio": round(r / s, 2)}
        print(f"{k:>14} {len(roll[k]):>5} {r:>11.2f}ms {s:>11.3f}ms {r / s:>8.1f}×")

    if not args.no_log:
        runlog.save(
            "replay_error_scale", args.tag or "rollout",
            {"deals": args.deals, "seed": args.seed},
            {"by_cards_left": summary},
            models=[bid_path, dmc_path], took_s=t.s)


def cmd_errors(args):
    """Combien de décisions coûtent réellement, par donne et par siège."""
    bid_path, dmc_path = _models()
    out = {}
    with runlog.Timer() as t:
        for label, weak in (("4x_doudou50", None), ("sud_heuristique", 2)):
            rng = random.Random(args.seed)
            per_deal, per_seat, decisions = [], [0] * 4, [0] * 4
            for d in range(args.deals):
                dealer, hands, actions, _ = play_one_deal(
                    bid_path, dmc_path, rng, weak)
                count = 0
                for r in decisions_of(dealer, hands, actions):
                    decisions[r["player"]] += 1
                    if r["cost_ds"] > 0:
                        count += 1
                        per_seat[r["player"]] += 1
                per_deal.append(count)
                print(f"\r{label} — donne {d + 1}/{args.deals}", end="", flush=True)
            hist = {}
            for c in per_deal:
                hist[c] = hist.get(c, 0) + 1
            out[label] = {
                "mean_per_deal": round(sum(per_deal) / args.deals, 2),
                "median_per_deal": med(per_deal), "max_per_deal": max(per_deal),
                "deals_without_error": hist.get(0, 0),
                "per_seat": per_seat, "decisions_per_seat": decisions,
                "histogram": {str(k): hist[k] for k in sorted(hist)},
            }
            print(f"\n{label} — moyenne {out[label]['mean_per_deal']}, "
                  f"médiane {out[label]['median_per_deal']}, "
                  f"max {out[label]['max_per_deal']}, "
                  f"donnes sans erreur {hist.get(0, 0)}/{args.deals}")
            print("  par siège : " + "  ".join(
                f"{SEATS[s]} {per_seat[s]}/{decisions[s]}" for s in range(4)))

    if not args.no_log:
        runlog.save(
            "replay_error_scale", args.tag or "errors",
            {"deals": args.deals, "seed": args.seed},
            out, models=[bid_path, dmc_path], took_s=t.s)


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    for name, fn, default_deals in (("scale", cmd_scale, 60),
                                    ("variation", cmd_variation, 12),
                                    ("rollout", cmd_rollout, 8),
                                    ("errors", cmd_errors, 40)):
        s = sub.add_parser(name, help=fn.__doc__.splitlines()[0])
        s.add_argument("--deals", type=int, default=default_deals)
        s.add_argument("--seed", type=int, default=42)
        s.add_argument("--tag", default=None)
        s.add_argument("--no-log", action="store_true")
        s.set_defaults(func=fn)
    s = sub.add_parser("example", help=cmd_example.__doc__.splitlines()[0])
    s.add_argument("--tries", type=int, default=400)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--want", choices=("both", "bon", "faute"), default="both")
    s.set_defaults(func=cmd_example)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
