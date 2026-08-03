#!/usr/bin/env python3
"""Combien un seul siège peut-il déplacer une donne ? — plan factoriel 2⁴.

Sur une donne dont **l'enchère est figée**, on rejoue le jeu de la carte dans les
**16 configurations** possibles : chacun des quatre sièges est tenu soit par DouDou50,
soit par l'Oracle DD. Les deux joueurs étant déterministes, les 16 résultats sont 16
nombres exacts — aucun bruit d'échantillonnage à l'intérieur d'une donne. On en tire
l'**effet principal** de chaque siège (le remplacer change le score de combien ?) et les
**interactions** (mon partenaire compte-t-il autant que moi ? un adversaire fort
annule-t-il mon avantage ?).

## Pourquoi ce chiffre-là

C'est le plafond de vitesse de tout classement. Si remplacer un siège par un joueur
parfait ne déplace le score que de µ points en moyenne, alors qu'une donne varie de
±316 points pour des raisons qui ne doivent rien à personne (écart-type mesuré,
`deal_margin_scale`), aucun système de notation ne peut extraire plus de µ/316 de signal
par donne. Le reste est de la chance de distribution. C'est la question qui traîne
derrière tout le dossier `docs/classement_et_scoring.md`, et elle se mesure ici
directement au lieu de se déduire d'un modèle.

## Les trois choix qui décident de ce qui est mesuré

1. **L'enchère est figée**, et rejouée à l'identique dans les 16 configurations.
   `oracle_dd` est une méthode de *jeu* ; son enchère resterait celle du TOML. Laisser
   les 16 configurations enchérir librement produirait 16 contrats différents et le
   plan factoriel ne mesurerait plus rien. Ici les deux specs ne diffèrent que par
   **une seule ligne** (`[play] method`), ce qui est exactement la définition d'un
   facteur. Corollaire : ce script mesure l'influence d'un siège **sur le jeu de la
   carte**, pas sur l'enchère.

2. **On ventile par rôle.** Les sièges ne sont pas interchangeables : le preneur pèse
   bien plus qu'un défenseur. Un effet moyenné sur les quatre sièges serait une moyenne
   de choses incomparables.

3. **L'Oracle n'est pas une borne supérieure.** Le DD suppose que *la défense joue
   parfaitement aussi* ; contre un adversaire imparfait, un joueur exploitant peut faire
   mieux. Le script mesure donc, et affiche, la fréquence à laquelle une configuration
   bat le tout-Oracle. Si elle est non négligeable, tout raisonnement qui traite
   l'Oracle comme un plafond est faux — y compris la version « bornes » de R4.

## Deux détails de reproductibilité

- **Le départage entre cartes DD-équivalentes compte.** 57,8 % des positions ont
  plusieurs cartes optimales, et dans une configuration mixte le choix de l'Oracle
  change ce que voient les DouDou d'en face. `--tiebreak` est donc un paramètre
  enregistré, pas un détail (défaut `order`, celui de la production).
- **Les donnes sont tirées au hasard, et c'est ici la bonne façon de faire.** La règle
  du dépôt (« mesurer sur des donnes réellement jouées ») vise les *positions* de jeu,
  qui dépendent de qui a joué avant. Une main, elle, est uniforme par les règles : un
  paquet battu n'a pas de distribution « réaliste » différente. Ce qui doit être
  réaliste est le **contrat**, et il l'est puisqu'il sort du bidder de référence (v6)
  aux quatre sièges.

    uv run python scripts/analysis/seat_influence.py --deals 400 --jobs 8
    uv run python scripts/analysis/seat_influence.py --from data/analysis/seat_influence/<f>.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import runlog  # noqa: E402

BID_MODEL = "models/bid_v6_isdd.bin"
PLAY_MODEL = "models/dmc_50.bin"

N_SEATS = 4
N_CONF = 1 << N_SEATS  # bit s = l'Oracle tient le siège s
ALL_DD, ALL_OR = 0, N_CONF - 1

# Les deux specs ne diffèrent que par `[play] method`. Le bloc `[bid]` est identique et
# n'est **jamais consulté** — l'enchère est rejouée depuis les actions enregistrées —
# mais le garder identique des deux côtés est ce qui fait qu'il n'y a qu'un facteur.
_BID = f"""
[bid]
strategy = "nn"
model = "{BID_MODEL}"
hidden = 512
"""

SPEC_DD = _BID + f"""
[play]
method = "dmc"
model = "{PLAY_MODEL}"
residual = true
"""


def spec_or(tiebreak: str) -> str:
    return _BID + f"""
[play]
method = "oracle_dd"
tiebreak = "{tiebreak}"
"""


# ---------------------------------------------------------------------------
# Un travailleur = un jeu d'agents (8 : DouDou et Oracle à chacun des 4 sièges),
# construits une fois. DouDou50 et l'Oracle sont sans état entre deux donnes, donc
# les réutiliser d'une configuration à l'autre est sûr — `init_deal` est appelé de
# toute façon, au cas où un jour ils en auraient un.
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(tiebreak: str, seed: int) -> None:
    import colver

    _W["colver"] = colver
    _W["dd"] = [colver.Agent(SPEC_DD, s, seed + s) for s in range(N_SEATS)]
    _W["or"] = [colver.Agent(spec_or(tiebreak), s, seed + 100 + s) for s in range(N_SEATS)]


def _auction(colver, dealer, hands, bidders):
    """Joue l'enchère et rend (actions, contrat, siège preneur) — ou None si passée."""
    env = colver.Env.deal_with_hands(dealer, hands)
    for a in bidders:
        a.init_deal(env)
    actions, taker = [], None
    while env.phase() == 0 and not env.is_terminal():
        seat = env.current_player()
        act = int(bidders[seat].decide(env)["action"])
        if 1 <= act <= 40:  # une vraie annonce (pas passe / coinche / surcoinche)
            taker = seat
        for a in bidders:
            a.observe(env, act)
        env.step(act)
        actions.append(act)
    contract = env.get_contract()
    if not contract.get("value") or taker is None:
        return None
    return actions, contract, taker


def _play(colver, dealer, hands, bid_actions, occupants):
    """Rejoue l'enchère figée puis déroule les 8 plis. Rend (marge, points cartes N-S)."""
    env = colver.Env.deal_with_hands(dealer, hands)
    for p in occupants:
        p.init_deal(env)
    for act in bid_actions:
        for p in occupants:
            p.observe(env, act)
        env.step(act)
    while not env.is_terminal():
        seat = env.current_player()
        act = int(occupants[seat].decide(env)["action"])
        for p in occupants:
            p.observe(env, act)
        env.step(act)
    rw = env.rewards()
    pts = env.get_points()
    return float(rw[0]) - float(rw[1]), (int(pts[0]), int(pts[1]))


def _dd_value_ns(colver, dealer, hands, bid_actions):
    """Valeur DD (points cartes N-S) à l'entame — le contrôle du tout-Oracle."""
    env = colver.Env.deal_with_hands(dealer, hands)
    for act in bid_actions:
        env.step(act)
    scores = env.solve_scores()["scores"]
    ns_to_play = env.current_player() % 2 == 0
    vals = [v for _, v in scores]
    return max(vals) if ns_to_play else min(vals)


def sweep(seed: int, check: bool = False) -> dict | None:
    colver = _W["colver"]
    rng = random.Random(seed)
    deck = list(range(32))
    rng.shuffle(deck)
    hands = [sorted(deck[i * 8:(i + 1) * 8]) for i in range(N_SEATS)]
    dealer = rng.randrange(N_SEATS)

    auc = _auction(colver, dealer, hands, _W["dd"])
    if auc is None:
        return None
    bid_actions, contract, taker = auc

    margins, card_margins, pts_ns = [], [], []
    for mask in range(N_CONF):
        occ = [_W["or"][s] if (mask >> s) & 1 else _W["dd"][s] for s in range(N_SEATS)]
        m, (ns, ew) = _play(colver, dealer, hands, bid_actions, occ)
        margins.append(m)
        card_margins.append(ns - ew)
        pts_ns.append(ns)

    rec = {
        "seed": seed, "dealer": dealer, "taker": taker,
        "value": int(contract["value"]), "trump": int(contract["trump"]),
        "team": int(contract["team"]), "coinche": int(contract["coinche"]),
        "margins": margins,
        # Les mêmes 16 configurations en **écart de points cartes** (N-S − E-O). C'est la
        # même partie jouée, lue sans le barème : comparer les deux échelles isole ce que
        # la marche du contrat ajoute à elle seule.
        "card_margins": card_margins,
    }
    if check:
        # Invariant fort sur toute la machinerie de rejeu : quatre Oracles réalisent
        # exactement la valeur DD de la position d'entame.
        rec["dd_ns"] = _dd_value_ns(colver, dealer, hands, bid_actions)
        rec["all_or_ns"] = pts_ns[ALL_OR]
    return rec


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

ROLES = ("preneur", "partenaire", "défenseur")
MARGIN_SD = 316.0  # écart-type d'une marge de donne (deal_margin_scale, 2 999 donnes)


def _role(seat: int, taker: int) -> str:
    if seat == taker:
        return "preneur"
    return "partenaire" if seat == (taker ^ 2) else "défenseur"


def _gain(margin: float, seat: int) -> float:
    """La marge est N-S − E-O ; on la relit du côté de l'équipe du siège."""
    return margin if seat % 2 == 0 else -margin


def _mean_sd(xs):
    if not xs:
        return 0.0, 0.0, 0.0
    m = statistics.fmean(xs)
    sd = statistics.pstdev(xs) if len(xs) > 1 else 0.0
    return m, sd, sd / len(xs) ** 0.5 if xs else 0.0


def report(recs: list[dict], tiebreak: str) -> dict:
    n = len(recs)
    print(f"\n{n} donnes notables · départage Oracle = {tiebreak!r}\n")

    # --- contrôle d'exactitude -------------------------------------------------
    checked = [r for r in recs if "dd_ns" in r]
    if checked:
        bad = [r for r in checked if r["dd_ns"] != r["all_or_ns"]]
        print(f"Contrôle · quatre Oracles réalisent la valeur DD : "
              f"{len(checked) - len(bad)}/{len(checked)}"
              + (f"  ⚠️ {len(bad)} écarts" if bad else " ✓"))

    # --- repères ---------------------------------------------------------------
    dd = [r["margins"][ALL_DD] for r in recs]
    orc = [r["margins"][ALL_OR] for r in recs]
    print("\n── Les deux repères (marge N-S − E-O) " + "─" * 36)
    print(f"  tout-DouDou : {statistics.fmean(dd):+8.1f}   (le datum de R4)")
    print(f"  tout-Oracle : {statistics.fmean(orc):+8.1f}   (jeu parfait des deux côtés)")

    # Marge de manœuvre : de combien la donne bouge selon qui la joue.
    spreads = [max(r["margins"]) - min(r["margins"]) for r in recs]
    figees = sum(1 for s in spreads if s == 0)
    print("\n── Marge de manœuvre par donne (max − min sur les 16) " + "─" * 19)
    print(f"  moyenne {statistics.fmean(spreads):7.1f}   médiane {statistics.median(spreads):7.1f}"
          f"   p90 {statistics.quantiles(spreads, n=10)[8]:7.1f}")
    print(f"  donnes où les 16 configurations donnent le même score : "
          f"{figees} ({100*figees/n:.1f} %)")

    # --- effet principal par siège, ventilé par rôle ---------------------------
    # Effet principal = moyenne des 8 différences appariées « ce siège passe de
    # DouDou à Oracle, les trois autres inchangés », relue du côté de son équipe.
    by_role: dict[str, list[float]] = {r: [] for r in ROLES}
    deltas_all: list[float] = []
    for r in recs:
        m = r["margins"]
        for s in range(N_SEATS):
            bit = 1 << s
            d = statistics.fmean([_gain(m[k | bit] - m[k], s)
                                  for k in range(N_CONF) if not k & bit])
            by_role[_role(s, r["taker"])].append(d)
            deltas_all.append(d)

    print("\n── Effet principal d'un siège (DouDou → Oracle), pour son équipe " + "─" * 8)
    print(f"  {'rôle':<12} {'n':>6} {'effet':>9} {'± err':>8} {'écart-type':>11} "
          f"{'>0':>7} {'p10':>7} {'p90':>7}")
    for role in ROLES:
        xs = by_role[role]
        mean, sd, err = _mean_sd(xs)
        pos = 100 * sum(1 for x in xs if x > 0) / len(xs) if xs else 0
        q = statistics.quantiles(xs, n=10) if len(xs) > 9 else [float("nan")] * 9
        print(f"  {role:<12} {len(xs):>6} {mean:>+9.1f} {err:>8.1f} {sd:>11.1f} "
              f"{pos:>6.0f} % {q[0]:>+7.0f} {q[8]:>+7.0f}")
    mean_all, sd_all, err_all = _mean_sd(deltas_all)
    print(f"  {'tous':<12} {len(deltas_all):>6} {mean_all:>+9.1f} {err_all:>8.1f} "
          f"{sd_all:>11.1f}")

    # --- interactions ----------------------------------------------------------
    # L'effet d'un siège dépend-il de qui l'entoure ? Deux découpes : le partenaire
    # (1 bit) et les adversaires (2 bits).
    part: dict[int, list[float]] = {0: [], 1: []}
    opp: dict[int, list[float]] = {0: [], 1: [], 2: []}
    for r in recs:
        m = r["margins"]
        for s in range(N_SEATS):
            bit, pbit = 1 << s, 1 << (s ^ 2)
            obits = [1 << o for o in range(N_SEATS) if o % 2 != s % 2]
            for k in range(N_CONF):
                if k & bit:
                    continue
                d = _gain(m[k | bit] - m[k], s)
                part[1 if k & pbit else 0].append(d)
                opp[sum(1 for b in obits if k & b)].append(d)

    print("\n── Interactions " + "─" * 57)
    print("  effet du siège selon son entourage :")
    for k, lbl in ((0, "partenaire DouDou"), (1, "partenaire Oracle")):
        mean, _, err = _mean_sd(part[k])
        print(f"    {lbl:<22} {mean:>+8.1f} ± {err:.1f}")
    for k in (0, 1, 2):
        mean, _, err = _mean_sd(opp[k])
        print(f"    {k} adversaire(s) Oracle {'':<1} {mean:>+8.1f} ± {err:.1f}")

    # --- la même chose en points cartes ---------------------------------------
    # Contrôle causal : si la complémentarité disparaît sans le barème, c'est la
    # marche au seuil du contrat qui la produit, pas une synergie de jeu.
    card = None
    if all("card_margins" in r for r in recs):
        cpart: dict[int, list[float]] = {0: [], 1: []}
        crole: dict[str, list[float]] = {r: [] for r in ROLES}
        for r in recs:
            cm = r["card_margins"]
            for s in range(N_SEATS):
                bit, pbit = 1 << s, 1 << (s ^ 2)
                ds = [_gain(cm[k | bit] - cm[k], s) for k in range(N_CONF) if not k & bit]
                crole[_role(s, r["taker"])].append(statistics.fmean(ds))
                for k in range(N_CONF):
                    if not k & bit:
                        cpart[1 if k & pbit else 0].append(_gain(cm[k | bit] - cm[k], s))
        print("\n  les mêmes échanges lus en **points cartes** (sans le barème) :")
        for role in ROLES:
            mean, _, err = _mean_sd(crole[role])
            print(f"    effet {role:<12} {mean:>+8.1f} ± {err:.1f}")
        c0, _, e0 = _mean_sd(cpart[0])
        c1, _, e1 = _mean_sd(cpart[1])
        print(f"    partenaire DouDou      {c0:>+8.1f} ± {e0:.1f}")
        print(f"    partenaire Oracle      {c1:>+8.1f} ± {e1:.1f}")
        p0, _, _ = _mean_sd(part[0])
        p1, _, _ = _mean_sd(part[1])
        if p0 > 0 and c0 > 0:
            print(f"    → amplification par un bon partenaire : ×{p1/p0:.2f} au barème, "
                  f"×{c1/c0:.2f} en points cartes")
        card = {"effect": {r: round(_mean_sd(crole[r])[0], 1) for r in ROLES},
                "partner_dd": round(c0, 1), "partner_or": round(c1, 1)}

    # --- traduction en Elo -----------------------------------------------------
    # Le seul chiffre directement comparable aux ancres de `elo.py`. Il se calcule
    # sur la **distribution** des marges, jamais en injectant une marge moyenne dans
    # l'écrasement : `E[s(m)] ≠ s(E[m])`, et le dépôt s'est déjà fait avoir là-dessus.
    def _s(m):
        return 1.0 / (1.0 + 10 ** (-m / MARGIN_SD))

    def _elo(p):
        p = min(max(p, 1e-9), 1 - 1e-9)
        return 400 * math.log10(p / (1 - p))

    duo = [x for r in recs for x in
           (_s(_gain(r["margins"][0b0101], 0)), _s(_gain(r["margins"][0b1010], 1)))]
    solo = [_s(_gain(r["margins"][1 << s], s)) for r in recs for s in range(N_SEATS)]
    p_duo, p_solo = statistics.fmean(duo), statistics.fmean(solo)
    print("\n── Traduction en Elo (échelle de `elo.py`, marge / 316) " + "─" * 18)
    print(f"  2 Oracles contre 2 DouDou : score {p_duo:.4f} → "
          f"{_elo(p_duo):+.0f} Elo (les deux partenaires changent)")
    print(f"  1 Oracle  contre 0        : score {p_solo:.4f} → "
          f"{2*_elo(p_solo):+.0f} Elo (un seul change : ×2 pour la dilution)")
    print("  Les deux estiment le même écart individuel DouDou → jeu parfait.")
    print("  Un écart entre elles est la signature de la complémentarité ci-dessus :")
    print("  le modèle additif d'Elo (équipe = moyenne des deux) ne la représente pas.")

    # --- l'Oracle n'est pas un plafond ----------------------------------------
    # La bonne question est **appariée** : à opposants et partenaire figés, le siège
    # fait-il mieux tenu par l'Oracle ? « Une configuration bat le tout-Oracle »
    # serait trivialement vrai (il suffit que la défense d'en face soit DouDou).
    worse = sum(1 for d in deltas_all if d < -1e-9)
    equal = sum(1 for d in deltas_all if abs(d) <= 1e-9)
    print("\n── L'Oracle est-il un plafond ? (échanges appariés) " + "─" * 22)
    print(f"  échanges où l'Oracle fait **moins bien** que DouDou au même siège : "
          f"{worse}/{len(deltas_all)} ({100*worse/len(deltas_all):.1f} %)")
    print(f"  échanges sans effet : {100*equal/len(deltas_all):.1f} %")
    print("  Le DD suppose une défense parfaite ; contre DouDou, jouer la ligne")
    print("  DD-optimale n'est donc pas la meilleure réponse. Un écart au jeu de")
    print("  l'Oracle n'est pas une erreur — c'est ce qui interdit de s'en servir")
    print("  comme d'une borne, ou comme dénominateur d'un score normalisé.")

    # --- ce que ça dit du classement ------------------------------------------
    mt, sdt, _ = _mean_sd(by_role["preneur"])
    print("\n── Plafond de vitesse d'un classement " + "─" * 35)
    print(f"  effet d'un siège au rôle le plus lourd (preneur) : {mt:+.1f} pts / donne")
    print(f"  bruit d'une donne, hors joueur (écart-type)      : {MARGIN_SD:.0f} pts")
    if mt > 0:
        print(f"  rapport signal / bruit                          : {mt/MARGIN_SD:.3f}")
        print("\n  Donnes pour distinguer ces deux occupants à 2 erreurs-types :")
        print(f"    non apparié — ce que subit l'Elo en prod : {(2*MARGIN_SD/mt)**2:>7.0f}")
        print(f"    apparié     — ce que fait l'arène (duplicate) : {(2*sdt/mt)**2:>4.0f}")
        print(f"  L'appariement vaut un facteur {(MARGIN_SD/sdt)**2:.0f} : c'est la même")
        print("  donne rejouée, donc la chance de distribution s'annule. Un classement")
        print("  en ligne ne peut pas apparier — d'où l'écart entre les deux colonnes.")

    return {
        "n": n,
        "all_dd_margin": round(statistics.fmean(dd), 1),
        "all_or_margin": round(statistics.fmean(orc), 1),
        "spread_mean": round(statistics.fmean(spreads), 1),
        "spread_median": round(statistics.median(spreads), 1),
        "frozen_deals_pct": round(100 * figees / n, 1),
        "effect": {role: round(_mean_sd(by_role[role])[0], 1) for role in ROLES},
        "effect_err": {role: round(_mean_sd(by_role[role])[2], 1) for role in ROLES},
        "effect_sd": {role: round(_mean_sd(by_role[role])[1], 1) for role in ROLES},
        "effect_partner_dd": round(_mean_sd(part[0])[0], 1),
        "effect_partner_or": round(_mean_sd(part[1])[0], 1),
        "effect_by_opp_oracles": {k: round(_mean_sd(opp[k])[0], 1) for k in opp},
        "oracle_worse_pct": round(100 * worse / len(deltas_all), 1),
        "no_effect_pct": round(100 * equal / len(deltas_all), 1),
        "margin_sd_reference": MARGIN_SD,
        "card_points": card,
        "elo_2v2": round(_elo(p_duo)),
        "elo_1v0_x2": round(2 * _elo(p_solo)),
        "deals_to_separate_unpaired": round((2 * MARGIN_SD / mt) ** 2) if mt > 0 else None,
    }


# ---------------------------------------------------------------------------


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=400)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--tiebreak", default="order",
                    choices=["order", "lowest", "highest", "cheapest", "dearest"])
    ap.add_argument("--check-every", type=int, default=25,
                    help="une donne sur N vérifie que le tout-Oracle réalise la valeur DD")
    ap.add_argument("--tag", default="default")
    ap.add_argument("--no-log", action="store_true")
    ap.add_argument("--from", dest="from_file",
                    help="re-lire un run précédent au lieu de recalculer")
    args = ap.parse_args()

    if args.from_file:
        blob = json.loads(Path(args.from_file).read_text())
        report(blob["payload"]["deals"], blob["params"].get("tiebreak", "?"))
        return 0

    seeds = [args.seed * 1_000_003 + i for i in range(args.deals)]
    checks = {s for i, s in enumerate(seeds) if args.check_every and i % args.check_every == 0}

    recs: list[dict] = []
    with runlog.Timer() as timer:
        if args.jobs > 1:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(args.jobs, initializer=_init_worker,
                          initargs=(args.tiebreak, args.seed)) as pool:
                for i, rec in enumerate(pool.imap_unordered(
                        _sweep_star, [(s, s in checks) for s in seeds], chunksize=4)):
                    if rec is not None:
                        recs.append(rec)
                    if (i + 1) % 50 == 0:
                        print(f"  {i+1}/{len(seeds)} donnes  "
                              f"({timer.elapsed/(i+1)*1000:.0f} ms/donne)", file=sys.stderr)
        else:
            _init_worker(args.tiebreak, args.seed)
            for i, s in enumerate(seeds):
                rec = sweep(s, check=s in checks)
                if rec is not None:
                    recs.append(rec)
                if (i + 1) % 25 == 0:
                    print(f"  {i+1}/{len(seeds)} donnes  "
                          f"({timer.elapsed/(i+1)*1000:.0f} ms/donne)", file=sys.stderr)

    if not recs:
        print("aucune donne notable", file=sys.stderr)
        return 1
    recs.sort(key=lambda r: r["seed"])
    summary = report(recs, args.tiebreak)

    if not args.no_log:
        runlog.save(
            script="seat_influence",
            tag=args.tag,
            params={"deals": args.deals, "seed": args.seed, "jobs": args.jobs,
                    "tiebreak": args.tiebreak, "bidder": "bid_v6 (figé, rejoué)",
                    "occupants": "DouDou50 vs oracle_dd", "configs": N_CONF},
            summary=summary,
            payload={"deals": recs},
            models=[BID_MODEL, PLAY_MODEL],
            took_s=timer.s,
        )
    return 0


def _sweep_star(arg):
    return sweep(*arg)


if __name__ == "__main__":
    raise SystemExit(main())
