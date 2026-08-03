#!/usr/bin/env python3
"""Combien un seul siège peut-il déplacer une donne ? — plan factoriel 2⁴.

Sur une donne dont **l'enchère est figée**, on rejoue le jeu de la carte dans les
**16 configurations** possibles : chacun des quatre sièges est tenu soit par l'occupant
**A** (`--a`, le plus faible), soit par l'occupant **B** (`--b`). Les deux étant
déterministes, les 16 résultats sont 16 nombres exacts — aucun bruit d'échantillonnage à
l'intérieur d'une donne. On en tire l'**effet principal** de chaque siège (le remplacer
change le score de combien ?) et les **interactions** (mon partenaire compte-t-il autant
que moi ? un adversaire fort annule-t-il mon avantage ?).

Deux couples utiles, et ils ne répondent pas à la même question :

- `--a doudou50 --b oracle` (défaut) — l'**enveloppe** : le plus grand écart de niveau
  qui existe dans ce jeu. C'est le bon couple pour dimensionner un classement, parce
  qu'aucun joueur réel ne peut faire mieux.
- `--a doudou35 --b doudou50` — le **régime réaliste** : deux joueurs imparfaits, comme
  deux humains. C'est le bon couple pour savoir si les conclusions tirées de l'enveloppe
  survivent à l'échelle où vivent les gens.

## Pourquoi ce chiffre-là

C'est le plafond de vitesse de tout classement. Si remplacer un siège par un joueur
parfait ne déplace le score que de µ points en moyenne, alors qu'une donne varie de
±316 points pour des raisons qui ne doivent rien à personne (écart-type mesuré,
`deal_margin_scale`), aucun système de notation ne peut extraire plus de µ/316 de signal
par donne. Le reste est de la chance de distribution. C'est la question qui traîne
derrière tout le dossier `docs/classement_et_scoring.md`, et elle se mesure ici
directement au lieu de se déduire d'un modèle.

## Les trois choix qui décident de ce qui est mesuré

1. **L'enchère est figée**, et rejouée à l'identique dans les 16 configurations. Laisser
   les 16 configurations enchérir librement produirait 16 contrats différents et le plan
   factoriel ne mesurerait plus rien. Les deux specs ne diffèrent donc que par leur bloc
   `[play]` — le `[bid]` est le même (v6) des deux côtés, et n'est de toute façon jamais
   consulté. Corollaire : ce script mesure l'influence d'un siège **sur le jeu de la
   carte**, pas sur l'enchère.

2. **On ventile par rôle.** Les sièges ne sont pas interchangeables : le preneur pèse
   bien plus qu'un défenseur. Un effet moyenné sur les quatre sièges serait une moyenne
   de choses incomparables.

3. **L'Oracle n'est pas une borne supérieure.** Le DD suppose que *la défense joue
   parfaitement aussi* ; contre un adversaire imparfait, un joueur exploitant peut faire
   mieux. Le script mesure donc, et affiche, la fréquence à laquelle l'occupant fort fait
   **moins bien** que le faible au même siège, à entourage figé. Si elle est non
   négligeable, tout raisonnement qui traite l'Oracle comme un plafond est faux — y
   compris la version « bornes » de R4. La question doit être posée **appariée** :
   « une configuration bat-elle le tout-Oracle ? » serait trivialement vraie, il suffit
   que la défense d'en face soit faible.

## Deux détails de reproductibilité

- **Le départage entre cartes DD-équivalentes compte.** ~50 % des positions ont plusieurs
  cartes optimales, et dans une configuration mixte le choix de l'Oracle change ce que
  voient les DouDou d'en face. `--tiebreak` est donc un paramètre enregistré, pas un
  détail. Le défaut `order` est celui de la production, et **vaut aujourd'hui
  `cheapest`** : depuis `1edd349` c'est le solveur lui-même qui préfère la carte la moins
  chère, donc les deux ne diffèrent sur aucune décision. Le vrai contraste est `dearest`
  (45,9 % de cartes différentes) ou `highest` (45,3 %).
- **L'extension PyO3 est de la provenance.** Elle porte le code du solveur, elle n'est pas
  dans git, et `uv sync` ne la recompile pas toujours. Son sha256 part donc au journal, et
  le script refuse de tourner si un `.rs` du cœur est plus récent qu'elle — voir
  `_check_extension_is_fresh`, écrit après avoir perdu trois runs de 4 000 donnes.
- **Les donnes sont tirées au hasard, et c'est ici la bonne façon de faire.** La règle
  du dépôt (« mesurer sur des donnes réellement jouées ») vise les *positions* de jeu,
  qui dépendent de qui a joué avant. Une main, elle, est uniforme par les règles : un
  paquet battu n'a pas de distribution « réaliste » différente. Ce qui doit être
  réaliste est le **contrat**, et il l'est puisqu'il sort du bidder de référence (v6)
  aux quatre sièges.

    uv run python scripts/analysis/seat_influence.py --deals 4000 --jobs 8
    uv run python scripts/analysis/seat_influence.py --deals 4000 --jobs 8 \
        --a doudou35 --b doudou50 --tag dd35_vs_dd50
    uv run python scripts/analysis/seat_influence.py --from data/analysis/seat_influence/<f>.json
"""

from __future__ import annotations

import argparse
import json
import math
import random
import statistics
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "scripts" / "analysis"))

import runlog  # noqa: E402

BID_MODEL = "models/bid_v6_isdd.bin"

N_SEATS = 4
N_CONF = 1 << N_SEATS  # bit s = l'occupant **B** tient le siège s
ALL_A, ALL_B = 0, N_CONF - 1

# Les deux occupants ne diffèrent que par le bloc `[play]`. Le bloc `[bid]` est identique
# et n'est **jamais consulté** — l'enchère est rejouée depuis les actions enregistrées —
# mais le garder identique des deux côtés est ce qui fait qu'il n'y a qu'un facteur.
_BID = f"""
[bid]
strategy = "nn"
model = "{BID_MODEL}"
hidden = 512
"""

# A doit être le plus faible des deux, pour que l'effet mesuré soit positif.
OCCUPANTS = {
    # Les deux barreaux bas de l'échelle : des fonctions pures, instantanées, dont
    # personne ne doute qu'elles jouent moins bien qu'un réseau. Elles sont là pour
    # **donner un ordre connu d'avance** — c'est ce qui manque à un couple serré comme
    # DouDou35/DouDou50, dont l'ordre a lui-même dû être mesuré.
    "heuristic": ("Heuristiq", 'method = "heuristic"'),
    "rule": ("Regles", 'method = "rule"'),
    "doudou35": ("DouDou35", 'method = "dmc"\nmodel = "models/dmc_35.bin"'),
    "doudou50": ("DouDou50", 'method = "dmc"\nmodel = "models/dmc_50.bin"\nresidual = true'),
    # Variante de contrôle : le même fichier avec le passage résiduel forcé. `residual`
    # est indétectable depuis le poids (CLAUDE.md), donc c'est le seul réglage libre du
    # couple — et s'il était mal posé, « DouDou35 est plus faible » mesurerait une
    # configuration ratée et non un niveau.
    "doudou35res": ("DD35+res", 'method = "dmc"\nmodel = "models/dmc_35.bin"\nresidual = true'),
    "oracle": ("Oracle", 'method = "oracle_dd"'),
}


# --- l'autre facteur : l'enchère -------------------------------------------------
#
# Le jeu de la carte ne fait qu'une **moitié** d'une donne, et les donnes du site couvrent
# les deux. Une échelle de force mesurée à enchère figée décrit donc la moitié la moins
# décisive du jeu. Mêmes 16 configurations, facteur inversé : le bloc `[bid]` varie, le
# `[play]` est DouDou50 partout.
#
# Différence de nature avec le facteur « carte » : chaque configuration **produit son
# propre contrat**, puisque c'est précisément le travail d'un enchérisseur. On ne peut donc
# pas figer l'enchère — et une donne passée par une configuration marque 0, ce qui est la
# bonne note pour un enchérisseur qui laisse filer une main jouable.
BIDDERS = {
    "b_heuristic": ("BidHeur", 'strategy = "heuristic"'),
    "b_improved": ("BidImpr", 'strategy = "improved"'),
    "b_improved_v2": ("BidImpV2", 'strategy = "improved_v2"'),
    "b_roro": ("BidRoro", 'strategy = "roro"'),
    "b_petit_bide": ("BidPetit", 'strategy = "petit_bide"'),
    "b_moelleux": ("BidMoell", 'strategy = "moelleux"'),
    "b_maxi": ("BidMaxi", 'strategy = "maxi"'),
    "b_v6": ("BidV6", f'strategy = "nn"\nmodel = "{BID_MODEL}"\nhidden = 512'),
}

_PLAY_FIXED = '\n[play]\nmethod = "dmc"\nmodel = "models/dmc_50.bin"\nresidual = true\n'


def is_bidder(name: str) -> bool:
    return name in BIDDERS


def label_of(name: str) -> str:
    return (BIDDERS if is_bidder(name) else OCCUPANTS)[name][0]


def spec_for(name: str, tiebreak: str) -> str:
    if is_bidder(name):
        return "\n[bid]\n" + BIDDERS[name][1] + "\n" + _PLAY_FIXED
    body = OCCUPANTS[name][1]
    if name == "oracle":
        body += f'\ntiebreak = "{tiebreak}"'
    return _BID + "\n[play]\n" + body + "\n"


def model_of(name: str) -> str | None:
    table = BIDDERS if is_bidder(name) else OCCUPANTS
    for line in table[name][1].splitlines():
        if line.startswith("model = "):
            return line.split('"')[1]
    return None


# ---------------------------------------------------------------------------
# Un travailleur = un jeu de 8 agents (A et B à chacun des 4 sièges), construits une
# fois. DouDou et l'Oracle sont sans état entre deux donnes, donc les réutiliser d'une
# configuration à l'autre est sûr — `init_deal` est appelé de toute façon, au cas où un
# jour ils en auraient un.
# ---------------------------------------------------------------------------

_W: dict = {}


def _init_worker(a: str, b: str, tiebreak: str, seed: int) -> None:
    import colver

    _W["colver"] = colver
    _W["b_is_oracle"] = b == "oracle"
    # Les deux occupants doivent varier le **même** facteur, sinon on ne mesure rien
    # d'interprétable : un couple mixte ferait bouger l'enchère et la carte ensemble.
    if is_bidder(a) != is_bidder(b):
        raise SystemExit(f"--a {a} et --b {b} ne varient pas le même facteur")
    _W["factor"] = "bid" if is_bidder(a) else "play"
    _W["a"] = [colver.Agent(spec_for(a, tiebreak), s, seed + s) for s in range(N_SEATS)]
    _W["b"] = [colver.Agent(spec_for(b, tiebreak), s, seed + 100 + s) for s in range(N_SEATS)]


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

    margins, card_margins, pts_ns = [], [], []

    if _W["factor"] == "bid":
        # L'enchère est le facteur : chaque configuration mène la sienne, donc chaque
        # configuration a son propre contrat — et parfois aucun. Une donne passée marque
        # 0-0, la note qui convient à un enchérisseur qui laisse filer une main jouable.
        taker, contract, bid_actions = None, {}, []
        bid = 0
        for mask in range(N_CONF):
            occ = [_W["b"][s] if (mask >> s) & 1 else _W["a"][s] for s in range(N_SEATS)]
            auc = _auction(colver, dealer, hands, occ)
            if auc is None:
                margins.append(0.0)
                card_margins.append(0)
                pts_ns.append(0)
                continue
            bid += 1
            acts, ctr, tkr = auc
            if not bid_actions:  # métadonnées de la configuration de référence
                bid_actions, contract = acts, ctr
                taker = tkr
            m, (ns, ew) = _play(colver, dealer, hands, acts, occ)
            margins.append(m)
            card_margins.append(ns - ew)
            pts_ns.append(ns)
        if bid == 0:
            return None  # aucune configuration n'a pris : la donne ne dit rien
    else:
        auc = _auction(colver, dealer, hands, _W["a"])
        if auc is None:
            return None
        bid_actions, contract, taker = auc
        for mask in range(N_CONF):
            occ = [_W["b"][s] if (mask >> s) & 1 else _W["a"][s] for s in range(N_SEATS)]
            m, (ns, ew) = _play(colver, dealer, hands, bid_actions, occ)
            margins.append(m)
            card_margins.append(ns - ew)
            pts_ns.append(ns)

    rec = {
        "seed": seed, "dealer": dealer,
        "taker": None if _W["factor"] == "bid" else taker,
        "value": int(contract.get("value", 0)), "trump": int(contract.get("trump", -1)),
        "team": int(contract.get("team", -1)), "coinche": int(contract.get("coinche", 0)),
        # Sous le facteur enchère le preneur **change d'une configuration à l'autre** :
        # ventiler par rôle mélangerait des rôles différents dans une même différence
        # appariée. On le met donc à None, et le rapport n'affiche qu'un agrégat.
        "factor": _W["factor"],
        "margins": margins,
        # Les mêmes 16 configurations en **écart de points cartes** (N-S − E-O). C'est la
        # même partie jouée, lue sans le barème : comparer les deux échelles isole ce que
        # la marche du contrat ajoute à elle seule.
        "card_margins": card_margins,
    }
    if check and _W["b_is_oracle"]:
        # Invariant fort sur toute la machinerie de rejeu : quatre Oracles réalisent
        # exactement la valeur DD de la position d'entame. N'a de sens que si B est
        # l'Oracle — un DouDou, si fort soit-il, n'a aucune raison de l'atteindre.
        rec["dd_ns"] = _dd_value_ns(colver, dealer, hands, bid_actions)
        rec["all_or_ns"] = pts_ns[ALL_B]
    return rec


# ---------------------------------------------------------------------------
# Lecture
# ---------------------------------------------------------------------------

ROLES = ("preneur", "partenaire", "défenseur")
MARGIN_SD = 316.0  # écart-type d'une marge de donne (deal_margin_scale, 2 999 donnes)


def _role(seat: int, taker: int | None) -> str:
    if taker is None:  # facteur enchère : le preneur varie selon la configuration
        return "tous sièges"
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


def report(recs: list[dict], tiebreak: str, a: str = "doudou50", b: str = "oracle") -> dict:
    n = len(recs)
    la, lb = label_of(a), label_of(b)
    tb = f" · départage Oracle = {tiebreak!r}" if b == "oracle" else ""
    facteur = "enchère" if is_bidder(a) else "jeu de la carte"
    roles = ("tous sièges",) if is_bidder(a) else ROLES
    print(f"\n{n} donnes notables · facteur = {facteur} · {la} → {lb}{tb}\n")

    # --- contrôle d'exactitude -------------------------------------------------
    checked = [r for r in recs if "dd_ns" in r]
    if checked:
        bad = [r for r in checked if r["dd_ns"] != r["all_or_ns"]]
        print(f"Contrôle · quatre Oracles réalisent la valeur DD : "
              f"{len(checked) - len(bad)}/{len(checked)}"
              + (f"  ⚠️ {len(bad)} écarts" if bad else " ✓"))

    # --- repères ---------------------------------------------------------------
    dd = [r["margins"][ALL_A] for r in recs]
    orc = [r["margins"][ALL_B] for r in recs]
    print("\n── Les deux repères (marge N-S − E-O) " + "─" * 36)
    print(f"  tout-{la:<9}: {statistics.fmean(dd):+8.1f}   (le datum de R4)")
    print(f"  tout-{lb:<9}: {statistics.fmean(orc):+8.1f}")

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
    by_role: dict[str, list[float]] = {r: [] for r in roles}
    deltas_all: list[float] = []
    for r in recs:
        m = r["margins"]
        for s in range(N_SEATS):
            bit = 1 << s
            d = statistics.fmean([_gain(m[k | bit] - m[k], s)
                                  for k in range(N_CONF) if not k & bit])
            by_role[_role(s, r["taker"])].append(d)
            deltas_all.append(d)

    print(f"\n── Effet principal d'un siège ({la} → {lb}), pour son équipe " + "─" * 8)
    print(f"  {'rôle':<12} {'n':>6} {'effet':>9} {'± err':>8} {'écart-type':>11} "
          f"{'>0':>7} {'p10':>7} {'p90':>7}")
    for role in roles:
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
    for k, lbl in ((0, f"partenaire {la}"), (1, f"partenaire {lb}")):
        mean, _, err = _mean_sd(part[k])
        print(f"    {lbl:<22} {mean:>+8.1f} ± {err:.1f}")
    for k in (0, 1, 2):
        mean, _, err = _mean_sd(opp[k])
        print(f"    {k} adversaire(s) {lb:<9} {mean:>+8.1f} ± {err:.1f}")

    # --- la même chose en points cartes ---------------------------------------
    # Contrôle causal : si la complémentarité disparaît sans le barème, c'est la
    # marche au seuil du contrat qui la produit, pas une synergie de jeu.
    card = None
    if all("card_margins" in r for r in recs):
        cpart: dict[int, list[float]] = {0: [], 1: []}
        crole: dict[str, list[float]] = {r: [] for r in roles}
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
        for role in roles:
            mean, _, err = _mean_sd(crole[role])
            print(f"    effet {role:<12} {mean:>+8.1f} ± {err:.1f}")
        c0, _, e0 = _mean_sd(cpart[0])
        c1, _, e1 = _mean_sd(cpart[1])
        print(f"    partenaire {la:<11}{c0:>+8.1f} ± {e0:.1f}")
        print(f"    partenaire {lb:<11}{c1:>+8.1f} ± {e1:.1f}")
        p0, _, _ = _mean_sd(part[0])
        p1, _, _ = _mean_sd(part[1])
        if p0 > 0 and c0 > 0:
            print(f"    → amplification par un bon partenaire : ×{p1/p0:.2f} au barème, "
                  f"×{c1/c0:.2f} en points cartes")
        card = {"effect": {r: round(_mean_sd(crole[r])[0], 1) for r in roles},
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
    print(f"  2 {lb} contre 2 {la} : score {p_duo:.4f} → "
          f"{_elo(p_duo):+.0f} Elo (les deux partenaires changent)")
    print(f"  1 {lb} seul               : score {p_solo:.4f} → "
          f"{2*_elo(p_solo):+.0f} Elo (un seul change : ×2 pour la dilution)")
    print(f"  Les deux estiment le même écart individuel {la} → {lb}.")
    print("  Un écart entre elles est la signature de la complémentarité ci-dessus :")
    print("  le modèle additif d'Elo (équipe = moyenne des deux) ne la représente pas.")

    # --- l'Oracle n'est pas un plafond ----------------------------------------
    # La bonne question est **appariée** : à opposants et partenaire figés, le siège
    # fait-il mieux tenu par l'Oracle ? « Une configuration bat le tout-Oracle »
    # serait trivialement vrai (il suffit que la défense d'en face soit DouDou).
    worse = sum(1 for d in deltas_all if d < -1e-9)
    equal = sum(1 for d in deltas_all if abs(d) <= 1e-9)
    print(f"\n── {lb} est-il un plafond ? (échanges appariés) " + "─" * 22)
    print(f"  échanges où {lb} fait **moins bien** que {la} au même siège : "
          f"{worse}/{len(deltas_all)} ({100*worse/len(deltas_all):.1f} %)")
    print(f"  échanges sans effet : {100*equal/len(deltas_all):.1f} %")
    if b == "oracle":
        print("  Le DD suppose une défense parfaite ; contre un adversaire imparfait, la")
        print("  ligne DD-optimale n'est donc pas la meilleure réponse.")
    print(f"  Un écart au jeu de {lb} n'est pas une erreur — c'est ce qui interdit de")
    print("  s'en servir comme borne, ou comme dénominateur d'un score normalisé.")

    # --- ce que ça dit du classement ------------------------------------------
    mt, sdt, _ = _mean_sd(by_role[roles[0]])
    print("\n── Plafond de vitesse d'un classement " + "─" * 35)
    print(f"  effet d'un siège ({roles[0]}) : {mt:+.1f} pts / donne")
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
        "factor": "bid" if is_bidder(a) else "play",
        "effect": {role: round(_mean_sd(by_role[role])[0], 1) for role in roles},
        "effect_err": {role: round(_mean_sd(by_role[role])[2], 1) for role in roles},
        "effect_sd": {role: round(_mean_sd(by_role[role])[1], 1) for role in roles},
        "occupants": [a, b],
        "effect_partner_weak": round(_mean_sd(part[0])[0], 1),
        "effect_partner_strong": round(_mean_sd(part[1])[0], 1),
        "effect_by_strong_opponents": {k: round(_mean_sd(opp[k])[0], 1) for k in opp},
        "strong_worse_pct": round(100 * worse / len(deltas_all), 1),
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
    ap.add_argument("--a", default="doudou50", choices=sorted(OCCUPANTS) + sorted(BIDDERS),
                    help="occupant de reference (le plus faible des deux)")
    ap.add_argument("--b", default="oracle", choices=sorted(OCCUPANTS) + sorted(BIDDERS),
                    help="occupant substitue (le plus fort) ; meme facteur que --a")
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
        occ = blob["params"].get("occupants", ["doudou50", "oracle"])
        report(blob["payload"]["deals"], blob["params"].get("tiebreak", "?"), *occ)
        return 0

    _check_extension_is_fresh()

    seeds = [args.seed * 1_000_003 + i for i in range(args.deals)]
    checks = {s for i, s in enumerate(seeds) if args.check_every and i % args.check_every == 0}

    recs: list[dict] = []
    with runlog.Timer() as timer:
        if args.jobs > 1:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            with ctx.Pool(args.jobs, initializer=_init_worker,
                          initargs=(args.a, args.b, args.tiebreak, args.seed)) as pool:
                for i, rec in enumerate(pool.imap_unordered(
                        _sweep_star, [(s, s in checks) for s in seeds], chunksize=4)):
                    if rec is not None:
                        recs.append(rec)
                    if (i + 1) % 50 == 0:
                        print(f"  {i+1}/{len(seeds)} donnes  "
                              f"({timer.elapsed/(i+1)*1000:.0f} ms/donne)", file=sys.stderr)
        else:
            _init_worker(args.a, args.b, args.tiebreak, args.seed)
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
    summary = report(recs, args.tiebreak, args.a, args.b)

    if not args.no_log:
        runlog.save(
            script="seat_influence",
            tag=args.tag,
            params={"deals": args.deals, "seed": args.seed, "jobs": args.jobs,
                    "tiebreak": args.tiebreak, "bidder": "bid_v6 (figé, rejoué)",
                    "occupants": [args.a, args.b], "configs": N_CONF},
            summary=summary,
            payload={"deals": recs},
            # Le `.so` est de la provenance au même titre qu'un fichier de poids :
            # il porte le code du solveur, et il n'est pas dans git.
            models=[BID_MODEL, model_of(args.a), model_of(args.b),
                    str(_extension_path())],
            took_s=timer.s,
        )
    return 0


def _sweep_star(arg):
    return sweep(*arg)


def _extension_path():
    import colver._colver as ext

    return Path(ext.__file__)


def _check_extension_is_fresh() -> None:
    """Refuser de mesurer avec un `.so` plus vieux que le cœur Rust.

    Le 2026-08-03, trois runs de 4 000 donnes ont été calculés avec une extension
    compilée une heure avant le commit qui changeait le départage des ex æquo du
    solveur : l'Oracle mesuré n'était pas celui de la production, et le `--tiebreak`
    passé en ligne de commande n'atteignait même pas le parseur. Le piège est
    documenté dans CLAUDE.md — `uv sync` ne recompile pas toujours — mais rien ne
    l'attrapait ici, et **un A/B qui rend des chiffres identiques au bit près n'est
    pas un résultat nul, c'est une panne**. C'est ce signal-là qui l'a révélé.
    """
    so = _extension_path()
    newest, newest_src = 0.0, None
    for root in (ROOT / "colver-core" / "src", ROOT / "colver-py" / "src"):
        for src in root.rglob("*.rs"):
            # `colver-core/src/bin/` sont des binaires autonomes : ils ne sont pas liés
            # dans l'extension, donc en toucher un ne périme rien ici. Sans cette
            # exclusion la garde se déclenche sur le travail d'un autre agent et devient
            # du bruit — une garde qu'on apprend à ignorer ne garde plus rien.
            if "bin" in src.parts:
                continue
            m = src.stat().st_mtime
            if m > newest:
                newest, newest_src = m, src
    if newest > so.stat().st_mtime:
        rel = newest_src.relative_to(ROOT) if newest_src else "?"
        raise SystemExit(
            f"\n⚠️  L'extension PyO3 est périmée.\n"
            f"    {so.relative_to(ROOT)} date du "
            f"{datetime.fromtimestamp(so.stat().st_mtime):%Y-%m-%d %H:%M}\n"
            f"    mais {rel} a changé le "
            f"{datetime.fromtimestamp(newest):%Y-%m-%d %H:%M}.\n"
            f"    Mesurer maintenant produirait des chiffres pour du code qui n'existe plus.\n"
            f"    Recompiler :  env -u CONDA_PREFIX uv run maturin develop --release\n"
        )


if __name__ == "__main__":
    raise SystemExit(main())
