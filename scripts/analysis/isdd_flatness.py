#!/usr/bin/env python3
"""Le plat de `PlayObjective::DealScore` : fréquence, justesse, et ce qu'il coûte.

Le score de donne est **plat sur deux paliers entiers** du barème : toute chute
vaut `162 + contrat×mult` quel que soit le partage des plis, et tout contrat
contré tenu de même. Seul « réussi non contré » a une pente. Quand tous les
mondes d'IS-DD tombent du même côté du seuil, l'objectif ne distingue donc plus
aucune carte — et la décision revenait à l'indice légal le plus bas avant le
départage lexicographique du 2026-08-06 (`IsDdSearch::prefers`).

Trois modes, du moins cher au plus cher :

  truth   un solve DD par décision, sur la donne réelle. Aucun GPU. Donne la
          fréquence du plat **dans le vrai monde** — une borne supérieure sur
          celle d'IS-DD, dont le plat est une conjonction sur ses mondes.

  agent   IS-DD à chaque décision, contre le solve du vrai monde. **La** mesure :
          à quelle fréquence Dédé est plat, à quelle fréquence il a tort de
          l'être, et ce que son départage coûte alors réellement.

  ab      `DealScore` contre `CardPoints`, appariés à la même décision et jugés
          tous deux au coût en score de donne dans le vrai monde. C'est la mesure
          « appariée à la décision » qu'un h2h d'arène ne peut pas faire : son
          RNG distribue *et* alimente la recherche, donc le premier coup qui
          diverge décale toutes les donnes suivantes.

Le corpus est un COLVGM01/02 de donnes **réellement jouées** (`isdd_games_v2.bin`
= bid v6 + IS-DD) : la distribution de positions doit être celle où Dédé décide,
pas celle d'un jeu aléatoire.

`agent` et `ab` ont besoin du sidecar playgen (`playgen-up`) — sans lui les
mondes retombent en contraints-uniformes et la mesure porte sur un autre agent.
Et **rendre la VRAM à la fin** (`playgen-down`).

    scripts/analysis/isdd_flatness.py truth data/training/isdd_games_v2.bin --deals 300
    COLVER_ISDD_DETS=64 scripts/analysis/isdd_flatness.py agent data/training/isdd_games_v2.bin --deals 200
    COLVER_ISDD_DETS=64 scripts/analysis/isdd_flatness.py ab data/training/isdd_games_v2.bin --deals 100
"""

import argparse
import concurrent.futures as cf
import os
import statistics
import sys
import time

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", ".."))
sys.path.insert(0, os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))), "python"))

import colver  # noqa: E402
import colver.web.agents as agents  # noqa: E402

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

EPS = 1e-6


def read_corpus(path, n_deals):
    """`[(dealer, hands, actions)]` depuis un COLVGM01 ou 02.

    Le pas dépend du magic : COLVGM02 insère `score_ns`/`score_ew` (2×u16) entre
    les mains et le compte d'actions. Lire un v2 au pas de v1 ne lève rien — ça
    décale chaque donne de 4 octets et rend des statistiques entièrement fausses.
    """
    raw = np.memmap(path, dtype=np.uint8, mode="r")
    magic = raw[:8].tobytes()
    if magic not in (b"COLVGM01", b"COLVGM02"):
        raise SystemExit(f"magic inattendu : {magic!r}")
    score_len = 4 if magic == b"COLVGM02" else 0
    total = int(np.frombuffer(raw[8:16].tobytes(), dtype="<u8")[0])
    out, p = [], 16
    for _ in range(min(n_deals, total)):
        dealer = int(raw[p])
        hands = [[i for i in range(32) if int(h) >> i & 1]
                 for h in np.frombuffer(raw[p + 1:p + 17].tobytes(), dtype="<u4")]
        head = p + 17 + score_len
        na = int(raw[head])
        acts = raw[head + 1:head + 1 + na].tolist()
        p = head + 1 + na
        out.append((dealer, hands, acts))
    return out, magic.decode()


def _true_costs(env, seat, action_of_agent):
    """Coût réel, en score de donne, de la carte choisie — orienté vers son camp."""
    r = env.solve_scores()
    deal = {int(c): int(v) for c, v in r["deal_scores"]}
    pts = {int(c): int(v) for c, v in r["scores"]}
    team = seat % 2
    best = max(deal.values()) if team == 0 else min(deal.values())
    return {
        "deal": deal, "pts": pts,
        "spread_deal": max(deal.values()) - min(deal.values()),
        "spread_pts": max(pts.values()) - min(pts.values()),
        "cost": abs(deal[action_of_agent] - best) if action_of_agent in deal else None,
        "made": [bool(v) for _, v in r["contract_made"]],
    }


# ── truth ────────────────────────────────────────────────────────────────────

def run_truth(deals, workers):
    def one(job):
        dealer, hands, acts = job
        env = colver.Env.deal_with_hands(dealer, hands)
        out = []
        for a in acts:
            if env.is_terminal():
                break
            if int(env.phase()) == 1:
                legals = list(env.legal_actions())
                if a not in legals:
                    return []          # journal incohérent : on jette la donne
                if len(legals) > 1:
                    t = _true_costs(env, int(env.current_player()), a)
                    out.append({
                        "trick": int(sum(env.get_tricks_won())),
                        "coinche": int(env.get_contract()["coinche"]),
                        "flat": t["spread_deal"] == 0,
                        "spread_pts": t["spread_pts"],
                        "decided": all(t["made"]) or not any(t["made"]),
                    })
            env.step(a)
        return out

    rows = []
    with cf.ThreadPoolExecutor(max_workers=workers) as ex:
        for r in ex.map(one, deals):
            rows.extend(r)
    return rows


def report_truth(rows):
    n = len(rows)
    flat = [r for r in rows if r["flat"]]
    print(f"\n{n} décisions non forcées, {len(flat)} plates = {100 * len(flat) / n:.1f} %\n")
    print("par pli :")
    for t in range(8):
        s = [r for r in rows if r["trick"] == t]
        if s:
            f = sum(r["flat"] for r in s)
            print(f"  pli {t + 1} : {f:5d}/{len(s):5d} = {100 * f / len(s):5.1f} %")
    print("\npar niveau de contre :")
    by_coinche = {}
    for k, lab in ((0, "normal"), (1, "contré"), (2, "surcontré")):
        s = [r for r in rows if r["coinche"] == k]
        if not s:
            continue
        f = sum(r["flat"] for r in s)
        by_coinche[lab] = round(100 * f / len(s), 1)
        print(f"  {lab:10s} : {f:6d}/{len(s):6d} = {100 * f / len(s):5.1f} % "
              f"({100 * len(s) / n:.1f} % des décisions)")
    dec = sum(r["decided"] for r in flat)
    sp = sorted(r["spread_pts"] for r in flat)
    print(f"\nissue déjà scellée sur les plates : {dec}/{len(flat)} = "
          f"{100 * dec / max(1, len(flat)):.1f} %")
    print(f"écart en POINTS CARTES sur ces mêmes décisions : "
          f"moyenne {statistics.fmean(sp) if sp else 0:.1f}, "
          f"médiane {sp[len(sp) // 2] if sp else 0}, max {sp[-1] if sp else 0}")
    return {
        "decisions": n,
        "flat_pct": round(100 * len(flat) / n, 1),
        "flat_pct_by_coinche": by_coinche,
        "decided_pct_of_flat": round(100 * dec / max(1, len(flat)), 1),
        "card_pts_spread_mean_on_flat": round(statistics.fmean(sp), 1) if sp else 0.0,
    }


# ── agent / ab ───────────────────────────────────────────────────────────────

def _specs(mode, belief):
    base = agents.spec_for("dede", belief_model=belief)
    if mode == "agent":
        return {"deal": base}
    pts = base.replace('method = "isdd"', 'method = "isdd"\nobjective = "card_points"')
    assert 'objective = "card_points"' in pts, "la spec CardPoints n'a pas pris"
    return {"deal": base, "pts": pts}


def run_agents(deals, mode, belief, progress):
    """Une décision = une ligne. Séquentiel : IS-DD parallélise déjà en interne."""
    specs = _specs(mode, belief)
    rows = []
    for k, (dealer, hands, acts) in enumerate(deals):
        env = colver.Env.deal_with_hands(dealer, hands)
        # Un agent par siège : IS-DD est *seat-bound*, l'interroger depuis un
        # autre siège lui donnerait une information que ce siège n'avait pas.
        tables = {kind: [colver.Agent(spec, s) for s in range(4)]
                  for kind, spec in specs.items()}
        for tbl in tables.values():
            for b in tbl:
                b.init_deal(env)
        for a in acts:
            if env.is_terminal():
                break
            if int(env.phase()) == 1:
                seat = int(env.current_player())
                legals = list(env.legal_actions())
                if a not in legals:
                    break
                if len(legals) > 1:
                    d = tables["deal"][seat].decide(env)
                    vals = {int(c): float(v) for c, v in d.get("candidates", [])}
                    pick = int(d["action"])
                    t = _true_costs(env, seat, pick)
                    row = {
                        "trick": int(sum(env.get_tricks_won())),
                        "coinche": int(env.get_contract()["coinche"]),
                        "isdd_flat": bool(vals) and
                        (max(vals.values()) - min(vals.values())) <= EPS,
                        "true_spread": t["spread_deal"],
                        "cost_deal": t["cost"],
                    }
                    if "pts" in tables:
                        d2 = tables["pts"][seat].decide(env)
                        alt = int(d2["action"])
                        row["cost_pts"] = _true_costs(env, seat, alt)["cost"]
                        row["same"] = alt == pick
                    rows.append(row)
            for tbl in tables.values():
                for b in tbl:
                    b.observe(env, a)
            env.step(a)
        if progress and (k + 1) % progress == 0:
            print(f"  {k + 1}/{len(deals)} donnes, {len(rows)} décisions", flush=True)
    return rows


def report_agent(rows):
    n = len(rows)
    flat = [r for r in rows if r["isdd_flat"]]
    bad = [r for r in flat if r["true_spread"] > 0]
    costs = [r["cost_deal"] for r in bad]
    nz = [c for c in costs if c > 0]
    leak = sum(costs) / n if n else 0.0
    ctrl = [r["cost_deal"] for r in rows if not r["isdd_flat"]]
    print(f"\n{n} décisions non forcées ; Dédé plat sur {len(flat)} = "
          f"{100 * len(flat) / n:.1f} %")
    print(f"  dont le vrai monde n'était PAS plat : {len(bad)}/{len(flat)} = "
          f"{100 * len(bad) / max(1, len(flat)):.1f} %")
    if costs:
        print(f"  coût du départage sur ces cas : moyenne {statistics.fmean(costs):.1f}, "
              f"max {max(costs)} ; réellement coûteux {len(nz)}/{len(bad)}")
    print(f"  ⚠ fuite rapportée à TOUTES les décisions : {leak:.2f} pt de score de donne "
          f"— portée par {len(nz)} incident(s), donc à ne pas lire comme une moyenne stable")
    print(f"\ntémoin — Dédé non plat ({len(ctrl)}) : coût moyen "
          f"{statistics.fmean(ctrl) if ctrl else 0:.1f}")
    return {
        "decisions": n,
        "isdd_flat_pct": round(100 * len(flat) / n, 1),
        "wrong_flat_pct_of_flat": round(100 * len(bad) / max(1, len(flat)), 2),
        "costly_events": len(nz),
        "leak_per_decision": round(leak, 3),
        "control_cost_when_not_flat": round(statistics.fmean(ctrl), 1) if ctrl else 0.0,
    }


def report_ab(rows):
    def block(sub, lab):
        if not sub:
            return None
        n = len(sub)
        diff = [r["cost_pts"] - r["cost_deal"] for r in sub]
        m = statistics.fmean(diff)
        se = (statistics.pstdev(diff) / n ** 0.5) if n > 1 else 0.0
        cd = statistics.fmean(r["cost_deal"] for r in sub)
        cp = statistics.fmean(r["cost_pts"] for r in sub)
        same = 100 * sum(r["same"] for r in sub) / n
        print(f"{lab:22s} n={n:5d}  DealScore {cd:6.2f}  CardPoints {cp:6.2f}  "
              f"écart {m:+6.2f} ± {1.96 * se:5.2f}  même carte {same:4.1f} %")
        return {"n": n, "deal": round(cd, 2), "pts": round(cp, 2),
                "diff": round(m, 2), "ci95": round(1.96 * se, 2), "same_pct": round(same, 1)}

    print()
    return {
        "all": block(rows, "toutes décisions"),
        "flat": block([r for r in rows if r["isdd_flat"]], "  Dédé plat"),
        "not_flat": block([r for r in rows if not r["isdd_flat"]], "  Dédé non plat"),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("mode", choices=["truth", "agent", "ab"])
    ap.add_argument("corpus")
    ap.add_argument("--deals", type=int, default=200)
    ap.add_argument("--workers", type=int, default=16, help="mode truth seulement")
    ap.add_argument("--progress", type=int, default=10, help="0 = silencieux")
    ap.add_argument("--tag", default="")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    deals, magic = read_corpus(args.corpus, args.deals)
    belief = colver.belief_model_path()
    belief = str(belief) if belief else None
    print(f"{len(deals)} donnes de {args.corpus} ({magic})")
    if args.mode != "truth":
        if not agents.SIDECAR_URL:
            print("⚠ aucun sidecar playgen : les mondes seront contraints-uniformes, "
                  "donc ce n'est PAS l'agent de production qui est mesuré", file=sys.stderr)
        print(f"mondes/décision = {agents.ISDD_DETS or 'mode temps'}")

    t0 = time.time()
    if args.mode == "truth":
        rows = run_truth(deals, args.workers)
        summary = report_truth(rows)
    else:
        rows = run_agents(deals, args.mode, belief, args.progress)
        summary = report_agent(rows) if args.mode == "agent" else report_ab(rows)
    took = time.time() - t0

    if not args.no_log:
        path = runlog.save(
            "isdd_flatness", args.tag or args.mode,
            {"mode": args.mode, "corpus": args.corpus, "deals": len(deals),
             "dets": agents.ISDD_DETS, "sidecar": bool(agents.SIDECAR_URL)},
            summary, payload={"rows": rows},
            models=[belief] if belief else (), took_s=took)
        print(f"\njournalisé : {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
