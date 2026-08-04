#!/usr/bin/env python3
"""Mesure B : le préfixe d'enchère déplace-t-il l'étiquette de la couche de scores ?

Enveloppe de `bench_prefix_label`. Le binaire fait le travail ; ce script l'exécute,
dépouille et **journalise** — un run coûte des dizaines de minutes de GPU, et un run
qui n'écrit que sur stdout se repaie à chaque question posée.

La mesure A ([docs/data_gen/isdd_score_layer_v2.md](../../docs/data_gen/isdd_score_layer_v2.md) §4)
a classé les préfixes possibles en quatre rangs par ce qu'ils **mentent** — or (rien),
argent (une relance retirée), bronze (une ouverture retirée, un siège devient muet), fer
(tout le préfixe construit). C'est une hypothèse ordonnée ; B mesure si l'ordre du
mensonge est l'ordre du coût en points cartes.

**Tout se lit contre le bras témoin** — le même préfixe étiqueté deux fois avec deux
graines. C'est le bruit propre d'IS-DD à ce budget de mondes ; un écart plus petit que
lui ne veut rien dire, quel que soit son z.

Usage :
    uv run python scripts/analysis/prefix_label.py --deals 2000 --tag couche-v2-mesure-B
    uv run python scripts/analysis/prefix_label.py --reuse <fichier.json>

Le sidecar playgen doit être debout, et redescendre après (`playgen-down`).
"""

import argparse
import json
import math
import os
import subprocess
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import runlog  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
BIN = os.path.join(ROOT, "target/release/bench_prefix_label")
BOT = "arena/bots/gen_isdd_cardpts.toml"
BID_MODEL = "models/bid_v6_isdd_resume/bid_nn_final.bin"
PLAYGEN = "models/playgen/playgen_v2_final.bin"
GAMES = "data/training/isdd_games_v1.bin"

CONTRASTS = [
    ("or_minus_fer", "or − fer            (case t₁)"),
    ("peel_minus_fer", "épluchage − fer     (case t₂)"),
    ("silver_minus_fer", "  dont argent (relance)"),
    ("bronze_minus_fer", "  dont bronze (ouverture)"),
]

# ⚠️ Les contrastes ci-dessus sont en points **N-S**, et c'est la mauvaise lecture.
# L'étiquette est un total N-S, mais l'effet du préfixe porte sur **le camp qui prend** :
# un préfixe réaliste place correctement la force, donc le preneur joue mieux. En N-S
# l'effet change de signe selon qui prend (+6,7 quand N-S prend, −2,1 quand c'est E-O) et
# s'annule presque à la moyenne — +2,2 au lieu de +4,4. Exactement le piège de
# `bid_contract_ranks` (rang « de la donne » contre rang « vu du preneur »), deuxième fois.
def _side(ns_dd):
    """Camp qui peut tenir le contrat à cet atout : 0 = N-S. Les points cartes étant à
    somme constante, un seul des deux le peut."""
    return 0 if ns_dd > 81 else 1


def taker_oriented(rows):
    """Les mêmes contrastes, comptés du côté du preneur plausible.

    `rows` : [t1, t2, peel_is_raise, or1, or2, fer1, peel, fer2, dd1, dd2].
    """
    import statistics as st

    def agg(xs):
        m = sum(xs) / len(xs)
        se = st.stdev(xs) / len(xs) ** 0.5
        return {"mean": round(m, 3), "se": round(se, 3),
                "z": round(m / se, 2) if se else None,
                "sd": round(st.stdev(xs), 3), "n": len(xs)}

    sign1 = [1 if _side(r[8]) == 0 else -1 for r in rows]
    sign2 = [1 if _side(r[9]) == 0 else -1 for r in rows]
    out = {
        "or_minus_fer": agg([s * (r[3] - r[5]) for s, r in zip(sign1, rows, strict=True)]),
        "peel_minus_fer": agg([s * (r[6] - r[7]) for s, r in zip(sign2, rows, strict=True)]),
    }
    for name, keep in [("silver_minus_fer", True), ("bronze_minus_fer", False)]:
        sel = [s * (r[6] - r[7]) for s, r in zip(sign2, rows, strict=True) if bool(r[2]) == keep]
        if len(sel) > 2:
            out[name] = agg(sel)
    return out


def run(deals, threads, dets, out_json):
    cmd = [BIN, "--games", GAMES, "--bot", BOT, "--bid-model", BID_MODEL,
           "--deals", str(deals), "--threads", str(threads), "--json", out_json]
    if dets:
        cmd += ["--dets", str(dets)]
    print(f"[run] {' '.join(cmd)}", file=sys.stderr)
    subprocess.run(cmd, cwd=ROOT, check=True)
    with open(out_json, encoding="utf-8") as fh:
        return json.load(fh)


def digest(raw):
    sd0 = raw["control_sd"]
    out = {
        "deals": raw["deals"],
        "secs": round(raw["secs"], 1),
        "labels_per_s": round(5 * raw["deals"] / raw["secs"], 2),
        "control_sd": round(sd0, 3),
        "control_mean": round(raw["control_mean"], 3),
        # La lecture qui compte. Gardée à part de `*_ns` pour que l'index versionné
        # porte les deux et qu'on voie l'écart entre elles.
        "taker_oriented": taker_oriented(raw["rows"]),
    }
    for key, _ in CONTRASTS:
        c = raw[key]
        se = c["se"]
        out[key] = {
            "mean": round(c["mean"], 3),
            "se": round(se, 3),
            "z": round(c["mean"] / se, 2) if se else None,
            "sd": round(c["sd"], 3),
            # Le seul chiffre qui compte vraiment : l'écart rapporté au bruit propre
            # de l'étiqueteur. Un z énorme sur un effet de 0,3 point ne change rien.
            "in_control_sd": round(c["mean"] / sd0, 3) if sd0 else None,
            "n": c["n"],
        }
    return out


def show(d):
    print("\n=== ORIENTÉ PRENEUR — la lecture qui compte ===")
    print(f"  {'contraste':<26}{'n':>7}{'moyenne':>11}{'±':>8}{'z':>8}{'en σ témoin':>14}")
    for key, label in CONTRASTS:
        c = d["taker_oriented"].get(key)
        if not c:
            continue
        ratio = c["mean"] / d["control_sd"] if d["control_sd"] else 0
        print(f"  {label.split('(')[0].strip():<26}{c['n']:>7}{c['mean']:>+11.2f}"
              f"{c['se']:>8.2f}{c['z']:>+8.1f}{ratio:>+14.2f}")
    print("\n  (ci-dessous : les mêmes en points N-S bruts — l'effet s'y annule à moitié,")
    print("   parce qu'il change de signe selon le camp qui prend)")


def show_ns(d):
    print(f"\n=== {d['deals']} donnes, {d['secs']:.0f} s ({d['labels_per_s']:.1f} étiquetages/s) ===")
    print(f"\nTÉMOIN — même préfixe, deux graines : "
          f"moyenne {d['control_mean']:+.2f}, **écart-type apparié {d['control_sd']:.2f} points**")
    print("  C'est le bruit propre de l'étiqueteur. Rien en dessous ne veut dire quoi que ce soit.\n")
    print(f"  {'contraste':<36}{'n':>7}{'moyenne':>11}{'±':>8}{'z':>8}{'en σ témoin':>14}")
    for key, label in CONTRASTS:
        c = d[key]
        z = f"{c['z']:+.1f}" if c["z"] is not None else "—"
        print(f"  {label:<36}{c['n']:>7}{c['mean']:>+11.2f}{c['se']:>8.2f}{z:>8}"
              f"{c['in_control_sd']:>+14.2f}")


def verdict(d):
    """La lecture qu'on veut retrouver dans six mois, pas la table brute.

    **Un écart petit devant le bruit n'est pas un écart négligeable.** Le bruit de
    l'étiqueteur (σ ≈ 24 pts) est *non biaisé* : il se moyenne sur des centaines de
    milliers de donnes. Un décalage systématique de 4 points ne se moyenne pas — il
    reste dans chaque étiquette, dans le même sens. C'est pourquoi on lit le z, et pas
    seulement le rapport au plancher.
    """
    sd0 = d["control_sd"]
    print("\n=== LECTURE ===")
    for key, label in CONTRASTS[:2]:
        c = d["taker_oriented"][key]
        lo, hi = c["mean"] - 2 * c["se"], c["mean"] + 2 * c["se"]
        if c["z"] is None or abs(c["z"]) < 2:
            note = "non résolu à cet échantillon"
        else:
            note = (f"décalage SYSTÉMATIQUE de {c['mean']:+.1f} pt pour le preneur "
                    f"({abs(c['mean']) / sd0:.2f}× le bruit par étiquette, mais il ne "
                    f"se moyenne pas)")
        print(f"  {label.split('(')[0].strip():<22} [{lo:+.1f} ; {hi:+.1f}] pts à 2σ — {note}")

    ag = d["taker_oriented"].get("silver_minus_fer")
    br = d["taker_oriented"].get("bronze_minus_fer")
    if ag and br:
        gap = ag["mean"] - br["mean"]
        gse = math.hypot(ag["se"], br["se"])
        print(f"\n  argent − bronze : {gap:+.2f} ±{2 * gse:.2f} (2σ) — "
              + ("l'ordre du mensonge se voit dans les points"
                 if abs(gap) > 2 * gse else "les deux rangs ne se distinguent pas"))

    o = d["taker_oriented"]["or_minus_fer"]["mean"]
    p = d["taker_oriented"]["peel_minus_fer"]["mean"]
    print(f"\n  Hiérarchie mesurée : or {o:+.2f}  >  épluchage {p:+.2f}  >  fer 0 (référence)")
    print("  Conséquence pour une couche MIXTE : ses cases ne sont pas comparables entre")
    print("  elles — celle que v6 annonce est mieux étiquetée que les autres, ce qui")
    print("  incline vers la politique qu'on cherche à dépasser. D'où le fichier de rangs.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--deals", type=int, default=2000)
    ap.add_argument("--threads", type=int, default=96,
                    help="96 mesuré sans erreur ; 256 noie le sidecar")
    ap.add_argument("--dets", type=int)
    ap.add_argument("--tag", default="prefix-label")
    ap.add_argument("--reuse", help="fichier json déjà produit")
    ap.add_argument("--no-log", action="store_true")
    args = ap.parse_args()

    with runlog.Timer() as t:
        if args.reuse:
            with open(args.reuse, encoding="utf-8") as fh:
                raw = json.load(fh)
        else:
            with tempfile.TemporaryDirectory() as tmp:
                raw = run(args.deals, args.threads, args.dets,
                          os.path.join(tmp, "b.json"))

    d = digest(raw)
    show(d)
    show_ns(d)
    verdict(d)

    if not args.no_log:
        runlog.save(
            script="prefix_label",
            tag=args.tag,
            params={"deals": args.deals, "threads": args.threads, "dets": args.dets,
                    "bot": BOT, "games": GAMES, "reused": bool(args.reuse)},
            summary=d,
            payload=raw,   # `rows` porte les 5 étiquettes de chaque donne
            models=[os.path.join(ROOT, BID_MODEL), os.path.join(ROOT, PLAYGEN)],
            took_s=t.s,
        )


if __name__ == "__main__":
    main()
