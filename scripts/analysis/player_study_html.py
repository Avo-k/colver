#!/usr/bin/env python3
"""Page HTML partageable à partir du JSON de `player_study.py`.

Le rapport Markdown s'adresse à quelqu'un qui connaît le projet ; cette page
s'adresse aux **joueurs eux-mêmes**. Mêmes chiffres, mais le vocabulaire est
celui de la table (« preneur », « coupe », « annonce ») et chaque mesure dit ce
qu'elle vaut : un classement sans son incertitude ferait passer un écart de
bruit pour un verdict.

La page est autonome — aucune ressource externe, la CSP des artefacts bloque
les CDN — et suit les tokens de l'app (or `#d4af37`, tapis vert, bleu d'équipe)
pour se lire comme une page de Colver. Les deux paires de séries ont été
validées (bande de clarté, plancher de chroma, séparation daltonienne,
contraste) séparément pour le thème clair et pour le sombre.

Usage :
    uv run python scripts/analysis/player_study_html.py etude.json -o etude.html
"""

import argparse
import html
import json
from pathlib import Path

SUITS = ["♠", "♥", "♦", "♣"]

# Bots qu'on montre comme repères. Les autres (2 donnes) ne disent rien.
REF_BOTS = {"dede": "Dédé", "doudou": "DouDou50"}


# ---------------------------------------------------------------------------
# Formatage
# ---------------------------------------------------------------------------


def fr(x, digits=1):
    """Nombre à la française — la virgule décimale, comme sur une feuille."""
    if x is None:
        return "—"
    return f"{x:.{digits}f}".replace(".", ",")


def pc(x, digits=0):
    """Le % est collé par une espace fine insécable (U+202F) : sans elle il
    passe seul à la ligne suivante au milieu d'une phrase."""
    if x is None:
        return "—"
    return f"{x * 100:.{digits}f}\u202f%".replace(".", ",")


def sg(x, digits=1):
    if x is None:
        return "—"
    return f"{'+' if x >= 0 else '−'}{abs(x):.{digits}f}".replace(".", ",")


def e(s):
    """Pour un attribut — les guillemets comptent."""
    return html.escape(str(s), quote=True)


def t(s):
    """Pour du texte. Le français est plein d'apostrophes : les échapper les
    rendrait illisibles à la source sans rien changer au rendu."""
    return html.escape(str(s), quote=False)


def fr_date(iso):
    """2026-02-18 → 18/02/2026."""
    try:
        y, m, d = iso.split("-")
        return f"{d}/{m}/{y}"
    except (ValueError, AttributeError):
        return iso


# ---------------------------------------------------------------------------
# Traits de style, en français de table
# ---------------------------------------------------------------------------


def traits(p, med, dede):
    """3 à 6 observations sur un joueur, chacune comparée au corpus.

    Rédigé sans pronom sujet — registre télégraphique de fiche d'observation.
    On ne connaît pas le genre des joueurs, et « il » par défaut se tromperait
    sur une partie d'entre eux.
    """
    out = []

    pr, m = p["pass_rate"], med.get("pass_rate")
    if pr is not None and m is not None:
        if pr > m + 0.04:
            out.append(("Prudent aux annonces",
                        f"Passe {pc(pr)} de ses prises de parole, contre "
                        f"{pc(m)} pour la médiane du groupe."))
        elif pr < m - 0.04:
            out.append(("Offensif aux annonces",
                        f"Ne passe que {pc(pr)} de ses prises de parole, "
                        f"contre {pc(m)} pour la médiane du groupe."))
        else:
            out.append(("Annonces dans la norme",
                        f"Passe {pc(pr)} de ses prises de parole, comme la "
                        f"médiane du groupe ({pc(m)})."))

    ov, mo = p["overbid"], med.get("overbid")
    if ov is not None and mo is not None and p["declared"]:
        if ov > mo + 3:
            label, txt = "Annonce haut", "plus haut que les autres"
        elif ov < mo - 3:
            label, txt = "Annonce bas", "plus bas que les autres"
        else:
            label, txt = "Annonce juste", "comme les autres"
        out.append((label,
                    f"Preneur sur {p['declared']} donnes, avec une annonce "
                    f"moyenne de {fr(p['mean_declared_value'], 0)} pour un "
                    f"maximum théorique de {fr(p['dd_makeable'], 0)} — {txt}. "
                    f"Contrat tenu {pc(p['contract_success'])}."))

    if p["ruff_chances"] and p["ruff_chances"] >= 20:
        rp, dp = p["ruff_on_partner"], dede.get("ruff_on_partner")
        if rp is not None and dp is not None:
            if rp > dp + 0.05:
                out.append(("Dépensier en atout",
                            f"Sur les plis que son partenaire tenait déjà, "
                            f"coupe quand même {pc(rp)} du temps — Dédé se "
                            f"retient davantage ({pc(dp)})."))
            elif rp < dp - 0.05:
                out.append(("Économe en atout",
                            f"Ne coupe que {pc(rp)} des plis que son "
                            f"partenaire tenait déjà, là où Dédé coupe "
                            f"{pc(dp)}."))

    lt, mlt = p["opening_lead_trump"], med.get("opening_lead_trump")
    if lt is not None and mlt is not None and abs(lt - mlt) > 0.05:
        out.append(("Entame à l'atout" if lt > mlt else "Entame rarement atout",
                    f"Ouvre la donne à l'atout {pc(lt)} du temps "
                    f"(médiane du groupe {pc(mlt)})."))

    og, mg = p["opp_pts"], med.get("opp_pts")
    if og is not None and mg is not None:
        if og < mg - 0.3:
            out.append(("Ne donne rien",
                        f"Quand l'adversaire tient le pli, ne lâche que "
                        f"{fr(og)} point par coup (médiane {fr(mg)})."))
        elif og > mg + 0.3:
            out.append(("Lâche des points",
                        f"Quand l'adversaire tient le pli, lui laisse "
                        f"{fr(og)} point par coup (médiane {fr(mg)})."))

    att, dfn = p["dd_cost_att"], p["dd_cost_def"]
    if att is not None and dfn is not None and p["dd_cost_att_n"] >= 40 and p["dd_cost_def_n"] >= 40:
        if att > dfn * 1.5:
            out.append(("Meilleur en défense",
                        f"Perd {fr(att, 2)} point par décision en prenant, mais "
                        f"seulement {fr(dfn, 2)} en défense. C'est le contrat "
                        "joué, pas la défense, qui coûte."))
        elif dfn > att * 1.5:
            out.append(("Meilleur preneur",
                        f"Perd {fr(dfn, 2)} point par décision en défense "
                        f"contre {fr(att, 2)} en prenant le contrat."))
    return out[:6]


# ---------------------------------------------------------------------------
# Fragments
# ---------------------------------------------------------------------------


def bar_rows(rows, vmax):
    """Barres horizontales à série unique — une seule teinte, pas de légende.

    Les robots sont des lignes comme les autres, en gris et en italique : ils
    servent de repère. Pas de règle verticale en plus — elle répéterait
    exactement ce que leur barre montre déjà.
    """
    out = []
    for r in rows:
        v = r["value"]
        w = 0 if v is None or not vmax else max(1.5, v / vmax * 100)
        cls = " is-you" if r.get("human") else " is-bot"
        out.append(
            f'<div class="bar-row{cls}" tabindex="0" '
            f'data-tip="{e(r["tip"])}">'
            f'<div class="bar-name">{t(r["name"])}</div>'
            f'<div class="bar-track">'
            f'<div class="bar-fill" style="width:{w:.2f}%"></div>'
            f'</div>'
            f'<div class="bar-val">{t(r["label"])}</div>'
            f'</div>')
    return '<div class="bars">' + "".join(out) + "</div>"


def grouped_rows(rows, vmax):
    """Deux séries par joueur — preneur / défense."""
    out = []
    for r in rows:
        a, d = r["att"], r["dfn"]
        wa = 0 if a is None or not vmax else max(1.5, a / vmax * 100)
        wd = 0 if d is None or not vmax else max(1.5, d / vmax * 100)
        cls = " is-you" if r.get("human") else " is-bot"
        out.append(
            f'<div class="grp{cls}">'
            f'<div class="bar-name">{t(r["name"])}</div>'
            f'<div class="grp-bars">'
            f'<div class="grp-line" tabindex="0" data-tip="{e(r["tip_att"])}">'
            f'<div class="bar-track"><div class="bar-fill s-att" '
            f'style="width:{wa:.2f}%"></div></div>'
            f'<div class="bar-val">{t(fr(a, 2))}</div></div>'
            f'<div class="grp-line" tabindex="0" data-tip="{e(r["tip_def"])}">'
            f'<div class="bar-track"><div class="bar-fill s-def" '
            f'style="width:{wd:.2f}%"></div></div>'
            f'<div class="bar-val">{t(fr(d, 2))}</div></div>'
            f'</div></div>')
    return '<div class="groups">' + "".join(out) + "</div>"


def table_block(headers, rows, caption):
    th = "".join(f"<th>{t(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{t(c)}</td>" for c in r) + "</tr>"
                   for r in rows)
    return (f'<details class="tbl"><summary>{t(caption)}</summary>'
            f'<div class="tbl-scroll"><table><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody></table></div></details>")


# ---------------------------------------------------------------------------
# Page
# ---------------------------------------------------------------------------


def build(data):
    corpus = data["corpus"]
    players = data["players"]
    humans = [p for p in players if p["human"]]
    bots = {p["label"]: p for p in players if not p["human"]}
    dede = bots.get("dede", {})
    doudou = bots.get("doudou")

    med = {}
    for k in ("pass_rate", "overbid", "opp_pts", "partner_pts",
              "opening_lead_trump", "dd_avg_cost", "agree_isdd"):
        vals = sorted(p[k] for p in humans if p.get(k) is not None)
        med[k] = vals[len(vals) // 2] if vals else None

    ranked = sorted([p for p in humans if p["dd_avg_cost"] is not None],
                    key=lambda p: p["dd_avg_cost"])
    best = ranked[0] if ranked else None

    # --- classement principal ---------------------------------------------
    rank_rows = []
    for p in ranked:
        rank_rows.append({
            "name": p["label"], "human": True, "value": p["dd_avg_cost"],
            "label": fr(p["dd_avg_cost"], 2),
            "tip": (f"{p['label']} — {fr(p['dd_avg_cost'], 2)} point perdu par "
                    f"décision, sur {p['dd_decisions']} décisions "
                    f"(± {fr(p['dd_cost_sem'], 2)})"),
        })
    for key, nice in REF_BOTS.items():
        b = bots.get(key)
        if b and b["dd_avg_cost"] is not None:
            rank_rows.append({
                "name": nice, "human": False, "value": b["dd_avg_cost"],
                "label": fr(b["dd_avg_cost"], 2),
                "tip": (f"{nice} (robot) — {fr(b['dd_avg_cost'], 2)} point par "
                        f"décision sur {b['dd_decisions']} décisions"),
            })
    rank_rows.sort(key=lambda r: r["value"])
    vmax = max(r["value"] for r in rank_rows) * 1.12

    rank_tbl = table_block(
        ["Joueur", "Coût par décision", "Incertitude", "Décisions",
         "Coups parfaits", "Erreurs et fautes"],
        [[p["label"] + ("" if p["human"] else " (robot)"),
          fr(p["dd_avg_cost"], 2), "± " + fr(p["dd_cost_sem"], 2),
          p["dd_decisions"], pc(p["dd_perfect"]), pc(p["dd_blunder"])]
         for p in sorted(
             [x for x in players if x["dd_decisions"] >= 50],
             key=lambda x: x["dd_avg_cost"] or 9e9)],
        "Voir les chiffres en tableau")

    # --- preneur / défense -------------------------------------------------
    grp = []
    for p in ranked:
        grp.append({
            "name": p["label"], "human": True,
            "att": p["dd_cost_att"], "dfn": p["dd_cost_def"],
            "tip_att": (f"{p['label']} preneur — {fr(p['dd_cost_att'], 2)} point "
                        f"par décision sur {p['dd_cost_att_n']} décisions"),
            "tip_def": (f"{p['label']} en défense — {fr(p['dd_cost_def'], 2)} point "
                        f"par décision sur {p['dd_cost_def_n']} décisions"),
        })
    if dede.get("dd_cost_att") is not None:
        grp.append({
            "name": "Dédé", "human": False,
            "att": dede["dd_cost_att"], "dfn": dede["dd_cost_def"],
            "tip_att": f"Dédé preneur — {fr(dede['dd_cost_att'], 2)} point par décision",
            "tip_def": f"Dédé en défense — {fr(dede['dd_cost_def'], 2)} point par décision",
        })
    gmax = max([v for r in grp for v in (r["att"], r["dfn"]) if v is not None]) * 1.12

    grp_tbl = table_block(
        ["Joueur", "Preneur", "Décisions (preneur)", "Défense", "Décisions (défense)"],
        [[r["name"], fr(r["att"], 2),
          next((p["dd_cost_att_n"] for p in players if p["label"] == r["name"]
                or (r["name"] == "Dédé" and p["label"] == "dede")), "—"),
          fr(r["dfn"], 2),
          next((p["dd_cost_def_n"] for p in players if p["label"] == r["name"]
                or (r["name"] == "Dédé" and p["label"] == "dede")), "—")]
         for r in grp],
        "Voir les chiffres en tableau")

    # --- annonces ----------------------------------------------------------
    bid_rows = []
    for p in sorted(humans, key=lambda x: -(x["overbid"] or -999)):
        if p["overbid"] is None:
            continue
        bid_rows.append([
            p["label"], p["declared"], fr(p["mean_declared_value"], 0),
            fr(p["dd_makeable"], 0), sg(p["overbid"]),
            pc(p["contract_success"]), pc(p["pass_rate"]),
            pc(p["bid_agree_v6"]),
        ])
    if dede.get("overbid") is not None:
        bid_rows.append([
            "Dédé (robot)", dede["declared"], fr(dede["mean_declared_value"], 0),
            fr(dede["dd_makeable"], 0), sg(dede["overbid"]),
            pc(dede["contract_success"]), pc(dede["pass_rate"]),
            pc(dede["bid_agree_v6"]),
        ])
    bid_head = ["Joueur", "Preneur", "Annonce", "Maximum", "Écart",
                "Tenu", "Passe", "Accord robot"]

    # --- style -------------------------------------------------------------
    sty_rows = []
    for p in sorted(humans, key=lambda x: -x["cards"]):
        sty_rows.append([
            p["label"], p["cards"], pc(p["opening_lead_trump"]),
            pc(p["lead_ace_rate"]), pc(p["ruff_rate"]),
            pc(p["ruff_on_partner"]), fr(p["partner_pts"]), fr(p["opp_pts"]),
        ])
    if dede:
        sty_rows.append([
            "Dédé (robot)", dede["cards"], pc(dede["opening_lead_trump"]),
            pc(dede["lead_ace_rate"]), pc(dede["ruff_rate"]),
            pc(dede["ruff_on_partner"]), fr(dede["partner_pts"]),
            fr(dede["opp_pts"]),
        ])
    sty_head = ["Joueur", "Cartes", "Entame atout", "Ouvre à l'As",
                "Coupe", "Coupe s/ part.", "Pts partenaire", "Pts adversaire"]

    # --- fiches joueurs ----------------------------------------------------
    cards = []
    for i, p in enumerate(ranked, 1):
        ts = traits(p, med, dede)
        trait_html = "".join(
            f'<li><span class="tr-k">{e(k)}</span>'
            f'<span class="tr-v">{t(v)}</span></li>' for k, v in ts)
        notes = []
        if p["deals"] < 60:
            notes.append("Peu de donnes : à lire comme une tendance, pas "
                         "comme un verdict.")
        if "nonyme" in p["label"]:
            # Ce n'est pas un joueur mais un seau : toutes les parties jouées
            # sans compte y tombent, sans moyen de les séparer.
            notes.append("Toutes les parties jouées sans être connecté sont "
                         "regroupées ici — ce n'est pas forcément la même "
                         "personne d'une donne à l'autre.")
        thin = "".join(f'<p class="thin">{t(n)}</p>' for n in notes)
        agree = ""
        if p["agree_isdd"] is not None:
            agree = (f'<div class="stat"><span class="s-n">'
                     f'{pc(p["agree_isdd"])}</span>'
                     f'<span class="s-l">même carte que Dédé</span></div>')
        cards.append(f"""
      <article class="player">
        <header class="p-head">
          <span class="p-rank">{i}</span>
          <h3>{t(p['label'])}</h3>
          <span class="p-deals">{p['deals']} donnes</span>
        </header>
        <div class="p-stats">
          <div class="stat"><span class="s-n">{fr(p['dd_avg_cost'], 2)}</span>
            <span class="s-l">point perdu par décision</span></div>
          <div class="stat"><span class="s-n">{pc(p['dd_perfect'])}</span>
            <span class="s-l">de coups parfaits</span></div>
          {agree}
        </div>
        {thin}
        <ul class="traits">{trait_html}</ul>
      </article>""")

    headline = ""
    if best:
        gap = best["dd_avg_cost"] - (dede.get("dd_avg_cost") or 0)
        headline = (
            f"<strong>{t(best['label'])}</strong> est le plus proche du jeu "
            f"parfait : {fr(best['dd_avg_cost'], 2)} point perdu par décision, "
            f"soit {fr(abs(gap), 2)} de plus que Dédé, le robot qui vous "
            f"donne la réplique.")

    suits = " ".join(f'<span class="s{i}">{s}</span>'
                     for i, s in enumerate(SUITS))

    return PAGE.format(
        suits=suits,
        deals=corpus["deals"],
        d_from=fr_date(corpus["from"]), d_to=fr_date(corpus["to"]),
        n_players=len(humans),
        with_review=corpus["with_review"],
        headline=headline,
        rank_bars=bar_rows(rank_rows, vmax),
        rank_tbl=rank_tbl,
        grp_bars=grouped_rows(grp, gmax),
        grp_tbl=grp_tbl,
        bid_tbl=table_block(bid_head, bid_rows, "Voir le détail des annonces"),
        bid_rows=render_table(bid_head, bid_rows),
        sty_rows=render_table(sty_head, sty_rows),
        cards="".join(cards),
        truncated=corpus["truncated"],
        doudou_cost=fr((doudou or {}).get("dd_avg_cost"), 2),
    )


def render_table(headers, rows):
    th = "".join(f"<th>{t(h)}</th>" for h in headers)
    body = "".join("<tr>" + "".join(f"<td>{t(c)}</td>" for c in r) + "</tr>"
                   for r in rows)
    return (f'<div class="tbl-scroll"><table><thead><tr>{th}</tr></thead>'
            f"<tbody>{body}</tbody></table></div>")


PAGE = """<title>Qui joue le mieux ? — étude des joueurs Colver</title>
<style>
:root {{
  --or:      #9c7f24;
  --bleu:    #3574c4;
  --tapis:   #1f4d1c;
  --ground:  #f2f3ee;
  --panel:   #ffffff;
  --line:    rgba(24, 34, 22, 0.13);
  --ink:     #191c19;
  --ink-2:   #4c534a;
  --ink-3:   #6e756b;
  --bien:    #2f7d33;
  --mal:     #b3312e;
  --rail:    rgba(24, 34, 22, 0.07);
  --serif:   'Iowan Old Style', 'Palatino Linotype', Palatino, 'Book Antiqua', Georgia, serif;
  --sans:    'Segoe UI', system-ui, -apple-system, 'Helvetica Neue', sans-serif;
  --mono:    ui-monospace, 'SF Mono', 'Cascadia Mono', Menlo, Consolas, monospace;
}}
@media (prefers-color-scheme: dark) {{
  :root {{
    --or: #ab8a2c; --bleu: #4288d8; --tapis: #163a14;
    --ground: #14161a; --panel: #1b1e21;
    --line: rgba(232, 240, 230, 0.12);
    --ink: #e8e8e6; --ink-2: #b0b4ac; --ink-3: #8e948b;
    --bien: #6bbf6e; --mal: #ef5350;
    --rail: rgba(232, 240, 230, 0.07);
  }}
}}
:root[data-theme="dark"] {{
  --or: #ab8a2c; --bleu: #4288d8; --tapis: #163a14;
  --ground: #14161a; --panel: #1b1e21;
  --line: rgba(232, 240, 230, 0.12);
  --ink: #e8e8e6; --ink-2: #b0b4ac; --ink-3: #8e948b;
  --bien: #6bbf6e; --mal: #ef5350;
  --rail: rgba(232, 240, 230, 0.07);
}}
:root[data-theme="light"] {{
  --or: #9c7f24; --bleu: #3574c4; --tapis: #1f4d1c;
  --ground: #f2f3ee; --panel: #ffffff;
  --line: rgba(24, 34, 22, 0.13);
  --ink: #191c19; --ink-2: #4c534a; --ink-3: #6e756b;
  --bien: #2f7d33; --mal: #b3312e;
  --rail: rgba(24, 34, 22, 0.07);
}}

/* Sans ça, `width: 100%` + un padding horizontal déborde de la largeur du
   padding : la page entière défile latéralement au téléphone. Le reset du
   conteneur d'artefact ne le garantit pas — on le pose donc soi-même. */
*, *::before, *::after {{ box-sizing: border-box; }}

body {{
  background: var(--ground); color: var(--ink);
  overflow-x: hidden;
  font-family: var(--sans); font-size: 16px; line-height: 1.55;
  -webkit-text-size-adjust: 100%;
}}
.wrap {{ width: 100%; max-width: 900px; margin: 0 auto; padding: 0 20px 72px; }}

/* ---------- bandeau ---------- */
.mast {{
  background: var(--tapis); color: #f0efe6;
  margin-bottom: 40px;
  border-bottom: 3px solid var(--or);
}}
.mast-in {{
  width: 100%; max-width: 900px; margin: 0 auto; padding: 40px 20px 32px;
  display: flex; flex-direction: column; gap: 18px;
}}
.eyebrow {{
  font-size: 12px; letter-spacing: 0.16em; text-transform: uppercase;
  color: #c8b978; display: flex; align-items: center; gap: 10px;
}}
.eyebrow .s0, .eyebrow .s3 {{ color: #d9d6c6; }}
.eyebrow .s1, .eyebrow .s2 {{ color: #dd8a7e; }}
.mast h1 {{
  font-family: var(--serif); font-weight: 600;
  font-size: clamp(2rem, 6vw, 3.1rem); line-height: 1.05;
  text-wrap: balance; letter-spacing: -0.01em;
}}
.lede {{
  font-size: clamp(1rem, 2.4vw, 1.15rem); max-width: 62ch;
  color: #ddddcf;
}}
.lede strong {{ color: #fff; font-weight: 600; }}
.facts {{
  display: flex; flex-wrap: wrap; gap: 10px 28px;
  border-top: 1px solid rgba(255,255,255,0.16); padding-top: 16px;
  font-size: 13px; color: #c3c8ba;
}}
.facts b {{
  font-family: var(--mono); font-variant-numeric: tabular-nums;
  color: #f0efe6; font-weight: 600;
}}

/* ---------- sections ---------- */
section {{ margin-bottom: 56px; }}
h2 {{
  font-family: var(--serif); font-size: clamp(1.45rem, 3.6vw, 2rem);
  font-weight: 600; margin-bottom: 6px; text-wrap: balance;
}}
.sub {{ color: var(--ink-2); max-width: 66ch; margin-bottom: 26px; font-size: 15px; }}
.sub + .sub {{ margin-top: -18px; }}
p + p {{ margin-top: 12px; }}

/* ---------- barres ---------- */
.bars {{ position: relative; display: flex; flex-direction: column; gap: 10px; }}
.bar-row {{
  display: grid; grid-template-columns: 132px 1fr 54px;
  align-items: center; gap: 12px; position: relative;
}}
.bar-row:focus-visible, .grp-line:focus-visible {{
  outline: 2px solid var(--or); outline-offset: 3px; border-radius: 3px;
}}
.bar-name {{
  font-size: 14px; font-weight: 600; text-align: right;
  overflow: hidden; text-overflow: ellipsis; white-space: nowrap;
}}
.is-bot .bar-name {{ font-weight: 400; color: var(--ink-3); font-style: italic; }}
.bar-track {{ background: var(--rail); border-radius: 3px; height: 15px; min-width: 0; }}
.bar-fill {{
  height: 100%; background: var(--or);
  border-radius: 3px 4px 4px 3px; min-width: 3px;
}}
.is-bot .bar-fill {{ background: var(--ink-3); }}
.bar-val {{
  font-family: var(--mono); font-variant-numeric: tabular-nums;
  font-size: 13px; color: var(--ink-2);
}}
/* ---------- barres groupées ---------- */
.groups {{ display: flex; flex-direction: column; gap: 16px; }}
.grp {{ display: grid; grid-template-columns: 132px 1fr; gap: 12px; align-items: center; }}
.grp-bars {{ display: flex; flex-direction: column; gap: 2px; min-width: 0; }}
.grp-line {{ display: grid; grid-template-columns: 1fr 54px; gap: 12px; align-items: center; }}
.s-att {{ background: var(--or); }}
.s-def {{ background: var(--bleu); }}
.legend {{
  display: flex; gap: 20px; margin-bottom: 18px; font-size: 13px;
  color: var(--ink-2); flex-wrap: wrap;
}}
.legend span {{ display: flex; align-items: center; gap: 7px; }}
.chip {{ width: 12px; height: 12px; border-radius: 3px; display: inline-block; }}

/* ---------- tableaux ---------- */
.tbl {{ margin-top: 22px; }}
.tbl summary {{
  cursor: pointer; font-size: 13px; color: var(--ink-2);
  padding: 7px 0; user-select: none;
}}
.tbl summary:hover {{ color: var(--ink); }}
.tbl-scroll {{ overflow-x: auto; max-width: 100%; margin-top: 10px; }}
table {{ border-collapse: collapse; width: max-content; min-width: 100%;
         font-size: 13.5px; }}
th, td {{
  text-align: right; padding: 8px 10px; border-bottom: 1px solid var(--line);
  white-space: nowrap;
}}
th:first-child, td:first-child {{ text-align: left; }}
thead th {{
  font-size: 11.5px; letter-spacing: 0.06em; text-transform: uppercase;
  color: var(--ink-3); font-weight: 600; border-bottom: 1px solid var(--ink-3);
}}
td {{ font-family: var(--mono); font-variant-numeric: tabular-nums; }}
td:first-child {{ font-family: var(--sans); font-weight: 600; }}
tbody tr:hover {{ background: var(--rail); }}

/* ---------- fiches ---------- */
.players {{ display: flex; flex-direction: column; gap: 18px; }}
.player {{
  background: var(--panel); border: 1px solid var(--line);
  border-radius: 4px; padding: 22px 24px;
  border-left: 3px solid var(--or);
}}
.p-head {{ display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }}
.p-rank {{
  font-family: var(--mono); font-size: 12px; font-weight: 700;
  color: var(--panel); background: var(--or);
  width: 22px; height: 22px; border-radius: 50%;
  display: inline-flex; align-items: center; justify-content: center;
  flex: none; align-self: center;
}}
.p-head h3 {{ font-family: var(--serif); font-size: 1.35rem; font-weight: 600; }}
.p-deals {{ font-size: 12.5px; color: var(--ink-3); margin-left: auto; }}
.p-stats {{
  display: flex; flex-wrap: wrap; gap: 12px 34px;
  margin: 16px 0 4px; padding-bottom: 16px; border-bottom: 1px solid var(--line);
}}
.stat {{ display: flex; flex-direction: column; gap: 1px; }}
.s-n {{
  font-family: var(--mono); font-variant-numeric: tabular-nums;
  font-size: 1.5rem; font-weight: 600; line-height: 1.1;
}}
.s-l {{ font-size: 12px; color: var(--ink-3); }}
.thin {{ font-size: 12.5px; color: var(--mal); margin-top: 12px; }}
.traits {{ list-style: none; margin-top: 16px; display: flex; flex-direction: column; gap: 11px; }}
.traits li {{ display: flex; flex-direction: column; gap: 1px; }}
.tr-k {{
  font-size: 11.5px; letter-spacing: 0.07em; text-transform: uppercase;
  color: var(--or); font-weight: 700;
}}
.tr-v {{ font-size: 14.5px; color: var(--ink-2); max-width: 68ch; }}

/* ---------- méthode ---------- */
.method {{
  border-top: 1px solid var(--line); padding-top: 28px;
  font-size: 13.5px; color: var(--ink-2);
}}
.method h2 {{ font-size: 1.15rem; margin-bottom: 14px; }}
.method h3 {{
  font-size: 12px; letter-spacing: 0.08em; text-transform: uppercase;
  color: var(--ink); margin: 18px 0 5px; font-weight: 700;
}}
.method p {{ max-width: 72ch; }}

/* ---------- infobulle ---------- */
#tip {{
  position: fixed; z-index: 50; pointer-events: none; opacity: 0;
  background: var(--ink); color: var(--ground);
  padding: 7px 11px; border-radius: 4px; font-size: 12.5px;
  max-width: 280px; transition: opacity .12s ease;
  box-shadow: 0 4px 16px rgba(0,0,0,.28);
}}
#tip.on {{ opacity: 1; }}

@media (max-width: 640px) {{
  .bar-row, .grp {{ grid-template-columns: 92px 1fr 48px; }}
  .grp {{ grid-template-columns: 92px 1fr; }}
  .p-deals {{ margin-left: 0; width: 100%; }}
}}
@media (prefers-reduced-motion: reduce) {{
  * {{ transition: none !important; animation: none !important; }}
}}
</style>

<header class="mast">
  <div class="mast-in">
    <div class="eyebrow">{suits} <span>Colver · Belote contrée</span></div>
    <h1>Qui joue le mieux&nbsp;?</h1>
    <p class="lede">{headline}</p>
    <div class="facts">
      <span><b>{deals}</b> donnes analysées</span>
      <span><b>{n_players}</b> joueurs</span>
      <span>du <b>{d_from}</b> au <b>{d_to}</b></span>
      <span><b>{with_review}</b> donnes rejouées par les robots</span>
    </div>
  </div>
</header>

<div class="wrap">

<section>
  <h2>Le classement</h2>
  <p class="sub">Après chaque partie, un solveur rejoue la donne en voyant les
  quatre mains et calcule le meilleur coup possible. On compte alors ce que
  chaque carte jouée a coûté par rapport à ce meilleur coup. <strong>Plus la
  barre est courte, mieux c'est.</strong></p>
  <p class="sub">C'est une mesure sévère : le solveur triche, il connaît le jeu
  de tout le monde. Un coup parfaitement raisonnable peut lui déplaire parce
  qu'il savait, lui, où était la carte. À lire comme un écart au jeu parfait,
  pas comme une liste de fautes.</p>
  {rank_bars}
  {rank_tbl}
</section>

<section>
  <h2>Preneur ou défenseur&nbsp;?</h2>
  <p class="sub">Prendre le contrat et le défendre ne demandent pas les mêmes
  décisions. Séparer les deux révèle des profils que le classement général
  cache&nbsp;: on peut être excellent en défense et perdre ses donnes dès qu'on
  prend.</p>
  <div class="legend">
    <span><i class="chip" style="background:var(--or)"></i> Quand il prend le contrat</span>
    <span><i class="chip" style="background:var(--bleu)"></i> Quand il défend</span>
  </div>
  {grp_bars}
  {grp_tbl}
</section>

<section>
  <h2>Les annonces</h2>
  <p class="sub">« Maximum théorique » = le meilleur contrat que son camp aurait
  tenu si tout le monde voyait toutes les cartes. <strong>Personne ne l'atteint,
  et c'est normal</strong> — le robot non plus. Ce qui compte, c'est de se
  comparer entre vous&nbsp;: un écart proche de zéro veut dire qu'on annonce
  haut, un écart très négatif qu'on est prudent.</p>
  {bid_rows}
</section>

<section>
  <h2>Les habitudes</h2>
  <p class="sub">Ni bien ni mal&nbsp;: ce sont des manières de jouer. Chaque taux
  est rapporté à ses <em>occasions</em> — « coupe quand il peut » ne compte que
  les fois où le joueur n'avait plus la couleur demandée et tenait de l'atout.</p>
  {sty_rows}
</section>

<section>
  <h2>Les joueurs</h2>
  <p class="sub">Classés du plus proche au plus éloigné du jeu parfait.</p>
  <div class="players">{cards}</div>
</section>

<section class="method">
  <h2>Comment c'est mesuré</h2>

  <h3>Le coût par décision</h3>
  <p>Pour chaque carte, le solveur double-mort résout la position avec les
  quatre mains visibles et compare la carte jouée à la meilleure. La différence
  est comptée en points. Les cartes forcées — quand il n'y a qu'un coup légal —
  ne comptent pas.</p>

  <h3>« Même carte que Dédé »</h3>
  <p>Dédé est le robot contre lequel vous jouez sur Colver. À chaque carte, on
  lui demande ce qu'il aurait joué <em>à votre place</em>, en ne lui montrant
  que ce que votre siège pouvait voir. C'est la mesure la plus juste, parce
  qu'elle ne suppose aucune information cachée. Repère utile&nbsp;: DouDou50,
  l'autre robot, ne rejoue le coup de Dédé qu'environ une fois sur deux — donc
  être à 55&nbsp;% n'a rien de médiocre.</p>

  <h3>Ce que ces chiffres ne disent pas</h3>
  <p>Les points marqués par donne ne mesurent pas le niveau&nbsp;: ils dépendent
  des partenaires et des adversaires, et tout le monde n'a pas joué contre les
  mêmes. Les joueurs ayant peu de donnes ont des chiffres instables — une seule
  mauvaise donne y pèse lourd. Enfin, {truncated} donnes ont un enregistrement
  incohérent et s'arrêtent en cours de route&nbsp;; elles comptent pour le style,
  pas pour le résultat.</p>
</section>

</div>

<div id="tip" role="status"></div>
<script>
(function () {{
  var tip = document.getElementById('tip');
  function show(el) {{
    var t = el.getAttribute('data-tip');
    if (!t) return;
    tip.textContent = t;
    tip.classList.add('on');
    var r = el.getBoundingClientRect();
    var top = r.top - tip.offsetHeight - 9;
    tip.style.top = (top < 6 ? r.bottom + 9 : top) + 'px';
    var left = Math.min(
      Math.max(8, r.left), window.innerWidth - tip.offsetWidth - 8);
    tip.style.left = left + 'px';
  }}
  function hide() {{ tip.classList.remove('on'); }}
  document.querySelectorAll('[data-tip]').forEach(function (el) {{
    el.addEventListener('mouseenter', function () {{ show(el); }});
    el.addEventListener('mouseleave', hide);
    el.addEventListener('focus', function () {{ show(el); }});
    el.addEventListener('blur', hide);
  }});
  window.addEventListener('scroll', hide, {{passive: true}});
}})();
</script>
"""


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("json_in")
    ap.add_argument("-o", "--out", default="etude.html")
    args = ap.parse_args()
    data = json.loads(Path(args.json_in).read_text(encoding="utf-8"))
    Path(args.out).write_text(build(data), encoding="utf-8")
    print(f"→ {args.out}")


if __name__ == "__main__":
    main()
