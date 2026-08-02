#!/usr/bin/env python3
"""Génère le graphique des paliers d'enchère 170/180 (SVG autonome, clair + sombre).

    python3 scripts/analysis/bid_ceiling_chart.py > docs/rules-survey/matrices/bid-ceiling.svg

Chiffres produits par `bid_ceiling_pool.py` sur `data/deals/base_5M.bin` et ses deux couches
de scores. Échelle **linéaire** volontairement : l'argument est précisément que les paliers
170/180 ne sont rien à côté du capot, et un axe logarithmique le masquerait.
"""
import sys

SERIES = [  # (nom, couleur claire, couleur sombre) — slots catégoriels 1, 2, 3
    ("DD — jeu parfait", "#2a78d6", "#3987e5"),
    ("DouDou (DMC)", "#eb6834", "#d95926"),
    ("IS-DD", "#1baf7a", "#199e70"),
]
GROUPS = [  # (étiquette, [DD, DouDou, IS-DD], étiqueter les valeurs ?)
    ("capot réalisé", [16.076, 4.817, 7.013], True),
    ("160 – 169", [2.362, 1.508, 2.216], False),
    ("170 – 179", [0.240, 0.339, 0.340], True),
    ("180 et plus", [0.0039, 0.0146, 0.0128], True),
]

W, H = 880, 452
L, R, T, B = 150, 86, 96, 46      # marges
XMAX = 17.0                        # % max de l'axe
BAR, GAP, GGAP = 18, 2, 24         # hauteur de barre, écart intra-groupe, écart inter-groupes
PW = W - L - R


def x(v):
    return L + PW * v / XMAX


def fmt(v):
    """2 décimales au-dessus de 1 %, 3 en dessous — même précision dans un groupe."""
    return (f"{v:.2f}" if v >= 1 else f"{v:.3f}").replace(".", ",") + " %"


def main():
    o = []
    a = o.append
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" '
      f'role="img" aria-label="Fréquence des paliers d\'enchère 170 et 180 comparée au capot">')
    a("""<style>
  .surface{fill:#fcfcfb}
  .t1{fill:#0b0b0b}.t2{fill:#52514e}.t3{fill:#7a7973}
  .grid{stroke:#e6e5e1;stroke-width:1}
  .s0{fill:#2a78d6}.s1{fill:#eb6834}.s2{fill:#1baf7a}
  @media (prefers-color-scheme:dark){
    .surface{fill:#1a1a19}
    .t1{fill:#ffffff}.t2{fill:#c3c2b7}.t3{fill:#8f8e85}
    .grid{stroke:#333330}
    .s0{fill:#3987e5}.s1{fill:#d95926}.s2{fill:#199e70}
  }
</style>""")
    a(f'<rect class="surface" width="{W}" height="{H}"/>')

    # titre
    a(f'<text class="t1" x="24" y="34" font-size="16" font-weight="600">'
      f"Ouvrir l'échelle d'enchères au-dessus de 160 : ce que ça ajouterait</text>")
    a(f'<text class="t2" x="24" y="55" font-size="12.5">'
      f'% de donnes où le meilleur atout d\'un camp atteint la tranche — 5 000 000 donnes</text>')

    # légende
    lx = 24
    for i, (name, _, _) in enumerate(SERIES):
        a(f'<rect class="s{i}" x="{lx}" y="70" width="9" height="9" rx="2"/>')
        a(f'<text class="t2" x="{lx + 15}" y="79" font-size="12">{name}</text>')
        lx += 15 + len(name) * 6.6 + 22

    # grille + axe x
    ybot = T + len(GROUPS) * (3 * BAR + 2 * GAP) + (len(GROUPS) - 1) * GGAP
    for v in (0, 5, 10, 15):
        a(f'<line class="grid" x1="{x(v):.1f}" y1="{T - 10}" x2="{x(v):.1f}" y2="{ybot + 6}"/>')
        a(f'<text class="t3" x="{x(v):.1f}" y="{ybot + 24}" font-size="11.5" '
          f'text-anchor="middle">{v} %</text>')

    y = T
    for label, vals, do_label in GROUPS:
        gh = 3 * BAR + 2 * GAP
        a(f'<text class="t1" x="{L - 16}" y="{y + gh / 2 + 4.5:.1f}" font-size="12.5" '
          f'text-anchor="end" font-weight="500">{label}</text>')
        for i, v in enumerate(vals):
            by = y + i * (BAR + GAP)
            w = max(PW * v / XMAX, 1.6)
            if w >= 8:  # extrémité arrondie 4px, ancrée à la ligne de base
                d = (f"M{L},{by} H{L + w - 4} a4,4 0 0 1 4,4 v{BAR - 8} "
                     f"a4,4 0 0 1 -4,4 H{L} Z")
                a(f'<path class="s{i}" d="{d}"/>')
            else:
                a(f'<rect class="s{i}" x="{L}" y="{by}" width="{w:.2f}" height="{BAR}"/>')
            if do_label:
                a(f'<text class="t2" x="{L + w + 7:.1f}" y="{by + BAR / 2 + 4:.1f}" '
                  f'font-size="11.5">{fmt(v)}</text>')
        y += gh + GGAP

    a(f'<line class="grid" x1="{L}" y1="{T - 10}" x2="{L}" y2="{ybot + 6}"/>')
    a(f'<text class="t3" x="24" y="{H - 12}" font-size="11.5">'
      f"Un palier à 180 n'est tenable que dans une donne sur 7 000 — et quand il l'est, "
      f"le capot l'est presque toujours aussi.</text>")
    a("</svg>")
    sys.stdout.write("\n".join(o) + "\n")


if __name__ == "__main__":
    main()
