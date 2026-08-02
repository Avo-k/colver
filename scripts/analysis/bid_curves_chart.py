#!/usr/bin/env python3
"""Courbes : ce qui est annoncé, ce qui est fait, ce qui était atteignable.

    python3 scripts/analysis/bid_curves_chart.py > docs/rules-survey/matrices/bid-curves.svg

Chiffres produits par `bid_distribution.py` sur 60 000 donnes de `playgen_games_9M.bin` :
mêmes donnes, même camp preneur, même atout — donc les trois courbes sont comparables.
Le capot est une colonne détachée : à 252 points il est isolé de 90 points du reste de
l'échelle, et l'inclure en ligne écraserait tout le reste.
"""
import sys

X = [60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160]
SERIES = [
    ("Contrat annoncé", [0.00, 0.00, 2.30, 8.90, 12.89, 23.04, 22.71, 19.45, 8.09, 2.05, 0.54], 0.03),
    ("Points faits par le preneur", [2.98, 4.61, 6.30, 8.35, 10.37, 12.32, 14.07, 13.74, 12.51, 2.49, 0.10], 8.90),
    ("Atteignable en jeu parfait (DD)", [2.63, 3.77, 5.49, 7.36, 9.51, 10.57, 12.86, 12.95, 14.17, 1.84, 0.01], 15.44),
]
# Une seule étiquette directe : les deux autres courbes se frôlent, la légende et les
# valeurs de la colonne capot suffisent à les distinguer.
LABEL_AT = [110, None, None]
LABEL_DY = [-16, 0, 0]

W, H = 880, 500
L, R, T, B = 62, 34, 108, 84
YMAX = 24.0
PW, PH = W - L - R, H - T - B
CAPW = 96                            # largeur réservée à la colonne capot, à droite
LW = PW - CAPW - 26                  # largeur de la partie continue


def px(v):
    return L + LW * (v - X[0]) / (X[-1] - X[0])


def py(v):
    return T + PH * (1 - v / YMAX)


def main():
    o = []
    a = o.append
    xcap = L + PW - CAPW / 2
    a(f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" height="{H}" '
      f'font-family="-apple-system,BlinkMacSystemFont,Segoe UI,Helvetica,Arial,sans-serif" '
      f'role="img" aria-label="Distribution des contrats annoncés, des points faits et des '
      f'points atteignables en jeu parfait">')
    a("""<style>
  .surface{fill:#fcfcfb}.band{fill:#f2f1ed}
  .t1{fill:#0b0b0b}.t2{fill:#52514e}.t3{fill:#7a7973}
  .grid{stroke:#e6e5e1;stroke-width:1}
  .s0{stroke:#2a78d6}.s1{stroke:#eb6834}.s2{stroke:#1baf7a}
  .f0{fill:#2a78d6}.f1{fill:#eb6834}.f2{fill:#1baf7a}
  .ring{stroke:#fcfcfb;stroke-width:2}
  @media (prefers-color-scheme:dark){
    .surface{fill:#1a1a19}.band{fill:#242422}
    .t1{fill:#ffffff}.t2{fill:#c3c2b7}.t3{fill:#8f8e85}
    .grid{stroke:#333330}
    .s0{stroke:#3987e5}.s1{stroke:#d95926}.s2{stroke:#199e70}
    .f0{fill:#3987e5}.f1{fill:#d95926}.f2{fill:#199e70}
    .ring{stroke:#1a1a19}
  }
  .line{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}
</style>""")
    a(f'<rect class="surface" width="{W}" height="{H}"/>')

    a(f'<text class="t1" x="24" y="36" font-size="16" font-weight="600">'
      f"Ce qui est annoncé, ce qui est fait, ce qui était là</text>")
    a(f'<text class="t2" x="24" y="57" font-size="12.5">'
      f"60 000 donnes jouées — mêmes donnes, même camp preneur, même atout</text>")

    lx = 24
    for i, (name, _, _) in enumerate(SERIES):
        a(f'<rect class="f{i}" x="{lx}" y="72" width="9" height="9" rx="2"/>')
        a(f'<text class="t2" x="{lx + 15}" y="81" font-size="12">{name}</text>')
        lx += 15 + len(name) * 6.5 + 24

    # bande de la colonne capot, détachée de l'échelle continue
    a(f'<rect class="band" x="{L + PW - CAPW:.1f}" y="{T - 12}" width="{CAPW}" '
      f'height="{PH + 12}" rx="4"/>')

    for v in range(0, int(YMAX) + 1, 6):
        a(f'<line class="grid" x1="{L}" y1="{py(v):.1f}" x2="{L + PW}" y2="{py(v):.1f}"/>')
        a(f'<text class="t3" x="{L - 10}" y="{py(v) + 4:.1f}" font-size="11.5" '
          f'text-anchor="end">{v} %</text>')

    for v in X:
        if v % 20 == 0:
            a(f'<text class="t3" x="{px(v):.1f}" y="{T + PH + 22:.1f}" font-size="11.5" '
              f'text-anchor="middle">{v}</text>')
    a(f'<text class="t3" x="{xcap:.1f}" y="{T + PH + 22:.1f}" font-size="11.5" '
      f'text-anchor="middle" font-weight="600">capot</text>')
    a(f'<text class="t3" x="{L + LW / 2:.1f}" y="{T + PH + 42:.1f}" font-size="11.5" '
      f'text-anchor="middle">points de la donne</text>')

    for i, (name, ys, cap) in enumerate(SERIES):
        pts = " ".join(f"{px(x):.1f},{py(y):.1f}" for x, y in zip(X, ys))
        a(f'<polyline class="line s{i}" points="{pts}"/>')
        for x, y in zip(X, ys):
            a(f'<circle class="f{i} ring" cx="{px(x):.1f}" cy="{py(y):.1f}" r="3.4"/>')
        # colonne capot : marqueur isolé + valeur
        a(f'<circle class="f{i} ring" cx="{xcap:.1f}" cy="{py(cap):.1f}" r="4.2"/>')
        val = (f"{cap:.2f}" if cap < 0.1 else f"{cap:.1f}").replace(".", ",") + " %"
        a(f'<text class="t2" x="{xcap - 11:.1f}" y="{py(cap) + 4:.1f}" font-size="11.5" '
          f'text-anchor="end">{val}</text>')
        # étiquette directe sur la courbe
        lx_ = LABEL_AT[i]
        if lx_ is None:
            continue
        j = X.index(lx_)
        a(f'<text class="t2" x="{px(lx_):.1f}" y="{py(ys[j]) + LABEL_DY[i]:.1f}" '
          f'font-size="11.5" text-anchor="middle" font-weight="500">{name}</text>')

    a(f'<line class="grid" x1="{L}" y1="{T - 12}" x2="{L}" y2="{T + PH}"/>')
    a(f'<text class="t3" x="24" y="{H - 26}" font-size="11.5">'
      f"Les preneurs annoncent 110-120 quand le jeu parfait en offre 140 — et convertissent "
      f"8,9 % de capots sur les 15,4 % disponibles.</text>")
    a(f'<text class="t3" x="24" y="{H - 9}" font-size="10.5">'
      f"Donnes sous 60 points exclues (~3 %). Au-delà de 160, plus aucune annonce : "
      f"l'échelle s'arrête là où le jeu s'arrête.</text>")
    a("</svg>")
    sys.stdout.write("\n".join(o) + "\n")


if __name__ == "__main__":
    main()
