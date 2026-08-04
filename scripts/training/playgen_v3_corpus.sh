#!/usr/bin/env bash
# Le corpus de **parties** dont un playgen v3 a besoin.
#
# Tout le corpus playgen actuel (`playgen_games_9M.bin`, COLVGM01) est fait de
# donnes tirées indépendamment, donc jouées à 0-0. Un modèle qui lit le score y
# verrait toujours le même jeton et n'apprendrait rien : **le corpus est le
# préalable, pas le modèle**.
#
#   usage: scripts/training/playgen_v3_corpus.sh [donnes] [threads]
#
# Le joueur est `arena/bots/gen_match.toml` — bid v6 (score-aware, c'est lui qui
# met de l'information dans les jetons de score) + DouDou50 à la carte. Voir
# l'en-tête du TOML pour l'arbitrage contre IS-DD.
#
# Mesuré le 2026-08-04 : **159 donnes/s à 16 threads** sur cette machine (32
# cœurs), 62 o/donne. Un run de 3 M de donnes coûte donc ~5,2 h et 190 Mo.
#
# Pourquoi 3 M et pas 9 M comme le corpus v2 : ce qu'un entraînement consomme
# n'est pas des donnes mais des **échantillons**, et une donne en fournit 96
# (4 observateurs × 24 permutations de couleurs). 3 M de donnes = 288 M
# d'échantillons distincts, dont un run à 30,7 M ne prend que 11 %. Ce sont donc
# les heures de génération, pas la diversité, qui fixent la taille.
set -uo pipefail
cd "$(dirname "$0")/../.."

DEALS=${1:-3000000}
THREADS=${2:-32}
OUT=${OUT:-data/training/playgen_matches_${DEALS}.bin}

echo "corpus de parties : $DEALS donnes, $THREADS threads -> $OUT"
./target/release/gen_games_isdd \
  --bot arena/bots/gen_match.toml \
  --match-mode \
  --deals "$DEALS" \
  --threads "$THREADS" \
  --seed 2026 \
  --progress-every 50000 \
  --out "$OUT"

# `--check` rejoue tout le corpus et rend l'histogramme des écarts de score.
# Ce n'est pas décoratif : c'est le **dénominateur** qui convertit une pénalité
# par bande en gain attendu, et il est gratuit ici.
./target/release/gen_games_isdd --check "$OUT"

# Corpus retenu, **autre seed** — un entraînement et son évaluation ne doivent
# pas partager une donne. 20 000 donnes suffisent : le bench est déterministe.
HELD=${HELD:-data/training/heldout_matches_20k.bin}
if [ ! -f "$HELD" ]; then
  echo "corpus retenu : 20000 donnes -> $HELD"
  ./target/release/gen_games_isdd \
    --bot arena/bots/gen_match.toml --match-mode \
    --deals 20000 --threads "$THREADS" --seed 90211 --out "$HELD"
  ./target/release/gen_games_isdd --check "$HELD"
fi
