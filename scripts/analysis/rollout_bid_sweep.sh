#!/usr/bin/env bash
# Trier les architectures du bidder par simulation, avant de valider la survivante.
#
# ## Le témoin
#
# `v6_isdd_75M` est **ce bot moins la simulation** : même réseau d'enchère,
# même joueur de cartes. Chaque h2h mesure donc la simulation et rien d'autre —
# c'est la seule comparaison dont on puisse tirer une conclusion sur elle.
#
# ## Quelle statistique lire
#
# Pas le « RESULT: x% » des matchs : à 30 matchs par direction, son erreur type
# est de ~6,5 points de pourcentage, donc il ne voit qu'un désastre. Lire la
# ligne **« Par donne »**, qui porte son IC95 — ~780 donnes à ce budget, soit
# ±3,5 pp. C'est ce qu'un criblage peut faire : éliminer ce qui est franchement
# moins bon, **jamais** couronner un gagnant. Un écart de 2 pp par donne survit
# à ce filtre sans rien prouver.
#
# ## Pourquoi `--no-save`
#
# Un criblage n'a pas sa place dans `arena/results/matches.csv` : ses lignes s'y
# liraient plus tard comme des mesures. Seule la validation finale s'écrit, et
# elle se lance à la main.
#
# ## ⚠️ Le sidecar
#
# Les bots par défaut échantillonnent leurs mondes sur playgen : `playgen-up`
# avant, `playgen-down` après. Sans lui ils refusent de se construire
# (`fallback = "strict"`), ce qui est voulu — un repli silencieux sur des mondes
# uniformes ferait mesurer un autre bot que celui du fichier.
#
#     playgen-up && scripts/analysis/rollout_bid_sweep.sh 30 ; playgen-down
#     REF=v6_isdd_75M BOTS="rollout_probe_512" scripts/analysis/rollout_bid_sweep.sh 300 --save
set -euo pipefail

MATCHES=${1:-30}
SAVE=${2:-}
REF=${REF:-v6_isdd_75M}
# Trois A/B, un par levier : le budget (128 contre 512), l'origine des mondes
# (playgen contre uniforme), et la présélection (sondage contre top-K).
BOTS=${BOTS:-"rollout_probe_128 rollout_probe_512 rollout_probe_512_unif rollout_top_512"}
THREADS=${THREADS:-$(nproc)}
SEED=${SEED:-42}

SAVE_FLAG="--no-save"
[ "$SAVE" = "--save" ] && SAVE_FLAG=""

cargo build --release --bin arena

echo "témoin : $REF   |   $MATCHES matchs/direction   |   $THREADS fils   |   graine $SEED"
[ -n "$SAVE_FLAG" ] && echo "criblage : rien n'est écrit dans matches.csv"
echo

for bot in $BOTS; do
  echo "════════════════════════════════════════════════════════════════"
  ./target/release/arena h2h "$bot" "$REF" \
      --matches "$MATCHES" --threads "$THREADS" --seed "$SEED" $SAVE_FLAG \
    | grep -E "ARENA H2H|RESULT|Avg margin|Par donne|Note à la marge|Wall"
  echo
done
