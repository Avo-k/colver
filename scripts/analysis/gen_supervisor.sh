#!/usr/bin/env bash
# Relance `gen_score_layer` s'il disparaît, pendant un run non surveillé de plusieurs
# jours.
#
# Le chien de garde du sidecar (`sidecar_watchdog.sh`) couvre l'autre moitié du
# problème : si le sidecar meurt, il revient. Mais si c'est le GÉNÉRATEUR qui tombe —
# panique, OOM, budget d'erreurs dépassé après une longue coupure — plus rien ne
# calcule, et une nuit entière peut se perdre entre deux regards. La reprise rend le
# redémarrage gratuit : la couche est réécrite tous les `--checkpoint` donnes et un
# relancement repart du préfixe dense.
#
# **Non invasif** : le script ne lance rien tant qu'un `gen_score_layer` tourne, donc il
# s'attache à un run déjà en cours.
#
# **Borné** : un plafond de redémarrages et un délai croissant. Sans ça, une cause
# permanente — pool illisible, sidecar définitivement mort, disque plein — produirait
# une boucle qui remplit les journaux et masque la panne au lieu de la signaler.
#
#   scripts/analysis/gen_supervisor.sh <log-du-generateur> [max-relances]
#
set -u
LOG="${1:?usage: gen_supervisor.sh <log> [max]}"
MAX="${2:-20}"
REPO="${COLVER_REPO:-$HOME/code/colver}"
SUP="${LOG%.log}_supervisor.log"

# La commande exacte du run. Gardée ici en toutes lettres plutôt que reconstruite : une
# relance qui change un paramètre en silence produirait une couche à deux régimes, et
# rien dans le fichier ne le dirait.
ARGS=(--pool data/deals/base_5M.bin --offset 0 --count 500000
      --threads 160 --checkpoint 500
      --out data/deals/scores_isdd_v2.sc
      --url "http://localhost:8003,http://localhost:8003,http://localhost:8003,http://192.168.1.23:8003")

stamp() { date +'%Y-%m-%d %H:%M:%S'; }
n=0

echo "$(stamp) superviseur armé (max $MAX relances)" >>"$SUP"
while true; do
  # `pgrep -x` compare au nom tronqué à 15 caractères par Linux ; « gen_score_layer »
  # en fait exactement 15, donc il correspond. Un binaire plus long ne correspondrait
  # pas et le superviseur relancerait en boucle sur un processus bien vivant.
  if ! pgrep -x gen_score_layer >/dev/null 2>&1; then
    if [ "$n" -ge "$MAX" ]; then
      echo "$(stamp) $MAX relances atteintes — arrêt du superviseur, la panne est permanente" >>"$SUP"
      exit 1
    fi
    n=$((n + 1))
    echo "$(stamp) générateur absent — relance n°$n" >>"$SUP"
    ( cd "$REPO" && setsid nohup ./target/release/gen_score_layer "${ARGS[@]}" \
        >>"$LOG" 2>&1 </dev/null & )
    # Laisser le temps du chargement du pool (105 Mo) avant de reconclure à l'absence,
    # et espacer les tentatives si elles échouent coup sur coup.
    sleep $((30 + n * 15))
  fi
  sleep 30
done
