#!/usr/bin/env bash
# Chien de garde du sidecar playgen local.
#
# Une génération de couche de scores tourne des jours. Si le sidecar meurt, chaque thread
# du générateur rend une erreur, le budget saute et le run s'arrête — en pleine nuit, ça
# coûte toutes les heures restantes. Ce script le relance.
#
# Il ne surveille QUE le sidecar local : celui de moxxi est un service systemd géré par
# la prod, et le relancer d'ici serait s'immiscer dans un déploiement qu'on ne pilote pas.
#
#   scripts/analysis/sidecar_watchdog.sh <fichier-de-log> &
#
set -u
LOG="${1:-/tmp/sidecar_watchdog.log}"
URL="${COLVER_PLAYGEN_GPU_URL:-http://localhost:8003}"
REPO="${COLVER_REPO:-$HOME/code/colver}"
MODEL="${COLVER_PLAYGEN_MODEL:-$REPO/models/playgen/playgen_v2_final.bin}"
BIN="$REPO/target/release/playgen_gpu_server"

stamp() { date +'%Y-%m-%d %H:%M:%S'; }

while true; do
  if ! curl -s -m 5 "$URL/health" >/dev/null 2>&1; then
    echo "$(stamp) sidecar injoignable — relance" >>"$LOG"
    setsid nohup "$BIN" --playgen "$MODEL" --port 8003 \
      >>"${LOG%.log}_sidecar.log" 2>&1 </dev/null &
    # Le chargement du modèle et le contexte CUDA prennent une dizaine de secondes ;
    # revérifier trop tôt lancerait un second processus sur le même port.
    for _ in $(seq 1 30); do
      sleep 2
      curl -s -m 3 "$URL/health" >/dev/null 2>&1 && break
    done
    if curl -s -m 3 "$URL/health" >/dev/null 2>&1; then
      echo "$(stamp) sidecar de retour : $(curl -s -m 3 "$URL/health")" >>"$LOG"
    else
      echo "$(stamp) ÉCHEC de la relance" >>"$LOG"
    fi
  fi
  sleep 60
done
