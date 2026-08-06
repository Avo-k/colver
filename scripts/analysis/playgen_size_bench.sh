#!/usr/bin/env bash
# Ce que la **taille du sampler playgen** coûte en débit de génération de donnes.
#
# La question : `gen_games_isdd` est limité par le GPU (96 % du temps de thread
# est de l'attente du sidecar, cf. docs/data_gen/isdd_games.md). Le sampler est
# donc le levier, et sa taille le paramètre le plus direct. Trois architectures,
# toutes au format COLVPG02 — seuls `d` et `L` changent :
#
#   v2-belote-small  d=256 L=4   3,2 M paramètres
#   v2        d=384 L=6  10,7 M   ← celui de la prod
#   v2-belote-large  d=512 L=8  25,3 M
#
# ## Trois disciplines, toutes payées d'avance ailleurs
#
#  1. **Jamais deux exécutions séquentielles.** La charge dérive de 20 % sur ces
#     machines, plus que la plupart des écarts mesurés. Les trois modèles sont
#     donc **alternés**, l'ordre tournant à chaque tour, et on lit la médiane.
#  2. **Le client tourne sur la machine de dev, le sidecar sur l'hôte GPU.**
#     Moxxi n'a que 8 cœurs physiques : y mettre aussi les solves DD ferait une
#     mesure bornée par le CPU, c'est-à-dire précisément celle qui ne répond pas
#     à la question. C'est aussi la configuration des chiffres publiés.
#  3. **Port 8013, jamais 8003.** 8003 est le sidecar de production, sur la même
#     carte. Ce script ne tue que le sidecar qu'il a lui-même démarré, Ctrl-C
#     compris — un sidecar oisif ne rend jamais sa VRAM.
#
# Usage :
#   scripts/analysis/playgen_size_bench.sh [--deals 250] [--rounds 3] [--dets 40]
set -euo pipefail
cd "$(dirname "$0")/../.."

GPU_HOST="${COLVER_GEN_GPU_HOST:-moxxi}"
GPU_ADDR="${COLVER_GEN_GPU_ADDR:-192.168.1.23}"
PORT=8013
DEALS=250
ROUNDS=3
DETS=40
THREADS=256
LANES=512
MATCH_MODE=1
OUTDIR="${COLVER_BENCH_OUTDIR:-/tmp/playgen_size_bench}"

while [ $# -gt 0 ]; do
  case "$1" in
    --deals) DEALS="$2"; shift 2 ;;
    --rounds) ROUNDS="$2"; shift 2 ;;
    --dets) DETS="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --lanes) LANES="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --no-match-mode) MATCH_MODE=0; shift ;;
    --outdir) OUTDIR="$2"; shift 2 ;;
    *) echo "argument inconnu : $1" >&2; exit 2 ;;
  esac
done

[ "$PORT" != "8003" ] || { echo "❌ 8003 est le sidecar de PRODUCTION" >&2; exit 1; }

# nom|chemin distant du .bin
MODELS=(
  "v2belote_small|/home/claude/playgen/models-bench/v2belote_small_120000.bin"
  "v2|/home/claude/playgen/models/playgen_v2_final.bin"
  "v2belote_large|/home/claude/playgen/models-bench/v2belote_large2_80000.bin"
)
SIDECAR_BIN="/home/claude/colver-bench/target/release/playgen_gpu_server"
URL="http://${GPU_ADDR}:${PORT}"

mkdir -p "$OUTDIR"
SIDECAR_PID=""

stop_sidecar() {
  [ -n "$SIDECAR_PID" ] || return 0
  ssh "$GPU_HOST" "kill $SIDECAR_PID 2>/dev/null || true"
  # Attendre la restitution effective de la VRAM : un `kill` rend la main bien
  # avant que le contexte CUDA soit démonté, et le sidecar suivant mesurerait
  # alors une carte encore occupée par le précédent.
  for _ in $(seq 30); do
    ssh "$GPU_HOST" "kill -0 $SIDECAR_PID 2>/dev/null" || break
    sleep 1
  done
  SIDECAR_PID=""
}
trap 'stop_sidecar' EXIT INT TERM

start_sidecar() {  # $1 = chemin du modèle
  local model="$1"
  local free
  free=$(ssh "$GPU_HOST" "nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits")
  if [ "$free" -lt 8000 ]; then
    echo "❌ ${free} Mio libres sur le GPU — un sidecar d'expérience traîne ?" >&2
    exit 1
  fi
  SIDECAR_PID=$(ssh "$GPU_HOST" "setsid nohup nice -n 5 $SIDECAR_BIN \
      --playgen $model --port $PORT --lane-budget $LANES \
      > /tmp/bench_sidecar_${PORT}.log 2>&1 < /dev/null & echo \$!")
  for _ in $(seq 60); do
    if curl -s -m 2 "${URL}/health" > /dev/null 2>&1; then return 0; fi
    sleep 1
  done
  echo "❌ le sidecar n'a pas répondu — log distant /tmp/bench_sidecar_${PORT}.log" >&2
  ssh "$GPU_HOST" "tail -20 /tmp/bench_sidecar_${PORT}.log" >&2
  exit 1
}

echo "▸ construction du client"
cargo build --release --features parallel --bin gen_games_isdd -q

RESULTS="$OUTDIR/results.tsv"
: > "$RESULTS"
printf 'round\tmodel\tdeals\twall_s\tdeals_s\tactions_s\twait_pct\tsolve_pct\n' >> "$RESULTS"

for r in $(seq 1 "$ROUNDS"); do
  # Rotation de l'ordre : la dérive de charge ne doit favoriser aucun modèle.
  n=${#MODELS[@]}
  for k in $(seq 0 $((n - 1))); do
    idx=$(( (k + r - 1) % n ))
    name="${MODELS[$idx]%%|*}"
    path="${MODELS[$idx]#*|}"
    log="$OUTDIR/${name}_r${r}.log"
    out="$OUTDIR/${name}_r${r}.bin"
    rm -f "$out" "$out".[0-9]*

    echo "── tour $r · $name ────────────────────────────────"
    start_sidecar "$path"
    set +e
    nice -n 10 ./target/release/gen_games_isdd \
      --deals "$DEALS" --dets "$DETS" --threads "$THREADS" \
      --url "$URL" --out "$out" --seed $((1000 + r)) \
      $([ "$MATCH_MODE" = 1 ] && echo --match-mode) \
      > "$log" 2>&1
    rc=$?
    set -e
    stop_sidecar
    if [ $rc -ne 0 ]; then
      echo "⚠️  run en échec (rc=$rc) — voir $log" >&2
      tail -5 "$log" >&2
      continue
    fi

    line=$(grep -oP '^\d+ donnes en [\d.]+s — [\d.]+ donnes/s, \d+ actions/s' "$log" | tail -1)
    nd=$(echo "$line" | grep -oP '^\d+')
    wall=$(echo "$line" | grep -oP 'en \K[\d.]+')
    dps=$(echo "$line" | grep -oP '— \K[\d.]+')
    aps=$(echo "$line" | grep -oP ', \K\d+(?= actions)')
    wait_p=$(grep -oP 'part attente sidecar : \K[\d.]+' "$log" | tail -1)
    solve_p=$(grep -oP 'part solve DD : \K[\d.]+' "$log" | tail -1)
    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$r" "$name" "$nd" "$wall" "$dps" "$aps" "${wait_p:-}" "${solve_p:-}" >> "$RESULTS"
    echo "   $nd donnes, ${wall}s → ${dps} donnes/s (attente sidecar ${wait_p:-?} %)"
  done
done

echo
echo "▸ résultats bruts : $RESULTS"
column -t "$RESULTS"
