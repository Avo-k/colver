#!/usr/bin/env bash
# Génère un corpus de donnes complètes jouées par IS-DD, de bout en bout :
# sidecar → génération → vérification → restitution de la VRAM.
#
# Les trois disciplines que ce script existe pour ne pas oublier, toutes payées
# une fois cette nuit-là (docs/data_gen/isdd_games.md) :
#
#   1. **Un sidecar oisif ne rend jamais sa VRAM.** Trois d'entre eux occupaient
#      21 Go des 24 d'une carte, et le run actif n'a pas planté — il a tourné
#      30 % moins vite. Le script vérifie la VRAM libre AVANT de commencer, et
#      ne tue que le sidecar qu'il a lui-même démarré.
#   2. **Ne pas partager le GPU d'un run en cours.** Un second processus sur la
#      même carte a suffi à faire expirer les lectures et à arrêter une
#      génération de 28 000 donnes à 5 076. Le script refuse de démarrer si une
#      génération tourne déjà.
#   3. **Un corpus se valide chez son consommateur.** `--check` dit que c'est un
#      corpus ; `validate_games_file` dit qu'il est utilisable, en vérifiant que
#      la carte jouée est toujours dans le masque visible par l'observateur.
#
# Usage :
#   scripts/training/gen_isdd_corpus.sh --deals 100000 [options]
#
#   --deals N        nombre de donnes (obligatoire)
#   --dets D         mondes par décision (défaut 40)
#   --out PATH       fichier COLVGM01 (défaut data/training/isdd_games_<N>.bin)
#   --threads T      défaut 256 — très au-dessus de nproc, exprès
#   --gpu-host H     hôte du sidecar (défaut : $COLVER_GEN_GPU_HOST ou localhost)
#   --port P         port du sidecar de génération (défaut 8013 — PAS 8003, la prod)
#   --model PATH     modèle playgen sur l'hôte GPU
#                    ($COLVER_GEN_SIDECAR_BIN pour le binaire du sidecar)
#   --keep-sidecar   ne pas l'arrêter à la fin
set -euo pipefail
cd "$(dirname "$0")/../.."

DEALS=""; DETS=40; OUT=""; THREADS=256
GPU_HOST="${COLVER_GEN_GPU_HOST:-localhost}"
PORT=8013
MODEL="${COLVER_GEN_PLAYGEN_MODEL:-models/playgen/playgen_v2_final.bin}"
# Le binaire du sidecar SUR L'HÔTE GPU. Il se construit là-bas (candle+CUDA), pas
# ici : `cargo build --release --bin playgen_gpu_server --features gpu_server`
# avec `nvcc` dans le PATH et `CUDARC_CUDA_VERSION` posé.
SIDECAR_BIN="${COLVER_GEN_SIDECAR_BIN:-\$HOME/colver-bench/target/release/playgen_gpu_server}"
KEEP=0

while [ $# -gt 0 ]; do
  case "$1" in
    --deals) DEALS="$2"; shift 2 ;;
    --dets) DETS="$2"; shift 2 ;;
    --out) OUT="$2"; shift 2 ;;
    --threads) THREADS="$2"; shift 2 ;;
    --gpu-host) GPU_HOST="$2"; shift 2 ;;
    --port) PORT="$2"; shift 2 ;;
    --model) MODEL="$2"; shift 2 ;;
    --keep-sidecar) KEEP=1; shift ;;
    *) echo "argument inconnu : $1" >&2; exit 2 ;;
  esac
done
[ -n "$DEALS" ] || { echo "usage: $0 --deals N [options]" >&2; exit 2; }
OUT="${OUT:-data/training/isdd_games_${DEALS}.bin}"
URL="http://${GPU_HOST}:${PORT}"

# `ssh localhost` marche rarement ; sur place on exécute directement.
run_gpu() { if [ "$GPU_HOST" = "localhost" ]; then bash -c "$1"; else ssh "$GPU_HOST" "$1"; fi; }

# ── Discipline 2 : une seule génération à la fois ──────────────────────────
if pgrep -f "gen_games_isdd --deals" > /dev/null; then
  echo "❌ une génération tourne déjà — la laisser finir (cf. l'en-tête de ce script)" >&2
  exit 1
fi
if [ -e "${OUT}.0000" ]; then
  echo "❌ des éclats d'un run précédent existent en ${OUT}.* — les rassembler d'abord :" >&2
  echo "   cargo run --release --bin gen_games_isdd -- --merge $OUT --out $OUT" >&2
  exit 1
fi

echo "▸ construction"
cargo build --release --features parallel --bin gen_games_isdd -q

# ── Discipline 1 : de la VRAM libre, et un sidecar qu'on possède ───────────
STARTED=0
if curl -s -m 3 "$URL/health" > /dev/null 2>&1; then
  echo "▸ sidecar déjà en ligne sur $URL — on ne le touchera pas"
else
  FREE=$(run_gpu 'nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits' | head -1)
  echo "▸ VRAM libre sur $GPU_HOST : ${FREE} Mio"
  if [ "${FREE:-0}" -lt 10000 ]; then
    echo "❌ moins de 10 Go libres — un lot de 4096 lanes en demande ~9. Tuer les sidecars oisifs." >&2
    run_gpu 'nvidia-smi --query-compute-apps=pid,used_memory --format=csv' >&2
    exit 1
  fi
  echo "▸ démarrage du sidecar sur $URL"
  run_gpu "nohup setsid $SIDECAR_BIN \
      --playgen '$MODEL' --bind 0.0.0.0 --port $PORT --lane-budget 4096 --handlers 512 \
      > /tmp/playgen_gen_${PORT}.log 2>&1 < /dev/null & disown" || true
  for _ in $(seq 30); do
    sleep 2
    curl -s -m 3 "$URL/health" > /dev/null 2>&1 && break
  done
  curl -s -m 3 "$URL/health" > /dev/null || { echo "❌ le sidecar n'a pas répondu" >&2; exit 1; }
  STARTED=1
fi
# Rendre la VRAM quoi qu'il arrive — y compris sur Ctrl-C — mais seulement si
# c'est nous qui l'avons prise.
cleanup() {
  if [ "$STARTED" = 1 ] && [ "$KEEP" = 0 ]; then
    echo "▸ arrêt du sidecar (VRAM rendue)"
    run_gpu "pkill -f 'port $PORT'" || true
  fi
}
trap cleanup EXIT INT TERM

echo "▸ $DEALS donnes, $DETS mondes/décision, $THREADS threads → $OUT"
COLVER_PLAYGEN_GPU_URL="$URL" ./target/release/gen_games_isdd \
  --deals "$DEALS" --dets "$DETS" --threads "$THREADS" \
  --shard 5000 --progress-every $(( DEALS / 20 > 0 ? DEALS / 20 : 1 )) --out "$OUT"

# ── Discipline 3 : valider chez le consommateur ────────────────────────────
echo "▸ relecture"
./target/release/gen_games_isdd --check "$OUT"
echo "▸ validation par le tokeniseur playgen"
COLVER_GAMES="$PWD/$OUT" cargo test -p colver-core --release --lib \
  validate_games_file -- --ignored --nocapture 2>&1 | grep -E "^validated|test result"

echo "✅ $OUT prêt pour : train_playgen --games $OUT"
