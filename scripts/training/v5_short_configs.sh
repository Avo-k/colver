#!/usr/bin/env bash
# v5 short stabilization configs: 3 runs × 2M steps, sequential on GPU.
# Compares: A=baseline (v2 features only), B=+reward clip, C=+clip+EMA.

set -e
cd "$(dirname "$0")/../.."

POOL=data/deals/archive/dd_1.5M_max_dmc_isdd.bin
COMMON=(
  --num-envs 256
  --hidden 512
  --layers 3
  --lr 1e-4
  --pool-file "$POOL"
  --reward real
  --score-aware
  --sa-features-v2
  --eps-start 0.25
  --eps-end 0.02
  --eps-decay-steps 1500000
  --steps 2000000
  --eval-freq 500000
  --eval-matches 200
  --save-freq 1000000
  --buffer-size 200000
  --min-buffer 5000
)

run_config() {
  local name=$1
  shift
  local outdir="models/v5_short_${name}"
  local log="logs/v5_short_${name}.log"
  mkdir -p "$outdir" logs
  echo
  echo "=========================================="
  echo "Config $name → $outdir"
  echo "Extra flags: $*"
  echo "Started: $(date '+%F %T')"
  echo "=========================================="
  ./target/release/train_bid_nn \
    "${COMMON[@]}" \
    --save-dir "$outdir" \
    "$@" \
    2>&1 | tee "$log"
  echo "Finished $name: $(date '+%F %T')"
}

run_config A
run_config B --reward-clip 1.0
run_config C --reward-clip 1.0 --ema-tau 0.005

echo
echo "All three configs done."
