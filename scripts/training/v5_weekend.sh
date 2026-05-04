#!/usr/bin/env bash
# v5 weekend training pipeline.
#
# Phase 1 (parallel, ~20h):
#   - CPU: enrich dd_1M_isdd.bin from dd_1.5M_base.bin (~14h)
#   - GPU: train v5_max 20M steps on existing max pool (~20h concurrent)
#
# Phase 2 (sequential, ~26h):
#   - GPU: train v5_isdd 25M steps on new ISDD-pure pool
#
# All steps use: sa-features-v2 + reward-clip 1.0 + ema-tau 0.005 + cosine lr decay.

set -u  # unset var = error; not -e so failure in one stage doesn't kill the pipeline
cd "$(dirname "$0")/../.."

mkdir -p logs models data/pools

LOG_MASTER=logs/v5_weekend_master.log
LOG_ENRICH=logs/v5_weekend_enrich.log
LOG_MAX=logs/v5_weekend_max.log
LOG_ISDD=logs/v5_weekend_isdd.log

MAX_POOL=data/deals/archive/dd_1.5M_max_dmc_isdd.bin
ISDD_POOL=data/deals/archive/dd_1M_isdd.bin
BASE_POOL=data/deals/archive/dd_1.5M_base.bin

COMMON_TRAIN=(
  --num-envs 256
  --hidden 512
  --layers 3
  --lr 3e-4
  --lr-end 3e-5
  --eps-start 0.30
  --eps-end 0.02
  --reward real
  --score-aware
  --sa-features-v2
  --reward-clip 1.0
  --ema-tau 0.005
  --eval-matches 500
  --buffer-size 500000
  --min-buffer 10000
)

log() {
  echo "[$(date '+%F %T')] $*" | tee -a "$LOG_MASTER"
}

log "=== v5 WEEKEND PIPELINE START ==="
log "Phase 1: enrich (CPU) + train v5_max 20M (GPU) in parallel"

# ---- Phase 1a: Enrichment (CPU background) ----
log "Starting enrichment: $ISDD_POOL (1M deals @ 20ms)"
./target/release/enrich_pool_isdd \
  --pool "$BASE_POOL" \
  --output "$ISDD_POOL" \
  --deals 1000000 \
  --time-ms 20 \
  --dets 20 \
  --seed 42 \
  > "$LOG_ENRICH" 2>&1 &
ENRICH_PID=$!
log "Enrichment PID: $ENRICH_PID"

# Small delay so enrichment grabs its thread budget before training launches
sleep 5

# ---- Phase 1b: Train v5_max (GPU parallel) ----
log "Starting v5_max training: 20M steps on $MAX_POOL"
./target/release/train_bid_nn \
  "${COMMON_TRAIN[@]}" \
  --pool-file "$MAX_POOL" \
  --steps 20000000 \
  --eps-decay-steps 15000000 \
  --eval-freq 2000000 \
  --save-freq 2000000 \
  --save-dir models/bid_v5_max \
  > "$LOG_MAX" 2>&1 &
MAX_PID=$!
log "v5_max PID: $MAX_PID"

# Wait for BOTH phase-1 jobs to finish
log "Waiting for enrichment (PID $ENRICH_PID) and v5_max training (PID $MAX_PID)..."
wait "$ENRICH_PID"
ENRICH_RC=$?
log "Enrichment finished (rc=$ENRICH_RC)"
wait "$MAX_PID"
MAX_RC=$?
log "v5_max training finished (rc=$MAX_RC)"

# ---- Phase 2: Train v5_isdd on clean pool ----
if [ ! -f "$ISDD_POOL" ]; then
  log "ERROR: $ISDD_POOL does not exist after Phase 1 (enrich rc=$ENRICH_RC). Falling back to MAX_POOL."
  PHASE2_POOL="$MAX_POOL"
else
  PHASE2_POOL="$ISDD_POOL"
  log "Phase 2: using clean pool $ISDD_POOL"
fi

log "Starting v5_isdd training: 25M steps on $PHASE2_POOL"
./target/release/train_bid_nn \
  "${COMMON_TRAIN[@]}" \
  --pool-file "$PHASE2_POOL" \
  --steps 25000000 \
  --eps-decay-steps 18000000 \
  --eval-freq 2500000 \
  --save-freq 2500000 \
  --save-dir models/bid_v5_isdd \
  > "$LOG_ISDD" 2>&1
ISDD_RC=$?
log "v5_isdd training finished (rc=$ISDD_RC)"

log "=== v5 WEEKEND PIPELINE DONE ==="
log "Results:"
log "  Enrichment: rc=$ENRICH_RC, pool=$ISDD_POOL"
log "  v5_max:     rc=$MAX_RC, dir=models/bid_v5_max"
log "  v5_isdd:    rc=$ISDD_RC, dir=models/bid_v5_isdd (pool=$PHASE2_POOL)"
