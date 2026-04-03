#!/bin/bash
# Triforge training: bootstrap play from old bid NN, then alternate.
#
# Phase 0 (bootstrap): Train PLAY against frozen old bid NN (256×2 arch)
# Phase 1+: Alternate bid-only / play-only with new architecture (512×3)
#
# Usage:
#   ./scripts/triforge.sh [--cycles N] [--play-steps M] [--bid-steps M]
#
# Results saved to models/triforge/

set -e

# --- Defaults ---
CYCLES=3
PLAY_STEPS=10000000
BID_STEPS=5000000
NUM_ENVS=256
PLAY_HIDDEN=1024
BID_HIDDEN=512
BID_LAYERS=3
EVAL_MATCHES=500
SAVE_FREQ=500000
EVAL_FREQ=1000000

# Old bid NN for bootstrap (256×2 architecture)
OLD_BID_SAFETENSORS="models/bid_nn_final.safetensors"
OLD_BID_HIDDEN=256
OLD_BID_LAYERS=2

# Eval opponent (fixed reference)
EVAL_PLAY_CHECKPOINT="models/dmc_35.bin"
EVAL_BID_MODEL="models/bid_nn_final.bin"

BASE_DIR="models/triforge"

# --- Parse args ---
while [[ $# -gt 0 ]]; do
    case $1 in
        --cycles) CYCLES="$2"; shift 2;;
        --play-steps) PLAY_STEPS="$2"; shift 2;;
        --bid-steps) BID_STEPS="$2"; shift 2;;
        --num-envs) NUM_ENVS="$2"; shift 2;;
        *) echo "Unknown arg: $1"; exit 1;;
    esac
done

echo "=== Triforge Training ==="
echo "Cycles: $CYCLES"
echo "Play steps/phase: $((PLAY_STEPS / 1000000))M | Bid steps/phase: $((BID_STEPS / 1000000))M"
echo ""

# Common eval args (same for all phases)
EVAL_ARGS="--eval-random-matches $EVAL_MATCHES \
    --eval-checkpoint-matches $EVAL_MATCHES \
    --eval-play-checkpoint $EVAL_PLAY_CHECKPOINT \
    --eval-bid-model $EVAL_BID_MODEL \
    --save-freq $SAVE_FREQ \
    --eval-freq $EVAL_FREQ"

# ==========================================================
#  PHASE 0: Bootstrap PLAY against frozen old bid NN
# ==========================================================
PHASE_DIR="$BASE_DIR/phase0_play"
echo ""
echo "=========================================="
echo "  PHASE 0: Bootstrap PLAY (frozen bid: $OLD_BID_SAFETENSORS)"
echo "  Bid arch: ${OLD_BID_HIDDEN}×${OLD_BID_LAYERS} (old) | Save: $PHASE_DIR"
echo "=========================================="

cargo run -p colver-core --bin train_joint --features dmc_train --release -- \
    --mode play-only \
    --steps $PLAY_STEPS \
    --resume-bid "$OLD_BID_SAFETENSORS" \
    --save-dir "$PHASE_DIR" \
    --num-envs $NUM_ENVS \
    --play-hidden $PLAY_HIDDEN \
    --bid-hidden $OLD_BID_HIDDEN \
    --bid-layers $OLD_BID_LAYERS \
    --play-eps-start 0.15 \
    --play-eps-decay $PLAY_STEPS \
    $EVAL_ARGS

CURRENT_PLAY="$PHASE_DIR/play_final.safetensors"
echo "  Bootstrap play: $CURRENT_PLAY"

# ==========================================================
#  CYCLES 1..N: Alternate bid-only / play-only (new arch)
# ==========================================================
for cycle in $(seq 1 $CYCLES); do
    echo ""
    echo "=========================================="
    echo "  CYCLE $cycle / $CYCLES (new arch: bid ${BID_HIDDEN}×${BID_LAYERS})"
    echo "=========================================="

    # --- Phase A: Train BID against frozen PLAY ---
    PHASE_DIR="$BASE_DIR/cycle${cycle}_bid"
    echo ""
    echo "--- Phase ${cycle}A: Train BID (frozen play: $CURRENT_PLAY) ---"

    cargo run -p colver-core --bin train_joint --features dmc_train --release -- \
        --mode bid-only \
        --steps $BID_STEPS \
        --resume-play "$CURRENT_PLAY" \
        --save-dir "$PHASE_DIR" \
        --num-envs $NUM_ENVS \
        --play-hidden $PLAY_HIDDEN \
        --bid-hidden $BID_HIDDEN \
        --bid-layers $BID_LAYERS \
        --warmup-steps 0 \
        --handover-steps $((BID_STEPS / 4)) \
        --heuristic-floor 0.20 \
        --bid-eps-start 0.30 \
        --bid-eps-decay $BID_STEPS \
        $EVAL_ARGS

    CURRENT_BID="$PHASE_DIR/bid_final.safetensors"
    echo "  Bid v$cycle: $CURRENT_BID"

    # --- Phase B: Train PLAY against frozen BID ---
    PHASE_DIR="$BASE_DIR/cycle${cycle}_play"
    echo ""
    echo "--- Phase ${cycle}B: Train PLAY (frozen bid: $CURRENT_BID) ---"

    cargo run -p colver-core --bin train_joint --features dmc_train --release -- \
        --mode play-only \
        --steps $PLAY_STEPS \
        --resume-play "$CURRENT_PLAY" \
        --resume-bid "$CURRENT_BID" \
        --save-dir "$PHASE_DIR" \
        --num-envs $NUM_ENVS \
        --play-hidden $PLAY_HIDDEN \
        --bid-hidden $BID_HIDDEN \
        --bid-layers $BID_LAYERS \
        --play-eps-start 0.15 \
        --play-eps-decay $PLAY_STEPS \
        $EVAL_ARGS

    CURRENT_PLAY="$PHASE_DIR/play_final.safetensors"
    echo "  Play v$cycle: $CURRENT_PLAY"
done

echo ""
echo "=== Triforge complete ==="
echo "Final play: $CURRENT_PLAY"
echo "Final bid:  $CURRENT_BID"
