#!/usr/bin/env bash
# Bumblebid reward mix sweep: DD vs real DouDou50 points
# Uses d=64 L=2 H=4 (proven best architecture)
set -euo pipefail

POOL="data/pools/bumblebid_5M_enriched.bin"
DD_POOL="data/pools/dd_5M_enriched.bin"
OPPONENT="models/bid_v2/bid_nn_final.bin"
BASE_DIR="models/bumblebid/reward_sweep"
STEPS=1000000
EVAL_FREQ=50000
SAVE_FREQ=500000
N_ENVS=512
BATCH=1024

mkdir -p "$BASE_DIR"

run_mix() {
    local name="$1"
    local mix="$2"
    local log="$BASE_DIR/${name}.log"
    mkdir -p "$BASE_DIR/$name"
    echo "=========================================="
    echo "  Experiment: $name (reward_mix=$mix)"
    echo "  Log: $log"
    echo "=========================================="

    PYTHONUNBUFFERED=1 PYTHONPATH=scripts \
        python -m bumblebid.train \
        --pool-file "$POOL" --dd-pool "$DD_POOL" --opponent "$OPPONENT" \
        --d-model 64 --n-layers 2 --n-heads 4 \
        --lr 3e-4 --eps-decay-steps 200000 --buffer-size 500000 \
        --n-envs $N_ENVS --batch-size $BATCH --steps $STEPS \
        --eval-freq $EVAL_FREQ --save-freq $SAVE_FREQ \
        --save-dir "$BASE_DIR/$name" \
        --opponent-decay-steps 10000000 \
        --reward-mix "$mix" \
        2>&1 | tee "$log"

    echo ""
}

# 100% DD (baseline — should match previous sweep results)
run_mix "mix_100dd" 1.0

# 100% real DouDou50 points
run_mix "mix_100real" 0.0

# 50/50 blend
run_mix "mix_50_50" 0.5

# 75% real / 25% DD
run_mix "mix_75real" 0.25

echo "=== SWEEP COMPLETE ==="
echo ""

# Now run arena eval on each
echo "=== ARENA EVALUATION ==="
for name in mix_100dd mix_100real mix_50_50 mix_75real; do
    model="$BASE_DIR/$name/latest.pt"
    if [ -f "$model" ]; then
        echo "--- $name ---"
        PYTHONPATH=scripts python scripts/bumblebid/arena_eval.py \
            --model "$model" \
            --d-model 64 --n-layers 2 --n-heads 4 \
            --play-method dmc --play-model models/dmc_50.bin \
            --matches 200 2>&1 | tee -a "$BASE_DIR/${name}_arena.log"
        echo ""
    fi
done

echo "=== ALL DONE ==="
