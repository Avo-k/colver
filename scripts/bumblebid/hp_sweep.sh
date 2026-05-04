#!/usr/bin/env bash
# Hyperparam sweep on best architecture (d=64 L=2 H=4)
set -euo pipefail

POOL="data/deals/archive/bumblebid_2.5M.bin"
DD_POOL="data/deals/archive/dd_2.5M.bin"
OPPONENT="models/bid_v2/bid_nn_final.bin"
BASE_DIR="models/bumblebid/sweep"
STEPS=500000
EVAL_FREQ=50000
N_ENVS=512
BATCH=1024

# Best arch from round 1
D=64; L=2; H=4

run_hp() {
    local name="$1"; shift
    local log="$BASE_DIR/${name}.log"
    mkdir -p "$BASE_DIR/$name"
    echo "=== $name ==="
    PYTHONUNBUFFERED=1 PYTHONPATH=scripts \
        python -m bumblebid.train \
        --pool-file "$POOL" --dd-pool "$DD_POOL" --opponent "$OPPONENT" \
        --d-model $D --n-layers $L --n-heads $H \
        --n-envs $N_ENVS --batch-size $BATCH --steps $STEPS \
        --eval-freq $EVAL_FREQ --save-freq $STEPS \
        --save-dir "$BASE_DIR/$name" \
        --opponent-decay-steps 10000000 \
        "$@" \
        2>&1 | tee "$log"
}

# 1. Higher LR
run_hp "hp_lr1e3" --lr 1e-3 --eps-decay-steps 200000 --buffer-size 500000

# 2. Lower LR
run_hp "hp_lr1e4" --lr 1e-4 --eps-decay-steps 200000 --buffer-size 500000

# 3. Slower epsilon decay (more exploration)
run_hp "hp_eps500k" --lr 3e-4 --eps-decay-steps 500000 --buffer-size 500000

# 4. Larger buffer
run_hp "hp_buf2M" --lr 3e-4 --eps-decay-steps 200000 --buffer-size 2000000

echo "=== HP SWEEP DONE ==="
