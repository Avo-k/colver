#!/usr/bin/env bash
# Bumblebid architecture + hyperparameter sweep
# Run from repo root: bash scripts/bumblebid/sweep.sh
set -euo pipefail

POOL="data/pools/bumblebid_2.5M.bin"
DD_POOL="data/pools/dd_2.5M.bin"
OPPONENT="models/bid_v2/bid_nn_final.bin"
BASE_DIR="models/bumblebid/sweep"
STEPS=500000
EVAL_FREQ=50000
SAVE_FREQ=500000
N_ENVS=512
BATCH=1024

mkdir -p "$BASE_DIR"

# Summary file
SUMMARY="$BASE_DIR/summary.txt"
echo "=== Bumblebid Sweep — $(date) ===" > "$SUMMARY"
echo "" >> "$SUMMARY"

run_experiment() {
    local name="$1"
    local d_model="$2"
    local n_layers="$3"
    local n_heads="$4"
    local lr="$5"
    local eps_decay="$6"
    local buffer="$7"
    local extra_args="${8:-}"

    local save_dir="$BASE_DIR/$name"
    local log="$BASE_DIR/${name}.log"
    mkdir -p "$save_dir"

    echo "=========================================="
    echo "  Experiment: $name"
    echo "  d=$d_model L=$n_layers H=$n_heads lr=$lr eps_decay=$eps_decay buf=$buffer"
    echo "  Log: $log"
    echo "=========================================="

    PYTHONUNBUFFERED=1 PYTHONPATH=scripts \
        python -m bumblebid.train \
        --pool-file "$POOL" \
        --dd-pool "$DD_POOL" \
        --opponent "$OPPONENT" \
        --d-model "$d_model" \
        --n-layers "$n_layers" \
        --n-heads "$n_heads" \
        --lr "$lr" \
        --eps-decay-steps "$eps_decay" \
        --buffer-size "$buffer" \
        --n-envs "$N_ENVS" \
        --batch-size "$BATCH" \
        --steps "$STEPS" \
        --eval-freq "$EVAL_FREQ" \
        --save-freq "$SAVE_FREQ" \
        --save-dir "$save_dir" \
        --opponent-decay-steps 10000000 \
        $extra_args \
        2>&1 | tee "$log"

    # Extract last eval score
    local last_eval
    last_eval=$(grep ">>> Eval" "$log" | tail -1 | grep -oP '[+-]\d+\.\d+' || echo "N/A")
    echo "$name | d=$d_model L=$n_layers H=$n_heads lr=$lr eps=$eps_decay buf=$buffer | eval=$last_eval" >> "$SUMMARY"
    echo ""
}

echo ""
echo "============================================"
echo "  ROUND 1: Architecture sweep"
echo "  (fixed: lr=3e-4, eps_decay=200K, buf=500K)"
echo "============================================"
echo ""

# ~105K params
run_experiment "arch_d64_L2_H4" \
    64 2 4 \
    3e-4 200000 500000

# ~232K params
run_experiment "arch_d96_L2_H4" \
    96 2 4 \
    3e-4 200000 500000

# ~408K params
run_experiment "arch_d128_L2_H4" \
    128 2 4 \
    3e-4 200000 500000

# ~604K params (matches bid_v2 param count)
run_experiment "arch_d128_L3_H4" \
    128 3 4 \
    3e-4 200000 500000

# ~1.6M params (half of original)
run_experiment "arch_d256_L2_H8" \
    256 2 8 \
    3e-4 200000 500000

echo ""
echo "============================================"
echo "  ROUND 1 COMPLETE — checking best arch"
echo "============================================"
echo ""

# Pick best architecture by final eval score
best_name=""
best_score=-999
for log_file in "$BASE_DIR"/arch_*.log; do
    name=$(basename "$log_file" .log)
    score=$(grep ">>> Eval" "$log_file" | tail -1 | grep -oP '[+-]\d+\.\d+' || echo "-999")
    echo "  $name: last eval = $score"
    if (( $(echo "$score > $best_score" | bc -l) )); then
        best_score="$score"
        best_name="$name"
    fi
done
echo ""
echo "  Best arch: $best_name (eval=$best_score)"
echo "" >> "$SUMMARY"
echo "Best arch: $best_name (eval=$best_score)" >> "$SUMMARY"

# Extract best arch params
best_d=$(echo "$best_name" | grep -oP 'd\K\d+')
best_l=$(echo "$best_name" | grep -oP 'L\K\d+')
best_h=$(echo "$best_name" | grep -oP 'H\K\d+')

echo ""
echo "============================================"
echo "  ROUND 2: Hyperparameter sweep on $best_name"
echo "  (d=$best_d L=$best_l H=$best_h)"
echo "============================================"
echo ""
echo "" >> "$SUMMARY"
echo "ROUND 2: Hyperparams on $best_name" >> "$SUMMARY"

# Try higher LR
run_experiment "hp_lr1e3" \
    "$best_d" "$best_l" "$best_h" \
    1e-3 200000 500000

# Try lower LR
run_experiment "hp_lr1e4" \
    "$best_d" "$best_l" "$best_h" \
    1e-4 200000 500000

# Try slower epsilon decay (more exploration)
run_experiment "hp_eps500k" \
    "$best_d" "$best_l" "$best_h" \
    3e-4 500000 500000

# Try larger buffer
run_experiment "hp_buf2M" \
    "$best_d" "$best_l" "$best_h" \
    3e-4 200000 2000000

echo ""
echo "============================================"
echo "  ROUND 2 COMPLETE"
echo "============================================"
echo ""

# Final summary
echo ""
echo "============================================"
echo "  FINAL SUMMARY"
echo "============================================"
cat "$SUMMARY"
echo ""
echo "Detailed logs in: $BASE_DIR/"
echo "Done at $(date)"
