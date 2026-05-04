#!/usr/bin/env bash
# Bid v3 reward mix sweep: compare DD vs real vs blend vs curriculum
# Each experiment trains a 512x3 MLP for 5M steps, then evaluates in arena.
#
# Usage:
#   ./scripts/training/bid_v3_sweep.sh [--steps 5000000] [--matches 200]
#
set -euo pipefail

# === Configuration ===
STEPS="${STEPS:-5000000}"
MATCHES="${MATCHES:-200}"
NUM_ENVS=64
HIDDEN=512
LAYERS=3
POOL="data/deals/archive/dd_5M_enriched.bin"
EVAL_FREQ=500000
SAVE_FREQ=1000000
REFERENCE_BOT="nn_v2_dmc50"
PLAY_MODEL="models/play_v2/play_final.bin"
BASE_DIR="models/bid_v3_exp"

# Parse args
while [[ $# -gt 0 ]]; do
    case "$1" in
        --steps) STEPS="$2"; shift 2 ;;
        --matches) MATCHES="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 1 ;;
    esac
done

# === Experiment definitions ===
# Format: name:reward_flag
EXPERIMENTS=(
    "exp_a_dd:dd"
    "exp_b_blend75:blend:0.75"
    "exp_c_blend50:blend:0.5"
    "exp_d_blend25:blend:0.25"
    "exp_e_curriculum:curriculum:0.95:0.3"
)

RESULTS_FILE="${BASE_DIR}/sweep_results.txt"
mkdir -p "$BASE_DIR"

# Header
echo "=================================================================" | tee "$RESULTS_FILE"
echo "  Bid v3 Reward Mix Sweep" | tee -a "$RESULTS_FILE"
echo "  $(date '+%Y-%m-%d %H:%M')" | tee -a "$RESULTS_FILE"
echo "  Steps: $STEPS | Envs: $NUM_ENVS | Hidden: $HIDDEN | Layers: $LAYERS" | tee -a "$RESULTS_FILE"
echo "  Pool: $POOL" | tee -a "$RESULTS_FILE"
echo "  Reference: $REFERENCE_BOT (h2h $MATCHES matches)" | tee -a "$RESULTS_FILE"
echo "=================================================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

SWEEP_START=$(date +%s)

for entry in "${EXPERIMENTS[@]}"; do
    # Parse name and reward flag (first colon separates name from reward)
    EXP_NAME="${entry%%:*}"
    REWARD_FLAG="${entry#*:}"
    SAVE_DIR="${BASE_DIR}/${EXP_NAME}"

    echo "" | tee -a "$RESULTS_FILE"
    echo "=== ${EXP_NAME} (reward: ${REWARD_FLAG}) ===" | tee -a "$RESULTS_FILE"
    echo "  Started: $(date '+%H:%M:%S')" | tee -a "$RESULTS_FILE"
    EXP_START=$(date +%s)

    # --- Train ---
    # Skip training if already done (resume-safe)
    if [[ -f "${SAVE_DIR}/bid_nn_final.bin" ]]; then
        echo "  SKIP training (${SAVE_DIR}/bid_nn_final.bin exists)" | tee -a "$RESULTS_FILE"
    else
        echo "  Training ${STEPS} steps..."
        mkdir -p "$SAVE_DIR"
        cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- \
            --hidden "$HIDDEN" --layers "$LAYERS" \
            --steps "$STEPS" --num-envs "$NUM_ENVS" \
            --pool-file "$POOL" \
            --reward "$REWARD_FLAG" \
            --save-dir "$SAVE_DIR" \
            --eval-freq "$EVAL_FREQ" --save-freq "$SAVE_FREQ" \
            2>&1 | tee "${SAVE_DIR}/training.log"

        EXP_TRAIN_END=$(date +%s)
        TRAIN_MINS=$(( (EXP_TRAIN_END - EXP_START) / 60 ))
        echo "  Training done in ${TRAIN_MINS}m" | tee -a "$RESULTS_FILE"
    fi

    # --- Check model exists ---
    FINAL_BIN="${SAVE_DIR}/bid_nn_final.bin"
    if [[ ! -f "$FINAL_BIN" ]]; then
        echo "  ERROR: ${FINAL_BIN} not found, skipping arena eval" | tee -a "$RESULTS_FILE"
        continue
    fi

    # --- Create arena bot TOML ---
    BOT_NAME="bid_v3_${EXP_NAME}"
    BOT_TOML="arena/bots/${BOT_NAME}.toml"
    cat > "$BOT_TOML" <<EOF
[bid]
strategy = "nn"
model = "${FINAL_BIN}"
hidden = ${HIDDEN}

[play]
method = "dmc"
model = "${PLAY_MODEL}"
residual = true
EOF
    echo "  Created bot: ${BOT_TOML}"

    # --- Arena H2H vs reference ---
    echo "  Arena: ${BOT_NAME} vs ${REFERENCE_BOT} (${MATCHES} matches)..."
    ARENA_OUTPUT=$(cargo run --bin arena --release -- h2h "$BOT_NAME" "$REFERENCE_BOT" --matches "$MATCHES" 2>&1)
    echo "$ARENA_OUTPUT" | tail -10 | tee -a "$RESULTS_FILE"

    # --- Also evaluate intermediate checkpoints if they exist ---
    for ckpt in "${SAVE_DIR}"/bid_nn_*.bin; do
        [[ "$ckpt" == *final* ]] && continue
        [[ "$ckpt" == *latest* ]] && continue
        [[ ! -f "$ckpt" ]] && continue

        STEP_NUM=$(echo "$ckpt" | sed 's/.*bid_nn_\([0-9]*\)\.bin/\1/')
        CKPT_BOT="bid_v3_${EXP_NAME}_${STEP_NUM}"
        CKPT_TOML="arena/bots/${CKPT_BOT}.toml"
        cat > "$CKPT_TOML" <<EOF
[bid]
strategy = "nn"
model = "${ckpt}"
hidden = ${HIDDEN}

[play]
method = "dmc"
model = "${PLAY_MODEL}"
residual = true
EOF
        echo "  Arena: ${CKPT_BOT} vs ${REFERENCE_BOT}..."
        CKPT_RESULT=$(cargo run --bin arena --release -- h2h "$CKPT_BOT" "$REFERENCE_BOT" --matches "$MATCHES" 2>&1)
        echo "    checkpoint ${STEP_NUM}:" | tee -a "$RESULTS_FILE"
        echo "$CKPT_RESULT" | tail -5 | tee -a "$RESULTS_FILE"
    done

    EXP_END=$(date +%s)
    TOTAL_MINS=$(( (EXP_END - EXP_START) / 60 ))
    echo "  Total time for ${EXP_NAME}: ${TOTAL_MINS}m" | tee -a "$RESULTS_FILE"
    echo "" | tee -a "$RESULTS_FILE"
done

# === Final summary ===
SWEEP_END=$(date +%s)
SWEEP_HOURS=$(( (SWEEP_END - SWEEP_START) / 3600 ))
SWEEP_MINS=$(( ((SWEEP_END - SWEEP_START) % 3600) / 60 ))

echo "" | tee -a "$RESULTS_FILE"
echo "=================================================================" | tee -a "$RESULTS_FILE"
echo "  SWEEP COMPLETE — ${SWEEP_HOURS}h${SWEEP_MINS}m total" | tee -a "$RESULTS_FILE"
echo "  $(date '+%Y-%m-%d %H:%M')" | tee -a "$RESULTS_FILE"
echo "=================================================================" | tee -a "$RESULTS_FILE"
echo "" | tee -a "$RESULTS_FILE"

# Round-robin of all sweep bots
echo "=== Final Round-Robin ===" | tee -a "$RESULTS_FILE"
SWEEP_BOTS=""
for entry in "${EXPERIMENTS[@]}"; do
    EXP_NAME="${entry%%:*}"
    BOT_NAME="bid_v3_${EXP_NAME}"
    # Only include if the bot TOML exists
    if [[ -f "arena/bots/${BOT_NAME}.toml" ]]; then
        if [[ -n "$SWEEP_BOTS" ]]; then
            SWEEP_BOTS="${SWEEP_BOTS},${BOT_NAME}"
        else
            SWEEP_BOTS="${BOT_NAME}"
        fi
    fi
done

if [[ -n "$SWEEP_BOTS" ]]; then
    # Add reference bot to round-robin
    SWEEP_BOTS="${SWEEP_BOTS},${REFERENCE_BOT}"
    echo "  Bots: ${SWEEP_BOTS}"
    cargo run --bin arena --release -- round-robin --matches "$MATCHES" --bots "$SWEEP_BOTS" 2>&1 | tee -a "$RESULTS_FILE"
fi

echo ""
echo "Results saved to: ${RESULTS_FILE}"
echo "Training logs in: ${BASE_DIR}/exp_*/training.log"
