#!/bin/bash
# Post-training pipeline for bid_v4_score_aware:
# 1. Wait for training to finish
# 2. Re-eval each 2M checkpoint with 1000 matches
# 3. Run arena round-robin with new scoring rules

set -e
cd /home/avok/code/colver

LOG="models/bid_v4_score_aware/post_training.log"
mkdir -p models/bid_v4_score_aware
exec > >(tee -a "$LOG") 2>&1

echo "=== Post-training pipeline started $(date) ==="

# ──────────────────────────────────────────────────────────────────
# Step 1: Wait for training to finish (check for final checkpoint)
# ──────────────────────────────────────────────────────────────────
echo "Waiting for training to complete..."
while [ ! -f models/bid_v4_score_aware/bid_nn_final.safetensors ]; do
    sleep 120
done
echo "Training complete at $(date)"
sleep 10  # let files flush

# ──────────────────────────────────────────────────────────────────
# Step 2: Re-eval each 2M checkpoint with 1000 full matches
# ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Re-evaluating checkpoints (1000 matches each) ==="

PLAY_MODEL="models/play_v2/play_final.bin"
BASELINE_BID="models/bid_v3_max_20M/bid_nn_final.bin"

# Build the eval binary (same as calibrate_winprob but for h2h)
# We'll use the arena binary for this since it supports full match play
cargo build --bin arena --release 2>/dev/null

# Create bot configs for each checkpoint
for STEP in 2000000 4000000 6000000 8000000 10000000 12000000 14000000 16000000 18000000 20000000; do
    SAFETENSORS="models/bid_v4_score_aware/bid_nn_${STEP}.safetensors"
    BIN="models/bid_v4_score_aware/bid_nn_${STEP}.bin"

    if [ ! -f "$BIN" ] && [ ! -f "$SAFETENSORS" ]; then
        echo "  Skipping step $STEP (no checkpoint)"
        continue
    fi

    if [ ! -f "$BIN" ]; then
        echo "  Skipping step $STEP (no .bin export)"
        continue
    fi

    # Create bot TOML for this checkpoint
    TOML="arena/bots/v4_sa_${STEP}.toml"
    cat > "$TOML" << EOF
# bid_v4_score_aware checkpoint at ${STEP} steps
[bid]
strategy = "nn"
model = "$BIN"
hidden = 512
score_aware = true

[play]
method = "dmc"
model = "$PLAY_MODEL"
residual = true
EOF

    echo "  Evaluating step $STEP..."
    cargo run --bin arena --release -- h2h "v4_sa_${STEP}" nn_v2_dmc50 --matches 500 2>&1 | grep -E "Result|winner|Win"
    echo ""
done

# Also create a bot for the final model
if [ -f "models/bid_v4_score_aware/bid_nn_final.bin" ]; then
    cat > arena/bots/v4_score_aware.toml << 'EOF'
# bid_v4_score_aware final model (score-aware bidder + DouDou50)
[bid]
strategy = "nn"
model = "models/bid_v4_score_aware/bid_nn_final.bin"
hidden = 512
score_aware = true

[play]
method = "dmc"
model = "models/play_v2/play_final.bin"
residual = true
EOF
fi

# ──────────────────────────────────────────────────────────────────
# Step 3: Arena round-robin with key bots (new scoring rules)
# ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Arena round-robin (new scoring rules) ==="
echo "Started at $(date)"

# Core bots to re-evaluate
BOTS="nn_v2_dmc50,nn_v2_dmc35,nn_dmc35"

# Add v4_score_aware final if available
if [ -f "models/bid_v4_score_aware/bid_nn_final.bin" ]; then
    BOTS="$BOTS,v4_score_aware"
fi

# Add nn_v2_isdd bots if they exist
for BOT in nn_v2_isdd nn_v2_isdd_no_belief nn_isdd bid_v3_max_20M_20000000; do
    if [ -f "arena/bots/${BOT}.toml" ]; then
        BOTS="$BOTS,$BOT"
    fi
done

echo "Bots: $BOTS"
cargo run --bin arena --release -- round-robin --matches 200 --bots "$BOTS" 2>&1

echo ""
echo "=== Leaderboard ==="
cargo run --bin arena --release -- results 2>&1

echo ""
echo "=== Phase 1 complete at $(date) ==="

# ──────────────────────────────────────────────────────────────────
# Step 4: Continue training for 20M more steps (40M total)
# ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Continuing training for 20M more steps ==="
echo "Started at $(date)"

# Resume from final checkpoint with low epsilon (fine-tune, not restart)
cargo run -p colver-core --bin train_bid_nn --features dmc_train --release -- \
    --hidden 512 --layers 3 \
    --num-envs 64 --batch-size 512 \
    --steps 20000000 \
    --pool-file data/deals/archive/dd_1.5M_max_dmc_isdd.bin \
    --reward real \
    --score-aware \
    --score-dist data/winprob_points.csv \
    --sa-uniform-ratio 0.2 \
    --save-dir models/bid_v4_score_aware_40M \
    --eval-freq 2000000 --save-freq 2000000 \
    --eval-matches 200 \
    --resume models/bid_v4_score_aware/bid_nn_final.safetensors \
    --eps-start 0.05 --eps-end 0.01 --eps-decay-steps 4000000 2>&1

echo ""
echo "=== 40M training complete at $(date) ==="

# ──────────────────────────────────────────────────────────────────
# Step 5: Eval the 40M model
# ──────────────────────────────────────────────────────────────────
echo ""
echo "=== Evaluating 40M checkpoints ==="

if [ -f "models/bid_v4_score_aware_40M/bid_nn_final.bin" ]; then
    cat > arena/bots/v4_score_aware_40M.toml << 'EOF'
# bid_v4_score_aware 40M steps
[bid]
strategy = "nn"
model = "models/bid_v4_score_aware_40M/bid_nn_final.bin"
hidden = 512

[play]
method = "dmc"
model = "models/play_v2/play_final.bin"
residual = true
EOF

    echo "H2H: v4_score_aware_40M vs nn_v2_dmc50 (500 matches)"
    cargo run --bin arena --release -- h2h v4_score_aware_40M nn_v2_dmc50 --matches 500 2>&1
    echo ""
    echo "H2H: v4_score_aware_40M vs v4_score_aware (500 matches)"
    cargo run --bin arena --release -- h2h v4_score_aware_40M v4_score_aware --matches 500 2>&1
fi

echo ""
echo "=== Full pipeline complete at $(date) ==="
