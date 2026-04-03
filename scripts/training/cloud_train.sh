#!/bin/bash
# Universal cloud training entrypoint for RunPod pods.
# Reads configuration from environment variables.
set -ex

RELEASE_BASE="https://github.com/Avo-k/colver/releases/download/train-v1"

echo "=== Colver Belief Net Training ==="
echo "Race: ${RACE_NAME:-unknown}"
echo "Args: ${RACE_ARGS}"
echo "Common: epochs=${EPOCHS:-50} bs=${BATCH_SIZE:-512} lr=${LR:-3e-4}"

# Download training binary
echo "Downloading training binary..."
curl -sL "${RELEASE_BASE}/train_belief_net" -o /usr/local/bin/train_belief_net
chmod +x /usr/local/bin/train_belief_net

# Download replay data
echo "Downloading replay data..."
mkdir -p /data /models
curl -sL "${RELEASE_BASE}/games_500k.bin" -o /data/games_500k.bin

echo "Binary: $(ls -lh /usr/local/bin/train_belief_net)"
echo "Data:   $(ls -lh /data/games_500k.bin)"

# Check GPU
nvidia-smi || echo "WARNING: no GPU detected"

# Run training
echo "Starting training..."
train_belief_net \
    --replays /data/games_500k.bin \
    --epochs "${EPOCHS:-50}" \
    --batch-size "${BATCH_SIZE:-512}" \
    --lr "${LR:-3e-4}" \
    --cosine-lr \
    --warmup-epochs "${WARMUP:-5}" \
    --seed "${SEED:-42}" \
    --val-split 0.05 \
    ${RACE_ARGS} \
    --output "/models/race_${RACE_NAME:-unknown}.bin"

echo ""
echo "=== Training complete ==="
ls -lh /models/

# Keep pod alive so logs/files can be retrieved
echo "Pod staying alive for 10 min for file retrieval..."
sleep 600
