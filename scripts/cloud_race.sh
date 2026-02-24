#!/bin/bash
# cloud_race.sh — Run belief net training races in parallel on cloud GPUs
#
# Prerequisites:
#   1. Build and push the training image:
#        docker build -f Dockerfile.train -t colver-train .
#        docker tag colver-train <registry>/colver-train:latest
#        docker push <registry>/colver-train:latest
#
#   2. Upload replay data to a public URL or cloud storage:
#        # Option A: RunPod network volume
#        # Option B: S3/GCS bucket
#        # Option C: GitHub release asset
#
#   3. Install runpodctl or vastai CLI
#
# Usage:
#   ./scripts/cloud_race.sh [provider]   # provider = runpod|vast|local
#
# Each race runs as a separate container/pod with its own GPU.
# Results are written to mounted volume or downloaded after completion.

set -euo pipefail

IMAGE="${TRAIN_IMAGE:-colver-train:latest}"
DATA_PATH="${DATA_PATH:-/data/games_500k.bin}"
OUTPUT_DIR="${OUTPUT_DIR:-/models}"
EPOCHS="${EPOCHS:-50}"
SEED="${SEED:-42}"

# Common training args
COMMON="--replays $DATA_PATH --epochs $EPOCHS --batch-size 512 --lr 3e-4 \
  --cosine-lr --warmup-epochs 5 --seed $SEED --val-split 0.05"

# Define all race variants
declare -A RACES
RACES[baseline]="--v2 --augment --output $OUTPUT_DIR/race_baseline.bin"
RACES[v3]="--v3 --augment --output $OUTPUT_DIR/race_v3.bin"
RACES[crossattn]="--v2 --augment --variant cross_attn --output $OUTPUT_DIR/race_crossattn.bin"
RACES[auxloss]="--v2 --augment --variant aux_loss --output $OUTPUT_DIR/race_auxloss.bin"
RACES[wide]="--v2 --augment --variant var_mlp --num-layers 1 --hidden 768 --output $OUTPUT_DIR/race_wide.bin"
RACES[narrow]="--v2 --augment --variant var_mlp --num-layers 3 --hidden 256 --output $OUTPUT_DIR/race_narrow.bin"
RACES[suitshared]="--v2 --variant suit_shared --output $OUTPUT_DIR/race_suitshared.bin"
RACES[countreg]="--v2 --augment --count-reg 0.1 --output $OUTPUT_DIR/race_count_reg.bin"

PROVIDER="${1:-local}"

run_local() {
    echo "=== Running all races locally (sequential) ==="
    for name in "${!RACES[@]}"; do
        echo ""
        echo "=== Race: $name ==="
        docker run --gpus all \
            -v "$(pwd)/data:/data" \
            -v "$(pwd)/models:/models" \
            "$IMAGE" $COMMON ${RACES[$name]}
    done
}

run_runpod() {
    echo "=== Launching races on RunPod (parallel) ==="
    echo "Requires: RUNPOD_API_KEY env var, runpodctl installed"
    echo "Image: $IMAGE (must be pushed to a registry)"
    echo ""

    PIDS=()
    for name in "${!RACES[@]}"; do
        echo "Launching pod: race-$name"
        # RunPod GPU pod via API
        # Adjust gpu_type_id for your needs:
        #   NVIDIA RTX 3090: "NVIDIA GeForce RTX 3090"
        #   NVIDIA RTX 4090: "NVIDIA GeForce RTX 4090"
        #   NVIDIA A4000:    "NVIDIA RTX A4000"
        runpodctl create pod \
            --name "race-$name" \
            --gpuType "NVIDIA GeForce RTX 3090" \
            --gpuCount 1 \
            --imageName "$IMAGE" \
            --volumeSize 50 \
            --args "$COMMON ${RACES[$name]}" \
            &
        PIDS+=($!)
    done

    echo ""
    echo "Waiting for all pods to launch..."
    for pid in "${PIDS[@]}"; do wait "$pid"; done
    echo "All pods launched. Monitor at https://www.runpod.io/console/pods"
}

run_vast() {
    echo "=== Launching races on vast.ai (parallel) ==="
    echo "Requires: vastai CLI installed and configured"
    echo "Image: $IMAGE (must be pushed to a registry)"
    echo ""

    for name in "${!RACES[@]}"; do
        echo "Searching for GPU and launching: race-$name"
        # Find cheapest 3090 offer
        OFFER=$(vastai search offers 'gpu_name=RTX 3090 num_gpus=1 dph<0.30 inet_down>200' \
            --order 'dph' --limit 1 --raw | head -1 | awk '{print $1}')

        if [ -n "$OFFER" ]; then
            vastai create instance "$OFFER" \
                --image "$IMAGE" \
                --disk 50 \
                --label "race-$name" \
                --args "$COMMON ${RACES[$name]}"
            echo "  → Launched on offer $OFFER"
        else
            echo "  → No suitable offer found for race-$name"
        fi
    done
    echo ""
    echo "Monitor at https://cloud.vast.ai/instances/"
}

case "$PROVIDER" in
    local)  run_local ;;
    runpod) run_runpod ;;
    vast)   run_vast ;;
    *)      echo "Usage: $0 [local|runpod|vast]"; exit 1 ;;
esac

echo ""
echo "=== Race launcher complete ==="
