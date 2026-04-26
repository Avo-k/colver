#!/usr/bin/env bash
# v6 ISDD bid training — RESUME from 45M checkpoint, continue to 75M total.
#
# Original run stopped at 45M / 75M with ε=0.064, LR=2.3e-4 (cosine 60%).
# This continues 30M more steps with matched ε/LR to avoid policy degradation:
#   * eps-start 0.064 → eps-end 0.02 over 30M (matches schedule tail)
#   * lr 2.3e-4 → 3e-5 over 30M (fresh cosine, end matches original)
#   * save-dir is new (no clobbering of old checkpoints)
#
# Caveat: replay buffer starts empty (~10k step warmup), EMA target reinit.
# Effectively a brief fine-tune restart at the loaded weights. Should be minor.

set -u
cd "$(dirname "$0")/../.."

mkdir -p logs models

LOG=logs/v6_isdd_resume.log
SAVE_DIR=models/bid_v6_isdd_resume
RESUME_CKPT=models/bid_v6_isdd/bid_nn_45000000.safetensors

POOL=data/deals/base_5M.bin
SCORES=data/deals/scores_isdd_5M.sc

BASELINE_BID=models/bid_v5_isdd/bid_nn_final.bin
BASELINE_HIDDEN=512

echo "[$(date '+%F %T')] === v6 ISDD RESUME training START ===" | tee -a "$LOG"
echo "  resume:   $RESUME_CKPT" | tee -a "$LOG"
echo "  pool:     $POOL" | tee -a "$LOG"
echo "  scores:   $SCORES" | tee -a "$LOG"
echo "  save_dir: $SAVE_DIR" | tee -a "$LOG"
echo "  baseline: $BASELINE_BID (h=$BASELINE_HIDDEN)" | tee -a "$LOG"

./target/release/train_bid_nn \
  --num-envs 256 \
  --hidden 512 \
  --layers 3 \
  --resume "$RESUME_CKPT" \
  --lr 2.3e-4 \
  --lr-end 3e-5 \
  --eps-start 0.064 \
  --eps-end 0.02 \
  --eps-decay-steps 30000000 \
  --steps 30000000 \
  --reward real \
  --score-aware \
  --sa-features-v3 \
  --match-sim \
  --reward-clip 1.0 \
  --ema-tau 0.005 \
  --buffer-size 500000 \
  --min-buffer 10000 \
  --pool-file "$POOL" \
  --scores "$SCORES" \
  --eval-freq 2500000 \
  --save-freq 2500000 \
  --eval-matches 1000 \
  --save-dir "$SAVE_DIR" \
  --eval-baseline-bid "$BASELINE_BID" \
  --eval-baseline-hidden "$BASELINE_HIDDEN" \
  2>&1 | tee -a "$LOG"

RC=$?
echo "[$(date '+%F %T')] === v6 ISDD RESUME training END (rc=$RC) ===" | tee -a "$LOG"
exit $RC
