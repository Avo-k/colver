#!/usr/bin/env bash
# v6 bid training — ISDD pool, 75M steps, with belote + match-sim fixes.
#
# Changes vs v5_isdd (25M, 1M deals):
#   * Pool: data/deals/base_5M.bin + data/deals/scores_isdd_5M.sc (5× data).
#   * Obs: --sa-features-v3 (113 → 117 dim: v2 extras + 4 self-belote bits).
#   * Reward: compute_scores now credits Q+K-of-trump same-hand belote (+20).
#     Transparent — no flag — but changes reward magnitudes on ~11 % of deals.
#   * --match-sim: cumulative scores + dealer rotation 0→1→2→3 across deals
#     until one team reaches 2000. Replaces the uniform random-score injection
#     at reset. The model sees realistic match trajectories rather than an
#     artificial uniform distribution of (ns, ew) states.
#   * Steps: 25M → 75M (ratio passes-per-deal goes from ~50k to ~30k).
#   * Baseline eval: v5_isdd@25M (real champion, verified by 1000-match arena),
#     512 hidden. 1000 matches per eval to cut noise.
#
# Everything else (arch 512×3, EMA 0.005, cosine LR 3e-4→3e-5, reward-clip 1.0,
# PER α=0.6, diversity 0.40→0.15, ε 0.30→0.02) matches v5 so any delta is
# attributable to the data + obs + reward + match-sim changes.

set -u
cd "$(dirname "$0")/../.."

mkdir -p logs models

LOG=logs/v6_isdd.log
SAVE_DIR=models/bid_v6_isdd

POOL=data/deals/base_5M.bin
SCORES=data/deals/scores_isdd_5M.sc

BASELINE_BID=models/bid_v5_isdd/bid_nn_final.bin
BASELINE_HIDDEN=512

echo "[$(date '+%F %T')] === v6 ISDD training START ===" | tee -a "$LOG"
echo "  pool:     $POOL" | tee -a "$LOG"
echo "  scores:   $SCORES" | tee -a "$LOG"
echo "  save_dir: $SAVE_DIR" | tee -a "$LOG"
echo "  baseline: $BASELINE_BID (h=$BASELINE_HIDDEN)" | tee -a "$LOG"

./target/release/train_bid_nn \
  --num-envs 256 \
  --hidden 512 \
  --layers 3 \
  --lr 3e-4 \
  --lr-end 3e-5 \
  --eps-start 0.30 \
  --eps-end 0.02 \
  --eps-decay-steps 55000000 \
  --steps 75000000 \
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
echo "[$(date '+%F %T')] === v6 ISDD training END (rc=$RC) ===" | tee -a "$LOG"
exit $RC
