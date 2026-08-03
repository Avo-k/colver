#!/usr/bin/env bash
# Do finer move-ordering rules actually shrink the tree?
#
# The confusion table (`bench_dd ordering`) says ~70 % of ordering failures are *within* a
# move category: the card tried and the card that should have been tried do the same kind of
# thing. So a coarse Contrée rule ("prefer ruffs", "lead trump early") cannot fix them, and
# the candidates here all discriminate inside a category instead. What each one targets is in
# the doc comment of `move_order_score_v`.
#
# Node counts are the metric — exact, so sequential runs are safe. Wall time is reported but
# a 5 % difference in it means nothing on a shared box.
#
# EVERY variant must produce identical values. Reordering moves changes the shape of the
# search, never its result; a mismatch here is a bug in the variant, not a faster solver.
#
# Usage: scripts/analysis/dd_order_variants.sh [corpus] [threads]
set -euo pipefail

CORPUS="${1:-data/analysis/dd_corpus_v1.bin}"
THREADS="${2:-32}"
OUT="${TMPDIR:-/tmp}/dd_order"
mkdir -p "$OUT"

BIN=target/release/bench_dd
FEATURES="parallel solver_stats solver_ablation"
[ -x "$BIN" ] || cargo build --release --features "$FEATURES" --bin bench_dd

echo "corpus $CORPUS | $THREADS threads | node counts are the metric"
echo

for v in 0 1 2 3 4; do
    echo "=== order v$v ==="
    COLVER_DD_ORDER="$v" "$BIN" run \
        --corpus "$CORPUS" --threads "$THREADS" \
        --values "$OUT/v$v.vals" --json "$OUT/v$v.json" \
        2>&1 | grep -E "^ *(shape|full|mid|end|worlds|ALL)|^wall"
    echo
done

echo "=== exactness gate ==="
fail=0
for v in 1 2 3 4; do
    printf "  v%-11s " "$v"
    if "$BIN" diff --a "$OUT/v0.vals" --b "$OUT/v$v.vals" 2>&1 | grep -q "EXACT MATCH"; then
        echo "EXACT MATCH"
    else
        echo "MISMATCH — the variant changed an answer"
        fail=1
    fi
done
[ "$fail" -eq 0 ] || exit 1

echo
echo "=== nodes vs v0 (>1 means the variant is worse) ==="
python3 - "$OUT" <<'PY'
import json, sys, os
out = sys.argv[1]
base = json.load(open(os.path.join(out, "v0.json")))["nodes"]
for v in range(5):
    n = json.load(open(os.path.join(out, f"v{v}.json")))["nodes"]
    print(f"  v{v}: {n:>14,}  {n/base:6.3f}x")
PY
