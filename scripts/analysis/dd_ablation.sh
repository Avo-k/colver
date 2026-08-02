#!/usr/bin/env bash
# What do PVS, killer moves and the history heuristic actually buy?
#
# The three carry folklore gains (+37 %, +38 %, +16 %) that predate every harness in this
# repo. They are also the reason "the tail is a move-ordering failure" is a hypothesis: if
# ordering were already near-optimal, the hard 10 % would be hard on their own merits.
#
# Node counts are the metric, and they are exact — so unlike a wall-clock A/B these
# configurations can safely run one after another. Wall time is reported but is a 32-thread
# number on a shared box; do not read a 5 % difference in it.
#
# Every ablation must produce IDENTICAL values: turning a heuristic off changes the order the
# search visits moves, never the answer. The diff at the end is not a formality — it is the
# same gate that would have caught `quick_tricks`.
#
# Usage: scripts/analysis/dd_ablation.sh [corpus] [threads]
set -euo pipefail

CORPUS="${1:-data/analysis/dd_corpus_v1.bin}"
THREADS="${2:-32}"
OUT="${TMPDIR:-/tmp}/dd_ablation"
mkdir -p "$OUT"

BIN=target/release/bench_dd
FEATURES="parallel solver_stats solver_ablation"

if [ ! -x "$BIN" ]; then
    echo "build: cargo build --release --features \"$FEATURES\" --bin bench_dd" >&2
    cargo build --release --features "$FEATURES" --bin bench_dd
fi

# name : env assignments
CONFIGS=(
    "baseline:"
    "no_pvs:COLVER_DD_NO_PVS=1"
    "no_killers:COLVER_DD_NO_KILLERS=1"
    "no_history:COLVER_DD_NO_HISTORY=1"
    "none:COLVER_DD_NO_PVS=1 COLVER_DD_NO_KILLERS=1 COLVER_DD_NO_HISTORY=1"
)

echo "corpus $CORPUS | $THREADS threads | node counts are the metric"
echo

for cfg in "${CONFIGS[@]}"; do
    name="${cfg%%:*}"
    envs="${cfg#*:}"
    echo "=== $name ==="
    # The flags are read once per process into a OnceLock, so each configuration needs its
    # own run — there is no way to switch mid-process, by design.
    env $envs "$BIN" run \
        --corpus "$CORPUS" --threads "$THREADS" \
        --values "$OUT/$name.vals" --json "$OUT/$name.json" \
        2>&1 | grep -E "heuristics:|^ *(shape|full|mid|end|worlds|ALL)|^wall|throughput"
    echo
done

echo "=== exactness gate ==="
# An ablation that changes a value is not a slower solver, it is a broken one.
fail=0
for cfg in "${CONFIGS[@]:1}"; do
    name="${cfg%%:*}"
    printf "  %-12s " "$name"
    if "$BIN" diff --a "$OUT/baseline.vals" --b "$OUT/$name.vals" 2>&1 | grep -q "EXACT MATCH"; then
        echo "EXACT MATCH"
    else
        echo "MISMATCH — the ablation changed an answer"
        fail=1
    fi
done
[ "$fail" -eq 0 ] || exit 1

echo
echo "json summaries in $OUT/"
