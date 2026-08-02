#!/usr/bin/env bash
# A/B two git revisions of the solver by ALTERNATING two binaries.
#
# Building both and running them back to back is not enough: a single binary measured twice
# on this machine varied by 20 %, which is bigger than most of the wins we are chasing.
# Alternating N times and keeping the minimum per configuration is what makes the comparison
# survive whatever else the box is doing.
#
# Usage:  scripts/analysis/dd_ab_revs.sh <rev-A> [rounds]
#         rev-A is the baseline; the working tree is the candidate.
set -euo pipefail
cd "$(dirname "$0")/../.."

REV="${1:?usage: dd_ab_revs.sh <baseline-rev> [rounds]}"
ROUNDS="${2:-3}"
CORPUS="data/analysis/dd_corpus_v1.bin"
FEATURES="parallel solver_stats"
export RUSTFLAGS="-C target-cpu=native"

echo "building candidate (working tree)..."
cargo build --release --features "$FEATURES" --bin bench_dd -q
cp target/release/bench_dd /tmp/bench_dd.cand

echo "building baseline ($REV)..."
WT=$(mktemp -d)
git worktree add -q --detach "$WT" "$REV"
( cd "$WT" && cargo build --release --features "$FEATURES" --bin bench_dd -q \
    --target-dir "$WT/target" )
cp "$WT/target/release/bench_dd" /tmp/bench_dd.base

for r in $(seq 1 "$ROUNDS"); do
  echo "--- round $r/$ROUNDS"
  /tmp/bench_dd.base run --corpus "$CORPUS" --threads 1 --values /tmp/dd.base.vals \
      > "/tmp/dd.base.$r.txt"
  /tmp/bench_dd.cand run --corpus "$CORPUS" --threads 1 --values /tmp/dd.cand.vals \
      > "/tmp/dd.cand.$r.txt"
done

echo
echo "=== EXACTNESS ==="
/tmp/bench_dd.cand diff --a /tmp/dd.base.vals --b /tmp/dd.cand.vals | tail -6

echo
echo "=== SPEED (minimum over $ROUNDS alternating rounds) ==="
python3 - "$ROUNDS" <<'PY'
import re, sys
rounds = int(sys.argv[1])
ROW = re.compile(r"^\s*(full|mid|end|worlds|ALL)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)")
def best(tag):
    out = {}
    for r in range(1, rounds + 1):
        for line in open(f"/tmp/dd.{tag}.{r}.txt"):
            m = ROW.match(line)
            if m:
                k = m.group(1)
                v = float(m.group(5))
                n = int(m.group(4))
                if k not in out or v < out[k][0]:
                    out[k] = (v, n)
    return out
b, c = best("base"), best("cand")
print(f"{'shape':>8} {'nodes/pos':>12} {'base us':>12} {'cand us':>12} {'speedup':>9}")
for k in ("full", "mid", "end", "worlds", "ALL"):
    if k in b and k in c:
        same = "" if b[k][1] == c[k][1] else f"  !! nodes {b[k][1]} -> {c[k][1]}"
        print(f"{k:>8} {c[k][1]:>12} {b[k][0]:>12.1f} {c[k][0]:>12.1f} {b[k][0]/c[k][0]:>8.3f}x{same}")
PY

git worktree remove --force "$WT" 2>/dev/null || true
