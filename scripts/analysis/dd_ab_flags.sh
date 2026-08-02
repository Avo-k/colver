#!/usr/bin/env bash
# A/B the *compilation target* of the DD solver by ALTERNATING binaries built from the
# SAME source with different RUSTFLAGS.
#
# Same reasoning as dd_ab_revs.sh: a single binary measured twice on this box varied by 20 %,
# so running config A then config B measures the machine's mood, not the flag. We build each
# config once, then alternate them round after round and keep the MINIMUM per config —
# competing load only ever adds time, so the fastest observation is the least disturbed one.
#
# Extra guard this script gets for free: the source is identical across configs, so the node
# counts MUST match exactly. If they don't, something is wrong with the harness, not the flag.
#
# Usage:  scripts/analysis/dd_ab_flags.sh [rounds]
set -euo pipefail
cd "$(dirname "$0")/../.."

ROUNDS="${1:-5}"
CORPUS="data/analysis/dd_corpus_v1.bin"
FEATURES="parallel solver_stats"
OUT=/tmp/ddflags
mkdir -p "$OUT"

# name          rustflags
CONFIGS=(
  "base:"
  "v3:-C target-cpu=x86-64-v3"
  "native:-C target-cpu=native"
)

echo "### build"
for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%:*}"; flags="${cfg#*:}"
  echo "  $name  RUSTFLAGS='$flags'"
  # Separate target dirs: cargo fingerprints on RUSTFLAGS, so sharing one dir would force a
  # full LTO rebuild on every switch. They live outside the repo (64 MB each) so there is
  # nothing to gitignore and nothing to clean up.
  RUSTFLAGS="$flags" cargo build --release --features "$FEATURES" --bin bench_dd -q \
      --target-dir "$OUT/target-$name"
  cp "$OUT/target-$name/release/bench_dd" "$OUT/bench_dd.$name"
done

echo
echo "### proof the flags reached the compiler"
# This check is the whole point: "no difference between three identical binaries" is a
# vacuous result. Do NOT try to verify with `rustc --print cfg` — it IGNORES the RUSTFLAGS
# env var (that is a cargo mechanism), so it reports the same 3 features for every config
# and proves nothing. Only the disassembly does.
#
# Read the tzcnt column before concluding anything about a speedup: it is already 32 in the
# baseline build, because LLVM emits the F3-prefixed encoding that pre-BMI1 CPUs decode as
# plain bsf. The dominant bit primitive here never needed the flag.
printf "  %-8s %8s %8s %8s %8s %8s %10s\n" cfg tzcnt popcnt blsr andn vpxor md5
for cfg in "${CONFIGS[@]}"; do
  name="${cfg%%:*}"
  d=$(objdump -d --no-show-raw-insn "$OUT/bench_dd.$name" 2>/dev/null || true)
  printf "  %-8s %8s %8s %8s %8s %8s %10s\n" "$name" \
    "$(grep -c '\btzcnt\b' <<<"$d")"  "$(grep -c '\bpopcnt\b' <<<"$d")" \
    "$(grep -c '\bblsr\b'  <<<"$d")"  "$(grep -c '\bandn\b'   <<<"$d")" \
    "$(grep -c '\bvpxor\b' <<<"$d")"  "$(md5sum "$OUT/bench_dd.$name" | cut -c1-8)"
done

echo
echo "### run: $ROUNDS alternating rounds, 1 thread"
for r in $(seq 1 "$ROUNDS"); do
  printf "  round %s/%s:" "$r" "$ROUNDS"
  for cfg in "${CONFIGS[@]}"; do
    name="${cfg%%:*}"
    printf " %s" "$name"
    "$OUT/bench_dd.$name" run --corpus "$CORPUS" --threads 1 \
        --values "$OUT/$name.vals" > "$OUT/$name.$r.txt"
  done
  printf "  (load %s)\n" "$(cut -d' ' -f1 /proc/loadavg)"
done

echo
echo "### EXACTNESS (values must be identical — same source)"
for cfg in "${CONFIGS[@]:1}"; do
  name="${cfg%%:*}"
  printf "  base vs %-8s " "$name"
  "$OUT/bench_dd.base" diff --a "$OUT/base.vals" --b "$OUT/$name.vals" | grep -Ei "match|differ" | head -2 | tr '\n' ' '
  echo
done

echo
echo "### SPEED (minimum over $ROUNDS rounds)"
python3 - "$ROUNDS" "$OUT" <<'PY'
import re, sys
rounds, out = int(sys.argv[1]), sys.argv[2]
names = ["base", "v3", "native"]
ROW = re.compile(r"^\s*(full|mid|end|worlds|ALL)\s+(\d+)\s+([\d.]+)\s+(\d+)\s+([\d.]+)")
def best(tag):
    o = {}
    for r in range(1, rounds + 1):
        for line in open(f"{out}/{tag}.{r}.txt"):
            m = ROW.match(line)
            if m:
                k, n, v = m.group(1), int(m.group(4)), float(m.group(5))
                if k not in o or v < o[k][0]:
                    o[k] = (v, n)
    return o
d = {n: best(n) for n in names}
print(f"{'shape':>8} {'nodes/pos':>12} " + "".join(f"{n+' us':>12}" for n in names)
      + f"{'v3':>9}{'native':>9}")
for k in ("full", "mid", "end", "worlds", "ALL"):
    if not all(k in d[n] for n in names):
        continue
    nodes = {d[n][k][1] for n in names}
    warn = "" if len(nodes) == 1 else f"  !! node counts differ: {sorted(nodes)}"
    b = d["base"][k][0]
    print(f"{k:>8} {d['base'][k][1]:>12} "
          + "".join(f"{d[n][k][0]:>12.1f}" for n in names)
          + f"{b/d['v3'][k][0]:>8.3f}x{b/d['native'][k][0]:>8.3f}x{warn}")
PY
