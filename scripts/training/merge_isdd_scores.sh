#!/usr/bin/env bash
# Wait for enrich PID to finish, then merge the 5 per-1M IS-DD score files into
# one scores_isdd_5M.sc, and move the individual files to data/deals/archive/.
set -euo pipefail

ENRICH_PID="${1:-}"
MERGE_LOG="logs/merge_isdd_5M.log"

exec > >(tee -a "$MERGE_LOG") 2>&1

echo "[$(date '+%F %T')] === merge_isdd_scores start (waiting on PID=$ENRICH_PID) ==="

if [[ -n "$ENRICH_PID" ]]; then
  while kill -0 "$ENRICH_PID" 2>/dev/null; do
    sleep 30
  done
  echo "[$(date '+%F %T')] enrichment PID $ENRICH_PID exited"
fi

# Confirm the last piece landed
for f in data/deals/scores_isdd_1M.sc \
         data/deals/scores_isdd_1M-2M.sc \
         data/deals/scores_isdd_2M-3M.sc \
         data/deals/scores_isdd_3M-4M.sc \
         data/deals/scores_isdd_4M-5M.sc; do
  if [[ ! -f "$f" ]]; then
    echo "[$(date '+%F %T')] FAIL: missing $f — aborting merge"
    exit 1
  fi
done

echo "[$(date '+%F %T')] all 5 .sc files present — merging"

python3 - <<'PY'
import struct, pathlib

files = [
    "data/deals/scores_isdd_1M.sc",
    "data/deals/scores_isdd_1M-2M.sc",
    "data/deals/scores_isdd_2M-3M.sc",
    "data/deals/scores_isdd_3M-4M.sc",
    "data/deals/scores_isdd_4M-5M.sc",
]

all_scores = bytearray()
for i, path in enumerate(files):
    data = pathlib.Path(path).read_bytes()
    assert data[:8] == b"COLVSC01", f"{path}: bad magic"
    name_len = struct.unpack_from("<H", data, 8)[0]
    name = data[10:10+name_len]
    assert name == b"isdd", f"{path}: name={name!r}"
    off = 10 + name_len
    count  = struct.unpack_from("<I", data, off)[0]
    offset = struct.unpack_from("<I", data, off+4)[0]
    expected = i * 1_000_000
    assert offset == expected, f"{path}: offset {offset} != {expected}"
    assert count  == 1_000_000, f"{path}: count {count}"
    scores = data[off+8:]
    assert len(scores) == count * 4, f"{path}: bytes {len(scores)} != {count*4}"
    all_scores.extend(scores)

out = pathlib.Path("data/deals/scores_isdd_5M.sc")
with out.open("wb") as f:
    f.write(b"COLVSC01")
    f.write(struct.pack("<H", 4))
    f.write(b"isdd")
    f.write(struct.pack("<I", 5_000_000))
    f.write(struct.pack("<I", 0))
    f.write(all_scores)

print(f"Wrote {out} ({out.stat().st_size:,} bytes, {len(all_scores)//4:,} scores)")
PY

echo "[$(date '+%F %T')] merged → data/deals/scores_isdd_5M.sc"

mkdir -p data/deals/archive
for f in scores_isdd_1M.sc \
         scores_isdd_1M-2M.sc \
         scores_isdd_2M-3M.sc \
         scores_isdd_3M-4M.sc \
         scores_isdd_4M-5M.sc; do
  mv -v "data/deals/$f" "data/deals/archive/$f"
done

echo "[$(date '+%F %T')] === merge_isdd_scores DONE ==="
ls -la data/deals/*.sc
