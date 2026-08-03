#!/usr/bin/env bash
# Does the choice among DD-equal cards change playing strength?
#
# 57,8 % of positions have more than one card at the best DD value, and which one the solver
# reports depends on its internal move ordering. Nobody has measured whether that matters.
#
# **It cannot be measured against a perfect opponent**: two DD-optimal players both realise the
# DD value, so any tie-break scores identically by construction. It can only show up against an
# imperfect one — and the effect should grow as the opponent gets worse, since a "DD-equal" card
# only differentiates itself by how well it exploits mistakes. Hence the opponent sweep, from a
# fixed heuristic up to DouDou50.
#
# Match win rate saturates (the oracle beats the weak bots ~95 % of the time whatever it does),
# so the metric is the **average margin in points**. Three seeds per cell give the error bar —
# without one, a 50-point difference is unreadable.
#
# Usage: scripts/analysis/dd_tiebreak_arena.sh [matches_per_direction] [out.csv]
set -euo pipefail

MATCHES="${1:-400}"
OUT="${2:-/tmp/tiebreak_arena.csv}"
BIN=target/release/arena
[ -x "$BIN" ] || cargo build --release --features parallel --bin arena

TIEBREAKS=(order lowest highest cheapest dearest)
OPPONENTS=(heuristic rule ismcts dmc50)
SEEDS=(11 22 33)

echo "tiebreak,opponent,seed,win_pct,margin" > "$OUT"
total=$(( ${#TIEBREAKS[@]} * ${#OPPONENTS[@]} * ${#SEEDS[@]} ))
i=0
for opp in "${OPPONENTS[@]}"; do
  for tb in "${TIEBREAKS[@]}"; do
    for sd in "${SEEDS[@]}"; do
      i=$((i+1))
      # --no-save: these are throwaway probe bots, they have no business in matches.csv.
      line=$("$BIN" h2h "tb_$tb" "opp_$opp" --matches "$MATCHES" --seed "$sd" --no-save 2>&1)
      win=$(grep -oP "tb_$tb \K[0-9.]+(?=%)" <<<"$line" | head -1)
      mrg=$(grep -oP 'Avg margin: \K[+-][0-9]+' <<<"$line")
      echo "$tb,$opp,$sd,$win,$mrg" >> "$OUT"
      echo "[$i/$total] $tb vs $opp seed=$sd -> ${win}% margin ${mrg}"
    done
  done
done
echo "done -> $OUT"
