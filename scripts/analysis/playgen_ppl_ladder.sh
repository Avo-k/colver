#!/usr/bin/env bash
# Échelle de perplexité : exporte tout checkpoint v3-small pas encore exporté,
# puis passe le bench sur lui et sur les .bin de v2 déjà disponibles.
#
# L'axe honnête est le nombre d'**échantillons vus** (steps × batch), pas les
# steps : v2 s'entraîne à batch 192, v3-small à 256. Deux courbes indexées sur
# les steps se compareraient à budgets différents.
#
#   usage: scripts/analysis/playgen_ppl_ladder.sh [n_donnes]
set -uo pipefail
cd "$(dirname "$0")/../.."

N=${1:-500}
GAMES=data/training/heldout_20k_s90210.bin
OUT=data/analysis/playgen_ppl
mkdir -p "$OUT"

# v3-small : d=256 L=4 H=8, batch 256.
for ckpt in models/playgen_v3_small/playgen_[0-9]*.safetensors; do
    [ -e "$ckpt" ] || continue
    step=$(basename "$ckpt" .safetensors | sed 's/playgen_//')
    bin="models/playgen_v3_small/v3s_${step}.bin"
    if [ ! -f "$bin" ]; then
        echo "--- export $ckpt -> $bin"
        CUDARC_CUDA_VERSION=13010 ./target/release/export_playgen \
            "$ckpt" "$bin" --d-model 256 --layers 4 --heads 8 --v2 >/dev/null || continue
    fi
    log="$OUT/v3s_${step}_n${N}.txt"
    [ -f "$log" ] || ./target/release/bench_playgen_ppl \
        --model "$bin" --games "$GAMES" --n "$N" > "$log"
    echo "== v3-small step $step  ($((step * 256 / 1000))k échantillons)"
    grep -E "^    [1-8] \|" "$log" | awk -F'|' '{gsub(/ /,"",$4); printf "%s ", $4}'
    echo
done

# v2 : d=384 L=6 H=8, batch 192. Déjà exportés.
# v2_half.bin et v2_60k.bin sont le même fichier (md5 f3307df9…) : un seul point.
for pair in "models/playgen_v2/playgen_v2_half.bin:60000" \
            "models/playgen/playgen_v2_120k.bin:120000" \
            "models/playgen/playgen_v2_final.bin:160000"; do
    bin="${pair%%:*}"; step="${pair##*:}"
    [ -f "$bin" ] || continue
    log="$OUT/v2_${step}_n${N}.txt"
    [ -f "$log" ] || ./target/release/bench_playgen_ppl \
        --model "$bin" --games "$GAMES" --n "$N" > "$log"
    echo "== v2 step $step  ($((step * 192 / 1000))k échantillons)"
    grep -E "^    [1-8] \|" "$log" | awk -F'|' '{gsub(/ /,"",$4); printf "%s ", $4}'
    echo
done
