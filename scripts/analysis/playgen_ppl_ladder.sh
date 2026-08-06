#!/usr/bin/env bash
# Échelle de perplexité : exporte tout checkpoint pas encore exporté, puis passe
# le bench dessus et sur les .bin de v2 déjà disponibles.
#
# L'axe honnête est le nombre d'**échantillons vus** (steps × batch), pas les
# steps : les runs n'ont pas le même batch, donc deux courbes indexées sur les
# steps compareraient des budgets différents.
#
#   usage: scripts/analysis/playgen_ppl_ladder.sh [n_donnes]
#
# Les runs à parcourir sont déclarés ici — <dir>:<d_model>:<layers>:<heads>:<batch>.
set -uo pipefail
cd "$(dirname "$0")/../.."

# <dir>:<d>:<L>:<H>:<batch>[:<offset>] — l'offset porte les **reprises**, dont le
# compteur de steps repart de zéro, souvent avec un autre batch. Sans lui le
# budget affiché est celui d'un run neuf, donc faux.
#
# `v2belote_*` sont des **COLVPG02** (`--v2`), pas des v3 : la seule chose qui
# les sépare de `playgen_v2_final` est la belote dans le masque des sièges
# cachés. Aucun ne lit le score de partie — ces courbes mesurent la capacité et
# le budget d'échantillons, jamais l'apport du score.
RUNS=${RUNS:-"models/playgen_v2belote_small:256:4:8:256 models/playgen_v2belote_large:512:8:8:128 models/playgen_v2belote_large2:512:8:8:96:3840000 models/playgen_v2belote_large3:512:8:8:96:11520000"}
N=${1:-500}
GAMES=data/training/heldout_20k_s90210.bin
OUT=data/analysis/playgen_ppl
mkdir -p "$OUT"

# Une ligne de résultats depuis un log de bench.
emit() {
    local log="$1" head="$2"
    echo "$head"
    printf '   encheres '; grep -E "^     [1-6] \|" "$log" \
        | awk -F'|' '{gsub(/ /,"",$4); gsub(/ /,"",$2); printf "%s(n=%s) ", $4, $2}'
    printf '\n   jeu/carte'; grep -E "^    [1-8] \|" "$log" \
        | awk -F'|' '{gsub(/ /,"",$4); printf " %s", $4}'
    # La ligne qui décide : un écart par carte se **compose** sur toutes les
    # cartes restantes, donc c'est en continuations cumulées que se lit ce que
    # le modèle fait gagner à IS-DD, pas en rapport par carte. Ancrer sur
    # l'en-tête du tableau : les colonnes `n` des deux autres ont la même forme.
    printf '\n   mondes   '; sed -n '/^  depuis pli/,$p' "$log" | tail -n +2 \
        | awk -F'|' '{gsub(/ /,"",$2); printf " %s", $2}'
    echo
}

for run in $RUNS; do
    IFS=: read -r dir d l h batch offset <<< "$run"
    offset=${offset:-0}
    tag=$(basename "$dir" | sed 's/playgen_//')
    for ckpt in "$dir"/playgen_[0-9]*.safetensors; do
        [ -e "$ckpt" ] || continue
        step=$(basename "$ckpt" .safetensors | sed 's/playgen_//')
        bin="$dir/${tag}_${step}.bin"
        if [ ! -f "$bin" ]; then
            echo "--- export $ckpt -> $bin" >&2
            CUDARC_CUDA_VERSION=13010 ./target/release/export_playgen \
                "$ckpt" "$bin" --d-model "$d" --layers "$l" --heads "$h" --v2 \
                >/dev/null || continue
        fi
        log="$OUT/${tag}_${step}_n${N}.txt"
        [ -f "$log" ] || ./target/release/bench_playgen_ppl \
            --model "$bin" --games "$GAMES" --n "$N" > "$log"
        emit "$log" "== $tag step $step  ($(( (offset + step * batch) / 1000 ))k échantillons)"
    done
done

# v2 : d=384 L=6 H=8, batch 192, déjà exportés.
# v2_half.bin et v2_60k.bin sont le même fichier (md5 f3307df9…) : un seul point.
for pair in "models/playgen_v2/playgen_v2_half.bin:60000" \
            "models/playgen/playgen_v2_120k.bin:120000" \
            "models/playgen/playgen_v2_final.bin:160000"; do
    bin="${pair%%:*}"; step="${pair##*:}"
    [ -f "$bin" ] || continue
    log="$OUT/v2_${step}_n${N}.txt"
    [ -f "$log" ] || ./target/release/bench_playgen_ppl \
        --model "$bin" --games "$GAMES" --n "$N" > "$log"
    emit "$log" "== v2 step $step  ($((step * 192 / 1000))k échantillons)"
done
