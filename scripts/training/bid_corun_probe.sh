#!/usr/bin/env bash
# Combien de bras d'entraînement d'annonce cette machine fait-elle tourner de front ?
#
# Une campagne d'ablation, c'est N bras au même budget. En séquentiel le mur est N × la
# durée d'un bras ; de front il pourrait être celle d'un seul. Entre les deux il y a une
# mesure, et elle ne se devine pas : `train_bid_nn` ne prend qu'environ un cœur (les envs
# sont séquentiels) mais il tape le GPU à chaque pas, donc le partage peut être gratuit
# comme il peut être une division exacte.
#
# Le protocole tient à une règle : **on ne compare pas deux runs successifs**. Chaque
# palier tourne le même temps, on jette la phase de remplissage du replay (les premiers
# pas ne rétropropagent rien et affichent un débit qui n'est pas celui du régime établi),
# et on lit la médiane de ce qui reste.
#
#   scripts/training/bid_corun_probe.sh [secondes_par_palier] [paliers...]
#   scripts/training/bid_corun_probe.sh 180 1 2 3

set -euo pipefail
cd "$(dirname "$0")/../.."

SECS=${1:-180}
shift || true
LEVELS=("${@:-1 2 3}")
[ $# -eq 0 ] && LEVELS=(1 2 3)

BIN=./target/release/train_bid_nn
OUT=$(mktemp -d)
trap 'pkill -f "$BIN" 2>/dev/null || true; rm -rf "$OUT"' EXIT

if [ ! -x "$BIN" ]; then
    echo "manque $BIN — cargo build -p colver-core --bin train_bid_nn --features dmc_train --release" >&2
    exit 1
fi

COMMON=(--hidden 512 --layers 3 --num-envs 256
        --pool-file data/deals/base_5M.bin --scores data/deals/scores_isdd_5M.sc
        --score-aware --match-sim --steps 100000000
        --save-freq 100000000 --eval-freq 100000000)

# Les trois bras de la campagne, pour que la mesure porte sur ce qu'on lancera vraiment.
arm_flags() {
    case $1 in
        0) echo "--sa-features-v3" ;;
        1) echo "--sa-features-v3 --canonical" ;;
        *) echo "--sa-features-v7 --canonical" ;;
    esac
}

# Débit en régime établi : on saute les 8 premières lignes (remplissage du replay) et on
# prend la médiane du reste.
throughput() {
    awk -F'|' '/^ +[0-9]+ \|/ {gsub(/ /,"",$NF); print $NF}' "$1" \
        | tail -n +9 | sort -n | awk '{a[NR]=$1} END {if (NR) print (NR%2 ? a[(NR+1)/2] : int((a[NR/2]+a[NR/2+1])/2)); else print "n/a"}'
}

echo "palier  secondes  débit par bras (pas/s)   total"
for n in "${LEVELS[@]}"; do
    for i in $(seq 0 $((n - 1))); do
        # shellcheck disable=SC2046
        setsid nohup "$BIN" "${COMMON[@]}" $(arm_flags "$i") \
            --seed $((42 + i)) --save-dir "$OUT/n${n}_a${i}" \
            > "$OUT/n${n}_a${i}.log" 2>&1 < /dev/null &
    done
    sleep "$SECS"
    pkill -f "$BIN" 2>/dev/null || true
    sleep 4

    rates=()
    for i in $(seq 0 $((n - 1))); do rates+=("$(throughput "$OUT/n${n}_a${i}.log")"); done
    total=0
    for r in "${rates[@]}"; do [ "$r" != "n/a" ] && total=$((total + r)); done
    printf "%6s  %8s  %-22s  %5s\n" "$n" "$SECS" "${rates[*]}" "$total"
done
