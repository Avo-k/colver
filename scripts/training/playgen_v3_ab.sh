#!/usr/bin/env bash
# L'A/B qui décide si un playgen v3 vaut la peine : **même corpus, même config,
# même budget, seul `--v3` change**.
#
# Pourquoi un témoin et pas juste « v3 contre playgen_v2_final ». Un v3 diffère
# de v2 par **deux** choses à la fois : il lit le score, et il est entraîné sur
# un corpus de parties (donc sur des enchères de fin de partie que le corpus
# 0-0 ne contient pas du tout). Comparer v3 à `playgen_v2_final` attribuerait au
# score ce qui vient du corpus. Le témoin est un v2 entraîné sur **le même
# corpus de parties** : la seule différence restante est les deux jetons.
#
#   usage: scripts/training/playgen_v3_ab.sh <corpus.bin> [steps]
#
# ## Pourquoi la petite config
#
# L'échelle de perplexité du 2026-08-04 a établi que **la tête d'enchère est
# saturée en capacité et en données** : d=256 L=4 (3,22 M) à 12,8 M
# d'échantillons est à 1,6 % de d=384 L=6 (10,74 M) à 30,7 M. Or le score ne
# touche que la tête d'enchère — DouDou50 joue la carte sans jamais le lire.
# L'A/B se fait donc à la petite config, et sa réponse transfère.
#
# C'est le rendement de cette mesure : ~6 h de GPU au lieu de ~40 h pour la même
# décision. Le gros modèle ne se justifie qu'**après**, en production, et pour
# la tête de jeu.
#
# ## Budget
#
# 60 000 pas × 256 = 15,36 M échantillons. La tête d'enchère est à 1,6 % de sa
# valeur finale dès 2,56 M, donc 15,36 M la voit convergée. Mesuré : 5,0 pas/s,
# soit ~3,3 h par bras, ~6,6 h les deux.
set -uo pipefail
cd "$(dirname "$0")/../.."

CORPUS=${1:?usage: playgen_v3_ab.sh <corpus.bin> [steps]}
STEPS=${2:-60000}
COMMON=(--games "$CORPUS" --steps "$STEPS" --batch-size 256
        --d-model 256 --layers 4 --heads 8 --lr 3e-4 --warmup 500
        --save-freq 10000 --eval-freq 5000)

for arm in v3 v2; do
  dir="models/playgen_v3ab_$arm"
  echo "=== bras $arm -> $dir"
  CUDARC_CUDA_VERSION=13010 ./target/release/train_playgen \
    "--$arm" "${COMMON[@]}" --save-dir "$dir" || exit 1
  CUDARC_CUDA_VERSION=13010 ./target/release/export_playgen \
    "$dir/playgen_final.safetensors" "$dir/${arm}_final.bin" \
    --d-model 256 --layers 4 --heads 8 "--$arm" || exit 1
done

# La lecture. `--model-b` bootstrappe l'écart **apparié sur les donnes**, et la
# table par bande d'écart de score est celle qui décide : un gain global de
# 0,004 est indistinguable du bruit alors que le même effet vaut 0,1 sur les
# 4 % de donnes à plus de 1200 d'écart. Mesurer là où la chose s'applique.
HELD=${HELD:-data/training/heldout_matches_20k.bin}
./target/release/bench_playgen_ppl \
  --model models/playgen_v3ab_v2/v2_final.bin \
  --model-b models/playgen_v3ab_v3/v3_final.bin \
  --games "$HELD" --n 20000 | tee data/analysis/playgen_ppl/v3ab_verdict.txt
