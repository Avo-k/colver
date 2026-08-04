#!/usr/bin/env bash
# Attend la fin du run v3-large en cours (11,52 M échantillons), puis le
# prolonge jusqu'aux 30,72 M de v2 — le seul budget qui permette un
# « meilleur contre meilleur » plutôt qu'une comparaison à budget apparié.
#
# 200 000 steps × 96 = 19,2 M échantillons de plus. Rien ne s'y oppose :
#   - le corpus offre 864 M échantillons distincts (9 M donnes × 4 observateurs
#     × 24 permutations), dont 1,3 % consommés ;
#   - `train_playgen` n'a qu'un warmup, pas de décroissance de LR, donc une
#     reprise ne casse aucun calendrier ;
#   - train 0,899 contre éval 0,919 : aucun sur-apprentissage.
set -uo pipefail
cd "$(dirname "$0")/../.."

while pgrep -f "train_playgen.*playgen_v3_large2" >/dev/null; do sleep 30; done
sleep 10   # laisser écrire le checkpoint final

exec env CUDARC_CUDA_VERSION=13010 ./target/release/train_playgen \
  --v2 --games data/training/playgen_games_9M.bin \
  --steps 200000 --batch-size 96 --d-model 512 --layers 8 --heads 8 \
  --lr 2e-4 --warmup 200 --save-freq 20000 \
  --resume models/playgen_v3_large2/playgen_final.safetensors \
  --save-dir models/playgen_v3_large3
