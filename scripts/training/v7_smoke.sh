#!/usr/bin/env bash
# Test de fumée de la chaîne v7 — à passer AVANT d'engager 58 h d'entraînement.
#
# **Pourquoi il existe.** La campagne v7 (§4bis de docs/bid/bid_v7_plan.md) est trois bras
# de ~19 h, jugés par une échelle d'instruments. Le premier passage de ce script, le
# 2026-08-06, a trouvé que **le code d'entraînement marchait et que trois des quatre
# instruments de jugement ne pouvaient pas lire un réseau canonique** — sans lever
# d'erreur. Découvrir ça après 58 h aurait coûté la campagne entière.
#
# Il tourne sur une couche **partielle** : c'est délibéré, la question posée est « la
# chaîne est-elle branchée », pas « le modèle est-il bon ». ~10 min, un checkpoint jetable.
#
#   scripts/training/v7_smoke.sh [couche] [pas]
#
# ⚠️ Il consomme le GPU. Mesuré le 2026-08-06 : à côté d'une génération de couche, le
# trainer tombe de 511 à 115 pas/s — le CPU n'est pas en cause (le bras a son cœur
# entier, charge 11,6/32), c'est le GPU à 90 %. Les deux ne cohabitent pas.

set -euo pipefail
cd "$(dirname "$0")/../.."

LAYER="${1:-data/deals/scores_isdd_v2.sc}"
STEPS="${2:-40000}"
POOL_SRC="${POOL_SRC:-data/deals/base_5M.bin}"
WORK="${WORK:-/tmp/v7smoke}"
V6="${V6:-models/bid_v6_isdd_resume/bid_nn_final.bin}"

mkdir -p "$WORK"
say() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

say "1. instantané de la couche (elle peut être en cours d'écriture)"
uv run python scripts/analysis/snapshot_score_layer.py "$LAYER" "$WORK/layer.sc" --ranks
N=$(python3 -c "
import struct
d = open('$WORK/layer.sc','rb').read(32)
nl = struct.unpack('<H', d[8:10])[0]
print(struct.unpack('<I', d[10+nl:14+nl])[0])")

say "2. format et qualité de la couche"
uv run python scripts/analysis/check_score_layer.py "$WORK/layer.sc"

say "3. troncature du pool à $N donnes"
# `--pool-size` est obligatoire en aval : sans lui `DealPool::load_or_generate` REGÉNÈRE
# jusqu'à 1 000 000 (le défaut) et annule la troncature en silence. Cf. §9 du plan couche.
uv run python scripts/analysis/truncate_pool.py "$POOL_SRC" "$N" "$WORK/base_$N.bin"

say "4. entraînement court ($STEPS pas, bras C = 117 canonique)"
rm -rf "$WORK/model"
./target/release/train_bid_nn \
  --num-envs 256 --hidden 512 --layers 3 \
  --lr 3e-4 --lr-end 3e-5 \
  --eps-start 0.30 --eps-end 0.02 --eps-decay-steps 55000000 \
  --steps "$STEPS" \
  --reward real --score-aware --sa-features-v3 --canonical \
  --match-sim --reward-clip 1.0 --ema-tau 0.005 \
  --buffer-size 500000 --min-buffer 10000 \
  --pool-size "$N" --pool-file "$WORK/base_$N.bin" \
  --scores "$WORK/layer.sc" \
  --eval-freq 100000000 --save-freq "$STEPS" \
  --save-dir "$WORK/model" 2>&1 | tee "$WORK/train.log" | grep -E \
  "Loaded .* deals|generating more|Activated score layer|Canonical suit ordering|obs_dim|Reward mode|^ *[0-9]+ \|" | head -20

say "5. les trois lignes de contrôle du démarrage"
# « generating more » est le contrôle décisif : il dit que la troncature a été annulée et
# que 99 % des épisodes retomberaient sur des dd_pts périmées, sans un mot dans le log.
for pat in "Loaded $N deals" "Activated score layer" "Canonical suit ordering"; do
  if grep -q "$pat" "$WORK/train.log"; then echo "  ✔ $pat"; else echo "  ✗ MANQUANT : $pat"; fi
done
if grep -q "generating more" "$WORK/train.log"; then
  echo "  ✗ « generating more » PRÉSENT — la troncature a été annulée, run invalide"
else
  echo "  ✔ pas de « generating more »"
fi

say "6. contrôle de largeur (niveau 0 de l'échelle d'acceptation)"
# Un dueling 512³ pèse 2 445 488 o à 117 et 2 457 776 o à 123. C'est le seul contrôle qui
# attrape une largeur fausse avant des heures de calcul.
SZ=$(stat -c%s "$WORK/model/bid_nn_final.bin")
case "$SZ" in
  2445488) echo "  ✔ $SZ o = obs 117" ;;
  2457776) echo "  ✔ $SZ o = obs 123" ;;
  *)       echo "  ✗ $SZ o — largeur inattendue" ;;
esac

say "7. câblage canonique : deux implémentations indépendantes doivent concorder"
uv run python scripts/analysis/canonical_roundtrip.py \
  --net "$WORK/model/bid_nn_final.bin" --deals 120 --cross-check

say "8. ce que coûterait un oubli de \`canonical = true\` (mesuré sur v6, entraîné)"
uv run python scripts/analysis/canonical_roundtrip.py --net "$V6" --deals 200

say "9. équivariance (niveau 1) — lire le critère dans bid_v7_plan.md §« Ce qui juge »"
echo "  attendu : erreur de Q à 0.0000 en médiane ET p90 ; bascules dans la fourchette"
echo "  des contrôles (roro 1,3 % … improved_v2 3,9 %), le résidu étant les 7,5 % de"
echo "  mains à symétrie de couleur. Un 0,0 % de bascule n'est PAS atteignable."
uv run python scripts/analysis/bid_equivariance.py \
  --bid-model "$WORK/model/bid_nn_final.bin" --canonical --deals 120 --no-log

say "terminé — artefacts dans $WORK"
