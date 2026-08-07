#!/usr/bin/env bash
# Relance `gen_score_layer` s'il disparaît, pendant un run non surveillé de plusieurs
# jours.
#
# Le chien de garde du sidecar (`sidecar_watchdog.sh`) couvre l'autre moitié du
# problème : si le sidecar meurt, il revient. Mais si c'est le GÉNÉRATEUR qui tombe —
# panique, OOM, budget d'erreurs dépassé après une longue coupure — plus rien ne
# calcule, et une nuit entière peut se perdre entre deux regards. La reprise rend le
# redémarrage gratuit : la couche est réécrite tous les `--checkpoint` donnes et un
# relancement repart du préfixe dense.
#
# **Non invasif** : le script ne lance rien tant qu'un `gen_score_layer` tourne, donc il
# s'attache à un run déjà en cours.
#
# **Borné** : un plafond de redémarrages et un délai croissant. Sans ça, une cause
# permanente — pool illisible, sidecar définitivement mort, disque plein — produirait
# une boucle qui remplit les journaux et masque la panne au lieu de la signaler.
#
# **Il surveille aussi le préfixe gelé, et c'est la panne la plus coûteuse des deux.**
# Vécue le 2026-08-06 à 17:53. Une donne dont l'étiquetage échoue est laissée à zéro
# (`gen_score_layer.rs`, branche `if !ok`), et le fichier n'est écrit que jusqu'à
# `done.iter().position(|&x| !x)` — donc **une seule** donne ratée gèle le compteur pour
# tout le reste du processus, pendant que 160 threads continuent de calculer des donnes
# qui ne seront jamais persistées. Le processus a l'air parfaitement sain : CPU à 1550 %,
# aucune erreur, le compteur d'« abouties » monte. Seul celui d'« écrites » est figé.
# Deux mille donnes ont été perdues avant qu'on le voie ; une nuit en aurait coûté dix
# mille. Le redémarrage est la parade, puisque la reprise repart du préfixe dense et
# réessaie la donne fautive — un échec de sidecar sous contention est transitoire.
#
#   scripts/analysis/gen_supervisor.sh <log-du-generateur> [max-relances]
#
set -u
LOG="${1:?usage: gen_supervisor.sh <log> [max]}"
MAX="${2:-20}"
REPO="${COLVER_REPO:-$HOME/code/colver}"
SUP="${LOG%.log}_supervisor.log"

# Couche surveillée, et délai au-delà duquel un compteur figé n'est plus une lenteur.
# À 1,0 donne/s un checkpoint de 500 tombe toutes les ~8 min ; 25 min laissent donc trois
# checkpoints de marge avant de conclure au gel.
LAYER="${COLVER_GEN_LAYER:-$REPO/data/deals/scores_isdd_v2.sc}"
STALL_S="${COLVER_GEN_STALL_S:-1500}"

# Second détecteur, **indépendant du débit** : l'écart entre « abouties » et « écrites »
# du dernier checkpoint. En régime sain il vaut le nombre de donnes en vol chez les 160
# threads — mesuré entre 38 et 125 sur deux jours. Dès qu'une donne manque, il croît sans
# borne au rythme de la production. 800 laisse un facteur 6 sur le maximum observé et
# trahit le trou en ~10 min à 1,3 donne/s, contre 25 pour le délai de gel.
#
# Les deux détecteurs sont gardés : celui-ci voit vite mais dépend du format du log,
# celui du temps est grossier mais ne dépend de rien. Le premier des deux qui parle gagne.
GAP_MAX="${COLVER_GEN_GAP_MAX:-800}"

# Écart du dernier checkpoint écrit, ou vide si le processus COURANT n'en a pas encore
# écrit. Deux précautions, chacune pour un piège rencontré :
#
# 1. Le grep vise la ligne de checkpoint et non la dernière ligne du log — une ligne
#    d'erreur arrivée entre deux checkpoints ferait rendre du vide, donc perdrait le
#    détecteur au moment précis où il sert.
# 2. **La lecture démarre après la dernière ligne de reprise.** Le log est en mode ajout,
#    donc juste après un redémarrage la dernière ligne de checkpoint est celle de
#    l'ANCIEN processus et porte encore son gros écart : sans cette borne, le superviseur
#    tuerait le processus neuf à la seconde suivante, en boucle, et le run n'avancerait
#    plus jamais.
layer_gap() {
  local start e
  start=$(grep -n "donnes déjà étiquetées" "$LOG" 2>/dev/null | tail -1 | cut -d: -f1)
  [ -z "$start" ] && start=1
  e=$(tail -n +"$start" "$LOG" 2>/dev/null | grep "donnes écrites" | tail -1 \
        | sed -n 's/.*✔ \([0-9]*\) donnes écrites (\([0-9]*\) abouties.*/\2-\1/p')
  [ -n "$e" ] && echo $((e))
}

# Le binaire est **épinglé** hors de `target/`. Un `cargo build --release` lancé par
# quelqu'un d'autre dans ce dépôt remplace `target/release/gen_score_layer` sans rien
# dire ; le processus en cours n'en souffre pas (Linux garde l'inode), mais la relance
# suivante prendrait un binaire différent et la couche aurait deux régimes sans qu'aucun
# octet du fichier ne le signale. Le nom de base doit rester « gen_score_layer » : c'est
# lui que `pgrep -x` compare.
BIN="${COLVER_GEN_BIN:-$REPO/.gen_pin/gen_score_layer}"
[ -x "$BIN" ] || BIN="$REPO/target/release/gen_score_layer"

# Nombre de donnes réellement persistées. L'en-tête COLVSC01 est de longueur variable
# (magic[8] + name_len:u16 + name + count:u32 + offset:u32), d'où la lecture du nom.
layer_count() {
  python3 - "$LAYER" <<'PY' 2>/dev/null || echo -1
import struct, sys
try:
    d = open(sys.argv[1], 'rb').read(64)
    nl = struct.unpack('<H', d[8:10])[0]
    print(struct.unpack('<I', d[10 + nl:14 + nl])[0])
except Exception:
    print(-1)
PY
}

# La commande exacte du run. Gardée ici en toutes lettres plutôt que reconstruite : une
# relance qui change un paramètre en silence produirait une couche à deux régimes, et
# rien dans le fichier ne le dirait.
ARGS=(--pool data/deals/base_5M.bin --offset 0 --count 500000
      --threads 160 --checkpoint 500
      --out data/deals/scores_isdd_v2.sc
      --games data/training/isdd_games_v2.bin
      --url "http://localhost:8003,http://localhost:8003,http://localhost:8003,http://192.168.1.23:8003")

stamp() { date +'%Y-%m-%d %H:%M:%S'; }
n=0
last_count=$(layer_count)
last_move=$(date +%s)

echo "$(stamp) superviseur armé (max $MAX relances, binaire $BIN, gel > ${STALL_S}s)" >>"$SUP"
while true; do
  # `pgrep -x` compare au nom tronqué à 15 caractères par Linux ; « gen_score_layer »
  # en fait exactement 15, donc il correspond. Un binaire plus long ne correspondrait
  # pas et le superviseur relancerait en boucle sur un processus bien vivant.
  if ! pgrep -x gen_score_layer >/dev/null 2>&1; then
    if [ "$n" -ge "$MAX" ]; then
      echo "$(stamp) $MAX relances atteintes — arrêt du superviseur, la panne est permanente" >>"$SUP"
      exit 1
    fi
    n=$((n + 1))
    echo "$(stamp) générateur absent — relance n°$n" >>"$SUP"
    ( cd "$REPO" && setsid nohup "$BIN" "${ARGS[@]}" \
        >>"$LOG" 2>&1 </dev/null & )
    # Laisser le temps du chargement du pool (105 Mo) avant de reconclure à l'absence,
    # et espacer les tentatives si elles échouent coup sur coup.
    sleep $((30 + n * 15))
    # Le compteur vient de repartir d'un préfixe dense : remettre l'horloge du gel à
    # zéro, sinon la relance suivante se déclencherait sur le temps de chargement.
    last_count=$(layer_count)
    last_move=$(date +%s)
  else
    # Générateur vivant : le compteur d'écrites doit avancer. S'il ne bouge plus, la
    # donne en tête du trou a échoué et **rien de ce qui se calcule ne sera gardé**.
    c=$(layer_count)
    g=$(layer_gap)
    why=""
    if [ -n "$g" ] && [ "$g" -gt "$GAP_MAX" ]; then
      why="écart écrites/abouties à $g (> $GAP_MAX)"
    fi
    if [ "$c" != "$last_count" ]; then
      last_count=$c
      last_move=$(date +%s)
    elif [ "$c" != "-1" ] && [ $(( $(date +%s) - last_move )) -gt "$STALL_S" ]; then
      why="compteur figé depuis ${STALL_S}s"
    fi
    if [ -n "$why" ]; then
      echo "$(stamp) préfixe GELÉ à $c — $why — redémarrage (tout ce qui suit le trou est perdu de toute façon)" >>"$SUP"
      kill $(pgrep -x gen_score_layer) 2>/dev/null
      # Pas de relance ici : le tour suivant constate l'absence et s'en charge, donc
      # le plafond `MAX` compte aussi les redémarrages pour gel. Un gel qui se répète
      # trente fois est une panne permanente, pas une malchance.
      last_move=$(date +%s)
    fi
  fi
  sleep 30
done
