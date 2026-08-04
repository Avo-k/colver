#!/bin/sh
# Pré-charge le modèle de jeu au démarrage du conteneur, pour que la première
# partie ne paie pas le téléchargement.
#
# Passe par `colver.download_model()` et non par une copie locale de sa logique :
# c'est lui qui sait d'où viennent les poids (le Hub, avec repli GitHub Releases)
# et qui vérifie le cache. Une reimplementation ici s'est deja desynchronisee une
# fois — elle importait `_model._DEFAULT_URL`, un nom prive disparu au passage au
# Hub, et le `|| echo` ci-dessous transformait l'ImportError en simple
# avertissement : le conteneur demarrait sans DouDou50, donc le mode « rapide »
# retombait sur Dede sans que personne le voie.
MODEL_PATH="${COLVER_MODEL_PATH:-/app/models/dmc_50.bin}"
MODEL_DIR="$(dirname "$MODEL_PATH")"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ] || [ "${COLVER_UPDATE_MODEL:-0}" = "1" ]; then
    echo "[entrypoint] Téléchargement du modèle de jeu..."
    COLVER_DEST="$MODEL_PATH" python -c "
import os, shutil
import colver

dest = os.environ['COLVER_DEST']
src = colver.download_model(force=os.environ.get('COLVER_UPDATE_MODEL') == '1')
if os.path.realpath(src) != os.path.realpath(dest):
    shutil.copyfile(src, dest)
print(f'  {src} -> {dest}')
" && echo "[entrypoint] Modèle prêt : $MODEL_PATH" \
  || echo "[entrypoint] ATTENTION : téléchargement échoué, DouDou50 indisponible (mode « rapide » dégradé)"
else
    echo "[entrypoint] Modèle déjà présent : $MODEL_PATH"
fi

exec "$@"
