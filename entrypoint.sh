#!/bin/sh
# Auto-download the latest DMC model if not present
MODEL_PATH="${COLVER_MODEL_PATH:-/app/models/dmc_final.bin}"
MODEL_DIR="$(dirname "$MODEL_PATH")"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ] || [ "${COLVER_UPDATE_MODEL:-0}" = "1" ]; then
    echo "[entrypoint] Downloading latest model..."
    python -c "
from colver._model import _DEFAULT_URL
from urllib.request import urlretrieve
import sys
dest = '$MODEL_PATH'
print(f'  {_DEFAULT_URL} -> {dest}')
urlretrieve(_DEFAULT_URL, dest)
print('  Done.')
" && echo "[entrypoint] Model ready at $MODEL_PATH" \
  || echo "[entrypoint] WARNING: model download failed, continuing without DouDou"
else
    echo "[entrypoint] Model found at $MODEL_PATH"
fi

exec "$@"
