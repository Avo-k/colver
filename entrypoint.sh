#!/bin/sh
# Auto-download the latest DMC model if not present or if update requested
MODEL_PATH="${COLVER_MODEL_PATH:-/app/models/dmc_50.bin}"
MODEL_DIR="$(dirname "$MODEL_PATH")"

mkdir -p "$MODEL_DIR"

if [ ! -f "$MODEL_PATH" ] || [ "${COLVER_UPDATE_MODEL:-0}" = "1" ]; then
    echo "[entrypoint] Downloading latest model..."
    python -c "
from colver._model import _DEFAULT_URL
from urllib.request import urlretrieve
import os, tempfile
dest = '$MODEL_PATH'
print(f'  {_DEFAULT_URL} -> {dest}')
# Download to temp file first, then atomic move (avoids corrupt partial files)
fd, tmp = tempfile.mkstemp(dir=os.path.dirname(dest), suffix='.tmp')
os.close(fd)
try:
    urlretrieve(_DEFAULT_URL, tmp)
    os.replace(tmp, dest)
    print('  Done.')
except Exception:
    os.unlink(tmp)
    raise
" && echo "[entrypoint] Model ready at $MODEL_PATH" \
  || echo "[entrypoint] WARNING: model download failed, continuing without DouDou50"
else
    echo "[entrypoint] Model found at $MODEL_PATH"
fi

exec "$@"
