"""Model weight download and discovery."""

import os
import sys
from pathlib import Path
from urllib.request import urlretrieve

_DEFAULT_URL = "https://github.com/Avo-k/colver/releases/download/v0.2.0/dmc_final.bin"
_CACHE_DIR = Path.home() / ".cache" / "colver" / "models"


def model_path(name: str = "dmc_final.bin") -> Path | None:
    """Find a model weights file.

    Checks ``COLVER_MODEL_PATH`` env-var first, then ``~/.cache/colver/models/``.
    Returns *None* if the file is not found.
    """
    env = os.environ.get("COLVER_MODEL_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    p = _CACHE_DIR / name
    if p.is_file():
        return p
    return None


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    if total_size > 0:
        pct = min(100, block_num * block_size * 100 // total_size)
        mb = block_num * block_size / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)


def download_model(
    url: str | None = None,
    name: str = "dmc_final.bin",
    force: bool = False,
) -> Path:
    """Download model weights to ``~/.cache/colver/models/``.

    Returns the local path to the downloaded file.
    """
    dest = _CACHE_DIR / name
    if dest.is_file() and not force:
        print(f"Model already cached at {dest}")
        return dest

    url = url or _DEFAULT_URL
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {url}")
    try:
        urlretrieve(url, dest, reporthook=_progress_hook)
        print()  # newline after progress
    except Exception:
        if dest.exists():
            dest.unlink()
        raise
    print(f"Saved to {dest}")
    return dest
