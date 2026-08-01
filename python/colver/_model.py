"""Model weight download and discovery."""

import logging
import os
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

_DEFAULT_URL = "https://github.com/Avo-k/colver/releases/download/v0.4.0/dmc_50.bin"
_DEFAULT_BID_URL = "https://github.com/Avo-k/colver/releases/download/v0.7.0/bid_v6_isdd.bin"
_DEFAULT_BELIEF_URL = "https://github.com/Avo-k/colver/releases/download/v0.7.0/belief_v4_fix_v2.bin"
_DEFAULT_PLAYGEN_URL = "https://github.com/Avo-k/colver/releases/download/v0.8.0/playgen_v2_final.bin"
_CACHE_DIR = Path.home() / ".cache" / "colver" / "models"


def model_path(name: str = "dmc_50.bin") -> Path | None:
    """Find a model weights file.

    Checks ``COLVER_MODEL_PATH`` env-var first, then ``~/.cache/colver/models/``,
    then ``./models/`` relative to the current working directory (dev fallback).
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
    # Dev fallback: check ./models/ relative to CWD
    p = Path.cwd() / "models" / name
    if p.is_file():
        return p
    return None


def bid_model_path(name: str = "bid_v6_isdd.bin") -> Path | None:
    """Find a bid model weights file.

    Checks ``COLVER_BID_MODEL_PATH`` env-var first, then ``~/.cache/colver/models/``,
    then ``./models/`` relative to the current working directory (dev fallback).
    Returns *None* if the file is not found.
    """
    env = os.environ.get("COLVER_BID_MODEL_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    p = _CACHE_DIR / name
    if p.is_file():
        return p
    # Dev fallback: check ./models/ relative to CWD
    p = Path.cwd() / "models" / name
    if p.is_file():
        return p
    return None


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    # La barre `\r` n'a de sens que sur un terminal : dans un collecteur de
    # logs ligne à ligne (docker logs), elle s'accumulerait en une ligne géante.
    if total_size > 0 and sys.stdout.isatty():
        pct = min(100, block_num * block_size * 100 // total_size)
        mb = block_num * block_size / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)


def download_model(
    url: str | None = None,
    name: str = "dmc_50.bin",
    force: bool = False,
) -> Path:
    """Download model weights to ``~/.cache/colver/models/``.

    Returns the local path to the downloaded file.
    Downloads to a temp file first, then atomically moves into place
    to avoid leaving corrupt partial files on interrupted downloads.
    """
    dest = _CACHE_DIR / name
    if dest.is_file() and not force:
        logger.info("Model already cached at %s", dest)
        return dest

    url = url or _DEFAULT_URL
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
    os.close(fd)
    try:
        urlretrieve(url, tmp, reporthook=_progress_hook)
        if sys.stdout.isatty():
            print()  # newline after progress
        os.replace(tmp, dest)
    except Exception:
        os.unlink(tmp)
        raise
    logger.info("Saved to %s", dest)
    return dest


def download_bid_model(
    url: str | None = None,
    name: str = "bid_v6_isdd.bin",
    force: bool = False,
) -> Path:
    """Download bid model weights to ``~/.cache/colver/models/``.

    Returns the local path to the downloaded file.
    """
    dest = _CACHE_DIR / name
    if dest.is_file() and not force:
        logger.info("Bid model already cached at %s", dest)
        return dest

    url = url or _DEFAULT_BID_URL
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
    os.close(fd)
    try:
        urlretrieve(url, tmp, reporthook=_progress_hook)
        if sys.stdout.isatty():
            print()  # newline after progress
        os.replace(tmp, dest)
    except Exception:
        os.unlink(tmp)
        raise
    logger.info("Saved to %s", dest)
    return dest


def belief_model_path(name: str = "belief_v4_fix_v2.bin") -> Path | None:
    """Find a belief net model weights file.

    Checks ``COLVER_BELIEF_MODEL_PATH`` env-var first, then ``~/.cache/colver/models/``,
    then ``./models/`` relative to the current working directory (dev fallback).
    Returns *None* if the file is not found.
    """
    env = os.environ.get("COLVER_BELIEF_MODEL_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    p = _CACHE_DIR / name
    if p.is_file():
        return p
    p = Path.cwd() / "models" / name
    if p.is_file():
        return p
    return None


def download_belief_model(
    url: str | None = None,
    name: str = "belief_v4_fix_v2.bin",
    force: bool = False,
) -> Path:
    """Download belief net model weights to ``~/.cache/colver/models/``.

    Returns the local path to the downloaded file.
    """
    dest = _CACHE_DIR / name
    if dest.is_file() and not force:
        logger.info("Belief model already cached at %s", dest)
        return dest

    url = url or _DEFAULT_BELIEF_URL
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
    os.close(fd)
    try:
        urlretrieve(url, tmp, reporthook=_progress_hook)
        if sys.stdout.isatty():
            print()  # newline after progress
        os.replace(tmp, dest)
    except Exception:
        os.unlink(tmp)
        raise
    logger.info("Saved to %s", dest)
    return dest


def playgen_model_path(name: str = "playgen_v2_final.bin") -> Path | None:
    """Find a playgen world-sampler model weights file.

    Checks ``COLVER_PLAYGEN_MODEL_PATH`` env-var first, then
    ``~/.cache/colver/models/``, then ``./models/playgen/`` relative to the
    current working directory (dev fallback).
    Returns *None* if the file is not found.
    """
    env = os.environ.get("COLVER_PLAYGEN_MODEL_PATH")
    if env:
        p = Path(env)
        if p.is_file():
            return p
    p = _CACHE_DIR / name
    if p.is_file():
        return p
    p = Path.cwd() / "models" / "playgen" / name
    if p.is_file():
        return p
    return None


def download_playgen_model(
    url: str | None = None,
    name: str = "playgen_v2_final.bin",
    force: bool = False,
) -> Path:
    """Download playgen model weights to ``~/.cache/colver/models/``.

    Returns the local path to the downloaded file.
    """
    dest = _CACHE_DIR / name
    if dest.is_file() and not force:
        logger.info("Playgen model already cached at %s", dest)
        return dest

    url = url or _DEFAULT_PLAYGEN_URL
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading %s", url)
    fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
    os.close(fd)
    try:
        urlretrieve(url, tmp, reporthook=_progress_hook)
        if sys.stdout.isatty():
            print()  # newline after progress
        os.replace(tmp, dest)
    except Exception:
        os.unlink(tmp)
        raise
    logger.info("Saved to %s", dest)
    return dest
