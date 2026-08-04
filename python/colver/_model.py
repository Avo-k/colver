"""Model weight download and discovery.

Les poids vivent sur le Hub Hugging Face, un dépôt par modèle. Le repli reste
GitHub Releases : ce sont les URL qu'utilisaient les versions ≤ 0.9.1, elles
restent servies, et elles dépannent quand le Hub est injoignable.

Ce qui ne change pas, et qui est ce qui rend le dépôt de dev silencieux :
``COLVER_*_MODEL_PATH`` et le repli ``./models/`` sont consultés **avant** tout
téléchargement, par les fonctions ``*_path``.
"""

import logging
import os
import shutil
import sys
import tempfile
from pathlib import Path
from urllib.request import urlretrieve

logger = logging.getLogger(__name__)

# nom local -> (dépôt HF, fichier dans le dépôt, URL de repli GitHub Releases)
_MODELS: dict[str, tuple[str, str, str]] = {
    "dmc_50.bin": (
        "Avo-k/colver-doudou50",
        "dmc_50.bin",
        "https://github.com/Avo-k/colver/releases/download/v0.4.0/dmc_50.bin",
    ),
    "bid_v6_isdd.bin": (
        "Avo-k/colver-bid-v6",
        "bid_v6_isdd.bin",
        "https://github.com/Avo-k/colver/releases/download/v0.7.0/bid_v6_isdd.bin",
    ),
    "belief_v4_fix_v2.bin": (
        "Avo-k/colver-belief-v4",
        "belief_v4_fix_v2.bin",
        "https://github.com/Avo-k/colver/releases/download/v0.7.0/belief_v4_fix_v2.bin",
    ),
    "playgen_v2_final.bin": (
        "Avo-k/colver-playgen-v2",
        "playgen_v2_final.bin",
        "https://github.com/Avo-k/colver/releases/download/v0.8.0/playgen_v2_final.bin",
    ),
}

_CACHE_DIR = Path.home() / ".cache" / "colver" / "models"


def _find(name: str, env_var: str, subdir: str = "") -> Path | None:
    """Chercher un modèle sans jamais le télécharger."""
    env = os.environ.get(env_var)
    if env:
        p = Path(env)
        if p.is_file():
            return p
    p = _CACHE_DIR / name
    if p.is_file():
        return p
    # Repli dev : ./models/ (ou ./models/<subdir>/) relatif au répertoire courant
    p = Path.cwd() / "models" / subdir / name if subdir else Path.cwd() / "models" / name
    if p.is_file():
        return p
    return None


def model_path(name: str = "dmc_50.bin") -> Path | None:
    """Trouver les poids du réseau de jeu (DouDou50). *None* si absent."""
    return _find(name, "COLVER_MODEL_PATH")


def bid_model_path(name: str = "bid_v6_isdd.bin") -> Path | None:
    """Trouver les poids du réseau d'enchère (Bid v6). *None* si absent."""
    return _find(name, "COLVER_BID_MODEL_PATH")


def belief_model_path(name: str = "belief_v4_fix_v2.bin") -> Path | None:
    """Trouver les poids du réseau de croyances. *None* si absent."""
    return _find(name, "COLVER_BELIEF_MODEL_PATH")


def playgen_model_path(name: str = "playgen_v2_final.bin") -> Path | None:
    """Trouver les poids de l'échantillonneur de mondes playgen. *None* si absent."""
    return _find(name, "COLVER_PLAYGEN_MODEL_PATH", subdir="playgen")


def _progress_hook(block_num: int, block_size: int, total_size: int) -> None:
    # La barre `\r` n'a de sens que sur un terminal : dans un collecteur de
    # logs ligne à ligne (docker logs), elle s'accumulerait en une ligne géante.
    if total_size > 0 and sys.stdout.isatty():
        pct = min(100, block_num * block_size * 100 // total_size)
        mb = block_num * block_size / 1_048_576
        total_mb = total_size / 1_048_576
        print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct}%)", end="", flush=True)


def _install(src: Path, dest: Path) -> Path:
    """Poser `src` en `dest`, sans laisser de fichier à moitié écrit.

    Lien dur d'abord — les deux caches sont sous ``~/.cache``, donc en général le
    même système de fichiers, et le Hub garde déjà ses propres octets. La copie
    ne sert que si les deux caches sont montés séparément.

    **`resolve()` est obligatoire.** `hf_hub_download` rend un chemin de
    ``snapshots/`` qui est un lien symbolique *relatif* vers ``../../blobs/<sha>``.
    `os.link` dessus duplique le lien, pas le fichier : déplacé ici, son
    ``../../blobs/`` ne désigne plus rien et le cache se remplit de liens morts.
    On lie donc le blob réel.
    """
    src = src.resolve()
    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
    os.close(fd)
    os.unlink(tmp)
    try:
        try:
            os.link(src, tmp)
        except OSError:
            shutil.copyfile(src, tmp)
        os.replace(tmp, dest)
    except Exception:
        # `lexists` et non `exists` : un lien mort doit être nettoyé lui aussi.
        if os.path.lexists(tmp):
            os.unlink(tmp)
        raise
    return dest


def _fetch(name: str, url: str | None, force: bool, revision: str | None) -> Path:
    """Rapatrier un modèle dans ``~/.cache/colver/models/``.

    Hub d'abord, GitHub Releases en repli. Une `url` explicite court-circuite
    les deux : c'est la porte de sortie pour servir ses propres poids.
    """
    dest = _CACHE_DIR / name
    if dest.is_file() and not force:
        logger.info("Modèle déjà en cache : %s", dest)
        return dest

    entry = _MODELS.get(name)

    if url is None and entry is not None:
        repo_id, filename, _ = entry
        try:
            from huggingface_hub import hf_hub_download

            logger.info("Téléchargement depuis le Hub : %s/%s", repo_id, filename)
            got = hf_hub_download(repo_id=repo_id, filename=filename, revision=revision)
            return _install(Path(got), dest)
        except ImportError:
            logger.warning("huggingface_hub absent — repli sur GitHub Releases")
        except Exception as e:
            logger.warning("Hub injoignable (%s) — repli sur GitHub Releases", e)

    if url is None:
        if entry is None:
            raise ValueError(f"Modèle inconnu : {name!r}. Passez une `url` explicite.")
        url = entry[2]

    _CACHE_DIR.mkdir(parents=True, exist_ok=True)
    logger.info("Téléchargement de %s", url)
    fd, tmp = tempfile.mkstemp(dir=_CACHE_DIR, suffix=".tmp")
    os.close(fd)
    try:
        urlretrieve(url, tmp, reporthook=_progress_hook)
        if sys.stdout.isatty():
            print()  # newline after progress
        os.replace(tmp, dest)
    except Exception:
        if os.path.exists(tmp):
            os.unlink(tmp)
        raise
    logger.info("Enregistré dans %s", dest)
    return dest


def download_model(
    url: str | None = None,
    name: str = "dmc_50.bin",
    force: bool = False,
    revision: str | None = None,
) -> Path:
    """Télécharger les poids du réseau de jeu (DouDou50).

    Depuis `Avo-k/colver-doudou50 <https://huggingface.co/Avo-k/colver-doudou50>`_,
    avec repli GitHub Releases. Rend le chemin local.
    """
    return _fetch(name, url, force, revision)


def download_bid_model(
    url: str | None = None,
    name: str = "bid_v6_isdd.bin",
    force: bool = False,
    revision: str | None = None,
) -> Path:
    """Télécharger les poids du réseau d'enchère (Bid v6).

    Depuis `Avo-k/colver-bid-v6 <https://huggingface.co/Avo-k/colver-bid-v6>`_,
    avec repli GitHub Releases. Rend le chemin local.
    """
    return _fetch(name, url, force, revision)


def download_belief_model(
    url: str | None = None,
    name: str = "belief_v4_fix_v2.bin",
    force: bool = False,
    revision: str | None = None,
) -> Path:
    """Télécharger les poids du réseau de croyances.

    Depuis `Avo-k/colver-belief-v4 <https://huggingface.co/Avo-k/colver-belief-v4>`_,
    avec repli GitHub Releases. Rend le chemin local.
    """
    return _fetch(name, url, force, revision)


def download_playgen_model(
    url: str | None = None,
    name: str = "playgen_v2_final.bin",
    force: bool = False,
    revision: str | None = None,
) -> Path:
    """Télécharger les poids de l'échantillonneur de mondes playgen.

    Depuis `Avo-k/colver-playgen-v2 <https://huggingface.co/Avo-k/colver-playgen-v2>`_,
    avec repli GitHub Releases. Rend le chemin local.
    """
    return _fetch(name, url, force, revision)
