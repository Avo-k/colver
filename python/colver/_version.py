"""Single source of truth for the package version: pyproject.toml.

This used to be a hardcoded string, which silently drifted 11 releases behind
(0.9.0 shipped reporting itself as "0.3.1"). Reading it back from the installed
distribution metadata makes that drift structurally impossible.
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("colver")
except PackageNotFoundError:
    # Source checkout with the package not installed (e.g. plain PYTHONPATH use).
    __version__ = "0.0.0+unknown"
