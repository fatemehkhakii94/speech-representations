"""Tools for reproducible analysis of speech model representations."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("speech-representations")
except PackageNotFoundError:  # pragma: no cover - source checkout before installation
    __version__ = "0+unknown"

__all__ = ["__version__"]
