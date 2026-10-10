"""Erised: walk-forward backtesting for daily-bar equity and multi-asset strategies."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("erised")
except PackageNotFoundError:  # running from a source tree without installation
    __version__ = "0.0.0"

__all__ = ["__version__"]
