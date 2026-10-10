"""Data loading API.

Every downstream component (features, backtester, strategies) reads
data through this module. This is the single place where we enforce
the holdout boundary.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

import pandas as pd

from ..config import DATA_DIR
from .schema import validate_bars

logger = logging.getLogger(__name__)


# Filename convention: SYMBOL.parquet for the full series,
# SYMBOL_research.parquet for the research-only slice,
# SYMBOL_holdout.parquet for the sealed holdout.
def _full_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}.parquet"


def _research_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}_research.parquet"


def _holdout_path(symbol: str) -> Path:
    return DATA_DIR / f"{symbol.upper()}_holdout.parquet"


def load_full(symbol: str) -> pd.DataFrame:
    """Load the full series.

    Use only for data ingestion and one-time holdout sealing.
    DO NOT use this in strategy research — use load_research().
    """
    path = _full_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"No data file at {path}. Run `erised data fetch` first.")
    df = pd.read_parquet(path)
    validate_bars(df, symbol)
    return df


def load_research(symbol: str) -> pd.DataFrame:
    """Load the research-only slice. Use this for ALL strategy development.

    The holdout is excluded. Calling this is safe and routine.
    """
    path = _research_path(symbol)
    if not path.exists():
        raise FileNotFoundError(
            f"No research file at {path}. Run `erised data seal` to create one."
        )
    df = pd.read_parquet(path)
    validate_bars(df, symbol)
    return df


def load_holdout(symbol: str, _i_understand_this_is_final_evaluation: bool = False) -> pd.DataFrame:
    """Load the sealed holdout.

    DO NOT CALL THIS during research. The flag is intentionally clunky
    so accidental usage in a notebook is obvious in code review.
    """
    if not _i_understand_this_is_final_evaluation:
        raise RuntimeError(
            "load_holdout requires the explicit acknowledgment flag. "
            "This dataset is for final evaluation only. If you're sure, "
            "pass _i_understand_this_is_final_evaluation=True."
        )
    path = _holdout_path(symbol)
    if not path.exists():
        raise FileNotFoundError(f"No holdout file at {path}.")
    df = pd.read_parquet(path)
    validate_bars(df, symbol)
    return df


def file_hash(path: Path) -> str:
    """SHA-256 of a file's contents, for the holdout seal."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()
