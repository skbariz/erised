"""Tests for the data layer."""

from __future__ import annotations

import pandas as pd
import pytest

from erised.data.loader import load_holdout, load_research
from erised.data.schema import validate_bars


def test_validate_bars_rejects_tz_aware():
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=10, tz="UTC"))
    df = pd.DataFrame(
        {
            "open": 100.0,
            "high": 101.0,
            "low": 99.0,
            "close": 100.5,
            "adj_close": 100.5,
            "volume": 1000,
        },
        index=idx,
    )
    with pytest.raises(ValueError, match="tz-naive"):
        validate_bars(df)


def test_validate_bars_rejects_unsorted():
    idx = pd.DatetimeIndex(["2020-01-03", "2020-01-02", "2020-01-04"])
    df = pd.DataFrame(
        {
            "open": [100.0, 100.0, 100.0],
            "high": [101.0, 101.0, 101.0],
            "low": [99.0, 99.0, 99.0],
            "close": [100.5, 100.5, 100.5],
            "adj_close": [100.5, 100.5, 100.5],
            "volume": [1000, 1000, 1000],
        },
        index=idx,
    )
    with pytest.raises(ValueError, match="sorted ascending"):
        validate_bars(df)


def test_validate_bars_rejects_inconsistent_ohlc():
    idx = pd.DatetimeIndex(pd.bdate_range("2020-01-01", periods=1))
    df = pd.DataFrame(
        {
            "open": [100.0],
            "high": [99.0],  # high < open
            "low": [98.0],
            "close": [100.5],
            "adj_close": [100.5],
            "volume": [1000],
        },
        index=idx,
    )
    with pytest.raises(ValueError, match="OHLC inconsistencies"):
        validate_bars(df)


def test_holdout_loader_requires_explicit_flag():
    """The holdout cannot be loaded without the acknowledgment flag."""
    with pytest.raises(RuntimeError, match="acknowledgment flag"):
        load_holdout("SPY")


def test_research_loader_works():
    """The research loader works without ceremony."""
    df = load_research("SPY")
    assert len(df) > 1000
    assert df.index.max().year < 2023  # holdout starts 2023-01-01
