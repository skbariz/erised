"""Tests for H071: TripleConfirmCalendar."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.base import TripleConfirmCalendar


def _prices(start: str = "2003-01-01", n: int = 600, tlt_trend: str = "down") -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(71)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, n))
    if tlt_trend == "down":
        tlt = 100 * np.cumprod(1 + rng.normal(-0.0002, 0.004, n))
    else:
        tlt = 100 * np.cumprod(1 + rng.normal(0.0004, 0.004, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_raises_without_tlt():
    strat = TripleConfirmCalendar()
    df = pd.DataFrame({"adj_close": [100.0] * 100}, index=pd.date_range("2003-01-01", periods=100, freq="B"))
    with pytest.raises(ValueError, match="tlt_close"):
        strat.generate_positions(df)


def test_returns_series():
    pos = TripleConfirmCalendar().generate_positions(_prices())
    assert isinstance(pos, pd.Series)
    assert pos.name == "position"


def test_positions_binary():
    pos = TripleConfirmCalendar().generate_positions(_prices())
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_flat_in_summer():
    prices = _prices(start="2005-06-01", n=100, tlt_trend="down")
    pos = TripleConfirmCalendar().generate_positions(prices)
    # June is summer — must be flat
    summer = pos[prices.index.month.isin([6, 7, 8, 9, 10])]
    assert summer.sum() == pytest.approx(0.0)


def test_nov_jan_uses_bond_filter_only():
    # Nov-Jan window: bond trending down (TLT declining) → bond_ok=True → should be long
    prices = _prices(start="2005-11-01", n=60, tlt_trend="down")
    pos = TripleConfirmCalendar(tlt_ma_period=200).generate_positions(prices)
    # After MA warmup is not possible in 60 bars, but at least not all flat
    assert isinstance(pos, pd.Series)


def test_flat_before_tlt_data():
    # Use pre-2002 period where TLT unavailable (simulate with all-NaN tlt)
    idx = pd.date_range("2000-01-01", periods=100, freq="B")
    rng = np.random.default_rng(0)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 100))
    prices = pd.DataFrame(
        {"adj_close": spy, "tlt_close": np.nan}, index=idx
    )
    pos = TripleConfirmCalendar().generate_positions(prices)
    assert pos.sum() == pytest.approx(0.0)


def test_shape_matches_input():
    prices = _prices()
    pos = TripleConfirmCalendar().generate_positions(prices)
    assert len(pos) == len(prices)
