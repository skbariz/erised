"""Focused tests for H031: RealizedVsImpliedVol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import RealizedVsImpliedVol


def _prices(spy_rets, vix_levels, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in spy_rets:
        px.append(px[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": px}, index=idx)
    assert len(vix_levels) == n + 1
    df["vix_close"] = vix_levels
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    vix = (18 + rng.normal(0, 4, 701)).clip(10, 60).tolist()
    prices = _prices(spy, vix)
    assert_no_lookahead(RealizedVsImpliedVol().generate_positions, prices)


def test_long_when_vix_exceeds_realized():
    """VIX >> realized vol -> position 1.0 (vol risk premium positive)."""
    spy = [0.0] * 200  # zero returns -> realized vol ~= 0
    vix = [20.0] * 201  # VIX=20 >> realized vol
    prices = _prices(spy, vix)
    pos = RealizedVsImpliedVol(realized_window=20).generate_positions(prices)
    assert (pos.iloc[22:] == 1.0).all()


def test_flat_when_realized_exceeds_vix():
    """Realized vol >> VIX -> position 0.0 (VRP negative)."""
    # Extremely volatile SPY returns -> high realized vol
    spy = [0.05 if i % 2 == 0 else -0.05 for i in range(200)]
    # Low VIX (10 < realized vol ~= 80% annualized)
    vix = [10.0] * 201
    prices = _prices(spy, vix)
    pos = RealizedVsImpliedVol(realized_window=20).generate_positions(prices)
    assert (pos.iloc[22:] == 0.0).all()


def test_position_is_binary():
    rng = np.random.default_rng(4)
    spy = rng.normal(0, 0.01, 300).tolist()
    vix = (20 + rng.normal(0, 4, 301)).clip(10, 60).tolist()
    prices = _prices(spy, vix)
    pos = RealizedVsImpliedVol().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_realized_window_param():
    """Different realized_window values should produce valid binary positions."""
    spy = [0.001] * 300
    vix = [20.0] * 301
    prices = _prices(spy, vix)
    for rw in [10, 20, 60]:
        pos = RealizedVsImpliedVol(realized_window=rw).generate_positions(prices)
        assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_zero_during_warmup():
    """Before realized vol warmup, position should be 0."""
    spy = [0.001] * 100
    vix = [20.0] * 101
    prices = _prices(spy, vix)
    pos = RealizedVsImpliedVol(realized_window=20).generate_positions(prices)
    # First 20 rows: no valid realized vol yet
    assert (pos.iloc[:20] == 0.0).all()


def test_requires_vix_close_column():
    """Raises ValueError if vix_close column is missing."""
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="vix_close"):
        RealizedVsImpliedVol().generate_positions(prices)
