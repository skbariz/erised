"""Focused tests for H030: VIXTrendSignal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies.base import VIXTrendSignal


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
    assert_no_lookahead(VIXTrendSignal().generate_positions, prices)


def test_long_when_vix_declining():
    """VIX below its MA (declining trend) -> long SPY (+1)."""
    spy = [0.001] * 300
    # VIX falls monotonically: starts at 25, ends at 10
    vix = [25 - 0.05 * i for i in range(301)]
    prices = _prices(spy, vix)
    pos = VIXTrendSignal(ma_period=20).generate_positions(prices)
    # Well after warmup, VIX is declining -> long
    assert pos.iloc[100] == pytest.approx(1.0)


def test_short_when_vix_rising():
    """VIX above its MA (rising trend) -> short SPY (-1)."""
    spy = [0.001] * 300
    # VIX rises monotonically: starts at 10, ends at 40
    vix = [10 + 0.1 * i for i in range(301)]
    prices = _prices(spy, vix)
    pos = VIXTrendSignal(ma_period=20).generate_positions(prices)
    assert pos.iloc[100] == pytest.approx(-1.0)


def test_valid_range_after_warmup():
    """After warmup, positions should only be +1 or -1."""
    rng = np.random.default_rng(3)
    spy = rng.normal(0, 0.01, 300).tolist()
    vix = (20 + rng.normal(0, 5, 301)).clip(10, 60).tolist()
    prices = _prices(spy, vix)
    pos = VIXTrendSignal(ma_period=20).generate_positions(prices)
    valid = pos.iloc[21:]
    assert set(valid.values.round(6)).issubset({-1.0, 1.0})


def test_zero_before_warmup():
    """Before MA warmup, position should be 0."""
    spy = [0.001] * 100
    vix = [20.0] * 101
    prices = _prices(spy, vix)
    pos = VIXTrendSignal(ma_period=20).generate_positions(prices)
    assert (pos.iloc[:20] == 0.0).all()


def test_ma_period_param():
    """Longer MA period should produce fewer crossovers (fewer sign changes)."""
    rng = np.random.default_rng(7)
    spy = rng.normal(0, 0.01, 400).tolist()
    vix = (20 + rng.normal(0, 5, 401)).clip(10, 60).tolist()
    prices = _prices(spy, vix)
    pos10 = VIXTrendSignal(ma_period=10).generate_positions(prices)
    pos60 = VIXTrendSignal(ma_period=60).generate_positions(prices)
    changes10 = (pos10.diff() != 0).sum()
    changes60 = (pos60.diff() != 0).sum()
    assert changes60 < changes10


def test_requires_vix_close_column():
    """Raises ValueError if vix_close column is missing."""
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="vix_close"):
        VIXTrendSignal().generate_positions(prices)
