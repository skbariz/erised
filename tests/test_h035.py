"""Focused tests for H035: GoldFearGauge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies.base import GoldFearGauge


def _prices(spy_rets, gld_levels, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in spy_rets:
        px.append(px[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": px}, index=idx)
    assert len(gld_levels) == n + 1
    df["gld_close"] = gld_levels
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    gld = (150 + rng.normal(0, 5, 701)).tolist()
    prices = _prices(spy, gld)
    assert_no_lookahead(GoldFearGauge().generate_positions, prices)


def test_long_when_gld_below_ma():
    """GLD drops sharply while MA still reflects prior high -> position 1.0."""
    spy = [0.001] * 200
    # GLD high for 80 bars, then drops to 100; check 5 bars after the drop
    # while the 30d MA is still anchored near 160
    gld = [160.0] * 80 + [100.0] * 121
    prices = _prices(spy, gld)
    pos = GoldFearGauge(ma_period=30).generate_positions(prices)
    # At bar 85, the 30d MA covers bars 56-85: 25 × 160 + 5 × 100 ≈ 150 > 100
    assert pos.iloc[85] == pytest.approx(1.0)


def test_flat_when_gld_above_ma():
    """GLD spikes sharply while MA still reflects prior low -> position 0.0."""
    spy = [0.001] * 200
    # GLD low for 80 bars, then spikes to 160; check 5 bars after spike
    gld = [100.0] * 80 + [160.0] * 121
    prices = _prices(spy, gld)
    pos = GoldFearGauge(ma_period=30).generate_positions(prices)
    # At bar 85, MA is still near 100; GLD at 160 > MA
    assert pos.iloc[85] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(11)
    spy = rng.normal(0, 0.01, 400).tolist()
    gld = (150 + rng.normal(0, 8, 401)).tolist()
    prices = _prices(spy, gld)
    pos = GoldFearGauge(ma_period=20).generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_zero_during_warmup():
    """Before ma_period warmup, position should be 0."""
    spy = [0.001] * 100
    gld = [150.0] * 101
    prices = _prices(spy, gld)
    pos = GoldFearGauge(ma_period=50).generate_positions(prices)
    assert (pos.iloc[:50] == 0.0).all()


def test_requires_gld_close_column():
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="gld_close"):
        GoldFearGauge().generate_positions(prices)
