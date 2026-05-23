"""Focused tests for H033: BondTrendEquityTimer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import BondTrendEquityTimer


def _prices(spy_rets, tlt_levels, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in spy_rets:
        px.append(px[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": px}, index=idx)
    assert len(tlt_levels) == n + 1
    df["tlt_close"] = tlt_levels
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    tlt = (100 + rng.normal(0, 2, 701)).tolist()
    prices = _prices(spy, tlt)
    assert_no_lookahead(BondTrendEquityTimer().generate_positions, prices)


def test_long_when_tlt_below_ma():
    """TLT drops sharply while MA still reflects prior high -> position 1.0."""
    spy = [0.001] * 300
    # TLT high for 100 bars, then drops to 70; check 5 bars after the drop
    # while the 50d MA is still anchored around 120
    tlt = [120.0] * 100 + [70.0] * 201
    prices = _prices(spy, tlt)
    pos = BondTrendEquityTimer(ma_period=50).generate_positions(prices)
    # At bar 105, the 50d MA covers bars 56-105: 44 × 120 + 6 × 70 ≈ 112 > 70
    assert pos.iloc[105] == pytest.approx(1.0)


def test_flat_when_tlt_above_ma():
    """TLT spikes sharply while MA still reflects prior low -> position 0.0."""
    spy = [0.001] * 300
    # TLT low for 100 bars, then spikes to 130; check 5 bars after spike
    tlt = [70.0] * 100 + [130.0] * 201
    prices = _prices(spy, tlt)
    pos = BondTrendEquityTimer(ma_period=50).generate_positions(prices)
    # At bar 105, MA is still near 70; TLT at 130 > MA
    assert pos.iloc[105] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(7)
    spy = rng.normal(0, 0.01, 400).tolist()
    tlt = (100 + rng.normal(0, 3, 401)).tolist()
    prices = _prices(spy, tlt)
    pos = BondTrendEquityTimer(ma_period=50).generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_zero_during_warmup():
    """Before ma_period warmup, position should be 0."""
    spy = [0.001] * 100
    tlt = [100.0] * 101
    prices = _prices(spy, tlt)
    pos = BondTrendEquityTimer(ma_period=50).generate_positions(prices)
    assert (pos.iloc[:50] == 0.0).all()


def test_requires_tlt_close_column():
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="tlt_close"):
        BondTrendEquityTimer().generate_positions(prices)
