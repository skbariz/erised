"""Focused tests for H027: SeasonalityTrendCombo."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import SeasonalityTrendCombo


def _prices(rets, start="2010-01-04"):
    n = len(rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in rets:
        px.append(px[-1] * (1 + r))
    return pd.DataFrame({"adj_close": px}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    prices = _prices(rng.normal(0.0005, 0.01, 700).tolist())
    assert_no_lookahead(SeasonalityTrendCombo().generate_positions, prices)


def test_unfavorable_months_always_flat():
    """May-Oct should always produce position 0, regardless of trend."""
    prices = _prices([0.002] * 800)  # Sustained uptrend → MA golden cross
    pos = SeasonalityTrendCombo().generate_positions(prices)
    summer_mask = prices.index.month.isin([5, 6, 7, 8, 9, 10])
    assert (pos[summer_mask] == 0.0).all()


def test_downtrend_in_favorable_months_is_flat():
    """Death cross in Nov-Apr → position 0 (both conditions must be true)."""
    # Sustained decline ensures fast MA < slow MA
    prices = _prices([-0.003] * 800)
    pos = SeasonalityTrendCombo().generate_positions(prices)
    # After warmup, all bars should be 0 (downtrend kills the trend signal)
    assert (pos.iloc[200:] == 0.0).all()


def test_uptrend_in_favorable_months_is_long():
    """Golden cross during Nov-Apr → position 1.0."""
    prices = _prices([0.002] * 800)
    pos = SeasonalityTrendCombo().generate_positions(prices)
    winter_mask = prices.index.month.isin([11, 12, 1, 2, 3, 4])
    # After MA warmup (200 bars), check late winter months
    late_winter = prices.index[winter_mask & (prices.index.year >= 2012)]
    if len(late_winter) > 0:
        assert (pos[late_winter] == 1.0).all()


def test_position_is_binary():
    """Positions must be 0 or 1 only."""
    prices = _prices([0.001] * 600)
    pos = SeasonalityTrendCombo().generate_positions(prices)
    unique_vals = set(pos.values.round(6))
    assert unique_vals.issubset({0.0, 1.0})


def test_ma_params_accepted():
    """Different MA parameter combinations should produce valid binary positions."""
    prices = _prices([0.001] * 500)
    for fast, slow in [(20, 100), (50, 200), (50, 150)]:
        pos = SeasonalityTrendCombo(fast=fast, slow=slow).generate_positions(prices)
        unique_vals = set(pos.values.round(6))
        assert unique_vals.issubset({0.0, 1.0})


def test_warmup_period_is_zero():
    """Before slow MA is ready, position should be 0."""
    prices = _prices([0.001] * 400, start="2010-01-04")
    pos = SeasonalityTrendCombo(fast=50, slow=200).generate_positions(prices)
    # First 199 bars have no valid slow MA
    assert (pos.iloc[:199] == 0.0).all()


def test_returns_series():
    prices = _prices([0.001] * 400)
    pos = SeasonalityTrendCombo().generate_positions(prices)
    assert isinstance(pos, pd.Series)
    assert len(pos) == len(prices)
