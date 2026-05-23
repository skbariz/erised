"""Focused tests for H039: DeepDrawdownCrisisBuyer."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import DeepDrawdownCrisisBuyer


def _prices(levels, start="2000-01-03"):
    idx = pd.bdate_range(start=start, periods=len(levels))
    return pd.DataFrame({"adj_close": levels}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    px = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 800))
    prices = _prices(px.tolist())
    assert_no_lookahead(DeepDrawdownCrisisBuyer().generate_positions, prices)


def test_enters_long_below_entry_threshold():
    """Position becomes 1 once drawdown from rolling peak exceeds entry_dd."""
    # Price rises to 100, then falls to 75 (25% drawdown) — should be long
    px = [100.0] * 10 + [75.0] * 5
    prices = _prices(px)
    pos = DeepDrawdownCrisisBuyer(entry_dd=-0.20, exit_dd=-0.05, lookback=20).generate_positions(
        prices
    )
    assert pos.iloc[-1] == pytest.approx(1.0)


def test_flat_above_entry_threshold():
    """No entry when drawdown is only 5%."""
    px = [100.0] * 10 + [95.0] * 5
    prices = _prices(px)
    pos = DeepDrawdownCrisisBuyer(entry_dd=-0.20, exit_dd=-0.05, lookback=20).generate_positions(
        prices
    )
    assert pos.iloc[-1] == pytest.approx(0.0)


def test_exits_after_recovery():
    """Position exits once price recovers above exit_dd threshold."""
    # Enter at 75 (25% DD), then recover to 96 (4% DD = above -5% exit)
    px = [100.0] * 10 + [75.0] * 10 + [96.0] * 5
    prices = _prices(px)
    pos = DeepDrawdownCrisisBuyer(entry_dd=-0.20, exit_dd=-0.05, lookback=30).generate_positions(
        prices
    )
    assert pos.iloc[-1] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(7)
    px = 100 * np.cumprod(1 + rng.normal(0.0002, 0.015, 500))
    prices = _prices(px.tolist())
    pos = DeepDrawdownCrisisBuyer().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})
