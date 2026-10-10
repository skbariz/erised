"""Focused tests for H040: ShortTermWeeklyReversal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies import ShortTermWeeklyReversal


def _prices(rets, start="2000-01-03"):
    n = len(rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in rets:
        px.append(px[-1] * (1 + r))
    return pd.DataFrame({"adj_close": px}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0003, 0.01, 800).tolist()
    prices = _prices(rets)
    assert_no_lookahead(ShortTermWeeklyReversal().generate_positions, prices)


def test_long_after_sharp_weekly_decline():
    """After a very bad 5-day return in bottom history, position should be 1."""
    rng = np.random.default_rng(10)
    # Build long history with small returns, then one sharp drop
    rets = rng.normal(0.0003, 0.005, 600).tolist()
    # Replace last 5 bars with a -5% per day decline
    rets[-5:] = [-0.05] * 5
    prices = _prices(rets)
    pos = ShortTermWeeklyReversal(hold_days=5, threshold=10).generate_positions(prices)
    assert pos.iloc[-1] == pytest.approx(1.0)


def test_flat_when_recent_return_is_high():
    """Build history with variance, then end with strong gains -> above p10 -> flat."""
    rng = np.random.default_rng(20)
    # 700 bars of normal variation to establish a distribution with negative tail
    rets = rng.normal(0.0003, 0.01, 700).tolist()
    # Final 5 bars: very strong gains, clearly above bottom decile
    rets[-5:] = [0.05] * 5
    prices = _prices(rets)
    pos = ShortTermWeeklyReversal(hold_days=5, threshold=10).generate_positions(prices)
    # Last bar: 5-day return is massive positive -> not in bottom 10% -> flat
    assert pos.iloc[-1] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(99)
    rets = rng.normal(0.0, 0.015, 1000).tolist()
    prices = _prices(rets)
    pos = ShortTermWeeklyReversal().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_zero_during_warmup():
    """Before history_min bars, strategy should be flat."""
    rets = [0.001] * 300
    prices = _prices(rets)
    pos = ShortTermWeeklyReversal(history_min=252).generate_positions(prices)
    assert (pos.iloc[:252] == 0.0).all()
