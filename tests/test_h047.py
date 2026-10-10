"""Focused tests for H047: PreEarningsDrift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies.base import PreEarningsDrift


def _prices(rets, start="2010-01-04"):
    n = len(rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in rets:
        px.append(px[-1] * (1 + r))
    return pd.DataFrame({"adj_close": px}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, 600).tolist()
    prices = _prices(rets)
    assert_no_lookahead(PreEarningsDrift().generate_positions, prices)


def test_long_in_early_earnings_month():
    """First N days of January -> position 1.0."""
    rets = [0.001] * 300
    prices = _prices(rets, start="2010-01-04")  # starts on first business day of Jan
    pos = PreEarningsDrift(n_days=5, earnings_months=(1, 4, 7, 10)).generate_positions(prices)
    # First bar is Jan 4, which is within the first 5 trading days of January
    assert pos.iloc[0] == pytest.approx(1.0)


def test_flat_outside_earnings_months():
    """February (non-earnings month) -> position 0.0."""
    rets = [0.001] * 300
    prices = _prices(rets, start="2010-02-01")  # starts in February
    pos = PreEarningsDrift(n_days=5, earnings_months=(1, 4, 7, 10)).generate_positions(prices)
    assert pos.iloc[0] == pytest.approx(0.0)


def test_flat_in_mid_earnings_month():
    """Mid-January of second year (outside n_days window) -> position 0.0."""
    rets = [0.001] * 600
    # Start from Jan 4 to anchor the calendar, then check mid-Jan of 2nd year
    prices = _prices(rets, start="2010-01-04")
    pos = PreEarningsDrift(n_days=5, earnings_months=(1, 4, 7, 10)).generate_positions(prices)
    # Find Jan 2011 dates that are after the 5th trading day
    jan_2011 = pos[(pos.index.year == 2011) & (pos.index.month == 1)]
    if len(jan_2011) > 5:
        assert jan_2011.iloc[5] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(7)
    rets = rng.normal(0, 0.01, 600).tolist()
    prices = _prices(rets)
    pos = PreEarningsDrift().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_active_in_all_four_earnings_months():
    """Confirm positions fire in Jan, Apr, Jul, Oct."""
    rets = [0.001] * 2000
    prices = _prices(rets, start="2010-01-04")
    pos = PreEarningsDrift(n_days=5).generate_positions(prices)
    for month in (1, 4, 7, 10):
        month_early = pos[(pos.index.month == month)]
        if len(month_early) > 0:
            assert month_early.iloc[0] == pytest.approx(1.0)
