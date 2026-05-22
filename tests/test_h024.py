"""Focused tests for H024: TurnOfMonth."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import TurnOfMonth


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
    assert_no_lookahead(TurnOfMonth().generate_positions, prices)


def test_position_is_binary():
    """Positions must be 0 or 1 only."""
    prices = _prices([0.001] * 600)
    pos = TurnOfMonth().generate_positions(prices)
    unique_vals = set(pos.values.round(6))
    assert unique_vals.issubset({0.0, 1.0})


def test_first_days_of_month_are_active():
    """First n_days of each month should have position 1."""
    prices = _prices([0.001] * 800)
    n = 3
    pos = TurnOfMonth(n_days=n).generate_positions(prices)
    idx = prices.index

    for year in [2011, 2012]:
        for month in range(1, 13):
            month_days = idx[(idx.year == year) & (idx.month == month)]
            if len(month_days) >= n:
                for d in month_days[:n]:
                    assert pos[d] == pytest.approx(1.0), f"Day {d} should be active"


def test_last_days_of_month_are_active():
    """Last n_days of each month should have position 1."""
    prices = _prices([0.001] * 800)
    n = 3
    pos = TurnOfMonth(n_days=n).generate_positions(prices)
    idx = prices.index

    for year in [2011, 2012]:
        for month in range(1, 13):
            month_days = idx[(idx.year == year) & (idx.month == month)]
            if len(month_days) >= n:
                for d in month_days[-n:]:
                    assert pos[d] == pytest.approx(1.0), f"Day {d} should be active"


def test_mid_month_days_are_inactive():
    """Mid-month days (not near month boundaries) should have position 0."""
    prices = _prices([0.001] * 800)
    pos = TurnOfMonth(n_days=2).generate_positions(prices)
    idx = prices.index

    for year in [2011, 2012]:
        for month in range(1, 13):
            month_days = idx[(idx.year == year) & (idx.month == month)]
            if len(month_days) >= 8:
                mid = month_days[3:-3]
                assert (pos[mid] == 0.0).all()


def test_n_days_parameter():
    """Different n_days values should produce valid binary positions."""
    prices = _prices([0.001] * 600)
    for nd in [2, 3, 5]:
        pos = TurnOfMonth(n_days=nd).generate_positions(prices)
        unique_vals = set(pos.values.round(6))
        assert unique_vals.issubset({0.0, 1.0})


def test_returns_series():
    prices = _prices([0.001] * 400)
    pos = TurnOfMonth().generate_positions(prices)
    assert isinstance(pos, pd.Series)
    assert len(pos) == len(prices)
