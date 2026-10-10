"""Focused tests for H026: QuarterEndDrift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies import QuarterEndDrift


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
    assert_no_lookahead(QuarterEndDrift().generate_positions, prices)


def test_position_is_binary():
    """Positions must be 0 or 1 only."""
    prices = _prices([0.001] * 600)
    pos = QuarterEndDrift().generate_positions(prices)
    unique_vals = set(pos.values.round(6))
    assert unique_vals.issubset({0.0, 1.0})


def test_last_days_of_quarter_are_active():
    """Last n_days of each quarter-end month should have position 1."""
    prices = _prices([0.001] * 800)
    n = 5
    pos = QuarterEndDrift(n_days=n).generate_positions(prices)
    idx = prices.index
    quarter_end_months = [3, 6, 9, 12]

    for year in [2011, 2012]:
        for month in quarter_end_months:
            month_days = idx[(idx.year == year) & (idx.month == month)]
            if len(month_days) >= n:
                for d in month_days[-n:]:
                    assert pos[d] == pytest.approx(1.0), f"Day {d} should be active"


def test_non_quarter_end_months_are_inactive():
    """Jan, Feb, Apr, May, Jul, Aug, Oct, Nov should always be flat."""
    prices = _prices([0.001] * 800)
    pos = QuarterEndDrift(n_days=5).generate_positions(prices)
    non_qe_months = [1, 2, 4, 5, 7, 8, 10, 11]
    non_qe_mask = prices.index.month.isin(non_qe_months)
    assert (pos[non_qe_mask] == 0.0).all()


def test_n_days_parameter():
    """Different n_days values should produce valid binary positions."""
    prices = _prices([0.001] * 600)
    for nd in [3, 5, 10]:
        pos = QuarterEndDrift(n_days=nd).generate_positions(prices)
        unique_vals = set(pos.values.round(6))
        assert unique_vals.issubset({0.0, 1.0})


def test_more_days_means_more_active_bars():
    """n_days=10 should have more active bars than n_days=3."""
    prices = _prices([0.001] * 800)
    pos3 = QuarterEndDrift(n_days=3).generate_positions(prices)
    pos10 = QuarterEndDrift(n_days=10).generate_positions(prices)
    assert pos10.sum() > pos3.sum()


def test_returns_series():
    prices = _prices([0.001] * 400)
    pos = QuarterEndDrift().generate_positions(prices)
    assert isinstance(pos, pd.Series)
    assert len(pos) == len(prices)
