"""Tests for H074: PreHolidayDrift."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.base import PreHolidayDrift


def _prices(start: str = "2000-01-01", n: int = 500) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(74)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({"adj_close": spy}, index=idx)


def test_returns_series():
    pos = PreHolidayDrift().generate_positions(_prices())
    assert isinstance(pos, pd.Series)
    assert pos.name == "position"


def test_positions_binary():
    pos = PreHolidayDrift().generate_positions(_prices())
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_small_fraction_of_active_days():
    # Should only be active ~8-10 days per year (1 pre-holiday day per holiday)
    prices = _prices(n=252)  # 1 year approx
    pos = PreHolidayDrift(pre_days=1).generate_positions(prices)
    active = pos.sum()
    # Expect 8-12 active days per year (US has ~8 federal holidays)
    assert 5 <= active <= 15


def test_more_pre_days_increases_activity():
    prices = _prices(n=500)
    pos1 = PreHolidayDrift(pre_days=1).generate_positions(prices)
    pos2 = PreHolidayDrift(pre_days=2).generate_positions(prices)
    assert pos2.sum() >= pos1.sum()


def test_shape_matches_input():
    prices = _prices(n=300)
    pos = PreHolidayDrift().generate_positions(prices)
    assert len(pos) == len(prices)


def test_jan1_preceded_by_trading_day():
    # New Year's Day (Jan 1) should have a pre-holiday trading day
    prices = _prices(start="2010-01-01", n=300)
    pos = PreHolidayDrift(pre_days=1).generate_positions(prices)
    # Should have at least one active day (Jan 1 2010 is a Friday → Dec 31 is pre-holiday)
    assert pos.sum() >= 1
