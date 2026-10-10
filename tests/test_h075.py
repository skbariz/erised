"""Tests for H075: EarningsSeasonDrift."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.base import EarningsSeasonDrift


def _prices(start: str = "2000-01-01", n: int = 500) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(75)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({"adj_close": spy}, index=idx)


def test_returns_series():
    pos = EarningsSeasonDrift().generate_positions(_prices())
    assert isinstance(pos, pd.Series)
    assert pos.name == "position"


def test_positions_binary():
    pos = EarningsSeasonDrift().generate_positions(_prices())
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_active_only_in_earnings_months():
    prices = _prices(n=500)
    pos = EarningsSeasonDrift().generate_positions(prices)
    non_earnings = pos[~prices.index.month.isin([1, 4, 7, 10])]
    assert non_earnings.sum() == pytest.approx(0.0)


def test_active_within_first_n_days():
    prices = _prices(start="2010-01-01", n=252)
    strat = EarningsSeasonDrift(active_days=5)
    pos = strat.generate_positions(prices)
    # All active days should be within the first 5 trading days of Jan/Apr/Jul/Oct
    active_idx = prices.index[pos == 1.0]
    for dt in active_idx:
        assert dt.month in [1, 4, 7, 10]


def test_active_fraction_reasonable():
    # With active_days=15 and 4 earnings months, should be ~25% of trading days
    prices = _prices(n=1260)  # ~5 years
    pos = EarningsSeasonDrift(active_days=15).generate_positions(prices)
    fraction = pos.mean()
    assert 0.15 <= fraction <= 0.35


def test_different_active_days_changes_activity():
    prices = _prices(n=500)
    pos5 = EarningsSeasonDrift(active_days=5).generate_positions(prices)
    pos15 = EarningsSeasonDrift(active_days=15).generate_positions(prices)
    assert pos15.sum() > pos5.sum()


def test_shape_matches_input():
    prices = _prices(n=300)
    pos = EarningsSeasonDrift().generate_positions(prices)
    assert len(pos) == len(prices)
