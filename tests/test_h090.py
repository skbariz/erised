"""Tests for H090: MonthlyRebalancingRiskParity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.base import MonthlyRebalancingRiskParity


def _prices(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(90)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = MonthlyRebalancingRiskParity().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_shape_matches_input():
    prices = _prices(n=400)
    w = MonthlyRebalancingRiskParity().generate_weights(prices)
    assert w.shape == (400, 2)


def test_weights_nonnegative():
    w = MonthlyRebalancingRiskParity().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_sum_at_most_one():
    w = MonthlyRebalancingRiskParity().generate_weights(_prices())
    assert (w.sum(axis=1).round(6) <= 1.0).all()


def test_fewer_unique_rows_than_daily():
    # Monthly rebalancing → many consecutive rows identical (forward-filled)
    # H021 daily → almost every row unique
    prices = _prices(n=500)
    w = MonthlyRebalancingRiskParity().generate_weights(prices)
    # After warmup, check that weight changes are infrequent (≈ monthly)
    tail = w.iloc[100:]
    spy_changes = tail["SPY"].diff().abs()
    # Number of non-zero changes should be ~monthly (≈ 19 in 400 rows)
    n_changes = (spy_changes > 1e-8).sum()
    # Daily rebalancing would have ~400 changes; monthly should have ~19
    assert n_changes < 50


def test_weights_constant_within_month():
    # Within a calendar month (after warmup), weights should be constant
    prices = _prices(n=500)
    w = MonthlyRebalancingRiskParity().generate_weights(prices)
    # Pick a month far past warmup (e.g., 2004-06)
    mask = (prices.index.year == 2004) & (prices.index.month == 6)
    month_rows = w.loc[mask]
    if len(month_rows) > 1:
        spy_vals = month_rows["SPY"].round(8)
        assert spy_vals.nunique() == 1
        tlt_vals = month_rows["TLT"].round(8)
        assert tlt_vals.nunique() == 1
