"""Tests for H078: SeasonalRiskParity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies import SeasonalRiskParity


def _prices(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(78)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = SeasonalRiskParity().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_spy_zero_in_summer():
    # May-Oct: SPY should be 0
    prices = _prices(n=500)
    w = SeasonalRiskParity(vol_lookback=60).generate_weights(prices)
    summer_mask = prices.index.month.isin([5, 6, 7, 8, 9, 10])
    summer_spy = w["SPY"][summer_mask]
    assert (summer_spy.round(6) == 0.0).all()


def test_tlt_one_in_summer_after_warmup():
    # May-Oct: TLT = 1.0 (when data available and past warmup)
    prices = _prices(n=500)
    w = SeasonalRiskParity(vol_lookback=60).generate_weights(prices)
    summer_mask = prices.index.month.isin([5, 6, 7, 8, 9, 10]) & (prices.index >= prices.index[70])
    summer_tlt = w["TLT"][summer_mask]
    assert (summer_tlt.round(6) == 1.0).all()


def test_risk_parity_in_winter():
    # Nov-Apr: both SPY and TLT should have positive weights
    prices = _prices(n=500)
    w = SeasonalRiskParity(vol_lookback=60).generate_weights(prices)
    winter_mask = prices.index.month.isin([11, 12, 1, 2, 3, 4]) & (prices.index >= prices.index[70])
    assert (w["SPY"][winter_mask] > 0).any()
    assert (w["TLT"][winter_mask] > 0).any()


def test_winter_weights_sum_to_one():
    prices = _prices(n=500)
    w = SeasonalRiskParity(vol_lookback=60).generate_weights(prices)
    winter_mask = prices.index.month.isin([11, 12, 1, 2, 3, 4]) & (prices.index >= prices.index[70])
    sums = (w["SPY"] + w["TLT"])[winter_mask]
    assert (sums.round(4) == 1.0).all()


def test_shape_matches_input():
    prices = _prices(n=300)
    w = SeasonalRiskParity().generate_weights(prices)
    assert w.shape == (300, 2)
