"""Tests for H089: H043H021RegimeBlend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.base import H043H021RegimeBlend


def _prices(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(89)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = H043H021RegimeBlend().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_shape_matches_input():
    prices = _prices(n=400)
    w = H043H021RegimeBlend().generate_weights(prices)
    assert w.shape == (400, 2)


def test_weights_nonnegative():
    w = H043H021RegimeBlend().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_sum_at_most_one():
    w = H043H021RegimeBlend().generate_weights(_prices())
    assert (w.sum(axis=1).round(6) <= 1.0).all()


def test_summer_months_half_invested():
    # In summer (May-Oct): H043 is flat → blend is 0.5*H021; sum = 0.5
    prices = _prices(n=500)
    w = H043H021RegimeBlend(blend=0.5).generate_weights(prices)
    summer_mask = prices.index.month.isin([5, 6, 7, 8, 9, 10]) & (prices.index >= prices.index[220])
    summer_sums = w.loc[summer_mask].sum(axis=1).round(4)
    # Summer: H043=0, so w_spy=0.5*w021_spy, w_tlt=0.5*w021_tlt, sum=0.5
    assert (summer_sums == 0.5).all()


def test_winter_more_invested_than_summer():
    prices = _prices(n=500)
    w = H043H021RegimeBlend(blend=0.5).generate_weights(prices)
    summer = prices.index.month.isin([5, 6, 7, 8, 9, 10])
    winter = prices.index.month.isin([11, 12, 1, 2, 3, 4])
    late = prices.index >= prices.index[220]
    # Winter should on average have more invested than summer
    assert w.loc[winter & late].sum(axis=1).mean() > w.loc[summer & late].sum(axis=1).mean()
