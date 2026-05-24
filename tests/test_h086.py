"""Tests for H086: LowVolTargetPortfolio."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.base import LowVolTargetPortfolio


def _prices(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(86)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = LowVolTargetPortfolio().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_shape_matches_input():
    prices = _prices(n=300)
    w = LowVolTargetPortfolio().generate_weights(prices)
    assert w.shape == (300, 2)


def test_weights_nonnegative():
    w = LowVolTargetPortfolio().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_at_most_one():
    w = LowVolTargetPortfolio().generate_weights(_prices())
    row_sums = w.sum(axis=1)
    assert (row_sums.round(6) <= 1.0).all()


def test_scale_down_in_high_vol():
    # High vol prices → portfolio vol > 8% → scale < 1 → weights sum < 1
    idx = pd.date_range("2003-01-01", periods=400, freq="B")
    rng = np.random.default_rng(0)
    spy = 100 * np.cumprod(1 + rng.normal(0.0, 0.04, 400))  # very high vol
    tlt = 100 * np.cumprod(1 + rng.normal(0.0, 0.02, 400))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = LowVolTargetPortfolio(vol_target=0.08).generate_weights(prices)
    tail = w.iloc[90:]
    # In high-vol regime, total exposure should be scaled below 1.0 on average
    assert tail.sum(axis=1).mean() < 0.95


def test_no_scale_in_low_vol():
    # Low vol prices → portfolio vol < 8% → scale = 1 → weights sum ≈ 1
    idx = pd.date_range("2003-01-01", periods=400, freq="B")
    rng = np.random.default_rng(1)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.001, 400))  # very low vol
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.0005, 400))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = LowVolTargetPortfolio(vol_target=0.08).generate_weights(prices)
    tail = w.iloc[90:]
    assert (tail.sum(axis=1).round(4) == 1.0).all()
