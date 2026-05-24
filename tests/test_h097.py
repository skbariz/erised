"""Tests for H097: CorrelationRiskParityHybrid."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.base import CorrelationRiskParityHybrid


def _prices(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(97)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = CorrelationRiskParityHybrid().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_shape_matches_input():
    prices = _prices(n=300)
    w = CorrelationRiskParityHybrid().generate_weights(prices)
    assert w.shape == (300, 2)


def test_weights_nonnegative():
    w = CorrelationRiskParityHybrid().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_sum_at_most_one():
    w = CorrelationRiskParityHybrid().generate_weights(_prices())
    assert (w.sum(axis=1).round(6) <= 1.0).all()


def test_crisis_weights_when_correlation_positive():
    # When SPY and TLT move together (positive correlation) → crisis weights
    # Construct prices where both move up together: positive correlation
    n = 400
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    common = rng.normal(0.001, 0.01, n)
    spy = 100 * np.cumprod(1 + common)
    tlt = 100 * np.cumprod(1 + common * 0.8)  # strongly correlated with SPY
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = CorrelationRiskParityHybrid(corr_lookback=60, crisis_tlt=0.90).generate_weights(prices)
    # After warmup, correlation is positive → TLT weight should be 0.90
    tail = w.iloc[100:]
    assert (tail["TLT"].round(6) == 0.90).mean() > 0.8


def test_normal_market_roughly_h021_weights():
    # When SPY and TLT have negative correlation (normal market), weights follow H021
    n = 400
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(1)
    spy_noise = rng.normal(0.001, 0.01, n)
    # TLT moves opposite to SPY: negative correlation
    tlt_noise = -0.5 * spy_noise + rng.normal(0.0, 0.002, n)
    spy = 100 * np.cumprod(1 + spy_noise)
    tlt = 100 * np.cumprod(1 + tlt_noise)
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = CorrelationRiskParityHybrid(corr_lookback=60).generate_weights(prices)
    # When no crisis: both SPY and TLT should have positive weights (H021 parity)
    tail = w.iloc[100:]
    assert (tail["SPY"] > 0).all() and (tail["TLT"] > 0).all()
