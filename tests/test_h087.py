"""Tests for H087: RatchetDrawdownProtection."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.base import RatchetDrawdownProtection


def _prices(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(87)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = RatchetDrawdownProtection().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_shape_matches_input():
    prices = _prices(n=300)
    w = RatchetDrawdownProtection().generate_weights(prices)
    assert w.shape == (300, 2)


def test_weights_nonnegative():
    w = RatchetDrawdownProtection().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_sum_at_most_one():
    w = RatchetDrawdownProtection().generate_weights(_prices())
    assert (w.sum(axis=1).round(6) <= 1.0).all()


def test_full_scale_at_peak():
    # Rising market → always at peak → scale = 1 → weights sum ≈ 1 (after warmup)
    idx = pd.date_range("2003-01-01", periods=400, freq="B")
    spy = 100 * np.cumprod(np.full(400, 1.001))  # monotonically rising
    tlt = 100 * np.cumprod(np.full(400, 1.0005))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = RatchetDrawdownProtection(drawdown_limit=0.15).generate_weights(prices)
    tail = w.iloc[80:]
    assert (tail.sum(axis=1).round(4) == 1.0).all()


def test_reduced_scale_after_crash():
    # Portfolio crashes → drawdown > 0 → scale < 1 → weights sum < 1
    n = 400
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    # Rise then crash
    spy = np.concatenate([
        100 * np.cumprod(np.full(200, 1.001)),
        100 * np.cumprod(np.full(200, 0.994)) * 100 * np.cumprod(np.full(200, 1.001))[-1] / 100,
    ])
    tlt = 100 * np.cumprod(1 + np.random.default_rng(0).normal(0.0001, 0.002, n))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = RatchetDrawdownProtection(drawdown_limit=0.15).generate_weights(prices)
    crash_rows = w.iloc[220:260]
    assert crash_rows.sum(axis=1).mean() < 0.95
