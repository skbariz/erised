"""Tests for H096: PCARiskRegime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.base import PCARiskRegime


def _prices(n: int = 600, spy_drift: float = 0.0003, tlt_drift: float = 0.0001) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(96)
    spy = 100 * np.cumprod(1 + rng.normal(spy_drift, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(tlt_drift, 0.005, n))
    gld = 100 * np.cumprod(1 + rng.normal(0.0001, 0.008, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "gld_close": gld}, index=idx)


def test_returns_spy_tlt_columns():
    w = PCARiskRegime().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_shape_matches_input():
    prices = _prices(n=500)
    w = PCARiskRegime().generate_weights(prices)
    assert w.shape == (500, 2)


def test_weights_nonnegative():
    w = PCARiskRegime().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_binary_sum():
    # Each row is either (1,0), (0,1), or (0,0) — never partial
    w = PCARiskRegime().generate_weights(_prices())
    row_sums = w.sum(axis=1)
    assert ((row_sums == 0) | (row_sums.round(6) == 1.0)).all()


def test_signal_eventually_fires():
    # After pca_lookback + signal_days warmup, some rows should be non-zero
    w = PCARiskRegime(pca_lookback=252, signal_days=21).generate_weights(_prices(n=600))
    tail = w.iloc[280:]
    assert tail.sum(axis=1).sum() > 0


def test_rising_spy_favors_spy_position():
    # Strong SPY drift with noise → signal points to SPY → SPY weight > TLT after warmup
    n = 600
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    spy = 100 * np.cumprod(1 + rng.normal(0.002, 0.01, n))  # strong uptrend
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))  # near-flat
    gld = 100 * np.cumprod(1 + rng.normal(0.0001, 0.008, n))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "gld_close": gld}, index=idx)
    w = PCARiskRegime(pca_lookback=252, signal_days=21).generate_weights(prices)
    tail = w.iloc[290:]
    assert tail["SPY"].mean() > tail["TLT"].mean()
