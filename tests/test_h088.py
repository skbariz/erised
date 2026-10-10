"""Tests for H088: H001H021DirectBlend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.base import H001H021DirectBlend


def _prices(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(88)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = H001H021DirectBlend().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_shape_matches_input():
    prices = _prices(n=400)
    w = H001H021DirectBlend().generate_weights(prices)
    assert w.shape == (400, 2)


def test_weights_nonnegative():
    w = H001H021DirectBlend().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_sum_at_most_one():
    w = H001H021DirectBlend().generate_weights(_prices())
    assert (w.sum(axis=1).round(6) <= 1.0).all()


def test_blend_50_50_when_h001_long():
    # Monotonically rising → H001 always long → w_spy = 0.5 + 0.5*w021_spy
    idx = pd.date_range("2003-01-01", periods=500, freq="B")
    spy = 100 * np.cumprod(np.full(500, 1.002))
    tlt = 100 * np.cumprod(np.full(500, 1.001))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = H001H021DirectBlend(blend=0.5).generate_weights(prices)
    tail = w.iloc[220:]
    # When H001 is always long, SPY weight must be > 0.5 (0.5 from H001 + some from H021)
    assert (tail["SPY"] > 0.5).all()


def test_tlt_always_nonzero_after_warmup():
    # TLT weight = 0.5 * w021_tlt > 0 (after warmup)
    prices = _prices(n=500)
    w = H001H021DirectBlend(blend=0.5).generate_weights(prices)
    tail = w.iloc[220:]
    assert (tail["TLT"] > 0).all()


def test_blend_100_h001_equals_single_asset():
    # blend=1.0 → 100% H001; TLT weight = 0 always; SPY = 0 or 1
    idx = pd.date_range("2003-01-01", periods=500, freq="B")
    spy = 100 * np.cumprod(np.full(500, 1.002))
    tlt = 100 * np.cumprod(np.full(500, 1.001))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = H001H021DirectBlend(blend=1.0).generate_weights(prices)
    assert (w["TLT"].round(6) == 0.0).all()
    tail = w.iloc[220:]
    assert set(tail["SPY"].round(6).unique()).issubset({0.0, 1.0})
