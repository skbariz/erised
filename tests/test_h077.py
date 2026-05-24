"""Tests for H077: RiskParityMomentumFilter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.base import RiskParityMomentumFilter


def _prices(n: int = 400, spy_up: bool = True, tlt_up: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(77)
    spy_d = 0.0004 if spy_up else -0.0004
    tlt_d = 0.0002 if tlt_up else -0.0003
    spy = 100 * np.cumprod(1 + rng.normal(spy_d, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(tlt_d, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = RiskParityMomentumFilter().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_weights_nonnegative():
    w = RiskParityMomentumFilter().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_sum_at_most_one():
    w = RiskParityMomentumFilter().generate_weights(_prices())
    total = w["SPY"] + w["TLT"]
    assert (total <= 1.0 + 1e-10).all()


def test_flat_when_both_negative_momentum():
    # Both assets deterministically declining → all weights 0 after warmup
    idx = pd.date_range("2003-01-01", periods=400, freq="B")
    spy = 100 * np.cumprod(np.full(400, 0.998))  # -0.2%/day, guaranteed negative mom
    tlt = 100 * np.cumprod(np.full(400, 0.997))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = RiskParityMomentumFilter(mom_days=126).generate_weights(prices)
    tail = w.iloc[140:]
    assert (tail["SPY"].round(6) == 0.0).all()
    assert (tail["TLT"].round(6) == 0.0).all()


def test_spy_only_when_tlt_negative():
    # SPY deterministically rising, TLT deterministically falling
    idx = pd.date_range("2003-01-01", periods=400, freq="B")
    spy = 100 * np.cumprod(np.full(400, 1.002))  # +0.2%/day, guaranteed positive mom
    tlt = 100 * np.cumprod(np.full(400, 0.997))  # -0.3%/day, guaranteed negative mom
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = RiskParityMomentumFilter(mom_days=126).generate_weights(prices)
    tail = w.iloc[140:]
    assert (tail["TLT"].round(6) == 0.0).all()
    assert (tail["SPY"].round(6) > 0).all()


def test_both_positive_gives_parity_like_weights():
    # Both deterministically rising → all weights sum to 1.0
    idx = pd.date_range("2003-01-01", periods=400, freq="B")
    spy = 100 * np.cumprod(np.full(400, 1.002))  # guaranteed positive mom
    tlt = 100 * np.cumprod(np.full(400, 1.001))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = RiskParityMomentumFilter(mom_days=63).generate_weights(prices)
    tail = w.iloc[70:]
    total = tail["SPY"] + tail["TLT"]
    assert (total.round(4) == 1.0).all()


def test_shape_matches_input():
    prices = _prices(n=250)
    w = RiskParityMomentumFilter().generate_weights(prices)
    assert w.shape == (250, 2)
