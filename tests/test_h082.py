"""Tests for H082: RiskAdjustedMomentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.base import RiskAdjustedMomentum


def _prices(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2004-01-01", periods=n, freq="B")
    rng = np.random.default_rng(82)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    gld = 100 * np.cumprod(1 + rng.normal(0.0002, 0.008, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "gld_close": gld}, index=idx)


def test_returns_correct_columns():
    w = RiskAdjustedMomentum().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT", "GLD"}


def test_shape_matches_input():
    prices = _prices(n=400)
    w = RiskAdjustedMomentum().generate_weights(prices)
    assert w.shape[0] == 400


def test_weights_nonnegative():
    w = RiskAdjustedMomentum().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_each_row_at_most_one_asset():
    w = RiskAdjustedMomentum().generate_weights(_prices())
    # Each row either flat (0) or fully invested in one asset (sum=1)
    row_sums = w.sum(axis=1)
    assert ((row_sums.round(6) == 0.0) | (row_sums.round(6) == 1.0)).all()


def test_flat_before_warmup():
    prices = _prices(n=500)
    w = RiskAdjustedMomentum(mom_days=252).generate_weights(prices)
    assert (w.iloc[:252].sum(axis=1) == 0.0).all()


def test_high_sharpe_asset_wins():
    # SPY has much higher return than TLT/GLD with same small noise → SPY wins by risk-adj score
    idx = pd.date_range("2004-01-01", periods=500, freq="B")
    rng = np.random.default_rng(999)
    noise = rng.normal(0, 0.001, 500)  # tiny noise so vol != 0
    spy = 100 * np.cumprod(1 + 0.005 + noise)  # +0.5%/day dominant
    tlt = 100 * np.cumprod(1 + 0.0002 + noise * 0.5)
    gld = 100 * np.cumprod(1 + 0.0003 + noise * 0.5)
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "gld_close": gld}, index=idx)
    w = RiskAdjustedMomentum().generate_weights(prices)
    tail = w.iloc[270:]
    # SPY has far higher risk-adjusted momentum → wins consistently
    assert (tail["SPY"] == 1.0).mean() > 0.95
    assert (tail["TLT"] == 0.0).all()
