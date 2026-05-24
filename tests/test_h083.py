"""Tests for H083: ContrarianMultiAssetRotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.base import ContrarianMultiAssetRotation


def _prices(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2004-01-01", periods=n, freq="B")
    rng = np.random.default_rng(83)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    gld = 100 * np.cumprod(1 + rng.normal(0.0002, 0.008, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "gld_close": gld}, index=idx)


def test_returns_correct_columns():
    w = ContrarianMultiAssetRotation().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT", "GLD"}


def test_shape_matches_input():
    prices = _prices(n=400)
    w = ContrarianMultiAssetRotation().generate_weights(prices)
    assert w.shape[0] == 400


def test_weights_nonnegative():
    w = ContrarianMultiAssetRotation().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_each_row_at_most_one_asset():
    w = ContrarianMultiAssetRotation().generate_weights(_prices())
    row_sums = w.sum(axis=1)
    assert ((row_sums.round(6) == 0.0) | (row_sums.round(6) == 1.0)).all()


def test_flat_when_loser_still_falling():
    # TLT is clearly the worst 12m performer AND still falling (3m negative) → flat
    idx = pd.date_range("2004-01-01", periods=500, freq="B")
    spy = 100 * np.cumprod(np.full(500, 1.002))   # clearly winning
    tlt = 100 * np.cumprod(np.full(500, 0.997))   # clearly losing AND still falling
    gld = 100 * np.cumprod(np.full(500, 1.001))   # middle
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "gld_close": gld}, index=idx)
    w = ContrarianMultiAssetRotation().generate_weights(prices)
    # TLT keeps falling → confirm_days return still negative → no allocation
    tail = w.iloc[300:]
    assert (tail.sum(axis=1) == 0.0).all()


def test_loser_with_reversal_gets_allocated():
    # SPY crashes then recovers; TLT rises; SPY is worst 12m but 3m positive after recovery
    n = 600
    idx = pd.date_range("2004-01-01", periods=n, freq="B")
    # SPY: crash for first 300 days, recover for next 300
    spy_crash = 100 * np.cumprod(np.full(300, 0.996))
    spy_recov = spy_crash[-1] * np.cumprod(np.full(300, 1.003))
    spy = np.concatenate([spy_crash, spy_recov])
    tlt = 100 * np.cumprod(np.full(n, 1.001))
    gld = 100 * np.cumprod(np.full(n, 1.0005))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "gld_close": gld}, index=idx)
    w = ContrarianMultiAssetRotation(mom_days=252, confirm_days=63).generate_weights(prices)
    # After recovery starts (row 370+): SPY worst 12m AND 3m positive → allocated
    tail = w.iloc[370:]
    assert (tail["SPY"] == 1.0).any()
