"""Tests for H076: FourAssetRiskParity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.base import FourAssetRiskParity


def _prices(n: int = 300, include_dbc: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2007-01-01", periods=n, freq="B")
    rng = np.random.default_rng(76)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    gld = 100 * np.cumprod(1 + rng.normal(0.0002, 0.008, n))
    dbc = 100 * np.cumprod(1 + rng.normal(0.0001, 0.012, n))
    df = pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "gld_close": gld}, index=idx)
    if include_dbc:
        df["dbc_close"] = dbc
    return df


def test_returns_four_columns_when_all_available():
    w = FourAssetRiskParity().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT", "GLD", "DBC"}


def test_returns_three_columns_without_dbc():
    w = FourAssetRiskParity().generate_weights(_prices(include_dbc=False))
    assert set(w.columns) == {"SPY", "TLT", "GLD"}


def test_weights_sum_to_one_after_warmup():
    w = FourAssetRiskParity(vol_lookback=60).generate_weights(_prices(n=300))
    tail = w.iloc[65:]
    sums = tail.sum(axis=1).round(6)
    assert (sums == 1.0).all()


def test_weights_nonnegative():
    w = FourAssetRiskParity().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_high_vol_asset_gets_lower_weight():
    # DBC has much higher vol (0.03) than TLT (0.003) → DBC should get lower weight
    idx = pd.date_range("2007-01-01", periods=300, freq="B")
    rng = np.random.default_rng(0)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 300))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.003, 300))
    gld = 100 * np.cumprod(1 + rng.normal(0.0002, 0.008, 300))
    dbc = 100 * np.cumprod(1 + rng.normal(0.0001, 0.030, 300))
    prices = pd.DataFrame(
        {"adj_close": spy, "tlt_close": tlt, "gld_close": gld, "dbc_close": dbc}, index=idx
    )
    w = FourAssetRiskParity(vol_lookback=60).generate_weights(prices)
    # TLT (lowest vol) should have highest weight; DBC (highest vol) lowest
    tail = w.iloc[70:]
    assert (tail["TLT"] > tail["DBC"]).mean() > 0.9


def test_shape_matches_input():
    prices = _prices(n=200)
    w = FourAssetRiskParity().generate_weights(prices)
    assert w.shape[0] == 200
