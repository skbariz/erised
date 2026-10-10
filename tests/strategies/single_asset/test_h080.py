"""Tests for H080: RiskParityYieldCurveOverlay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies import RiskParityYieldCurveOverlay


def _prices(n: int = 300, curve_inverted: bool = False) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(80)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    if curve_inverted:
        shy = 100 * np.cumprod(1 + rng.normal(0.0005, 0.001, n))  # SHY outperforms TLT
    else:
        shy = 100 * np.cumprod(1 + rng.normal(-0.0001, 0.001, n))  # SHY underperforms
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "shy_close": shy}, index=idx)


def _prices_no_shy(n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(0)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = RiskParityYieldCurveOverlay().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_normal_curve_matches_risk_parity():
    # When curve is not inverted, weights should be standard H021
    prices = _prices(n=300, curve_inverted=False)
    w = RiskParityYieldCurveOverlay(vol_lookback=60, curve_days=63).generate_weights(prices)
    tail = w.iloc[70:]
    sums = (tail["SPY"] + tail["TLT"]).round(6)
    assert (sums == 1.0).all()


def test_inverted_curve_reduces_spy_weight():
    # When curve inverted, SPY weight should be lower (scaled by spy_scale=0.50)
    prices_normal = _prices(n=300, curve_inverted=False)
    prices_inverted = _prices(n=300, curve_inverted=True)
    strat = RiskParityYieldCurveOverlay(vol_lookback=60, curve_days=63, spy_scale=0.50)
    w_normal = strat.generate_weights(prices_normal)
    w_inverted = strat.generate_weights(prices_inverted)
    tail_n = w_normal["SPY"].iloc[70:]
    tail_i = w_inverted["SPY"].iloc[70:]
    # Inverted curve should result in lower average SPY weight
    assert tail_i.mean() < tail_n.mean()


def test_weights_sum_to_one():
    prices = _prices(n=300, curve_inverted=True)
    w = RiskParityYieldCurveOverlay(vol_lookback=60).generate_weights(prices)
    tail = w.iloc[70:]
    sums = (tail["SPY"] + tail["TLT"]).round(6)
    assert (sums == 1.0).all()


def test_no_shy_uses_normal_parity():
    # Without SHY, curve signal is always False → standard H021 weights
    prices = _prices_no_shy(n=300)
    w = RiskParityYieldCurveOverlay(vol_lookback=60).generate_weights(prices)
    tail = w.iloc[70:]
    sums = (tail["SPY"] + tail["TLT"]).round(6)
    assert (sums == 1.0).all()


def test_weights_nonnegative():
    w = RiskParityYieldCurveOverlay().generate_weights(_prices(curve_inverted=True))
    assert (w >= -1e-10).all().all()


def test_shape_matches_input():
    prices = _prices(n=200)
    w = RiskParityYieldCurveOverlay().generate_weights(prices)
    assert w.shape == (200, 2)
