"""Tests for H073: SellInMayYieldCurveConfirm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.base import SellInMayYieldCurveConfirm


def _prices(start: str = "2003-01-01", n: int = 600, curve_normal: bool = True) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(73)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, n))
    # TLT declining → price below its 200MA → bond_ok=True
    tlt = 100 * np.cumprod(1 + rng.normal(-0.0002, 0.004, n))
    if curve_normal:
        # Normal curve: SHY also declining but slower than TLT → TLT_ret < SHY_ret is still
        # negative for TLT; but we need shy_r < tlt_r which means SHY underperforms TLT.
        # Achievable when TLT declines less than SHY, OR both rising with TLT rising more.
        # Use: both decline, but SHY declines faster → shy_r < tlt_r ✓
        shy = 100 * np.cumprod(1 + rng.normal(-0.0005, 0.001, n))
    else:
        # Inverted: SHY outperforms TLT → shy_r > tlt_r
        shy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.001, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "shy_close": shy}, index=idx)


def test_raises_without_tlt():
    strat = SellInMayYieldCurveConfirm()
    df = pd.DataFrame(
        {"adj_close": [100.0] * 100, "shy_close": [100.0] * 100},
        index=pd.date_range("2003-01-01", periods=100, freq="B"),
    )
    with pytest.raises(ValueError, match="tlt_close"):
        strat.generate_positions(df)


def test_returns_series():
    pos = SellInMayYieldCurveConfirm().generate_positions(_prices())
    assert isinstance(pos, pd.Series)
    assert pos.name == "position"


def test_positions_binary():
    pos = SellInMayYieldCurveConfirm().generate_positions(_prices())
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_flat_in_summer():
    prices = _prices(start="2005-06-01", n=100)
    pos = SellInMayYieldCurveConfirm().generate_positions(prices)
    summer = pos[prices.index.month.isin([6, 7, 8, 9, 10])]
    assert summer.sum() == pytest.approx(0.0)


def test_flat_without_shy():
    # When SHY unavailable, not_inverted is always False → flat
    idx = pd.date_range("2003-11-01", periods=120, freq="B")
    rng = np.random.default_rng(5)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, 120))
    tlt = 100 * np.cumprod(1 + rng.normal(-0.0002, 0.004, 120))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    pos = SellInMayYieldCurveConfirm().generate_positions(prices)
    assert pos.sum() == pytest.approx(0.0)


def test_shape_matches_input():
    prices = _prices()
    pos = SellInMayYieldCurveConfirm().generate_positions(prices)
    assert len(pos) == len(prices)


def test_can_be_long_in_winter_normal_curve():
    # Start in November, TLT declining (bond_ok), curve normal (SHY slow)
    prices = _prices(start="2005-11-01", n=500, curve_normal=True)
    pos = SellInMayYieldCurveConfirm(tlt_ma=200, curve_days=63).generate_positions(prices)
    winter = pos[prices.index.month.isin([11, 12, 1, 2, 3, 4])]
    assert winter.sum() > 0
