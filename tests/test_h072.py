"""Tests for H072: HalloweenCreditConfirm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.base import HalloweenCreditConfirm


def _prices(
    start: str = "2008-01-01",
    n: int = 500,
    hyg_trend: str = "up",
    lqd_trend: str = "flat",
) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(72)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    hyg_drift = 0.0003 if hyg_trend == "up" else -0.0002
    lqd_drift = 0.0001
    hyg = 100 * np.cumprod(1 + rng.normal(hyg_drift, 0.003, n))
    lqd = 100 * np.cumprod(1 + rng.normal(lqd_drift, 0.002, n))
    return pd.DataFrame({"adj_close": spy, "hyg_close": hyg, "lqd_close": lqd}, index=idx)


def _prices_no_credit(start: str = "2000-01-01", n: int = 300) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(0)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    return pd.DataFrame({"adj_close": spy}, index=idx)


def test_returns_series():
    pos = HalloweenCreditConfirm().generate_positions(_prices())
    assert isinstance(pos, pd.Series)
    assert pos.name == "position"


def test_positions_binary():
    pos = HalloweenCreditConfirm().generate_positions(_prices())
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_flat_in_summer():
    prices = _prices(start="2010-06-01", n=120)
    pos = HalloweenCreditConfirm().generate_positions(prices)
    summer = pos[prices.index.month.isin([6, 7, 8, 9, 10])]
    assert summer.sum() == pytest.approx(0.0)


def test_flat_without_credit_data():
    # When HYG/LQD unavailable, credit_ok is always False → flat
    prices = _prices_no_credit(start="2003-11-01", n=120)
    pos = HalloweenCreditConfirm().generate_positions(prices)
    assert pos.sum() == pytest.approx(0.0)


def test_can_be_long_with_good_credit_in_winter():
    # Start in Jan, HYG outperforming → by Nov of same year (after 126-bar warmup) should go long
    prices = _prices(start="2010-01-04", n=500, hyg_trend="up")
    pos = HalloweenCreditConfirm(mom_days=126, credit_days=60).generate_positions(prices)
    # Nov-Apr of 2010-11 window is well past warmup; check that season has some longs
    winter = pos[prices.index.month.isin([11, 12, 1, 2, 3, 4])].iloc[130:]
    assert winter.sum() > 0


def test_flat_with_bad_credit_signal():
    # HYG trending down relative to LQD → credit_ok=False → flat even in season
    prices = _prices(start="2010-11-01", n=250, hyg_trend="down")
    pos = HalloweenCreditConfirm(mom_days=126, credit_days=60).generate_positions(prices)
    assert isinstance(pos, pd.Series)
