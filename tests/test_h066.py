"""Tests for H066: VixRiskParityEnhanced."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.base import VixRiskParityEnhanced


def _prices(n: int = 300, vix_val: float = 20.0, spike_at: int | None = None) -> pd.DataFrame:
    idx = pd.date_range("2002-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    spy = 100 * np.cumprod(1 + rng.normal(0.0004, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0002, 0.005, n))
    vix = np.full(n, vix_val)
    if spike_at is not None:
        vix[spike_at : spike_at + 20] = 40.0
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "vix_close": vix}, index=idx)


def test_returns_spy_tlt_columns():
    strat = VixRiskParityEnhanced()
    w = strat.generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_weights_sum_to_one_in_normal_regime():
    strat = VixRiskParityEnhanced()
    w = strat.generate_weights(_prices(n=300, vix_val=20.0))
    # After warmup, weights should sum to ~1
    tail = w.iloc[70:]
    sums = (tail["SPY"] + tail["TLT"]).round(6)
    assert (sums == 1.0).all()


def test_crisis_weights_on_vix_spike():
    strat = VixRiskParityEnhanced(vix_spike=35.0, vix_resume=30.0, crisis_tlt=0.80)
    prices = _prices(n=300, vix_val=20.0, spike_at=150)
    w = strat.generate_weights(prices)
    # During the spike window, TLT weight should be 0.80
    crisis_rows = w.iloc[151:170]
    assert (crisis_rows["TLT"].round(2) == 0.80).all()
    assert (crisis_rows["SPY"].round(2) == 0.20).all()


def test_resumes_risk_parity_after_spike():
    strat = VixRiskParityEnhanced(vix_spike=35.0, vix_resume=30.0)
    prices = _prices(n=400, vix_val=20.0, spike_at=150)
    w = strat.generate_weights(prices)
    # After spike ends (VIX back to 20 < 30), should revert to risk-parity (not 0.80)
    post_spike = w.iloc[175:]
    assert not (post_spike["TLT"].round(2) == 0.80).all()


def test_no_data_before_warmup():
    strat = VixRiskParityEnhanced(vol_lookback=60)
    w = strat.generate_weights(_prices(n=300))
    # vol_lookback=60, min_periods=60, shift(1) → first valid row is index 60
    # rows 0..59 should be 0
    assert w.iloc[:60]["SPY"].sum() == pytest.approx(0.0)


def test_shape_matches_input():
    prices = _prices(n=200)
    w = VixRiskParityEnhanced().generate_weights(prices)
    assert w.shape == (200, 2)
