"""Focused tests for H021: RiskParity."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead_multi
from quant.strategies.base import RiskParity


def _prices(spy_rets, tlt_rets, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]; tlt = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    for r in tlt_rets:
        tlt.append(tlt[-1] * (1 + r))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    n = 700
    spy_rets = rng.normal(0.0005, 0.01, n).tolist()
    tlt_rets = rng.normal(0.0002, 0.005, n).tolist()
    prices = _prices(spy_rets, tlt_rets)
    assert_no_lookahead_multi(RiskParity().generate_weights, prices)


def test_weights_sum_to_one_after_warmup():
    prices = _prices([0.001] * 300, [0.0005] * 300)
    w = RiskParity(vol_lookback=60).generate_weights(prices)
    valid = (w["SPY"] + w["TLT"]) > 0
    sums = (w["SPY"] + w["TLT"])[valid]
    assert np.allclose(sums.values, 1.0)


def test_higher_vol_gets_lower_weight():
    """Higher-volatility asset should receive smaller weight."""
    rng = np.random.default_rng(0)
    spy_rets = rng.normal(0, 0.025, 300).tolist()   # high vol
    tlt_rets = rng.normal(0, 0.005, 300).tolist()   # low vol
    prices = _prices(spy_rets, tlt_rets)
    w = RiskParity(vol_lookback=60).generate_weights(prices)
    # TLT (low vol) should have higher weight than SPY (high vol)
    assert w["TLT"].iloc[-1] > w["SPY"].iloc[-1]


def test_equal_vol_gives_50_50():
    """When both assets have equal volatility, weights should be ~50/50."""
    rng = np.random.default_rng(1)
    rets = rng.normal(0, 0.01, 300).tolist()
    prices = _prices(rets, rets)  # identical vol
    w = RiskParity(vol_lookback=60).generate_weights(prices)
    valid = (w["SPY"] + w["TLT"]) > 0
    assert np.allclose(w["SPY"][valid].values, 0.5, atol=0.02)
    assert np.allclose(w["TLT"][valid].values, 0.5, atol=0.02)


def test_weights_non_negative():
    """Risk parity weights must always be ≥ 0 (long-only)."""
    rng = np.random.default_rng(2)
    spy_rets = rng.normal(0, 0.01, 300).tolist()
    tlt_rets = rng.normal(0, 0.008, 300).tolist()
    prices = _prices(spy_rets, tlt_rets)
    w = RiskParity().generate_weights(prices)
    assert (w >= 0).all().all()


def test_columns():
    prices = _prices([0.001] * 200, [0.001] * 200)
    w = RiskParity().generate_weights(prices)
    assert list(w.columns) == ["SPY", "TLT"]


def test_vol_lookback_parameter():
    """Different vol lookbacks should produce valid weights."""
    prices = _prices([0.001] * 300, [0.0005] * 300)
    for vl in [20, 60, 120]:
        w = RiskParity(vol_lookback=vl).generate_weights(prices)
        valid = (w["SPY"] + w["TLT"]) > 0
        if valid.any():
            assert np.allclose((w["SPY"] + w["TLT"])[valid].values, 1.0)
