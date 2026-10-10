"""Focused tests for H019: SPYGLDRotation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead_multi
from erised.strategies import SPYGLDRotation


def _prices(spy_rets, gld_rets, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    gld = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    for r in gld_rets:
        gld.append(gld[-1] * (1 + r))
    return pd.DataFrame({"adj_close": spy, "gld_close": gld}, index=idx)


def test_no_lookahead():
    prices = _prices([0.001] * 700, [0.0005] * 700)
    assert_no_lookahead_multi(SPYGLDRotation().generate_weights, prices)


def test_spy_when_momentum_positive():
    prices = _prices([0.002] * 300, [0.001] * 300)
    w = SPYGLDRotation().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(1.0)
    assert w["GLD"].iloc[-1] == pytest.approx(0.0)


def test_gld_when_spy_momentum_negative():
    prices = _prices([-0.001] * 300, [0.001] * 300)
    w = SPYGLDRotation().generate_weights(prices)
    assert w["GLD"].iloc[-1] == pytest.approx(1.0)
    assert w["SPY"].iloc[-1] == pytest.approx(0.0)


def test_always_fully_invested_post_warmup():
    """Sum of weights must be 1.0 after lookback window warms up."""
    prices = _prices([0.001] * 300, [-0.001] * 300)
    w = SPYGLDRotation().generate_weights(prices)
    valid = (w["SPY"] + w["GLD"]) > 0
    assert np.allclose((w["SPY"] + w["GLD"])[valid].values, 1.0)


def test_no_cash_allocation():
    """Strategy has no cash state — must always choose SPY or GLD."""
    prices = _prices([-0.001] * 300, [-0.001] * 300)
    w = SPYGLDRotation().generate_weights(prices)
    # Even when both assets fall, GLD gets allocated (SPY negative → GLD)
    assert w["GLD"].iloc[-1] == pytest.approx(1.0)


def test_weights_binary():
    prices = _prices([0.001] * 300, [0.002] * 300)
    w = SPYGLDRotation().generate_weights(prices)
    unique_vals = set(w.values.ravel().round(6))
    assert unique_vals.issubset({0.0, 1.0})


def test_lookback_parameter_accepted():
    prices = _prices([0.001] * 300, [0.001] * 300)
    for lm in [3, 6, 9, 12]:
        w = SPYGLDRotation(lookback_months=lm).generate_weights(prices)
        assert w.columns.tolist() == ["SPY", "GLD"]
