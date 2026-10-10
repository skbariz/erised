"""Focused tests for H063: DefensiveSectorOverlay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.backtest.engine import assert_no_lookahead_multi
from erised.strategies.base import DefensiveSectorOverlay


def _prices(spy_rets, vix_vals=None, xlp_rets=None, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    if vix_vals is not None:
        df["vix_close"] = ([vix_vals[0]] + vix_vals)[: n + 1]
    if xlp_rets is not None:
        xlp = [100.0]
        for r in xlp_rets:
            xlp.append(xlp[-1] * (1 + r))
        df["xlp_close"] = xlp
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    n = 600
    prices = _prices(
        rng.normal(0.0005, 0.01, n).tolist(),
        vix_vals=[20.0 + rng.uniform(-8, 15) for _ in range(n)],
        xlp_rets=rng.normal(0.0003, 0.005, n).tolist(),
    )
    assert_no_lookahead_multi(DefensiveSectorOverlay().generate_weights, prices)


def test_full_spy_when_uptrend_and_calm():
    """SPY rising (above MA) + VIX calm (<25) → 100% SPY."""
    n = 400
    prices = _prices([0.001] * n, vix_vals=[15.0] * n, xlp_rets=[0.0005] * n)
    w = DefensiveSectorOverlay(spy_ma=200, vix_thresh=25).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0
    assert w["XLP"].iloc[-1] == 0.0


def test_half_xlp_when_uptrend_but_stressed():
    """SPY rising + VIX stressed (>25) → 50/50 SPY+XLP."""
    n = 400
    prices = _prices([0.001] * n, vix_vals=[35.0] * n, xlp_rets=[0.0005] * n)
    w = DefensiveSectorOverlay(spy_ma=200, vix_thresh=25).generate_weights(prices)
    assert abs(w["SPY"].iloc[-1] - 0.5) < 1e-9
    assert abs(w["XLP"].iloc[-1] - 0.5) < 1e-9


def test_full_xlp_when_downtrend():
    """SPY in downtrend (below MA) → 100% XLP."""
    n = 400
    prices = _prices([-0.001] * n, vix_vals=[20.0] * n, xlp_rets=[0.0005] * n)
    w = DefensiveSectorOverlay(spy_ma=200, vix_thresh=25).generate_weights(prices)
    assert w["XLP"].iloc[-1] == 1.0
    assert w["SPY"].iloc[-1] == 0.0


def test_defaults_spy_without_xlp():
    """Without XLP column, can't go defensive — stays in SPY."""
    n = 400
    prices = _prices([0.001] * n, vix_vals=[35.0] * n)
    w = DefensiveSectorOverlay().generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0


def test_columns_present():
    prices = _prices([0.001] * 300, vix_vals=[20.0] * 300, xlp_rets=[0.0005] * 300)
    w = DefensiveSectorOverlay().generate_weights(prices)
    assert set(w.columns) == {"SPY", "XLP"}


def test_weights_sum_to_one():
    rng = np.random.default_rng(3)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        vix_vals=[15 + rng.uniform(0, 30) for _ in range(n)],
        xlp_rets=rng.normal(0, 0.005, n).tolist(),
    )
    w = DefensiveSectorOverlay().generate_weights(prices)
    assert (w.sum(axis=1).round(9) == 1.0).all()
