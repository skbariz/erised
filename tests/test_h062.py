"""Focused tests for H062: SmallCapCyclical."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.engine import assert_no_lookahead_multi
from quant.strategies.base import SmallCapCyclical


def _prices(spy_rets, iwm_rets=None, hyg_rets=None, lqd_rets=None, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    for col, rets in [("iwm_close", iwm_rets), ("hyg_close", hyg_rets), ("lqd_close", lqd_rets)]:
        if rets is not None:
            px = [100.0]
            for r in rets:
                px.append(px[-1] * (1 + r))
            df[col] = px
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    n = 600
    prices = _prices(
        rng.normal(0.0005, 0.01, n).tolist(),
        rng.normal(0.0006, 0.012, n).tolist(),
        rng.normal(0.0004, 0.006, n).tolist(),
        rng.normal(0.0003, 0.004, n).tolist(),
    )
    assert_no_lookahead_multi(SmallCapCyclical().generate_weights, prices)


def test_long_iwm_when_both_signals_positive():
    """Healthy credit AND IWM outperforms SPY → long IWM."""
    n = 400
    prices = _prices(
        [0.001] * n,    # SPY moderate
        [0.002] * n,    # IWM stronger → size momentum positive
        [0.003] * n,    # HYG rising faster
        [0.001] * n,    # LQD slower → credit ok
    )
    w = SmallCapCyclical(credit_days=63, mom_days=126).generate_weights(prices)
    assert w["IWM"].iloc[-1] == 1.0
    assert w["SPY"].iloc[-1] == 0.0


def test_long_spy_when_credit_stressed():
    """LQD outperforms HYG → credit stress → SPY."""
    n = 400
    prices = _prices(
        [0.001] * n,
        [0.002] * n,    # IWM outperforming
        [0.001] * n,    # HYG slower than LQD
        [0.003] * n,    # LQD rising → credit stress
    )
    w = SmallCapCyclical(credit_days=63, mom_days=126).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0
    assert w["IWM"].iloc[-1] == 0.0


def test_long_spy_when_iwm_lags():
    """IWM underperforms SPY → no size premium → SPY even with healthy credit."""
    n = 400
    prices = _prices(
        [0.002] * n,    # SPY stronger
        [0.001] * n,    # IWM weaker
        [0.003] * n,    # HYG fine
        [0.001] * n,
    )
    w = SmallCapCyclical(credit_days=63, mom_days=126).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0


def test_fallback_to_spy_without_credit_data():
    """Without HYG/LQD: no credit signal → SPY (safe fallback)."""
    n = 400
    prices = _prices([0.001] * n, [0.002] * n)
    w = SmallCapCyclical().generate_weights(prices)
    assert (w["SPY"] == 1.0).all()
    assert (w["IWM"] == 0.0).all()


def test_columns_present():
    prices = _prices([0.001] * 300, [0.001] * 300, [0.001] * 300, [0.001] * 300)
    w = SmallCapCyclical().generate_weights(prices)
    assert set(w.columns) == {"SPY", "IWM"}


def test_weights_sum_to_one():
    rng = np.random.default_rng(2)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.012, n).tolist(),
        rng.normal(0, 0.006, n).tolist(),
        rng.normal(0, 0.004, n).tolist(),
    )
    w = SmallCapCyclical().generate_weights(prices)
    assert (w.sum(axis=1).round(9) == 1.0).all()
