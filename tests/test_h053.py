"""Focused tests for H053: CreditSpreadRegime."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import CreditSpreadRegime


def _prices(spy_rets, hyg_rets=None, lqd_rets=None, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    for col, rets in [("hyg_close", hyg_rets), ("lqd_close", lqd_rets)]:
        if rets is not None:
            px = [100.0]
            for r in rets:
                px.append(px[-1] * (1 + r))
            df[col] = px
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    n = 500
    prices = _prices(
        rng.normal(0.0005, 0.01, n).tolist(),
        rng.normal(0.0004, 0.006, n).tolist(),
        rng.normal(0.0003, 0.004, n).tolist(),
    )
    assert_no_lookahead(CreditSpreadRegime().generate_positions, prices)


def test_long_when_hyg_outperforms_lqd():
    """HYG rising faster than LQD → credit appetite healthy → long SPY."""
    n = 200
    prices = _prices([0.001] * n, [0.003] * n, [0.001] * n)
    pos = CreditSpreadRegime(momentum_days=63).generate_positions(prices)
    assert pos.iloc[-1] == 1.0


def test_flat_when_lqd_outperforms_hyg():
    """LQD rising faster than HYG → credit stress → flat."""
    n = 200
    prices = _prices([0.001] * n, [0.0005] * n, [0.003] * n)
    pos = CreditSpreadRegime(momentum_days=63).generate_positions(prices)
    assert pos.iloc[-1] == 0.0


def test_flat_without_credit_data():
    """Without hyg_close or lqd_close, defaults to flat (safe)."""
    prices = _prices([0.001] * 200)
    pos = CreditSpreadRegime().generate_positions(prices)
    assert (pos == 0.0).all()


def test_flat_before_warmup():
    """Before momentum_days of data, position must be 0."""
    n = 300
    prices = _prices([0.001] * n, [0.001] * n, [0.001] * n)
    pos = CreditSpreadRegime(momentum_days=63).generate_positions(prices)
    assert (pos.iloc[:63] == 0.0).all()


def test_positions_binary():
    rng = np.random.default_rng(9)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.006, n).tolist(),
        rng.normal(0, 0.004, n).tolist(),
    )
    pos = CreditSpreadRegime().generate_positions(prices)
    assert set(pos.round(9).unique()).issubset({0.0, 1.0})
