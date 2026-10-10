"""Focused tests for H054: YieldCurveProxy."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.backtest.engine import assert_no_lookahead
from erised.strategies import YieldCurveProxy


def _prices(spy_rets, shy_rets=None, tlt_rets=None, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    for col, rets in [("shy_close", shy_rets), ("tlt_close", tlt_rets)]:
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
        rng.normal(0.0002, 0.002, n).tolist(),
        rng.normal(0.0003, 0.005, n).tolist(),
    )
    assert_no_lookahead(YieldCurveProxy().generate_positions, prices)


def test_long_when_tlt_outperforms_shy():
    """TLT rising faster than SHY → long duration outperforming → curve ok → long SPY."""
    n = 200
    prices = _prices([0.001] * n, [0.001] * n, [0.004] * n)
    pos = YieldCurveProxy(mom_days=63).generate_positions(prices)
    assert pos.iloc[-1] == 1.0


def test_flat_when_shy_outperforms_tlt():
    """SHY rising faster than TLT → short end outperforming → curve inverted → flat."""
    n = 200
    prices = _prices([0.001] * n, [0.004] * n, [0.001] * n)
    pos = YieldCurveProxy(mom_days=63).generate_positions(prices)
    assert pos.iloc[-1] == 0.0


def test_defaults_long_without_shy_column():
    """Without shy_close, strategy defaults to always long SPY."""
    prices = _prices([0.001] * 200)
    pos = YieldCurveProxy().generate_positions(prices)
    assert (pos == 1.0).all()


def test_flat_before_warmup():
    n = 300
    prices = _prices([0.001] * n, [0.001] * n, [0.001] * n)
    pos = YieldCurveProxy(mom_days=63).generate_positions(prices)
    assert (pos.iloc[:63] == 0.0).all()


def test_positions_binary():
    rng = np.random.default_rng(11)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.002, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
    )
    pos = YieldCurveProxy().generate_positions(prices)
    assert set(pos.round(9).unique()).issubset({0.0, 1.0})
