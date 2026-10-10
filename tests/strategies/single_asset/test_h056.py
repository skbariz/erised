"""Focused tests for H056: YieldCurveRotation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.backtest.engine import assert_no_lookahead_multi
from erised.strategies import YieldCurveRotation


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
    assert_no_lookahead_multi(YieldCurveRotation().generate_weights, prices)


def test_long_spy_when_curve_normal():
    """TLT rising faster than SHY → normal curve → long SPY."""
    n = 200
    prices = _prices([0.001] * n, [0.001] * n, [0.004] * n)
    w = YieldCurveRotation(mom_days=63).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0
    assert w["TLT"].iloc[-1] == 0.0


def test_long_tlt_when_curve_inverted():
    """SHY rising faster than TLT → inverted curve → long TLT."""
    n = 200
    prices = _prices([0.001] * n, [0.004] * n, [0.001] * n)
    w = YieldCurveRotation(mom_days=63).generate_weights(prices)
    assert w["TLT"].iloc[-1] == 1.0
    assert w["SPY"].iloc[-1] == 0.0


def test_flat_before_data():
    """Without SHY/TLT data, all weights are zero."""
    prices = _prices([0.001] * 200)
    w = YieldCurveRotation().generate_weights(prices)
    assert (w.sum(axis=1) == 0.0).all()


def test_weights_sum_to_one_or_zero():
    """Each row is either fully invested or flat — never partial."""
    rng = np.random.default_rng(7)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.002, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
    )
    w = YieldCurveRotation().generate_weights(prices)
    row_sums = w.sum(axis=1)
    assert ((row_sums == 0.0) | (row_sums == 1.0)).all()


def test_columns_present():
    prices = _prices([0.001] * 200, [0.001] * 200, [0.001] * 200)
    w = YieldCurveRotation().generate_weights(prices)
    assert set(w.columns) == {"SPY", "TLT"}


def test_weights_non_negative():
    rng = np.random.default_rng(8)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.002, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
    )
    w = YieldCurveRotation().generate_weights(prices)
    assert (w >= 0).all().all()
