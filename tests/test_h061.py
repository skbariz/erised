"""Focused tests for H061: ValueGrowthRotation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.engine import assert_no_lookahead_multi
from quant.strategies.base import ValueGrowthRotation


def _prices(spy_rets, shy_rets=None, tlt_rets=None, ive_rets=None, ivw_rets=None, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    for col, rets in [("shy_close", shy_rets), ("tlt_close", tlt_rets),
                      ("ive_close", ive_rets), ("ivw_close", ivw_rets)]:
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
        rng.normal(0.0004, 0.009, n).tolist(),
        rng.normal(0.0005, 0.011, n).tolist(),
    )
    assert_no_lookahead_multi(ValueGrowthRotation().generate_weights, prices)


def test_long_ivw_when_curve_normal_and_spy_up():
    """Normal curve + rising SPY → long IVW (growth)."""
    n = 400
    prices = _prices([0.001] * n, [0.001] * n, [0.004] * n, [0.001] * n, [0.001] * n)
    w = ValueGrowthRotation(mom_days=63, spy_ma=200).generate_weights(prices)
    assert w["IVW"].iloc[-1] == 1.0
    assert w["SPY"].iloc[-1] == 0.0
    assert w["IVE"].iloc[-1] == 0.0


def test_long_ive_when_curve_inverted():
    """Inverted curve → long IVE (value/defensive)."""
    n = 400
    prices = _prices([0.001] * n, [0.004] * n, [0.001] * n, [0.001] * n, [0.001] * n)
    w = ValueGrowthRotation(mom_days=63, spy_ma=200).generate_weights(prices)
    assert w["IVE"].iloc[-1] == 1.0
    assert w["SPY"].iloc[-1] == 0.0
    assert w["IVW"].iloc[-1] == 0.0


def test_long_spy_without_curve_data():
    """Without SHY/TLT data → default to SPY."""
    prices = _prices([0.001] * 400, ive_rets=[0.001] * 400, ivw_rets=[0.001] * 400)
    w = ValueGrowthRotation().generate_weights(prices)
    assert (w["SPY"] == 1.0).all()


def test_long_spy_when_curve_normal_but_spy_down():
    """Normal curve but SPY in downtrend → stay in SPY (not growth)."""
    n = 400
    prices = _prices([-0.001] * n, [0.001] * n, [0.004] * n, [0.001] * n, [0.001] * n)
    w = ValueGrowthRotation(mom_days=63, spy_ma=200).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0
    assert w["IVW"].iloc[-1] == 0.0


def test_columns_present():
    prices = _prices([0.001] * 300, [0.001] * 300, [0.001] * 300, [0.001] * 300, [0.001] * 300)
    w = ValueGrowthRotation().generate_weights(prices)
    assert set(w.columns) == {"SPY", "IVW", "IVE"}


def test_weights_sum_to_one():
    rng = np.random.default_rng(1)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.002, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
        rng.normal(0, 0.009, n).tolist(),
        rng.normal(0, 0.011, n).tolist(),
    )
    w = ValueGrowthRotation().generate_weights(prices)
    assert (w.sum(axis=1).round(9) == 1.0).all()
