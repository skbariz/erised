"""Focused tests for H060: CarryMomentumSPYTLT."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.engine import assert_no_lookahead_multi
from quant.strategies.base import CarryMomentumSPYTLT


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
    n = 700
    prices = _prices(
        rng.normal(0.0005, 0.01, n).tolist(),
        rng.normal(0.0002, 0.002, n).tolist(),
        rng.normal(0.0003, 0.005, n).tolist(),
    )
    assert_no_lookahead_multi(CarryMomentumSPYTLT().generate_weights, prices)


def test_long_spy_when_curve_steep_and_trend_up():
    """Steep curve AND SPY above MA → 100% SPY."""
    n = 400
    # SPY rising → above any MA; TLT outperforming SHY
    prices = _prices([0.001] * n, [0.001] * n, [0.004] * n)
    w = CarryMomentumSPYTLT(mom_days=63, spy_ma=200).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0
    assert w["TLT"].iloc[-1] == 0.0


def test_long_tlt_when_curve_inverted():
    """Inverted curve → 100% TLT regardless of SPY trend."""
    n = 400
    prices = _prices([0.001] * n, [0.004] * n, [0.001] * n)
    w = CarryMomentumSPYTLT(mom_days=63, spy_ma=200).generate_weights(prices)
    assert w["TLT"].iloc[-1] == 1.0
    assert w["SPY"].iloc[-1] == 0.0


def test_flat_when_curve_steep_but_spy_below_ma():
    """Steep curve but SPY in downtrend → flat (no equity, no bonds)."""
    n = 400
    # SPY declining → below MA; normal curve
    prices = _prices([-0.001] * n, [0.001] * n, [0.004] * n)
    w = CarryMomentumSPYTLT(mom_days=63, spy_ma=200).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 0.0
    assert w["TLT"].iloc[-1] == 0.0


def test_flat_before_data():
    """Without SHY/TLT columns: flat."""
    prices = _prices([0.001] * 400)
    w = CarryMomentumSPYTLT().generate_weights(prices)
    assert (w.sum(axis=1) == 0.0).all()


def test_columns_present():
    prices = _prices([0.001] * 200, [0.001] * 200, [0.001] * 200)
    w = CarryMomentumSPYTLT().generate_weights(prices)
    assert set(w.columns) == {"SPY", "TLT"}


def test_weights_non_negative():
    rng = np.random.default_rng(13)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.002, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
    )
    w = CarryMomentumSPYTLT().generate_weights(prices)
    assert (w >= 0).all().all()


def test_weights_at_most_one():
    rng = np.random.default_rng(14)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.002, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
    )
    w = CarryMomentumSPYTLT().generate_weights(prices)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()
