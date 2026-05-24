"""Focused tests for H057: BondCreditDualConfirm."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import BondCreditDualConfirm


def _prices(spy_rets, tlt_rets=None, hyg_rets=None, lqd_rets=None, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    for col, rets in [("tlt_close", tlt_rets), ("hyg_close", hyg_rets), ("lqd_close", lqd_rets)]:
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
        rng.normal(0.0002, 0.005, n).tolist(),
        rng.normal(0.0004, 0.006, n).tolist(),
        rng.normal(0.0003, 0.004, n).tolist(),
    )
    assert_no_lookahead(BondCreditDualConfirm().generate_positions, prices)


def test_long_when_both_signals_ok():
    """TLT falling (below MA) AND HYG outperforms LQD → long SPY."""
    n = 400
    prices = _prices(
        [0.001] * n,
        [-0.001] * n,   # TLT falling → below MA → risk-on signal
        [0.003] * n,    # HYG rising faster
        [0.001] * n,    # LQD rising slower
    )
    pos = BondCreditDualConfirm(tlt_ma=200, credit_days=63).generate_positions(prices)
    assert pos.iloc[-1] == 1.0


def test_flat_when_tlt_above_ma():
    """TLT above MA → risk-off per bond trend → flat regardless of credit."""
    n = 400
    prices = _prices(
        [0.001] * n,
        [0.003] * n,    # TLT rising → above MA
        [0.003] * n,    # HYG fine
        [0.001] * n,
    )
    pos = BondCreditDualConfirm(tlt_ma=200, credit_days=63).generate_positions(prices)
    assert pos.iloc[-1] == 0.0


def test_flat_when_credit_stressed():
    """TLT below MA but LQD outperforms HYG → credit stress → flat."""
    n = 400
    prices = _prices(
        [0.001] * n,
        [-0.001] * n,   # TLT falling → risk-on by bond signal
        [0.001] * n,    # HYG slower
        [0.003] * n,    # LQD faster → credit stress
    )
    pos = BondCreditDualConfirm(tlt_ma=200, credit_days=63).generate_positions(prices)
    assert pos.iloc[-1] == 0.0


def test_fallback_to_tlt_only_without_credit_data():
    """Without HYG/LQD, falls back to TLT-only signal (H033 behavior)."""
    n = 400
    prices = _prices([0.001] * n, [-0.001] * n)  # TLT falling → risk-on
    pos = BondCreditDualConfirm(tlt_ma=200).generate_positions(prices)
    assert pos.iloc[-1] == 1.0


def test_positions_binary():
    rng = np.random.default_rng(10)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
        rng.normal(0, 0.006, n).tolist(),
        rng.normal(0, 0.004, n).tolist(),
    )
    pos = BondCreditDualConfirm().generate_positions(prices)
    assert set(pos.round(9).unique()).issubset({0.0, 1.0})
