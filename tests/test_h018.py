"""Focused tests for H018: SPYTLTDualMomentum."""

from __future__ import annotations

import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead_multi
from erised.strategies.base import SPYTLTDualMomentum


def _prices(spy_rets, tlt_rets, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    tlt = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    for r in tlt_rets:
        tlt.append(tlt[-1] * (1 + r))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_no_lookahead():
    prices = _prices([0.001] * 700, [-0.0005] * 700)
    assert_no_lookahead_multi(SPYTLTDualMomentum().generate_weights, prices)


def test_hold_spy_when_spy_positive():
    prices = _prices([0.002] * 300, [-0.001] * 300)
    w = SPYTLTDualMomentum().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(1.0)
    assert w["TLT"].iloc[-1] == pytest.approx(0.0)


def test_hold_tlt_when_spy_negative_tlt_positive():
    prices = _prices([-0.001] * 300, [0.002] * 300)
    w = SPYTLTDualMomentum().generate_weights(prices)
    assert w["TLT"].iloc[-1] == pytest.approx(1.0)
    assert w["SPY"].iloc[-1] == pytest.approx(0.0)


def test_cash_when_both_negative():
    prices = _prices([-0.001] * 300, [-0.001] * 300)
    w = SPYTLTDualMomentum().generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(0.0)
    assert w["TLT"].iloc[-1] == pytest.approx(0.0)


def test_weights_binary():
    """Weights must be 0 or 1 only — no fractional allocations."""
    prices = _prices([0.001] * 300, [-0.001] * 300)
    w = SPYTLTDualMomentum().generate_weights(prices)
    unique_vals = set(w.values.ravel().round(6))
    assert unique_vals.issubset({0.0, 1.0})


def test_no_simultaneous_spy_tlt():
    """SPY and TLT must never both be non-zero at the same time."""
    prices = _prices(
        [0.001 * ((-1) ** i) for i in range(300)], [0.001 * ((-1) ** (i + 1)) for i in range(300)]
    )
    w = SPYTLTDualMomentum().generate_weights(prices)
    both_active = (w["SPY"] > 0) & (w["TLT"] > 0)
    assert not both_active.any()


def test_weight_sum_never_exceeds_one():
    """Total allocation must never exceed 1.0."""
    prices = _prices([0.001] * 300, [0.002] * 300)
    w = SPYTLTDualMomentum().generate_weights(prices)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_lookback_sensitivity():
    """Shorter lookback should produce non-identical but valid weights."""
    prices = _prices([0.001] * 300, [-0.001] * 300)
    w12 = SPYTLTDualMomentum(lookback_months=12).generate_weights(prices)
    w6 = SPYTLTDualMomentum(lookback_months=6).generate_weights(prices)
    # Both valid but may differ at some dates
    assert set(w12.values.ravel().round(6)).issubset({0.0, 1.0})
    assert set(w6.values.ravel().round(6)).issubset({0.0, 1.0})
