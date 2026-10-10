"""Focused tests for H065: EqualWeightDefensiveFactor."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.backtest.engine import assert_no_lookahead_multi
from erised.strategies.base import EqualWeightDefensiveFactor


def _prices(spy_rets, xlp_rets=None, xlu_rets=None, ive_rets=None, start="2005-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    for col, rets in [("xlp_close", xlp_rets), ("xlu_close", xlu_rets), ("ive_close", ive_rets)]:
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
        rng.normal(0.0003, 0.005, n).tolist(),
        rng.normal(0.0003, 0.006, n).tolist(),
        rng.normal(0.0004, 0.009, n).tolist(),
    )
    assert_no_lookahead_multi(EqualWeightDefensiveFactor().generate_weights, prices)


def test_equal_weight_all_four():
    """All 4 assets available → 25% each."""
    n = 300
    prices = _prices([0.001] * n, [0.001] * n, [0.001] * n, [0.001] * n)
    w = EqualWeightDefensiveFactor().generate_weights(prices)
    assert abs(w["SPY"].iloc[-1] - 0.25) < 1e-9
    assert abs(w["XLP"].iloc[-1] - 0.25) < 1e-9
    assert abs(w["XLU"].iloc[-1] - 0.25) < 1e-9
    assert abs(w["IVE"].iloc[-1] - 0.25) < 1e-9


def test_renormalizes_without_ive():
    """Without IVE column → 3 assets at 33.3% each; IVE not in output."""
    n = 300
    prices = _prices([0.001] * n, [0.001] * n, [0.001] * n)
    w = EqualWeightDefensiveFactor().generate_weights(prices)
    assert set(w.columns) == {"SPY", "XLP", "XLU"}
    assert abs(w["SPY"].iloc[-1] - 1 / 3) < 1e-9
    assert abs(w["XLP"].iloc[-1] - 1 / 3) < 1e-9
    assert abs(w["XLU"].iloc[-1] - 1 / 3) < 1e-9


def test_spy_only_without_factors():
    """Without any factor columns → 100% SPY."""
    prices = _prices([0.001] * 300)
    w = EqualWeightDefensiveFactor().generate_weights(prices)
    assert (w["SPY"] == 1.0).all()


def test_weights_always_sum_to_one():
    rng = np.random.default_rng(5)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
        rng.normal(0, 0.006, n).tolist(),
        rng.normal(0, 0.009, n).tolist(),
    )
    w = EqualWeightDefensiveFactor().generate_weights(prices)
    assert (w.sum(axis=1).round(9) == 1.0).all()


def test_columns_present():
    prices = _prices([0.001] * 300, [0.001] * 300, [0.001] * 300, [0.001] * 300)
    w = EqualWeightDefensiveFactor().generate_weights(prices)
    assert set(w.columns) == {"SPY", "XLP", "XLU", "IVE"}


def test_non_negative():
    rng = np.random.default_rng(6)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
        rng.normal(0, 0.006, n).tolist(),
        rng.normal(0, 0.009, n).tolist(),
    )
    w = EqualWeightDefensiveFactor().generate_weights(prices)
    assert (w >= 0).all().all()
