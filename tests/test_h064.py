"""Focused tests for H064: TechDefensiveMomentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.engine import assert_no_lookahead_multi
from quant.strategies.base import TechDefensiveMomentum


def _prices(spy_rets, qqq_rets=None, xlu_rets=None, start="2005-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    for col, rets in [("qqq_close", qqq_rets), ("xlu_close", xlu_rets)]:
        if rets is not None:
            px = [100.0]
            for r in rets:
                px.append(px[-1] * (1 + r))
            df[col] = px
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    n = 800
    prices = _prices(
        rng.normal(0.0005, 0.01, n).tolist(),
        rng.normal(0.0007, 0.015, n).tolist(),
        rng.normal(0.0003, 0.007, n).tolist(),
    )
    assert_no_lookahead_multi(TechDefensiveMomentum().generate_weights, prices)


def test_long_qqq_when_tech_leads():
    """QQQ with stronger momentum → long QQQ."""
    n = 600
    prices = _prices([0.001] * n, [0.003] * n, [0.001] * n)
    w = TechDefensiveMomentum(long_days=252, skip_days=21).generate_weights(prices)
    assert w["QQQ"].iloc[-1] == 1.0
    assert w["XLU"].iloc[-1] == 0.0
    assert w["SPY"].iloc[-1] == 0.0


def test_long_xlu_when_defensive_leads():
    """XLU with stronger momentum → long XLU."""
    n = 600
    prices = _prices([0.001] * n, [0.0005] * n, [0.003] * n)
    w = TechDefensiveMomentum(long_days=252, skip_days=21).generate_weights(prices)
    assert w["XLU"].iloc[-1] == 1.0
    assert w["QQQ"].iloc[-1] == 0.0


def test_long_spy_when_both_negative():
    """Both QQQ and XLU have negative momentum AND SPY positive → SPY."""
    n = 600
    prices = _prices([0.001] * n, [-0.001] * n, [-0.001] * n)
    w = TechDefensiveMomentum(long_days=252, skip_days=21).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0
    assert w["QQQ"].iloc[-1] == 0.0
    assert w["XLU"].iloc[-1] == 0.0


def test_flat_when_all_negative():
    """All three assets with negative momentum → flat."""
    n = 600
    prices = _prices([-0.001] * n, [-0.001] * n, [-0.001] * n)
    w = TechDefensiveMomentum(long_days=252, skip_days=21).generate_weights(prices)
    assert w.sum(axis=1).iloc[-1] == 0.0


def test_columns_present():
    prices = _prices([0.001] * 500, [0.001] * 500, [0.001] * 500)
    w = TechDefensiveMomentum().generate_weights(prices)
    assert set(w.columns) == {"SPY", "QQQ", "XLU"}


def test_weights_at_most_one():
    rng = np.random.default_rng(4)
    n = 600
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.015, n).tolist(),
        rng.normal(0, 0.007, n).tolist(),
    )
    w = TechDefensiveMomentum().generate_weights(prices)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()
    assert (w >= 0).all().all()
