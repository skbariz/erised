"""Focused tests for H059: MacroFearComposite."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.backtest.engine import assert_no_lookahead
from erised.strategies import MacroFearComposite


def _prices(
    spy_rets,
    vix_vals=None,
    gld_rets=None,
    hyg_rets=None,
    lqd_rets=None,
    shy_rets=None,
    tlt_rets=None,
    start="2010-01-04",
):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    if vix_vals is not None:
        df["vix_close"] = ([vix_vals[0]] + vix_vals)[: n + 1]
    for col, rets in [
        ("gld_close", gld_rets),
        ("hyg_close", hyg_rets),
        ("lqd_close", lqd_rets),
        ("shy_close", shy_rets),
        ("tlt_close", tlt_rets),
    ]:
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
        vix_vals=[20.0 + rng.normal(0, 5) for _ in range(n)],
        gld_rets=rng.normal(0.0003, 0.008, n).tolist(),
        hyg_rets=rng.normal(0.0004, 0.006, n).tolist(),
        lqd_rets=rng.normal(0.0003, 0.004, n).tolist(),
        shy_rets=rng.normal(0.0002, 0.002, n).tolist(),
        tlt_rets=rng.normal(0.0003, 0.005, n).tolist(),
    )
    assert_no_lookahead(MacroFearComposite().generate_positions, prices)


def test_fully_long_when_no_fear():
    """Zero fear signals → position = 1.0."""
    n = 200
    prices = _prices(
        [0.001] * n,
        vix_vals=[15.0] * n,  # VIX below 25
        gld_rets=[-0.001] * n,  # GLD falling (below MA)
        hyg_rets=[0.003] * n,  # HYG outperforming
        lqd_rets=[0.001] * n,
        shy_rets=[0.001] * n,
        tlt_rets=[0.004] * n,  # TLT outperforming → curve normal
    )
    pos = MacroFearComposite(vix_thresh=25, gld_ma=50, mom_days=63).generate_positions(prices)
    assert pos.iloc[-1] == 1.0


def test_flat_when_multiple_fear_signals():
    """2+ fear signals → position = 0.0."""
    n = 200
    prices = _prices(
        [0.001] * n,
        vix_vals=[35.0] * n,  # VIX above 25 (fear)
        gld_rets=[0.003] * n,  # GLD above MA (fear)
        hyg_rets=[0.001] * n,  # HYG lagging LQD (fear)
        lqd_rets=[0.003] * n,
        shy_rets=[0.001] * n,
        tlt_rets=[0.004] * n,  # curve normal (no fear here)
    )
    pos = MacroFearComposite(vix_thresh=25, gld_ma=50, mom_days=63).generate_positions(prices)
    assert pos.iloc[-1] == 0.0


def test_positions_in_range():
    """All positions must be in [0, 1]."""
    rng = np.random.default_rng(12)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        vix_vals=[15 + rng.uniform(0, 30) for _ in range(n)],
        gld_rets=rng.normal(0, 0.008, n).tolist(),
        hyg_rets=rng.normal(0, 0.006, n).tolist(),
        lqd_rets=rng.normal(0, 0.004, n).tolist(),
        shy_rets=rng.normal(0, 0.002, n).tolist(),
        tlt_rets=rng.normal(0, 0.005, n).tolist(),
    )
    pos = MacroFearComposite().generate_positions(prices)
    assert (pos >= 0.0).all()
    assert (pos <= 1.0).all()


def test_missing_components_handled():
    """Strategy must not crash with only some columns present."""
    n = 200
    prices = _prices([0.001] * n, vix_vals=[20.0] * n)
    pos = MacroFearComposite().generate_positions(prices)
    assert len(pos) == n + 1
    assert (pos >= 0).all()
