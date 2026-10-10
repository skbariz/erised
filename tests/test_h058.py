"""Focused tests for H058: InvertedCurveTreasuryLadder."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.backtest.engine import assert_no_lookahead_multi
from erised.strategies.base import InvertedCurveTreasuryLadder


def _prices(spy_rets, shy_rets=None, tlt_rets=None, ief_rets=None, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    for col, rets in [("shy_close", shy_rets), ("tlt_close", tlt_rets), ("ief_close", ief_rets)]:
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
        rng.normal(0.00025, 0.004, n).tolist(),
    )
    assert_no_lookahead_multi(InvertedCurveTreasuryLadder().generate_weights, prices)


def test_long_spy_when_curve_normal():
    """Normal curve throughout → all SPY."""
    n = 300
    prices = _prices([0.001] * n, [0.001] * n, [0.004] * n, [0.002] * n)
    w = InvertedCurveTreasuryLadder(mom_days=63, sustain_days=21).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0
    assert w["TLT"].iloc[-1] == 0.0
    assert w["IEF"].iloc[-1] == 0.0


def test_ladder_after_sustained_inversion():
    """Sustained inversion (>21d) → 75% TLT + 25% IEF."""
    n = 400
    # Inversion: SHY (0.004) > TLT (0.001) throughout
    prices = _prices([0.001] * n, [0.004] * n, [0.001] * n, [0.002] * n)
    w = InvertedCurveTreasuryLadder(mom_days=63, sustain_days=21).generate_weights(prices)
    assert abs(w["TLT"].iloc[-1] - 0.75) < 1e-9
    assert abs(w["IEF"].iloc[-1] - 0.25) < 1e-9
    assert w["SPY"].iloc[-1] == 0.0


def test_still_spy_during_brief_inversion():
    """Brief inversion (fewer than sustain_days) → still SPY."""
    n = 200
    # Normal curve for first 100 rows, then brief inversion for 10 rows
    shy_rets = [0.001] * n
    tlt_rets = [0.004] * 100 + [0.001] * 100  # TLT decelerates
    prices = _prices([0.001] * n, shy_rets, tlt_rets, [0.002] * n)
    w = InvertedCurveTreasuryLadder(mom_days=10, sustain_days=50).generate_weights(prices)
    # inversion not sustained → SPY
    assert w["SPY"].iloc[-1] == 1.0


def test_columns_present():
    prices = _prices([0.001] * 200, [0.001] * 200, [0.001] * 200, [0.001] * 200)
    w = InvertedCurveTreasuryLadder().generate_weights(prices)
    assert set(w.columns) == {"SPY", "TLT", "IEF"}


def test_weights_sum_to_at_most_one():
    rng = np.random.default_rng(4)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.002, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
        rng.normal(0, 0.004, n).tolist(),
    )
    w = InvertedCurveTreasuryLadder().generate_weights(prices)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_weights_non_negative():
    rng = np.random.default_rng(5)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.002, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
        rng.normal(0, 0.004, n).tolist(),
    )
    w = InvertedCurveTreasuryLadder().generate_weights(prices)
    assert (w >= 0).all().all()
