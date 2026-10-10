"""Focused tests for H022: DefensiveEquityHedge."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead_multi
from erised.strategies.base import DefensiveEquityHedge


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
    rng = np.random.default_rng(42)
    n = 700
    spy_rets = rng.normal(0.0005, 0.01, n).tolist()
    tlt_rets = rng.normal(0.0002, 0.005, n).tolist()
    prices = _prices(spy_rets, tlt_rets)
    assert_no_lookahead_multi(DefensiveEquityHedge().generate_weights, prices)


def test_uptrend_fully_in_spy():
    """Steady uptrend: fast MA > slow MA → 100% SPY, 0% TLT."""
    prices = _prices([0.002] * 500, [0.001] * 500)
    w = DefensiveEquityHedge(fast=50, slow=200).generate_weights(prices)
    # After warmup, in sustained uptrend, SPY weight should be 1.0
    assert w["SPY"].iloc[-1] == pytest.approx(1.0)
    assert w["TLT"].iloc[-1] == pytest.approx(0.0)


def test_downtrend_splits_50_50():
    """Sustained downtrend: fast MA < slow MA → 50% SPY, 50% TLT."""
    # Start with a rally to build warm MA values, then decline
    rally = [0.005] * 250
    decline = [-0.003] * 450
    prices = _prices(rally + decline, [0.001] * 700)
    w = DefensiveEquityHedge(fast=50, slow=200).generate_weights(prices)
    assert w["SPY"].iloc[-1] == pytest.approx(0.5)
    assert w["TLT"].iloc[-1] == pytest.approx(0.5)


def test_weights_sum_to_one_after_warmup():
    """After warmup, SPY + TLT must always sum to 1.0."""
    prices = _prices([0.001] * 400, [0.0005] * 400)
    w = DefensiveEquityHedge(fast=50, slow=200).generate_weights(prices)
    valid = (w["SPY"] + w["TLT"]) > 0
    sums = (w["SPY"] + w["TLT"])[valid]
    assert np.allclose(sums.values, 1.0)


def test_weights_non_negative():
    """All weights must be ≥ 0 (long-only)."""
    rng = np.random.default_rng(7)
    spy_rets = rng.normal(0, 0.01, 400).tolist()
    tlt_rets = rng.normal(0, 0.005, 400).tolist()
    prices = _prices(spy_rets, tlt_rets)
    w = DefensiveEquityHedge().generate_weights(prices)
    assert (w >= 0).all().all()


def test_columns():
    prices = _prices([0.001] * 300, [0.001] * 300)
    w = DefensiveEquityHedge().generate_weights(prices)
    assert list(w.columns) == ["SPY", "TLT"]


def test_ma_params_accepted():
    """Different MA parameter combinations should produce valid weights."""
    prices = _prices([0.001] * 400, [0.0005] * 400)
    for fast, slow in [(20, 100), (50, 200), (50, 150)]:
        w = DefensiveEquityHedge(fast=fast, slow=slow).generate_weights(prices)
        valid = (w["SPY"] + w["TLT"]) > 0
        if valid.any():
            assert np.allclose((w["SPY"] + w["TLT"])[valid].values, 1.0)


def test_warmup_period_is_zero():
    """Before slow MA is ready, all weights should be 0."""
    prices = _prices([0.001] * 300, [0.001] * 300)
    w = DefensiveEquityHedge(fast=50, slow=200).generate_weights(prices)
    # rolling(200) first valid at index 199; rows 0-198 are pre-warmup zeros
    pre_warmup = w.iloc[:199]
    assert (pre_warmup == 0.0).all().all()
