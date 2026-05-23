"""Focused tests for H036: EquityVsSafeHavenMomentum."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import EquityVsSafeHavenMomentum


def _prices(spy_rets, tlt_rets, gld_rets, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy_px = [100.0]
    for r in spy_rets:
        spy_px.append(spy_px[-1] * (1 + r))
    tlt_px = [100.0]
    for r in tlt_rets:
        tlt_px.append(tlt_px[-1] * (1 + r))
    gld_px = [100.0]
    for r in gld_rets:
        gld_px.append(gld_px[-1] * (1 + r))
    return pd.DataFrame(
        {"adj_close": spy_px, "tlt_close": tlt_px, "gld_close": gld_px}, index=idx
    )


def test_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    tlt = rng.normal(0.0002, 0.005, 700).tolist()
    gld = rng.normal(0.0003, 0.007, 700).tolist()
    prices = _prices(spy, tlt, gld)
    assert_no_lookahead(EquityVsSafeHavenMomentum().generate_positions, prices)


def test_long_when_spy_outperforms():
    """SPY return >> TLT and GLD -> position 1.0."""
    n = 200
    # SPY strong uptrend, safe havens flat
    spy = [0.005] * n
    tlt = [0.0] * n
    gld = [0.0] * n
    prices = _prices(spy, tlt, gld)
    pos = EquityVsSafeHavenMomentum(lookback_days=20).generate_positions(prices)
    assert pos.iloc[30] == pytest.approx(1.0)


def test_flat_when_safe_havens_outperform():
    """Safe havens >> SPY -> position 0.0."""
    n = 200
    spy = [0.0] * n
    tlt = [0.005] * n  # bonds rallying
    gld = [0.005] * n  # gold rallying
    prices = _prices(spy, tlt, gld)
    pos = EquityVsSafeHavenMomentum(lookback_days=20).generate_positions(prices)
    assert pos.iloc[30] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(13)
    spy = rng.normal(0, 0.01, 400).tolist()
    tlt = rng.normal(0, 0.005, 400).tolist()
    gld = rng.normal(0, 0.007, 400).tolist()
    prices = _prices(spy, tlt, gld)
    pos = EquityVsSafeHavenMomentum(lookback_days=21).generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_zero_during_warmup():
    """Before lookback_days warmup, position should be 0."""
    n = 100
    prices = _prices([0.001] * n, [0.0] * n, [0.0] * n)
    pos = EquityVsSafeHavenMomentum(lookback_days=30).generate_positions(prices)
    assert (pos.iloc[:30] == 0.0).all()


def test_requires_both_columns():
    prices_no_tlt = pd.DataFrame(
        {"adj_close": [100.0, 101.0], "gld_close": [150.0, 151.0]},
        index=pd.bdate_range("2010-01-04", periods=2),
    )
    with pytest.raises(ValueError, match="tlt_close"):
        EquityVsSafeHavenMomentum().generate_positions(prices_no_tlt)

    prices_no_gld = pd.DataFrame(
        {"adj_close": [100.0, 101.0], "tlt_close": [90.0, 91.0]},
        index=pd.bdate_range("2010-01-04", periods=2),
    )
    with pytest.raises(ValueError, match="gld_close"):
        EquityVsSafeHavenMomentum().generate_positions(prices_no_gld)
