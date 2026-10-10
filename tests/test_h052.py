"""Focused tests for H052: CommodityMomentumOverlay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.backtest.engine import assert_no_lookahead
from erised.strategies.base import CommodityMomentumOverlay


def _prices(spy_rets, dbc_rets=None, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    if dbc_rets is not None:
        dbc = [100.0]
        for r in dbc_rets:
            dbc.append(dbc[-1] * (1 + r))
        df["dbc_close"] = dbc
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    n = 700
    prices = _prices(
        rng.normal(0.0005, 0.01, n).tolist(),
        rng.normal(0.0003, 0.012, n).tolist(),
    )
    assert_no_lookahead(CommodityMomentumOverlay().generate_positions, prices)


def test_long_when_dbc_below_ma():
    """Steady rising SPY, DBC trending down → always long."""
    n = 300
    prices = _prices([0.001] * n, [-0.002] * n)  # DBC falling, below any MA
    pos = CommodityMomentumOverlay(ma_period=50).generate_positions(prices)
    # After warmup, should be long
    assert pos.iloc[-1] == 1.0


def test_flat_when_dbc_above_ma():
    """DBC steadily rising → above MA → flat."""
    n = 300
    prices = _prices([0.001] * n, [0.003] * n)  # DBC rising sharply
    pos = CommodityMomentumOverlay(ma_period=50).generate_positions(prices)
    # After warmup, DBC above MA → flat
    assert pos.iloc[-1] == 0.0


def test_defaults_long_without_dbc_column():
    """Without dbc_close, strategy defaults to long SPY (conservative fallback)."""
    prices = _prices([0.001] * 200)
    pos = CommodityMomentumOverlay().generate_positions(prices)
    assert (pos == 1.0).all()


def test_flat_before_warmup():
    """Before MA is computable, position must be 0."""
    n = 300
    prices = _prices([0.001] * n, [0.001] * n)
    pos = CommodityMomentumOverlay(ma_period=200).generate_positions(prices)
    # First 200 bars: no MA available → flat
    assert (pos.iloc[:200] == 0.0).all()


def test_positions_binary():
    """All positions must be 0.0 or 1.0."""
    rng = np.random.default_rng(7)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.015, n).tolist(),
    )
    pos = CommodityMomentumOverlay().generate_positions(prices)
    assert set(pos.round(9).unique()).issubset({0.0, 1.0})
