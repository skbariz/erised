"""Focused tests for H023: SellInMay."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import SellInMay


def _prices(rets, start="2010-01-04"):
    n = len(rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in rets:
        px.append(px[-1] * (1 + r))
    return pd.DataFrame({"adj_close": px}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    prices = _prices(rng.normal(0.0005, 0.01, 700).tolist())
    assert_no_lookahead(SellInMay().generate_positions, prices)


def test_favorable_months_get_position_one():
    """Nov–Apr days should have position 1.0."""
    prices = _prices([0.001] * 600)
    pos = SellInMay().generate_positions(prices)
    favorable_mask = prices.index.month.isin([11, 12, 1, 2, 3, 4])
    assert (pos[favorable_mask] == 1.0).all()


def test_unfavorable_months_get_position_zero():
    """May–Oct days should have position 0.0."""
    prices = _prices([0.001] * 600)
    pos = SellInMay().generate_positions(prices)
    unfavorable_mask = prices.index.month.isin([5, 6, 7, 8, 9, 10])
    assert (pos[unfavorable_mask] == 0.0).all()


def test_position_is_binary():
    """Positions must be 0 or 1 only."""
    prices = _prices([0.001] * 600)
    pos = SellInMay().generate_positions(prices)
    unique_vals = set(pos.values.round(6))
    assert unique_vals.issubset({0.0, 1.0})


def test_custom_favorable_months():
    """Oct-Apr window should include October and exclude May."""
    prices = _prices([0.001] * 800)
    pos = SellInMay(favorable_months=(10, 11, 12, 1, 2, 3, 4)).generate_positions(prices)
    oct_mask = prices.index.month == 10
    may_mask = prices.index.month == 5
    assert (pos[oct_mask] == 1.0).all()
    assert (pos[may_mask] == 0.0).all()


def test_returns_series():
    prices = _prices([0.001] * 400)
    pos = SellInMay().generate_positions(prices)
    assert isinstance(pos, pd.Series)
    assert len(pos) == len(prices)


def test_index_alignment():
    """Position index should match price index."""
    prices = _prices([0.001] * 400)
    pos = SellInMay().generate_positions(prices)
    assert pos.index.equals(prices.index)
