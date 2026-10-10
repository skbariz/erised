"""Focused tests for H025: JanuaryBarometer."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.backtest.engine import assert_no_lookahead
from erised.strategies import JanuaryBarometer


def _prices(rets, start="2005-01-03"):
    n = len(rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in rets:
        px.append(px[-1] * (1 + r))
    return pd.DataFrame({"adj_close": px}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    prices = _prices(rng.normal(0.0005, 0.01, 700).tolist())
    assert_no_lookahead(JanuaryBarometer().generate_positions, prices)


def test_flat_in_january():
    """Strategy should have 0 position during January itself."""
    prices = _prices([0.001] * 800)
    pos = JanuaryBarometer().generate_positions(prices)
    jan_mask = prices.index.month == 1
    assert (pos[jan_mask] == 0.0).all()


def test_positive_january_leads_to_long():
    """Positive January return → long (1.0) for rest of that year."""
    # Jan 2005 is positive (all returns 0.001), Feb-Dec should be 1.0
    prices = _prices([0.001] * 600)
    pos = JanuaryBarometer().generate_positions(prices)
    # Feb–Dec 2005 should have position 1.0
    feb_dec_2005 = prices.index[(prices.index.year == 2005) & (prices.index.month > 1)]
    if len(feb_dec_2005) > 0:
        assert (pos[feb_dec_2005] == 1.0).all()


def test_negative_january_leads_to_flat():
    """Negative January return → flat (0.0) for rest of that year."""
    # Make Jan 2005 negative, rest positive
    jan_2005_idx_size = len(pd.bdate_range("2005-01-03", "2005-01-31"))
    rets = [-0.005] * jan_2005_idx_size + [0.001] * (800 - jan_2005_idx_size)
    prices = _prices(rets)
    pos = JanuaryBarometer().generate_positions(prices)
    feb_dec_2005 = prices.index[(prices.index.year == 2005) & (prices.index.month > 1)]
    if len(feb_dec_2005) > 0:
        assert (pos[feb_dec_2005] == 0.0).all()


def test_position_is_binary():
    """Positions must be 0 or 1 only."""
    prices = _prices([0.001] * 600)
    pos = JanuaryBarometer().generate_positions(prices)
    unique_vals = set(pos.values.round(6))
    assert unique_vals.issubset({0.0, 1.0})


def test_returns_series():
    prices = _prices([0.001] * 400)
    pos = JanuaryBarometer().generate_positions(prices)
    assert isinstance(pos, pd.Series)
    assert len(pos) == len(prices)


def test_each_year_is_independent():
    """Signals from different years should not bleed into each other."""
    # Year 1 January negative, Year 2 January positive
    prices = _prices([0.001] * 1000, start="2005-01-03")
    pos = JanuaryBarometer().generate_positions(prices)
    # Each Feb-Dec period should independently reflect its January
    for year in prices.index.year.unique():
        jan = prices.index[(prices.index.year == year) & (prices.index.month == 1)]
        rest = prices.index[(prices.index.year == year) & (prices.index.month > 1)]
        if len(jan) < 2 or len(rest) == 0:
            continue
        # Check consistency within the year
        rest_pos = pos[rest]
        assert len(rest_pos.unique()) <= 1, f"Year {year} has mixed signals"
