"""Focused tests for H045: TurnOfMonthLowVol."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import TurnOfMonthLowVol


def _prices(spy_rets, vix_levels, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in spy_rets:
        px.append(px[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": px}, index=idx)
    assert len(vix_levels) == n + 1
    df["vix_close"] = vix_levels
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 600).tolist()
    vix = (18 + rng.uniform(0, 6, 601)).tolist()
    prices = _prices(spy, vix)
    assert_no_lookahead(TurnOfMonthLowVol().generate_positions, prices)


def test_long_at_turn_when_vix_low():
    """At month-end/start with VIX below threshold -> position 1.0."""
    spy = [0.001] * 400
    vix = [15.0] * 401  # always calm
    prices = _prices(spy, vix, start="2010-01-29")  # starts near month end
    pos = TurnOfMonthLowVol(n_days=3, vix_threshold=25).generate_positions(prices)
    # First bar is Jan 29 (last few days of Jan) — should be in TOM window
    assert pos.iloc[1] == pytest.approx(1.0)


def test_flat_at_turn_when_vix_high():
    """At month-end/start with VIX above threshold -> position 0.0."""
    spy = [0.001] * 400
    vix = [40.0] * 401  # always elevated
    prices = _prices(spy, vix, start="2010-01-29")
    pos = TurnOfMonthLowVol(n_days=3, vix_threshold=25).generate_positions(prices)
    assert pos.iloc[1] == pytest.approx(0.0)


def test_flat_in_mid_month():
    """Mid-month days should never be long regardless of VIX."""
    spy = [0.001] * 600
    vix = [10.0] * 601  # VIX very low, would allow all turns
    prices = _prices(spy, vix, start="2010-01-15")  # start mid-month
    pos = TurnOfMonthLowVol(n_days=3, vix_threshold=25).generate_positions(prices)
    # First few bars are mid-January (rank_start ≥ 10, rank_end ≥ 10) — should be flat
    assert pos.iloc[0] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(7)
    spy = rng.normal(0, 0.01, 500).tolist()
    vix = np.clip(20 + rng.normal(0, 8, 501), 5, 80).tolist()
    prices = _prices(spy, vix)
    pos = TurnOfMonthLowVol().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_requires_vix_close_column():
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="vix_close"):
        TurnOfMonthLowVol().generate_positions(prices)
