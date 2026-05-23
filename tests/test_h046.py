"""Focused tests for H046: HalloweenMomentumConfirm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import HalloweenMomentumConfirm


def _prices(rets, start="2010-01-04"):
    n = len(rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in rets:
        px.append(px[-1] * (1 + r))
    return pd.DataFrame({"adj_close": px}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, 700).tolist()
    prices = _prices(rets)
    assert_no_lookahead(HalloweenMomentumConfirm().generate_positions, prices)


def test_long_in_season_with_positive_momentum():
    """November with positive 6m momentum -> position 1.0."""
    # Build 200 days of positive returns before reaching November
    rets = [0.003] * 200
    prices = _prices(rets, start="2010-05-01")
    pos = HalloweenMomentumConfirm(momentum_days=63).generate_positions(prices)
    # After 200 bars from May = ~Nov; momentum is strongly positive
    assert pos.iloc[-1] == pytest.approx(1.0)


def test_flat_in_season_with_negative_momentum():
    """November but negative 6m momentum -> position 0.0."""
    # 200 bars of strong declines -> negative 63d momentum
    rets = [-0.003] * 200
    prices = _prices(rets, start="2010-05-01")
    pos = HalloweenMomentumConfirm(momentum_days=63).generate_positions(prices)
    assert pos.iloc[-1] == pytest.approx(0.0)


def test_flat_out_of_season():
    """May-Oct is always flat regardless of momentum."""
    rets = [0.003] * 300
    prices = _prices(rets, start="2010-05-01")  # starts in season-off months
    pos = HalloweenMomentumConfirm(momentum_days=63).generate_positions(prices)
    # First bar is May (month 5) — out of season
    assert pos.iloc[0] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(99)
    rets = rng.normal(0.0002, 0.01, 600).tolist()
    prices = _prices(rets)
    pos = HalloweenMomentumConfirm().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})
