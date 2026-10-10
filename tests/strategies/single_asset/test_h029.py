"""Focused tests for H029: VIXRegimeFilter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies import VIXRegimeFilter


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
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    vix = (18 + rng.normal(0, 4, 701)).clip(10, 50).tolist()
    prices = _prices(spy, vix)
    assert_no_lookahead(VIXRegimeFilter().generate_positions, prices)


def test_long_when_vix_low():
    """VIX below threshold -> position 1.0."""
    spy = [0.001] * 100
    vix = [12.0] * 101
    prices = _prices(spy, vix)
    pos = VIXRegimeFilter(vix_threshold=20.0).generate_positions(prices)
    assert (pos.iloc[1:] == 1.0).all()


def test_flat_when_vix_high():
    """VIX above threshold -> position 0.0."""
    spy = [0.001] * 100
    vix = [35.0] * 101
    prices = _prices(spy, vix)
    pos = VIXRegimeFilter(vix_threshold=20.0).generate_positions(prices)
    assert (pos == 0.0).all()


def test_switches_on_threshold():
    """Position switches cleanly when VIX crosses threshold."""
    spy = [0.001] * 100
    vix = [15.0] * 50 + [25.0] * 51
    prices = _prices(spy, vix)
    pos = VIXRegimeFilter(vix_threshold=20.0).generate_positions(prices)
    # First 50 bars: VIX=15 -> long (after 1-bar shift)
    assert pos.iloc[1] == pytest.approx(1.0)
    # After VIX crosses 20 -> flat
    assert pos.iloc[52] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(2)
    spy = rng.normal(0, 0.01, 200).tolist()
    vix = (20 + rng.normal(0, 5, 201)).clip(10, 50).tolist()
    prices = _prices(spy, vix)
    pos = VIXRegimeFilter().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_threshold_param():
    """Lower threshold means fewer active days."""
    spy = [0.001] * 300
    vix = [22.0] * 301
    prices = _prices(spy, vix)
    pos15 = VIXRegimeFilter(vix_threshold=15.0).generate_positions(prices)
    pos25 = VIXRegimeFilter(vix_threshold=25.0).generate_positions(prices)
    assert pos25.sum() > pos15.sum()


def test_requires_vix_close_column():
    """Raises ValueError if vix_close column is missing."""
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="vix_close"):
        VIXRegimeFilter().generate_positions(prices)
