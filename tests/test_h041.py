"""Focused tests for H041: PostCrashRecovery."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies.base import PostCrashRecovery


def _prices(spy_rets, vix_levels, start="2000-01-03"):
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
    spy = rng.normal(0.0003, 0.01, 500).tolist()
    vix = (20 + rng.uniform(0, 10, 501)).tolist()
    prices = _prices(spy, vix)
    assert_no_lookahead(PostCrashRecovery().generate_positions, prices)


def test_long_after_vix_spike_and_delay():
    """Position becomes 1 exactly delay_days after VIX spikes above threshold."""
    spy = [0.0001] * 200
    # VIX spikes at bar 50
    vix = [20.0] * 50 + [45.0] + [20.0] * 150
    prices = _prices(spy, vix)
    pos = PostCrashRecovery(vix_threshold=40, delay_days=10, hold_days=30).generate_positions(
        prices
    )
    # Spike at bar 50, delay=10 -> entry at bar 61; hold_days=30, so long bars 61-90
    assert pos.iloc[62] == pytest.approx(1.0)


def test_flat_when_vix_never_spikes():
    """If VIX stays below threshold the whole time, position is always 0."""
    spy = [0.0001] * 200
    vix = [20.0] * 201
    prices = _prices(spy, vix)
    pos = PostCrashRecovery(vix_threshold=40, delay_days=10, hold_days=30).generate_positions(
        prices
    )
    assert (pos == 0.0).all()


def test_position_is_binary():
    rng = np.random.default_rng(7)
    spy = rng.normal(0.0002, 0.01, 600).tolist()
    vix = np.clip(20 + rng.normal(0, 8, 601), 10, 80).tolist()
    prices = _prices(spy, vix)
    pos = PostCrashRecovery().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_requires_vix_close_column():
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="vix_close"):
        PostCrashRecovery().generate_positions(prices)
