"""Focused tests for H028: VIXSpikeEntry."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies import VIXSpikeEntry


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
    vix = (20 + rng.normal(0, 5, 701)).clip(10, 80).tolist()
    prices = _prices(spy, vix)
    assert_no_lookahead(VIXSpikeEntry().generate_positions, prices)


def test_flat_before_spike():
    """Position should be 0 when VIX has never crossed threshold."""
    spy = [0.001] * 100
    vix = [15.0] * 101
    prices = _prices(spy, vix)
    pos = VIXSpikeEntry(vix_threshold=30.0, hold_days=20).generate_positions(prices)
    assert (pos == 0.0).all()


def test_active_immediately_after_spike():
    """Position should activate in the bar following a VIX spike."""
    spy = [0.001] * 100
    vix = [15.0] * 50 + [35.0] + [15.0] * 50
    prices = _prices(spy, vix)
    pos = VIXSpikeEntry(vix_threshold=30.0, hold_days=20).generate_positions(prices)
    # VIX spike at index 50; with 1-bar shift, active from index 51
    assert pos.iloc[51] == pytest.approx(1.0)


def test_active_within_hold_window():
    """Position should remain 1 for hold_days after spike."""
    spy = [0.001] * 100
    vix = [15.0] * 40 + [35.0] + [15.0] * 60
    prices = _prices(spy, vix)
    pos = VIXSpikeEntry(vix_threshold=30.0, hold_days=10).generate_positions(prices)
    # Should be active for 10 bars after spike
    assert pos.iloc[41] == pytest.approx(1.0)
    assert pos.iloc[50] == pytest.approx(1.0)
    assert pos.iloc[51] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(1)
    spy = rng.normal(0, 0.01, 200).tolist()
    vix = (20 + rng.normal(0, 8, 201)).clip(10, 80).tolist()
    prices = _prices(spy, vix)
    pos = VIXSpikeEntry().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_vix_threshold_param():
    """Lower threshold should produce more active days."""
    spy = [0.001] * 300
    vix = [25.0] * 301
    prices = _prices(spy, vix)
    pos20 = VIXSpikeEntry(vix_threshold=20.0).generate_positions(prices)
    pos30 = VIXSpikeEntry(vix_threshold=30.0).generate_positions(prices)
    assert pos20.sum() > pos30.sum()


def test_requires_vix_close_column():
    """Raises ValueError if vix_close column is missing."""
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="vix_close"):
        VIXSpikeEntry().generate_positions(prices)
