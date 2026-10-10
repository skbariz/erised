"""Tests for H068: VrpCalendarHybrid."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.base import VrpCalendarHybrid


def _prices(start: str = "2000-01-01", n: int = 300, vix_val: float = 25.0) -> pd.DataFrame:
    idx = pd.date_range(start, periods=n, freq="B")
    rng = np.random.default_rng(99)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    vix = np.full(n, vix_val)
    return pd.DataFrame({"adj_close": spy, "vix_close": vix}, index=idx)


def test_raises_without_vix():
    strat = VrpCalendarHybrid()
    df = pd.DataFrame(
        {"adj_close": [100.0] * 50}, index=pd.date_range("2000-01-01", periods=50, freq="B")
    )
    with pytest.raises(ValueError, match="vix_close"):
        strat.generate_positions(df)


def test_returns_series():
    pos = VrpCalendarHybrid().generate_positions(_prices())
    assert isinstance(pos, pd.Series)
    assert pos.name == "position"


def test_positions_binary():
    pos = VrpCalendarHybrid().generate_positions(_prices())
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_flat_in_summer_even_when_vrp_fires():
    # Use July which is NOT in Nov-Apr window
    prices = _prices(start="2005-07-01", n=60, vix_val=30.0)
    pos = VrpCalendarHybrid(realized_window=20).generate_positions(prices)
    # VIX=30 >> realized ~1.6% annualized, so VRP fires — but summer → flat
    assert pos.iloc[25:].sum() == pytest.approx(0.0)


def test_long_in_winter_when_vrp_fires():
    # Use January (in Nov-Apr), VIX=30, realized_vol will be low (small daily moves)
    prices = _prices(start="2005-01-03", n=80, vix_val=30.0)
    pos = VrpCalendarHybrid(realized_window=20).generate_positions(prices)
    # After warmup (20 bars), VIX=30 > realized vol → in season → should be long
    assert pos.iloc[25:].sum() > 0


def test_flat_when_vrp_off():
    # VIX = 5 (very low) → VIX < realized vol → VRP signal off → flat regardless of season
    prices = _prices(start="2005-01-03", n=80, vix_val=5.0)
    pos = VrpCalendarHybrid(realized_window=20).generate_positions(prices)
    assert pos.iloc[25:].sum() == pytest.approx(0.0)
