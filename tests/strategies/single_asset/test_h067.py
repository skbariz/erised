"""Tests for H067: VixPercentileRegime."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies import VixPercentileRegime


def _prices(vix_arr: list[float] | None = None, n: int = 300) -> pd.DataFrame:
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    rng = np.random.default_rng(7)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    vix = np.array(vix_arr) if vix_arr is not None else np.full(n, 15.0)
    return pd.DataFrame({"adj_close": spy, "vix_close": vix}, index=idx)


def test_raises_without_vix():
    strat = VixPercentileRegime()
    df = pd.DataFrame(
        {"adj_close": [100.0] * 100}, index=pd.date_range("2000-01-01", periods=100, freq="B")
    )
    with pytest.raises(ValueError, match="vix_close"):
        strat.generate_positions(df)


def test_returns_series_named_position():
    pos = VixPercentileRegime().generate_positions(_prices())
    assert isinstance(pos, pd.Series)
    assert pos.name == "position"


def test_positions_binary():
    pos = VixPercentileRegime().generate_positions(_prices())
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_flat_before_min_periods():
    strat = VixPercentileRegime(quantile=0.33)
    pos = strat.generate_positions(_prices(n=200))
    # Expanding quantile needs min_periods=63; first 64 bars should be 0
    assert pos.iloc[:64].sum() == pytest.approx(0.0)


def test_low_vix_produces_long():
    # VIX history: high during warmup, then drops well below the 33rd percentile
    n = 300
    vix = [25.0] * 150 + [8.0] * 150
    pos = VixPercentileRegime(quantile=0.33).generate_positions(_prices(vix_arr=vix, n=n))
    # After the drop, VIX=8 < 33rd pctile of history (mostly 25) → should be long
    assert pos.iloc[160:].sum() > 0


def test_high_vix_produces_flat():
    # Ramp VIX from 10 to 50 over history, then set last 50 bars to 80
    n = 300
    vix = list(np.linspace(10, 25, 250)) + [80.0] * 50
    pos = VixPercentileRegime(quantile=0.33).generate_positions(_prices(vix_arr=vix, n=n))
    # Last bars with VIX=80 well above historical 33rd pctile should be flat
    assert pos.iloc[-10:].sum() == pytest.approx(0.0)
