"""Tests for H081: DualFrequencyMomentum."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies import DualFrequencyMomentum


def _prices(n: int = 500, trend: float = 0.0003, noise: float = 0.01) -> pd.DataFrame:
    idx = pd.date_range("2000-01-01", periods=n, freq="B")
    rng = np.random.default_rng(81)
    spy = 100 * np.cumprod(1 + rng.normal(trend, noise, n))
    return pd.DataFrame({"adj_close": spy}, index=idx)


def test_returns_position_series():
    pos = DualFrequencyMomentum().generate_positions(_prices())
    assert isinstance(pos, pd.Series)
    assert pos.name == "position"


def test_shape_matches_input():
    prices = _prices(n=400)
    pos = DualFrequencyMomentum().generate_positions(prices)
    assert len(pos) == 400


def test_values_are_zero_or_one():
    pos = DualFrequencyMomentum().generate_positions(_prices())
    assert set(pos.unique()).issubset({0.0, 1.0})


def test_flat_before_warmup():
    prices = _prices(n=500)
    pos = DualFrequencyMomentum(mom_long=252, mom_skip=21, mom_short=21).generate_positions(prices)
    # First 252 rows: mom_long signal not yet defined → all flat
    assert (pos.iloc[:252] == 0.0).all()


def test_flat_when_long_trend_down():
    # Deterministically declining → 12-1m always negative → all flat
    idx = pd.date_range("2000-01-01", periods=500, freq="B")
    spy = 100 * np.cumprod(np.full(500, 0.998))
    prices = pd.DataFrame({"adj_close": spy}, index=idx)
    pos = DualFrequencyMomentum().generate_positions(prices)
    assert (pos.iloc[280:] == 0.0).all()


def test_long_when_both_positive():
    # Deterministically rising → both signals positive → should be long
    idx = pd.date_range("2000-01-01", periods=500, freq="B")
    spy = 100 * np.cumprod(np.full(500, 1.001))
    prices = pd.DataFrame({"adj_close": spy}, index=idx)
    pos = DualFrequencyMomentum().generate_positions(prices)
    assert (pos.iloc[280:] == 1.0).all()
