"""Focused tests for H037: DualSafeHavenConfirmation."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies.base import DualSafeHavenConfirmation


def _prices(spy_rets, tlt_levels, gld_levels, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in spy_rets:
        px.append(px[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": px}, index=idx)
    assert len(tlt_levels) == n + 1
    assert len(gld_levels) == n + 1
    df["tlt_close"] = tlt_levels
    df["gld_close"] = gld_levels
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    tlt = (100 + rng.normal(0, 2, 701)).tolist()
    gld = (150 + rng.normal(0, 4, 701)).tolist()
    prices = _prices(spy, tlt, gld)
    assert_no_lookahead(DualSafeHavenConfirmation().generate_positions, prices)


def test_long_when_both_below_ma():
    """Both TLT and GLD drop sharply while MAs still reflect prior highs -> 1.0."""
    n = 200
    spy = [0.001] * n
    # Both safe havens high for 80 bars then drop; check 5 bars after drop
    tlt = [120.0] * 80 + [70.0] * 121
    gld = [160.0] * 80 + [90.0] * 121
    prices = _prices(spy, tlt, gld)
    pos = DualSafeHavenConfirmation(ma_period=30).generate_positions(prices)
    # At bar 85, both MAs still anchored at old highs while prices dropped
    assert pos.iloc[85] == pytest.approx(1.0)


def test_flat_when_tlt_above_ma():
    """TLT spikes above its MA even if GLD is below its MA -> position 0.0."""
    n = 200
    spy = [0.001] * n
    tlt = [70.0] * 80 + [130.0] * 121  # TLT spikes above MA
    gld = [160.0] * 80 + [90.0] * 121  # GLD drops below MA
    prices = _prices(spy, tlt, gld)
    pos = DualSafeHavenConfirmation(ma_period=30).generate_positions(prices)
    assert pos.iloc[85] == pytest.approx(0.0)


def test_flat_when_gld_above_ma():
    """GLD spikes above its MA even if TLT is below its MA -> position 0.0."""
    n = 200
    spy = [0.001] * n
    tlt = [120.0] * 80 + [70.0] * 121  # TLT drops below MA
    gld = [100.0] * 80 + [160.0] * 121  # GLD spikes above MA
    prices = _prices(spy, tlt, gld)
    pos = DualSafeHavenConfirmation(ma_period=30).generate_positions(prices)
    assert pos.iloc[85] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(15)
    spy = rng.normal(0, 0.01, 400).tolist()
    tlt = (100 + rng.normal(0, 3, 401)).tolist()
    gld = (150 + rng.normal(0, 5, 401)).tolist()
    prices = _prices(spy, tlt, gld)
    pos = DualSafeHavenConfirmation(ma_period=50).generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_zero_during_warmup():
    """Before ma_period warmup, position should be 0."""
    n = 100
    prices = _prices([0.001] * n, [100.0] * (n + 1), [150.0] * (n + 1))
    pos = DualSafeHavenConfirmation(ma_period=50).generate_positions(prices)
    assert (pos.iloc[:50] == 0.0).all()


def test_requires_both_columns():
    prices_no_tlt = pd.DataFrame(
        {"adj_close": [100.0, 101.0], "gld_close": [150.0, 151.0]},
        index=pd.bdate_range("2010-01-04", periods=2),
    )
    with pytest.raises(ValueError, match="tlt_close"):
        DualSafeHavenConfirmation().generate_positions(prices_no_tlt)

    prices_no_gld = pd.DataFrame(
        {"adj_close": [100.0, 101.0], "tlt_close": [90.0, 91.0]},
        index=pd.bdate_range("2010-01-04", periods=2),
    )
    with pytest.raises(ValueError, match="gld_close"):
        DualSafeHavenConfirmation().generate_positions(prices_no_gld)
