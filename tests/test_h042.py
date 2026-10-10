"""Focused tests for H042: VolOfVolCalm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies import VolOfVolCalm


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
    spy = rng.normal(0.0003, 0.01, 600).tolist()
    vix = (20 + rng.uniform(0, 5, 601)).tolist()
    prices = _prices(spy, vix)
    assert_no_lookahead(VolOfVolCalm().generate_positions, prices)


def test_long_during_calm_vol_regime():
    """After chaotic VIX history, tiny final VoV is in bottom tercile -> long."""
    rng = np.random.default_rng(5)
    spy = rng.normal(0.0003, 0.01, 1000).tolist()
    # High VoV history to push p33 up, then extremely calm VIX at the end
    vix_high = (20 + rng.normal(0, 8, 701)).tolist()
    vix_calm = [20.0] * 300
    vix = vix_high + vix_calm
    prices = _prices(spy, vix)
    pos = VolOfVolCalm(vov_window=20, history_min=252, percentile=33).generate_positions(prices)
    assert pos.iloc[-1] == pytest.approx(1.0)


def test_flat_during_volatile_vol_regime():
    """Highly volatile VIX -> VoV in top tercile -> flat."""
    rng = np.random.default_rng(6)
    spy = rng.normal(0.0003, 0.01, 1000).tolist()
    # Calm VIX first to anchor p33 low, then chaotic VIX pushes current VoV above p33
    vix_calm = [20.0] * 601
    vix_high = (20 + rng.normal(0, 20, 400)).tolist()
    vix = vix_calm + vix_high
    prices = _prices(spy, vix)
    pos = VolOfVolCalm(vov_window=20, history_min=252, percentile=33).generate_positions(prices)
    assert pos.iloc[-1] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(99)
    spy = rng.normal(0.0002, 0.01, 800).tolist()
    vix = np.clip(20 + rng.normal(0, 5, 801), 5, 80).tolist()
    prices = _prices(spy, vix)
    pos = VolOfVolCalm().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_zero_during_warmup():
    """Before history_min bars, strategy should be flat."""
    spy = [0.0003] * 400
    vix = [20.0] * 401
    prices = _prices(spy, vix)
    pos = VolOfVolCalm(vov_window=20, history_min=252).generate_positions(prices)
    assert (pos.iloc[:252] == 0.0).all()


def test_requires_vix_close_column():
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="vix_close"):
        VolOfVolCalm().generate_positions(prices)
