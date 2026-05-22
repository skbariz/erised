"""Focused tests for H032: VIXMeanReversion."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import VIXMeanReversion


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
    vix = (18 + rng.normal(0, 4, 701)).clip(10, 60).tolist()
    prices = _prices(spy, vix)
    assert_no_lookahead(VIXMeanReversion().generate_positions, prices)


def test_active_on_extreme_vix_spike():
    """Extreme VIX spike (z > n_sigma) -> position 1.0."""
    spy = [0.001] * 200
    # Low stable VIX for lookback, then extreme spike
    vix = [15.0] * 160 + [60.0] + [15.0] * 40
    prices = _prices(spy, vix)
    pos = VIXMeanReversion(lookback=60, n_sigma=2.0).generate_positions(prices)
    # Day after spike (1-bar shift) should be active
    assert pos.iloc[161] == pytest.approx(1.0)


def test_flat_during_normal_vix():
    """Normal VIX -> position 0."""
    spy = [0.001] * 200
    vix = [18.0] * 201
    prices = _prices(spy, vix)
    pos = VIXMeanReversion(lookback=60, n_sigma=2.0).generate_positions(prices)
    # After warmup, flat VIX should produce no signal
    assert (pos.iloc[62:] == 0.0).all()


def test_position_is_binary():
    rng = np.random.default_rng(5)
    spy = rng.normal(0, 0.01, 300).tolist()
    vix = (20 + rng.normal(0, 5, 301)).clip(10, 60).tolist()
    prices = _prices(spy, vix)
    pos = VIXMeanReversion().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_lower_sigma_more_signals():
    """Lower n_sigma threshold -> more active days."""
    rng = np.random.default_rng(6)
    spy = rng.normal(0, 0.01, 400).tolist()
    vix = (20 + rng.normal(0, 5, 401)).clip(10, 60).tolist()
    prices = _prices(spy, vix)
    pos1 = VIXMeanReversion(n_sigma=1.0).generate_positions(prices)
    pos2 = VIXMeanReversion(n_sigma=2.0).generate_positions(prices)
    pos3 = VIXMeanReversion(n_sigma=3.0).generate_positions(prices)
    assert pos1.sum() >= pos2.sum() >= pos3.sum()


def test_zero_during_warmup():
    """Before lookback warmup, position should be 0."""
    spy = [0.001] * 100
    vix = [20.0] * 101
    prices = _prices(spy, vix)
    pos = VIXMeanReversion(lookback=60).generate_positions(prices)
    assert (pos.iloc[:60] == 0.0).all()


def test_requires_vix_close_column():
    """Raises ValueError if vix_close column is missing."""
    prices = pd.DataFrame({"adj_close": [100.0, 101.0, 102.0]},
                          index=pd.bdate_range("2010-01-04", periods=3))
    with pytest.raises(ValueError, match="vix_close"):
        VIXMeanReversion().generate_positions(prices)
