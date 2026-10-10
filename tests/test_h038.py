"""Focused tests for H038: LongTermReversal."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies.base import LongTermReversal


def _prices(rets, start="1993-02-01"):
    n = len(rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in rets:
        px.append(px[-1] * (1 + r))
    return pd.DataFrame({"adj_close": px}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    rets = rng.normal(0.0005, 0.01, 2000).tolist()
    prices = _prices(rets)
    assert_no_lookahead(LongTermReversal().generate_positions, prices)


def test_long_when_5yr_return_in_bottom_tercile():
    """After a long decline the 5yr rolling return should be in the bottom tercile -> long."""
    rng = np.random.default_rng(1)
    # Build 2500 bars: mostly small gains, then a sharp decline at the end
    rets = rng.normal(0.0003, 0.005, 2500).tolist()
    # Final 250 bars: strong losses to push 5yr return into bottom percentile
    rets[-250:] = [-0.002] * 250
    prices = _prices(rets)
    pos = LongTermReversal(lookback_days=1260, percentile=33).generate_positions(prices)
    # After enough history, strategy should be long at the end
    assert pos.iloc[-1] == pytest.approx(1.0)


def test_flat_when_5yr_return_high():
    """After a long strong rally the 5yr rolling return is in the top tercile -> flat."""
    rng = np.random.default_rng(2)
    # Mostly small gains, then a big rally pushing 5yr return to the top
    rets = rng.normal(0.0002, 0.005, 2500).tolist()
    rets[-250:] = [0.003] * 250
    prices = _prices(rets)
    pos = LongTermReversal(lookback_days=1260, percentile=33).generate_positions(prices)
    assert pos.iloc[-1] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(99)
    rets = rng.normal(0.0003, 0.01, 3000).tolist()
    prices = _prices(rets)
    pos = LongTermReversal().generate_positions(prices)
    assert set(pos.dropna().values.round(6)).issubset({0.0, 1.0})


def test_zero_during_warmup():
    """Before lookback_days + history_min bars, strategy should be flat."""
    rets = [0.001] * 500
    prices = _prices(rets)
    pos = LongTermReversal(lookback_days=252, history_min=252).generate_positions(prices)
    # Need at least lookback_days + history_min bars before first signal
    assert (pos.iloc[:252] == 0.0).all()
