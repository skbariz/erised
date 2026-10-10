"""Focused tests for H034: SPYTLTCorrelationRegime."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies import SPYTLTCorrelationRegime


def _prices(spy_rets, tlt_rets, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy_px = [100.0]
    for r in spy_rets:
        spy_px.append(spy_px[-1] * (1 + r))
    tlt_px = [100.0]
    for r in tlt_rets:
        tlt_px.append(tlt_px[-1] * (1 + r))
    return pd.DataFrame({"adj_close": spy_px, "tlt_close": tlt_px}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    tlt = rng.normal(0.0002, 0.005, 700).tolist()
    prices = _prices(spy, tlt)
    assert_no_lookahead(SPYTLTCorrelationRegime().generate_positions, prices)


def test_long_when_correlation_negative():
    """Perfectly negatively correlated returns -> position 1.0."""
    n = 200
    base = [0.005 if i % 2 == 0 else -0.005 for i in range(n)]
    spy = base
    tlt = [-r for r in base]  # perfect negative correlation
    prices = _prices(spy, tlt)
    pos = SPYTLTCorrelationRegime(corr_window=20).generate_positions(prices)
    assert pos.iloc[25] == pytest.approx(1.0)


def test_flat_when_correlation_positive():
    """Perfectly positively correlated returns -> position 0.0."""
    n = 200
    base = [0.005 if i % 2 == 0 else -0.005 for i in range(n)]
    spy = base
    tlt = base  # perfect positive correlation
    prices = _prices(spy, tlt)
    pos = SPYTLTCorrelationRegime(corr_window=20).generate_positions(prices)
    assert pos.iloc[25] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(9)
    spy = rng.normal(0, 0.01, 400).tolist()
    tlt = rng.normal(0, 0.005, 400).tolist()
    prices = _prices(spy, tlt)
    pos = SPYTLTCorrelationRegime(corr_window=30).generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_zero_during_warmup():
    """Before corr_window warmup, position should be 0."""
    spy = [0.001] * 100
    tlt = [-0.001] * 100
    prices = _prices(spy, tlt)
    pos = SPYTLTCorrelationRegime(corr_window=40).generate_positions(prices)
    assert (pos.iloc[:40] == 0.0).all()


def test_requires_tlt_close_column():
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="tlt_close"):
        SPYTLTCorrelationRegime().generate_positions(prices)
