"""Focused tests for H043: SellInMayBondConfirm."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.backtest.engine import assert_no_lookahead
from erised.strategies.base import SellInMayBondConfirm


def _prices(spy_rets, tlt_levels, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    px = [100.0]
    for r in spy_rets:
        px.append(px[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": px}, index=idx)
    assert len(tlt_levels) == n + 1
    df["tlt_close"] = tlt_levels
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    spy = rng.normal(0.0005, 0.01, 700).tolist()
    tlt = (100 + rng.normal(0, 2, 701)).tolist()
    prices = _prices(spy, tlt)
    assert_no_lookahead(SellInMayBondConfirm().generate_positions, prices)


def test_long_in_season_with_bond_riskoff():
    """In Nov-Apr AND TLT below MA (risk-on) -> position 1.0."""
    # 300 bars starting in November: in season; TLT low so MA > price
    spy = [0.001] * 300
    # TLT was high, then dropped; 5 bars after drop the MA should still be above
    tlt = [120.0] * 100 + [70.0] * 201
    prices = _prices(spy, tlt, start="2010-11-01")
    pos = SellInMayBondConfirm(tlt_ma_period=50).generate_positions(prices)
    assert pos.iloc[105] == pytest.approx(1.0)


def test_flat_in_season_with_bond_uptrend():
    """In Nov-Apr BUT TLT above MA (risk-off) -> position 0.0."""
    spy = [0.001] * 300
    # TLT was low, then spiked; MA still below price
    tlt = [70.0] * 100 + [130.0] * 201
    prices = _prices(spy, tlt, start="2010-11-01")
    pos = SellInMayBondConfirm(tlt_ma_period=50).generate_positions(prices)
    assert pos.iloc[105] == pytest.approx(0.0)


def test_flat_out_of_season():
    """May-Oct is always flat regardless of TLT."""
    spy = [0.001] * 300
    tlt = [70.0] * 301  # TLT low (would signal long if in season)
    prices = _prices(spy, tlt, start="2010-05-01")  # starts in May (out of season)
    pos = SellInMayBondConfirm(tlt_ma_period=50).generate_positions(prices)
    # May, June, July, Aug, Sep, Oct = flat (months 5-10)
    may_oct = prices.index.month.isin({5, 6, 7, 8, 9, 10})
    assert (pos[may_oct] == 0.0).all()


def test_position_is_binary():
    rng = np.random.default_rng(7)
    spy = rng.normal(0, 0.01, 500).tolist()
    tlt = (100 + rng.normal(0, 3, 501)).tolist()
    prices = _prices(spy, tlt)
    pos = SellInMayBondConfirm(tlt_ma_period=50).generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_requires_tlt_close_column():
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="tlt_close"):
        SellInMayBondConfirm().generate_positions(prices)
