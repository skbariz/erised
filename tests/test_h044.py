"""Focused tests for H044: JanuaryBarometerGoldFilter."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.engine import assert_no_lookahead
from quant.strategies.base import JanuaryBarometerGoldFilter


def _prices_with_gld(n_years=5, spy_jan_ret=0.03, gld_trend="flat", start_year=2010):
    """Build a multi-year price series with controllable January SPY and GLD behavior."""
    trading_days = []
    spy_px = [100.0]
    gld_px_list = [80.0]

    for year in range(start_year, start_year + n_years):
        jan_days = pd.bdate_range(f"{year}-01-02", f"{year}-01-31")
        feb_dec_days = pd.bdate_range(f"{year}-02-01", f"{year}-12-31")
        trading_days.extend(jan_days.tolist() + feb_dec_days.tolist())

        # SPY: positive January return
        n_jan = len(jan_days)
        per_day_jan = (1 + spy_jan_ret) ** (1 / n_jan) - 1
        for _ in range(n_jan):
            spy_px.append(spy_px[-1] * (1 + per_day_jan))

        # SPY: flat Feb-Dec
        for _ in range(len(feb_dec_days)):
            spy_px.append(spy_px[-1] * 1.001)

        # GLD: rising (fear) or flat
        for _ in range(n_jan):
            if gld_trend == "rising":
                gld_px_list.append(gld_px_list[-1] * 1.005)
            else:
                gld_px_list.append(gld_px_list[-1])
        for _ in range(len(feb_dec_days)):
            gld_px_list.append(gld_px_list[-1])

    # Build DataFrame aligned with spy (one extra for the first reference bar)
    idx = pd.DatetimeIndex(trading_days)
    spy_arr = spy_px[1 : len(idx) + 1]
    gld_arr = gld_px_list[1 : len(idx) + 1]
    return pd.DataFrame({"adj_close": spy_arr, "gld_close": gld_arr}, index=idx)


def test_no_lookahead():
    rng = np.random.default_rng(42)
    idx = pd.bdate_range("2005-01-03", periods=1000)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 1000))
    gld = 80 * np.cumprod(1 + rng.normal(0.0002, 0.008, 1000))
    prices = pd.DataFrame({"adj_close": spy, "gld_close": gld}, index=idx)
    assert_no_lookahead(JanuaryBarometerGoldFilter().generate_positions, prices)


def test_long_when_positive_january_and_gold_flat():
    """Positive Jan + GLD flat (not trending up) -> long Feb-Dec."""
    prices = _prices_with_gld(n_years=3, spy_jan_ret=0.03, gld_trend="flat", start_year=2010)
    pos = JanuaryBarometerGoldFilter(gld_ma_period=20).generate_positions(prices)
    # Feb 2011 should be long (after the first full year with enough GLD history)
    feb_2011 = pos[(pos.index.year == 2011) & (pos.index.month == 2)]
    if len(feb_2011) > 0:
        assert feb_2011.iloc[0] == pytest.approx(1.0)


def test_flat_when_gold_rising_in_january():
    """Positive Jan BUT GLD trending up (fear signal) -> flat Feb-Dec."""
    prices = _prices_with_gld(n_years=3, spy_jan_ret=0.03, gld_trend="rising", start_year=2010)
    pos = JanuaryBarometerGoldFilter(gld_ma_period=20).generate_positions(prices)
    # Gold is rising — signal should be blocked
    feb_2011 = pos[(pos.index.year == 2011) & (pos.index.month == 2)]
    if len(feb_2011) > 0:
        assert feb_2011.iloc[0] == pytest.approx(0.0)


def test_position_is_binary():
    rng = np.random.default_rng(99)
    idx = pd.bdate_range("2005-01-03", periods=2000)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, 2000))
    gld = 80 * np.cumprod(1 + rng.normal(0.0001, 0.008, 2000))
    prices = pd.DataFrame({"adj_close": spy, "gld_close": gld}, index=idx)
    pos = JanuaryBarometerGoldFilter().generate_positions(prices)
    assert set(pos.values.round(6)).issubset({0.0, 1.0})


def test_requires_gld_close_column():
    prices = pd.DataFrame(
        {"adj_close": [100.0, 101.0, 102.0]}, index=pd.bdate_range("2010-01-04", periods=3)
    )
    with pytest.raises(ValueError, match="gld_close"):
        JanuaryBarometerGoldFilter().generate_positions(prices)
