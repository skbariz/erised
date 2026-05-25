"""Tests for Batch 1 cross-sectional strategies (H150-H154).

Tests run on a small synthetic panel (no real data required).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.backtest.panel_engine import count_stock_trades, run_panel_backtest
from quant.strategies.cross_sectional import (
    HighWaterMomentum,
    MomentumLongOnly,
    MomentumMarketNeutral,
    SectorNeutralMomentum,
    TrendGatedMomentum,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small panel: 10 tickers, 500 trading days."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2010-01-01", periods=500)
    tickers = [f"T{i:02d}" for i in range(10)]
    returns = rng.normal(0.0005, 0.015, size=(500, 10))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    membership = pd.DataFrame(True, index=dates, columns=tickers)
    # T08 and T09 join late (simulate late additions)
    membership.iloc[:100, -2:] = False
    return panel, membership


@pytest.fixture
def spy_prices(synthetic_panel) -> pd.Series:
    panel, _ = synthetic_panel
    return panel.iloc[:, 0].rename("SPY")


# ---------------------------------------------------------------------------
# Panel engine tests
# ---------------------------------------------------------------------------


def test_panel_backtest_long_only_no_cost(synthetic_panel):
    panel, membership = synthetic_panel
    # All-in first ticker, no rebalancing
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    weights["T00"] = 1.0
    result = run_panel_backtest(panel, weights, commission_bps=0, slippage_bps=0)
    assert result.equity.iloc[-1] > 0


def test_panel_backtest_zero_weights_flat(synthetic_panel):
    panel, _ = synthetic_panel
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    result = run_panel_backtest(panel, weights)
    assert (result.net_returns == 0).all()
    assert result.equity.iloc[-1] == pytest.approx(1.0)


def test_count_stock_trades(synthetic_panel):
    panel, _ = synthetic_panel
    weights = pd.DataFrame(0.0, index=panel.index, columns=panel.columns)
    weights.iloc[10:20, :3] = 1 / 3
    n = count_stock_trades(weights)
    assert n > 0


# ---------------------------------------------------------------------------
# H150: MomentumLongOnly
# ---------------------------------------------------------------------------


def test_h150_weights_sum_leq_one(synthetic_panel):
    panel, membership = synthetic_panel
    strat = MomentumLongOnly(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    row_sums = weights.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-9).all(), "Long-only weights must sum to ≤ 1.0"


def test_h150_weights_non_negative(synthetic_panel):
    panel, membership = synthetic_panel
    strat = MomentumLongOnly(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights >= -1e-9).all().all(), "Long-only weights must be non-negative"


def test_h150_respects_membership(synthetic_panel):
    panel, membership = synthetic_panel
    strat = MomentumLongOnly(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    # T08/T09 should have zero weight in first 100 rows (membership=False)
    early_weights_t08 = weights.iloc[:100]["T08"]
    assert (early_weights_t08 == 0).all(), "Non-member stocks must have zero weight"


def test_h150_no_lookahead(synthetic_panel):
    panel, membership = synthetic_panel
    strat = MomentumLongOnly(long_days=60, skip_days=5)
    w_full = strat.generate_weights(panel, membership)
    # Truncate at row 300 and recompute; weights at row 300 must be identical
    w_trunc = strat.generate_weights(panel.iloc[:301], membership.iloc[:301])
    date = panel.index[300]
    pd.testing.assert_series_equal(w_full.loc[date], w_trunc.loc[date], check_names=False)


def test_h150_monthly_rebalance_stable_intramonth(synthetic_panel):
    panel, membership = synthetic_panel
    strat = MomentumLongOnly(long_days=60, skip_days=5, monthly_rebalance=True)
    weights = strat.generate_weights(panel, membership)
    # Within any single calendar month, all rows must have identical weights
    months = weights.index.to_series().dt.to_period("M")
    for _month, grp in weights.groupby(months):
        first_row = grp.iloc[0]
        for _, row in grp.iterrows():
            pd.testing.assert_series_equal(row, first_row, check_names=False)


# ---------------------------------------------------------------------------
# H151: MomentumMarketNeutral
# ---------------------------------------------------------------------------


def test_h151_gross_exposure_leq_one(synthetic_panel):
    panel, membership = synthetic_panel
    strat = MomentumMarketNeutral(long_days=60, skip_days=5)
    weights = strat.generate_weights(panel, membership)
    gross = weights.abs().sum(axis=1)
    # Gross should be ≤ 1.0 (0.5 long + 0.5 short)
    assert (gross <= 1.0 + 1e-6).all()


def test_h151_net_exposure_near_zero(synthetic_panel):
    panel, membership = synthetic_panel
    strat = MomentumMarketNeutral(long_days=60, skip_days=5)
    weights = strat.generate_weights(panel, membership)
    # After warmup, net should be near zero (long leg cancels short leg)
    net = weights.sum(axis=1)
    active = weights.abs().sum(axis=1) > 0.01
    if active.any():
        assert net[active].abs().max() < 0.1


def test_h151_monthly_rebalance_stable_intramonth(synthetic_panel):
    panel, membership = synthetic_panel
    strat = MomentumMarketNeutral(long_days=60, skip_days=5, monthly_rebalance=True)
    weights = strat.generate_weights(panel, membership)
    months = weights.index.to_series().dt.to_period("M")
    for _month, grp in weights.groupby(months):
        first_row = grp.iloc[0]
        for _, row in grp.iterrows():
            pd.testing.assert_series_equal(row, first_row, check_names=False)


# ---------------------------------------------------------------------------
# H152: HighWaterMomentum
# ---------------------------------------------------------------------------


def test_h152_weights_sum_leq_one(synthetic_panel):
    panel, membership = synthetic_panel
    strat = HighWaterMomentum(lookback=60, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_h152_weights_non_negative(synthetic_panel):
    panel, membership = synthetic_panel
    strat = HighWaterMomentum(lookback=60, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights >= -1e-9).all().all()


def test_h152_monthly_rebalance_stable_intramonth(synthetic_panel):
    panel, membership = synthetic_panel
    strat = HighWaterMomentum(lookback=60, top_quantile=0.3, monthly_rebalance=True)
    weights = strat.generate_weights(panel, membership)
    months = weights.index.to_series().dt.to_period("M")
    for _month, grp in weights.groupby(months):
        first_row = grp.iloc[0]
        for _, row in grp.iterrows():
            pd.testing.assert_series_equal(row, first_row, check_names=False)


# ---------------------------------------------------------------------------
# H153: SectorNeutralMomentum
# ---------------------------------------------------------------------------


def test_h153_weights_sum_leq_one(synthetic_panel):
    panel, membership = synthetic_panel
    tickers = panel.columns.tolist()
    # Assign 2 sectors of 5 tickers each
    sector_map = pd.Series(["SectorA"] * 5 + ["SectorB"] * 5, index=tickers, name="sector")
    strat = SectorNeutralMomentum(sector_map=sector_map, long_days=60, skip_days=5)
    weights = strat.generate_weights(panel, membership)
    assert (weights.sum(axis=1) <= 1.0 + 1e-6).all()
    assert (weights >= -1e-9).all().all()


def test_h153_equal_sector_weights(synthetic_panel):
    panel, membership = synthetic_panel
    tickers = panel.columns.tolist()
    sector_map = pd.Series(["SectorA"] * 5 + ["SectorB"] * 5, index=tickers, name="sector")
    strat = SectorNeutralMomentum(sector_map=sector_map, long_days=60, skip_days=5)
    weights = strat.generate_weights(panel, membership)
    # On active rows, each sector's total weight should be approximately equal
    active_rows = weights.sum(axis=1) > 1e-6
    if active_rows.any():
        sector_a_total = weights.loc[active_rows, tickers[:5]].sum(axis=1)
        sector_b_total = weights.loc[active_rows, tickers[5:]].sum(axis=1)
        # Both sectors get equal weight (0.5 each when both active)
        diff = (sector_a_total - sector_b_total).abs()
        assert (diff < 1e-9).all(), "Sector weights must be equal"


# ---------------------------------------------------------------------------
# H154: TrendGatedMomentum
# ---------------------------------------------------------------------------


def test_h154_zero_exposure_below_ma(synthetic_panel, spy_prices):
    panel, membership = synthetic_panel
    # Create a monotonically declining series so prices are always below their MA
    dates = spy_prices.index
    spy_down = pd.Series(np.linspace(1000, 100, len(dates)), index=dates)
    strat = TrendGatedMomentum(spy_prices=spy_down, long_days=60, skip_days=5, ma_long=50)
    weights = strat.generate_weights(panel, membership)
    # After warmup (ma_long + a few days buffer), exposure must be zero
    post_warmup = weights.iloc[55:]
    assert (post_warmup.abs().sum(axis=1) < 1e-9).all()


def test_h154_full_exposure_above_ma(synthetic_panel, spy_prices):
    panel, membership = synthetic_panel
    # Monotonically increasing series: price is always above its rolling mean
    dates = spy_prices.index
    spy_up = pd.Series(np.linspace(100, 10000, len(dates)), index=dates)
    strat = TrendGatedMomentum(spy_prices=spy_up, long_days=60, skip_days=5, ma_long=50)
    w_gated = strat.generate_weights(panel, membership)
    # Use daily rebalance for direct signal comparison (TrendGatedMomentum is daily)
    strat_base = MomentumLongOnly(long_days=60, skip_days=5, monthly_rebalance=False)
    w_base = strat_base.generate_weights(panel, membership)
    # Post-warmup, gated should match base
    post = slice(55, None)
    pd.testing.assert_frame_equal(
        w_gated.iloc[post].reset_index(drop=True),
        w_base.iloc[post].reset_index(drop=True),
        check_like=True,
        atol=1e-9,
    )


def test_h154_internal_trend_no_spy(synthetic_panel):
    panel, membership = synthetic_panel
    # Without spy_prices, should use equal-weight S&P 500 index internally
    strat = TrendGatedMomentum(long_days=60, skip_days=5, ma_long=50)
    weights = strat.generate_weights(panel, membership)
    # Weights must be valid (non-negative, sum ≤ 1)
    assert (weights >= -1e-9).all().all()
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()
