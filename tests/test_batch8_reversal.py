"""Tests for H185-H189 short-term reversal and mean-reversion strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.cross_sectional import (
    LiquidityFilteredReversal,
    MonthlyReversalLong,
    ResidualOUMeanReversion,
    ResidualWeeklyReversal,
    WeeklyReversalSectorNeutral,
)

N_DATES = 1400
TICKERS = ["LOSER", "WINNER", "VOLATILE", "STABLE", "NEUTRAL"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def sectors() -> pd.Series:
    return pd.Series(
        {
            "LOSER": "Tech",
            "WINNER": "Tech",
            "VOLATILE": "Finance",
            "STABLE": "Finance",
            "NEUTRAL": "Healthcare",
        },
        name="sector",
    )


@pytest.fixture
def spy_rets(dates: pd.DatetimeIndex) -> pd.Series:
    np.random.seed(42)
    return pd.Series(np.random.randn(N_DATES) * 0.01, index=dates, name="SPY")


@pytest.fixture
def prices(dates: pd.DatetimeIndex, spy_rets: pd.Series) -> pd.DataFrame:
    """VOLATILE has high daily vol, STABLE has low daily vol."""
    np.random.seed(99)
    spy = spy_rets.values
    vols = {"LOSER": 0.01, "WINNER": 0.01, "VOLATILE": 0.025, "STABLE": 0.003, "NEUTRAL": 0.01}
    data = {}
    for t, v in vols.items():
        idio = np.random.randn(N_DATES) * v
        ret = 1.0 * spy + idio
        data[t] = 50.0 * np.cumprod(1 + ret)
    return pd.DataFrame(data, index=dates)


# ---------------------------------------------------------------------------
# WeeklyReversalSectorNeutral (H185)
# ---------------------------------------------------------------------------


class TestWeeklyReversalSectorNeutral:
    def test_weights_sum_to_one(self, prices, membership, sectors, dates):
        strat = WeeklyReversalSectorNeutral(sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[200:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_no_nan_weights(self, prices, membership, sectors):
        strat = WeeklyReversalSectorNeutral(sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_weekly_rebalance_structure(self, prices, membership, sectors):
        """Weights should be constant for 5-day blocks (periodic rebalance)."""
        strat = WeeklyReversalSectorNeutral(sectors, rebalance_period=5)
        w = strat.generate_weights(prices, membership)
        # Days 1-4 should match day 0; days 6-9 should match day 5
        for start in [0, 5, 10]:
            block = w.iloc[start : start + 5]
            if len(block) == 5:
                first_row = block.iloc[0]
                assert (block - first_row).abs().max().max() < 1e-10

    def test_loser_bias(self, prices, membership, sectors, dates):
        """Construct a deterministic case: force LOSER to have worst 5d return."""
        # Create prices where LOSER always declines for 5 days
        np.random.seed(5)
        n = N_DATES
        data = {}
        for t in TICKERS:
            ret = np.zeros(n)
            if t == "LOSER":
                # Periodic -2% every 5 days followed by bounce
                for i in range(0, n, 5):
                    ret[i : i + 5] = -0.004  # 5d * -0.4% = -2% per 5d
            elif t == "WINNER":
                for i in range(0, n, 5):
                    ret[i : i + 5] = 0.004
            data[t] = 50.0 * np.cumprod(1 + ret)
        prices_det = pd.DataFrame(data, index=dates)
        strat = WeeklyReversalSectorNeutral(sectors)
        w = strat.generate_weights(prices_det, membership)
        late = w.loc[dates[200:]]
        # LOSER (worst return) should dominate over WINNER in Tech sector
        assert late["LOSER"].mean() >= late["WINNER"].mean()


# ---------------------------------------------------------------------------
# ResidualWeeklyReversal (H186)
# ---------------------------------------------------------------------------


class TestResidualWeeklyReversal:
    def test_weights_sum_to_one(self, prices, membership, spy_rets, sectors, dates):
        strat = ResidualWeeklyReversal(spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[400:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_no_nan_weights(self, prices, membership, spy_rets, sectors):
        strat = ResidualWeeklyReversal(spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_weekly_structure(self, prices, membership, spy_rets, sectors):
        """Weights held constant within each 5-day period."""
        strat = ResidualWeeklyReversal(spy_rets, sectors, rebalance_period=5)
        w = strat.generate_weights(prices, membership)
        for start in [10, 15, 20]:
            block = w.iloc[start : start + 5]
            if len(block) == 5:
                first = block.iloc[0]
                assert (block - first).abs().max().max() < 1e-10


# ---------------------------------------------------------------------------
# LiquidityFilteredReversal (H187)
# ---------------------------------------------------------------------------


class TestLiquidityFilteredReversal:
    def test_weights_sum_to_one(self, prices, membership, sectors, dates):
        strat = LiquidityFilteredReversal(sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[200:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_no_nan_weights(self, prices, membership, sectors):
        strat = LiquidityFilteredReversal(sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_volatile_over_stable(self, prices, membership, sectors, dates):
        """VOLATILE (high realized vol) should appear more than STABLE (low vol)."""
        strat = LiquidityFilteredReversal(sectors, illiq_quantile=0.50)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[200:]]
        assert late["VOLATILE"].mean() >= late["STABLE"].mean()


# ---------------------------------------------------------------------------
# MonthlyReversalLong (H188)
# ---------------------------------------------------------------------------


class TestMonthlyReversalLong:
    def test_weights_sum_to_one(self, prices, membership, sectors, dates):
        strat = MonthlyReversalLong(sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[100:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_no_nan_weights(self, prices, membership, sectors):
        strat = MonthlyReversalLong(sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_monthly_rebalance_structure(self, prices, membership, sectors):
        """Weights must be constant within each calendar month."""
        strat = MonthlyReversalLong(sectors)
        w = strat.generate_weights(prices, membership)
        month_label = w.index.to_period("M")
        for month in month_label.unique()[2:7]:
            rows = w[month_label == month]
            if len(rows) > 1:
                first = rows.iloc[0]
                assert (rows - first).abs().max().max() < 1e-10


# ---------------------------------------------------------------------------
# ResidualOUMeanReversion (H189)
# ---------------------------------------------------------------------------


class TestResidualOUMeanReversion:
    def test_weights_sum_to_one_or_zero(self, prices, membership, spy_rets, sectors, dates):
        strat = ResidualOUMeanReversion(spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        row_sums = w.loc[dates[400:]].sum(axis=1)
        # OU strategy may have no qualifying stocks on some days
        assert ((row_sums.round(6) == 1.0) | (row_sums.round(6) == 0.0)).all()

    def test_no_nan_weights(self, prices, membership, spy_rets, sectors):
        strat = ResidualOUMeanReversion(spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_entry_z_controls_selectivity(self, prices, membership, spy_rets, sectors, dates):
        """Tighter entry_z (-2.0) should select fewer stocks than looser (-1.0)."""
        strat_tight = ResidualOUMeanReversion(spy_rets, sectors, entry_z=-2.0)
        strat_loose = ResidualOUMeanReversion(spy_rets, sectors, entry_z=-1.0)
        w_tight = strat_tight.generate_weights(prices, membership)
        w_loose = strat_loose.generate_weights(prices, membership)
        late = dates[500:]
        # Tight threshold selects fewer non-zero positions
        n_tight = (w_tight.loc[late] > 0).sum(axis=1).mean()
        n_loose = (w_loose.loc[late] > 0).sum(axis=1).mean()
        assert n_tight <= n_loose

    def test_weekly_structure(self, prices, membership, spy_rets, sectors):
        """Weights held for 5-day blocks."""
        strat = ResidualOUMeanReversion(spy_rets, sectors, rebalance_period=5)
        w = strat.generate_weights(prices, membership)
        for start in [20, 25, 30]:
            block = w.iloc[start : start + 5]
            if len(block) == 5:
                first = block.iloc[0]
                assert (block - first).abs().max().max() < 1e-10
