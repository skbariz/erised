"""Tests for H180-H184 earnings drift and fundamental event strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.cross_sectional import (
    CashEarningsDrift,
    EarningsMomentumComposite,
    EarningsTrendLong,
    FilingDateGapContinuation,
    RevenueEarningsDoubleGrowth,
)

N_DATES = 1400
TICKERS = ["GROW", "FLAT", "SHRINK", "CASH", "ACCRUAL"]


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
            "GROW": "Tech",
            "FLAT": "Tech",
            "SHRINK": "Finance",
            "CASH": "Finance",
            "ACCRUAL": "Healthcare",
        },
        name="sector",
    )


@pytest.fixture
def prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    np.random.seed(42)
    data = {}
    for t in TICKERS:
        ret = np.random.randn(N_DATES) * 0.01 + 0.0003
        data[t] = 100.0 * np.cumprod(1 + ret)
    return pd.DataFrame(data, index=dates)


def _make_fund_panel(dates, values_by_ticker, update_every=63):
    """Create quarterly-updating fundamental panel."""
    df = pd.DataFrame(np.nan, index=dates, columns=TICKERS)
    for t, vals in values_by_ticker.items():
        for i, v in enumerate(vals):
            idx = i * update_every
            if idx < len(dates):
                df.loc[dates[idx], t] = v
    return df.ffill()


@pytest.fixture
def net_income(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """GROW: growing NI, FLAT: flat NI, SHRINK: declining NI."""
    ni_values = {
        "GROW": [100 + i * 15 for i in range(N_DATES // 63 + 1)],
        "FLAT": [100] * (N_DATES // 63 + 1),
        "SHRINK": [200 - i * 15 for i in range(N_DATES // 63 + 1)],
        "CASH": [80 + i * 10 for i in range(N_DATES // 63 + 1)],
        "ACCRUAL": [60 + i * 8 for i in range(N_DATES // 63 + 1)],
    }
    return _make_fund_panel(dates, ni_values)


@pytest.fixture
def revenue(dates: pd.DatetimeIndex) -> pd.DataFrame:
    rev_values = {
        "GROW": [500 + i * 30 for i in range(N_DATES // 63 + 1)],
        "FLAT": [500] * (N_DATES // 63 + 1),
        "SHRINK": [600 - i * 20 for i in range(N_DATES // 63 + 1)],
        "CASH": [400 + i * 20 for i in range(N_DATES // 63 + 1)],
        "ACCRUAL": [300 + i * 15 for i in range(N_DATES // 63 + 1)],
    }
    return _make_fund_panel(dates, rev_values)


@pytest.fixture
def operating_cf(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """CASH: high OCF (cash-backed), ACCRUAL: low OCF (accrual-heavy)."""
    ocf_values = {
        "GROW": [90 + i * 12 for i in range(N_DATES // 63 + 1)],
        "FLAT": [90] * (N_DATES // 63 + 1),
        "SHRINK": [150 - i * 12 for i in range(N_DATES // 63 + 1)],
        "CASH": [75 + i * 10 for i in range(N_DATES // 63 + 1)],   # OCF ~ NI: cash quality
        "ACCRUAL": [10 + i * 1 for i in range(N_DATES // 63 + 1)], # OCF << NI: accrual-heavy
    }
    return _make_fund_panel(dates, ocf_values)


# ---------------------------------------------------------------------------
# EarningsTrendLong (H180)
# ---------------------------------------------------------------------------


class TestEarningsTrendLong:
    def test_weights_sum_to_one(self, prices, membership, net_income, sectors, dates):
        strat = EarningsTrendLong(net_income, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_growers_preferred(self, prices, membership, net_income, sectors, dates):
        """GROW (steadily rising NI) should have higher weight than SHRINK."""
        strat = EarningsTrendLong(net_income, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]]
        assert late["GROW"].mean() >= late["SHRINK"].mean()

    def test_no_nan_weights(self, prices, membership, net_income, sectors):
        strat = EarningsTrendLong(net_income, sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_early_folds_sparse(self, prices, membership, net_income, sectors, dates):
        """Before EDGAR coverage (~252 days), weights can be zero."""
        strat = EarningsTrendLong(net_income, sectors)
        w = strat.generate_weights(prices, membership)
        # At the very beginning (before yoy_window), all weights should be 0
        assert (w.iloc[:100].sum(axis=1) == 0.0).all()


# ---------------------------------------------------------------------------
# RevenueEarningsDoubleGrowth (H181)
# ---------------------------------------------------------------------------


class TestRevenueEarningsDoubleGrowth:
    def test_weights_sum_to_one_or_zero(self, prices, membership, net_income, revenue, sectors, dates):
        strat = RevenueEarningsDoubleGrowth(net_income, revenue, sectors)
        w = strat.generate_weights(prices, membership)
        row_sums = w.loc[dates[700:]].sum(axis=1)
        # Either fully invested or zero (if no double-positive stocks in a sector)
        assert ((row_sums.round(6) == 1.0) | (row_sums.round(6) == 0.0)).all()

    def test_double_growers_selected(self, prices, membership, net_income, revenue, sectors, dates):
        """GROW has both NI and revenue growing; SHRINK has both declining."""
        strat = RevenueEarningsDoubleGrowth(net_income, revenue, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]]
        assert late["GROW"].mean() >= late["SHRINK"].mean()

    def test_no_nan_weights(self, prices, membership, net_income, revenue, sectors):
        strat = RevenueEarningsDoubleGrowth(net_income, revenue, sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()


# ---------------------------------------------------------------------------
# FilingDateGapContinuation (H182)
# ---------------------------------------------------------------------------


class TestFilingDateGapContinuation:
    def test_weights_sum_to_one_or_zero(self, prices, membership, net_income, dates):
        strat = FilingDateGapContinuation(net_income)
        w = strat.generate_weights(prices, membership)
        row_sums = w.sum(axis=1)
        assert ((row_sums.round(6) == 1.0) | (row_sums.round(6) == 0.0)).all()

    def test_no_nan_weights(self, prices, membership, net_income):
        strat = FilingDateGapContinuation(net_income)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_positions_held_after_filing(self, membership, net_income, dates):
        """Once entered, a stock should stay for up to hold_days."""
        np.random.seed(7)
        n = N_DATES
        # Build prices where GROW always gaps up on filing dates
        base_rets = np.random.randn(n) * 0.005
        data = {t: 100.0 * np.cumprod(1 + base_rets) for t in TICKERS}
        prices_df = pd.DataFrame(data, index=dates)
        # Force a +5% return on first NI update date for GROW
        first_update = dates[63]
        idx = dates.get_loc(first_update)
        prices_df.iloc[idx, prices_df.columns.get_loc("GROW")] = (
            prices_df.iloc[idx - 1, prices_df.columns.get_loc("GROW")] * 1.05
        )
        strat = FilingDateGapContinuation(net_income, gap_threshold=0.02, hold_days=20)
        w = strat.generate_weights(prices_df, membership)
        # GROW should appear in portfolio shortly after the filing date
        post_filing = w.loc[dates[64 : 64 + 20]]
        assert post_filing["GROW"].sum() > 0


# ---------------------------------------------------------------------------
# CashEarningsDrift (H183)
# ---------------------------------------------------------------------------


class TestCashEarningsDrift:
    def test_weights_sum_to_one_or_zero(
        self, prices, membership, net_income, operating_cf, sectors, dates
    ):
        strat = CashEarningsDrift(net_income, operating_cf, sectors)
        w = strat.generate_weights(prices, membership)
        row_sums = w.loc[dates[700:]].sum(axis=1)
        assert ((row_sums.round(6) == 1.0) | (row_sums.round(6) == 0.0)).all()

    def test_cash_quality_filter(
        self, prices, membership, net_income, operating_cf, sectors, dates
    ):
        """ACCRUAL has low OCF/NI ratio and should be excluded."""
        strat = CashEarningsDrift(net_income, operating_cf, sectors, cash_ratio_min=0.5)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]]
        # CASH has OCF ~ NI (cash-backed), ACCRUAL has OCF << NI
        assert late["CASH"].mean() >= late["ACCRUAL"].mean()

    def test_no_nan_weights(self, prices, membership, net_income, operating_cf, sectors):
        strat = CashEarningsDrift(net_income, operating_cf, sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()


# ---------------------------------------------------------------------------
# EarningsMomentumComposite (H184)
# ---------------------------------------------------------------------------


class TestEarningsMomentumComposite:
    def test_weights_sum_to_one(
        self, prices, membership, net_income, sectors, dates
    ):
        strat = EarningsMomentumComposite(net_income, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_no_nan_weights(self, prices, membership, net_income, sectors):
        strat = EarningsMomentumComposite(net_income, sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_uses_both_signals(self, prices, membership, net_income, sectors, dates):
        """Composite should differ from pure-momentum weights (NI contributes)."""
        strat_em = EarningsMomentumComposite(net_income, sectors)
        from quant.strategies.cross_sectional import SectorNeutralMomentum

        strat_m = SectorNeutralMomentum(sectors)
        w_em = strat_em.generate_weights(prices, membership)
        w_m = strat_m.generate_weights(prices, membership)
        late = dates[700:]
        diff = (w_em.loc[late] - w_m.loc[late]).abs().mean().mean()
        assert diff > 1e-6
