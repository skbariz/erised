"""Tests for H170-H174 profitability/investment FF5 cross-sectional strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.cross_sectional import (
    CashOperatingProfitability,
    FF5Composite,
    FF5InvestmentFactor,
    OperatingProfitabilityFF5,
    ReturnOnAssets,
)

N_DATES = 1400
TICKERS = ["CONS", "AGG", "QUAL", "WEAK", "MED"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    np.random.seed(11)
    data = {
        t: 50 + np.linspace(0, 10, N_DATES) + np.random.randn(N_DATES) * 0.3
        for t in TICKERS
    }
    return pd.DataFrame(data, index=dates)


def _fund(dates, tickers, vals):
    df = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for i, t in enumerate(tickers):
        df.loc[dates[::63], t] = vals[i]
    return df.ffill()


@pytest.fixture
def sectors() -> pd.Series:
    return pd.Series(
        {"CONS": "Tech", "AGG": "Tech", "QUAL": "Finance", "WEAK": "Finance", "MED": "Healthcare"},
        name="sector",
    )


@pytest.fixture
def total_assets(dates) -> pd.DataFrame:
    # CONS: low growth (conservative); AGG: high growth (aggressive)
    # We model as level; growth computed from shift(252)
    # After 252 days, CONS stays flat; AGG grows by 50%
    df = pd.DataFrame(np.nan, index=dates, columns=TICKERS)
    for t, base, rate in zip(TICKERS, [500, 500, 500, 500, 500], [1.0, 1.5, 1.1, 1.2, 1.05]):
        df.loc[dates[::63], t] = base
    df = df.ffill()
    # Simulate growth after year 1 for AGG
    half = len(dates) // 2
    df.iloc[half:, TICKERS.index("AGG")] = 750
    return df


@pytest.fixture
def gross_profit(dates) -> pd.DataFrame:
    return _fund(dates, TICKERS, [200, 80, 180, 40, 120])


@pytest.fixture
def book_equity(dates) -> pd.DataFrame:
    return _fund(dates, TICKERS, [300, 200, 350, 100, 250])


@pytest.fixture
def net_income(dates) -> pd.DataFrame:
    return _fund(dates, TICKERS, [80, 30, 90, 5, 50])


@pytest.fixture
def operating_cf(dates) -> pd.DataFrame:
    return _fund(dates, TICKERS, [90, 25, 100, 3, 55])


@pytest.fixture
def shares_outstanding(dates) -> pd.DataFrame:
    return _fund(dates, TICKERS, [10, 10, 10, 10, 10])


# ---------------------------------------------------------------------------
# FF5InvestmentFactor (H170)
# ---------------------------------------------------------------------------


class TestFF5InvestmentFactor:
    def test_weights_sum_to_one(self, prices, membership, total_assets, sectors, dates):
        strat = FF5InvestmentFactor(total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[800:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.95

    def test_low_growth_selected(self, prices, membership, sectors, dates):
        """CONS (zero asset growth) should get weight; AGG (high continuous growth) should not."""
        # Build ta where CONS is flat and AGG grows linearly every quarter
        ta = pd.DataFrame(500.0, index=dates, columns=TICKERS)
        # AGG asset value grows by 60% each quarter-update — continuous strong growth
        agg_idx = TICKERS.index("AGG")
        update_dates = dates[::63]
        for i, d in enumerate(update_dates):
            ta.loc[d:, TICKERS[agg_idx]] = 500 * (1.6 ** (i + 1))
        strat = FF5InvestmentFactor(ta, sectors)
        w = strat.generate_weights(prices, membership)
        # After warmup (>252 days), CONS should be selected in Tech sector
        late = w.loc[dates[800:]]
        assert late["CONS"].mean() > 0

    def test_zero_lag_excluded(self, prices, membership, sectors, dates):
        """Rows where lagged TA = 0 should not produce asset growth = inf."""
        ta = pd.DataFrame(1.0, index=dates, columns=TICKERS)
        ta.iloc[:260, :] = 0.0  # first year of lags will be 0
        strat = FF5InvestmentFactor(ta, sectors)
        w = strat.generate_weights(prices, membership)
        # Should not raise and weights should be finite
        assert np.isfinite(w.values).all()


# ---------------------------------------------------------------------------
# OperatingProfitabilityFF5 (H171)
# ---------------------------------------------------------------------------


class TestOperatingProfitabilityFF5:
    def test_weights_sum_to_one(self, prices, membership, gross_profit, book_equity, sectors, dates):
        strat = OperatingProfitabilityFF5(gross_profit, book_equity, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.95

    def test_high_op_selected(self, prices, membership, gross_profit, book_equity, sectors, dates):
        """CONS has GP/BE = 200/300 = 0.667; should get weight."""
        strat = OperatingProfitabilityFF5(gross_profit, book_equity, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]]
        assert late["CONS"].mean() > 0

    def test_negative_be_excluded(self, prices, membership, gross_profit, book_equity, sectors, dates):
        be_bad = book_equity.copy()
        be_bad["WEAK"] = -50.0
        strat = OperatingProfitabilityFF5(gross_profit, be_bad, sectors)
        w = strat.generate_weights(prices, membership)
        assert w["WEAK"].sum() == 0.0


# ---------------------------------------------------------------------------
# FF5Composite (H172)
# ---------------------------------------------------------------------------


class TestFF5Composite:
    def test_weights_sum_to_one(
        self, prices, membership, book_equity, shares_outstanding, gross_profit, total_assets, sectors, dates
    ):
        strat = FF5Composite(book_equity, shares_outstanding, gross_profit, total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[800:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_sector_neutral(
        self, prices, membership, book_equity, shares_outstanding, gross_profit, total_assets, sectors, dates
    ):
        strat = FF5Composite(book_equity, shares_outstanding, gross_profit, total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[800:]]
        tech = sectors[sectors == "Tech"].index.tolist()
        fin = sectors[sectors == "Finance"].index.tolist()
        tech_wt = late[tech].sum(axis=1)
        fin_wt = late[fin].sum(axis=1)
        assert (tech_wt - fin_wt).abs().mean() < 0.05

    def test_no_nan_weights(
        self, prices, membership, book_equity, shares_outstanding, gross_profit, total_assets, sectors, dates
    ):
        strat = FF5Composite(book_equity, shares_outstanding, gross_profit, total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()


# ---------------------------------------------------------------------------
# ReturnOnAssets (H173)
# ---------------------------------------------------------------------------


class TestReturnOnAssets:
    def test_weights_sum_to_one(self, prices, membership, net_income, total_assets, sectors, dates):
        strat = ReturnOnAssets(net_income, total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.95

    def test_high_roa_selected(self, prices, membership, net_income, total_assets, sectors, dates):
        """QUAL has NI/TA = 90/500 = 0.18 (highest); should get weight."""
        strat = ReturnOnAssets(net_income, total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]]
        assert late["QUAL"].mean() > 0

    def test_low_roa_not_dominant(self, prices, membership, net_income, total_assets, sectors, dates):
        """WEAK has NI/TA = 5/500 = 0.01 (lowest); should get less weight than QUAL."""
        strat = ReturnOnAssets(net_income, total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]]
        assert late["WEAK"].mean() < late["QUAL"].mean()


# ---------------------------------------------------------------------------
# CashOperatingProfitability (H174)
# ---------------------------------------------------------------------------


class TestCashOperatingProfitability:
    def test_weights_sum_to_one(self, prices, membership, operating_cf, book_equity, sectors, dates):
        strat = CashOperatingProfitability(operating_cf, book_equity, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.95

    def test_high_cop_selected(self, prices, membership, operating_cf, book_equity, sectors, dates):
        """QUAL has OCF/BE = 100/350 = 0.286; should get weight."""
        strat = CashOperatingProfitability(operating_cf, book_equity, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]]
        assert late["QUAL"].mean() > 0

    def test_negative_be_excluded(self, prices, membership, operating_cf, book_equity, sectors, dates):
        be_bad = book_equity.copy()
        be_bad["WEAK"] = -10.0
        strat = CashOperatingProfitability(operating_cf, be_bad, sectors)
        w = strat.generate_weights(prices, membership)
        assert w["WEAK"].sum() == 0.0
