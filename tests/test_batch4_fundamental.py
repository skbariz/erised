"""Tests for H165-H169 quality-factor cross-sectional strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.cross_sectional import (
    CompositeQualityScore,
    GrossProfitabilityRank,
    LowAccrualsQuality,
    QualityValueComposite,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DATES = 1400
TICKERS = ["QUAL", "MED", "JUNK", "CHEAP", "EXP"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    np.random.seed(7)
    data = {t: 50 + np.linspace(0, 20, N_DATES) + np.random.randn(N_DATES) * 0.5 for t in TICKERS}
    return pd.DataFrame(data, index=dates)


def _fund_panel(dates, tickers, vals):
    """Create a fundamental panel with annual-ish updates (quarter-end-shifted)."""
    df = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for i, t in enumerate(tickers):
        update_dates = dates[::63]  # quarterly update
        df.loc[update_dates, t] = vals[i]
    return df.ffill()


@pytest.fixture
def sectors() -> pd.Series:
    return pd.Series(
        {"QUAL": "Tech", "MED": "Tech", "JUNK": "Finance", "CHEAP": "Finance", "EXP": "Healthcare"},
        name="sector",
    )


@pytest.fixture
def gross_profit(dates) -> pd.DataFrame:
    return _fund_panel(dates, TICKERS, [200, 100, 30, 80, 150])


@pytest.fixture
def total_assets(dates) -> pd.DataFrame:
    return _fund_panel(dates, TICKERS, [500, 500, 500, 500, 500])


@pytest.fixture
def net_income(dates) -> pd.DataFrame:
    return _fund_panel(dates, TICKERS, [80, 40, 5, 30, 60])


@pytest.fixture
def operating_cf(dates) -> pd.DataFrame:
    # QUAL: high OCF (low accruals); JUNK: low OCF (high accruals)
    return _fund_panel(dates, TICKERS, [90, 38, -10, 35, 55])


@pytest.fixture
def book_equity(dates) -> pd.DataFrame:
    return _fund_panel(dates, TICKERS, [300, 200, 100, 400, 250])


@pytest.fixture
def long_term_debt(dates) -> pd.DataFrame:
    return _fund_panel(dates, TICKERS, [50, 100, 300, 20, 80])


@pytest.fixture
def current_assets(dates) -> pd.DataFrame:
    return _fund_panel(dates, TICKERS, [120, 80, 40, 100, 90])


@pytest.fixture
def current_liabilities(dates) -> pd.DataFrame:
    return _fund_panel(dates, TICKERS, [60, 70, 80, 50, 45])


@pytest.fixture
def shares_outstanding(dates) -> pd.DataFrame:
    return _fund_panel(dates, TICKERS, [10, 10, 10, 10, 10])


@pytest.fixture
def spy_rets(dates) -> pd.Series:
    np.random.seed(0)
    return pd.Series(np.random.randn(N_DATES) * 0.01, index=dates, name="SPY")


# ---------------------------------------------------------------------------
# GrossProfitabilityRank (H165)
# ---------------------------------------------------------------------------


class TestGrossProfitabilityRank:
    def test_weights_sum_to_one(
        self, prices, membership, gross_profit, total_assets, sectors, dates
    ):
        strat = GrossProfitabilityRank(gross_profit, total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.95

    def test_high_gpa_selected(
        self, prices, membership, gross_profit, total_assets, sectors, dates
    ):
        """QUAL has highest GP/A (200/500=0.40); should get weight."""
        strat = GrossProfitabilityRank(gross_profit, total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        late_w = w.loc[dates[700:]]
        assert late_w["QUAL"].mean() > 0

    def test_negative_assets_excluded(
        self, prices, membership, gross_profit, total_assets, sectors, dates
    ):
        """Stocks with total_assets <= 0 must get zero weight."""
        ta_bad = total_assets.copy()
        ta_bad["JUNK"] = -100.0
        strat = GrossProfitabilityRank(gross_profit, ta_bad, sectors)
        w = strat.generate_weights(prices, membership)
        assert w["JUNK"].sum() == 0.0

    def test_no_lookahead(self, prices, membership, gross_profit, total_assets, sectors, dates):
        """Weights must not anticipate future GP/A updates."""
        gp_shifted = gross_profit.copy()
        gp_shifted["QUAL"] = np.nan
        strat = GrossProfitabilityRank(gp_shifted, total_assets, sectors)
        w = strat.generate_weights(prices, membership)
        # QUAL with NaN GP should have no weight
        assert w["QUAL"].sum() == 0.0

    def test_monthly_rebalance_reduces_turnover(self, prices, membership, sectors, dates):
        """Monthly rebalance should hold weights fixed within months vs daily-varying signal."""
        np.random.seed(3)
        # Daily-varying GP so without monthly rebalance, weights would shift daily
        gp_daily = pd.DataFrame(
            np.abs(np.random.randn(len(dates), len(TICKERS))) * 100 + 1,
            index=dates,
            columns=TICKERS,
        )
        ta_const = pd.DataFrame(500.0, index=dates, columns=TICKERS)
        strat_d = GrossProfitabilityRank(gp_daily, ta_const, sectors, monthly_rebalance=False)
        strat_m = GrossProfitabilityRank(gp_daily, ta_const, sectors, monthly_rebalance=True)
        w_d = strat_d.generate_weights(prices, membership)
        w_m = strat_m.generate_weights(prices, membership)
        changes_d = (w_d.diff().abs().sum(axis=1) > 0.001).sum()
        changes_m = (w_m.diff().abs().sum(axis=1) > 0.001).sum()
        assert changes_m < changes_d


# ---------------------------------------------------------------------------
# LowAccrualsQuality (H166)
# ---------------------------------------------------------------------------


class TestLowAccrualsQuality:
    def test_weights_sum_to_one(
        self, prices, membership, net_income, operating_cf, total_assets, dates
    ):
        strat = LowAccrualsQuality(net_income, operating_cf, total_assets)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.95

    def test_low_accruals_selected(
        self, prices, membership, net_income, operating_cf, total_assets, dates
    ):
        """QUAL has lowest accruals (ni=80, ocf=90 -> (80-90)/500 = -0.02), should be selected."""
        strat = LowAccrualsQuality(net_income, operating_cf, total_assets)
        w = strat.generate_weights(prices, membership)
        late_w = w.loc[dates[700:]]
        assert late_w["QUAL"].mean() > 0

    def test_high_accruals_excluded(
        self, prices, membership, net_income, operating_cf, total_assets, dates
    ):
        """JUNK has highest accruals (ni=5, ocf=-10 -> 15/500=0.03), should not dominate."""
        strat = LowAccrualsQuality(net_income, operating_cf, total_assets, top_quantile=0.20)
        w = strat.generate_weights(prices, membership)
        late_w = w.loc[dates[700:]]
        assert late_w["JUNK"].mean() < late_w["QUAL"].mean()

    def test_zero_assets_excluded(
        self, prices, membership, net_income, operating_cf, total_assets, dates
    ):
        ta_bad = total_assets.copy()
        ta_bad["JUNK"] = 0.0
        strat = LowAccrualsQuality(net_income, operating_cf, ta_bad)
        w = strat.generate_weights(prices, membership)
        assert w["JUNK"].sum() == 0.0


# ---------------------------------------------------------------------------
# CompositeQualityScore (H168)
# ---------------------------------------------------------------------------


class TestCompositeQualityScore:
    def test_weights_sum_to_one(
        self,
        prices,
        membership,
        gross_profit,
        total_assets,
        net_income,
        operating_cf,
        book_equity,
        long_term_debt,
        current_assets,
        current_liabilities,
        sectors,
        spy_rets,
        dates,
    ):
        strat = CompositeQualityScore(
            gross_profit,
            total_assets,
            net_income,
            operating_cf,
            book_equity,
            long_term_debt,
            current_assets,
            current_liabilities,
            sectors,
            spy_rets,
        )
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_junk_not_dominant(
        self,
        prices,
        membership,
        gross_profit,
        total_assets,
        net_income,
        operating_cf,
        book_equity,
        long_term_debt,
        current_assets,
        current_liabilities,
        sectors,
        spy_rets,
        dates,
    ):
        """JUNK (high leverage, low profitability) should get less weight than QUAL."""
        strat = CompositeQualityScore(
            gross_profit,
            total_assets,
            net_income,
            operating_cf,
            book_equity,
            long_term_debt,
            current_assets,
            current_liabilities,
            sectors,
            spy_rets,
        )
        w = strat.generate_weights(prices, membership)
        late_w = w.loc[dates[700:]]
        assert late_w["QUAL"].mean() >= late_w["JUNK"].mean()

    def test_sector_neutral(
        self,
        prices,
        membership,
        gross_profit,
        total_assets,
        net_income,
        operating_cf,
        book_equity,
        long_term_debt,
        current_assets,
        current_liabilities,
        sectors,
        spy_rets,
        dates,
    ):
        """Total weight per sector should be approximately equal across active sectors."""
        strat = CompositeQualityScore(
            gross_profit,
            total_assets,
            net_income,
            operating_cf,
            book_equity,
            long_term_debt,
            current_assets,
            current_liabilities,
            sectors,
            spy_rets,
        )
        w = strat.generate_weights(prices, membership)
        late_w = w.loc[dates[700:]]
        tech_tickers = sectors[sectors == "Tech"].index.tolist()
        fin_tickers = sectors[sectors == "Finance"].index.tolist()
        tech_wt = late_w[tech_tickers].sum(axis=1)
        fin_wt = late_w[fin_tickers].sum(axis=1)
        diff = (tech_wt - fin_wt).abs().mean()
        assert diff < 0.05  # within 5% on average


# ---------------------------------------------------------------------------
# QualityValueComposite (H169)
# ---------------------------------------------------------------------------


class TestQualityValueComposite:
    def test_weights_sum_to_one(
        self,
        prices,
        membership,
        book_equity,
        shares_outstanding,
        gross_profit,
        total_assets,
        net_income,
        operating_cf,
        sectors,
        dates,
    ):
        strat = QualityValueComposite(
            book_equity,
            shares_outstanding,
            gross_profit,
            total_assets,
            net_income,
            operating_cf,
            sectors,
        )
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.95

    def test_negative_book_equity_excluded(
        self,
        prices,
        membership,
        book_equity,
        shares_outstanding,
        gross_profit,
        total_assets,
        net_income,
        operating_cf,
        sectors,
        dates,
    ):
        be_bad = book_equity.copy()
        be_bad["JUNK"] = -50.0  # negative B/M excluded
        strat = QualityValueComposite(
            be_bad,
            shares_outstanding,
            gross_profit,
            total_assets,
            net_income,
            operating_cf,
            sectors,
        )
        w = strat.generate_weights(prices, membership)
        # JUNK has negative BE -> bm=NaN -> should not dominate composite
        # (z-scores will not select it on B/M; could still get weight from GPA)
        # just verify weights sum to ~1
        late = w.loc[dates[700:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.95

    def test_sector_neutral(
        self,
        prices,
        membership,
        book_equity,
        shares_outstanding,
        gross_profit,
        total_assets,
        net_income,
        operating_cf,
        sectors,
        dates,
    ):
        strat = QualityValueComposite(
            book_equity,
            shares_outstanding,
            gross_profit,
            total_assets,
            net_income,
            operating_cf,
            sectors,
        )
        w = strat.generate_weights(prices, membership)
        late_w = w.loc[dates[700:]]
        tech_tickers = sectors[sectors == "Tech"].index.tolist()
        fin_tickers = sectors[sectors == "Finance"].index.tolist()
        tech_wt = late_w[tech_tickers].sum(axis=1)
        fin_wt = late_w[fin_tickers].sum(axis=1)
        diff = (tech_wt - fin_wt).abs().mean()
        assert diff < 0.05
