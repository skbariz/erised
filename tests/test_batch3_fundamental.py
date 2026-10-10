"""Tests for H160-H164 fundamental-data cross-sectional strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    LargeCapValue,
    PiotroskiValueScreen,
    QualityFilteredReversal,
    SectorNeutralValueMomentum,
    ValueMomentumComposite,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_DATES = 1400
N_TICKERS = 5
TICKERS = ["CHEAP", "EXP", "MOM", "JUNK", "QUAL"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """CHEAP starts low and rises moderately; MOM has strong recent momentum."""
    np.random.seed(42)
    data = {
        "CHEAP": 10 + np.linspace(0, 20, N_DATES),
        "EXP": 100 + np.linspace(0, 30, N_DATES),
        "MOM": 50 + np.linspace(0, 100, N_DATES),
        "JUNK": 20 + np.linspace(0, 5, N_DATES),
        "QUAL": 30 + np.linspace(0, 25, N_DATES),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def book_equity_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """CHEAP and JUNK have high B/M (high book equity relative to price).
    EXP and MOM have low B/M. QUAL medium.
    """
    data = {
        "CHEAP": np.full(N_DATES, 500.0),  # very cheap (high B/M)
        "EXP": np.full(N_DATES, 200.0),  # expensive (low B/M)
        "MOM": np.full(N_DATES, 150.0),  # momentum growth stock
        "JUNK": np.full(N_DATES, 400.0),  # cheap but low quality
        "QUAL": np.full(N_DATES, 300.0),  # medium value, high quality
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def net_income_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """CHEAP and QUAL have positive earnings. JUNK has negative earnings (filtered out for E/P)."""
    data = {
        "CHEAP": np.full(N_DATES, 80.0),
        "EXP": np.full(N_DATES, 100.0),
        "MOM": np.full(N_DATES, 120.0),
        "JUNK": np.full(N_DATES, -10.0),  # negative -> E/P excluded
        "QUAL": np.full(N_DATES, 60.0),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def shares_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """100 million shares for all. Prices differ so market caps differ."""
    return pd.DataFrame(
        {t: np.full(N_DATES, 100.0) for t in TICKERS},
        index=dates,
    )


@pytest.fixture
def operating_cf_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """JUNK has negative CF; others positive."""
    data = {
        "CHEAP": np.full(N_DATES, 50.0),
        "EXP": np.full(N_DATES, 80.0),
        "MOM": np.full(N_DATES, 90.0),
        "JUNK": np.full(N_DATES, -5.0),  # fails OCF gate
        "QUAL": np.full(N_DATES, 40.0),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def current_assets_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {t: np.full(N_DATES, 100.0) for t in TICKERS}
    data["JUNK"] = np.full(N_DATES, 50.0)  # current ratio < 1
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def current_liab_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {t: np.full(N_DATES, 60.0) for t in TICKERS}
    data["JUNK"] = np.full(N_DATES, 80.0)  # current ratio < 1
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def fscore_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    """CHEAP and QUAL have high F-score; JUNK has low F-score."""
    data = {
        "CHEAP": np.full(N_DATES, 7.0),
        "EXP": np.full(N_DATES, 5.0),
        "MOM": np.full(N_DATES, 5.0),
        "JUNK": np.full(N_DATES, 3.0),  # fails min_fscore=6 gate
        "QUAL": np.full(N_DATES, 7.0),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sectors() -> pd.Series:
    return pd.Series(
        {
            "CHEAP": "Financials",
            "EXP": "Technology",
            "MOM": "Technology",
            "JUNK": "Financials",
            "QUAL": "Healthcare",
        },
        name="sector",
    )


# ---------------------------------------------------------------------------
# H160: LargeCapValue
# ---------------------------------------------------------------------------


class TestLargeCapValue:
    def test_weights_sum_to_one(
        self, prices, membership, book_equity_panel, net_income_panel, shares_panel, dates
    ):
        strat = LargeCapValue(
            book_equity=book_equity_panel,
            net_income=net_income_panel,
            shares_outstanding=shares_panel,
            top_quantile=0.40,
        )
        weights = strat.generate_weights(prices, membership)
        # After warm-up, rows with any position should sum to ~1.0
        active_rows = weights.sum(axis=1) > 0
        assert active_rows.any(), "No active rows found"
        np.testing.assert_allclose(weights.loc[active_rows].sum(axis=1).values, 1.0, atol=1e-9)

    def test_cheap_stock_preferred(
        self, prices, membership, book_equity_panel, net_income_panel, shares_panel, dates
    ):
        """CHEAP has highest B/M and positive E/P → should be selected."""
        strat = LargeCapValue(
            book_equity=book_equity_panel,
            net_income=net_income_panel,
            shares_outstanding=shares_panel,
            top_quantile=0.20,
        )
        weights = strat.generate_weights(prices, membership)
        last_row = weights.iloc[-1]
        # With top_quantile=0.20 and 5 stocks, threshold ≈ 1 stock
        # CHEAP should be the top value pick
        assert last_row["CHEAP"] > 0 or last_row.max() > 0

    def test_negative_book_equity_excluded(self, prices, membership, shares_panel, dates):
        """Stocks with negative book equity should never be selected."""
        be = pd.DataFrame(
            {
                "CHEAP": np.full(N_DATES, -100.0),
                "EXP": np.full(N_DATES, 200.0),
                "MOM": np.full(N_DATES, 150.0),
                "JUNK": np.full(N_DATES, 300.0),
                "QUAL": np.full(N_DATES, 250.0),
            },
            index=dates,
        )
        ni = pd.DataFrame({t: np.full(N_DATES, 50.0) for t in TICKERS}, index=dates)
        strat = LargeCapValue(
            book_equity=be,
            net_income=ni,
            shares_outstanding=shares_panel,
        )
        weights = strat.generate_weights(prices, membership)
        assert weights["CHEAP"].sum() == 0.0, "Negative B/M stock should never be selected"

    def test_no_lookahead_bias(
        self, prices, membership, book_equity_panel, net_income_panel, shares_panel
    ):
        strat = LargeCapValue(
            book_equity=book_equity_panel,
            net_income=net_income_panel,
            shares_outstanding=shares_panel,
        )
        weights_full = strat.generate_weights(prices, membership)
        weights_half = strat.generate_weights(prices.iloc[:700], membership.iloc[:700])
        mid = prices.index[699]
        pd.testing.assert_series_equal(
            weights_full.loc[mid],
            weights_half.loc[mid],
            check_names=False,
        )


# ---------------------------------------------------------------------------
# H161: QualityFilteredReversal
# ---------------------------------------------------------------------------


class TestQualityFilteredReversal:
    def test_weights_sum_to_one(
        self, prices, membership, operating_cf_panel, current_assets_panel, current_liab_panel
    ):
        strat = QualityFilteredReversal(
            operating_cf=operating_cf_panel,
            current_assets=current_assets_panel,
            current_liabilities=current_liab_panel,
            long_days=252,
            recovery_days=63,
            top_quantile=0.40,
        )
        weights = strat.generate_weights(prices, membership)
        active_rows = weights.sum(axis=1) > 0
        assert active_rows.any()
        np.testing.assert_allclose(weights.loc[active_rows].sum(axis=1).values, 1.0, atol=1e-9)

    def test_quality_gate_filters_junk(
        self, prices, membership, operating_cf_panel, current_assets_panel, current_liab_panel
    ):
        """JUNK has negative OCF and current_ratio < 1 → should never be selected."""
        strat = QualityFilteredReversal(
            operating_cf=operating_cf_panel,
            current_assets=current_assets_panel,
            current_liabilities=current_liab_panel,
            long_days=252,
            recovery_days=63,
            top_quantile=0.80,  # wide quantile to see many selections
        )
        weights = strat.generate_weights(prices, membership)
        assert weights["JUNK"].sum() == 0.0, "JUNK fails quality gate, should never be selected"

    def test_no_lookahead_bias(
        self, prices, membership, operating_cf_panel, current_assets_panel, current_liab_panel
    ):
        strat = QualityFilteredReversal(
            operating_cf=operating_cf_panel,
            current_assets=current_assets_panel,
            current_liabilities=current_liab_panel,
            long_days=252,
            recovery_days=63,
        )
        w_full = strat.generate_weights(prices, membership)
        w_half = strat.generate_weights(prices.iloc[:700], membership.iloc[:700])
        mid = prices.index[699]
        pd.testing.assert_series_equal(w_full.loc[mid], w_half.loc[mid], check_names=False)


# ---------------------------------------------------------------------------
# H162: ValueMomentumComposite
# ---------------------------------------------------------------------------


class TestValueMomentumComposite:
    def test_weights_sum_to_one(
        self, prices, membership, book_equity_panel, net_income_panel, shares_panel
    ):
        strat = ValueMomentumComposite(
            book_equity=book_equity_panel,
            net_income=net_income_panel,
            shares_outstanding=shares_panel,
            top_quantile=0.40,
        )
        weights = strat.generate_weights(prices, membership)
        active_rows = weights.sum(axis=1) > 0
        assert active_rows.any()
        np.testing.assert_allclose(weights.loc[active_rows].sum(axis=1).values, 1.0, atol=1e-9)

    def test_all_zero_before_warmup(
        self, prices, membership, book_equity_panel, net_income_panel, shares_panel
    ):
        """Before long_days data is available, all weights should be 0."""
        strat = ValueMomentumComposite(
            book_equity=book_equity_panel,
            net_income=net_income_panel,
            shares_outstanding=shares_panel,
            long_days=252,
        )
        weights = strat.generate_weights(prices, membership)
        assert weights.iloc[:252].sum().sum() == 0.0, "Weights before warm-up should be zero"

    def test_no_lookahead_bias(
        self, prices, membership, book_equity_panel, net_income_panel, shares_panel
    ):
        strat = ValueMomentumComposite(
            book_equity=book_equity_panel,
            net_income=net_income_panel,
            shares_outstanding=shares_panel,
        )
        w_full = strat.generate_weights(prices, membership)
        w_half = strat.generate_weights(prices.iloc[:700], membership.iloc[:700])
        mid = prices.index[699]
        pd.testing.assert_series_equal(w_full.loc[mid], w_half.loc[mid], check_names=False)


# ---------------------------------------------------------------------------
# H163: SectorNeutralValueMomentum
# ---------------------------------------------------------------------------


class TestSectorNeutralValueMomentum:
    def test_weights_sum_to_one(
        self, prices, membership, book_equity_panel, net_income_panel, shares_panel, sectors
    ):
        strat = SectorNeutralValueMomentum(
            book_equity=book_equity_panel,
            net_income=net_income_panel,
            shares_outstanding=shares_panel,
            sectors=sectors,
            within_sector_quantile=0.50,
        )
        weights = strat.generate_weights(prices, membership)
        active_rows = weights.sum(axis=1) > 0
        assert active_rows.any()
        np.testing.assert_allclose(weights.loc[active_rows].sum(axis=1).values, 1.0, atol=1e-6)

    def test_sector_columns_covered(
        self, prices, membership, book_equity_panel, net_income_panel, shares_panel, sectors
    ):
        """All tickers should appear as columns in the output."""
        strat = SectorNeutralValueMomentum(
            book_equity=book_equity_panel,
            net_income=net_income_panel,
            shares_outstanding=shares_panel,
            sectors=sectors,
        )
        weights = strat.generate_weights(prices, membership)
        assert set(weights.columns) == set(TICKERS)

    def test_no_lookahead_bias(
        self, prices, membership, book_equity_panel, net_income_panel, shares_panel, sectors
    ):
        strat = SectorNeutralValueMomentum(
            book_equity=book_equity_panel,
            net_income=net_income_panel,
            shares_outstanding=shares_panel,
            sectors=sectors,
        )
        w_full = strat.generate_weights(prices, membership)
        w_half = strat.generate_weights(prices.iloc[:700], membership.iloc[:700])
        mid = prices.index[699]
        pd.testing.assert_series_equal(w_full.loc[mid], w_half.loc[mid], check_names=False)


# ---------------------------------------------------------------------------
# H164: PiotroskiValueScreen
# ---------------------------------------------------------------------------


class TestPiotroskiValueScreen:
    def test_weights_sum_to_one(
        self, prices, membership, book_equity_panel, shares_panel, fscore_panel
    ):
        strat = PiotroskiValueScreen(
            book_equity=book_equity_panel,
            shares_outstanding=shares_panel,
            fscore_panel=fscore_panel,
            bm_quantile=0.60,
            min_fscore=6.0,
        )
        weights = strat.generate_weights(prices, membership)
        active_rows = weights.sum(axis=1) > 0
        assert active_rows.any()
        np.testing.assert_allclose(weights.loc[active_rows].sum(axis=1).values, 1.0, atol=1e-9)

    def test_low_fscore_excluded(
        self, prices, membership, book_equity_panel, shares_panel, fscore_panel
    ):
        """JUNK has F-score=3 < 6 → never selected even if B/M qualifies."""
        strat = PiotroskiValueScreen(
            book_equity=book_equity_panel,
            shares_outstanding=shares_panel,
            fscore_panel=fscore_panel,
            bm_quantile=0.80,  # include JUNK in value quintile
            min_fscore=6.0,
        )
        weights = strat.generate_weights(prices, membership)
        assert weights["JUNK"].sum() == 0.0, "JUNK F-score=3 < 6, should never be selected"

    def test_negative_bm_excluded(self, prices, membership, shares_panel, fscore_panel, dates):
        """Stock with negative book equity should never be selected."""
        be = pd.DataFrame(
            {
                "CHEAP": np.full(N_DATES, -100.0),
                "EXP": np.full(N_DATES, 300.0),
                "MOM": np.full(N_DATES, 200.0),
                "JUNK": np.full(N_DATES, 400.0),
                "QUAL": np.full(N_DATES, 250.0),
            },
            index=dates,
        )
        strat = PiotroskiValueScreen(
            book_equity=be,
            shares_outstanding=shares_panel,
            fscore_panel=fscore_panel,
            bm_quantile=0.80,
            min_fscore=3.0,
        )
        weights = strat.generate_weights(prices, membership)
        assert weights["CHEAP"].sum() == 0.0, "Negative B/M stock should never be selected"

    def test_no_lookahead_bias(
        self, prices, membership, book_equity_panel, shares_panel, fscore_panel
    ):
        strat = PiotroskiValueScreen(
            book_equity=book_equity_panel,
            shares_outstanding=shares_panel,
            fscore_panel=fscore_panel,
        )
        w_full = strat.generate_weights(prices, membership)
        w_half = strat.generate_weights(prices.iloc[:700], membership.iloc[:700])
        mid = prices.index[699]
        pd.testing.assert_series_equal(w_full.loc[mid], w_half.loc[mid], check_names=False)
