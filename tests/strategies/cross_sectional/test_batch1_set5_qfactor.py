"""Tests for H351-H355: Set 5 Batch 1 q-factor strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    QFactorComposite,
    QInvestmentRank,
    QInvestProfCombo,
    QResidualMomentum,
    QROELevel,
)

N_DATES = 1200
N_TICKERS = 6
TICKERS = ["HIGH_INV", "LOW_INV", "HIGH_ROE", "LOW_ROE", "MOM_HI", "MOM_LO"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    np.random.seed(42)
    data = {
        "HIGH_INV": 100 + np.cumsum(np.random.randn(N_DATES) * 0.5),
        "LOW_INV": 100 + np.cumsum(np.random.randn(N_DATES) * 0.5),
        "HIGH_ROE": 100 + np.linspace(0, 80, N_DATES),
        "LOW_ROE": 100 + np.linspace(0, 5, N_DATES),
        "MOM_HI": 100 + np.linspace(0, 60, N_DATES),
        "MOM_LO": 100 + np.linspace(0, 2, N_DATES),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def total_assets_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    # HIGH_INV grows fast; LOW_INV stays flat
    base = {
        "HIGH_INV": 500.0,
        "LOW_INV": 500.0,
        "HIGH_ROE": 300.0,
        "LOW_ROE": 300.0,
        "MOM_HI": 400.0,
        "MOM_LO": 400.0,
    }
    growth = {
        "HIGH_INV": 0.001,
        "LOW_INV": 0.0001,
        "HIGH_ROE": 0.0002,
        "LOW_ROE": 0.0002,
        "MOM_HI": 0.0003,
        "MOM_LO": 0.0003,
    }
    data = {t: base[t] * (1 + growth[t]) ** np.arange(N_DATES) for t in TICKERS}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def net_income_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {
        "HIGH_INV": np.full(N_DATES, 20.0),
        "LOW_INV": np.full(N_DATES, 15.0),
        "HIGH_ROE": np.full(N_DATES, 90.0),
        "LOW_ROE": np.full(N_DATES, 5.0),
        "MOM_HI": np.full(N_DATES, 30.0),
        "MOM_LO": np.full(N_DATES, 10.0),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def book_equity_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {
        "HIGH_INV": np.full(N_DATES, 200.0),
        "LOW_INV": np.full(N_DATES, 200.0),
        "HIGH_ROE": np.full(N_DATES, 100.0),
        "LOW_ROE": np.full(N_DATES, 500.0),
        "MOM_HI": np.full(N_DATES, 150.0),
        "MOM_LO": np.full(N_DATES, 150.0),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def gross_profit_panel(dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {t: np.full(N_DATES, 50.0) for t in TICKERS}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sectors() -> pd.Series:
    return pd.Series(
        {
            "HIGH_INV": "Tech",
            "LOW_INV": "Tech",
            "HIGH_ROE": "Finance",
            "LOW_ROE": "Finance",
            "MOM_HI": "Health",
            "MOM_LO": "Health",
        }
    )


def _weights_valid(w: pd.DataFrame) -> None:
    assert isinstance(w, pd.DataFrame)
    assert (w >= 0).all().all(), "negative weights found"
    assert (w.sum(axis=1) <= 1.001).all(), "weights sum > 1"


# ---------------------------------------------------------------------------
# H351 QInvestmentRank
# ---------------------------------------------------------------------------


def test_q_investment_rank_shape(dates, membership, prices, total_assets_panel, sectors):
    strat = QInvestmentRank(total_assets=total_assets_panel, sectors=sectors)
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_q_investment_rank_prefers_low_inv(dates, membership, prices, total_assets_panel, sectors):
    strat = QInvestmentRank(total_assets=total_assets_panel, sectors=sectors)
    w = strat.generate_weights(prices, membership)
    late = w.iloc[600:]
    assert late["LOW_INV"].mean() >= late["HIGH_INV"].mean(), "should prefer low-investment stocks"


# ---------------------------------------------------------------------------
# H352 QROELevel
# ---------------------------------------------------------------------------


def test_q_roe_level_shape(dates, membership, prices, net_income_panel, book_equity_panel, sectors):
    strat = QROELevel(net_income=net_income_panel, book_equity=book_equity_panel, sectors=sectors)
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_q_roe_level_prefers_high_roe(
    dates, membership, prices, net_income_panel, book_equity_panel, sectors
):
    strat = QROELevel(net_income=net_income_panel, book_equity=book_equity_panel, sectors=sectors)
    w = strat.generate_weights(prices, membership)
    late = w.iloc[600:]
    assert late["HIGH_ROE"].mean() >= late["LOW_ROE"].mean(), "should prefer high-ROE stocks"


# ---------------------------------------------------------------------------
# H353 QInvestProfCombo
# ---------------------------------------------------------------------------


def test_q_invest_prof_combo_shape(
    dates, membership, prices, total_assets_panel, net_income_panel, book_equity_panel, sectors
):
    strat = QInvestProfCombo(
        total_assets=total_assets_panel,
        net_income=net_income_panel,
        book_equity=book_equity_panel,
        sectors=sectors,
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


# ---------------------------------------------------------------------------
# H354 QResidualMomentum
# ---------------------------------------------------------------------------


def test_q_residual_momentum_shape(
    dates, membership, prices, total_assets_panel, net_income_panel, book_equity_panel, sectors
):
    strat = QResidualMomentum(
        total_assets=total_assets_panel,
        net_income=net_income_panel,
        book_equity=book_equity_panel,
        sectors=sectors,
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_q_residual_momentum_no_lookahead(
    dates, membership, prices, total_assets_panel, net_income_panel, book_equity_panel, sectors
):
    strat = QResidualMomentum(
        total_assets=total_assets_panel,
        net_income=net_income_panel,
        book_equity=book_equity_panel,
        sectors=sectors,
    )
    w = strat.generate_weights(prices, membership)
    # Early rows (before enough history) should be all-zero
    assert (w.iloc[:20].sum(axis=1) == 0).all(), "weights before signal warm-up should be zero"


# ---------------------------------------------------------------------------
# H355 QFactorComposite
# ---------------------------------------------------------------------------


def test_q_factor_composite_shape(
    dates,
    membership,
    prices,
    total_assets_panel,
    net_income_panel,
    book_equity_panel,
    gross_profit_panel,
    sectors,
):
    strat = QFactorComposite(
        total_assets=total_assets_panel,
        net_income=net_income_panel,
        book_equity=book_equity_panel,
        gross_profit=gross_profit_panel,
        sectors=sectors,
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)
