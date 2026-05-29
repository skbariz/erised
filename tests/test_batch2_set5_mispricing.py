"""Tests for H356-H360: Set 5 Batch 2 mispricing strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.cross_sectional import (
    ManagementMispricing,
    MispricingAvoidFilter,
    MispricingCombo,
    MispricingQualityComposite,
    PerformanceMispricing,
)

N_DATES = 1200
N_TICKERS = 6
TICKERS = ["CLEAN", "DIRTY", "HIGH_ROA", "LOW_ROA", "HIGH_GP", "LOW_GP"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    np.random.seed(7)
    data = {t: 100 + np.cumsum(np.random.randn(N_DATES) * 0.5) for t in TICKERS}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def total_assets(dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {t: np.full(N_DATES, 500.0) for t in TICKERS}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def net_income(dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {
        "CLEAN": np.full(N_DATES, 40.0),
        "DIRTY": np.full(N_DATES, 5.0),
        "HIGH_ROA": np.full(N_DATES, 80.0),
        "LOW_ROA": np.full(N_DATES, 2.0),
        "HIGH_GP": np.full(N_DATES, 30.0),
        "LOW_GP": np.full(N_DATES, 5.0),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def operating_cf(dates: pd.DatetimeIndex) -> pd.DataFrame:
    # CLEAN: OCF close to NI (low accruals); DIRTY: OCF much lower (high accruals)
    data = {
        "CLEAN": np.full(N_DATES, 38.0),
        "DIRTY": np.full(N_DATES, -10.0),
        "HIGH_ROA": np.full(N_DATES, 75.0),
        "LOW_ROA": np.full(N_DATES, 1.0),
        "HIGH_GP": np.full(N_DATES, 28.0),
        "LOW_GP": np.full(N_DATES, 3.0),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def shares_outstanding(dates: pd.DatetimeIndex) -> pd.DataFrame:
    # CLEAN: no dilution; DIRTY: growing shares
    base = {t: 100.0 for t in TICKERS}
    growth = {"CLEAN": 0.0, "DIRTY": 0.0005, "HIGH_ROA": 0.0, "LOW_ROA": 0.0001, "HIGH_GP": 0.0, "LOW_GP": 0.0001}
    data = {t: base[t] * (1 + growth[t]) ** np.arange(N_DATES) for t in TICKERS}
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def gross_profit(dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {
        "CLEAN": np.full(N_DATES, 100.0),
        "DIRTY": np.full(N_DATES, 100.0),
        "HIGH_ROA": np.full(N_DATES, 100.0),
        "LOW_ROA": np.full(N_DATES, 100.0),
        "HIGH_GP": np.full(N_DATES, 250.0),  # high GP/TA
        "LOW_GP": np.full(N_DATES, 20.0),    # low GP/TA
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sectors() -> pd.Series:
    return pd.Series(
        {"CLEAN": "Tech", "DIRTY": "Tech", "HIGH_ROA": "Finance",
         "LOW_ROA": "Finance", "HIGH_GP": "Health", "LOW_GP": "Health"}
    )


def _weights_valid(w: pd.DataFrame) -> None:
    assert isinstance(w, pd.DataFrame)
    assert (w >= 0).all().all()
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H356 ManagementMispricing
# ---------------------------------------------------------------------------

def test_management_mispricing_shape(dates, membership, prices, total_assets, net_income, operating_cf, shares_outstanding, sectors):
    strat = ManagementMispricing(
        total_assets=total_assets, net_income=net_income,
        operating_cf=operating_cf, shares_outstanding=shares_outstanding, sectors=sectors
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_management_mispricing_prefers_clean(dates, membership, prices, total_assets, net_income, operating_cf, shares_outstanding, sectors):
    strat = ManagementMispricing(
        total_assets=total_assets, net_income=net_income,
        operating_cf=operating_cf, shares_outstanding=shares_outstanding, sectors=sectors
    )
    w = strat.generate_weights(prices, membership)
    late = w.iloc[600:]
    assert late["CLEAN"].mean() >= late["DIRTY"].mean(), "should prefer clean management over diluting firms"


# ---------------------------------------------------------------------------
# H357 PerformanceMispricing
# ---------------------------------------------------------------------------

def test_performance_mispricing_shape(dates, membership, prices, total_assets, net_income, operating_cf, sectors):
    strat = PerformanceMispricing(
        total_assets=total_assets, net_income=net_income, operating_cf=operating_cf, sectors=sectors
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_performance_mispricing_prefers_high_roa(dates, membership, prices, total_assets, net_income, operating_cf, sectors):
    strat = PerformanceMispricing(
        total_assets=total_assets, net_income=net_income, operating_cf=operating_cf, sectors=sectors
    )
    w = strat.generate_weights(prices, membership)
    late = w.iloc[600:]
    assert late["HIGH_ROA"].mean() >= late["LOW_ROA"].mean(), "should prefer high-ROA firms"


# ---------------------------------------------------------------------------
# H358 MispricingCombo
# ---------------------------------------------------------------------------

def test_mispricing_combo_shape(dates, membership, prices, total_assets, net_income, operating_cf, shares_outstanding, sectors):
    strat = MispricingCombo(
        total_assets=total_assets, net_income=net_income,
        operating_cf=operating_cf, shares_outstanding=shares_outstanding, sectors=sectors
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


# ---------------------------------------------------------------------------
# H359 MispricingAvoidFilter
# ---------------------------------------------------------------------------

def test_mispricing_avoid_filter_shape(dates, membership, prices, total_assets, net_income, operating_cf, shares_outstanding, sectors):
    strat = MispricingAvoidFilter(
        total_assets=total_assets, net_income=net_income,
        operating_cf=operating_cf, shares_outstanding=shares_outstanding, sectors=sectors
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_mispricing_avoid_filter_excludes_some(dates, membership, prices, total_assets, net_income, operating_cf, shares_outstanding, sectors):
    strat = MispricingAvoidFilter(
        total_assets=total_assets, net_income=net_income,
        operating_cf=operating_cf, shares_outstanding=shares_outstanding, sectors=sectors,
        exclude_pct=0.20,
    )
    w = strat.generate_weights(prices, membership)
    late = w.iloc[600:]
    # With 20% exclusion, at least some stocks should have zero weight consistently
    zero_cols = (late.mean(axis=0) == 0).sum()
    assert zero_cols >= 1, "at least one ticker should be excluded"


# ---------------------------------------------------------------------------
# H360 MispricingQualityComposite
# ---------------------------------------------------------------------------

def test_mispricing_quality_composite_shape(dates, membership, prices, total_assets, net_income, operating_cf, shares_outstanding, gross_profit, sectors):
    strat = MispricingQualityComposite(
        total_assets=total_assets, net_income=net_income,
        operating_cf=operating_cf, shares_outstanding=shares_outstanding,
        gross_profit=gross_profit, sectors=sectors
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)
