"""Tests for H381-H385: Set 5 Batch 7 governance proxy strategies."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from quant.strategies.cross_sectional import (
    BuybackQualityGovernance, CapitalDisciplineGovernance,
    GovernanceMDRComposite, GovernanceQualityComposite, LowDilutionGovernance,
)
N_DATES, N_TICKERS = 1200, 5
TICKERS = ["BUYBACK", "DILUTER", "HIGH_GP", "LOW_GP", "NEUTRAL"]

@pytest.fixture
def dates(): return pd.bdate_range("2010-01-01", periods=N_DATES)
@pytest.fixture
def membership(dates): return pd.DataFrame(True, index=dates, columns=TICKERS)
@pytest.fixture
def prices(dates):
    np.random.seed(3)
    return pd.DataFrame({t: 100+np.cumsum(np.random.randn(N_DATES)*0.5) for t in TICKERS}, index=dates)

@pytest.fixture
def shares_outstanding(dates):
    sh = {"BUYBACK": 100*(0.9997**np.arange(N_DATES)), "DILUTER": 100*(1.0005**np.arange(N_DATES)), "HIGH_GP": np.full(N_DATES, 100.), "LOW_GP": np.full(N_DATES, 100.), "NEUTRAL": np.full(N_DATES, 100.)}
    return pd.DataFrame(sh, index=dates)
@pytest.fixture
def total_assets(dates): return pd.DataFrame({t: np.full(N_DATES, 500.) for t in TICKERS}, index=dates)
@pytest.fixture
def net_income(dates): return pd.DataFrame({t: np.full(N_DATES, 30.) for t in TICKERS}, index=dates)
@pytest.fixture
def operating_cf(dates): return pd.DataFrame({t: np.full(N_DATES, 28.) for t in TICKERS}, index=dates)
@pytest.fixture
def gross_profit(dates): return pd.DataFrame({"BUYBACK": np.full(N_DATES, 100.), "DILUTER": np.full(N_DATES, 100.), "HIGH_GP": np.full(N_DATES, 300.), "LOW_GP": np.full(N_DATES, 30.), "NEUTRAL": np.full(N_DATES, 150.)}, index=dates)
@pytest.fixture
def sectors(): return pd.Series({"BUYBACK": "Tech", "DILUTER": "Tech", "HIGH_GP": "Finance", "LOW_GP": "Finance", "NEUTRAL": "Health"})

def _valid(w):
    assert (w >= 0).all().all(); assert (w.sum(axis=1) <= 1.001).all()

def test_low_dilution_governance(dates, membership, prices, shares_outstanding, sectors):
    w = LowDilutionGovernance(shares_outstanding=shares_outstanding, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)
    late = w.iloc[600:]; assert late["BUYBACK"].mean() >= late["DILUTER"].mean()

def test_buyback_quality_governance(dates, membership, prices, shares_outstanding, sectors):
    w = BuybackQualityGovernance(shares_outstanding=shares_outstanding, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)
    late = w.iloc[600:]; assert late["BUYBACK"].mean() >= late["DILUTER"].mean()

def test_capital_discipline_governance(dates, membership, prices, shares_outstanding, total_assets, net_income, operating_cf, sectors):
    w = CapitalDisciplineGovernance(shares_outstanding=shares_outstanding, total_assets=total_assets, net_income=net_income, operating_cf=operating_cf, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)

def test_governance_quality_composite(dates, membership, prices, shares_outstanding, total_assets, net_income, operating_cf, gross_profit, sectors):
    w = GovernanceQualityComposite(shares_outstanding=shares_outstanding, total_assets=total_assets, net_income=net_income, operating_cf=operating_cf, gross_profit=gross_profit, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)

def test_governance_mdr_composite(dates, membership, prices, shares_outstanding, total_assets, net_income, operating_cf, gross_profit, sectors):
    w = GovernanceMDRComposite(shares_outstanding=shares_outstanding, total_assets=total_assets, net_income=net_income, operating_cf=operating_cf, gross_profit=gross_profit, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)
