"""Tests for H386-H390: Set 5 Batch 8 ESG proxy strategies."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from quant.strategies.cross_sectional import (
    ControversyProxyAvoid, ESGProxyComposite, GreenProxyQuality,
    LowCarbonSectorNeutral, SinExclusionPremium,
)
N_DATES, N_TICKERS = 800, 8
TICKERS = ["STAPLES1", "TECH1", "TECH2", "UTIL1", "ENERGY1", "HIGH_GP", "HIGH_LTD", "NEUTRAL"]

@pytest.fixture
def dates(): return pd.bdate_range("2010-01-01", periods=N_DATES)
@pytest.fixture
def membership(dates): return pd.DataFrame(True, index=dates, columns=TICKERS)
@pytest.fixture
def prices(dates):
    np.random.seed(77)
    return pd.DataFrame({t: 100+np.cumsum(np.random.randn(N_DATES)*0.5) for t in TICKERS}, index=dates)
@pytest.fixture
def gross_profit(dates):
    gp = {t: np.full(N_DATES, 100.) for t in TICKERS}
    gp["TECH1"] = np.full(N_DATES, 300.)  # highest GP in green universe
    gp["UTIL1"] = np.full(N_DATES, 250.)
    return pd.DataFrame(gp, index=dates)
@pytest.fixture
def total_assets(dates): return pd.DataFrame({t: np.full(N_DATES, 500.) for t in TICKERS}, index=dates)
@pytest.fixture
def long_term_debt(dates): return pd.DataFrame({"STAPLES1": np.full(N_DATES, 50.), "TECH1": np.full(N_DATES, 50.), "TECH2": np.full(N_DATES, 50.), "UTIL1": np.full(N_DATES, 50.), "ENERGY1": np.full(N_DATES, 50.), "HIGH_GP": np.full(N_DATES, 50.), "HIGH_LTD": np.full(N_DATES, 400.), "NEUTRAL": np.full(N_DATES, 50.)}, index=dates)
@pytest.fixture
def shares_outstanding(dates): return pd.DataFrame({t: np.full(N_DATES, 100.) for t in TICKERS}, index=dates)
@pytest.fixture
def sectors(): return pd.Series({"STAPLES1": "Consumer Staples", "TECH1": "Information Technology", "TECH2": "Information Technology", "UTIL1": "Utilities", "ENERGY1": "Energy", "HIGH_GP": "Financials", "HIGH_LTD": "Financials", "NEUTRAL": "Health Care"})

def _valid(w):
    assert (w >= 0).all().all(); assert (w.sum(axis=1) <= 1.001).all()

def test_sin_exclusion_premium(dates, membership, prices, sectors):
    w = SinExclusionPremium(sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)
    assert (w["STAPLES1"] == 0).all(), "Consumer Staples should be excluded"

def test_low_carbon_sector_neutral(dates, membership, prices, gross_profit, total_assets, sectors):
    w = LowCarbonSectorNeutral(gross_profit=gross_profit, total_assets=total_assets, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)
    assert (w["ENERGY1"] == 0).all(), "Energy should be excluded"

def test_green_proxy_quality(dates, membership, prices, gross_profit, total_assets, sectors):
    w = GreenProxyQuality(gross_profit=gross_profit, total_assets=total_assets, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)
    late = w.iloc[200:]
    assert late[["TECH1", "TECH2", "UTIL1"]].sum(axis=1).mean() > 0.5, "Green sectors should dominate"

def test_controversy_proxy_avoid(dates, membership, prices, gross_profit, total_assets, long_term_debt, shares_outstanding, sectors):
    w = ControversyProxyAvoid(gross_profit=gross_profit, total_assets=total_assets, long_term_debt=long_term_debt, shares_outstanding=shares_outstanding, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)

def test_esg_proxy_composite(dates, membership, prices, gross_profit, total_assets, long_term_debt, shares_outstanding, sectors):
    w = ESGProxyComposite(gross_profit=gross_profit, total_assets=total_assets, long_term_debt=long_term_debt, shares_outstanding=shares_outstanding, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)
