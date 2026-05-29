"""Tests for H371-H375: Set 5 Batch 5 ML model ladder strategies."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from quant.strategies.cross_sectional import (
    ElasticNetRankStrong, GradientBoostingRank, LinearCrossSectionalRank,
    MacroStateGBRank, MLEnsembleRank,
)
N_DATES, N_TICKERS = 1200, 5
TICKERS = ["A", "B", "C", "D", "E"]

@pytest.fixture
def dates(): return pd.bdate_range("2010-01-01", periods=N_DATES)
@pytest.fixture
def membership(dates): return pd.DataFrame(True, index=dates, columns=TICKERS)
@pytest.fixture
def prices(dates):
    np.random.seed(42)
    return pd.DataFrame({t: 100+np.cumsum(np.random.randn(N_DATES)*0.5) for t in TICKERS}, index=dates)
@pytest.fixture
def fund(dates): return pd.DataFrame({t: np.full(N_DATES, 100.) for t in TICKERS}, index=dates)
@pytest.fixture
def sectors(): return pd.Series({t: "Tech" for t in TICKERS})
@pytest.fixture
def spy_prices(dates): return pd.Series(100+np.linspace(0,30,N_DATES), index=dates)

def _valid(w):
    assert (w >= -1e-9).all().all()
    assert (w.sum(axis=1) <= 1.001).all()

def test_linear_cs_rank(dates, membership, prices, fund, sectors):
    w = LinearCrossSectionalRank(gross_profit=fund, total_assets=fund, net_income=fund, book_equity=fund, operating_cf=fund, long_term_debt=fund, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)

def test_elastic_net_rank_strong(dates, membership, prices, fund, sectors):
    w = ElasticNetRankStrong(gross_profit=fund, total_assets=fund, net_income=fund, book_equity=fund, operating_cf=fund, long_term_debt=fund, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)

def test_gradient_boosting_rank(dates, membership, prices, fund, sectors):
    w = GradientBoostingRank(gross_profit=fund, total_assets=fund, net_income=fund, book_equity=fund, operating_cf=fund, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)

def test_ml_ensemble_rank(dates, membership, prices, fund, sectors):
    w = MLEnsembleRank(gross_profit=fund, total_assets=fund, net_income=fund, book_equity=fund, operating_cf=fund, long_term_debt=fund, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)

def test_macro_state_gb_rank(dates, membership, prices, fund, sectors, spy_prices):
    w = MacroStateGBRank(gross_profit=fund, total_assets=fund, net_income=fund, book_equity=fund, operating_cf=fund, spy_prices=spy_prices, sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape; _valid(w)
