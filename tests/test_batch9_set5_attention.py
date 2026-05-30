"""Tests for H391-H395: Set 5 Batch 9 attention/vol proxy strategies."""
from __future__ import annotations
import numpy as np
import pandas as pd
import pytest
from quant.strategies.cross_sectional import (
    ConsistentMomentumPort,
    LowAttentionValuePort,
    LowVolDisagreementAvoid,
    ReturnSpikeRiskFilter,
    VolatilityAwareComposite,
)

N_DATES, N_TICKERS = 1000, 6
TICKERS = ["HIGH_GP", "LOW_GP", "HIGH_MOM", "LOW_MOM", "HIGH_VOL", "NEUTRAL"]


@pytest.fixture
def dates():
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates):
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates):
    np.random.seed(42)
    p = {}
    for t in TICKERS:
        p[t] = 100 + np.cumsum(np.random.randn(N_DATES) * 0.5)
    # HIGH_MOM gets a strong upward trend
    p["HIGH_MOM"] = 50 + np.linspace(0, 200, N_DATES) + np.random.randn(N_DATES) * 2
    # HIGH_VOL gets larger returns
    p["HIGH_VOL"] = 100 + np.cumsum(np.random.randn(N_DATES) * 3.0)
    return pd.DataFrame(p, index=dates)


@pytest.fixture
def gross_profit(dates):
    gp = {t: np.full(N_DATES, 100.0) for t in TICKERS}
    gp["HIGH_GP"] = np.full(N_DATES, 400.0)
    gp["LOW_GP"] = np.full(N_DATES, 10.0)
    return pd.DataFrame(gp, index=dates)


@pytest.fixture
def total_assets(dates):
    return pd.DataFrame({t: np.full(N_DATES, 500.0) for t in TICKERS}, index=dates)


@pytest.fixture
def net_income(dates):
    return pd.DataFrame({t: np.full(N_DATES, 30.0) for t in TICKERS}, index=dates)


@pytest.fixture
def operating_cf(dates):
    return pd.DataFrame({t: np.full(N_DATES, 28.0) for t in TICKERS}, index=dates)


@pytest.fixture
def sectors():
    return pd.Series({
        "HIGH_GP": "Tech",
        "LOW_GP": "Tech",
        "HIGH_MOM": "Finance",
        "LOW_MOM": "Finance",
        "HIGH_VOL": "Energy",
        "NEUTRAL": "Health",
    })


def _valid(w):
    assert (w >= 0).all().all()
    assert (w.sum(axis=1) <= 1.001).all()


def test_low_attention_value_port(dates, membership, prices, gross_profit, total_assets, sectors):
    w = LowAttentionValuePort(
        gross_profit=gross_profit, total_assets=total_assets, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)
    # HIGH_GP should tend to have higher weight than LOW_GP
    late = w.iloc[200:]
    assert late["HIGH_GP"].mean() >= late["LOW_GP"].mean()


def test_consistent_momentum_port(dates, membership, prices, sectors):
    w = ConsistentMomentumPort(sectors=sectors).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_low_vol_disagreement_avoid(dates, membership, prices, gross_profit, total_assets, sectors):
    w = LowVolDisagreementAvoid(
        gross_profit=gross_profit, total_assets=total_assets, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)
    # HIGH_VOL (wider vol) should get lower weight than NEUTRAL
    late = w.iloc[500:]
    assert late["HIGH_VOL"].mean() <= late["NEUTRAL"].mean() + 0.1


def test_return_spike_risk_filter(dates, membership, prices, gross_profit, total_assets, net_income, operating_cf, sectors):
    w = ReturnSpikeRiskFilter(
        gross_profit=gross_profit, total_assets=total_assets,
        net_income=net_income, operating_cf=operating_cf, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_volatility_aware_composite(dates, membership, prices, gross_profit, total_assets, net_income, operating_cf, sectors):
    w = VolatilityAwareComposite(
        gross_profit=gross_profit, total_assets=total_assets,
        net_income=net_income, operating_cf=operating_cf, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)
