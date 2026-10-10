"""Tests for H396-H399: Set 5 Batch 10 execution-aware strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    ExecutionOptimizedComposite,
    LiquidityRegimeCapPort,
    LowTurnoverGP,
    SpreadAwarePositionSizer,
)

N_DATES, N_TICKERS = 1200, 5
TICKERS = ["HIGH_GP", "LOW_GP", "HIGH_VOL", "LOW_VOL", "NEUTRAL"]


@pytest.fixture
def dates():
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates):
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates):
    np.random.seed(99)
    p = {}
    for t in TICKERS:
        p[t] = 100 + np.cumsum(np.random.randn(N_DATES) * 0.5)
    p["HIGH_VOL"] = 100 + np.cumsum(np.random.randn(N_DATES) * 3.0)
    p["LOW_VOL"] = 100 + np.cumsum(np.random.randn(N_DATES) * 0.1)
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
    return pd.Series(
        {
            "HIGH_GP": "Tech",
            "LOW_GP": "Tech",
            "HIGH_VOL": "Energy",
            "LOW_VOL": "Energy",
            "NEUTRAL": "Health",
        }
    )


def _valid(w):
    assert (w >= 0).all().all()
    assert (w.sum(axis=1) <= 1.001).all()


def test_low_turnover_gp(dates, membership, prices, gross_profit, total_assets, sectors):
    w = LowTurnoverGP(
        gross_profit=gross_profit, total_assets=total_assets, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)
    # HIGH_GP should tend to get higher weight
    late = w.iloc[300:]
    assert late["HIGH_GP"].mean() >= late["LOW_GP"].mean()


def test_spread_aware_position_sizer(
    dates, membership, prices, gross_profit, total_assets, net_income, operating_cf, sectors
):
    w = SpreadAwarePositionSizer(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        operating_cf=operating_cf,
        sectors=sectors,
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_liquidity_regime_cap_port(dates, membership, prices, gross_profit, total_assets, sectors):
    w = LiquidityRegimeCapPort(
        gross_profit=gross_profit, total_assets=total_assets, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_execution_optimized_composite(
    dates, membership, prices, gross_profit, total_assets, net_income, operating_cf, sectors
):
    w = ExecutionOptimizedComposite(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        operating_cf=operating_cf,
        sectors=sectors,
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)
