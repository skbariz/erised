"""Tests for H376-H380: Set 5 Batch 6 earnings-event strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    EarningsConfirmedQuality,
    EarningsSurpriseDrift,
    EarningsWindowQualityBoost,
    EventAwareComposite,
    QuietPeriodMomentum,
)

N_DATES, N_TICKERS = 1200, 5
TICKERS = ["HIGH_GP", "LOW_GP", "SURP_POS", "SURP_NEG", "MOM_HI"]


@pytest.fixture
def dates():
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates):
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates):
    np.random.seed(5)
    return pd.DataFrame(
        {
            "HIGH_GP": 100 + np.linspace(0, 50, N_DATES),
            "LOW_GP": 100 + np.linspace(0, 5, N_DATES),
            "SURP_POS": 100 + np.linspace(0, 30, N_DATES),
            "SURP_NEG": 100 + np.linspace(0, 3, N_DATES),
            "MOM_HI": 100 + np.linspace(0, 80, N_DATES),
        },
        index=dates,
    )


@pytest.fixture
def gross_profit(dates):
    return pd.DataFrame(
        {
            "HIGH_GP": np.full(N_DATES, 300.0),
            "LOW_GP": np.full(N_DATES, 30.0),
            "SURP_POS": np.full(N_DATES, 100.0),
            "SURP_NEG": np.full(N_DATES, 100.0),
            "MOM_HI": np.full(N_DATES, 100.0),
        },
        index=dates,
    )


@pytest.fixture
def total_assets(dates):
    return pd.DataFrame({t: np.full(N_DATES, 500.0) for t in TICKERS}, index=dates)


@pytest.fixture
def net_income(dates):
    # SURP_POS: NI rises every 63 days; SURP_NEG: NI falls
    ni = {t: np.full(N_DATES, 50.0) for t in TICKERS}
    for i in range(0, N_DATES, 63):
        ni["SURP_POS"][i:] = 50.0 + (i // 63) * 5
        ni["SURP_NEG"][i:] = 50.0 - (i // 63) * 2
    return pd.DataFrame(ni, index=dates)


@pytest.fixture
def sectors():
    return pd.Series(
        {
            "HIGH_GP": "Tech",
            "LOW_GP": "Tech",
            "SURP_POS": "Finance",
            "SURP_NEG": "Finance",
            "MOM_HI": "Health",
        }
    )


def _valid(w):
    assert isinstance(w, pd.DataFrame)
    assert (w >= 0).all().all()
    assert (w.sum(axis=1) <= 1.001).all()


def test_earnings_window_quality_boost(
    dates, membership, prices, gross_profit, total_assets, net_income, sectors
):
    w = EarningsWindowQualityBoost(
        gross_profit=gross_profit, total_assets=total_assets, net_income=net_income, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_earnings_surprise_drift(dates, membership, prices, total_assets, net_income, sectors):
    w = EarningsSurpriseDrift(
        total_assets=total_assets, net_income=net_income, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_quiet_period_momentum(dates, membership, prices, net_income, sectors):
    w = QuietPeriodMomentum(net_income=net_income, sectors=sectors).generate_weights(
        prices, membership
    )
    assert w.shape == prices.shape
    _valid(w)


def test_earnings_confirmed_quality(
    dates, membership, prices, gross_profit, total_assets, net_income, sectors
):
    w = EarningsConfirmedQuality(
        gross_profit=gross_profit, total_assets=total_assets, net_income=net_income, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)
    late = w.iloc[600:]
    # Stocks with positive NI drift should be preferred
    assert late["SURP_POS"].mean() >= late["SURP_NEG"].mean()


def test_event_aware_composite(
    dates, membership, prices, gross_profit, total_assets, net_income, sectors
):
    w = EventAwareComposite(
        gross_profit=gross_profit, total_assets=total_assets, net_income=net_income, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)
