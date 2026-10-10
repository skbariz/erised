"""Tests for H366-H370: Set 5 Batch 4 anomaly-decay strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    DecayAdjustedComposite,
    EigenportfolioCharacteristic,
    IncrementalAlphaFilter,
    PostPublicationHaircutComposite,
    RecentEpochOnlyGP,
)

N_DATES, N_TICKERS = 1200, 5
TICKERS = ["HIGH_GP", "MED_GP", "LOW_GP", "HIGH_MOM", "LOW_MOM"]


@pytest.fixture
def dates():
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates):
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates):
    np.random.seed(1)
    return pd.DataFrame(
        {
            "HIGH_GP": 100 + np.linspace(0, 50, N_DATES),
            "MED_GP": 100 + np.linspace(0, 20, N_DATES),
            "LOW_GP": 100 + np.linspace(0, 5, N_DATES),
            "HIGH_MOM": 100 + np.linspace(0, 80, N_DATES),
            "LOW_MOM": 100 + np.linspace(0, 2, N_DATES),
        },
        index=dates,
    )


@pytest.fixture
def gross_profit(dates):
    return pd.DataFrame(
        {
            "HIGH_GP": np.full(N_DATES, 300.0),
            "MED_GP": np.full(N_DATES, 150.0),
            "LOW_GP": np.full(N_DATES, 30.0),
            "HIGH_MOM": np.full(N_DATES, 100.0),
            "LOW_MOM": np.full(N_DATES, 100.0),
        },
        index=dates,
    )


@pytest.fixture
def total_assets(dates):
    return pd.DataFrame({t: np.full(N_DATES, 500.0) for t in TICKERS}, index=dates)


@pytest.fixture
def net_income(dates):
    return pd.DataFrame({t: np.full(N_DATES, 30.0) for t in TICKERS}, index=dates)


@pytest.fixture
def book_equity(dates):
    return pd.DataFrame({t: np.full(N_DATES, 200.0) for t in TICKERS}, index=dates)


@pytest.fixture
def operating_cf(dates):
    return pd.DataFrame({t: np.full(N_DATES, 28.0) for t in TICKERS}, index=dates)


@pytest.fixture
def sectors():
    return pd.Series(
        {
            "HIGH_GP": "Tech",
            "MED_GP": "Tech",
            "LOW_GP": "Finance",
            "HIGH_MOM": "Finance",
            "LOW_MOM": "Health",
        }
    )


def _valid(w):
    assert isinstance(w, pd.DataFrame)
    assert (w >= 0).all().all()
    assert (w.sum(axis=1) <= 1.001).all()


def test_recent_epoch_only_gp(dates, membership, prices, gross_profit, total_assets, sectors):
    w = RecentEpochOnlyGP(
        gross_profit=gross_profit, total_assets=total_assets, sectors=sectors
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_post_publication_haircut(
    dates,
    membership,
    prices,
    gross_profit,
    total_assets,
    net_income,
    book_equity,
    operating_cf,
    sectors,
):
    w = PostPublicationHaircutComposite(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_incremental_alpha_filter(
    dates,
    membership,
    prices,
    gross_profit,
    total_assets,
    net_income,
    book_equity,
    operating_cf,
    sectors,
):
    w = IncrementalAlphaFilter(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_eigenportfolio_characteristic(
    dates,
    membership,
    prices,
    gross_profit,
    total_assets,
    net_income,
    book_equity,
    operating_cf,
    sectors,
):
    w = EigenportfolioCharacteristic(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)


def test_decay_adjusted_composite(
    dates,
    membership,
    prices,
    gross_profit,
    total_assets,
    net_income,
    book_equity,
    operating_cf,
    sectors,
):
    w = DecayAdjustedComposite(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
    ).generate_weights(prices, membership)
    assert w.shape == prices.shape
    _valid(w)
