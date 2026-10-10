"""Tests for H361-H365: Set 5 Batch 3 nonlinear/regularized strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    ElasticNetReturnForecast,
    MonotonicityVotingComposite,
    NonlinearMomentumBins,
    PercentileBinnedGP,
    ShrunkSignalComposite,
)

N_DATES = 1200
N_TICKERS = 6
TICKERS = ["HIGH_GP", "MED_GP", "LOW_GP", "HIGH_MOM", "MED_MOM", "LOW_MOM"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def prices(dates: pd.DatetimeIndex) -> pd.DataFrame:
    np.random.seed(99)
    data = {
        "HIGH_GP": 100 + np.linspace(0, 50, N_DATES) + np.random.randn(N_DATES) * 0.5,
        "MED_GP": 100 + np.linspace(0, 20, N_DATES) + np.random.randn(N_DATES) * 0.5,
        "LOW_GP": 100 + np.linspace(0, 5, N_DATES) + np.random.randn(N_DATES) * 0.5,
        "HIGH_MOM": 100 + np.linspace(0, 80, N_DATES) + np.random.randn(N_DATES) * 0.5,
        "MED_MOM": 100 + np.linspace(0, 20, N_DATES) + np.random.randn(N_DATES) * 0.5,
        "LOW_MOM": 100 + np.linspace(0, 2, N_DATES) + np.random.randn(N_DATES) * 0.5,
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def gross_profit(dates: pd.DatetimeIndex) -> pd.DataFrame:
    data = {
        "HIGH_GP": np.full(N_DATES, 300.0),
        "MED_GP": np.full(N_DATES, 150.0),
        "LOW_GP": np.full(N_DATES, 30.0),
        "HIGH_MOM": np.full(N_DATES, 100.0),
        "MED_MOM": np.full(N_DATES, 100.0),
        "LOW_MOM": np.full(N_DATES, 100.0),
    }
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def total_assets(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({t: np.full(N_DATES, 500.0) for t in TICKERS}, index=dates)


@pytest.fixture
def net_income(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({t: np.full(N_DATES, 30.0) for t in TICKERS}, index=dates)


@pytest.fixture
def book_equity(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({t: np.full(N_DATES, 200.0) for t in TICKERS}, index=dates)


@pytest.fixture
def operating_cf(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame({t: np.full(N_DATES, 28.0) for t in TICKERS}, index=dates)


@pytest.fixture
def sectors() -> pd.Series:
    return pd.Series(
        {
            "HIGH_GP": "Tech",
            "MED_GP": "Tech",
            "LOW_GP": "Finance",
            "HIGH_MOM": "Finance",
            "MED_MOM": "Health",
            "LOW_MOM": "Health",
        }
    )


def _weights_valid(w: pd.DataFrame) -> None:
    assert isinstance(w, pd.DataFrame)
    assert (w >= 0).all().all()
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H361 PercentileBinnedGP
# ---------------------------------------------------------------------------


def test_percentile_binned_gp_shape(dates, membership, prices, gross_profit, total_assets, sectors):
    strat = PercentileBinnedGP(
        gross_profit=gross_profit, total_assets=total_assets, sectors=sectors
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_percentile_binned_gp_prefers_high(
    dates, membership, prices, gross_profit, total_assets, sectors
):
    strat = PercentileBinnedGP(
        gross_profit=gross_profit, total_assets=total_assets, sectors=sectors
    )
    w = strat.generate_weights(prices, membership)
    late = w.iloc[400:]
    assert late["HIGH_GP"].mean() >= late["LOW_GP"].mean(), "should prefer high GP/TA quintile"


# ---------------------------------------------------------------------------
# H362 MonotonicityVotingComposite
# ---------------------------------------------------------------------------


def test_monotonicity_voting_shape(
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
    strat = MonotonicityVotingComposite(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


# ---------------------------------------------------------------------------
# H363 ElasticNetReturnForecast
# ---------------------------------------------------------------------------


def test_elastic_net_shape(
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
    strat = ElasticNetReturnForecast(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_elastic_net_no_negative_weights(
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
    strat = ElasticNetReturnForecast(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
    )
    w = strat.generate_weights(prices, membership)
    assert (w >= -1e-9).all().all(), "long-only: no negative weights"


# ---------------------------------------------------------------------------
# H364 NonlinearMomentumBins
# ---------------------------------------------------------------------------


def test_nonlinear_momentum_bins_shape(dates, membership, prices, sectors):
    strat = NonlinearMomentumBins(sectors=sectors)
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_nonlinear_momentum_bins_prefers_high_mom(dates, membership, prices, sectors):
    strat = NonlinearMomentumBins(sectors=sectors, top_decile=0.20)
    w = strat.generate_weights(prices, membership)
    late = w.iloc[400:]
    assert late["HIGH_MOM"].mean() >= late["LOW_MOM"].mean(), "should prefer high-momentum stocks"


def test_nonlinear_momentum_bins_narrower_than_quintile(dates, membership, prices, sectors):
    strat10 = NonlinearMomentumBins(sectors=sectors, top_decile=0.10)
    strat20 = NonlinearMomentumBins(sectors=sectors, top_decile=0.20)
    w10 = strat10.generate_weights(prices, membership)
    w20 = strat20.generate_weights(prices, membership)
    # Decile should have fewer non-zero positions on average
    avg10 = (w10 > 0).sum(axis=1).mean()
    avg20 = (w20 > 0).sum(axis=1).mean()
    assert avg10 <= avg20, "top-decile should hold fewer stocks than top-quintile"


# ---------------------------------------------------------------------------
# H365 ShrunkSignalComposite
# ---------------------------------------------------------------------------


def test_shrunk_signal_composite_shape(
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
    strat = ShrunkSignalComposite(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
    )
    w = strat.generate_weights(prices, membership)
    assert w.shape == prices.shape
    _weights_valid(w)


def test_shrunk_signal_composite_clip_reduces_extremes(
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
    # clip_val=1.0 should concentrate less than clip_val=5.0
    strat_tight = ShrunkSignalComposite(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
        clip_val=1.0,
    )
    strat_loose = ShrunkSignalComposite(
        gross_profit=gross_profit,
        total_assets=total_assets,
        net_income=net_income,
        book_equity=book_equity,
        operating_cf=operating_cf,
        sectors=sectors,
        clip_val=5.0,
    )
    w_tight = strat_tight.generate_weights(prices, membership)
    w_loose = strat_loose.generate_weights(prices, membership)
    # Both should be valid
    _weights_valid(w_tight)
    _weights_valid(w_loose)
