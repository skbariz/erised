"""Tests for H175-H179 low-beta, defensive, and vol-managed strategies."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.cross_sectional import (
    BettingAgainstBeta,
    DefensiveQualityMomentum,
    LowBetaQuality,
    LowVolatilityAnomaly,
    VolTargetedMomentum,
)

N_DATES = 1400
TICKERS = ["LBETA", "MBETA", "HBETA", "SAFE", "RISKY"]


@pytest.fixture
def dates() -> pd.DatetimeIndex:
    return pd.bdate_range("2010-01-01", periods=N_DATES)


@pytest.fixture
def membership(dates: pd.DatetimeIndex) -> pd.DataFrame:
    return pd.DataFrame(True, index=dates, columns=TICKERS)


@pytest.fixture
def spy_rets(dates: pd.DatetimeIndex) -> pd.Series:
    np.random.seed(42)
    return pd.Series(np.random.randn(N_DATES) * 0.01, index=dates, name="SPY")


@pytest.fixture
def prices(dates: pd.DatetimeIndex, spy_rets: pd.Series) -> pd.DataFrame:
    """Construct prices with known betas.
    LBETA: beta~0.3, MBETA: beta~1.0, HBETA: beta~1.8
    SAFE: low vol, RISKY: high vol.
    """
    np.random.seed(99)
    spy = spy_rets.values
    betas = [0.3, 1.0, 1.8, 0.4, 1.5]
    vols = [0.005, 0.01, 0.015, 0.004, 0.02]  # idiosyncratic vols
    data = {}
    for t, b, v in zip(TICKERS, betas, vols, strict=True):
        idio = np.random.randn(N_DATES) * v
        ret = b * spy + idio
        price = 50.0 * np.cumprod(1 + ret)
        data[t] = price
    return pd.DataFrame(data, index=dates)


@pytest.fixture
def sectors() -> pd.Series:
    return pd.Series(
        {
            "LBETA": "Tech",
            "MBETA": "Tech",
            "HBETA": "Finance",
            "SAFE": "Finance",
            "RISKY": "Healthcare",
        },
        name="sector",
    )


def _fund(dates, tickers, vals):
    df = pd.DataFrame(np.nan, index=dates, columns=tickers)
    for i, t in enumerate(tickers):
        df.loc[dates[::63], t] = vals[i]
    return df.ffill()


@pytest.fixture
def gross_profit(dates) -> pd.DataFrame:
    return _fund(dates, TICKERS, [150, 100, 60, 180, 40])


@pytest.fixture
def total_assets(dates) -> pd.DataFrame:
    return _fund(dates, TICKERS, [500, 500, 500, 500, 500])


# ---------------------------------------------------------------------------
# BettingAgainstBeta (H175)
# ---------------------------------------------------------------------------


class TestBettingAgainstBeta:
    def test_weights_sum_to_one(self, prices, membership, spy_rets, sectors, dates):
        strat = BettingAgainstBeta(spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_low_beta_selected(self, prices, membership, spy_rets, sectors, dates):
        """LBETA (beta~0.3) should get positive weight vs HBETA (beta~1.8)."""
        strat = BettingAgainstBeta(spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]]
        assert late["LBETA"].mean() >= late["HBETA"].mean()

    def test_no_nan_weights(self, prices, membership, spy_rets, sectors, dates):
        strat = BettingAgainstBeta(spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_monthly_rebalance_flag(self, prices, membership, spy_rets, sectors, dates):
        """With monthly rebalance, weights on non-month-start days must equal prior month-start."""
        strat_m = BettingAgainstBeta(spy_rets, sectors, monthly_rebalance=True)
        w_m = strat_m.generate_weights(prices, membership)
        # Check: within any given month, weights must be constant (no intra-month changes)
        late = w_m.loc[dates[600:]]
        month_label = late.index.to_period("M")
        for month in month_label.unique()[:5]:
            month_rows = late[month_label == month]
            if len(month_rows) > 1:
                # All rows in this month should be identical
                first_row = month_rows.iloc[0]
                assert (month_rows - first_row).abs().max().max() < 1e-10


# ---------------------------------------------------------------------------
# LowVolatilityAnomaly (H176)
# ---------------------------------------------------------------------------


class TestLowVolatilityAnomaly:
    def test_weights_sum_to_one(self, prices, membership, sectors, dates):
        strat = LowVolatilityAnomaly(sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_low_vol_selected(self, prices, membership, sectors, dates):
        """SAFE (low idio vol) should get weight; RISKY (high vol) should not dominate."""
        strat = LowVolatilityAnomaly(sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]]
        assert late["SAFE"].mean() >= late["RISKY"].mean()

    def test_no_nan_weights(self, prices, membership, sectors):
        strat = LowVolatilityAnomaly(sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()


# ---------------------------------------------------------------------------
# VolTargetedMomentum (H177)
# ---------------------------------------------------------------------------


class TestVolTargetedMomentum:
    def test_weights_sum_to_one(self, prices, membership, dates):
        strat = VolTargetedMomentum()
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_no_nan_weights(self, prices, membership):
        strat = VolTargetedMomentum()
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()

    def test_high_vol_scaled_down(self, prices, membership, dates):
        """High-vol stocks should get reduced weight relative to equal-weight momentum."""
        strat_scaled = VolTargetedMomentum(target_vol=0.10)
        strat_base = VolTargetedMomentum(target_vol=1.0)  # no effective scaling
        w_s = strat_scaled.generate_weights(prices, membership)
        w_b = strat_base.generate_weights(prices, membership)
        # RISKY should have proportionally lower weight when scaled
        late_s = w_s.loc[dates[600:]]
        late_b = w_b.loc[dates[600:]]
        # Among stocks that appear in both, high-vol stocks should have lower relative weight
        # Check that RISKY/LBETA ratio is lower when target_vol is tight
        risky_ratio_scaled = late_s["RISKY"].mean()
        risky_ratio_base = late_b["RISKY"].mean()
        # Scaled version should weight RISKY less than base when it appears at all
        # (may be 0 in both if RISKY has bad momentum)
        assert risky_ratio_scaled <= risky_ratio_base + 0.01  # allow small tolerance


# ---------------------------------------------------------------------------
# DefensiveQualityMomentum (H178)
# ---------------------------------------------------------------------------


class TestDefensiveQualityMomentum:
    def test_weights_sum_to_one(
        self, prices, membership, gross_profit, total_assets, spy_rets, sectors, dates
    ):
        strat = DefensiveQualityMomentum(gross_profit, total_assets, spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_sector_neutral(
        self, prices, membership, gross_profit, total_assets, spy_rets, sectors, dates
    ):
        strat = DefensiveQualityMomentum(gross_profit, total_assets, spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]]
        tech_tickers = sectors[sectors == "Tech"].index.tolist()
        fin_tickers = sectors[sectors == "Finance"].index.tolist()
        diff = (late[tech_tickers].sum(axis=1) - late[fin_tickers].sum(axis=1)).abs().mean()
        assert diff < 0.05

    def test_no_nan_weights(
        self, prices, membership, gross_profit, total_assets, spy_rets, sectors
    ):
        strat = DefensiveQualityMomentum(gross_profit, total_assets, spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        assert not w.isnull().any().any()


# ---------------------------------------------------------------------------
# LowBetaQuality (H179)
# ---------------------------------------------------------------------------


class TestLowBetaQuality:
    def test_weights_sum_to_one(
        self, prices, membership, gross_profit, total_assets, spy_rets, sectors, dates
    ):
        strat = LowBetaQuality(gross_profit, total_assets, spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]].sum(axis=1)
        assert (late.round(6) == 1.0).mean() > 0.90

    def test_low_beta_high_quality_selected(
        self, prices, membership, gross_profit, total_assets, spy_rets, sectors, dates
    ):
        """LBETA has low beta; SAFE also low vol; HBETA/RISKY should be deprioritized."""
        strat = LowBetaQuality(gross_profit, total_assets, spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]]
        # HBETA (high beta) should not dominate
        assert late["HBETA"].mean() < 0.5

    def test_sector_neutral(
        self, prices, membership, gross_profit, total_assets, spy_rets, sectors, dates
    ):
        strat = LowBetaQuality(gross_profit, total_assets, spy_rets, sectors)
        w = strat.generate_weights(prices, membership)
        late = w.loc[dates[600:]]
        tech_tickers = sectors[sectors == "Tech"].index.tolist()
        fin_tickers = sectors[sectors == "Finance"].index.tolist()
        diff = (late[tech_tickers].sum(axis=1) - late[fin_tickers].sum(axis=1)).abs().mean()
        assert diff < 0.05
