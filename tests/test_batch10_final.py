"""Tests for Batch 10 strategies: 52w-high, idio-momentum, skewness, earnings accel, composite (H195-H199)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.cross_sectional import (
    EarningsAcceleration,
    FiftyTwoWeekHighMomentum,
    IdiosyncraticMomentum,
    LowSkewnessAnomaly,
    MultiSignalComposite,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

DATES = pd.date_range("2008-01-01", periods=600, freq="B")
TICKERS = ["AAPL", "MSFT", "GOOG", "JPM", "GS", "BAC", "XOM", "CVX", "JNJ", "PFE"]
SECTORS = pd.Series(
    {
        "AAPL": "Tech",
        "MSFT": "Tech",
        "GOOG": "Tech",
        "JPM": "Financials",
        "GS": "Financials",
        "BAC": "Financials",
        "XOM": "Energy",
        "CVX": "Energy",
        "JNJ": "Healthcare",
        "PFE": "Healthcare",
    }
)


def make_prices(seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rets = rng.normal(0.0003, 0.015, (len(DATES), len(TICKERS)))
    prices = 100 * np.cumprod(1 + rets, axis=0)
    return pd.DataFrame(prices, index=DATES, columns=TICKERS)


def make_membership(prices: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=prices.index, columns=prices.columns)


def make_net_income(prices: pd.DataFrame, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    ni_vals = rng.normal(1e9, 2e8, (len(prices), len(prices.columns)))
    ni = pd.DataFrame(ni_vals, index=prices.index, columns=prices.columns)
    ni.iloc[: len(prices) // 2] = np.nan  # simulate EDGAR gaps early on
    return ni


@pytest.fixture
def prices():
    return make_prices()


@pytest.fixture
def membership(prices):
    return make_membership(prices)


@pytest.fixture
def net_income(prices):
    return make_net_income(prices)


# ---------------------------------------------------------------------------
# H195: FiftyTwoWeekHighMomentum
# ---------------------------------------------------------------------------


class TestFiftyTwoWeekHighMomentum:
    def test_weights_sum_to_one(self, prices, membership):
        s = FiftyTwoWeekHighMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        assert active.any()
        totals = w[active].sum(axis=1)
        assert (totals - 1.0).abs().max() < 1e-9

    def test_no_negative_weights(self, prices, membership):
        s = FiftyTwoWeekHighMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert (w >= -1e-12).all().all()

    def test_shape_matches_input(self, prices, membership):
        s = FiftyTwoWeekHighMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert w.shape == prices.shape

    def test_early_rows_zero_before_window(self, prices, membership):
        s = FiftyTwoWeekHighMomentum(sectors=SECTORS, high_window=252)
        w = s.generate_weights(prices, membership)
        # Should have no weights until at least half the window
        assert w.iloc[:60].sum().sum() == 0.0

    def test_periodic_rebalance_holds(self, prices, membership):
        s = FiftyTwoWeekHighMomentum(sectors=SECTORS, rebalance_period=21)
        w = s.generate_weights(prices, membership)
        active_idx = w[w.sum(axis=1) > 0].index
        if len(active_idx) < 50:
            return
        # Within each 21-day block, weights must be constant
        block_start = w.iloc[::21]
        aligned = block_start.reindex(w.index, method="ffill")
        diff = (w - aligned).abs().max(axis=1)
        assert (diff < 1e-10).all()

    def test_custom_window(self, prices, membership):
        s = FiftyTwoWeekHighMomentum(sectors=SECTORS, high_window=126)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        assert active.any()

    def test_membership_respected(self, prices):
        mem = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        mem[["AAPL", "MSFT", "XOM", "JNJ"]] = True
        s = FiftyTwoWeekHighMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, mem)
        non_members = ["GOOG", "JPM", "GS", "BAC", "CVX", "PFE"]
        assert (w[non_members] == 0.0).all().all()


# ---------------------------------------------------------------------------
# H196: IdiosyncraticMomentum
# ---------------------------------------------------------------------------


class TestIdiosyncraticMomentum:
    def test_weights_sum_to_one(self, prices, membership):
        s = IdiosyncraticMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        assert active.any()
        totals = w[active].sum(axis=1)
        assert (totals - 1.0).abs().max() < 1e-9

    def test_no_negative_weights(self, prices, membership):
        s = IdiosyncraticMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert (w >= -1e-12).all().all()

    def test_shape_matches_input(self, prices, membership):
        s = IdiosyncraticMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert w.shape == prices.shape

    def test_early_rows_zero(self, prices, membership):
        s = IdiosyncraticMomentum(sectors=SECTORS, mom_window=252, skip_window=21)
        w = s.generate_weights(prices, membership)
        # Need 252 + 21 days before any signal; first rows must be zero
        assert w.iloc[:200].sum().sum() == 0.0

    def test_sector_neutrality(self, prices, membership):
        s = IdiosyncraticMomentum(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        if not active.any():
            return
        active_w = w[active]
        tech_w = active_w[["AAPL", "MSFT", "GOOG"]].sum(axis=1)
        fin_w = active_w[["JPM", "GS", "BAC"]].sum(axis=1)
        # Each sector should have equal allocation within tolerance
        assert (tech_w - fin_w).abs().max() < 0.05

    def test_custom_params(self, prices, membership):
        s = IdiosyncraticMomentum(sectors=SECTORS, mom_window=126, skip_window=5)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        assert active.any()


# ---------------------------------------------------------------------------
# H197: LowSkewnessAnomaly
# ---------------------------------------------------------------------------


class TestLowSkewnessAnomaly:
    def test_weights_sum_to_one(self, prices, membership):
        s = LowSkewnessAnomaly(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        assert active.any()
        totals = w[active].sum(axis=1)
        assert (totals - 1.0).abs().max() < 1e-9

    def test_no_negative_weights(self, prices, membership):
        s = LowSkewnessAnomaly(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert (w >= -1e-12).all().all()

    def test_shape_matches_input(self, prices, membership):
        s = LowSkewnessAnomaly(sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert w.shape == prices.shape

    def test_early_rows_zero_before_half_window(self, prices, membership):
        s = LowSkewnessAnomaly(sectors=SECTORS, skew_window=252)
        w = s.generate_weights(prices, membership)
        # min_periods = skew_window // 2 = 126
        assert w.iloc[:100].sum().sum() == 0.0

    def test_periodic_rebalance(self, prices, membership):
        s = LowSkewnessAnomaly(sectors=SECTORS, rebalance_period=21)
        w = s.generate_weights(prices, membership)
        block_start = w.iloc[::21]
        aligned = block_start.reindex(w.index, method="ffill")
        diff = (w - aligned).abs().max(axis=1)
        assert (diff < 1e-10).all()

    def test_custom_window(self, prices, membership):
        s = LowSkewnessAnomaly(sectors=SECTORS, skew_window=126)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        assert active.any()

    def test_membership_respected(self, prices):
        mem = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        mem[["AAPL", "MSFT", "XOM", "JNJ"]] = True
        s = LowSkewnessAnomaly(sectors=SECTORS)
        w = s.generate_weights(prices, mem)
        assert (w[["GOOG", "JPM", "GS", "BAC", "CVX", "PFE"]] == 0.0).all().all()


# ---------------------------------------------------------------------------
# H198: EarningsAcceleration
# ---------------------------------------------------------------------------


class TestEarningsAcceleration:
    def test_weights_sum_to_one_when_active(self, prices, membership, net_income):
        s = EarningsAcceleration(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        if not active.any():
            pytest.skip("No active rows (EDGAR gap)")
        totals = w[active].sum(axis=1)
        assert (totals - 1.0).abs().max() < 1e-9

    def test_no_negative_weights(self, prices, membership, net_income):
        s = EarningsAcceleration(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert (w >= -1e-12).all().all()

    def test_shape_matches_input(self, prices, membership, net_income):
        s = EarningsAcceleration(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert w.shape == prices.shape

    def test_zero_weights_when_no_fundamentals(self, prices, membership):
        # All NaN net_income → no signal → all zero weights
        ni_all_nan = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        s = EarningsAcceleration(net_income=ni_all_nan, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert w.sum().sum() == 0.0

    def test_monthly_rebalance_structure(self, prices, membership, net_income):
        s = EarningsAcceleration(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active_rows = w[w.sum(axis=1) > 0]
        if len(active_rows) < 40:
            pytest.skip("Not enough active rows")
        # Weights change only at month boundaries
        w_diff = w.diff().abs().sum(axis=1)
        change_days = w_diff[w_diff > 1e-10].index
        if len(change_days) == 0:
            pytest.skip("No weight changes found")
        # All change days should be within the first few days of a month
        for d in change_days:
            assert d.day <= 5, f"Weight changed on {d} (not near month start)"

    def test_custom_accel_window(self, prices, membership, net_income):
        s = EarningsAcceleration(net_income=net_income, sectors=SECTORS, accel_window=42)
        w = s.generate_weights(prices, membership)
        assert w.shape == prices.shape


# ---------------------------------------------------------------------------
# H199: MultiSignalComposite
# ---------------------------------------------------------------------------


class TestMultiSignalComposite:
    def test_weights_sum_to_one_when_active(self, prices, membership, net_income):
        s = MultiSignalComposite(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        if not active.any():
            pytest.skip("No active rows")
        totals = w[active].sum(axis=1)
        assert (totals - 1.0).abs().max() < 1e-9

    def test_no_negative_weights(self, prices, membership, net_income):
        s = MultiSignalComposite(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert (w >= -1e-12).all().all()

    def test_shape_matches_input(self, prices, membership, net_income):
        s = MultiSignalComposite(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        assert w.shape == prices.shape

    def test_works_without_earnings_signal(self, prices, membership):
        # NaN net_income → composite falls back to price-only signals
        ni_nan = pd.DataFrame(np.nan, index=prices.index, columns=prices.columns)
        s = MultiSignalComposite(net_income=ni_nan, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        # Price signals (mom + reversal) should still produce weights after warm-up
        assert active.any()

    def test_sector_neutrality(self, prices, membership, net_income):
        s = MultiSignalComposite(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        active = w.sum(axis=1) > 0
        if not active.any():
            pytest.skip("No active rows")
        active_w = w[active]
        tech_w = active_w[["AAPL", "MSFT", "GOOG"]].sum(axis=1)
        fin_w = active_w[["JPM", "GS", "BAC"]].sum(axis=1)
        assert (tech_w - fin_w).abs().max() < 0.05

    def test_monthly_rebalance_structure(self, prices, membership, net_income):
        s = MultiSignalComposite(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, membership)
        w_diff = w.diff().abs().sum(axis=1)
        change_days = w_diff[w_diff > 1e-10].index
        if len(change_days) == 0:
            pytest.skip("No weight changes found")
        for d in change_days:
            assert d.day <= 5, f"Weight changed on {d} (not near month start)"

    def test_custom_reversal_window(self, prices, membership, net_income):
        s = MultiSignalComposite(net_income=net_income, sectors=SECTORS, reversal_window=3)
        w = s.generate_weights(prices, membership)
        assert w.shape == prices.shape

    def test_membership_respected(self, prices, net_income):
        mem = pd.DataFrame(False, index=prices.index, columns=prices.columns)
        mem[["AAPL", "MSFT", "XOM", "JNJ"]] = True
        s = MultiSignalComposite(net_income=net_income, sectors=SECTORS)
        w = s.generate_weights(prices, mem)
        assert (w[["GOOG", "JPM", "GS", "BAC", "CVX", "PFE"]] == 0.0).all().all()
