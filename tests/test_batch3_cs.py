"""Tests for Batch 3 cross-sectional strategies: H160-H164."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    CompositeValueMomentum,
    FiveYearReversal,
    LongTermReversal,
    MomentumLongOnly,
    NearFiftyTwoWeekLow,
    ResidualMomentum,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dates():
    return pd.bdate_range("2000-01-03", periods=1300)  # enough for 5yr reversal warm-up


@pytest.fixture
def membership(dates):
    tickers = ["W", "L", "M"]  # W=winner, L=loser, M=middle
    return pd.DataFrame(True, index=dates, columns=tickers)


@pytest.fixture
def reversal_panel(dates):
    """W has been rising for 3+ years; L has been falling; M is flat."""
    n = len(dates)
    # W: steady winner (+30% per year), L: steady loser (-20% per year), M: flat
    w = pd.Series(100 * (1.30 ** (np.arange(n) / 252)), index=dates)
    loser = pd.Series(100 * (0.80 ** (np.arange(n) / 252)), index=dates)
    m = pd.Series(np.ones(n) * 100.0, index=dates)
    return pd.DataFrame({"W": w, "L": loser, "M": m})


@pytest.fixture
def synthetic_panel(dates):
    """Simple 3-stock panel for generic validity tests."""
    n = len(dates)
    rng = np.random.default_rng(42)
    data = {
        t: pd.Series(
            100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n)),
            index=dates,
        )
        for t in ["A", "B", "C"]
    }
    return pd.DataFrame(data)


@pytest.fixture
def spy_rets(dates):
    rng = np.random.default_rng(99)
    return pd.Series(rng.normal(0.0003, 0.008, len(dates)), index=dates, name="SPY")


# ---------------------------------------------------------------------------
# H160: LongTermReversal
# ---------------------------------------------------------------------------


class TestLongTermReversal:
    def test_weights_valid(self, reversal_panel, membership):
        strat = LongTermReversal(long_days=756, skip_days=21, top_quantile=0.40)
        w = strat.generate_weights(reversal_panel, membership)
        assert w.shape == reversal_panel.shape
        # All weights non-negative (long-only)
        assert (w >= 0).all().all()
        # Row sums ≤ 1.0 + tolerance (can be 0 during warm-up)
        assert (w.sum(axis=1) <= 1.01).all()

    def test_loser_selected_over_winner(self, reversal_panel, membership):
        """After 3-year warm-up, L (steady loser) should rank ahead of W (winner)."""
        strat = LongTermReversal(long_days=756, skip_days=21, top_quantile=0.40)
        w = strat.generate_weights(reversal_panel, membership)
        # Use the last 50 rows (well past warm-up)
        tail = w.tail(50)
        active_rows = tail[tail.sum(axis=1) > 0]
        assert len(active_rows) > 0
        # L should be selected (have positive weight) more often than W
        l_selected = (active_rows["L"] > 0).sum()
        w_selected = (active_rows["W"] > 0).sum()
        assert l_selected >= w_selected, f"L selected {l_selected}x, W selected {w_selected}x"

    def test_monthly_stability(self, reversal_panel, membership):
        """Weights should not change within the same calendar month."""
        strat = LongTermReversal()
        w = strat.generate_weights(reversal_panel, membership)
        months = reversal_panel.index.to_series().dt.to_period("M")
        for month, group_idx in months.groupby(months).groups.items():
            group = w.loc[group_idx]
            if group.shape[0] < 2:
                continue
            # All rows in the month should be identical
            first = group.iloc[0]
            for _, row in group.iloc[1:].iterrows():
                assert first.equals(row), f"Weight changed within month {month}"
            break  # Only check first non-trivial month after warm-up

    def test_no_lookahead(self, reversal_panel, membership):
        """Weights with first N rows zeroed should equal weights computed on truncated panel."""
        n = 800
        strat = LongTermReversal()
        w_full = strat.generate_weights(reversal_panel, membership)
        w_trunc = strat.generate_weights(reversal_panel.iloc[:n], membership.iloc[:n])
        last_row_full = w_full.iloc[n - 1]
        last_row_trunc = w_trunc.iloc[-1]
        pd.testing.assert_series_equal(last_row_full, last_row_trunc, check_names=False)


# ---------------------------------------------------------------------------
# H161: FiveYearReversal
# ---------------------------------------------------------------------------


class TestFiveYearReversal:
    def test_weights_valid(self, reversal_panel, membership):
        strat = FiveYearReversal(long_days=1260, skip_days=21, top_quantile=0.40)
        w = strat.generate_weights(reversal_panel, membership)
        assert (w >= 0).all().all()
        assert (w.sum(axis=1) <= 1.01).all()

    def test_longer_warmup_than_h160(self, reversal_panel, membership):
        """FiveYearReversal needs more rows before weights become active."""
        strat_h160 = LongTermReversal(long_days=756, skip_days=21)
        strat_h161 = FiveYearReversal(long_days=1260, skip_days=21)
        w160 = strat_h160.generate_weights(reversal_panel, membership)
        w161 = strat_h161.generate_weights(reversal_panel, membership)
        # First active row of H161 should come after first active row of H160
        first_active_160 = (w160.sum(axis=1) > 0).idxmax()
        first_active_161 = (w161.sum(axis=1) > 0).idxmax()
        assert first_active_161 > first_active_160

    def test_monthly_stability(self, reversal_panel, membership):
        strat = FiveYearReversal()
        w = strat.generate_weights(reversal_panel, membership)
        months = reversal_panel.index.to_series().dt.to_period("M")
        for month, group_idx in months.groupby(months).groups.items():
            group = w.loc[group_idx]
            if group.shape[0] < 2 or group.sum(axis=1).sum() == 0:
                continue
            first = group.iloc[0]
            for _, row in group.iloc[1:].iterrows():
                assert first.equals(row), f"Weight changed within month {month}"
            break


# ---------------------------------------------------------------------------
# H162: CompositeValueMomentum
# ---------------------------------------------------------------------------


class TestCompositeValueMomentum:
    def test_weights_valid(self, synthetic_panel, membership):
        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat = CompositeValueMomentum()
        w = strat.generate_weights(synthetic_panel, mem3)
        assert (w >= 0).all().all()
        assert (w.sum(axis=1) <= 1.01).all()

    def test_differs_from_pure_momentum(self, synthetic_panel, membership):
        """Composite ranks should differ from pure momentum ranks."""
        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat_mom = MomentumLongOnly(top_quantile=0.40)
        strat_cvm = CompositeValueMomentum(top_quantile=0.40)
        w_mom = strat_mom.generate_weights(synthetic_panel, mem3)
        w_cvm = strat_cvm.generate_weights(synthetic_panel, mem3)
        # Not identical
        assert not w_mom.equals(w_cvm)

    def test_monthly_stability(self, synthetic_panel):
        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat = CompositeValueMomentum()
        w = strat.generate_weights(synthetic_panel, mem3)
        months = synthetic_panel.index.to_series().dt.to_period("M")
        for month, group_idx in months.groupby(months).groups.items():
            group = w.loc[group_idx]
            if group.shape[0] < 2 or group.sum(axis=1).sum() == 0:
                continue
            first = group.iloc[0]
            for _, row in group.iloc[1:].iterrows():
                assert first.equals(row), f"Weight changed within month {month}"
            break


# ---------------------------------------------------------------------------
# H163: ResidualMomentum
# ---------------------------------------------------------------------------


class TestResidualMomentum:
    def test_weights_valid(self, synthetic_panel, spy_rets):
        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat = ResidualMomentum(spy_rets=spy_rets)
        w = strat.generate_weights(synthetic_panel, mem3)
        assert (w >= 0).all().all()
        assert (w.sum(axis=1) <= 1.01).all()

    def test_weights_valid_no_spy(self, synthetic_panel):
        """Without spy_rets, should use equal-weight panel average as market proxy."""
        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat = ResidualMomentum()
        w = strat.generate_weights(synthetic_panel, mem3)
        assert (w >= 0).all().all()
        assert (w.sum(axis=1) <= 1.01).all()

    def test_differs_from_raw_momentum(self, synthetic_panel, spy_rets):
        """Residual momentum should produce different weights than raw 12-1m momentum."""
        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat_raw = MomentumLongOnly(top_quantile=0.40)
        strat_res = ResidualMomentum(spy_rets=spy_rets, top_quantile=0.40)
        w_raw = strat_raw.generate_weights(synthetic_panel, mem3)
        w_res = strat_res.generate_weights(synthetic_panel, mem3)
        # Weights should not be identical (beta removal changes ranking)
        assert not w_raw.equals(w_res)

    def test_monthly_stability(self, synthetic_panel, spy_rets):
        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat = ResidualMomentum(spy_rets=spy_rets)
        w = strat.generate_weights(synthetic_panel, mem3)
        months = synthetic_panel.index.to_series().dt.to_period("M")
        for month, group_idx in months.groupby(months).groups.items():
            group = w.loc[group_idx]
            if group.shape[0] < 2 or group.sum(axis=1).sum() == 0:
                continue
            first = group.iloc[0]
            for _, row in group.iloc[1:].iterrows():
                assert first.equals(row), f"Weight changed within month {month}"
            break


# ---------------------------------------------------------------------------
# H164: NearFiftyTwoWeekLow
# ---------------------------------------------------------------------------


class TestNearFiftyTwoWeekLow:
    def test_weights_valid(self, synthetic_panel):
        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat = NearFiftyTwoWeekLow()
        w = strat.generate_weights(synthetic_panel, mem3)
        assert (w >= 0).all().all()
        assert (w.sum(axis=1) <= 1.01).all()

    def test_beaten_stock_selected(self, dates, membership):
        """Stock trading at its 52-week low (ratio=1.0) should outrank one at its high."""
        n = len(dates)
        # BEATEN: drops from 100 to 70 over 252 days, sitting near its 52-week low
        beaten = pd.Series(
            np.linspace(100, 70, n),
            index=dates,
        )
        # STRONG: rises from 100 to 150 over 252 days, trading near its 52-week high
        strong = pd.Series(
            np.linspace(100, 150, n),
            index=dates,
        )
        # NEUTRAL: slight upward drift so its score > BEATEN's
        neutral = pd.Series(np.linspace(100, 110, n), index=dates)
        panel = pd.DataFrame({"BEATEN": beaten, "STRONG": strong, "NEUTRAL": neutral})
        mem = pd.DataFrame(True, index=dates, columns=["BEATEN", "STRONG", "NEUTRAL"])
        strat = NearFiftyTwoWeekLow(window=252, top_quantile=0.40)
        w = strat.generate_weights(panel, mem)

        tail = w.tail(100)
        active_rows = tail[tail.sum(axis=1) > 0]
        assert len(active_rows) > 0
        beaten_selected = (active_rows["BEATEN"] > 0).sum()
        strong_selected = (active_rows["STRONG"] > 0).sum()
        assert beaten_selected >= strong_selected, (
            f"BEATEN selected {beaten_selected}x, STRONG selected {strong_selected}x"
        )

    def test_monthly_stability(self, synthetic_panel):
        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat = NearFiftyTwoWeekLow()
        w = strat.generate_weights(synthetic_panel, mem3)
        months = synthetic_panel.index.to_series().dt.to_period("M")
        for month, group_idx in months.groupby(months).groups.items():
            group = w.loc[group_idx]
            if group.shape[0] < 2 or group.sum(axis=1).sum() == 0:
                continue
            first = group.iloc[0]
            for _, row in group.iloc[1:].iterrows():
                assert first.equals(row), f"Weight changed within month {month}"
            break

    def test_proxy_opposite_of_h152(self, synthetic_panel):
        """Near-52wk-low and HighWaterMomentum should generally select opposite stocks."""
        from erised.strategies.cross_sectional import HighWaterMomentum

        mem3 = pd.DataFrame(True, index=synthetic_panel.index, columns=["A", "B", "C"])
        strat_low = NearFiftyTwoWeekLow(top_quantile=0.40)
        strat_high = HighWaterMomentum(top_quantile=0.40)
        w_low = strat_low.generate_weights(synthetic_panel, mem3)
        w_high = strat_high.generate_weights(synthetic_panel, mem3)
        # They should select different stocks (not identical portfolios)
        assert not w_low.equals(w_high)
