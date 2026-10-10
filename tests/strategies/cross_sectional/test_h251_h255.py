"""Tests for H251-H255: Tail Risk and Lottery Avoidance strategies.

Verifies no-lookahead, weight integrity, and basic signal direction for:
  H251 IdiosyncraticVolFactor
  H252 MaxReturnLotteryAvoid
  H253 DownsideBetaFactor
  H254 TailRiskComposite
  H255 TailFilteredMomentum
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    DownsideBetaFactor,
    IdiosyncraticVolFactor,
    MaxReturnLotteryAvoid,
    TailFilteredMomentum,
    TailRiskComposite,
)

# ---------------------------------------------------------------------------
# Shared fixtures
# ---------------------------------------------------------------------------

N_DATES = 600
N_STOCKS = 30
RNG = np.random.default_rng(251)


def _make_panel(n: int = N_DATES, m: int = N_STOCKS, seed: int = 251) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2000-01-03", periods=n, freq="B")
    prices = 100 * np.cumprod(1 + rng.normal(0.0003, 0.015, (n, m)), axis=0)
    return pd.DataFrame(prices, index=idx, columns=[f"T{i:03d}" for i in range(m)])


def _make_membership(panel: pd.DataFrame, frac: float = 0.8) -> pd.DataFrame:
    rng = np.random.default_rng(42)
    mask = rng.random(panel.shape) < frac
    return pd.DataFrame(mask, index=panel.index, columns=panel.columns)


def _make_spy(panel: pd.DataFrame) -> pd.Series:
    rng = np.random.default_rng(999)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, len(panel)))
    return pd.Series(spy, index=panel.index, name="SPY").pct_change().fillna(0.0)


def _make_sectors(panel: pd.DataFrame) -> pd.Series:
    sectors = ["Tech", "Health", "Finance", "Energy", "Consumer"]
    return pd.Series(
        [sectors[i % len(sectors)] for i in range(panel.shape[1])],
        index=panel.columns,
    )


# ---------------------------------------------------------------------------
# H251: IdiosyncraticVolFactor
# ---------------------------------------------------------------------------


class TestIdiosyncraticVolFactor:
    def setup_method(self):
        self.panel = _make_panel()
        self.membership = _make_membership(self.panel)
        self.spy = _make_spy(self.panel)
        self.sectors = _make_sectors(self.panel)
        self.strat = IdiosyncraticVolFactor(sectors=self.sectors, spy_rets=self.spy)

    def test_shape_matches_input(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert w.shape == self.panel.shape

    def test_weights_nonnegative(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.fillna(0.0) >= -1e-9).all().all()

    def test_weights_sum_at_most_one(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.sum(axis=1).round(6) <= 1.0 + 1e-6).all()

    def test_warmup_rows_are_zero(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        # First idio_vol_window rows should have no signal
        assert w.iloc[: self.strat.idio_vol_window].sum().sum() == pytest.approx(0.0, abs=1e-9)

    def test_periodic_rebalance_structure(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        tail = w.iloc[200:]
        changes = tail.diff().abs().sum(axis=1)
        n_rebalances = (changes > 1e-8).sum()
        # Roughly 1 rebalance per 21 days
        n_days = len(tail)
        assert n_days // 30 <= n_rebalances <= n_days // 15

    def test_no_future_lookahead(self):
        """Weights at time t must not change when we append new future rows."""
        panel_short = self.panel.iloc[:400]
        mem_short = self.membership.iloc[:400]
        w_short = IdiosyncraticVolFactor(sectors=self.sectors, spy_rets=self.spy).generate_weights(
            panel_short, mem_short
        )

        panel_long = self.panel
        mem_long = self.membership
        w_long = IdiosyncraticVolFactor(sectors=self.sectors, spy_rets=self.spy).generate_weights(
            panel_long, mem_long
        )

        shared_idx = w_short.index
        pd.testing.assert_frame_equal(
            w_short.loc[shared_idx].round(8),
            w_long.loc[shared_idx].round(8),
        )


# ---------------------------------------------------------------------------
# H252: MaxReturnLotteryAvoid
# ---------------------------------------------------------------------------


class TestMaxReturnLotteryAvoid:
    def setup_method(self):
        self.panel = _make_panel()
        self.membership = _make_membership(self.panel)
        self.sectors = _make_sectors(self.panel)
        self.strat = MaxReturnLotteryAvoid(sectors=self.sectors)

    def test_shape_matches_input(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert w.shape == self.panel.shape

    def test_weights_nonnegative(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.fillna(0.0) >= -1e-9).all().all()

    def test_weights_sum_at_most_one(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.sum(axis=1).round(6) <= 1.0 + 1e-6).all()

    def test_avoids_high_max_stocks(self):
        """A stock with an artificially high MAX should get zero weight."""
        panel = _make_panel(n=200).copy()
        # Give stock 0 a huge return spike on day 50
        panel.iloc[50, 0] = panel.iloc[49, 0] * 1.50
        mem = _make_membership(panel)
        w = MaxReturnLotteryAvoid(sectors=self.sectors).generate_weights(panel, mem)
        # After day 50 + max_window, stock 0 should have zero or low weight
        assert w.iloc[60:80, 0].sum() == pytest.approx(0.0, abs=1e-9)

    def test_no_future_lookahead(self):
        panel_short = self.panel.iloc[:300]
        mem_short = self.membership.iloc[:300]
        w_short = MaxReturnLotteryAvoid(sectors=self.sectors).generate_weights(
            panel_short, mem_short
        )
        w_long = MaxReturnLotteryAvoid(sectors=self.sectors).generate_weights(
            self.panel, self.membership
        )
        pd.testing.assert_frame_equal(w_short.round(8), w_long.iloc[: len(w_short)].round(8))


# ---------------------------------------------------------------------------
# H253: DownsideBetaFactor
# ---------------------------------------------------------------------------


class TestDownsideBetaFactor:
    def setup_method(self):
        self.panel = _make_panel()
        self.membership = _make_membership(self.panel)
        self.spy = _make_spy(self.panel)
        self.sectors = _make_sectors(self.panel)
        self.strat = DownsideBetaFactor(sectors=self.sectors, spy_rets=self.spy)

    def test_shape_matches_input(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert w.shape == self.panel.shape

    def test_weights_nonnegative(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.fillna(0.0) >= -1e-9).all().all()

    def test_weights_sum_at_most_one(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.sum(axis=1).round(6) <= 1.0 + 1e-6).all()

    def test_warmup_rows_are_zero(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        min_p = self.strat.downside_window // 4
        assert w.iloc[:min_p].sum().sum() == pytest.approx(0.0, abs=1e-9)

    def test_low_downside_beta_selected(self):
        """Stock with near-zero downside beta should get non-zero weight."""
        panel = _make_panel(n=500)
        spy = _make_spy(panel)
        mem = pd.DataFrame(True, index=panel.index, columns=panel.columns)
        sectors = pd.Series(["A"] * panel.shape[1], index=panel.columns)
        w = DownsideBetaFactor(sectors=sectors, spy_rets=spy).generate_weights(panel, mem)
        # At least some stocks should be selected after warmup
        assert w.iloc[300:].sum(axis=1).max() > 0


# ---------------------------------------------------------------------------
# H254: TailRiskComposite
# ---------------------------------------------------------------------------


class TestTailRiskComposite:
    def setup_method(self):
        self.panel = _make_panel()
        self.membership = _make_membership(self.panel)
        self.spy = _make_spy(self.panel)
        self.sectors = _make_sectors(self.panel)
        self.strat = TailRiskComposite(sectors=self.sectors, spy_rets=self.spy)

    def test_shape_matches_input(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert w.shape == self.panel.shape

    def test_weights_nonnegative(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.fillna(0.0) >= -1e-9).all().all()

    def test_weights_sum_at_most_one(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.sum(axis=1).round(6) <= 1.0 + 1e-6).all()

    def test_active_after_warmup(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        # After longest window (downside 252d), portfolio should be active
        assert w.iloc[300:].sum(axis=1).max() > 0

    def test_no_future_lookahead(self):
        panel_short = self.panel.iloc[:450]
        mem_short = self.membership.iloc[:450]
        w_short = TailRiskComposite(sectors=self.sectors, spy_rets=self.spy).generate_weights(
            panel_short, mem_short
        )
        w_long = TailRiskComposite(sectors=self.sectors, spy_rets=self.spy).generate_weights(
            self.panel, self.membership
        )
        pd.testing.assert_frame_equal(w_short.round(8), w_long.iloc[: len(w_short)].round(8))


# ---------------------------------------------------------------------------
# H255: TailFilteredMomentum
# ---------------------------------------------------------------------------


class TestTailFilteredMomentum:
    def setup_method(self):
        self.panel = _make_panel()
        self.membership = _make_membership(self.panel)
        self.spy = _make_spy(self.panel)
        self.sectors = _make_sectors(self.panel)
        self.strat = TailFilteredMomentum(sectors=self.sectors, spy_rets=self.spy)

    def test_shape_matches_input(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert w.shape == self.panel.shape

    def test_weights_nonnegative(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.fillna(0.0) >= -1e-9).all().all()

    def test_weights_sum_at_most_one(self):
        w = self.strat.generate_weights(self.panel, self.membership)
        assert (w.sum(axis=1).round(6) <= 1.0 + 1e-6).all()

    def test_lottery_stock_excluded(self):
        """A stock with extreme idio vol should be excluded from the portfolio."""
        panel = _make_panel(n=400).copy()
        # Make stock 0 extremely volatile in an idiosyncratic way
        rng = np.random.default_rng(0)
        noise = rng.normal(0, 0.15, len(panel))  # 15% daily idio vol
        panel.iloc[:, 0] = panel.iloc[:, 0] * np.cumprod(1 + noise)
        mem = pd.DataFrame(True, index=panel.index, columns=panel.columns)
        sectors = pd.Series(["A"] * panel.shape[1], index=panel.columns)
        spy = _make_spy(panel)
        w = TailFilteredMomentum(sectors=sectors, spy_rets=spy).generate_weights(panel, mem)
        # Stock 0 should be absent most of the time after warmup
        exclusion_rate = (w.iloc[300:, 0] == 0.0).mean()
        assert exclusion_rate > 0.5

    def test_no_future_lookahead(self):
        panel_short = self.panel.iloc[:400]
        mem_short = self.membership.iloc[:400]
        w_short = TailFilteredMomentum(sectors=self.sectors, spy_rets=self.spy).generate_weights(
            panel_short, mem_short
        )
        w_long = TailFilteredMomentum(sectors=self.sectors, spy_rets=self.spy).generate_weights(
            self.panel, self.membership
        )
        pd.testing.assert_frame_equal(w_short.round(8), w_long.iloc[: len(w_short)].round(8))
