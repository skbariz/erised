"""Tests for H276-H280: Skewness/Tail Risk batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.cross_sectional import (
    CoskewnessPortfolioPenalty,
    CrashSensitivityRegime,
    LeftTailLossFactor,
    NegativeSkewnessAvoidance,
    SkewAdjustedMomentum,
)


def _make_panel(n_days: int = 400, n_stocks: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, (n_days, n_stocks)), axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_membership(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=panel.index, columns=panel.columns)


def _make_spy(panel: pd.DataFrame, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    spy_prices = 100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, len(panel)), axis=0))
    return pd.Series(spy_prices, index=panel.index, name="SPY")


def _make_sectors(panel: pd.DataFrame) -> pd.Series:
    n = len(panel.columns)
    secs = ["Tech", "Health", "Finance", "Energy", "Consumer"]
    return pd.Series([secs[i % len(secs)] for i in range(n)], index=panel.columns)


# ---------------------------------------------------------------------------
# H276: NegativeSkewnessAvoidance
# ---------------------------------------------------------------------------


def test_neg_skew_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = NegativeSkewnessAvoidance(sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_neg_skew_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = NegativeSkewnessAvoidance(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_neg_skew_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = NegativeSkewnessAvoidance(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_neg_skew_warmup_zero():
    panel = _make_panel(n_days=50)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = NegativeSkewnessAvoidance(sectors, skew_window=252)
    w = strat.generate_weights(panel, mem)
    assert w.sum().sum() == 0.0


# ---------------------------------------------------------------------------
# H277: CoskewnessPortfolioPenalty
# ---------------------------------------------------------------------------


def test_coskewness_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = CoskewnessPortfolioPenalty(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_coskewness_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = CoskewnessPortfolioPenalty(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_coskewness_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = CoskewnessPortfolioPenalty(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_coskewness_warmup_zero():
    panel = _make_panel(n_days=50)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = CoskewnessPortfolioPenalty(spy, sectors, window=252)
    w = strat.generate_weights(panel, mem)
    assert w.sum().sum() == 0.0


# ---------------------------------------------------------------------------
# H278: LeftTailLossFactor
# ---------------------------------------------------------------------------


def test_cvar_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = LeftTailLossFactor(sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_cvar_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = LeftTailLossFactor(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_cvar_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = LeftTailLossFactor(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_cvar_selects_low_tail_loss():
    # Stock with artificially low CVaR (constant price = zero returns) vs high CVaR
    rng = np.random.default_rng(7)
    n_days, n_stocks = 300, 10
    dates = pd.date_range("2005-01-03", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.02, (n_days, n_stocks)), axis=0))
    # Make last 5 stocks have extreme daily losses
    prices[:, 5:] = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.10, (n_days, 5)), axis=0))
    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    mem = pd.DataFrame(True, index=dates, columns=tickers)
    sectors = pd.Series(["Tech"] * n_stocks, index=tickers)
    strat = LeftTailLossFactor(sectors, window=252)
    w = strat.generate_weights(panel, mem)
    after_warmup = w.iloc[260:]
    # First 5 stocks (low vol) should have more weight than last 5 (high vol, high CVaR)
    assert after_warmup[tickers[:5]].sum().sum() > after_warmup[tickers[5:]].sum().sum()


# ---------------------------------------------------------------------------
# H279: CrashSensitivityRegime
# ---------------------------------------------------------------------------


def test_crash_regime_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = CrashSensitivityRegime(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_crash_regime_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = CrashSensitivityRegime(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_crash_regime_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = CrashSensitivityRegime(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# H280: SkewAdjustedMomentum
# ---------------------------------------------------------------------------


def test_skew_adj_mom_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = SkewAdjustedMomentum(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_skew_adj_mom_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = SkewAdjustedMomentum(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_skew_adj_mom_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = SkewAdjustedMomentum(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_skew_adj_mom_warmup_zero():
    panel = _make_panel(n_days=100)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = SkewAdjustedMomentum(spy, sectors, mom_window=252)
    w = strat.generate_weights(panel, mem)
    assert w.sum().sum() == 0.0
