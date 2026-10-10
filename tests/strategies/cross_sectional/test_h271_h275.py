"""Tests for H271-H275: Volume/Momentum Lifecycle batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.cross_sectional import (
    AttentionReversalScreen,
    EarlyStageMomentum,
    MomentumVolatilityScore,
    PriceBreakoutMomentum,
    VolatilitySurgeReturnPremium,
)


def _make_panel(n_days: int = 400, n_stocks: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2005-01-03", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, (n_days, n_stocks)), axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_membership(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=panel.index, columns=panel.columns)


def _make_fund(panel: pd.DataFrame, scale: float = 1e9, seed: int = 1) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    vals = np.abs(rng.normal(scale, scale * 0.3, panel.shape))
    return pd.DataFrame(vals, index=panel.index, columns=panel.columns)


def _make_sectors(panel: pd.DataFrame) -> pd.Series:
    n = len(panel.columns)
    secs = ["Tech", "Health", "Finance", "Energy", "Consumer"]
    return pd.Series([secs[i % len(secs)] for i in range(n)], index=panel.columns)


# ---------------------------------------------------------------------------
# H271: EarlyStageMomentum
# ---------------------------------------------------------------------------


def test_early_stage_momentum_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = EarlyStageMomentum(sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_early_stage_momentum_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = EarlyStageMomentum(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_early_stage_momentum_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = EarlyStageMomentum(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_early_stage_momentum_warmup_zero():
    # With 80 days and 252-day mom window, no weights should be generated
    panel = _make_panel(n_days=80)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = EarlyStageMomentum(sectors, mom_window=252)
    w = strat.generate_weights(panel, mem)
    assert w.sum().sum() == 0.0


# ---------------------------------------------------------------------------
# H272: VolatilitySurgeReturnPremium
# ---------------------------------------------------------------------------


def test_vol_surge_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = VolatilitySurgeReturnPremium(sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_vol_surge_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = VolatilitySurgeReturnPremium(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_vol_surge_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = VolatilitySurgeReturnPremium(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_vol_surge_excludes_negative_return_stocks():
    # Make all stocks have negative 21-day returns — no stocks flagged
    n_days, n_stocks = 300, 10
    dates = pd.date_range("2005-01-03", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    # Monotonically declining prices
    prices = np.ones((n_days, n_stocks)) * 100
    for d in range(1, n_days):
        prices[d] = prices[d - 1] * 0.999
    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    mem = pd.DataFrame(True, index=dates, columns=tickers)
    sectors = pd.Series(["Tech"] * n_stocks, index=tickers)
    strat = VolatilitySurgeReturnPremium(sectors, vol_ratio_threshold=0.01)
    w = strat.generate_weights(panel, mem)
    # All returns are negative, so no stocks should be flagged
    assert w.sum().sum() == 0.0


# ---------------------------------------------------------------------------
# H273: PriceBreakoutMomentum
# ---------------------------------------------------------------------------


def test_price_breakout_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = PriceBreakoutMomentum(sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_price_breakout_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = PriceBreakoutMomentum(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_price_breakout_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = PriceBreakoutMomentum(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_price_breakout_warmup_zero():
    # With 50 days and 63-day window, no breakout possible (rolling max requires full window)
    panel = _make_panel(n_days=50)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = PriceBreakoutMomentum(sectors, breakout_window=63)
    w = strat.generate_weights(panel, mem)
    assert w.sum().sum() == 0.0


# ---------------------------------------------------------------------------
# H274: AttentionReversalScreen
# ---------------------------------------------------------------------------


def test_attention_reversal_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 10e9, 2)
    strat = AttentionReversalScreen(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_attention_reversal_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 10e9, 2)
    strat = AttentionReversalScreen(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_attention_reversal_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 10e9, 2)
    strat = AttentionReversalScreen(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


# ---------------------------------------------------------------------------
# H275: MomentumVolatilityScore
# ---------------------------------------------------------------------------


def test_momentum_vol_score_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = MomentumVolatilityScore(sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_momentum_vol_score_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = MomentumVolatilityScore(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_momentum_vol_score_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = MomentumVolatilityScore(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_momentum_vol_score_warmup_zero():
    panel = _make_panel(n_days=100)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = MomentumVolatilityScore(sectors, mom_window=252)
    w = strat.generate_weights(panel, mem)
    assert w.sum().sum() == 0.0
