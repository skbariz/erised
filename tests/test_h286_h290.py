"""Tests for H286-H290: Capital Discipline batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.cross_sectional import (
    AssetGrowthPenalty,
    CapitalEfficiencyQualityComposite,
    InvestmentEfficiencyRatio,
    LowCapexGrowthFactor,
    ROICCapexDiscipline,
)


def _make_panel(n_days: int = 600, n_stocks: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2004-01-02", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    prices = 100 * np.exp(np.cumsum(rng.normal(0.0004, 0.015, (n_days, n_stocks)), axis=0))
    return pd.DataFrame(prices, index=dates, columns=tickers)


def _make_membership(panel: pd.DataFrame) -> pd.DataFrame:
    return pd.DataFrame(True, index=panel.index, columns=panel.columns)


def _make_sectors(panel: pd.DataFrame) -> pd.Series:
    n = len(panel.columns)
    secs = ["Tech", "Health", "Finance", "Energy", "Consumer"]
    return pd.Series([secs[i % len(secs)] for i in range(n)], index=panel.columns)


def _make_fund(panel: pd.DataFrame, base: float = 1e9, growth: float = 0.03, seed: int = 7) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_days, n_stocks = panel.shape
    quarters = n_days // 63 + 2
    qvals = base * np.cumprod(1 + rng.normal(growth, 0.08, (quarters, n_stocks)), axis=0)
    daily = np.full((n_days, n_stocks), np.nan)
    for q in range(quarters):
        start, end = q * 63, min((q + 1) * 63, n_days)
        if start < n_days:
            daily[start:end] = qvals[q]
    return pd.DataFrame(daily, index=panel.index, columns=panel.columns)


# ---------------------------------------------------------------------------
# H286: LowCapexGrowthFactor
# ---------------------------------------------------------------------------


def test_low_capex_growth_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, base=10e9)
    ca = _make_fund(panel, base=3e9, seed=8)
    strat = LowCapexGrowthFactor(ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_low_capex_growth_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, base=10e9)
    ca = _make_fund(panel, base=3e9, seed=8)
    strat = LowCapexGrowthFactor(ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_low_capex_growth_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, base=10e9)
    ca = _make_fund(panel, base=3e9, seed=8)
    strat = LowCapexGrowthFactor(ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H287: InvestmentEfficiencyRatio
# ---------------------------------------------------------------------------


def test_inv_efficiency_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    rev = _make_fund(panel, base=5e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    strat = InvestmentEfficiencyRatio(rev, ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_inv_efficiency_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    rev = _make_fund(panel, base=5e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    strat = InvestmentEfficiencyRatio(rev, ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_inv_efficiency_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    rev = _make_fund(panel, base=5e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    strat = InvestmentEfficiencyRatio(rev, ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H288: ROICCapexDiscipline
# ---------------------------------------------------------------------------


def test_roic_capex_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    cl = _make_fund(panel, base=2e9, seed=9)
    strat = ROICCapexDiscipline(ni, ta, ca, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_roic_capex_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    cl = _make_fund(panel, base=2e9, seed=9)
    strat = ROICCapexDiscipline(ni, ta, ca, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_roic_capex_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    cl = _make_fund(panel, base=2e9, seed=9)
    strat = ROICCapexDiscipline(ni, ta, ca, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H289: AssetGrowthPenalty
# ---------------------------------------------------------------------------


def test_asset_growth_penalty_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, base=10e9)
    ca = _make_fund(panel, base=3e9, seed=8)
    strat = AssetGrowthPenalty(ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_asset_growth_penalty_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, base=10e9)
    ca = _make_fund(panel, base=3e9, seed=8)
    strat = AssetGrowthPenalty(ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_asset_growth_penalty_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, base=10e9)
    ca = _make_fund(panel, base=3e9, seed=8)
    strat = AssetGrowthPenalty(ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H290: CapitalEfficiencyQualityComposite
# ---------------------------------------------------------------------------


def test_capital_efficiency_composite_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    rev = _make_fund(panel, base=5e9, seed=2)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    cl = _make_fund(panel, base=2e9, seed=9)
    strat = CapitalEfficiencyQualityComposite(ni, rev, ta, ca, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_capital_efficiency_composite_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    rev = _make_fund(panel, base=5e9, seed=2)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    cl = _make_fund(panel, base=2e9, seed=9)
    strat = CapitalEfficiencyQualityComposite(ni, rev, ta, ca, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_capital_efficiency_composite_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    rev = _make_fund(panel, base=5e9, seed=2)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    cl = _make_fund(panel, base=2e9, seed=9)
    strat = CapitalEfficiencyQualityComposite(ni, rev, ta, ca, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()
