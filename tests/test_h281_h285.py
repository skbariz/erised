"""Tests for H281-H285: Fundamental Growth batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.cross_sectional import (
    FundamentalMomentumComposite,
    GrossProfitImprovementFactor,
    LowCoverageMomentumProxy,
    OCFGrowthFactor,
    RevenueGrowthFactor,
)


def _make_panel(n_days: int = 600, n_stocks: int = 20, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    dates = pd.date_range("2004-01-02", periods=n_days, freq="B")
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


def _make_fund_panel(
    panel: pd.DataFrame,
    base_val: float = 1e9,
    growth_rate: float = 0.05,
    seed: int = 7,
) -> pd.DataFrame:
    """Simulate quarterly-updated fundamental panel (forward-filled like EDGAR)."""
    rng = np.random.default_rng(seed)
    n_days, n_stocks = panel.shape
    # Generate quarterly values
    quarters = n_days // 63 + 2
    qvals = base_val * np.cumprod(1 + rng.normal(growth_rate, 0.1, (quarters, n_stocks)), axis=0)
    # Expand to daily, updating every 63 days
    daily = np.full((n_days, n_stocks), np.nan)
    for q in range(quarters):
        start = q * 63
        end = min((q + 1) * 63, n_days)
        if start < n_days:
            daily[start:end] = qvals[q]
    return pd.DataFrame(daily, index=panel.index, columns=panel.columns)


# ---------------------------------------------------------------------------
# H281: GrossProfitImprovementFactor
# ---------------------------------------------------------------------------


def test_gp_improvement_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund_panel(panel)
    strat = GrossProfitImprovementFactor(gp, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_gp_improvement_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund_panel(panel)
    strat = GrossProfitImprovementFactor(gp, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_gp_improvement_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund_panel(panel)
    strat = GrossProfitImprovementFactor(gp, sectors)
    w = strat.generate_weights(panel, mem)
    active = w.sum(axis=1)
    assert (active <= 1.001).all()


def test_gp_improvement_warmup_zero():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund_panel(panel)
    strat = GrossProfitImprovementFactor(gp, sectors, yoy_window=252)
    w = strat.generate_weights(panel, mem)
    assert w.iloc[:252].sum().sum() == 0.0


# ---------------------------------------------------------------------------
# H282: RevenueGrowthFactor
# ---------------------------------------------------------------------------


def test_rev_growth_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    rev = _make_fund_panel(panel, base_val=5e9)
    strat = RevenueGrowthFactor(rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_rev_growth_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    rev = _make_fund_panel(panel, base_val=5e9)
    strat = RevenueGrowthFactor(rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_rev_growth_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    rev = _make_fund_panel(panel, base_val=5e9)
    strat = RevenueGrowthFactor(rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


def test_rev_growth_warmup_zero():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    rev = _make_fund_panel(panel, base_val=5e9)
    strat = RevenueGrowthFactor(rev, sectors, yoy_window=252)
    w = strat.generate_weights(panel, mem)
    assert w.iloc[:252].sum().sum() == 0.0


# ---------------------------------------------------------------------------
# H283: OCFGrowthFactor
# ---------------------------------------------------------------------------


def test_ocf_growth_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ocf = _make_fund_panel(panel, base_val=8e8, growth_rate=0.04)
    strat = OCFGrowthFactor(ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_ocf_growth_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ocf = _make_fund_panel(panel, base_val=8e8)
    strat = OCFGrowthFactor(ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_ocf_growth_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ocf = _make_fund_panel(panel, base_val=8e8)
    strat = OCFGrowthFactor(ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


def test_ocf_growth_warmup_zero():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ocf = _make_fund_panel(panel, base_val=8e8)
    strat = OCFGrowthFactor(ocf, sectors, yoy_window=252)
    w = strat.generate_weights(panel, mem)
    assert w.iloc[:252].sum().sum() == 0.0


# ---------------------------------------------------------------------------
# H284: LowCoverageMomentumProxy
# ---------------------------------------------------------------------------


def test_low_coverage_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = LowCoverageMomentumProxy(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_low_coverage_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = LowCoverageMomentumProxy(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_low_coverage_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy = _make_spy(panel)
    strat = LowCoverageMomentumProxy(spy, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H285: FundamentalMomentumComposite
# ---------------------------------------------------------------------------


def test_fund_mom_composite_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund_panel(panel)
    ni = _make_fund_panel(panel, base_val=3e8, seed=13)
    strat = FundamentalMomentumComposite(gp, ni, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_fund_mom_composite_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund_panel(panel)
    ni = _make_fund_panel(panel, base_val=3e8, seed=13)
    strat = FundamentalMomentumComposite(gp, ni, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_fund_mom_composite_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund_panel(panel)
    ni = _make_fund_panel(panel, base_val=3e8, seed=13)
    strat = FundamentalMomentumComposite(gp, ni, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


def test_fund_mom_composite_warmup_zero():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund_panel(panel)
    ni = _make_fund_panel(panel, base_val=3e8, seed=13)
    strat = FundamentalMomentumComposite(gp, ni, sectors, yoy_window=252)
    w = strat.generate_weights(panel, mem)
    assert w.iloc[:252].sum().sum() == 0.0
