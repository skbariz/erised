"""Tests for H291-H295: Execution-Aware / Low-Turnover batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.strategies.cross_sectional import (
    CostAwareMomentumLongHold,
    ExecutionEfficientComposite,
    LowTurnoverQualityFactor,
    LowTurnoverValueFactor,
    TurnoverCappedProfitabilityFactor,
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
# H291: LowTurnoverValueFactor
# ---------------------------------------------------------------------------


def test_low_turnover_value_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    sh = _make_fund(panel, base=1e9, seed=6)
    strat = LowTurnoverValueFactor(ni, sh, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_low_turnover_value_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    sh = _make_fund(panel, base=1e9, seed=6)
    strat = LowTurnoverValueFactor(ni, sh, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_low_turnover_value_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    sh = _make_fund(panel, base=1e9, seed=6)
    strat = LowTurnoverValueFactor(ni, sh, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


def test_low_turnover_value_fewer_trades():
    """Buy/hold band should produce fewer weight changes than monthly rebalance."""
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ni = _make_fund(panel, base=5e8, seed=5)
    sh = _make_fund(panel, base=1e9, seed=6)
    strat = LowTurnoverValueFactor(ni, sh, sectors)
    w = strat.generate_weights(panel, mem)
    # Count days with weight changes
    changes = (w.diff().abs().sum(axis=1) > 1e-8).sum()
    # Should have weight changes primarily on rebalance days (every 21 days)
    # Total possible days / 21 ≈ 28 rebalance days; changes should be ≤ n_days/21 + warmup
    assert changes <= len(panel) // 10  # conservative: much less than daily changes


# ---------------------------------------------------------------------------
# H292: LowTurnoverQualityFactor
# ---------------------------------------------------------------------------


def test_low_turnover_quality_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat = LowTurnoverQualityFactor(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_low_turnover_quality_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat = LowTurnoverQualityFactor(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_low_turnover_quality_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat = LowTurnoverQualityFactor(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H293: CostAwareMomentumLongHold
# ---------------------------------------------------------------------------


def test_cost_aware_momentum_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = CostAwareMomentumLongHold(sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_cost_aware_momentum_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = CostAwareMomentumLongHold(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_cost_aware_momentum_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    strat = CostAwareMomentumLongHold(sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H294: TurnoverCappedProfitabilityFactor
# ---------------------------------------------------------------------------


def test_turnover_capped_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat = TurnoverCappedProfitabilityFactor(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_turnover_capped_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat = TurnoverCappedProfitabilityFactor(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_turnover_capped_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat = TurnoverCappedProfitabilityFactor(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H295: ExecutionEfficientComposite
# ---------------------------------------------------------------------------


def test_exec_efficient_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    rev = _make_fund(panel, base=5e9, seed=2)
    strat = ExecutionEfficientComposite(gp, ta, ca, rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_exec_efficient_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    rev = _make_fund(panel, base=5e9, seed=2)
    strat = ExecutionEfficientComposite(gp, ta, ca, rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_exec_efficient_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ca = _make_fund(panel, base=3e9, seed=8)
    rev = _make_fund(panel, base=5e9, seed=2)
    strat = ExecutionEfficientComposite(gp, ta, ca, rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()
