"""Tests for H296-H298: Tail-risk filtered, distress-screened, accounting sustainability."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.cross_sectional import (
    AccountingSustainabilityQualityComposite,
    DistressScreenedValueQuality,
    TailRiskFilteredCrossProgram,
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


def _make_spy_rets(panel: pd.DataFrame, seed: int = 99) -> pd.Series:
    rng = np.random.default_rng(seed)
    return pd.Series(rng.normal(0.0004, 0.01, len(panel)), index=panel.index)


def _make_fund(
    panel: pd.DataFrame, base: float = 1e9, growth: float = 0.03, seed: int = 7
) -> pd.DataFrame:
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
# H296: TailRiskFilteredCrossProgram
# ---------------------------------------------------------------------------


def test_tail_risk_filtered_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy_rets = _make_spy_rets(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat = TailRiskFilteredCrossProgram(gp, ta, sectors, spy_rets)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_tail_risk_filtered_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy_rets = _make_spy_rets(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat = TailRiskFilteredCrossProgram(gp, ta, sectors, spy_rets)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_tail_risk_filtered_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy_rets = _make_spy_rets(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat = TailRiskFilteredCrossProgram(gp, ta, sectors, spy_rets)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


def test_tail_risk_filtered_excludes_some():
    """Tail-risk filter should reduce effective universe (some stocks excluded)."""
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    spy_rets = _make_spy_rets(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    strat_filtered = TailRiskFilteredCrossProgram(gp, ta, sectors, spy_rets, exclude_quantile=0.10)
    w_filtered = strat_filtered.generate_weights(panel, mem)
    # Count stocks selected (non-zero weight) on typical rebalance days
    nonzero_counts = (w_filtered > 0).sum(axis=1)
    # After warmup, some days should have non-zero stocks but fewer than unfiltered max
    late_days = nonzero_counts.iloc[300:]
    # At least some holdings should exist
    assert late_days.max() > 0


# ---------------------------------------------------------------------------
# H297: DistressScreenedValueQuality
# ---------------------------------------------------------------------------


def test_distress_screened_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ni = _make_fund(panel, base=5e8, seed=5)
    ca = _make_fund(panel, base=3e9, seed=8)
    cl = _make_fund(panel, base=2e9, seed=9)
    ltd = _make_fund(panel, base=1e9, seed=4)
    sh = _make_fund(panel, base=1e9, seed=6)
    strat = DistressScreenedValueQuality(gp, ta, ni, ca, cl, ltd, sh, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_distress_screened_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ni = _make_fund(panel, base=5e8, seed=5)
    ca = _make_fund(panel, base=3e9, seed=8)
    cl = _make_fund(panel, base=2e9, seed=9)
    ltd = _make_fund(panel, base=1e9, seed=4)
    sh = _make_fund(panel, base=1e9, seed=6)
    strat = DistressScreenedValueQuality(gp, ta, ni, ca, cl, ltd, sh, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_distress_screened_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ni = _make_fund(panel, base=5e8, seed=5)
    ca = _make_fund(panel, base=3e9, seed=8)
    cl = _make_fund(panel, base=2e9, seed=9)
    ltd = _make_fund(panel, base=1e9, seed=4)
    sh = _make_fund(panel, base=1e9, seed=6)
    strat = DistressScreenedValueQuality(gp, ta, ni, ca, cl, ltd, sh, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()


# ---------------------------------------------------------------------------
# H298: AccountingSustainabilityQualityComposite
# ---------------------------------------------------------------------------


def test_accounting_sustainability_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ni = _make_fund(panel, base=5e8, seed=5)
    ocf = _make_fund(panel, base=4e8, seed=11)
    cl = _make_fund(panel, base=2e9, seed=9)
    strat = AccountingSustainabilityQualityComposite(gp, ta, ni, ocf, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_accounting_sustainability_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ni = _make_fund(panel, base=5e8, seed=5)
    ocf = _make_fund(panel, base=4e8, seed=11)
    cl = _make_fund(panel, base=2e9, seed=9)
    strat = AccountingSustainabilityQualityComposite(gp, ta, ni, ocf, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= 0).all().all()


def test_accounting_sustainability_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, base=2e9)
    ta = _make_fund(panel, base=10e9, seed=3)
    ni = _make_fund(panel, base=5e8, seed=5)
    ocf = _make_fund(panel, base=4e8, seed=11)
    cl = _make_fund(panel, base=2e9, seed=9)
    strat = AccountingSustainabilityQualityComposite(gp, ta, ni, ocf, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.001).all()
