"""Tests for H261-H265: Accounting Sustainability batch."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.strategies.cross_sectional import (
    AccrualReliabilityFactor,
    ConservativeCapexInvestment,
    LowNetOperatingAssets,
    SustainableEarningsComposite,
    SustainabilityFilteredProfitability,
)


def _make_panel(n_days: int = 300, n_stocks: int = 20, seed: int = 42) -> pd.DataFrame:
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
# H261: LowNetOperatingAssets
# ---------------------------------------------------------------------------


def test_lnoa_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    cl = _make_fund(panel, 8e8, 2)
    strat = LowNetOperatingAssets(ta, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_lnoa_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    cl = _make_fund(panel, 8e8, 2)
    strat = LowNetOperatingAssets(ta, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_lnoa_row_sum_leq_one():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    cl = _make_fund(panel, 8e8, 2)
    strat = LowNetOperatingAssets(ta, cl, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-6).all()


def test_lnoa_avoids_high_noa():
    """Stocks with high NOA should not be selected."""
    rng = np.random.default_rng(77)
    n_days, n_stocks = 300, 30
    dates = pd.date_range("2005-01-03", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    panel = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n_days, n_stocks)), axis=0)),
        index=dates,
        columns=tickers,
    )
    mem = pd.DataFrame(True, index=panel.index, columns=panel.columns)
    sectors = pd.Series(["Tech"] * n_stocks, index=tickers)

    ta = pd.DataFrame(np.full(panel.shape, 5e9), index=panel.index, columns=panel.columns)
    cl_vals = np.full(panel.shape, 8e8)
    # First 5 stocks: very low current liabilities → high NOA
    cl_vals[:, :5] = 1e6
    cl = pd.DataFrame(cl_vals, index=panel.index, columns=panel.columns)

    strat = LowNetOperatingAssets(ta, cl, sectors)
    w = strat.generate_weights(panel, mem)
    last = w.iloc[-1]
    assert last.iloc[:5].sum() < 1e-6, "High-NOA stocks should be excluded"


# ---------------------------------------------------------------------------
# H262: AccrualReliabilityFactor
# ---------------------------------------------------------------------------


def test_accrual_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    ni = _make_fund(panel, 3e8, 2)
    ocf = _make_fund(panel, 4e8, 3)
    strat = AccrualReliabilityFactor(ta, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_accrual_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    ni = _make_fund(panel, 3e8, 2)
    ocf = _make_fund(panel, 4e8, 3)
    strat = AccrualReliabilityFactor(ta, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_accrual_selects_low_accrual_stocks():
    """Stocks with NI much higher than OCF (high accruals) should be avoided."""
    rng = np.random.default_rng(55)
    n_days, n_stocks = 400, 30
    dates = pd.date_range("2005-01-03", periods=n_days, freq="B")
    tickers = [f"T{i:02d}" for i in range(n_stocks)]
    panel = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0.0003, 0.012, (n_days, n_stocks)), axis=0)),
        index=dates,
        columns=tickers,
    )
    mem = pd.DataFrame(True, index=panel.index, columns=panel.columns)
    sectors = pd.Series(["Tech"] * n_stocks, index=tickers)

    ta = pd.DataFrame(np.full(panel.shape, 5e9), index=panel.index, columns=panel.columns)
    ni_vals = np.full(panel.shape, 3e8)
    ocf_vals = np.full(panel.shape, 3e8)
    # First 5 stocks: NI >> OCF (high accruals = earnings not cash-backed)
    ni_vals[:, :5] = 1e9
    ocf_vals[:, :5] = 1e7
    ni = pd.DataFrame(ni_vals, index=panel.index, columns=panel.columns)
    ocf = pd.DataFrame(ocf_vals, index=panel.index, columns=panel.columns)

    strat = AccrualReliabilityFactor(ta, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    last = w.iloc[-1]
    assert last.iloc[:5].sum() < 1e-6, "High-accrual stocks should be excluded"


# ---------------------------------------------------------------------------
# H263: ConservativeCapexInvestment
# ---------------------------------------------------------------------------


def test_capex_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    ca = _make_fund(panel, 1e9, 2)
    strat = ConservativeCapexInvestment(ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_capex_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    ca = _make_fund(panel, 1e9, 2)
    strat = ConservativeCapexInvestment(ta, ca, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_capex_warmup_zero():
    panel = _make_panel(n_days=100)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    ca = _make_fund(panel, 1e9, 2)
    strat = ConservativeCapexInvestment(ta, ca, sectors, window=252)
    w = strat.generate_weights(panel, mem)
    assert w.sum(axis=1).sum() == 0.0


# ---------------------------------------------------------------------------
# H264: SustainableEarningsComposite
# ---------------------------------------------------------------------------


def test_sec_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    ca = _make_fund(panel, 1e9, 2)
    cl = _make_fund(panel, 8e8, 3)
    ni = _make_fund(panel, 3e8, 4)
    ocf = _make_fund(panel, 4e8, 5)
    strat = SustainableEarningsComposite(ta, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_sec_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    ca = _make_fund(panel, 1e9, 2)
    cl = _make_fund(panel, 8e8, 3)
    ni = _make_fund(panel, 3e8, 4)
    ocf = _make_fund(panel, 4e8, 5)
    strat = SustainableEarningsComposite(ta, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_sec_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    ca = _make_fund(panel, 1e9, 2)
    cl = _make_fund(panel, 8e8, 3)
    ni = _make_fund(panel, 3e8, 4)
    ocf = _make_fund(panel, 4e8, 5)
    strat = SustainableEarningsComposite(ta, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# H265: SustainabilityFilteredProfitability
# ---------------------------------------------------------------------------


def test_sfp_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 5e9, 2)
    ca = _make_fund(panel, 1e9, 3)
    cl = _make_fund(panel, 8e8, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    strat = SustainabilityFilteredProfitability(gp, ta, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_sfp_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 5e9, 2)
    ca = _make_fund(panel, 1e9, 3)
    cl = _make_fund(panel, 8e8, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    strat = SustainabilityFilteredProfitability(gp, ta, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_sfp_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 5e9, 2)
    ca = _make_fund(panel, 1e9, 3)
    cl = _make_fund(panel, 8e8, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    strat = SustainabilityFilteredProfitability(gp, ta, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-6).all()
