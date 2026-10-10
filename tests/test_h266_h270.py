"""Tests for H266-H270: Intangible Quality (Gross Margin Proxies) batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.cross_sectional import (
    GrossMarginChangeFactor,
    GrossMarginIntensityFactor,
    IntangibleQualityComposite,
    IntangibleValueScreen,
    KnowledgeReturnProxy,
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
# H266: GrossMarginIntensityFactor
# ---------------------------------------------------------------------------


def test_gm_intensity_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    strat = GrossMarginIntensityFactor(gp, rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_gm_intensity_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    strat = GrossMarginIntensityFactor(gp, rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_gm_intensity_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    strat = GrossMarginIntensityFactor(gp, rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-6).all()


def test_gm_intensity_selects_high_margin():
    """Stocks with high gross margin should appear in the selection."""
    rng = np.random.default_rng(33)
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

    rev = pd.DataFrame(np.full(panel.shape, 4e9), index=panel.index, columns=panel.columns)
    gp_vals = np.full(panel.shape, 1e9)
    # Last 5 stocks: 90% gross margin (very high)
    gp_vals[:, -5:] = 3.6e9
    gp = pd.DataFrame(gp_vals, index=panel.index, columns=panel.columns)

    strat = GrossMarginIntensityFactor(gp, rev, sectors)
    w = strat.generate_weights(panel, mem)
    last = w.iloc[-1]
    # At least some of the high-margin stocks should have positive weight
    assert last.iloc[-5:].sum() > 0, "High gross margin stocks should be selected"


# ---------------------------------------------------------------------------
# H267: GrossMarginChangeFactor
# ---------------------------------------------------------------------------


def test_gm_change_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    strat = GrossMarginChangeFactor(gp, rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_gm_change_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    strat = GrossMarginChangeFactor(gp, rev, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_gm_change_warmup_zero():
    panel = _make_panel(n_days=100)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    strat = GrossMarginChangeFactor(gp, rev, sectors, window=252)
    w = strat.generate_weights(panel, mem)
    assert w.sum(axis=1).sum() == 0.0


# ---------------------------------------------------------------------------
# H268: KnowledgeReturnProxy
# ---------------------------------------------------------------------------


def test_krp_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 5e9, 2)
    strat = KnowledgeReturnProxy(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_krp_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 5e9, 2)
    strat = KnowledgeReturnProxy(gp, ta, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_krp_top_decile():
    """With 10% quantile, selected stocks should be at most 10% of universe."""
    panel = _make_panel(n_stocks=50)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 5e9, 2)
    strat = KnowledgeReturnProxy(gp, ta, sectors, within_sector_quantile=0.10)
    w = strat.generate_weights(panel, mem)
    last = w.iloc[-1]
    n_selected = (last > 0).sum()
    assert n_selected <= 10, f"Expected at most 10 stocks selected, got {n_selected}"


# ---------------------------------------------------------------------------
# H269: IntangibleValueScreen
# ---------------------------------------------------------------------------


def test_ivs_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    ta = _make_fund(panel, 5e9, 3)
    be = _make_fund(panel, 2e9, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    shs = _make_fund(panel, 1e8, 7)
    strat = IntangibleValueScreen(gp, rev, ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_ivs_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    ta = _make_fund(panel, 5e9, 3)
    be = _make_fund(panel, 2e9, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    shs = _make_fund(panel, 1e8, 7)
    strat = IntangibleValueScreen(gp, rev, ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_ivs_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    ta = _make_fund(panel, 5e9, 3)
    be = _make_fund(panel, 2e9, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    shs = _make_fund(panel, 1e8, 7)
    strat = IntangibleValueScreen(gp, rev, ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-6).all()


# ---------------------------------------------------------------------------
# H270: IntangibleQualityComposite
# ---------------------------------------------------------------------------


def test_iqc_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    ni = _make_fund(panel, 3e8, 3)
    ocf = _make_fund(panel, 4e8, 4)
    strat = IntangibleQualityComposite(gp, rev, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_iqc_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    ni = _make_fund(panel, 3e8, 3)
    ocf = _make_fund(panel, 4e8, 4)
    strat = IntangibleQualityComposite(gp, rev, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_iqc_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    ni = _make_fund(panel, 3e8, 3)
    ocf = _make_fund(panel, 4e8, 4)
    strat = IntangibleQualityComposite(gp, rev, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w.sum(axis=1) <= 1.0 + 1e-6).all()


def test_iqc_warmup_zero():
    panel = _make_panel(n_days=100)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    rev = _make_fund(panel, 4e9, 2)
    ni = _make_fund(panel, 3e8, 3)
    ocf = _make_fund(panel, 4e8, 4)
    strat = IntangibleQualityComposite(gp, rev, ni, ocf, sectors, window=252)
    w = strat.generate_weights(panel, mem)
    assert w.sum(axis=1).sum() == 0.0
