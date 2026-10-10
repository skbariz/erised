"""Tests for H256-H260: Distress Avoidance batch."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.cross_sectional import (
    CHSMarketDistress,
    DistressFilteredValue,
    DistressTrendFactor,
    LowDistressQualityMomentum,
    OhlsonOScoreDistress,
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
# H256: OhlsonOScoreDistress
# ---------------------------------------------------------------------------


def test_oscore_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ca = _make_fund(panel, 1e9, 3)
    cl = _make_fund(panel, 8e8, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    strat = OhlsonOScoreDistress(ta, be, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_oscore_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ca = _make_fund(panel, 1e9, 3)
    cl = _make_fund(panel, 8e8, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    strat = OhlsonOScoreDistress(ta, be, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_oscore_row_sum_leq_one():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ca = _make_fund(panel, 1e9, 3)
    cl = _make_fund(panel, 8e8, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    strat = OhlsonOScoreDistress(ta, be, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    row_sums = w.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-6).all()


def test_oscore_avoids_high_distress_stocks():
    """Bottom-quintile selection should avoid stocks with very high leverage."""
    rng = np.random.default_rng(99)
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
    be_vals = np.full(panel.shape, 2e9)
    # First 5 stocks: book equity near zero → high leverage
    be_vals[:, :5] = 1e7
    be = pd.DataFrame(be_vals, index=panel.index, columns=panel.columns)
    ca = pd.DataFrame(np.full(panel.shape, 1e9), index=panel.index, columns=panel.columns)
    cl = pd.DataFrame(np.full(panel.shape, 8e8), index=panel.index, columns=panel.columns)
    ni = pd.DataFrame(np.full(panel.shape, 3e8), index=panel.index, columns=panel.columns)
    ocf = pd.DataFrame(np.full(panel.shape, 4e8), index=panel.index, columns=panel.columns)

    strat = OhlsonOScoreDistress(ta, be, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    # High-leverage (high distress) stocks should have zero weight in last row
    last_row = w.iloc[-1]
    assert last_row.iloc[:5].sum() < 1e-6, "High-distress stocks should be excluded"


# ---------------------------------------------------------------------------
# H257: CHSMarketDistress
# ---------------------------------------------------------------------------


def test_chs_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ni = _make_fund(panel, 3e8, 3)
    ocf = _make_fund(panel, 4e8, 4)
    shs = _make_fund(panel, 1e8, 5)
    strat = CHSMarketDistress(ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_chs_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ni = _make_fund(panel, 3e8, 3)
    ocf = _make_fund(panel, 4e8, 4)
    shs = _make_fund(panel, 1e8, 5)
    strat = CHSMarketDistress(ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_chs_short_panel():
    """CHS can generate weights with a short panel (non-vol components don't need 252d)."""
    panel = _make_panel(n_days=100)
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ni = _make_fund(panel, 3e8, 3)
    ocf = _make_fund(panel, 4e8, 4)
    shs = _make_fund(panel, 1e8, 5)
    strat = CHSMarketDistress(ta, be, ni, ocf, shs, sectors, vol_window=252)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape
    assert (w >= -1e-9).all().all()


# ---------------------------------------------------------------------------
# H258: DistressTrendFactor
# ---------------------------------------------------------------------------


def test_distress_trend_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ca = _make_fund(panel, 1e9, 3)
    cl = _make_fund(panel, 8e8, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    strat = DistressTrendFactor(ta, be, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_distress_trend_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ca = _make_fund(panel, 1e9, 3)
    cl = _make_fund(panel, 8e8, 4)
    ni = _make_fund(panel, 3e8, 5)
    ocf = _make_fund(panel, 4e8, 6)
    strat = DistressTrendFactor(ta, be, ca, cl, ni, ocf, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


# ---------------------------------------------------------------------------
# H259: DistressFilteredValue
# ---------------------------------------------------------------------------


def test_distress_filtered_value_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ni = _make_fund(panel, 3e8, 3)
    ocf = _make_fund(panel, 4e8, 4)
    shs = _make_fund(panel, 1e8, 5)
    strat = DistressFilteredValue(ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_distress_filtered_value_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    ta = _make_fund(panel, 5e9)
    be = _make_fund(panel, 2e9, 2)
    ni = _make_fund(panel, 3e8, 3)
    ocf = _make_fund(panel, 4e8, 4)
    shs = _make_fund(panel, 1e8, 5)
    strat = DistressFilteredValue(ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


# ---------------------------------------------------------------------------
# H260: LowDistressQualityMomentum
# ---------------------------------------------------------------------------


def test_ldqm_shape():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 5e9, 2)
    be = _make_fund(panel, 2e9, 3)
    ni = _make_fund(panel, 3e8, 4)
    ocf = _make_fund(panel, 4e8, 5)
    shs = _make_fund(panel, 1e8, 6)
    strat = LowDistressQualityMomentum(gp, ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    assert w.shape == panel.shape


def test_ldqm_non_negative():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 5e9, 2)
    be = _make_fund(panel, 2e9, 3)
    ni = _make_fund(panel, 3e8, 4)
    ocf = _make_fund(panel, 4e8, 5)
    shs = _make_fund(panel, 1e8, 6)
    strat = LowDistressQualityMomentum(gp, ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    assert (w >= -1e-9).all().all()


def test_ldqm_row_sum():
    panel = _make_panel()
    mem = _make_membership(panel)
    sectors = _make_sectors(panel)
    gp = _make_fund(panel, 2e9)
    ta = _make_fund(panel, 5e9, 2)
    be = _make_fund(panel, 2e9, 3)
    ni = _make_fund(panel, 3e8, 4)
    ocf = _make_fund(panel, 4e8, 5)
    shs = _make_fund(panel, 1e8, 6)
    strat = LowDistressQualityMomentum(gp, ta, be, ni, ocf, shs, sectors)
    w = strat.generate_weights(panel, mem)
    row_sums = w.sum(axis=1)
    assert (row_sums <= 1.0 + 1e-6).all()
