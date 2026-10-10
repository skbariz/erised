"""Tests for Batch 2 cross-sectional strategies (H155-H159).

Tests run on a small synthetic panel (no real data required).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from erised.strategies.cross_sectional import (
    DrawdownCircuitBreaker,
    JumpPenalizedMomentum,
    MomentumLongOnly,
    SmoothMomentum,
    SmoothVolTargetMomentum,
    VolScaledMomentum,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def synthetic_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Small panel: 10 tickers, 500 trading days."""
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2010-01-01", periods=500)
    tickers = [f"T{i:02d}" for i in range(10)]
    returns = rng.normal(0.0005, 0.015, size=(500, 10))
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    membership = pd.DataFrame(True, index=dates, columns=tickers)
    return panel, membership


@pytest.fixture
def trending_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel with one ticker on a smooth uptrend and one on a jumpy path."""
    dates = pd.bdate_range("2010-01-01", periods=500)
    tickers = [f"T{i:02d}" for i in range(10)]
    rng = np.random.default_rng(7)
    returns = rng.normal(0.0002, 0.008, size=(500, 10))
    # T00: smooth uptrend — many small positive days
    returns[:, 0] = np.abs(rng.normal(0.003, 0.002, size=500))
    # T01: same cumulative return but from a few large jumps
    jumpy = np.zeros(500)
    jumpy[::50] = 0.15  # large gains every 50 days
    returns[:, 1] = jumpy
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    membership = pd.DataFrame(True, index=dates, columns=tickers)
    return panel, membership


@pytest.fixture
def crash_panel() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Panel with a simulated crash to trigger drawdown circuit breaker."""
    rng = np.random.default_rng(99)
    dates = pd.bdate_range("2010-01-01", periods=500)
    tickers = [f"T{i:02d}" for i in range(10)]
    returns = rng.normal(0.001, 0.01, size=(500, 10))
    # Introduce a sharp crash in the middle
    returns[200:230, :] = -0.03
    prices = 100.0 * np.exp(np.cumsum(returns, axis=0))
    panel = pd.DataFrame(prices, index=dates, columns=tickers)
    membership = pd.DataFrame(True, index=dates, columns=tickers)
    return panel, membership


# ---------------------------------------------------------------------------
# H155: SmoothMomentum
# ---------------------------------------------------------------------------


def test_h155_weights_sum_leq_one(synthetic_panel):
    panel, membership = synthetic_panel
    strat = SmoothMomentum(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_h155_weights_non_negative(synthetic_panel):
    panel, membership = synthetic_panel
    strat = SmoothMomentum(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights >= -1e-9).all().all()


def test_h155_smooth_ticker_outranks_jumpy(trending_panel):
    """T00 (smooth uptrend) should receive higher weight than T01 (jumpy)."""
    panel, membership = trending_panel
    strat = SmoothMomentum(long_days=120, skip_days=5, top_quantile=0.5, monthly_rebalance=False)
    weights = strat.generate_weights(panel, membership)
    post_warmup = weights.iloc[130:]
    active = post_warmup.abs().sum(axis=1) > 0
    if active.any():
        t00_total = post_warmup.loc[active, "T00"].sum()
        t01_total = post_warmup.loc[active, "T01"].sum()
        assert t00_total >= t01_total, "Smooth ticker should get >= weight vs jumpy ticker"


def test_h155_monthly_rebalance_stable_intramonth(synthetic_panel):
    panel, membership = synthetic_panel
    strat = SmoothMomentum(long_days=60, skip_days=5, monthly_rebalance=True)
    weights = strat.generate_weights(panel, membership)
    months = weights.index.to_series().dt.to_period("M")
    for _month, grp in weights.groupby(months):
        first_row = grp.iloc[0]
        for _, row in grp.iterrows():
            pd.testing.assert_series_equal(row, first_row, check_names=False)


def test_h155_no_lookahead(synthetic_panel):
    panel, membership = synthetic_panel
    strat = SmoothMomentum(long_days=60, skip_days=5)
    w_full = strat.generate_weights(panel, membership)
    w_trunc = strat.generate_weights(panel.iloc[:301], membership.iloc[:301])
    date = panel.index[300]
    pd.testing.assert_series_equal(w_full.loc[date], w_trunc.loc[date], check_names=False)


# ---------------------------------------------------------------------------
# H156: JumpPenalizedMomentum
# ---------------------------------------------------------------------------


def test_h156_weights_sum_leq_one(synthetic_panel):
    panel, membership = synthetic_panel
    strat = JumpPenalizedMomentum(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_h156_weights_non_negative(synthetic_panel):
    panel, membership = synthetic_panel
    strat = JumpPenalizedMomentum(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights >= -1e-9).all().all()


def test_h156_signal_differs_from_h150(synthetic_panel):
    """Trimmed momentum should produce different scores than raw momentum."""
    panel, membership = synthetic_panel
    strat_h150 = MomentumLongOnly(long_days=60, skip_days=5, monthly_rebalance=False)
    strat_h156 = JumpPenalizedMomentum(long_days=60, skip_days=5, monthly_rebalance=False)
    w_h150 = strat_h150.generate_weights(panel, membership)
    w_h156 = strat_h156.generate_weights(panel, membership)
    # Weights should not be identical (signals are different)
    assert not w_h150.equals(w_h156), "H156 trimmed signal must differ from H150 raw signal"


def test_h156_monthly_rebalance_stable_intramonth(synthetic_panel):
    panel, membership = synthetic_panel
    strat = JumpPenalizedMomentum(long_days=60, skip_days=5, monthly_rebalance=True)
    weights = strat.generate_weights(panel, membership)
    months = weights.index.to_series().dt.to_period("M")
    for _month, grp in weights.groupby(months):
        first_row = grp.iloc[0]
        for _, row in grp.iterrows():
            pd.testing.assert_series_equal(row, first_row, check_names=False)


# ---------------------------------------------------------------------------
# H157: VolScaledMomentum
# ---------------------------------------------------------------------------


def test_h157_weights_sum_leq_one(synthetic_panel):
    panel, membership = synthetic_panel
    strat = VolScaledMomentum(long_days=60, skip_days=5, top_quantile=0.3, vol_max=1.0)
    weights = strat.generate_weights(panel, membership)
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_h157_weights_non_negative(synthetic_panel):
    panel, membership = synthetic_panel
    strat = VolScaledMomentum(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights >= -1e-9).all().all()


def test_h157_scale_bounded(synthetic_panel):
    """Scaled weights must stay within [vol_min, vol_max] × base weights."""
    panel, membership = synthetic_panel
    vol_min, vol_max = 0.2, 1.5
    strat = VolScaledMomentum(
        long_days=60, skip_days=5, top_quantile=0.3, vol_min=vol_min, vol_max=vol_max
    )
    weights = strat.generate_weights(panel, membership)
    base_strat = MomentumLongOnly(long_days=60, skip_days=5, top_quantile=0.3)
    base_w = base_strat.generate_weights(panel, membership)
    # Row sums of scaled weights must be <= vol_max × row sums of base weights + tolerance
    scaled_sums = weights.sum(axis=1)
    base_sums = base_w.sum(axis=1)
    assert (scaled_sums <= base_sums * vol_max + 1e-6).all()


def test_h157_high_vol_reduces_exposure(crash_panel):
    """After a crash (high realized vol), scaled exposure should be lower than base."""
    panel, membership = crash_panel
    strat = VolScaledMomentum(
        long_days=60, skip_days=5, top_quantile=0.3, vol_target=0.10, vol_window=21
    )
    weights = strat.generate_weights(panel, membership)
    base_strat = MomentumLongOnly(long_days=60, skip_days=5, top_quantile=0.3)
    base_w = base_strat.generate_weights(panel, membership)
    # In the post-crash period, scaled exposure should average lower than base
    post_crash = slice(240, 280)
    scaled_avg = weights.iloc[post_crash].sum(axis=1).mean()
    base_avg = base_w.iloc[post_crash].sum(axis=1).mean()
    assert scaled_avg < base_avg + 1e-6, "High-vol period should reduce exposure"


# ---------------------------------------------------------------------------
# H158: DrawdownCircuitBreaker
# ---------------------------------------------------------------------------


def test_h158_weights_sum_leq_one(synthetic_panel):
    panel, membership = synthetic_panel
    strat = DrawdownCircuitBreaker(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_h158_weights_non_negative(synthetic_panel):
    panel, membership = synthetic_panel
    strat = DrawdownCircuitBreaker(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights >= -1e-9).all().all()


def test_h158_crash_reduces_exposure(crash_panel):
    """Drawdown circuit breaker must reduce exposure after crash period."""
    panel, membership = crash_panel
    strat = DrawdownCircuitBreaker(
        long_days=60,
        skip_days=5,
        top_quantile=0.3,
        dd_soft=-0.05,  # low threshold to trigger easily on small panel
        dd_hard=-0.10,
        dd_resume=-0.02,
        lookback=60,
    )
    weights = strat.generate_weights(panel, membership)
    base_strat = MomentumLongOnly(long_days=60, skip_days=5, top_quantile=0.3)
    base_w = base_strat.generate_weights(panel, membership)
    # Post-crash, circuit breaker should reduce average exposure
    post_crash = slice(230, 270)
    scaled_avg = weights.iloc[post_crash].sum(axis=1).mean()
    base_avg = base_w.iloc[post_crash].sum(axis=1).mean()
    assert scaled_avg <= base_avg + 1e-6, "Circuit breaker must not increase exposure after crash"


def test_h158_scale_values_valid(synthetic_panel):
    """Scale factors should be one of the valid states: full, soft, hard."""
    panel, membership = synthetic_panel
    strat = DrawdownCircuitBreaker(
        long_days=60, skip_days=5, top_quantile=0.5, scale_soft=0.5, scale_hard=0.25
    )
    weights = strat.generate_weights(panel, membership)
    base_strat = MomentumLongOnly(long_days=60, skip_days=5, top_quantile=0.5)
    base_w = base_strat.generate_weights(panel, membership)
    # Where base has nonzero weight, scaled weight should be base × {0.25, 0.5, 1.0}
    base_sums = base_w.sum(axis=1)
    scaled_sums = weights.sum(axis=1)
    active = base_sums > 1e-6
    if active.any():
        ratios = scaled_sums[active] / base_sums[active]
        valid_ratios = {0.25, 0.5, 1.0}
        for r in ratios:
            assert any(abs(r - v) < 1e-6 for v in valid_ratios), (
                f"Scale ratio {r} not in {valid_ratios}"
            )


# ---------------------------------------------------------------------------
# H159: SmoothVolTargetMomentum
# ---------------------------------------------------------------------------


def test_h159_weights_sum_leq_one(synthetic_panel):
    panel, membership = synthetic_panel
    strat = SmoothVolTargetMomentum(long_days=60, skip_days=5, top_quantile=0.3, vol_max=1.0)
    weights = strat.generate_weights(panel, membership)
    assert (weights.sum(axis=1) <= 1.0 + 1e-9).all()


def test_h159_weights_non_negative(synthetic_panel):
    panel, membership = synthetic_panel
    strat = SmoothVolTargetMomentum(long_days=60, skip_days=5, top_quantile=0.3)
    weights = strat.generate_weights(panel, membership)
    assert (weights >= -1e-9).all().all()


def test_h159_differs_from_h155_and_h157(synthetic_panel):
    """H159 = smooth signal + vol scaling; must differ from each component alone."""
    panel, membership = synthetic_panel
    strat_h155 = SmoothMomentum(long_days=60, skip_days=5, monthly_rebalance=False)
    strat_h157 = VolScaledMomentum(long_days=60, skip_days=5, monthly_rebalance=False)
    strat_h159 = SmoothVolTargetMomentum(long_days=60, skip_days=5, monthly_rebalance=False)
    w155 = strat_h155.generate_weights(panel, membership)
    w157 = strat_h157.generate_weights(panel, membership)
    w159 = strat_h159.generate_weights(panel, membership)
    assert not w159.equals(w155), "H159 must differ from H155"
    assert not w159.equals(w157), "H159 must differ from H157"


def test_h159_monthly_rebalance_stable_intramonth(synthetic_panel):
    """Stock selection (not vol scaling) is monthly; check intra-month row sums stable."""
    panel, membership = synthetic_panel
    strat = SmoothVolTargetMomentum(long_days=60, skip_days=5, monthly_rebalance=True)
    weights = strat.generate_weights(panel, membership)
    # Vol scaling is daily, so individual rows will differ intra-month,
    # but the SET of held tickers (non-zero cols) should be stable intra-month.
    months = weights.index.to_series().dt.to_period("M")
    for _month, grp in weights.groupby(months):
        first_nonzero = grp.iloc[0] > 0
        for _, row in grp.iterrows():
            pd.testing.assert_series_equal(
                (row > 0),
                first_nonzero,
                check_names=False,
                obj="Non-zero ticker set must be stable intra-month",
            )
