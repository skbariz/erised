"""Tests for the metrics module."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from quant.metrics.performance import (
    annualized_vol,
    cagr,
    max_drawdown,
    report,
    sharpe,
    sortino,
)


def _const_returns(daily_ret: float, n: int = 252 * 5) -> pd.Series:
    idx = pd.bdate_range("2000-01-03", periods=n)
    return pd.Series(daily_ret, index=idx)


def test_cagr_zero_when_zero_returns():
    r = _const_returns(0.0)
    assert cagr(r) == 0.0


def test_cagr_compounds_correctly():
    # 0.1% per day for 252 days should be ~28.6% annual
    r = _const_returns(0.001, n=252)
    expected = (1.001**252) - 1
    assert cagr(r) == pytest.approx(expected, rel=1e-6)


def test_annualized_vol_scales_with_sqrt_t():
    rng = np.random.default_rng(0)
    daily_vol = 0.01
    r = pd.Series(
        rng.normal(0, daily_vol, size=10000), index=pd.bdate_range("2000-01-01", periods=10000)
    )
    av = annualized_vol(r)
    assert av == pytest.approx(daily_vol * np.sqrt(252), rel=0.05)


def test_sharpe_zero_when_zero_returns():
    r = _const_returns(0.0)
    assert sharpe(r) == 0.0


def test_sharpe_positive_for_positive_drift():
    rng = np.random.default_rng(1)
    r = pd.Series(rng.normal(0.0004, 0.01, 5000), index=pd.bdate_range("2000-01-01", periods=5000))
    s = sharpe(r)
    assert s > 0
    # Empirical Sharpe for these params ~ 0.0004 / 0.01 * sqrt(252) ~ 0.635
    assert 0.4 < s < 0.9


def test_sortino_above_sharpe_for_skewed_returns():
    # Construct returns with positive skew: more small wins, occasional bigger losses
    # but overall positive — Sortino should exceed Sharpe.
    rng = np.random.default_rng(2)
    n = 5000
    base = rng.normal(0.0005, 0.005, n)
    # Add positive skew: occasional positive jumps
    jumps = rng.choice([0, 0.02], size=n, p=[0.99, 0.01])
    r = pd.Series(base + jumps, index=pd.bdate_range("2000-01-01", periods=n))
    assert sortino(r) > sharpe(r)


def test_max_drawdown_simple():
    # Equity: 1, 1.1, 1.21, 0.605 -> max DD = -50%
    returns = pd.Series([0.1, 0.1, -0.5])
    assert max_drawdown(returns) == pytest.approx(-0.5)


def test_max_drawdown_no_drawdown():
    returns = pd.Series([0.01, 0.01, 0.01])
    assert max_drawdown(returns) == pytest.approx(0.0)


def test_report_returns_complete_object():
    r = _const_returns(0.001, n=500)
    p = pd.Series(1.0, index=r.index)
    rep = report(r, positions=p)
    assert rep.n_periods == 500
    assert rep.sharpe > 0
    assert rep.max_drawdown <= 0
    assert rep.hit_rate > 0
