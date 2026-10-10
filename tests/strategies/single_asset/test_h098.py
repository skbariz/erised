"""Tests for H098: ReversionEntryRiskParity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies import ReversionEntryRiskParity


def _prices(n: int = 400) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(98)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    w = ReversionEntryRiskParity().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_shape_matches_input():
    prices = _prices(n=300)
    w = ReversionEntryRiskParity().generate_weights(prices)
    assert w.shape == (300, 2)


def test_weights_nonnegative():
    w = ReversionEntryRiskParity().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_sum_at_most_one():
    w = ReversionEntryRiskParity().generate_weights(_prices())
    assert (w.sum(axis=1).round(6) <= 1.0).all()


def test_normal_market_uses_h021_weights():
    # Steadily rising SPY → no drawdown → H021-like risk-parity weights throughout
    n = 400
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    spy = 100 * np.cumprod(np.full(n, 1.001))  # monotone up, drawdown never > 0
    tlt = 100 * np.cumprod(np.full(n, 1.0003))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = ReversionEntryRiskParity(vol_lookback=20, dd_trigger=0.10).generate_weights(prices)
    # No crisis fires → both assets have non-trivial weights after warmup
    tail = w.iloc[30:]
    assert (tail["SPY"] > 0).all() and (tail["TLT"] > 0).all()


def test_crash_triggers_crisis_weights():
    # Build a price path where SPY crashes > 10% to trigger crisis mode
    n = 400
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(42)
    spy_rets = rng.normal(0.0003, 0.005, n)
    # Insert a 20% drawdown starting at bar 200
    spy_rets[200:220] = -0.015  # 20 bars × -1.5% = -26% cumulative crash
    spy = 100 * np.cumprod(1 + spy_rets)
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.003, n))
    prices = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    w = ReversionEntryRiskParity(
        vol_lookback=20, dd_trigger=0.10, crisis_tlt=0.80, crisis_days=20
    ).generate_weights(prices)
    # During the crash window (bars 201-220), TLT weight should hit 0.80
    crisis_window = w.iloc[201:221]
    assert (crisis_window["TLT"] == 0.80).any()
