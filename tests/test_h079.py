"""Tests for H079: CrisisProtocolRiskParity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.base import CrisisProtocolRiskParity


def _prices(n: int = 300, crash_at: int | None = None) -> pd.DataFrame:
    idx = pd.date_range("2003-01-01", periods=n, freq="B")
    rng = np.random.default_rng(79)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.008, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.004, n))
    vix = np.full(n, 18.0)
    if crash_at is not None:
        # Simulate crash: SPY falls 15%, VIX spikes to 45
        spy[crash_at : crash_at + 30] *= np.cumprod(np.full(30, 0.994))
        vix[crash_at : crash_at + 30] = 45.0
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt, "vix_close": vix}, index=idx)


def test_returns_spy_tlt_columns():
    w = CrisisProtocolRiskParity().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_normal_weights_match_risk_parity_structure():
    # Without crash, weights should sum to 1 and both positive after warmup
    w = CrisisProtocolRiskParity(vol_lookback=60).generate_weights(_prices(n=300))
    tail = w.iloc[65:]
    sums = (tail["SPY"] + tail["TLT"]).round(6)
    assert (sums == 1.0).all()


def test_crisis_mode_on_crash():
    # Use a low crisis_dd threshold so it triggers quickly after VIX spike
    prices = _prices(n=300, crash_at=150)
    strat = CrisisProtocolRiskParity(
        drawdown_window=20, crisis_dd=-0.02, vix_spike=35.0, crisis_tlt=0.90
    )
    w = strat.generate_weights(prices)
    # VIX spikes at 150, shift(1) means crisis visible at 151+; check well inside crash window
    crisis_rows = w.iloc[155:175]
    assert (crisis_rows["TLT"].round(2) == 0.90).all()
    assert (crisis_rows["SPY"].round(2) == 0.10).all()


def test_resumes_normal_after_recovery():
    prices = _prices(n=400, crash_at=150)
    # VIX returns to 18 after crash window
    strat = CrisisProtocolRiskParity(
        vol_lookback=60,
        drawdown_window=20,
        crisis_dd=-0.10,
        resume_dd=-0.07,
        vix_spike=35.0,
        vix_resume=30.0,
        crisis_tlt=0.90,
    )
    w = strat.generate_weights(prices)
    # After recovery (crash was at 150-180), later rows should not be all 90/10
    late = w.iloc[220:]
    assert not (late["TLT"].round(2) == 0.90).all()


def test_weights_nonnegative():
    w = CrisisProtocolRiskParity().generate_weights(_prices(n=300, crash_at=100))
    assert (w >= -1e-10).all().all()


def test_shape_matches_input():
    prices = _prices(n=200)
    w = CrisisProtocolRiskParity().generate_weights(prices)
    assert w.shape == (200, 2)
