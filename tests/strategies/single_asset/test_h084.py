"""Tests for H084: MomentumCarryBlend."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies import MomentumCarryBlend


def _prices(n: int = 500) -> pd.DataFrame:
    idx = pd.date_range("2004-01-01", periods=n, freq="B")
    rng = np.random.default_rng(84)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    gld = 100 * np.cumprod(1 + rng.normal(0.0002, 0.008, n))
    shy = 100 * np.cumprod(1 + rng.normal(0.00005, 0.001, n))
    return pd.DataFrame(
        {"adj_close": spy, "tlt_close": tlt, "gld_close": gld, "shy_close": shy}, index=idx
    )


def test_returns_correct_columns():
    w = MomentumCarryBlend().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT", "GLD"}


def test_shape_matches_input():
    prices = _prices(n=400)
    w = MomentumCarryBlend().generate_weights(prices)
    assert w.shape[0] == 400


def test_weights_nonnegative():
    w = MomentumCarryBlend().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_sum_at_most_one():
    w = MomentumCarryBlend().generate_weights(_prices())
    assert (w.sum(axis=1).round(6) <= 1.0).all()


def test_no_shy_gives_momentum_only():
    # Without SHY, carry sub-signal is always flat → result = 0.5 * momentum
    prices = _prices(n=500).drop(columns=["shy_close"])
    w = MomentumCarryBlend(blend=0.5).generate_weights(prices)
    # Weights should all be ≤ 0.5 (half from momentum, carry part flat)
    tail = w.iloc[270:]
    assert (tail.max(axis=1).round(6) <= 0.5 + 1e-6).all()


def test_tlt_carry_when_tlt_outperforms_shy():
    # TLT rising strongly, SHY flat → TLT carry signal fires; carry sub gives TLT=1
    idx = pd.date_range("2004-01-01", periods=500, freq="B")
    spy = 100 * np.cumprod(np.full(500, 1.0005))
    tlt = 100 * np.cumprod(np.full(500, 1.002))  # TLT >> SHY
    gld = 100 * np.cumprod(np.full(500, 1.0005))
    shy = 100 * np.cumprod(np.full(500, 1.0001))
    prices = pd.DataFrame(
        {"adj_close": spy, "tlt_close": tlt, "gld_close": gld, "shy_close": shy}, index=idx
    )
    w = MomentumCarryBlend(blend=0.5).generate_weights(prices)
    # TLT should get at least 0.5 from carry sub-signal
    tail = w.iloc[270:]
    assert (tail["TLT"] >= 0.5 - 1e-6).all()
