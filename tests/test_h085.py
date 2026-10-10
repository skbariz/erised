"""Tests for H085: MomentumMacroGate."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies.base import MomentumMacroGate


def _prices(n: int = 500, include_dbc: bool = True, include_vnq: bool = True) -> pd.DataFrame:
    idx = pd.date_range("2007-01-01", periods=n, freq="B")
    rng = np.random.default_rng(85)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, 0.01, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, 0.005, n))
    gld = 100 * np.cumprod(1 + rng.normal(0.0002, 0.008, n))
    shy = 100 * np.cumprod(1 + rng.normal(0.00005, 0.001, n))
    df = pd.DataFrame(
        {"adj_close": spy, "tlt_close": tlt, "gld_close": gld, "shy_close": shy}, index=idx
    )
    if include_dbc:
        dbc = 100 * np.cumprod(1 + rng.normal(0.0001, 0.012, n))
        df["dbc_close"] = dbc
    if include_vnq:
        vnq = 100 * np.cumprod(1 + rng.normal(0.0002, 0.009, n))
        df["vnq_close"] = vnq
    return df


def test_returns_correct_columns():
    w = MomentumMacroGate().generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT", "GLD", "DBC", "VNQ"}


def test_returns_subset_without_dbc_vnq():
    w = MomentumMacroGate().generate_weights(_prices(include_dbc=False, include_vnq=False))
    assert set(w.columns) == {"SPY", "TLT", "GLD"}


def test_shape_matches_input():
    prices = _prices(n=400)
    w = MomentumMacroGate().generate_weights(prices)
    assert w.shape[0] == 400


def test_weights_nonnegative():
    w = MomentumMacroGate().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_each_row_at_most_one_asset():
    w = MomentumMacroGate().generate_weights(_prices())
    row_sums = w.sum(axis=1)
    assert ((row_sums.round(6) == 0.0) | (row_sums.round(6) == 1.0)).all()


def test_inverted_curve_forces_tlt():
    # SHY rising much faster than TLT → curve inverted → must hold TLT
    idx = pd.date_range("2007-01-01", periods=400, freq="B")
    spy = 100 * np.cumprod(np.full(400, 1.002))  # strong momentum
    tlt = 100 * np.cumprod(np.full(400, 1.0001))  # barely rising
    gld = 100 * np.cumprod(np.full(400, 1.003))  # also strong
    shy = 100 * np.cumprod(np.full(400, 1.003))  # SHY >> TLT → inverted
    dbc = 100 * np.cumprod(np.full(400, 1.001))
    vnq = 100 * np.cumprod(np.full(400, 1.001))
    prices = pd.DataFrame(
        {
            "adj_close": spy,
            "tlt_close": tlt,
            "gld_close": gld,
            "shy_close": shy,
            "dbc_close": dbc,
            "vnq_close": vnq,
        },
        index=idx,
    )
    w = MomentumMacroGate(curve_days=63).generate_weights(prices)
    # After warmup, curve is always inverted → always TLT
    tail = w.iloc[80:]
    assert (tail["TLT"] == 1.0).all()
    assert (tail["SPY"] == 0.0).all()


def test_normal_curve_follows_momentum():
    # TLT outperforms SHY → curve not inverted → follow momentum
    idx = pd.date_range("2007-01-01", periods=400, freq="B")
    spy = 100 * np.cumprod(np.full(400, 1.003))  # strongest momentum
    tlt = 100 * np.cumprod(np.full(400, 1.002))  # TLT > SHY → normal curve
    gld = 100 * np.cumprod(np.full(400, 1.001))
    shy = 100 * np.cumprod(np.full(400, 1.0001))
    dbc = 100 * np.cumprod(np.full(400, 1.0005))
    vnq = 100 * np.cumprod(np.full(400, 1.0005))
    prices = pd.DataFrame(
        {
            "adj_close": spy,
            "tlt_close": tlt,
            "gld_close": gld,
            "shy_close": shy,
            "dbc_close": dbc,
            "vnq_close": vnq,
        },
        index=idx,
    )
    w = MomentumMacroGate(curve_days=63).generate_weights(prices)
    tail = w.iloc[280:]
    # SPY has the best momentum → SPY should win when curve is normal
    assert (tail["SPY"] == 1.0).all()
