"""Focused tests for H055: FiveAssetRotation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.backtest.engine import assert_no_lookahead_multi
from erised.strategies import FiveAssetRotation


def _prices(
    spy_rets, tlt_rets=None, gld_rets=None, dbc_rets=None, vnq_rets=None, start="2010-01-04"
):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy}, index=idx)
    for col, rets in [
        ("tlt_close", tlt_rets),
        ("gld_close", gld_rets),
        ("dbc_close", dbc_rets),
        ("vnq_close", vnq_rets),
    ]:
        if rets is not None:
            px = [100.0]
            for r in rets:
                px.append(px[-1] * (1 + r))
            df[col] = px
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    n = 800
    prices = _prices(
        rng.normal(0.0005, 0.01, n).tolist(),
        rng.normal(0.0002, 0.005, n).tolist(),
        rng.normal(0.0003, 0.008, n).tolist(),
        rng.normal(0.0001, 0.012, n).tolist(),
        rng.normal(0.0004, 0.009, n).tolist(),
    )
    assert_no_lookahead_multi(FiveAssetRotation().generate_weights, prices)


def test_selects_top_momentum_asset():
    """SPY best momentum → all weight in SPY."""
    n = 400
    prices = _prices(
        [0.003] * n,  # SPY: strong positive
        [0.001] * n,  # TLT: weaker
        [0.0005] * n,  # GLD: weakest
        [0.001] * n,  # DBC
        [0.001] * n,  # VNQ
    )
    w = FiveAssetRotation(long_days=252, skip_days=21).generate_weights(prices)
    assert w["SPY"].iloc[-1] == 1.0
    assert w["TLT"].iloc[-1] == 0.0


def test_goes_to_cash_when_all_negative():
    """All assets with negative momentum → cash (all weights zero)."""
    n = 400
    prices = _prices(
        [-0.001] * n,
        [-0.0005] * n,
        [-0.0002] * n,
        [-0.001] * n,
        [-0.0008] * n,
    )
    w = FiveAssetRotation(long_days=252, skip_days=21).generate_weights(prices)
    assert (w.sum(axis=1).iloc[-1]) == 0.0


def test_weight_sum_at_most_one():
    rng = np.random.default_rng(5)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
        rng.normal(0, 0.008, n).tolist(),
        rng.normal(0, 0.012, n).tolist(),
        rng.normal(0, 0.009, n).tolist(),
    )
    w = FiveAssetRotation().generate_weights(prices)
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_weights_non_negative():
    rng = np.random.default_rng(6)
    n = 400
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
        rng.normal(0, 0.008, n).tolist(),
        rng.normal(0, 0.012, n).tolist(),
        rng.normal(0, 0.009, n).tolist(),
    )
    w = FiveAssetRotation().generate_weights(prices)
    assert (w >= 0).all().all()


def test_works_with_only_spy():
    """Should handle missing asset columns gracefully."""
    prices = _prices([0.001] * 400)
    w = FiveAssetRotation().generate_weights(prices)
    assert "SPY" in w.columns
    assert (w.sum(axis=1) <= 1.0 + 1e-9).all()


def test_columns_present():
    prices = _prices([0.001] * 400, [0.001] * 400, [0.001] * 400, [0.001] * 400, [0.001] * 400)
    w = FiveAssetRotation().generate_weights(prices)
    assert set(w.columns) == {"SPY", "TLT", "GLD", "DBC", "VNQ"}
