"""Focused tests for H051: ThreeAssetRiskParity."""

from __future__ import annotations

import numpy as np
import pandas as pd

from quant.backtest.engine import assert_no_lookahead_multi
from quant.strategies.base import ThreeAssetRiskParity


def _prices(spy_rets, tlt_rets, gld_rets=None, start="2010-01-04"):
    n = len(spy_rets)
    idx = pd.bdate_range(start=start, periods=n + 1)
    spy = [100.0]
    tlt = [100.0]
    for r in spy_rets:
        spy.append(spy[-1] * (1 + r))
    for r in tlt_rets:
        tlt.append(tlt[-1] * (1 + r))
    df = pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)
    if gld_rets is not None:
        gld = [100.0]
        for r in gld_rets:
            gld.append(gld[-1] * (1 + r))
        df["gld_close"] = gld
    return df


def test_no_lookahead():
    rng = np.random.default_rng(42)
    n = 700
    prices = _prices(
        rng.normal(0.0005, 0.01, n).tolist(),
        rng.normal(0.0002, 0.005, n).tolist(),
        rng.normal(0.0003, 0.008, n).tolist(),
    )
    assert_no_lookahead_multi(ThreeAssetRiskParity().generate_weights, prices)


def test_weights_sum_to_one_when_all_assets_available():
    rng = np.random.default_rng(1)
    n = 300
    prices = _prices(
        rng.normal(0, 0.01, n).tolist(),
        rng.normal(0, 0.005, n).tolist(),
        rng.normal(0, 0.008, n).tolist(),
    )
    w = ThreeAssetRiskParity(vol_lookback=60).generate_weights(prices)
    active = (w["SPY"] + w["TLT"] + w["GLD"]) > 0
    sums = (w["SPY"] + w["TLT"] + w["GLD"])[active]
    assert np.allclose(sums.values, 1.0)


def test_falls_back_to_two_assets_when_gld_missing():
    """Without gld_close, should behave like H021 (SPY+TLT only)."""
    prices = _prices([0.001] * 200, [0.0005] * 200)  # no gld_rets
    w = ThreeAssetRiskParity(vol_lookback=60).generate_weights(prices)
    assert "GLD" in w.columns
    assert (w["GLD"] == 0.0).all()
    active = (w["SPY"] + w["TLT"]) > 0
    sums = (w["SPY"] + w["TLT"])[active]
    assert np.allclose(sums.values, 1.0)


def test_higher_vol_gets_lower_weight():
    rng = np.random.default_rng(0)
    prices = _prices(
        rng.normal(0, 0.025, 300).tolist(),   # high vol SPY
        rng.normal(0, 0.005, 300).tolist(),   # low vol TLT
        rng.normal(0, 0.010, 300).tolist(),   # medium vol GLD
    )
    w = ThreeAssetRiskParity(vol_lookback=60).generate_weights(prices)
    # TLT (lowest vol) should have highest weight
    assert w["TLT"].iloc[-1] > w["SPY"].iloc[-1]
    assert w["TLT"].iloc[-1] > w["GLD"].iloc[-1]


def test_weights_non_negative():
    rng = np.random.default_rng(2)
    prices = _prices(
        rng.normal(0, 0.01, 300).tolist(),
        rng.normal(0, 0.008, 300).tolist(),
        rng.normal(0, 0.007, 300).tolist(),
    )
    w = ThreeAssetRiskParity().generate_weights(prices)
    assert (w >= 0).all().all()


def test_columns():
    prices = _prices([0.001] * 200, [0.001] * 200, [0.001] * 200)
    w = ThreeAssetRiskParity().generate_weights(prices)
    assert list(w.columns) == ["SPY", "TLT", "GLD"]


def test_equal_vol_gives_equal_thirds():
    rng = np.random.default_rng(3)
    rets = rng.normal(0, 0.01, 300).tolist()
    prices = _prices(rets, rets, rets)
    w = ThreeAssetRiskParity(vol_lookback=60).generate_weights(prices)
    active = (w.sum(axis=1)) > 0
    assert np.allclose(w["SPY"][active].values, 1 / 3, atol=0.02)
    assert np.allclose(w["TLT"][active].values, 1 / 3, atol=0.02)
    assert np.allclose(w["GLD"][active].values, 1 / 3, atol=0.02)
