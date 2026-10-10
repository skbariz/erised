"""Tests for H070: MultiAssetDynamicVolTarget."""

from __future__ import annotations

import numpy as np
import pandas as pd

from erised.strategies import MultiAssetDynamicVolTarget


def _prices(n: int = 300, spy_vol: float = 0.01, tlt_vol: float = 0.005) -> pd.DataFrame:
    idx = pd.date_range("2002-01-01", periods=n, freq="B")
    rng = np.random.default_rng(55)
    spy = 100 * np.cumprod(1 + rng.normal(0.0003, spy_vol, n))
    tlt = 100 * np.cumprod(1 + rng.normal(0.0001, tlt_vol, n))
    return pd.DataFrame({"adj_close": spy, "tlt_close": tlt}, index=idx)


def test_returns_spy_tlt_columns():
    strat = MultiAssetDynamicVolTarget()
    w = strat.generate_weights(_prices())
    assert set(w.columns) == {"SPY", "TLT"}


def test_shape_matches_input():
    prices = _prices(n=250)
    w = MultiAssetDynamicVolTarget().generate_weights(prices)
    assert w.shape == (250, 2)


def test_weights_nonnegative():
    w = MultiAssetDynamicVolTarget().generate_weights(_prices())
    assert (w >= -1e-10).all().all()


def test_weights_at_most_one_per_asset():
    w = MultiAssetDynamicVolTarget().generate_weights(_prices())
    assert (w["SPY"] <= 1.0 + 1e-10).all()
    assert (w["TLT"] <= 1.0 + 1e-10).all()


def test_total_weight_at_most_one():
    # After warmup, total weight should not exceed 1.0 (no leverage)
    w = MultiAssetDynamicVolTarget().generate_weights(_prices(n=300))
    total = w["SPY"] + w["TLT"]
    assert (total.iloc[80:] <= 1.0 + 1e-10).all()


def test_scales_down_in_high_vol_regime():
    # High-vol SPY should cause scale-down → total < 1.0
    prices = _prices(n=300, spy_vol=0.04, tlt_vol=0.02)
    strat = MultiAssetDynamicVolTarget(vol_target=0.05)
    w = strat.generate_weights(prices)
    total = w["SPY"] + w["TLT"]
    # Should see some scaling below 1.0 in later folds
    assert (total.iloc[100:] < 0.999).any()


def test_tlt_heavy_when_spy_more_volatile():
    # SPY more volatile → inverse-vol gives more weight to TLT
    prices = _prices(n=300, spy_vol=0.02, tlt_vol=0.005)
    w = MultiAssetDynamicVolTarget().generate_weights(prices)
    tail = w.iloc[80:]
    assert (tail["TLT"] > tail["SPY"]).mean() > 0.7
